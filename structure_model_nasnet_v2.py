# %%
# two_stage_nasnet_pipeline.py
import os, math, json, random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess


# %%

# =============== Config ===============
SEED = 42
random.seed(SEED); tf.random.set_seed(SEED)

TRAIN_DIR = "D:\Diux\hoctap\DoAn\ddr\\train_preprocess"        # chứa 5 thư mục con 0..4
VAL_DIR   = "D:\Diux\hoctap\DoAn\ddr\\val_preprocess"          # chứa 5 thư mục con 0..4

IMG_SIZE  = 331                       # chuẩn NASNetLarge
BATCH     = 16

# Warmup + fine-tune
HEAD_EPOCHS = 5                       # train head (đóng băng backbone)
FT_EPOCHS   = 20                      # fine-tune toàn bộ
LR_HEAD     = 3e-4
LR_FT       = 1e-4

OUT_DIR     = "outputs_two_stage"
os.makedirs(OUT_DIR, exist_ok=True)

# =============== Dataset loaders ===============
AUTOTUNE = tf.data.AUTOTUNE

def make_base_ds(root_dir, subset="train"):
    """
    Load DS 5 lớp 0..4 từ folder. Không augment ở đây (augment sẽ ở pipeline).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        labels="inferred",
        label_mode="int",
        class_names=['0','1','2','3','4'],
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        shuffle=True if subset=="train" else False,
        seed=SEED
    )
    return ds

# Tạo augmentation & preprocessing
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
], name="augment")

def preprocess(x):
    # x: float32 [0,255]
    return nasnet_preprocess(x)

def make_pipeline_for_module1(ds, training=True):
    """
    Module 1: No-DR (y=0) vs DR (y in 1..4)  -> nhãn nhị phân {0,1}
    """
    def map_to_bin(x, y):
        y_bin = tf.where(tf.equal(y, 0), tf.zeros_like(y), tf.ones_like(y))
        return x, tf.cast(y_bin, tf.float32)

    ds = ds.map(map_to_bin, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x,y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x,y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x,y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

def make_pipeline_for_module2(ds, training=True):
    """
    Module 2: chỉ giữ mẫu DR (y in 1..4), ánh xạ nhãn 1..4 -> 0..3 (softmax 4 lớp)
    """
    def filter_dr(x, y):
        keep = tf.not_equal(y, 0)
        return keep

    def map_to_4(x, y):
        y4 = y - 1  # 1..4 -> 0..3
        return x, tf.cast(y4, tf.int32)

    ds = ds.filter(filter_dr)
    ds = ds.map(map_to_4, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x,y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x,y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x,y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

# =============== Models ===============
def build_backbone(trainable=False):
    base = NASNetLarge(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg', weights='imagenet')
    base.trainable = trainable
    inp  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x    = base(inp, training=False)
    x    = layers.Dropout(0.3)(x)
    return inp, x, base

def build_module1():
    # Binary: No-DR (0) vs DR (1)
    inp, feat, base = build_backbone(trainable=False)
    x = layers.Dense(512, activation='relu')(feat)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name="bin_out")(x)
    model = models.Model(inp, out, name="NASNetL_Module1_NoDR_vs_DR")
    opt = tf.keras.optimizers.Adam(LR_HEAD)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model, base

def build_module2():
    # 4-class: 1..4 -> 0..3
    inp, feat, base = build_backbone(trainable=False)
    x = layers.Dense(512, activation='relu')(feat)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(4, activation='softmax', name="stage4_out")(x)
    model = models.Model(inp, out, name="NASNetL_Module2_Stages_1to4")
    opt = tf.keras.optimizers.Adam(LR_HEAD)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, base

def unfreeze_all_and_recompile(model, base, lr=LR_FT):
    base.trainable = True
    # Optionally: có thể chọn unfreeze từ 1 block cuối cùng nếu GPU yếu
    # for layer in base.layers[:-50]: layer.trainable = False
    opt = tf.keras.optimizers.Adam(lr)
    # loss/metrics giữ nguyên
    model.compile(optimizer=opt, loss=model.loss, metrics=model.metrics)
    return model

# =============== Training ===============
def train_two_stage():
    print("Loading base datasets...")
    train_base = make_base_ds(TRAIN_DIR, subset="train")
    val_base   = make_base_ds(VAL_DIR, subset="val")

    m1_best = os.path.join(OUT_DIR, "module1_ft_best.keras")
    m2_best = os.path.join(OUT_DIR, "module2_ft_best.keras")

    # ---------- Module 1 ----------
    if os.path.exists(m1_best):
        print("\n=== Module 1: SKIP (đã có module1_ft_best.keras) ===")
        m1 = tf.keras.models.load_model(m1_best, compile=False)
    else:
        print("\n=== Module 1: No-DR vs DR ===")
        tr1 = make_pipeline_for_module1(train_base, training=True)
        va1 = make_pipeline_for_module1(val_base,   training=False)

        m1, m1_base = build_module1()

        cbs1 = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
            ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6, monitor='val_auc', mode='max'),
            ModelCheckpoint(m1_best, monitor='val_auc', mode='max', save_best_only=True)
        ]

        print("Stage-1 (head) ...")
        m1.fit(tr1, epochs=HEAD_EPOCHS, validation_data=va1, callbacks=cbs1, verbose=1)

        print("Fine-tune all layers ...")
        m1 = unfreeze_all_and_recompile(m1, m1_base, lr=LR_FT)
        cbs1_ft = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
            ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7, monitor='val_auc', mode='max'),
            ModelCheckpoint(m1_best, monitor='val_auc', mode='max', save_best_only=True)
        ]
        m1.fit(tr1, epochs=FT_EPOCHS, validation_data=va1, callbacks=cbs1_ft, verbose=1)
        m1.save(os.path.join(OUT_DIR, "module1_final.keras"))
        print("Saved Module 1.")

    # ---------- Module 2 ----------
    if os.path.exists(m2_best):
        print("\n=== Module 2: SKIP (đã có module2_ft_best.keras) ===")
        m2 = tf.keras.models.load_model(m2_best, compile=False)
    else:
        print("\n=== Module 2: Stages 1..4 (4-class) ===")
        tr2 = make_pipeline_for_module2(train_base, training=True)
        va2 = make_pipeline_for_module2(val_base,   training=False)

        m2, m2_base = build_module2()

        cbs2 = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max'),
            ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6, monitor='val_accuracy', mode='max'),
            ModelCheckpoint(m2_best, monitor='val_accuracy', mode='max', save_best_only=True)
        ]

        print("Stage-1 (head) ...")
        m2.fit(tr2, epochs=HEAD_EPOCHS, validation_data=va2, callbacks=cbs2, verbose=1)

        print("Fine-tune all layers ...")
        m2 = unfreeze_all_and_recompile(m2, m2_base, lr=LR_FT)
        cbs2_ft = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max'),
            ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7, monitor='val_accuracy', mode='max'),
            ModelCheckpoint(m2_best, monitor='val_accuracy', mode='max', save_best_only=True)
        ]
        m2.fit(tr2, epochs=FT_EPOCHS, validation_data=va2, callbacks=cbs2_ft, verbose=1)
        m2.save(os.path.join(OUT_DIR, "module2_final.keras"))
        print("Saved Module 2.")


# =============== Inference (2-stage) ===============
def load_models_for_infer():
    m1 = tf.keras.models.load_model(os.path.join(OUT_DIR, "module1_ft_best.keras"), compile=False)
    m2 = tf.keras.models.load_model(os.path.join(OUT_DIR, "module2_ft_best.keras"), compile=False)
    return m1, m2

def load_and_preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x   = tf.keras.utils.img_to_array(img)
    x   = tf.cast(x, tf.float32)
    x   = nasnet_preprocess(x)
    return tf.expand_dims(x, 0)

def predict_stage(img_path, thr=0.5):
    """
    2-stage suy luận:
     - Module1 -> pDR = sigmoid(out). Nếu pDR < thr => dự đoán lớp 0 (No-DR).
     - Ngược lại -> Module2 (softmax 4 lớp 0..3) -> map về 1..4.
    """
    m1, m2 = load_models_for_infer()
    x = load_and_preprocess_image(img_path)

    p_dr = float(m1.predict(x, verbose=0)[0][0])
    if p_dr < thr:
        return 0, {"p_dr": p_dr, "module2": None}

    # DR: gọi module 2
    probs = m2.predict(x, verbose=0)[0]  # shape (4,)
    cls_0_3 = int(tf.argmax(probs).numpy())
    final_label = cls_0_3 + 1           # map về 1..4
    return final_label, {"p_dr": p_dr, "probs_1to4": probs.tolist()}

# =============== Main ===============
if __name__ == "__main__":
    train_two_stage()
    # Ví dụ suy luận:
    # label, info = predict_stage("some_image.jpg")
    # print(label, info)
