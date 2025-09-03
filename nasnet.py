# %%
import os, random, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocess


# %%
# ================== CONFIG ==================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

TRAIN_DIR = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/train_preprocess"   # <--- chỉnh nếu khác
VAL_DIR   = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/val_preprocess"     # <--- chỉnh nếu khác

IMG_SIZE = 331        # NASNetLarge
BATCH_SIZE = 16
HEAD_EPOCHS = 12      # train head (freeze backbone)
FT_EPOCHS   = 12      # fine-tune cuối
AUGMENT = False       # bạn đã tăng cường offline -> để False

OUT_DIR = "outputs_two_stage"
os.makedirs(OUT_DIR, exist_ok=True)

# Bật memory growth GPU (nếu có)
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

# %%
# ================== DATA ==================
def make_augmenter():
    # Nếu không augment, trả về 1 layer giữ nguyên đầu vào (identity)
    if not AUGMENT:
        return keras.Sequential([layers.Lambda(lambda x: x, name="identity")], name="augmenter")
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.05),
    ], name="augmenter")

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = nasnet_preprocess(x)  # [-1,1]
    return x, y

def build_ds_5cls(directory, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=["0","1","2","3","4"],  # cố định thứ tự
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

def build_ds_binary(directory, shuffle):
    # (0) vs (1..4)
    ds = build_ds_5cls(directory, shuffle)
    def to_binary(x, y):
        y_bin = tf.cast(y > 0, tf.float32)  # 0/1
        return x, tf.expand_dims(y_bin, -1) # shape (B,1) cho BCE
    ds = ds.map(to_binary, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

def build_ds_dr4(directory, shuffle):
    # Chỉ lấy lớp 1..4, Keras sẽ gán nhãn 0..3 theo class_names dưới
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=["1","2","3","4"],
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# %%
# ================== MODELS ==================
def build_stage1_binary_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = make_augmenter()(inp)
    base = NASNetLarge(include_top=False, weights="imagenet", input_tensor=x, pooling="avg", name="NASNet")
    base.trainable = False  # train head trước
    x=base(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="Stage1_NASNet_Binary")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")]
    )
    return model

def unfreeze_last_n(model, n_last=50, lr=1e-4):
    base = model.get_layer("NASNet")
    base.trainable = True
    # Đóng băng phần đầu, chỉ mở n_last layer cuối
    if n_last is not None and n_last > 0:
        for l in base.layers[:-n_last]:
            l.trainable = False
    # Thường đóng băng BN khi fine-tune
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"),
                 keras.metrics.BinaryAccuracy(name="acc")]
    )

def build_stage2_dr4_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = make_augmenter()(inp)
    base = NASNetLarge(include_top=False, weights="imagenet", input_tensor=x, pooling="avg", name="NASNet")
    base.trainable = False
    x=base(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(4, activation="softmax")(x)  # 4 lớp: 1..4 -> 0..3
    model = keras.Model(inp, out, name="Stage2_NASNet_DR4")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# %%
# ================== TRAIN & EVAL ==================
def train_stage1():
    print("\n=== Stage 1: Binary no-DR vs DR ===")
    ds_tr = build_ds_binary(TRAIN_DIR, shuffle=True)
    ds_va = build_ds_binary(VAL_DIR, shuffle=False)

    m1 = build_stage1_binary_model()
    cbs = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(os.path.join(OUT_DIR, "stage1_head_best.keras"), monitor="val_loss", save_best_only=True)
    ]
    m1.fit(ds_tr, validation_data=ds_va, epochs=HEAD_EPOCHS, callbacks=cbs)

    # Fine-tune
    unfreeze_last_n(m1, n_last=50, lr=1e-4)
    cbs_ft = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(os.path.join(OUT_DIR, "stage1_finetune_best.keras"), monitor="val_loss", save_best_only=True)
    ]
    m1.fit(ds_tr, validation_data=ds_va, epochs=FT_EPOCHS, callbacks=cbs_ft)

    # Lưu
    path_final = os.path.join(OUT_DIR, "stage1_final.keras")
    m1.save(path_final)
    print("Saved:", path_final)
    return m1

def train_stage2(m1_base=None):
    print("\n=== Stage 2: DR 4 lớp (1..4) ===")
    ds_tr = build_ds_dr4(TRAIN_DIR, shuffle=True)
    ds_va = build_ds_dr4(VAL_DIR, shuffle=False)

    m2 = build_stage2_dr4_model()

    # Khởi tạo base của Stage2 từ Stage1 (warm-start)
    if m1_base is not None:
        try:
            m2.get_layer("NASNet").set_weights(m1_base.get_layer("NASNet").get_weights())
            print(">> Warm-start Stage2 từ backbone Stage1.")
        except Exception as e:
            print(">> Không thể copy trọng số NASNet từ Stage1:", e)

    cbs = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(os.path.join(OUT_DIR, "stage2_head_best.keras"), monitor="val_loss", save_best_only=True)
    ]
    m2.fit(ds_tr, validation_data=ds_va, epochs=HEAD_EPOCHS, callbacks=cbs)

    # Fine-tune backbone
    # (mở n_last layer cuối — có thể chỉnh 50→100 nếu dữ liệu nhiều)
    base2 = m2.get_layer("NASNet")
    base2.trainable = True
    for l in base2.layers[:-50]:
        l.trainable = False
    for l in base2.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    m2.compile(optimizer=keras.optimizers.Adam(1e-4),
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
    cbs_ft = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(os.path.join(OUT_DIR, "stage2_finetune_best.keras"), monitor="val_loss", save_best_only=True)
    ]
    m2.fit(ds_tr, validation_data=ds_va, epochs=FT_EPOCHS, callbacks=cbs_ft)

    path_final = os.path.join(OUT_DIR, "stage2_final.keras")
    m2.save(path_final)
    print("Saved:", path_final)
    return m2

# %%
# ================== 5-CLASS INFERENCE (tuỳ chọn) ==================
def predict_image_5class(img_path, model_stage1, model_stage2, thr=0.5):
    """Trả về nhãn 0..4 theo pipeline 2 giai đoạn."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = nasnet_preprocess(x)
    x = np.expand_dims(x, 0)

    p_dr = float(model_stage1.predict(x, verbose=0)[0][0])
    if p_dr < thr:
        return 0
    # DR -> 1..4 (model_stage2 output 0..3)
    p4 = model_stage2.predict(x, verbose=0)[0]
    return int(np.argmax(p4)) + 1

# %%
def main():
    m1 = train_stage1()
    m2 = train_stage2(m1_base=m1)

    # Lưu tóm tắt
    with open(os.path.join(OUT_DIR, "training_done.json"), "w") as f:
        json.dump({"stage1": "done", "stage2": "done"}, f, indent=2)
    print("\n==> Đã huấn luyện xong. Mô hình nằm trong:", OUT_DIR)
    print("   - stage1_final.keras")
    print("   - stage2_final.keras")

# %%
main()

# %%
m1 = build_stage1_binary_model()
print(m1.get_layer("NASNet"))

# %%
print([l.name for l in m1.layers[:20]])

# %%
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ===== Config =====
TEST_DIR = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/test_preprocess"   # thư mục test gồm 0..4
IMG_SIZE = 331      # NASNetLarge input
BATCH_SIZE = 16
THRESHOLD = 0.5     # ngưỡng phân biệt stage1 (0 vs DR)

# ===== Dataset =====
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=["0", "1", "2", "3", "4"],
    color_mode="rgb",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = nasnet_preprocess(x)
    return x, y

test_ds = test_ds.map(preprocess)

# ===== Load models =====
stage1 = tf.keras.models.load_model(
    "outputs_two_stage/stage1_final.keras",
    safe_mode=False
)
stage2 = tf.keras.models.load_model(
    "outputs_two_stage/stage2_final.keras",
    safe_mode=False
)
# ===== Predict pipeline =====
y_true = np.concatenate([y for _, y in test_ds], axis=0)

y_pred = []
for x_batch, _ in test_ds:
    # Stage1 dự đoán xác suất DR
    p_stage1 = stage1.predict(x_batch, verbose=0).ravel()

    # Stage2 dự đoán chi tiết cho các ảnh DR
    p_stage2 = stage2.predict(x_batch, verbose=0)

    for i in range(len(p_stage1)):
        if p_stage1[i] < THRESHOLD:
            y_pred.append(0)
        else:
            cls = np.argmax(p_stage2[i]) + 1  # shift 0..3 -> 1..4
            y_pred.append(cls)

y_pred = np.array(y_pred)

# ===== Metrics =====
print("=== Classification Report (Stage1+Stage2 pipeline) ===")
print(classification_report(y_true, y_pred, digits=4))

acc = accuracy_score(y_true, y_pred)
print("Overall Accuracy:", acc)

# ===== Confusion Matrix =====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0","1","2","3","4"],
            yticklabels=["0","1","2","3","4"])
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Pipeline 2 stage)")
plt.show()


# %%
!ipynb-py-convert structure_model_nasnet.ipynb nasnet.py