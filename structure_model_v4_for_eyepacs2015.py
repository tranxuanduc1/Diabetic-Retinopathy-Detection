# %%
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# %%
"""

"""

# %%
import tensorflow as tf, gc
tf.keras.backend.clear_session(); gc.collect()

# Bật memory growth (rất quan trọng)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# %%
# ================== CONFIG ==================
TRAIN_DIR = "eyepacs_2015/train_preprocess_ben_graham"
VAL_DIR   = "eyepacs_2015/val_preprocess_ben_graham"

IMAGE_SIZE = 448       # 448 hoặc 512 cho CPU; 600 sẽ rất chậm trên CPU
BATCH_SIZE = 8
SEED = 42

FREEZE_BACKBONE = True   # freeze giai đoạn đầu cho CPU
DROP_RATE = 0.4          # dropout trong head
DENSE_UNITS = 1024

LR = 3e-4
WEIGHT_DECAY = 1e-4
LOSS_W_SOFTMAX = 1.0
LOSS_W_ORDINAL = 0.5

EPOCHS = 15              # ví dụ (bạn tăng sau)

# %%
AUTOTUNE = tf.data.AUTOTUNE
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB4

# %%
# ========== Ordinal utils ==========
def ordinal_encode_tf(y_int):
    """y_int: (B,) int32 0..4 -> (B,4) float32: [y>=1, y>=2, y>=3, y>=4]"""
    y_int = tf.cast(y_int, tf.int32)
    thresholds = tf.constant([1, 2, 3, 4], dtype=tf.int32)  # (4,)
    y_exp = tf.expand_dims(y_int, axis=-1)                  # (B,1)
    return tf.cast(y_exp >= thresholds, tf.float32)         # (B,4)

def map_preprocess(image, label):
    # image_dataset_from_directory trả image uint8 [0..255]; EfficientNet preprocess sẽ scale
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # -> [0..1] cho EfficientNet
    return image, ordinal_encode_tf(label)




# %%
# ========== Dataset loaders (không cache vào RAM) ==========
def make_ds(data_dir, subset="train"):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=["0","1","2","3","4"],   # cố định thứ tự nhãn
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=(subset=="train"),
        seed=SEED
    )
    ds = ds.map(map_preprocess, num_parallel_calls=AUTOTUNE)
    # ds = ds.map(map_dual_targets, num_parallel_calls=AUTOTUNE)
    # Không dùng .cache() để tiết kiệm RAM; chỉ prefetch
    ds = ds.prefetch(AUTOTUNE)
    return ds

# %%
def sca_block(x, ratio=8, name="sca"):
    """Simple Channel Attention: GAP -> 1x1 Conv (reduce) -> 1x1 Conv (gate sigmoid) -> multiply."""
    ch = int(x.shape[-1])
    mid = max(ch // ratio, 1)

    gap = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
    red = layers.Conv2D(mid, 1, padding="same", activation="relu",
                        use_bias=True, name=f"{name}_reduce")(gap)
    gate = layers.Conv2D(ch, 1, padding="same", activation="sigmoid",
                         use_bias=True, name=f"{name}_gate")(red)
    out = layers.Multiply(name=f"{name}_mul")([x, gate])
    return out

# %%
# ========== Model builder ==========
def build_model(img_size=IMAGE_SIZE, freeze_backbone=FREEZE_BACKBONE):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Backbone EfficientNet-B4 (ImageNet)
    base = EfficientNetB4(include_top=False, weights="imagenet", input_tensor=inputs)
    base.trainable = not freeze_backbone

    x = base.output
    # CBAM ở feature map cuối (nhẹ)
    x = sca_block(x, ratio=8, name="sca")

    # Global pooling + head chung
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROP_RATE)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROP_RATE * 0.75)(x)

    # Head A: Softmax 5 lớp
    out_soft = layers.Dense(5, activation="softmax", name="softmax")(x)
    # Head B: Ordinal (≥1..4), sigmoid
    out_ord  = layers.Dense(4, activation="sigmoid", name="ordinal")(x)

    model = models.Model(inputs, out_ord, name="EffB4_CBAM_DualHead")
    return model

# %%
# ========== Compile ==========
def compile_model(model,
                  lr=LR,
                  wd=WEIGHT_DECAY,
                  ):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
   

    losses = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    metrics = [tf.keras.metrics.AUC(name="auc", multi_label=True)]

    model.compile(optimizer=optimizer, loss=losses,
                   metrics=metrics)
    return model

# %%
# ================== MAIN ==================

print("Loading datasets...")
ds_train = make_ds(TRAIN_DIR, subset="train")
ds_val   = make_ds(VAL_DIR, subset="val")

print("Building model...")
model = build_model()
model = compile_model(model)





# %%
# ===== Hàm mở block và train từng stage =====
def unfreeze_blocks_by_prefix(model, prefixes):
    """Mở các block theo prefix như 'block6', 'block7'"""
    for layer in model.layers:
        if any(layer.name.startswith(pref) for pref in prefixes):
            layer.trainable = True
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

def run_finetune_stage(model, stage_idx, prefixes, lr, save_path):
    print(f"\n=== Stage {stage_idx}: Unfreeze {prefixes} ===")
    unfreeze_blocks_by_prefix(model, prefixes)

   
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)


    losses = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.AUC(name="auc", multi_label=True)]

    model.compile(optimizer=optimizer, loss=losses,
                   metrics=metrics)

    callbacks_ft = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=10,
        callbacks=callbacks_ft,
        verbose=1
    )
    return history

# %%
# Stage definitions: mở block từ 7 → 1
stage_blocks = [
    ["block6", "block7"],
    ["block4", "block5", "block6", "block7"],
    ["block2", "block3", "block4", "block5", "block6", "block7"],
]
stage_lrs = [1e-4, 5e-5, 3e-5]

# Stage 1 checkpoint
stage1_path = "models/effb4_eyespacs2015_ordinal_v4_stage1.keras"

if os.path.exists(stage1_path):
    print(f"Stage 1 checkpoint found: {stage1_path}, loading...")
    model = tf.keras.models.load_model(stage1_path, compile=False)
else:
    print("\n=== Stage 1: Train head (backbone frozen) ===")
    history1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(stage1_path, monitor="val_loss", save_best_only=True)
        ],
        verbose=1
    )
    model.save(stage1_path)

# Loop qua stage 2–8
for i, (blocks, lr) in enumerate(zip(stage_blocks, stage_lrs), start=2):
    save_path = f"models/effb4_eyespacs2015_ordinal_v4_stage{i}.keras"
    if os.path.exists(save_path):
        print(f"Stage {i} checkpoint found: {save_path}, loading...")
        model = tf.keras.models.load_model(save_path, compile=False)
        continue

    model = tf.keras.models.load_model(f"models/effb4_eyespacs2015_ordinal_v4_stage{i-1}.keras", compile=False)
    run_finetune_stage(model, i, blocks, lr, save_path)

print("\n=== Training pipeline completed ===")

# %%