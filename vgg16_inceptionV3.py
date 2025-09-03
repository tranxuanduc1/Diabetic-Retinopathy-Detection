# %%
import os, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16, inception_v3

# ================== CONFIG ==================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

TRAIN_DIR = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/train_preprocess"
VAL_DIR   = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/val_preprocess"
OUT_DIR   = "outputs_hybrid"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 299
BATCH_SIZE = 16
HEAD_EPOCHS = 10
FT_EPOCHS   = 15

# ================== DATASET ==================
def load_ds(directory, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=["0","1","2","3","4"],
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )
    return ds.prefetch(tf.data.AUTOTUNE)

# ================== MODEL ==================
def build_hybrid_model(img_size=299, num_classes=5, dropout=0.5):
    inp = layers.Input(shape=(img_size, img_size, 3), name="input_img")

    # ----- VGG16 branch -----
    vgg_base = vgg16.VGG16(include_top=False, weights="imagenet",
                           input_shape=(img_size, img_size, 3))
    vgg_base.trainable = False
    vgg_feat = vgg_base(inp)
    vgg_feat = layers.GlobalAveragePooling2D(name="vgg_gap")(vgg_feat)

    # ----- InceptionV3 branch -----
    inc_base = inception_v3.InceptionV3(include_top=False, weights="imagenet",
                                        input_shape=(img_size, img_size, 3))
    inc_base.trainable = False
    inc_feat = inc_base(inp)
    inc_feat = layers.GlobalAveragePooling2D(name="inc_gap")(inc_feat)

    # ----- Concatenate features -----
    merged = layers.Concatenate(name="concat_feats")([vgg_feat, inc_feat])
    x = layers.Dropout(dropout)(merged)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inp, out, name="Hybrid_VGG16_InceptionV3")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ================== TRAIN HEAD ==================
def train_head():
    ds_train = load_ds(TRAIN_DIR, shuffle=True)
    ds_val   = load_ds(VAL_DIR, shuffle=False)

    model = build_hybrid_model(img_size=IMG_SIZE, num_classes=5)

    cbs = [
        keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "hybrid_head.keras"),
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=HEAD_EPOCHS, callbacks=cbs)
    model.save(os.path.join(OUT_DIR, "hybrid_head.keras"))
    return model

# ================== FINE-TUNING ==================
def unfreeze_backbones(model, n_last_vgg=4, n_last_inc=30, lr=1e-4):
    """Mở một số layer cuối cùng của backbone để fine-tune"""
    for layer in model.layers:
        layer.trainable = False  # mặc định freeze

    # VGG16
    vgg_base = None
    inc_base = None
    for l in model.layers:
        if isinstance(l, keras.Model) and l.name.startswith("vgg16"):
            vgg_base = l
        if isinstance(l, keras.Model) and l.name.startswith("inception_v3"):
            inc_base = l

    if vgg_base:
        for l in vgg_base.layers[-n_last_vgg:]:
            if not isinstance(l, layers.BatchNormalization):
                l.trainable = True
    if inc_base:
        for l in inc_base.layers[-n_last_inc:]:
            if not isinstance(l, layers.BatchNormalization):
                l.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def fine_tune():
    ds_train = load_ds(TRAIN_DIR, shuffle=True)
    ds_val   = load_ds(VAL_DIR, shuffle=False)

    print(">> Load lại model head")
    model = keras.models.load_model(os.path.join(OUT_DIR, "hybrid_head.keras"))

    model = unfreeze_backbones(model, n_last_vgg=4, n_last_inc=30, lr=1e-4)

    cbs = [
        keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "hybrid_finetune.keras"),
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=FT_EPOCHS, callbacks=cbs)
    model.save(os.path.join(OUT_DIR, "hybrid_final.keras"))
    return model

# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(os.path.join(OUT_DIR, "hybrid_head.keras")):
        train_head()
        tf.keras.backend.clear_session()

    fine_tune()
    print("✅ Huấn luyện xong. Model cuối cùng: hybrid_final.keras")


# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow import keras

# ================== LOAD TEST DATA ==================
TEST_DIR = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/test_preprocess"

def load_test_ds(directory):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=["0","1","2","3","4"],
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return ds

# ================== EVALUATE ==================
def evaluate_model(model_path, test_dir):
    print(f">> Load model từ {model_path}")
    model = keras.models.load_model(model_path)

    ds_test = load_test_ds(test_dir)

    # Lấy nhãn thật và dự đoán
    y_true = np.concatenate([y.numpy() for x, y in ds_test], axis=0)
    y_pred_prob = model.predict(ds_test, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Tính metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    f1   = f1_score(y_true, y_pred, average="macro")

    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0","1","2","3","4"],
                yticklabels=["0","1","2","3","4"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Hybrid Model")
    plt.show()

# ================== RUN ==================
if __name__ == "__main__":
    evaluate_model("outputs_hybrid/hybrid_final.keras", TEST_DIR)


# %%


# %%


# %%


# %%


# %%
!ipynb-py-convert structure_model_vgg16_inceptionV3.ipynb vgg16_inceptionV3.py