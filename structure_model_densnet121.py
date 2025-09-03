# %%
import os, json, random, math, pickle,cv2
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import AUTOTUNE
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
from skimage.color import rgb2gray

# %%
!pip show tensorflow keras

# %%
try:
    import cv2
except Exception:
    cv2 = None
try:
    from skimage.color import rgb2gray
    from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
except Exception:
    rgb2gray = greycomatrix = greycoprops = local_binary_pattern = None

# %%
# ================== CONFIG ==================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

TRAIN_DIR = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/train_preprocess"   # <--- chỉnh nếu khác
VAL_DIR   = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/val_preprocess"   

IMG_SIZE = 224        # DenseNet chuẩn 224
BATCH_SIZE = 16
HEAD_EPOCHS = 12      # train head với backbone đóng băng
FT_EPOCHS   = 12      # fine-tune backbone
AUGMENT = False       # bạn đã augment offline -> để False

OUT_DIR = "outputs_msedensenet"
os.makedirs(OUT_DIR, exist_ok=True)


# %%
# ================== DATA ==================
def make_augmenter():
    if not AUGMENT:
        # identity để tránh lỗi Sequential rỗng
        return keras.Sequential([layers.Lambda(lambda x: x, name="identity")], name="augmenter")
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.05),
    ], name="augmenter")

def load_ds_5cls(directory, shuffle):
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
    return ds.prefetch(AUTOTUNE)

# %%
# ================== HAND-CRAFTED FEATURES ==================
def _rgb_hsv_stats(img_u8):
    # img_u8: HxWx3 uint8
    # RGB mean/std
    means = img_u8.reshape(-1,3).mean(axis=0)
    stds  = img_u8.reshape(-1,3).std(axis=0) + 1e-6
    # HSV
    if cv2 is not None:
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        means_hsv = hsv.reshape(-1,3).mean(axis=0)
        stds_hsv  = hsv.reshape(-1,3).std(axis=0) + 1e-6
    else:
        # fallback nếu thiếu cv2
        means_hsv = np.zeros(3); stds_hsv = np.zeros(3)
    return np.concatenate([means, stds, means_hsv, stds_hsv], axis=0).astype(np.float32)

def _green_hist(img_u8, bins=16):
    g = img_u8[:,:,1]
    hist, _ = np.histogram(g, bins=bins, range=(0,255), density=True)
    return hist.astype(np.float32)

def _glcm_feats(img_u8, levels=32):
    if greycomatrix is None or greycoprops is None:
        return np.zeros(6, dtype=np.float32)  # contrast, dissimilarity, homogeneity, energy, correlation, ASM
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) if cv2 is not None else (rgb2gray(img_u8)*255).astype(np.uint8)
    # quantize to 'levels'
    q = (gray.astype(np.float32) * (levels/256.0)).astype(np.uint8)
    # distances & angles
    dists = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(q, distances=dists, angles=angles, levels=levels, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        v = greycoprops(glcm, prop)  # shape (len(dists), len(angles))
        feats.append(v.mean())
    return np.array(feats, dtype=np.float32)

def _lbp_hist(img_u8, P=8, R=1.0):
    if local_binary_pattern is None:
        return np.zeros(P+2, dtype=np.float32)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) if cv2 is not None else (rgb2gray(img_u8)*255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2), density=True)
    return hist.astype(np.float32)  # length P+2

def handcrafted_features_numpy(img_np):
    """
    img_np: float32 (H,W,3) in [0,255] (from tf.dataset), convert to uint8 for features.
    Returns 1D float32 vector.
    """
    if img_np.dtype != np.uint8:
        img_u8 = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_np
    f1 = _rgb_hsv_stats(img_u8)           # 12
    f2 = _green_hist(img_u8, bins=16)     # 16
    f3 = _glcm_feats(img_u8, levels=32)   # 6
    f4 = _lbp_hist(img_u8, P=8, R=1.0)    # 10
    feats = np.concatenate([f1, f2, f3, f4], axis=0)  # total 44
    return feats.astype(np.float32)

# Tính kích thước đặc trưng một lần để set_shape cho tf.data
_FEATURE_DIM = len(handcrafted_features_numpy(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)))

def add_handcrafted_to_batch(x, y):
    """
    x: (B,H,W,3) float32 [0,255] ; y: (B,)
    return: ((x, feats), y)  với feats shape (B, _FEATURE_DIM)
    """
    def _batch_feats(x_batch):
        x_np = x_batch.numpy()  # (B,H,W,3)
        feats = [handcrafted_features_numpy(img) for img in x_np]
        return np.stack(feats, axis=0).astype(np.float32)

    feats = tf.py_function(func=_batch_feats, inp=[x], Tout=tf.float32)
    feats.set_shape([None, _FEATURE_DIM])
    return (x, feats), y

# %%
# ================== MODEL ==================
def build_msedensenet_model():
    # Input 1: ảnh
    inp_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
    x = make_augmenter()(inp_img)
    x = layers.Rescaling(1./255)(x)
    base = DenseNet121(include_top=False, weights="imagenet", input_tensor=x, pooling="avg", name="densenet121")
    base.trainable = False
    feat_cnn = base.output                      # (None, 1024)

    # Head CNN nhỏ
    h = layers.Dropout(0.4)(feat_cnn)
    h = layers.Dense(512, activation="relu")(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)

    # Input 2: đặc trưng thủ công
    inp_feat = layers.Input(shape=(_FEATURE_DIM,), name="handcrafted_input")
    f = layers.BatchNormalization()(inp_feat)
    f = layers.Dense(128, activation="relu")(f)
    f = layers.Dropout(0.3)(f)

    # Hợp nhất
    z = layers.Concatenate()([h, f])
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(5, activation="softmax", name="pred")(z)

    model = keras.Model([inp_img, inp_feat], out, name="MSE_DenseNet121_Handcrafted")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def unfreeze_backbone(model, n_last=100, lr=1e-4):
   # lấy tất cả layer conv2, conv3, conv4, conv5 bên trong backbone
    backbone_layers = [l for l in model.layers if l.name.startswith(("conv", "pool", "bn", "relu", "avg_pool"))]

    # chỉ mở n_last layer cuối
    for l in backbone_layers[:-n_last]:
        l.trainable = False
    for l in backbone_layers[-n_last:]:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
        else:
            l.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

# %%
# ================== EVAL HELPERS ==================
def collect_predictions(model, ds):
    y_true_all, p_all = [], []
    for (x_img, x_feat), y in ds:
        p = model.predict([x_img, x_feat], verbose=0)
        p_all.append(p)
        y_true_all.append(y.numpy())
    return np.concatenate(y_true_all), np.concatenate(p_all, axis=0)

def report_metrics(y_true, p, title="VAL"):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    y_pred = np.argmax(p, axis=1)
    print(f"\n== {title} ==")
    print("Accuracy:", f"{accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

# %%
# ================== MAIN ==================
def main():
    print("== Tạo dataset gốc ==")
    ds_train_raw = load_ds_5cls(TRAIN_DIR, shuffle=True)
    ds_val_raw   = load_ds_5cls(VAL_DIR, shuffle=False)

    # Gắn đặc trưng thủ công vào mỗi batch
    print(f"Feature dim = {_FEATURE_DIM}")
    ds_train = ds_train_raw.map(add_handcrafted_to_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    ds_val   = ds_val_raw.map(add_handcrafted_to_batch,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # ====== Train head ======
    head_path = os.path.join(OUT_DIR, "head_best.keras")

    if os.path.exists(head_path):
        print("\n=== Load model từ head_best.keras ===")
        model = keras.models.load_model(head_path, compile=False,safe_mode=False)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    else:
        print("\n=== Huấn luyện head (freeze backbone) ===")
        model = build_msedensenet_model()
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(head_path, monitor="val_loss", save_best_only=True)
        ]
        model.fit(ds_train, validation_data=ds_val, epochs=HEAD_EPOCHS, callbacks=cbs)


    # ====== Fine-tune backbone ======
    print("\n=== Fine-tune DenseNet121 (mở ~100 layer cuối) ===")
    unfreeze_backbone(model, n_last=100, lr=1e-4)
    cbs_ft = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "finetune_best.keras"), monitor="val_loss", save_best_only=True)
    ]
    model.fit(ds_train, validation_data=ds_val, epochs=FT_EPOCHS, callbacks=cbs_ft)

    # Lưu
    final_path = os.path.join(OUT_DIR, "msedensenet_final.keras")
    model.save(final_path)
    print("Saved:", final_path)

    # Đánh giá nhanh trên VAL
    y_val, p_val = collect_predictions(model, ds_val)
    report_metrics(y_val, p_val, title="VAL (MSE-DenseNet+Handcrafted)")

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"done": True, "feature_dim": int(_FEATURE_DIM)}, f, ensure_ascii=False, indent=2)


# %%
# ================== INFERENCE 1 ẢNH ==================
def predict_one_image(img_path, model_path=os.path.join(OUT_DIR, "msedensenet_final.keras")):
    model = keras.models.load_model(model_path, compile=False)

    # 1) Ảnh cho nhánh CNN
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x_img = tf.keras.preprocessing.image.img_to_array(img)  # float32 [0..255]
    x_img = np.expand_dims(x_img, 0).astype(np.float32)

    # 2) Đặc trưng thủ công cho nhánh thứ 2
    feats = handcrafted_features_numpy(x_img[0])  # (F,)
    x_feat = np.expand_dims(feats, 0).astype(np.float32)

    p = model.predict([x_img, x_feat], verbose=0)[0]
    y_hat = int(np.argmax(p))
    return y_hat, p.tolist()


# %%
main()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model_on_test(model_path, test_dir, img_size=224, batch_size=16):
    """
    Đánh giá model trên tập test (chứa thư mục con 0..4).
    """
    # 1) Load model
    model = keras.models.load_model(model_path, compile=False,safe_mode=False)

    # 2) Load test dataset
    ds_test_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=["0", "1", "2", "3", "4"],
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False  # quan trọng để so sánh nhãn
    )
    ds_test = ds_test_raw.map(add_handcrafted_to_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # 3) Dự đoán
    y_true, y_pred, y_prob = [], [], []
    for (x_img, x_feat), y in ds_test:
        p = model.predict([x_img, x_feat], verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(p, axis=1))
        y_prob.extend(p)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4) Báo cáo Precision, Recall, F1-score
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))

    # 5) Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["0","1","2","3","4"], 
                yticklabels=["0","1","2","3","4"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Set")
    plt.show()


# %%
evaluate_model_on_test(
    model_path="outputs_msedensenet/msedensenet_final.keras",
    test_dir="/home/duc/Documents/DoAn/aptos2019-blindness-detection/test_preprocess",
    img_size=224,
    batch_size=16
)

# %%
!ipynb-py-convert structure_model_densnet121.ipynb densnet121.py

# %%
import os, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.data import AUTOTUNE

# ================== CONFIG ==================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

TRAIN_DIR = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/train_preprocess"
VAL_DIR   = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/val_preprocess"

IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 12
FT_EPOCHS   = 12
OUT_DIR = "outputs_densenet121"
os.makedirs(OUT_DIR, exist_ok=True)


# ================== DATA ==================
def make_augmenter():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.05),
    ], name="augmenter")

def load_ds(directory, shuffle=True):
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
    return ds.prefetch(AUTOTUNE)


# ================== MODEL ==================
def build_densenet121_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
    x = make_augmenter()(inp)
    x = layers.Rescaling(1./255)(x)

    base = DenseNet121(include_top=False, weights="imagenet",
                       input_tensor=x, pooling="avg", name="densenet121")
    base.trainable = False

    h = layers.Dropout(0.4)(base.output)
    h = layers.Dense(512, activation="relu")(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(5, activation="softmax", name="pred")(h)

    model = keras.Model(inp, out, name="DenseNet121_only")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def unfreeze_backbone(model, n_last=100, lr=1e-4):
    """
    Mở n_last layer cuối của backbone DenseNet121 để fine-tune.
    Không phụ thuộc vào tên 'densenet121' nữa, mà lọc các layer conv/pool/bn.
    """
    backbone_layers = [
        l for l in model.layers 
        if l.name.startswith(("conv", "pool", "bn", "relu", "avg_pool"))
    ]

    # Đóng băng hầu hết, chỉ mở n_last layer cuối
    for l in backbone_layers[:-n_last]:
        l.trainable = False
    for l in backbone_layers[-n_last:]:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
        else:
            l.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ================== MAIN ==================
def main():
    ds_train = load_ds(TRAIN_DIR, shuffle=True)
    ds_val   = load_ds(VAL_DIR, shuffle=False)

    # ====== Train head ======
    model = build_densenet121_model()
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "head_best.keras"), monitor="val_accuracy", save_best_only=True)
    ]
    model.fit(ds_train, validation_data=ds_val, epochs=HEAD_EPOCHS, callbacks=cbs)

    # ====== Fine-tune backbone ======
    unfreeze_backbone(model, n_last=100, lr=1e-4)
    cbs_ft = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "finetune_best.keras"), monitor="val_accuracy", save_best_only=True)
    ]
    model.fit(ds_train, validation_data=ds_val, epochs=FT_EPOCHS, callbacks=cbs_ft)

    # Lưu
    final_path = os.path.join(OUT_DIR, "densenet121_final.keras")
    model.save(final_path)
    print("Saved:", final_path)


if __name__ == "__main__":
    main()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================== EVALUATE ==================
def evaluate_model_on_test(model_path, test_dir, img_size=224, batch_size=16):
    """
    Đánh giá model trên tập test (chứa thư mục con 0..4).
    """
    # 1) Load model
    model = keras.models.load_model(model_path, compile=False)

    # 2) Load test dataset
    ds_test = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=["0", "1", "2", "3", "4"],
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    # 3) Dự đoán
    y_true = np.concatenate([y.numpy() for _, y in ds_test], axis=0)
    y_prob = model.predict(ds_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # 4) Báo cáo Precision, Recall, F1-score
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))

    # 5) Accuracy tổng thể
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")

    # 6) Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["0","1","2","3","4"], 
                yticklabels=["0","1","2","3","4"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - DenseNet121")
    plt.show()


# ================== RUN TEST EVAL ==================
if __name__ == "__main__":
    TEST_DIR = "/home/duc/Documents/DoAn/aptos2019-blindness-detection/test_preprocess"
    evaluate_model_on_test(
        model_path="outputs_densenet121/densenet121_final.keras",
        test_dir=TEST_DIR,
        img_size=224,
        batch_size=16
    )
