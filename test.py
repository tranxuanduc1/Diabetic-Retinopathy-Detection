# %%

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import cv2
AUTOTUNE = tf.data.AUTOTUNE
from tensorflow.keras.applications.efficientnet import preprocess_input

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import random
import time
import pickle
import os, glob, cv2, numpy as np
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, f1_score,
    confusion_matrix, classification_report
)

# %%
CLASS_DIRS = ["0","1","2","3","4"]
EXT = ".jpg"                       # ảnh đuôi .jpg
METHOD = "ben_graham"              # "ben_graham" | "clahe_lab" | "clahe_green"
TARGET_SIZE = 600                  # ảnh đầu ra (h = w = 600)
APPLY_CIRCULAR_MASK = True         # che góc đen
DO_CROP_FOV = True                 # crop sát vùng võng mạc trước khi resize
MARGIN_CIRCLE = 4                  # rìa trống khi vẽ mask tròn

def crop_fov(img, thr=10):
    """Crop sát vùng võng mạc: xác định bbox của các pixel > thr (loại biên đen)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > thr
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # phòng hờ biên
    y0 = max(0, y0-2); x0 = max(0, x0-2)
    y1 = min(img.shape[0]-1, y1+2); x1 = min(img.shape[1]-1, x1+2)
    return img[y0:y1+1, x0:x1+1]

def apply_circular_mask(img, margin=MARGIN_CIRCLE):
    """Giữ vùng tròn trung tâm, che góc đen cho ảnh võng mạc."""
    h, w = img.shape[:2]
    r = min(h, w)//2 - margin
    if r <= 0:
        return img
    cx, cy = w//2, h//2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    out = cv2.bitwise_and(img, img, mask=mask)
    return out

def preprocess_ben_graham(bgr, sigma=10):
    """
    Ben Graham preprocessing: giảm non-uniform illumination.
    Công thức phổ biến: out = 4*img - 4*GaussianBlur(img, sigma) + 128
    """
    gb = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(bgr, 4.0, gb, -4.0, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_square(img, size=TARGET_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def process_one(img_bgr):
    x = img_bgr
    # 1) Crop vùng FOV (loại viền đen, cắt gọn)
    if DO_CROP_FOV:
        x = crop_fov(x, thr=10)
    # 2) Resize tạm về vuông để mask tròn đẹp
    x = resize_square(x, TARGET_SIZE)
    # 3) Chọn 1 trong các phương pháp tiền xử lý
    x = preprocess_ben_graham(x, sigma=10)
    # 4) (tuỳ chọn) Circular mask để che góc
    if APPLY_CIRCULAR_MASK:
        x = apply_circular_mask(x, margin=MARGIN_CIRCLE)
    # 5) Đảm bảo kích thước đầu ra
    x = resize_square(x, TARGET_SIZE)
    return x


# %%
TEST_DIR = "archive/augmented_resized_V2/test/"  
IMAGE_SIZE = 448       # 448 hoặc 512 cho CPU; 600 sẽ rất chậm trên CPU
BATCH_SIZE = 8
SEED = 42

# %%
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
    return image, label
def map_dual_targets(image, y_int):
    # tạo 2 nhãn: softmax (int) và ordinal (4-dim)
    return image, {
        "softmax": tf.cast(y_int, tf.int32),
        "ordinal": ordinal_encode_tf(y_int)
    }


# %%
def make_ds(data_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=["0","1","2","3","4"],   # cố định thứ tự nhãn
        color_mode="rgb",
        batch_size=16,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=False,
        seed=SEED
    )
    ds = ds.map(map_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.map(map_dual_targets, num_parallel_calls=AUTOTUNE)
    # Không dùng .cache() để tiết kiệm RAM; chỉ prefetch
    ds = ds.prefetch(AUTOTUNE)
    return ds

# %%
model=tf.keras.models.load_model('models/effb4_dualhead_stage1.keras')

# %%
img=cv2.imread("archive/augmented_resized_V2/test/4/2fde69f20585-600-FA.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=process_one(img)
img = cv2.resize(img, (448, 448))
image = tf.cast(img, tf.float32)
image = preprocess_input(image) 
plt.imshow(img)

img = np.expand_dims(img, axis=0)
img.shape

# %%
pre=model.predict(img)



# %%
print(pre[0])
print(np.argmax(pre[0][0],axis=0))

# %%
TEST_DIR = "archive/augmented_resized_V2/val"          # test/0..4
CLASS_NAMES = ["0","1","2","3","4"]

IMAGE_SIZE = 448           # khớp với lúc train (448/512)
BATCH_SIZE = 8
SEED = 42

ALPHA_FUSE = 0.3           # tỉ lệ trộn: p_final = α*softmax + (1-α)*ordinal_dist
MODEL_PATH = None  

# %%
AUTOTUNE = tf.data.AUTOTUNE

# %%
def make_test_ds(data_dir=TEST_DIR):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,              # test: không shuffle
        seed=SEED
    )
    # Chuẩn hoá giống lúc train (EfficientNet preprocess_input)
    def _map(x, y):
        x = tf.cast(x, tf.float32)
        x = preprocess_input(x)
        return x, y
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds

def cum_to_dist(cum4):
    """
    cum4: (N,4) với [p>=1, p>=2, p>=3, p>=4]
    -> (N,5) phân phối lớp [0..4]
    """
    p_ge1, p_ge2, p_ge3, p_ge4 = [cum4[:, i] for i in range(4)]
    p0 = 1.0 - p_ge1
    p1 = p_ge1 - p_ge2
    p2 = p_ge2 - p_ge3
    p3 = p_ge3 - p_ge4
    p4 = p_ge4
    P = np.stack([p0, p1, p2, p3, p4], axis=1)
    # Chặn âm do không đơn điệu tuyệt đối, rồi renorm
    P = np.clip(P, 1e-7, None)
    P = P / P.sum(axis=1, keepdims=True)
    return P

def predict_dual(model, ds, alpha=ALPHA_FUSE):
    """Trả về y_true, p_soft, p_from_ord, p_final, y_pred_*"""
    y_true = []
    for _, y in ds:
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)

    # model.predict với dataset (x,y) sẽ chỉ dùng x; đầu ra: [softmax, ordinal]
    preds = model.predict(ds, verbose=1)
    p_soft = preds[0]        # (N,5)
    cum4   = preds[1]        # (N,4)
    p_ord  = cum_to_dist(cum4)

    if alpha <= 1e-8:
        p_final = p_soft
    elif alpha >= 0.9999:
        p_final = p_ord
    else:
        p_final = alpha * p_soft + (1.0 - alpha) * p_ord
        p_final = p_final / p_final.sum(axis=1, keepdims=True)  # renorm

    y_pred_soft  = p_soft.argmax(axis=1)
    y_pred_ord   = p_ord.argmax(axis=1)
    y_pred_final = p_final.argmax(axis=1)

    return y_true, p_soft, p_ord, p_final, y_pred_soft, y_pred_ord, y_pred_final

def print_main_metrics(y_true, p_final, y_pred_final):
    """Phương pháp mình chọn: QWK + Acc + Macro-F1 (trên dự đoán hợp nhất)"""
    qwk  = cohen_kappa_score(y_true, y_pred_final, weights="quadratic")
    acc  = accuracy_score(y_true, y_pred_final)
    f1m  = f1_score(y_true, y_pred_final, average="macro")
    print("\n====== MAIN METRICS (Fused) ======")
    print(f"Quadratic Weighted Kappa : {qwk:.5f}")
    print(f"Accuracy                 : {acc:.5f}")
    print(f"Macro F1                 : {f1m:.5f}")
    print("\nClassification report (per-class):")
    print(classification_report(y_true, y_pred_final, digits=4))

def show_confusion(y_true, y_pred, normalize=False):
    """Confusion matrix với sklearn.metrics.confusion_matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4], normalize='true' if normalize else None)
    print("\n====== CONFUSION MATRIX ======")
    print("(Hàng: nhãn thật, Cột: nhãn dự đoán)")
    print(np.array2string(cm, formatter={'float_kind':lambda x: f"{x:6.3f}"} if normalize else None))

    # (Tuỳ chọn) vẽ heatmap bằng matplotlib
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix ({'normalized' if normalize else 'counts'})")
    plt.colorbar(im)
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, CLASS_NAMES)
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # Ghi số trong ô (nếu không normalize)
    if not normalize:
        for i in range(5):
            for j in range(5):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9, color="w" if cm.max()>0 and cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.show()

# %%

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 2) Tạo test dataset
ds_test = make_test_ds(TEST_DIR)

    # 3) Dự đoán
y_true, p_soft, p_ord, p_final, y_pred_soft, y_pred_ord, y_pred_final = predict_dual(model, ds_test, alpha=ALPHA_FUSE)

    # 4) ĐÁNH GIÁ CHÍNH (mình chọn): QWK + Acc + Macro-F1 (dùng dự đoán hợp nhất)
print_main_metrics(y_true, p_final, y_pred_final)

    # 5) CONFUSION MATRIX (đếm số lượng)
show_confusion(y_true, y_pred_final, normalize=False)