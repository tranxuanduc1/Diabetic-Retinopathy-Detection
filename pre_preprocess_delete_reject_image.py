# %%
import cv2,glob, math
import numpy as np
import matplotlib.pyplot as plt
import os
import albumentations as A
from tqdm import tqdm
import pandas as pd
import shutil

# %%

# ================== CẤU HÌNH ==================
ROOT_DIR = "eyepascs_2015/train"               # thư mục gốc: train/
CLASS_DIRS = ["0","1","2","3","4"]
EXT = ".jpg"                     # ảnh .jpg 600x600 theo mô tả của bạn
PER_LABEL_SHOW = 100             # số ảnh mỗi nhãn để hiển thị (nếu đủ)
SAVE_PREVIEW_DIR = None          # ví dụ "preview_quality" để copy mẫu; None = không lưu
OUTPUT_CSV = "image_quality_labels.csv"

# Ngưỡng phân loại (bạn có thể chỉnh sau khi xem kết quả)
REJECT_RULES = {
    "min_lap_var": 80.0,        # mờ nặng nếu < 80
    "min_tenengrad": 120.0,     # mờ nặng nếu < 120
    "min_contrast": 22.0,       # tương phản quá thấp
    "min_brightness": 10.0,     # quá tối 35.0
    "max_brightness": 220.0,    # quá sáng
    "max_dark_frac": 0.60,      # phần trăm điểm rất tối quá cao
    "max_white_frac": 0.25,     # phần trăm điểm rất sáng quá cao
    "min_foreground_ratio": 0.45, # viền đen nhiều/quá ít tín hiệu
}

GOOD_RULES = {
    "min_lap_var": 180.0,
    "min_tenengrad": 250.0,
    "min_contrast": 32.0,
    "min_brightness": 60.0,
    "max_brightness": 190.0,
    "max_dark_frac": 0.35,
    "max_white_frac": 0.12,
    "min_foreground_ratio": 0.65,
}

MAX_SIDE_FOR_METRIC = 1024  # tính metric trên ảnh không quá lớn để ổn định
# ==============================================

def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_for_metric(img, max_side=MAX_SIDE_FOR_METRIC):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    nh, nw = int(round(h*s)), int(round(w*s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def compute_quality_metrics(path):
    img = imread_rgb(path)
    if img is None:
        return None
    rimg = resize_for_metric(img)
    gray8 = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
    if gray8.dtype != np.uint8:
        gray8 = cv2.normalize(gray8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Độ nét
    lap = cv2.Laplacian(gray8, cv2.CV_64F, ksize=3)
    lap_var = float(lap.var())
    gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx*gx + gy*gy))

    # Sáng / tương phản
    gray32 = gray8.astype(np.float32)
    brightness = float(gray32.mean())
    contrast   = float(gray32.std())

    # Tối/quá sáng
    dark_frac  = float((gray8 < 15).mean())
    white_frac = float((gray8 > 240).mean())

    # Foreground (viền đen)
    fg_ratio   = float((gray8 > 10).mean())

    return {
        "lap_var": lap_var,
        "tenengrad": tenengrad,
        "brightness": brightness,
        "contrast": contrast,
        "dark_frac": dark_frac,
        "white_frac": white_frac,
        "foreground_ratio": fg_ratio,
        "height": rimg.shape[0],
        "width": rimg.shape[1],
    }

def rate_quality(m, reject_rules=REJECT_RULES, good_rules=GOOD_RULES):
    # 1) Reject nếu vi phạm NGHIÊM TRỌNG bất kỳ tiêu chí nào
    if (m is None or
        m["lap_var"] < reject_rules["min_lap_var"] or
        m["tenengrad"] < reject_rules["min_tenengrad"] or
        m["contrast"] < reject_rules["min_contrast"] or
        m["brightness"] < reject_rules["min_brightness"] or
        m["brightness"] > reject_rules["max_brightness"] or
        m["dark_frac"] > reject_rules["max_dark_frac"] or
        m["white_frac"] > reject_rules["max_white_frac"] or
        m["foreground_ratio"] < reject_rules["min_foreground_ratio"]):
        return "reject"

    # 2) Good nếu đạt TẤT CẢ tiêu chí “đẹp”
    if (m["lap_var"] >= good_rules["min_lap_var"] and
        m["tenengrad"] >= good_rules["min_tenengrad"] and
        m["contrast"] >= good_rules["min_contrast"] and
        good_rules["min_brightness"] <= m["brightness"] <= good_rules["max_brightness"] and
        m["dark_frac"] <= good_rules["max_dark_frac"] and
        m["white_frac"] <= good_rules["max_white_frac"] and
        m["foreground_ratio"] >= good_rules["min_foreground_ratio"]):
        return "good"

    # 3) Còn lại là usable
    return "usable"

def collect_paths_from_classdirs(root_dir=ROOT_DIR, class_dirs=CLASS_DIRS, ext=EXT):
    paths, labels = [], []
    patterns = [f"*{ext}", f"*{ext.upper()}"]
    for c in class_dirs:
        cdir = os.path.join(root_dir, c)
        if not os.path.isdir(cdir):
            print(f"[Cảnh báo] Không thấy thư mục lớp: {cdir}")
            continue
        found = []
        for pat in patterns:
            found += glob.glob(os.path.join(cdir, pat))
        if not found:
            print(f"[Cảnh báo] Thư mục {cdir} không có ảnh {ext}.")
            continue
        paths.extend(sorted(found))
        labels.extend([c]*len(found))
    return paths, labels

def score_and_label(paths, labels):
    rows = []
    for p, lab in tqdm(list(zip(paths, labels)), desc="Scoring"):
        m = compute_quality_metrics(p)
        row = {"path": p, "label": lab}
        if m is None:
            row.update({k: np.nan for k in
                ["lap_var","tenengrad","brightness","contrast","dark_frac","white_frac","foreground_ratio","height","width"]
            })
            row["quality"] = "reject"
        else:
            row.update(m)
            row["quality"] = rate_quality(m)
        rows.append(row)
    return pd.DataFrame(rows)

def show_samples_for_quality(df, quality="reject", n=PER_LABEL_SHOW, save_dir=SAVE_PREVIEW_DIR):
    sub = df[df["quality"] == quality]
    if len(sub) == 0:
        print(f"[{quality}] Không có ảnh.")
        return
    k = min(n, len(sub))
    sample = sub.sample(k, random_state=42)

    # vẽ lưới ảnh
    cols = 10 if k >= 50 else 5
    rows = int(math.ceil(k / cols))
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, (_, r) in enumerate(sample.iterrows()):
        img = imread_rgb(r["path"])
        if img is None: continue
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        title = f"{quality} | {os.path.basename(r['path'])}"
        t2 = f"\nLAP={r['lap_var']:.0f} TEN={r['tenengrad']:.0f} C={r['contrast']:.1f} B={r['brightness']:.0f}"
        plt.title(title + t2, fontsize=7)
        plt.axis('off')

        # Lưu mẫu (tùy chọn)
        if save_dir:
            out_dir = os.path.join(save_dir, quality)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(r["path"])),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.tight_layout()
    plt.show()



# %%
def print_summary(df):
    print("\n===== TỔNG QUAN METRIC =====")
    cols = ["lap_var","tenengrad","brightness","contrast","dark_frac","white_frac","foreground_ratio"]
    print(df[cols].describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).round(2))
    print("\n===== PHÂN BỐ NHÃN CHẤT LƯỢNG =====")
    print(df["quality"].value_counts())

if __name__ == "__main__":
    paths, labels = collect_paths_from_classdirs(ROOT_DIR, CLASS_DIRS, EXT)
    print(f"Found {len(paths)} images under {ROOT_DIR}.")
    if len(paths) == 0:
        raise SystemExit("Không tìm thấy ảnh. Kiểm tra ROOT_DIR / EXT / tên thư mục lớp.")

    df = score_and_label(paths, labels)
    print_summary(df)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã lưu nhãn chất lượng vào: {OUTPUT_CSV}")

    # Hiển thị mẫu ~100 ảnh mỗi nhãn (nếu đủ)
    for q in ["reject", "usable", "good"]:
        print(f"\n=== XEM NHANH: {q.upper()} ===")
        show_samples_for_quality(df, quality=q, n=PER_LABEL_SHOW, save_dir=SAVE_PREVIEW_DIR)

# %%
pd.read_csv( "image_quality_labels.csv")['quality'].value_counts()


# %%
# Thống kê số lượng theo từng label × quality
label_quality_stats = (
    df.groupby(['label', 'quality'])
      .size()
      .unstack(fill_value=0)                    # cột: reject/usable/good
      .reindex(columns=['reject','usable','good'], fill_value=0)
)

# Sắp xếp label theo số (nếu label là chuỗi '0'..'4')
try:
    label_quality_stats = label_quality_stats.sort_index(key=lambda s: s.astype(int))
except Exception:
    label_quality_stats = label_quality_stats.sort_index()

# Thêm tổng theo từng label và tổng toàn cục
label_quality_stats['total'] = label_quality_stats.sum(axis=1)
label_quality_stats.loc['TOTAL'] = label_quality_stats.sum()

print(label_quality_stats)
label_quality_stats.to_csv('label_quality_stats.csv', encoding='utf-8-sig')

# %%
# =========xoas anh reject ==============
CSV_PATH = "image_quality_labels.csv"   # đổi nếu tên/đường dẫn khác

df = pd.read_csv(CSV_PATH)

# Lấy danh sách đường dẫn của ảnh bị đánh nhãn reject
reject_paths = df[df["quality"].str.lower().eq("reject")]["path"].astype(str).tolist()

print(f"Found {len(reject_paths)} reject files in CSV.")

deleted = 0
missing = 0
errors = 0

for p in reject_paths:
    try:
        if os.path.exists(p):
            os.remove(p)
            deleted += 1
        else:
            missing += 1
    except Exception as e:
        errors += 1
        print(f"[LỖI] {p}: {e}")

print(f"ĐÃ XÓA: {deleted}  |  KHÔNG TỒN TẠI: {missing}  |  LỖI: {errors}")

# %%
!ipynb-py-convert pre_preprocess_delete_reject_image.ipynb pre_preprocess_delete_reject_image.py