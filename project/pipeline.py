import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input

IMAGE_SIZE = 300
PREPROCESS_SIZE = 512


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """Decode raw image bytes to RGB uint8 NumPy array (H, W, 3)."""
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except UnidentifiedImageError:
        raise ValueError("Cannot decode image: unrecognized format")
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")


def crop_fov(img: np.ndarray, thr: int = 10) -> np.ndarray:
    """Remove black border around retinal FOV. Input: RGB uint8."""
    mask = np.any(img > thr, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    margin = 2
    r0 = max(0, r0 - margin)
    r1 = min(img.shape[0], r1 + margin + 1)
    c0 = max(0, c0 - margin)
    c1 = min(img.shape[1], c1 + margin + 1)
    return img[r0:r1, c0:c1]


def ben_graham(img: np.ndarray) -> np.ndarray:
    """Ben Graham illumination correction. Input/output: BGR uint8."""
    gb = cv2.GaussianBlur(img, (0, 0), sigmaX=10, sigmaY=10)
    out = cv2.addWeighted(img, 4.0, gb, -4.0, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_circular_mask(img: np.ndarray, margin: int = 4) -> np.ndarray:
    """Zero out image corners, keeping only the central circular FOV. Input: BGR uint8."""
    h, w = img.shape[:2]
    radius = min(h, w) // 2 - margin
    if radius <= 0:
        return img
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)


def preprocess_for_model(raw: bytes) -> np.ndarray:
    """
    Full preprocessing pipeline for one image.

    Steps:
      decode (RGB) -> crop_fov (RGB) -> resize 512 (RGB)
      -> RGB->BGR -> ben_graham (BGR) -> circular_mask (BGR)
      -> resize 300 (BGR) -> expand_dims -> preprocess_input

    Returns float32 array of shape (1, 300, 300, 3).
    """
    img = decode_image_bytes(raw)                                # RGB uint8
    img = crop_fov(img)                                          # RGB uint8
    img = cv2.resize(img, (PREPROCESS_SIZE, PREPROCESS_SIZE),
                     interpolation=cv2.INTER_CUBIC)              # RGB uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)                   # BGR uint8
    img = ben_graham(img)                                        # BGR uint8
    img = apply_circular_mask(img)                               # BGR uint8
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                     interpolation=cv2.INTER_CUBIC)              # BGR uint8
    batch = np.expand_dims(img.astype(np.float32), axis=0)       # (1,300,300,3)
    batch = preprocess_input(batch)                              # float32 normalized
    return batch


LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

TH_BIN = 0.5


def run_ensemble(models: dict, batch: np.ndarray) -> dict:
    """
    Run hierarchical ensemble prediction.

    models keys: 'low_high', 'zero_one', 'ordinal234'
    batch: float32 (1, 300, 300, 3)

    Workflow:
      M1 (low/high?) → low  → M2 → 0 or 1
                     → high → M3 → 2, 3, or 4

    Returns dict with: prediction, label, route, raw_outputs
    """
    p_m1_high = float(models["low_high"].predict(batch, verbose=0)[0, 0])

    if p_m1_high < TH_BIN:
        # Low severity branch → M2 distinguishes 0 vs 1
        p_m2_one = float(models["zero_one"].predict(batch, verbose=0)[0, 0])
        prediction = 1 if p_m2_one >= TH_BIN else 0
        route = "m1_low to m2"
        raw = {"p_m1_high": p_m1_high, "p_m2_one": p_m2_one}
    else:
        # High severity branch → M3 distinguishes 2/3/4
        m3_out = models["ordinal234"].predict(batch, verbose=0)[0]
        p_ge3, p_ge4 = float(m3_out[0]), float(m3_out[1])
        p2 = 1.0 - p_ge3
        p3 = p_ge3 - p_ge4
        p4 = p_ge4
        P = np.clip([p2, p3, p4], 1e-9, None)
        P = P / P.sum()
        prediction = int(np.argmax(P)) + 2  # {2, 3, 4}
        route = "m1_high to m3"
        raw = {"p_m1_high": p_m1_high, "p_ge3": p_ge3, "p_ge4": p_ge4}

    return {
        "prediction": prediction,
        "label": LABELS[prediction],
        "route": route,
        "raw_outputs": raw,
    }
