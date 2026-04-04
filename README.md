# Diabetic Retinopathy Detection

A FastAPI-based REST API that classifies diabetic retinopathy severity from retinal fundus images using a hierarchical ensemble of three EfficientNet-B3 models.

## Grading Scale

| Grade | Label              |
|-------|--------------------|
| 0     | No DR              |
| 1     | Mild DR            |
| 2     | Moderate DR        |
| 3     | Severe DR          |
| 4     | Proliferative DR   |

## Setup

### Requirements

- Python 3.11
- Trained model files in `output_three_models/`

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start the server

```bash
cd project
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## API Endpoints

### GET `/health`

Check that the server is running and all three models are loaded.

**Response**

```json
{
  "status": "ok",
  "models_loaded": true
}
```

---

### POST `/predict`

Classify a retinal fundus image.

**Request**

- Content-Type: `multipart/form-data`
- Field: `file` — an image file (JPEG, PNG, etc.)

**Response**

```json
{
  "prediction": 2,
  "label": "Moderate DR",
  "route": "m1_high to m3",
  "raw_outputs": {
    "p_m1_high": 0.812,
    "p_ge3": 0.341,
    "p_ge4": 0.097
  }
}
```

| Field           | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| `prediction`    | Predicted DR grade (0–4)                                           |
| `label`         | Human-readable label for the grade                                 |
| `route`         | Which ensemble branch was taken (`m1_low to m2` or `m1_high to m3`) |
| `raw_outputs`   | Raw model probabilities (fields vary by route, see below)          |

**`raw_outputs` fields by route**

| Route          | Fields present                              |
|----------------|---------------------------------------------|
| `m1_low to m2` | `p_m1_high`, `p_m2_one`                     |
| `m1_high to m3`| `p_m1_high`, `p_ge3`, `p_ge4`              |

- `p_m1_high` — probability that severity is high (grade ≥ 2)
- `p_m2_one`  — probability that grade is 1 (vs. 0), when low branch taken
- `p_ge3`     — probability that grade ≥ 3, when high branch taken
- `p_ge4`     — probability that grade = 4, when high branch taken

**Error responses**

| Status | Meaning                               |
|--------|---------------------------------------|
| 400    | File is not an image or is empty      |
| 422    | Image could not be preprocessed       |
| 500    | Model inference failed                |

---

## Usage Examples

### curl

```bash
# Health check
curl http://localhost:8000/health

# Predict from a file
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/retinal_image.jpg"
```

### Python (requests)

```python
import requests

# Health check
resp = requests.get("http://localhost:8000/health")
print(resp.json())

# Predict
with open("retinal_image.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("retinal_image.jpg", f, "image/jpeg")},
    )

result = resp.json()
print(f"Grade: {result['prediction']} — {result['label']}")
print(f"Route: {result['route']}")
print(f"Raw outputs: {result['raw_outputs']}")
```

### JavaScript (fetch)

```js
const form = new FormData();
form.append("file", fileInput.files[0]);

const resp = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: form,
});

const result = await resp.json();
console.log(result.prediction, result.label);
```

---

## Model Architecture

The ensemble uses a three-stage hierarchical decision tree:

```
Input image
    |
    M1 (low_high)
    |-- p < 0.5  --> M2 (zero_one)  --> Grade 0 or 1
    |-- p >= 0.5 --> M3 (ordinal234) --> Grade 2, 3, or 4
```

All three models are EfficientNet-B3 trained at 300×300 pixels with Ben Graham illumination preprocessing.
