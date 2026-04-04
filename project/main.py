import logging
import os
from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Optional

from pipeline import preprocess_for_model, run_ensemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "output_three_models")
)

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models from %s ...", MODELS_DIR)
    models["low_high"]   = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "effb3_low_high_stage4.keras")
    )
    models["zero_one"]   = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "effb3_0_vs_1_stage4.keras")
    )
    models["ordinal234"] = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "effb3_234_ordinal2bit_stage4.keras")
    )
    logger.info("All 3 models loaded.")
    yield
    models.clear()


app = FastAPI(title="Diabetic Retinopathy Detection API", lifespan=lifespan)


class RawOutputs(BaseModel):
    p_m1_high: float
    p_m2_one: Optional[float] = None
    p_ge3: Optional[float] = None
    p_ge4: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    route: str
    raw_outputs: RawOutputs


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models) == 3}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got content-type: {file.content_type}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        batch = preprocess_for_model(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=422, detail="Image preprocessing failed")

    try:
        result = run_ensemble(models, batch)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed")

    return PredictionResponse(
        prediction=result["prediction"],
        label=result["label"],
        route=result["route"],
        raw_outputs=RawOutputs(**result["raw_outputs"]),
    )
