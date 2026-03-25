import logging
import os
import sys
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from aware.service import embed_watermark, detect_watermark
from aware.utils.models import load


logger = logging.getLogger(__name__)

app = FastAPI()

# Load AWARE models
try:
    logger.info("Loading AWARE models...")
    embedder, detector = load()
    model = {
        "embedder": embedder,
        "detector": detector,
    }
    logger.info("AWARE models loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load AWARE models: {e}. Application cannot start.")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class EmbedRequest(BaseModel):
    audio: List[float]
    watermark_data: List[int]
    sampling_rate: int


class DetectRequest(BaseModel):
    audio: List[float]
    sampling_rate: int


@app.post("/embed")
async def embed(request: EmbedRequest):
    """Embed a watermark in an audio file using AWARE."""
    audio = np.array(request.audio)
    watermark_data = np.array(request.watermark_data, dtype=np.int32)
    sampling_rate = request.sampling_rate

    embedder = model["embedder"]


    try:
        import torch
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        logger.warning(f"Failed to count FLOPs for embedding: {e}")

    try:
        watermarked_audio = embed_watermark(
            audio,
            sampling_rate,
            watermark_data,
            embedder
        )
    except Exception as e:
        logger.error(f"Error embedding watermark: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    # Sanitize watermarked audio to ensure JSON serialization works
    # Replace NaN and Inf values with 0
    watermarked_audio = np.nan_to_num(watermarked_audio, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "watermarked_audio": watermarked_audio.tolist(),
    }


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file using AWARE."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate

    detector = model["detector"]

    try:
        import torch
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        logger.warning(f"Failed to count FLOPs for detection: {e}")

    try:
        detected_watermark, confidence = detect_watermark(
            audio,
            sampling_rate,
            detector
        )
    except Exception as e:
        logger.error(f"Error detecting watermark: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    # Sanitize detected watermark and confidence to ensure JSON serialization works
    if detected_watermark is not None:
        detected_watermark = np.nan_to_num(detected_watermark, nan=0.0, posinf=1.0, neginf=0.0)


    return {
        "watermark": detected_watermark.tolist() if detected_watermark is not None else None,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    # Use the default as a fallback if AWARE_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 9004))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting AWARE server on {host}:{app_port}")
    uvicorn.run(app, host=host, port=app_port)