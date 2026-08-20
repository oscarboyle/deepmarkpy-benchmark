import logging
import os
import sys
from typing import List

import numpy as np
import torch
import uvicorn
from audioseal import AudioSeal
from fastapi import FastAPI
from pydantic import BaseModel

from utils.utils import load_config, resample_audio

logger = logging.getLogger(__name__)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# .eval() is a no-op for the current SEANet checkpoints (conv + weight-norm,
# no dropout or batchnorm), but it costs nothing and a future checkpoint that
# did carry them would otherwise produce silently non-deterministic output.
model = {
    "generator": AudioSeal.load_generator("audioseal_wm_16bits").to(device).eval(),
    "detector": AudioSeal.load_detector("audioseal_detector_16bits").to(device).eval(),
}

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)


class EmbedRequest(BaseModel):
    audio: List[float]
    watermark_data: List[int]
    sampling_rate: int


class DetectRequest(BaseModel):
    audio: List[float]
    sampling_rate: int


# Handlers are sync `def`, not `async def`: the forward pass blocks, and a
# blocking call in an async handler stalls the whole event loop. FastAPI runs
# sync handlers in a threadpool instead.
@app.post("/embed")
def embed(request: EmbedRequest):
    """Embed a watermark in an audio file."""
    audio = np.array(request.audio)
    watermark_data = np.array(request.watermark_data)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        # Feeds the raw request list rather than `audio`; kept as-is because
        # the golden fixtures encode this path.
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    generator = model["generator"]
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    msg = torch.from_numpy(watermark_data).unsqueeze(0).to(device)

    # Without no_grad the encoder/decoder activations are retained for a
    # backward pass that never comes -- hundreds of MB per request.
    with torch.no_grad():
        watermark = generator.get_watermark(wav, message=msg)
        watermarked_audio = wav + watermark

    watermarked_audio = np.squeeze(watermarked_audio.cpu().numpy())

    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(
            watermarked_audio, config["sampling_rate"], sampling_rate
        )

    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    # AudioSeal requires minimum audio length for the neural network
    # Kernel size is 7, but due to architecture we need more samples
    min_samples = 1000  # Safe minimum for AudioSeal
    if len(audio) < min_samples:
        logger.warning(
            f"Audio too short for detection ({len(audio)} samples), returning empty result"
        )
        return {"watermark": [], "confidence": 0.0}

    detector = model["detector"]
    watermarked_audio = np.expand_dims(audio, axis=[0, 1])
    watermarked_audio = torch.tensor(watermarked_audio, dtype=torch.float32).to(device)

    try:
        with torch.no_grad():
            confidence, message = detector.detect_watermark(watermarked_audio)
    except RuntimeError as e:
        logger.error(f"Detection failed: {e}")
        return {"watermark": [], "confidence": 0.0}

    message = message.squeeze().cpu().numpy()
    return {"watermark": message.tolist(), "confidence": float(confidence)}


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)