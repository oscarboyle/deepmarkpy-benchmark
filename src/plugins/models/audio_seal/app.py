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
# Add this near the top
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update the model dictionary
model = {
    "generator": AudioSeal.load_generator("audioseal_wm_16bits").to(device),
    "detector": AudioSeal.load_detector("audioseal_detector_16bits").to(device),
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

class WatermarkRequest(BaseModel):
    audio: List[float]
    watermark_data: List[int]
    sampling_rate: int
    mode: str = "native"  # "native" | "banded"

def _match_len(y: np.ndarray, n: int) -> np.ndarray:
    """resample_poly returns ceil(n*up/down); force exact length."""
    if len(y) > n:
        return y[:n]
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y

def _raw_watermark(audio: np.ndarray, msg: torch.Tensor) -> np.ndarray:
    """Run the generator, return the residual at the rate it was fed."""
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        wm = model["generator"].get_watermark(wav, message=msg)
    return wm.squeeze().cpu().numpy().astype(np.float64)

    
@app.post("/embed")
async def embed(request: EmbedRequest):
    """Embed a watermark in an audio file."""
    audio = np.array(request.audio)
    watermark_data = np.array(request.watermark_data)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    generator = model["generator"]
    wav = torch.tensor(audio, dtype=torch.float32)
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    msg = torch.from_numpy(watermark_data).unsqueeze(0).to(device)

    watermark = generator.get_watermark(
        wav, message=msg, sample_rate=config["sampling_rate"]
    )

    watermarked_audio = wav + watermark
    watermarked_audio = watermarked_audio.cpu().detach().numpy()
    watermarked_audio = np.squeeze(watermarked_audio)

    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(watermarked_audio, config["sampling_rate"], sampling_rate)

    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    # AudioSeal requires minimum audio length for the neural network
    # Kernel size is 7, but due to architecture we need more samples
    min_samples = 1000  # Safe minimum for AudioSeal
    if len(audio) < min_samples:
        logger.warning(f"Audio too short for detection ({len(audio)} samples), returning empty result")
        return {"watermark": [], "confidence": 0.0}

    detector = model["detector"]
    watermarked_audio = np.expand_dims(audio, axis=[0, 1])
    watermarked_audio = torch.tensor(watermarked_audio, dtype=torch.float32).to(device)

    try:
        confidence, message = detector.detect_watermark(watermarked_audio, sampling_rate)
    except RuntimeError as e:
        logger.error(f"Detection failed: {e}")
        return {"watermark": [], "confidence": 0.0}

    message = message.squeeze().cpu().numpy()
    return {"watermark": message if message is None else message.tolist(),
            "confidence": float(confidence)}


@app.post("/watermark")
async def watermark(request: WatermarkRequest):
    """Return the raw additive watermark residual, at the CALLER's sample rate.
 
    mode="native"  -> feed the model the audio as-is. Learned filters have a
                      fixed impulse response in samples, so at 44.1 kHz the
                      watermark's spectral content shifts up by 44100/16000.
    mode="banded"  -> downsample, watermark, upsample the residual. The
                      anti-aliasing filter confines the watermark to 0-8 kHz.
    """
    audio = np.nan_to_num(np.asarray(request.audio, dtype=np.float64),
                          nan=0.0, posinf=1.0, neginf=-1.0)
    sr = request.sampling_rate
    msg = torch.from_numpy(np.asarray(request.watermark_data)).unsqueeze(0).to(device)
 
    if request.mode == "banded" and sr != MODEL_SR:
        g = np.gcd(sr, MODEL_SR)
        up, down = sr // g, MODEL_SR // g          # 44100/16000 -> 441/160
        wm16 = _raw_watermark(resample_poly(audio, down, up), msg)
        wm = _match_len(resample_poly(wm16, up, down), len(audio))
    elif request.mode == "banded":
        wm = _raw_watermark(audio, msg)            # already at model rate
    elif request.mode == "native":
        wm = _raw_watermark(audio, msg)
    else:
        raise ValueError(f"unknown mode {request.mode!r}")
 
    return {"watermark_signal": wm.tolist(), "sampling_rate": sr, "mode": request.mode}
 


if __name__ == "__main__":
    # Use the default as a fallback if APP_PORT is not set in the environment
    app_port = int(os.getenv("APP_PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host=host, port=app_port)