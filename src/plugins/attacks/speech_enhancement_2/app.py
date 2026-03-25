import logging
import os
import sys
import tempfile
from typing import List
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from clearvoice import ClearVoice
from utils.utils import load_config
import soundfile as sf

logger = logging.getLogger(__name__)
app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    model_name: str


@app.post("/attack")
async def attack(request: AttackRequest):
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)
    
    # Create a temporary file with .wav extension (not .mp3)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Add noise before saving
        noise_strength = config.get("noise_strength_se2", 0.0)
        if noise_strength > 0:
            noisy = audio + noise_strength * np.random.normal(0, 1, size=(len(audio)))
            audio = noisy

        # Save the audio array to the temporary WAV file
        logger.info(f"Saving audio to: {tmp_path} with sampling rate: {sampling_rate}")
        sf.write(tmp_path, audio, sampling_rate)
        logger.info(f"Audio saved successfully")
        
        # Pass the temporary file path to ClearVoice
        logger.info(f"Processing with ClearVoice model: {request.model_name}")
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=[request.model_name])
        audio_cv = myClearVoice(input_path=tmp_path, online_write=False)
        
        logger.info(f"Successfully processed audio with {request.model_name}")
        return {"audio": audio_cv.tolist()}
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return {"error": str(e), "audio": None}
    
    finally:
        # Delete the temporary file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info(f"Deleted temporary file: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {tmp_path}: {str(e)}")
