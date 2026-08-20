import logging
import os

import numpy as np
from scipy.signal import resample_poly

from core.base_model import BaseModel

logger = logging.getLogger(__name__)

MODEL_SR = 16000  # AudioSeal's native rate; the service only ever sees this


class AudioSeal44kModel(BaseModel):
    """AudioSeal at 44.1 kHz via band-splitting.

    The watermark is generated at 16 kHz and its residual upsampled back to
    the caller's rate, so the perturbation stays confined to 0-8 kHz and the
    original high band survives untouched. Talks to the stock AudioSeal
    service on AUDIOSEAL_PORT -- no server-side changes, no second container.
    """

    def __init__(self):
        super().__init__()

        port = os.getenv("AUDIOSEAL_PORT", "5001")

        if not port:
            logger.error("AUDIOSEAL_PORT environment variable not set and no default provided.")
            raise ValueError("AUDIOSEAL_PORT must be set")

        self.base_url = f"http://localhost:{port}"
        logger.info(f"AudioSeal44kModel initialized. Target API: {self.base_url}")

    @staticmethod
    def _ratio(sr: int):
        """Rational resampling factors between `sr` and the model rate."""
        g = np.gcd(int(sr), MODEL_SR)
        return int(sr) // g, MODEL_SR // g          # 44100 -> (441, 160)

    @staticmethod
    def _match_len(y: np.ndarray, n: int) -> np.ndarray:
        """resample_poly returns ceil(n*up/down); force exact length."""
        if len(y) > n:
            return y[:n]
        if len(y) < n:
            return np.pad(y, (0, n - len(y)))
        return y

    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """Embeds a band-limited watermark into the audio."""
        # Sanitize audio: replace NaN with 0 and clip Inf to valid float range
        x = np.nan_to_num(np.asarray(audio, dtype=np.float64),
                          nan=0.0, posinf=1.0, neginf=-1.0)

        if int(sampling_rate) == MODEL_SR:
            x_low, up, down = x, 1, 1
        else:
            up, down = self._ratio(sampling_rate)
            x_low = resample_poly(x, down, up)

        # Always tell the service 16 kHz: if we send 44100 it does its own
        # lossy resample round-trip internally and the band-split is undone.
        payload = {
            "audio": x_low.tolist(),
            "watermark_data": np.asarray(watermark_data).tolist(),
            "sampling_rate": MODEL_SR,
        }

        response_data = self._make_request(endpoint="/embed", json_data=payload, method="POST")

        if "watermarked_audio" not in response_data:
            logger.error("'/embed' response did not contain 'watermarked_audio' key.")
            raise KeyError("Missing 'watermarked_audio' in response from /embed")

        y_low = np.asarray(response_data["watermarked_audio"], dtype=np.float64)

        # Recovering the residual by subtraction only holds if the service
        # returned a sample-aligned signal. A silent off-by-one here would
        # corrupt every watermark without raising anything.
        if len(y_low) != len(x_low):
            raise RuntimeError(
                f"/embed changed length: sent {len(x_low)}, got {len(y_low)}. "
                "Residual extraction assumes sample alignment."
            )

        wm_low = y_low - x_low

        if up == down == 1:
            return (x + wm_low).astype(np.float32)

        wm = self._match_len(resample_poly(wm_low, up, down), len(x))
        return (x + wm).astype(np.float32)

    def detect(self, audio: np.ndarray, sampling_rate: int):
        """Detects a watermark, downsampling to the model rate first."""
        # Sanitize audio: replace NaN with 0 and clip Inf to valid float range
        x = np.nan_to_num(np.asarray(audio, dtype=np.float64),
                          nan=0.0, posinf=1.0, neginf=-1.0)

        # Use the same resampler as embed rather than the service's, so the
        # residual sees a matched filter pair on the way out and back in.
        if int(sampling_rate) != MODEL_SR:
            up, down = self._ratio(sampling_rate)
            x = resample_poly(x, down, up)

        payload = {"audio": x.tolist(), "sampling_rate": MODEL_SR}

        response_data = self._make_request(endpoint="/detect", json_data=payload, method="POST")

        if "watermark" not in response_data:
            logger.error("'/detect' response did not contain 'watermark' key.")
            raise KeyError("Missing 'watermark' in response from /detect")

        return np.array(response_data["watermark"]), float(response_data.get("confidence", 0.0))

    def is_watermarked(self, detect_output) -> bool:
        """AudioSeal returns (watermark, confidence). Compare to threshold."""
        _watermark, confidence = detect_output
        threshold = self._config.get("detection_threshold", 0.5)
        return float(confidence) >= threshold