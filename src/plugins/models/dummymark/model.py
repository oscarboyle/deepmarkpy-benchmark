import numpy as np
from core.base_model import BaseModel

class DummyModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Initialize as an instance variable
        self.correct_watermark = np.random.randint(0, 2, size=16)

    def generate_watermark(self):
        """Generates a dummy 16-bit watermark array."""
        return self.correct_watermark

    def embed(self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Dummy embedder: instantly returns the original audio untouched."""
        # Save whatever watermark is being embedded so detect() always knows the right answer
        self.correct_watermark = watermark_data
        return audio.copy()

    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Dummy detector: always returns the exact watermark that was embedded."""
        return self.correct_watermark