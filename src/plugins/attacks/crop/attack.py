import numpy as np

from core.base_attack import BaseAttack

import logging
logger = logging.getLogger(__name__)


class CropAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a crop attack by cropping a percentage of the beginning of the audio.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - crop_percentage (float): Total percentage to crop. Default is 10.

        Returns:
            np.ndarray: Audio signal with portions cropped from both ends.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        crop_percentage = kwargs.get(
            "crop_percentage", self.config.get("crop_percentage", 10)
        )

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        total_samples = len(audio)

        samples_to_crop_per_side = int(total_samples * (crop_percentage / 100.0))

        start_index = samples_to_crop_per_side
        cropped_audio = audio[start_index:]

        logger.info(f"Crop: removed {samples_to_crop_per_side} samples from each end, keeping {len(cropped_audio)} samples")

        return cropped_audio
