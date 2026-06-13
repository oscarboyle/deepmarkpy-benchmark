import inspect
import logging
import os

import numpy as np
import soundfile as sf
import librosa
import json

from plugin_manager import PluginManager
from utils.utils import load_audio, snr
from utils.metrics import si_sdr, psnr #pesq_wrapper, stoi_wrapper, 


logger = logging.getLogger(__name__)

def to_json_safe(obj):
    """
    Recursively convert numpy types to native Python types
    so json.dump does not crash.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


class Benchmark:
    """
    A class to perform various attacks on watermarking models and benchmark their performance.
    """

    def __init__(self):
        """
        Initialize Benchmark class with PluginManager.
        """
        self.plugin_manager = PluginManager()
        # Now these are dicts of the form { "class_name": {"class": ActualClass, "config": {...}} }
        self.attacks = self.plugin_manager.get_attacks()
        self.models = self.plugin_manager.get_models()

    def get_available_args(self):
        valid_args = {}
        models = self.models.keys()
        attacks = self.attacks.keys()
        for attack in attacks:
            config = self.attacks[attack]["config"]
            if config is not None:
                for key, value in config.items():
                    if key in valid_args and valid_args[key] != value:
                        logger.warning(
                            f"Config parameter '{key}' defined by multiple attacks with "
                            f"different defaults. Last value wins. Consider using unique "
                            f"parameter names (e.g., '{key}_{attack.lower()}')."
                        )
                    valid_args[key] = value
        return list(models), list(attacks), valid_args

    def show_available_plugins(self):
        """
        Print out all discovered models and attacks, including any __init__ parameters
        and key-value pairs from config.json (defaults).
        """
        logger.info("===== Available Models =====")
        for model_name, model_entry in self.models.items():
            model_cls = model_entry["class"]
            config = model_entry.get("config") or {}

            signature = inspect.signature(model_cls.__init__)
            params = [p for p in signature.parameters.values() if p.name != "self"]

            init_params = {
                p.name: (None if p.default is inspect.Parameter.empty else p.default)
                for p in params
            }

            logger.info(f"\nModel: {model_name}")
            logger.info(f"  - Constructor parameters: {init_params}")

            logger.info("  - Arguments defaults:")
            if config:
                for key, val in config.items():
                    logger.info(f"    {key}: {val}")
            else:
                logger.info("    (none found)")

        logger.info("\n===== Available Attacks =====")
        for attack_name, attack_entry in self.attacks.items():
            attack_cls = attack_entry["class"]
            config = attack_entry.get("config") or {}

            signature = inspect.signature(attack_cls.__init__)
            params = [p for p in signature.parameters.values() if p.name != "self"]
            init_params = {
                p.name: (None if p.default is inspect.Parameter.empty else p.default)
                for p in params
            }

            logger.info(f"\nAttack: {attack_name}")
            logger.info(f"  - Constructor parameters: {init_params}")

            logger.info("  - Argument defaults:")
            if config:
                for key, val in config.items():
                    logger.info(f"    {key}: {val}")
            else:
                logger.info("    (none found)")

    def run(
        self,
        filepaths,
        wm_model,
        watermark_data=None,
        attack_types=None,
        sampling_rate=None,
        verbose=False,
        save_audio= False,
        output_dir="audio_processed",
        calculate_quality_metrics=True,
        results_filename=None,
        **kwargs,
    ):
        """
        Benchmark the watermarking models against selected attacks.

        Args:
            filepaths (str or list): Path(s) to the audio file(s) to benchmark.
            wm_model (str): The model to benchmark (e.g., 'AudioSeal', 'WavMark', 'SilentCipher').
            watermark_data (np.ndarray, optional): The binary watermark data to embed. Defaults to random message.
            attack_types (list, optional): A list of attack types to perform. Defaults to all available attacks.
            sampling_rate (int, optional): Target sampling rate for loading audio. Defaults to None.
            verbose (bool, optional): Print verbose info. Defaults to False.
            save_audio (bool, optional): Whether to save processed audio files. Defaults to False.
            output_dir (str, optional): Directory to save processed audio. Defaults to "audio_processed".
            **kwargs: Additional parameters for specific attacks.

        Returns:
            dict: A dictionary containing benchmark results for each file and attack.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # Create output directory if it doesn't exist
        if save_audio:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Audio will be saved to: {output_dir}")

        # If user doesn't specify attacks, use them all
        attack_types = attack_types or list(self.attacks.keys())
        results = {}
        if results_filename and os.path.exists(results_filename):
            try:
                with open(results_filename, "r") as fp:
                    results = json.load(fp)
                logger.info(f"Loaded existing progress from {results_filename}. Checking for missing attacks...")
            except Exception as e:
                logger.warning(f"Could not load existing results from {results_filename}. Starting fresh. Error: {e}")

        if wm_model not in self.models:
            raise ValueError(
                f"Model '{wm_model}' not found. Available: {list(self.models.keys())}"
            )

        model_cls = self.models[wm_model]["class"]
        model_instance = model_cls()
        model_config = self.models[wm_model]["config"] or {}
        returns_confidence = model_config.get("returns_confidence", False)
        is_zero_bit = model_config.get("is_zero_bit", False)

        if sampling_rate is None:
            sampling_rate = self.models[wm_model]["config"]["sampling_rate"]
            logger.info(f"Using default sampling rate {sampling_rate} for model {wm_model}")

        attack_kwargs = {
            **kwargs,
            "model": model_instance,
            "watermark_data": watermark_data,
            "sampling_rate": sampling_rate,
            "models": self.models,
        }

        for filepath in filepaths:

            file_results = results.get(filepath, {})
            
            # Figure out which attacks actually need to be run
            attacks_to_run = []
            for atk in attack_types:
                if atk not in file_results:
                    attacks_to_run.append(atk)
                elif atk  in file_results and verbose:
                    logger.info(f"  Skipping attack '{atk}' for {os.path.basename(filepath)} (already processed).")

            # If all requested attacks are already processed, skip the file entirely!
            if not attacks_to_run:
                if verbose:
                    logger.info(f"Skipping {os.path.basename(filepath)} entirely. All requested attacks are complete.")
                continue

            if verbose:
                logger.info(f"\nProcessing file: {filepath} with attacks: {attacks_to_run}")
            
            # If the file wasn't in the JSON at all, initialize its dictionary.
            # If it was, we retain the old data and will just append to it later.
            if filepath not in results:
                results[filepath] = {}

            # Get base filename without extension
            base_filename = os.path.splitext(os.path.basename(filepath))[0]

            # Generate a fresh watermark for each file if none was supplied by the user
            file_watermark = watermark_data if watermark_data is not None else model_instance.generate_watermark()
            attack_kwargs["watermark_data"] = file_watermark

            # Load audio
            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)
            logger.info(f"Sampling rate is: {sampling_rate}")
            attack_kwargs["orig_audio"] = audio

            # Embed watermark
            watermarked_audio = model_instance.embed(
                audio=audio, watermark_data=file_watermark, sampling_rate=sampling_rate
            ) 
            psnr_embed = "N/A"
            si_sdr_embed = "N/A"
            if calculate_quality_metrics:
                sr_scalar = int(sampling_rate) if isinstance(sampling_rate, (np.ndarray, list)) else sampling_rate
                psnr_embed = psnr(audio, watermarked_audio)
                si_sdr_embed = si_sdr(audio, watermarked_audio)
                
            results[filepath]["embedding_metrics"] = {
                      
                "psnr": psnr_embed,
                "si_sdr": si_sdr_embed
            }


            # Save watermarked audio
            if save_audio:
                watermarked_filename = f"{base_filename}_watermarked.wav"
                watermarked_path = os.path.join(output_dir, watermarked_filename)
                sf.write(watermarked_path, watermarked_audio, sampling_rate) 

            # Apply each attack and compute metrics
            for attack_name in attacks_to_run:
                if attack_name not in self.attacks:
                    logger.warning(f"Attack '{attack_name}' not found. Skipping.")
                    continue

                if verbose:
                    logger.info(f"  Applying attack: {attack_name}")

                attack_instance = self.attacks[attack_name]["class"]()

                if (attack_name =="CrossModelAttack"):
                    
                    different_model_name = kwargs.get("different_model_name")
                    logger.info(f"Different model is chosen and it's {different_model_name}")
                    different_model_cls = self.models[different_model_name]["class"]
                    different_model_instance = different_model_cls()

                    attacked_audio, different_watermark = attack_instance.apply(
                        watermarked_audio, **attack_kwargs
                    )
                    if calculate_quality_metrics:
                        attacked_audio_metrics, _ = attack_instance.apply(
                            audio, **attack_kwargs
                        )


                #in case of the collusion mod attack
                elif (attack_name == "ZeroBitCollusionAttack"):
                    attack_kwargs["original_audio_collusion"] = audio

                    attacked_audio = attack_instance.apply(
                        watermarked_audio, **attack_kwargs
                    )
                    if calculate_quality_metrics:
                        attacked_audio_metrics = attack_instance.apply(
                            audio, **attack_kwargs
                        )

                else:
                    attacked_audio = attack_instance.apply(
                        watermarked_audio, **attack_kwargs
                    )
                    if calculate_quality_metrics:
                        attacked_audio_metrics = attack_instance.apply(
                            audio, **attack_kwargs
                        )

                # Ensure consistent shape for all attacks
                if isinstance(attacked_audio, np.ndarray):
                    attacked_audio = np.squeeze(attacked_audio)
                if calculate_quality_metrics and isinstance(attacked_audio_metrics, np.ndarray):
                    attacked_audio_metrics = np.squeeze(attacked_audio_metrics)

                # Save attacked audio
                if save_audio:
                    audio_to_save = attacked_audio
                    if audio_to_save.ndim == 1:
                        audio_to_save = np.expand_dims(audio_to_save, axis=1)

                    attacked_filename = f"{base_filename}_{attack_name}.wav"
                    attacked_path = os.path.join(output_dir, attacked_filename)
                    sf.write(attacked_path, audio_to_save, sampling_rate)
                    if verbose:
                        logger.info(f"Saved attacked audio: {attacked_filename}")
                
                confidence = None
                if returns_confidence:
                    detected_message, confidence = model_instance.detect(attacked_audio, sampling_rate)
                else:
                    detected_message = model_instance.detect(attacked_audio, sampling_rate)

                if (attack_name =="CrossModelAttack"):
                    different_detected_message = different_model_instance.detect(attacked_audio, sampling_rate)
                    diff_model_config = self.models.get(different_model_name, {}).get("config") or {}
                    diff_is_zero_bit = diff_model_config.get("is_zero_bit", False)
                    diff_returns_confidence = diff_model_config.get("returns_confidence", False)
                    if diff_is_zero_bit:
                        if isinstance(different_detected_message, np.ndarray):
                            different_accuracy = different_detected_message.tolist()
                        else:
                            different_accuracy = different_detected_message
                    elif diff_returns_confidence:
                        different_watermark_detected, _ = different_detected_message
                        different_accuracy = self.compare_watermarks(different_watermark, different_watermark_detected)
                    else:
                        different_accuracy = self.compare_watermarks(different_watermark, different_detected_message)
                

                snr_val = snr(audio, attacked_audio)
                

                sr_scalar = int(sampling_rate) if isinstance(sampling_rate, (np.ndarray, list)) else sampling_rate
                # stoi_val = "N/A"
                # pesq_val = "N/A"
                psnr_atk_val = "N/A"
                si_sdr_atk_val = "N/A"
                if calculate_quality_metrics:
                    
                    # Resample to 16kHz if needed (PESQ/STOI only support 8kHz/16kHz)
                    metrics_sr = 16000 if sr_scalar not in [8000, 16000] else sr_scalar
                    # ref = librosa.resample(audio, orig_sr=sr_scalar, target_sr=metrics_sr) if metrics_sr != sr_scalar else audio
                    # deg = librosa.resample(attacked_audio_metrics, orig_sr=sr_scalar, target_sr=metrics_sr) if metrics_sr != sr_scalar else attacked_audio_metrics

                    # stoi_val = stoi_wrapper(ref, deg, metrics_sr)
                    # pesq_val = pesq_wrapper(ref, deg, metrics_sr, 'wb')

                    psnr_atk_val = psnr(audio, attacked_audio_metrics)
                    si_sdr_atk_val = si_sdr(audio, attacked_audio_metrics)


                if is_zero_bit:
                    if isinstance(detected_message, np.ndarray):
                        accuracy = detected_message.tolist()
                    else:
                        accuracy = detected_message
                else:
                    accuracy = self.compare_watermarks(file_watermark, detected_message)
                    
                results[filepath][attack_name] = {
                    "accuracy": accuracy,
                    "psnr_attack": psnr_atk_val,
                    "si_sdr_attack": si_sdr_atk_val,
                    }

                # Add confidence for models that return it
                if confidence is not None:
                    results[filepath][attack_name]["confidence"] = confidence

                if attack_name == "CrossModelAttack":
                    results[filepath][attack_name]["accuracy_cross_model"] = different_accuracy

        # Incrementally save the results after all attacks on the current file finish
            if results_filename:
                try:
                    with open(results_filename, "w") as fp:
                        json.dump(to_json_safe(results), fp, indent=4)
                    if verbose:
                        logger.info(f"Incrementally saved progress to {results_filename}")
                except Exception as e:
                    logger.error(f"Failed to incrementally save results for {filepath}: {e}")

        return results

    def compute_mean_accuracy(self, results):
        """
        Compute mean accuracy and FNR/FPR for each attack.

        Args:
            results: Dictionary of results from run()
            confidence_threshold: Threshold for watermark detection (default 0.5)

        Returns:
            Dictionary with mean accuracies and FNR/FPR rates
        """
        attack_accuracies = {}

        for _, attack_dict in results.items():
            for attack_name, metrics in attack_dict.items():

                if "accuracy" not in metrics:
                    continue

                if attack_name not in attack_accuracies:
                    attack_accuracies[attack_name] = {
                        "accuracy": [],
                        "accuracy_cross_model": [],
                        "confidence": []
                    }
                
                
                attack_accuracies[attack_name]["accuracy"].append(metrics["accuracy"])

                if "accuracy_cross_model" in metrics:
                    attack_accuracies[attack_name]["accuracy_cross_model"].append(
                        metrics["accuracy_cross_model"]
                    )

                if "confidence" in metrics:
                    attack_accuracies[attack_name]["confidence"].append(metrics["confidence"])

        mean_accuracies = {}

        for attack_name, acc in attack_accuracies.items():
            mean_accuracies[attack_name] = {}

            mean_accuracies[attack_name]["accuracy_mean"] = float(
                np.mean([a for a in acc["accuracy"] if a is not None])
            )

            if acc["accuracy_cross_model"]:
                mean_accuracies[attack_name]["accuracy_cross_model_mean"] = float(
                    np.mean([a for a in acc["accuracy_cross_model"] if a is not None])
                )

        return mean_accuracies


    def compare_watermarks(self, original, detected):
        """
        Compare the original and detected watermarks.

        Args:
            original (np.ndarray): The original binary watermark.
            detected (np.ndarray): The detected binary watermark.

        Returns:
            float: The accuracy of the detected watermark (percentage), or None if invalid.
        """
        if detected is None:
            return 50.00
        if isinstance(detected, np.ndarray) and detected.ndim == 0:
            return 50.00
        if isinstance(detected, (list, np.ndarray)) and len(detected) == 0:
            return 50.00
        if np.any(detected == np.array(None)):
            return 50.00
        # Check for shape mismatch
        if len(original) != len(detected):
            return 50.00
        matches = np.sum(original == detected)
        return (matches / len(original)) * 100