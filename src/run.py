import argparse
import json
import os

import logging

from benchmark import Benchmark, to_json_safe
from utils.report_generator import generate_benchmark_report
import numpy as np
import sys 

# Set up logging to go to BOTH the terminal and a file called "benchmark.log"
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("benchmark.log", mode="a"), # Appends to a log file
        logging.StreamHandler(sys.stdout)               # Still prints to the terminal
    ]
)
logger = logging.getLogger(__name__)



# def to_json_safe(obj):
#     """
#     Recursively convert numpy types to native Python types
#     so json.dump does not crash.
#     """
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (np.float32, np.float64)):
#         return float(obj)
#     if isinstance(obj, (np.int32, np.int64)):
#         return int(obj)
#     if isinstance(obj, dict):
#         return {k: to_json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [to_json_safe(v) for v in obj]
#     return obj
    
def main():
    benchmark = Benchmark()

    models, attacks, valid_args = benchmark.get_available_args()

    parser = argparse.ArgumentParser(description="Run DeepMark Benchmark CLI")

    # Add model and attack selection
    parser.add_argument(
        "--wav_files_dir",
        type=str,
        help="Path to the directory containing .wav files.",
        required=True,
    )
    parser.add_argument(
        "--wm_model",
        type=str,
        choices=models,
        required=True,
        help="Watermarking model to use.",
    )
    parser.add_argument(
        "--attack_types",
        type=str,
        nargs="*",
        choices=attacks,
        default=None,
        metavar="ATTACK",
        help="List of attacks to apply. Allowed values: " + ", ".join(attacks),
    )

    # Add verbose flag
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    # Dynamically add configuration parameters from the available plugins
    for arg, default_value in valid_args.items():
        if isinstance(default_value, bool):
            # Use BooleanOptionalAction to support both --flag and --no-flag
            parser.add_argument(
                f"--{arg}",
                action=argparse.BooleanOptionalAction,
                default=default_value,
                help=f"Enable/disable {arg} (default: {default_value})",
            )
        else:
            parser.add_argument(
                f"--{arg}",
                type=type(default_value),
                default=default_value,
                help=f"Set {arg} (default: {default_value})",
            )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        #Mute noisy libraries
        logging.getLogger("numba").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger.debug("Verbose logging enabled.")

    args_dict = vars(args)

    try:
        all_files = os.listdir(args.wav_files_dir)
        filepaths = [
            os.path.join(args.wav_files_dir, f)
            for f in all_files
            if f.lower().endswith(".wav") or f.lower().endswith(".mp3")
        ]
        if not filepaths:
            logger.error(f"No .wav files found in directory: {args.wav_files_dir}")
            return  # Exit if no files found
        logger.info(f"Found {len(filepaths)} .wav files to process.")
    except FileNotFoundError:
        logger.error(f"Audio directory not found: {args.wav_files_dir}")
        return
    except Exception as e:
        logger.error(f"Error accessing audio directory {args.wav_files_dir}: {e}")
        return

    folder_name = os.path.basename(os.path.normpath(args.wav_files_dir))

    results_filename = f"benchmark_results_{args.wm_model}_{folder_name}.json"
    stats_filename = f"benchmark_stats_{args.wm_model}_{folder_name}.json"

    results = benchmark.run(
        filepaths=filepaths, 
        results_filename=results_filename, 
        **args_dict
    )

    with open(results_filename, "w") as fp:
        json.dump(to_json_safe(results), fp, indent=4)

    logger.info(f"Benchmark completed. Results saved to {results_filename}")

    stats = benchmark.compute_mean_accuracy(results)
    flattened_stats = {attack: metrics["accuracy_mean"] for attack, metrics in stats.items()}
    
    with open(stats_filename, "w") as fp:
        json.dump(to_json_safe(flattened_stats), fp, indent=4)

    logger.info(f"Benchmark statistics saved to {stats_filename}")

    try:
        logger.info("Generating benchmark report...")
        # Pass the dynamic stats_filename to the report generator
        latex_path, chart_path = generate_benchmark_report(
            stats_file=stats_filename, 
            model_name=args.wm_model,
            report_dir="report"
        )


        logger.info(f"Benchmark report generated: {latex_path}")
        logger.info(f"Chart saved: {chart_path}")
    except Exception as e:
        logger.error(f"Failed to generate benchmark report: {e}")
        logger.info("Benchmark data is still available in JSON files")



if __name__ == "__main__":
    main()
