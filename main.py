import argparse
import os
import subprocess

from nicetoolbox.detectors.main import main as detectors_main
from nicetoolbox.evaluation.main import main as evaluation_main
from nicetoolbox.utils.calibration_gui.calibration_converter import (
    calibration_converter,
)
from nicetoolbox.visual.media.main import main as visual_media_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="detectors",
        type=str,
        required=False,
        choices=[
            "detectors",
            "evaluation",
            "visual_media",
            "visual_stats",
            "calibration_converter",
        ],
    )
    args = parser.parse_args()

    working_directory = os.getcwd()
    machine_specifics = f"{working_directory}/machine_specific_paths.toml"

    if args.run == "detectors":
        run_config = f"{working_directory}/nicetoolbox/detectors/configs/run_file.toml"
        detectors_main(run_config, machine_specifics)
        
    elif args.run == "evaluation":
        eval_config = f"{working_directory}/nicetoolbox/evaluation/configs/evaluation_config.toml"
        evaluation_main(eval_config, machine_specifics)
        
    elif args.run == "visual_media":
        visual_media_main()

    elif args.run == "visual_stats":
    
        main = "main_statistics.py"
        script_command = f"streamlit run {main}"
        venv_path = "envs/nicetoolbox/bin/activate"
        command = f"source {venv_path} && {script_command}"
        
        subprocess.run(
            command, 
            text=True,
            shell=True,
            executable="/bin/bash"
        )

    elif args.run == "calibration_converter":
        calibration_converter()

    else:
        print("Unknown argument! Please call 'main.py' with one of the following "
              "arguments: 'detectors', 'visual_media', 'visual_stats', or "
              "'calibration_converter'.")
    