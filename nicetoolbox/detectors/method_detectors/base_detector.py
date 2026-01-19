"""
A template class for Detectors.
"""

import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from nicetoolbox_core.entrypoint import SubprocessError

from ...utils.config import save_config
from ...utils.system import detect_os_type


class BaseDetector(ABC):
    """
    Abstract class to setup and run existing computer vision research code, called
    method detectors.

    Attributes:
        components (list): A list of components associated with the method detector.
        algorithm (str): The algorithm used for detecting the component.
        out_folder (str): The output folder.
        viz_folder (str): The visualization folder.
        subjects_descr (str): The subjects description.
        config_path (str): The path to the configuration file.
        conda_path (str): The path to the conda installation.
        framework (str): The name of the framework used for the method detector.
        script_path (str): The path to the script used for the method detector.
        venv (str): The type of virtual environment used for the method detector.
        env_name (str): The name of the virtual environment used for the method
            detector.
        venv_path (str): The path to the virtual environment used for the method
            detector.
    """

    def __init__(self, config, io, data, requires_out_folder=True) -> None:
        """
        Sets up the input and output folders required for each method based on the
        provided configurations. Saves a copy of the configuration file for the method
        detector.

        Args:
            config (dict): Configuration parameters for the detector.
            io (IO): An instance of the IO class for input/output operations.
            data (Data): An instance of the Data class for accessing data.
            requires_out_folder (bool, optional): Indicates whether an output folder
                is required. Defaults to True.
        """
        # (1) Store general detector/data/io information
        self.config = config
        self.io = io
        self.data = data

        self.results_folders = dict(
            (comp, io.get_detector_output_folder(comp, self.algorithm, "result")) for comp in self.components
        )
        main_component = self.components[0]
        self.out_folder, self.viz_folder = None, None
        if requires_out_folder:
            self.out_folder = io.get_detector_output_folder(main_component, self.algorithm, "output")
        if self.config["visualize"]:
            self.viz_folder = io.get_detector_output_folder(main_component, self.algorithm, "visualization")

        self.subjects_descr = data.subjects_descr

        # (2) Extend the content of the detector config (used during venv inference)
        self.config["log_file"], self.config["log_level"] = io.get_log_file_level()  # Get log file and level

        self.config["result_folders"] = self.results_folders
        self.config["out_folder"] = self.out_folder

        self.config["algorithm"] = self.algorithm
        self.config["calibration"] = data.calibration
        self.config["subjects_descr"] = data.subjects_descr
        self.config["cam_sees_subjects"] = data.camera_mapping["cam_sees_subjects"]

        self.config.update(data.get_input_recipe())  # Add data recipe to config for dataloader during inference

        # (3) Save the detector config (for venv inference)
        self.config_path = os.path.join(
            io.get_detector_output_folder(main_component, self.algorithm, "run_config"),
            "run_config.toml",
        )
        save_config(self.config, self.config_path)

        # (4) Prepare OS specific venv/conda inference settings
        self.os_type = detect_os_type()
        self.conda_path = io.get_conda_path()

        framework = config.get("framework", self.algorithm)
        self.venv, self.env_name = config["env_name"].split(":")

        self.script_path = io.get_inference_path(main_component, framework)
        if self.venv == "venv":
            self.venv_path = io.get_venv_path(framework, self.env_name)

    def __str__(self):
        """
        Returns a description of the method detector for printing.

        Returns:
            str: A string representation of the method detector, including its
                components, and the associated algorithm.
        """
        return f"Instance of component {self.components} \n\t" f"algorithm = {self.algorithm} \n\t" + " \n\t".join(
            [f"{attr} = {value}" for (attr, value) in self.__dict__.items()]
        )

    def run_inference(self) -> None:
        """
        Runs the inference of the method detector in a separate terminal/cmd
        window using the specified virtual environment or conda environment.
        Captures the output and logs the success or failure of the inference.
        """
        logging.info(f"INFERENCE: Launching {self.algorithm} subprocess...")

        # (1) Create the command to run the method detector
        command = self._create_command()

        # (2) Run the command in a separate terminal/cmd window
        if self.os_type == "windows":
            cmd_result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=True,
                check=False,
            )
        else:
            cmd_result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=True,
                executable="/bin/bash",
                check=False,
            )

        # (3) Check the return code and log the result
        if cmd_result.returncode == 0:
            logging.info(f"INFERENCE: {self.algorithm} finished successfully (Exit 0).")
            self.post_inference()
            return

        # (4) Handle subprocess failure
        logging.error(f"INFERENCE: {self.algorithm} subprocess failed (Exit 1).")

        config_path = Path(self.config_path)
        error_file = config_path.parent / "error.json"

        if error_file.exists():
            try:
                with open(error_file) as file:
                    remote_exc_json = json.load(file)

                remote_exc = SubprocessError(**remote_exc_json)

                # Raise. Now we need to figure out where to catch and
                # ignore this raise of the exception based on ErrorLevel.
                raise RuntimeError(
                    f"Subprocess raised {remote_exc.exception_type}: {remote_exc.message}\n\n"
                    f"--- Remote Traceback ---\n\n{remote_exc.traceback}"
                )

            except (json.JSONDecodeError, KeyError) as err:
                raise RuntimeError(
                    f"Subprocess failed and error report is corrupt: {err}\nStderr: {cmd_result.stderr}"
                ) from err
        else:
            # The script died before it could catch the exception and write the json.
            raise RuntimeError(f"Subprocess Hard Crash (No Error Report):\n{cmd_result.stderr}")

    def _create_command(self) -> str:
        """
        Creates the command to run the method detector in a separate terminal/cmd
        window using the specified virtual environment or conda environment.

        Returns:
            str: The command to run the method detector.
        """
        if self.venv == "conda":
            if self.os_type == "windows":
                command = (
                    f"deactivate && "
                    f'cmd "/c conda activate {self.env_name} && '
                    f'python {self.script_path} {self.config_path}"'
                )
            elif self.os_type == "linux":
                conda_path = os.path.join(self.conda_path, "bin/activate")
                python_path = os.path.join(self.conda_path, "envs", self.env_name, "bin/python")
                command = (
                    f"conda init bash && source ~/.bashrc && "
                    f"{conda_path} {self.env_name} && "
                    f"{python_path} {self.script_path} {self.config_path}"
                )
        elif self.venv == "venv":
            if self.os_type == "windows":
                command = f'cmd "/c {self.venv_path} && ' f'python {self.script_path} {self.config_path}"'
            elif self.os_type == "linux":
                command = f"source {self.venv_path} && " f"python {self.script_path} {self.config_path}"
        else:
            print(f"WARNING! venv '{self.venv}' is not known. " f"Detector not running.")
        return command

    @abstractmethod
    def post_inference(self) -> None:  # noqa: B027
        """
        Post-processing after inference.

        This method is called after the inference step and is used for any
        post-processing tasks that need to be performed.
        """
        pass

    @property
    @abstractmethod
    def components(self) -> list[str]:
        """
        Abstract property that returns the components of the method detector.

        This property should be implemented in the derived classes to specify the
        components that the method detector is associated with.

        Returns:
            list: A list of strings representing the components associated with the
                method detector.

        Raises:
            NotImplementedError: If the property is not set in the derived classes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm(self) -> str:
        """
        Abstract property that returns the algorithm of the method detector.

        This property should be implemented in the derived classes to specify the
        algorithm that the method detector is associated with.

        Returns:
            str: A string representing the algorithm associated with the method detector

        Raises:
            NotImplementedError: If the property is not set in the derived classes.
        """
        raise NotImplementedError

    @abstractmethod
    def visualization(self, data) -> None:
        """
        Abstract method to visualize the output of the method, preferably as a video.

        This method is intended to generate a visual representation of the method
        detector's output. The visualization should be saved in the self.viz_folder.

        Args:
            data (any): The data to be visualized. The type and content of this
                parameter depend on the specific implementation of the method detector.

        Returns:
            None. This method does not return any value. However, it should save the
                visualization in the self.viz_folder.

        Raises:
            NotImplementedError: If this method is not implemented in the derived
                classes.
        """
        pass
