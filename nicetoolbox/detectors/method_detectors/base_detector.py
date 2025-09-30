"""
A template class for Detectors.
"""

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from ...utils import system
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
        # log file
        config["log_file"], config["log_level"] = io.get_log_file_level()

        # output folders for inference and visualizations
        config["result_folders"] = dict(
            (comp, io.get_detector_output_folder(comp, self.algorithm, "result"))
            for comp in self.components
        )
        main_component = self.components[0]
        if requires_out_folder:
            config["out_folder"] = io.get_detector_output_folder(
                main_component, self.algorithm, "output"
            )
            self.out_folder = config["out_folder"]
        if config["visualize"]:
            self.viz_folder = io.get_detector_output_folder(
                main_component, self.algorithm, "visualization"
            )

        config["algorithm"] = self.algorithm
        config["calibration"] = data.calibration
        config["subjects_descr"] = data.subjects_descr
        self.subjects_descr = data.subjects_descr
        config["cam_sees_subjects"] = data.camera_mapping["cam_sees_subjects"]

        # Todo SPIGA post-inference and inference depends on this order
        ordered_views = []
        if config.get("frames_list"):
            for path in config.get("frames_list")[
                0
            ]:  # getting order from the first frame
                parts = path.split(os.sep)
                if "frames" in parts:
                    idx = parts.index("frames")
                    ordered_views.append(parts[idx - 1])
                else:
                    ordered_views.append(Path(path).parent.parent.name)
            self.camera_order = ordered_views.copy()
            config["camera_order"] = self.camera_order

        # save this method config that will be given to the third party detector
        self.config_path = os.path.join(
            io.get_detector_output_folder(main_component, self.algorithm, "run_config"),
            "run_config.toml",
        )
        save_config(config, self.config_path)

        self.conda_path = io.get_conda_path()
        self.framework = (
            config["framework"] if "framework" in config.keys() else self.algorithm  # noqa: SIM118
        )
        self.script_path = data.get_inference_path(main_component, self.framework)

        # specify the virtual environment for the third party method/detector
        self.venv, self.env_name = config["env_name"].split(":")
        if self.venv == "venv":
            self.venv_path = data.get_venv_path(self.framework, self.env_name)

    def __str__(self):
        """
        Returns a description of the method detector for printing.

        Returns:
            str: A string representation of the method detector, including its
                components, and the associated algorithm.
        """
        return (
            f"Instance of component {self.components} \n\t"
            f"algorithm = {self.algorithm} \n\t"
            + " \n\t".join(
                [f"{attr} = {value}" for (attr, value) in self.__dict__.items()]
            )
        )

    def run_inference(self):
        # detect operating system
        os_type = detect_os_type()

        if self.venv == "conda":
            # create terminal command
            if os_type == "windows":
                command = (
                    f"deactivate && "
                    f'cmd "/c conda activate {self.env_name} && '
                    f'python {self.script_path} {self.config_path}"'
                )
            elif os_type == "linux":
                conda_path = os.path.join(self.conda_path, "bin/activate")
                python_path = os.path.join(
                    self.conda_path, "envs", self.env_name, "bin/python"
                )
                command = (
                    f"conda init bash && source ~/.bashrc && "
                    f"{conda_path} {self.env_name} && "
                    f"{python_path} {self.script_path} {self.config_path}"
                )
                # command = f"{python_path} {self.script_path} {self.config_path}"

        elif self.venv == "venv":
            # create terminal command
            if os_type == "windows":
                command = (
                    f'cmd "/c {self.venv_path} && '
                    f'python {self.script_path} {self.config_path}"'
                )
            elif os_type == "linux":
                command = (
                    f"source {self.venv_path} && "
                    f"python {self.script_path} {self.config_path}"
                )

        else:
            print(
                f"WARNING! venv '{self.venv}' is not known. " f"Detector not running."
            )

        # run in terminal/cmd
        if system.detect_os_type() == "windows":
            cmd_result = subprocess.run(
                command, capture_output=True, text=True, shell=True, check=False
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

        if cmd_result.returncode == 0:
            logging.info("INFERENCE Pipeline - SUCCESS.")
            self.post_inference()

        else:
            logging.error(
                f"INFERENCE Pipeline - ERROR occurred with return code "
                f"{cmd_result.returncode}"
            )
            logging.error(f"INFERENCE Pipeline - ERROR: {cmd_result.stderr}")
            logging.info(f"INFERENCE Pipeline - Terminal OUTPUT {cmd_result.stdout}")

    def post_inference(self):  # noqa: B027
        """
        Post-processing after inference.

        This method is called after the inference step and is used for any
        post-processing tasks that need to be performed.
        """
        pass

    @property
    @abstractmethod
    def components(self):
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
    def algorithm(self):
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
    def visualization(self, data):
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
