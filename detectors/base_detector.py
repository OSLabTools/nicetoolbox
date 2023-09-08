"""
A template class for Detectors.
"""

import os
from abc import ABC, abstractmethod
import subprocess
from oslab_utils.system import detect_os_type
from oslab_utils.config import save_config


class BaseDetector(ABC):
    """Class to setup and run existing computer vision research code.
    """

    def __init__(self, config, io, data) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        # output folders for inference and visualizations
        config['out_folder'] = io.get_output_folder(self.name, 'output')
        config['result_folder'] = io.get_output_folder(self.name, 'result')
        self.viz_folder = io.get_output_folder(self.name, 'visualization')

        # save this method config that will be given to the third party detector
        self.config_path = os.path.join(io.get_output_folder(self.name, 'tmp'),
                                        'run_config.toml')
        save_config(config, self.config_path)

        # get the path of the third party inference script
        self.script_path = data.get_inference_path(self.name)

        # specify the virtual environment for the third party method/detector
        self.venv, self.env_name = config['env_name'].split(':')
        if self.venv == 'venv':
            self.env_name = data.get_venv_path(self.name, self.env_name)

    def __str__(self):
        """ description of the class instance for printing

        Returns
        -------
        str
            all attributes and their values, written in plain text
        """
        return f"Instance of class {self.name} \n\t" \
               f"behavior = {self.behavior} \n\t" + \
               " \n\t".join([f"{attr} = {value}"
                             for (attr, value) in self.__dict__.items()])

    def run_inference(self):
        # detect operating system
        os_type = detect_os_type()

        if self.venv == 'conda':
            # create terminal command
            if os_type == 'windows':
                command = f'cmd "/c conda activate {self.env_name} && ' \
                          f'python {self.script_path} {self.config_path}"'
            elif os_type == "linux":
                command = f"conda activate {self.env_name} && " \
                          f"python {self.script_path} {self.config_path}"

        if self.venv == 'venv':
            # create terminal command
            if os_type == 'windows':
                command = f'cmd "/c source {self.env_name} && ' \
                          f'python {self.script_path} {self.config_path}"'
            elif os_type == "linux":
                command = f"source {self.env_name} && " \
                          f"python {self.script_path} {self.config_path}"

        else:
            print(f"WARNING! venv '{self.venv}' is not known. "
                  f"Detector not running.")

        # run in terminal/cmd
        cmd_result = subprocess.run(command, capture_output=True, text=True)

        if cmd_result.returncode != 0:
            print(f"Error occurred with return code {cmd_result.returncode}")
            print(cmd_result.stderr)
        return cmd_result.stdout

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def behavior(self):
        raise NotImplementedError

    @abstractmethod
    def visualization(self, data):
        """ Visualize the output of the method, preferably as a video

        Should save the visualization in the self.viz_folder
        """
        pass

