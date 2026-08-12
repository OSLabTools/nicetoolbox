"""
Base class for Method Detectors.
Method detectors run external inference scripts in separate virtual environments.
"""

import inspect
import json
import logging
import os
import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import final

from nicetoolbox_core.entrypoint import SubprocessError, get_subprocess_error_path

from ...configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ...utils.base_detectors import flatten_inference_config
from ...utils.config import save_config
from ...utils.hf_token import effective_hf_hub_token
from ...utils.system import detect_os_type
from ..base_detector import BaseDetector
from ..data import SequenceData
from ..detector_outputs import DetectorOutput
from ..in_out import SequenceIO
from ..subsequence_context import SubsequenceContext


class BaseMethod(BaseDetector):
    """
    Abstract base class for method detectors.

    Method detectors run inference in external virtual environments via subprocess.
    """

    # If set, used for get_inference_path instead of components[0].
    inference_package_name: str | None = None

    runtime: MethodDetectorRuntime

    # subprocess settings
    os_type: str
    conda_path: Path
    venv: str
    env_name: str
    script_path: str

    # method specific settings
    visualize: bool
    requires_out_folder: bool
    out_folders: dict[str, str]
    result_folders: dict[str, str]
    viz_folders: dict[str, str]
    config_paths: list[Path]

    @final
    def __init__(
        self,
        io: SequenceIO,
        data: SequenceData,
        subsequence_context: SubsequenceContext,
        algorithm_instance: str,
    ) -> None:
        """
        Initialize base method detector with references.
        """
        # (1) Call BaseDetector __init__()
        super().__init__(io, data, subsequence_context, algorithm_instance)

        logging.info(
            f"Initializing method detector {self.__class__.__name__} with instance '{self.algorithm_instance}' "
            f"and component(s) {self.components}."
        )

        # (2) Setup subprocess execution settings
        self._setup_subprocess_settings()

        # (3) Setup method detector (builds runtime, validates, saves config)
        self.requires_out_folder = True
        self.out_folders = self.compute_output_folders(self.requires_out_folder)
        self.result_folders = self.compute_result_folders()
        self.viz_folders = self.compute_viz_folders(self.visualize)

        self.runtime = self._initialize_detector()  # This will be overloaded by child detectors!

        # (4) Flatten config for subprocess
        inference_config = flatten_inference_config(self.detector_config, self.runtime)

        # Pre-map single component out_folder and viz_folder for backward compatibility... TODO
        if len(self.components) == 1:
            comp = self.components[0]
            self.out_folder = self.out_folders[comp]
            self.viz_folder = self.viz_folders[comp]
            inference_config["out_folder"] = self.out_folder
            inference_config["viz_folder"] = self.viz_folder

        # (5) Save config for subprocess for each component
        folder = self.io.get_detector_output_folder(self.components[0], self.algorithm_instance, "run_config")
        self.config_path = folder / "run_config.toml"
        save_config(inference_config, self.config_path)
        for comp in self.components[1:]:
            dup = self.io.get_detector_output_folder(comp, self.algorithm_instance, "run_config") / "run_config.toml"
            save_config(inference_config, dup)

        logging.info("Inference preparation completed.\n")

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Create runtime configuration.

        Override in subclasses that need extended runtime fields.
        The override should:
        1. Call super()._create_runtime() to get base runtime
        2. Return an instance of the detector's RuntimeConfig class with additional fields

        Returns:
            MethodDetectorRuntime or subclass with detector-specific fields
        """
        return MethodDetectorRuntime(
            nicetoolbox_root=str(self.io.code_folder),
            result_folders=self.result_folders,
            out_folders=self.out_folders,
            viz_folders=self.viz_folders,
            algorithm=self.algorithm_instance,
            visualize=self.visualize,
            subjects_descr=self.data.subjects_descr,
            log_file=str(self.subsequence_context.log_file),
            log_level=self.subsequence_context.log_level,
            calibration=self.data.calibration,
            cam_sees_subjects=self.data.cam_sees_subjects,
            input_recipes=self.data.get_input_recipes(),
        )

    def _resolve_inference_script(self) -> Path:
        """
        Resolve the inference script path relative to the concrete subclass's file.
        Convention: <same_folder_as_class>/<algorithm_type>_inference.py
        """
        class_file = Path(inspect.getfile(type(self)))
        script = class_file.parent / f"{self.detector_config.algorithm_type}_inference.py"
        if not script.exists():
            logging.error(f"Detector inference file {script} does not exist!")
            raise FileNotFoundError(script)
        return script

    def _setup_subprocess_settings(self) -> None:
        """Setup OS-specific subprocess execution settings."""
        self.os_type = detect_os_type()
        self.conda_path = self.io.get_conda_path()

        env_name = getattr(self.detector_config, "env_name", "venv:nicetoolbox")  # Default to nicetoolbox env
        self.venv, self.env_name = env_name.split(":")

        self.script_path = self._resolve_inference_script()

        if self.venv == "venv":
            self.venv_path = self.io.get_venv_path(self.detector_config.algorithm_type, self.env_name)

    def _subprocess_env(self) -> dict:
        """Copy os.environ and set HF_TOKEN from machine_specific_paths.toml when configured."""
        env = os.environ.copy()
        tok = effective_hf_hub_token(self.subsequence_context.machine)
        if tok:
            env["HF_TOKEN"] = tok
        return env

    # -------------------------------------------------------------------------
    # BaseDetector Interface Implementation
    # -------------------------------------------------------------------------

    def run(self):
        """
        Execute method detector: run subprocess inference + post_inference.

        post_inference may return a DetectorOutput (migrated detectors); if so, it is validated
        against the declared outputs and saved here, then returned for visualization. Legacy
        detectors return None/int and save themselves — that value is passed through untouched.
        """
        result: DetectorOutput = self._run_inference()
        # TODO: after migration finished - unify with base_feature
        if isinstance(result, DetectorOutput):
            # validate all outputs against defined outputs and schemas
            result.validate(self.declared_outputs)
            # validate that all npz have valid axes
            subjects, cameras, frames = self.data.canonical_axes
            result.validate_canonical_axes(subjects, cameras, frames)
            # dump results on drive
            result.save(self.io, self.algorithm_instance)
        return result

    def _run_inference(self) -> None:
        """Run the inference subprocess."""
        logging.info(f"INFERENCE: Launching {self.algorithm_instance} subprocess...")
        # debugging shortcut
        if self.subsequence_context.run.skip_inference:
            logging.warning(
                f"INFERENCE: skip_inference is set - reusing existing {self.algorithm_instance} "
                "inference output and running post-processing only."
            )
            return self.post_inference()

        command = self._create_command()

        # delete error from previois runs, before starting processing
        error_file = get_subprocess_error_path(self.config_path)
        error_file.unlink(missing_ok=True)

        sub_env = self._subprocess_env()
        if self.os_type == "windows":
            cmd_result = subprocess.run(command, capture_output=True, text=True, shell=True, check=False, env=sub_env)
        else:
            cmd_result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=True,
                executable="/bin/bash",
                check=False,
                env=sub_env,
            )

        if cmd_result.returncode == 0:
            logging.info(f"INFERENCE: {self.algorithm_instance} finished successfully (Exit 0).")
            return self.post_inference()

        logging.error(f"INFERENCE: {self.algorithm_instance} subprocess failed (Exit 1).")
        self._handle_subprocess_error(cmd_result)
        return None

    def _create_command(self) -> str:
        """Create the shell command to run inference."""
        script = f'"{self.script_path}"'
        config = f'"{self.config_path}"'

        if self.venv == "conda":
            if self.os_type == "windows":
                conda_env_path = os.path.join(self.io.code_folder, "envs", self.env_name)
                # fmt: off
                command = (
                    f"deactivate && "
                    f'cmd /s /c "conda activate {conda_env_path} && '
                    f'python {script} {config}"'
                )
                # fmt: on
            else:
                python_path = os.path.join(self.io.code_folder, "envs", self.env_name, "bin/python")
                command = f"'{python_path}' {script} {config}"
        elif self.venv == "venv":
            if self.os_type == "windows":
                command = f'cmd /s /c ""{self.venv_path}" && python {script} {config}"'
            else:
                command = f"source '{self.venv_path}' && " f"python {script} {config}"
        else:
            raise ValueError(f"Unknown venv type '{self.venv}'. Expected 'conda' or 'venv'.")
        return command

    def _handle_subprocess_error(self, cmd_result) -> None:
        """Handle subprocess failure by checking for error.json."""
        error_file = get_subprocess_error_path(self.config_path)

        if error_file.exists():
            try:
                with open(error_file) as file:
                    remote_exc_json = json.load(file)
                remote_exc = SubprocessError(**remote_exc_json)
                raise RuntimeError(
                    f"Subprocess raised {remote_exc.exception_type}: {remote_exc.message}\n\n"
                    f"--- Remote Traceback ---\n\n{remote_exc.traceback}"
                )
            except (json.JSONDecodeError, KeyError) as err:
                raise RuntimeError(
                    f"Subprocess failed and error report is corrupt: {err}\n" f"Stderr: {cmd_result.stderr}"
                ) from err
        else:
            raise RuntimeError(f"Subprocess Hard Crash (No Error Report):\n{cmd_result.stderr}")

    @abstractmethod
    def post_inference(self) -> None:
        """Post-processing after inference completes."""
        pass
