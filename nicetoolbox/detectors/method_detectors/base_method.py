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

from nicetoolbox_core.entrypoint import (
    SUBPROCESS_MANIFEST_ADAPTER,
    SubprocessFail,
    SubprocessManifest,
    SubprocessResult,
    get_subprocess_result_path,
)

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
    env_name: str
    env_root: Path
    script_path: str

    # method specific settings
    visualize: bool
    requires_out_folder: bool
    algorithm_out_folder: str
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

        # (3) Component-Centric Result Folders (<component_name>/)
        self.result_folders = self.compute_result_folders()
        self.viz_folders = self.compute_viz_folders(self.visualize)

        # (4) Algorithm-Centric Sandbox Paths (_native_outputs/<cur_algorithm_name>/)
        algorithm_folder_path = self.io.get_algorithm_output_folder(self.algorithm_instance, "detector_native_folder")
        self.algorithm_folder = str(algorithm_folder_path)
        algorithm_out_folder_path = self.io.get_algorithm_output_folder(self.algorithm_instance, "output")
        self.algorithm_out_folder = str(algorithm_out_folder_path)
        native_viz_folder = self.io.get_algorithm_output_folder(self.algorithm_instance, "native_visualization")
        self.native_viz_folder = str(native_viz_folder)
        self.subprocess_log_path = algorithm_folder_path / "subprocess.log"

        self.runtime = self._initialize_detector()  # Overloaded by child detectors

        # (5) Flatten config for subprocess
        inference_config = flatten_inference_config(self.detector_config, self.runtime)

        # (6) Save config for subprocess ONCE under algorithm sandbox (_native_outputs/)
        config_folder = self.io.get_algorithm_output_folder(self.algorithm_instance, "run_config")
        self.config_path = config_folder / "run_config.toml"
        save_config(inference_config, self.config_path)

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
            algorithm_folder=self.algorithm_folder,
            algorithm_out_folder=self.algorithm_out_folder,
            native_viz_folder=self.native_viz_folder,
            viz_folders=self.viz_folders,
            algorithm=self.algorithm_instance,
            visualize=self.visualize,
            subjects_descr=self.data.subjects_descr,
            log_level=self.subsequence_context.log_level,
            calibration=self.data.calibration,
            input_recipes=self.data.get_input_recipes(),
        )

    def get_inference_script(self) -> Path | str:
        """
        Overridable helper method hook for downstream detectors to specify
        their inference script path in code.
        """
        return f"{self.detector_config.algorithm_type}_inference.py"

    def _resolve_inference_script(self) -> Path:
        """
        Resolve the inference script path relative to the concrete subclass's file.
        Default convention: <same_folder_as_class>/<algorithm_type>_inference.py
        Uses get_inference_script() hook to override it
        """
        class_file = Path(inspect.getfile(type(self)))
        detector_folder = class_file.parent
        file_name = self.get_inference_script()

        script = detector_folder / file_name
        if not script.is_file():
            logging.error(f"Detector '{self.algorithm_instance}' inference script not found at: {script}")
            raise FileNotFoundError(script)
        return script

    def _resolve_env_root(self) -> Path:
        """
        Resolve the folder holding this detector's environment.

        Third-party detectors are not installed by default, so a missing environment
        is an expected user-facing state rather than a bug.

        Returns:
            Path: The environment folder.

        Raises:
            FileNotFoundError: If the environment is not installed.
        """
        env_root = self.io.code_folder / "envs" / self.env_name
        if not env_root.is_dir():
            raise FileNotFoundError(
                f"Detector '{self.algorithm_instance}': the environment "
                f"'{self.env_name}' is not installed.\n\n"
                f"Expected at: {env_root}"
            )
        return env_root

    def _setup_subprocess_settings(self) -> None:
        """Setup OS-specific subprocess execution settings."""
        self.os_type = detect_os_type()

        self.env_name = getattr(self.detector_config, "env_name", "nicetoolbox")  # Default to nicetoolbox env

        self.script_path = self._resolve_inference_script()
        self.env_root = self._resolve_env_root()

    def _subprocess_env(self) -> dict:
        """Copy os.environ and set HF_TOKEN from machine_specific_paths.toml when configured."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        tok = effective_hf_hub_token(self.subsequence_context.machine)
        if tok:
            env["HF_TOKEN"] = tok
        return env

    def _load_result_manifest(self) -> SubprocessManifest:
        """Load and validate result.json written by entrypoint. Raises if missing or unparsable."""
        result_file = get_subprocess_result_path(self.config_path)
        with open(result_file) as f:
            return SUBPROCESS_MANIFEST_ADAPTER.validate_python(json.load(f))

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

    def _run_inference(self) -> DetectorOutput | None:
        """Run the inference subprocess."""
        logging.info(f"INFERENCE: Launching {self.algorithm_instance} subprocess...")

        # debugging shortcut
        if self.subsequence_context.run.skip_inference:
            logging.warning(
                f"INFERENCE: skip_inference is set - reusing existing {self.algorithm_instance} "
                "inference output and running post-processing only."
            )
            result_manifest = self._load_result_manifest()
            return self.post_inference(result_manifest)

        command = self._create_command()

        # delete result manifest from previous runs, before starting processing
        result_file = get_subprocess_result_path(self.config_path)
        result_file.unlink(missing_ok=True)

        sub_env = self._subprocess_env()
        if self.os_type == "windows":
            popen_kwargs = {"shell": True, "env": sub_env}
        else:
            popen_kwargs = {"shell": True, "executable": "/bin/bash", "env": sub_env}

        with open(self.subprocess_log_path, "w", encoding="utf-8") as f_log:
            process = subprocess.Popen(
                command,
                stdout=f_log,
                stderr=f_log,
                **popen_kwargs,
            )
            returncode = process.wait()

        if returncode == 0:
            logging.info(f"INFERENCE: {self.algorithm_instance} finished successfully (Exit 0).")
            result_manifest = self._load_result_manifest()
            return self.post_inference(result_manifest)

        logging.error(f"INFERENCE: {self.algorithm_instance} subprocess failed (Exit 1).")
        self._handle_subprocess_error()
        return None

    def _create_command(self) -> str:
        """Create the shell command to run inference."""
        script = f'"{self.script_path}"'
        config = f'"{self.config_path}"'

        python_bin = self.env_root / ("Scripts/python.exe" if self.os_type == "windows" else "bin/python")

        if self.os_type == "windows":
            command = f'"{python_bin}" {script} {config}'
        else:
            command = f"'{python_bin}' {script} {config}"
        return command

    def _handle_subprocess_error(self) -> None:
        """Handle subprocess failure by checking result.json for error report."""
        if get_subprocess_result_path(self.config_path).exists():
            manifest = self._load_result_manifest()
            if isinstance(manifest, SubprocessFail):
                remote_exc = manifest.error
                raise RuntimeError(
                    f"Subprocess raised {remote_exc.exception_type}: {remote_exc.message}\n\n"
                    f"--- Remote Traceback ---\n\n{remote_exc.traceback}"
                )

        # In the case of a hard crash (out of memory, segFault, etc.),
        # the subprocess may not have written a result.json file.
        # In that case, we try to read the last 30 lines of the log file for clues.
        try:
            log_text = self.subprocess_log_path.read_text(encoding="utf-8")
            stderr = "\n".join(log_text.splitlines()[-30:])
        except Exception:
            pass
        raise RuntimeError(f"Subprocess Hard Crash (No Error Report):\n{stderr}")

    @abstractmethod
    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput | None:
        """Post-processing after inference completes."""
        pass
