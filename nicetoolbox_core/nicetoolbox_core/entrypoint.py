import json
import logging
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import toml
from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Annotated, Literal

# =============================================================================
# Subprocess results schemas
# =============================================================================


class SubprocessResult(BaseModel):
    status: Literal["success"] = "success"
    artifacts: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_artifacts(cls, **artifacts: str) -> "SubprocessResult":
        return cls(artifacts=artifacts)

    def get_raw_output(self) -> Optional[str]:
        """Convenience accessor for the conventional single-file artifact."""
        return self.artifacts.get("raw_output")


class SubprocessError(BaseModel):
    """A safe-to-json wrapper for exceptions raised in subprocesses."""

    exception_type: str
    message: str
    traceback: str


class SubprocessFail(BaseModel):
    status: Literal["error"] = "error"
    error: SubprocessError


SubprocessManifest = Annotated[
    Union[SubprocessResult, SubprocessFail],
    Field(discriminator="status"),
]

SUBPROCESS_MANIFEST_ADAPTER = TypeAdapter(SubprocessManifest)

# =============================================================================
# Entry point functions
# =============================================================================


def get_subprocess_result_path(config_path: Path) -> Path:
    return config_path.parent / "result.json"


def _save_result_manifest(config_path: Path, manifest: dict) -> None:
    result_file = get_subprocess_result_path(config_path)
    try:
        with open(result_file, "w") as file:
            json.dump(manifest, file, indent=4)
    except Exception as io_err:
        print(f"CRITICAL: Failed to write result manifest to json: {io_err}", file=sys.stderr)
        raise


def run_inference_entrypoint(main_function: Callable) -> None:
    """
    Decorator for the main inference function.

    Handles CLI args, Config loading, Logging, and Error Pickling.

    Args:
        main_function (Callable): The main inference function.

    Raises:
        SystemExit: Exits with code 0 on success, 1 on failure.
    """

    @wraps(main_function)  # Helper decorator to preserve metadata of original func
    def wrapper():
        if main_function.__module__ != "__main__":
            logging.warning(
                "You are accidentally importing an inference script inside the nicetoolbox code. "
                "It should always be run in its own virtual environment"
            )
            return

        if len(sys.argv) < 2:
            print(
                "Invalid inference script call. Usage: python script.py <config_path>",
                file=sys.stderr,
            )
            sys.exit(1)

        config_path = Path(sys.argv[1])

        try:
            # 1. Load Config
            config = toml.load(config_path)

            # 2. Setup Logging
            logging.basicConfig(
                level=config.get("log_level", "INFO"),
                format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
            )

            # 3. Execute
            subprocess_result = main_function(config)

            _save_result_manifest(config_path, subprocess_result.model_dump(exclude_none=True))
            sys.exit(0)

        except Exception as e:
            # 1. Log locally
            logging.critical("Inference Failed", exc_info=True)

            # 2. Wrap in Pydantic error model
            safe_error = SubprocessError(
                exception_type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
            )

            # 3. Save error manifest inside result.json
            error_manifest = SubprocessFail(error=safe_error)
            try:
                _save_result_manifest(config_path, error_manifest.model_dump(exclude_none=True))
            except Exception as io_err:
                print(f"CRITICAL: Failed to write error manifest to json: {io_err}", file=sys.stderr)
            sys.exit(1)

    return wrapper()
