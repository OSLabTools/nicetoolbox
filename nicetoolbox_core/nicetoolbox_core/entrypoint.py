import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict

import toml


@dataclass
class SubprocessError:
    """
    A safe-to-json wrapper for exceptions raised in subprocesses.
    """

    exception_type: str
    message: str
    traceback: str


def run_inference_entrypoint(main_function: Callable[[Dict[str, Any]], None]) -> None:
    """
    Decorator for the main inference function.

    Handles CLI args, Config loading, Logging, and Error Pickling.

    Args:
        main_function (Callable[[Dict[str, Any]], None]): The main inference function.

    Raises:
        SystemExit: Exits with code 0 on success, 1 on failure.
    """

    @wraps(main_function)  # Helper decorator to preserve metadata of original func
    def wrapper():
        if main_function.__module__ != "__main__":
            logging.warning(
                "You are accidentally importing an inference script inside the nicetoolbox code. "
                "It should always be run in its own venv/conda"
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
                filename=config.get("log_file"),
                level=config.get("log_level", "INFO"),
                format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
            )

            # 3. Execute
            main_function(config)

            sys.exit(0)

        except Exception as e:
            # 1. Log locally
            logging.critical("Inference Failed", exc_info=True)

            # 2. Wrap in dataclass
            safe_error = SubprocessError(
                exception_type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
            )

            # Save error.json next to the config file (standard detector output location)
            error_file = config_path.parent / "error.json"
            try:
                with open(error_file, "w") as file:
                    json.dump(asdict(safe_error), file, indent=4)
            except Exception as io_err:
                print(
                    f"CRITICAL: Failed to write error to json: {io_err}",
                    file=sys.stderr,
                )

            sys.exit(1)

    return wrapper()
