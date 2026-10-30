"""
Shared utility functions, constants, and execution helpers for NICE Toolbox installers.
"""

import datetime
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

INSTALLATION_DIR = Path(__file__).resolve().parent  # <repo>/nicetoolbox/installation
REPO_ROOT = INSTALLATION_DIR.parents[1]

# set our working directory to the parent folder, aka nicetoolbox root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENVS_DIR = REPO_ROOT / "envs"
NICETOOLBOX_CORE_DIR = REPO_ROOT / "nicetoolbox_core"
TRASH_DIR = ENVS_DIR / ".trash"

_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
LOG_FILE = REPO_ROOT / f"installation_error_{_ts}.log"

IS_WINDOWS = platform.system() == "Windows"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


# ---------------------------------------------------------------------------
# File system & venv helpers
# ---------------------------------------------------------------------------


def _remove_readonly(func, path, _excinfo) -> None:
    """Error handler for shutil.rmtree on Windows: clears read-only flag and retries."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree(path: Path) -> None:
    """Cross-platform rmtree that handles read-only files (e.g. inside .git dirs on Windows)."""
    shutil.rmtree(path, onerror=_remove_readonly)


def remove_venv(venv_dir: Path) -> None:
    """Delete an existing venv directory if present, handling read-only files on Windows.

    Windows refuses to delete a running executable (e.g. an IDE language server started from
    the venv's python.exe) but allows renaming its folder. In that case the venv is moved to
    envs/.trash, which is emptied on later runs once nothing uses it anymore.
    """
    shutil.rmtree(TRASH_DIR, ignore_errors=True)  # leftovers from earlier runs, if now unlocked

    if not venv_dir.exists():
        return
    print(f"Removing existing virtual environment at {venv_dir} ...")
    try:
        rmtree(venv_dir)
    except PermissionError:
        target = TRASH_DIR / f"{venv_dir.name}_{_ts}"
        try:
            TRASH_DIR.mkdir(parents=True, exist_ok=True)
            venv_dir.rename(target)
        except OSError:
            raise RuntimeError(
                f"Cannot remove {venv_dir}: some of its files are in use.\n"
                "Close programs using this environment (VS Code, Jupyter, terminals with it activated)\n"
                "and re-run the installer."
            ) from None
        print("WARNING: some files are in use, moved the old environment to")
        print(f"{target}")
        print("Restart programs that use it after the installation. It is deleted on the next run.")


def separator(title: str = "") -> None:
    print()
    print("*" * 60)
    if title:
        print(f"{title}")
        print("*" * 60)
    print()


def run(cmd: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming stdout live. On failure, stderr is written to LOG_FILE.

    Paths in `cmd` must be wrapped in double quotes, e.g. f'uv venv "{venv_dir}"', so that
    paths containing spaces stay a single argument.
    """
    print(f"$ {cmd}")
    # env variable below is necessary so that uv considers all links for downloading a package,
    # if you delete it uv will first look at --index_url
    # which will throw an error for torch cuda installations
    env = os.environ.copy()
    env["UV_INDEX_STRATEGY"] = "unsafe-best-match"
    try:
        return subprocess.run(
            # Windows parses the quotes itself when given a command string (shlex's non-posix
            # mode would keep them as part of the path); on Linux split it like a shell does.
            cmd if IS_WINDOWS else shlex.split(cmd),
            cwd=str(cwd or REPO_ROOT),
            check=check,
            env=env,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        err_text = exc.stderr.decode(errors="replace")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"FAILED COMMAND: {cmd}\n")
            f.write(f"EXIT CODE: {exc.returncode}\n")
            f.write(f"STDERR:\n{err_text}\n")
        print(err_text)
        raise


def venv_root(venv_name: str) -> Path:
    """Return the root folder of a named venv, e.g. envs/spiga."""
    return ENVS_DIR / venv_name


def venv_exe(venv_name: str, exe: str) -> Path:
    """Return the path to an executable inside a named venv."""
    bin_dir = "Scripts" if IS_WINDOWS else "bin"
    return venv_root(venv_name) / bin_dir / f"{exe}{EXE_SUFFIX}"


def venv_python(venv_name: str) -> Path:
    """Return the path to the Python executable inside a named venv."""
    return venv_exe(venv_name, "python")


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def check_git() -> bool:
    return shutil.which("git") is not None


def check_nvidia_driver() -> bool:
    """Best-effort NVIDIA driver check: nvidia-smi ships with the driver (no CUDA Toolkit needed)."""
    return shutil.which("nvidia-smi") is not None


def method_detector_file(name: str) -> Path:
    """Return the path to a file next to the calling script, e.g. its requirements file."""
    caller_file = sys._getframe(1).f_code.co_filename
    return Path(caller_file).resolve().with_name(name)


def detector_installer(install_fn: Callable[[], None]) -> Callable[[], None]:
    """Decorator for a detector's install() function.

    Detector installers are loaded and called by nicetoolbox/installation/install.py. If the file is run
    directly instead, its module is "__main__": print a notice and exit.
    """
    if install_fn.__module__ == "__main__":
        name = Path(install_fn.__code__.co_filename).parent.name
        print(f"Detector installer '{name}' is intended to be called by the NICE Toolbox installer.")
        sys.exit(1)
    return install_fn
