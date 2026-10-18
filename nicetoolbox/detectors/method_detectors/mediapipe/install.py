from ....installation.installer_utils import (
    NICETOOLBOX_CORE_DIR,
    detector_installer,
    method_detector_file,
    remove_venv,
    run,
    separator,
    venv_python,
    venv_root,
)

METADATA = {
    "name": "mediapipe",
    "label": "MediaPipe",
    "description": "Face landmarks",
}


@detector_installer
def install() -> None:
    """Install MediaPipe detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("mediapipe")
    remove_venv(venv_dir)

    print("Installing virtual environment for algorithm 'MediaPipe'...")
    print("Creating virtual environment...")
    run(f'uv venv "{venv_dir}" --python 3.10')
    print(f"Virtual environment created in {venv_dir}")

    py = venv_python("mediapipe")

    print("Installing requirements for 'MediaPipe'...")
    reqs = method_detector_file("mediapipe_requirements.txt")
    run(f'uv pip install --python "{py}" setuptools<82 -r "{reqs}"')
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'MediaPipe' environment setup completed successfully.")
