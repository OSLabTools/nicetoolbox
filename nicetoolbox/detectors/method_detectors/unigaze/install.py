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
    "name": "unigaze",
    "label": "UniGaze",
    "description": "Gaze estimation",
}


@detector_installer
def install() -> None:
    """Install UniGaze detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("unigaze")
    remove_venv(venv_dir)

    print("Creating virtual environment for algorithm 'UniGaze'...")
    run(f'uv venv "{venv_dir}" --python 3.10')
    print(f"Virtual environment created in {venv_dir}")

    py = venv_python("unigaze")

    print("Installing requirements for 'UniGaze'...")
    run(
        f'uv pip install --python "{py}"'
        " torch==2.0.1+cu118 torchvision==0.15.2+cu118"
        " --index-url https://download.pytorch.org/whl/cu118"
        " --extra-index-url https://pypi.org/simple"
    )
    reqs = method_detector_file("unigaze_requirements.txt")
    run(f'uv pip install --python "{py}" -r "{reqs}"')
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("UniGaze environment setup completed successfully.")
