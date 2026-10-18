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
    "name": "spiga",
    "label": "SPIGA",
    "description": "Head pose & facial landmarks",
}


@detector_installer
def install() -> None:
    """Install SPIGA detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("spiga")
    remove_venv(venv_dir)

    print("Installing virtual environment for algorithm 'SPIGA'...")
    print("Creating virtual environment...")
    run(f'uv venv "{venv_dir}" --python 3.10')
    print(f"Virtual environment created in {venv_dir}")

    py = venv_python("spiga")

    print("Installing requirements for 'SPIGA'...")
    run(
        f'uv pip install --python "{py}"'
        " setuptools<82 torch==2.1.0+cu118 torchvision==0.16.0+cu118"
        " --index-url https://download.pytorch.org/whl/cu118"
        " --extra-index-url https://pypi.org/simple"
    )
    reqs = method_detector_file("spiga_requirements.txt")
    run(f'uv pip install --python "{py}" -r "{reqs}"')
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'SPIGA' environment setup completed successfully.")
