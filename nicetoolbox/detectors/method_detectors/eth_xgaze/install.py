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
    "name": "eth_xgaze",
    "label": "ETH-XGaze",
    "description": "Gaze estimation",
}


@detector_installer
def install() -> None:
    """Install ETH-XGaze detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("eth_xgaze")
    remove_venv(venv_dir)

    print("Creating virtual environment for submodule 'ETH-XGaze'...")
    run(f'uv venv "{venv_dir}" --python 3.10')
    print(f"Virtual environment created in {venv_dir}")

    py = venv_python("eth_xgaze")

    print("Installing requirements for 'ETH-XGaze'...")
    run(
        f'uv pip install --python "{py}"'
        " torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118"
        " --index-url https://download.pytorch.org/whl/cu118"
        " --extra-index-url https://pypi.org/simple"
    )
    reqs = method_detector_file("eth_xgaze_requirements.txt")
    run(f'uv pip install --python "{py}" -r "{reqs}"')
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'ETH-XGaze' environment setup completed successfully.")
