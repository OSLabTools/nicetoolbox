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
    "name": "py_feat",
    "label": "Py-Feat",
    "description": "Facial action units & emotion detection",
}


@detector_installer
def install() -> None:
    """Install Py-Feat detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("py_feat")
    remove_venv(venv_dir)

    print("Installing virtual environment for algorithm 'Py-Feat'...")
    run(f'uv venv "{venv_dir}" --python 3.11')
    print(f"Virtual environment created in {venv_dir}")

    py = venv_python("py_feat")

    print("Installing requirements for 'Py-Feat'...")
    run(
        f'uv pip install --python "{py}"'
        " torch==2.11.0+cu126 torchvision==0.26.0+cu126"
        " --index-url https://download.pytorch.org/whl/cu126"
        " --extra-index-url https://pypi.org/simple"
    )
    run(
        f'uv pip install --python "{py}"'
        " torchcodec==0.16.0+cpu"
        " --index-url https://download.pytorch.org/whl/cpu"
        " --extra-index-url https://pypi.org/simple"
    )
    reqs = method_detector_file("py_feat_requirements.txt")
    run(f'uv pip install --python "{py}" -r "{reqs}"')
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'Py-Feat' environment setup completed successfully.")
