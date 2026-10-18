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
    "name": "insight_face",
    "label": "InsightFace",
    "description": "Face detection",
}


@detector_installer
def install() -> None:
    """Install InsightFace detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("insight_face")
    remove_venv(venv_dir)

    print("Installing virtual environment for algorithm 'InsightFace'...")
    print("Creating virtual environment...")
    run(f'uv venv "{venv_dir}" --python 3.10')
    print(f"Virtual environment created in {venv_dir}")

    py = venv_python("insight_face")

    print("Installing Pytorch...")
    run(
        f'uv pip install --python "{py}"'
        " torch==2.9.0+cu129"
        " --index-url https://download.pytorch.org/whl/cu129"
        " --extra-index-url https://pypi.org/simple"
    )
    print("Installing requirements for 'InsightFace'...")
    reqs = method_detector_file("insight_face_requirements.txt")
    run(f'uv pip install --python "{py}" -r "{reqs}"')
    run(f'uv pip install --python "{py}" --no-deps onnxruntime-gpu==1.23.2')
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'InsightFace' environment setup completed successfully.")
