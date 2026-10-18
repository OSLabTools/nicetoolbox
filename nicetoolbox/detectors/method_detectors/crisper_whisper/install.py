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
    "name": "crisper_whisper",
    "label": "CrisperWhisper",
    "description": "Precise speech transcription",
}


@detector_installer
def install() -> None:
    """Install CrisperWhisper detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("crisper_whisper")
    remove_venv(venv_dir)

    print("Installing virtual environment for algorithm 'CrisperWhisper'...")
    print("Creating virtual environment...")
    run(f'uv venv "{venv_dir}" --python 3.10')
    print(f"Virtual environment created in {venv_dir}")

    py = venv_python("crisper_whisper")

    print("Installing requirements for 'CrisperWhisper'...")
    run(
        f'uv pip install --python "{py}"'
        " torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126"
        " --index-url https://download.pytorch.org/whl/cu126"
        " --extra-index-url https://pypi.org/simple"
    )
    run(f'uv pip install --python "{py}" onnxruntime<1.24.0')
    reqs = method_detector_file("crisper_whisper_requirements.txt")
    run(f'uv pip install --python "{py}" -r "{reqs}"')
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'CrisperWhisper' environment setup completed successfully.")
