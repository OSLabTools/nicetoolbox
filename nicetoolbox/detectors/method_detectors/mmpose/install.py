from ....installation.installer_utils import (
    NICETOOLBOX_CORE_DIR,
    detector_installer,
    method_detector_file,
    remove_venv,
    run,
    separator,
    venv_exe,
    venv_python,
    venv_root,
)

METADATA = {
    "name": "openmmlab",
    "label": "MMPose",
    "description": "Body joint estimation via MMPose",
}


@detector_installer
def install() -> None:
    """Install MMPose / openmmlab detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("openmmlab")
    remove_venv(venv_dir)

    print("Installing environment for submodule 'MMPose'...")
    print("Creating virtual environment at envs/openmmlab with Python 3.8 using uv...")
    run(f'uv venv "{venv_dir}" --python 3.8')

    py = venv_python("openmmlab")

    # Install PyTorch with CUDA
    print("Installing PyTorch and dependencies...")
    run(
        f'uv pip install --python "{py}"'
        " torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0"
        " --index-url https://download.pytorch.org/whl/cu118"
        " --extra-index-url https://pypi.org/simple"
    )

    # Install nicetoolbox-core
    print("Installing nicetoolbox-core dependencies...")
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    # Install MMPose and its dependencies
    print("Installing MMPose and dependencies...")
    run(f'uv pip install --python "{py}" setuptools pip openmim fsspec mmengine pyparsing six toml protobuf==3.20.3')

    mim = venv_exe("openmmlab", "mim")
    print("Installing openmmlab packages via mim...")
    run(f'"{mim}" install mmcv==2.1.0')
    run(f'"{mim}" install mmdet>=3.1.0,<3.3.0')
    run(f'"{mim}" install mmpretrain>=1.0.0rc8')

    print("Installing MMPose from the OSLabTools fork...")
    reqs = method_detector_file("mmpose_requirements.txt")
    run(f'uv pip install --python "{py}" --no-build-isolation -r "{reqs}"')

    print("'MMPose' environment setup completed successfully.")
