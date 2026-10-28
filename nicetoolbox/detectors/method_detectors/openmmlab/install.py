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
    "name": "openmmlab",
    "label": "MMPose, MMDet (OpenMMLab)",
    "description": "Person detection and body joint estimation",
}


@detector_installer
def install() -> None:
    """Install the OpenMMLab (MMDet, MMPose) virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("openmmlab")
    remove_venv(venv_dir)

    print("Installing environment for 'OpenMMLab' (MMDet, MMPose)...")
    print("Creating virtual environment at envs/openmmlab with Python 3.8 using uv...")
    run(f'uv venv "{venv_dir}" --python 3.8')

    py = venv_python("openmmlab")

    # Install PyTorch with CUDA
    print("Installing PyTorch (2.1.0, cu118)...")
    run(
        f'uv pip install --python "{py}"'
        " torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118"
        " --index-url https://download.pytorch.org/whl/cu118"
        " --extra-index-url https://pypi.org/simple"
    )

    # mmcv ships compiled native ops, install the prebuilt wheel matching torch/cuda
    print("Installing MMCV (2.1.0, cu118/torch2.1)...")
    run(
        f'uv pip install --python "{py}" mmcv==2.1.0'
        " --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html"
    )

    # mmdet and mmpose (git) and chumpy (sdist only) are built from source, all pure python;
    # their setup.py import torch/pip, so they are built inside the venv (--no-build-isolation)
    print("Installing MMDet, MMPose and their requirements...")
    run(f'uv pip install --python "{py}" setuptools pip')
    reqs = method_detector_file("openmmlab_requirements.txt")
    run(f'uv pip install --python "{py}" --no-build-isolation -r "{reqs}"')

    # Install nicetoolbox-core
    print("Installing nicetoolbox-core dependencies...")
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'OpenMMLab' environment setup completed successfully.")
