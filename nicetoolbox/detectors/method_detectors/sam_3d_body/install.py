from ....installation.installer_utils import (
    NICETOOLBOX_CORE_DIR,
    check_git,
    detector_installer,
    method_detector_file,
    remove_venv,
    run,
    separator,
    venv_python,
    venv_root,
)

METADATA = {
    "name": "sam_3d_body",
    "label": "SAM 3D Body",
    "description": "3D body mesh estimation",
}


@detector_installer
def install() -> None:
    """Install SAM 3D Body detector virtual environment and dependencies."""
    separator(f"Installing {METADATA['label']}")

    venv_dir = venv_root("sam_3d_body")
    remove_venv(venv_dir)

    print("Creating SAM 3D Body venv at sam_3d_body ...")
    run(f'uv venv "{venv_dir}" --python 3.10')

    # Clone the SAM 3D Body fork if not already present
    repo_dir = venv_root("sam_3d_body") / "src" / "sam-3d-body"
    if not repo_dir.exists():
        if not check_git():
            raise RuntimeError(
                "git is required to install SAM 3D Body but was not found.\n"
                "Download git from https://git-scm.com/downloads and re-run."
            )
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print("Cloning SAM 3D Body fork...")
        run(f'git clone https://github.com/OSLabTools/sam-3d-body "{repo_dir}"')
        run(f'git -C "{repo_dir}" checkout --detach b5c765a0d89d789985e186d396315e7590887b94')

    py = venv_python("sam_3d_body")

    print("Installing PyTorch (2.8.0, cu129)...")
    run(
        f'uv pip install --python "{py}"'
        " setuptools<82 torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129"
        " --index-url https://download.pytorch.org/whl/cu129"
        " --extra-index-url https://pypi.org/simple"
    )
    print("Installing SAM 3D Body dependencies...")
    reqs = method_detector_file("sam_3d_body_pip_requirements.txt")
    run(f'uv pip install --python "{py}" -r "{reqs}"')
    print("Installing Detectron2...")
    run(
        f'uv pip install --python "{py}"'
        " detectron2==0.6+fd27788pt2.8.0cu129"
        " --extra-index-url https://miropsota.github.io/torch_packages_builder"
        " --no-deps"
    )
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("'SAM 3D Body' environment setup completed successfully.")
