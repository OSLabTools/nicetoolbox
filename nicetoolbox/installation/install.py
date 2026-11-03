"""
NICE Toolbox Installer
======================
Main installer script for:
  - Create venv and install nice toolbox
  - Call detector install scripts
  - Config file generation (machine_specific_paths.toml, nice_project.toml)
  - Example dataset download
  - Model asset download

Flags:
    -y, --yes            Don't prompt; accept defaults (all detectors unless --detectors is given)
    --detectors=a,b,c    Install only these detectors, without prompting
    --core-only         Skip all detectors
    --dev                Also install dev extras (nicetoolbox is always installed editable)
    --skip-dataset       Skip example dataset download
    --skip-assets        Skip model asset download
"""

import argparse
import importlib.util
import platform
import sys
import urllib.request
import zipfile

from .installer_utils import (
    IS_WINDOWS,
    LOG_FILE,
    NICETOOLBOX_CORE_DIR,
    REPO_ROOT,
    check_git,
    check_nvidia_driver,
    remove_venv,
    run,
    separator,
    venv_exe,
    venv_python,
    venv_root,
)

# ---------------------------------------------------------------------------
# Dynamic detector discovery
# ---------------------------------------------------------------------------


def discover_detectors() -> tuple[list[dict], dict[str, callable]]:
    """
    Dynamically discover all detector install scripts in
    nicetoolbox/detectors/method_detectors/*/install.py.
    """
    detectors_dir = REPO_ROOT / "nicetoolbox" / "detectors" / "method_detectors"
    discovered_meta: dict[str, dict] = {}
    discovered_fn: dict[str, callable] = {}

    if detectors_dir.exists():
        for script in sorted(detectors_dir.glob("*/install.py")):
            module_name = f"nicetoolbox.detectors.method_detectors.{script.parent.name}.install"
            try:
                # find the script and load it as a module with module_name
                spec = importlib.util.spec_from_file_location(module_name, script)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                meta = getattr(module, "METADATA", None)
                install_fn = getattr(module, "install", None)
                if meta and isinstance(meta, dict) and callable(install_fn):
                    name = meta.get("name")
                    if name:
                        discovered_meta[name] = meta
                        discovered_fn[name] = install_fn
            except Exception as e:
                print(f"WARNING: Failed to load detector installer at {script}: {e}")

    ordered_list = sorted(discovered_meta.values(), key=lambda m: m.get("label", m["name"]).lower())

    return ordered_list, discovered_fn


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------


def check_prerequisites() -> None:
    """Check for required tools (stop if missing) and optional ones (warn if missing)."""
    separator("Checking prerequisites")

    # TODO: install git, same style as uv/ffmpeg?
    if not check_git():
        # uv calls the system git for every git+https:// requirement, used by most detectors
        print("ERROR: git not found. It is required to install the detectors.")
        print("Download: https://git-scm.com/downloads")
        print("Install git, open a new terminal and re-run the installer.")
        sys.exit(1)

    # PyTorch wheels ship their own CUDA runtime, so only the driver is needed.
    # A missing GPU does not break the install, only running GPU detectors later.
    if not check_nvidia_driver():
        print("WARNING: nvidia-smi not found. You may need to update an NVIDIA driver.")
        print("Download: https://www.nvidia.com/en-us/drivers/")
        return

    # TODO: check long paths on Windows?

    print("All prerequisites found.")


# ---------------------------------------------------------------------------
# Config file creation
# ---------------------------------------------------------------------------

DEFAULT_MACHINE_SPECIFICS = REPO_ROOT / "machine_specific_paths.toml"
DEFAULT_PROJECT_CONFIG = REPO_ROOT / "nice_project.toml"

DEFAULT_CONFIG_FOLDER = "<project_folder_path>/configs"
DEFAULT_DATASETS_FOLDER = "../datasets"
DEFAULT_OUTPUTS_FOLDER = "../outputs"


def create_project_config() -> None:
    # TODO: refactor with project wizard
    if DEFAULT_PROJECT_CONFIG.exists():
        print(f"{DEFAULT_PROJECT_CONFIG.name} already exists, skipping.")
        return
    separator("Creating nice_project.toml")
    DEFAULT_PROJECT_CONFIG.write_text(
        "# Project-specific paths configuration.\n"
        "# Use <project_folder_path> to reference paths relative to this file's folder.\n"
        "\n"
        "# Path to the directory in which all configuration files are stored\n"
        f"configs_folder_path = '{DEFAULT_CONFIG_FOLDER}'\n"
        "\n"
        "# Path to the directory in which all datasets are stored\n"
        f"datasets_folder_path = '{DEFAULT_DATASETS_FOLDER}'\n"
        "\n"
        "# Directory for saving toolbox output\n"
        f"output_folder_path = '{DEFAULT_OUTPUTS_FOLDER}'\n"
    )
    print(f"Created {DEFAULT_PROJECT_CONFIG}")


def create_machine_specifics() -> None:
    if DEFAULT_MACHINE_SPECIFICS.exists():
        print(f"{DEFAULT_MACHINE_SPECIFICS.name} already exists, skipping.")
        return
    separator("Creating machine_specific_paths.toml")
    DEFAULT_MACHINE_SPECIFICS.write_text(
        "# Optional Hugging Face token for gated Hub models (e.g. sam_3d_body). Leave empty if unused.\n"
        "hugging_face_token = ''\n"
        "# Hardware acceleration for video decoding (ffmpeg '-hwaccel'), e.g. 'cuda', 'auto', 'd3d11va'.\n"
        "# Set to 'none' to decode on the CPU, e.g. on machines without a usable NVIDIA GPU.\n"
        "ffmpeg_hwaccel = 'cuda'\n"
    )
    print(f"Created {DEFAULT_MACHINE_SPECIFICS}")


# ---------------------------------------------------------------------------
# Main nicetoolbox venv
# ---------------------------------------------------------------------------


def install_nicetoolbox_venv(dev: bool = False) -> None:
    separator("Installing nicetoolbox venv")
    venv_dir = venv_root("nicetoolbox")

    remove_venv(venv_dir)

    print(f"Creating Python 3.10 venv at {venv_dir} ...")
    run(f'uv venv "{venv_dir}" --python 3.10')

    py = venv_python("nicetoolbox")

    print("Installing nicetoolbox-core (editable)...")
    run(f'uv pip install --python "{py}" -e "{NICETOOLBOX_CORE_DIR}"')

    print("Installing nicetoolbox (editable)...")
    extras = ""
    if dev:
        print("Dev mode: including dev extras (testing, linting, docs).")
        extras = "[dev]"

    run(f'uv pip install --python "{py}" -e ".{extras}"')
    print("nicetoolbox venv installed successfully.")


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

# TODO: move this to assets manager?
EXAMPLE_DATASET_NAME = "communication_multiview"
EXAMPLE_DATASET_URL = "https://keeper.mpdl.mpg.de/seafhttp/f/bbf0b44a2df34af685af/?op=view"


def download_dataset() -> None:
    # Same folder nice_project.toml points to: relative to the repo root, not the working directory
    datasets_dir = (REPO_ROOT / DEFAULT_DATASETS_FOLDER).resolve()
    dataset_dir = datasets_dir / EXAMPLE_DATASET_NAME
    if dataset_dir.exists():
        print(f"Example dataset already present at {dataset_dir}, skipping.")
        return

    separator("Downloading example dataset")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    zip_path = REPO_ROOT / f"{EXAMPLE_DATASET_NAME}.zip"

    print(f"Downloading from {EXAMPLE_DATASET_URL} ...")
    urllib.request.urlretrieve(EXAMPLE_DATASET_URL, zip_path)

    print(f"Extracting to {datasets_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(datasets_dir)

    zip_path.unlink()
    print(f"Example dataset ready at {dataset_dir}")


# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------


def download_assets() -> None:
    separator("Downloading model assets")
    exe = venv_exe("nicetoolbox", "download_assets")
    if not exe.exists():
        print(f"ERROR: {exe} not found. Run the nicetoolbox venv install first.")
        return
    run(f'"{exe}"')


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def select_detectors_interactive(detectors: list[dict]) -> list[dict]:
    """Print a numbered checklist and let the user toggle detectors."""
    print()
    print("Select detectors to install (all selected by default).")
    print("Enter numbers to toggle, 'a' to select all, 'n' to select none,")
    print("or press Enter to confirm the current selection.")
    print()

    selected = [True] * len(detectors)

    while True:
        for i, det in enumerate(detectors):
            mark = "x" if selected[i] else " "
            print(f"[{mark}] {i + 1:2d}. {det['label']:<20} - {det['description']}")
        print()
        choice = input("Toggle (number), 'a'=all, 'n'=none, Enter=confirm: ").strip().lower()

        if choice == "":
            break
        elif choice == "a":  # noqa: RET508
            selected = [True] * len(detectors)
        elif choice == "n":
            selected = [False] * len(detectors)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(detectors):
                    selected[idx] = not selected[idx]
                else:
                    print(f"Invalid number. Enter 1-{len(detectors)}.")
            except ValueError:
                print("Invalid input.")
        print()

    chosen = [det for det, sel in zip(detectors, selected) if sel]
    print()
    if chosen:
        print(f"Will install: {', '.join(d['label'] for d in chosen)}")
    else:
        print("No detectors selected.")
    print()
    return chosen


def select_detectors_from_args(names: str, detectors: list[dict]) -> list[dict]:
    """Parse a comma-separated list of detector names (from --detectors)."""
    name_set = {n.strip() for n in names.split(",") if n.strip()}
    chosen = [d for d in detectors if d["name"] in name_set]
    unknown = name_set - {d["name"] for d in detectors}
    if unknown:
        print(f"WARNING: Unknown detector names ignored: {', '.join(sorted(unknown))}")
    return chosen


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(detectors: list[dict]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NICE Toolbox installer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Non-interactive mode. Installs all detectors unless --detectors is given.",
    )
    available_names = ", ".join(d["name"] for d in detectors)
    parser.add_argument(
        "--detectors",
        default=None,
        help=f"Comma-separated list of detector names to install, without prompting. Available: {available_names}",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip venvs that are already fully installed. By default, selected venvs are rebuilt.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Install only the main nicetoolbox venv; skip all detectors.",
    )
    parser.add_argument(
        "-d",
        "--dev",
        action="store_true",
        help="Also install dev extras (testing, linting, docs).",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip downloading the example dataset.",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip downloading model assets after install.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first detector that fails to install, instead of continuing with the rest (e.g. for CI).",
    )
    return parser.parse_args()


def main() -> None:
    detectors, install_fn_map = discover_detectors()
    args = parse_args(detectors)
    interactive = not (args.yes or args.core_only or args.detectors is not None)

    separator("NICE Toolbox Installer")
    print(f"Install root : {REPO_ROOT}")
    print(f"Platform     : {platform.system()} {platform.machine()}")
    print(f"Mode         : {'interactive' if interactive else 'non-interactive'}")
    print(f"Error log    : {LOG_FILE}  (written only on failure)")
    print()

    # --- Prerequisites ---
    check_prerequisites()

    # --- Config files ---
    create_machine_specifics()
    create_project_config()

    # --- Main venv ---
    if args.only_missing and venv_root("nicetoolbox").exists():
        print("nicetoolbox venv already installed, skipping (--only-missing).")
    else:
        install_nicetoolbox_venv(dev=args.dev)

    # --- Example dataset ---
    if not args.skip_dataset:
        download_dataset()

    # --- Detector selection ---
    if args.core_only:
        detectors_to_install: list[dict] = []
        print("--core-only specified: skipping all detectors.")
    elif args.detectors is not None:
        detectors_to_install = select_detectors_from_args(args.detectors, detectors)
    elif args.yes:
        detectors_to_install = detectors
        print("--yes specified: installing all detectors.")
    else:
        detectors_to_install = select_detectors_interactive(detectors)

    # --- Install detectors ---
    # Detector venvs live in envs/<METADATA name>
    failed: list[str] = []
    skipped: list[str] = []
    for det in detectors_to_install:
        if args.only_missing and venv_root(det["name"]).exists():
            print(f"{det['label']} already installed, skipping (--only-missing).")
            skipped.append(det["label"])
            continue
        try:
            install_fn = install_fn_map[det["name"]]
            install_fn()
        except Exception as exc:
            print(f"\nERROR: {det['label']} installation failed: {exc}")
            failed.append(det["label"])
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(f"{'=' * 60}\n")
                f.write(f"FAILED DETECTOR: {det['label']}\n")
                f.write(f"ERROR: {exc}\n")
            if args.fail_fast:
                print("Stopping (--fail-fast).")
                sys.exit(1)
            print("Continuing with remaining detectors...\n")

    # --- Download model assets ---
    if not args.skip_assets and not args.core_only:
        download_assets()

    # --- Summary ---
    separator("Installation complete")
    installed = [d["label"] for d in detectors_to_install if d["label"] not in failed + skipped]
    if installed:
        print(f"Installed: {', '.join(installed)}")
    if skipped:
        print(f"Skipped (already installed): {', '.join(skipped)}")
    if failed:
        print(f"ERROR: Failed: {', '.join(failed)}")
        if LOG_FILE.exists():
            print(f"Error details written to: {LOG_FILE}")
    print()
    print("To run the toolbox:")
    print(f"cd {REPO_ROOT}")
    if IS_WINDOWS:
        print(r"envs\nicetoolbox\Scripts\activate")
    else:
        print("source envs/nicetoolbox/bin/activate")
    print("run_detectors")
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
