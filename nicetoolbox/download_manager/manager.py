import logging
import time
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

from ..configs.schemas.asset_manifest import AssetManifest
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..utils import logging_utils as log_ut
from ..utils.hf_token import effective_hf_hub_token


class AssetManager:
    def __init__(self, config):
        """
        Initializes the manager by extracting paths directly from the active configuration.
        """
        self.assets_root = Path(config.run_config.io.assets)

        manifest_path = config.run_config.io.asset_manifest
        manifest_model = config.cfg_loader.load_config(manifest_path, AssetManifest)

        self.manifest = {k: v.model_dump() for k, v in manifest_model.root.items()}

    def download_file(self, url: str, dest_path: Path, desc: str):
        """
        Streams a file from a URL to a local destination with a progress bar.
        Incorporates resume (.tmp) and robust retry logic for network drops.
        """

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.parent / (dest_path.name + ".tmp")

        max_retries = 3

        for attempt in range(max_retries):
            resume_header = {}
            mode = "wb"
            downloaded_bytes = 0

            # Calculate temp file size inside the loop so retries pick up exactly where they left off
            if temp_path.exists():
                downloaded_bytes = temp_path.stat().st_size
                if downloaded_bytes > 0:
                    resume_header = {"Range": f"bytes={downloaded_bytes}-"}
                    mode = "ab"

            try:
                # Added strict read timeouts (10s to connect, 30s max wait for next packet)
                response = requests.get(url, stream=True, headers=resume_header, timeout=(10, 30))

                if response.status_code == 416:
                    resume_header = {}
                    mode = "wb"
                    downloaded_bytes = 0
                    response = requests.get(url, stream=True, timeout=(10, 30))

                if response.status_code == 200:
                    downloaded_bytes = 0
                    mode = "wb"

                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                if response.status_code == 206:
                    total_size += downloaded_bytes
                else:
                    # Ensure total_size is accurate if server ignores Range and sends 200
                    total_size = int(response.headers.get("content-length", 0))

                with open(temp_path, mode) as file, tqdm(
                    desc=desc if attempt == 0 else f"{desc} (Resume attempt {attempt})",
                    total=total_size,
                    initial=downloaded_bytes,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = file.write(chunk)
                            bar.update(size)

                # If the loop finishes without an exception, the file is fully downloaded!
                temp_path.rename(dest_path)
                return  # Exit the function completely

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Network drop detected while downloading '{desc}'.")
                    logging.warning(f"Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
                else:
                    logging.error(f"Connection failed while downloading '{desc}'. Details: {e}")
                    logging.error(f"Please check internet or manually place file at: {dest_path}")
                    raise

    def _to_manifest_key(self, asset: str | Path) -> str:
        path = Path(asset)
        # Already a relative manifest key
        if not path.is_absolute():
            logging.debug(f"Asset path '{path}' is likely already a manifest key.")
            return path.as_posix()
        try:
            return path.relative_to(self.assets_root).as_posix()
            # as_posix makes it a string and keeps it Windows-friendly
        except ValueError as e:
            raise ValueError(f"Asset path '{path}' is outside assets root: {self.assets_root}") from e

    def verify_and_download(self, asset_keys: List[str]):
        """
        Checks if required assets exist, downloads them if missing.
        """
        log_ut.log_banner("Download Manager Assets Check")

        for key in set(asset_keys):
            key = self._to_manifest_key(key)
            if key not in self.manifest:
                logging.warning(f"Asset key '{key}' not found in asset_manifest.toml.")
                continue

            asset_info = self.manifest[key]
            dest_path = self.assets_root / key

            if not dest_path.exists():
                logging.info(f"Missing asset: {key}. Downloading...")
                self.download_file(asset_info["url"], dest_path, desc=key)
            else:
                logging.info(f"Asset '{key}' verified.")

    def ensure_assets_for_config(self, config):
        """
        Determines which assets are needed for the active run and downloads them.
        """
        required_assets = []

        active_algos = set(config.run_config.algorithms)

        # required_assets for those specific algorithms
        algos_dict = config.detectors_config.algorithms
        for algo in active_algos:
            algo_model = algos_dict.get(algo)
            if algo_model and hasattr(algo_model, "required_assets"):
                for val in algo_model.required_assets.values():
                    try:
                        # val will be cleaned in verify_and_download
                        required_assets.append(val)
                    except ValueError:
                        logging.warning(f"Path '{val}' is not inside the assets root!")

        logging.info(f"AssetManager check: Found {len(required_assets)} required assets for this run.")
        self.verify_and_download(required_assets)
        self._log_hf_only_models(active_algos, config.machine_specific_config)

    def _log_hf_only_models(self, active_algos: set, machine: MachineSpecificConfig) -> None:
        """Log Hugging Face token status for algorithms not on the Keeper manifest."""
        if "sam_3d_body" not in active_algos:
            return
        if effective_hf_hub_token(machine):
            logging.info("AssetManager: sam_3d_body uses Hugging Face weights (token configured).")
        else:
            logging.error(
                "AssetManager: sam_3d_body needs hugging_face_token in machine_specific_paths.toml "
                "(and Hub license acceptance)."
            )
