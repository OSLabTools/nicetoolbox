import logging
import shutil
import time
import zipfile
from pathlib import Path
from typing import List

import requests
from huggingface_hub import hf_hub_download, parse_hf_uri, snapshot_download
from huggingface_hub import logging as hf_logging
from huggingface_hub.errors import GatedRepoError, HfUriError, RepositoryNotFoundError
from huggingface_hub.file_download import repo_folder_name
from tqdm import tqdm

from ..configs.schemas.asset_manifest import AssetManifest
from ..utils import logging_utils as log_ut
from ..utils.hf_token import effective_hf_hub_token

# Silence Hugging Face Hub warnings/info and underlying httpx network logs
hf_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)


class AssetManager:
    def __init__(self, config):
        """
        Initializes the manager by extracting paths directly from the active configuration.
        """
        self.config = config  # Storing config to access machine config for tokens
        self.assets_root = Path(config.run_config.io.assets)

        manifest_path = config.run_config.io.asset_manifest
        manifest_model = config.cfg_loader.load_config(manifest_path, AssetManifest)

        self.manifest = {k: v.model_dump() for k, v in manifest_model.root.items()}

    def download_file(self, url: str, dest_path: Path, desc: str):
        """
        Streams a file from a URL to a local destination with a progress bar.
        Incorporates resume (.tmp) and robust retry logic for network drops.

        huggingface.co URLs are routed through the Hub client instead, which handles token
        auth, resume and the Hub's own transfer backend rather than the single-connection
        stream below.
        """
        if self.is_hf_url(url):
            self.download_hf_file(url, dest_path, desc=desc)
            return

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

    def download_and_extract_zip(self, url: str, dest_path: Path, desc: str, flatten_single_root: bool = False):
        zip_path = dest_path.parent / (dest_path.name + ".zip")
        staging_path = dest_path.parent / (dest_path.name + ".incomplete")

        self.download_file(url, zip_path, desc=desc)

        shutil.rmtree(staging_path, ignore_errors=True)
        try:
            staging_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                self._assert_safe_zip(archive, desc)
                archive.extractall(staging_path)
            if flatten_single_root:
                self._flatten_single_root(staging_path, desc)
            staging_path.rename(dest_path)
        except zipfile.BadZipFile as err:
            shutil.rmtree(staging_path, ignore_errors=True)
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded archive for '{desc}' is not a valid zip file.") from err
        except Exception:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise
        finally:
            zip_path.unlink(missing_ok=True)

        logging.info(f"Asset '{desc}' extracted to {dest_path}.")

    @staticmethod
    def _flatten_single_root(folder: Path, desc: str) -> None:
        entries = list(folder.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            logging.warning(
                f"Asset '{desc}' set flatten_single_root, but the archive does not have a single "
                f"top-level folder. Extracted contents left as-is."
            )
            return

        nested = entries[0]
        # Move via a sibling temp name: renaming the child onto its own parent is not portable.
        lifted = folder.parent / (folder.name + ".lifted")
        shutil.rmtree(lifted, ignore_errors=True)
        nested.rename(lifted)
        folder.rmdir()
        lifted.rename(folder)

    @staticmethod
    def _assert_safe_zip(archive: zipfile.ZipFile, desc: str) -> None:
        """
        Rejects archives whose members would escape the extraction folder.

        Guards against path traversal ('../') and absolute paths in member names, which
        `extractall` would otherwise honour and write outside the assets tree.
        """
        for member in archive.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Archive for '{desc}' contains an unsafe path: '{member}'.")

    def download_hf_repo(self, repo_id: str, cache_dir: Path, desc: str):
        """
        Downloads a Hugging Face repository using snapshot_download.
        Leverages HF's native resume and caching mechanisms.
        """
        token = effective_hf_hub_token(self.config.machine_specific_config)
        if not token:
            logging.warning(f"No HF token found in config. Attempting public download for '{repo_id}'.")

        logging.info(f"Downloading Hugging Face repository: {repo_id} to cache: {cache_dir}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # cache_dir is the parent folder (e.g. <assets>/whisperx)
                # huggingface_hub automatically structures it internally.
                snapshot_download(repo_id=repo_id, cache_dir=cache_dir, token=token)
                logging.info(f"HF repo '{repo_id}' successfully downloaded.")
                return
            except GatedRepoError:
                logging.error(
                    f"HF repo '{repo_id}' is gated. Accept the license on Hugging Face "
                    "and ensure a valid token is configured."
                )
                # HF will leave empty folder, which breaks exist asset check
                # we will try to delete it if exists
                repo_dir = Path(cache_dir) / repo_folder_name(repo_id=repo_id, repo_type="model")
                shutil.rmtree(repo_dir, ignore_errors=True)
                raise
            except RepositoryNotFoundError:
                logging.error(f"HF repo '{repo_id}' was not found. Check the repo ID in asset_manifest.toml.")
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Network drop while downloading HF repo '{desc}'. "
                        f"Retrying in 5s... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(5)
                else:
                    logging.error(f"Failed HF repo '{desc}' after {max_retries} attempts. Details: {e}")
                    raise

    @staticmethod
    def is_hf_url(url: str) -> bool:
        # check if this link is hugging face file
        try:
            parse_hf_uri(url)
            return True
        except HfUriError:
            return False

    def download_hf_file(self, url: str, dest_path: Path, desc: str):
        """
        Downloads a single file from a Hugging Face repo straight to its manifest path.
        The repo id, revision and in-repo path are parsed out of the URL.
        """
        uri = parse_hf_uri(url)

        token = effective_hf_hub_token(self.config.machine_specific_config)
        if not token:
            logging.warning(f"No HF token found in config. Attempting public download for '{uri.id}'.")

        # local_dir writes the real file (not a symlink into a blob cache), so the asset lands
        # where required_assets expects it. HF names it after its in-repo path, which must
        # therefore match the manifest key's basename.
        expected = dest_path.parent / Path(uri.path_in_repo).name
        if expected != dest_path:
            raise RuntimeError(
                f"Asset '{desc}' maps to in-repo file '{uri.path_in_repo}', which would be written "
                f"to '{expected}' instead of the manifest path '{dest_path}'. Rename the manifest key "
                f"to match the file name in the repo."
            )
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading HF file '{uri.path_in_repo}' from {uri.id} (revision: {uri.revision or 'main'})")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                hf_hub_download(
                    repo_id=uri.id,
                    filename=uri.path_in_repo,
                    revision=uri.revision,
                    repo_type=uri.type,
                    local_dir=dest_path.parent,
                    token=token,
                )
                logging.info(f"HF file '{desc}' successfully downloaded.")
                return
            except GatedRepoError:
                logging.error(
                    f"HF repo '{uri.id}' is gated. Accept the license on Hugging Face "
                    "and ensure a valid token is configured."
                )
                raise
            except RepositoryNotFoundError:
                logging.error(f"HF repo '{uri.id}' was not found. Check the URL in asset_manifest.toml.")
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Network drop while downloading HF file '{desc}'. "
                        f"Retrying in 5s... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(5)
                else:
                    logging.error(f"Failed HF file '{desc}' after {max_retries} attempts. Details: {e}")
                    raise

    def verify_and_download(self, asset_keys: List[str]):
        """
        Checks if required assets exist, downloads them if missing.
        """
        log_ut.log_banner("Download Manager Assets Check")

        for key in set(asset_keys):
            if key not in self.manifest:
                logging.warning(f"Asset key '{key}' not found in asset_manifest.toml.")
                continue

            asset_info = self.manifest[key]
            dest_path = self.assets_root / key

            source = asset_info.get("source")
            url = asset_info.get("url")

            if source == "huggingface":
                if not dest_path.exists():
                    logging.info(f"Missing HF asset: {key}. Downloading...")
                    self.download_hf_repo(repo_id=url, cache_dir=dest_path.parent, desc=key)
                else:
                    logging.info(f"HF Asset '{key}' verified.")

            elif source == "url":
                if not dest_path.exists():
                    logging.info(f"Missing asset: {key}. Downloading...")
                    self.download_file(url, dest_path, desc=key)
                else:
                    logging.info(f"Asset '{key}' verified.")

            elif source == "zip":
                if not dest_path.exists():
                    logging.info(f"Missing asset: {key}. Downloading and extracting...")
                    flatten = asset_info.get("flatten_single_root", False)
                    self.download_and_extract_zip(url, dest_path, desc=key, flatten_single_root=flatten)
                else:
                    logging.info(f"Asset '{key}' verified.")
            else:
                logging.warning(f"Manifest entry '{key}' is missing valid source type. Found: '{source}'")

    def _is_gated(self, key: str) -> bool:
        """True if the manifest marks this asset key as gated (restricted-access)."""
        info = self.manifest.get(key)
        return bool(info and info.get("access") == "gated")

    def _asset_present(self, key: str) -> bool:
        """True if the asset already exists locally. Mirrors verify_and_download's check."""
        return (self.assets_root / key).exists()

    def _gated_assets_by_algorithm(self) -> dict[str, list[str]]:
        """Maps each algorithm instance to its gated required asset keys (empty algos omitted)."""
        result: dict[str, list[str]] = {}
        for algo_name, algo_model in self.config.detectors_config.algorithms.items():
            if not hasattr(algo_model, "required_assets"):
                continue
            gated = []
            for val in algo_model.required_assets.values():
                try:
                    key = str(Path(val).relative_to(self.assets_root)).replace("\\", "/")
                except ValueError:
                    continue
                if self._is_gated(key):
                    gated.append(key)
            if gated:
                result[algo_name] = gated
        return result

    def unobtainable_gated_algorithms(self) -> set[str]:
        """
        Gated algorithms whose weights are unobtainable: at least one gated asset is missing
        locally AND no HF token is available to download it. Weights that are already present
        (e.g. download manually) are obtainable even without a token.
        """
        has_token = bool(effective_hf_hub_token(self.config.machine_specific_config))
        skip = set()
        for algo_name, gated_keys in self._gated_assets_by_algorithm().items():
            unobtainable = (not has_token) and any(not self._asset_present(k) for k in gated_keys)
            if unobtainable:
                skip.add(algo_name)
        return skip

    def ensure_assets_for_config(self) -> set[str]:
        """
        Ensure assets for the configured algorithms, skipping gated ones when no HF token
        is available. Returns the set of algorithms that remain runnable.
        """
        # check algos that can't be obtained
        selected_algorithms = set(self.config.run_config.algorithms)
        unobtainable_algorithms = self.unobtainable_gated_algorithms()
        skipped = selected_algorithms & unobtainable_algorithms
        if skipped:
            msg = (
                "Gated models are required but unavailable (weights missing locally and no HF token): "
                f"{', '.join(sorted(skipped))}.\n"
                "Configure HF_TOKEN or hugging_face_token and accept the model license agreements."
            )
            if self.config.run_config.skip_gated_models_errors:
                logging.warning(msg)
            else:
                raise RuntimeError(msg)

        # remove skipped models from active algorithms list
        active_algos = selected_algorithms - skipped
        logging.info(f"Active algorithms for this config: {', '.join(sorted(active_algos)) or '(none)'}")

        required_assets = []
        algos_dict = self.config.detectors_config.algorithms
        for algo in active_algos:
            algo_model = algos_dict.get(algo)
            if algo_model and hasattr(algo_model, "required_assets"):
                for val in algo_model.required_assets.values():
                    try:
                        # 'val' is the fully resolved absolute path.
                        # This extracts just the relative part to match the manifest keys
                        clean_key = str(Path(val).relative_to(self.assets_root))
                        clean_key = clean_key.replace("\\", "/")  # Safe fallback for Windows
                        required_assets.append(clean_key)
                    except ValueError:
                        logging.warning(f"Path '{val}' is not inside the assets root!")

        logging.info(f"AssetManager check: Found {len(required_assets)} required assets for this run.")

        self.verify_and_download(required_assets)
        return active_algos
