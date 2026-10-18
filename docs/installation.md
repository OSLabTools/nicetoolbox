# Installation

```{contents} Contents
:depth: 3
:local:
```

```{attention}
NICE Toolbox requires third-party dependencies during installation. These are not maintained by the NICE Toolbox authors and are provided as-is by their respective owners, under their own separate license. We assume no liability for any issues arising from these dependencies. **Use at your own risk**. For more information, see [the Section 15, 16, 17 of the NICE Toolbox License](https://github.com/OSLabTools/nicetoolbox/blob/main/LICENSE) and [LICENSES_ALGORITHMS.md](https://github.com/OSLabTools/nicetoolbox/blob/main/LICENSES_ALGORITHMS.md).
```

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 11 or Ubuntu 22.04+ |
| **RAM** | 32 GB or more |
| **GPU** | NVIDIA GPU with 24 GB VRAM or more, CUDA 11.8+ compatible |
| **Storage** | 120 GB SSD or more |

## Prerequisites

### Git

[Git](https://git-scm.com/) is a free tool for downloading, updating and managing source code. It can be used to download NICE Toolbox and required for some of its dependencies.

**Windows**

1. Open the [Git for Windows download page](https://git-scm.com/downloads/win) and click **"Click here to download"** to get the latest 64-bit installer.
2. Run the downloaded file. If Windows asks whether the app may make changes to your device, click **Yes**.
3. Click **Next** through all pages to keep the default settings, then click **Install**.

You can also install Git with the console command:

```powershell
winget install --id Git.Git -e --source winget
```

**Linux (Ubuntu)**

Run the following commands in your terminal and enter your password when asked:

```bash
sudo apt update
sudo apt install git
```

For other systems and more information, see the [official Git installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### NVIDIA GPU Driver

The NVIDIA GPU driver lets NICE Toolbox use your graphics card. Most likely your system already has a compatible driver. If you want to update it, see the official [NVIDIA driver downloads](https://www.nvidia.com/en-us/drivers/) (Windows) or the [Ubuntu NVIDIA driver guide](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/) (Linux).

### Enable Long Paths (Windows only)

We recommend enabling long path support on Windows. It helps to avoid installation problems and errors when working with results stored in deeply nested folders. To enable it:

1. Open Windows Settings: click **Start**, type `Settings`, and press **Enter**.
2. Type `Enable long paths` into the search bar and click the first result.
3. Switch **Enable long paths** on.

Alternatively, open PowerShell as administrator and run:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

See [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation) for more details.

## Download

You can download the [latest NICE Toolbox release](https://github.com/OSLabTools/nicetoolbox/releases/latest) from GitHub by clicking **Source code (zip)** under **Assets**, and unzip the downloaded file to a folder of your choice.

If you want to use a specific commit or older version, clone the repository with Git:

```bash
git clone https://github.com/OSLabTools/nicetoolbox.git
cd nicetoolbox
```

To learn more about using Git, check [the documentation here](https://git-scm.com/learn).

## Installation Script

NICE Toolbox comes with interactive installer scripts for Windows (`install_windows.bat`) and Linux (`install_linux.sh`). They install NICE Toolbox, its dependencies and the detectors you select, and download an example dataset. Depending on your internet connection and the number of detectors, the installation can take from several minutes to about an hour.

**Windows**

Open the NICE Toolbox folder and double-click `install_windows.bat`. A window opens and shows the installation progress.

Alternatively, open a terminal in the NICE Toolbox folder (in File Explorer, right-click the folder and select **Open in Terminal**) and run:

```powershell
.\install_windows.bat
```

**Linux**

Open a terminal and run:

```bash
cd path/to/nicetoolbox
bash install_linux.sh
```

When the installation is finished, follow [getting started](getting_started.md) to activate a virtual environment and run detectors.

### Installation Options

You can pass options to the installation script when you start it from a terminal. The most useful ones are:

| Option | Description |
|--------|-------------|
| `-y`, `--yes` | Non-interactive mode. Install all detectors without showing the menu |
| `--detectors=a,b,c` | Install only the listed detectors, without showing the menu, e.g. `--detectors=spiga,py_feat` |
| `--only-missing` | Don't reinstall detectors that are already installed |
| `--core-only` | Install only the NICE Toolbox itself, without any detectors |

For example, to add the SPIGA detector to an existing installation:

```
# Windows
.\install_windows.bat --only-missing --detectors=spiga
# Linux
bash install_linux.sh --only-missing --detectors=spiga
```

To see all available options and detector names, run the script with `--help`.

## Update to Newer Version

To update NICE Toolbox, simply [download](#download) the new release into a new folder, [install](#installation-script) it as usual and delete the old NICE Toolbox folder.

## Hugging Face Access Token

Some detectors use models that are gated on [Hugging Face](https://huggingface.co/) and require you to accept their terms of use before downloading:

- **WhisperX** uses [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) for speaker diarization.
- **SAM 3D Body** uses [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3).

To get access:

1. Create a [Hugging Face account](https://huggingface.co/join) if you don't have one.
2. Visit each model page linked above and click **Agree and access repository** to accept the terms.
3. Create a Hugging Face access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with at least **Read** permissions.
4. Set the token in your `machine_specific_paths.toml`:

```toml
hugging_face_token = "your_token_here"
```

The model weights will be downloaded automatically on the first run and cached in the `assets` directory.

## Docker

You can install NICE Toolbox using Docker. The image ships the core toolbox together with all the system prerequisites, so you do not have to set any of them up yourself.

To use it, install [Docker](https://docs.docker.com/get-started/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), which lets Docker containers use your GPU. On Windows, use them inside [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

The third-party detectors are not part of the image. You can choose which of them to install:

```shell
# download the base container from Docker Hub
docker pull mpioslab/nicetoolbox

# start a base container
docker run --name my-nice-toolbox --gpus all -it mpioslab/nicetoolbox
# inside the container, install the detectors you need
bash install_linux.sh --only-missing --detectors=py_feat,openmmlab
# when finished, stop the container
exit

# start it again to do experiments
docker start -ai my-nice-toolbox

# or start it again with disconnected network access
docker network disconnect bridge my-nice-toolbox
docker start -ai my-nice-toolbox
```

When finished, follow [getting started](getting_started.md) to activate a virtual environment and run detectors. For more information about installation options, see [here](#installation-options).

## Additional Notes

Please check [Rerun's privacy policies](https://www.rerun.io/privacy).
Although rerun.io is used in local mode, the application will be collecting user information. To disable these analytics, open a terminal in the NICE Toolbox folder and run:

```
# Windows
envs\nicetoolbox\Scripts\rerun analytics disable
# Linux
envs/nicetoolbox/bin/rerun analytics disable
```