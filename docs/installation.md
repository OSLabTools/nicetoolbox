# Installation

We conducted tests of the installation on Windows 11 and Ubuntu versions 20 and 22, using CUDA 11.8.

<!-- TOC -->
- [Installation](#installation)
  - [1. Clone the repositories](#1-clone-the-repositories)
  - [2. Download assets](#2-download-assets)
  - [2. Prerequisites](#2-prerequisites)
    - [a. Python 3.10](#a-python-310)
    - [b. Conda](#b-conda)
    - [c. Cuda 11.8](#c-cuda-118)
    - [d. FFmpeg](#d-ffmpeg)
  - [3. Install the necessary libraries](#3-install-the-necessary-libraries)
    - [**How-to:**](#how-to)
      - [3.1. In Linux:](#31-in-linux)
      - [3.2. In Windows:](#32-in-windows)
      - [4. Additional Notes:](#4-additional-notes)
<!-- TOC -->

## 1. Clone the repositories
Clone the isa-tool repository and navigate into the isa-tool folder:
#TODO: UPDATE RENAME OF THE REPOSITORIES
```
git clone git@gitlab.tuebingen.mpg.de:cschmitt/isa-tool.git 
cd isa-tool
```
## 2. Download assets
Assets are uploaded to [keeper](https://keeper.mpdl.mpg.de/d/a9f91e7e60e84da69fc0/).
Please download the folder put it in `isa-tool/detectors/assets`.

## 2. Prerequisites

### a. Python 3.10
see https://www.python.org/downloads/

### b. Conda
see https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### c. Cuda 11.8 
- For windows, see https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- For Ubuntu, see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

#TODO: if the code works without CUDA - specify here - and perhaps create openmmlab installation 
have pytorch - cpu only installation 

### d. FFmpeg
- For windows: 
1. Visit the official [FFmpeg website](https://ffmpeg.org/download.html) to get the latest version 
of the FFmpeg package and binary files. 
2. Hover over the Windows icon with your mouse and click on 'Windows builds from gyan.dev'
3. This redirects you to a page having FFmpeg binaries. Install the latest git master branch build, 
e.g., ffmpeg-git-essentials.7z.
4. Extract the downloaded files and rename the extracted folder as ffmpeg
5. Move the folder to the root of the C drive or the folder of your choice.
6. Add FFmpeg to PATH in Windows environment variables.

Source: https://phoenixnap.com/kb/ffmpeg-windows

- For Ubuntu: 
see https://phoenixnap.com/kb/install-ffmpeg-ubuntu 


## 3. Install the necessary libraries 

The installation scripts presume that Conda, CUDA 11.8, and Python 3.10 are already installed on 
your system. Installation via Conda is mandatory because of openmmlab installation (pose detector). 
However, if you wish to use different versions of Python and CUDA, you can modify the corresponding 
lines in the installation files.

The installation script (`installation/install_all`): 
1. Setup isa-tool environment
2. Setup conda environment for Openmmlab (pose detector)
3. Setup venv environment for Gaze Detector

`install_all.sh` is the installation script for Linux and `install_all.bat` the installation script for Windows. 


### **How-to:**
#### 3.1. In Linux:
Open a terminal and navigate to main directory where isa-tool then type the followings into terminal

```
cd /path/to/isa-tool/
chmod +x ./installation/installation_all.sh  # to add executable permission to the script
./installation/install_all.sh                # to install all necessary libraries
```

#### 3.2. In Windows:  
#TODO - Needs to be tested !!

Open a cmd and navigate to main directory where isa-tool then type the following into cmd
```
cd \path\to\isa-tool
installation\install_all.bat
```

#### 4. Additional Notes: 
Please check rerun privacy policies. 

https://www.rerun.io/privacy

It will be used in local mode. Still, the application will be collecting the user information. To disable these analytics: 

1. activate envs/ under isa-tool, then run: 
```
rerun analytics config ##to see current configuration
rerun analytics disable 
rerun analytics config ## to check if the change is applied

```
For more information: https://github.com/rerun-io/rerun/blob/main/crates/re_analytics/README.md