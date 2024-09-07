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

## 1. Clone the repository
1. Before cloning the repository, install git into your machine if it is not installed yet.
See https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

2. Clone the isa-tool repository and navigate into the isa-tool folder:
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

If you are a windows user, please add python to PATH variables. 
The detailed explanation can be found under https://www.educative.io/answers/how-to-add-python-to-path-variable-in-windows 


### b. Conda
see https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Conda can be installed through different distributors. One of them is Anaconda. The "defaults" channel in Anaconda, which is used for the base environment, is subject to specific licensing. Another option is using Miniforge, which uses the conda-forge channel by default, offering only free and open-source packages.

If you installed Conda through Anaconda, you can switch to the free conda-forge channel by following these steps:

```
# check what is currently set
conda config --show channels

# remove all channels other than conda-forge conda config --remove channels defaults

# add conda-forge if not already present
conda config --add channels conda-forge
``` 
To install Miniforge you can use https://github.com/conda-forge/miniforge

**Important Notice for Conda Installation**

During the installation of Conda, it is **crucial not to select** the option to register Conda's Python as the default Python interpreter. 
This is because **NiceToolbox** requires Python version **3.10** to be set as the default.

![miniforge-defaultpyhthon.PNG](graphics%2Fminiforge-defaultpyhthon.PNG)

Selecting this option during installation may result in errors or conflicts, 
as Conda's Python version may differ from the required version for NiceToolbox. To ensure proper functionality, make sure Python 3.10 remains your default version.


### c. Cuda 11.8 
- For windows, see https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- For Ubuntu, see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

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

### e. Turn Developer Mode on in Windows
Nice Toolbox creates symlinks. In Windows, enable Developer Mode to make it posible.
Go to Settings > Updates&Security > For Developers and turn Developer Mode to on.
TODO -- check if it creates any security risk (windows gives a warning about) and whether there is a better way to do it.


## 3. Install the necessary libraries

The installation scripts presume that Conda, CUDA 11.8, and Python 3.10 are already installed on 
your system. Installation via Conda is mandatory because of openmmlab installation (pose detector). 
However, if you wish to use different versions of Python and CUDA, you can modify the corresponding 
lines in the installation files.

The installation script (`installation/install_all`) does the follows: 
1. Setup Nice Toolbox environment
2. Setup conda environment for Openmmlab (pose detector)
3. Setup venv environment for Gaze Detector

`install_all.sh` is the installation script for Linux and `install_all.bat` the installation script for Windows. 


### **How-to:**
#### 3.1. In Linux:
Open a terminal and navigate to main directory where Nice Toolbox is then type the followings into terminal

```
cd /path/to/nice-toolbox/
chmod +x ./installation/install_all.sh  # to add executable permission to the script
./installation/install_all.sh                # to install all necessary libraries
```

#### 3.2. In Windows:  
#TODO - Needs to be tested !!

Open a cmd and navigate to main directory where Nice Toolbox is then type the following into cmd
```
cd \path\to\nice-toolbox
installation\install_all.bat # to install all necessary libraries
```

#### 4. Additional Notes: 
Please check rerun privacy policies. 

https://www.rerun.io/privacy

It will be used in local mode. Still, the application will be collecting the user information. To disable these analytics: 

1. activate envs/ under nice-toolbox, then run: 
```
rerun analytics config ##to see current configuration
rerun analytics disable 
rerun analytics config ## to check if the change is applied

```
For more information: https://github.com/rerun-io/rerun/blob/main/crates/re_analytics/README.md