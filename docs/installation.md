# Installation

We conducted tests of the installation on Windows 11 and Ubuntu versions 20 and 22, using CUDA 11.8.

<!-- TOC -->
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
    - [Python 3.10](#python-310)
    - [Conda](#conda)
    - [Cuda 11.8](#cuda-118)
    - [FFmpeg](#ffmpeg)
    - [Git](#git)
    - [On Windows: Microsoft Visual C++](#on-windows-microsoft-visual-c)
    - [On Windows: Turn developer mode on](#on-windows-turn-developer-mode-on)
    - [On Windows: make](#on-windows-make)
  - [Clone the repository](#clone-the-repository)
  - [Makefile installation](#makefile-installation)
  - [Additional notes](#additional-notes)
<!-- TOC -->

## Prerequisites

### Python 3.10

Please find the download links under the [official python](https://www.python.org/downloads/) pages. The latest installer of a stable release of Python 3.10.10 can be downloaded [from here](https://www.python.org/downloads/release/python-31011/).

If you are a Windows user, please add python to your `PATH` variable as explained on [educative.io](https://www.educative.io/answers/how-to-add-python-to-path-variable-in-windows).

### Conda

Conda can be installed through different installers, see [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). A popular one is the Anaconda Distribution -- it uses the anaconda channel by default which is subject to specific licensing.
An alternative option is the [Miniforge](https://github.com/conda-forge/miniforge) installer, which uses the [conda-forge](https://conda-forge.org/) channel and comes with open-source packages.
Please find instructions to install Miniforge on their [official website](https://github.com/conda-forge/miniforge).

If you installed Conda through Anaconda, you can switch to the free conda-forge channel following these steps:

```bash
# check what is currently set
conda config --show channels

# remove all channels other than conda-forge
conda config --remove channels defaults

# add conda-forge if not already present
conda config --add channels conda-forge
```

```{important}
During the installation of Conda, it is **crucial not to select** the option to register Conda's Python as the default Python interpreter.
This is because the Nice Toolbox requires **Python version 3.10** to be set as the default.

![miniforge_default_python.png](graphics/miniforge_default_python.png)

Selecting this option during installation may result in errors or conflicts, as Conda's Python version may differ from the required version for NiceToolbox. To ensure proper functionality, make sure Python 3.10 remains your default version.
```

```{important}
On Windows, after installing Conda, ensure that the Conda paths are added to the SYSTEM environment variables. For details see: https://saturncloud.io/blog/solving-the-conda-command-not-recognized-issue-on-windows-10/#step-2-add-conda-to-the-path
```

### Cuda 11.8

Please find installation instructions on the official websites: for [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### FFmpeg

On Linux Ubuntu, please find detailed instructions [here](https://phoenixnap.com/kb/install-ffmpeg-ubuntu).

On Windows, you can follow [phoenixnap.com](https://phoenixnap.com/kb/ffmpeg-windows):

1. Visit the official [FFmpeg website](https://ffmpeg.org/download.html) to get the latest version
of the FFmpeg package and binary files.
2. Hover over the Windows icon with your mouse and click on 'Windows builds from gyan.dev'
3. This redirects you to a page having FFmpeg binaries. Install the latest git master branch build,
e.g., ffmpeg-git-essentials.7z.
4. Extract the downloaded files and rename the extracted folder as ffmpeg.
5. Move the folder to the root of the C drive or the folder of your choice.
6. Add FFmpeg to `PATH` in Windows SYSTEM environment variables.

### Git 

Ensure that Git is installed on your system. You can find installation instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### On Windows: Microsoft Visual C++

Microsoft Visual C++ 14.0 or greater is required for compiling some of the dependencies. Get it with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### On Windows: Turn developer mode on

Nice Toolbox creates symlinks. To enable these, please activate Developer Mode on Windows.
This can be done by going to `Settings` > `Updates & Security` > `For Developers` and turning Developer Mode on.

TODO -- check if it creates any security risk (windows gives a warning about) and whether there is a better way to do it.

### On Windows: Make

Nice Toolbox uses Makefiles for simple installation process. Follow these steps to install `make` on Windows for use with **Git Bash**:

**Step 1:** Download `make` for Windows
- Go to the official **ezwinports** SourceForge page:  
   🔗 [https://sourceforge.net/projects/ezwinports/files/](https://sourceforge.net/projects/ezwinports/files/)
- Download the latest version of **make**:  
   - Look for a file named:  `make-<latest_version>-without-guile-w32-bin.zip`

**Step 2:** Extract the ZIP File
- Unzip the downloaded `make-<latest_version>-without-guile-w32-bin.zip` file.

**Step 3:** Copy the Files to Git Bash’s MinGW64 Folder
- Navigate to: `C:\Program Files\Git\mingw64`
- Copy the contents of the extracted folder (copy all folders) into `C:\Program Files\Git\mingw64`. 
- **IMPORTANT:** Do NOT overwrite or replace any existing files.

**Note:**  
After copying the files, you must **restart Git Bash** for the changes to take effect.


## Clone the repository

Clone the NICE Toolbox repository and navigate to its directory:

```bash
git clone --recurse-submodules git@github.com:OSLabTools/nicetoolbox.git
cd /path/to/nicetoolbox
```

The `--recurse-submodules` flag ensures that all submodules are are automatically initialized and updated. 
Alternatively, you can run the following commands after having cloned the repository without this flag:
```bash
git submodule init           # to initialize your local configuration file
git submodule update         # to fetch all the files from the submodules and check out the appropriate commit
git submodule update --init  # to combine the git submodule init and git submodule update steps
```

## Makefile installation

The NICE Toolbox includes a Makefile that handles the installation of all required libraries and dependencies. It also downloads assets and an example dataset, and generates a machine-specific configuration file. Available commands include:

- `make` or `make all`  - Run all the commands below.
- `make create_machine_specifics` - Generate the configuration file.
- `make download_assets` - Check and download assets.
- `make download_dataset` - Check and download the example dataset.
- `make install` - Install all dependencies.

```{note}
Conda is required for installing the OpenMMLab environment (human pose estimation framework).
If you need to use different versions of Python or CUDA, you can adjust the relevant lines in the `Makefile` accordingly.
```

In case of errors during installation, you can run `make clean_all` to remove all virtual environments. After that, you can restart the installation.

### On Linux

Open a **terminal** (on Linux) or **Git Bash** (on Windows) and navigate to the directory of the repository, then run the command `make`:

```bash
cd /path/to/nicetoolbox/
make        
```

## Additional notes

Please check [rerun privacy policies](https://www.rerun.io/privacy).
Although rerun.io is used in local mode, the application will be collecting user information. To disable these analytics, activate the code environment in `env/` and then run:

```bash
rerun analytics config   ##to see current configuration
rerun analytics disable
rerun analytics config   ## to check if the change is applied
```
