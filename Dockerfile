FROM docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 

# install python 3.10, make, git + some essential stuff
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt update && \
    apt install -y python3 python3-venv python3-dev \
    build-essential unzip git-all ffmpeg wget

# conda from miniforge
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh

# copy the repo + submodules
COPY . /nicetoolbox
WORKDIR /nicetoolbox

# do all toolbox installation steps
ARG NICETOOLBOX_DEV
RUN make create_machine_specifics && \
    make download_assets && \
    make download_dataset && \
    make install DEV=${NICETOOLBOX_DEV}

# for version tracking we need to get git repository metadata 
ARG NICETOOLBOX_GIT_HASH
ARG NICETOOLBOX_GIT_SUMMARY

# we set git hash and summary into env variables
# they will be resolved in runtime by toolbox 
ENV NICETOOLBOX_GIT_HASH=${NICETOOLBOX_GIT_HASH}
ENV NICETOOLBOX_GIT_SUMMARY=${NICETOOLBOX_GIT_SUMMARY}