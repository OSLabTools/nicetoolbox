# syntax=docker/dockerfile:1
# check=skip=SecretsUsedInArgOrEnv

FROM docker.io/nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS base

# install make, git, ffmpeg + some essential stuff
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt update && \
    apt install -y build-essential \
    unzip git-all ffmpeg wget curl

# install uv (manages its own Python installations)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# copy the repo + submodules
COPY . /nicetoolbox
WORKDIR /nicetoolbox

# run minimalistic NICE Toolbox installation (only core environment)
ARG NICETOOLBOX_DEV
RUN make install DEV=${NICETOOLBOX_DEV}

# for version tracking we need to get git repository metadata 
ARG NICETOOLBOX_GIT_HASH
ARG NICETOOLBOX_GIT_SUMMARY

# we set git hash and summary into env variables
# they will be resolved in runtime by toolbox 
ENV NICETOOLBOX_GIT_HASH=${NICETOOLBOX_GIT_HASH}
ENV NICETOOLBOX_GIT_SUMMARY=${NICETOOLBOX_GIT_SUMMARY}

# now install the rest of the NICE Toolbox (detectors venv, gated models, etc.)
FROM base AS full

# The token itself is mounted as a secret (never baked into the image). BuildKit
# excludes secret contents from the cache key, so a changed token would otherwise
# be ignored on rebuild. HF_TOKEN_SUFFIX (last 6 chars of the token) is referenced
# below purely so the cache invalidates when the token changes.
ARG HF_TOKEN_SUFFIX="none"

# ! don't remove echo HF token echo, important for cache invalidation
# ! NICETOOLBOX_DEV is lost here
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) && \
    echo "Optional HF token: ${HF_TOKEN_SUFFIX}" && \
    make install_all_detectors download_assets