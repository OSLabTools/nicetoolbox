#!/bin/bash
set -e
TAG=${1:-"latest"}
DEV=${2:-false}
INCLUDE_GATED_HF_MODELS=${3:-false}

GIT_HASH=$(git rev-parse HEAD)
GIT_SUMMARY=$(git log -1 --pretty=%s)

echo "Building Docker image, tag: $TAG"
echo "Building with commit: $GIT_HASH"
echo "Commit message: $GIT_SUMMARY"
echo "Dev mode: $DEV"
echo "Include gated HF models: $INCLUDE_GATED_HF_MODELS"

# Only pass the HF token secret when gated models are requested and a token is
# available. When omitted, the Dockerfile's optional secret mount is empty and
# 'make all' skips gated models.
# HF_TOKEN_SUFFIX is the last 6 chars of the token, passed as a build arg so the
# build cache invalidates when the token changes (BuildKit excludes the secret
# itself from the cache key). Empty when no gated build, so it has no effect.
SECRET_ARGS=()
HF_TOKEN_SUFFIX="none"
if [ "$INCLUDE_GATED_HF_MODELS" = "true" ]; then
  if [ -n "$HF_TOKEN" ]; then
    SECRET_ARGS=(--secret id=hf_token,env=HF_TOKEN)
    HF_TOKEN_SUFFIX="...${HF_TOKEN: -6}"
    echo "HF_TOKEN present — gated Hugging Face models will be included."
  else
    echo "ERROR: INCLUDE_GATED_HF_MODELS=true but HF_TOKEN is not set." >&2
    echo "Run 'export HF_TOKEN=\"your_token\"' before running this script." >&2
    exit 1
  fi
fi

# remove current nicetoolbox to avoid dangling images
docker rmi -f mpioslab/nicetoolbox:$TAG 2>/dev/null || true

# start rebuilding the docker
# ensure BuildKit is enabled and pass the secret
DOCKER_BUILDKIT=1 docker build \
  --build-arg NICETOOLBOX_GIT_HASH="$GIT_HASH" \
  --build-arg NICETOOLBOX_GIT_SUMMARY="$GIT_SUMMARY" \
  --build-arg NICETOOLBOX_DEV="$DEV" \
  --build-arg HF_TOKEN_SUFFIX="$HF_TOKEN_SUFFIX" \
  "${SECRET_ARGS[@]}" \
  -t mpioslab/nicetoolbox:$TAG .