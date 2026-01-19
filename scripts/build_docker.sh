#!/bin/bash
set -e
TAG=${1:-"latest"}
DEV=${2:-false}

GIT_HASH=$(git rev-parse HEAD)
GIT_SUMMARY=$(git log -1 --pretty=%s)

echo "Building Docker image, tag: $TAG"
echo "Building with commit: $GIT_HASH"
echo "Commit message: $GIT_SUMMARY"
echo "Dev mode: $DEV"

# remove current nicetoolbox to avoid dangling images
docker rmi -f mpioslab/nicetoolbox:$TAG 2>/dev/null || true

# start rebuilding the docker
docker build \
  --build-arg NICETOOLBOX_GIT_HASH="$GIT_HASH" \
  --build-arg NICETOOLBOX_GIT_SUMMARY="$GIT_SUMMARY" \
  --build-arg NICETOOLBOX_DEV="$DEV" \
  -t mpioslab/nicetoolbox:$TAG .