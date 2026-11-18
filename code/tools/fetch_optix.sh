#!/usr/bin/env bash
# Downloads the official OptiX HEADERS (public NVIDIA/optix-dev repository, header-only) into
# code/third_party/optix-dev, outside Git (.gitignore). Idempotent: if it already exists at the pinned commit, it does nothing.
# The denoiser runtime (libnvoptix) ships with the NVIDIA driver; this script does NOT install anything on the system.
set -euo pipefail
TAG="v9.1.0"
COMMIT="f1f6dd803f3159992d248178f6e09421c6eb8b6d"   # refs/tags/v9.1.0 (git ls-remote, 2026-11-12)
DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/optix-dev"
if [ -d "$DIR/.git" ] && [ "$(git -C "$DIR" rev-parse HEAD)" = "$COMMIT" ]; then
  echo "optix-dev $TAG ($COMMIT) already present in $DIR"; exit 0
fi
rm -rf "$DIR"; mkdir -p "$(dirname "$DIR")"
git clone --quiet --depth 1 --branch "$TAG" https://github.com/NVIDIA/optix-dev.git "$DIR"
GOT="$(git -C "$DIR" rev-parse HEAD)"
if [ "$GOT" != "$COMMIT" ]; then echo "unexpected commit: $GOT (expected $COMMIT)" >&2; exit 1; fi
test -f "$DIR/include/optix.h" || { echo "optix.h is missing" >&2; exit 1; }
echo "optix-dev $TAG ($COMMIT) in $DIR"
