#!/usr/bin/env bash
# Dear ImGui (MIT, https://github.com/ocornut/imgui) for the interactive UI of v3_1-kerr-cuda-ui (v3.1). Idempotent; pins the tag.
set -euo pipefail
TAG=v1.93.0
COMMIT=c1aaac65e896a45f81f47a4d68f9959e0f86c859   # refs/tags/v1.93.0 (cloned on 2026-11-15)
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/imgui"
if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" describe --tags --exact-match 2>/dev/null || true)" = "$TAG" ]; then echo "imgui $TAG already present in $DEST"; exit 0; fi
rm -rf "$DEST"; git clone --quiet --depth 1 --branch "$TAG" https://github.com/ocornut/imgui "$DEST"
echo "imgui $TAG (commit $(git -C "$DEST" rev-parse HEAD)) in $DEST"
