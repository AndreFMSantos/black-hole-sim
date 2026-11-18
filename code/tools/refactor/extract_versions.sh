#!/usr/bin/env bash
# Extracts each milestone of the CUDA app from Git history into its own folder under apps/ (2026-11-16 refactor).
# Each folder is the code AS IT WAS at that commit, compiled against the current (backwards-compatible) core/.
# Usage: tools/refactor/extract_versions.sh   (run from code/; idempotent; overwrites the folders)
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=01-Projects/black-hole-sim/code/apps/v06b-kerr-cuda   # the app's path in the history (all commits)
ROOT=$(git rev-parse --show-toplevel)
# folder             commit/tag   prefix for targets/tests
VERSIONS=(
  "v0_6b-kerr-cuda    bhs-v0.6c   v0_6b"
  "v0_7-cinematic-cuda bhs-v0.7e  v0_7"
  "v0_8-flight-cuda   5fddaeb     v0_8"
  "v0_9-disk-cuda     e2df574     v0_9"
  "v2_x-variants-cuda 4b60f84     v2_x"
)
# the HEAD of today's app is apps/v3_1-kerr-cuda-ui (already its own folder)
for row in "${VERSIONS[@]}"; do
  read -r dir rev pre <<<"$row"
  dest="apps/$dir"; rm -rf "$dest"; mkdir -p "$dest"
  git -C "$ROOT" archive "$rev" "$SRC" | tar -x -C "$dest" --strip-components=5
  # unique target and test names per folder
  sed -i -e "s/v06b-kerr-cuda/$dir/g" -e "s/NAME v06b_/NAME ${pre}_/g" -e "s/NAME v07d_regression/NAME ${pre}_v07d_regression/" -e "s/NAME v08c_no_optix/NAME ${pre}_no_optix/" -e "s/NAME v06b_selftest/NAME ${pre}_selftest/" "$dest/CMakeLists.txt"
  if [ "$dir" = "v2_x-variants-cuda" ]; then  # backport (2026-11-16): v2.7 swallowed `k.phi = ...` inside a // comment (the camera azimuth never reached the kernel)
    python3 - "$dest/render.cu" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
# NOTE: the string below must match the ORIGINAL (Portuguese) comment as it was in the Git history being extracted.
old = "limite 0.001 graus k.phi = p.phi_deg * M_PI / 180.0;"
assert old in s
open(p, "w").write(s.replace(old, "limite 0.001 graus\n  k.phi = p.phi_deg * M_PI / 180.0;"))
PY
  fi
  printf '%s %s\n' "$dir" "$(git -C "$ROOT" rev-parse --short "$rev")" >>"$dest/.extracted-from"
  echo "$dir <- $rev ($(ls "$dest" | wc -l) files)"
done
