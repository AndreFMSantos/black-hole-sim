#!/usr/bin/env bash
# Downloads the HDR sky map (NASA SVS "Deep Star Maps 2020", 8k, ~125 MB) into assets/sky/.
# Idempotent: if the file exists with the right sha256, it does nothing. The EXR does not go to Git.
set -euo pipefail
URL="https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_8k.exr"
SHA="dc6c4f413e85707a29a25a9451148154554ecca2c996f84fa8f47b65ef9ff7c4"
DIR="$(cd "$(dirname "$0")/../.." && pwd)/assets/sky"
OUT="$DIR/starmap_2020_8k.exr"
sum() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
mkdir -p "$DIR"
if [ -f "$OUT" ] && [ "$(sum "$OUT")" = "$SHA" ]; then echo "ok: $OUT already exists and the sha256 matches"; exit 0; fi
curl -fL --retry 3 -o "$OUT.part" "$URL"
if [ "$(sum "$OUT.part")" != "$SHA" ]; then echo "sha256 does NOT match, deleting" >&2; rm -f "$OUT.part"; exit 1; fi
mv "$OUT.part" "$OUT"
echo "downloaded and verified: $OUT"
