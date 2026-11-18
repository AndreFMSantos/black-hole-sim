#!/usr/bin/env bash
# Downloads ONE snapshot of the Illinois GRMHD library (v3, CC BY 4.0, cite Dhruv et al. 2025, ApJS 277, 16) into the data folder (not in Git).
# Usage: tools/fetch_grmhd.sh MAD|SANE SPIN FRAME      e.g.: tools/fetch_grmhd.sh MAD 0 5500   (FRAME 5000..6000, 144 MB each)
# SPIN: -0.94 | -0.5 | 0 | +0.5 | +0.94 (SANE also -0.94_new). Destination: ${BHS_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}/illinois-grmhd-v3/<FLUX>_a<SPIN>/
# Checks the size against the server listing; idempotent. Does NOT download more than requested (a whole simulation is ~144 GB).
set -euo pipefail
flux=${1:?MAD|SANE}; spin=${2:?spin}; frame=${3:?frame 5000..6000}
base=https://thz.astro.illinois.edu
dir="${BHS_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}/illinois-grmhd-v3/${flux}_a${spin}"
mkdir -p "$dir"
name=$(printf 'torus.out0.%05d.h5' "$frame")
enc=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$spin")
path=$(curl -fsS "$base/fetch_files_v3_grmhd.php?flux=$flux&spin=$enc" | python3 -c "
import sys,json
n=sys.argv[1]
for f in json.load(sys.stdin):
    if f['name']==n: print(f['path'] + '\t' + f['size']); break
else: sys.exit('file not listed: '+n)" "$name")
url_path=${path%%$'\t'*}; size=${path#*$'\t'}
echo "listed: $url_path ($size)"
if [ -s "$dir/$name" ]; then echo "already exists: $dir/$name ($(stat -c %s "$dir/$name") bytes)"; exit 0; fi
curl -fS -o "$dir/$name.part" "$base$url_path"
mv "$dir/$name.part" "$dir/$name"
echo "ok: $dir/$name ($(stat -c %s "$dir/$name") bytes) sha256 $(sha256sum "$dir/$name" | cut -d' ' -f1)"
