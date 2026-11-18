#!/usr/bin/env bash
# Regression v0.7d -> v0.8: the PHYSICAL look (without --cinematic or polish flags) must stay byte-identical.
# Usage: tools/regress_v07d.sh [--update]   (BHS_BIN=path of the executable; default ./build/...)
#    (md5 of the PNGs of 3 reference renders, 640x360)
set -euo pipefail
cd "$(dirname "$0")/.."
B="${BHS_BIN:-./build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui}"   # ctest passes the executable path
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
run() { $B --shot "$T/$1.png" --w 640 --h 360 --runs 1 "${@:2}" >/dev/null; printf "%s  %s\n" "$(md5sum < "$T/$1.png" | cut -d' ' -f1)" "$1"; }
{
  run A_float_n4_nt        --a 0.9 --theta 80 --prec float --samples 4 --seed 3
  run B_double_n1_newt     --a 0.998 --theta 86 --prec double --profile newtonian
  run C_float_aces_bloom   --a 0.9 --theta 80 --prec float --samples 2 --tonemap aces --bloom 0.12 --observer orbit --dist 12
} > "$T/now.txt"
if [ "${1:-}" = "--update" ]; then cp "$T/now.txt" tools/regress_v07d.golden; echo "golden updated"; exit 0; fi
if diff -u tools/regress_v07d.golden "$T/now.txt"; then echo "v0.7d regression OK (physical look byte-identical)"; else echo "REGRESSION: the physical look changed" >&2; exit 1; fi
