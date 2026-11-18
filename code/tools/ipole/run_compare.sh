#!/usr/bin/env bash
# Same snapshot, same physics in ipole (https://github.com/AFD-Illinois/ipole, built separately) and in our renderer; see docs/physics/41-ipole-comparison.md.
# Needs: the Illinois GRMHD library snapshot MAD a = 0 frame 5500 (code/tools/fetch_grmhd.sh MAD 0 5500), its .bgm conversion (code/tools/convert_grmhd.py),
# IPOLE=<path to the ipole binary>, and the v4_0 app built.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd); DATA=${BHS_DATA:-$ROOT/data}
H5=$DATA/illinois-grmhd-v3/MAD_a0/torus.out0.05500.h5; BGM=$DATA/bgm/MAD_a0_5500.bgm; APP=$ROOT/code/build/apps/v4_0-metric-api/v4_0-metric-api
OUT=${1:-$DATA/cmp_ipole}; mkdir -p "$OUT"; cd "$OUT"
# electrons: constant T_p/T_e = 3 in ipole's convention (gamma_e = 4/3, gamma_p = 5/3); the snapshot has gam = 4/3, so Theta_i = u/(3 rho) and
# our proportional model needs th-e-ratio = (m_p/m_e) * (2/9) / (5/3) * 3 = 734.461069372
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} "$IPOLE" --dump="$H5" --thetacam=17 --phicam=0 --freqcgs=230e9 --MBH=6.5e9 --M_unit=1e25 --dsource=16.8e6 --fov=160 --nx=160 --ny=160 \
  --trat_large=3 --trat_small=3 --only_unpolarized=1 --outfile=ipole.h5 > ipole.log 2>&1
grep Ftot ipole.log
"$APP" --disk-model grmhd --grmhd "$BGM" --munit 1e25 --mbh 6.5e9 --electrons prop --th-e-ratio 734.461069372 --theta-e-min 0 --nu-ghz 230 --grmhd-sigma-cut 1 \
  --grmhd-rcap 100 --theta 17 --phi 0 --dist 1000 --fov 2.3994257002047 --w 160 --h 160 --prec double --tol 1e-8 --nosky --radio-cal --d-pc 16.8e6 --radio-out ours.pfm | grep "total flux"
python3 "$HERE/compare_images.py" ipole.h5 ours.pfm
