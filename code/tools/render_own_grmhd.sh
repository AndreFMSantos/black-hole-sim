#!/usr/bin/env bash
# Converts the AthenaK dumps (mhd_w_bcc) into .bgm and renders one radio frame (T_b, M87*) per dump; then joins them into an mp4.
# Usage: tools/render_own_grmhd.sh [theta=17] [out.mp4]
set -euo pipefail
TH=${1:-17}; OUT=${2:-${BHS_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}/athenak/run_d/own_mad_th${TH}.mp4}
RUN=${BHS_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}/athenak/run_d; BIN=$RUN/T10k_d/bin; W=$RUN/frames_th$TH; mkdir -p $RUN/bgm $W
HERE=$(cd "$(dirname "$0")/.." && pwd); EXE=$HERE/build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui
for f in $BIN/torus.mhd_w_bcc.*.bin; do
  n=$(basename $f .bin); n=${n##*.}
  [ -s $RUN/bgm/mad_$n.bgm ] || nice uv run --with h5py --with numpy python -I $HERE/tools/convert_athenak.py $f $RUN/bgm/mad_$n.bgm > /dev/null
  [ -s $W/f_$n.png ] || $EXE --nohud --runs 1 --shot $W/f_$n.png --grmhd $RUN/bgm/mad_$n.bgm --disk-model grmhd --electrons rhigh --rhigh 20 \
      --dist 150 --fov 16 --w 720 --h 720 --prec float --samples 6 --theta $TH --munit 7.361e24 --emission leung --tb-range 8.5,11 > /dev/null
done
ffmpeg -y -loglevel error -framerate 20 -i $W/f_%05d.png -c:v libx264 -crf 14 -pix_fmt yuv420p $OUT
echo "ok $OUT"
