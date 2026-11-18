#!/usr/bin/env bash
# converts (incrementally) the AthenaK mhd_w_bcc dumps of T600_d to .bgm in bgm_hi/mad_NNNNN.bgm (numbering 0..)
RUN=${BHS_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}/athenak/run_d; mkdir -p $RUN/bgm_hi
HERE=$(cd "$(dirname "$0")/.." && pwd)
for f in $(ls $RUN/T600_d/bin/torus.mhd_w_bcc.*.bin | sort); do
  n=${f##*.mhd_w_bcc.}; n=${n%.bin}
  [ -s $RUN/bgm_hi/mad_$n.bgm ] && continue
  # only convert complete dumps: the next one already exists (or the run finished)
  nxt=$(printf "%05d" $((10#$n + 1)))
  [ -e $RUN/T600_d/bin/torus.mhd_w_bcc.$nxt.bin ] || [ -e $RUN/T600.done ] || continue
  nice uv run --with h5py --with numpy python -I $HERE/tools/convert_athenak.py $f $RUN/bgm_hi/mad_$n.bgm > /dev/null 2>&1
done
