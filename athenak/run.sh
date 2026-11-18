#!/usr/bin/env bash
# Runs the MAD torus to t = 10 000 M (about 18 h on an RTX 5080). Output in $BHS_DATA/athenak/run.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${BHS_DATA:-$HERE/../data}
SRC=${ATHENAK:-$DATA/athenak}
RUN=$DATA/athenak/run; mkdir -p "$RUN"; cd "$RUN"
"$SRC/build_d/src/athena" -i "$HERE/mad_10k.athinput" meshblock/nx1=16 meshblock/nx2=16 meshblock/nx3=16 \
  time/tlim=10000 time/nlim=-1 output1/dt=10 output2/dt=50 -d "$RUN" 2>&1 | tee "$RUN/run.log"
