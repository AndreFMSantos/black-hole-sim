#!/bin/bash
# renders all the segments (resumable) and assembles the final video; log in WORK/full.log
cd "$(dirname "$0")"
W=${1:-${BHS_DATA:-$(cd "$(dirname "$0")/../../.." && pwd)/data}/timelapse}
mkdir -p "$W"
{ python3 render.py --workdir "$W" && python3 assemble.py --workdir "$W"; echo "rc=$? $(date)"; } > "$W/full.log" 2>&1
