#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJ_DIR="$(cd "$CODE_DIR/.." && pwd)"

BIN="${BIN:-$CODE_DIR/build/apps/v0_6b-kerr-metal/v0_6b-kerr-metal}"
SKY="${SKY:-$PROJ_DIR/assets/sky/starmap_2020_8k.exr}"
OUT="${OUT:-$PROJ_DIR/assets/evolution_timelapse.mp4}"
WORK="${WORK:-/tmp/bhs_timelapse}"

W="${W:-1280}"
H="${H:-720}"
FPS="${FPS:-30}"
SAMPLES="${SAMPLES:-2}"
XFADE="0.5"

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found in $BIN" >&2
  exit 1
fi
if [[ ! -f "$SKY" ]]; then
  echo "Error: sky map not found in $SKY" >&2
  exit 1
fi

mkdir -p "$WORK" "$(dirname "$OUT")"

echo "[1/4] Compiling the caption generator (Swift/CoreGraphics)..."
swiftc -O "$SCRIPT_DIR/render_overlay.swift" -o "$WORK/render_overlay"

echo "[2/4] Generating JSON routes and overlays..."
cat > "$WORK/route_v03_schwarzschild.json" <<EOF
{
  "name": "v03-schwarzschild", "fps": $FPS, "seconds": 8, "ease": "smooth",
  "spin": 0.0, "fov": 40, "exposure": 3,
  "ship": {"timescale": 50,
           "waypoints": [{"t": 0, "r": 40, "theta": 85, "phi": 0},
                         {"t": 4, "r": 34, "theta": 78, "phi": 90},
                         {"t": 8, "r": 40, "theta": 82, "phi": 180}]},
  "lookat": {"target": "hole", "roll": 0}
}
EOF

cat > "$WORK/route_v04_newtonian.json" <<EOF
{
  "name": "v04-newtonian", "fps": $FPS, "seconds": 9, "ease": "smooth",
  "spin": 0.0, "fov": 40, "exposure": 3,
  "ship": {"timescale": 50,
           "waypoints": [{"t": 0, "r": 40, "theta": 98, "phi": 0},
                         {"t": 4.5, "r": 36, "theta": 80, "phi": 120},
                         {"t": 9, "r": 40, "theta": 65, "phi": 240}]},
  "lookat": {"target": "hole", "roll": 0}
}
EOF

cat > "$WORK/route_v22b_radio.json" <<EOF
{
  "name": "v22b-radio", "fps": $FPS, "seconds": 10, "ease": "smooth",
  "spin": 0.9, "fov": 18, "exposure": 3,
  "ship": {"timescale": 50,
           "waypoints": [{"t": 0, "r": 90, "theta": 82, "phi": 0},
                         {"t": 5, "r": 85, "theta": 62, "phi": 180},
                         {"t": 10, "r": 90, "theta": 74, "phi": 360}]},
  "lookat": {"target": "hole", "roll": 0}
}
EOF

"$WORK/render_overlay" "$WORK/ov1.png" "v0.3"   "Schwarzschild Shadow (a = 0)"          "Null geodesics (RK4/DP45) + gravitational lensing of the 8K sky"
"$WORK/render_overlay" "$WORK/ov2.png" "v0.4"   "Accretion Disk + Doppler"            "Newtonian profile (r_in = 6M) with gravitational shift and beaming"
"$WORK/render_overlay" "$WORK/ov3.png" "v0.6b"  "Kerr + Novikov-Thorne (a: 0 -> 0.998)" "Frame dragging and ISCO contraction"
"$WORK/render_overlay" "$WORK/ov4.png" "v0.7c"  "Observer in a Relativistic Orbit"      "Special-relativistic aberration + Doppler boost in the ZAMO frame"
"$WORK/render_overlay" "$WORK/ov5.png" "v0.8d"  "Cinematic HDR Pipeline"          "ACES tone mapping, 6-level bloom, 6-point glare, grain and chroma"
"$WORK/render_overlay" "$WORK/ov6.png" "v0.9b"  "3D Volumetric Disk"                  "Gaussian layer (H/r = 0.08, tau = 10) and radiative transfer"
"$WORK/render_overlay" "$WORK/ov7.png" "v0.9c2" "3D Turbulence, Hot Spot and Slow Light"  "Differential rotation, orbiting flare and finite propagation delay"
"$WORK/render_overlay" "$WORK/ov8.png" "v2.2b"  "EHT Radio Mode (230 GHz)"              "Fishbone-Moncrief torus + thermal synchrotron (brightness temperature)"

echo "[3/4] Rendering 8 chapters on the Metal GPU ($W x $H @ $FPS fps, $SAMPLES samples)..."

"$BIN" --route "$WORK/route_v03_schwarzschild.json" --route-out "$WORK/c1.mp4" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --shutter 0 --nodisk --sky "$SKY"

"$BIN" --route "$WORK/route_v04_newtonian.json" --route-out "$WORK/c2.mp4" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --shutter 0 --profile newtonian --sky "$SKY"

"$BIN" --video "$WORK/c3.mp4" --clip spin --seconds 10 --fps "$FPS" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --profile nt --sky "$SKY"

"$BIN" --video "$WORK/c4.mp4" --clip orbit --seconds 10 --fps "$FPS" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --profile nt --observer orbit --sky "$SKY"

"$BIN" --video "$WORK/c5.mp4" --clip orbit --seconds 10 --fps "$FPS" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --profile nt --cinematic --sky "$SKY"

"$BIN" --video "$WORK/c6.mp4" --clip orbit --seconds 10 --fps "$FPS" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --hr 0.08 --tau 10 --vol-sub 4 --edge-soft 0.25 --cinematic --sky "$SKY"

"$BIN" --video "$WORK/c7.mp4" --clip disk --seconds 11 --fps "$FPS" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --hr 0.08 --tau 10 --vol-sub 4 \
  --turb 0.35 --spot 8,0,0.9,2.5 --spot-flare 45,12,3.0 --slowlight --disk-rate 12 --cinematic --sky "$SKY"

"$BIN" --route "$WORK/route_v22b_radio.json" --route-out "$WORK/c8.mp4" \
  --w "$W" --h "$H" --prec float --samples "$SAMPLES" --shutter 0 \
  --disk-model torus --torus 7,12,1.3333 --ne0 1e5 --plasma-beta 3 --theta-e-max 20 --nu-ghz 230

echo "[4/4] Composing captions and transitions (xfade ${XFADE}s) into $OUT..."
# Durations: 8, 9, 10, 10, 10, 10, 11, 10 -> offsets: 7.5, 16.0, 25.5, 35.0, 44.5, 54.0, 64.5
ffmpeg -y \
  -i "$WORK/c1.mp4" -i "$WORK/c2.mp4" -i "$WORK/c3.mp4" -i "$WORK/c4.mp4" \
  -i "$WORK/c5.mp4" -i "$WORK/c6.mp4" -i "$WORK/c7.mp4" -i "$WORK/c8.mp4" \
  -i "$WORK/ov1.png" -i "$WORK/ov2.png" -i "$WORK/ov3.png" -i "$WORK/ov4.png" \
  -i "$WORK/ov5.png" -i "$WORK/ov6.png" -i "$WORK/ov7.png" -i "$WORK/ov8.png" \
  -filter_complex "
    [0:v][8:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v0];
    [1:v][9:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v1];
    [2:v][10:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v2];
    [3:v][11:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v3];
    [4:v][12:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v4];
    [5:v][13:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v5];
    [6:v][14:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v6];
    [7:v][15:v]overlay=0:0:format=auto,format=yuv420p,fps=$FPS,settb=AVTB[v7];
    [v0][v1]xfade=transition=fade:duration=$XFADE:offset=7.5[x1];
    [x1][v2]xfade=transition=fade:duration=$XFADE:offset=16.0[x2];
    [x2][v3]xfade=transition=fade:duration=$XFADE:offset=25.5[x3];
    [x3][v4]xfade=transition=fade:duration=$XFADE:offset=35.0[x4];
    [x4][v5]xfade=transition=fade:duration=$XFADE:offset=44.5[x5];
    [x5][v6]xfade=transition=fade:duration=$XFADE:offset=54.0[x6];
    [x6][v7]xfade=transition=fade:duration=$XFADE:offset=64.5[vout]
  " \
  -map "[vout]" -c:v libx264 -preset medium -crf 18 -movflags +faststart "$OUT"

echo "Done: $OUT"
