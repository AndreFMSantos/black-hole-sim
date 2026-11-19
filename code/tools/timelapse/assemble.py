#!/usr/bin/env python3
"""Assembles the time-lapse (v1.1): one clip per segment with a title card (drawtext), 1 s cross-fades, 10-bit H.265.

  assemble.py [--draft] [--workdir DIR] [--out FILE]

The segments already rendered (render.py) are read from WORKDIR/{draft|full}/sNN. Each one is scaled to the video resolution (lanczos); the
final segment (native 4K) is not rescaled. The cards show the version and a key number (segments.py); they appear 0.6 s after the start of the
segment and leave before the cross-fade.
"""
import os as _os_bhs, pathlib as _pl_bhs
import os
_BHS_ROOT = str(_pl_bhs.Path(__file__).resolve().parents[3])  # the folder holding code/, data/ and assets/
_BHS_DATA = _os_bhs.environ.get('BHS_DATA', _os_bhs.path.join(_BHS_ROOT, 'data'))
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import importlib
S = importlib.import_module(os.environ.get('BHS_SEGMENTS', 'segments'))

FONT_B = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
FONT_R = "/usr/share/fonts/noto/NotoSans-Regular.ttf"


def sh(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, cmd))}\n{r.stderr[-1500:]}")


def ship_line(meta):
    m = re.search(r"max proper thrust ([0-9.e+-]+) c/M.*max velocity beta = ([0-9.]+)", meta.get("route_report", ""))
    return f"max thrust {float(m.group(1)):.3f} c/M · max β {float(m.group(2)):.2f} · 32 samples per frame, physical motion blur" if m else ""


def clip(seg, d, W, H, out, threads, hdr=False):
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    meta["ship_line"] = ship_line(meta)
    title, tmpl = seg.title, seg.key
    key = tmpl.format(**meta) if "{" in tmpl else tmpl
    (d / f"title.txt").write_text(title)
    (d / f"key.txt").write_text(key)
    D = seg.dur
    alpha = f"if(lt(t,0.6),0,if(lt(t,1.4),(t-0.6)/0.8,if(lt(t,{D}-1.2),1,if(lt(t,{D}-0.4),({D}-0.4-t)/0.8,0))))"
    fs1, fs2, x, y = round(H * 0.040), round(H * 0.026), round(W * 0.035), round(H * 0.85)
    box = "box=1:boxcolor=black@0.45:boxborderw=%d" % round(H * 0.012)
    # HDR: the text stays at 203 cd/m2 (graphics white, BT.2408): PQ code 0.5807 -> grey 0x94 (drawtext converts the RGB to limited-range YCbCr)
    fc1, fc2 = ("0x949494", "0x949494@0.92") if hdr else ("white", "white@0.92")
    src = ["-i", d / "seg.mov"] if (d / "seg.mov").exists() else ["-framerate", S.FPS, "-i", d / "frame_%05d.png"]
    if hdr and not (d / "seg.mov").exists():  # SDR source (8-bit PNG, gamma 2.2, Rec.709): white 1.0 -> 203 cd/m2, PQ + BT.2020 (zimg)
        pre = f"scale={W}:{H}:flags=lanczos,zscale=rin=full:min=gbr:tin=bt470m:pin=bt709:r=tv:m=2020_ncl:t=smpte2084:p=bt2020:npl=203,format=yuv444p10le,"
    else:
        pre = f"scale={W}:{H}:flags=lanczos,format=yuv444p10le,"
    vf = (pre +
          f"drawtext=fontfile={FONT_B}:textfile={d / f'title.txt'}:expansion=none:fontcolor={fc1}:fontsize={fs1}:x={x}:y={y}:{box}:alpha='{alpha}',"
          f"drawtext=fontfile={FONT_R}:textfile={d / f'key.txt'}:expansion=none:fontcolor={fc2}:fontsize={fs2}:x={x}:y={y + round(fs1 * 1.5)}:{box}:alpha='{alpha}',"
          f"format=yuv422p10le")
    tags = ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-color_range", "tv"] if hdr else []
    sh(["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-threads", threads, *src, "-vf", vf, "-r", S.FPS, "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", *tags, out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--workdir", default=os.path.join(_BHS_DATA, "timelapse"))
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", default="6")
    ap.add_argument("--hdr", action="store_true", help="HDR10 (PQ, BT.2020, metadata): the route/final segments already come in PQ from the renderer (render.py --hdr)")
    a = ap.parse_args()
    work = Path(a.workdir) / ("draft" if a.draft else "full")
    W, H = (1920, 1080) if a.draft else (3840, 2160)
    out = Path(a.out) if a.out else work / ("timelapse" + ("_hdr" if a.hdr else "") + ".mp4")
    clips = []
    for s in S.SEGMENTS:
        c = work / s.id / ("titled" + ("_hdr" if a.hdr else "") + ".mov")
        if not c.exists():
            print(f"clip {s.id}", flush=True)
            clip(s, work / s.id, W, H, c, a.threads, a.hdr)
        clips.append(c)
    starts, total = S.starts()
    ins = []
    for c in clips:
        ins += ["-i", c]
    fc, prev = [], "[0:v]"
    for i in range(1, len(clips)):
        nxt = f"[v{i}]"
        fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={S.OVERLAP}:offset={starts[S.SEGMENTS[i].id]}{nxt}")
        prev = nxt
    x265p = f"pools={a.threads}"
    tags = []
    if a.hdr:  # static HDR10: primaries/transfer/matrix + P3-D65 mastering display of 1000 cd/m2 and MaxCLL/MaxFALL measured from the luma (1500/35, approximate)
        x265p += (":colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:hdr10=1:hdr10-opt=1:repeat-headers=1:"
                  "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1500,35")
        tags = ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-color_range", "tv"]
    cmd = ["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-threads", a.threads, *ins, "-filter_complex", ";".join(fc), "-map", prev, "-r", S.FPS,
           "-c:v", "libx265", "-preset", "medium" if a.draft else "slow", "-crf", "20" if a.draft else "14", "-pix_fmt", "yuv420p10le", "-x265-params", x265p, "-tag:v", "hvc1", *tags, out]
    print("encoding", out, flush=True)
    sh(cmd)
    print("pronto", out, f"({total:.0f} s)")


if __name__ == "__main__":
    main()
