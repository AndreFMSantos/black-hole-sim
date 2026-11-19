#!/usr/bin/env python3
"""Renders the time-lapse segments (v1.1) with the global camera C(t). Resumable: skips what already exists.

  render.py [--only s01,s02] [--draft] [--workdir DIR] [--list]

Output per segment in WORKDIR/sNN/: PNG frames (rec2d, cpu, v05) or seg.mov (route/final, 10-bit ProRes) + meta.json with the measured numbers.
--draft: resolution /4 and few samples, to validate the whole pipeline in minutes.
"""
import os as _os_bhs, pathlib as _pl_bhs
_BHS_ROOT = str(_pl_bhs.Path(__file__).resolve().parents[3])  # the folder holding code/, data/ and assets/
_BHS_DATA = _os_bhs.environ.get('BHS_DATA', _os_bhs.path.join(_BHS_ROOT, 'data'))
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import importlib
C = importlib.import_module(os.environ.get('BHS_CAMERA', 'camera'))
S = importlib.import_module(os.environ.get('BHS_SEGMENTS', 'segments'))
V4 = os.environ.get('BHS_SEGMENTS') == 'segments_v4'
GPU = 'v4_0-metric-api' if V4 else 'v3_1-kerr-cuda-ui'
BW, BH = (3840, 2160) if V4 else (1920, 1080)   # v4: the GPU segments are rendered natively in 4K

CODE = Path(__file__).resolve().parents[2]
BIN = CODE / "build" / "apps"
DEFAULT_WORK = (Path(_BHS_DATA) / "timelapse")


def nframes(seg, last):
    return round((seg.dur + (0.0 if last else S.OVERLAP)) * S.FPS)


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"failed ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout + r.stderr


def route_json(seg, start, total, dur_s, cam, draft):
    wps = []
    step = 0.5
    k = 0
    while k * step < dur_s + step:
        t = k * step
        r, th, ph = cam.state(start + t)
        wps.append({"t": round(t, 6), "r": r, "theta": th, "phi": ph})
        k += 1
    disk = dict(seg.args.get("disk", {}))
    if "turb" in disk or seg.args.get("timed"):
        disk["time"] = {"mode": "rate", "rate": 20, "t0": start * 20}
    return {"name": seg.id, "fps": S.FPS, "seconds": dur_s, "ease": "linear", "spin": seg.args.get("spin", 0.9), "fov": C.FOV, "exposure": seg.args.get("exposure", 3),
            "ship": {"camera_path": True, "timescale": 1, "waypoints": wps}, "disk": disk}


def final_route_json(seg, start, dur_s):
    r, th, ph0 = C.Camera(1).state(start)   # (r, theta, phi) of the global camera at the entry instant (theta already at 80 degrees)
    th = C.THETA1
    pts = [(0, 40, th, ph0), (12, 33, 78, ph0 + 52), (26, 26, 75, ph0 + 140), (dur_s, 20, 72, ph0 + 250)]   # ends outside the disk (rout = 16)
    disk = dict(next(x for x in S.SEGMENTS if x.id == "s11").args["disk"])
    disk["time"] = {"mode": "rate", "rate": 20, "t0": start * 20}
    return {"name": seg.id, "fps": S.FPS, "seconds": dur_s, "ease": "smooth", "spin": 0.9, "fov": C.FOV, "exposure": [{"t": 0, "v": 3}, {"t": dur_s, "v": 2.2}],
            "ship": {"timescale": 25, "waypoints": [{"t": t, "r": r, "theta": a, "phi": p} for t, r, a, p in pts]},
            "lookat": {"target": "hole", "roll": 0}, "disk": disk}


def render_segment(seg, start, total, last, work, draft, hdr=False):
    out = work / seg.id
    out.mkdir(parents=True, exist_ok=True)
    meta_f = out / "meta.json"
    n = nframes(seg, last)
    cam = C.Camera(C.REF_TOTAL or total)
    div = 4 if draft else 1
    times = [start + (k + 0.5) / S.FPS for k in range(n)]
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    W, H = BW // div, BH // div
    if seg.tool == "rec2d":
        if (out / f"frame_{n - 1:05d}.png").exists():
            return
        cams = out / "cams.txt"
        cams.write_text("\n".join(f"{cam.theta(t):.6f} {cam.phi(t):.6f}" for t in times) + "\n")
        o = run([BIN / "v1_1-record-2d" / "v1_1-record-2d", "--scene", seg.args["scene"], "--cams", cams, "--out", out, "--w", W, "--h", H, "--steps", seg.args.get("steps", 8)])
        meta["recorder"] = o.strip()
        m = re.search(r"Euler r = ([0-9.]+).*", o)
        if m:
            meta["euler_r"] = f"{float(m.group(1)):.3f}".replace(".", ",")
            meta["t"] = re.search(r"t = ([0-9.]+)", o).group(1).replace(".", ",")
        m = re.search(r"(\d+) rays, (\d+) captured", o)
        if m:
            meta["n"], meta["captured"] = m.group(1), m.group(2)
    elif seg.tool == "radio":  # v2.2b torus / v2.2g own GRMHD: false colour (T_b), one process per frame; GRMHD: linear blend of two consecutive snapshots
        exe = BIN / GPU / GPU
        bgm = (Path(_BHS_DATA) / "athenak/run_d/bgm")
        from PIL import Image
        import numpy as np
        for k, t in enumerate(times):
            f = out / f"frame_{k:05d}.png"
            if f.exists():
                continue
            base = [exe, "--nohud", "--runs", 1, "--w", W, "--h", H, "--prec", "float", "--samples", 2 if draft else 6, "--dist", 150, "--fov", 16 if seg.args["kind"] == "torus" else 11,
                    "--theta", f"{cam.theta(t):.6f}", "--phi", f"{cam.phi(t):.6f}", "--nu-ghz", 230, "--mbh", 6.5e9, "--tb-range", "8.5,11"]
            if seg.args["kind"] == "torus":
                run(base + ["--shot", f, "--a", 0.9, "--disk-model", "torus", "--torus", "7,12", "--ne0", 3e4, "--plasma-beta", 3, "--theta-e-max", 20, "--tonemap", "aces"])
            else:
                if seg.args["kind"] == "grmhd_hi":
                    bgm = (Path(_BHS_DATA) / "athenak/run_d/bgm_hi")
                    d0, d1 = 201, 201 + seg.args["n"] - 1  # the dump numbering continues from the restart (201..500 = t 10002..10600)
                else:
                    d0, d1 = seg.args["d0"], seg.args["d1"]
                u = d0 + (d1 - d0) * k / max(1, n - 1)
                i0 = min(int(u), d1 - 1); fr = u - i0
                if V4:  # v4.0 time interpolation between snapshots i0 and i0 + 1 (pattern speed 0.2 Keplerian) instead of blending PNGs
                    run(base + ["--shot", f, "--grmhd", bgm / f"mad_{i0:05d}.bgm", "--grmhd-next", bgm / f"mad_{i0 + 1:05d}.bgm", "--grmhd-tau", f"{fr:.5f}", "--grmhd-dt", 2, "--grmhd-pattern", 0.2,
                                "--disk-model", "grmhd", "--electrons", "rhigh", "--rhigh", 20, "--munit", 8.589e24, "--emission", "leung"])
                    continue
                imgs = []
                for di in (i0, i0 + 1):
                    tmp = out / f"tmp_{k:05d}_{di}.png"
                    run(base + ["--shot", tmp, "--grmhd", bgm / f"mad_{di:05d}.bgm", "--disk-model", "grmhd", "--electrons", "rhigh", "--rhigh", 20, "--munit", 8.589e24 if V4 else 7.361e24, "--emission", "leung"])
                    imgs.append(np.asarray(Image.open(tmp).convert("RGB")).astype(np.float64))
                    tmp.unlink()
                Image.fromarray(np.round((1 - fr) * imgs[0] + fr * imgs[1]).astype(np.uint8)).save(f)
    elif seg.tool == "var":  # v2.1: parameter ramps (wormhole, charge, Lambda): one process per frame; SDR frames in PNG
        exe = BIN / GPU / GPU
        import math
        for k, t in enumerate(times):
            f = out / f"frame_{k:05d}.png"
            if f.exists():
                continue
            x = k / max(1, n - 1)
            sm = x * x * (3 - 2 * x)
            base = [exe, "--nohud", "--runs", 1, "--w", W, "--h", H, "--samples", 2 if draft else 6, "--shot", f, "--tonemap", "aces", "--theta", f"{cam.theta(t):.6f}", "--phi", f"{cam.phi(t):.6f}"]
            kind = seg.args["kind"]
            if kind == "wh":
                L = math.exp(math.log(14.0) + (math.log(2.4) - math.log(14.0)) * sm)
                run(base + ["--wormhole", "interstellar", "--wh-l", f"{L:.5f}", "--fov", 50, "--sky2", os.path.join(_BHS_ROOT, "assets/sky/wise/wise_8k.exr"), "--exposure", 2, "--sky-exposure", 3])
            elif kind == "whtravel":  # v4.0: the trip from one sky to the other, l = 14 cos(pi x) rho from +14 to -14 (slow at the ends, fastest at the throat)
                L = 14.0 * math.cos(math.pi * x)
                run(base + ["--wormhole", "interstellar", "--wh-l", f"{L:.5f}", "--fov", 60, "--sky2", os.path.join(_BHS_ROOT, "assets/sky/wise/wise_8k.exr"), "--exposure", 2, "--sky-exposure", 3])
            elif kind == "rn" and V4:  # Kerr-Newman Novikov-Thorne disk (own ISCO), Q up to 0.95 (a real horizon)
                run(base + ["--a", 0, "--charge", f"{0.95 * sm:.5f}", "--dist", 40, "--fov", 40, "--rout", 14, "--prec", "double", "--tol", "1e-8", "--bloom", 0.1, "--exposure", 2])
            elif kind == "rn":
                run(base + ["--a", 0, "--charge", f"{1.1 * sm:.5f}", "--dist", 30, "--fov", 34, "--rin", 6, "--rout", 14, "--profile", "newtonian", "--nodoppler", "--prec", "double", "--tol", "1e-8", "--bloom", 0.1, "--exposure", 2])
            elif V4:  # camera at 7 M, FOV 100, Lambda 0 -> 0.04 (r_c = 7.4 M at the end): the shadow stays smaller than the frame
                run(base + ["--a", 0, "--lambda", f"{0.04 * sm:.6f}", "--dist", 7, "--fov", 100, "--nodisk", "--prec", "double", "--tol", "1e-8", "--exposure", 2])
            else:
                run(base + ["--a", 0, "--lambda", f"{0.045 * sm:.6f}", "--dist", 5.5, "--fov", 70, "--nodisk", "--prec", "double", "--tol", "1e-8", "--exposure", 2])
    elif seg.tool == "view":  # v4.0 views: spherical inflow, jet walls, X-ray lamppost (Kerr a = 0.9375, false colour)
        exe = BIN / GPU / GPU
        for k, t in enumerate(times):
            f = out / f"frame_{k:05d}.png"
            if f.exists():
                continue
            base = [exe, "--nohud", "--runs", 1, "--w", W, "--h", H, "--prec", "float", "--samples", 2 if draft else 6, "--shot", f, "--a", 0.9375, "--theta", f"{cam.theta(t):.6f}",
                    "--phi", f"{cam.phi(t):.6f}", "--dist", 40, "--fov", 30, "--nosky"]
            kind = seg.args["kind"]
            if kind == "sph":
                run(base + ["--disk-model", "spherical", "--sph-flow", "infall"])
            elif kind == "cones":
                run(base + ["--disk-model", "cones", "--fov", 50])
            else:
                run(base + ["--disk-model", "lamppost", "--lamppost-h", 6, "--lamppost-plunge"])
    elif seg.tool in ("cpu", "v05"):
        for k, t in enumerate(times):
            f = out / f"frame_{k:05d}.png"
            if f.exists():
                continue
            if seg.tool == "cpu":
                a = seg.args
                w, h = a["w"] // div, a["h"] // div
                run([BIN / a["app"] / a["app"], "--out", f, "--w", w, "--h", h, "--ssaa", 1 if draft else a["ssaa"], "--dist", C.DIST, "--fov", C.FOV,
                     "--theta", f"{cam.theta(t):.6f}", "--phi", f"{cam.phi(t):.6f}"])
            else:
                cmd = [BIN / "v0_5-realtime-gpu" / "v0_5-realtime-gpu", "--shot", f, "--w", W, "--h", H, "--dist", C.DIST, "--fov", C.FOV,
                       "--theta", f"{cam.theta(t):.6f}", "--phi", f"{cam.phi(t):.6f}"]
                if not seg.args["sky"]:
                    cmd.append("--nosky")
                run(cmd)
    else:  # route / final
        mov = out / "seg.mov"
        if mov.exists():
            return
        js = route_json(seg, start, total, n / S.FPS, cam, draft) if seg.tool == "route" else final_route_json(seg, start, n / S.FPS)
        (out / "route.json").write_text(json.dumps(js, indent=1))
        w, h = (seg.args.get("w", BW) // div, seg.args.get("h", BH) // div)
        cmd = [BIN / GPU / GPU, "--route", out / "route.json", "--route-out", mov, "--codec", "prores", "--workdir", out / "chunks", "--resume",
               "--w", w, "--h", h, "--samples", 2 if draft else seg.args["samples"], "--a", seg.args.get("spin", 0.9)] + (["--hdr10", 1000] if hdr else []) + seg.args.get("flags", [])
        o = run(cmd)
        for line in o.splitlines():
            if line.startswith("waypoints:"):
                meta["route_report"] = line
                mm = re.search(r"max proper thrust ([0-9.e+-]+) c/M.*max velocity beta = ([0-9.]+)", line)
                if mm:
                    meta["ship_line"] = f"max thrust {float(mm.group(1)):.3f} c/M · max β {float(mm.group(2)):.2f} · {seg.args['samples']} samples per frame, physical motion blur"
    meta_f.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"{seg.id} ok ({n} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--workdir", default=str(DEFAULT_WORK))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--hdr", action="store_true", help="GPU segments (route/final) in HDR10 (PQ BT.2020); use with its own --workdir")
    a = ap.parse_args()
    starts, total = S.starts()
    if a.list:
        for s in S.SEGMENTS:
            print(f"{s.id} {s.tool:6s} start {starts[s.id]:6.1f}s dur {s.dur:4.0f}s  {s.title}")
        print(f"total {total:.0f} s")
        return
    work = Path(a.workdir) / ("draft" if a.draft else "full")
    only = set(filter(None, a.only.split(",")))
    for i, s in enumerate(S.SEGMENTS):
        if only and s.id not in only:
            continue
        render_segment(s, starts[s.id], total, i == len(S.SEGMENTS) - 1, work, a.draft, a.hdr)


if __name__ == "__main__":
    main()
