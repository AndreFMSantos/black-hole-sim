#!/usr/bin/env python3
"""Robustness sweep of the interactive app: one small image per parameter set, over every UI scenario.

For each scenario (the 14 non-GRMHD ones of ui.cpp, as CLI flags) every parameter the UI exposes is varied one at a time over its
slider range (plus the extremes), in float and double. Each run is checked for:
  - crash / non-zero exit / timeout / CUDA error text on stderr
  - NaN/Inf text in the output
  - blank image (all black or all white)
  - isolated saturated pixels (bright pixel >= 250 whose 7x7 median is < 180; run with --nosky so stars cannot count)
  - float vs double disagreement (pixels with |diff| > 60)
Usage: sweep_params.py [--out DIR] [--only SUBSTR] [--quick] [--timeout S]
Writes DIR/report.md, DIR/results.json and the images in DIR/img/ (only for flagged runs unless --keep-all).
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui"  # --bin overrides (v4_0: build/apps/v4_0-metric-api/v4_0-metric-api)

# scenario -> base flags (what ui.cpp's kScenarios do); "nosky" False for the wormholes (the 2nd sky is the point)
SCEN = {
    "schwarzschild-shadow": dict(a=0, nodisk=True, theta=80, dist=40),
    "thin-newtonian": dict(a=0, profile="newtonian", theta=80, dist=40),
    "kerr-a09-newtonian": dict(a=0.9, profile="newtonian", theta=80, dist=40),
    "novikov-thorne": dict(a=0.9, theta=80, dist=40),
    "orbiting-observer": dict(a=0.9, theta=80, dist=12, observer="orbit"),
    "thick-disk": dict(a=0.9, theta=80, dist=40, hr=0.06, tau=4, flare=0.15),
    "turbulence-spot": dict(a=0.9, theta=75, dist=40, hr=0.06, tau=4, flare=0.15, turb=1.2, turb_scale=1.2, edge_soft=0.12, spot="9,200,1.2,5", slowlight=True, time=100),
    "reissner-nordstrom": dict(a=0, charge=0.6, theta=78, dist=30),
    "kerr-newman": dict(a=0.5, charge=0.5, theta=78, dist=40),
    "naked-singularity": dict(a=0, charge=1.1, profile="newtonian", rin=6, theta=78, dist=30),
    "schwarzschild-de-sitter": dict(a=0, lambda_=0.01, nodisk=True, theta=90, dist=8, fov=70),
    "ellis-wormhole": dict(wormhole="ellis,1", wh_l=4, nodisk=True, fov=50, theta=90, dist=4, sky=True),
    "interstellar-wormhole": dict(wormhole="interstellar", wh_l=4, nodisk=True, fov=50, theta=90, dist=4, sky=True),
    "interstellar-preset": dict(preset="interstellar", theta=82, dist=40, fov=30),
    "radio-riaf": dict(a=0.9, disk_model="riaf", dist=150, fov=16, theta=17, mbh=6.5e9, nu_ghz=230, ne0=8e3, plasma_beta=10, theta_e_max=12, tonemap="aces"),
    "spherical-inflow": dict(a=0.9, disk_model="spherical", dist=100, fov=24, theta=60),
    "jet-cones": dict(a=0.5, disk_model="cones", dist=100, fov=24, theta=80),
    "radio-torus": dict(a=0.9, disk_model="torus", torus="7,12", ne0=3e4, plasma_beta=3, theta_e_max=20, dist=150, fov=16, theta=17, mbh=6.5e9, nu_ghz=230, tonemap="aces"),
}

# UI slider ranges (ui.cpp) + extremes; only applied where the scenario has that parameter's metric
AXES = {
    "dist": [1.5, 1.6, 2, 3, 5, 8, 12, 30, 100, 500, 2000],
    "theta": [0.1, 1, 10, 45, 89.9, 90, 90.1, 135, 179.9],
    "phi": [0, 90, 180, 359.9],
    "fov": [2, 10, 40, 100, 150],
    "a": [0, 0.01, 0.5, 0.5214, 0.9, 0.998, 0.9999],
    "charge": [0.01, 0.3, 0.5, 0.6, 0.8, 0.95, 0.9999, 1.0, 1.001, 1.5],
    "lambda_": [1e-4, 1e-3, 0.01, 0.05, 0.11],
    "wh_l": [1.05, 1.5, 4, 10, 60],
    "rout": [8, 16, 40, 80],
    "hr": [0.01, 0.06, 0.2],
    "tau": [0.5, 4, 10],
    "flare": [0, 0.15, 0.5],
    "exposure": [0.03, 1, 3, 30],
    "bloom": [0, 0.3, 0.6],
    "observer": ["static", "orbit", "infall"],
    "ne0": [1e3, 3e4, 1e7],
    "plasma_beta": [0.5, 3, 30],
    "theta_e_max": [5, 20, 40],
}


def applicable(name, base, key):
    if key == "a":
        return not (base.get("lambda_") or base.get("wormhole"))
    if key == "charge":
        return "charge" in base
    if key == "lambda_":
        return "lambda_" in base
    if key == "wh_l":
        return "wormhole" in base
    if key in ("rout",):
        return not base.get("nodisk") and "disk_model" not in base
    if key in ("hr", "tau", "flare"):
        return "hr" in base
    if key in ("ne0", "plasma_beta", "theta_e_max"):
        return "disk_model" in base
    if key in ("a",) and base.get("preset"):
        return True
    if key == "observer":
        return not (base.get("wormhole") or "charge" in base or "lambda_" in base)
    if key == "dist":
        return "wormhole" not in base
    return True


def ui_semantics(p):
    """What the UI does on top of the raw values: naked Kerr-Newman (a^2 + Q^2 >= 1) switches to the Newtonian disk with r_in = 6."""
    p = dict(p)
    if "wormhole" not in p:  # the UI slider only offers the camera range of the metric (camera_limits.hpp); mirror it
        import math
        lo, hi = 1.5, 2000.0
        if p.get("lambda_"):
            th = math.acos(-3 * math.sqrt(p["lambda_"])) / 3; k = 2 / math.sqrt(p["lambda_"])
            r = sorted(k * math.cos(th - 2 * math.pi * i / 3) for i in range(3)); lo, hi = r[1] * 1.02, r[2] * 0.98
        else:
            d = 1 - p.get("a", 0) ** 2 - p.get("charge", 0) ** 2
            if d >= 0:
                lo = max(lo, (1 + math.sqrt(d)) * 1.02)
            else:
                lo = max(lo, 3.0)  # naked singularity: trapped rays make closer cameras prohibitively slow
            if p.get("observer") == "infall":
                lo = max(lo, 2.25)
            if p.get("observer") == "orbit" and not p.get("charge"):
                lo = max(lo, 2 * (1 + math.cos(2 / 3 * math.acos(-min(1, p.get("a", 0))))) * 1.02)
        p["dist"] = min(max(p.get("dist", 40), lo), hi)
    if "charge" in p and p.get("a", 0) ** 2 + p["charge"] ** 2 >= 1.0:
        p["profile"] = "newtonian"; p.setdefault("rin", 6)
    return p


def flags(p, prec, w, h):
    p = ui_semantics(p)
    f = ["--nohud", "--w", str(w), "--h", str(h), "--samples", "1", "--prec", prec, "--max-steps", "10000"]  # the interactive window uses 10000
    if not p.get("sky"):
        f.append("--nosky")
    if prec == "float":
        f += ["--tol", "2e-5"]
    m = {"a": "--a", "theta": "--theta", "phi": "--phi", "dist": "--dist", "fov": "--fov", "charge": "--charge", "lambda_": "--lambda", "rin": "--rin", "rout": "--rout",
         "hr": "--hr", "tau": "--tau", "flare": "--flare", "turb": "--turb", "turb_scale": "--turb-scale", "edge_soft": "--edge-soft", "spot": "--spot", "time": "--time",
         "profile": "--profile", "observer": "--observer", "wormhole": "--wormhole", "wh_l": "--wh-l", "disk_model": "--disk-model", "torus": "--torus", "ne0": "--ne0",
         "plasma_beta": "--plasma-beta", "preset": "--preset", "theta_e_max": "--theta-e-max", "mbh": "--mbh", "nu_ghz": "--nu-ghz", "tonemap": "--tonemap", "exposure": "--exposure", "bloom": "--bloom"}
    for k, v in p.items():
        if k in m:
            f += [m[k], str(v)]
        elif k == "nodisk" and v:
            f.append("--nodisk")
        elif k == "slowlight" and v:
            f.append("--slowlight")
    return f


def analyse(path):
    a = np.asarray(Image.open(path).convert("RGB"), float).mean(2)
    med = median_filter(a, size=7)
    iso = int(((a >= 250) & (med < 180)).sum())
    return dict(mean=float(a.mean()), iso=iso, black=bool(a.max() < 3), white=bool(a.min() > 252)), a


def run(p, prec, out, timeout, w, h):
    cmd = [str(BIN)] + flags(p, prec, w, h) + ["--shot", str(out)]
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, err = r.returncode, (r.stderr + r.stdout)
    except subprocess.TimeoutExpired:
        rc, err = "timeout", ""
    dt = time.time() - t
    low = err.lower()
    bad = [k for k in ("nan", "inf ", "cuda error", "illegal", "terminate", "assert", "error:") if k in low]
    return rc, bad, dt, " ".join(cmd[1:]), err[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent.parent / "data/sweep"))
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true", help="3 values per axis (min, middle, max)")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--size", default="320x180")
    ap.add_argument("--random", type=int, default=0, help="extra random combinations per scenario")
    ap.add_argument("--bin", default="", help="executable to test (default: the v3_1 build)")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    global BIN
    if args.bin:
        BIN = Path(args.bin).resolve()
    out = Path(args.out); (out / "img").mkdir(parents=True, exist_ok=True)
    results = []
    rng = np.random.default_rng(12345)
    for name, base in SCEN.items():
        if args.only and args.only not in name:
            continue
        sets = [("base", dict(base))]
        for key, vals in AXES.items():
            if not applicable(name, base, key):
                continue
            vv = [vals[0], vals[len(vals) // 2], vals[-1]] if args.quick else vals
            for v in vv:
                p = dict(base); p[key] = v
                sets.append((f"{key}={v}", p))
        for k in range(args.random):
            p = dict(base)
            for key, vals in AXES.items():
                if applicable(name, base, key) and rng.random() < 0.5:
                    p[key] = vals[int(rng.integers(len(vals)))]
            sets.append((f"random{k}", p))
        for label, p in sets:
            row = dict(scenario=name, label=label, issues=[])
            imgs = {}
            for prec in ("float", "double"):
                f = out / "img" / f"{name}__{label.replace('=', '-').replace(',', '_')}__{prec}.png"
                if f.exists():
                    f.unlink()
                rc, bad, dt, cmd, tail = run(p, prec, f, args.timeout, w, h)
                row[prec] = dict(rc=rc, secs=round(dt, 2), cmd=cmd)
                if rc != 0:
                    row["issues"].append(f"{prec}: exit {rc} {tail.strip().splitlines()[-1] if tail.strip() else ''}")
                    continue
                if bad:
                    row["issues"].append(f"{prec}: output mentions {bad}")
                if dt > (4 if prec == "float" else 20):  # 320x180 x 1 sample: anything slower would exceed the display watchdog in the UI (~1900x1040)
                    row["issues"].append(f"{prec}: slow ({dt:.1f} s for {w}x{h}, 1 sample)")
                if not f.exists():
                    row["issues"].append(f"{prec}: no image written")
                    continue
                st, a = analyse(f)
                row[prec].update(st); imgs[prec] = (f, a)
                near_or_tiny = p.get("dist", 40) < 8 or p.get("fov", 40) < 5  # shadow larger than the field of view: black is expected
                if st["black"] and not near_or_tiny and not p.get("nodisk") is True and "wormhole" not in p and p.get("disk_model") is None and name not in ("schwarzschild-shadow", "schwarzschild-de-sitter"):
                    row["issues"].append(f"{prec}: black image")
                if st["white"]:
                    row["issues"].append(f"{prec}: all-white image")
                if st["iso"] > 0:
                    row["issues"].append(f"{prec}: {st['iso']} isolated saturated px")
            if len(imgs) == 2:
                d = np.abs(imgs["float"][1] - imgs["double"][1])
                row["float_vs_double_px60"] = int((d > 60).sum())
                if row["float_vs_double_px60"] > 0.002 * w * h:
                    row["issues"].append(f"float vs double differ at {row['float_vs_double_px60']} px (>60)")
            if not row["issues"] and not args.keep_all:
                for prec, (f, _) in imgs.items():
                    f.unlink()
            results.append(row)
            print(f"{name:26s} {label:22s} {'OK' if not row['issues'] else 'ISSUE: ' + '; '.join(row['issues'])}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    bad = [r for r in results if r["issues"]]
    lines = [f"# Parameter sweep report\n", f"{len(results)} parameter sets x 2 precisions, {w}x{h}, 1 sample. Flagged: {len(bad)}.\n", "| scenario | set | issues |", "|---|---|---|"]
    for r in bad:
        lines.append(f"| {r['scenario']} | `{r['label']}` | {'; '.join(r['issues'])} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(bad)} flagged of {len(results)}; report in {out / 'report.md'}")


if __name__ == "__main__":
    main()
