#!/usr/bin/env python3
"""Validates the Kerr render in CUDA (v3_1-kerr-cuda-ui) against the CPU reference (v0_6-kerr-cpu, --ssaa 1).
Usage: uv run --with numpy --with pillow python tools/validate_kerr_cuda.py [--build build] [--out DIR] [--prec double,float] [--w 1920 --h 1080]
For each view: mean absolute error (0-255), % of pixels with maximum-per-channel error > 4 and > 16, split into
disk (interior), disk edge (+-3 px) and sky/shadow. The disk mask comes from a GPU render with --sky-exposure 0.
It also measures the CPU time (omarchy, all threads) and the GPU time (ms/frame)."""
import argparse, os, re, subprocess, sys, numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--build", default="build"); ap.add_argument("--out", default="/tmp/bhs-validate")
ap.add_argument("--prec", default="double,mixed:1e-6,float", help="list of prec[:tol]"); ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
ap.add_argument("--reuse-cpu", action="store_true")
ap.add_argument("--profile", default="newtonian", help="nt|newtonian (passed to CPU and GPU)")
ap.add_argument("--rawsky", action="store_true", help="GPU without mipmaps/anisotropy (geometry only, same as the CPU)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
CPU = f"{A.build}/apps/v0_6-kerr-cpu/v0_6-kerr-cpu"; GPU = f"{A.build}/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui"
SKY = "../assets/sky/starmap_2020_8k.exr"
views = [(a, th, False) for a in (0.0, 0.9, 0.998) for th in (80, 86)] + [(a, 90, True) for a in (0.0, 0.9, 0.998)]

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

def dil(m, n):
    for _ in range(n):
        g = m.copy(); g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]; m = g
    return m
def ero(m, n): return ~dil(~m, n)

def rep(d, mx, m):
    if not m.any(): return (0, 0, 0, 0)
    return (100 * m.mean(), d[m].mean(), 100 * (mx[m] > 4).mean(), 100 * (mx[m] > 16).mean())

print("| a | theta | disk | prec | region | % px | MAE | >4 | >16 |\n|---|---|---|---|---|---|---|---|---|")
times = []
for a, th, nod in views:
    tag = f"a{a}_t{th}_{'nodisk' if nod else 'disk'}"
    common = ["--a", str(a), "--theta", str(th), "--w", str(A.w), "--h", str(A.h), "--dist", "40", "--exposure", "3", "--sky-exposure", "2"] + (["--nodisk"] if nod else [])
    cpu_png = f"{A.out}/cpu_{tag}.png"
    if not (A.reuse_cpu and os.path.exists(cpu_png)):
        o = run([CPU, *common, "--ssaa", "1", "--profile", A.profile, "--sky", SKY, "--out", cpu_png])
        cpu_s = float(re.search(r"threads, ([\d.]+) s", o).group(1))
    else: cpu_s = float("nan")
    mask_png = f"{A.out}/mask_{tag}.png"
    mask = None
    if not nod:
        run([GPU, "--shot", mask_png, *common, "--sky-exposure", "0", "--runs", "1", "--prec", "float", "--profile", A.profile])
        mask = np.asarray(Image.open(mask_png).convert("L")).astype(np.int32) > 2
    c = np.asarray(Image.open(cpu_png).convert("RGB")).astype(np.int32)
    for spec in A.prec.split(","):
        prec, _, ptol = spec.partition(":")
        extra = (["--tol", ptol] if ptol else []) + (["--rawsky"] if A.rawsky else []) + ["--profile", A.profile]
        gpu_png = f"{A.out}/gpu_{spec.replace(":","_")}_{tag}.png"
        o = run([GPU, "--shot", gpu_png, *common, "--prec", prec, "--runs", "10", *extra])
        ms = float(re.search(r"mean of \d+: ([\d.]+) ms", o).group(1))
        steps = re.search(r"mean ([\d.]+), max (\d+)", o)
        g = np.asarray(Image.open(gpu_png).convert("RGB")).astype(np.int32)
        d = np.abs(c - g); mx = d.max(axis=2)
        regs = [("total", np.ones(mx.shape, bool))]
        if mask is not None:
            grow = dil(mask, 3); inner = ero(mask, 3)
            regs += [("disk", inner), ("edge", grow & ~inner), ("sky", ~grow)]
        for name, m in regs:
            p, mae, g4, g16 = rep(d, mx, m)
            print(f"| {a} | {th} | {'no' if nod else 'yes'} | {prec} | {name} | {p:.1f} | {mae:.3f} | {g4:.2f} % | {g16:.2f} % |")
        times.append((a, th, nod, spec, ms, cpu_s, float(steps.group(1)), int(steps.group(2))))
        Image.fromarray(np.clip(mx * 4, 0, 255).astype(np.uint8)).save(f"{A.out}/diff_{spec.replace(":","_")}_{tag}.png")
print("\n| a | theta | disk | prec | GPU ms/frame | CPU s (16 thr) | speedup | mean steps | max |\n|---|---|---|---|---|---|---|---|---|")
for a, th, nod, prec, ms, cs, sm, sx in times:
    print(f"| {a} | {th} | {'no' if nod else 'yes'} | {prec} | {ms:.2f} | {cs:.2f} | {cs * 1000 / ms:.0f}x | {sm:.0f} | {sx} |")
