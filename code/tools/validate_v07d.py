#!/usr/bin/env python3
"""Validation of v0.7d (cinematic GPU) against the CPU reference (v0_6-kerr-cpu).
Usage: uv run --with numpy --with pillow python tools/validate_v07d.py [--build build] [--out DIR] [--w 1920 --h 1080]
 1. convergence: GPU with N = 1, 4, 16, 64 samples (--rawsky, a = 0.9, theta = 80, nt) vs CPU --ssaa 8 (64 samples/pixel),
    with 'ldr' accumulation (Reinhard per sample, same as the CPU SSAA) and 'hdr' (HDR mean, tone mapping afterwards)
 2. moving observer: orbit (r = 12, theta = 85) and infall (r = 10), CPU --ssaa 1 vs GPU 1 sample (--rawsky), nt and newtonian
Prints MAE (0-255 per channel) and % of pixels with maximum-per-channel error > 4 and > 16."""
import argparse, os, re, subprocess, numpy as np
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--build", default="build"); ap.add_argument("--out", default="/tmp/bhs-v07d")
ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
ap.add_argument("--ssaa", type=int, default=8); ap.add_argument("--prec", default="float")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
CPU = f"{A.build}/apps/v0_6-kerr-cpu/v0_6-kerr-cpu"; GPU = f"{A.build}/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui"
SKY = "../assets/sky/starmap_2020_8k.exr"
run = lambda cmd: subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.int32)
def err(a, b):
    d = np.abs(a - b); mx = d.max(axis=2)
    return d.mean(), 100 * (mx > 4).mean(), 100 * (mx > 16).mean()
def ms(o): return float(re.search(r"mean of \d+: ([\d.]+) ms", o).group(1))
common = ["--w", str(A.w), "--h", str(A.h), "--exposure", "3", "--sky-exposure", "2"]

print(f"## 1. Convergence (a=0.9, theta=80, D=40, nt, {A.prec}) vs CPU --ssaa {A.ssaa}")
cpu = f"{A.out}/cpu_ssaa{A.ssaa}.png"
o = run([CPU, "--a", "0.9", "--theta", "80", *common, "--ssaa", str(A.ssaa), "--profile", "nt", "--sky", SKY, "--out", cpu])
print("CPU:", re.search(r"threads, ([\d.]+) s", o).group(0))
c8 = load(cpu)
print("| N | accumulation | MAE | > 4 | > 16 | ms/frame |\n|---|---|---|---|---|---|")
for mode in ("ldr", "hdr"):
    for n in (1, 4, 16, 64):
        f = f"{A.out}/gpu_{mode}_{n}.png"
        o = run([GPU, "--shot", f, "--a", "0.9", "--theta", "80", *common, "--prec", A.prec, "--rawsky", "--samples", str(n), "--accum", mode, "--runs", "3"])
        m, g4, g16 = err(c8, load(f))
        print(f"| {n} | {mode} | {m:.3f} | {g4:.2f} % | {g16:.2f} % | {ms(o):.1f} |")

print("\n## 2. Moving observer: CPU --ssaa 1 vs GPU 1 sample (--rawsky)")
print("| observer | profile | prec | MAE | > 4 | > 16 |\n|---|---|---|---|---|---|")
for obs, dist in (("orbit", 12), ("infall", 10)):
    for prof in ("newtonian", "nt"):
        cf = f"{A.out}/cpu_{obs}_{prof}.png"
        base = ["--a", "0.9", "--theta", "85", "--dist", str(dist), *common, "--observer", obs, "--profile", prof]
        run([CPU, *base, "--ssaa", "1", "--sky", SKY, "--out", cf])
        for prec in ("double", "float"):
            gf = f"{A.out}/gpu_{obs}_{prof}_{prec}.png"
            run([GPU, "--shot", gf, *base, "--prec", prec, "--rawsky", "--runs", "1"])
            m, g4, g16 = err(load(cf), load(gf))
            print(f"| {obs} (r={dist}) | {prof} | {prec} | {m:.3f} | {g4:.2f} % | {g16:.2f} % |")
