#!/usr/bin/env python3
"""Temporal flicker of the denoiser (v0.8c): mean difference between consecutive frames, with and without the denoiser.
 (A) static camera, two noise realisations (different seeds): noise only -> lower = less flicker
 (B) one step of the spin clip (a 0.5 -> 0.5 + 0.998/600), same seed: real change + flicker; compared with the "true" change
     (64-sample reference without denoiser)
Usage: uv run --with numpy --with pillow python tools/flicker_v08c.py [--build build] [--n 2] [--w 1920 --h 1080]"""
import argparse, os, subprocess, numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--build", default="build"); ap.add_argument("--out", default="/tmp/bhs-flicker")
ap.add_argument("--n", type=int, default=2); ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
GPU = f"{A.build}/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui"
def shot(tag, a, n, seed, off, ex=()):
    p = f"{A.out}/{tag}.png"
    subprocess.run([GPU, "--shot", p, "--a", str(a), "--theta", "80", "--w", str(A.w), "--h", str(A.h), "--prec", "float", "--runs", "1",
                    "--samples", str(n), "--seed", str(seed), "--sample-offset", str(off), *ex], capture_output=True, check=True)
    return np.asarray(Image.open(p).convert("RGB")).astype(float)
d = lambda x, y: np.abs(x - y).mean()
print("| case | without denoiser | optix |\n|---|---|---|")
na, nb = shot("A1n", 0.5, A.n, 1, 100), shot("A2n", 0.5, A.n, 2, 300)
da, db = shot("A1d", 0.5, A.n, 1, 100, ("--denoise", "optix")), shot("A2d", 0.5, A.n, 2, 300, ("--denoise", "optix"))
print(f"| (A) static camera, different noise (N={A.n}): MAE between frames | {d(na, nb):.3f} | {d(da, db):.3f} |")
a2 = 0.5 + 0.998 / 600
n1, n2 = shot("B1n", 0.5, A.n, 1, 100), shot("B2n", a2, A.n, 1, 100)
d1, d2 = shot("B1d", 0.5, A.n, 1, 100, ("--denoise", "optix")), shot("B2d", a2, A.n, 1, 100, ("--denoise", "optix"))
r1, r2 = shot("R1", 0.5, 64, 9, 5000), shot("R2", a2, 64, 9, 5000)
print(f"| (B) one step of the spin clip (N={A.n}, same seed): MAE between frames | {d(n1, n2):.3f} | {d(d1, d2):.3f} |")
print(f"|     true change (reference, 64 samples) | {d(r1, r2):.3f} | {d(r1, r2):.3f} |")
print(f"|     excess over the true change | {d(n1, n2) - d(r1, r2):+.3f} | {d(d1, d2) - d(r1, r2):+.3f} |")
