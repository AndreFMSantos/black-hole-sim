#!/usr/bin/env python3
"""Validation of v0.8c: OptiX denoiser vs samples. Reference = --samples 256 without denoiser (independent seed/offset).
For N = 1, 2, 4, 8, 16 (and 32, 64 without the denoiser only) measures, with and without the denoiser (and with albedo), the error vs the reference:
HDR (the average before bloom/tone mapping: MAE and PSNR with peak = 99.9th percentile of the reference) and sRGB (after tone mapping:
MAE 0-255, PSNR(255), % of pixels with error > 16), and the ms per frame (trace + denoiser).
Usage: uv run --with numpy --with pillow python tools/validate_v08c.py [--build build] [--out DIR] [--w 1920 --h 1080] [--a 0.9 --theta 80]"""
import argparse, os, re, subprocess, numpy as np
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--build", default="build"); ap.add_argument("--out", default="/tmp/bhs-v08c")
ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
ap.add_argument("--a", default="0.9"); ap.add_argument("--theta", default="80"); ap.add_argument("--extra", default="", help="flags extra (ex.: --cinematic)")
ap.add_argument("--refn", type=int, default=256)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
GPU = f"{A.build}/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui"
base = ["--a", A.a, "--theta", A.theta, "--w", str(A.w), "--h", str(A.h), "--prec", "float", "--runs", "3"] + A.extra.split()

def pfm(path):
    with open(path, "rb") as f:
        f.readline(); w, h = map(int, f.readline().split()); f.readline()
        d = np.frombuffer(f.read(), dtype="<f4").reshape(h, w, 3)
    return d[::-1]
def run(tag, n, seed, off, extra=()):
    png, hdr = f"{A.out}/{tag}.png", f"{A.out}/{tag}.pfm"
    o = subprocess.run([GPU, "--shot", png, "--hdr", hdr, "--samples", str(n), "--seed", str(seed), "--sample-offset", str(off), *base, *extra],
                       capture_output=True, text=True, check=True).stdout
    ms = float(re.search(r"mean of \d+: ([\d.]+) ms", o).group(1))
    dn = float(re.search(r"denoiser ([\d.]+) ms", o).group(1))
    return png, hdr, ms, dn
def metrics(ref_png, ref_hdr, png, hdr, peak):
    r, t = np.asarray(Image.open(ref_png).convert("RGB")).astype(float), np.asarray(Image.open(png).convert("RGB")).astype(float)
    d = t - r; mae = np.abs(d).mean(); mse = (d ** 2).mean(); p16 = 100 * (np.abs(d).max(axis=2) > 16).mean()
    rh, th = pfm(ref_hdr), pfm(hdr); dh = th - rh
    return dict(hmae=np.abs(dh).mean(), hpsnr=10 * np.log10(peak ** 2 / (dh ** 2).mean()), smae=mae, spsnr=10 * np.log10(255 ** 2 / mse), p16=p16)

rp, rh, rms, _ = run("ref", A.refn, 1, 5000)
lum = pfm(rh) @ np.array([0.2126, 0.7152, 0.0722]); peak = float(np.percentile(pfm(rh), 99.9))
print(f"ref: {A.refn} samples ({rms:.0f} ms), HDR peak (p99.9) = {peak:.3f}\n")
print("| N | mode | ms/frame | denoiser ms | HDR MAE | HDR PSNR (dB) | sRGB MAE | sRGB PSNR (dB) | > 16 |\n|---|---|---|---|---|---|---|---|---|")
rows = {}
for n in (1, 2, 4, 8, 16, 32, 64):
    modes = [("none", ())] + ([("optix", ("--denoise", "optix")), ("optix+albedo", ("--denoise", "optix", "--denoise-albedo"))] if n <= 16 else [])
    for name, ex in modes:
        png, hdr, ms, dn = run(f"n{n}_{name}", n, 2, 100, ex)
        m = metrics(rp, rh, png, hdr, peak); rows[(n, name)] = m | dict(ms=ms)
        print(f"| {n} | {name} | {ms:.1f} | {dn:.2f} | {m['hmae']:.4f} | {m['hpsnr']:.2f} | {m['smae']:.3f} | {m['spsnr']:.2f} | {m['p16']:.2f} % |")
tgt = rows[(64, "none")]
print(f"\ntarget: 64 samples without denoiser: sRGB PSNR {tgt['spsnr']:.2f} dB, HDR PSNR {tgt['hpsnr']:.2f} dB, {tgt['ms']:.1f} ms")
for name in ("optix", "optix+albedo"):
    ok = [n for n in (1, 2, 4, 8, 16) if rows[(n, name)]["spsnr"] >= tgt["spsnr"]]
    print(f"  {name}: smallest N matching the target's sRGB PSNR: {ok[0] if ok else 'none (<= 16)'}")
