#!/usr/bin/env python3
"""Compares two PNGs (CPU v0.4 vs GPU v0.5): mean absolute error per channel (0-255) and % of pixels with a large error.
Usage: compare_cpu_gpu.py cpu.png gpu.png [diff.png] [--mask disk_mask.png]
  --mask: PNG rendered with --sky-exposure 0 (disk only); pixels with luminance > 2 = disk region.
  With a mask, it reports separately: disk (interior), edge (3 px ring around the disk/sky boundary), sky."""
import sys, numpy as np
from PIL import Image
args = [a for a in sys.argv[1:] if not a.startswith("--")]
mask_path = None
if "--mask" in sys.argv:
    mask_path = sys.argv[sys.argv.index("--mask") + 1]
    args.remove(mask_path)
a = np.asarray(Image.open(args[0]).convert("RGB")).astype(np.int32)
b = np.asarray(Image.open(args[1]).convert("RGB")).astype(np.int32)
d = np.abs(a - b)
mx = d.max(axis=2)

def report(name, m):
    if not m.any(): return
    print(f"{name:34s} {100*m.mean():6.2f}% pixels  MAE {d[m].mean():.3f}  >4: {100*(mx[m]>4).mean():.3f}%  >16: {100*(mx[m]>16).mean():.3f}%  >32: {100*(mx[m]>32).mean():.3f}%")

print(f"shape {a.shape}  MAE por canal R,G,B: {[round(float(d[..., k].mean()), 3) for k in range(3)]}")
report("total", np.ones(mx.shape, bool))
if mask_path:
    disk = np.asarray(Image.open(mask_path).convert("L")).astype(np.int32) > 2
    grow = disk.copy()
    for _ in range(3):  # dilata 3 px
        g = grow.copy()
        g[1:, :] |= grow[:-1, :]; g[:-1, :] |= grow[1:, :]; g[:, 1:] |= grow[:, :-1]; g[:, :-1] |= grow[:, 1:]
        grow = g
    shrink = ~grow.copy()  # unused
    er = disk.copy()
    for _ in range(3):  # erode 3 px
        e = er.copy()
        e[1:, :] &= er[:-1, :]; e[:-1, :] &= er[1:, :]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]
        er = e
    edge = grow & ~er
    report("disk (interior)", er)
    report("disk/sky edge (+-3 px)", edge)
    report("sky (far from the disk)", ~grow)
if len(args) > 2:
    Image.fromarray(np.clip(mx * 4, 0, 255).astype(np.uint8)).save(args[2])
