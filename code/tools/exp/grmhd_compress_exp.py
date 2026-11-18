# Experiment (v2.2f-a2): classical compression of an Illinois GRMHD snapshot. Usage: uv run --with h5py --with numpy python -I tools/exp/grmhd_compress_exp.py <torus.out0.NNNNN.h5>
import sys, time, zlib, h5py, numpy as np
f = h5py.File(sys.argv[1], 'r'); P = f['prims'][...]; H = f['header']
x1 = H['geom']['startx1'][()] + (np.arange(288) + 0.5) * H['dx1'][()]; r = np.exp(x1)
raw = P.nbytes
print(f"raw: {raw/1e6:.1f} MB  (4.72 M cells x 8 float32 variables)")
rho = P[..., 0].astype(np.float64); uu = P[..., 1].astype(np.float64)
lrho = np.log10(rho)
print(f"log10 rho: {lrho.min():.2f} .. {lrho.max():.2f}  (range of {lrho.max()-lrho.min():.1f} dex)")

# 1. lossless compression (byte-shuffle + zlib): how much fits without touching the values
t0 = time.time()
b = np.ascontiguousarray(P).view(np.uint8).reshape(-1, 4)
sh = np.ascontiguousarray(b.T).tobytes()
c = zlib.compress(sh, 6)
print(f"1. lossless (shuffle+zlib 6): {len(c)/1e6:.1f} MB  = {raw/len(c):.2f}x  ({time.time()-t0:.0f} s)")

# 2. float16 / 8-10 bit in log: error in dex in log10 rho and relative error in Theta_i
def q_err(a, bits):
    lo, hi = a.min(), a.max(); n = (1 << bits) - 1
    q = np.round((a - lo) / (hi - lo) * n); return (q / n * (hi - lo) + lo) - a
for bits in (16, 12, 10, 8):
    e = q_err(lrho, bits)
    print(f"2. log10 rho in {bits:2d} bits: rms error {np.sqrt((e**2).mean()):.4f} dex, max {np.abs(e).max():.4f} dex  ({raw/8*bits/8/1e6:.1f} MB for this variable alone)")
e16 = (lrho.astype(np.float16).astype(np.float64) - lrho)
print(f"   float16 of log10 rho: rms error {np.sqrt((e16**2).mean()):.4f} dex, max {np.abs(e16).max():.4f} dex")

# 3. radial cut: what remains with r < R
mass = rho * r[:, None, None]**3
for R in (30, 60, 100, 200):
    n = int(np.searchsorted(r, R)); frac_cells = n / 288
    print(f"3. cut r < {R:3d} (i < {n:3d}): {frac_cells*100:5.1f} % of the cells, {mass[:n].sum()/mass.sum()*100:5.1f} % of the mass (rho r^3)")

# 4. spatial subsampling by 2 (block mean) and trilinear reconstruction: error in log10 rho
from numpy import add
d = lrho.reshape(144, 2, 64, 2, 64, 2).mean(axis=(1, 3, 5))
def up(a):
    for ax in range(3):
        n = a.shape[ax]; idx = (np.arange(2 * n) + 0.5) / 2 - 0.5
        i0 = np.clip(np.floor(idx).astype(int), 0, n - 1); i1 = np.clip(i0 + 1, 0, n - 1); w = (idx - np.floor(idx)).reshape([-1 if k == ax else 1 for k in range(3)])
        a = np.take(a, i0, axis=ax) * (1 - w) + np.take(a, i1, axis=ax) * w
    return a
rec = up(d)
e = rec - lrho
print(f"4. 2x subsampling per axis (8x fewer cells, {raw/8/1e6:.1f} MB for all vars): rms error {np.sqrt((e**2).mean()):.4f} dex, p99.9 {np.percentile(np.abs(e),99.9):.3f} dex")
