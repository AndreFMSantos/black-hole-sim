#!/usr/bin/env python3
"""Quantises the 12 fields of a .bgm (see tools/convert_grmhd.py) to N bits and writes another (dequantised) .bgm, to measure the error on the FINAL IMAGE.
rho, theta_i, |b|, beta: uniform quantisation in log10 between the field's minimum and maximum; u^mu and bhat^mu: uniform between the minimum and maximum.
Usage: uv run --with numpy python -I tools/exp/quantize_bgm.py in.bgm out.bgm N"""
import struct, sys
import numpy as np
src, dst, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
raw = open(src, 'rb').read()
n1, n2, n3, nf, _ = struct.unpack_from('<5I', raw, 4)
D = np.frombuffer(raw, dtype=np.float32, offset=256).reshape(n1, n2, n3, nf).astype(np.float64)
Q = np.empty_like(D)
levels = (1 << N) - 1
for c in range(nf):
    x = D[..., c]
    logf = c in (0, 1, 2, 3)
    v = np.log10(np.maximum(x, 1e-300)) if logf else x
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo: Q[..., c] = x; continue
    k = np.round((v - lo) / (hi - lo) * levels)
    r = lo + k / levels * (hi - lo)
    Q[..., c] = 10.0 ** r if logf else r
    err = np.abs(r - v).max()
    print(f"field {c:2d}: {'log10' if logf else 'lin  '} [{lo:.3g}, {hi:.3g}] max |error| {err:.3g}{' dex' if logf else ''}")
open(dst, 'wb').write(raw[:256] + Q.astype(np.float32).tobytes())
print(f"{src} -> {dst} ({N} bits)")
