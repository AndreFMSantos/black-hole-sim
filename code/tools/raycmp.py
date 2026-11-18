#!/usr/bin/env python3
"""Compare two per-ray dumps (`--shot x --dump-rays f.bin`) of the v4_0 app: candidate against reference.

Refinement phase, rule 2 (notes/plan-refinement-phase.md): an optimisation that changes the step sequence or the end of
the ray is judged against a double reference (e.g. --prec double --tol 1e-9 --rfar 1e8), not by md5.

Reported per quantity: max, p99 and median error over the rays where both dumps agree on the outcome, plus the rays whose
fate (escaped / captured / orbiting) or disk hit differ. Sky direction error = angle between the two exit directions (rad).

    raycmp.py ref.bin cand.bin [--max-dir RAD] [--max-disk-r REL] [--max-fate-frac F]

Exit status 1 when a given threshold is exceeded (for ctest). Standard library only.
"""
import argparse
import math
import struct
import sys

REC = struct.Struct("<4i4d")


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"BHSRAYS1":
        sys.exit(f"{path}: not a BHSRAYS1 dump")
    nx, ny = struct.unpack_from("<2i", data, 8)
    off, n = 16, nx * ny
    if len(data) != off + n * REC.size:
        sys.exit(f"{path}: size does not match {nx}x{ny}")
    return nx, ny, [REC.unpack_from(data, off + i * REC.size) for i in range(n)]


def direction(mu, phi):
    st = math.sqrt(max(0.0, 1.0 - mu * mu))
    return (st * math.cos(phi), st * math.sin(phi), mu)


def angle(a, b):
    # atan2(|a x b|, a.b): accurate also for tiny angles
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def summary(values):
    if not values:
        return "n/a (no rays)"
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]
    return f"max {v[-1]:.3e}  p99 {pick(0.99):.3e}  median {pick(0.5):.3e}  ({len(v)} rays)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--max-dir", type=float, help="fail if the max sky-direction error (rad) exceeds this")
    ap.add_argument("--max-disk-r", type=float, help="fail if the max relative disk_r error exceeds this")
    ap.add_argument("--max-fate-frac", type=float, default=None, help="fail if the fraction of rays with a different fate/disk hit exceeds this")
    a = ap.parse_args()
    nx, ny, ref = load(a.ref)
    nx2, ny2, cand = load(a.cand)
    if (nx, ny) != (nx2, ny2):
        sys.exit(f"grids differ: {nx}x{ny} vs {nx2}x{ny2}")
    dirs, dr, dphi, fate_diff = [], [], [], 0
    for r, c in zip(ref, cand):
        if r[0] != c[0] or r[1] != c[1]:
            fate_diff += 1
            continue
        if r[0] == 0:
            dirs.append(angle(direction(r[4], r[5]), direction(c[4], c[5])))
        if r[1]:
            dr.append(abs(c[6] - r[6]) / max(abs(r[6]), 1e-300))
            dphi.append(abs(c[7] - r[7]))
    n = len(ref)
    steps = lambda d: sum(x[2] for x in d) / len(d)
    rej = lambda d: 100.0 * sum(x[3] for x in d) / max(1, sum(x[2] + x[3] for x in d))
    print(f"grid {nx}x{ny} = {n} rays")
    print(f"steps/ray: ref {steps(ref):.1f} ({rej(ref):.2f} % rejected), cand {steps(cand):.1f} ({rej(cand):.2f} % rejected)")
    print(f"fate or disk hit differs: {fate_diff} rays ({100.0 * fate_diff / n:.3f} %)")
    print(f"sky direction (rad):   {summary(dirs)}")
    print(f"disk_r (relative):     {summary(dr)}")
    print(f"disk_phi (rad):        {summary(dphi)}")
    bad = []
    if a.max_dir is not None and dirs and max(dirs) > a.max_dir:
        bad.append(f"sky direction {max(dirs):.3e} > {a.max_dir:g}")
    if a.max_disk_r is not None and dr and max(dr) > a.max_disk_r:
        bad.append(f"disk_r {max(dr):.3e} > {a.max_disk_r:g}")
    if a.max_fate_frac is not None and fate_diff / n > a.max_fate_frac:
        bad.append(f"fate differs on {fate_diff / n:.3e} of the rays > {a.max_fate_frac:g}")
    for b in bad:
        print("FAILED:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
