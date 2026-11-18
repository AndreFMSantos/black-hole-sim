#!/usr/bin/env python3
"""O4c: the GPU paths against the analytic Kerr oracle (an independent truth), per ray.

  oracle_rays.py BINARY

`--dump-oracle` writes the same per-ray dump as the GPU (tools/raycmp.py format) from core/include/bhs/kerr_analytic.hpp: closed-form polar motion, radial Mino integrals
by tanh-sinh, exact at infinity. Not an integrator: errors of the GPU kernels are not shared with it. Measured on a = 0.9, i = 70, 192x108 rays:
double (DOP853, tol 1e-9) sky 2.5e-10, disk r 2.5e-10; float (production) sky 2.2e-4, disk r 2.0e-4, disk phi 5.7e-4 (maxima).
Checked per scene: no fate/disk-hit difference (double), double max errors < 1e-8, float p99 sky and disk r < 4e-4 and fate differences <= 0.05 %.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import raycmp  # noqa: E402

B = sys.argv[1]
SCENES = {
    "a0.9 i70": ["--a", "0.9", "--theta", "70", "--dist", "40", "--fov", "30", "--rout", "20"],
    "a0.5 i30": ["--a", "0.5", "--theta", "30", "--dist", "60", "--fov", "24", "--rout", "30"],
    "a0.998 i85": ["--a", "0.998", "--theta", "85", "--dist", "40", "--fov", "30", "--rout", "16"],
    "a0 i60": ["--a", "0", "--theta", "60", "--dist", "30", "--fov", "40", "--rout", "20"],
}
COMMON = ["--nohud", "--nosky", "--w", "640", "--h", "360", "--shot", "x", "--dump-grid", "128"]


def dump(tmp, name, args):
    path = os.path.join(tmp, name + ".bin")
    p = subprocess.run([B] + COMMON + ["--dump-rays", path] + args, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{name}: exit {p.returncode}\n{p.stdout}{p.stderr}")
    return raycmp.load(path)[2]


def errors(ref, cand):
    fate, dirs, dr = 0, [], []
    for r, c in zip(ref, cand):
        if r[0] != c[0] or r[1] != c[1]:
            fate += 1
            continue
        if r[0] == 0:
            dirs.append(raycmp.angle(raycmp.direction(r[4], r[5]), raycmp.direction(c[4], c[5])))
        if r[1]:
            dr.append(abs(c[6] - r[6]) / r[6])
    dirs.sort(); dr.sort()
    return fate, dirs, dr


q = lambda v, f: v[int(f * (len(v) - 1))] if v else 0.0
bad = 0
with tempfile.TemporaryDirectory() as tmp:
    for name, args in SCENES.items():
        key = name.replace(" ", "_")
        ref = dump(tmp, key + "-oracle", args + ["--dump-oracle"])
        dbl = dump(tmp, key + "-d", args + ["--prec", "double", "--tol", "1e-9"])
        flt = dump(tmp, key + "-f", args + ["--prec", "float"])
        fd, dd, rd = errors(ref, dbl)
        ff, df, rf = errors(ref, flt)
        n = len(ref)
        print(f"{name:11s} double: fate diff {fd} | sky max {dd[-1]:.2e} | disk r max {rd[-1]:.2e}   float: fate diff {ff} | sky p99 {q(df, .99):.2e} | disk r p99 {q(rf, .99):.2e}  ({n} rays)")
        for ok, what in [(fd == 0, "double: no fate/disk-hit difference"), (dd[-1] < 1e-8 and rd[-1] < 1e-8, "double: sky and disk r max < 1e-8"),
                         (ff <= 0.0005 * n, "float: fate/disk-hit differences <= 0.05 %"), (q(df, .99) < 4e-4 and q(rf, .99) < 4e-4, "float: p99 sky and disk r < 4e-4")]:
            if not ok:
                print(f"FAILED {name}: {what}")
                bad += 1
print("oracle rays OK" if not bad else f"oracle rays: {bad} checks FAILED")
sys.exit(1 if bad else 0)
