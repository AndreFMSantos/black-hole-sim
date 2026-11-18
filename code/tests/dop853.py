#!/usr/bin/env python3
"""O3 (refinement phase): the DOP853 integrator of the double thin-disk/sky path.

  dop853.py BINARY

For Kerr, Kerr-Newman, a far camera and Schwarzschild-de Sitter:
  1. two independent tight references agree (DOP853 tol 1e-13 vs DP45 tol 1e-11): sky direction and disk r within 1e-8. A wrong coefficient
     in the generated tableau (tools/gen_dop853.py) would drop the order and break this;
  2. the default double run (DOP853, calibrated at tol/10) is no worse than DP45 at the same --tol in EVERY quantile (max, p99, median) of the
     sky direction, disk r and disk phi, against the DOP853 1e-13 reference (rule 2 of notes/plan-refinement-phase.md);
  3. it takes fewer accepted steps per ray.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import raycmp  # noqa: E402

B = sys.argv[1]
CASES = {"kerr": ["--a", "0.9"], "kerr-newman": ["--a", "0.5", "--charge", "0.5", "--rin", "6"], "kerr-far-camera": ["--a", "0.7", "--dist", "500", "--fov", "4"],
         # SdS: the exit at 0.9999 r_c is interpolated, and the initial dr/dtau now includes Lambda (before the fix the two references differed by 6e-3 rad)
         "sds": ["--a", "0", "--lambda", "0.01", "--nodisk", "--theta", "90", "--dist", "8", "--fov", "70"]}


def dump(tmp, name, args):
    path = os.path.join(tmp, name + ".bin")
    p = subprocess.run([B, "--shot", "x", "--dump-rays", path, "--dump-grid", "128", "--prec", "double", "--max-steps", "400000"] + args, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{name}: exit {p.returncode}\n{p.stdout}{p.stderr}")
    return raycmp.load(path)[2]


def quantiles(ref, cand):
    q = {"dir": [], "disk_r": [], "disk_phi": []}
    fate = 0
    for r, c in zip(ref, cand):
        if r[0] != c[0] or r[1] != c[1]:
            fate += 1
            continue
        if r[0] == 0:
            q["dir"].append(raycmp.angle(raycmp.direction(r[4], r[5]), raycmp.direction(c[4], c[5])))
        if r[1]:
            q["disk_r"].append(abs(c[6] - r[6]) / r[6])
            q["disk_phi"].append(abs(c[7] - r[7]))
    out = {}
    for k, v in q.items():
        v.sort()
        out[k] = (v[-1], v[int(0.99 * (len(v) - 1))], v[len(v) // 2]) if v else (0.0, 0.0, 0.0)
    return fate, out


bad = 0
with tempfile.TemporaryDirectory() as tmp:
    for name, args in CASES.items():
        ref = dump(tmp, name + "-r853", args + ["--integrator", "dop853", "--tol", "1e-13"])
        ref45 = dump(tmp, name + "-r45", args + ["--integrator", "dp45", "--tol", "1e-11"])
        d45 = dump(tmp, name + "-45", args + ["--integrator", "dp45"])
        d853 = dump(tmp, name + "-853", args + ["--integrator", "dop853"])
        f0, qr = quantiles(ref, ref45)
        f45, q45 = quantiles(ref, d45)
        f853, q853 = quantiles(ref, d853)
        s45 = sum(x[2] for x in d45) / len(d45)
        s853 = sum(x[2] for x in d853) / len(d853)
        print(f"{name}: references agree: fate diff {f0}, dir max {qr['dir'][0]:.2e}, disk_r max {qr['disk_r'][0]:.2e}; steps/ray dp45 {s45:.1f}, dop853 {s853:.1f}")
        checks = [(f0 == 0 and qr["dir"][0] < 1e-8 and qr["disk_r"][0] < 1e-8, "independent references agree within 1e-8"),
                  (f853 <= f45, "dop853: no more fate/disk-hit differences than dp45"),
                  (s853 < s45, "dop853: fewer accepted steps")]
        for k in ("dir", "disk_r", "disk_phi"):
            print(f"  {k:8s} dp45 max/p99/median {q45[k][0]:.2e} {q45[k][1]:.2e} {q45[k][2]:.2e} | dop853 {q853[k][0]:.2e} {q853[k][1]:.2e} {q853[k][2]:.2e}")
            checks.append((all(q853[k][i] <= q45[k][i] for i in range(3)), f"dop853 no worse than dp45 in every quantile of {k}"))
        for ok, what in checks:
            if not ok:
                print(f"FAILED {name}: {what}")
                bad += 1
print("dop853 OK" if not bad else f"dop853: {bad} checks FAILED")
sys.exit(1 if bad else 0)
