#!/usr/bin/env python3
"""O1 (refinement phase): the analytic escape tail must be at least as accurate as integrating to r_far.

  escape_tail.py BINARY

Reference: double, tol 1e-11, tail OFF, integrated to r_far = 1e10 (its own finite-r error is ~b/r_far ~ 1e-9 rad).
For Kerr and Kerr-Newman (and a far camera):
  - the tail itself: double tol 1e-11 WITH the tail matches the reference to < 2e-8 rad (measured 3.5e-9);
  - double with the tail at the default tolerance: same fates/disk hits, max error at least 5x below the old default
    (tail off, r_far = 1e6, whose O(b/r_far) error dominates) and a smaller median;
  - float with the tail: no worse than float without it (max within 1.2x), same fates.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import raycmp  # noqa: E402

B = sys.argv[1]
CASES = {
    "kerr": ["--a", "0.9"],
    "kerr-newman": ["--a", "0.5", "--charge", "0.5", "--rin", "6"],
    "kerr-far-camera": ["--a", "0.7", "--dist", "500", "--fov", "4"],
}


def dump(tmp, name, args):
    path = os.path.join(tmp, name + ".bin")
    p = subprocess.run([B, "--shot", "x", "--dump-rays", path, "--dump-grid", "128"] + args, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{name}: exit {p.returncode}\n{p.stdout}{p.stderr}")
    return raycmp.load(path)[2]


def errors(ref, cand):
    dirs, fate = [], 0
    for r, c in zip(ref, cand):
        if r[0] != c[0] or r[1] != c[1]:
            fate += 1
        elif r[0] == 0:
            dirs.append(raycmp.angle(raycmp.direction(r[4], r[5]), raycmp.direction(c[4], c[5])))
    dirs.sort()
    return fate, dirs[-1], dirs[len(dirs) // 2]


bad = 0
with tempfile.TemporaryDirectory() as tmp:
    for name, args in CASES.items():
        ref = dump(tmp, name + "-ref", args + ["--prec", "double", "--tol", "1e-11", "--rtail", "0", "--rfar", "1e10", "--max-steps", "400000"])
        res = {
            "tight tail": dump(tmp, name + "-tt", args + ["--prec", "double", "--tol", "1e-11"]),
            "double tail": dump(tmp, name + "-dt", args + ["--prec", "double"]),
            "double no tail": dump(tmp, name + "-dn", args + ["--prec", "double", "--rtail", "0"]),
            "float tail": dump(tmp, name + "-ft", args + ["--prec", "float"]),
            "float no tail": dump(tmp, name + "-fn", args + ["--prec", "float", "--rtail", "0"]),
        }
        e = {k: errors(ref, v) for k, v in res.items()}
        steps = {k: sum(x[2] for x in v) / len(v) for k, v in res.items()}
        for k in res:
            print(f"{name:16s} {k:15s} fate diff {e[k][0]:3d}  dir max {e[k][1]:.3e}  median {e[k][2]:.3e}  steps {steps[k]:6.1f}")
        checks = [
            (e["double tail"][0] == 0, "double tail: same fates/disk hits"),
            (e["tight tail"][0] == 0 and e["tight tail"][1] < 2e-8, "tail at tol 1e-11: max sky direction error < 2e-8 rad"),
            (e["double tail"][1] < e["double no tail"][1] / 5, "double tail: max error at least 5x below the old default"),
            (e["double tail"][2] < e["double no tail"][2], "double tail: median error below the old default (r_far = 1e6)"),
            (e["float tail"][0] <= e["float no tail"][0], "float tail: no more fate differences than without"),
            (e["float tail"][1] <= 1.2 * e["float no tail"][1], "float tail: max error within 1.2x of without"),
            (steps["float tail"] < steps["float no tail"] and steps["double tail"] < steps["double no tail"], "fewer steps with the tail"),
        ]
        for ok, what in checks:
            if not ok:
                print(f"FAILED {name}: {what}")
                bad += 1
print("escape tail OK" if not bad else f"escape tail: {bad} checks FAILED")
sys.exit(1 if bad else 0)
