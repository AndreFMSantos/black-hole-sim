#!/usr/bin/env python3
"""O2 (refinement phase): the float step control must not waste attempts, and must stay as accurate as before.

  step_control.py BINARY

Before O2 the float integrator rejected 34 % of its step attempts (on the way out, r(tau) runs into a pole at finite Mino time and the
error-based controller lagged one step behind). Checks, on the default scene and on Kerr-Newman, against a double reference (tol 1e-11):
  - float: rejected attempts < 10 % of all attempts;
  - float accuracy as before O2 (measured on this grid, old -> new controller; Kerr: dir p99 2.78e-4 -> 2.77e-4, median 9.7e-6 -> 1.0e-5;
    KN: p99 3.07e-4 both): fate/disk-hit differences <= 0.05 % of the rays, sky direction p99 < 4e-4 rad and median < 2e-5 rad,
    disk_r p99 < 4e-4. Maxima are not tested: near-critical rays (photon ring) amplify the float error exponentially (one ray per grid,
    the same with both controllers: KN max 2.3e-2 rad).
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import raycmp  # noqa: E402

B = sys.argv[1]
CASES = {"kerr": ["--a", "0.9"], "kerr-newman": ["--a", "0.5", "--charge", "0.5", "--rin", "6"]}


def dump(tmp, name, args):
    path = os.path.join(tmp, name + ".bin")
    p = subprocess.run([B, "--shot", "x", "--dump-rays", path, "--dump-grid", "160"] + args, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{name}: exit {p.returncode}\n{p.stdout}{p.stderr}")
    return raycmp.load(path)[2]


bad = 0
with tempfile.TemporaryDirectory() as tmp:
    for name, args in CASES.items():
        ref = dump(tmp, name + "-ref", args + ["--prec", "double", "--tol", "1e-11", "--max-steps", "400000"])
        cand = dump(tmp, name + "-f", args + ["--prec", "float"])
        acc = sum(x[2] for x in cand)
        rej = sum(x[3] for x in cand)
        frac = rej / max(1, acc + rej)
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
        q = lambda v, f: v[int(f * (len(v) - 1))]
        print(f"{name:12s} float: {acc / len(cand):.1f} accepted steps/ray, rejected {100 * frac:.2f} % | fate diff {fate} | dir p99 {q(dirs, .99):.3e} median {q(dirs, .5):.3e} | disk_r p99 {q(dr, .99):.3e}")
        for ok, what in [(frac < 0.10, "rejected attempts < 10 %"), (fate <= 0.0005 * len(ref), "fate/disk-hit differences <= 0.05 %"),
                         (q(dirs, .99) < 4e-4 and q(dirs, .5) < 2e-5, "sky direction p99 < 4e-4 and median < 2e-5 rad"), (q(dr, .99) < 4e-4, "disk_r p99 < 4e-4")]:
            if not ok:
                print(f"FAILED {name}: {what}")
                bad += 1
print("step control OK" if not bad else f"step control: {bad} checks FAILED")
sys.exit(1 if bad else 0)
