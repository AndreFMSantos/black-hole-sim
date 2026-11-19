#!/usr/bin/env python3
"""Golden images for the refactor (v4.0): one small deterministic render per supported (metric, disk, precision) combination.

  golden.py make  BINARY   write images/ and manifest.json from BINARY (do this ONCE, with the frozen v3_1 app, before touching the code)
  golden.py check BINARY   render again and compare pixel by pixel with images/ (exit 1 on any difference)

All renders: 256x144, 1 sample, fixed seed, --nohud; `--nosky` except where the sky is the point (wormholes, one Kerr with sky).
Comparison is bit-exact on the 8-bit PNG; a refactor that changes float arithmetic may need a written exception (max abs diff reported).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = ["--nohud", "--w", "256", "--h", "144", "--samples", "1", "--seed", "7"]
F = ["--prec", "float", "--tol", "2e-5"]
D = ["--prec", "double"]
NOSKY = ["--nosky"]
CASES = {
    # Kerr, thin disks
    "kerr-nt-a09-float": NOSKY + F + ["--a", "0.9", "--theta", "80", "--dist", "40"],
    "kerr-nt-a09-double": NOSKY + D + ["--a", "0.9", "--theta", "80", "--dist", "40"],
    "kerr-newtonian-a09-float": NOSKY + F + ["--a", "0.9", "--theta", "80", "--dist", "40", "--profile", "newtonian"],
    "schwarzschild-shadow-float": F + ["--a", "0", "--theta", "80", "--dist", "40", "--nodisk"],
    "kerr-nt-a0998-edgeon-double": NOSKY + D + ["--a", "0.998", "--theta", "88", "--dist", "40"],
    "kerr-nt-sky-float": F + ["--a", "0.9", "--theta", "80", "--dist", "40"],
    # Kerr, other disks / observers
    "kerr-thick-layer-float": NOSKY + F + ["--a", "0.9", "--theta", "80", "--dist", "40", "--hr", "0.06", "--tau", "4", "--flare", "0.15"],
    "kerr-turbulence-spot-float": NOSKY + F + ["--a", "0.9", "--theta", "75", "--dist", "40", "--hr", "0.06", "--tau", "4", "--flare", "0.15", "--turb", "1.2", "--turb-scale", "1.2", "--edge-soft", "0.12", "--spot", "9,200,1.2,5", "--time", "100"],
    "kerr-orbit-observer-float": NOSKY + F + ["--a", "0.9", "--theta", "80", "--dist", "12", "--observer", "orbit"],
    "kerr-infall-observer-float": NOSKY + F + ["--a", "0.9", "--theta", "80", "--dist", "12", "--observer", "infall"],
    "kerr-torus-radio-float": NOSKY + F + ["--a", "0.9", "--disk-model", "torus", "--torus", "7,12", "--ne0", "3e4", "--plasma-beta", "3", "--theta-e-max", "20", "--dist", "150", "--fov", "16", "--theta", "17", "--mbh", "6.5e9", "--nu-ghz", "230", "--tonemap", "aces"],
    # Kerr-Newman family
    "reissner-nordstrom-float": NOSKY + F + ["--a", "0", "--charge", "0.6", "--theta", "78", "--dist", "30"],
    "kerr-newman-float": NOSKY + F + ["--a", "0.5", "--charge", "0.5", "--theta", "78", "--dist", "40"],
    "kerr-newman-double": NOSKY + D + ["--a", "0.5", "--charge", "0.5", "--theta", "78", "--dist", "40"],
    "naked-singularity-float": NOSKY + F + ["--a", "0", "--charge", "1.1", "--profile", "newtonian", "--rin", "6", "--theta", "78", "--dist", "30"],
    # Schwarzschild-de Sitter
    "sds-shadow-float": F + ["--a", "0", "--lambda", "0.01", "--nodisk", "--theta", "90", "--dist", "8", "--fov", "70"],
    "sds-shadow-double": D + ["--a", "0", "--lambda", "0.01", "--nodisk", "--theta", "90", "--dist", "8", "--fov", "70"],
    "sds-disk-float": NOSKY + F + ["--a", "0", "--lambda", "0.01", "--profile", "newtonian", "--rin", "6", "--theta", "78", "--dist", "9"],
    # wormholes (sky needed: the second sky is the point)
    "wormhole-ellis-float": F + ["--wormhole", "ellis,1", "--wh-l", "4", "--fov", "50", "--theta", "90"],
    "wormhole-interstellar-float": F + ["--wormhole", "interstellar", "--wh-l", "4", "--fov", "50", "--theta", "90"],
}


# every float case gets a double twin (same scene, --prec double) so a refactor that changes float rounding can be judged against the double reference
for _n in [n for n in CASES if n.endswith("-float")]:
    _t = _n[:-6] + "-double"
    if _t not in CASES:
        _a = list(CASES[_n]); _i = _a.index("--prec"); _a[_i + 1] = "double"
        if "--tol" in _a:
            _j = _a.index("--tol"); del _a[_j:_j + 2]
        CASES[_t] = _a

# float rounding tolerance of a refactor: a float image may differ from its golden in the last bits (FMA contraction) only if it is not further from the
# double reference than the golden float was
FLOAT_SLACK_REL, FLOAT_SLACK_ABS = 1.02, 0.02


# v4.0 changes the physics of these cases on purpose: the infall observer uses the exact Kerr velocity (lorentz::infall_beta_kerr) instead of sqrt(2/r).
# Their reference images come from images_v4/ (made with `golden.py make-v4 BINARY`); every other case still compares with the v3_1 oracle in images/.
V4_CHANGED = {"kerr-infall-observer-float", "kerr-infall-observer-double",
              # O1 (refinement phase, 2026-11-17): the analytic escape tail makes the exit direction asymptotic (error ~1e-9 rad instead of
              # O(b / r_far) ~ 1.6e-5 rad at r_far = 1e6), so the stars move by a fraction of a pixel in the double images that show the sky.
              # The new images are the more accurate ones (tests/escape_tail.py checks the tail against a reference integrated to r_far = 1e10).
              "schwarzschild-shadow-double", "kerr-nt-sky-double",
              # O3 + fixes (2026-11-17, notes/physics/33-dop853.md): DOP853 on the double thin-disk path (calibrated, no worse than DP45 in every quantile)
              # and the reprojection only where well conditioned: 1-level changes in a few pixels of the double images (more accurate rays)
              "kerr-newman-double", "kerr-orbit-observer-double", "reissner-nordstrom-double", "naked-singularity-double",
              # Backport of O3 reprojection to v3_1 (2026-11-19) shifted 1 pixel by 1 level vs v4_0 (O1/O2/O6)
              "kerr-thick-layer-double",
              # Schwarzschild-de Sitter: the initial dr/dtau lacked the Lambda term since v2.1c (sky errors up to 7e-2 rad in the disk scene, 270 rays with a
              # wrong fate in 36864) and the exit at 0.9999 r_c was not interpolated: both images of each SdS case are new (physics fix)
              "sds-shadow-double", "sds-shadow-float", "sds-disk-double", "sds-disk-float"}


V4_BINARY = False  # set in main(): only the v4_0 binary is compared with images_v4 (the frozen v3_1 keeps the oracle images)


def ref_dir(name, img):
    return HERE / "images_v4" if V4_BINARY and name in V4_CHANGED and (HERE / "images_v4").exists() else img


def render(binary, name, out):
    cmd = [str(binary)] + BASE + CASES[name] + ["--shot", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not Path(out).exists():
        raise RuntimeError(f"{name}: exit {r.returncode}\n{r.stdout[-300:]}{r.stderr[-300:]}")
    return " ".join(cmd[1:])


def main():
    global V4_BINARY
    mode, binary = sys.argv[1], Path(sys.argv[2]).resolve()
    V4_BINARY = binary.name.startswith("v4_0")
    img = HERE / "images"; img.mkdir(exist_ok=True)
    man = HERE / "manifest.json"
    if mode == "make-v4":
        d = HERE / "images_v4"; d.mkdir(exist_ok=True)
        for n in sorted(V4_CHANGED):
            render(binary, n, d / f"{n}.png"); print("made (v4)", n)
        return 0
    if mode == "make":
        meta = {"binary": binary.name, "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(), "cases": {}}
        for n in CASES:
            meta["cases"][n] = render(binary, n, img / f"{n}.png").replace(str(img), "images")
            print("made", n)
        man.write_text(json.dumps(meta, indent=1))
        return 0
    bad = 0
    tmp = HERE / "_check"; tmp.mkdir(exist_ok=True)
    for n in CASES:
        render(binary, n, tmp / f"{n}.png")
        a = np.asarray(Image.open(ref_dir(n, img) / f"{n}.png").convert("RGB"), int); b = np.asarray(Image.open(tmp / f"{n}.png").convert("RGB"), int)
        d = np.abs(a - b)
        ok = a.shape == b.shape and d.max() == 0
        note = ""
        if not ok and n.startswith("wormhole") and d.max() <= 1 and (d.max(2) > 0).mean() < 0.001:  # the wormhole tracer is float-only code (the --prec flag does not reach it): last-bit differences from a changed instantiation
            ok = True; note = "  last-bit (wormhole tracer is float-only)"
        if not ok and n.endswith("-float") and (ref_dir(n[:-6] + "-double", img) / (n[:-6] + "-double.png")).exists():
            ref = np.asarray(Image.open(ref_dir(n[:-6] + "-double", img) / (n[:-6] + "-double.png")).convert("RGB"), int)
            e_old, e_new = np.abs(a - ref).mean(), np.abs(b - ref).mean()
            ok = e_new <= FLOAT_SLACK_REL * e_old + FLOAT_SLACK_ABS
            note = f"  float rounding: MAE vs double {e_old:.4f} -> {e_new:.4f}, vs golden {d.mean():.4f}"
        bad += not ok
        print(f"{'ok  ' if ok and not note else ('ok~ ' if ok else 'DIFF')} {n}" + ("" if ok and not note else f"  max abs diff {d.max()}, {int((d.max(2) > 0).sum())} px{note}"))
        (tmp / f"{n}.png").unlink()
    tmp.rmdir()
    print(f"{len(CASES) - bad}/{len(CASES)} pass (identical, or float within the rounding criterion)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
