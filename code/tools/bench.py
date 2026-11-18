#!/usr/bin/env python3
"""Fixed performance bench for the v4_0 app (refinement phase R0.1, notes/plan-refinement-phase.md).

Runs every scenario with `--shot` (N runs after warm-up), parses ms/frame, steps per ray and warp efficiency, writes
`data/bench/<date>-<commit>.csv` and prints the change against the previous CSV in that folder.

    bench.py [--bin PATH] [--runs 5] [--only name,name] [--no-save]

Scenarios needing data (GRMHD) are skipped when the file is missing. Standard library only.
"""
import argparse
import csv
import datetime
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PROJECT = os.path.dirname(CODE)
DEFAULT_BIN = os.path.join(CODE, "build", "apps", "v4_0-metric-api", "v4_0-metric-api")
GRMHD = os.path.join(PROJECT, "data", "bgm", "MAD_a0_5500.bgm")
OUT_DIR = os.path.join(PROJECT, "data", "bench")

HD = ["--w", "1920", "--h", "1080"]
SQ = ["--w", "700", "--h", "700"]
FLOAT = ["--prec", "float"]
# name -> (args, needs file or None)
SCENARIOS = {
    "thin-float-1080p": (HD + FLOAT, None),
    "thin-double-1080p": (HD + ["--prec", "double"], None),
    "layer-float-1080p": (HD + FLOAT + ["--hr", "0.1"], None),
    "torus-float-700": (SQ + FLOAT + ["--disk-model", "torus"], None),
    "riaf-float-700": (SQ + FLOAT + ["--disk-model", "riaf"], None),
    "spherical-float-700": (SQ + FLOAT + ["--disk-model", "spherical", "--a", "0.9", "--theta", "60", "--dist", "100", "--fov", "24"], None),
    "cones-float-700": (SQ + FLOAT + ["--disk-model", "cones", "--a", "0.5", "--theta", "80", "--dist", "100", "--fov", "24"], None),
    "grmhd-mad-float-720p": (["--w", "1280", "--h", "720"] + FLOAT + ["--disk-model", "grmhd", "--grmhd", GRMHD], GRMHD),
    "kn-float-1080p": (HD + FLOAT + ["--a", "0.5", "--charge", "0.5", "--rin", "6"], None),
    "sds-float-1080p": (HD + FLOAT + ["--a", "0", "--lambda", "0.01", "--rin", "6", "--dist", "7"], None),
    "wormhole-float-1080p": (HD + FLOAT + ["--wormhole", "ellis"], None),
}

RE_MS = re.compile(r"mean of \d+: ([0-9.]+) ms/frame")
RE_STEPS = re.compile(r"steps per ray: mean ([0-9.]+), max (\d+); warp efficiency \(sum/sum of max\*32\) ([0-9.]+)")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", CODE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def run(binary, name, args, runs, tmp):
    cmd = [binary, "--shot", os.path.join(tmp, name + ".png"), "--runs", str(runs)] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    ms, st = RE_MS.search(out), RE_STEPS.search(out)
    if p.returncode != 0 or not ms:
        print(f"{name}: FAILED (exit {p.returncode})\n{out[-600:]}", file=sys.stderr)
        return None
    row = {"scenario": name, "ms": float(ms.group(1))}
    if st:
        row.update(steps_mean=float(st.group(1)), steps_max=int(st.group(2)), warp_eff=float(st.group(3)))
    return row


def previous_csv(exclude):
    if not os.path.isdir(OUT_DIR):
        return None
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".csv") and os.path.join(OUT_DIR, f) != exclude)
    return os.path.join(OUT_DIR, files[-1]) if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    names = [n for n in SCENARIOS if not a.only or n in a.only.split(",")]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in names:
            args, needs = SCENARIOS[n]
            if needs and not os.path.exists(needs):
                print(f"{n}: skipped (missing {needs})")
                continue
            r = run(a.bin, n, args, a.runs, tmp)
            if r:
                rows.append(r)
                print(f"{n:24s} {r['ms']:9.3f} ms  steps {r.get('steps_mean', float('nan')):7.1f}  warp {r.get('warp_eff', float('nan')):.2f}", flush=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    path = os.path.join(OUT_DIR, f"{stamp}-{git_commit()}.csv")
    prev = previous_csv(path)
    if prev:
        with open(prev) as f:
            old = {r["scenario"]: r for r in csv.DictReader(f)}
        print(f"\nagainst {os.path.basename(prev)}:")
        for r in rows:
            o = old.get(r["scenario"])
            if o:
                print(f"  {r['scenario']:24s} {float(o['ms']):9.3f} -> {r['ms']:9.3f} ms  ({float(o['ms']) / r['ms']:.2f}x)")
    if not a.no_save and rows:
        os.makedirs(OUT_DIR, exist_ok=True)
        keys = ["scenario", "ms", "steps_mean", "steps_max", "warp_eff"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")
    return 0 if len(rows) == len([n for n in names if not SCENARIOS[n][1] or os.path.exists(SCENARIOS[n][1])]) else 1


if __name__ == "__main__":
    sys.exit(main())
