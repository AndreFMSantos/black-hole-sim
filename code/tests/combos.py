#!/usr/bin/env python3
"""v4.0: the command line accepts exactly the (metric, disk) combinations of apps/v4_0-metric-api/combos.hpp and refuses the others.

  combos.py BINARY
Supported combinations must render (exit 0); unsupported ones must exit with an error. (The GRMHD column needs a dump and is covered by v4_0_grmhd_blend.)
"""
import subprocess
import sys

B = sys.argv[1]
METRICS = {
    "Kerr": ["--a", "0.9"], "KerrNewman": ["--a", "0.5", "--charge", "0.5"], "NakedKN": ["--a", "0.5", "--charge", "1.1", "--rin", "6"],
    "SdS": ["--a", "0", "--lambda", "0.01", "--rin", "6", "--dist", "8"], "Wormhole": ["--wormhole", "ellis,1"],
}
DISKS = {
    "None": ["--nodisk"], "NovikovThorne": ["--profile", "nt"], "Newtonian": ["--profile", "newtonian"], "Thick": ["--hr", "0.06", "--tau", "4"],
    "Torus": ["--disk-model", "torus", "--torus", "7,12", "--dist", "150", "--fov", "16"],
    "Riaf": ["--disk-model", "riaf", "--dist", "150", "--fov", "16"],
    "Spherical": ["--disk-model", "spherical", "--dist", "100", "--fov", "24"],
    "JetCones": ["--disk-model", "cones", "--dist", "100", "--fov", "24"],
    "Lamppost": ["--disk-model", "lamppost", "--lamppost-samples", "300"],
}
SUPPORTED = {
    "Kerr": {"None", "NovikovThorne", "Newtonian", "Thick", "Torus", "Riaf", "Spherical", "JetCones", "Lamppost"}, "KerrNewman": {"None", "NovikovThorne", "Newtonian"},
    "NakedKN": {"None", "Newtonian"}, "SdS": {"None", "Newtonian"}, "Wormhole": {"None"},
}
bad = 0
for m, mf in METRICS.items():
    for d, df in DISKS.items():
        flags = mf + [x for x in df]
        if "--dist" in df and "--dist" in mf:
            flags = [x for x in flags]  # the disk's --dist wins only for Kerr/torus; for SdS keep a valid camera
        if m == "SdS" and "--dist" in df:
            flags = mf + [x for x in df if x not in ("--dist", "150")]
        if m == "Wormhole" and d != "None":
            expect_ok = False  # explicit disk flags with a wormhole: refused (the wormhole is sky only)
        else:
            expect_ok = d in SUPPORTED[m]
        r = subprocess.run([B, "--nohud", "--nosky", "--w", "32", "--h", "18", "--samples", "1", "--prec", "float", "--tol", "2e-5"] + flags + ["--shot", "/tmp/combos.png"], capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0
        flag = "" if ok == expect_ok else "   <-- MISMATCH"
        bad += ok != expect_ok
        print(f"{m:11s} {d:14s} {'renders' if ok else 'refused':8s} expected {'ok' if expect_ok else 'refusal'}{flag}")
sys.exit(1 if bad else 0)
