#!/usr/bin/env python3
"""Can a human see a black hole with the naked eye? Angular diameter of the shadow (Schwarzschild, 2 sqrt(27) GM/c^2 = 10.39 rg; Kerr differs by a few %)
against the eye's resolution (~1 arcmin, foveal acuity 20/20), and the distance at which the shadow reaches 1 arcmin. Masses/distances of the systems: [memory]
(Gaia BH1 9.62 Msun at 480 pc; Gaia BH3 33 Msun at 590 pc; Sgr A* 4.14e6 Msun at 8127 pc; M87* 6.5e9 Msun at 16.8 Mpc). Usage: python3 tools/exp/eye_feasibility.py"""
import math
G = 6.674e-11; c = 2.998e8; Msun = 1.989e30; pc = 3.0857e16; AU = 1.496e11; ly = 9.461e15; Rmoon = 3.844e8
arcmin = math.radians(1 / 60); arcsec = math.radians(1 / 3600); uas = arcsec * 1e-6
print(f"{'system':14s} {'rg (km)':>10s} {'shadow (km)':>12s} {'distance':>12s} {'ang. diameter':>16s} {'vs eye (1 arcmin)':>20s} {'d for 1 arcmin':>22s}")
for nm, M, d in (("Gaia BH1", 9.62, 480 * pc), ("Gaia BH3", 33, 590 * pc), ("Sgr A*", 4.14e6, 8127 * pc), ("M87*", 6.5e9, 16.8e6 * pc)):
    rg = G * M * Msun / c**2; D = 2 * math.sqrt(27) * rg
    th = D / d
    dn = D / arcmin
    unit = f"{dn/Rmoon:.3g} Lua" if dn < 1e3 * Rmoon else f"{dn/AU:.4g} UA"
    print(f"{nm:14s} {rg/1e3:10.4g} {D/1e3:12.4g} {d/ly:9.4g} ly {th/uas:12.4g} uas {th/arcmin:18.3g}x {unit:>22s}")
print("\nreference: the EHT resolved M87* (42 uas) with an Earth-sized interferometer at 230 GHz; the eye at 550 nm has a diffraction limit ~ 1.22 lambda/D with D=5 mm: %.0f arcsec" % (1.22 * 550e-9 / 5e-3 / arcsec))
print("tidal (10 Msun, at 1 Moon distance ~ 3.8e5 km): gradient ~ 2GM L / d^3 for L = 2 m: %.3g m/s^2" % (2 * G * 9.62 * Msun * 2 / (Rmoon**3)))
