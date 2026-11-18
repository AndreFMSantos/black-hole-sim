#!/usr/bin/env python3
"""Order-of-magnitude estimate (v3.0, design): how much visible light reaches the eye from a thin accretion disk (Shakura-Sunyaev, face-on,
without relativistic effects, i.e. a reasonable UPPER BOUND for not-too-bright disks) in several scenarios. It is NOT GR photometry: it serves to say
which regime we are in (dazzling / comfortable / invisible) before implementing --view eye. CGS constants. Usage: python3 tools/exp/eye_scenarios.py"""
import numpy as np
G = 6.674e-8; c = 2.998e10; sig = 5.6704e-5; h = 6.626e-27; k = 1.381e-16; Msun = 1.989e33; AU = 1.496e13
Ledd1 = 1.26e38  # erg/s per solar mass

def ybar(lam_nm):  # CIE 1931 y (Wyman, Sloan & Shirley 2013, multi-lobe fit) [memory]
    g = lambda l, mu, s1, s2: np.exp(-0.5 * ((l - mu) / np.where(l < mu, s1, s2)) ** 2)
    return 0.821 * g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * g(lam_nm, 530.9, 16.3, 31.1)

def scenario(name, M_sun, mdot_edd, d_cm, eta=0.057):
    M = M_sun * Msun; rg = G * M / c**2
    Mdot = mdot_edd * Ledd1 * M_sun / (eta * c**2)
    rin = 6 * rg
    r = np.geomspace(rin, 5e3 * rg, 4000)
    T = (3 * G * M * Mdot / (8 * np.pi * sig * r**3) * (1 - np.sqrt(rin / r))) ** 0.25
    lam = np.linspace(380, 780, 401) * 1e-7   # cm
    Bl = 2 * h * c**2 / lam[None, :]**5 / np.expm1(h * c / (lam[None, :] * k * T[:, None] + 1e-300))   # erg/s/cm2/sr/cm
    # flux at d, face-on: F_lam = (1/d^2) * integral B_lam(T(r)) 2 pi r dr  (cos i = 1)
    Fl = np.trapezoid(Bl * (2 * np.pi * r)[:, None], r, axis=0) / d_cm**2           # erg/s/cm2/cm
    # illuminance: 683 lm/W * integral ybar F_lam dlam ; erg/s/cm2 -> W/m2 = 1e-3
    E_lux = 683.0 * np.trapezoid(ybar(lam * 1e7) * Fl, lam) * 1e-3
    # luminance of the brightest zone (cd/m2): L = 683 * int ybar B_lam dlam [W/m2/sr] -> 1e-3
    Lum = 683.0 * np.trapezoid(ybar(lam * 1e7) * Bl.max(axis=0) * 0 + ybar(lam * 1e7) * Bl[np.argmax(T)], lam) * 1e-3
    ang = 2 * 20 * rg / d_cm   # angular diameter of ~ 20 r_g (inner disk), rad
    print(f"{name:52s} d = {d_cm/rg:9.3g} rg  T_max = {T.max():9.3g} K  E_visible = {E_lux:9.3g} lux  L_disk = {Lum:9.3g} cd/m2  ~{np.degrees(ang):.3g} deg")
    return E_lux

if __name__ == '__main__':
    print("references: noon Sun 1.2e5 lux; full Moon 0.25 lux; starlight 2e-3 lux; scotopic threshold ~1e-6 cd/m2; HDR screen 1e3 cd/m2; Sun 1.6e9 cd/m2")
    scenario("10 Msun, mdot = 0.1 Edd, a 1000 km", 10, 0.1, 1e8)
    scenario("10 Msun, mdot = 0.1 Edd, a 1e5 km", 10, 0.1, 1e10)
    scenario("10 Msun, mdot = 0.1 Edd, a 1 AU", 10, 0.1, AU)
    scenario("10 Msun, mdot = 1e-4 Edd, at 1e5 km (cold disk)", 10, 1e-4, 1e10)
    scenario("4.14e6 Msun (Sgr A*), mdot = 1e-3 Edd (hipotetico), a 100 AU", 4.14e6, 1e-3, 100 * AU)
    scenario("4.14e6 Msun, mdot = 1e-6 Edd (hipotetico), a 100 AU", 4.14e6, 1e-6, 100 * AU)
