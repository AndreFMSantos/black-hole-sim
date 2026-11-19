#!/usr/bin/env python3
"""Generates a SYNTHETIC .bgm from the analytic Fishbone-Moncrief torus (any spin a: because the torus is axisymmetric
with u^r = b^r = 0, contravariant u^t, u^phi, b^t, b^phi coincide in Kerr-Schild and Boyer-Lindquist),
on the same FMKS grid as the Illinois dumps, to validate the GRMHD reader end to end against the analytic renderer (--disk-model torus).
Implementation INDEPENDENT of the C++ (numpy): l_K(r_max), |u_t|, enthalpy, 4-velocity, toroidal field of constant beta.

Usage: uv run --with numpy python -I tools/make_torus_dump.py out.bgm [rin=7 rmax=12 theta_e_max=20 beta=3 a=0.0]
Dump units: rho = rho~ (maximum 1); the render uses M_unit such that rho_unit = ne0 (m_p + m_e) (see notes/physics/22f-grmhd-snapshot.md).
|b|^2 = 2 rho (m_p thI + m_e thE) / (beta (m_p + m_e))  =>  B_G = |b| c sqrt(4 pi rho_unit) = sqrt(8 pi P / beta) with P = n_e c^2 (m_p thI + m_e thE).
"""
import struct, sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location('cg', __import__('os').path.join(__import__('os').path.dirname(__file__), 'convert_grmhd.py'))
cg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cg)

MP, ME = 1.67262192369e-24, 9.1093837015e-28
out = sys.argv[1]
rin = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
rmax = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
th_e_max = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
beta = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
gam = 4.0 / 3.0
a = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
p = dict(a=a, hslope=0.3, mks_smooth=0.5, poly_alpha=14.0, poly_xt=0.82, startx1=0.5721548105021682)
n1, n2, n3 = 288, 128, 128
dx1, dx2, dx3 = 0.02199861273777767, 1.0 / 128, 2 * np.pi / 128

def abs_ut(r, th, ell):
    s2 = np.sin(th) ** 2; Sig = r * r + a * a * np.cos(th) ** 2
    gtt = -(1 - 2 * r / Sig); gtp = -2 * a * r * s2 / Sig; gpp = (r * r + a * a + 2 * a * a * r * s2 / Sig) * s2
    det = gtt * gpp - gtp ** 2
    itt, itp, ipp = gpp / det, -gtp / det, gtt / det
    D = itt - 2 * ell * itp + ell * ell * ipp
    ok = D < 0
    return np.where(ok, 1.0 / np.sqrt(np.where(ok, -D, 1.0)), 0.0), (gtt, gtp, gpp, itt, itp, ipp)

ell = (rmax ** 2 - 2 * a * np.sqrt(rmax) + a * a) / (rmax ** 1.5 - 2 * np.sqrt(rmax) + a)
ut_in = float(abs_ut(np.float64(rin), np.float64(np.pi / 2), ell)[0]); ut_max = float(abs_ut(np.float64(rmax), np.float64(np.pi / 2), ell)[0])
hmax = ut_in / ut_max
assert ut_in < 1.0 and hmax > 1.0, "open or invalid torus"
thI_max = (gam - 1) / gam * (hmax - 1)
print(f"torus a = {a}: r_in {rin}, r_max {rmax}, ell = {ell:.6f}, |u_t|_in = {ut_in:.6f}, h_max = {hmax:.6f}, Theta_i,max = {thI_max:.5f}, Theta_e,max = {th_e_max}")

X1 = p['startx1'] + (np.arange(n1) + 0.5) * dx1
X2 = (np.arange(n2) + 0.5) * dx2
X1g, X2g = np.meshgrid(X1, X2, indexing='ij')
r = np.exp(X1g); th = cg.fmks_theta(X1g, X2g, p)
ut_abs, (gtt, gtp, gpp, itt, itp, ipp) = abs_ut(r, th, ell)
h = np.where(ut_abs > 0, ut_in / np.where(ut_abs > 0, ut_abs, 1.0), 0.0)
# only the bound component of the torus: the formula gives h > 1 also next to the black hole (|u_t| -> 0, h -> infinity), an UNBOUND region of the torus
# (measured: rho~ = 7e3 at r = 2.3); the analytic renderer excludes r < 0.7 r_in, the same is done here
inside = (h > 1.0) & (r > 0.7 * rin)
rho = np.where(inside, ((h - 1) / (hmax - 1)) ** (1.0 / (gam - 1)), 0.0)
thI = np.where(inside, (gam - 1) / gam * (h - 1), 0.0)
thE = th_e_max * np.where(inside, (h - 1) / (hmax - 1), 0.0)
absb2 = 2 * rho * (MP * thI + ME * thE) / (beta * (MP + ME))
ut_low = -ut_abs
# outside the torus: ZAMO four-velocity (valid down to the horizon even inside the ergosphere for a != 0)
Sig = r * r + a * a * np.cos(th) ** 2
Del = np.maximum(1e-6, r * r - 2 * r + a * a)
A_k = (r * r + a * a) ** 2 - a * a * Del * (np.sin(th) ** 2)
alpha_z = np.sqrt(np.maximum(1e-6, Sig * Del / A_k))
omega_z = 2 * a * r / A_k
ut_up = np.where(inside, ut_low * (itt - ell * itp), 1.0 / alpha_z)
uph_up = np.where(inside, ut_low * (itp - ell * ipp), omega_z / alpha_z)
bn = np.sqrt(np.maximum(gtt * ell * ell + 2 * gtp * ell + gpp, 1e-30))
floor = 1e-30
D = np.zeros((n1, n2, n3, 12), dtype=np.float32)
D[..., 0] = np.where(inside, np.maximum(rho, floor), floor)[:, :, None]
D[..., 1] = np.where(inside, thI, floor)[:, :, None]
D[..., 2] = np.sqrt(np.maximum(absb2, floor))[:, :, None]
D[..., 3] = beta
D[..., 4] = ut_up[:, :, None]; D[..., 5] = 0.0; D[..., 6] = 0.0; D[..., 7] = uph_up[:, :, None]
D[..., 8] = np.where(inside, ell / bn, 0.0)[:, :, None]; D[..., 9] = 0.0; D[..., 10] = 0.0; D[..., 11] = np.where(inside, 1.0 / bn, 1.0 / np.maximum(np.sqrt(gpp), 1e-30))[:, :, None]
hdr = bytearray(256); hdr[0:4] = b'BGM1'
struct.pack_into('<5I', hdr, 4, n1, n2, n3, 12, 0)
struct.pack_into('<16d', hdr, 24, p['startx1'], dx1, dx2, dx3, a, p['hslope'], p['mks_smooth'], p['poly_alpha'], p['poly_xt'], gam, 0.0, 0, 0, 0, 0, 0)
open(out, 'wb').write(bytes(hdr) + D.tobytes())
print(f"{out}: {D.nbytes/1e6:.1f} MB; cells inside the torus: {int(inside.sum()) * n3} of {n1*n2*n3}; ")
print(f"for the render: --munit M_unit with rho_unit = ne0 (m_p + m_e); --electrons prop --th-e-ratio {th_e_max/thI_max:.6f} (Theta_e = ce Theta_i, ce = Theta_e,max/Theta_i,max)")
