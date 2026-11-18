#!/usr/bin/env python3
"""Converts an Illinois GRMHD dump (iharm3d/KHARMA, HDF5, FMKS coordinates) to the binary .bgm format of black-hole-sim.

Usage: uv run --with h5py --with numpy python -I tools/convert_grmhd.py in.h5 out.bgm

Conventions (pyharm, commit fda4277, see notes/references.md): r = exp(X1), FMKS theta, phi = phi_KS; cell centre X = startx + (i + 0.5) dx;
gamma = sqrt(1 + g_ij u~^i u~^j), u^0 = gamma/alpha, u^i = u~^i - gamma alpha g^{0i} (native FMKS basis); b^0 = B^i u_i, b^i = (B^i + b^0 u^i)/u^0,
Pb = b^2/2 (Heaviside-Lorentz), Pg = (gam - 1) UU, Theta_i = Pg/rho, beta = Pg/Pb. All in float64; output in float32.

.bgm format: 256-byte header (magic 'BGM1', uint32 n1 n2 n3 nfields, float64[16]: startx1 dx1 dx2 dx3 a hslope mks_smooth poly_alpha poly_xt gam t ...)
and a float32 payload [n1][n2][n3][12] with the per-cell fields:
  0 rho, 1 theta_i, 2 |b|, 3 beta, 4..7 u^mu (t, r, theta, phi) in spherical KERR-SCHILD, 8..11 bhat^mu = b^mu/|b| (t, r, theta, phi) in KS.
"""
import struct
import sys

import numpy as np

NF = 12


def fmks_theta(X1, X2, p):
    pn = 0.5 * np.pi / (1.0 + 1.0 / (p['poly_alpha'] + 1.0) / p['poly_xt'] ** p['poly_alpha'])
    th_g = np.pi * X2 + 0.5 * (1.0 - p['hslope']) * np.sin(2 * np.pi * X2)
    y = 2 * X2 - 1
    th_j = pn * y * (1.0 + (y / p['poly_xt']) ** p['poly_alpha'] / (p['poly_alpha'] + 1.0)) + 0.5 * np.pi
    w = np.exp(p['mks_smooth'] * (p['startx1'] - X1))
    return th_g + w * (th_j - th_g)


def fmks_jac(X1, X2, p):
    """dtheta/dX1, dtheta/dX2 analytic (the rest of the Jacobian is trivial: dr/dX1 = r)."""
    pn = 0.5 * np.pi / (1.0 + 1.0 / (p['poly_alpha'] + 1.0) / p['poly_xt'] ** p['poly_alpha'])
    y = 2 * X2 - 1
    th_g = np.pi * X2 + 0.5 * (1.0 - p['hslope']) * np.sin(2 * np.pi * X2)
    th_j = pn * y * (1.0 + (y / p['poly_xt']) ** p['poly_alpha'] / (p['poly_alpha'] + 1.0)) + 0.5 * np.pi
    dg = np.pi + (1.0 - p['hslope']) * np.pi * np.cos(2 * np.pi * X2)
    dj = 2 * pn * (1.0 + (y / p['poly_xt']) ** p['poly_alpha'])
    w = np.exp(p['mks_smooth'] * (p['startx1'] - X1))
    return -p['mks_smooth'] * w * (th_j - th_g), dg + w * (dj - dg)


def gcov_ks(a, r, th):
    s2 = np.sin(th) ** 2; rho2 = r * r + a * a * np.cos(th) ** 2
    g = np.zeros(np.broadcast(r, th).shape + (4, 4))
    g[..., 0, 0] = -1 + 2 * r / rho2
    g[..., 0, 1] = g[..., 1, 0] = 2 * r / rho2
    g[..., 0, 3] = g[..., 3, 0] = -2 * a * r * s2 / rho2
    g[..., 1, 1] = 1 + 2 * r / rho2
    g[..., 1, 3] = g[..., 3, 1] = -a * s2 * (1 + 2 * r / rho2)
    g[..., 2, 2] = rho2
    g[..., 3, 3] = s2 * (rho2 + a * a * s2 * (1 + 2 * r / rho2))
    return g


def convert(path_in, path_out, chunk=8):
    import h5py
    f = h5py.File(path_in, 'r')
    H = f['header']
    p = dict(a=float(H['geom']['fmks']['a'][()]), hslope=float(H['geom']['fmks']['hslope'][()]), mks_smooth=float(H['geom']['fmks']['mks_smooth'][()]),
             poly_alpha=float(H['geom']['fmks']['poly_alpha'][()]), poly_xt=float(H['geom']['fmks']['poly_xt'][()]), startx1=float(H['geom']['startx1'][()]))
    gam = float(H['gam'][()]); t = float(H['t'][()])
    n1, n2, n3 = int(H['n1'][()]), int(H['n2'][()]), int(H['n3'][()])
    dx1, dx2, dx3 = float(H['dx1'][()]), float(H['dx2'][()]), float(H['dx3'][()])
    assert abs(p['startx1'] - float(H['geom']['startx1'][()])) < 1e-15
    prims = f['prims']
    X2 = (np.arange(n2) + 0.5) * dx2
    out = np.empty((n1, n2, n3, NF), dtype=np.float32)
    worst_uu = worst_bu = 0.0
    for i0 in range(0, n1, chunk):
        i1 = min(i0 + chunk, n1); c = i1 - i0
        X1 = p['startx1'] + (np.arange(i0, i1) + 0.5) * dx1
        r = np.exp(X1)[:, None]                                   # (c,1)
        X1g, X2g = np.meshgrid(X1, X2, indexing='ij')             # (c,n2)
        th = fmks_theta(X1g, X2g, p)
        dth1, dth2 = fmks_jac(X1g, X2g, p)
        rr = np.broadcast_to(r, th.shape)
        gks = gcov_ks(p['a'], rr, th)                              # (c,n2,4,4)
        J = np.zeros(th.shape + (4, 4))                            # dx^mu_KS / dX^nu_native
        J[..., 0, 0] = 1; J[..., 1, 1] = rr; J[..., 2, 1] = dth1; J[..., 2, 2] = dth2; J[..., 3, 3] = 1
        gn = np.einsum('...ai,...ab,...bj->...ij', J, gks, J)      # native metric
        gn = np.broadcast_to(gn[:, :, None], (c, n2, n3, 4, 4))
        gc = np.linalg.inv(gn)
        P = prims[i0:i1].astype(np.float64)                        # (c,n2,n3,8)
        rho, uu = P[..., 0], P[..., 1]; ut_ = P[..., 2:5]; B = P[..., 5:8]
        alpha = 1.0 / np.sqrt(-gc[..., 0, 0])
        q = np.einsum('...i,...ij,...j->...', ut_, gn[..., 1:, 1:], ut_)
        gamma_L = np.sqrt(1.0 + q)
        ucon = np.empty(rho.shape + (4,))
        ucon[..., 0] = gamma_L / alpha
        ucon[..., 1:] = ut_ - (gamma_L * alpha)[..., None] * gc[..., 0, 1:]
        ucov = np.einsum('...ij,...j->...i', gn, ucon)
        b0 = np.einsum('...i,...i->...', B, ucov[..., 1:])
        bcon = np.empty_like(ucon)
        bcon[..., 0] = b0
        bcon[..., 1:] = (B + b0[..., None] * ucon[..., 1:]) / ucon[..., 0:1]
        bsq = np.einsum('...i,...ij,...j->...', bcon, gn, bcon)
        Pg = (gam - 1.0) * uu
        absb = np.sqrt(np.maximum(bsq, 1e-300))
        JJ = np.broadcast_to(J[:, :, None], (c, n2, n3, 4, 4))
        uks = np.einsum('...ij,...j->...i', JJ, ucon)
        bks = np.einsum('...ij,...j->...i', JJ, bcon) / absb[..., None]
        gks_b = np.broadcast_to(gks[:, :, None], (c, n2, n3, 4, 4))
        worst_uu = max(worst_uu, float(np.abs(np.einsum('...i,...ij,...j->...', uks, gks_b, uks) + 1.0).max()))
        worst_bu = max(worst_bu, float((np.abs(np.einsum('...i,...ij,...j->...', bks, gks_b, uks))).max()))
        out[i0:i1, ..., 0] = rho; out[i0:i1, ..., 1] = Pg / rho; out[i0:i1, ..., 2] = absb; out[i0:i1, ..., 3] = Pg / (0.5 * bsq)
        out[i0:i1, ..., 4:8] = uks; out[i0:i1, ..., 8:12] = bks
    hdr = bytearray(256)
    hdr[0:4] = b'BGM1'
    struct.pack_into('<5I', hdr, 4, n1, n2, n3, NF, 0)
    struct.pack_into('<16d', hdr, 24, p['startx1'], dx1, dx2, dx3, p['a'], p['hslope'], p['mks_smooth'], p['poly_alpha'], p['poly_xt'], gam, t, 0, 0, 0, 0, 0)
    with open(path_out, 'wb') as fo:
        fo.write(hdr); fo.write(np.ascontiguousarray(out).tobytes())
    print(f"{path_in} -> {path_out}: {n1}x{n2}x{n3}x{NF} float32 = {out.nbytes/1e6:.1f} MB; a = {p['a']}, t = {t}")
    print(f"invariants (float64, BEFORE quantising to float32): max |u.u + 1| = {worst_uu:.2e}, max |bhat.u| = {worst_bu:.2e}")
    return out


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
