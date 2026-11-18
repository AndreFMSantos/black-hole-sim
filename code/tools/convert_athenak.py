#!/usr/bin/env python3
"""Converts an AthenaK `mhd_w_bcc` dump (.bin, CARTESIAN Kerr-Schild, mesh with static refinement) into black-hole-sim's .bgm format.

Usage: uv run --with h5py --with numpy python -I tools/convert_athenak.py in.bin out.bgm [--a 0.9375] [--n 192,128,128] [--rin 1.4] [--rout 100]
(h5py is only needed because AthenaK's vis/python/bin_convert.py imports it; AthenaK at $BHS_DATA/athenak, or $ATHENAK.)

Resamples onto a uniform spherical grid in (X1 = ln r, X2 = theta/pi, X3 = phi), i.e. MKS with hslope = 1 (theta = pi X2 exactly; mks_smooth = 1e4 cancels the FMKS polynomial
term in the reader), cell centres X = startx + (i + 0.5) dx, like the Illinois .bgm. Trilinear interpolation at the cell centres, in the finest block
that covers the point (the primitives are interpolated; the derived quantities are computed afterwards, at the point).

AthenaK conventions (src/eos/ideal_c2p_mhd.hpp, SingleP2C_IdealGRMHD): w.v = u~^i = u^i + alpha gamma g^{0i} (gamma = sqrt(1 + g_ij u~^i u~^j), u^0 = gamma/alpha,
alpha = 1/sqrt(-g^{00})), w.e = internal energy per volume (Pg = (Gamma - 1) e), bcc = B^i (the CT field, the same as HARM), b^0 = B^i u_i, b^i = (B^i + b^0 u^i)/u^0,
Pb = b^2/2 (Heaviside-Lorentz). Cartesian Kerr-Schild metric: g_mn = eta + 2H l_m l_n, g^{mn} = eta - 2H l^m l^n, H = r^3/(r^4 + a^2 z^2), l_m = (1, (r x + a y)/(r^2 + a^2),
(r y - a x)/(r^2 + a^2), z/r); x + i y = (r + i a) sin(theta) e^{i phi_KS}.
Output: the 12 per-cell fields of the .bgm (see convert_grmhd.py): rho, theta_i, |b|, beta, u^mu and bhat^mu in SPHERICAL KS (t, r, theta, phi).
"""
import os as _os_bhs, pathlib as _pl_bhs
_BHS_ROOT = str(_pl_bhs.Path(__file__).resolve().parents[2])  # the folder holding code/, data/ and assets/
_BHS_DATA = _os_bhs.environ.get('BHS_DATA', _os_bhs.path.join(_BHS_ROOT, 'data'))
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get('ATHENAK', os.path.join(_BHS_DATA, 'athenak')), 'vis', 'python'))
from convert_grmhd import NF, gcov_ks  # noqa: E402

GAM = 1.4444444444444444  # <mhd> gamma of the input (checked in the .bin header)


def build_lookup(d):
    """Per level: sorted keys (ix,iy,iz) -> block index; block size and nx per block."""
    geo = np.asarray(d['mb_geometry']); lg = np.asarray(d['mb_logical'])  # lg = [lx1, lx2, lx3, level]
    levels = {}
    for m in range(len(geo)):
        L = int(lg[m, 3])
        levels.setdefault(L, []).append(m)
    out = {}
    for L, ms in levels.items():
        ms = np.array(ms)
        size = geo[ms[0], 1] - geo[ms[0], 0]
        keys = (lg[ms, 0].astype(np.int64) << 40) | (lg[ms, 1].astype(np.int64) << 20) | lg[ms, 2].astype(np.int64)
        o = np.argsort(keys)
        out[L] = (keys[o], ms[o], size)
    return out


def sample(d, lk, pts, fields):
    """Trilinear at pts (N,3) (x,y,z) in the finest block covering each point. Returns (N, len(fields)) and a coverage mask."""
    geo = np.asarray(d['mb_geometry']); lg = np.asarray(d['mb_logical'])
    nx = d['nx1_mb']; assert nx == d['nx2_mb'] == d['nx3_mb']
    N = len(pts)
    res = np.zeros((N, len(fields))); done = np.zeros(N, bool)
    x0 = np.array([d['x1min'], d['x2min'], d['x3min']])
    for L in sorted(lk, reverse=True):
        keys, ms, size = lk[L]
        todo = np.where(~done)[0]
        if not len(todo):
            break
        ij = np.floor((pts[todo] - x0) / size).astype(np.int64)
        ok = (ij >= 0).all(1)
        key = (ij[:, 0] << 40) | (ij[:, 1] << 20) | ij[:, 2]
        pos = np.searchsorted(keys, key)
        pos = np.minimum(pos, len(keys) - 1)
        hit = ok & (keys[pos] == key)
        idx = todo[hit]
        if not len(idx):
            continue
        mb = ms[pos[hit]]
        dx = size / nx
        s = (pts[idx] - geo[mb][:, [0, 2, 4]]) / dx - 0.5      # coordinate in cell-centre indices, (n,3) = (x1,x2,x3)
        i0 = np.clip(np.floor(s).astype(np.int64), 0, nx - 2)
        f = np.clip(s - i0, 0.0, 1.0)
        for q, name in enumerate(fields):
            arr = d['mb_data'][name]                          # [n_mbs, nz, ny, nx]
            v = 0.0
            for dz in (0, 1):
                for dy in (0, 1):
                    for dxx in (0, 1):
                        w = (f[:, 2] if dz else 1 - f[:, 2]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 0] if dxx else 1 - f[:, 0])
                        v = v + w * arr[mb, i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dxx]
            res[idx, q] = v
        done[idx] = True
    return res, done


def convert(path_in, path_out, a=0.9375, n=(192, 128, 128), rin=1.4, rout=100.0):
    import bin_convert
    d = bin_convert.read_binary(path_in)
    d['mb_data'] = {k: np.asarray(v) for k, v in d['mb_data'].items()}
    lk = build_lookup(d)
    n1, n2, n3 = n
    startx1 = np.log(rin); dx1 = (np.log(rout) - startx1) / n1; dx2 = 1.0 / n2; dx3 = 2 * np.pi / n3
    th = (np.arange(n2) + 0.5) * dx2 * np.pi
    ph = (np.arange(n3) + 0.5) * dx3
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    out = np.empty((n1, n2, n3, NF), dtype=np.float32)
    fields = ['dens', 'eint', 'velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3']
    worst_uu = worst_bu = 0.0; uncovered = 0
    for i in range(n1):
        r = np.exp(startx1 + (i + 0.5) * dx1)
        x = np.sin(TH) * (r * np.cos(PH) - a * np.sin(PH)); y = np.sin(TH) * (r * np.sin(PH) + a * np.cos(PH)); z = r * np.cos(TH) + 0 * PH
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
        P, done = sample(d, lk, pts, fields)
        uncovered += int((~done).sum())
        rho, e = P[:, 0], P[:, 1]; ut = P[:, 2:5]; B = P[:, 5:8]
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        H = r ** 3 / (r ** 4 + a * a * Z * Z)
        lc = np.stack([(r * X + a * Y) / (r * r + a * a), (r * Y - a * X) / (r * r + a * a), Z / r], 1)     # l_i (covariant, spatial)
        # Cartesian KS metric: g_ij = delta_ij + 2H l_i l_j ; g^{0i} = 2H l_i ; g^{00} = -(1 + 2H)
        g3 = np.eye(3)[None] + 2 * H[:, None, None] * lc[:, :, None] * lc[:, None, :]
        alpha = 1.0 / np.sqrt(1.0 + 2 * H)
        q = np.einsum('ni,nij,nj->n', ut, g3, ut)
        gamma = np.sqrt(1.0 + q)
        u0 = gamma / alpha
        u3 = ut - (gamma * alpha * 2 * H)[:, None] * lc                          # u^i = u~^i - alpha gamma g^{0i}
        lu = u0 + np.einsum('ni,ni->n', lc, u3)                                    # l_mu u^mu
        ucov3 = u3 + (2 * H * lu)[:, None] * lc                                    # u_i = u^i + 2H l_i (l.u)
        b0 = np.einsum('ni,ni->n', B, ucov3)
        b3 = (B + b0[:, None] * u3) / u0[:, None]
        # Cartesian -> spherical KS Jacobian: d(r,th,ph)/d(x,y,z) = inverse of d(x,y,z)/d(r,th,ph) (analytic)
        thv = np.repeat(th[:, None], n3, 1).ravel(); phv = PH.ravel()
        st, ct, sp, cp = np.sin(thv), np.cos(thv), np.sin(phv), np.cos(phv)
        Jf = np.zeros((len(thv), 3, 3))                                            # rows x,y,z; columns r,th,ph
        Jf[:, 0, 0] = st * cp; Jf[:, 0, 1] = ct * (r * cp - a * sp); Jf[:, 0, 2] = st * (-r * sp - a * cp)
        Jf[:, 1, 0] = st * sp; Jf[:, 1, 1] = ct * (r * sp + a * cp); Jf[:, 1, 2] = st * (r * cp - a * sp)
        Jf[:, 2, 0] = ct; Jf[:, 2, 1] = -r * st; Jf[:, 2, 2] = 0.0
        Ji = np.linalg.inv(Jf)
        us = np.einsum('nij,nj->ni', Ji, u3); bs = np.einsum('nij,nj->ni', Ji, b3)
        ucon = np.concatenate([u0[:, None], us], 1); bcon = np.concatenate([b0[:, None], bs], 1)
        gks = gcov_ks(a, np.full(len(thv), r), thv)
        bsq = np.einsum('ni,nij,nj->n', bcon, gks, bcon)
        absb = np.sqrt(np.maximum(bsq, 1e-300))
        Pg = (GAM - 1.0) * e
        bh = bcon / absb[:, None]
        worst_uu = max(worst_uu, float(np.abs(np.einsum('ni,nij,nj->n', ucon, gks, ucon) + 1.0).max()))
        worst_bu = max(worst_bu, float(np.abs(np.einsum('ni,nij,nj->n', bh, gks, ucon)).max()))
        o = np.empty((len(thv), NF), dtype=np.float32)
        o[:, 0] = rho; o[:, 1] = Pg / np.maximum(rho, 1e-300); o[:, 2] = absb; o[:, 3] = np.minimum(Pg / (0.5 * np.maximum(bsq, 1e-300)), 1e30); o[:, 4:8] = ucon; o[:, 8:12] = bh
        out[i] = o.reshape(n2, n3, NF)
    t = float(d['time'])
    hdr = bytearray(256)
    hdr[0:4] = b'BGM1'
    struct.pack_into('<5I', hdr, 4, n1, n2, n3, NF, 0)
    struct.pack_into('<16d', hdr, 24, startx1, dx1, dx2, dx3, a, 1.0, 1e4, 14.0, 0.82, GAM, t, 0, 0, 0, 0, 0)
    with open(path_out, 'wb') as fo:
        fo.write(hdr); fo.write(np.ascontiguousarray(out).tobytes())
    print(f"{path_in} -> {path_out}: {n1}x{n2}x{n3}x{NF} float32 = {out.nbytes/1e6:.1f} MB; a = {a}, t = {t}, r in [{rin}, {rout}]")
    print(f"points outside the mesh: {uncovered}; invariants (float64): max |u.u + 1| = {worst_uu:.2e}, max |bhat.u| = {worst_bu:.2e}")
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    opts = dict(zip(args[2::2], args[3::2]))
    convert(args[0], args[1], a=float(opts.get('--a', 0.9375)), n=tuple(int(v) for v in opts.get('--n', '192,128,128').split(',')),
            rin=float(opts.get('--rin', 1.4)), rout=float(opts.get('--rout', 100.0)))
