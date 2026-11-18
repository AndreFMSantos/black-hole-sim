import os as _os_bhs, pathlib as _pl_bhs
import os
_BHS_ROOT = str(_pl_bhs.Path(__file__).resolve().parents[3])  # the folder holding code/, data/ and assets/
_BHS_DATA = _os_bhs.environ.get('BHS_DATA', _os_bhs.path.join(_BHS_ROOT, 'data'))
# Experiment (v2.2f-a2): reconstruct frame 5500 from 5499 and 5501 (linear vs following the rotation). Needs the 3 files from tools/fetch_grmhd.sh MAD 0 <frame>
import h5py, numpy as np
D = os.path.join(_BHS_DATA, "illinois-grmhd-v3/MAD_a0/torus.out0.%05d.h5")
def L(n):
    f = h5py.File(D % n, 'r'); H = f['header']
    return np.log10(f['prims'][..., 0].astype(np.float64)), float(H['t'][()]), H
a, ta, H = L(5499); b, tb, _ = L(5500); c, tc, _ = L(5501)
print("times t_g:", ta, tb, tc, " (interval 5499->5501 =", round(tc - ta, 3), ")")
x1 = H['geom']['startx1'][()] + (np.arange(288) + 0.5) * H['dx1'][()]; r = np.exp(x1)
n3 = 128
def shift_phi(v, s):          # shifts v (..., n3) by s cells in phi (periodic, Fourier, fractional)
    k = np.fft.fftfreq(n3) * n3
    return np.real(np.fft.ifft(np.fft.fft(v, axis=-1) * np.exp(-2j * np.pi * k * s / n3), axis=-1))
# empirical rotation per shell: shift (in phi cells) that aligns a(5499) with c(5501), maximising the correlation
s_est = np.zeros(288)
for i in range(288):
    A = a[i] - a[i].mean(); C = c[i] - c[i].mean()            # (128 theta, 128 phi)
    cc = np.real(np.fft.ifft(np.fft.fft(C, axis=-1) * np.conj(np.fft.fft(A, axis=-1)), axis=-1)).sum(axis=0)
    k = int(np.argmax(cc)); y0, y1, y2 = cc[(k - 1) % n3], cc[k], cc[(k + 1) % n3]
    den = y0 - 2 * y1 + y2; frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    s = k + frac; s_est[i] = s - n3 if s > n3 / 2 else s
dphi = 2 * np.pi / n3
for r0 in (3, 6, 10, 20, 50):
    i = int(np.searchsorted(r, r0)); w = s_est[i] * dphi / (tc - ta)
    print(f"  r = {r[i]:6.2f}: shift {s_est[i]:+6.2f} cells in {tc-ta:.0f} t_g  -> Omega ~ {w:.4f}  (Keplerian 1/r^1.5 = {r[i]**-1.5:.4f})")
# reconstruction of the middle frame
lin = 0.5 * (a + c)
adv = np.empty_like(lin)
for i in range(288):
    adv[i] = 0.5 * (shift_phi(a[i], 0.5 * s_est[i]) + shift_phi(c[i], -0.5 * s_est[i]))
def rms(x, sel=slice(None)): e = (x - b)[sel]; return np.sqrt((e**2).mean()), np.percentile(np.abs(e), 99.9)
print("\nreconstruction of log10 rho(5500) from 5499 and 5501:")
for name, sel in (("whole domain", slice(None)), ("r < 30", slice(0, int(np.searchsorted(r, 30)))), ("r < 10", slice(0, int(np.searchsorted(r, 10))))):
    l, a1 = rms(lin, sel), rms(adv, sel); base = rms(a, sel)
    print(f"  {name:15s} copy of the neighbour: rms {base[0]:.3f} dex | linear: rms {l[0]:.3f} (p99.9 {l[1]:.2f}) | advected: rms {a1[0]:.3f} (p99.9 {a1[1]:.2f})  dex")
