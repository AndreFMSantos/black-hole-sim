#!/usr/bin/env python3
"""Ring diameter of a radio image (PFM in Jy/uas^2 from --radio-out), in the spirit of the EHT's REx (Paper IV): for each candidate centre
on a grid, azimuth-averaged radial profile; the ring is the peak of that profile; the chosen centre maximises the peak (sharpest ring).
Also gives the full width at half maximum and the contrast. Usage: python tools/ring_diameter.py img.pfm fov_deg dist_M [uas_per_M=3.819]"""
import sys
import numpy as np


def read_pfm(p):
    with open(p, 'rb') as f:
        assert f.readline().strip() == b'Pf'
        w, h = map(int, f.readline().split()); sc = float(f.readline())
        d = np.frombuffer(f.read(), dtype='<f4' if sc < 0 else '>f4').reshape(h, w)
    return d[::-1].astype(np.float64)


def rex(img, M_per_px, search=3.0, step=0.15, rmax=9.0, nb=90):
    h, w = img.shape
    y, x = (np.mgrid[0:h, 0:w] - np.array([[[h / 2 - 0.5]], [[w / 2 - 0.5]]])) * M_per_px
    best = None
    for cy in np.arange(-search, search + 1e-9, step):
        for cx in np.arange(-search, search + 1e-9, step):
            r = np.hypot(x - cx, y - cy)
            idx = np.minimum((r / rmax * nb).astype(int), nb)
            s = np.bincount(idx.ravel(), img.ravel(), nb + 1)[:nb]; n = np.bincount(idx.ravel(), minlength=nb + 1)[:nb]
            prof = s / np.maximum(n, 1)
            k = np.argmax(prof[8:]) + 8
            if best is None or prof[k] > best[0]:
                best = (prof[k], (k + 0.5) * rmax / nb, cx, cy, prof)
    pk, rr, cx, cy, prof = best
    half = pk / 2; k = int(rr / rmax * nb)
    lo = k
    while lo > 0 and prof[lo] > half: lo -= 1
    hi = k
    while hi < nb - 1 and prof[hi] > half: hi += 1
    return 2 * rr, cx, cy, (hi - lo) * rmax / nb, pk / max(prof[:k].min(), 1e-30)


if __name__ == '__main__':
    img = read_pfm(sys.argv[1]); fov = float(sys.argv[2]); dist = float(sys.argv[3]); upm = float(sys.argv[4]) if len(sys.argv) > 4 else 3.819
    M_per_px = 2 * dist * np.tan(np.radians(fov / 2)) / img.shape[1]
    d, cx, cy, fwhm, contrast = rex(img, M_per_px)
    print(f"diameter = {d:.2f} M = {d * upm:.1f} uas | centre ({cx:.2f}, {cy:.2f}) M | width {fwhm:.2f} M | peak/inner minimum {contrast:.1f}")
