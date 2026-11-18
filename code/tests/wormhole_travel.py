#!/usr/bin/env python3
"""v4.0 wormhole travel checks (camera position l can be negative: the camera went through the throat).

  wormhole_travel.py BINARY SKY SKY2
1. mirror: with the same sky on both sides the image at l = +4 equals the image at l = -4 bit for bit (the metric is symmetric under l -> -l and the tracer is
   odd/even exactly), which tests the sign handling of the viewing direction;
2. far side: with two different skies, the centre of the view at l = +0.3 (looking at the throat from the +l side) shows the 2nd sky and at l = -0.3 the 1st one
   (what lies on the other side of the throat), so the two centre colours must differ clearly.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

binary, sky, sky2 = sys.argv[1:4]
base = [binary, "--nohud", "--w", "160", "--h", "90", "--samples", "1", "--prec", "float", "--tol", "2e-5", "--wormhole", "ellis,1", "--fov", "70", "--theta", "90"]
tmp = Path(tempfile.mkdtemp())


def shot(l, extra, name):
    out = tmp / name
    r = subprocess.run(base + ["--wh-l", str(l)] + extra + ["--shot", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.asarray(Image.open(out).convert("RGB"), float)


a, b = shot(4, ["--sky", sky], "p4.png"), shot(-4, ["--sky", sky], "m4.png")
assert (a == b).all(), "mirror: images at l = +4 and l = -4 differ"
print("mirror ok: l=+4 and l=-4 identical")
p, m = shot(0.3, ["--sky", sky, "--sky2", sky2], "p03.png"), shot(-0.3, ["--sky", sky, "--sky2", sky2], "m03.png")
cp, cm = p[35:55, 70:90].mean((0, 1)), m[35:55, 70:90].mean((0, 1))
d = np.abs(cp - cm).sum()
print(f"far side: centre colour l=+0.3 {cp.round(1)}  l=-0.3 {cm.round(1)}  |diff| {d:.1f}")
assert d > 10, "the two sides of the throat must show different skies"

# 3. aberration of the travelling observer: velocity along -e_l at l = +4 (towards the throat) is the mirror of velocity along +e_l at l = -4; and it changes the image
v1, v2 = shot(4, ["--sky", sky, "--beta", "-0.8,0,0"], "v1.png"), shot(-4, ["--sky", sky, "--beta", "0.8,0,0"], "v2.png")
assert (v1 == v2).all(), "mirror with velocity: images differ"
assert not (v1 == a).all(), "the velocity must change the image (aberration)"
print("aberration ok: mirror with velocity identical, differs from the static image")
