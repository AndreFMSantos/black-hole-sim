#!/usr/bin/env python3
"""Converts an 8-bit equirectangular image (sRGB, e.g. WISE PIA15482) into a linear half EXR for --sky2.
Usage: uv run --with pillow --with numpy --with openexr python -I tools/img_to_exr.py in.tif out.exr [width=8192] [target_mean=0.015]
The linear value = sRGB^-1(x) * gain, with the gain chosen so that the mean of the three channels is target_mean (NASA's star map has a mean ~0.004-0.005);
it is an aesthetic choice (the original image is a false-colour display image, not radiometry)."""
import sys
import numpy as np
from PIL import Image
import OpenEXR, Imath
Image.MAX_IMAGE_PIXELS = None
src, dst = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
target = float(sys.argv[4]) if len(sys.argv) > 4 else 0.015
im = Image.open(src).convert('RGB').resize((W, W // 2), Image.LANCZOS)
x = np.asarray(im).astype(np.float64) / 255.0
lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
gain = target / lin.mean()
lin = np.clip(lin * gain, 0, 65000)
print(f"{src} -> {dst}: {W}x{W//2}, linear mean before {lin.mean()/gain:.4f}, gain {gain:.3f}, final mean {lin.mean():.4f}, max {lin.max():.3f}")
h = OpenEXR.Header(W, W // 2)
h['channels'] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)) for c in 'RGB'}
f = OpenEXR.OutputFile(dst, h)
f.writePixels({c: lin[..., i].astype(np.float16).tobytes() for i, c in enumerate('RGB')})
f.close()
