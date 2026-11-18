---
tags: [coding, black-hole-sim, cuda]
created: 2026-11-12
---

# v0.8d — Cinematic polish (`--cinematic`)

Sequel to [07d-cinematic-gpu](07d-cinematic-gpu.md). **All of this is aesthetic, not physical**, and only turns on with `--cinematic` or the explicit flags: the default (physical) look **does not change** and is verified byte for byte (see Regression).

## Effects (`render.cu`, resolve stage)
| Effect | Where in the chain | Implementation | Flag (value in `--cinematic`) |
|---|---|---|---|
| Star-shaped **glare** | HDR, added before tone mapping | *top-hat*: pixel minus the coarse level (1/8) of the bloom pyramid, above a threshold → 3 streaks at 0°/60°/120° in both directions (6 points) with `exp(−d/τ)` decay, τ = 16 px at half resolution (≈ 32 px), 48 taps/point | `--glare s` (0.10), `--glare-thr` (0.5) |
| Radial **chromatic aberration** | HDR, when reading the buffer | R sampled at `r(1 + c·rn)`, B at `r(1 − c·rn)` (`rn` = distance to the centre / half-diagonal; bilinear) | `--chroma c` (0.003) |
| **Vignette** | HDR, after bloom and glare | `1 − v·smoothstep((rn − 0.35)/0.75)` | `--vignette v` (0.30) |
| **Film grain** | **after** tone mapping and gamma | triangular hash noise `(x, y, seed ^ frame)`, amplitude `g·(0.35 + 0.65·4L(1−L))` (more in the mid-tones); fixed seed ⇒ reproducible; the video passes the frame index | `--grain g` (0.025); `--grain 0` turns it off |
No depth of field (the camera is a pinhole). The preset also keeps v0.7d's ACES, bloom 0.12, `--beaming 0.5` and clamp 50. Flags after `--cinematic` override it (e.g. `--cinematic --glare 0`).

**Why the *top-hat* in the glare:** with a plain threshold on the luminance the glare came off the whole disk (a wide, bright smear) and the stars, which in HDR only reach ~2, never triggered it. Subtracting the coarse pyramid level leaves only the compact points: stars and the hot core of the disk.

## Regression (mandatory)
`tools/regress_v07d.sh` (also in `ctest`, `v07d_regression`): md5 of the PNGs of 3 640×360 renders of the physical look (float 4 samples NT; double newtonian a = 0.998 θ = 86°; float aces + bloom with an orbit observer). The golden (`tools/regress_v07d.golden`) was generated with the v0.7d binary **before** writing the polish and **is still byte-identical**. The selftest adds: polish at 0 = identical bytes; glare raises the mean luminance; grain reproducible with a fixed seed and `frame_index` changes the noise; vignette does not brighten the corner; chroma changes the image.

## Cost (1080p, float, a = 0.9)
Bloom + glare stage: 0.048 ms with bloom only → **0.67 ms** with glare (≈ 0.62 ms is the glare: 6 points × 48 taps at half resolution); final resolve +0.02 ms. The frame is still dominated by the trace (8.1 ms/sample); 32 samples + polish = 0.26 s.

## Images (`assets/v08d-*`, 32 samples, float)
`v08d-a09-no-polish.png` / `v08d-a09-polish.png` (a = 0.9, θ = 80°) and `v08d-a0998-no-polish.png` / `v08d-a0998-polish.png` (a = 0.998, θ = 86°): "sem" (without) = `--cinematic` with glare, grain, vignette and chroma at 0 (ACES/bloom/beaming only); "com" (with) = full preset. 6 s clip with the polish: `assets/v08d-spin-polish.mp4` (spin 0 → 0.998, 360 frames, 8 samples/frame, 31.0 s of rendering, 86 ms/frame, 4.6 MB; the grain varies per frame).

## Limitations
Cross-convolution glare without real diffraction (no per-point spectrum); the hot core of the disk also generates glare (visible as an "X" in the brightest area at a = 0.998), which is stylistic but debatable; chroma only acts on the final HDR (no dispersion of the bloom); the grain is not temporally coherent (new noise per frame, like film). The preset parameters were chosen by eye. Not seen by hand: the clip in continuous playback.
