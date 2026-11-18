---
tags: [coding, black-hole-sim, cuda, physics]
created: 2026-11-12
---

# v0.9a2 — Soft, irregular outer edge

André's question (2026-11-12): "the limit is a perfect curved line; is that what happens in real life?". **No.** The hard `r < r_out` cut of v0.8 was an artefact of the model (the Novikov-Thorne emission falls as r⁻³ but the code stopped abruptly at a radius where the brightness was still appreciable), and the outline came out as an exact geometric ellipse. This phase only deals with the edge; the thickness (H/r) and the 3D turbulence stay for v0.9b/v0.9c (roadmap-v0.9-v1.1). **The brightness inside the ISCO (plunging region) was not done**: it needs its own physics (infalling gas, v0.9b / GRMHD).

## What changed
| Piece | Description | Where |
|---|---|---|
| **Edge weight** | `w(r) = 1 − smoothstep(r_e(1−s), r_e(1+s), r)`; equals 1/2 at `r_e`; `s = 0` is the hard cut | `disk_emission.hpp` (`edge_weight`) |
| **Irregular edge** | `r_e(φ) = r_out (1 + a (2n−1))`, `n` = 2-octave value noise over the azimuth circle (fixed in the disk frame) | `edge_radius`, kernel |
| **Semi-transparent disk in the fringe** | a ray crossing the fringe takes `w` of the emission and **continues behind it** with transmittance `1−w`; up to 3 layers (the 3rd closes the budget); the sky (or a 2nd disk layer, or the horizon) fills the rest | kernel: `skip` in the trace (`kerr::Disk::skip`, `bk::trace(..., skip)`) |

Flags: `--edge-soft s` (0 to 0.9; 0 = hard, default) and `--edge-noise a` (0 to 0.5; 0 = circle, default). Recommended values: `--edge-soft 0.12 --edge-noise 0.1` with the default `--rout` (16); for a more extended disk `--rout 22 --edge-soft 0.2 --edge-noise 0.12`. `s = 0.3` with `rout = 14` already eats the bright core (the decay zone goes from 9.8 to 18.2) and gets blurry.

**Why `skip` and not "paint it black":** fading to black at the last impact point darkens the edge against the sky, the opposite of what a tenuous gas does. With the ray continuing, the sky shows **through** the fringe, and the lensed image of the other side of the disk stops being hidden by an opaque edge (visible in the images: the arc under the hole).

## Oracles
Core (`test_disk_emission.cpp`, 88 cases, all green): `edge_weight` with `s = 0` is the hard cut; with `s > 0` it is 1 below `r(1−s)`, 0 above `r(1+s)`, 1/2 at `r`, monotonic and with no jumps > 0.01 on a 2000-point grid; `edge_radius` spans `rout(1 ± a)`; **`skip = 1` returns the 2nd crossing**: for 8 rays with 2 crossings, `trace(skip = 1)` matches to 1e-9 a clipped disk (ring) that excludes the 1st and keeps the 2nd; `skip = 0` equal to the old behaviour.

GPU (`--selftest`, section `v0.9a2`): `skip = 1` GPU vs `kerr::trace` (466 rays with a 2nd crossing, 0 with a different fate): p99(|Δr|) = 5.7e-6 at tol 1e-8. This p99 **scales with the tolerance** (1.9e-4, 5.7e-6, 1.2e-7 for tol = 1e-6, 1e-8, 1e-10): these are higher-order images, sensitive to the integrator, not a systematic error; the test limit is 2000·tol. Image: `--edge-soft 0 --edge-noise 0` equals the default (identical mean luminance); `--edge-soft 0.3` gives 99.8 % of the hard one's mean brightness and > 58 % of pixels unchanged (those far from the edge); with noise the HDR is finite. **Regression: the default look stays byte-identical** (`ctest` 3/3, `v07d_regression`).

A slip in the core test: the first sweep of rays with 2 crossings used a window that was too wide and found none; it moved to a fine 60×60 grid near the shadow and handles both cases (2nd crossing inside or outside the 1st).

## Cost (a = 0.9, θ = 75°, 1080p float, 32 samples, rout 16)
10.15 ms/sample (hard) · 11.57 (`--edge-soft 0.12`, +14 %) · 12.01 (+ `--edge-noise 0.1`, +18 %) · 12.36 (`rout 22`, soft 0.2, noise 0.12). The cost comes from the rays crossing the fringe (a 2nd trace) and from the noise.

Images: `assets/v09a2-comparison.png` = hard (top-left), `--edge-soft 0.12` (top-right), `+ --edge-noise 0.1` (bottom-left), `--rout 22 --edge-soft 0.2 --edge-noise 0.12` (bottom-right); individual ones in `assets/v09a2/`.

```bash
build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui --shot k.png --a 0.9 --theta 75 --prec float --samples 32 --rout 16 --edge-soft 0.12 --edge-noise 0.1
```

## Limitations
- It is still a **thin** disk: the edge is soft in the plane, there is no thickness; the fringe only has an effective optical depth `1−w`, not a radiative-transfer integral (that is v0.9b).
- The edge noise is **static** (it does not rotate with Ω(r)); v0.9c advects it.
- The 3-layer limit discards the rest (< 0.1 % transmittance) instead of integrating it.
- Opt-in: the default look did not change. Making the soft edge the default requires updating the regression golden (`tools/regress_v07d.sh --update`); left for André to decide.
