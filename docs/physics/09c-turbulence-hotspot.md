---
tags: [coding, black-hole-sim, cuda, physics]
created: 2026-11-12
---

# v0.9c — 3D turbulence, hot spot and flare

Third piece of roadmap-v0.9-v1.1 (on top of [09b-thick-disk](09b-thick-disk.md)). It only acts in volume mode (`--hr > 0`); with `--turb 0` and no `--spot` the result is that of v0.9b (equal to ~1e-6).

## Model
| Piece | Description | Where |
|---|---|---|
| **Disk time** | `--time T` (in M); the turbulence pattern and the spot rotate with the Kerr `Ω(r) = 1/(r^{3/2} + a)`: **differential rotation** (inner rings run ahead) | `core/include/bhs/turbulence.hpp`, `render.cu` |
| **Turbulence** | lognormal density factor `exp(a·n − a²σ²/2)`, `n` = 3-octave fbm (3D value noise, weights 1, ½, ¼) in `[-1, 1]`, `σ² = 0.0584` (measured with `tools/turb_var.cpp`); coordinates in the rotating frame `φ' = φ − Ω t`: `(cos φ', sin φ')/(s h)` (continuous in φ) and `(ln r + 1.7 μ)/(s h)`; cells of ≈ `s·H(r)` (`--turb-scale`, default 2) | `turb::density_factor`, `turb_density_f` (float) |
| **Hot spot** | `--spot r,φ0,size,amp`: Gaussian in Euclidean distance `d² = r² + r_h² − 2 r r_h cos(φ−φ_h) + z²` orbiting with `Ω(r_h)`; raises the emissivity: `T → T (1 + amp·G)^{1/4}`, so the bolometric radiance `∝ T⁴` rises by exactly `1 + amp·G` | `turb::spot_*` |
| **Flare** | `--spot-flare t,w,boost`: `amp(t) = amp (1 + boost·exp(−(t−t_f)²/2w²))` | `spot_amp_t` |
| **Video** | `--disk-rate R` (M per video second) and `--clip disk` (camera fixed at 75°; if `--hr` was not given: `--hr 0.08 --flare 0.15 --tau 4 --edge-soft 0.12 --turb 1.2 --turb-scale 1.2`; 20 M/s) | `main.cpp` |

Flags: `--turb a [--turb-scale 2] --spot r,phi_deg,size,amp --spot-flare t,w,boost --time T [--disk-rate R]`.

## Approximation to declare: instantaneous light (solved with `--slowlight`, see [09c2-slow-light](09c2-slow-light.md))
The disk time is the **same at all points of the ray** (it ignores the propagation delay). Light takes ~`2r` M to cross the disk, and the period at `r = 9` is 175 M: ~20 % of an orbit. For the turbulence (which unfolds over many orbits) it is irrelevant; for the light curve of a flare and for the apparent position of the spot it **is not**, and it stays as the next step (*slow light*: integrate `dt/dMino` along the geodesic, with an oracle in Schwarzschild).

## Oracles
Core (`test_turbulence.cpp`, 8 new cases, **102 in total**, green; 611 932 assertions):
1. `fbm` deterministic, in `[-1, 1]`, mean ≈ 0 (< 0.01) and variance equal to the tabulated one (3 %).
2. `amp = 0` ⇒ factor exactly 1.
3. **No drift in the rotating frame:** `f(r, φ + Ω(r) t, μ, t) = f(r, φ, μ, 0)` to 1e-9, for r = 6.5, 9, 14 and t = 0, 37, 250.
4. **Differential rotation:** following the inner ring at the outer ring gives a different pattern (the rotation is not rigid), with `(Ω₁ − Ω₂) t > 1 rad` at t = 90.
5. Mean density factor ≈ 1 (3 %) over the disk.
6. **Spot period** `2π/Ω(r_h) = 2π(r_h^{3/2} + a)` (equal phase modulo 2π, T factor equal to 1e-12).
7. `(1 + amp G)` exact in the bolometric radiance; `G = 1` at the centre, `e^{−1/2}` at 1σ in r and in z; spot 20 M from the centre ≈ 1; `amp = 0` ⇒ 1.
8. Flare: peak `amp(1+boost)` at `t_f`, `e^{-1/2}` at 1w, → `amp` far away.

GPU (`--selftest`, section `v0.9c`, in `ctest`):
| Test | Result | Limit |
|---|---|---|
| GPU (float) density factor vs core (double), t = 137, 3000 points | 1.5e-5 | 5e-3 |
| GPU spot T factor vs core (with flare) | 1.6e-6 | 1e-4 |
| **spot: image(t) = image(t + 2π/Ω(r_h))** (T = 175.30 M) | **difference 0**; at half a period 18 % | 1e-4 / > 0.5 % |
| **rotation direction consistent with the Doppler:** the spot, starting from the far side, moves towards the approaching (brighter) side | spot centroid −13.9 → −33.7 px in 0.5 rad; bright side −27.9 px | same sign, > 2 px |
| turbulence preserves the mean luminance | ×0.9986 | ±10 % |
| the image evolves with the disk time (t = 0 vs 60) | 1.2 % | > 0.1 % |

## Cost (a = 0.9, θ = 75°, 1080p float, 32 samples, `--hr 0.08 --flare 0.15 --tau 6 --edge-soft 0.12`)
25.30 ms/sample (smooth) · 25.82 (`--turb 0.7`, +2 %) · 25.98 and 26.01 (with spot, t = 0 and 40). The 3-octave noise in float only costs +0.5 ms.

## Images and clip
`assets/v09c-comparison.jpg` (smooth / `--turb 0.7` t = 0 / t = 60 / spot t = 0, 40, 80) and `assets/v09c-turbulence.jpg` (`--turb 1.2 --turb-scale 1.5`; `--turb 1.6 --turb-scale 1 --tau 3`: fine filaments). With `--turb 0.7` the texture is discreet because the integral along the ray averages many cells; that is why the clip uses 1.2 with scale 1.2. Clip `assets/v09c-disk.mp4` (outside Git, like the earlier videos; regenerate with the command below): 8 s, 1080p60, 480 frames, 8 samples/frame, `--spot 8,200,1.2,5 --spot-flare 80,12,3`; **109.6 s of render+encode** (228.4 ms/frame, GPU 225.1), 1.5 MB.

```bash
build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui --video disk.mp4 --clip disk --seconds 8 --samples 8 --spot 8,200,1.2,5 --spot-flare 80,12,3 --prec float --a 0.9
```

## Limitations
- Instantaneous light (above); `dt/dMino` along the geodesic is the next step.
- The turbulence is density-only (the temperature does not fluctuate) and statistically isotropic in `(φ, ln r, μ)` coordinates, with no spiral/MRI structure.
- Value `fbm` with 3 octaves has subtle grid artefacts; a gradient noise (Perlin/simplex) would be better if it shows.
- The spot is rigid (not sheared by the differential rotation): it is a Gaussian at `r_h`, with size ≲ `H`.
