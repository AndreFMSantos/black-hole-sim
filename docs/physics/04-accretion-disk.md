---
tags: [coding, physics, black-hole-sim]
created: 2026-11-11
---

# v0.4 — Accretion disk

## Geometry: where the ray crosses the disk plane
The ray lives in its plane (v0.3), with position `r(cos φ ĉ + sin φ t̂)`. The disk is `z = 0`. That gives `ĉ_z cos φ + t̂_z sin φ = 0`, hence `φ_k = atan2(−ĉ_z, t̂_z) + kπ`. At each `φ_k`, `u` is interpolated between two RK4 steps and `r_in < r < r_out` is tested. **Opaque disk**: the first crossing inside the ring ends the ray. Rays that miss the disk on the first pass (for example, pass inside the ISCO) can bend and hit it later: this is how the image of the disk above the shadow and the arc below it are born (higher-order images). Code: `trace_with_disk` in `schwarzschild.hpp`.

## Emission
- **Temperature** (Shakura-Sunyaev with zero torque at the ISCO): `T⁴ ∝ r⁻³ (1 − √(r_in/r))`, normalised to 1 at the maximum `r = 49/36 · r_in`. Newtonian form: still without the relativistic Novikov-Thorne corrections (v0.7 if it is fun).
- **Frequency shift** `g = ν_obs/ν_emit` for an emitter on a circular orbit (`Ω = √(M/r³)`, `u^t = 1/√(1−3M/r)`) and a static observer at `D`:
  `g = √(1 − 3M/r) / ( √(1 − 2M/D) · (1 − Ω L_z/E) )`
  with `L_z/E` of the photon (travelling from the emitter to the observer) = `−b (ĉ × t̂)_z`. It contains, all at once, the gravitational redshift, the transverse Doppler and the longitudinal Doppler.
- **Colour**: since `I_ν/ν³` is invariant, the observed spectrum of a black body is still a black body, with `T_obs = g · T_emit` (and brightness `∝ g⁴` bolometrically). The colour comes from integrating Planck with the CIE 1931 functions (Wyman-Sloan-Shirley fit) → XYZ → linear sRGB (`blackbody.hpp`), with a log-spaced LUT in T so as not to integrate per pixel.
- Reinhard tone mapping on the luminance (per sample, so there are no burnt points in the SSAA) + gamma 2.2.
- Turbulence: 3D value noise in (azimuth, log r) with a slight spiral. Purely aesthetic.

## Oracles
| Oracle | Expected | Obtained |
|---|---|---|
| T profile | 0 at the ISCO, maximum 1 at 49/36·r_in, ∝ r⁻³ᐟ⁴ far away | passes (test) |
| `g` without Doppler | √(1−3/r)/√(1−2/D) | exact |
| `g` with Doppler, nearly flat space (r = 1e5, 15 geometries) | special-relativistic Doppler factor `1/(γ(1−n·β))` | within 1e-3 (test) |
| Black body | 6500 K ≈ white, 3000 K red, 15000 K blue | passes |
| Crossing without curvature (b = 50, D = 1000) | r ≈ 50.0 | within 3% (gravity deflects a little) |
| Crossing convergence (b = 6, step 1e-3 vs 5e-4) | equal | within 1e-4 |
| Doppler asymmetry in the image (mean luminance, left vs right side) | without Doppler: equal; with Doppler: approaching side brighter | without: 93.2 / 93.5; with: 180.7 / 74.8 |

The disk rotates counter-clockwise seen from +z, with the observer at +x: the left side of the screen approaches us.

## Results (assets/)
`v04-disk.png` (inclination 10° above the plane), `v04-disk-edgeon.png` (4°), `v04-disk-nodoppler.png` (same render, with `--nodoppler`). All 1920×1080, SSAA 3×3, ~12 s on 11 threads.

## Honest limitations
Infinitely thin and opaque disk; non-relativistic T profile; no scattering or thick disk; no photon ring resolved beyond orders 1-2; non-rotating hole (Kerr is v0.6). The "Interstellar" colour is much more subdued than the physical one: there the Doppler shift was deliberately softened.

## Next
v0.5: real time. Move this to a GLSL 4.6 compute shader on the RTX 5080 (omarchy) with a free camera.
