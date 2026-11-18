---
tags: [coding, physics, black-hole-sim]
created: 2026-11-12
---

# v0.7c — Moving observer: aberration and Doppler

So far the camera was a **ZAMO** observer (zero angular momentum), almost at rest. A real probe orbits or falls. Since spacetime is locally flat, it is enough to compose the Kerr geometry with **special relativity** in the local ZAMO frame: `(e_r, e_θ, e_φ)`.

## Transformation (`core/include/bhs/lorentz.hpp`)
The observer moves with velocity `β` (vector, in c) in the ZAMO frame. A photon of direction `n'` in the observer frame and `n` in the ZAMO frame, with 4-momentum `p = E(1, n)`:

```
E  = γ E' (1 + β·n')        p_∥ = γ (p'_∥ + β E')      p_⊥ = p'_⊥      → n = p/E
E' = γ E  (1 − β·n)                                                      (inverse)
dop = E'/E
```

In the image: the viewing direction of each pixel gives `n'`; the boost gives `n` (aberration), which enters the Kerr `trace` as before; the factor `dop` multiplies the disk `g` (`g_total = g_ZAMO · dop`). The Kerr **circular-orbit** velocity relative to the ZAMO is `β = ϖ(Ω − ω)/α` (at `a = 0`: `√(M/(r−2M))`); the **free-fall** one from rest at infinity is `√(2M/r)` (exact in Schwarzschild; in Kerr it is an approximation).

## Oracles (tests `core/tests/test_boost.cpp`, 6 new, 63 in total)
Zero velocity is the identity; aberration of a perpendicular photon: `n = (β, 1/γ)` and `dop = 1/γ` (transverse Doppler); head-on photon: `dop = √((1+β)/(1−β))` (= 2 at β = 0.6); round trip observer → ZAMO → observer to 1e-12; `circular_beta(0, r) = √(M/(r−2M))` to 1e-12 (r = 8, 20, 40); `infall_beta(2M) = 1` at the horizon.
Image: `--beta 0,0,0` is identical (MAE 0.0000) to the static observer.

## Usage
`v0_6-kerr-cpu --observer static|orbit|infall` or `--beta br,bθ,bφ` (components in c, in the ZAMO frame).

## Images (assets/)
| Image | Observer | |
|---|---|---|
| `v07c-static-r12.png` | static, r = 12, θ = 85°, a = 0.9 | reference |
| `v07c-orbit-r12.png` | circular orbit, r = 12, β = 0.2957 | the hole moves forward (right), the disk "falls" onto the camera; the stars stretch into streaks |
| `v07c-fast-beta07.png` | β = 0.7 along +φ, r = 40 (hypothetical) | the hole ends up at the right edge of the screen: pure relativistic aberration |
| `v07c-infall-r10.png` | radial infall, r = 10, β = −√(2/10) = −0.447 | the disk fills the field; the light concentrates ahead |

## Limitations
The **sky** undergoes aberration but **not** Doppler/colour (the map is RGB, not a spectrum; a spectrum model per star would be needed); the sky brightness is not scaled by `dop⁴`. The observer keeps the axes aligned with `(e_r, e_θ, e_φ)` (camera without its own rotation). Free fall in Kerr uses `√(2M/r)` relative to the ZAMO, which is only approximate. The observer does not "advance" in time (one image = one instant).

## Next
v0.7d (omarchy): cinematic GPU with the NT LUT, progressive supersampling, bloom and filmic tone mapping; then the flight video (v0.7e).
