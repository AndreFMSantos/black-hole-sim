---
tags: [coding, physics, black-hole-sim]
created: 2026-11-11
---

# v0.6a — Kerr (rotating black hole), CPU reference

CUDA (v0.6b) is not yet installed on omarchy; so this phase first closes the **double-precision CPU reference**, tested against analytic oracles. It is the same order we followed from v0.3 → v0.5: the slow, verifiable version is born before the fast one.

## What changes compared with Schwarzschild
With rotation (`a` = angular momentum per unit mass, M = 1) spacetime drags the frames (Lense-Thirring effect). Spherical symmetry is lost: **rays stop being planar**, so the trick of reducing each pixel to a 2D ODE (v0.3) no longer works. There are three constants of motion per photon: `E` (= 1), `L = L_z` and the **Carter constant `Q`**.

## Equations (`core/include/bhs/kerr.hpp`)
Boyer-Lindquist coordinates, `μ = cos θ`, `Δ = r² − 2r + a²`. In Mino time `dτ = dλ/Σ` the equations separate:

```
(dr/dτ)² = R(r)  = [(r²+a²) − aL]² − Δ[(L−a)² + Q]
(dμ/dτ)² = M(μ)  = (1−μ²)(Q + a²μ²) − L²μ²
dφ/dτ    = L/(1−μ²) − a + a(r²+a²−aL)/Δ
```

To avoid tripping over the turning points (where `√R` changes sign) the smooth 2nd-order form is integrated: `r'' = R'(r)/2`, `μ'' = M'(μ)/2`. Using `μ` instead of `θ` removes the singularity on the axis. The ray is traced **backwards** (from the observer to the sky), in `τ' = −τ`.

## ZAMO observer
Camera = zero-angular-momentum observer at `(r, θ)`: `α = √(ΣΔ/A)`, `ω = 2ar/A`, `ϖ = √(A/Σ) sin θ`, `A = (r²+a²)² − a²Δ sin²θ`. From the photon's local direction `n = (n_r, n_θ, n_φ)` come `E_loc = 1/(α + ωϖ n_φ)`, `L = ϖ E_loc n_φ`, `p_r`, `p_θ` and `Q = p_θ² + μ²(L²/(1−μ²) − a²)`. In the image: `fwd = −e_r`, `up = −e_θ`, `right = +e_φ`, and the arriving photon is the opposite of the viewing direction.

## Integrator
Adaptive Dormand-Prince 5(4) (`rtol 1e-8`, `atol 1e-9`; 1e-9/1e-10 by default in `core`). The tableau coefficients are covered by a test (harmonic oscillator). **Lesson:** the 2nd-order form does not correct the error in the first integrals. The accumulated **absolute** error at large r, where `R ~ r⁴` is huge, destroyed the relative precision near the hole (with an observer at r = 5000 the ray was "captured" by mistake, relative violation 1.0). Fix: after each step, reproject the velocities onto `(dr/dτ)² = R(r)` and `(dμ/dτ)² = M(μ)`, keeping the sign. `max_inv_err` records the violation *before* the projection (honest local error): ≤ 1e-6 in the test.

## Escape and shadow
The ray "escaped" when `r > r_far = 10⁶` moving away; the sky direction is `(θ, φ)` at that point. With `r_far = 10⁴` the error of ~b/r_far (0.03°) shifted the stars ~0.4 px: MAE 6.0/255 against v0.4. With 10⁶: 0.135.

## Disk
**Prograde** equatorial circular orbits: `Ω = 1/(r^{3/2} + a)`, `u^t = (r^{3/2} + a)/(r^{3/4} √(r^{3/2} − 3r^{1/2} + 2a))`. Inner edge at the Kerr ISCO (BPT 1972): 6M (a = 0), 2.321M (a = 0.9), 1.237M (a = 0.998).
Redshift: `g = E_loc,obs / (u^t (1 − ΩL))`, with `E_loc,obs = (1 − ωL)/α` at the observer. At `a = 0` it reduces exactly to v0.4's `g`. The crossing of the plane `μ = 0` is located with cubic Hermite interpolation of the step (value + derivative at both ends), opaque disk as before.

## Oracles (tests `core/tests/test_kerr.cpp`, 11 new, 48 in total, all green)
| Oracle | Expected | Obtained |
|---|---|---|
| Horizon, ISCO, photon orbit at the limits (a = 0, 1) | 2, 6, 3 · 1, 9 (retro), 1, 4 | exact |
| ISCO is the minimum of the circular-orbit energy (independent of the formula) | dE/dr = 0 | \|dE\| < 1e-6 (a = 0.3, 0.9, 0.998) |
| Equatorial circular photon orbit is a double root of R(r) (a = 0.5, 0.9, pro and retro) | R = R′ = 0 | 1e-9 |
| Constants from the ZAMO satisfy the null condition | R(r₀) = (Δp_r)² · M(μ₀) = (sin θ p_θ)² | 1e-9 |
| Dormand-Prince on a harmonic oscillator (t = 10) | error drops with the tolerance | 1e-4 (tol 1e-6) → 1e-8 (tol 1e-10) |
| a = 0: deflection of equatorial rays vs `schw::trace_from` (b = 8, 12, 30) | equal | 2e-3 |
| a = 0: capture threshold | √27 ≈ 5.196 | 5.1 captures, 5.3 escapes |
| **a = 0.9: shadow edges in the equatorial plane (Bardeen 1973)** | \|L\| = ξ(r_ph) pro and retro | 2e-3 (observer at r = 5000) |
| a = 0: disk crossing vs `schw::trace_with_disk` | equal | 2e-4 |
| a = 0: Kerr `g` vs `disk::redshift` | equal | 1e-12 |

**Image comparison (a = 0 vs v0.4, 960×540, SSAA 1, same real sky):** MAE 0.135/255, 0.010 % of pixels with error > 4, 0.000 % > 16. The residual difference is the finite `r_far` and the RK45 tolerance.

## Results (assets/)
- `v06-kerr-shadow-a0.png` / `v06-kerr-shadow-a0998.png`: shadow only, observer on the equator, no disk. The a = 0.998 one has the **D shape**: flattened on the approaching (prograde) side, widened on the opposite one. It is Bardeen's shape.
- `v06-kerr-a09.png` (θ = 80°) and `v06-kerr-a0998-edgeon.png` (θ = 86°): disk down to the Kerr ISCO. The prograde disk gets much closer to the horizon; in the extreme case the Doppler and redshift leave the disk almost entirely dark except the approaching side. Exposure 3×. Nothing was artificially smoothed (the film smoothed the beaming).
- Cost: 1080p, SSAA 3×3, 11 threads (Mac): 44–54 s per image (v0.4 with Schwarzschild: 12 s).

## Limitations
Thin, opaque disk with a Newtonian temperature profile; ZAMO observer (not orbiting); no negative `a` in the CLI (a retrograde disk would be `a < 0`, not tested in the images).

## Next
**v0.6b (done):** [Kerr on CUDA](06b-kerr-cuda.md), validated against this reference.
