---
tags: [coding, physics, black-hole-sim]
created: 2026-11-12
---

# v0.8a — The pilot: a Kerr timelike geodesic with thrust

Up to v0.7 the observer sat on a sphere of radius `D` (orbital camera) or had a *given* velocity (`--beta`). To **fly** around the black hole the observer's own trajectory has to be physical: a Kerr **timelike geodesic**, which the pilot can alter with thrusters. This phase is the core (CPU, double precision); the interactive interface on CUDA is v0.8b.

## Equations (`core/include/bhs/pilot.hpp`)
Unit mass; constants `E` (energy), `L` (`L_z`) and `Q` (Carter). In Mino time (`dτ_proper = Σ dλ`) the equations separate, with the mass terms that light does not have:

```
(dr/dλ)² = R(r)  = [E(r²+a²) − aL]² − Δ [ r² + (L − aE)² + Q ]
(dμ/dλ)² = M(μ) = (1−μ²) Q − a²(1−E²) μ²(1−μ²) − L² μ²          μ = cos θ
dφ/dλ    = −aE + L/(1−μ²) + a [E(r²+a²) − aL]/Δ
dt/dλ    = −a (aE(1−μ²) − L) + (r²+a²)[E(r²+a²) − aL]/Δ
```

Integrated in the 2nd-order form `r'' = R'/2`, `μ'' = M'/2` (smooth at turning points) with the same adaptive Dormand-Prince 5(4) and the same **reprojection onto the first integrals** as light (kerr.hpp), plus the proper time `τ` and the coordinate time `t` as states. The step is limited so the accumulated `τ` lands exactly on the requested instant: `advance(dτ)` advances *dτ* of proper time.

## Pilot state
Position `(r, θ, φ)` and velocity `β` (vector in c) in the local ZAMO frame `(e_r, e_θ, e_φ)`. The constants come from the velocities:
`E = γ(α + ωϖβ_φ)`, `L = γϖβ_φ`, `p_r = γβ_r√(Σ/Δ)`, `p_θ = γβ_θ√Σ`, `Q = p_θ² + μ²[a²(1−E²) + L²/(1−μ²)]`; the inverse conversion recovers `β` to 1e-12 (test). Orientation: that of the ZAMO boosted by the velocity (**not** Fermi-Walker transported: an approximation the pilot does not notice at moderate speeds; kept as a limitation).

## Thrust
In the pilot's rest frame the 4-velocity `(1, 0)` becomes `(√(1+δ²), δ)` with `δ = a_thrust · dτ`; boost back to the ZAMO and recompute `E, L, Q`. Applied in short steps it is a constant proper acceleration (addition of rapidities).

## Oracles (tests `core/tests/test_pilot.cpp`, 7 new, 75 in total)
| Oracle | Expected | Obtained |
|---|---|---|
| state → constants → state | identity | 1e-12 |
| Kerr circular orbit (a = 0.9, r = 8, ~100 of τ) | r constant; E conserved | 1e-8 / 1e-9 |
| `dφ/dτ` of the circular orbit in Schwarzschild (r = 10) | `r^{-3/2}/√(1−3/r)` | 1e-8 |
| Radial fall from rest at infinity, `r(τ)^{3/2} = r₀^{3/2} − (3/2)√(2M) τ` | exact | 1e-8 (E = 1 to 1e-12) |
| Thrust in the flat limit (r = 10⁶, a = 0.01, τ = 100) | `β = tanh(a τ) = 0.76159` | 2e-5 |
| Orbit below the ISCO (r = 5.4, v = 0.98 of Keplerian) | falls into the horizon | captured |
| Generic inclined Kerr orbit (100 × 3 of τ) | E, L, Q constant; E reconstructed from the state | 1e-14 / 1e-7 |

## Scenario numbers (scratch, from the same class)
- **Radial fall from r = 40 in Schwarzschild:** reaches the horizon (r = 2.0009) after **τ = 117.92** of proper time (analytic 117.92), but only after **t = 149.1** of coordinate time (which diverges logarithmically as r → 2M).
- **Circular orbit at r = 10:** one turn takes τ = 166.24 (Schwarzschild) and τ = 172.87 (a = 0.9, prograde); `t/τ = 1.19523 = 1/√(1−3/r)` at a = 0 (gravitational time dilation + kinematics), 1.18212 at a = 0.9.
- **Radial fall in Kerr (a = 0.9) from r = 40:** τ = 118.08 down to r = 1.4368 (horizon 1.4359).
The cost is negligible: thousands of steps per second on the CPU, well within the budget of a frame.

## Limitations
No Fermi-Walker transport of the orientation (the camera stays aligned with the local ZAMO, not with a gyroscopic frame); no regularisation of the axis (passing through the pole makes `φ` jump by π, as in Boyer-Lindquist coordinates; it only affects the camera azimuth); inside the horizon the Boyer-Lindquist coordinates stop working: the simulation **stops** (`captured()`) at `r < r₊ + 10⁻³`.

## Next
v0.8b (omarchy): the flight interface on CUDA, which uses this header: orbital vs flight mode, looking around, thrusters, HUD (r, β, γ, proper time, frequency shift).
