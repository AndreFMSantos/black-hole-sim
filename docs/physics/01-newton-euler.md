---
tags: [coding, physics, black-hole-sim]
created: 2026-11-11
---

# v0.1 — Newton + integrators

## Two different things
1. **The physical law**: Newtonian gravity, `a = -GM r / |r|^3` (units GM = 1). This is the *model*.
2. **The integrator**: how we advance time (`x(t+dt)`, `v(t+dt)`). This is the *numerical method*. Euler, Verlet and RK4 solve the **same** law with different errors.

## Integrators (in `core/include/bhs/integrators.hpp`)
| Method | Order | Symplectic | Behaviour on the circular orbit (dt = 0.01, 10 orbits) |
|---|---|---|---|
| Explicit Euler | 1 | no | r: 1.0 → 1.648; ΔE = +0.203; ΔL = +0.298. Spirals outwards. |
| Velocity Verlet | 2 | yes | r = 1.000000; max ΔE 1.25e-9 (oscillates, does not drift); ΔL ~ 1e-15 |
| RK4 | 4 | no | r = 1.000000; ΔE 8.7e-11 (drifts only over a very long time) |

Numbers copied from the real run (`scratch/probe`, 2026-11-11), fixed as thresholds in `core/tests/test_integrators.cpp`.

## Why Euler fails
It uses the old state to update both position and velocity. At each step the position advances along the tangent, that is, slightly outside the circle, and the velocity is updated with the acceleration of the old point. The error always has the same sign, so it **accumulates** as extra energy. Verlet is time-reversible and preserves the phase-space area (symplectic), so the error stays bounded.

## Oracles
- Circular orbit: `v = sqrt(GM/r)`, `T = 2π sqrt(r³/GM)`, `E = -GM/(2r)`, `L = sqrt(GM r)`.
- E and L conserved; Verlet reversible (reversing v and integrating back returns the initial point).

## Next
v0.2: light does not follow Newton. Schwarzschild null geodesics: `u'' + u = 3Mu²`, with `u = 1/r`.
