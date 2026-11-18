---
tags: [coding, physics, black-hole-sim]
created: 2026-11-11
---

# v0.2 — Light in Schwarzschild (2D)

## Model
Schwarzschild metric, equatorial plane, units G = c = M = 1 (horizon r = 2M). Light follows **null geodesics**. With `u = 1/r` and `φ` as the independent variable:

```
d²u/dφ² + u = 3 M u²        (relativistic Binet equation)
(du/dφ)² = 1/b² − u²(1 − 2Mu)    (first integral; b = L/E, impact parameter)
```

Without the `3Mu²` term we get `u'' + u = 0`: a straight line (Newtonian light without gravity). The extra term is the curvature of spacetime.

## Numerics
RK4 in φ (step 1e-3). The ray starts at `u = 0, w = du/dφ = 1/b` (coming from infinity). Captured when `u ≥ 1/(2M)`; escapes when `u` returns to 0 with `w < 0` (crossing φ by linear interpolation). Deflection = `φ_end − π`. Code: `core/include/bhs/schwarzschild.hpp`.

## Verified oracles (tests `core/tests/test_schwarzschild.cpp`)
| Oracle | Expected | Obtained |
|---|---|---|
| Photon sphere: u = 1/3 is a fixed point | u'' = 0 | 0 (exact) |
| Shadow, captured/escapes bisection | b_c = √27 = 5.19615242 | 5.19615242 |
| Weak deflection, b = 400 | series 4/b + 15π/(4b²) + 128/(3b³) = 0.01007430 | 0.01007430 |
| Weak deflection, b = 100 | 0.04122076 | 0.04122254 (remainder of order b⁻⁴) |
| b = 1.0001·b_c | deflection > 2π (loops round the photon sphere) | 8.81 rad (1.40 turns) |
| Convergence: dφ 1e-3 vs 5e-4 | equal | up to 1e-8 |

Note: the classical weak-deflection limit is 4M/b (twice the Newtonian value 2M/b, the famous 1919 Eddington test).

## Next
v0.3: go from 2D to an image: launch one ray per pixel, convert (b, φ) to a 3D direction and see where each ray ends up (horizon = black, or star field).
