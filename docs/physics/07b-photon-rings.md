---
tags: [coding, physics, black-hole-sim]
created: 2026-11-12
---

# v0.7b — Photon rings: images by order and the Lyapunov exponent

## Idea
On a thin disk, each ray traced backwards crosses the disk plane `n` times before escaping or being captured. `n = 1`: direct image; `n = 2`: the back face of the disk (strong lensing); `n ≥ 3`: **photon rings**, ever thinner. Near the critical impact parameter `b_c = √27 M`, the total sweep of the ray grows as `φ_total ≈ −ln(b − b_c) + const`: rays loop around the photon sphere (r = 3M) before leaving. If `b_k` is the `b` where `φ_total = π/2 + kπ` (top-down observer), then

```
(b_{k+1} − b_c) / (b_k − b_c) → e^{−γ},     γ = π   (Schwarzschild; Lyapunov exponent of the photon sphere)
```

Each ring is `e^{−π} ≈ 4.32 %` of the previous one: the image of the black hole is a logarithmic fractal of rings, which the Event Horizon Telescope tries to resolve (`n = 3` not yet).

## Implementation
`core/include/bhs/photon_rings.hpp`: `count_crossings` (RK4 in φ, as in v0.2; counts the crossings of φ₀ + kπ) and `ring_boundary(r0, k)` (bisection in `b` until `φ_total = π/2 + kπ`). App `apps/v0_7b-photon-rings` (Schwarzschild, geometry only: no disk or sky): colour by order, captured rays dimmed; prints the boundaries measured on the centre line and the reference `ring_boundary(D, k)`.

## Oracles (tests `core/tests/test_rings.cpp`, 4 new, 57 in total)
Ratios `(b_{k+1} − b_c)/(b_k − b_c)`, observer at infinity (`r0 = 10⁵`):

| k | b_k | b_k − b_c | ratio |
|---|---|---|---|
| 1 | 6.167489 | 9.713e-1 | — |
| 2 | 5.227937 | 3.179e-2 | 0.03272 |
| 3 | 5.197505 | 1.353e-3 | 0.04257 |
| 4 | 5.196211 | 5.842e-5 | 0.04318 |
| 5 | 5.196155 | 2.524e-6 | **0.04321** |
| 6 | 5.196153 | 1.091e-7 | **0.04321** |
| limit | | | e^{−π} = 0.043214 |

The tests require 2 % for k = 3..5 and check that `count_crossings` gives `k` for `b_k + ε` and `k + 1` for `b_k − ε`.

**Measured in the image, centre line, observer at D = 40** (the distance changes the boundaries, so the reference is `ring_boundary(40, k)`):

| zoom | transition | measured b (pixel) | `ring_boundary(40, k)` |
|---|---|---|---|
| fov 20° | n: 1 → 2 | 5.985656 | 5.998084 (1 pixel of difference) |
| fov 20° | n: 2 → 3 | 5.217050 | 5.223987 |
| fov 0.12° | n: 2 → 3 | 5.224038 | 5.223987 |
| fov 0.12° | n: 3 → 4 | 5.197320 | 5.197340 |
| fov 0.12° | n: 4 → 5 | 5.196155 | 5.196204 |

At the zoom (≈ 0.0000021° per pixel) the boundaries agree to 1e-4 in b.

## Images (assets/)
- `v07b-orders-overview.png`: fov 20°. Gold = direct image (n = 1); blue = back face (n = 2); magenta thread = first photon ring (n = 3); inside the shadow, the dimmed colours are captured rays (brown: already crossed the plane once; black: none).
- `v07b-orders-zoom.png`: fov 0.12°, centred on the shadow edge: the bands n = 2, 3, 4, 5 (blue, pink, thin green, …) show the narrowing by e^{−π}. It is a diagnostic image, almost "graphical".

## Limitations
Schwarzschild only. In Kerr the Lyapunov exponent depends on the spin and the orientation, and the shadow is not circular; it would come after v0.7d has Kerr with crossing counts (requires touching the Kerr `trace`, which was being fixed in v0.6c). (Done later: see `07b2-kerr-lyapunov.md`.)

## Next
v0.7c: the observer stops being static (aberration and Doppler).
