---
tags: [coding, physics, black-hole-sim]
created: 2026-11-12
---

# v0.7b2 — The Lyapunov exponent in Kerr: spin changes how the rings shrink

In v0.7b I measured that, in Schwarzschild, each photon ring is `e^{−π} ≈ 4.32 %` of the previous one. With rotation the number changes **and depends on the direction** (prograde or retrograde). This note shows an **analytic** prediction (linearisation) against a **numerical** measurement (geodesic integration).

## Prediction (`core/include/bhs/kerr_lyapunov.hpp`)
Near the equatorial circular photon orbit (double root of `R(r)`), in Mino time `dr/dτ = √R ≈ κ(r − r_ph)` with `κ = √(R''/2)`, and the azimuth advances at `dφ/dτ = Φ(r_ph)`. A half turn (`Δφ = π`) multiplies the deviation by `e^γ`:

```
γ = π κ / |Φ|     (R'' = 4A + 8r² − 2K, A = r²+a²−aL, K = (L−a)² + Q;  Q = 0, L = ξ(r_ph) on the equator)
```

`γ = π` at `a = 0` (tested to 1e-9). In Kerr: the prograde orbit (co-rotating with the hole) is **less** unstable (smaller `γ`, wider rings), the retrograde one **more**.

## Measurement (test `core/tests/test_kerr_rings.cpp`)
Equatorial rays of a ZAMO observer at `D = 10⁵`, exact `L` (inverting `L → n_φ` with `kerr::nphi_for_L`, because `L = ϖE_loc n_φ` and the factor `1/α` shifts `L` by ~1e-5 at D = 10⁵; the first version of the test ignored this and saturated), bisection in `L` for sweeps `|φ_end| = kπ`, ratio between `L_{k+1} − L_c` and `L_k − L_c`:

| a | direction | r_ph | \|L_c\| | γ (linearisation) | e^{−γ} | measured k6/k5 | measured k7/k6 |
|---|---|---|---|---|---|---|---|
| 0 | — | 3.0000 | 5.1962 | 3.1416 | 0.04321 | 0.04321 | 0.04321 |
| 0.5 | pro | 2.3473 | 4.0963 | 2.3925 | 0.09140 | 0.09138 | 0.09140 |
| 0.5 | retro | 3.5321 | 6.1382 | 3.6656 | 0.02559 | 0.02559 | 0.02559 |
| 0.9 | pro | 1.5579 | 2.8444 | 1.2160 | 0.29641 | 0.29445 | 0.29580 |
| 0.9 | retro | 3.9103 | 6.8323 | 4.0041 | 0.01824 | 0.01824 | 0.01824 |
| 0.998 | pro | 1.0739 | 2.1109 | 0.1940 | 0.82362 | 0.71332 | 0.74153 |
| 0.998 | retro | 3.9982 | 6.9967 | 4.0795 | 0.01692 | 0.01692 | 0.01692 |

The test (3 %) covers a = 0, 0.9 pro and 0.9 retro. Prediction and measurement agree to 4–5 digits where convergence is fast; in the near-extremal prograde case (a = 0.998) the measurement **converges slowly** (0.713 → 0.742 → ... towards 0.824): the decay per ring is only 18 %, and the asymptotics need a much larger k (and much longer integrations).

## Physical reading
With `a = 0.9`, each prograde ring is 29.6 % of the previous one (against 4.3 % in Schwarzschild), but the retrograde one only 1.8 %. At `a = 0.998` the prograde rings barely shrink (82 %): light takes many turns to escape. This is why the image of a fast-spinning hole, seen from the approaching side, has such a much richer structure of thin rings.

## Limitations
Equatorial rays only (circular orbit in the plane); for an off-axis observer the factor depends on the inclination (Johnson et al. 2020). No crossing count in Kerr (the image by order only exists in Schwarzschild, v0.7b).
