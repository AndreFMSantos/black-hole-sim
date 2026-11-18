---
tags: [coding, black-hole-sim, performance, physics]
created: 2026-11-17
status: done
---
# O1 — Analytic escape tail (refinement phase)

Plan: plan-refinement-phase. App: `apps/v4_0-metric-api` (`kerr_cuda.cuh`, `escape_tail`).

## The problem (measured)
An escaping ray was integrated until r_far = 1e6. In Mino time r(τ) ends in a pole: r → ∞ at a finite τ_∞. The adaptive step can only cover a fixed fraction of the distance to that pole per step, so the number of steps grows like log(r_far / r).
- On the default scene (Kerr a = 0.9, θ = 80°, D = 40) **about 45 % of the steps of an escaping ray** were spent between r ≈ 1e3 and 1e6:
  - float: 56.9 steps per ray, 31.7 when stopping at 1e3;
  - double: 246.6 steps per ray, 131.6 when stopping at 1e3.
- Stopping early without a correction is wrong. The position angles at finite r are not the asymptotic direction (error O(b/r)): at r = 1e3, 76 % of the pixels changed, and the error reached 2.3e-2 rad (`tools/raycmp.py`).
- Even r_far = 1e6 carried that error: the double exit direction was off by 1.6e-5 rad (median), which was the whole double "precision" of the sky.

## The method
For an E = 1 photon there is at most one radial turning point outside the horizon. An outgoing ray (dr/dτ > 0) therefore never turns again, and its future radial motion is fixed. Carter separability (the same structure the integrator uses) splits the remaining evolution:
- **Remaining Mino time:** T = ∫_{r_t}^∞ dr/√R(r).
- **Radial part of φ:** Φ_r = −∫_{r_t}^∞ g(r) dr/√R(r), with g = a(r² + a² − aL)/Δ − a = a(2r − q² − aL)/Δ.

With u = 1/r both become smooth integrals on [0, 1/r_t]:
- dr/√R = du/√P(u), with P(u) = R u⁴ = (1 + (a² − aL)u²)² − K u²(1 − 2u + (a² + q²)u²), and P(0) = 1;
- g dr/√R = a u(2 − (q² + aL)u)/(1 − 2u + (a² + q²)u²) · du/√P.

They are evaluated with **8-point Gauss-Legendre**.
- **Angular part.** μ, dμ/dτ, σ = 1 − μ² and the angular part of φ (dφ = −L/σ dτ) advance by T with the angular equations alone. This uses an adaptive DP45 with the same tolerances, the same pole-crossing step cap and the same reprojection onto Θ(μ) as the main loop.
- **Scope.** Kerr and Kerr-Newman. SdS rays already end at 0.9999 r_c, and the Ellis wormhole has its own tracer (left as is; it could get the same treatment).
- **Where it starts.** `--rtail R` (default 1000). The kernel raises it to the largest of 10 rout, 40 torus rmax (the torus is sampled up to 4 rmax), 10 riaf_rout and 10 grmhd_rcap, so no emitting region is ever skipped. `--rtail 0` restores the old behaviour.

## Verification (`tests/escape_tail.py`, ctest `v4_0_escape_tail`)
The reference is double, tol 1e-11, tail off, integrated to r_far = 1e10 (its own finite-r error is ~1e-9 rad). Grid 128 × 72, three cases.

| Case | quantity | double tail | double no tail (old default) | float tail | float no tail |
|---|---|---|---|---|---|
| Kerr a = 0.9 | dir max | 1.26e-6 | 2.41e-5 | 7.00e-4 | 7.20e-4 |
| | dir median | **1.0e-9** | 1.65e-5 | 9.9e-6 | 1.2e-5 |
| | steps/ray | 131.5 | 246.5 | 31.6 | 56.9 |
| KN a = 0.5, Q = 0.5 | dir median | 1.0e-9 | 1.64e-5 | 9.7e-6 | 1.2e-5 |
| Kerr a = 0.7, D = 500 | dir median | 2.9e-9 | 2.04e-5 | 1.1e-5 | 1.4e-5 |

- **The tail itself.** At tol 1e-11 with the tail, the maximum error is **2.4e-9 to 3.5e-9 rad** against the 1e10 reference. A separate check of the 1e8 integrated reference against the 1e10 one showed 3.5e-7, which is exactly its O(b/r_far) error. The ~1e-6 maxima left in "double tail" are the tol 1e-8 integration error of the near part, not the tail.
- **No ray changes fate** (escaped, captured, disk hit) in any case.
- **Updated tests:**
  - the CPU-vs-GPU equivalence selftests (both stop at the same r_far) now run with `r_tail = 0`;
  - the golden images `schwarzschild-shadow-double` and `kerr-nt-sky-double` moved to `images_v4/`, because the stars move by a fraction of a pixel (max 8 levels, 2200 px at 256 × 144); the float twins pass the "not further from double" criterion;
  - ctest 35/35.

## Speed (`tools/bench.py`, RTX 5080, 5 runs)
| Scenario | before | after | gain |
|---|---|---|---|
| thin disk float 1080p | 6.97 ms | 4.51 ms | 1.55× |
| thin disk double 1080p | 787.6 ms | 442.4 ms | 1.78× |
| thick layer float 1080p | 22.36 ms | 12.59 ms | 1.78× |
| radio torus float 700² | 17.59 ms | 13.78 ms | 1.28× |
| RIAF float 700² | 26.95 ms | 23.68 ms | 1.14× |
| GRMHD MAD float 720p | 46.30 ms | 37.58 ms | 1.23× |
| Kerr-Newman float 1080p | 7.23 ms | 4.49 ms | 1.61× |
| SdS / wormhole | unchanged | | 1.00× |

## Also found (R0)
- In float, **42 % of the step attempts are rejected** (0.9 % in double). That is O2's target (the step controller).
- The "warp efficiency" counter only counts accepted steps, so it does not see rejection divergence.
- `ctest -j 4` can fail `v4_0_golden` through GPU contention; run the suite serially (35/35 in 142 s).
