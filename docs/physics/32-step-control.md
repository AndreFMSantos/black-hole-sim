---
tags: [coding, black-hole-sim, performance]
created: 2026-11-17
status: done
---
# O2 — Step control on the way out (refinement phase)

Plan: plan-refinement-phase. Previous step: [30-analytic-escape-tail](30-analytic-escape-tail.md).

## Symptom (R0 counter)
The float integrator (tol 2e-5) rejected **34 % of its step attempts** (42 % before O1). Double (tol 1e-8) rejected 1.75 %.

## Diagnosis
1. **Instrumented which state component set the error on each rejection** (temporary atomic counters, since removed). The radius r caused 537 k of the 604 k rejections, half of them at r > 100.
2. **Step-by-step trace of one ray.** Once outgoing, the sequence was always "accept with err ≈ 0.56, then reject with err ≈ 1.8":
   ```
   r=104.3  h=0.002287 err=1.797  -> rejected
   r=104.3  h=0.001831 err=0.558  -> accepted
   r=128.8  h=0.001851 err=1.796  -> rejected
   r=128.8  h=0.001482 err=0.558  -> accepted
   ```
3. **Cause.** In Mino time r(τ) runs into a pole at finite τ (r ≈ 1/(τ_∞ − τ)), so the step that meets the tolerance shrinks like 1/r. Measured: h·r ≈ 0.19 at every accepted step. The controller (0.9 err^{-1/5}) only sees the current error. After a good step it proposes h × 1.01, while the next step needs h × 0.81, so every second attempt fails.
4. **Generic fixes do not help.** A Gustafsson/Hairer PI controller brought it to 31 %, and "no growth right after a rejection" to 34 %. The problem is a trend in the solution, not controller oscillation.

## Fix
After an accepted step on the way out (dr/dτ > 0), the proposal is also multiplied by min(1, r_prev/r), which keeps h·r instead of h. Acceptance (err ≤ 1) and the tolerances are unchanged, so accuracy does not depend on it.
- **Only in the loose-tolerance regime** (rtol ≥ 1e-6). In double the shrink per step is small and the plain controller keeps up (1.7 % rejections). There the scaling would cost +2.6 % accepted steps, so double is left bit-identical.

## Verification (`tests/step_control.py`, ctest `v4_0_step_control`; reference double tol 1e-11)
| float, 160 × 90 grid | before | after |
|---|---|---|
| Kerr: rejected attempts | 34.1 % | **3.27 %** |
| Kerr: accepted steps/ray | 31.7 | 31.4 |
| Kerr: sky dir p99 / median | 2.78e-4 / 9.7e-6 | 2.77e-4 / 1.0e-5 |
| KN: rejected attempts | (similar) | 3.89 % |
| KN: sky dir p99 | 3.07e-4 | 3.04e-4 |
| fate differences | 1 ray | 1 ray (the same near-critical ray) |

- **Correction (later the same day, [33-dop853](33-dop853.md) §4):** most of those float outliers were not intrinsic to near-critical rays. They came from the velocity reprojection at turning points. After that fix the KN float max is 1.7e-4, not 2.3e-2. The O2 conclusions (rejections, speed) are unchanged.
- **Maxima are not tested.** Near-critical rays (the photon ring) amplify the float error exponentially. One ray per grid shows it, with **both** controllers (KN max 2.3e-2 rad). I first wrote the test with maxima and it failed for that reason. Comparing the old binary with the new one on the same grid showed the problem was in my threshold, not in the change.
- **Overall.** ctest 36/36 serial; the golden float images pass the "not further from double" criterion; double unchanged.

## Speed (`tools/bench.py`, RTX 5080, against O1)
| Scenario | after O1 | after O2 | gain | total vs baseline |
|---|---|---|---|---|
| thin disk float 1080p | 4.51 ms | 3.92 ms | 1.15× | 1.78× (6.97 ms) |
| Kerr-Newman float 1080p | 4.49 ms | 3.81 ms | 1.18× | 1.90× (7.23 ms) |
| thick layer float 1080p | 12.59 ms | 11.31 ms | 1.11× | 1.98× (22.36 ms) |
| GRMHD MAD float 720p | 37.58 ms | 34.72 ms | 1.08× | 1.33× (46.30 ms) |
| torus float 700² | 13.78 ms | 13.41 ms | 1.03× | 1.31× |
| RIAF float 700² | 23.68 ms | 23.43 ms | 1.01× | 1.15× |
| thin disk double 1080p | 442.4 ms | 441.8 ms | 1.00× | 1.78× |

The volume models gain less, because their own per-model step caps (path length per step inside the gas) set the step there.

## Open
- The Ellis wormhole has its own tracer and was not checked for the same pattern.
- SdS rays end at 0.9999 r_c, where the pole structure differs; it is unchanged (2.8 ms).
