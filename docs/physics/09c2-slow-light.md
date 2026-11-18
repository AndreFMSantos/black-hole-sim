---
tags: [coding, black-hole-sim, cuda, physics]
created: 2026-11-12
---

# v0.9c2 — Slow light (propagation delay)

Closes the "instantaneous light" approximation declared in [09c-turbulence-hotspot](09c-turbulence-hotspot.md). Opt-in: `--slowlight` (with `--hr` and time). Without the flag, everything is as in v0.9c.

## Why it matters
Light takes ~`2r` M to cross the disk and the period at `r = 9` is 175 M: ~20 % of an orbit. For the turbulence it is irrelevant, but the apparent position of the spot and the **light curve of a flare** depend on the delay, which varies from point to point (the side nearest the camera is seen later).

## Model
| Piece | Description |
|---|---|
| **Flight time** | `dt/dMino = (r²+a²)(r²+a²−aL)/Δ + a(L − a sin²θ)` (E = 1). The ray is traced backwards (from the camera to the scene) and `T(s) = ∫ dt/dMino ds` is the flight time since the camera |
| **Integration** | steps far from the layer: Simpson `h/6 (f₀ + 4 f_m + f₁)` (with `r`, `σ = 1−μ²` by Hermite); inside the layer: midpoint per sub-sample (`T += ½ f dh` before and after the `add`) |
| **Emission time** | `t_emit = t_obs − (T − T_ref)`; the turbulence (`φ − Ω t_emit`), the spot phase and the flare use `t_emit` per sample |
| **Reference `T_ref`** | `r*(D) − r*(10)`, `r* = r + 2 ln(r/2 − 1)` (Schwarzschild): `t_obs = --time` is the instant at which light emitted at `r = 10` along the line of sight would arrive. It only fixes the origin of time; in Kerr it is a convention |
| **CPU** | `kerr::Trace::t_disk` (the time `t` is the 6th state variable of the RK45), oracle for the GPU |

## Oracles
Core (`test_slowlight.cpp`, 3 new cases, **105 in total**, 611 955 assertions):
1. **Schwarzschild, independent quadrature:** `t_disk` from `kerr::trace` vs `∫ dr / ((1−2/r) √(1 − b²(1−2/r)/r²))` (Simpson on 2e5 points, `b = √(Q+L²)`): 5 face-on rays (r_e from 4.7 to 7.3, D = 60) agree to **2e-7**. The oracle is only valid with monotonic radial motion: the test requires `r_e > r_periastro + 0.1` (a ray with `r_e` hugging the periastron, margin 0.02, failed by 0.8 % because the integrand has an integrable singularity there and Simpson's rule loses precision: an oracle error, not a code one).
2. Flat limit (D = 4000, r = 1500): `t_disk` = Euclidean distance to 5e-3, and `> ` the distance (the gravitational delay only increases the time).
3. `dt/dτ` reduces to `r³/(r−2)` at a = 0, and to `(r²+a²)²/Δ − a²` on the equator with L = 0.

GPU (`--selftest`, section `v0.9c2`, in `ctest`):
| Test | Result | Limit |
|---|---|---|
| GPU-accumulated `T` (where the transmittance drops to ½, thin opaque layer) vs CPU `t_disk`, a = 0, D = 60 | max relative error **5.4e-4** (174 rays) | 3e-3 |
| same a = 0.9, θ = 70° | **1.5e-3** (180 rays) | 3e-3 |
| **uniform flare** (giant spot, `--spot-flare 100,12,20`): the peak of each pixel at `t_f + t_flight(pixel) − T_ref`, with `t_flight` from the CPU; 22 pixels with offsets ≥ 6.2 M | max error **0.72 M** | 1 M |
| without slow light: the peak is at `t_f` in all pixels | 2e-5 | 1 M |
| without turbulence or spot slow light does not change the image | difference 0 | 1e-6 |

About the `T_half` deviation: with `H/r = 0.004` the error was 2.2e-3 (a = 0) and 1.2e-2 (a = 0.9, θ = 70°, 0.53 M) and did **not** change with 8 or 64 sub-samples. The cause is physical: `T_half` is the time where the transmittance drops to ½, ~`2.5 H` **above** the plane (whose distance along a ray inclined at 70° is `2.5 H / 0.34`); the CPU gives the time in the plane. With `H/r = 0.0005` the error drops ×8 (2.7e-4 and 1.5e-3), as predicted. A first hypothesis (sub-sample granularity) was discarded by this measurement.

## Cost
1280×720, 16 samples, `--hr 0.05 --turb 1.0 --spot …`: 13.82 ms/sample (instantaneous light) → **14.33** (slow light, +3.7 %). The cost is the `dt/dMino` per step, in float.

## Two errors that cost time
1. **Variable shadowing:** inside `VolAcc::add()` the local temperature was called `T`, the same as the member `T` (accumulated time), and `T_half` stored 3200 (the temperature) instead of 65 (the time). The render was right (it uses the member before the local declaration), so the flare oracle passed while the test hook failed with a ×38 error. Renamed to `Tobs`.
2. **Boolean flag ignored:** `is()` requires an argument after it, and `--slowlight` was the last argument; booleans use `flag()`. The symptom was two byte-identical images. Remember: flags without a value → `flag()`.

## Images
`assets/v09c2-slowlight.jpg`: `--a 0.9 --theta 80 --hr 0.05 --tau 3 --turb 1.0 --spot 8,60,1.0,12 --time 100` without (left) and with (right) `--slowlight`; mean difference 0.48/255, max 133.

## Limitations
- `T_ref` is a Schwarzschild convention; it only fixes the origin of time.
- Inside the layer `t_emit` varies per sample, but the layer is thin; the thin disk (`--hr 0`) has no time.
- The moving observer does not change the delay (it uses the camera's coordinate `t`).
- `T` in float accumulates an error ~1e-6 M, negligible.
