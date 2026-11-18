---
tags: [coding, black-hole-sim, physics]
created: 2026-11-17
status: done
---
# E2 — Jet walls: the dual-cone model of Papoutsis et al. 2023

Plan: plan-refinement-phase. App: `--disk-model cones` (volume model 5), UI type "Jet walls (dual cone)".

## Source and model (read via ar5iv)
Papoutsis, Bauböck, Chang & Gammie, *Jets and Rings in Images of Spinning Black Holes*, ApJ (2023), arXiv:2212.06281.
- **Emissivity.** j_ν = C r^{-α}[exp(−((θ − θ_c)/w)²) + exp(−((θ − (π − θ_c))/w)²)], with w = 1/128 rad and α = 5. That is a near-delta ridge on each cone wall.
- **Frequency independent** (near the synchrotron peak), so J = j/ν² is the invariant and I = ∫ g³ j ds, the same weight as the radio models.
- **Gas** in free fall: u_t = −1, u_θ = u_φ = 0. This is the same velocity as E1, and the code now shares `infall_E`.
- **Optically thin.**
- **Parameters.** `--cone-theta 30` (deg), `--cone-width 1/128` (rad), `--cone-alpha 5`, `--cone-rout 60`, display `--cone-range -6,-1` (log10 I, C = 1). Validated ranges: 0.5 < θ_c < 89.5, 1e-4 < w < 0.5, 0 < α ≤ 10.
- **Kerr only** (the volume tracer is Kerr-only); the combos table says so.

## Tracer change: an emitting surface at |μ| = μ_s
- **Why.** The thick layer is a Gaussian around μ = 0. The step cap (never skip the Gaussian) and the "relevant step" test (do the sub-sampling) were written for μ = 0.
- **Generalisation.** They now use the distance to the surface, ||μ| − μ_s|, plus the crossing of |μ| = μ_s. The old crossing of the equator is kept.
- **Old models are unchanged.** They return μ_s = 0, so ||μ| − 0| = |μ| gives the same values: the golden images and selftests are unchanged.
- **For the cones,** σ (the width in μ) = w sin θ_c, so the step cap keeps μ changes near the ridge below 0.6 of that width.

## Oracles (`--selftest`, section "E2 dual-cone jet walls")
| Check | Result | Limit |
|---|---|---|
| a = 0, i = 20° (observer inside the cone): ring on the critical curve b = √27, peak in 4 directions | 1.8e-3 (grid 0.002) | 4e-3 |
| a = 0, i = 80°: left/right mirror symmetry | 0 (exact) | 1e-6 |
| a = 0.5, i = 80°: no non-finite or negative values (4096 rays, float and double) | 0 | 0 |
| a = 0.5, i = 80°: total intensity float vs double | 1.7e-5 | 2e-3 |
| outside the cone, i = 80°: jet centreline (midpoint of the two wall ridges 12 M above the hole) at a = 0 | 9e-13 M | 5e-3 |
| the same at a = 0.9 minus a = 0 (paper: "exactly spin-independent") | **0** | 1e-2 |

- **Not tested: the ring shift ≈ 2a sin i.** Along the horizontal line through the image centre (β = 0) the near-critical rays orbit in the equatorial plane and never cross the walls at θ_c = 30°. There is no ring peak there to measure, the same lesson as the r_in = 6 case of E1. It would need a 2D ring fit; that shift is a property of the critical curve, already covered by the lensing tests.
- **Robustness.** Combos test (Kerr renders, others refused) and sweep (`jet-cones`, 76 sets) with 0 flagged; ctest 37/37.

## Cost
700² float, a = 0.5, i = 80°: **2.5 ms/frame**, 75.5 steps per ray.

## Image
`assets/e2-jet-cones-i80-i20.png` (a = 0.5, D = 100, FOV 12°, log range −8…−3):
- **left, i = 80°:** the two walls. The ring shows only near the cone bases, because equatorial near-critical rays never cross the walls.
- **right, i = 20° (inside the cone):** the thin ring around the inner shadow, which is what the paper describes for observers inside the cone.

## Limits (honest)
- These are jet *walls* near the hole, not an outflowing plume: no outflow velocity and no large-scale jet.
- **Richer options, after reading the papers:** Broderick & Loeb 2009 (a force-free jet with non-thermal electrons) and the 2024 semi-analytic funnel models. A GRMHD funnel already exists (E3, with the σ cut off).
