---
tags: [coding, black-hole-sim, physics, grmhd]
created: 2026-11-17
status: done
---
# E3 — The σ cut of the GRMHD emission (jet funnel)

Plan: plan-refinement-phase.

## What and why
- **Magnetisation.** σ = b²/ρ, in code units (b² = b_μb^μ, as stored in the .bgm by `convert_grmhd.py` / `convert_athenak.py`).
- **Why the funnel is cut.** In the jet funnel of a MAD, ρ sits at the numerical floor of the simulation, so its density, and therefore its emission ∝ n_e, is not physical.
- **Common practice [summary of a web search, sources below].**
  - Imaging codes zero the emission where σ > σ_cut, typically 1.
  - EHT M87 Papers V and VIII use 1, and VIII also tested 10 with unchanged metric distributions.
  - Chael et al. 2019 used 25.
  - The cut introduces a discontinuity in the emissivity.
- **The flag.** `--grmhd-sigma-cut S`, default 0 = off, so the earlier tests and renders are unchanged. The UI sets **1** when GRMHD is picked (the EHT choice) and adds a slider (0–30) with a tooltip.

## Measured (selftest "E3 GRMHD sigma cut": θ = 17°, 48 × 48 rays over 16°, 230 GHz)
| Dump | σ > 1 cells | flux removed, cut 25 | cut 10 | cut 1 |
|---|---|---|---|---|
| Illinois MAD a = 0 (frame 5500) | 29 % | 6.7 % | 8.1 % | **32.8 %** |
| own AthenaK MAD a = 0.9375 (mad_00100) | 61 % | 15.1 % | 33.1 % | **70.4 %** |

- **Our earlier GRMHD images included 33–70 % of funnel emission.** The cut is a model choice that changes the image a lot; the UI badge for GRMHD already says "model-dependent".
- **The Illinois dump has σ capped near 100** (its σ_max floor); the AthenaK run reaches σ ≈ 3e9 near the axis.
- **Image.** `assets/e3-grmhd-sigma-cut-0-vs-1.png` (Illinois MAD, θ = 17°): the diffuse outer emission and the faint central blob disappear with cut 1.

## Tests
- `cut = 1e30` gives a total bit-identical to no cut.
- 0 < I(1) ≤ I(10) ≤ I(25) ≤ I(off), and the cut removes some emission of a MAD.
- **A test-hygiene bug found on the way.** The first version used whichever dump was on the GPU. The end of `selftest_v22f` loads the synthetic torus (σ ≤ 1 everywhere), so the cut looked like a no-op (0 of 1 012 520 samples with σ > 1). The test now loads the MAD itself (`BHS_GRMHD_TEST`, else the Illinois MAD).
- ctest 37/37.

## Sources [summary]
- Common σ_cut = 1, EHT Papers V and VIII, and Chael et al. 2019 at 25: search results including [arXiv:2604.11869](https://arxiv.org/pdf/2604.11869), [EHT VIII arXiv:2105.01173](https://arxiv.org/pdf/2105.01173) and [arXiv:2501.14862](https://arxiv.org/pdf/2501.14862).
- Not opened; confirm the exact wording before publishing.
