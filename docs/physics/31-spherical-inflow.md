---
tags: [coding, black-hole-sim, physics]
created: 2026-11-17
status: done
---
# E1 — Spherical inflow ("Bondi-like"), Narayan, Johnson & Gammie 2019

Plan: plan-refinement-phase. App: `apps/v4_0-metric-api`, `--disk-model spherical` (volume model 4), and the UI disk type "Spherical inflow (bolometric)".

## Source and model
**Narayan, Johnson & Gammie 2019**, *The Shadow of a Spherically Accreting Black Hole*, ApJL 885, L33, arXiv:1910.02957 (read via ar5iv).
- Optically thin and **bolometric**. The gas-frame emitted power per unit volume is P(r) = 1/[(1 − 2/r)^{1/2} 4π r⁴] for r > 2, and j = P/4π.
- The paper uses the same P for the gas at rest and for the infalling gas.
- Flows:
  - **gas at rest** (static observers);
  - **free fall from infinity**, u_t = −1, so β = √(2/r) relative to static observers.
- The paper says the normalisation and functional form "are unimportant" for the shadow result.

Our implementation:
- **Transport.** I_obs = ∫ g⁴ j ds, where ds = E Σ dτ_Mino is the proper length in the gas frame (E = −k·u, with the affine dλ = Σ dτ). This follows from integrating the invariant I_ν/ν³ over frequency. The radio models use g³ j ds because they work at one observed frequency.
- **Kerr generalisation of the free fall.** u_t = −1, u_θ = u_φ = 0, as Papoutsis et al. 2023 do for their cones. This gives:
  - u^t = A/(ΣΔ) and u^φ = 2ar/(ΣΔ);
  - u^r = −√(2r(r² + a²))/Σ, from u·u = −1 and A − ΣΔ = 2r(r² + a²).
- **Gas at rest in Kerr.** u^t = 1/√(−g_tt). It does not exist inside the ergoregion, which emits nothing in this option.
- **Emissivity:**
  - `--sph-profile narayan` is the paper's exact profile (a = 0, r > 2);
  - the default is j = r^{-α}/(16π²), with α = 4 (the paper's leading behaviour) as a free parameter.
- **Inner cut.** The horizon by default (2 with the Narayan profile); `--sph-rin` overrides it.
- **Outer cut.** `--sph-rout` (200). The escape tail (O1) starts beyond 10 sph_rout.
- **Display.** False colour (afmhot) of log10 I in `--sph-range lo,hi` (default −7, −3), in code units with M = 1.
- **Combinations:**
  - Kerr only. Kerr-Newman is defined (neutral gas, the same free fall), but the volume tracer is Kerr-only; the combos table says so.
  - Not SdS: there is no infinity to fall from.
  - Not the wormhole.
- **UI badge.** Model-dependent: a toy model whose emissivity profile is chosen.

## Oracles (`--selftest`, section "E1 spherical inflow")
| Check | Result | Limit |
|---|---|---|
| b = 0, gas at rest, double: I vs host quadrature (closed form 1/(1536π²) = 6.596e-5; paper 6.60e-5) | 4.4e-7 | 1e-4 |
| b = 0, gas at rest, float | 1.6e-4 | 2e-3 |
| b = 0, free fall, double: I = 4.15856e-6 (paper 4.16e-6, about 16× darker) | 9.4e-7 | 1e-4 |
| b = 0, free fall, float | 1.4e-5 | 2e-3 |
| shadow edge (steepest rise of I(b)) vs √27, r_in = 2 | 1.3e-3 (grid 0.005) | 5e-3 |
| the same, r_in = 6 (no gas near the photon orbit, no peak, the edge still at √27) | 1.3e-3 | 5e-3 |
| gas at rest, luminosity at infinity 4π ∫ I 2πb db = 19/60 (whole image: lensing + redshift) | **3.0e-5** (0.31666) | 5e-3 |
| Kerr a = 0.9 free fall, 1681 rays: non-finite or non-positive | 0 | 0 |
| Kerr a = 0.9 float vs double | 8.2e-5 | 5e-3 |

- **Host reference.** Composite Simpson after r = 2 + s², which removes the (1 − 2/r)^{-1/2} singularity. It includes the factor (1 − 2/D)^{-2} of a static camera at D = 1e4.
- **First check of the model.** The b = 0 values were first checked against the paper with a throwaway script (`code/tools/narayan_b0.py`, `uv run --with scipy python code/tools/narayan_b0.py`): 6.596431e-5 and 4.156904e-6.
- **A mistake in my own test.** I first tested the *peak* of I(b) at √27. With r_in = 6 there is no peak there, because no gas sits near r = 3. What the paper states is that the *shadow edge* (the jump) stays at √27, and that is the test now.
- **Robustness.** The sweep (`tools/sweep_params.py --only spherical-inflow`, 76 sets including 20 random) flagged 0. The combos test includes the new kind. ctest 35/35.

## Cost
700² float, a = 0.9, D = 100: **6.5 ms/frame**, 73.7 steps per ray (step cap ds ≈ r/10 inside sph_rout).

## Image
`assets/e1-spherical-inflow-kerr-a09.png` (a = 0.9, θ = 60°, D = 100, FOV 24°): the dark region bounded by the bright photon ring, displaced sideways by the spin.

## Not done
- The Chael, Johnson & Lupsasca 2021 Kerr statement ("the brightness depression coincides with the rays that end on the horizon") is not a pixel test yet: only the abstract was read.
- The Michel (1972) thermodynamic profile with synchrotron (a radio variant) was not done; that paper has not been read.
