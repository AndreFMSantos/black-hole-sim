---
tags: [coding, black-hole-sim, cuda, physics, radio]
created: 2026-11-13
---

# v2.2b3 — Radio-mode calibration: flux in Jy, scale in µas, Leung fit

The radio mode ([22b-radio-torus](22b-radio-torus.md), [22f-b-grmhd-reader](22f-b-grmhd-reader.md)) produced false colour in T_b. It now measures **total flux (Jy)**, **intensity (Jy/µas²)** and **ring diameter (µas)** and fits the density to the observed flux, like the EHT (𝓜 adjusted to the mean flux).

## What was done
| Piece | Description |
|---|---|
| `--radio-cal --d-pc D [--source m87\|sgra] [--flux-target Jy --fit-ne0] [--radio-out f.pfm]` | renders the image in cgs intensity, converts to Jy and Jy/µas², fits the normalisation knob by log-log secant (torus: `ne0`; GRMHD: `M_unit`), 3 iterations, ~1.5 s at 256² |
| Angular scale | each pixel has the celestial coordinates (α, β) = (−L/sin θ_o, −p_θ) of the **ray** (constants of motion, E = 1), with no flat-field approximation; the pixel area is the Jacobian by finite differences; `dΩ = area · (GM/c²/D)²` |
| `--source m87` | M = 6.5e9 M☉, D = 16.8 Mpc, 0.5 Jy, ring 42 µas (EHT 2019 Papers VI, V, I) |
| `--source sgra` | M = 4.14e6 M☉, D = 8.127 kpc, 2.4 Jy, ring 51.8 µas (EHT 2022 Papers V, I) |
| `--emission dexter\|leung`, `--theta-e-min` | Dexter 2016 (default; ultra-relativistic) or **Leung et al. 2011** with `K₂(1/Θ_e)` (GPU: K₂ by Abramowitz & Stegun polynomials in float, `core`: `std::cyl_bessel_k`) |

Sources and reading status in `notes/references.md`. **The Leung 2011 paper was read** (PDF supplied by André) and confirms the implementation: eq. 72 with `K₂(1/Θ_e)` (the `K₂(1/Θ²)` typo of the Pandya PDF is confirmed), `ν_s = (2/9) ν_c Θ_e² sin θ`, `ν_c = eB/(2π m_e c)` (eq. 2), Kirchhoff for α (eq. 44). Declared validity: Θ_e ≳ 0.5 (abstract); the error becomes of order 1 for Θ_e ≃ (ν/(ν_c sin θ))^(-1/5) (rough estimate; ≈ 0.10–0.21 for B = 1–30 G at 230 GHz); below γ ~ 1 cyclotron streaks appear that the fit does not reproduce. The paper is a general numerical scheme (`harmony` code) and only eq. 72 is the thermal fit: there are no tables or angle-dependent coefficients (those came from other fits).

## Oracles
| Test | Result |
|---|---|
| `|(α, β)|` = `r sin α_loc / √(1−2/r)` (a = 0, 1024 pixels) | 4.6e-16 |
| Thin limit: `F(2M)/F(M)` | 8.000 (expected `M³`) |
| `F(2D)/F(D)` | 0.250 |
| Leung: j and α GPU vs core (polynomial K₂ vs `cyl_bessel_k`) | 6e-10 |
| core: `K₂(1/Θ) → 2Θ²`; ultra-relativistic limit of the fit; Kirchhoff | pass |
| core: Leung/Dexter at Θ_e = 10, x = ν/ν_c ∈ [3, 300] | ratio between 0.5 and 2 (different constructions) |
| Byte-identical regression (`ctest`) and full selftest | OK |

## Results
Scale: M87* θ_g = 3.819 µas (paper: 3.8 ± 0.4); Sgr A* 5.03 µas (paper, Table 1: 4.8 +1.4/−0.7). Flux fitted to < 0.1 %.

| Model | Source | Fitted normalisation | peak T_b | Ring (r ≤ 8 M) |
|---|---|---|---|---|
| analytic torus (a = 0.9375, i = 17°, β = 3, Θ_e,max = 20) | M87* | ne0 = 1.3e4 cm⁻³ | 3.5e9 K | 60.7 µas (not a ring; see below) |
| same | Sgr A* | **ne0 = 5.2e5 cm⁻³** | 6.5e9 K | 24.6 µas (same) |
| **Illinois GRMHD MAD a = 0, frame 5500**, R_high = 20, i = 17°, Leung | M87* | M_unit = 4.7e25 g | 2.7e10 K | **46.2 µas** (EHT 42 ± 3) |
| same, i = 60° | M87* | M_unit = 3.7e25 g | 5.5e10 K | 35.5 µas |
| same, i = 17° / 30° | Sgr A* | M_unit = 1.3e18 g | 3.4–3.9e10 K | 65.9 / 44.8 µas (EHT 51.8 ± 2.3) |

- **Independent order-of-magnitude check:** the torus `ne0` fitted for Sgr A* (5.2e5 cm⁻³) is of the same order as an EHT one-zone estimate (n_e ≈ 1e6 cm⁻³ for 2.4 Jy; Paper V, §II.2). It is not the same definition of density (torus peak vs typical value), so only the order of magnitude counts.
- **Leung vs Dexter and the Θ_e cut** (MAD a = 0 frame, M87*, i = 17°, `M_unit` fitted): R_high = 1, 5, 20 give differences of 0.8 %, 0.8 % and 0.4 % in `M_unit`; peak T_b identical to 0.1–0.3 %. Varying `--theta-e-min` from 0.1 to 1.0 **changes nothing** (4 digits; only at Θ_e,min = 3 and 10 do 0.0 % and 0.3 % appear): the emission at 230 GHz all comes from Θ_e ≫ 1 in this frame, even with R_high = 1, so the 0.3 cut (below the declared 0.5) is harmless here. It may matter in other frames/frequencies or with colder electrons (not tested).
- **The ring diameter is indicative, not a test:** it is the radius of the maximum of the mean radial profile (around the flux centroid) in a **single turbulent snapshot**, and varies from 35 to 66 µas with the inclination and the frame. The EHT measures it with fits to averaged images. The analytic torus has no dominant photon ring (the outer emission dominates), so the value has no meaning.
- **Brightness temperature:** the peak T_b (1e10–5e10 K) are above what is usually quoted for the EHT (~1e10 K or less in the final images, blurred to the telescope's resolution); here the image is at the integrator's resolution, unblurred. Not compared with published values: pending.

## Limitations
- One snapshot (a = 0, R_high = 20 fixed); the real EHT calibration is over time series and a library of models. M_unit **changes with the frame** (the flux fluctuates).
- **Updated 2026-11-19 (`ṁ` & EHT Papers II–IV):** from our AthenaK `10⁴ M` MAD history (`torus.user.hst`, `t ∈ [5000, 10000] M`), `⟨Φ/√ṁ⟩ = 14.94` (HL) = `53.0` (Gaussian), giving `Ṁ = 7.21e-5 M☉/yr` (`5.0e-7 Ṁ_Edd`) for M87* (`M_unit = 8.589e24 g`) and `6.59e-9 M☉/yr` for Sgr A* (`M_unit = 5.003e17 g`, inside EHT Paper V best-bet `(5.2–9.5)e-9 M☉/yr`). EHT Sgr A* Papers II–V and M87* IV–VI, VIII have been read (`lit-verification.md`).

## Commands
```bash
v3_1-kerr-cuda-ui --radio-cal --grmhd MAD_a0_5500.bgm --disk-model grmhd --electrons rhigh --rhigh 20 --dist 150 --fov 14 --w 320 --h 320 \
  --prec float --samples 2 --theta 17 --source m87 --fit-ne0 --munit 1e26 --emission leung --radio-out m87.pfm
```
