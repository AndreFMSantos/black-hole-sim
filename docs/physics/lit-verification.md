# Verifying the code against the literature (2026-11-15)

Reading directed by subagents on the PDFs in `data/papers/` (only the relevant pages; it is **not** a full reading). Each row says whether what the code uses **matches** the paper, with the proof. Status: [read] = excerpt read in the PDF and compared.

| item | in the code | in the paper | result |
|---|---|---|---|
| **R_high** (`--electrons rhigh`) | `Ti/Te = (R_high β² + 1)/(1 + β²)` (`render.cu`, GRMHD mode) | Mościbrodzka, Falcke & Shiokawa 2016 (A&A 586, A38), eq. (1) p. 3: `Tp/Te = R_high b²/(1+b²) + R_low/(1+b²)`, `b = β/β_crit`, `β_crit = 1`, `R_low = 1` | **matches** (identical with R_low = 1) [read] |
| R_high values | we used 20 for M87* | Table 1: 1, 5, 10, 20, 40, 100 (R_low = 1) | **within the paper's range** [read] |
| adiabatic index | 13/9 (dump header, same as the AthenaK input) | §4.4 p. 10: γ = 13/9 in the GRMHD | **matches** [read] |
| Θ_e from the gas | `Θ_e = Θ (m_p/m_e)/(1 + Ti/Te)` | **does not appear** in this paper (the agent searched m_p, 1836, "mass ratio") | **derived by us**: `p = n k (Ti + Te)`, `ρ ≈ n m_p` ⇒ `Θ_e = Θ (m_p/m_e)/(1 + Ti/Te)`; coherent, but **without confirmation in the literature** |
| hardening factor | `modified_planck(ν, T, f) = B_ν(f T)/f⁴` (`disk_emission.hpp`) | Shimura & Takahara 1995, eq. (1), p. 780: `F_ν = (1/f⁴) π B_ν(f T_eff)` | **matches** [read] |
| value of f | `--fcol` (default 1 = pure black body; the notes said 1.5–2 [memory]) | f ≈ **1.8–2.0** near Eddington (α ~ 0.1), almost independent of R and M; ~1.7 only in the inner region at 10 % of Eddington; for ṁ < 1 % of Eddington **no f fits** the spectrum; 1.5 was the old hypothesis | **correct the note**: 1.5 is the old hypothesis; the computed range is 1.7–2.0; and the NT disk in black-body colour **does not** apply at low accretion rates |
| torus ℓ (F&M) | `ell_kepler = (r² − 2a√r + a²)/(r^{3/2} − 2√r + a)` (`torus.hpp`) | the agent read eq. (3.8) as `l = ±(M/r³)^{1/2}[r⁴ + r²a² − 2Mra² ∓ a(Mr)^{1/2}(r² − a²)]/[r² − 3Mr ± 2a(Mr)^{1/2}]` | **the scan's transcription is wrong**: it does not match Kerr's L/E (a = 0, r = 8: 4.525 vs 3.771; the denominator should be `r² − 2Mr` for a = 0). Our `ell_kepler` **matches** the L/E of Kerr circular orbits (Bardeen–Press–Teukolsky) computed separately (script of this session). **F&M (3.8) to be re-read directly from the image** |
| torus parametrisation | `l = l_K(r_max)`, density normalised at r_max | F&M use `l = κ^{1/2} l_in`, `1 < κ < 2`; minimum r_in (Fig. 1): 5.62 Schwarzschild, 2.74 co-rotating extremal Kerr | **equivalent** (the condition `l = l_K(r_max)` is the same seen from the pressure maximum: the agent's inference, coherent with our equilibrium verified by finite derivatives) |
| SdS shadow, static observer | `sin²α = 27 M² f(r_O)/(r_O²(1 − 9ΛM²))` (`sds.hpp`) | Perlick, Tsupko & Bisnovatyi-Kogan 2018, eq. (14) p. 4: `sin²θ = f(r_O)/[(1/(27m²) − Λ/3) r_O²]` | **identical** (algebra: `(1/(27m²) − Λ/3) r_O² = r_O²(1 − 9Λm²)/(27m²)`) [read]; the GPU already matches the analytic to 5e-7 |
| SdS comoving | **not implemented** | eq. (36) p. 6 (∓ signs per side of r = 3m), limits (38)–(40) | **available** as an oracle for when the comoving observer is implemented |
| "sky" beyond r_c (SdS) | direction (θ, φ) at 0.9999 r_c | the paper says rays going to r_H2 end on I⁺ and that the sky there is "bright" without discussing sources; a static observer never sees beyond r_H2 | **our "sky" at the cosmological horizon is our own convention**, not the paper's (confirms the v4 backlog warning) |


## EHT (Sgr A* II–V and M87* IV–VI), reading directed by a subagent (text extracted with pdftotext; `sgra-V` is the arXiv v1 of 17-Nov-2023, not the published one; the figures (per-model approval maps) were **not** read)
| number we use | in the paper | result |
|---|---|---|
| Sgr A* M = 4.14e6 M☉, D = 8127 pc | Paper V §2.1: **values adopted in the simulations** (Do+2019, GRAVITY 2019), θg = 5.03 µas, no uncertainty. The **EHT measures** θg = 4.8 +1.4/−0.7 µas; with D = 8.15 ± 0.15 kpc it gives M = 4.0 +1.1/−0.6 ×10⁶ M☉ (Paper IV) | **matches the simulations**; it is not the EHT measurement (make this clear in the captions) |
| Sgr A* ring 51.8 µas | 51.8 ± 2.3 µas (68 %) (Paper IV) | **matches** |
| Sgr A* flux 2.4 Jy | 2.4 ± 0.2 Jy (mean 213–229 GHz, Paper II Tab. 2); ALMA gain ±10 % | **matches** |
| M87* M = 6.5e9, D = 16.8 Mpc, ring 42 µas | Paper VI: M = 6.5 ± 0.2(stat) ± 0.7(sys) ×10⁹ M☉, D = 16.8 +0.8/−0.7 Mpc, θg = 3.8 ± 0.4 µas, ring **42 ± 3 µas** | **matches** |
| **M87* flux 0.5 Jy** | **0.5 Jy is the MODEL SCALE value of Paper V**, not the measurement: Paper IV measures the compact flux 0.66 +0.16/−0.10 Jy (1.2 Jy at arcsec scale); Paper VI quotes ~0.7 Jy | **difference to note**: we calibrate to 0.5 Jy (like Paper V), not to 0.66 |
| M87* library scale | Paper V uses M = 6.2e9 M☉, D = 16.9 Mpc (θg = 3.62 µas), not 6.5e9/16.8 (θg = 3.8 µas) | **5 % difference** in the angular scale; our µas use 3.82 µas/M |
| MAD a = 0.9375, i = 17°, R_high = 20 (our run) | M87* Paper V: MAD a* ∈ {−0.94, −0.5, 0, 0.5, 0.75, 0.94}, i = 12°, 17°, 22°…, R_high ∈ {1, 10, 20, 40, 80, 160}; the jet angle to the line of sight is 17°; MAD preferred over SANE, non-zero spin, low R_high disfavoured; Ṁ fitted to a constant flux | **within the library**, and the calibration method (mass scale to the flux) is the same |
| **ring diameter / scale** | Paper VI: factor α = d/θg ≈ **11.5** (MAD only ≈ **11.0–11.1**, SANE ≈ 12.0) measured on GRMHD images; the Kerr critical curve is 9.6–10.4 GM/c² | **our ring is 9.7–9.9 GM/c² (α ≈ 9.8)**: it coincides with the critical curve (37.4 µas) but sits **~10–12 % below** the EHT's α for MAD models. Possible causes, untested: the EHT measures on images blurred by the ~20 µas beam and with another definition of diameter; our electron model and a single snapshot |
| Sgr A* library and best bet | Paper V: no model passes the 11 constraints; "best-bet" = **MAD, R_high = 160, positive a* (0.5 or 0.94), i ≲ 30°**, Ṁ = (5.2–9.5)e-9 M☉/yr; variability favours SANE (MAD too variable) | **opportunity**: our MAD a = 0.9375 + R_high = 160 + i = 30° + 2.4 Jy is a "Sgr A* best bet" scenario |

### Consequences recorded in the plan
1. Measure the ring of our GRMHD **blurred with the EHT beam** (~20 µas) to see whether α approaches 11 (cheap, light CPU/GPU).
2. Document in the caption/UI that Sgr A* (4.14e6 M☉ and 8127 pc) are the **simulations'** values, and that M87* at 0.5 Jy is the model scale of Paper V.
3. "Sgr A* best bet" scenario with our dump (R_high = 160, i = 30°, 2.4 Jy); only render it when the UI asks for it.


## Ring test against the EHT α (2026-11-15)
The EHT calibrates the M87\* MAD models with d/θ_g ≈ 11.0–11.1 (Paper VI); our ring (radial-profile peak, REx method) is 9.7–9.9 GM/c², coinciding with the Kerr critical curve (9.79). Tests done (dump t = 10 000 M, θ = 17°, flux fitted to 0.5 Jy):
- **Blurring with the EHT beam** (FWHM 10 and 20 µas) **does not allow measuring** the ring by radial peak: the centre fills with light and the peak falls to the minimum radius of the search; the EHT uses model fits (crescents), not peaks.
- **Fit of an asymmetric Gaussian ring** (with and without 20 µas blur; own simple implementation, **not** the EHT's `xs-ringauss`): d = 7.6–8.9 M (29–34 µas), i.e. **even smaller** (α ≈ 8–9) than by the peak, and insensitive to the blur.
- **R_high dependence** (1, 20, 80, 160, flux always at 0.5 Jy): by the peak, 9.9 M in all; by the fit, 9.1 → 8.7 M (decreases with R_high). So R_high does not explain the difference from α ≈ 11.
**Conclusion:** the difference from the EHT α (~20–30 % with this fit) **remains unexplained** (candidates, untested: the EHT's exact fitting model (xs-ringauss) and its flux treatment; time average instead of a single snapshot; polarisation/Faraday; funnel emission; our Leung emissivity).
**Sgr A\* (MAD, R_high = 160, i = 30°; the EHT Paper V "best bet"):** calibrated to **2.402 Jy** with `M_unit = 3.546e17 g` (M = 4.14e6 M☉, D = 8127 pc, 5.03 µas/M) the ring measures **10.1 M = 50.8 µas against 51.8 ± 2.3 µas** from the EHT (radial peak): compatible. Scenario in the UI.


## Fishbone & Moncrief 1976, eq. (3.8) re-read directly from the page (2026-11-15)
Read on p. 968 of the PDF (image): l(r) = ±(M/r³)^½ [r⁴ + r²a² − 2Mra² ∓ a(Mr)^½(r²−a²)] / [r² − 3Mr ± 2a(Mr)^½]; the text says the denominator vanishes at the photon-orbit radius. Compared by script (`scratchpad/fm38.py`) with the L/E of Kerr circular orbits (Bardeen 1972): **it does not match**, not even in Schwarzschild (r = 6: 4.899 against 3.674 = r^{3/2}/(r−2M); for a = 0 the formula gives r^{3/2}/(r−3M)). Since the L/E of a circular orbit is finite at r_ph and the l of (3.8) diverges there, **(3.8) is not the L/E of circular geodesics**; I did not identify what it is (it may be a quantity of their formalism that I did not understand, or a misreading on my part). **Not adopted**: nothing in the code uses (3.8) (the search in the code found no references); the torus comes from AthenaK and the NT disk uses Page–Thorne/Bardeen, validated by independent oracles. Recorded as a point to be clarified, with no impact on the results.
