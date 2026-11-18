---
tags: [coding, black-hole-sim, grmhd, data]
created: 2026-11-12
---

# v2.2f-a — The real Illinois GRMHD snapshot: format, conventions and reader plan

André's option 1 (2026-11-12): explore the real GRMHD data before plugging it into the render. Sources and verification status in [references](../REFERENCES.md). Current radio mode: [22b-radio-torus](22b-radio-torus.md).

## What was done
1. The data page (`thz.astro.illinois.edu/v3_grmhd.html`) lists the files through JavaScript; the listing JSON (`fetch_files_v3_grmhd.php?flux=MAD&spin=0`) gave **1001 files per simulation, 144.04 MB each (≈144 GB per simulation)**. I downloaded **just one** (MAD, a = 0, frame 5500, 151 038 200 bytes) with `code/tools/fetch_grmhd.sh MAD 0 5500` into `data/illinois-grmhd-v3/MAD_a0/` inside the project (outside Git). The whole library is ≈1.4 TB: I will **not** download it; the plan is to work with one or a few snapshots.
2. Read with `h5py` in an isolated Python (`-I`), treating the file as untrusted data.

## The file (read, not inferred)
- HDF5: `prims` `float32` **(288, 128, 128, 8)** (r, θ, φ, variable); `header/prim_names = RHO, UU, U1, U2, U3, B1, B2, B3`; `gam = 1.333333`; **`electrons = False`** (the electron temperature does not exist: it is a model); `coordinates = FMKS`, `base = SPHERICAL_KS`; `file_type = RST`; `t = 27 500.0000`; `a = 0`; `r_eh = 2`, `r_in = 1.772`, `r_out = 1000`, `startx1 = ln r_in = 0.5722`, `dx1 = 0.02200`, `dx2 = 0.0078125`, `dx3 = 0.04909`; `hslope 0.3`, `mks_smooth 0.5`, `poly_alpha 14`, `poly_xt 0.82`.
- Values (finite): `RHO` 3.7e-7 … 14.1; `UU` 3.5e-12 … 7.6; `U1` −0.19 … 0.47; `U3` −0.07 … 0.51; `B1` ±1.3.
- Radial profile (shell mean, `Θ_i = (Γ−1) UU/ρ` weighted by ρ): `r = 2`: 0.178; 3: 0.072; 5: 0.033; 8: 0.019; 12: 0.013; 20: 0.008; 40: 0.004; 100: 0.0015. **The real gas is considerably hotter than the analytic torus** (`Θ_i,max = 0.0056`) and the maximum `ρ` (14.1) is next to the horizon (r = 1.79).

## Conventions (pyharm, read in the code by a subagent; see `notes/references.md`)
- **FMKS:** `r = exp(X1)`, `φ_KS = X3`, `θ = θ_g + exp(mks_smooth (startx1 − X1)) (θ_j − θ_g)` with `θ_g = π X2 + ((1−hslope)/2) sin(2π X2)` and `θ_j = poly_norm y (1 + (y/poly_xt)^poly_alpha/(poly_alpha+1)) + π/2`, `y = 2 X2 − 1`; cell centre `X = startx + (i + ½) dx`.
- **Velocity:** `γ = √(1 + g_ij ũ^i ũ^j)` (FMKS basis), `u^0 = γ/α`, `u^i = ũ^i − γ α g^{0i}`.
- **Field:** `b^0 = B^i u_i`, `b^i = (B^i + b^0 u^i)/u^0`, `b² = b_μ b^μ`, **`P_mag = b²/2` (Heaviside-Lorentz)**, `β = P_g/P_mag`, `P_g = (Γ−1) UU`.
- **Units:** `L = GM/c²`, `ρ_unit = M_unit/L³`, `B_unit = c √(4π ρ_unit)`, `n_e = ρ ρ_unit/(m_p + m_e)`, `B_G = √(b²) B_unit`. **`M_unit` is a free parameter** ("fitted by imaging").
- **Not verified:** that the files' `U1..U3` really are `ũ^i` (pyharm assumes it, did not confirm it in the C code).

## Reader plan (v2.2f-b), with oracles
1. **Load** `prims` to the GPU (151 MB, float) and sample with trilinear interpolation in `(X1, X2, X3)` (hardware 3D texture; the `r,θ → X1,X2` map needs to invert FMKS by Newton in `θ`).
2. **Coordinates:** the integrator is in Boyer-Lindquist (`t_BL, φ_BL`), the data in KS (`φ_KS = φ_BL + ∫ a/Δ dr`): convert `φ` and the velocity. With **a = 0** (the first snapshot) KS and BL coincide in `r, θ, φ`, which gives a first step **without this complication**. Oracle: `a = 0` reproduces the Schwarzschild integrator.
3. **Fluid and field:** `u^μ`, `b^μ` and `b²` by the formulas above (native KS metric); go to `Θ_e` through a model (e.g. `Θ_e = Θ_i (m_p/m_e)/(1 + Ti/Te)` or the `R_high/R_low` of the EHT models) and `ne = ρ ρ_unit/(m_p + m_e)`.
4. **Emission and transport:** those of the radio mode (v2.2b); the angle `θ_B` goes from "toroidal field" to `cos θ_B = b·k/(|b| E)` with the real `b^μ` (in the fluid frame).
5. **Planned oracles:** (a) the sampled density reproduces a point chosen directly from the file (coordinate round trip); (b) `u·u = −1` and `b·u = 0` with the KS metric; (c) mean `β` and `σ` within the published range (to confirm in the paper); (d) the Fishbone-Moncrief torus as the reader's test case (the `prims` of an initial condition must reproduce the analytic density); (e) the image of a snapshot does not depend on the camera azimuth except statistically (there is no axisymmetry: it is a test that φ is not swapped).
6. **Risks:** convention of `U1..U3` not verified; `M_unit` and the electron rule are free, so the image **is not calibrated**; one snapshot is a turbulent state, not an average; 144 MB per snapshot and the temporal interpolation for video need several (≈ 144 MB × N).

None of the reader is implemented yet: this step only inspected and documented.
