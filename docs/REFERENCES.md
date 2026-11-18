# References

Papers, codes and data used by the project, as recorded while working (the markers say how far each one was read: [read], [summary], [memory]). Papers are cited, never copied into the repository.


# black-hole-sim references (for the blog seeds)

Convention: **[read]** = the content was read/verified in this work session (page or PDF obtained); **[memory]** = citation from general knowledge, **not verified**, confirm before publishing; **[summary]** = only the summary of a web search, without opening the paper. Never cite as verified what is not marked [read].

## Data we will use
| Resource | Notes | Status |
|---|---|---|
| **Illinois GRMHD library (v3)**, Dhruv et al. 2025, ApJS 277, 16, [arXiv:2411.12647](https://arxiv.org/abs/2411.12647); data at http://thz.astro.illinois.edu/ (page `v3_grmhd.html`) | 10 ideal GRMHD simulations (MAD and SANE × spins −0.94, −0.5, 0, +0.5, +0.94), FMKS grid 288×128×128, 30 000 t_g, **only the 25 000–30 000 t_g times are hosted** (the rest "on reasonable request"). Licence **CC BY 4.0**, cite the paper. Download through the page (selection in the browser, no Globus or registration). The `v5` repository is "coming soon". **Not verified:** file format, variables, cadence, sizes (the file table only shows up with JavaScript) | [read] (subagent, 3 pages) |
| **Real snapshot inspected** (MAD, a = 0, frame 5500, t = 27 500 t_g) | `code/tools/fetch_grmhd.sh MAD 0 5500` (144 MB; the listing comes from `fetch_files_v3_grmhd.php`, the site's own JSON). **HDF5**, `prims` float32 (288, 128, 128, 8) = 151 038 200 bytes, `prim_names = RHO, UU, U1, U2, U3, B1, B2, B3`, `gam = 1.333333`, `coordinates = FMKS` (`base = SPHERICAL_KS`, `hslope 0.3`, `mks_smooth 0.5`, `poly_alpha 14`, `poly_xt 0.82`, `r_in = 1.772`, `r_out = 1000`, `startx1 = ln r_in`), `electrons = False` (no electron temperature: it is a model), `file_type = RST`. **1001 files per simulation** (frames 5000–6000, 5 t_g cadence), 144.04 MB each ⇒ **≈144 GB per simulation, ≈1.4 TB the library**. File sha256: `46f854ec…4cf3`. Stored in `data/illinois-grmhd-v3/` inside the project (outside Git) | [read] (real file) |
| **Dump conventions (pyharm, commit `fda4277`, read in the code by a subagent, nothing executed)** | **FMKS** (`coordinates.py:638-655`): `r = exp(X1)`, `φ = X3` (Kerr-Schild φ), `θ_g = π X2 + ((1−hslope)/2) sin(2π X2)`, `y = 2 X2 − 1`, `θ_j = poly_norm y (1 + (y/poly_xt)^poly_alpha/(poly_alpha+1)) + π/2`, `poly_norm = (π/2)/(1 + 1/(poly_alpha+1)/poly_xt^poly_alpha)`, `θ = θ_g + exp(mks_smooth (startx1 − X1)) (θ_j − θ_g)`; cell centre `X = startx + (i + 0.5) dx`. **Velocity** (`variables.py:150-169`): `γ = √(1 + g_ij ũ^i ũ^j)` with `ũ = (U1,U2,U3)` (FMKS basis), `u^0 = γ/α`, `u^i = ũ^i − γ α g^{0i}`, `α = 1/√(−g^{00})`. **Field** (`:171-183`): `b^0 = B^i u_i`, `b^i = (B^i + b^0 u^i)/u^0`, `b² = b_μ b^μ`, **`P_mag = b²/2` (Heaviside-Lorentz, no 4π)**, `β = P_g/P_mag`, `P_g = (Γ−1) UU`, `Θ = P_g/ρ`. **Units** (`units.py`): `L = G M/c²`, `ρ_unit = M_unit/L³`, `B_unit = c √(4π ρ_unit)`, `n_e = ρ ρ_unit/(m_p + m_e)`, `B_G = √(b²) B_unit`; `M_unit` is not fixed in the code ("fitted by imaging, e.g. with ipole"); `get_units_M87` (6.2e9 M☉) and `_SgrA` (4.14e6 M☉). **KS metric** (`:143-172`): `g_tt = −1 + 2r/ρ²`, `g_tr = 2r/ρ²`, `g_tφ = −2 a r sin²θ/ρ²`, `g_rr = 1 + 2r/ρ²`, `g_rφ = −a sin²θ (1 + 2r/ρ²)`, `g_θθ = ρ²`, `g_φφ = sin²θ(ρ² + a² sin²θ(1 + 2r/ρ²))`. KS→BL: `φ_BL = φ_KS − ∫ a/Δ dr`, `t_BL = t_KS − ∫ 2r/Δ dr` (implicit in the matrices of `coordinates.py:795-806`). **Not verified:** that the files' `U1..U3` are exactly `ũ^i` (pyharm assumes it, did not confirm it against the iharm3d C code) | [read] (public code) |
| **pyharm**, https://github.com/AFD-Illinois/pyharm | Python package the paper points to for analysing the data | [read] (mentioned in the paper) |
| **NASA star map** (SVS, starmap_2020_8k.exr) and **Gaia DR2** (ESA/Gaia/DPAC) | HDR background sky, already in use (v0.5b); see `notes/physics/05b-real-sky.md` | [read] |

## Physics: emission and transport
| Source | What for | Status |
|---|---|---|
| **Dexter 2016, MNRAS 462, 115**, [arXiv:1602.03184](https://arxiv.org/abs/1602.03184), Appendix A (grtrans) | **thermal** synchrotron emissivity: `j_I = n e² ν /(2√3 c Θ_e²) · I_I(x)`, `x = ν/ν_c`, `ν_c = (3/2) ν_B sinθ_B Θ_e²`, `ν_B = eB/(2π m c)` (eq. A12); fit `I_I(x) = 2.5651 (1 + 1.92 x^{-1/3} + 0.9977 x^{-2/3}) exp(−1.8899 x^{1/3})` (eq. A18). Stated accuracy: ~1 % against the integral, ~20 % against the full emissivities (symphony) for `Θ_e > 3` and `ν/ν_c > 1`. Absorption by Kirchhoff, `α = j/B_ν(T_e)`. The `K₂` normalisation does not appear (ultra-relativistic limit) | [read] (PDF + pdftotext, subagent) |
| **Mahadevan, Narayan & Yi 1996** | isotropic form with `K₂` and `M(x_M)`; the subagent read Sect. 4.1 (eqs. 16–18: constant `4.43e-30`, `M(x_M) = 4.0505/x_M^{1/6}(1 + 0.40/x_M^{1/4} + 0.5316/x_M^{1/2}) exp(−1.8899 x_M^{1/3})`) in a paper that is **not** the original MNY96 (it is astro-ph/9609107, Mahadevan 1997 type); **do not mix with Dexter** | [read] partial, secondary source |
| **Leung, Gammie & Noble 2011, ApJ 737, 21** (magnetobremsstrahlung emissivity/absorptivity), DOI 10.1088/0004-637X/737/1/21 | the j_ν fit with K₂(1/Θ_e) (eq. 72, section 7), `--emission leung`. PDF supplied by André and read (text of the PDF; figures 9–11 only from the captions) | [read] |
| **Shimura & Takahara 1995** | hardening factor: definition `F_ν = (1/f⁴) π B_ν(f T_eff)` (eq. 1) and f ≈ 1.8–2.0 near Eddington (v0.9a); **correction:** 1.5 was the old hypothesis; for ṁ < 1 % of Eddington no f fits. See `notes/physics/lit-verification.md` | [read] (pp. 1–6, scan) |
| **Eddington (limb darkening)** `I ∝ 1 + 2.06 μ` | v0.9a | [memory] |
| **Fishbone & Moncrief 1976, ApJ 207, 962** | equilibrium torus with constant `l` (v2.2b); the implementation was verified by independent hydrostatic equilibrium, **not** by comparison with the paper | [memory] |
| **Novikov & Thorne 1973**; **Page & Thorne 1974**; **Bardeen, Press & Teukolsky 1972** (ISCO) | relativistic thin disk | [memory] |
| **Wyman, Sloan & Shirley 2013** | analytic fit of the CIE 1931 functions (black-body colour) | [memory] |

## Existing tools (and how we will use them)
| Tool | Notes | Status |
|---|---|---|
| **KHARMA** (GRMHD on GPU) | the search shows a paper running ipole on KHARMA snapshots; I did **not** verify licence, dependencies or requirements | [summary] ([arXiv:2302.02733](https://arxiv.org/abs/2302.02733) is about ipole on GPU) |
| **AthenaK** (Kokkos, GRMHD and numerical relativity), [arXiv:2409.10384](https://arxiv.org/abs/2409.10384) | candidate public code to run our own simulation; licence not verified | [summary] |
| **ipole**, **grtrans** (polarised radiative transfer) | image comparison (not done yet: we do not have the data) | [summary]/[read] (grtrans via Dexter) |
| **Ozaki schemes** (FP64 emulation on tensor cores), [arXiv:2511.13778](https://arxiv.org/abs/2511.13778) | only for matrix multiplication; **not** applicable to the geodesic integrator | [summary] |
| **RT cores / OptiX for empty-space skipping**: Wald, Zellmann & Morrical 2021 (https://www.sci.utah.edu/~wald/Publications/2021/spaceskip/ss.pdf); Ganter & Manzke 2019 (https://ganterd.github.io/media/bvhdvr_authorcopy.pdf); reply on the NVIDIA forum ("small" gain for typical data) | argument of seed 18 and of the decision to use 3D textures + a min-max map instead of RT cores | [summary] |
| **James, von Tunzelmann, Franklin & Thorne 2015, CQG 32, 065001** ([arXiv:1502.03808](https://arxiv.org/abs/1502.03808); Gargantua/DNGR) | Interstellar technique: thin disk at a uniform T of 4500 K, frequency shift omitted for artistic reasons, spin 0.6, ray bundles with geodesic deviation, flare by PSF | [read] (by a subagent, arXiv text) |
| **James et al. 2015, *Visualizing Interstellar's Wormhole*, Am. J. Phys. 83, 486** ([arXiv:1502.03809](https://arxiv.org/abs/1502.03809)) | metric `r(l)`, film parameters (ρ, a, M, W), geodesics (appendix), camera and two skies; **the wormhole is not in 1502.03808** | [read] (by a subagent, eq. 1–7 and appendix) |
| Cunha & Herdeiro, [arXiv:1801.00860](https://arxiv.org/abs/1801.00860); Perlick & Tsupko, [arXiv:2105.07101](https://arxiv.org/abs/2105.07101); Hioki & Maeda, [arXiv:0904.3575](https://arxiv.org/abs/0904.3575); Perlick, Tsupko & Bisnovatyi-Kogan, [arXiv:1804.04898](https://arxiv.org/abs/1804.04898) (Schwarzschild-de Sitter, shadow) | reviews of shadows, naked Kerr, wormhole, SdS | [summary] (1804.04898 partly read by the subagent; the others only summaries) |
| Müller 2004, *Visual appearance of a Morris-Thorne wormhole*, Am. J. Phys. 72, 1045; Müller 2008, PRD 77, 044043; Müller & Grave, [arXiv:0904.4184](https://arxiv.org/abs/0904.4184) (**correction 2026-11-15:** this ID is the *Catalogue of Spacetimes*, a catalogue of metrics, and not a paper about wormholes; Müller 2004 is not on arXiv) | visual appearance of the wormhole (images and videos at https://www.vis.uni-stuttgart.de/forschung/wissenschaftliche-visualisierung/relativistische-visualisierung/mtvis) | [summary] (only the page and summaries) |
| Ferwerda et al. 1996 (SIGGRAPH); Kirk & O'Brien 2011 (ACM TOG 30(4):42); Reinhard et al. 2002 (SIGGRAPH); Fritz et al. 2011, [arXiv:1105.2822](https://arxiv.org/abs/1105.2822) (extinction towards the Galactic Centre) | human vision model in low light; Sgr A* invisible in the optical (A_V ≈ 30–44 mag) | [summary]/[memory] (only search results) |

## Running our own GRMHD (v2.2g; read by a subagent in the clones, nothing compiled)
| Source | Notes | Status |
|---|---|---|
| **KHARMA**, https://github.com/parthenon-hpc-lab/kharma (commit `3c8bd19`) | BSD-3; FP64 practically mandatory; problems `pars/tori_3d/mad.par`, `sane.par`; thesis at [arXiv:2408.01361](https://arxiv.org/pdf/2408.01361) (~35 Mzcps on an A100, 21 on a V100) | [read] (clone) / [summary] (speeds) |
| **AthenaK**, https://github.com/IAS-Astrophysics/athenak (commit `c5a0d7f`) | BSD-3; `-DAthena_SINGLE_PRECISION=ON`; Kokkos 4.7.2 with `BLACKWELL120`; `inputs/grmhd/gr_fm_torus_mad_8_4.athinput`; Stone et al. [arXiv:2409.16053](https://arxiv.org/abs/2409.16053) (~120 Mzcps on an A100) and [arXiv:2409.10384](https://arxiv.org/pdf/2409.10384) (laptop RTX 3070 ~1e6 zcps, another problem) | [read] (clone) / [summary] (speeds; the PDF exceeded the fetch limit) |
| **iharm3d**, https://github.com/AFD-Illinois/iharm3d | CPU/OpenMP code, predecessor of KHARMA; not examined | [summary] |

## Neural fields and volume compression (exploration v2.2f-a2)
| Source | What for | Status |
|---|---|---|
| **Müller, Evans, Schied & Keller 2022, "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"** (SIGGRAPH) | the *hash-grid* encoding used in the experiment `tools/exp/grmhd_neural_field_exp.py` (own PyTorch implementation, naive, not their code) | [memory] |
| **Sitzmann et al. 2020, SIREN** (networks with sinusoidal activations) | alternative neural field; not used | [memory] |
| **Volume-compression work with networks (e.g. neurcomp, fV-SRN)** | context; not read | [memory] |
| **Video/frame interpolation (RIFE, FILM)** | the idea of learned, self-supervised temporal interpolation, for the snapshots 5 t_g apart | [memory] |
| Own experiments, numbers in `notes/physics/22f2-neural-compression.md` | classical compression vs neural field; temporal reconstruction | [read] (measured in this session, 1 snapshot, only ρ) |

## Radio-mode calibration (v2.2b3)
| Source | What for | Status |
|---|---|---|
| **Leung, Gammie & Noble 2011, ApJ 737, 21** (magnetobremsstrahlung emissivity/absorptivity) | the j_ν fit with K₂(1/Θ_e), `--emission leung`. The paper was **not read at first** (IOP/ADS blocked; not found on arXiv); the formula came from Pandya et al. 2016. (Read later from the PDF André supplied; see the row above.) | [summary] (paper at the time) / [read] (the formula, in Pandya) |
| **Pandya, Zhang, Chandra & Gammie 2016, arXiv:1602.08749**, §4.4 eq. 24–25 | quotes the fit: `j = n e² √2 π ν_s /(3 K₂(1/Θ_e) c) (X^{1/2} + 2^{11/12} X^{1/6})² exp(−X^{1/3})`, `ν_s = (2/9) ν_c Θ_e² sin θ`; Kirchhoff for α. The PDF prints `K₂(1/Θ²)`: **typo confirmed** against eq. 72 of the paper (`K₂(1/Θ_e)`) | [read] (PDF text) |
| **Dexter 2016, MNRAS, arXiv:1602.03184**, appendix A | origin of `I_I(x)` (A18): corresponds to I(x_M) of Mahadevan, Narayan & Yi 1996, ultra-relativistic limit (K₂ → 2Θ²); ~1 % against the integral, ≲ 20 % against the full emissivities for Θ_e > 3 | [read] |
| **EHT 2019 Paper V (arXiv:1906.11242)** | total M87* flux at 1.3 mm = 0.5 Jy; library with M = 6.2e9, D = 16.9 Mpc; R_high = 1…160 (eq. 7–8); 𝓜 fitted to 0.5 Jy | [read] (PDF text) |
| **EHT 2019 Paper VI (arXiv:1906.11243)** and **Paper I (arXiv:1906.11238)** | M87*: M = 6.5 ± 0.7 × 10⁹ M☉, D = 16.8 Mpc, θ_g = 3.8 ± 0.4 µas; ring 42 ± 3 µas | [read] (abstracts) |
| **EHT 2022 Sgr A\* Paper I (arXiv:2311.08680)** and **Paper V (arXiv:2311.09478)** | ring 51.8 ± 2.3 µas; Paper V: M = 4.14e6 M☉, D = 8.127 kpc, flux 2.4 Jy at 230 GHz (𝓜 fitted to ≈2.4 Jy), R_low = 1, R_high = 1…160, MAD with i ≤ 30° promising; one-zone estimate: n_e ≈ 1e6 cm⁻³, B ≈ 29 G for 2.4 Jy | [read] (PDF/ar5iv, up to ~60 % of Paper V) |
| EHT Papers II–IV of Sgr A\*, IV of M87* | not read at first (read later: see the PDFs section) | — |

## Background skies (improving the sky and choosing a 2nd sky; research of 2026-11-13, by a subagent, nothing downloaded)
| Source | What it is | Licence / credits | Status |
|---|---|---|---|
| **NASA SVS Deep Star Maps 2020**, https://svs.gsfc.nasa.gov/4851/ | what we already use (8K); also **16K (423 MB), 32K (1.4 GB), 64K (3.8 GB)**; layers: stars, *Milky Way background* (diffuse light from unresolved stars), bright stars; half-float EXR, plate carrée, ICRF or galactic (the galactic ones were redone on 2021-01-04); Hipparcos-2 + Tycho-2 + Gaia DR2 | credits "NASA/GSFC Scientific Visualization Studio" and "ESA/Gaia/DPAC"; the page does not state a licence (NASA's general policy, commercial use allowed without implying sponsorship) | [read] (by the subagent; the 16K/32K/64K dimensions by naming convention, not verified) |
| **ESA/Gaia EDR3 colour map** | equirectangular 8-bit PNG up to 16000×8000, non-linear, saturated stars | CC BY-SA 3.0 IGO (share-alike) | [read]; weak for lensing (no HDR) |
| **ESO/Brunier Milky Way** (eso0932a) | 6000×3000 photo, TIFF; the 800-Mpixel original only on request | CC BY 4.0 ("ESO/S. Brunier") | [read]; resolution below the current one |
| **WISE equirectangular** (PIA15482, NASA/JPL-Caltech/UCLA) | false-colour infrared (3.4 µm cyan, 12 µm green, 22 µm red), ~19000×9500, 8 bit | NASA's general policy | [read]; file sizes inconsistent across sources |
| **Mellinger All-Sky Panorama 2.0** | 648 Mpixel FITS | **licence not verified** (historically: non-commercial only) | [summary] |
| **Gaia Sky** (ARI Heidelberg), Gaia DR3 catalogue | view of the sky from another position in the Galaxy, on-screen only; feasible only at ~1–2 kpc (parallaxes) | data CC BY-SA 3.0 IGO; Gaia Sky licence **not verified** | [summary] |
| Edenhofer et al. 2024, [arXiv:2308.01295](https://arxiv.org/abs/2308.01295) | 3D dust map up to 1.25 kpc | — | [summary] |

## Early phases (v0.1–v0.8) — all [memory]: not re-verified in this session, confirm before publishing
| Source | What for | Status |
|---|---|---|
| **James, von Tunzelmann, Franklin & Thorne 2015**, *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*, Class. Quantum Grav. 32, 065001 (DNGR) | design reference for the whole project (lensing, disk, camera) | [memory] |
| **Carter 1968**, Phys. Rev. 174, 1559 | separable constants of motion of Kerr geodesics (`Q`), used with Mino time in v0.6 | [memory] |
| **Dormand & Prince 1980**, J. Comput. Appl. Math. 6, 19 | adaptive Runge–Kutta 5(4) used in v0.6a/b | [memory] |
| **Bardeen, Press & Teukolsky 1972**, ApJ 178, 347 | ZAMO observer, ISCO and circular-orbit quantities in Kerr | [memory] |
| **Novikov & Thorne 1973**; **Page & Thorne 1974** | relativistic thin-disk temperature profile (v0.7a) | [memory] |
| **Narkowicz 2016** (ACES filmic fit) and **Reinhard et al. 2002** | tone mappers of v0.7d | [memory] |
| **NVIDIA OptiX 9.1 denoiser** documentation | optional denoiser of v0.8c (measured: not worth it here) | [memory] |

## Points to remember in the seeds
- The Illinois data are of a **hot, tenuous flow (RIAF)**: the image is **radio** (EHT), in false colour; it is not an "optical" disk like Interstellar's.
- The electron temperature **does not come** from the simulation: it is a model (Ti/Te ratio); it is declared.
- Calibrate with `M_unit` and `ν_obs` (e.g. 230 GHz) and show the image as brightness temperature.

## Skies downloaded (2026-11-15, by a subagent; files in `assets/sky/`, outside this repository's Git)
- **NASA SVS 4851 Deep Star Maps 2020** [read]: `starmap_2020_16k.exr` (443 485 587 B, 16384×8192, HALF RGB, ZIP) and `milkyway_2020_16k.exr` (434 041 232 B, same); separate layers `starmap`, `milkyway` (diffuse background), `hiptyc` (bright Hipparcos/Tycho stars), also in galactic coordinates (`_gal`, corrected on 2021-01-04). No pixel > 1.0 (maximum exactly 1.0 on the stars: **saturated, not real HDR**). R means: 16K 0.00217; 8K (what we use) 0.00497: **the 8K is ~2.3× brighter**; measured on renders of the same framing (mean 15.5 vs 10.5 in 8 bit; pixels > 60: 0.85 % vs 0.15 %): to match the brightness with the 16K use `--sky-exposure` ≈ 2.3× the value (default 2 → ~4.6). The cause of the difference was not verified; the subagent also did not verify whether starmap = sum of the layers. (The 16K files were later deleted; only the 8K stays.)
- **WISE PIA15482 hires** [read]: `PIA15482_hires.tif`, 328 194 416 B, 19000×9500, RGB 8 bit; projection and frame **not verified**; credit "NASA/JPL-Caltech/UCLA". Converted to `wise_8k.exr` with `tools/img_to_exr.py` (gain 0.869, mean 0.015). Now at `assets/sky/wise/wise_8k.exr` and loaded by the UI as the wormhole's 2nd sky.
- **Conditions:** SVS 4851: credit "NASA/Goddard Space Flight Center Scientific Visualization Studio" and ESA/Gaia/DPAC (Gaia DR2); points to NASA's reproduction guidelines (NASA content generally without copyright in the US; acknowledge NASA as the source; do not imply commercial endorsement). Only summaries of the pages were read, not the full texts; additional Gaia DR2 conditions beyond citation **not verified**.

## PDFs obtained on 2026-11-15 (`data/papers/`, outside this repository's Git; downloaded by a subagent, IDs verified in the arXiv API and title/stamp on the 1st page)
Status: **EHT, Mościbrodzka 2016, Shimura & Takahara, Fishbone & Moncrief and Perlick et al. 2018: directed reading done** (only the sections/equations used; results in `notes/physics/lit-verification.md`); **Kerr–Newman and Bardeen/Page–Thorne: used in v2.1e**. Mark [read] only what was really read.
| Folder / file | Reference |
|---|---|
| eht/sgra-II-calibration | EHT Sgr A* Paper II, arXiv:2311.08679 (30 p.; on arXiv "EHT and Multi-wavelength Observations, Data Processing, and Calibration") |
| eht/sgra-III-imaging | EHT Sgr A* Paper III, arXiv:2311.09479 (81 p.) |
| eht/sgra-IV-variability-mass | EHT Sgr A* Paper IV, arXiv:2311.08697 (65 p.) |
| eht/sgra-V-astrophysical-models | EHT Sgr A* Paper V, arXiv:2311.09478 (62 p.) |
| eht/m87-IV-imaging, m87-V-asymmetric-ring, m87-VI-shadow-mass | EHT M87* Papers IV–VI, arXiv:1906.11241 / 1906.11242 / 1906.11243 |
| kn-disk/pugliese-quevedo-ruffini-2013 | Pugliese, Quevedo & Ruffini 2013, equatorial circular orbits of neutral particles in Kerr-Newman, arXiv:1303.6250 (the agent called it 1304.2940 from memory: the verified ID is 1303.6250) |
| kn-disk/hackmann-xu-2013 | Hackmann & Xu 2013, charged particle in Kerr-Newman, arXiv:1304.2142 |
| kn-disk/guo-2024 | Guo, Huang, Liang et al. 2024, ApJ 975:237, Kerr-Newman image with a thin disk, arXiv:2411.07914 |
| kn-disk/hassani-2026 | Hassani, Nozari & Saghafi 2026, thin disks in rotating charged black holes with effective higher curvature, arXiv:2605.13797 (phenomenological KN deformation; α → 0 gives KN) |
| shadows/cunha-herdeiro-2018, perlick-tsupko-2022, perlick-tsupko-bk-2018 | reviews of shadows (1801.00860, 2105.07101) and the Kottler/SdS shadow (1804.04898) |
| shadows/mueller-grave-catalogue-of-spacetimes | arXiv:0904.4184 (catalogue of metrics) |
| shadows/moscibrodzka-falcke-shiokawa-2016 | Mościbrodzka, Falcke & Shiokawa 2016, A&A 586, A38, "GRMHD simulations of the jet in M87", arXiv:1510.07243 (contains R_high; authors = Mościbrodzka, Falcke, Shiokawa; **not** Dexter/Davelaar; definition verified in `lit-verification.md`) |
| classics/ | Bardeen, Press & Teukolsky 1972 (ApJ 178:347), Page & Thorne 1974 (ApJ 191:499), Fishbone & Moncrief 1976 (ApJ 207:962), Shimura & Takahara 1995 (ApJ 445:780): open ADS scans |

## Refinement phase (2026-11-17, see plan-refinement-phase)
| Source | What for | Status |
|---|---|---|
| Narayan, Johnson & Gammie 2019, ApJL 885, L33, *The Shadow of a Spherically Accreting Black Hole*, [arXiv:1910.02957](https://arxiv.org/abs/1910.02957) | E1 spherical accretion: P(r) = 1/[(1−2/r)^½ 4π r⁴], static vs free fall (u_t = −1); I(b=0) 6.60e-5 static, 4.16e-6 infall; shadow edge √27 for r_in 2–6; L∞ = 19/60 | [read] (ar5iv HTML via fetch) |
| Chael, Johnson & Lupsasca 2021, ApJ, *Observing the Inner Shadow of a Black Hole*, [arXiv:2106.00683](https://arxiv.org/abs/2106.00683) | E1 Kerr: spherical optically thin inflow ⇒ brightness depression = rays ending on the horizon; log divergence at the critical curve | [summary] (abstract only) |
| Papoutsis, Bauböck, Chang & Gammie 2023, ApJ, *Jets and Rings in Images of Spinning Black Holes*, [arXiv:2212.06281](https://arxiv.org/abs/2212.06281) | E2 dual-cone: j = C r^-5 [Gaussian ridges at θ_c, π−θ_c], w = 1/128, u_t = −1 infall, optically thin; ring shift ≈ 2a sin i | [read] (ar5iv HTML via fetch) |
| Broderick & Loeb 2009, [arXiv:0812.0366](https://arxiv.org/abs/0812.0366) | force-free jet with outflow, non-thermal leptons (later option for E2) | [summary] |
| Radio images inside jet funnels, semi-analytic GRMHD models (2024), [arXiv:2406.00657](https://arxiv.org/abs/2406.00657); Kawashima et al. 2021 jet-base model [arXiv:2009.08641](https://arxiv.org/abs/2009.08641) | richer jet options | [summary] |
| Gralla & Lupsasca 2020, PRD 101, 044032, *The Null Geodesics of the Kerr Exterior*, [arXiv:1910.12881](https://arxiv.org/abs/1910.12881); companion *Lensing by Kerr Black Holes* [arXiv:1910.12873](https://arxiv.org/abs/1910.12873) | O4 closed-form Kerr geodesics (Jacobi elliptic, Legendre integrals) | [summary] |
| Cárdenas-Avendaño, Lupsasca & Zhu 2023, PRD 107, 043030, AART, [arXiv:2211.07469](https://arxiv.org/abs/2211.07469), github.com/iaart/aart (MIT) | O4 analytic ray tracing with an adaptive grid | [summary] |
| relxill family: Dauser et al.; García et al.; [arXiv:1905.08012](https://arxiv.org/abs/1905.08012), [arXiv:1708.03205](https://arxiv.org/abs/1708.03205) (ray-traced corona emissivity profiles) | E4 lamppost corona | [summary] |
| Chan, Psaltis & Özel 2013, GRay, [arXiv:1303.5057](https://arxiv.org/abs/1303.5057); Pu et al. 2016, Odyssey, [arXiv:1601.02063](https://arxiv.org/abs/1601.02063) | GPU throughput reference (~1 ns per photon per step) | [summary] |
| Hairer, Nørsett & Wanner (DOP853); Hairer & Wanner vol. II §IV.2 (PI step control, Gustafsson) | O2/O3 | [memory] |
| EHT M87 Paper VIII, [arXiv:2105.01173](https://arxiv.org/abs/2105.01173); sensitivities of GRMHD images [arXiv:2604.11869](https://arxiv.org/abs/2604.11869); limb-brightened jet [arXiv:2501.14862](https://arxiv.org/abs/2501.14862) | E3 σ_cut: typically 1 (EHT V/VIII; 10 tested), 25 in Chael et al. 2019 | [summary] |
