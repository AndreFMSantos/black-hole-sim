# 41 — Comparison with ipole (first pass, 2026-11-18)

Goal: compare our GRMHD radio images with an established code. **ipole** (Mościbrodzka & Gammie 2018; https://github.com/AFD-Illinois/ipole, commit `7f7a482`) reads the Illinois GRMHD library snapshots natively, so both codes can image the same file.

## Setup (identical on both sides)
- Snapshot: Illinois library v3, MAD a = 0, `torus.out0.05500.h5` (ours: the converted `data/bgm/MAD_a0_5500.bgm`).
- M = 6.5e9 Msun, D = 16.8 Mpc, M_unit = 1e25 g, 230 GHz, inclination 17°, phi = 0, camera at r = 1000 M, FOV 160 µas (41.88 M), 160 × 160 pixels, σ cut 1, r_max 100 M, unpolarised transfer (ipole `--only_unpolarized=1`).
- Electron temperature: **constant T_p/T_e = 3** in ipole (`trat_small = trat_large = 3`), which ipole turns into Θe = (m_p/m_e)·2/(3(2+R))·u/ρ (EHT formula, γe = 4/3, γp = 5/3); ours `--electrons prop --th-e-ratio 550.8458` reproduces exactly that Θe from our Θi = (γ−1)u/ρ (γ = 13/9).
- Ours in double, tol 1e-8. Commands and the comparison script: `data/cmp_ipole/` (`cmp.py`).

## Result (2026-11-18): agreement to 1.6 % in flux, image differences at the pixel-sampling level
| | ipole | ours | ours / ipole |
|---|---|---|---|
| total flux, 160² | 0.03799 Jy | 0.03741 Jy | 0.984 |
| total flux, ours at 640² (2×2 → 160²) | | 0.03776 Jy | 0.993 |
| image L1 (normalised), 160² | | | 0.065 |
| pixels above 10 % of the peak, ratio ours/ipole | | | p10 0.945, median 1.009, p90 1.083 |
| same, both images blurred 3×3 | | | L1 0.044; p10 0.953, p90 1.065 |

**The residual is sampling, not physics:** our own image at 160² against our own image at 640² averaged to 160² differs more (L1 0.084, p10–p90 0.84–1.11) than ours against ipole (L1 0.065). Both codes point-sample the pixel centres, and the MAD snapshot has structure finer than a 1 µas pixel (the traced pixel, ipole's brightest, is 0.38 of ours there: a sub-pixel filament). Orientation: ipole's image is ours transposed and flipped vertically (L1 0.075 against ≥ 0.6 for the other 7 orientations).

## How the first pass was wrong (a reference trap again)
The first pass gave ×2 in flux, not uniform. Investigation, in order:
1. **Point, emissivity:** our Dexter (2016) fit evaluated in Python with ipole's own n_e, Θe, B, ν (fluid frame) and angle along the traced ray: ours / ipole = 0.980 median (p5 0.862, p95 0.994) for `j_unpol·ν²`; absorption (Kirchhoff, α = j/B_ν) the same ratio for `k_unpol/ν`. Not the cause.
2. **Sampling along the ray:** flux 0.0188 / 0.01883 / 0.01883 Jy for `--grmhd-step` 0.05 / 0.01 / 0.005. Converged; not the cause.
3. **Point, plasma state:** our `.bgm` read in Python at ipole's trace points (nearest cell): n_e 0.991, B 1.003, β 0.989 (medians), but **Θe 0.754**. The snapshot's adiabatic index is **γ = 4/3**, not 13/9 as I had assumed when deriving `--th-e-ratio` (550.85); with γ = 4/3 the matching value is 734.461. Our code was right; my reference setting was wrong.
4. **σ cut:** along the traced ray only 1.6 % of the emission is at 0.9 < σ < 1, 0 % above: the cut boundary is not the cause.

## Our R_high electron model is not the EHT convention
Our `--electrons rhigh` uses Θe = Θi (m_p/m_e)/(1 + R) with Θi = (γ−1)u/ρ; ipole and the EHT GRRT formula list use Θe = (m_p/m_e)(γe−1)(γp−1)/((γp−1)+(γe−1)R)·u/ρ with γe = 4/3, γp = 5/3. At R = 20 ours is 0.52 of the EHT Θe for a γ = 4/3 snapshot and 0.70 for γ = 13/9. Both conventions are in the literature, but the EHT one is the standard: candidate change (it moves our M_unit calibration, so re-calibrate after).

## Reproduce
`code/tools/ipole/run_compare.sh [outdir]` with `IPOLE=<ipole binary>` (ipole commit `7f7a482`, built with its makefile; needs GSL and HDF5) and the snapshot + its `.bgm`; prints both fluxes and the image statistics (`compare_images.py`, numpy + h5py).
