---
tags: [coding, black-hole-sim, grmhd, research]
created: 2026-11-12
---

# v2.2g — Running our own GRMHD on an RTX 5080: what the sources say

André's option 3 (2026-11-12). Research done by a subagent that **read the code** (clones in `/tmp`, nothing compiled or executed) and the web. Everything marked "inference" is not measured. Continues from [22f-b-grmhd-reader](22f-b-grmhd-reader.md); sources in [references](../REFERENCES.md).

## Summary
| | KHARMA | AthenaK |
|---|---|---|
| URL | github.com/parthenon-hpc-lab/kharma (the AFD-Illinois URL redirects) | github.com/IAS-Astrophysics/athenak |
| Licence | BSD-3 | BSD-3 |
| Precision | **FP64 practically mandatory** (`GReal` fixed to `double`, comment "lots of work for Real != double") | **`-DAthena_SINGLE_PRECISION=ON`** exists (`CMakeLists.txt:34`); double by default; **no source says the GRMHD torus was validated in float** |
| Kokkos / Blackwell | the stable build with an old Kokkos "may not support" Blackwell; `scripts/update_kokkos.sh` bumps to 4.7.04 (the latest compatible) | Kokkos 4.7.2 pinned, which has `BLACKWELL120` (sm_120) and CUDA 13 support in the CHANGELOG |
| Ready-made problem | `pars/tori_3d/mad.par`, `sane.par` (FMKS, `r_out = 1000`, torus `rin = 20, rmax = 41`, `beta_min = 100`, 192×128×128 in MAD, `hdf5` with configurable `dt`, `single_precision_output`) | `inputs/grmhd/gr_fm_torus_mad_8_4.athinput` and `sane`; Cartesian Kerr-Schild with excision, 32³ root mesh with 9 levels of static refinement, `plm`/`hlle`/`rk2`, `cfl = 0.3`, `tlim = 10000`, binary output |
| Published speed | ~21 Mzcps (V100), ~35 Mzcps (A100) at 128³ (KHARMA thesis, arXiv:2408.01361); "a few days for 10 000 t_g" at ~256³ on several GPU nodes | ~120 Mzcps on an A100 (Stone et al., arXiv:2409.16053, via a research summary); in a dynamical-spacetime test on a laptop RTX 3070 ~1e6 zcps (~10× slower) |
| 288×128×128 up to t = 10 000 on one GPU | **not stated** | **not stated** |
| Output cadence | `dt` (e.g. 0.5 t_g) | `dt` |

## Feasibility on the RTX 5080 (inference, not measured)
- **Compiling:** AthenaK looks feasible with `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_BLACKWELL120=ON`. KHARMA needs `update_kokkos.sh` (a patch the script references was not in the listing: check).
- **FP64 is the bottleneck:** the card has FP64 at 1/64 of FP32 (~0.9 TFLOP/s, against ~9.7 on an A100). No source talks about FP64 on consumer GPUs for these codes. A guess, broad and unverified: a few Mzcps in KHARMA (5–10× below the A100), which would give **weeks** for 10 000 t_g (with ~1e6 time steps, also a guess).
- **AthenaK in float** would avoid the FP64, but it is exactly the regime (near the horizon, high spin, density floor 1e-8) where float rounding and primitive recovery can fail. It is an **experiment**, not a result.
- **Memory:** 16 GB seem enough for ~3 M zones (bytes per zone not stated).
- **Risks:** MADs need t ≳ 5000 t_g to saturate; fine disk output is large; both codes assume HPC and there are no results on consumer Blackwell.

## Not stated nor verified
Blackwell/sm_120 in the projects' documentation; any result on an RTX 50 series; a guide to FP64/FP32 on consumer GPUs; float validation of the AthenaK torus; number of zones and times for 288×128×128 up to 10 000 t_g; memory per zone.

## Measurement plan (executed, see below)
1. Compile AthenaK with `BLACKWELL120` (CUDA 13.4 in `/opt/cuda`, which already compiles this project with GCC 16.2).
2. Run a **small test** of the torus and measure **Mzcps in double and in float** on this card.
3. Only with that number estimate the cost of a whole simulation.

## Measurement (2026-11-13, RTX 5080, AthenaK c5a0d7f, CUDA 13.4, GCC 16.2)

Compilation: `cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_BLACKWELL120=ON -DPROBLEM=fluids/gr_torus` (`gr_torus` is a *user problem*: without `-DPROBLEM` the executable starts but fails with "Problem generator name could not be found"). `make -j6` took ~25 min.

Input `gr_fm_torus_mad_8_4.athinput` reduced to 4 refinement levels (copy in `data/athenak/run_d/small.athinput`, `pgen_name = gr_torus`), 200–300 cycles:

| mesh | blocks | cells | Mzcps (double) |
|---|---|---|---|
| base 32, 8³ blocks | 288 | 147 thousand | 3.19 |
| base 64, 16³ blocks | 288 | 1.18 M | **8.35** |

- The original input (9 levels, 4096 blocks of 8³) **does not fit** in 16 GB in double: with `nghost=4` each 8³ block has 16³ cells with ghosts (ratio 8×) and it fails to allocate 9.8 GiB (`prim`).
- **Float not measured**: `-DAthena_SINGLE_PRECISION=ON` does not compile upstream. I fixed 1 error (`std::min(Real, double)` in `diffusion/viscosity.cpp`), but another appeared (`eos/primitive-solver/eos_compose.cpp`, `double*` pointers to `Real*`). I stopped there: the option is not maintained and patching modules the torus does not use is not a good direction. To be measured only if it pays off.
- Estimate (script `code/tools/exp/athenak_cost.py`): original input, dt = 0.3·dx_fine = 0.0375 M, 2.1 M cells, no *local time stepping*. T = 10⁴ M = 2.67e5 cycles = 5.6e11 zone-cycles = **18.6 h at 8.35 Mzcps** (49 h at the small mesh's rate); T = 2·10⁴ M = 37 h. It assumes the large mesh keeps 8.35 Mzcps (not measured) and does not fit in double without larger blocks.
- Conclusion: **running a full MAD on the 5080 is possible but is a project of days, not minutes**, and at low resolution (8 cells/M at the horizon). Realistic option: SANE or a low-resolution MAD, ~10⁴ M, double, left running overnight; or reuse the Illinois snapshots. Decision for André.

## Update 2026-11-13: AthenaK in float (work of a second AI agent), evaluated
The agent left in the clone (`data/athenak`, outside this repository) 16 modified files and a `build_f/` (`Athena_SINGLE_PRECISION=ON`, BLACKWELL120) that compiles. Changes: `Real(...)` literals and `SIGN`/`ONE_3RD` etc. in `athena.hpp`, `cartesian_ks.hpp`, `eos*.hpp`, `hlle_grmhd.hpp`, `gr_torus.cpp`; `tol = 1e-6f` in the c2p with `SINGLE_PRECISION_ENABLED`; `floor_den = 1e-30f`; nuclear EOS tables (`eos_compose`) kept in `double*`; `pow(r,3)` → `r*r*r`. Mechanical and without changing the physics in double (my `viscosity.cpp` is there too).

Measured by me (same inputs, RTX 5080):
| mesh | double | float | gain |
|---|---|---|---|
| 4 levels, 147 thousand cells | 3.19 Mzcps | 18.6 | 5.8× |
| 4 levels, 1.18 M cells | 8.35 | 49.4 | 5.9× |
| **original MAD (9 levels), 8³ blocks** | **does not fit** (9.8 GiB) | 17.9 (10.3 GB GPU) | — |
| **original MAD, 16³ blocks** (same resolution) | 8.45 | **47.1** | 5.6× |

With 16³ blocks (512 blocks, 2.1 M cells, dt = 0.0375 M): **10⁴ M ≈ 3.3 h in float against ≈ 18.4 h in double**; 2·10⁴ M ≈ 6.6 h.

Float vs double correctness (reduced torus, 2000 cycles, t = 1200 M): density rms ≤ 3e-4, internal energy ≤ 1.5e-3, B ≤ 6e-4; in the dense gas (54 % of cells) the velocity agrees to 1e-4–1e-3; **in the low-density atmosphere (ρ < 1e-3 ρ_max, 32 % of cells) the velocities diverge 5–55 %**; time series (ṁ, ė, Φ) equal to 1e-4 at t ≤ 120 and with deviations of 1–15 % in small fluxes later (chaos). To be proved: the full MAD (magnetised funnel, saturation of the flux Φ), c2p failure counts, divB.

## Result 2026-11-13: full MAD float vs double (16³ blocks, up to 1000 M)

Same input (`gr_fm_torus_mad_8_4`, 512 blocks of 16³, dt = 0.0375 M), one run in float (`M_f`, 33 min) and one in double (`M_d`, 1 h 57 min; ~3.5× slower on this input, not 5.6×).

- **The float run blows up.** Both end with rc=0, but the float history has NaN from **t ≈ 40 M** (last clean value at t = 30 M). After that ṁ stays constant (2.24e-7) and ė, Φ = NaN up to 1000 M. The double continues normally (ṁ(1.3 M) ≈ 5e-5, Φ(24 M) grows from 0.76 to 5.5).
- Up to t = 30 M the float agrees with the double to 5 digits (ṁ(1.3 M) = 1.06247e-4 in both). The divergence is abrupt, not a gradual drift: consistent with a c2p/atmosphere failure at a point of the funnel/horizon, not with accumulated rounding error.
- **Exit code 0 does not indicate success**: AthenaK does not abort on NaN. Any long run must check the history (NaN) during execution.
- This **contradicts** the earlier conclusion that float serves for the full MAD: the other agent's validation (reduced torus up to t = 120 M) and mine (up to 1200 M) were of a torus without the funnel magnetisation of the stock MAD. The 5.6× gain only counts if the float survives.
- Cause not investigated (candidates: float c2p with very low/high β or σ in the funnel, density/pressure atmosphere floors, divB). Without that resolved, **the 10⁴ M run of the stock MAD has to be in double (~18 h)**, or in float only with reinforced floors/c2p and validated again against double.

### Investigation of the float NaN (2026-11-13): localised, not solved

Repetitions of the stock MAD (16³) with history and fine dumps, float against double:

- **Where**: the NaN is born in B (the cell-centred field), not in the density or velocity. At t = 36.41 M the float has |B| up to 2.5e6 (normal ≈ 4e-3) and v up to 8; at t = 36.83 there are already 9e4 NaN cells. The maxima are at r ≈ 0.45–0.65, **inside the excised region** (r_KS ≤ 1, ρ at the 1e-8 floor).
- **Comparison with the double** (maximum |B| per radius): identical to 3 digits until t = 33.6 M (r<1: 8.05e-19 float vs 8.09e-19 double). At t = 36.0 M the float has 1.25e-8 at r<1 (and only 1.5e-15 at r∈[1.5; 3)), the double 1.41e-17, smooth. The initial B in the interior is tiny (1e-40…1e-16 in float, already in the denormal range), grows exponentially through noise and the float loses it in a jump of ~9 decades in < 1 M.
- **Plausible origin, not proved**: numerical instability of the field in excised cells, where only the primitives are forced (`eos/ideal_grmhd.cpp`: ρ = dexcise, v = 0) but the field (faces, CT) keeps evolving, with an ill-conditioned KS metric near r → 0 in float.
- **Mitigations tried (3), all failed** (NaN at t ≈ 36–42 M): (1) `flux_excise_r = 1.3`; (2) `flux_excise_r` and `rexcise` (now read from the input, local patch) = 1.3 (still inside the horizon, r_H ≈ 1.35). The trigger at ~36 M is the same in all, so it does not depend on the excision radius.
- Hypotheses to test: reset B = 0 on the excised faces; compile only the metric/c2p in double (mixed float); floors in `ideal_c2p_mhd.hpp`.

**Verdict**: AthenaK's float is not usable on the stock MAD without additional investigation work; the 5.6× does not apply. Decision (André): proceed with the 10⁴ M run **in double** (16³ blocks, ~19 h).

### 10⁴ M run in double (started 2026-11-13 14:14 WET)

`data/athenak/run_d/run10k.sh` → `T10k_d/` (input `mad_10k.athinput` = stock + `gr_torus` + restart every 500 M; 16³ blocks; history dt = 10 M, `mhd_w_bcc` dumps dt = 50 M ≈ 27 GB). Measured rate: 6.6 s/M → ≈ 18.3 h (expected end ≈ 08:30 on 2026-11-14). End flagged in `T10k.done`. Note: `rexcise` is now read from the input in `coordinates.cpp` (local patch, default 1.0, no effect in double).

### Run result and conversion (2026-11-14)

- **Run completed:** `T10k.done` at 12:55 (start 2026-11-13 14:14 = 22 h 41 min of wall clock), t = 10 000 M, **0 NaN** in the history (dt = 0.0375 M, double, 16³ blocks), 201 `mhd_w_bcc` dumps (67 MB each, 22 GB with restarts), a = 0.9375, Γ = 13/9. On 2026-11-14 the "exit code 0" was accompanied by the NaN check (see above).
- **Converter `tools/convert_athenak.py`** (`.bin` Cartesian KS with 9 refinement levels → spherical KS `.bgm` 192×128×128, r ∈ [1.4; 100], X1 = ln r, X2 = θ/π, i.e. MKS with `hslope = 1` and `mks_smooth = 10⁴`, which cancels the polynomial term of FMKS in the reader). Conventions read in AthenaK's source (`ideal_c2p_mhd.hpp`): `velx` = ũ^i = u^i + αγ g^{0i}, `eint` = internal energy per volume, B^i from the CT; Cartesian Kerr-Schild metric `g = η + 2H l l`.
- **Oracles** (all verified):
  - `u·u = −1` and `b̂·u = 0` in float64 to **4.5e-13** and **3.6e-13**; 0 points outside the mesh; 0 NaN in the `.bgm`;
  - the resampler returns the cell values **exactly** at the cell centres (2000 points, error 0 — `tools/check_convert_athenak.py`);
  - away from the centres the trilinear differs from the nearest cell (median 6 %, p90 41 % in ρ): it is interpolation of a turbulent field, not a lookup error; **not** validated against a second independent solution;
  - physical profile in the equatorial plane: ρ ∝ r⁻¹ (12.7 at r = 1.5; 1.46 at r = 12), u^r < 0 at all radii, β = 6–80.
- **M87\* radio calibration** (`--radio-cal --fit-ne0`, 230 GHz, θ = 17°, `rhigh = 20`, Leung): `M_unit = 7.361e24 g` for 0.5004 Jy (target 0.5). **Ring: the initial 22.5 µas was a defect of the metric** (peak of the radial profile around the flux centroid: gave 1.1–5.9 M depending on the dump, catching the base of the jet). With the REx-type metric (centre that maximises the radial-profile peak, r ≥ 2.5 M; `tools/ring_diameter.py` and `--radio-cal`): **9.70–9.90 M = 37.0–37.8 µas** in 6 of 6 dumps (100–200), centre shifted 0.6 M, width ~0.5 M; the **Kerr critical curve** (a = 0.9375, i = 17°, 3.819 µas/M) measures 9.79 M = 37.4 µas, i.e. the ring coincides with theory to ~1 %. The EHT's 42 µas (±3) come from the real M, D and spin of M87*, which our model does not fit. The flux varies 0.40–0.68 Jy between dumps (±25 %): the calibration at 0.5 Jy is from a snapshot.
- **Images and film:** `assets/v22g-own-mad-th17.png`, `…-th60.png` (1000², 8 samples, T_b 10^8.5–10^11 K) and `assets/v22g-own-mad-th17.mp4` (201 dumps, 720², 20 fps, fixed `M_unit`; `tools/render_own_grmhd.sh`; ~10 min). The film shows the transient (at t < ~3000 M the disk is still forming, dim) and then the turbulent state with a jet at 17°. A photograph, not a continuous time series (dt = 50 M between dumps is 1.3 times the inner disk's scale).


### Cause of the float NaN (2026-11-15): B in the excised cells; mitigation by freezing
**Hypothesis tested:** the NaN is born in the magnetic field of the **excised** cells (r_KS ≤ 1: only the primitives are forced to ρ = 1e-8, v = 0; B keeps evolving by CT), where, with ρ at the floor, the Alfvén speed is ~ c and in float the field grows (|B| up to 2.5e6 at r ≈ 0.45–0.65 before the NaN, t ≈ 36 M).
**Test:** in `src/mhd/mhd_ct.cpp` (AthenaK clone, outside this repository; original in `scratchpad/mhd_ct.cpp.orig`) the three face updates (CT b1, b2, b3) skip the faces whose adjacent cells are in the mask.
| variant | result |
|---|---|
| original float | NaN at t ≈ 36–40 M (already measured) |
| freeze with the floor mask (r_KS ≤ 1) | **0 NaN up to 1000 M** (rc = 0, 101 history lines); ṁ(1.3 M) agrees with the double to ≤ 1.6 % in 300 M averages (ratio 1.000 / 0.997 / 0.984), ṁ(12 M) to 0.3 %; **but Φ(1.3 M) (magnetic flux at the horizon) stays 8–15× larger** (per-window means), Φ(12) and Φ(24) agree to 0.3 % up to t = 100 M |
| freeze with the flux mask, `flux_excise_r` = 0.7 | 0 NaN up to 100 M; Φ(12), Φ(24) to 0.3 %; ṁ(1.3) to 2 % |
| freeze with `flux_excise_r` = 0.85 | 0 NaN up to 100 M; same; run up to 1000 M **in progress** |
| freeze with `flux_excise_r` = 0.85, **1000 M** | 0 NaN; float/double ratios of the per-window means (100–400 / 400–700 / 700–1000 M): ṁ(1.3) 1.003 / 1.055 / 1.202; Φ(12) 1.000 / 1.003 / 1.000; Φ(24) 1.000; Φ(1.3) 32.6 / 26.5 / 18.0 |
| freeze r_KS ≤ 1, **1000 M** | 0 NaN; ṁ(1.3) 1.000 / 0.997 / 0.984; Φ(12), Φ(24) 1.000; Φ(1.3) 15.6 / 11.5 / 8.2 |
**Conclusion:** the **localised cause** is the numerical growth of the magnetic field B **inside the excised region** (r_KS ≤ 1, ρ at the floor): letting it evolve in float produces the NaN at t ≈ 36 M; freezing the faces there (`tools/exp/athenak_freeze_excised_B.patch`, only 3 *guards* in `mhd_ct.cpp`) avoids the NaN **up to 1000 M in both variants** and leaves the global diagnostics equal to the double (Φ at 12 and 24 M to 0.3 %; ṁ(12) to 0.3 %; ṁ(1.3) to ≤ 1.6 % with r ≤ 1, up to 20 % with r ≤ 0.85). **To be resolved:** Φ(1.3 M) (absolute magnetic flux ∫|B^r| at the surface r = 1.3, hence inside the horizon) stays 8–33× above the double in both variants; I do not know whether it is float B^r noise near the boundary (the integrand is an absolute value) or a real accumulation. It is also not tested for 10⁴ M. **Decision:** do not use float to extract fluxes near the horizon without further validation; the project's 10⁴ M run remains the **double** one (data of the video and the images). The NaN **does not affect the interactive app**: the app reads ready-made dumps and was audited separately (`27-float-robustness.md`).
