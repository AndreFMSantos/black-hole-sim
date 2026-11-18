---
tags: [coding, black-hole-sim, performance]
created: 2026-11-13
status: done
---
# v2.2h — Kernel optimisation (integration of the `v06b-opt-lab` lab)

## Origin
Another agent wrote a fork (`apps/v06b-opt-lab`, 2389 lines, without editing existing files) with optimisations for the RTX 5080 (sm_120). I reproduced its numbers and **integrated only what was justified, in `render.cu`**, without duplicating versions. The fork was not committed and was deleted on 2026-11-13 at André's request (everything of interest is in `render.cu`).

## What the fork said and what was confirmed
Reproduced (they match its summary): thin 13.0→9.2 ms, GRMHD 700² 106→36.6 ms, etc. But its "baseline" is the fork itself with flags off, **slower than the real `render.cu`** (thin 13.0 against 10.5 ms; GRMHD 106 against 68 ms). Against our code the real gains are smaller.

| Optimisation | Verdict | Measurement |
|---|---|---|
| Specialised kernels (precision × model, `if constexpr`) | **integrated** | 238 → 106–142 registers |
| Radiative transport in float (GRMHD, torus) | **integrated** | GRMHD 41→24.6 ms; torus 22.8→18.9 ms |
| 2D LUT + 2 Newton steps for `f_inv_theta` | **integrated** | GRMHD 26.7→21.6 ms; 2.7e-5 cells vs exact, also at the poles |
| FSAL (k7→k1) | rejected | thin +3 %, thick +1 %, GRMHD 0; in thick it changes 5 % of the pixels (max 86 levels) because it changes the step sequence |
| 8³ brick skipping | rejected | 1.00× in its own lab; risk of cutting faint emission |
| FP16 textures | rejected | ~1 % gain, PSNR 74.7→72.0 dB; ρ spans decades |
| Async streams + pinned memory | postponed | 1.23× only at 1 sample/frame; on a route (32+ samples, 0.8 s/frame) the D2H copy is negligible |
| Adaptive supersampling | postponed | 3.6× at 64 spp with MAE 0.48, but the mask may flicker in video |

What the fork had wrong or imprecise: the header of `kerr_opt.cuh` says FSAL is "bit-exact", but `k1[4]` (dφ/dτ) was computed with the σ before the reprojection; it is only exact if recomputed.

## Results (RTX 5080, float, mean of 3–5 runs, before = commit 3cb3e6b)
| Case | before | after | gain |
|---|---|---|---|
| thin disk 1080p | 10.5 ms | 7.2 ms | 1.47× |
| thick disk 1080p | 50.2 ms | 22.1 ms | 2.27× |
| radio torus 700² | 43.3 ms | 18.9 ms | 2.29× |
| GRMHD MAD 700² (step 0.05) | 68.5 ms | 21.7 ms | 3.16× |
| GRMHD MAD 1080p | 226.9 ms | 74.5 ms | 3.05× |
| GRMHD MAD 720p (step 0.15 r) | 90.1 ms | 30.5 ms | 2.95× (≈33 fps) |

For the interactive mode (André's question): **static GRMHD at 720p ≈ 33 fps, at 1080p ≈ 13 fps** with 1 sample per pixel. The cost of two snapshots with temporal interpolation is still missing.

## Verification
- `ctest` 3/3 (includes the v0.7d golden md5: the double and thin-disk paths did not change).
- `--selftest` OK; GRMHD torus oracle unchanged (ratio−1 = 2.77e-3, per-ray median 2.2e-3, p95 5.7e-3).
- Torus `j` float vs double: 6e-4 (limit 5e-3).
- Images: thin disk and GRMHD identical in 8 bit; thick disk 4 pixels differing by 1 level (PSNR 110 dB).
- New test: LUT vs exact bisection with 6000 points, 1/3 of them within 0.02 of X2 from each pole.

## How it works (for the blog)
The single kernel had all modes and precisions side by side (238 registers, 72 B of stack): SM occupancy was low even for the thin disk. It became `template<int PREC, int MODEL> kernel`, with the host choosing the instantiation; the tests still use run-time dispatch (`VolAcc` = `VolAccT<-1>`). The rest of the GRMHD/torus gain comes from taking FP64 off the hot path (1/64 of the rate on the 5080), keeping only the `Irad` accumulation in double.

## To do
- Measure the cost of two snapshots + interpolation (v2.2f-c).
- If the interactive mode needs more: low resolution + reprojection.
