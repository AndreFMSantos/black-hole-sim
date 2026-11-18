---
tags: [coding, black-hole-sim, performance]
created: 2026-11-17
status: measured, no code changed yet
---
# First Nsight Compute profile (RTX 5080, v4_0, commit c35c613)

Reports (git-ignored, 16–31 MB each): `data/ncu/{thin-float,torus-float,layer-float}.ncu-repz`, captured with
`ncu --set full -c 4 -o data/ncu/<name> build/apps/v4_0-metric-api/v4_0-metric-api --shot x.png --runs 1 <scenario args>`.
`ncu` runs as a normal user on this machine (no `ERR_NVGPUCTRPERM`), so no driver option or reboot was needed.

## Per-kernel numbers (main trace kernel, one launch)

| Scenario | Registers/thread | Occupancy limit (regs) | Active warps | Kernel time | Issue slots busy | Highest pipeline |
|---|---|---|---|---|---|---|
| thin disk, float, 1080p | 121 | 4 blocks | 32.2 % | 4.07 ms | 46.8 % | **FP64 72.3 %** |
| torus, float, 700² | 110 | 4 blocks | 32.3 % | 14.84 ms | 29.9 % | **FP64 83.8 %** |
| thick layer, float, 1080p | 128 | 4 blocks | 32.0 % | 12.11 ms | 23.3 % | **FP64 85.1 %** |

- No local-memory spilling (0 bytes) and branch efficiency 99.7 %: the kernel is not memory bound and not divergent (31 of 32 threads active).
- Per-ray helpers are cheap: `k_avg` 0.04 ms, `k_final` 0.1 ms.

## Finding: the "float" path still executes FP64

Only 2.5 % of the executed instructions are FP64 (thin disk: ~44 M of 1.75 G warp instructions: DFMA 17.9 M, DMUL 7.6 M,
DADD 4.4 M, ~9 M DSETP, 1.7 M F2F.F32.F64, 1.2 M MUFU.RCP64H), but GeForce runs FP64 at 1/64 of the FP32 rate, so that
2.5 % keeps the FP64 pipe at 72–85 % and is the top stall source (short scoreboard, 47–66 % of the cycles between issues).

Where it comes from (SASS lines grouped by source line, thin disk; line numbers are from `-lineinfo`, files not shown by ncu, resolved by hand):

| Source | Share of FP64 instructions | Nature |
|---|---|---|
| per-ray setup: `init_from_zamo` (sin/cos/sqrt in double), camera ray (`render.cu` ~802), `dop853_step` | ~35 % | once per ray, intentionally double (accuracy of the initial conditions) |
| inside the step loop: `(y[2] > S(0))`, `r_tail > 0.0`, `fmin(S(1), r_prev / y[0])` (`kerr_cuda.cuh` 428/527/547) | ~12 % | per step, probably double only because of literals / `r_tail`, `r_prev` types |
| post-processing: `acos(mu_end)`, LUT `log`, turbulence `vnoise`, `psi` | ~20 % | once per pixel |

**Not yet proven:** how much time removing it would save. Upper bound from the scheduler: issue slots are 47 % busy
(thin) and 23–30 % (torus, layer), so the FP64 pipe, not the FP32 issue rate, sets the time. The per-step part grows with
steps per ray, so it matters most for the long torus/layer paths.

## Occupancy

Registers cap the kernel at 4 resident blocks per SM (theoretical occupancy 33 %, ncu estimates a 66 % local speedup if it
were fixed, but the stalls are on the FP64/MIO pipe, not on a lack of warps; `__launch_bounds__` or `-maxrregcount` would be a
second experiment, only after the FP64 one).

## Next experiments (each measured with `tools/bench.py` and checked with `tools/raycmp.py`)

1. Per-step FP64 → float where the value is a comparison or a step-size factor (literals as `S(...)`, no semantic change); expect identical sky/disk quantiles.
2. Post-processing in float where the error budget allows (`acos`, LUT log), keep ray setup in double.
3. Only then `__launch_bounds__(…, 5)` to raise occupancy.

## Result of the first two experiments (O6a, O6b), 2026-11-17

Baseline `2026-11-17-1140-2f5ccad.csv`, new `2026-11-17-2310-16532f9.csv` (+ the O6b edit), `tools/bench.py --runs 5`:

| Scenario | Before | After | Speed-up | What changed |
|---|---|---|---|---|
| thin float 1080p | 4.011 ms | 3.473 | 1.15× | O6a |
| thin double 1080p | 355.7 | 355.2 | 1.00× | untouched (double already) |
| layer float 1080p | 11.655 | 5.556 | **2.10×** | O6a + O6b |
| torus float 700² | 14.138 | 8.310 | **1.70×** | O6a + O6b |
| RIAF float 700² | 23.690 | 11.998 | **1.97×** | O6a + O6b |
| spherical float 700² | 6.236 | 2.991 | **2.08×** | O6a + O6b |
| GRMHD float 720p | 34.942 | 31.866 | 1.10× | O6a + O6b |
| KN float 1080p | 3.892 | 3.397 | 1.15× | O6a |
| SdS float 1080p | 3.023 | 2.318 | 1.30× | O6a |
| wormhole float 1080p | 14.995 | 15.007 | 1.00× | separate tracer, untouched |

- **O6a: observer constants out of the per-ray code.** `init_from_zamo` computed `sin`, `cos`, three `sqrt` and several divisions in FP64 for every ray, from inputs that are the same for the whole frame (a, r, θ, q², Λ). `zamo_const()` now runs once per launch in a 1-thread device kernel (`zamo_kernel`) and the per-ray part keeps only the arithmetic that depends on the ray direction. Same device functions and the same expressions, so the bits are unchanged (a host `sin` would not be bit-identical to the device `sin`, hence the kernel). FP64 instructions in the thin float kernel: 51.1 M → 35.5 M.
- **O6b: one division.** In the volume sampling loop, `S((q + 0.5) / K)` has an `int` and a `double` literal: a **double division per sub-sample**. 82 M of the 187 M FP64 instructions of the torus kernel (44 %) were this one line. Now `(S(q) + S(0.5)) / S(K)` in the state type. `q + 0.5` and `K` are exact in float and IEEE division is correctly rounded, so the result differs from the old route (double divide, then round to float) only if the double rounding changes the last bit.
- Plus `r_tail > 0.0 && lam == 0.0` evaluated once before the step loop (two double compares per step).
- **Verification:** `ctest` 37/37 after each step (goldens included); layer 1080p and torus 700² float images **byte-identical** (same MD5) to the ones rendered by the pre-O6 binary.
- **How the culprit was found:** `ncu --page source --csv --print-source cuda,sass` gives the executed instructions per SASS line with its CUDA line; summing the DFMA/DMUL/DADD/DSETP/MUFU.RCP64H/F2F.F64 lines per CUDA line ranks the FP64 sources. Use `-k kernel` (the base name), not a regex on the mangled name: `zamo_kernel` and `k_avg` also match `regex:kernel`.

Still FP64 in the float path (thin disk, after O6a): camera ray (`KerrZamoObserver::ray`, ~11 %), per-step comparisons at the disk crossing test and step update (~12 %), `acos(mu_end)` and the colour LUT `log` (~7 %), turbulence/noise (`vnoise`, `cos`, `sin`) when enabled. Next: O6c, camera ray and post-processing, where float would change the result and needs `raycmp.py`.


## O6d (sprint S4, 2026-11-18): the FP64 that was left in the thin-disk float path

Profile `data/ncu/thin-float-O6d*.ncu-repz` (git-ignored), thin disk, float, 1080p, a = 0.9, i = 70°, `ncu --set full -k kernel -c 1`. Only changes that keep the image **byte-identical** were made:

| Change | What it removes |
|---|---|
| `redshift()` reads the observer's `alpha`, `omega` from the per-ray constants (`p.zamo`, hoisted in O6, same expressions) | `sin`, `cos`, `sqrt`, divisions per disk hit |
| `nt_temp()`: `ln r_in` and `(n - 1)/(ln r_out - ln r_in)` computed once on the host | two double `log` and a division per lookup |
| `lut_lookup()`: `(kN - 1)/ln(kT1/kT0)` in `__constant__` memory | one double `log` and a division per lookup |

| | before | after |
|---|---|---|
| kernel time (ncu, clocks locked) | 4.86 ms | 4.12 ms (-15 %) |
| FP64 instructions | 62.0 M | 52.6 M (-15 %) |
| image md5 (same scene) | | identical (`cmp`) |
| ctest | | 38/38 (goldens included, the new `v4_0_oracle_rays` too) |

What is left (52.6 M): the end-of-kernel accumulation `double -> float` and the finite/luminance checks on the double `col` (17 %), the colour LUT interpolation in double (7.6 %), and the cosmetic surface pattern (`psi`, `vnoise`, ~14 %). The pattern is the only sizeable one that is not physics, but doing it in float changes the picture (its noise would be evaluated at a different precision), so it is left as is: an aesthetic change should be an explicit choice, not a side effect of an optimisation.
