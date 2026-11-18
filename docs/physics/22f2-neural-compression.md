---
tags: [coding, black-hole-sim, grmhd, data, exploration]
created: 2026-11-12
---

# v2.2f-a2 — A neural network to compress the GRMHD data? An exploration with numbers

André's question (2026-11-12): instead of storing the data, train a neural network that reproduces it and thus reduce the space. Decision on direction **postponed** until we explored the question. Context: [22f-grmhd-snapshot](22f-grmhd-snapshot.md). Scripts in `code/tools/exp/` (reproducible). References in [references](../REFERENCES.md).

## Is there really a space problem?
Measured/computed with the real snapshot (151 MB, 4.72 M cells × 8 `float32` variables; 1001 per simulation):

| Case | Snapshots | Raw `float32` | 10 bits (all variables) |
|---|---|---|---|
| 2 min video at 20 M/s (2400 t_g) | 480 | **72.5 GB** | 22.6 GB |
| 3 min video at 20 M/s (3600 t_g) | 720 | **108.7 GB** | 34.0 GB |
| one whole simulation | 1001 | 151 GB | — |
| all 10 simulations | 10 010 | 1.51 TB | — |

Free space on the data disk: **1.8 TB**. Reading a snapshot at ~3 GB/s of NVMe takes ~50 ms, and an offline render is not limited by that. **For a 2–3 min time-lapse the space is not a blocker today** (72–109 GB from a single simulation); it becomes a problem if we want several simulations or the whole 4D.

## What each alternative gives (log10 ρ, MAD a=0 snapshot frame 5500)
| Method | Size (ρ only) | rms error | p99.9 error | Cost |
|---|---|---|---|---|
| lossless (shuffle + zlib) | — (whole file: 118.6 MB) | 0 | 0 | **only 1.27×**: turbulent `float32` data is almost incompressible |
| 10-bit log quantisation | 5.9 MB | 0.0021 dex | max 0.0037 | none |
| 8-bit log quantisation | 4.7 MB | 0.0086 dex | max 0.015 | none |
| `float16` of log ρ | 9.4 MB | 0.0005 dex | max 0.002 | none |
| 2× subsampling per axis + trilinear | 2.4 MB (8 bits) | 0.0396 dex | **0.394** | none |
| **neural field** hash grid 8 levels, T = 2^15 (530 k parameters) | **1.06 MB** (fp16) | 0.0347 dex | **0.150** | 29 s of training |
| **neural field** hash grid 12 levels, T = 2^17 (3.15 M parameters) | 6.30 MB (fp16) | 0.0157 dex | 0.070 | 45 s of training |

(1 dex = factor 10. 0.035 dex ≈ 8 % in ρ; 0.0021 dex ≈ 0.5 %.) A radial cut gives little (the grid is logarithmic: `r < 60` still has 56 % of the cells, 1.8×).

**Honest reading:** the network only wins at extreme compression (~18×, 1 MB per variable) and there it matches subsampling 2× in rms but has a 2.6× better p99.9 (0.15 vs 0.39 dex). With ~6 MB per variable **simple quantisation is ~7× more accurate** (0.0021 against 0.0157 dex) and needs no training. The network is **not** a "free" saving: at 4 GB for 480 snapshots (18×) the error is 8 % in ρ.

## What must not go unnoticed
1. **I only measured ρ, in a single snapshot.** The synchrotron emission depends **exponentially** on `Θ_e` and on `B` (via `exp(−1.89 x^{1/3})`); small errors in `UU` and `B`, and in the hot extremes, weigh more than the error in ρ. The number that matters is the error in the **final image**, which I did not measure.
2. **The training cost scales badly:** 29 s per variable and per snapshot with my naive implementation ⇒ 8 variables × 480 snapshots ≈ **31 h** (720: ≈ 46 h). Untuned.
3. **Physical constraints are lost:** a network does not preserve `∇·B = 0` nor the positivity of ρ and UU by construction (log solves the second).
4. **The turbulence decorrelates fast** (see below), so a neural field **per snapshot** does not exploit temporal coherence; a **4D** field (with time as input) would be the interesting network and is much harder to train.

## Temporal experiment: how much can be reconstructed from the neighbours?
Reconstructing frame 5500 (t = 27 500) from 5499 and 5501 (**10 t_g** apart, twice the real spacing):
- The gas rotates well below Keplerian (Ω measured by correlation in φ: 0.034 at r = 3 against 0.189 Keplerian; 0.0116 at r = 6 against 0.068; 0.0105 at r = 10 against 0.031; ≈ Keplerian only at r = 50, 0.0026 against 0.0028), as expected from a MAD.
- Error in log ρ: **copying the neighbour** 0.119 dex; **linear interpolation** 0.098 (p99.9 1.25); **following the rotation** (advection) **0.066** (p99.9 0.88). At `r < 30`: 0.146 → 0.098; at `r < 10`: 0.186 → 0.126.
- **Conclusion:** snapshots cannot be **thrown away** to save space (at 10 t_g there is already 0.1 dex rms and ~1 dex at the worst points), and the interpolation that follows the rotation beats the linear one by ~33 %. The error **at the real 5 t_g cadence** should be smaller, but I did not measure it (there is no ground truth at 2.5 t_g).

This suggests that the useful role of a network here is not compression, it is **interpolating in time** (60 fps at 20 M/s are 0.33 M per frame, against snapshots 5 t_g apart).

## Opinions (for us to decide)
1. **Do not build a compression network now.** There is no space blocker for the first goal (2–3 min time-lapse); 10-bit quantisation (3.2×, 0.002 dex) gives whatever modest saving we need, with no training and with controlled error.
2. **Reader v2.2f-b with `float`/log quantisation**, reading snapshots from NVMe (50 ms each) and **temporal interpolation that follows the rotation** (cheap and already shown better than linear).
3. **Optional research branch:** 4D neural field for temporal interpolation (self-supervised learning: predict the middle frame from the neighbours, as in video interpolation). It needs a measurement on the final image, not just on ρ.
4. **Another, separate route:** run the GRMHD ourselves (KHARMA/AthenaK) with whatever output cadence we want (0.5 t_g, say), which **eliminates the interpolation**; the cost is GPU time and not space, and I have not yet measured how long it takes.

## Limitations of these experiments
A single snapshot (MAD a=0), only `log10 ρ`, hash grid in naive PyTorch (2500 steps, untuned), comparison with simple quantisation, temporal test only at 10 t_g and only in ρ; none was evaluated on the final image. `uv` installed the CUDA 13 PyTorch wheels and it worked on the RTX 5080.


## Quantisation error on the FINAL IMAGE (2026-11-15; `tools/exp/quantize_bgm.py`)
Before, only the error in ρ had been measured (0.002 dex at 10 bits). Now it is measured on the image: each of the 12 fields of the `.bgm` of the t = 10 000 M dump is quantised (ρ, θ_i, |b|, β on a uniform log10 scale between minimum and maximum; u^μ and b̂^μ on a uniform linear scale), dequantised and rendered in float (512², 4 samples, `--munit 7.361e24`, M87*, 230 GHz, `rhigh 20`), comparing the intensity (PFM, Jy/µas²) with that of the original dump at pixels with I > 1 % of the maximum.
| bits | θ | flux (relative) | per-pixel relative error: median / p95 / max | rms/max |
|---|---|---|---|---|
| 12 | 17° | −1.7e-4 | 1.5e-3 / 7.5e-3 / 2.3e-2 | 5.6e-5 |
| 12 | 60° | −6.4e-5 | 9.4e-4 / 5.6e-3 / 1.8e-2 | 3.0e-5 |
| 10 | 17° | +2.9e-3 | 8.8e-3 / 5.6e-2 / 1.5e-1 | 3.4e-4 |
| 10 | 60° | −1.4e-4 | 4.9e-3 / 3.2e-2 / 9.9e-2 | 1.5e-4 |
| 8 | 17° | +7.6e-3 | 1.5e-1 / 6.3e-1 / 7.2e-1 | 3.5e-3 |
| 8 | 60° | +1.8e-2 | 4.9e-2 / 3.2e-1 / 1.2 | 1.5e-3 |
**Reading:** at 10 bits (3.2× less memory) the image changes by less than 1 % in the median and 0.3 % in the flux, which cannot be seen; at 8 bits with uniform quantisation the error is large (the range of b̂^μ, [−12, 21] in KS, wastes levels; non-uniform quantisation of u^μ and b̂^μ could help, not tested). A single snapshot and two angles, no temporal variability.
