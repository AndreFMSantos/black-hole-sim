---
tags: [coding, black-hole-sim, cuda, optix]
created: 2026-11-12
---

# v0.8c — Optional OptiX denoiser

## What it is and how it is wired in
OptiX's AI denoiser (`OPTIX_DENOISER_MODEL_KIND_HDR`), **optional in two layers**:
1. **Build:** `option(BHS_OPTIX ON)`; it only activates if the headers exist at `code/third_party/optix-dev/include/optix.h` (`code/tools/fetch_optix.sh`: `git clone --depth 1 --branch v9.1.0` of the public `NVIDIA/optix-dev` repository, commit `f1f6dd80…` verified, idempotent; recorded in `code/third_party/README.md`). Without headers (or `-DBHS_OPTIX=OFF`) a stub (`denoise_stub.cpp`) is compiled and the rest is the same; the Mac does not even build the CUDA target. The headers stay **outside Git** (NVIDIA SDK licence, accepted by André on 2026-11-12: do not redistribute the headers).
2. **Run time:** `--denoise none|optix` (default `none`), `--denoise-albedo` (albedo guide), key **N** in the window (title: "denoise OptiX"). Without OptiX the option fails with *"OptiX not available: built without BHS_OPTIX (run code/tools/fetch_optix.sh and rebuild)"*.
Pipeline: HDR average of the N samples → **`optixDenoiserInvoke`** (input/output `FLOAT4` RGBA32F; null `hdrIntensity` = OptiX auto-exposure; `blendFactor` 0; alpha copied) → bloom/glare → tone mapping. Context, denoiser, state and scratch are created **once** (`denoise::init`, idempotent; recreated only if the maximum capacity or the albedo changes); `optixDenoiserSetup` repeats only when the size changes (`--move-scale`).
Compatibility: **OptiX 9.1.0 headers (ABI 118)** with the **615.78.08** driver runtime: `optixInit`, context, denoiser and invocation worked without errors (nothing had to be worked around).
**Albedo guide:** own accumulator (normalised colour `c/max(c)` per sample; 0 where there is no light): the disk and the sky have colour but not "albedo" in the sense of a diffuse surface; measured below.

## Results (1080p, a = 0.9, θ = 80°, float, physical look; `tools/validate_v08c.py`)
Reference: **256 samples without the denoiser** (`--sample-offset 5000`, seed 1); the tests use seed 2 and `--sample-offset 100` (all samples *jittered*; the default sample 0 is the pixel centre and has no noise). HDR peak (p99.9 of the reference) = 6.23. sRGB PSNR = after tone mapping (255); HDR PSNR = HDR average before bloom/tone mapping.

| N | mode | ms/frame | denoiser ms | HDR PSNR (dB) | sRGB MAE | sRGB PSNR (dB) | > 16 |
|---|---|---|---|---|---|---|---|
| 1 | without | 8.2 | – | 56.49 | 1.651 | 37.93 | 0.57 % |
| 1 | optix | 13.1 | 4.91 | 56.92 | 1.398 | **39.00** | 0.46 % |
| 1 | optix+albedo | 13.7 | 5.37 | 57.00 | 1.365 | 39.36 | 0.40 % |
| 2 | without | 16.3 | – | 60.67 | 1.016 | 41.76 | 0.21 % |
| 2 | optix | 21.3 | 4.91 | 60.82 | 0.952 | 42.15 | 0.19 % |
| 2 | optix+albedo | 21.9 | 5.38 | 60.62 | 0.995 | 41.75 | 0.20 % |
| 4 | without | 32.6 | – | 67.12 | 0.489 | 47.56 | 0.03 % |
| 4 | optix | 37.5 | 4.91 | 67.15 | 0.479 | 47.69 | 0.03 % |
| 4 | optix+albedo | 38.2 | 5.38 | 66.52 | 0.568 | 45.90 | 0.06 % |
| 8 | without / optix / +albedo | 65.0 / 69.9 / 70.8 | 4.90 / 5.39 | 70.57 / 70.56 / 69.45 | 0.334 / 0.335 / 0.437 | 50.38 / 50.40 / 47.44 | 0.01 / 0.01 / 0.04 % |
| 16 | without / optix / +albedo | 129.8 / 134.7 / 136.1 | 4.90 / 5.39 | 72.79 / 72.78 / 71.46 | 0.237 / 0.244 / 0.347 | 52.64 / 52.55 / 48.77 | 0.00 / 0.00 / 0.03 % |
| 32 | without | 260.0 | – | 76.75 | 0.148 | 55.50 | 0.00 % |
| 64 | without | **520.0** | – | 83.60 | 0.072 | **59.12** | 0.00 % |

Second scene (a = 0.998, θ = 86°, `--cinematic --grain 0 --vignette 0`, sRGB PSNR): N = 1: 42.01 → 42.86 (optix) / 42.84 (+albedo); N = 2: 45.85 → 46.09; N = 4: 51.41 → 51.45; N ≥ 8 no gain; 64 samples without the denoiser = 62.85 dB.

### The question: how many samples + denoiser match 64 without the denoiser?
**None (N ≤ 16).** The best case (N = 16 + denoiser, 52.55 dB) is 6.6 dB below 64 samples (59.12 dB) and the denoiser does not improve N ≥ 8. The time gain is **zero or negative**: N = 1 + denoiser (13.1 ms) gives 39.0 dB, while N = 2 without the denoiser (16.3 ms) gives 41.8 dB and N ≈ 1.4 without the denoiser (≈ 11 ms) would already give 39 dB. **The denoiser does not pay off in this application.**
**Why:** the "noise" here is not Monte Carlo variance: the ray is an integrated, deterministic curve; what varies between samples is the sub-pixel position, and the N = 1 error is *aliasing* on thin edges (disk edge, ring arcs, stars) in an otherwise smooth image. The OptiX model was trained on *path-tracing* noise and can only smooth steps (a ~1 dB gain at N = 1), without reconstructing the missing detail. The albedo guide helps marginally at N = 1 and **hurts** at N ≥ 4 (−1.7 to −3.9 dB), because the normalised colour has false edges where the brightness drops to 0.

### Crops (N = 1 / N = 1 + optix / N = 4 / N = 4 + optix / reference 256; `assets/v08c-crops.png`)
- **Photon-ring arcs** (thin, bright): optix at N = 1 smooths the steps but **does not restore the thin stretches** that are missing; at N = 4 with and without the denoiser are almost equal to the reference.
- **Point star** (blue, brightest): no visible difference: it does not blur it.
- **Disk edge** (disk against sky): steps at N = 1; optix smooths them (the best effect observed).
- **Blue side of the Doppler shift** (smooth, bright area): no visible difference.
Nothing was visibly "ruined" (no *blur* of stars or arcs); the gain is only on edges at N = 1–2.

### Temporal flicker (`tools/flicker_v08c.py`, N = 1 / N = 2)
(A) static camera, different noise: frame-to-frame MAE **2.260 → 1.847** (N = 1) and 1.443 → 1.304 (N = 2) with the denoiser. (B) one step of the spin clip (same seed): excess over the true change (0.273, 64-sample reference) **+0.099 → +0.047** (N = 1) and +0.058 → +0.035 (N = 2). **The denoiser does not introduce flicker; it reduces it slightly.** (Measured on individual frames, not on an encoded video.)

### Cost
Denoiser at 1080p: **4.91 ms** (5.38 ms with albedo), constant with N. In flight (1 sample/frame, r = 12): 166 → 92 fps; with `--move-scale 0.5`: 555 → 278 fps; orbital mode r = 40: ~79 fps measured with the denoiser.

## Tensor cores: what was verified
- **According to NVIDIA's documentation**, the OptiX AI denoiser runs on RTX GPUs with tensor cores.
- **What I verified:** there is no `nsys`/`ncu` on this system; I used **CUPTI** (`--denoise-kernels`) to list the CUDA kernels that `optixDenoiserInvoke` launches (1080p, 2 invocations, 9 distinct kernels, ~9.7 ms of kernels): `optix_exp::conv_ampere_128_128_32_kernel`, `conv_ampere_256_64_32_kernel`, `conv_ampere_128_128_64_kernel`, `conv_ampere_fpool_128_128_32_kernel`, `fused_convolutions_kernel_sm80`, `k_tin3_mlp<64|96|128>` (which takes `const __half*`) and some "NVIDIA internal" ones. They are FP16 convolutions and MLP for Ampere+; **this is consistent with tensor cores but does not prove it** (I did not read hardware counters, e.g. `sm__pipe_tensor_op_*`). I claim no more than this.
- The rest of the render stays on the **CUDA cores**: there are no RT cores in our path (the ray is an integrated curve, with no triangles or BVH).

## Tests
`--selftest` (in `ctest`): synthetic image (gradient + blob, 25 % multiplicative Gaussian noise): **denoised MSE = 0.5 % of the noisy MSE**; full pipeline with `--denoise optix` (finite HDR, denoiser executed: 1.3 ms on a small scene, image different from `--denoise none`). `ctest` green in **both modes**: with OptiX (3/3) and with `-DBHS_OPTIX=OFF` (4/4, includes `v08c_no_optix`: `--denoise optix` fails with the clear message). Physical-look regression byte-identical in both (`v07d_regression`).

## Limitations
Without `--denoise` the code is the usual one; the measured gain only exists at N = 1–2 and is small; the HDR model assumes MC noise (not aliasing); no normals guide (there is no geometry); no OptiX temporal mode (there are no motion vectors; the camera changes every frame in flight). Not tested by hand: the `N` key in the window.
