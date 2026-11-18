---
tags: [coding, physics, black-hole-sim, cuda]
created: 2026-11-12
---

# v0.7d — Cinematic GPU (v0_6b-kerr-cuda; now `v0_7-cinematic-cuda` / `v3_1-kerr-cuda-ui`)

Sequel to [06b-kerr-cuda](06b-kerr-cuda.md), [07a-novikov-thorne](07a-novikov-thorne.md) and [07c-moving-observer](07c-moving-observer.md): the Kerr kernel gains the Novikov-Thorne profile, the moving observer, progressive supersampling, bloom and filmic tone mapping. `--help` describes everything.

## What went in
| Item | Implementation | Option |
|---|---|---|
| NT profile | `nt::Profile::table()` (2048 points in ln r) copied to an array; linear interpolation in `double`, `r^(-3/4)` tail; rebuilt when `a` changes (17 ms) | `--profile nt\|newtonian` (nt by default, like the CPU) |
| Moving observer | `lorentz::observer_to_zamo` ported to the device (`double`): `n'=(1,y,−x)/nrm` → `n` (aberration) which enters the trace; `dop = E'/E_zamo` multiplies the disk `g`. The sky only undergoes aberration | `--observer static\|orbit\|infall`, `--beta br,bth,bph` |
| Progressive supersampling | RGBA32F accumulation buffer; sample `k` = R2 jitter (`k·α + per-quad rotation`, α = (0.7549, 0.5698)); sample 0 is the pixel centre; fixed seed (`--seed`) ⇒ deterministic | `--samples N` (offline); window: accumulates up to 128 with the scene still and restarts on any change |
| Bloom | 6-level pyramid (2×2 reduction + separable Gaussian [1 4 6 4 1]) added to the HDR before tone mapping | `--bloom s` (0 = off), key `B` |
| Tone mapping | Reinhard on the luminance (default, comparable to the CPU) or ACES (Narkowicz fit, exposure 0.6), **after** the average | `--tonemap reinhard\|aces`, key `T` |
| Cinematic | preset: aces + bloom 0.12 + `--beaming 0.5` + luminance clamp 50 | `--cinematic`, key `C` |

**`--beaming s` and `--cinematic` are aesthetic choices, not physical.** `--beaming` replaces `g` by `g^s` only in the disk's apparent intensity/temperature (reduces the Doppler asymmetry, Interstellar style); it is labelled as such in `--help`, in the window title (`[CINEMATIC: aesthetic, not physical]`) and here. The default look is the physical one.

## Decisions
- **Same jitter in the 4 threads of a quad:** the sky uv gradient (`tex2DGrad`) comes from differences between quad neighbours via `__shfl_xor_sync`; if each pixel had a different jitter the derivative would be off by up to a factor 2. The jitter hash uses `(px>>1, py>>1, seed)`.
- **Average in HDR, tone map afterwards:** this is what the bloom requires. The price is a residual error against the CPU SSAA (which does Reinhard *per sample*): 0.175 vs 0.167 MAE at N = 64. The `--accum ldr` mode (Reinhard per sample before accumulating) exists only to validate against the CPU.
- **Fireflies:** the blue side of the Doppler shift reaches ~50× the reference brightness; `--clamp L` limits the luminance *per sample* (the cinematic preset uses 50). Off by default (exact regression).
- **NT profile on the GPU = table, not computation:** the flux integral costs 17 ms per spin on the CPU; the GPU only interpolates. Changing `a` in the window (`[`/`]`) rebuilds the table.

## Validation (1080p; `tools/validate_kerr_cuda.py`, `tools/validate_v07d.py`)
**Regression (1 sample, reinhard, bloom 0, `--rawsky`, CPU `--ssaa 1`), 9 views** (a = 0, 0.9, 0.998 × θ = 80°, 86° with disk, × θ = 90° without disk):

| Profile | Precision | MAE (0–255) | > 4 | > 16 |
|---|---|---|---|---|
| newtonian (= v0.6c) | double | 0.019 – 0.023 | 0.00 % | 0.00 % |
| newtonian | float | 0.186 – 0.242 | 0.50 – 0.68 % | 0.02 – 0.04 % |
| **nt** (vs `v0_6-kerr-cpu --profile nt`) | double | 0.019 – 0.023 | 0.00 % | 0.00 % |
| **nt** | float | 0.186 – 0.242 | 0.50 – 0.68 % | 0.02 – 0.04 % |
Identical to the v0.6c numbers: the NT LUT adds no measurable error.

**Convergence with the number of samples** (a = 0.9, θ = 80°, nt, float, `--rawsky`) against the CPU `--ssaa 8` (64 samples/pixel; 178 s on omarchy):

| N | accumulation | MAE | > 4 | > 16 | ms/frame GPU |
|---|---|---|---|---|---|
| 1 | ldr = hdr | 1.314 | 8.00 % | 0.38 % | 8.1 |
| 4 | ldr / hdr | 0.827 / 0.831 | 2.84 / 2.86 % | 0.14 / 0.15 % | 32.5 |
| 16 | ldr / hdr | 0.362 / 0.367 | 0.58 / 0.62 % | 0.02 / 0.03 % | 130 |
| 64 | ldr / hdr | **0.167 / 0.175** | 0.28 / 0.35 % | 0.01 / 0.02 % | 520 |
The error drops ~1/√N (8× less between 1 and 64) and the residual at N = 64 is that of float mode itself + the different jitter. 64 samples on the GPU (0.5 s) ≈ 178 s on the CPU.

**Moving observer** (CPU `--ssaa 1` vs GPU 1 sample, `--rawsky`, a = 0.9, θ = 85°): orbit r = 12: double MAE 0.003, float 0.038; infall r = 10: double 0.004, float 0.043 (> 4: ≤ 0.14 %), in the newtonian and nt profiles.

**CUDA selftest** (`--selftest`, also in `ctest`): NT LUT vs `nt::Profile::temperature` to 2.8e-8 (a = 0, 0.9, 0.998, r from r_isco to 300 and r < r_isco); boost vs `lorentz::observer_to_zamo` to 4e-16 (5 β × 200 directions); deterministic accumulation with a fixed seed (0 differing bytes), the seed changes the jitter, `samples=1` does not depend on the seed, bloom raises the mean luminance, aces ≠ reinhard.

## Performance (1080p, a = 0.9, θ = 80°, mean of 10 runs, ms/frame)
| Stage | float | double |
|---|---|---|
| newtonian, static, 1 sample (v0.6c) | 8.19 | 785 |
| + NT (LUT) | 8.14 | 784 |
| + `orbit` observer (r = 40) | 8.25 | 779 |
| + `infall` observer (β = −0.32) | 8.48 | 820 |
| + ACES | 8.15 | 784 |
| + bloom 0.12 | 8.21 (**bloom 0.048 ms**) | 784 |
| cinematic, 1 / 4 / 16 / 64 samples | 8.26 / 32.7 / 130 / 523 | 784 / 3140 / – / – |
Resolve (average + tone mapping) = 0.14 ms; the bloom pyramid 0.05 ms; **everything that is not the trace costs < 0.2 ms**. The cost is linear in the number of samples (8.1 ms each in float). Interactive window (float, 1896×1040): ~95 fps while moving; converging ~94 fps up to 128 samples (1.4 s), then it sleeps.
(Compared with v0.6c, float is +10–20 % slower: the sixth state σ and the v0.6c step limiter, not this phase.)

## Images (`assets/v07d-*`)
`v07d-a09-reinhard.png` / `v07d-a09-cinematic.png` (a = 0.9, θ = 80°, 64 samples), `v07d-a0998-reinhard.png` / `-cinematic.png` (a = 0.998, θ = 86°), `v07d-orbit-r12.png` / `-cinematic.png` (circular orbit r = 12, β = 0.296), `v07d-1sample.png` vs `v07d-64samples.png` and the 3× enlarged crop `v07d-aa-crop.png` (left 1 sample, right 64: the disk edge no longer has steps).

## Limitations
The bloom is a convolution with no threshold (stars glow too); no *lens flare* or *glare*; the per-quad jitter correlates the noise in 2×2 blocks (invisible with N ≥ 16); the sky has no Doppler/colour; `--cinematic` is aesthetic; no *denoising*; tone mapping after the average gives a ~5 % residual in the error against the CPU SSAA. Not tested by hand: mouse, keys `T B O C [ ]` and resizing (only fps/title in the window).
