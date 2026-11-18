---
tags: [coding, physics, black-hole-sim, cuda]
created: 2026-11-12
---

# v0.6b — Kerr on CUDA (RTX 5080)

Port of the Kerr render of [v0.6a](06-kerr-cpu.md) to a CUDA kernel. The reference is still `v0_6-kerr-cpu`; this phase measures *how much* is gained and *what it costs* in accuracy. (The app lives today in `apps/v0_6b-kerr-cuda/`, frozen at tag `bhs-v0.6c`; later features are in `v3_1-kerr-cuda-ui`.)

## Environment and the "GCC 16 problem"
omarchy (Arch): CUDA 13.4.3 in `/opt/cuda` (nvcc 13.4.92), driver 615.78, RTX 5080 = compute capability 12.0 (`-arch=sm_120`), GCC 16.2.1. **nvcc accepted GCC 16 with no flags** (`-ccbin`, `--allow-unsupported-compiler` were not needed; CMake detects "NVIDIA 13.4.92 with host compiler GNU 16.2.1"). Only tweak: `nvcc` is not on the `PATH`, so the top-level `CMakeLists.txt` uses `/opt/cuda/bin/nvcc` when `CUDACXX` is not set. If a future nvcc refuses GCC, the order to follow is: older host compiler (`-DCMAKE_CUDA_HOST_COMPILER=...`), and only as a last resort `--allow-unsupported-compiler`.

CMake: `check_language(CUDA)` + `find_package(CUDAToolkit)` inside `if(NOT APPLE)`; the `v0_6b-kerr-cuda` target only exists with `CMAKE_CUDA_COMPILER`. The Mac stays the same (no CUDA, not tested there). `CUDA_ARCHITECTURES 120` (`-DBHS_CUDA_ARCH=...` to change).

## Design (`apps/v0_6b-kerr-cuda/`)
- `kerr_cuda.cuh` — the physics, ported line by line from `kerr.hpp`: `init_from_zamo`, `R/dR/M/dM/dphi`, Dormand-Prince 5(4), velocity reprojection after every accepted step, cubic Hermite at the μ = 0 crossing, capture `r < r₊ + 1e-3`, `r_far = 1e6`. As a *template* `<S, F>`: `S` = state type, `F` = type of derivatives and increments (see Precision). Nothing is shared with `kerr.hpp` via `__host__ __device__` (the header uses `std::array`/lambdas and the Mac has to keep compiling); instead the **equivalence is tested**: `--selftest` (also in `ctest`, `v0_6b_selftest`) traces 6×2304 rays on the GPU and on the CPU (`kerr::trace`) and requires the same fate in ≥ 99.5 % and an angular error p99 < 1e-6.
- `render.cu` — the kernel (1 thread = 1 pixel), the sky and the interop. `main.cpp` — options, `--shot`, `--selftest`, GLFW window.
- Constants (`L`, `Q`, initial state) always computed in double in `init_from_zamo`; shading (redshift, temperature, disk noise, black-body LUT, per-sample Reinhard, gamma 2.2) in double.
- **Sky:** the EXR (half RGB) goes into an RGBA16F `cudaMipmappedArray` (14 levels; mips by 2×2 average in float on the host, ~1 s at start-up); `cudaTextureObject` with `Wrap` in u, `Clamp` in v, linear filter + linear mip, 16× anisotropy. Derivatives by 2×2 quads: the 128-thread block is mapped to 16×8 tiles in which **4 consecutive lanes = one quad** (horizontal neighbour = `lane^1`, vertical = `lane^2`); they exchange `(u,v)` with `__shfl_xor_sync` and call `tex2DGrad`. No early `return` (the shuffles are warp-uniform); `Δu` with *wrap* (`d − rint(d)`). If the quad neighbour did not escape, the gradient is 0 (it stays sharp: only affects the edge), same as v0.5b. `--rawsky` = `tex2DLod` level 0 (pure bilinear, like the CPU reference), only to validate geometry.
- Output in a `uchar4` buffer; in interactive mode `cudaGraphicsGLRegisterImage` on a `GL_RGBA8` texture + `cudaMemcpy2DToArray` + `glBlitFramebuffer` (flipped in y).

## Precision: double vs mixed vs float
| Mode | State | Derivatives/increments | tol | Notes |
|---|---|---|---|---|
| `double` | double | double | 1e-8 (atol 1e-9) | identical to the CPU |
| `mixed` | double | float (added to the double accumulator; reprojection in double) | 1e-6 | |
| `float` | float | float | 2e-5 | tol < 1e-6 is unreachable in float (the step controller would never accept): the CLI raises it to 2e-5 |
In every mode the reprojection `(dr/dτ)² = R`, `(dμ/dτ)² = M` is active. **The cost is FP64**, not divergence: the RTX 5080 does FP64 at 1/64 of FP32.

## Warp divergence
Measured in the kernel (`__reduce_max_sync` per warp): efficiency = Σsteps / Σ(max per warp × 32) = **0.99–1.00** in every view. Neighbouring pixels take almost the same number of steps (the image maximum, 472–665, is much larger than the maximum inside a warp). It is not a problem; there was no need to reorder rays.

## Validation (1080p, CPU `--ssaa 1`, 16 threads; `tools/validate_kerr_cuda.py --rawsky`)
Mean absolute error per channel (0–255) and % of pixels with maximum-per-channel error > 4 / > 16, **total**, GPU with pure bilinear sky (`--rawsky`, like the CPU). Ranges over the 9 views (a = 0, 0.9, 0.998 × θ = 80°, 86° with disk; × θ = 90° `--nodisk`):

| Mode | MAE | > 4 | > 16 |
|---|---|---|---|
| double | 0.019 – 0.023 | 0.00 % | 0.00 % |
| mixed (tol 1e-6) | 0.028 – 0.035 | 0.04 – 0.05 % | 0.00 – 0.01 % |
| float (tol 2e-5) | 0.230 – 0.296 | 0.80 – 1.08 % | 0.07 – 0.10 % |

By region (a = 0.998, θ = 86°): double — disk 0.000, edge 0.000, sky 0.021; mixed — 0.000 / 0.000 / 0.031; float — 0.004 / 0.003 / 0.258. **The disk is exact** (the float differences are in the sky: stars shifted by a fraction of a pixel). Full per-view table: `python tools/validate_kerr_cuda.py --rawsky` (output with `uv run --with numpy --with pillow`).

**With the default filter** (mipmaps + anisotropy) against the unfiltered CPU: MAE 0.93 – 1.94 (> 4: 6.5 – 15 %). This is **not a geometric error**: the CPU reference samples the sky with bilinear without mipmaps (aliasing on the stars), the GPU filters; with `--rawsky` the difference disappears (table above). In v0.5b, CPU SSAA 4×4 vs filtered GPU already gave MAE ≈ 0.7 for the same reason.

**a = 0 against v0.5 (Schwarzschild, GLSL compute shader, same quad filter, 1080p, `--rin 6`):**

| θ | CUDA double | CUDA float |
|---|---|---|
| 80° | MAE 0.144, > 4: 0.023 %, > 16: 0.001 % | 0.229, 0.69 %, 0.051 % |
| 86° | 0.147, 0.021 %, 0.000 % | 0.233, 0.69 %, 0.052 % |
(Two independent integrators — fixed φ step in GLSL, adaptive Carter in CUDA — and the same filtered real sky.)

## Performance (1080p, no SSAA, mean of 10 runs after 3 warm-up; filtered real sky)
| View | mean / max steps (double) | double | mixed 1e-6 | float 2e-5 | CPU 16 thr |
|---|---|---|---|---|---|
| a=0, θ=80° | 252 / 560 | 832 ms | 137 ms | 10.0 ms | 2.98 s |
| a=0, θ=86° | 267 / 542 | 884 ms | 145 ms | 10.3 ms | 3.15 s |
| a=0.9, θ=80° | 246 / 472 | 730 ms | 133 ms | 7.1 ms | 2.96 s |
| a=0.9, θ=86° | 261 / 506 | 776 ms | 141 ms | 7.3 ms | 3.26 s |
| a=0.998, θ=80° | 246 / 411 | 727 ms | 133 ms | 7.2 ms | 2.92 s |
| a=0.998, θ=86° | 261 / 487 | 772 ms | 141 ms | 7.2 ms | 3.10 s |
| a=0, θ=90° no disk | 293 / 594 | 946 ms | 142 ms | 10.8 ms | 3.40 s |
| a=0.9, θ=90° no disk | 297 / 601 | 859 ms | 144 ms | 7.6 ms | 3.45 s |
| a=0.998, θ=90° no disk | 298 / 665 | 862 ms | 145 ms | 7.7 ms | 3.46 s |

Speed-up against the 16-thread CPU of omarchy (same work, 1 sample/pixel): **double ≈ 4×, mixed ≈ 22–24×, float ≈ 300–456×**. Against the Mac (≈ 45 s, SSAA 3×3 = 9 samples): per sample ≈ 5 s, so double ≈ 7×, float ≈ 700× (by wall clock, with the GPU without SSAA: 62× / 6000×, not comparable). The GPU render **has no SSAA** (1 sample per pixel): the sky filter handles the aliasing of the stars, not of the disk edges.
`float` with fewer steps (≈ 57 vs 246) is as much tolerance (2e-5 vs 1e-8) as precision. v0.5 (Schwarzschild, fixed step) takes 1.1 ms; adaptive Kerr in float is ~6× more expensive, but with no step approximations.

## Interactive
`v0_6b-kerr-cuda` without `--shot`: GLFW window + CUDA-GL interop, orbital camera as in v0.5 (mouse, wheel), `[`/`]` spin 0 → 0.998 (steps of 0.01, the ISCO follows the spin), `D` Doppler, `G` disk, `+/-` exposure, `↑/↓` sky exposure, `P` double ↔ float, `ESC`. Runs in float by default. Measured with `--autoexit N` (quits and prints the fps): a = 0.9, 1896×1040 window, **104 fps** (kernel 6.8 ms) in float; in double 1.4 fps.
**Not tested by hand** (SSH session without input): mouse, wheel, keys (`[ ] D G + - P ↑ ↓`), resizing; nor did I see the window content (the interop was verified only by: opening, registering the texture, rendering 1200 frames with no CUDA error and the title with the fps). `--vsync` in a window without frame callbacks can block on Wayland.

## Bugs found in use (v0.6c)
André tested the interactive window in float and found two artefacts. **Same root cause: the rotation axis is a coordinate singularity of Boyer-Lindquist** (`φ` is not defined at `sin θ = 0`; `dφ/dτ = L/(1−μ²) + …` has a spike at every pole crossing). Fixed in `kerr.hpp` (CPU) and in the kernel; `tag bhs-v0.6c`.

### Bug 1 — vertical line in the centre of the image
- **Evidence:** `--w 1921` (central pixel with x = 0 exactly ⇒ L = 0): `v0_6-kerr-cpu` a=0.9, θ=80°, y=0.2 gives `φ_end = −0.109` with 336 steps, and `+3.032` for x = ±1e-6, ±1e-4 (the neighbours agree with each other; the L = 0 one is wrong by exactly π). With |L| = 4e-8 (x = 1e-9): **115 147 steps** and a wrong `φ_end`. It was the same in double and float (the line was in the double render), so it was not precision. Of the hypotheses: (a) this was it; (b) sky seam discarded (the line disappears without touching the sky/mipmaps; `Δu` with wrap is right); (c) it was not the *output* pole, it was the ray's passage through the pole.
- **Cause:** rays of the meridian plane cross the axis. With exact L = 0 the 2nd-order integrator in μ reaches μ = ±1, "reflects" and `φ` never gets the jump of π; with a tiny |L| the jump is stiff.
- **Fix:** regularise `|L| ≥ l_min = 1e-4` (`kerr::Options::l_min`; `--lmin` in CUDA). Measured (a = 0, y = 0.2): 1e-5 is still stiff (1100 steps at tol 1e-8, 1.7e5 at 1e-10, wrong result); 1e-4 and 1e-3 give the correct limit (2.494982 rad) with ~450 steps. It shifts the ray direction by `L/(ϖE)`: 2.5e-6 rad (0.002 px) at θ = 80°; 3e-4 rad (0.45 px, the worst case) with the camera at the pole, where ϖ → 0.
- **Before/after (deviation of the central column from its neighbours, mean luminance, sky regions):** 5.3 (float) / 7.9 (double) → 1.6 / 1.6 (control column: 2.65).

### Bug 2 — view from above/below: large black streak and fps dropping
- **Evidence:** float, θ = 0.5° (1080p, a=0.9, no disk): 245 ms/frame (4 fps), **max steps = 100 000, warp efficiency 0.75**, vertical black streak (`Orbiting` rays). In double 530 ms with no black rays. The window: 104 fps → 4 fps.
- **Causes (two, measured separately):**
  1. **Precision in float:** with μ ≈ 1 − 7.6e-5 and a resolution of 6e-8, `1 − μ²` loses its digits and `L/(1−μ²)` becomes noisy (relative error ~1e-3 in `k_φ`); the estimated error of `φ` never converges and the step drops until `max_steps`. Taking φ out of the control (`--phi-atol 1e9`) gave 6 ms but ruined the accuracy (MAE 3.5 vs double at θ = 0.5°; even in double MAE 1.2): discarded. **Solution:** `σ = 1 − μ²` as the 6th state (`dσ/dτ = −2μμ'`, relative precision near the pole), used in `dφ/dτ` and in `M(μ) = σ(Q + a²μ²) − L²μ²`; `μ` and `σ` are redundant and resynchronise after each step (σ < ½: σ rules; otherwise μ rules). Only in the kernel (the CPU in double does not need it).
  2. **Step skipping the spike:** with loose tolerances the step skips the narrow minimum of σ (width `√(σ/|σ''|)`) and the jump of π in φ is lost, **even in double**: `kerr::trace` with tol 2e-5 gave `φ_end` wrong by π (3.138 vs 0.000) at θ = 30° and 5°, and for tol 1e-4 too. The error estimated by the RK stages does not see a spike that falls between them. **Solution:** limit the step to `0.5·√(σ/|σ''|)` when σ < 0.2, with `σ'' = −(2M + μM')` (`kerr::Options::pole_kappa / pole_sigma`); it costs O(log) steps per crossing. With this all cases (θ = 30°, 5°, a = 0, 0.9, tol 1e-8 … 1e-4) agree.
  Ideas discarded with measurement: (2) sin θ weight in the φ error (does not attack the jump), (4) raising the camera's minimum θ (not needed; `kMinTheta` is still 0.5°).
- **ms/frame before → after (1080p, a = 0.9, `--prec float`):** θ = 0.5°: 245 → **7.6**; θ = 3°: 268 → 6.3 (θ = 5°: 7.2); θ = 80°: 7.1 → 8.5 (+20 %: 6th state and the limiter). Error vs double: MAE 0.11 (polar) and 0.19 (θ = 80°; before 0.23). Double: 0.53 → 0.59 s at θ = 0.5° (no black rays), 0.73 → 0.79 s at θ = 80°. Mixed: 110 ms (θ = 0.5°), MAE vs double 0.006–0.011 (before 0.03). Interactive window (float, 1896×1040): θ = 0.5°: ~4 → **98.6 fps**; θ = 80°: 104 → 90.8 fps.
- **CPU verification:** `core` 55 tests green (2 new: plane with L ≈ 0 crosses the pole without a jump in φ; a = 0 — the deflection does not depend on the camera θ, even at 0.5° and 179.5°, to 1e-5); a = 0 vs v0.4 is still at MAE **0.135**/255 (0.015 % > 4); `v0_6-kerr-cpu` vs GPU double in the polar views (θ = 0.5°, 5°; `--profile newtonian`): MAE 0.006.
- **GPU regression** (`--selftest`, in `ctest`): views with odd width and θ ∈ {80°, 30°, 5°, 0.5°, 179.5°}, a ∈ {0, 0.9}, double and float: 0 orbiting rays, max steps < 2000 (measured: 564 double, 208 float) and the direction of the x = 0 ray within < 0.02 rad of the neighbours at ±1 px (measured ≤ 1.3e-2). The GPU↔CPU equivalence now runs with tol 1e-10 (the old selftest, at tol 1e-8, compared two correct integrators with different errors: p99 2.6e-6 in `disk_r`; at 1e-10: 6e-8).
- **Note:** `v0_6-kerr-cpu` now uses Novikov-Thorne by default (v0.7a, another session); the CUDA kernel still had the Newtonian profile, so `tools/validate_kerr_cuda.py` uses `--profile newtonian` on the CPU.

## Limitations
No SSAA; procedural sky not ported (without an EXR the background is black); the `float` mode with tolerance 2e-5 was not tested in extreme views (D < 15, θ < 10°); `max_steps = 200000` (orbiting rays come out black); the 128 block and the tile mapping are fixed (I did not explore other configurations, nor FSAL, nor double's occupancy/registers).

## Reproduce
```
cd code && cmake -B build && cmake --build build -j && ctest --test-dir build
./build/apps/v0_6b-kerr-cuda/v0_6b-kerr-cuda --shot k.png --a 0.9 --theta 80 --prec double            # 0.73 s
./build/apps/v0_6b-kerr-cuda/v0_6b-kerr-cuda --shot k.png --a 0.998 --theta 86 --prec float            # 7 ms
./build/apps/v0_6b-kerr-cuda/v0_6b-kerr-cuda --shot k.png --a 0.998 --theta 90 --nodisk --prec mixed --tol 1e-6
uv run --with numpy --with pillow python tools/validate_kerr_cuda.py --rawsky
```
