---
tags: [coding, physics, black-hole-sim]
created: 2026-11-11
---

# v0.5 — Real time in a compute shader (GLSL 4.6)

Same physics as v0.4 ([04-accretion-disk](04-accretion-disk.md)), ported to a compute shader: **one thread per pixel**, `shaders/blackhole.comp`. Linux/omarchy only (macOS stays on OpenGL 4.1, with no compute shaders; the target only exists `if(NOT APPLE)`).

## Design
- **Pipeline**: the compute shader writes `RGBA32F` (already with per-sample Reinhard tone mapping and gamma 2.2) → `glBlitNamedFramebuffer` to the screen. In `--shot`, the image is read with `glGetTextureImage` and converted to 8 bit with the same rounding as the CPU (`v*255+0.5`).
- **Per pixel**: direction `d` → `b = D sinα/√(1−2/D)` → `trace_with_disk` (RK4 in φ on `u'' = −u + 3u²`, step 0.015 rad against 0.005 on the CPU, `maxPhi = 100`) with early exit (capture `u ≥ 1/2`, escape `w<0 ∧ u≤0`, disk hit) → `g` with Doppler → `T_obs = g·T_peak·profile(r)` → black-body LUT → disk noise; otherwise stars with direction `cosφ_end ĉ + sinφ_end t̂`.
- **LUT**: `bhs/lut.hpp` computes on the CPU 1024 log-spaced entries (500–100000 K) with `blackbody::rgb`; it goes into a 1D `RGB32F` texture, and the shader does `texture()` with `GL_LINEAR` at `(x+0.5)/1024`, the same interpolation as v0.4.
- **Hash/stars/noise**: ported from `stars.hpp` and v0.4's `vnoise` with `uint` (mod 2³² arithmetic is identical). No grid.
- **Camera** (`bhs/camera.hpp`, tested in `core`): `ĉ = (sinθ cosφ_c, sinθ sinφ_c, cosθ)`, `fwd = −ĉ`, `right = fwd×z`, `up = right×fwd`. With `φ_c = 0` it is exactly v0.4's; the azimuth rotates the basis around z. Mouse drag = θ/φ, wheel = D (min 10, max 400), `D` Doppler, `G` disk, `+`/`-` exposure, `ESC`.
- **Loader**: `gl_loader.hpp`, ~35 functions via `glfwGetProcAddress` with the `PFNGL*PROC` types from `<GL/glcorearb.h>` (Khronos, ships with Mesa). No glad/glew, nothing to generate or vendor. A new function = one line in `BHS_GL_FUNCS`.
- The FetchContent GLFW (3.4) is used as in the other phases; it links on Wayland/Hyprland without problems.

## CPU vs GPU validation
`v0_4-accretion-disk --ssaa 1` vs `v0_5-realtime-gpu --shot`, 1920×1080, same parameters, script `code/tools/compare_cpu_gpu.py` (error in 0–255 levels).

| View | MAE (R,G,B) | pixels with error > 16 |
|---|---|---|
| θ=80°, D=40 | 0.004, 0.003, 0.003 | 0.001 % |
| θ=10°, D=40 | 0.004, 0.004, 0.003 | 0.002 % |
| θ=4°, D=40 | 0.004, 0.004, 0.003 | 0.002 % |
| θ=88°, D=15 | 0.003, 0.003, 0.003 | 0.001 % |
| θ=60°, D=12 | 0.005, 0.004, 0.004 | 0.003 % |
| θ=30°, D=100 | 0.003, 0.003, 0.003 | 0.000 % |
| θ=80°, D=40, `--nodoppler` | 0.004, 0.003, 0.003 | ~0.001 % |
| θ=80°, D=40 vs CPU **SSAA 3×3** | 0.173, 0.142, 0.119 | 0.296 % (0.798 % > 4) |

With the same sampling (1 sample/pixel) the images are practically identical: RK4 is 4th order, so step 0.015 in `float` gives the same pixel as 0.005 in `double`; the few differences are stars/disk edges. Against the 3×3 SSAA the expected differences show up at the edges (anti-aliasing that the GPU does not do).

## Performance (RTX 5080, driver 615.78, `glFinish` per frame, mean of 99 frames)
| Scene | ms/frame | fps |
|---|---|---|
| 1080p, θ=80°, D=40 | 0.735 | 1361 |
| 1080p, θ=65°, φ=40°, D=25 | 0.525 | 1904 |
| 1080p, θ=88°, D=15 | 0.50 | 1996 |
| 1080p, θ=60°, D=12 (worst case measured) | 0.79 | 1271 |
| 4K (3840×2160), θ=80° | 2.84 | 352 |
| 1080p with step 0.005 (as on the CPU) | 2.11 | 474 |
| Real 1280×720 window in Hyprland (title) | 0.73 | ~1378 (no vsync) |

Compared with the CPU on the same omarchy (16 threads): SSAA 1 = 0.77 s → **~1050× faster**; SSAA 3×3 = 6.93 s (the "~12 s" were the Mac, 11 threads) → ~9400×. Target ≥60 fps: largely met (≈22× above).

## What broke / notes
- Nothing relevant broke: it compiled and was right the first time. `hyprctl`/`grim` need `HYPRLAND_INSTANCE_SIGNATURE`; `grim` hung in the SSH session and was killed by PID.
- Stars have `d2 < 0.018` in cells: in `float` the direction comes out with ~1e-7 error; enough for a minority of stars to differ slightly.
- I did not use `double` on the GPU. Near the photon orbit (b ≈ √27) rays go round many times; with `maxPhi = 100` the cost is bounded, and `step` and `maxPhi` are arguments (`--step`, `--maxphi`).

## Limitations / to verify
- Mouse and keyboard interaction was **not tested by hand** (SSH session without input): it was verified that the window opens, renders at >1000 fps and the title shows fps/ms; the camera maths is in unit tests, and the azimuth view (θ=70°, φ=50°) was inspected visually. To confirm: drag direction (chosen convention: dragging right makes φ decrease) and window resizing.
- The screen was not compared by eye with a compositor screenshot (`grim`).
- Same physical model as v0.4 (Schwarzschild, thin disk, Newtonian T). Kerr is v0.6.
