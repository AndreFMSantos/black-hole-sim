---
tags: [coding, physics, black-hole-sim]
created: 2026-11-11
---

# v0.5b — NASA real HDR sky, with filtering

Replaces the procedural sky with a real map, on the CPU (`v0_4-accretion-disk --sky`) and on the GPU (`v0_5-realtime-gpu`, by default if the file exists). Sequel to [05-realtime-gpu](05-realtime-gpu.md).

## The map
NASA SVS "Deep Star Maps 2020" (`starmap_2020_8k.exr`): OpenEXR half RGB, 8192×4096, linear, plate carrée, ICRF/J2000, based on Hipparcos, Tycho-2 and Gaia DR2. **Not in Git** (125 MB, `.gitignore`): `code/tools/fetch_sky.sh` downloads it and checks the sha256 (`dc6c4f41…9ff7c4`, confirmed). Credits: *NASA/Goddard Space Flight Center Scientific Visualization Studio; Gaia DR2: ESA/Gaia/DPAC* (see `assets/sky/README.md`).

## Direction → (u, v) mapping (`bhs/skymap.hpp`)
World: `+x` = RA 0h, `+y` = RA 6h, `+z` = north celestial pole. `RA = atan2(y,x)`, `dec = asin(z)`.
`u = 0.5 − RA/2π` (RA 0h at the centre, **grows to the left**; wrap in u), `v = 0.5 − dec/π` (top = dec +90°; clamp in v).
Tests: `+x → (0.5, 0.5)`, `+y → u=0.25`, `+z → v=0`, round trip direction→uv→direction (< 1e-12), exact bilinear at texel centres, linear between them, wrap at the seam, half→float (1, −2, 65504, subnormal, inf), and reading the real EXR (skipped if the file does not exist). 37 `core` tests green (28 + 4 from v0.5 + 5 from the sky).
Reader: tinyexr v1.0.8 (FetchContent with a non-existent `SOURCE_SUBDIR`, download only; `tinyexr.cc` + `miniz.c` are compiled into the `bhs_exr` lib; the project moved to `LANGUAGES C CXX`). It reads in **half** (201 MB as RGB16F, instead of 805 MB as float RGBA). Not yet compiled on the Mac.

## HDR exposure (`sky-stats`)
Luminance distribution (solid-angle weighted; the map is normalised to a maximum of **1.0**, stars saturated):

| Region | p50 | p90 | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|
| whole sky | 0.0015 | 0.0105 | 0.036 | 0.19 | 0.99 |
| Milky Way (|b|<5°) | 0.0096 | 0.027 | 0.069 | 0.32 | 1.0 |
| high latitude (|b|>60°) | 0.0002 | 0.0020 | 0.010 | 0.13 | 0.97 |

Fraction of the sky whose luminance×S exceeds 0.5 (the disk brightness before tone mapping): S=1: 0.036 %, **S=2: 0.075 %**, S=3: 0.11 %, S=4: 0.15 %.
**Choice: `--sky-exposure 2.0` by default** (CPU and GPU). With 1 the Milky Way is already visible (dust); with 3 the background brightness saturates. With 2 the Milky Way is clearly visible and only the cores of the brightest stars (0.075 % of the area) exceed the disk. Key **↑/↓** (×1.1) on the GPU adjusts the sky exposure, independent of the disk's `+`/`-`. The sky goes through the same per-sample Reinhard.

## Filtering
Lensing compresses the sky very anisotropically in the shadow and the Einstein ring, and the texture is 8192 px wide: without a filter there is aliasing and flickering brightness. Three modes (`--sky-mode`):
- **1 — two extra rays per pixel** (neighbours in x and y; fallback to the opposite neighbour if one does not escape): `textureGrad` + `GL_LINEAR_MIPMAP_LINEAR` + 16× anisotropy. Reference, costs 3 traces: **~3.0 ms**.
- **0 — 2×2 quad (default)**: the 4 threads of a quad (`GL_KHR_shader_subgroup_quad`) swap the sky `uv` and get the derivatives like a fragment shader's `dFdx/dFdy`, **with no extra rays**. Threads are remapped to 2×2 blocks (`gl_LocalInvocationIndex`) and there is no early `return` (quads must stay complete). If the quad neighbour did not escape (disk/capture), the gradient is 0 (it stays sharp: it only affects the edge). Difference from mode 1: MAE 0.018, 0.006 % of pixels > 16. Cost **≈ the same as not filtering**.
- **2 — no filter** (`textureLod` level 0): only for comparison.
Mipmaps by `glGenerateTextureMipmap` (14 levels) over linear HDR values. `u` with `GL_REPEAT`, `v` with `CLAMP_TO_EDGE`. Dependency: the `gl_LocalInvocationIndex` → quad mapping assumes the subgroup is made of consecutive local indices (true on NVIDIA with local size 64); it is validated by equality with mode 1.

## Cost (1080p, RTX 5080, mean of 99 frames; before 0.735 ms with the procedural sky)
| View | mode 0 (quad) | mode 2 (no filter) | mode 1 (extra rays) |
|---|---|---|---|
| A θ=80°, D=40 | 1.133 ms (883 fps) | 1.135 ms | 2.99 ms (335 fps) |
| B θ=86° | 1.195 ms (837 fps) | 1.196 ms | 3.17 ms |
| C θ=61°, φ=86°, D=40 (Milky Way) | 0.989 ms (1011 fps) | 0.989 ms | 2.40 ms |
The real sky costs ~+0.4 ms over the procedural one (accesses to the 201 MB texture), quad filtering costs ≈ 0. Loading the EXR (ZIP) + upload + mipmaps takes ~0.8 s.

## CPU (SSAA 4×4) vs GPU (mode 0) validation, 1920×1080, same parameters
Regions by mask (GPU render with `--sky-exposure 0`): disk = interior (eroded 3 px), edge = ±3 px around the disk/sky boundary, sky = the rest (includes the black shadow). `code/tools/compare_cpu_gpu.py --mask`.

| View | region | % pixels | MAE (0–255) | > 4 | > 16 |
|---|---|---|---|---|---|
| A θ=80° | total | 100 | 0.666 | 2.07 % | 0.26 % |
| | disk | 16.0 | 0.030 | 0.02 % | 0.01 % |
| | edge | 1.7 | 4.22 | 23.9 % | 9.4 % |
| | sky | 82.3 | 0.715 | 2.01 % | 0.12 % |
| B θ=86° | total | 100 | 0.720 | 2.30 % | 0.30 % |
| | disk | 10.1 | 0.069 | 0.03 % | 0.02 % |
| | edge | 1.9 | 4.38 | 25.1 % | 9.9 % |
| | sky | 88.1 | 0.715 | 2.07 % | 0.12 % |
| C θ=61°, φ=86° | total | 100 | 0.391 | 1.22 % | 0.19 % |
| | disk | 33.0 | 0.008 | 0.00 % | 0.00 % |
| | edge | 1.8 | 2.99 | 20.7 % | 6.6 % |
| | sky | 65.3 | 0.514 | 1.31 % | 0.11 % |

**Effect of the filtering** (same CPU reference): sky error in A without filter 1.668 (10.5 % > 4) → with filter 0.715 (2.0 % > 4); B 1.720 → 0.715; C 1.053 (4.8 % > 4) → 0.514 (1.3 %). The disk edge is SSAA error (the GPU has 1 sample/pixel), as in v0.5. Visually (`assets/v05b-filter-crops.jpg`: CPU 4×4 / filtered GPU / unfiltered GPU, 4× enlargement, next to the shadow in view C): the unfiltered one shows moiré in the ring streaks; the filtered one is smooth and equal to the reference. The rest of the sky error is point stars (SSAA 4×4 does a different integration).

## Limitations
- With FOV 40° at 1080p there are ~2.1 screen pixels per texel (22.8 texels/degree against 48 px/degree): away from the lensing the sky gets slightly soft, and a larger zoom would call for the 16k map (just swap the file).
- The camera fixes `up = +z` celestial: the Milky Way tilts ~60° relative to the screen when looking at the galactic centre (view C: θ=61.064°, φ=86.405°).
- Quad: gradient 0 where the neighbour does not escape. Not tested: Mac (no compute shaders for the GPU; `v04 --sky` + tinyexr should compile, to be verified), another GPU vendor, real-time interaction with the ↑/↓ key.
- Very bright stars: the map saturates at 1.0, so there is no "real" HDR above that.
