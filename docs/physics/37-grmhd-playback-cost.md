---
tags: [coding, black-hole-sim, performance, grmhd]
created: 2026-11-17
status: measured and optimised (O5, O6c)
---
# GRMHD playback cost (O5)

Tool: `v4_0-metric-api --bench-playback DIR [--play-first N --play-frames N --play-rate dumps/s --play-fps F --play-nopf] --w W --h H`
(the UI's play loop without a window: the dump clock advances by rate/fps per frame, a new snapshot is needed when its integer part changes, and each frame runs
`blend_step` + `render`). `BHS_TIMING=1` adds a breakdown of every snapshot upload. Data: `data/athenak/run_d/bgm` (201 dumps, 145 MB each), 1280×720 float, MAD, a = 0.94.

## Before (commit d9ca269)

| Case | blend_step on a snapshot change | render | frame mean / p95 / max |
|---|---|---|---|
| 10 dumps/s, prefetch on | 17.5 ms | 37.9 ms | 40.7 / 55.3 / 59.9 ms |
| 30 dumps/s, prefetch on | 17.6 ms | 37.3 ms | 46.1 / 55.6 / 62.6 |
| 30 dumps/s, prefetch off | 28.4 ms | 37.2 ms | 51.5 / 67.0 / 73.6 |
| 10 dumps/s, prefetch off (cold files) | 48.4 ms | 36.4 ms | 44.3 / 84.4 / 91.7 |
| 2 dumps/s, prefetch on | 17.5 ms | 36.0 ms | 36.5 / 36.2 / 54.3 |

The prefetch hides the file read and parse (13 ms) well. What stays on the render thread is the upload, `BHS_TIMING` breakdown of one upload (17.5 ms):
**repack 11.1 ms**, memcpy3D 5.5–6.0 ms, free 0.4, malloc3D 0.3, texture 0.0.

## Change 1 (O6c): float copies of the double parameters

The GRMHD kernel's top pipeline was FP64 (53.6 %, 259 M FP64 instructions at 720p): `float(p.g_startx1)`, `F(p.nu_obs)`… convert a `double` member of `KP` at run
time, in every sample, and a double → float conversion runs on the FP64 pipe. `struct PF` holds float copies (75 fields), filled once at the end of `make_kp`
with the same host conversion; device code reads `p.f.x` (or `pcv<F>(p.x, p.f.x)` where the type is generic). FP64 instructions 259 M → 94.7 M, and FP64 is no longer
the top pipeline of the kernel (59 % issue slots, "Shared FMA heavy" 34.7 %). Images byte-identical (layer, torus MD5), `ctest` 37/37.

| Scenario | O6a+b | + O6c |
|---|---|---|
| RIAF 700² | 11.998 ms | 9.817 (×1.22) |
| torus 700² | 8.310 | 7.197 (×1.15) |
| layer 1080p | 5.556 | 4.856 (×1.14) |
| GRMHD 720p | 31.866 | 29.840 (×1.07) |

(Intermediate attempt: converting only `float(p.x)` and not `F(p.x)` made the RIAF 9 % slower, 12.0 → 13.2 ms, with fewer registers: a scheduling effect, not occupancy. Doing both cleared it.)

## Change 2: the repack moves off the render thread

The 11 ms was a single-threaded strided gather (12 floats per cell → 3 float4 arrays, 4.7 M cells) on the render thread. `grm_pack` does it now on the prefetch worker
(next to the parse) and, when there is no prefetch, in 3 threads (one per array). The staged buffers are used by `grmhd_load_next` only if the worker produced them for
that path and the half-float setting did not change.

| Case (30 dumps/s unless noted) | Before | After |
|---|---|---|
| blend_step on a snapshot change, prefetch on | 17.5 ms | **7.9 ms** |
| frame mean / p95 / max | 46.1 / 55.6 / 62.6 | 40.4 / 44.3 / 63.3 |
| 10 dumps/s: frame mean / p95 / max | 40.7 / 55.3 / 59.9 | 37.6 / 44.2 / 46.7 |
| prefetch off (10 dumps/s) | 48.4 ms | 32.8 ms |
| prefetch off (30 dumps/s) | 28.4 ms | 31.8 ms (file read now in the same call, cached) |

Check: `--selftest-blend data/athenak/run_d/bgm` on real dumps, all 7 criteria pass (tau 0 / 1 equal the static snapshots, also after the swap).
Note: the ctest `v4_0_grmhd_blend` passes in 0.4 s because it runs without that data directory (it is skipped); run it by hand with the directory above.

## What is left

- Render is 36–37 ms at 720p in blend mode against 29.8 ms for one snapshot: sampling two snapshots costs +22 %. It dominates (about 27 fps), not the swap.
- Remaining swap cost (7.9 ms, every 1/6 to 1/2 frames): memcpy3D from pageable memory (6.9 ms). Options: pinned staging buffers and an asynchronous copy into a third texture set while the previous frames render (+226 MB VRAM). Not done; the spike is now 20 % of a frame.
- The GRMHD kernel is now FP32-issue bound (59 % issue slots, 112 registers, occupancy 32 %), so further gains need work on the sampling math, not on FP64.


## O5b (sprint S5, 2026-11-18): async upload into a third texture set

After O5 the swap still cost 7.4 ms on the render thread (`BHS_TIMING`: memcpy3D from pageable memory 6.8, free 0.3, malloc3D 0.2). Now the prefetch worker packs the next snapshot into **pinned** staging buffers (`cudaMallocHost`, 3 × one float4 volume) and enqueues `cudaMemcpy3DAsync` into a **spare** texture set on its own stream, with an event; `grmhd_load_next` only waits for the event and rotates `next <-> spare` (pointer swap; texture objects are created once per array). The old first snapshot becomes the new spare, protected by an event recorded on the default stream at the rotation (the next copy waits for the last frame that read it). Cost: +226 MB VRAM (one texture set) and +226 MB pinned host memory.
`BHS_NO_ASYNC_UPLOAD=1` restores the synchronous path (kept for the first load, cold snapshots, `--play-nopf` and any failed allocation: the pinned/spare allocation falls back instead of failing).

| 1280×720 float, MAD a = 0.94, warm files | blend_step p50 | frame mean | frame p50 | frame p95 |
|---|---|---|---|---|
| 10 dumps/s, synchronous upload | 0.01 ms (7.8 ms on the 8 swap frames) | 38.20 ms | 35.60 | 43.21 |
| 10 dumps/s, **async** | 0.00 ms (max 0.22) | **35.21 ms** | 35.11 | **35.85** |
| 30 dumps/s at 25 fps (1.2 dumps/frame), synchronous (O5) | 7.65 ms | 54.9 ms | 43.6 | 105.2 |
| 30 dumps/s at 25 fps, **async** | 0.17 ms | **48.8 ms** | **36.3** | 106.1 |

The mean frame is now the render time (35.7 ms) plus whatever the slow path costs. At 30 dumps/s a frame sometimes needs two new dumps (1.2 per frame): `blend_step` then cannot swap and falls back to `grmhd_load` + `grmhd_load_next` (≈ 70 ms, the p95); the async upload does not touch that path (O5c below).
Correctness: `--selftest-blend` (7 criteria: tau = 0 / 1 reproduce the static snapshots, after the swap no stale texture) passes with both paths; ctest 38/38.
O5c (not done): when the dump index jumps by 2, rotate the prefetched snapshot in as the *first* one (it is already on the GPU) and load only the second synchronously: ≈ 35 ms instead of 70.
