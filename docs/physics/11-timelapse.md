---
tags: [coding, black-hole-sim, timelapse, video]
created: 2026-11-13
---

# v1.1 — Time-lapse of the journey

André's decision (2026-11-12), in roadmap-v0.9-v1.1: **simple chronological style**, a single video in which the black hole evolves from version to version **always from the same perspective**: when the version changes the camera carries on where it was, only what is rendered changes. Target duration 2–3 min; the soft edge goes into the segments with a realistic disk; final format to be decided (suggestion 4K H.265).

## Global camera C(t)
`tools/timelapse/camera.py`: a function of the **global video time only**, the same for all segments.
- θ(t) = 0.5° + 79.5° · smoothstep(t / (0.6 · T)): starts looking from above at the equatorial plane (the view of the 2D simulations) and tilts to 80° at 60 % of the video (85.8 s), staying there;
- φ(t) = 4°/s, continuous rotation at a constant rate (572° in 143 s); distance 40 M and FOV 40° fixed (the hole has the same size in all versions, the same defaults as v0.4–v0.6).
- Tests (`test_camera.py`, 4/4): θ starts at 0.5° and ends at 80°; θ monotonic with derivative ≤ 1.6°/s; φ with a constant rate; the same t always gives the same state (independent of the segments).

## Segments (143 s in total; each renders `dur + 1 s` for the 1 s cross-fade)
| id | start (s) | dur | version | how it is rendered |
|---|---|---|---|---|
| s01 | 0 | 9 | v0.1 Newton and integrators | `v1_1-record-2d --scene newton` |
| s02 | 9 | 8 | v0.2 light in Schwarzschild | `v1_1-record-2d --scene light` |
| s03 | 17 | 9 | v0.3 3D shadow | `v0_3-raytrace-cpu` per frame, 1280×720 SSAA 2 |
| s04 | 26 | 9 | v0.4 accretion disk | `v0_4-accretion-disk` per frame, 1280×720 SSAA 2 |
| s05 | 35 | 8 | v0.5 real-time GLSL | `v0_5-realtime-gpu --nosky --shot` per frame |
| s06 | 43 | 7 | v0.5b NASA real sky | `v0_5-realtime-gpu --shot` per frame |
| s07 | 50 | 12 | v0.6 Kerr (a = 0.9) | `v3_1-kerr-cuda-ui --route` (camera_path), `--profile newtonian` |
| s08 | 62 | 12 | v0.7 Novikov–Thorne | same, `--profile nt` |
| s09 | 74 | 8 | v0.7d/v0.8d cinematic | same, `--cinematic` |
| s10 | 82 | 9 | v0.9b thick disk | same, `--cinematic`, `hr = 0.06, tau = 4` |
| s11 | 91 | 12 | v0.9c turbulence, hot spot, slow light, soft edge | same, full disk; continuous disk time (`t0 = start × 20`) |
| s12 | 103 | 40 | v1.0 offline route (4K) | `v3_1-kerr-cuda-ui --route` with waypoints followed by thrust (v1.0b), `lookat` on the hole, 32 samples, motion blur |

Each clip carries a card (version + key number) with a fade and leaves before the cross-fade; the numbers come from the notes or from measurements (those of s01/s02 and of the route are read from the programs' output: `meta.json`).

## Honest choices and deviations from the specification
1. **"Authentic" source:** the specification said to rebuild each version in a `git worktree` of the tag. Checked with `git diff tag HEAD`: the v0.1, v0.2, v0.3 and v0.5b apps are **identical** to those of the tags; v0.4 and v0.6a later gained `--phi` (azimuth) and `--observer`; v0.3 had no `--phi`, added now (with `--phi 0` the PNG is byte-identical to the original: `cmp`). So the executables from HEAD are used, which are the same physics with the azimuthal camera the time-lapse needs.
2. **CUDA versions (v0.6b to v0.9) = the current executable with the flags of the time** (`--profile newtonian` for v0.4–v0.6, `--cinematic`, `--hr`...), not recompiled from the tags. The earlier milestones recorded "byte-identical default look" in the regressions (ctest `v07d`), but **I did not repeat that comparison with the tags** for the time-lapse. (Since 2026-11-16 the code of each milestone also exists as its own folder: see `AGENTS.md` §3.)
3. **v0.1 and v0.2 were interactive OpenGL windows**: `v1_1-record-2d` uses the same `core` (same integrators, `schw::trace`), the same colours/dashes/horizon, with a software rasteriser (2×2 supersampling). It is not the original OpenGL.
4. **Virtual camera:** in the CUDA segments the camera is not a ship: the new mode `"camera_path": true` (route.hpp) follows the waypoint spline exactly, without dynamics, with a static observer; only the final segment uses the ship with thrust. Test: `route camera_path` (144 core tests).
5. **Resolution:** the container is 4K; the historical segments are rendered at 1080p (CPU at 720p) and upscaled by Lanczos; only s12 is native 4K.
6. The v2.x scenario (radio mode, GRMHD) is **not** included: the specification closes the video on the v1.0 route and says the variants only come in if André asks.

## Errors caught
- **Outdated executable:** the CUDA segments of the first draft showed only stars; `v0_6b-kerr-cuda` had not been recompiled after adding `camera_path` (I only rebuilt the tests target), so the waypoints ran in dynamic mode and the camera "flew". Lesson: rebuild the application target, not only the tests; the full draft caught it before the final render.
- **Missing glyphs:** the font (Noto Sans) has no `→` or `√`: they showed as boxes. The character set of all cards was checked against the font's `cmap` (fontTools); replaced by text.
- **Key number of v0.1:** at 12 s of simulated time Euler is at r = 1.117, not 1.648 (which is 10 orbits at dt = 0.01 in note 01); more steps per frame (32) were used and the card shows the value from the simulation itself (r = 1.349 at t = 48).
- The final route ended inside the disk (r = 13 < r_out = 16), with the whole image wrapped in fog; it now ends at r = 20 and the exposure drops from 3 to 2.2.

## Execution
```bash
cd code && cmake --build build -j8        # ALL targets (v1_1-record-2d, v3_1-kerr-cuda-ui, the old apps)
cd tools/timelapse
python3 test_camera.py                    # camera
python3 render.py --list                  # segments
python3 render.py --draft && python3 assemble.py --draft     # draft (1/4 resolution, 2 samples)
./run_full.sh                             # everything, resumable; log in data/timelapse/full.log
```
Full draft (480×270): rendering 5 min 40 s, assembly 2 min 3 s, a **143.000 s** video (correct `xfade` offsets), 1080p H.265 10 bit.

## Result (2026-11-13)
`data/timelapse/full/timelapse.mp4` (inside the project, outside Git): **3840×2160, H.265 10 bit (hvc1), 30 fps, 143.000 s, 499 MB**. Verified with `ffprobe` and with 13 frames extracted along the video (one per segment and one in the cross-fade): chronological progression, continuous camera (from above to the final tilt, always rotating), legible cards, no missing-glyph boxes, final route ending outside the disk.

Real times (RTX 5080 shared with the AthenaK run, which slowed from 6.6 to ~7 s/M while this ran): segments s01–s11 ≈ 40 min (includes two restarts for optimising the 2D recorder); **s12 (4K, 32 samples, 1200 frames) ≈ 1 h 36 min, ~5.9 min per block of 60 frames**; clips with cards ≈ 4 min; final x265 `slow` crf 14 encoding ≈ 12 min. Total ≈ 2 h 40 min; work area 33 GB (deletable: `data/timelapse/full/s*` except the `.mp4`).

Two performance problems solved midway (image verified equal before and after): the light scene of `v1_1-record-2d` took ~10 s/frame (captured rays with hundreds of thousands of sub-pixel segments → the ones shorter than 0.5 px are merged; MAE 1.2/255 from alpha accumulation, same structure), and then ~3.3 s/frame with the rotated camera (the bounding box of a diagonal line is almost the whole screen → only the strip of each segment is traversed; difference 0 and ~0.12 s/frame).

To be decided (André): keep 4K (or 1080p), royalty-free music (the video is silent), whether any segment should last longer/shorter, and whether the v2.x variants (radio/GRMHD) come in as an extra segment before the final route.

## Camera orbital speed: is it physically reasonable? (André's analysis, 2026-11-13; script `scratchpad/orbit_phys.py`)
**Conclusions:**
1. **The speed itself is possible and even modest.** A prograde Kerr circular geodesic orbit (a = 0.9) at r = 40 M has Ω = 0.00394 rad/M, period 1595 M and **v = 0.161c** (γ = 1.013), in free fall, without thrust. The video's 4°/s rotation is that orbit if **1 s of video = 17.7 M**; one turn takes 90 s of video. Closer is faster: r = 33: 0.177c; r = 26: 0.200c; r = 20: 0.229c.
2. **If the observer were a real ship** (the tables depend on the mass; M = GM/c³): for M87\* (6.5e9 M☉) 1 s of video = 5.7e5 s ≈ 6.6 days, the tides are ~6e-14 m/s² (null); for Sgr A\* 1 s = 361 s; for 10 M☉ r = 40 M is 591 km, the orbit lasts 0.079 s and the tide on a 2 m ship is 2.6e4 m/s² (≈ 2600 g: lethal). **Physically viable only for supermassive black holes.**
3. **The final route (v1.0b) is NOT a geodesic**: it needs a maximum thrust of 0.01255 c/M (rms 0.0016) to follow the waypoints. In g: M87\*: **12 g (rms 1.5 g)**, bearable for a trained crew for seconds; Sgr A\*: 1.9e4 g; 10 M☉: 7.8e9 g. So the video is plausible as a "ship" only for M87\*-like black holes.
4. **Inconsistency to declare:** in the CUDA segments the camera is virtual with a **static observer** (β = 0): there is no aberration or Doppler from the motion. A ship at 0.16c would have a maximum aberration of ~9° and would change the brightness of the approaching side (the `--observer orbit` option exists). Only the final route includes the ship's velocity (β max 0.17).
**About "it looks fast":** it is a pacing choice (90 s per turn), not an impossibility; speed and perceived speed are different things. **Suggestions for a future re-render (not done):** φ at 2–2.5°/s (turn in 144–180 s); final route closer to a geodesic (less thrust, e.g. `max_acc`); camera with `--observer orbit` to include aberration/Doppler.

## English version (2026-11-13)
The titles are only assembly text (drawtext), so **no frame needs re-rendering**: `assemble.py --lang en` recreates the clips with cards and the final video (`timelapse_en.mp4`; the clips stay cached as `titled_en.mov`, the frames/`seg.mov` are not touched). It runs on the CPU only (ffmpeg, `nice -n 19`, 6 threads), without the GPU. The Portuguese video is kept.

> Note (2026-11-16): GPU segments rendered with a non-zero `--phi` between v2.7 and v3.1 ignored the camera azimuth (bug: `k.phi` inside a comment); the GPU segments of the v3 video need re-rendering (see `tasks.md`).
