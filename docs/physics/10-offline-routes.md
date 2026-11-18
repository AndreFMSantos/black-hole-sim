---
tags: [coding, black-hole-sim, cuda, physics]
created: 2026-11-12
---

# v1.0 — High-quality offline routes

According to roadmap-v0.9-v1.1: define a **route** (motion of the ship *and* of the view) and render it offline to video, with many samples per frame, physical motion blur and 10-bit encoding. On top of the v0.8 pilot (`pilot.hpp`) and the v0.9 disk.

## Route file (JSON)
```json
{ "name": "demo", "fps": 24, "seconds": 4, "ease": "smooth",
  "spin": 0.9, "fov": [{"t": 0, "v": 45}, {"t": 4, "v": 30}], "exposure": 3,
  "ship": {"mode": "orbit", "r": 24, "theta": 82, "phi": 0, "timescale": 6,
           "burns": [{"t0": 1.5, "t1": 2.0, "acc": [0.03, 0, 0]}]},
  "view": [{"t": 0, "yaw": 0, "pitch": 0, "roll": 0}, {"t": 4, "yaw": 20, "pitch": -5, "roll": 0}],
  "disk": {"hr": 0.06, "tau": 4, "turb": 1.2, "spot": {"r": 9, "phi": 200, "size": 1.2, "amp": 5},
           "slowlight": true, "time": {"mode": "rate", "rate": 20, "t0": 0}} }
```
Example in the repository: `code/routes/demo-orbit-burn.json`.

| Field | Meaning |
|---|---|
| `spin`, `fov`, `exposure` | a number or a list of keyframes `{t, v}` (seconds of **video**); outside the ends they are held; between keyframes they are interpolated with smooth easing (`"ease": "linear"` turns it off). The spin can only vary with `ship.mode = "static"` (the ship's dynamics depend on it) |
| `ship.mode` | `orbit` (circular orbit), `infall` (radial fall), `zamo` (starts at rest) or `static` (fixed camera) |
| `ship.timescale` | proper time per second of video |
| `ship.burns` | thrusts `{t0, t1, acc: [fwd, right, up]}`: proper acceleration in c/M in the components of the camera's **default** basis (fwd = −e_r, right = e_φ, up = −e_θ), independent of the view orientation, as short impulses per grid step (same as flight mode) |
| `view` | `yaw/pitch/roll` keyframes in degrees (convention of `flight.hpp`); between them **slerp** of the quaternions (short arc, constant angular velocity) |
| `disk` | the v0.9 disk parameters (`hr, tau, flare, turb, turb_scale, edge_soft, edge_noise, spot, flare_pulse, slowlight`) and the disk time: `pilot` (the ship's coordinate time) or `rate` (M per second of video, from `t0`) |

The ship's trajectory is integrated **once**, on a grid of `1/(fps·16)` s, with the `Pilot` (geodesic + impulses) and interpolated linearly between grid points: deterministic and reproducible.

## Quality
- `N` samples per frame (`--samples`, default 64), sub-pixel jittered (R2) and **each at its own instant** inside the shutter (`--shutter`, default 0.5 of the frame; 0 turns it off): `t_k = (f + ½)/fps + shutter/fps·((k + ½)/N − ½)`. For each sample the ship, the view orientation and the disk time advance: **physical motion blur**. Spin, FOV and exposure stay constant inside the frame.
- Output at **16 bit per channel** (`k_final` also writes a `ushort4` buffer, the same chain: tonemap, bloom, optional grain, before quantising) → `ffmpeg` (`rgb48le`) → **10-bit H.265** (`libx265 -preset slow -crf 16 -pix_fmt yuv420p10le`) or ProRes 422 HQ (`--codec prores`). No denoiser (as the brief asks).
- The resolve parameters (`--tonemap`, `--bloom`, `--cinematic`, `--glare`, …) come from the command line and apply to all frames.

## Execution, resume and concatenation
```bash
build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui --route routes/demo-orbit-burn.json --route-out out.mp4 --w 3840 --h 2160 --samples 256 --chunk 60 [--resume] [--codec prores]
build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui --route r.json --route-frame 40 --route-shot f.png --samples 16   # preview of 1 frame (8-bit PNG)
```
The video is made in **blocks** of `--chunk` frames, each in its own file (`<out>.chunks/chunk_NNNN.mp4`) plus a `.done` marker; `--resume` skips the finished blocks and at the end concatenates with `ffmpeg -f concat -c copy` (no recompression). `--first/--last` render only a range (without concatenating).

## Oracles
Core (`test_json_quat.cpp`, `test_route.cpp`; **120 cases in total**, 15 new):
- **JSON:** numbers, strings with escapes and `\uXXXX`, nesting, and 9 invalid inputs rejected with a position.
- **Quaternions:** a 90° rotation about z takes x to y; **slerp** hits the ends and the angular distance to `a` is `t·total` to 1e-9 (constant angular velocity); by the short arc (`q` and `−q` equal); `from_bases` inverts the rotation to 1e-12.
- **Route:** validation (fps, modes, burns, variable spin only static), keyframes with the ends held, smooth easing (`smoothstep(0.25) = 0.15625`), **the orientation between yaw 0→90° keyframes has angle `π/2` and `π/4` at the midpoint**; a free orbit stays circular to 1e-6 and the proper time is `t_video × timescale` (30 M at 6 s); **E, L, Q conserved outside the burns** (1e-8, Q 1e-6) and **changed** by the burn; the trajectory is deterministic (two identical builds) and the lookup interpolates between grid points; `sample_time`: shutter 0 ⇒ all at mid-frame, mean at the middle, inside the shutter, one time per stratum.

GPU (`--selftest`, section `v1.0`, in `ctest`):
| Test | Result |
|---|---|
| same frame, same route, 2 renders: differing 16-bit values | **0** |
| offline frame (shutter 0) = interactive render (`render(p)`) with the same parameters | **0 differences** |
| 16 bit consistent with 8 bit: `\|round(v16/257) − v8\|` max | 0 |
| motion blur: shutter 1 differs from 0 on a fast orbit (`timescale 300`) | 2.4 % (limit 0.5 %) |
| motion blur preserves the mean luminance | ×0.994 (±5 %) |

**Resume, end to end:** a 96-frame route (4 s, 24 fps, 640×360, 8 samples, 3 blocks); block 1 deleted and repeated with `--resume`: only the 32 frames of that block were re-rendered and **the 96 decoded frames are identical** to those of the original render (`ffmpeg -f framemd5`). The output has `codec=hevc`, `pix_fmt=yuv420p10le`, 96 frames at 24/1.

## Cost (route `demo-orbit-burn`: volume + turbulence + slow light, float)
| Resolution | Samples/frame | s/frame | Remark |
|---|---|---|---|
| 4K (3840×2160) | 64 | **7.5** | measured |
| 4K | 256 | **29.4** | measured |
| 4K | 1024 | ≈ 117 | **extrapolated** (linear in N; not measured) |
| a 30 s clip at 60 fps (1800 frames) at 4K, 256 samples | — | ≈ 14.7 h | extrapolated from 29.4 s/frame |
About 115 ms per sample at 4K in this route; the thin-disk one is much cheaper (see v0.7d: 8.1 ms/sample at 1080p, ×4 at 4K).
**Demo** (`assets/v10-demo.mp4`, outside Git): `routes/demo-orbit-burn.json`, 1080p, 24 fps, 4 s = 96 frames, 32 samples, shutter 0.5, blocks of 24: **78.3 s** of render+encode (0.82 s/frame), HEVC `yuv420p10le`.

## Two defects found by the tests
1. **Accumulator reset before `set_size`:** `render_route_frame` called `accum_reset()` before the first `set_size()`; with a size different from the previous render it only cleared the old area and left garbage in the accumulator (12 620 differing 16-bit values between two renderings of the same frame). The determinism test caught it. The interactive `render()` already did `set_size` first.
2. **`angle_between` with `acos`:** it lost half the digits near 0 (`acos(1 − 1e-16) = 1.5e-8`); it became `2·atan2(|vector part|, |w|)` of the relative rotation.

## Limitations (to declare)
- **No waypoints/spline** (thrusts only) and no "look at" a target (orientation keyframes only): the brief foresaw both; they were done later (v1.0b below).
- The output is **tone-mapped at 16 bit**, not the linear HDR frames (EXR) of the brief; no intermediate frames are saved (only the H.265 blocks) (EXR frames: v1.0b).
- The intermediate blocks are lossy H.265 (`crf 16`); the concatenation is without recompression, so the final video is what x265 produced in each block.
- Slow-light `T_ref` and the time conventions: see [09c2-slow-light](09c2-slow-light.md).
- The `Pilot` state uses the spin at the start; a variable spin is only accepted with a static camera.

## v1.0b — Waypoints (converted into thrusts), "look at" and EXR frames (2026-11-13)

```json
"ship": {"timescale": 30, "max_acc": 0.05, "follow_wn": 0.07,
         "waypoints": [{"t": 0, "r": 40, "theta": 85, "phi": 0}, {"t": 2, "r": 28, "theta": 78, "phi": 40}, {"t": 4, "r": 18, "theta": 70, "phi": 100}]},
"lookat": {"target": "hole" | [r, theta, phi], "roll": 0}
```
Example: `routes/demo-waypoints.json`. `waypoints` and `burns` are mutually exclusive; the first waypoint has `t = 0` and gives the initial position; `t` in seconds of video (proper time = `t · timescale`).

**How it works** (`core/include/bhs/route.hpp`):
1. The waypoints (r, θ, φ) become pseudo-flat Cartesian and are interpolated with a C¹ Hermite spline whose tangents come from the derivative of the 2nd-degree Lagrange polynomial (exact for parabolas, ends included; 2 waypoints = a straight line).
2. The ship is a `Pilot` in Kerr geodesic free fall. At each step, a copy advances in free fall; the difference between the desired local four-velocity at the end of the step (the target's plus `wn ×` the predicted position error, in proper length) and that of the free fall gives the impulse, converted to the ship's rest frame (`rest_frame_impulse`, boost by −β) and applied by `Pilot::thrust`. It is a deadbeat velocity controller with a 1st-order position loop; default `wn` = 2/`timescale`.
3. Report: maximum and rms proper thrust (c/M), maximum position error (M), maximum velocity β, with a **warning if β > 0.7** (strong aberration and Doppler; the stars turn into streaks).
4. `lookat`: the direction is the pseudo-Cartesian straight line to the target, in the ship's (e_r, e_θ, e_φ) components; `up` = −e_θ projected; `roll` in degrees. For `"hole"` it is exactly −e_r (the default basis). **Approximation**: no spacetime curvature (only exact for the centre) and in the ZAMO frame: the aberration from the ship's velocity still shifts the image (the hole leaves the centre when the ship moves fast).

**Oracles** (`test_route.cpp`):
| Test | Measured |
|---|---|
| hover (3 equal waypoints, a = 0, r = 10): thrust = `M/(r² √(1−2M/r))` (closed formula, static observer) | 0.0111802 vs 0.0111803 (5e-6); position error 4.7e-5 M |
| waypoints on a Kerr circular geodesic orbit (a = 0.5, r = 12; φ(t) by `Ω u^t`, BPT 1972): small thrust | rms 7.3e-4–9.8e-4 c/M (~10 % of the hover = 7.6e-3), max error 3.4e-3 M |
| `max_acc` limit; rejection of malformed waypoints / with `burns` | passes |
| `lookat`: centre = default basis; known target; orthonormal, right-handed | passes |
| EXR: float32 write and read back, exact equality; file also read by the reference OpenEXR library | exact (0.001…1500) |

**Errors caught while measuring:** (1) the orbit test failed with `u^t = 1/√(1−3/r+2a r^{-3/2})`: the BPT numerator `(1 + a r^{-3/2})` was missing (a 1.2 % error in φ̇); (2) the 1st-order tangents at the ends left a start-up transient (β_r = −0.0115) and the rms was 5e-3; they moved to 2nd order (rms ÷ 6); (3) **it does not converge to 0 with denser waypoints** (rms 9.8e-4 / 7.9e-4 / 7.3e-4 for 0.5 / 0.25 / 0.125 s): the floor comes from the controller (linear ramp in time), not from the spline, and the test does not claim convergence; (4) the first demo asked for 23 M of displacement in 8 M of proper time (β ≈ 0.95, 4-velocity ≈ 3): a video of streaks and the hole out of the field; hence the β warning.

**EXR frames:** `--route r.json --route-exr DIR [--first F --last L]` writes `frame_NNNNN.exr` (linear float32 RGB, the average of the samples before bloom/tone mapping), resumes by skipping those that exist (writes to `.tmp` and renames). It can be combined with `--route-out`.

Limitations: no speed constraint (report only); `lookat` along a straight line; the ship's coordinate time is not a parameter (the spline is in proper time).
