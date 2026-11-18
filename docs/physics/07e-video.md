---
tags: [coding, black-hole-sim, cuda, video]
created: 2026-11-12
---

# v0.7e — Video (1080p60, `--video`)

`v0_6b-kerr-cuda --video out.mp4 --clip spin|orbit|infall` renders the frames **in memory** (GPU → RGB24 on the host) and pipes them (`popen`) to ffmpeg: `-f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 60 -i - -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart`. I chose the pipe (and not a script that calls `--shot` per frame) because it avoids 600–720 process launches, sky-map reloads (0.8 s each) and the intermediate PNG; the sky and the LUT stay resident. Options: `--fps 60`, `--seconds S`, `--crf 18`, `--samples N` (default 1; I used 8), `--cinematic`.
Float (`tol 2e-5`) by default in video mode, NT profile, 1080p.

## Clips (`assets/`)
| File | Clip | Frames | Look | Size | Render+encode time | ms/frame (GPU) |
|---|---|---|---|---|---|---|
| `v07e-spin-a0-0998.mp4` | **A**: `a` 0 → 0.998 (smoothstep), θ = 80°, D = 40, fixed camera, 10 s | 600 | cinematic | 7.5 MB | 51.0 s | 85.0 (66.0) |
| `v07e-spin-a0-0998-physical.mp4` | A, physical look (reinhard, no bloom, beaming 1) | 600 | physical | 10.7 MB | 50.8 s | 84.7 (65.6) |
| `v07e-orbit-a09.mp4` | **B**: a = 0.9, D = 40, θ 100° → 60° (smoothstep: from below, from the side, from above) and φ 0 → 360°, 10 s | 600 | cinematic | 21.1 MB | 40.3 s | 67.1 (63.6) |
| `v07e-infall-r40-r4.mp4` | **C**: radial infall r = 40 → 4, a = 0.9, θ = 75°, `--observer infall` updated per frame (β = −√(2/r): 0.224 → 0.707), 12 s | 720 | cinematic | 7.3 MB | 43.8 s | 60.8 (57.5) |
All 1920×1080, 60 fps, H.264 yuv420p, CRF 18, `+faststart`, 8 samples/frame. Verified with `ffprobe` (`h264`, `yuv420p`, 60/1 fps, 12.000 s in clip C). Rendering is ~4–5× slower than real time (16.7 ms/frame); none exceeds 40 MB (they stay in Git). In clip A the CPU spends ~19 ms/frame rebuilding the NT LUT (the spin changes every frame) and copying/sending the frame; that is why the wall clock (85 ms) exceeds the GPU (66 ms).

## Notes per clip
- **A (spin):** the disk shrinks (ISCO 6 → 1.24) and gets brighter as `a` rises; at `a → 0.998` the receding side is almost entirely dark (physical) — the `-physical` clip shows this without the aesthetic beaming. The shadow goes from a circle to a D shape.
- **B (orbit):** φ from 0 to 360° rotates the sky around the axis (the metric is axisymmetric: only the star field and the disk pattern turn); what changes the geometry is θ. The file is the largest (21 MB) because the moving star field does not compress.
- **C (infall):** **uniform proper time**, `r(τ) = (r₀^{3/2} − (3/2)√2 τ)^{2/3}` (free fall from rest at infinity, Schwarzschild), with τ_total = 115.5 M mapped to 12 s (≈ 9.6 M of proper time per second): it starts slow (r = 40) and accelerates towards the end (β = 0.707 at r = 4). β = √(2/r) is exact in Schwarzschild and **approximate** in Kerr (the ZAMO is not static). The "disk filling the screen" look is aberration + the fact that the camera (θ = 75°) sits inside the outer radius of the disk (16) from r < 16.

## Limitations
Static in time (each frame is an instant; the disk pattern does not rotate in coordinate time, only the camera/`a` vary); the fall does not cross the horizon (stops at r = 4, horizon at 1.44); the beaming in the cinematic clip is aesthetic (`--beaming 0.5`). **To check by hand:** the `.mp4` files were verified with `ffprobe` and by extracted frames (contact sheet), not watched in continuous playback (temporal flicker between frames, bloom jitter).
