# v2.5 — HDR10 video (time-lapse with the v2.x versions)

## What "HDR" was before
The 143 s video was 10 bit but **SDR**: no colour metadata (`color_transfer` unknown) and with the brightness compressed by *tone mapping* (ACES, gamma 2.2). The internal pipeline was already linear HDR (sample average, bloom and glare added before tone mapping), but the final result lost the dynamic range in the last step.

## Decisions
- **HDR10 output** = PQ (SMPTE ST 2084) + BT.2020 primaries, 10 bit, with static metadata (mastering display P3-D65 1000 cd/m², MaxCLL 1500 / MaxFALL 35 measured from the luma of the final file (peak 1304 and max FALL 28.3 cd/m² in 364 frames sampled at 2/s; approximate, the PQ luma is not the linear luminance)).
- **`--hdr10 [peak]`** in the renderer (`k_final`): the linear light (Rec.709, after bloom, glare, vignette and chroma) goes to cd/m² with `--hdr-nits` (default **120** cd/m² per unit; chosen so the mid-tones coincide with SDR ACES), soft luminance shoulder from 40 % of the peak up to `peak` (default 1000 cd/m², Y₀ + (peak − Y₀)(1 − e^{−(Y−Y₀)/(peak−Y₀)}), hue preserved), 709→2020 matrix, PQ OETF; no display tone mapping. The `--cinematic` grain is applied in PQ with half the amplitude. (The `--hdr` flag already existed: it exported the EXR/PFM of the HDR average; that is why the new one is called `--hdr10`.)
- **Old segments (CPU and v0.5, 8-bit SDR)** enter the HDR container through `zscale`: gamma 2.2, Rec.709 → PQ, BT.2020, SDR white = 203 cd/m² (BT.2408). They gain no dynamic range (they never had it).
- **GPU segments (v0.6 onwards)** are re-rendered in native HDR (ProRes 422 HQ 10 bit tagged BT.2020/PQ).
- **Text**: 203 cd/m² (grey 0x94 in PQ); 45 % black box.

## Checks
| oracle | result |
|---|---|
| PQ constants: PQ(100) / PQ(203) / PQ(1000) | 0.5081 / 0.5807 / 0.7518 (ST 2084 reference values) |
| `--hdr10` frame vs PQ computed in numpy from the linear EXR of the same frame (`tools/check_hdr10.py`, route-shot s10 frame 100, 480×270, 4 samples) | max difference **1** 8-bit code, mean 1e-4 |
| SDR→PQ conversion of `assemble` (`zscale`, `npl=203`): white / grey 128 / 32 (8 bit) | 203.7 / 44.5 / 2.1 cd/m² (expected 203.0 / 44.6 / 2.1) |
| `ctest` tests | 3/3 |

## Limits
There is no calibrated HDR screen on this computer: the on-screen look was not seen; the checks were numeric. The *tone-mapping* preview (hable) darkens a lot and does **not** represent the HDR. The 1000 cd/m² peak and the 120 cd/m² per unit are choices, not measurements.

## Result (2026-11-14)
`data/timelapse_hdr/full/timelapse_en_hdr.mp4` (inside the project, outside Git): 3840×2160, 30 fps, **182 s** (5460 frames), HEVC Main10, 323 MB, BT.2020/PQ with mastering display and MaxCLL/MaxFALL, English text. Order: v0.1 … v0.9c, **v2.2b** (radio torus, 12 s), **v2.2g** (our GRMHD, 14 s, 201 snapshots linearly blended), **v2.4** (cinematic disk, 13 s), v1.0 (final route, 40 s). The global camera uses the reference duration of 143 s, so segments v0.1–v0.5b (SDR, in an HDR container) were not re-rendered. Render: GPU ~3 h 10 min in the background (s12 at 4K 32 samples ~45 min), assembly ~17 min. Wormhole and Kerr-Newman were not included (no GPU port yet).

## Video v2 (2026-11-15)
`data/timelapse_hdr/full/timelapse_v2_en_hdr.mp4`: 3840×2160, 30 fps, **233 s**, 465 MB, HDR10 (BT.2020/PQ; MaxCLL 1500 / MaxFALL 35 measured, luma peak 1464 cd/m² and max FALL 34.8 in 466 sampled frames). Answers to André's feedback ("the GRMHD is too turbulent", "the disk does not look like Interstellar's; I expected arms/rings"):
- **GRMHD (20 s):** instead of 141 dumps of 50 M at ~500 M/s, a new run from the t = 10 000 M restart with 2 M dumps for 600 M (300 dumps, 0 NaN, rc = 0), shown at 30 M/s with a linear blend of 2 frames per dump. The run was measured at ~5–13 s/M depending on whether the GPU was shared. The dump numbering continues from the restart (201–500).
- **Disk:** v2.4 (filaments, 10 s) and **v2.6** (spiral arms + rings with gaps, 14 s).
- **New variants (SDR in an HDR container):** wormhole with two skies (14 s), charge ramp Q 0 → 1.1 (12 s) and Λ ramp 0 → 0.045 (8 s).
- **Text:** the video's Noto Sans font has no `≈ √ ≤ →`: replaced by `~`, `sqrt`, `<` (checked against the cmap table). The cached file name is `titled_en_hdr.mov` (an `rm` with another name left old clips; always check against `key_en.txt`).
- **Duration per segment:** s11b 12, s11c 20, s11d 10, s11e 14, s11f 14, s11g 12, s11h 8 s; total render ~6 h of GPU (double-precision charge ~20 s/frame; GRMHD ~7 s/frame; final 4K ~45 min).

## Video v3 (2026-11-15, after André's feedback)
`timelapse_v3_en_hdr.mp4`: 170 s (169 s of segments + 1 s tail of the last one, which was rendered with the overlap), 3840×2160, HDR10, English text. Removed: the disk with texture/arms/rings (s11d, s11e; "solid disk with texture applied", no physical basis), the final offline route (s12; "does not make sense to show": the observer motion is the global camera C(t) throughout the video) and the 16K sky (dropped, 2.3× darker). What stayed: v0.1 … v0.9c, radio (torus), own GRMHD (20 s, approved), wormhole, charge and Λ. The removed segments and the 233 s v2 no longer exist as files.

> Note (2026-11-16): GPU segments rendered with a non-zero camera azimuth (`--phi`) between v2.7 and v3.1 ignored it (`k.phi` inside a comment); see `tasks.md`.
