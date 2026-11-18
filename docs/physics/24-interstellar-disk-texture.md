# v2.4 — filamentary texture of the thin disk and glare fix (2026-11-14)

Origin: André compared our disk (v0.7d/0.8d) with the *Interstellar* Gargantua ("ours looks fake, smooth, with no detail") and saw flashes of blue streaks in the clip `v08d-spin-polish.mp4`.

## 1. Why our disk looked smooth
The stationary Novikov–Thorne only gives T(r): it has no azimuthal structure. The texture we had was a smooth modulation (`k = 0.55…1.15`, 2 octaves, frequencies 3/9 in azimuth and 9/28 in ln r, with a fixed spiral). In the film (DNGR, James et al. 2015, [memory]) the geometry is physical but the disk is artistic, with texture; the look comes from **fine azimuthal filaments** stretched by the differential rotation and magnified by the lens.

## 2. `--disk-tex amp[,fr,fphi,oct]`
**Thin** disk (default mode): `k = exposure · exp(amp · fbm)`, fbm of `oct` octaves (weight ×0.62, frequency ×2) of 3D value noise at `(fphi cos ψ, fphi sin ψ, fr ln r)`, with `fr ≫ fphi` (filaments stretched in azimuth) and `ψ = φ − 2.2 ln(r/r_in) − Ω(r) T` (Keplerian shear with the disk time `--time T`, Ω = 1/(r^{3/2}+a)). With `amp = 0` (default) the old code runs unchanged (the 3 `ctest` tests pass, incl. `v07d_regression`).
Example: `--cinematic --a 0.95 --theta 82 --dist 30 --fov 34 --rout 40 --exposure 4 --tpeak 7500 --bloom 0.35 --time 400 --disk-tex 1.5,60,5,6` → (image dropped; command kept for reference; 1080p, 24 samples, 27 s). **It is aesthetic**: the amplitude comes from no physical model; what is preserved is the geometry, the frequency shift and the differential rotation.
Limits: textures at the level of value noise (cubic grid visible by eye when zooming), phase kept only up to the chosen time (no slow light on the thin disk), no diffraction; cost equal to the old one.

## 3. Blue flashes (v0.8d): cause and fix
- **Where:** `assets/v08d-spin-polish.mp4`, frames 1, 7, 9, 33, 34, 38, 50, 66, 82, 96, … (isolated, clean neighbours). The main 143 s video (4290 frames, measured at 320×180) does **not** have them; they came from the separate clip.
- **Reproduction:** the regenerated clip (`--clip spin --fps 60 --seconds 6 --samples 8 --cinematic`) gives the **same** frames (deterministic).
- **Cause:** it is the star-shaped glare. With `--glare 0` they all disappear; with `--accum ldr` too. The photon ring and the disk core are compact and much brighter than a star, pass the *top-hat* filter and the streaks (0/60/120°) cross the shadow, in blue because the source is blue (approaching side). A *clamp* ceiling on the source does not fix it (the streaks stay, 0.040 of the image in blue vs 0.044); excluding the sources above `--glare-cap` (smooth between cap and 2 cap) fixes it: **cap = 1.5** leaves 0.0022 of blue in the worst frame (it was 0.044) and the stars keep the 6-point glare (stars ≤ ~1 in HDR).
- **New default:** `--glare-cap 1.5` (0 = old behaviour). `ctest` 3/3. The time-lapse video (s09–s11) was not regenerated, only the separate clip has the defect.

## v2.6 — spiral arms, rings with gaps and variable thickness (2026-11-15)
Origin: André found the cinematic disk (v2.4) still far from Interstellar and expected "debris rings or arms like in a galaxy, with randomness in thickness".
- `--disk-arms m,pitch,amp[,r]`: `f = 1 + amp cos(m ψ)`, `ψ = φ − Ω_p t − ln r / tan(pitch)` (logarithmic spiral rotating as a **rigid pattern** of Kerr angular velocity Ω(r_p), i.e. a density wave, not wound up by the differential rotation).
- `--disk-rings amp[,freq,gap]`: 1D noise in ln r (2 octaves) in `exp(amp (2n − 1))`, with gaps where `n < gap` (emission ∝ (n/gap)²): bands and gaps as in protoplanetary disks.
- `--hr-var amp[,scale]`: only in layer mode (`--hr`): thickness `H(r) exp(amp · n)` with `n ∈ [0,1]` (it only increases: the step limit is based on σ(r) and would be unsafe with thinner layers). In layer mode the effect got diluted; the convincing look is that of the **thin disk**.
- They apply to the thin disk (factor on the surface luminosity) and to the layer density. **Aesthetic**; with amp = 0 (default) the old code is identical (`ctest` 3/3). (example image dropped: no physical basis) (`--cinematic --a 0.95 --theta 62 --dist 45 --fov 40 --rout 38 --exposure 3.5 --tpeak 7500 --bloom 0.3 --time 300 --disk-tex 0.9,60,5,5 --disk-arms 3,16,0.8,14 --disk-rings 0.8,9,0.25`).

## Decision (André, 2026-11-15): drop the aesthetic texture from the video
"The disk with shadows/arms does not look realistic: I still see a solid disk with texture applied." Without a physical simulation that generates it (an optically thick thin disk with MRI would require radiation-MHD, out of reach), **`--disk-tex`, `--disk-arms`, `--disk-rings` and `--hr-var` no longer appear in the video** (segments s11d/s11e removed) and stay in the code only as aesthetic options, off by default, with no physical claim. The only source of physical structure we have is GRMHD (radio, hot plasma), which André approved.
