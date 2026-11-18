# v4.0 additions: the Interstellar preset, a parametric RIAF, disk animation (2026-11-17)

## Interstellar preset (`--preset interstellar`, UI scenario "Interstellar look", badge: aesthetic)
The documented choices of James et al. (2015, DNGR) for Gargantua [read earlier, see `references.md`]: a = 0.6; a thin **smooth** disk at a **uniform temperature (4500 K black body)**; **no frequency shift** (neither Doppler nor gravitational); wide, luminous disk (r from the ISCO 3.83 M to 20 M); film-like colour (ACES) and flare. It is an artistic rendering: the paper itself says the disk is not Novikov-Thorne and the colour shift was dropped for the audience. The realistic counterpart is the default (a = 0.9, NT, everything physical). New flags (all usable alone): `--noshift`, `--disk-temp uniform [T]`, `--smooth-disk`.
- **Oracles (in `--selftest`, no reference image):** with g = 1 and a uniform T every ray that reaches the disk returns the same intensity (all orders of the image): max/min - 1 = 0.0 over 22 800 pixels (a = 0.6, theta = 60); at a = 0 with g = 1 the image is left/right symmetric (difference 0.0).
- Exposure: uniform T at the default exposure 3 saturates to flat white; the preset uses 0.9.

## Parametric RIAF (`--disk-model riaf`, UI "Radio: RIAF (parametric)", badge: model-dependent)
Radio mode (false colour in T_b, no sky) like the torus. Gas: n_e = ne0 (r/6)^-p exp(-z²/2H²) with H = (H/r) r, Θ_e = Θ_e,max (r/6)^-q, B from a constant plasma beta (as in the torus), toroidal field, fluid on circular orbits at ξ Ω_K (Ω_K = 1/(r^1.5 + a), the same at all heights), from the ISCO to `--riaf-rout` (40 M). `--riaf pn,pt,hr,xi` (defaults 1.1, 0.84, 0.3, 0.5) with the torus flags `--ne0 --plasma-beta --theta-e-max --ti-te`.
- **Honesty:** the project has **no verified source** for numerical values of a RIAF: the power-law family is the standard shape used in the literature on Sgr A* models (Yuan et al. 2003; Broderick & Loeb 2006; both [memory], not re-read), but every exponent and amplitude here is a **free parameter** and the defaults only give a readable image (UI: ne0 = 8e3 cm^-3, Θ_e = 12, beta = 10). It is a variant of the *shape* of hot accretion flows, not a prediction.
- **Verification (in `--selftest`):** the emitter against the formulas that define the model: density profile, Θ_e profile, B(β) and the emission energy of circular orbits at a = 0 (relative error 3e-16, 0, 2e-16, 0); float vs double emissivity 3.8e-6; no emission below r_in or beyond riaf_rout. Not verified: absolute flux (no calibration against a target, no independent oracle for the transport beyond the shared torus code).
- Cost: 166 ms per 960x540 frame, 1 sample (step ~ a quarter of the local scale height).
- A Doppler-beamed crescent, the photon ring and the shadow appear as expected qualitatively (a = 0.9375, i = 60 degrees).

## Disk animation for every disk
The surface pattern of the thin disk (brightness noise) used to ignore the disk time, so the "play" button only did something for turbulence / hot spot. Now `psi -= disk_time / (r^1.5 + a)`: every ring turns with the Keplerian Omega (differential rotation) for all thin and layer disks, all metrics (an approximation outside Kerr). With disk_time = 0 the image is unchanged (goldens identical). Offline routes use the pilot's time by default, so route videos now show the rotation too.

## Not done, and why
- **Inclined / warped disk:** a disk misaligned with the spin is not a Novikov-Thorne disk (Lense-Thirring precession warps it, Bardeen-Petterson); there is no defined emission model to draw, so it is not offered (the same rule as the combinations table).
- Bondi flow and corona / jet: not started.
