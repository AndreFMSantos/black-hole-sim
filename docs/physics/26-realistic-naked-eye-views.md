# v3.0 — realistic views: feasibility and epistemic status (reframed on 2026-11-15)

> **Reframing (André, 2026-11-15):** the telescope was just a way of speaking; the question is whether **a human will ever be able to see a black hole with the naked eye**, at the eye's resolution. And the background warning: everything the app shows is a **physics-based simulation**; there is a lot of unknown physics and we have no way to confirm the real appearance. So point 2 stopped being a feature (`--view`) and became (a) this feasibility study and (b) an **epistemic-status label** per scenario, for the UI.

## Can a human see a black hole? (`tools/exp/eye_feasibility.py`)
Schwarzschild shadow: diameter 2√27 GM/c² = 10.39 r_g (Kerr differs by a few %). Eye: ~1 arcmin (foveal acuity 20/20); diffraction limit at 550 nm with a 5 mm pupil ≈ 28 arcsec. Masses and distances [memory].
| system | shadow | distance | angular diameter | vs eye (1 arcmin) | distance for 1 arcmin |
|---|---|---|---|---|---|
| Gaia BH1 (9.62 M☉) | 148 km | 1566 ly | 0.002 µas | 3.4e-11× | **1.3 Earth–Moon distances** |
| Gaia BH3 (33 M☉) | 507 km | 1924 ly | 0.006 µas | 9.6e-11× | 4.5 Earth–Moon distances |
| Sgr A* (4.14e6 M☉) | 6.4e7 km | 26 510 ly | 52 µas | 8.7e-7× | 1460 AU |
| M87* (6.5e9 M☉) | 1.0e11 km | 5.5e7 ly | 40 µas | 6.6e-7× | 2.3e6 AU |
**Reading:** from Earth it is impossible (6–11 orders of magnitude are missing; the EHT needs an Earth-sized interferometer at 230 GHz). With the naked eye one would only see a 1 arcmin shadow **by going there**: ~5·10⁵ km from a stellar hole (negligible tide: ~9e-5 m/s² over 2 m) or ~1460 AU from Sgr A*. An **isolated** hole would be a black dot deforming the star field (without a disk, no light of its own); with a disk, the brightness of any hot disk is greater than the Sun's (table below) and the view would be dominated by glare.

## Epistemic status of everything the app shows
| level | what | basis |
|---|---|---|
| **Observed / tested** | light bending and gravitational lensing; size and shape of the ring (EHT: M87* 42 µas, Sgr A* 51.8 µas, compatible with Kerr); ring measured in our GRMHD = Kerr critical curve | observations + general relativity |
| **Model-dependent** | Novikov–Thorne temperature profile (real low-accretion disks are RIAFs); electron temperature (Ti/Te, `R_high`); spin and inclination; magnetic field (MAD vs SANE); Doppler asymmetry; the n = 1 photon ring (predicted, not yet resolved) | plausible, non-unique models |
| **Hypothetical** | Kerr–Newman and RN (charge is neutralised quickly in plasma), naked singularity (cosmic censorship), Λ (the real value gives Λ M² ~ 1e-30 or less; in the videos we used up to 0.045, **exaggerated** so it can be seen), wormholes (no evidence, require exotic matter) | consistent mathematics, no observation |
| **Aesthetic** | `--cinematic`, glare, bloom, grain, disk texture/arms/rings, HDR exposure choices (1 unit = 120 cd/m²) | visual decision |
| **Unknown** | interior, singularity, quantum gravity | — |

## Brightness orders of magnitude (thin disk, `tools/exp/eye_scenarios.py`; not GR photometry)
André's request (2026-11-12): besides the "Interstellar" version, show what would be seen with the naked eye or with telescopes. This note is the **design and the orders of magnitude**; there is no `--view` yet. Script: `code/tools/exp/eye_scenarios.py` (face-on Shakura–Sunyaev thin disk, without relativistic effects; ȳ of Wyman, Sloan & Shirley 2013 [memory]; *not* GR photometry).

## Orders of magnitude (computed)
| scenario | T_max | visible illuminance | disk luminance | apparent size |
|---|---|---|---|---|
| 10 M☉, ṁ = 0.1 Edd, at 1000 km | 4.9e6 K | 9.9e15 lux | 3.2e13 cd/m² | ~34° |
| 10 M☉, ṁ = 0.1 Edd, at 10⁵ km | 4.9e6 K | 9.9e11 lux | 3.2e13 cd/m² | 0.34° |
| 10 M☉, ṁ = 0.1 Edd, at 1 AU | 4.9e6 K | 4.4e5 lux | 3.2e13 cd/m² | 0.0002° |
| 10 M☉, ṁ = 10⁻⁴ Edd, at 10⁵ km | 8.7e5 K | 9.3e10 lux | 5.5e12 cd/m² | 0.34° |
| 4.14e6 M☉ (hypothetical), ṁ = 10⁻³ Edd, at 100 AU | 6.1e4 K | 8.6e8 lux | 3.1e11 cd/m² | 0.94° |
| 4.14e6 M☉ (hypothetical), ṁ = 10⁻⁶ Edd, at 100 AU | 1.1e4 K | 3.4e6 lux | 1.6e10 cd/m² | 0.94° |
References: Sun at noon 1.2e5 lux and 1.6e9 cd/m²; full Moon 0.25 lux; HDR screen 1e3 cd/m².

**Conclusion:** the visible-band luminance of any part of a disk with T ≳ 10⁴ K exceeds the Sun's by 1–4 orders of magnitude (by Planck, a surface at 10⁴ K has a much larger visible surface brightness than one at 5800 K). An "Interstellar" disk (4500 K) would be ~1e8 cd/m² (dazzling, but below the Sun). So **the naked-eye view is dominated by adaptation and glare**, not by lack of light; the stars disappear. The real Sgr A* is a RIAF (hot, tenuous, mostly infrared/radio/X), and the `ṁ` above is hypothetical: the thin-disk model does not apply.
