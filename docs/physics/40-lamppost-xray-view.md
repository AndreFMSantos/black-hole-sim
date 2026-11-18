---
tags: [coding, black-hole-sim, physics, xray]
created: 2026-11-18
status: done (E4 part 2): GPU X-ray view, bolometric, Kerr only
---
# Lamppost X-ray view (E4 part 2)

Code: `apps/v4_0-metric-api` (`render.cu` `ensure_lp`, `lp_log10e`, `xray_rgb`; `interfaces.cuh` `ThinDisk::shade`), table in `core/include/bhs/lamppost.hpp` (`log_table`).
Physics of the profile: note 39. **Decision (André, 2026-11-18): bolometric reflected intensity** (no power-law band, no invented reflection fraction).

## What is drawn

A static point corona on the axis at BL radius h lights the thin disk; the disk re-emits the absorbed light isotropically (Lambertian). Per disk element:

`I_obs = g⁴ · I_em`, `I_em = ε_E(r) · albedo / π`

- `ε_E(r)`: energy emissivity from note 39 (photon number × the source-to-disk redshift), normalised so its **maximum is 1**; the albedo/π is folded into that normalisation, so the image is relative, not absolute.
- `g`: the observer's redshift of the disk element, exactly the one the thin disk already uses (Kerr, Doppler of the camera included; `--noshift`, `--nodoppler`, `--beaming` apply).
- Display: afmhot false colour of `log10 I_obs` between `--lamppost-range lo,hi` (default −4, 1; `I_obs > 1` is possible on the approaching side, g > 1). **The corona itself is not drawn** (a point).
- No surface pattern, no limb darkening, no temperature: it is the light, not a black body. The colour is a display choice, as in the radio / spherical views.

## Implementation

1. Host: `lamppost::emissivity(a, h, r_in, r_tab, samples)` (oracle, ~0.8 ms/photon) → `log_table` (log10 ε_E on 2048 points uniform in ln r, linear in (ln r, ln ε), exact for a power law, max = 0) → device array. `r_in` = ISCO (or `--rin`), `r_tab = max(2 rout, 100)`. Below the first radius the first value; beyond the last, the slope of the last segment.
2. Cache per (a, h, r_in, r_tab, samples). Offline/CLI: synchronous. UI: **asynchronous** (`std::async`): the new table is built in the background while the previous one is drawn, so dragging the spin does not freeze the frame; the first table is synchronous.
3. `ThinDisk::shade`: if `lp_n > 0`, colour = `xray_rgb(4 log10 g + log10 ε_rel(r))`; everything else of the thin-disk path (soft edge, layers behind the fringe) is unchanged.
4. `combos.hpp`: `DiskKind::Lamppost` is Kerr only (the photon orbits of Kerr-Newman / de Sitter / the wormhole were not worked out); the CLI refuses the others (`tests/combos.py`).

CLI: `--disk-model lamppost [--lamppost-h 6 --lamppost-range -4,1 --lamppost-samples 2000]`, `--rout` is the disk's outer edge. h must exceed the horizon.
UI: Disk → "X-ray: lamppost reflection (bolometric)"; h is applied when the slider is released.

## Verification

| Check | Result |
|---|---|
| Kernel table vs the oracle's own raw table (a = 0.9, h = 6, 11 radii, log10 ε differences) | 6.7e-7 (limit 5e-3) |
| Below the table / beyond the table (clamp, power-law continuation) | exact (0, limit 1e-6) |
| `log_table` of a power law (core test) | exact to 1e-5; slope_out −3 to 1e-9 |
| `log_table` of a real Kerr emissivity (a = 0.9, h = 6) | finite, ≤ 0, peak inside 6 r_in, slope −2…−4.5 |
| Combinations | Kerr renders; Kerr-Newman, naked, SdS, wormhole refused |
| Suites | core 182/182 (684 477 assertions); v4_0 ctest 37/37; `--selftest` OK |

Timing: the table costs 1.98 s end to end at 2000 samples (CLI, includes process start and a 64×36 render), 3.66 s at 4000. Rendering 960×540 with this model: steps per ray mean 26.3, max 120.

## Looking at it (a = 0.9, i = 70°, 960×540)

- h = 3: light concentrated near the horizon, a bright lobe on the approaching side, a dark outer disk. The disk fades quickly.
- h = 10: the emission fills the whole disk (the outer disk is lit by photons emitted upward and bent back, note 39), the inner peak is much weaker.
- This is the qualitative behaviour of Wilkins & Fabian (2012): a lower source concentrates the illumination inward. The quantitative check is note 39 (ε contrast 1438 → 126 → 7.4 for h = 3, 10, 30).

## Not done / limits

- Plunging region (r < r_in) is not lit; photons crossing the plane there continue (as in `kerr::Disk`). Lighting it needs a gas model inside the ISCO (plunging orbits for g and the area element), not only the table: out of scope for a bolometric picture.
- **Overlapping branches (found and fixed in E4b).** For a ≲ 0.5 and low h, rays that go round the hole land at the same radii as direct ones (a = 0.5, h = 3: two branches cover r = 4.3 … 511). The first table interleaved them in r instead of adding them. Exact check: `eps_N r dr = sin δ dδ = dx`, so `∫ eps_N r dr` must equal the measure of x = −cos δ that reaches the disk. Before: ratio 0.59 (a = 0, h = 3), 0.77 (a = 0.5, h = 3), 0.95–0.99 at h ≥ 6, 0.998–1.001 at a = 0.9. Now the table is a radial histogram (log cells, every segment of every branch spreads its dx with density ∝ r⁻², cell average), and the integral matches the measure within 2 % in 6 cases (core test "conserves photon number"). The first version with a uniform density gave an outer slope of exactly −1 (one wide segment at the edge of the sampling): the r⁻² density restores −2…−3.
- Frequency band (power law, reflection fraction, ionisation) is not modelled: this is the bolometric picture, the honest one for the data we have.
- The table build is not on the GPU; at h < ~2 r_+ with a ≈ 0.998 the sweeps get slow (a = 0, h = 3: 25 s, 618 points, note 39). The UI refuses nothing here; it just waits.


## S2: the plunging region (2026-11-18)

`--lamppost-plunge` (UI: checkbox): the disk reaches the horizon (r_+ + 1e-3) and the gas inside the ISCO falls on the geodesic with the E and L of the ISCO (Mummery & Balbus 2022):
`u^t = [(r²+a²+2a²/r)E − 2aL/r]/Δ`, `u^φ = [(1−2/r)L + 2aE/r]/Δ`, `(u^r)² = [(E(r²+a²) − aL)² − Δ(r² + (L−aE)²)]/r⁴`, `u^r < 0`.

What changes and what does not:
- **The photon-number emissivity does not change.** The number of photons crossing the equatorial hypersurface per gas proper time and proper area is the scalar `J^μ n_μ = J^θ √g_θθ`, and `dN = J^θ √−g dt dr dφ` with `√g_θθ/√−g = 1/r` on the equator: `ε_N = (dN/dt dr dφ)/r`, whatever the gas does. (So `ε_N ∝ sin δ |dδ/dr| / r` stays.)
- **The energy does.** `E_gas = −p·u = u^t − L u^φ − p_r u^r`, with `p_r = ±√R/Δ` and the sign of `dr` along the photon's path at the crossing. The oracle now returns that sign (`Result::pr_sign`: from `turn` and which leg the crossing is on), the GPU tracer stores `vr` = dr/dτ′ at the crossing (`p_r` has the sign of −vr).
- Source → gas: `g = E_gas √(Δ(h)/(h²+a²))`; camera → gas: `g = E_obs / E_gas` (`redshift_plunge`).

Checks:
| Check | Result |
|---|---|
| `u·u = −1` for 5 spins (incl. retrograde) at 6 radii from 0.2 to 4 r_isco; `u^t, u^φ` against the inverse metric; `(u^r)²` against the normalisation | 1e-9 / 1e-9 / 1e-6 |
| `u^r → 0` at the ISCO (continuity) | yes |
| Null condition of the photon with p_r from Carter's R | 1e-9 |
| Photon number over the table to the horizon (a = 0.9, h = 3) | integral = measure within 2 %; > 5 % of the photons land inside the ISCO |
| Kernel `g` against an independent route (inverse metric, `u^r` from normalisation, p_r from the null condition), 18 photons at 4 radii inside the ISCO | 8.7e-14 |
| Suites | core 188/188, ctest 37/37, `--selftest` OK |

Look (a = 0.9, i = 75°, h = 3): without it the ISCO leaves a black "inner shadow" with thin bright arcs of higher-order images at its left edge; with it the shadow edge is smooth (the true shadow), the region just inside the ISCO is lit and reddened (g < 1 as the gas falls away).
Not done: the plunging gas is Kerr-equatorial only (as everything in this view); the inner edge of the disk is the horizon, not a physical truncation.

## S3: power-law band and reflection fraction (2026-11-18)

`--lamppost-gamma G` (default 2) and `--lamppost-refl R` (default 0 = geometric); UI sliders. The second spectral treatment André asked for, on top of the bolometric one.

**Model** (relxill-like lamppost, grey albedo): the corona emits N(E) ∝ E^−Γ isotropically. In the disk frame the spectrum is blueshifted by g_sd = E_gas/E_source: `N_d(E) = ε_N g_sd^(Γ−1) E^−Γ` (photon number is conserved, `N_d(E) dE = N_s(E/g) d(E/g)`). Energy flux density per energy `F_E = E N_d ∝ ε_N g_sd^(Γ−1) E^(1−Γ)`. Reflected, isotropic: `I_em(E) = (albedo/π) F_E`. Seen from the camera, `I_E/E³` is invariant: `I_E,obs(E) = g³ I_em(E/g) = ε(r) g^(2+Γ) E^(1−Γ)`, with `ε(r) = ε_N <g_sd^(Γ−1)>` (the table's `eps_e`).

- **A band changes nothing in the image.** `∫E^(1−Γ) dE` over any band is one constant for all pixels, so the *relative* image is the same in any band; the band edges only rescale it. That is why there is no `--lamppost-band`: it would be a no-op (it would matter only with an energy-dependent albedo / line emission, which this model does not have).
- **Γ = 2 is exactly the bolometric view** (`g^4`, `ε_N g_sd`): the old image is the Γ = 2 case, not a separate path. Checked in the core test (Γ = 2 → `eps_e = eps_n g`; Γ = 1 → `eps_e = eps_n`; Γ = 3 → Jensen `<g²> ≥ <g>²`).
- Soft spectra are more beamed: g^(2+Γ) = g⁵ at Γ = 3 against g^3.5 at Γ = 1.5 (renders at a = 0.9, h = 6, i = 70°: Γ = 3 is strongly one-sided, Γ = 1.5 nearly even).
- **Reflection fraction** `R_f = N_disk / N_infinity` (Dauser et al. 2014): the number of photons that hit the disk (out to `--rout`, from the inner edge) over those that reach infinity, both from the same ray sample (`Table::refl_frac`). The geometry fixes it. `--lamppost-refl R` rescales the disk brightness by `R / R_geometric` (the relxill `refl_frac` convention: it moves the reflected component against the direct continuum, which is not drawn); 0 keeps the geometry. The CLI prints R_geometric.

Measured R_geometric (disk out to 100, ISCO inner edge unless stated):

| a | h | R_geo |
|---|---|---|
| 0 | 500 (disk to 1e5) | 0.998 (the flat-space value is 1) |
| 0.9 | 3 | 2.31 |
| 0.9 | 10 | 1.18 |
| 0.9 | 30 | 0.64 (a finite disk: photons landing beyond 100 are lost) |
| 0.9 | 6, `--rout 30`, ISCO / horizon inner edge | 1.19 / 1.29 |

Light bending makes R > 1 for low sources (Dauser et al. 2014, Fig. 1) and the finite disk makes R < 1 for high ones.
Checks: kernel table with Γ = 1.5 against the oracle table with the same Γ, 3.1e-6 in log10; Γ changes the profile; core 190/190 (711 881 assertions), ctest 37/37, `--selftest` OK.
Not done: the direct corona is not drawn (so R_f has no visible "ratio" in the image), no energy-dependent albedo or ionisation.
