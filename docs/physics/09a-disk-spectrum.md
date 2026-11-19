---
tags: [coding, black-hole-sim, cuda, physics]
created: 2026-11-12
---

# v0.9a — Disk: spectrum and colour (hardening factor, limb darkening, exposure)

First step of roadmap-v0.9-v1.1. **Everything is opt-in and physical; the default look did not change** (md5 of the v0.7d golden, `ctest` → `v07d_regression`, still byte-identical).

## What was added
| Piece | Formula | Where |
|---|---|---|
| **Hardened spectrum** | `I_ν = f⁻⁴ B_ν(f·T)`; the `f⁻⁴` keeps the bolometric radiance `∫I_ν dν = σT⁴/π` for any f. `f_col ≈ 1.7–2.0` near Eddington (Shimura & Takahara 1995, ApJ 445:780, eq. 1; does not apply at `ṁ < 1 %` Eddington) | `core/include/bhs/disk_emission.hpp` (`modified_planck_nu`, `rgb_hardened`), `lut::build(fcol)` |
| **Redshift** | `I_ν/ν³` invariant ⇒ the observed spectrum is the same modified black body at `T_obs = g·T` (v0.4 already used this), and the observed bolometric scales as `g⁴` | already in the kernel; now tested |
| **Electron-scattering limb darkening (Chandrasekhar 1960)** | `I(μ) = B (1 + 2.06 μ)/(1 + 2.06·2/3)` (Chandrasekhar 1960, *Radiative Transfer*, Table XXIV, p. 248: `I(1)/I(0) = 3.0631`; grey Eddington is `1 + 1.5 μ`). Normalisation leaves `2∫I μ dμ = πB` (= σT⁴) unchanged | `limb_chandrasekhar` (alias `limb_eddington`), `limb_eddington_grey`, kernel `render.cu` |
| **μ in the gas frame** | on the equator `Θ = Q`, so `p^(θ) = √Q/r`; a boost along `e_φ` (the gas motion) does not touch this component, only the energy `E_emit = −p·u = uᵗ(1 − ΩL)` ⇒ `μ = √Q / (r·uᵗ(1 − ΩL))` | `emitter_mu`; `kerr::Trace` and the device `Trace` now return `Q` |
| **Physical exposure** | `--expose relative` (default): brightness is renormalised to the peak of each LUT (as in v0.6a); `--expose physical`: fixed reference (black body f = 1 at `tpeak`), so `--fcol`/`--limb` change the brightness as in physics | `make_kp` in `render.cu` |

Flags: `--fcol f` · `--limb none|eddington` · `--expose relative|physical` (see `--help`). The turbulence noise (aesthetic) still multiplies everything; no appreciable cost (the LUT is 12 kB and is only recomputed when `fcol` changes).

## Oracles (tests)
Core (`test_disk_emission.cpp`, 9 new cases, 84 in total, all green):
1. `∫B_ν dν = σT⁴/π` (numerical integration in ln ν, T = 3e3, 1e4, 1e5 K) to 1e-4.
2. The same bolometric for `f⁻⁴B_ν(fT)`, f = 1, 1.5, 1.7, 2 (independent of f).
3. `I_obs(ν) = g³ I_em(ν/g)` ⇒ bolometric `g⁴·σT⁴/π` (g = 0.3, 0.8, 1.4) and the identity `B_mod(ν; gT) = g³ B_mod(ν/g; T)`.
4. `rgb_hardened(T, 1)` is **exactly** `blackbody::rgb(T)` (equality of doubles); `f ≠ 1` is `f⁻⁴·rgb(fT)`; `lut::build()` unchanged.
5. Limb darkening: `2∫₀¹ L(μ) μ dμ = 1` to 1e-9; edge/centre ratio `1 + 2.06`.
6. `emitter_mu`: `Q = 0 ⇒ μ = 0`; photon at radial rest with `L = 0` in Schwarzschild, `μ = √((1 − 3/r)/(1 − 2/r)) = 1/γ_orbital` (closed form, r = 6.5, 10, 40) to 1e-12.
7. `kerr::trace` returns the photon's Q.
8. Ray traced from a camera at D = 1e5, θ = 60° to the point (r = 1000, φ = 0) of the plane: the `μ` of the trace agrees with the flat angle to the normal (|Δ| < 3 %). **Note:** `trace()` takes the PHOTON's propagation direction, opposite to the viewing one; the first version of the test passed the viewing one and failed to hit the disk (a test error, not a code one).

GPU (`--selftest`, section `v0.9a`, in `ctest`): GPU `μ` vs `emission::emitter_mu` (a = 0, 0.9, 0.998; 2000 cases each) max 7.8e-16; GPU Q vs `kerr::trace` (400 rays) 2.1e-16; image: limb darkening face-on (θ = 5°) mean luminance ×1.194, edge-on (θ = 85°) ×0.730; `f_col = 1` = default (difference 0); `f_col = 1.7` changes the image.

## Measurements (a = 0.9, θ = 75°, 1080p float, 32 samples)
- Cost: 7.88 ms/sample (default) · 7.90 (`--fcol 1.7`) · 8.09 (`--limb eddington`): the limb costs +0.2 ms/sample (one double `emitter_mu` per disk pixel).
- `--fcol 1.7` with relative exposure: mean image luminance (selftest, θ = 60°) ×1.42; with `--expose physical` ×2.20. Colour: the disk gets whiter/bluer (see images).
- Mean RGB (sRGB, 0–1): default (0.096, 0.082, 0.067) · `--fcol 1.7` (0.121, 0.110, 0.099) · `--limb eddington` (0.092, 0.079, 0.065) · both + `--expose physical` (0.125, 0.114, 0.103).

Images (`assets/v09a/`, 960×540; contact sheet `assets/v09a-comparison.png`): the contact sheet = default (top-left), `--fcol 1.7` (top-right), `--limb eddington` (bottom-left), both with `--expose physical` (bottom-right); the four individual `*-960.png`.

```bash
build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui --shot k.png --a 0.9 --theta 75 --prec float --samples 32 --fcol 1.7 --limb eddington --expose physical
```

## Limitations
- `f_col` is **constant** over the disk (in reality it depends on T, density and radius); there is no `f_col(r)` profile yet.
- Eddington limb darkening is the grey approximation; real atmospheres have a frequency dependence (v0.9b with radiative transfer replaces this).
- The colour uses the CIE analytic fit of Wyman et al. (already from v0.4); `rgb` is radiance in the visible band, not bolometric: the bolometric is only verified in the tests (integral in ν), not in the image.
- The brightness comparison between `fcol` values depends on what "same exposure" means (hence the two `--expose` options).
