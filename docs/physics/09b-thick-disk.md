---
tags: [coding, black-hole-sim, cuda, physics]
created: 2026-11-12
---

# v0.9b — Thick disk: radiative transfer along the geodesic

Second piece of roadmap-v0.9-v1.1 (after [09a-disk-spectrum](09a-disk-spectrum.md) and [09a2-soft-edge](09a2-soft-edge.md)). **Opt-in** (`--hr h`); `--hr 0` (default) is the thin disk of the earlier versions, byte-identical (`ctest` → `v07d_regression`).

## Model
| Piece | Description | Where |
|---|---|---|
| **Layer** | Gaussian in `z = r·μ` with height scale `H = h(r)·r`, `h(r) = hr·(r/10)^flare`, **total** vertical optical depth `τ` (`∫α dz = τ`): `α(r,μ) = τ/(√(2π)H)·exp(−z²/2H²)·w_edge(r)`, 0 inside the ISCO | `core/include/bhs/volume.hpp` (`Slab`), `render.cu` (`VolAcc`) |
| **Transport** (grey, LTE, source = Planck at `T(r)`) | front→back: `I += trans·B·(1−e^{−dτ})`, `trans *= e^{−dτ}`; `dτ = α·E_emit·Σ·dMino` (the length in the gas frame is `E_emit·dλ`, `dλ = Σ·dMino`, `E_emit = uᵗ(1−ΩL)` as in the v0.9a `μ`); `B` = LUT colour at `T_obs = g·T(r)` | `volume::composite`, `VolAcc::add` |
| **Integration along the ray** | each accepted Dormand-Prince RK45 step generates `--vol-sub` (4) samples through a cubic Hermite of `r(τ)`, `μ(τ)`; the ray ends when `trans < 1e-3`; the equatorial-plane crossing (thin disk) is turned off | `bk::trace<S,F,Vol>` in `kerr_cuda.cuh` (`if constexpr (Vol::kActive)`: with `NoVol` the generated code is the old one) |
| **Step limit** | near the layer the step cannot jump over the Gaussian: `|Δμ| ≤ max(0.6σ, (|μ|−4σ)/2)` with `σ = h(r)`, limited by the speed `dμ/dτ'` and the curvature `μ''` (approaching from outside the distance to 4σ is reduced at most by half per step) | same |
| **Interchangeable source** | the density is just `Slab::alpha(r, μ)` / `VolAcc::add`: it is the point where v0.9c (3D turbulence) and v2.x (torus, RIAF, GRMHD data) plug in | — |

Flags: `--hr h [--tau 10 --flare 0 --vol-sub 4]`. `--edge-soft` applies as a **density taper**; `--fcol`, `--beaming`, `--profile`, `--nodoppler` and the moving observer work; `--edge-noise` and `--limb` do not (the aesthetic noise and the limb darkening belong to the thin disk; the physical turbulence is v0.9c).

## Two engineering details (what went wrong first)
1. **FP64 in every step cost 11×.** The 1st version did the step limit and the `add()` in `double`: 2.73 → 29.6 ms/sample (960×540). On GeForce cards FP64 is 1/64 of FP32. The step limit and the `add()` moved to `float`, keeping the integrator state; and the redshift observer factor, `E_obs = (1−ωL)/α` (which only depends on the radius, not on the point), is computed **once per ray** in `begin()`, with `g = E_obs·dop/e_emit` per sample. Result: **4.37 ms/sample** (thin disk 2.73). The image did not change (the oracles below were measured afterwards).
2. **Normalisation constant doubled.** The Gaussian's `1/√(2π)` and the `Σ` of `dλ` have to be right for the face-on oracle to match: it matches to 2e-4.

## Oracles
Core (`test_volume.cpp`, 6 new cases, **94 in total**, green): `H/r = hr (r/r_ref)^flare`; `∫α dz = τ` for flare 0 and 0.3 and r = 8, 12, 40 (numerical integral, 1e-6); α = 0 at the ISCO and at the hard cut, `w = 1/2` at `r_e` with the soft edge; front→back composition of a homogeneous layer `= B(1−e^{−τ})` for **any partition** (n = 1, 2, 7, 100; 1e-12), thick limit `→ B`, thin limit `→ Bτ`, two layers; face-on path in the gas `τ/μ`.

GPU (`--selftest`, section `v0.9b`, in `ctest`):
| Test | Result | Limit |
|---|---|---|
| **Face-on** (θ = 1°, D = 1000, a = 0): each pixel `= k·LUT(g·T)·(1−e^{−τ/μ_gas})`, with `r, L, Q` of the thin disk from `kerr::trace` (CPU) and `μ_gas = emission::emitter_mu` | τ = 0.5: max relative error **2.4e-4** (33 pixels); τ = 2: **8.1e-4** (40 pixels) | 3 % |
| τ → 0: `L(0.01)/L(0.02)` | 0.507 (the non-linearity of `1−e^{−τ}` explains the 1.4 %) | 0.5 ± 3 % |
| `vol-sub` 4 vs 16 (mean luminance) | 0.9999 | ±1 % |
| float vs double | 0.9998 | ±1 % |
| `H/r` 0.004 vs 0.002 (τ = 300): converges to the thin disk | 0.9957 | ±2 % |
| `H/r = 0.15`: finite and positive HDR | yes | — |

## Cost (a = 0.9, 1080p, float, 32 samples, rout 16)
| Scene | ms/sample |
|---|---|
| thin disk (θ = 75°) | 10.22 |
| `--hr 0.05 --tau 10` | 16.12 (+58 %) |
| `--hr 0.1 --flare 0.25 --tau 6 --edge-soft 0.12` | 25.45 |
| same at θ = 86° | 26.78 |

Images (`assets/v09b-comparison.png`): thin disk (top-left); `--hr 0.05 --tau 10` (top-right); `--hr 0.1 --flare 0.25 --tau 6 --edge-soft 0.12` (bottom-left; the `--edge-noise 0.1` that was also passed is ignored in volume mode); `θ = 86°` (bottom-right: the thick disk hides the far side).

```bash
build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui --shot k.png --a 0.9 --theta 75 --prec float --samples 32 --hr 0.1 --flare 0.25 --tau 6 --edge-soft 0.12
```

## Limitations (to declare)
- **Vertically isothermal:** `T(z) = T_NT(r)`; a real disk cools with height (atmosphere) and `f_col` depends on `T` and the density.
- **Keplerian gas at all heights** (no vertical shear or radial flow); `g` is computed with the `r` of the point.
- **`τ` constant with `r`** and grey opacity; the emission is isotropic (LTE), no limb darkening in the layer.
- **Cylindrical wall at the ISCO:** α = 0 for `r ≤ r_in`; there is no plunging region (visible at θ = 86°). Left for GRMHD/a plunge model.
- No noise: the "smooth" look of the volume is deliberate until v0.9c, which adds the 3D turbulence and the differential rotation.
- Validated for `H/r` from 0.002 to 0.15 (the selftest uses 0.002–0.004 with 4–8 sub-samples and 0.05–0.15 with 4); below 0.002 it was not measured.
