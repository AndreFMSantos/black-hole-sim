---
tags: [coding, black-hole-sim, cuda, physics, radio]
created: 2026-11-12
---

# v2.2b — Radio mode: Fishbone-Moncrief torus + thermal synchrotron

André's decision (2026-11-12): keep **two separate modes**: the "optical" disk of v0.x/v0.9 (black body, colour) and a **radio** mode (synchrotron emission, brightness-temperature image, no sky), like the EHT images. This is the first step of the radio mode and the basis of the future GRMHD simulation reader (v2.2f). References and the verification status of each: [references](../REFERENCES.md).

`--disk-model torus` (default `slab`: everything as before, `ctest`/regression byte-identical).

## Model
| Piece | Description | Where |
|---|---|---|
| **Torus** | hydrostatic equilibrium of a polytropic fluid of constant `l = −u_φ/u_t` (Fishbone & Moncrief 1976), `h = |u_t|_in/|u_t|(r,θ)`, `ρ̃ = ((h−1)/(h_max−1))^{1/(Γ−1)}`, `Θ_i = (Γ−1)(h−1)/Γ`; `l = l_Kepler(r_max)`. Refuses open tori (`|u_t|_in ≥ 1`: a = 0.9, r_max = 12 closes from r_in = 7) | `core/include/bhs/torus.hpp` |
| **Electrons** | `ne = ne0·ρ̃`; `Θ_e = θe_max (h−1)/(h_max−1)` (`--theta-e-max`) or `Θ_i (m_p/m_e)/(Ti/Te)`. **The electron temperature is a model parameter**: the analytic torus is cold (`Θ_i,max ≈ 0.0056` for r_in = 7, r_max = 12, which is already a thick torus, H/r ≈ 0.3), and real simulations heat the gas. `P = ne k (T_i + T_e)` | `render.cu` (`torus_emit`) |
| **Field** | toroidal, `B = √(8π P/β)` with constant `β` (`--plasma-beta`); `θ_B` between the field and the photon **in the fluid frame**: boost along e_φ, `cos θ_B = (n_φ − v)/(1 − v n_φ)`, `n_φ = (L/ϖ) α/(1 − ω L)`, `v = ϖ(Ω − ω)/α` | same |
| **Emission** | thermal synchrotron (Dexter 2016, eq. A12/A18): `j = ne e² ν /(2√3 c Θ_e²) I_I(x)`, `x = ν/ν_c`, `ν_c = (3/2) ν_B sin θ_B Θ_e²`; absorption by Kirchhoff `α = j/B_ν(T_e)`. Only with `Θ_e ≥ 0.3` (otherwise no emission; the fit is valid for `Θ_e > 3`, ~20 %) | `core/include/bhs/synchrotron.hpp`, `render.cu` |
| **Transport** | per geodesic step (the Hermite sub-samples of v0.9b): `dτ = α ds`, `ds = E_emit Σ dMino · (GM/c²)` in cm (`--mbh` in solar masses), `ΔI_obs = g³ j ds (1−e^{−dτ})/dτ · T`; `g = E_obs/E_emit`, `E_emit = u^t(1 − ΩL)`; `ν_em = ν_obs/g` | `VolAcc::add_torus` |
| **Image** | `T_b = I c²/(2kν²)` (Rayleigh-Jeans) → *afmhot* false colour in `log10 T_b ∈ [tb_lo, tb_hi]` (`--tb-range`), no sky | `radio_rgb` |
| **Step** | limit of ~0.6 M of path per step inside the gas region (`cap`), so as not to jump over the torus | `kerr_cuda.cuh`, `VolAcc::cap` |

Flags: `--disk-model torus --torus rin,rmax[,gamma] --ne0 N --plasma-beta β --theta-e-max Θ --ti-te R --nu-ghz F --mbh M --tb-range lo,hi --torus-step 1.0`. The camera has to be outside the gas (`--dist ≥ 80`).

## What it is NOT
It is not a GRMHD simulation (the torus is static, the initial condition of the simulations), nor is it calibrated against M87* or Sgr A*. It is a semi-analytic model, with free `ne0`, `β` and `Θ_e`. The image has a hole in the centre (the torus starts at r_in = 7) and so it shows the photon ring on the far side.

## Oracles
Core (`test_torus.cpp`, `test_synchrotron.cpp`; **132 cases**):
- **Torus:** `l_K(r)` matches the one derived from `Ω_K` through the metric (a = 0, 0.5, 0.9; r = 8, 12, 20) to 1e-12; `h = 1` at the inner edge; **pressure maximum at r_max** (`dh/dr = 0`); `u·u = −1` and `u_φ/u_t = −l`; **hydrostatic equilibrium** `∂_a ln h = ½ u^μ u^ν ∂_a g_μν` (a = r, θ; 12 points) with finite derivatives of the metric, **independent** of the torus formulas; `P/ρ` and the polytropic exponent.
- **Synchrotron:** the `I_I(x)` fit against the integral that defines it with `std::cyl_bessel_k` (Dexter eq. A15/A5): error ≤ **0.35 %** (x = 0.1…30; the paper says ~1 %); **total power** against `(4/3) σ_T c U_B ⟨γ²β²⟩` with `⟨γ²⟩ = 12Θ²`: ratio **0.9971** (Θ = 100, 300); `j ∝ n`; `j(λν; λν_B) = λ j(ν; ν_B)` (depends only on x); Kirchhoff and Rayleigh-Jeans `T_b`.

GPU (`--selftest`, section `v2.2b`):
| Test | Result |
|---|---|
| density and `Θ_e` GPU vs core | 4.8e-13 |
| field `B` GPU vs core | 3.2e-13 |
| `j` and `α` GPU vs core (same `g` and `sin θ_B`) | 7.4e-11 |
| `L = 0`: `sin θ_B = 1/γ`, `γ = α u^t` (aberration of the toroidal field) | 4.4e-16 |
| thin limit (`ne0 = 1e2`): `I(2M)/I(M) = 2` (path length) | 2.000 |
| thick limit (`ne0 = 1e9`): `T_b,max / T_e,max` between 0.1 and 2 (Planck) | 0.185 |
| axisymmetry: camera at φ = 0 and 100° | difference 0 |
| Doppler beaming: approaching side brighter (> 1.2×) | **4.5×** |

## Images and cost
`assets/v22b-radio.jpg`: a = 0.9, `--torus 7,12 --plasma-beta 3 --theta-e-max 20 --ne0 3e4 --mbh 6.5e9 --nu-ghz 230`, inclinations 17°, 60°, 80° (left → right). At 17° you see the ring around the shadow and the gas around it; at 60° and 80° growing Doppler asymmetry. With `ne0 = 1e4` the torus is optically thin and only the approaching side is seen (an arc); with `ne0 ≥ 1e5` it saturates at `T_b = g T_e`.
Cost (700×700, float, per sample), before and after the optimisation (v2.2b2):

| Inclination | Before (emission in `double`, step 0.6 M) | After (`float`, `--torus-step 1.0`) | Gain |
|---|---|---|---|
| 17° | 199.5 ms | **24.1 ms** | 8.3× |
| 60° | 308.7 ms | **33.1 ms** | 9.3× |
| 80° | 358.7 ms | **37.4 ms** | 9.6× |

What changed: (1) `torus_emit` is a `template <class F>`: **`float` in production** (FP64 is 1/64 of the throughput) and `double` as the reference of the tests; only `j`, `α` and `I` accumulate in `double` (dynamic range of tens of orders of magnitude); (2) the maximum path per step in the gas went from 0.6 M to **1.0 M** (`--torus-step`). Measured at 17° with `--samples 1`, against a fine reference (`--torus-step 0.3 --vol-sub 8`, 116.8 ms):

| `--torus-step` / `--vol-sub` | ms/sample | mean error vs reference (0..255) | max |
|---|---|---|---|
| 0.3 / 8 (reference) | 116.8 | 0 | 0 |
| 0.6 / 4 | 35.2 | 0.001 | 16 |
| **1.0 / 4 (default)** | **24.1** | 0.002 | 19 |
| 1.5 / 4 | 18.7 | 0.003 | 19 |
| 2.5 / 4 | 15.2 | 0.008 | 16 |
| 1.0 / 2 | 14.6 | 0.005 | 19 |

(The maximum of 16–19 is at isolated pixels of the photon ring.)

```bash
build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui --shot r.png --a 0.9 --theta 17 --dist 150 --fov 14 --w 700 --h 700 --prec float --samples 8 \
  --disk-model torus --torus 7,12 --ne0 3e4 --plasma-beta 3 --theta-e-max 20 --mbh 6.5e9 --nu-ghz 230 --tb-range 8,11 --tonemap aces
```

## Errors and lessons
1. **Entirely black image** with `Ti/Te = 10`: it was not a bug; `Θ_e ≈ 1` and `B ≈ 1.5 G` give `ν_c ≈ 6e6 Hz` and at 230 GHz `exp(−1.89 x^{1/3})` with `x ≈ 4e4` is ~1e-28. I measured the factor before concluding.
2. **Misleading parametrisation:** getting a high `Θ_e` with `Ti/Te < 1` would mean electrons hotter than the ions. There is now an explicit `--theta-e-max`, and the pressure uses the torus's `T_i` and the model's `T_e`.
3. The two false "errors" in the image (dark ring at b ≈ 7–10 M and uniform bright background) were physics: cold gas next to the inner edge (`Θ_e < 0.3`) and optically thick gas (`T_b = g T_e`).

## Optimisation errors (caught by the tests)
1. **Extra `k_B`:** when rewriting the absorption, `h ν/(k T_e)` became `h ν/(k_B Θ_e m_e c²)` (an extra `k_B` in the denominator). The exponential's argument became huge, `α = 0`, and the tests failed: `j`/`α` GPU vs core off by 100 % and the thick limit at 1.8e7.
2. **Underflow in `float`:** the Planck coefficient `2h/c² ≈ 1.5e-47` is smaller than the smallest `float` (1.4e-45): `B_ν = 0` and `α = j/0` gave NaN in every image. Rescaled with `ν = 1e10 ν_s` (coefficient 1.5e-17).
3. **False alarm in the float vs double test:** 3 points with 100 % error, all with `j ≈ 1e-71 … 1e-166` (**1e-53 of the maximum**), where `float` underflows to 0; the test only compares `j > 1e-12 j_max`. Maximum error measured in the rest: `ρ` 1.7e-4, `B` 1.1e-4, `j` 6.0e-4, `α` 5.9e-4.

## Limitations
- Static, analytic torus; toroidal field with constant `β`; electrons with `Θ_e ∝ Θ_i`. No polarisation.
- Dexter's fit is valid for `Θ_e > 3` (to ~20 %); below that (and below 0.3, where we cut) the emission is underestimated or absent. The Leung et al. 2011 fit, which keeps `K₂`, was not read (see `22b3-radio-calibration.md`: it was read later).
- Single frequency; false-colour image, not in Jy/µas² (calibrated later in v2.2b3).
- Without the optical mode's disk/sky (they are separate modes).
