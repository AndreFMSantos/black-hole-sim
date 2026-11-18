# v2.1a/c — Kerr-Newman, Reissner–Nordström, naked Kerr and Schwarzschild–de Sitter on the GPU

CPU core: roadmap-v2-variants-and-views §6 (`kerr.hpp` with `q2`, `sds.hpp`). This note describes the port to `kerr_cuda.cuh` / `render.cu`.

## What changes (and what does not)
- **Kerr-Newman** (`--charge Q`): only `Δ(r) = r² − 2r + a² + Q²` (also in `R(r) = (r²+a²−aL)² − ΔK`; `Δ' = 2r−2`, unchanged) and the ZAMO angular velocity `ω = a(2r − Q²)/A`. Horizon `1 + √(1 − a² − Q²)`; if `a² + Q² > 1` there is no horizon: rays reaching `r = 0.05` come out **black** (flagged non-physical on the CPU).
- **Schwarzschild–de Sitter** (`--lambda Λ`, `a = Q = 0`, `0 < Λ < 1/9`): `Δ = r² − 2r − Λr⁴/3`, `Δ' = 2r − 2 − 4Λr³/3`; the ray "escapes" on reaching `0.9999 r_c` (cosmological horizon, roots by the trigonometric method) and the `(θ, φ)` direction there is the sky one. The camera has to be between the horizons.
- **Assumed restrictions** (checked in `parse`): thin disk, static observer, Newtonian profile, no frequency shift (`redshift()` and the NT formulas are Kerr's); with `a ≠ 0` or `Λ > 0` and a disk `--rin` is needed (the KN ISCO is not implemented; RN `a = 0`: `rn_isco`). No volume, torus or GRMHD.
- **Kerr intact**: with `Q = 0`, `Λ = 0` the 3 cases of the byte-for-byte regression (`ctest v07d_regression`) are identical. A first port failed (float: 2 of 3 cases changed) because the Λ terms shared `r*r` with the Kerr code and altered the FMA contractions; solved with non-contractible multiplications (`__fmul_rn`/`__dmul_rn`) and only when `Λ ≠ 0`.

## Oracles (`--selftest`, `selftest_v21`, `selftest_v21c`; pass)
| test | result |
|---|---|
| KN a = 0.5 Q = 0.5; RN Q = 0.8 (GPU vs `kerr::trace`, 1600 rays, double, tol 1e-9) | 0 different fates; first disk crossing |dr| ≤ 7.6e-7; exit direction |dμ| ≤ 4.1e-5 |
| Kerr control a = 0.5, Q = 0 | 0 different fates; |dμ| 4.1e-5 (same: it is the last step at r_far = 5·10³, not the charge) |
| naked singularity a = 0.5 Q = 0.95 | **0** different fates (**correction 2026-11-15**: the earlier 139/1600 were not ill-conditioned rays, they were the `dphif` defect, see `27-float-robustness.md`) |
| SdS: angular radius of the shadow (GPU, bisection to 1e-6 in tan α) vs `sds::static_shadow_sin` | Λ = 0.002 (r_O = 10): 9.1e-8; Λ = 0.01 (r_O = 8): 4.9e-7; Λ = 0.03 (r_O = 5): 4.0e-7 (relative error) |

Images: `assets/v21-reissner-nordstrom-gpu.png` (Q = 0, 0.6, 0.9, 1.1; a = 0, thin Newtonian disk, θ = 78°; the shadow shrinks and at Q = 1.1 it disappears, leaving nested images of the disk), `assets/v21c-schwarzschild-de-sitter-gpu.png` (Λ = 0, 0.01, 0.03, 0.045; r_O = 5.5, no disk: the shadow shrinks with Λ).

## Limits
No consistent disk (NT, ISCO, KN redshift) in this phase (see `21e-kerr-newman-disk.md`); SdS only a = 0; Kerr–de Sitter and Λ < 0 do not exist; the naked singularity shows black where the physics is undefined.
