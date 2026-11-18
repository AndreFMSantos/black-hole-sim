# v2.1e — Novikov–Thorne/Page–Thorne thin disk in Kerr–Newman (2026-11-15)

André's request: "do it properly" for the disk of the charged variants (before: Newtonian profile, no redshift, wrong ISCO). Sources read by a subagent (see `lit-verification.md`): Page & Thorne 1974 (eq. 7, 11–13), Bardeen–Press–Teukolsky 1972 (Kerr), Pugliese, Quevedo & Ruffini 2013 (arXiv:1303.6250; E, L of circular orbits), Guo et al. 2024 (arXiv:2411.07914; redshift in KN), Hassani 2026 (general form of Ω).

## Physics (M = 1, a ≥ 0, q² = Q², a² + q² < 1, neutral particles, prograde equatorial circular orbits)
- Equatorial KN metric: `N = 2r − q²`; `g_tt = −1 + N/r²`, `g_tφ = −aN/r²`, `g_φφ = r² + a² + a²N/r²`; determinant of the (t, φ) block = −Δ, `Δ = r² − 2r + a² + q²`; `g_rr = r²/Δ`, `g_zz = 1` ⇒ `e^{ν+ψ+μ} = r`, as in Kerr (verified).
- Ω± = [−g_tφ,r ± √(g_tφ,r² − g_tt,r g_φφ,r)]/g_φφ,r; u^t = (−g_tt − 2Ωg_tφ − Ω²g_φφ)^{-1/2}; E = −(g_tt + Ωg_tφ)u^t; L = (g_tφ + Ωg_φφ)u^t.
- **ISCO** = (outermost) minimum of E(r): dE/dr = dL/dr = 0. **There is no closed formula in KN**: Pugliese only gives bounds (4 ≤ r_lsco ≤ 6 in RN; r_β ≤ r_lsco ≤ 6 with a ≠ 0) and the trend (r_lsco decreases with Q).
- **Flux** (Page–Thorne 11b + 12): `F = (1/(4π r)) (−Ω,r)(E − ΩL)⁻² ∫_{r_isco}^{r} (E − ΩL) L,r dr'`, per unit Ṁ; zero torque at the ISCO; matter inside the ISCO does not emit (as in the project's Kerr).
- **Redshift** (KN ZAMO observer, `ω = a(2r − q²)/A`, α = √(ΣΔ/A)): `g = [(1 − ωL)/α] / [u^t(1 − ΩL)]`, with the photon's `L`; the same design as `kerr::redshift`, with the KN emitter.

## Implementation
`core/include/bhs/novikov_thorne_kn.hpp` (`bhs::ntk`: `circ`, `isco`, `flux`, `luminosity`, `redshift`, `Profile`) and `render.cu` (`circ_kn`, `redshift(..., q2)`, `ensure_nt` with `ntk::Profile`, `rin` = `ntk::isco`). With `--charge Q` and `a² + Q² < 1` the disk uses the KN profile and redshift; with `Q = 0` the old Kerr code runs unchanged (byte-for-byte regression). SdS and naked KN (a² + Q² ≥ 1) keep the Newtonian profile, no frequency shift and `--rin`.

## Verification (CPU: `test_nt_kn.cpp`, 8 cases; GPU: `--selftest`)
| test | result |
|---|---|
| analytic metric derivatives = finite differences; det(t, φ) = −Δ | 1e-7 / 1e-12 |
| **Q = 0**: E, L, Ω, ISCO, flux vs closed Kerr formulas (`nt::`, `kerr::isco`) | 1e-11 / 1e-9 / 2e-6 |
| **a = 0**: ISCO vs root of the RN cubic (`kerr::rn_isco`) | 1e-8 |
| **E and L vs eq. (24)–(28) of Pugliese et al.** (eq. 28 with the missing factor `a` corrected; 5 pairs (a, Q) × 4 radii) | 1e-9 |
| **energy conservation** ∫4πrEF dr = 1 − E_isco (5 pairs incl. (0, 0.64), (0.5, 0.25), (0.9, 0.18), (0.3, 0.9)) | 1e-5 |
| ISCO decreases with charge; zero flux at the ISCO; values (a, Q) = (0.5; 0.5) → 3.732, (0.8; 0.3) → 2.665, (0.8; 0.5) → 2.108, (0.94; 0.3) → 1.530 | match an independent calculation (finite differences) to 3e-3 |
| near-extremal (a = 0.9, q² = 0.18): ISCO outside the horizon and efficiency larger than in Kerr | passes |
| redshift at Q = 0 = `kerr::redshift`; static limit of RN | 1e-10 / 1e-12 |
| **GPU vs CPU** (4 cases): T table (float) / redshift (double) | 3e-8 / 4e-16 |
ISCO and efficiency values η = 1 − E_isco (prograde, computed here): (a, q²) = (0, 0.64): 4.8908, η = 0.068716; (0.5, 0.25): 3.7321, 0.092499; (0.8, 0.09): 2.6653, 0.133200; (0.3, 0.90): 2.7166, 0.117466. **The Kerr–Newman PDFs contain no ISCO or efficiency table**: these numbers are ours, validated by the limits and the energy balance, not by comparison with published tables.

## Limits and warnings
- Neutral and prograde only (no retrograde, no charged particles); disk on the equator and thin; no emission inside the ISCO (the Guo/Chael plunge is not implemented); Eddington limb darkening off (it uses the Kerr gas frame).
- The flux of Hassani 2026 (arXiv:2605.13797) is **not KN** and its Table I does not match Kerr (a = 0.9: r_isco = 4.5 vs 2.321): it was not used as a reference.
- Guo 2024 uses a toy emissivity (not Novikov–Thorne): it only served for the redshift. Two of its points remain to be confirmed (and its claim that KN "is not asymptotically flat": KN is asymptotically flat).
- Q > M remains hypothetical and without astrophysical formation (see the epistemic status in `26-…`).
