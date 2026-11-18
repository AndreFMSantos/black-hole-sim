---
tags: [coding, physics, black-hole-sim]
created: 2026-11-12
---

# v0.7a — Novikov-Thorne (Page-Thorne) for Kerr

Up to v0.6 the radial brightness of the disk followed the **Newtonian** Shakura-Sunyaev shape, `T⁴ ∝ r⁻³(1 − √(r_in/r))`. In Kerr, near the horizon, that is wrong: the geometry changes the binding energy, the orbital velocity and the "drag". The right model for a thin disk of equatorial circular orbits with zero torque at the ISCO is **Novikov-Thorne (1973) / Page-Thorne (1974)**.

## Physics
For a circular orbit of radius r and spin a (M = 1, prograde):

```
E(r) = (r^{3/2} − 2 r^{1/2} + a) / (r^{3/4} √(r^{3/2} − 3 r^{1/2} + 2a))
L(r) = (r² − 2a r^{1/2} + a²)   / (r^{3/4} √(r^{3/2} − 3 r^{1/2} + 2a))
Ω(r) = 1 / (r^{3/2} + a)
```

The energy each ring releases while sinking (because the disk loses angular momentum outwards and binding energy inwards) gives the flux, **from the definition**:

```
F(r) = [1/(4π√−g)] · (−Ω,r)/(E − ΩL)² · ∫_{r_in}^{r} (E − ΩL) L,r dr'      √−g = r in the equatorial plane
```

`F(r_in) = 0` (zero torque); far from the hole `F → 3Ṁ/(8π r³)(1 − √(r_in/r))` (Newton). Implementation in `core/include/bhs/novikov_thorne.hpp` (`nt::flux`, `nt::Profile`): Gauss-Legendre integration over geometric segments; `Profile` tabulates `T ∝ F^{1/4}` normalised to 1 at the maximum, in ln r (2048 points), ready for a 1D texture on the GPU.

Note: I used the **integral of the definition** and not the closed Page-Thorne formula (very easy to memorise wrongly). The proof that it is right is independent: energy conservation.

## Oracle: radiative efficiency
The total luminosity seen at infinity, `L∞ = ∫ 4π r E F dr`, must equal the energy the matter loses down to the ISCO, `1 − E(r_isco)`:

| a | r_isco | 1 − E_isco | L∞ (integral of F) | r of the T peak (NT) | Newtonian peak 49/36·r_in |
|---|---|---|---|---|---|
| 0 | 6.000 | 0.057191 (5.72 %) | 0.057191 | 9.551 | 8.167 |
| 0.5 | 4.233 | 0.082118 | 0.082118 | 6.611 | 5.762 |
| 0.9 | 2.321 | 0.155753 | 0.155753 | 3.442 | 3.159 |
| 0.998 | 1.237 | 0.320994 (32 %) | 0.320994 | 1.581 | 1.684 |

Agreement to 1e-5 (test) for the four spins. The efficiency values match those in the literature (5.7 %, ~15.6 %, ~32 %); the Schwarzschild peak at ~9.5M too.

## Tests (`core/tests/test_nt.cpp`, 5 new, 53 in total)
E, L, Ω reduce to the closed Schwarzschild forms at a = 0 (via `u^t`); `F(r_isco) = 0` and F > 0 outside (a = 0, 0.5, 0.9, 0.998); Newtonian limit at r = 10⁵ (2e-3); L∞ = 1 − E_isco (1e-5); normalised profile with maximum 1.

## Integration
`v0_6-kerr-cpu --profile nt|newtonian` (nt by default). Comparison 960×540, a = 0.9, θ = 80°: MAE 1.18/255, 9.1 % of pixels > 4, 1.1 % > 16 (the NT profile is more compact and moves the peak to larger r at small a). Image: `assets/v07a-nt-a09.png`.

## Limitations
Thin, opaque disk; static observer; the colour still uses a black body with T ∝ F^{1/4} (no colour correction / hardening factor, no returning radiation); ΔT in extremal Kerr may need more resolution in ln r near the ISCO.

## Next
v0.7b: photon ring by image order. v0.7d (omarchy): this phase's LUT goes to the GPU.
