---
tags: [coding, black-hole-sim, physics, xray]
created: 2026-11-18
status: CPU physics done and verified (E4 part 1); GPU emission model and UI X-ray view NOT done
---
# Lamppost corona (E4), part 1: the emissivity profile on the analytic oracle

Code: `core/include/bhs/lamppost.hpp`, tests `core/tests/test_lamppost.cpp` (4 cases); it uses `kerr_analytic.hpp` (note 38). Source of the model: the relxill family
(Dauser et al. 2013; Wilkins & Fabian 2012). **This is an X-ray phenomenon**: in visible light the corona is not seen. The honest product of this module is an emissivity
profile / reflected-X-ray map, not "a prettier disk" (decided with André on 2026-11-17: an X-ray view, said so in the UI).

## The model and the derivation (each factor tested)

A static point source on the axis at BL radius h, emitting isotropically in its own frame; δ = polar angle of the photon's direction from +z in that frame.

| Quantity | Formula | How it was checked |
|---|---|---|
| Constants of motion | L = 0 (every geodesic through the axis), `Q = (h²+a²)² sin²δ/Δ(h) − a²`, `dr/ds = (h²+a²) cos δ` (E = 1, Mino time); `(dr/ds)² = R(h) = (h²+a²)² cos²δ` | against `kerr::init_from_zamo` at θ = 1e-4 (static ≈ ZAMO there), 3 spins × 3 heights × 3 angles × 2 azimuths; the first integral algebraically |
| Local energy of the source | `E_source = 1/α(h) = sqrt((h²+a²)/Δ(h))` per unit E at infinity | same test |
| Rays with Q < 0 | never reach the equator (they circle the axis and fall in) | `sin δ < a√Δ/(h²+a²)` finds no hit |
| Emissivity (photons per unit disk area as the gas measures it, per unit gas proper time) | `ε_N(r) ∝ sin δ |dδ/dr| / r`. Area element `γ√(g_rr g_φφ) dr dφ`; `u^t dt` for the time; `u^t/γ = 1/α_ZAMO`; the factors `√(Δ/A)` and `1/α` cancel | the flat-space limit below |
| Energy flux | × `g = E_disk/E_source = u^t(r) √(Δ(h)/(h²+a²))`, `u^t` of a prograde circular orbit (Bardeen et al. 1972), `E_disk = u^t(1 − ΩL) = u^t` since L = 0 | not independently tested (follows from the definitions) |

## Verification

- **Flat-space limit** (a = 0, h = 500, r from 0.3 h to 8 h): ε / [h/(h²+r²)^{3/2}] is constant to 3 %, and the slope between 40 h and 160 h is −3.00 ± 2 % (the Newtonian lamppost, `r⁻³` far away).
- **Schwarzschild, against a quadrature of the orbit equation**: the radius where a photon from the axis crosses the equatorial plane, `α(r) = ∫ b du/√(1 − b²u²(1−2u)) = π/2`, for 4 heights × 5 impact parameters (55 % to 99 % of b_c): 10+ cases, agreement 1e-9 relative.
- **Kerr**: ε > 0 and finite, the table is ordered, rays below the axis cone are lost, and a lower lamppost concentrates the emissivity inward (`ε(2r_in)/ε(20r_in)` larger for h = 3 than for h = 10; Wilkins & Fabian 2012).

Measured, 4000 samples, ~3.3 s per table (0.8 ms per photon):

| a | h | r_in | ε(2 r_in) / ε(20 r_in) | ε(20 r_in) / ε(200 r_in) | far slope (200 h–800 h) | table range of r |
|---|---|---|---|---|---|---|
| 0.9 | 3 | 2.32 | 1438 | 770 | −2.70 | 2.3 – 3.2e3 |
| 0.9 | 6 | 2.32 | 456 | 1129 | −2.91 | 2.3 – 3.7e3 |
| 0.9 | 10 | 2.32 | 126 | 1031 | −2.99 | 2.4 – 3.1e4 |
| 0.9 | 30 | 2.32 | 7.4 | 651 | −2.97 | 2.5 – 3.1e5 |
| 0.998 | 6 | 1.24 | 126 | 1081 | −3.01 | 1.2 – 9.6e3 |
| 0 | 30 | 6.0 | 66 | 945 | −2.97 | 6.1 – 3.7e5 |

(a = 0, h = 3 is the Schwarzschild photon-sphere radius: 25.8 s and only 618 points, the sweeps diverge logarithmically there.)

## Findings

- **The upward photons light the outer disk.** In flat space only photons emitted downward reach the plane. In a strong field the hole bends the nearly horizontal ones down (δ = π/2 lands at r = 45.6 for h = 10, a = 0.9; at r = 3.54 for h = 3), and everything beyond comes from photons emitted UPWARD and bent back. Sampling only δ > π/2 left the table silently without any radius beyond 3.54 for h = 3, and the contrast test returned 1.
- **That needed the oracle to follow a ray that moves out, turns at an outer root of P and falls back in** (`Radial::turn = 2`, substitution `u = ut + ξ²`; before, any root between the observer and infinity made the ray "unsupported"). Checked against long-double RK4 on 20 disk hits among 57 such rays: worst radius 1.5e-14 relative, azimuth 1.5e-14. A ray with roots on both sides of the observer (a bound radial motion) stays unsupported (negligible for the lamppost).
- **These photons are meridional rays** (L = 0): the case where the step-by-step integrator is unreliable (note 38). The closed-form polar motion handles L = 0 (u₊ = 1, `Pc` and `phi_lam` skipped, no `l_min` floor: `trace_consts`).

## Not done (E4 part 2: the X-ray view)

1. A GPU emission model that reads a table `ε(r)` (like the Novikov-Thorne LUT) and draws the reflected X-ray intensity `I_obs = g⁴ I_emit` with `I_emit ∝ ε_E(r) × albedo/π` on the thin disk; `--disk-model lamppost --lamppost-h H`.
2. The UI: a scenario / disk type labelled as X-ray, with h and spin; the table is cached per (a, h, r_in) (3 s at 4000 samples; 1000 samples would be ~1 s).
3. The disk inside the ISCO (plunging region) is not illuminated here (photons crossing the plane at r < r_in continue, as in `kerr::Disk`).
4. Decisions for André: the spectral treatment (bolometric reflected intensity vs a power-law X-ray band with a reflection fraction), and whether to draw only the reflected image or also the direct corona (a point: no).
