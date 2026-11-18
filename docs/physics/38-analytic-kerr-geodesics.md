---
tags: [coding, black-hole-sim, physics, oracle]
created: 2026-11-18
status: CPU oracle done and verified; closed-form radial part and any GPU kernel NOT done (decision below)
---
# Analytic Kerr null geodesics: a second, independent oracle (O4)

Code: `core/include/bhs/kerr_analytic.hpp` (`bhs::kerr_an`), tests `core/tests/test_kerr_analytic.cpp` (11 cases, 1290 checks; the whole core suite is 175/175).
Source: Gralla & Lupsasca, *The Null Geodesics of the Kerr Exterior*, PRD 101, 044032 (2020), arXiv:1910.12881. Same conventions as `kerr::trace`: the ray is followed
backwards from the observer, `s >= 0` is the backward Mino time, E = 1, λ = L, η = Q, `dφ/ds = -dφ/dτ`. Kerr only (q² = 0).

## What is closed form and what is not

| Part | Method | Status |
|---|---|---|
| Polar motion, η > 0 | `μ(s) = √u₊ sn(w | m)`, `w = w₀ + a√(-u₋) s`, `m = u₊/u₋ < 0`; `(dμ/ds)² = a²(u₊ - μ²)(μ² - u₋)`, `u₊u₋ = -η/a²` | closed form (Jacobi sn via AGM; negative parameter by the imaginary-modulus transformation) |
| Polar motion, η < 0 (vortical, near-axis rays of a polar observer) | `μ = ±√u₊ dn(w | k²)`, `k² = 1 - u₋/u₊`, `dw/ds = a√u₊`, `0 < u₋ < u₊`; no equator crossing | closed form |
| Polar part of φ | `∫λ/(1-μ²) ds` = incomplete Π (Carlson RF, RJ), period 2K handled by `2 Π_c` per period | closed form |
| Equatorial crossings | zeros of sn: `w = 2jK` | closed form |
| Radial motion | `u = 1/r`, `P(u) = R/r⁴ = 1 + c₂u² + c₃u³ + c₄u⁴`; Mino integrals `∫g du/√P` by tanh-sinh; the turning point `u_t` (smallest positive root, found from the monotone pieces of P) removed with `u = u_t - ξ²` | **numerical quadrature, not Legendre/Carlson forms** |
| `r(s)` at a crossing | inversion of the radial integral by bisection | numerical |
| Radial part of φ | `∫ a u(2 - aλu)/(1 - 2u + a²u²) du/√P` along the same path | numerical |

So it is independent of any step-by-step integrator (no ODE, no step control, no pole guard, no velocity reprojection), which is what an oracle needs. It is not a fast kernel.

## Verification (what each check is against)

1. **Special functions** against their definitions: Jacobi identities and derivatives for m ∈ {0.3, 0.9, -0.05, -0.7, -5, -40}; F, K, Π against tanh-sinh of the defining integrals (m down to -25, n up to 0.95); tanh-sinh against `∫x^-1/2`, `∫1/√(x(1-x))` (error 0 after extending the range to t = 6; with t = 3.4 the truncated tail cost 1.3e-10).
2. **Against `kerr::trace`** (DP45, atol 1e-14, rtol 1e-13, r_far 1e12), a = 0, 0.5, 0.9, 0.998, θ_o = 1°…90°, both looking toward and away from the hole: 1728 rays on the base grid, **0 fate mismatches** (escaped, captured, disk hit); sky direction worst 2.9e-11 = exactly b/r_far of the integrator's stop at 1e12 (b = 28), disk radius 3.8e-10 relative (the integrator's Hermite crossing), azimuth ≤ 5e-10.
3. **Against RK4 in long double** on the Carter equations (a third method, independent of both): polar μ(s) and φ to 1e-9 / 1e-8 for ordinary, vortical and meridional rays; the crossings are where RK4 finds μ = 0; disk crossings up to the 3rd (`skip` = 2): 278 hits, 215 with a radial turning point, 68 with `skip > 0`, worst radius 2e-9 relative, azimuth 8e-10.
4. **Critical curve**: escape and capture sweeps across the photon ring (4 spins), worst sky error 1e-11.

## Findings (bugs and traps, in the order they were hit)

- **`kerr::trace` convention.** `(nr, nth, nph)` is the direction of the PHOTON, opposite to the viewing one. Looking at the hole is `nr = +cos δ`. The first tests used `-cos δ`: only escaping rays, a trivially easy test. Both senses are tested now.
- **Outward rays and roots.** For a ray moving away, a root of P closer to the hole than the observer is irrelevant; treating it as a "bound region" rejected most of the image.
- **η < 0 rays are physical and common for polar observers** (θ_o = 30°, a ≥ 0.9, the centre of the image). They need the dn solution above.
- **The reference can be the wrong side.** RK4 of the SECOND-order equations does not enforce `(dr/ds)² = R`. Initial velocities from `init_from_zamo` are off the first integrals by 1.7e-16 relative; that grew to 4e-9 in the radius of a third-image crossing, which looked like an oracle error and was not (with velocities `√R`, `√M` in long double the RK4 converges to the oracle: 9e-11 at 2.4M steps, error ÷16 per doubling). Conversely, a ray that starts AT its turning latitude has `dμ/ds = 0` exactly (nth = 0); forcing `√M(μ₀)` (rounding noise ~1e-8) injected a spurious velocity into the reference. The reference now starts on the first integrals except when the data's velocity is ~0.
- **A real oracle bug found this way:** `asin(sn)` near sn = 1 is off by √(2e-16) = 1.4e-8; the integrand of Π near the pole is 1/(1-n) ≈ 3000, and φ gained 1.2e-6. The amplitude is now `atan2(sn, cn)` everywhere, and the initial phase uses `u₊ - μ₀² = M(μ₀)/(a²(μ₀² - u₋))` from the initial velocity instead of a difference of two nearly equal numbers.
- **`kerr::trace` (CPU DP45) is not reliable for meridional rays** (|L| < 1e-3, the `l_min` floor applies): at tolerances 1e-12 and 1e-14 it disagrees with itself by 8e-5 in the disk radius and 0.5 rad in φ (the turning latitude sits ~1e-9 from the pole; the reprojection onto √M is ill-conditioned, the same mechanism as O3 found for the GPU kernels in note 33). The oracle agrees with the long-double RK4 on these. The comparison test leaves them to the RK4 test (`degenerate` counter). Rays near the pole (θ_o = 1°) are also at 1.5e-8 in φ from the DP45 where the oracle matches RK4 to 3e-10.
- Near the critical curve, long-double for the root and the deflated cubic `C(u)` was tried (a root error of 1e-16 is amplified by 1/(gap) in `C(u_t)`); it changed nothing at the 1e-9 level (the cause was the reference, above) and was kept because it is free.

## Speed, and the decision

`tm.cpp` (2304 rays, a = 0.9, θ_o = 80°, disk to r = 20; same counts everywhere: 1879 disk hits, 144 escaped, 281 captured):

| Tracer | µs / ray (CPU, one core) |
|---|---|
| `kerr::trace`, DP45, tol 1e-8 | 9.1 |
| `kerr::trace`, DP45, tol 1e-11 | 35.5 |
| `kerr_an::trace` (this oracle) | 336.6 |

The oracle is 37× slower than the integrator at image accuracy, because every integral is a tanh-sinh quadrature (hundreds of evaluations) inside a bisection. **Decision: no GPU kernel from this code.** A kernel only makes sense after the radial part is also closed form (Carlson RF for `∫du/√P` over the real-root / complex-pair cases, Jacobi inversion for `r(s)` instead of bisection, Π by partial fractions for the radial φ), which is a separate spike (O4b) with unknown payoff: in double the DOP853 path is already 356 ms at 1080p, and the oracle's value today is as a reference.

## Use it for

Replacing "the double reference tracer" in checks of the GPU integrators: the accuracy claims of O1–O3 were measured against another integrator; this is an independent truth for escaping rays, disk crossings and higher-order images. A tool that writes the oracle result per pixel in the `--dump-rays` format would make `tools/raycmp.py` compare float/double GPU against it (task O4c).

## Addendum (sprint S1, 2026-11-18): why the CPU DP45 fails on meridional rays, and what was done

Measured on one ray (a = 0.9, θ_o = 60°, r_o = 40, L at the floor l_min = 1e-4, disk hit at r = 4.2054997543 per the oracle):

| atol | steps | error in the disk radius | note |
|---|---|---|---|
| 1e-8 | 211 | 4.5e-8 | |
| 1e-10 | 889 | 6e-10 | |
| 1e-12 | 189 200 | 5.2e-5 | same wrong value for every pole step limit (`pole_kappa` 0 … 0.5) |
| 1e-14 | 14.1 M | no hit | |

Cause: the integrator's state is μ = cos θ; the ray turns at σ = 1 − μ² = 3e-10, where μ (spacing 1.1e-16) carries σ to only ~7e-7 relative. Asking for 1e-12 cannot be honoured, and the step control chases noise (hence the step explosion). It is not the reprojection and not the pole step limit.
Done: `Trace::min_sigma` and `Trace::precision_limited(atol)` (σ_min < 1e-6 and atol < 1e-11); test over 3 spins × 2 angles × 3 meridional tilts: ≤ 1.4e-7 relative error at atol 1e-10, flag false at 1e-10, true at 1e-12.
Not done (S1b, optional): integrate χ = 1 − |μ| near the poles so the tolerance means what it says; the oracle is the reference for these rays meanwhile.

## Addendum (sprint S6a = O4c, 2026-11-18): the oracle as a per-ray truth for the GPU

`v4_0-metric-api --shot x --dump-rays f.bin --dump-oracle` writes the same BHSRAYS1 dump as the GPU, computed by `kerr_an::trace` on the CPU threads (exact at infinity, `r_far = 0`; disk from the ISCO or `--rin` to `--rout`; same camera rays). It makes `tools/raycmp.py` an independent comparison: the GPU integrators share no code with it.
`tests/oracle_rays.py` (ctest `v4_0_oracle_rays`), 4 scenes × 9216 rays:

| scene | double (DOP853, tol 1e-9): sky max / disk r max | float (production): sky p99 / disk r p99 |
|---|---|---|
| a = 0.9, i = 70° | 9.3e-11 / 2.7e-10 | 1.0e-5 / 1.8e-4 |
| a = 0.5, i = 30° | 6.9e-10 / 1.7e-10 | 2.5e-4 / 4.0e-5 |
| a = 0.998, i = 85° | 1.2e-10 / 1.1e-10 | 1.6e-5 / 1.8e-4 |
| a = 0, i = 60° | 3.6e-10 / 1.3e-10 | 7.9e-5 / 1.8e-4 |

No fate or disk-hit difference in double in any scene; one ray in 9216 differs in float (a = 0.9). Speed: 0.7 s for 20 736 rays on 11 threads.

## Addendum (sprint S6b = O4b, 2026-11-18): Newton instead of bisection, and why the closed form stays undone

The crossing radius is the root of `A(u) - s_j = 0` (or `T`), and bisection spent ~50 tanh-sinh quadratures per crossing. `detail::newton_bracket` uses the exact derivatives (`dA = 1/sqrt(P)`, `dT = 2/sqrt(sigma C)`, the integrands themselves) with the bracket as a safeguard: ~5 quadratures.
Same results (against the bisection dump: sky identical, disk r max 6.5e-15, disk phi 3.6e-15, 0 fate differences over 20 736 rays; core 190/190).

| | before | after |
|---|---|---|
| 20 736 rays, 11 threads, wall | 729 ms | 174 ms |
| single thread, 3600 rays (a = 0.9, i = 70°, to the disk) | 336 µs/ray (measured earlier on another ray set, 37× the DP45) | **75 µs/ray = 7.9× the DP45 (9.5 µs/ray at tol 1e-9)** |

A fully closed-form radial part (Carlson R_F / R_J for the four root configurations, Jacobi inversion of r(s), no quadrature at all) would remove the remaining quadratures (the `s_end` and `phi` integrals): expected gain 3–5×, i.e. roughly DP45 speed. It would only matter for a GPU kernel from the oracle, which has no consumer (the GPU already has DOP853 and the oracle already serves as the truth, 0.2 s for a 20 000-ray dump). Left undone on purpose; the task stays in the backlog with this measurement.
