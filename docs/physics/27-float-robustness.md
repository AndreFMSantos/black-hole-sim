# v2.7 — Float robustness: audit, defects found and fixes (2026-11-15)

Reason (André): the interactive app runs in **float** and must not stop because of NaN/inf. Tool: `v3_1-kerr-cuda-ui --audit-float N [--audit-seed S]`: N random scenarios (fixed seed) at 96×54 with 2 samples in float, plus fixed corners; it fails if the HDR has NaN/inf/negatives or if any pixel has to be "sanitised"; it warns if a ray does not terminate (steps = max_steps) or a frame takes > 3 s. It covers Kerr (a up to 0.9999), Kerr–Newman, naked KN, SdS, wormhole and radio torus, camera from near the horizon to 2000 M, poles, FOV 1–170°, relativistic observer (β up to 0.99), layer/turbulence/spot/slow light, bloom, glare and HDR10. A short version (120 scenarios, seed 777) runs inside `--selftest` (and therefore in `ctest`).

## What it found (first run: 16 failures in 400 scenarios)
| # | defect | effect | fix |
|---|---|---|---|
| 1 | **`dphif` (the φ equation) without q² in Δ** in the GPU Kerr–Newman integrator (my error in the v2.1a port) | wrong azimuth φ in all KN rays with a ≠ 0, and a division by ~0 at r = r₊ of Kerr (inside the KN horizon) ⇒ NaN; **my earlier explanation that the naked singularity was "ill-conditioned" was wrong**: those 139 different fates were this | `Δ = r² − 2r + a² + q² (− Λr⁴/3)` also in `dphif`; new test: GPU vs CPU azimuth (|Δφ| = 4e-5, equal to the Kerr control) and the naked case goes to **0** different fates |
| 2 | a **NaN step error was accepted as good** (`NaN > 1` is false) | corrupted rays carried on | rejected (`!isfinite`), and if it persists 40 times the ray ends absorbed (only affects cases with NaN: the normal path is identical) |
| 3 | slow-light `tref` with `log(r/2 − 1)`, NaN for r < 2 | whole frame NaN with the camera between the Schwarzschild and the Kerr horizon (high spin, `--hr`, `--slowlight`) | argument bounded below |
| 4 | layer: `u^t` with the root of a negative inside the circular photon orbit; `e_emit` ≈ 0 near the horizon | NaN/inf in pixels | no emission at those points |
| 5 | **exactly a = 1** (ISCO at the horizon, degenerate disk) | NaN | `--a` limited to 0.9999 (Thorne's limit is 0.998) |
| 6 | camera **exactly on the axis** (θ = 0°/180°) with extreme spin | 1–2 NaN pixels | camera θ limited to [0.001°, 179.999°] |
| 7 | (safety net) a single NaN contaminates the bloom pyramid and the glare and ruins the frame | whole frame | non-finite value ⇒ black, counted in `Stats::sanitized` (the audit requires 0) |

## Result
After the fixes: **9200 scenarios (23 seeds × 400), 0 failures, 0 warnings**; `ctest` 3/3 and Kerr byte-identical (the first version of the φ fix altered the bits of the float case; the new term went into a separate branch to preserve the old Kerr code).

## Limits
The audit is at 96×54 and 2 samples: it does not cover high resolution or all modes (GRMHD mode is only tested with a dump); the random scenarios are not exhaustive; the v3 video and the old images are not affected by defects 1–7 except the KN renders with a ≠ 0 (the image `v21e-kerr-newman-disk.png` was made **before** the fix of defect 1 and is regenerated in the asset clean-up). The charge ramp of the v3 video uses a = 0 (RN), where the φ equation lacks the missing term.
