# v2.1d — Wormhole (Ellis and Interstellar) on the GPU, with two skies

CPU core and theory: `core/include/bhs/wormhole.hpp` (James, von Tunzelmann, Franklin & Thorne 2015, arXiv:1502.03809 [read by a subagent]); `ds² = −dt² + dl² + r(l)² dΩ²`, every geodesic is planar.

## Implementation (`render.cu`: `wh_trace`, block `p.wh` in the main kernel)
- **Model in the main kernel**, not a separate kernel: camera, jitter, bloom, glare, HDR and tone mapping are those of the rest. Camera at rest at `l = --wh-l` (default 8 ρ), angular position `(--theta, --phi)`; the viewing direction `v = −n` (in the frame `e_l, e_θ, e_φ`) gives `cos α = v_l`, `B = r(l_c) sin α` and the transverse direction; integrate `dl/dt = p_l, dp_l/dt = B² r'/r³, dψ/dt = B/r²` (RK4, step `0.02 min(r, 5ρ + 0.2|l|)`, in float) until `|l| > l_max` (100 ρ) plus the flat tail `ψ += asin(B/r)`.
- **Direction on the sky:** `D = cos ψ P₀ + sin ψ T` (`P₀` the camera's angular position, `T` the transverse direction); `l_final > 0` → sky 1; `l_final < 0` → **sky 2** (`--sky2 EXR`; without it the 1st is repeated; the interactive UI loads `assets/sky/wise/wise_8k.exr` by itself since 2026-11-16, `--no-sky2` turns it off). The per-quad texture gradients do not cross the boundary between skies (no wrong *mip* on the crest).
- **CLI:** `--wormhole ellis[,ρ] | interstellar | ρ,a,M` (Interstellar: ρ = 1, 2a = 0.01ρ, W = 0.05ρ ⇒ a = 0.005, M = W/1.42953 = 0.03498), `--wh-l`, `--wh-lmax`, `--sky2`. No disk, charge, Λ or volume (`parse` refuses).
- **2nd sky:** WISE (NASA/JPL-Caltech/UCLA, PIA15482, false-colour infrared, 19000×9500 8 bit, downloaded by a subagent) converted by `tools/img_to_exr.py` (sRGB → linear, gain for a mean of 0.015; aesthetic choice). The map is a false-colour display image, not radiometry, and **I did not verify the frame** (galactic vs equatorial).

## Oracles (`--selftest`, `selftest_v21d`)
GPU (float) vs `bhs::wh::trace` (CPU, double, same `l_max`), 598 rays per case (excluding ±2e-3 around the critical ray):
| case | different sides | max \|Δψ\| | flat tail \|ψ(100) − ψ(10⁵)\| |
|---|---|---|---|
| Ellis ρ = 1, l_c = 8 | 0 | 1.3e-5 | 1.3e-6 |
| Interstellar (ρ = 1, a = 0.005, M = 0.03498), l_c = 8 | 0 | 5.0e-5 | 1.5e-5 |
Images: `assets/v21d-wormhole-interstellar.png` (l = 4), `assets/v21d-wormhole-ellis-vs-near.png` (Ellis at l = 4 and Interstellar at l = 2.2): a sphere with the other sky and a thin (Interstellar) or wide (Ellis) lens halo.

## Limits
No rotation (only the static wormhole; the film's does not rotate), no disk or objects; the camera only at rest; finite `l_max` (the flat tail ignores curvature beyond 100ρ, measured error 1e-5); the choice of skies is aesthetic.
