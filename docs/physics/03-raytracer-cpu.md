---
tags: [coding, physics, black-hole-sim]
created: 2026-11-11
---

# v0.3 — CPU ray tracer: the 3D shadow

## Idea: trace backwards
For each pixel we ask "where did this light come from?". We do the reverse path: start at the observer, follow the geodesic to infinity and read the sky colour in that direction. If the ray falls into the horizon, the pixel is black.

## Spherical symmetry = a 2D problem per pixel
Schwarzschild is spherically symmetric, so each ray stays in a plane that contains the hole and the observer. For the pixel direction `d`, with `ĉ` = hole→observer direction:
- `cos α = d·ĉ`, `t̂` = normalised transverse component of `d` (defines the plane).
- Impact parameter (static observer at r = D): `b = D sin α / sqrt(1 − 2M/D)`.
- `trace_from(D, b, inward)` integrates the Binet equation (v0.2) and returns `φ_end`.
- Direction the light came from: `v = cos φ_end · ĉ + sin φ_end · t̂`. Without gravity it would be φ_end = π, i.e. `v = −ĉ`.

## Background
Procedural stars (cells per cube face, hash) + an azimuth/elevation coordinate grid every 15°, with the equator in orange. The grid is what shows the lensing; without it only noise would be visible.

## Implementation
`core/include/bhs/{vec3,png,observer,stars}.hpp`, `trace_from` in `schwarzschild.hpp`, app `apps/v0_3-raytrace-cpu`. Hand-written PNG ("stored" deflate + CRC32/Adler32), no dependencies; multithreaded by rows (`std::thread` + atomic counter), configurable SSAA. Default RK4 step in φ = 5e-3.

## Oracles
| Oracle | Expected | Obtained |
|---|---|---|
| Shadow radius in the image (D = 30, fov 40°, 1280×720) | asin(b_c·sqrt(1−2/D)/D) → 167.87 px | 168.0 px (centre line, black pixels) |
| Shadow edge in the test | captured below / escapes above α_c (±0.1%) | passes |
| CRC32("123456789") / Adler32("Wikipedia") | 0xCBF43926 / 0x11E60398 | passes |
| trace_from(r0 = 1e4) vs trace(∞) | φ_end differs only by the initial arc asin(b/r0) | within 1e-4 |
| No hole (`--nobh`) | normal rectilinear grid | confirmed visually |

## What you see (assets/v03-shadow.png)
Black shadow; Einstein ring around it with the grid bent; multiple images (second, third... order) piling up near the edge; the stars stretch tangentially. The brown "tips" in the horizontal are the equator (orange line) being bent around the hole: it appears several times because of the higher-order images.

## Performance
1920×1080, SSAA 3×3, 11 threads (Mac): 13.7 s. Reference without the hole: 0.1 s. Cost dominated by rays near the shadow edge (many steps). This is what the GPU (v0.5) will turn into real time.

## Next
v0.4: thin accretion disk in the equatorial plane. Each ray crosses the plane z = 0 one or more times; at each crossing with r between the ISCO (6M) and an outer radius the disk is seen, with black-body colour, gravitational redshift and Doppler.
