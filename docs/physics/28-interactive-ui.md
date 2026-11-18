---
tags: [coding, physics, black-hole-sim]
created: 2026-11-15
updated: 2026-11-16
---

# v3.1 — Interactive UI with Dear ImGui (2026-11-15; fixes 2026-11-16)

André's request: a UI to choose among the versions/options and test them in real time. Decision: **Dear ImGui** (MIT) on top of the existing interactive loop (GLFW + OpenGL + CUDA), optional at build time: without `code/third_party/imgui` the app compiles with the bitmap HUD as before. The app lives in `apps/v3_1-kerr-cuda-ui/` (see `AGENTS.md` §3).

## How to run
```bash
code/tools/fetch_imgui.sh                 # Dear ImGui v1.93.0 (pinned commit) into code/third_party/imgui (outside Git)
cmake -S code -B code/build && cmake --build code/build -j --target v3_1-kerr-cuda-ui
code/build/apps/v3_1-kerr-cuda-ui/v3_1-kerr-cuda-ui      # interactive window: panel on the left, F1 hides/shows
```
The panel has: **Versions and variants** (16 scenarios, from v0.3 to v2.2g, each with a coloured epistemic-status badge and a note on hover), **Black hole** (metric: Kerr / Kerr–Newman–RN / Schwarzschild–de Sitter / wormhole; spin, charge, Λ, wormhole type and distance), **Disk** (none / thin NT / thin Newtonian / layer / radio torus / radio GRMHD with a selector and *play* of the 300 snapshots 2 M apart), **Camera and observer** (distance, θ, φ, FOV, static/orbit/infall), **Image** (exposure, sky, bloom, tone mapping, cinematic, float/double precision) and **Help**. The label at the top changes with the parameters (`ui::classify`): *observed*, *model-dependent*, *hypothetical* (e.g. charge, Λ, wormhole, naked singularity; it warns that the real Λ is ~1e-30), *aesthetic*.

## Implementation
`apps/v3_1-kerr-cuda-ui/ui.hpp/.cpp` (panel, scenarios, classification) and hooks in `main.cpp` (`#ifdef BHS_IMGUI`): the panel only edits `kcuda::Params` and the camera; the `key` vector of the loop (now with charge, Λ, wormhole, model, layer, radio, `dump_serial`…) detects the change and restarts the accumulation. The mouse/keyboard callbacks ignore events when ImGui captures them. Requests that need the main loop (apply a scenario, cinematic, load a GRMHD dump) go in a `ui::Request`. Dear ImGui is one `Combo`/`Slider` per parameter: there is no duplicated state.
Restrictions (as in the CLI): charge/Λ/wormhole only with a thin disk (or none); a² + Q² ≥ 1 turns the disk Newtonian with `r_in = 6`; Λ only with a = 0; radio needs the camera at ≥ 80–150 M.

## Verification
- `--ui-scenarios out.png` (also in `ctest` as `v3_1_ui_scenarios`): applies each scenario **without a window**, renders at 480×270 in float, requires a finite HDR and 0 sanitised pixels, prints the status and writes a contact sheet. **16 of 16 OK.**
- `v3_1_ui_wormhole_sky2` (with the real skies): the wormhole scenarios must change the image when the 2nd sky is on (`--no-sky2` turns it off).
- `v3_1_ui_english`: scans the UI/HUD strings for Portuguese.

## Fixes of 2026-11-16 (André's feedback)
1. **Mouse could not move horizontally.** Root cause: `render.cu` had `k.phi = p.phi_deg * M_PI / 180.0;` **inside a `//` comment** since v2.7 (the camera azimuth never reached the kernel; only `theta` and `dist` worked). Fix: the assignment on its own line; test `selftest_camera_phi` (a hot spot at a fixed azimuth makes the image depend on the camera φ without needing a sky: 0 → 90° changes the image by 3.8 %, 360° = 0° to 1e-9). Backported to the `v2x` snapshot (`tools/refactor/extract_versions.sh`); v06b–v09 were not affected. Side effect: GPU renders with `--phi ≠ 0` made since v2.7 (time-lapse, some assets) ignored the azimuth; see `tasks.md`.
2. **The wormhole showed only one sky.** Cause: the 2nd sky was only loaded by `--sky2` on the command line; the UI never called `load_sky2`. The UI (and `--ui-scenarios`) now load `assets/sky/wise/wise_8k.exr` and show a warning in the panel if it is missing; `--no-sky2` forces the 1st sky on both sides. The CLI keeps the old default so existing renders do not change.
3. **UI in English.** Panel, scenarios, badges, HUD, window title and `--help` (and the test-output words "FAILED"/"limit"). The check is automated (`v3_1_ui_english`).

## Not seen by me
I never saw the panel on screen (the session has no display); it compiled, started and rendered the scenarios. André's visual check is a task in `tasks.md`.

## To do
Apply the `Metric`/`DiskModel`/`Observer` refactor underneath (today the scenarios are sets of flags); configurable 2nd sky in the panel; save an image from the panel; show the status in the bitmap HUD too.
