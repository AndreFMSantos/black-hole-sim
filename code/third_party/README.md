# third_party (outside Git, except this file)

`code/third_party/optix-dev/` — official OptiX headers (public repository `https://github.com/NVIDIA/optix-dev`), **tag `v9.1.0`**,
commit **`f1f6dd803f3159992d248178f6e09421c6eb8b6d`** (OptiX 9.1.0, ABI 118). Fetched with `code/tools/fetch_optix.sh` (idempotent; pins the commit).

- Only the **headers** (header-only); the denoiser runtime is the `libnvoptix` that ships with the NVIDIA driver (tested with 615.78.08).
- Licence: *NVIDIA SOFTWARE DEVELOPER KITS, SAMPLES AND TOOLS LICENSE AGREEMENT (with distribution rights)* (`LICENSE.txt` in the cloned folder), which every user accepts when fetching it. Compliance: the headers are **not** copied or redistributed (this folder is in `.gitignore`); the repository only carries the script and our own code.
- Optional: without this folder, or with `-DBHS_OPTIX=OFF`, `v0_6b-kerr-cuda` builds without the denoiser (stub) and `--denoise optix` says it is not available.

`code/third_party/imgui/` — Dear ImGui (https://github.com/ocornut/imgui, **MIT** licence), **tag `v1.93.0`**, commit `c1aaac65e896a45f81f47a4d68f9959e0f86c859`; fetched with `code/tools/fetch_imgui.sh` (idempotent). For the interactive UI (backends `imgui_impl_glfw` + `imgui_impl_opengl3`). Outside Git (this folder is in `.gitignore`); optional: without it `v0_6b-kerr-cuda` builds with the bitmap HUD.
