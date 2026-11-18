# Credits and third-party material

This repository holds only our own code, our own images and the scripts that fetch everything else. Papers are cited, never copied: see [docs/REFERENCES.md](docs/REFERENCES.md).

## Software fetched at build time (not redistributed here)

| What | Licence | How it is used |
|---|---|---|
| [GLFW](https://github.com/glfw/glfw) 3.4 | zlib | window and input (CMake FetchContent) |
| [doctest](https://github.com/doctest/doctest) 2.4.11 | MIT | unit tests (CMake FetchContent) |
| [tinyexr](https://github.com/syoyo/tinyexr) 1.0.8 (+ miniz) | BSD-3-Clause | reading and writing EXR (CMake FetchContent) |
| [Dear ImGui](https://github.com/ocornut/imgui) 1.93.0 | MIT | interactive panel; fetched by `code/tools/fetch_imgui.sh` |
| Roboto Medium (shipped with Dear ImGui, `misc/fonts`) | Apache-2.0 | panel font, compiled into the binary from the fetched copy |
| [NVIDIA OptiX headers](https://github.com/NVIDIA/optix-dev) 9.1.0 | NVIDIA SDK licence (accepted by whoever fetches them) | optional denoiser; fetched by `code/tools/fetch_optix.sh`, never committed |
| [AthenaK](https://github.com/IAS-Astrophysics/athenak) (commit `c5a0d7f`) | BSD-3-Clause | our own GRMHD simulation; built and run from its repository, see [athenak/](athenak/README.md). Cite Stone et al. 2024, arXiv:2409.10384 |

## Data fetched by scripts (not redistributed here)

| What | Credit / licence | Script |
|---|---|---|
| NASA SVS *Deep Star Maps 2020* (`starmap_2020_8k.exr`), background sky | NASA/Goddard Space Flight Center Scientific Visualization Studio; Gaia DR2: ESA/Gaia/DPAC. NASA media guidelines | `code/tools/fetch_sky.sh` |
| WISE all-sky infrared mosaic (PIA15482), second sky of the wormhole | NASA/JPL-Caltech/UCLA. NASA media guidelines | download PIA15482 from NASA Photojournal, convert with `code/tools/img_to_exr.py` |
| Illinois GRMHD library v3 snapshots | Dhruv et al. 2025, ApJS 277, 16 (arXiv:2411.12647); CC BY 4.0 | `code/tools/fetch_grmhd.sh`, then `code/tools/convert_grmhd.py` |

## Methods and parameters taken from the literature

The physics follows published work, cited where it is used in the code and listed in [docs/REFERENCES.md](docs/REFERENCES.md). The main ones: Carter (1968) and Gralla & Lupsasca (2020) for Kerr geodesics; Novikov & Thorne (1973) and Page & Thorne (1974) for the thin disk; Fishbone & Moncrief (1976) for the torus; James, von Tunzelmann, Franklin & Thorne (2015) for the Interstellar black hole and wormhole; the Event Horizon Telescope papers on M87* and Sgr A* for the radio calibration; Narayan, Johnson & Gammie (2019), Papoutsis et al. (2023) and Dauser et al. (2014) for the spherical inflow, the jet walls and the lamppost reflection fraction.

## AI assistance

Most of the code and notes were written with AI coding agents under the author's direction; every result was checked against an independent oracle (see the physics notes).
