# Our own GRMHD simulation with AthenaK

The "own GRMHD" scenarios (a MAD flow around a black hole with a = 0.9375, used for the radio images calibrated to M87* and Sgr A*) come from a simulation we ran with **AthenaK**. AthenaK is not part of this repository: it is fetched from its own repository at a pinned commit, built, and run with the input file in this folder.

## Credit

- **AthenaK**: J. M. Stone et al. 2024, *AthenaK: A Performance-Portable Version of the Athena++ AMR Framework*, [arXiv:2409.10384](https://arxiv.org/abs/2409.10384). Code: <https://github.com/IAS-Astrophysics/athenak>, BSD 3-Clause licence. If you use the simulation or the snapshots, cite the AthenaK paper.
- The input is AthenaK's own `inputs/grmhd/gr_fm_torus_mad_8_4.athinput` (a Fishbone & Moncrief 1976 torus with a MAD field, the setup of the EHT code comparison, Porth et al. 2019, ApJS 243, 26) with two additions: `pgen_name = gr_torus` and a restart file every 500 M.
- The snapshot reader (`code/tools/convert_athenak.py`) imports AthenaK's `vis/python/bin_convert.py` from your AthenaK clone; it is not copied here.

## Reproduce

| Item | Value |
|---|---|
| AthenaK commit | `c5a0d7f9155a70149931bf0be5a4ffb673f2532a` (2026-09-11) |
| Build | double precision (default), Kokkos CUDA, `Kokkos_ARCH_BLACKWELL120=ON` (RTX 50 series; pick your GPU's architecture), `-DPROBLEM=fluids/gr_torus` |
| Mesh | as in the input, with 16³ mesh blocks (command-line override, see `run.sh`) |
| Length | 10 000 M (t_g); history every 10 M, `mhd_w_bcc` dumps every 50 M (~27 GB in total) |
| Cost measured | 6.6 s per M on an RTX 5080, ~18.3 h for 10 000 M |

```bash
./build.sh            # clones AthenaK at the pinned commit (with its Kokkos submodule) and builds it in double
./run.sh              # runs the MAD torus to t = 10 000 M into $BHS_DATA/athenak/run (BHS_DATA defaults to <repo>/data)
# convert dumps for the renderer (numpy and h5py; AthenaK's reader is imported from the clone):
python3 ../code/tools/convert_athenak.py $BHS_DATA/athenak/run/bin/torus.mhd_w_bcc.00200.bin mad_00200.bgm --a 0.9375
```

The UI looks for the converted snapshots in `$BHS_DATA/athenak/run_d/bgm_hi/mad_NNNNN.bgm` (the folder name of our run); copy or link them there, or type another folder in the panel.

## What did not work (for the record)

AthenaK has a single-precision build option. We tried it for a ~5.6× speed-up: it needed small type fixes to compile, and the MAD run then produced NaN in the magnetic field next to the excised region near the horizon from t ≈ 40 M, with every mitigation tried (three). The production run is therefore in **double precision on unmodified upstream code**; no AthenaK patch is needed to reproduce it.
