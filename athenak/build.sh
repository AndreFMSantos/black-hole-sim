#!/usr/bin/env bash
# Clones AthenaK at the pinned commit and builds it in double precision with Kokkos/CUDA (see README.md).
set -euo pipefail
COMMIT=c5a0d7f9155a70149931bf0be5a4ffb673f2532a
ARCH=${KOKKOS_ARCH:-BLACKWELL120}     # e.g. ADA89, AMPERE86, HOPPER90
DATA=${BHS_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
SRC=${ATHENAK:-$DATA/athenak}
if [ ! -d "$SRC/.git" ]; then git clone --recursive https://github.com/IAS-Astrophysics/athenak.git "$SRC"; fi
git -C "$SRC" checkout "$COMMIT" && git -C "$SRC" submodule update --init --recursive
cmake -S "$SRC" -B "$SRC/build_d" -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_${ARCH}=ON -DPROBLEM=fluids/gr_torus
cmake --build "$SRC/build_d" -j "$(nproc)"
echo "built: $SRC/build_d/src/athena"
