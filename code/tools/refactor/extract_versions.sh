#!/usr/bin/env bash
# Extracts each milestone of the CUDA app from Git history into its own folder under apps/ (2026-11-16 refactor).
# Each folder is the code AS IT WAS at that commit, compiled against the current (backwards-compatible) core/.
# Usage: tools/refactor/extract_versions.sh   (run from code/; idempotent; overwrites the folders)
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=01-Projects/black-hole-sim/code/apps/v06b-kerr-cuda   # the app's path in the history (all commits)
ROOT=$(git rev-parse --show-toplevel)
# folder             commit/tag   prefix for targets/tests
VERSIONS=(
  "v0_6b-kerr-cuda    bhs-v0.6c   v0_6b"
  "v0_7-cinematic-cuda bhs-v0.7e  v0_7"
  "v0_8-flight-cuda   5fddaeb     v0_8"
  "v0_9-disk-cuda     e2df574     v0_9"
  "v2_x-variants-cuda 4b60f84     v2_x"
)
# the HEAD of today's app is apps/v3_1-kerr-cuda-ui (already its own folder)
for row in "${VERSIONS[@]}"; do
  read -r dir rev pre <<<"$row"
  dest="apps/$dir"; rm -rf "$dest"; mkdir -p "$dest"
  git -C "$ROOT" archive "$rev" "$SRC" | tar -x -C "$dest" --strip-components=5
  # unique target and test names per folder; update renamed v0_5-realtime-gpu include path
  sed -i -e "s/v06b-kerr-cuda/$dir/g" -e "s/NAME v06b_/NAME ${pre}_/g" -e "s/NAME v07d_regression/NAME ${pre}_v07d_regression/" -e "s/NAME v08c_no_optix/NAME ${pre}_no_optix/" -e "s/NAME v06b_selftest/NAME ${pre}_selftest/" "$dest/CMakeLists.txt"
  sed -i 's|\.\./v05-realtime-gpu/|../v0_5-realtime-gpu/|g' "$dest"/*.cpp "$dest"/*.hpp 2>/dev/null || true
  # Backport (2026-11-19, from v4.0 O3 §4): conditional velocity reprojection near turning points (10-18x better float p99)
  python3 - "$dest/kerr_cuda.cuh" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = """    yn[1] = copysign(sqrt(fmax(Rf(yn[0], cS), S(0))), yn1);
    yn[3] = copysign(sqrt(fmax(Mf(yn[2], yn[5], cS), S(0))), yn3);"""
new = """    // Backport (v4.0 O3): conditional velocity reprojection (avoid ill-conditioned sqrt near turning points)
    {
      const S tolr = S(at) + S(rt) * fabs(yn[0]), tolm = S(at) + S(rt) * fabs(yn[2]);
      if (yn1 * yn1 >= S(200) * fabs(S(k7[1])) * tolr) yn[1] = copysign(sqrt(fmax(Rf(yn[0], cS), S(0))), yn1);
      if (yn3 * yn3 >= S(200) * fabs(S(k7[3])) * tolm || yn[5] < S(0.1)) yn[3] = copysign(sqrt(fmax(Mf(yn[2], yn[5], cS), S(0))), yn3);
    }"""
assert old in s, f"reprojection block not found in {p}"
open(p, "w").write(s.replace(old, new, 1))
PY
  if [ "$dir" = "v2_x-variants-cuda" ]; then
    # Backport (2026-11-16): v2.7 swallowed `k.phi = ...` inside a // comment; + (2026-11-19): EHT R_high convention (note 41)
    python3 - "$dest/render.cu" "$dest/kerr_cuda.cuh" "$dest/main.cpp" <<'PY'
import sys
# 1. render.cu: k.phi fix + EHT R_high convention
p = sys.argv[1]; s = open(p).read()
old_phi = "limite 0.001 graus k.phi = p.phi_deg * M_PI / 180.0;"
assert old_phi in s
s = s.replace(old_phi, "limite 0.001 graus\n  k.phi = p.phi_deg * M_PI / 180.0;", 1)
old_kp = "double g_startx1, g_dx1, g_hslope, g_ms, g_palpha, g_pxt, g_pnorm, g_rho_unit, g_b_unit, g_rhigh, g_ce, g_rcap, g_gam;"
assert old_kp in s
s = s.replace(old_kp, old_kp[:-1] + ", g_te_c;", 1)
old_el = "else if (p.el_model == 1) { const float b2 = o.beta * o.beta; const float tite = (float(p.g_rhigh) * b2 + 1.f) / (1.f + b2); thE = thI * float(synch::kMp / synch::kMe) / (1.f + tite); }"
new_el = """else if (p.el_model == 1) { const float b2 = o.beta * o.beta; const float R = (float(p.g_rhigh) * b2 + 1.f) / (1.f + b2); thE = thI * float(p.g_te_c) / (2.f + R); }
    else if (p.el_model == 3) { const float b2 = o.beta * o.beta; const float tite = (float(p.g_rhigh) * b2 + 1.f) / (1.f + b2); thE = thI * float(synch::kMp / synch::kMe) / (1.f + tite); }"""
assert old_el in s
s = s.replace(old_el, new_el, 1)
old_gam = "k.g_hslope = h.hslope; k.g_ms = h.mks_smooth; k.g_palpha = h.poly_alpha; k.g_pxt = h.poly_xt; k.g_pnorm = h.fmks().poly_norm(); k.g_gam = h.gam;"
new_gam = old_gam + "\n    k.g_te_c = (synch::kMp / synch::kMe) * (2.0 / 3.0) / std::max(1e-6, h.gam - 1.0);"
assert old_gam in s
s = s.replace(old_gam, new_gam, 1)
open(p, "w").write(s)

# 2. kerr_cuda.cuh: SdS initial dr/dtau with lam + r_c step cap + Hermite exit interpolation (v4.0 O3 §2-3)
pk = sys.argv[2]; sk = open(pk).read()
old_rc1 = "double r_plus = disc >= 0.0 ? 1.0 + sqrt(disc) : 0.0, r_far_eff = r_far;"
old_rc2 = "r_plus = mid; r_far_eff = 0.9999 * hi;"
assert old_rc1 in sk and old_rc2 in sk
sk = sk.replace(old_rc1, "double r_plus = disc >= 0.0 ? 1.0 + sqrt(disc) : 0.0, r_far_eff = r_far, r_c = r_far;", 1)
sk = sk.replace(old_rc2, "r_plus = mid; r_c = hi; r_far_eff = 0.9999 * hi;", 1)
old_y = "S y[6] = {S(r0), S(-delta_d(r0, a, q2) * in.pr), S(mu0), S(s0 * in.ptheta), S(0), S(s0 * s0)};"
assert old_y in sk
sk = sk.replace(old_y, "S y[6] = {S(r0), S(-delta_d(r0, a, q2, lam) * in.pr), S(mu0), S(s0 * in.ptheta), S(0), S(s0 * s0)};", 1)
old_cap = "    S yn[6];"
new_cap = "    if (lam > 0.0 && y[1] > S(0)) h = fmin(h, S(0.5) * (S(r_c) - y[0]) / y[1]);\n    S yn[6];"
assert old_cap in sk
sk = sk.replace(old_cap, new_cap, 1)
old_ex = "#pragma unroll\n    for (int i = 0; i < 6; ++i) y[i] = yn[i];"
new_ex = """    if (lam > 0.0 && yn[0] > rfar && yn[1] > S(0) && y[0] <= rfar) {
      auto hm = [&](int i, S s) { const S s2 = s * s, s3 = s2 * s; return (S(2) * s3 - S(3) * s2 + S(1)) * y[i] + (s3 - S(2) * s2 + s) * h * k1[i] + (S(-2) * s3 + S(3) * s2) * yn[i] + (s3 - s2) * h * k7[i]; };
      S lo = 0, hi = 1;
      for (int it = 0; it < (sizeof(S) == 8 ? 60 : 28); ++it) { const S m = S(0.5) * (lo + hi); (hm(0, m) < rfar ? lo : hi) = m; }
      const S s = S(0.5) * (lo + hi);
      out.fate = 0; out.mu_end = fmin(1.0, fmax(-1.0, double(hm(2, s)))); out.phi_end = double(hm(4, s));
      return;
    }
#pragma unroll
    for (int i = 0; i < 6; ++i) y[i] = yn[i];"""
assert old_ex in sk
sk = sk.replace(old_ex, new_ex, 1)
open(pk, "w").write(sk)

# 3. main.cpp: support --electrons rhigh-single
pm = sys.argv[3]; sm = open(pm).read()
old_m = 'else if (is("--electrons")) { const char* m = argv[++i]; p.electrons = std::strcmp(m, "rhigh") == 0 ? 1 : std::strcmp(m, "prop") == 0 ? 2 : 0; }'
new_m = 'else if (is("--electrons")) { const char* m = argv[++i]; p.electrons = std::strcmp(m, "rhigh") == 0 ? 1 : std::strcmp(m, "prop") == 0 ? 2 : std::strcmp(m, "rhigh-single") == 0 ? 3 : 0; }'
assert old_m in sm
open(pm, "w").write(sm.replace(old_m, new_m, 1))
PY
  fi
  printf '%s %s\n' "$dir" "$(git -C "$ROOT" rev-parse --short "$rev")" >>"$dest/.extracted-from"
  echo "$dir <- $rev ($(ls "$dest" | wc -l) files)"
done
