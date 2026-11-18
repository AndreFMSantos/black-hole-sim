#pragma once
// v4.0: the table of supported (metric, disk) combinations. Only combinations whose physics is defined are offered (decision of 2026-11-16, notes/v4-metric-api-design.md):
//  - Novikov-Thorne needs E(r), L(r), Omega(r) of circular orbits and an ISCO: defined for Kerr and Kerr-Newman (Page-Thorne generalisation, validated); Schwarzschild-de Sitter would need its own
//    derivation (to validate); a naked singularity has no ISCO.
//  - Ellis wormhole: no stable circular orbits (g_tt = -1: no potential; the only one, at the throat, is unstable): no accretion disk.
//  - Spherical inflow (E1): defined for Kerr-Newman too (neutral gas, same free fall), but the volume tracer is Kerr-only: offered in Kerr.
//  - Thick layer, radio torus, parametric RIAF and GRMHD data were built in Kerr: in another metric the emission would not solve the gas equations.
//  - Naked Kerr-Newman and SdS keep the simplified Newtonian profile (needs r_in), as before.
#include "render.hpp"

namespace bhs::kcuda {

enum class MetricKind { Kerr, KerrNewman, NakedKN, SdS, Wormhole };
enum class DiskKind { None, NovikovThorne, Newtonian, Thick, Torus, Grmhd, Riaf, Spherical, JetCones, Lamppost };

inline MetricKind metric_kind(const Params& p) {
  if (p.wormhole) return MetricKind::Wormhole;
  if (p.lambda > 0.0) return MetricKind::SdS;
  if (p.charge > 0.0) return p.a * p.a + p.charge * p.charge >= 1.0 ? MetricKind::NakedKN : MetricKind::KerrNewman;
  return MetricKind::Kerr;
}

inline bool disk_supported(MetricKind m, DiskKind d, const char** why = nullptr) {
  auto no = [&](const char* w) { if (why) *why = w; return false; };
  if (d == DiskKind::None) return true;
  if (d == DiskKind::Lamppost && m != MetricKind::Kerr) return no("the lamppost emissivity is derived for Kerr (the photon orbits of Kerr-Newman / de Sitter / the wormhole were not worked out)");
  switch (m) {
    case MetricKind::Kerr: return true;
    case MetricKind::KerrNewman:
      if (d == DiskKind::NovikovThorne || d == DiskKind::Newtonian) return true;
      return no(d == DiskKind::Spherical || d == DiskKind::JetCones ? "this inflow model is defined here, but the volume tracer is Kerr-only for now" : "in Kerr-Newman only the thin disk exists (the layer, the torus and the GRMHD data were built in Kerr)");
    case MetricKind::NakedKN:
      if (d == DiskKind::Newtonian) return true;
      return no(d == DiskKind::NovikovThorne ? "a naked singularity has no ISCO: Novikov-Thorne is undefined (use the Newtonian profile with --rin)" : "only the thin Newtonian disk (with --rin) exists for a naked singularity");
    case MetricKind::SdS:
      if (d == DiskKind::Newtonian) return true;
      return no(d == DiskKind::NovikovThorne ? "Novikov-Thorne in Schwarzschild-de Sitter is not derived/validated yet (use the Newtonian profile with --rin)" : d == DiskKind::Spherical ? "spherical inflow needs a start at infinity, which de Sitter does not have" : "only the thin Newtonian disk (with --rin) exists in Schwarzschild-de Sitter");
    case MetricKind::Wormhole: return no("the Ellis wormhole has no stable circular orbits (g_tt = -1): no accretion disk");
  }
  return false;
}

}  // namespace bhs::kcuda
