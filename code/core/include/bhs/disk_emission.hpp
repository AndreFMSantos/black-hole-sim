#pragma once
// Disk emission (v0.9a): hardened spectrum (hardening factor), limb darkening and
// emission angle in the gas frame.
//
// Spectrum: modified black body I_nu = f^-4 B_nu(f T). The f^-4 factor keeps the bolometric radiance
// integral(I_nu dnu) = sigma T^4 / pi independent of f (the atmosphere only redistributes the energy to higher
// frequencies; f_col ~ 1.7-2.0 near Eddington, Shimura & Takahara 1995 ApJ 445:780, eq. 1). With I_nu/nu^3 invariant,
// the spectrum seen after the shift g is the same modified black body at T_obs = g T (v0.4 already used this), and
// the observed bolometric scales as g^4.
//
// Limb darkening (mu = cos of the angle between the ray and the disk normal IN THE GAS FRAME):
// - Chandrasekhar (1960, Radiative Transfer, Table XXIV, p. 248) electron-scattering atmosphere:
//     I(1)/I(0) = 1.26938 / 0.41441 = 3.0631 -> linear approximation I(mu) = B (1 + 2.06 mu) / (1 + 2.06 * 2/3).
// - Grey Eddington approximation:
//     I(mu) = B (1 + 1.5 mu) / 2  (I(1)/I(0) = 2.5).
// Both normalisations leave the emergent flux 2 int_0^1 I(mu) mu dmu = pi B unchanged (= sigma T^4).
// Note: `limb_eddington` is kept as an alias of `limb_chandrasekhar` for backwards compatibility with v0.9a-v3.1.
#include <cmath>

#include "blackbody.hpp"
#include "vec3.hpp"

namespace bhs::emission {

constexpr double kSigmaSB = 5.670374419e-8;  // W m^-2 K^-4
constexpr double kH = 6.62607015e-34, kC = 2.99792458e8, kKB = 1.380649e-23;

// B_nu(T), SI (W m^-2 Hz^-1 sr^-1)
inline double planck_nu(double nu, double T) {
  const double x = kH * nu / (kKB * T);
  if (x > 700.0) return 0.0;
  return 2.0 * kH * nu * nu * nu / (kC * kC) / std::expm1(x);
}
inline double modified_planck_nu(double nu, double T, double fcol) { return planck_nu(nu, fcol * T) / std::pow(fcol, 4); }

// Colour (linear sRGB, absolute radiance on the blackbody::rgb scale) of the modified black body; f = 1 is v0.8.
inline Vec3 rgb_hardened(double T, double fcol) {
  if (fcol == 1.0) return blackbody::rgb(T);
  const Vec3 c = blackbody::rgb(fcol * T);
  const double k = 1.0 / std::pow(fcol, 4);
  return {k * c.x, k * c.y, k * c.z};
}

inline double limb_chandrasekhar(double mu) { return (1.0 + 2.06 * mu) / (1.0 + 2.06 * 2.0 / 3.0); }
inline double limb_eddington_grey(double mu) { return (1.0 + 1.5 * mu) / 2.0; }
// Backwards-compatible alias (v0.9a-v3.1 used 1 + 2.06 mu under this name; that coefficient is Chandrasekhar 1960 Table XXIV).
inline double limb_eddington(double mu) { return limb_chandrasekhar(mu); }

// mu in the gas frame (Kerr equatorial circular orbit) for a photon with L = L_z/E and Carter constant Q
// (E = 1 at infinity). On the equator Theta = Q, so the tetrad component p^(theta) = sqrt(Q)/r; a boost along
// e_phi (the gas motion) does not touch it, only the energy: E_emit = -p.u = u^t (1 - Omega L). mu = p^(theta)/E_emit.
inline double emitter_mu(double a, double r, double L, double Q) {
  const double r32 = std::pow(r, 1.5);
  const double ut = (r32 + a) / (std::pow(r, 0.75) * std::sqrt(r32 - 3.0 * std::sqrt(r) + 2.0 * a));
  const double Omega = 1.0 / (r32 + a);
  const double e_emit = ut * (1.0 - Omega * L);
  return std::min(1.0, std::sqrt(std::max(Q, 0.0)) / (r * std::abs(e_emit)));
}

// ---- soft outer edge (v0.9a2) ----
// A real disk does not end in a circle: the gas thins out and the emission decays. The hard cut `r < r_out` of v0.8 is an
// artefact. The emission weight decays from 1 to 0 with a smoothstep between r_e (1 - s) and r_e (1 + s) (equals 1/2 at r_e);
// s = 0 reproduces the hard cut. The edge radius r_e depends on the azimuth (noise n in [0, 1]): r_e = r_out (1 + amp (2n - 1)).
inline double smoothstep(double a, double b, double x) {
  const double t = std::min(1.0, std::max(0.0, (x - a) / (b - a)));
  return t * t * (3.0 - 2.0 * t);
}
inline double edge_weight(double r, double r_edge, double soft) {
  if (soft <= 0.0) return r < r_edge ? 1.0 : 0.0;
  return 1.0 - smoothstep(r_edge * (1.0 - soft), r_edge * (1.0 + soft), r);
}
inline double edge_radius(double rout, double amp, double n01) { return rout * (1.0 + amp * (2.0 * n01 - 1.0)); }
// radius up to which the trace has to look for the disk so as not to cut the soft/irregular edge
inline double edge_trace_limit(double rout, double amp, double soft) { return rout * (1.0 + amp) * (1.0 + soft); }

}  // namespace bhs::emission
