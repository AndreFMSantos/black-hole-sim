#pragma once
// E4 (refinement phase): lamppost corona, the CPU physics. See notes/physics/39-lamppost-corona.md.
//
// A point source on the rotation axis at Boyer-Lindquist radius h (theta = 0), static, emitting isotropically in its own frame. The photons are the rays that leave the axis:
//   * L = 0 (every geodesic through the axis has L_z = 0),  Q = (h^2 + a^2)^2 sin^2(delta) / Delta(h) - a^2,   dr/ds = (h^2 + a^2) cos(delta)   (E = 1, Mino time),
//     delta = polar angle of the photon's direction from +z in the source's static frame (delta > pi/2: toward the disk plane); (dr/ds)^2 = R(h) = (h^2+a^2)^2 cos^2(delta) holds.
//   * rays with Q < 0 never reach the equator ("vortical": they circle the axis and fall in).
//   * these are exactly the meridional rays (|L| < 1e-3) where the step-by-step integrator is unreliable (note 38), so the first equatorial crossing comes from
//     kerr_analytic.hpp (closed-form polar motion).
// Emissivity: photons per unit area of the disk measured by the gas (area element gamma sqrt(g_rr g_phi_phi) dr dphi) and per unit proper time of the gas (u^t dt):
//     eps_N(r)  ~  sin(delta) |d delta / d r| / r      (the factors sqrt(Delta/A) from the area and 1/alpha from u^t/gamma cancel: u^t = gamma / alpha_ZAMO)
// normalised so that in flat space it is the Newtonian lamppost  h / (h^2 + r^2)^(3/2)  (-> r^-3 far away). The energy flux carries one more factor g = E_disk / E_source:
//     g = u^t(r) sqrt(Delta(h) / (h^2 + a^2)),  u^t = (r^1.5 + a) / (r^0.75 sqrt(r^1.5 - 3 sqrt(r) + 2a))  (prograde circular orbit, Bardeen et al. 1972),
// since E_disk = -p.u = u^t (1 - Omega L) = u^t for L = 0 and the source's local energy is E_source = 1/alpha(h) = sqrt((h^2 + a^2)/Delta(h)) per unit E at infinity.
#include <algorithm>
#include <cmath>
#include <vector>

#include "kerr.hpp"
#include "kerr_analytic.hpp"

namespace bhs::lamppost {

inline double E_source(double a, double h) { return std::sqrt((h * h + a * a) / (h * h - 2.0 * h + a * a)); }
inline double carter_Q(double a, double h, double delta) {
  const double rr = h * h + a * a, Dl = h * h - 2.0 * h + a * a, s = std::sin(delta);
  return rr * rr * s * s / Dl - a * a;
}
inline double u_t_circular(double r, double a) {
  const double r15 = r * std::sqrt(r);
  return (r15 + a) / (std::pow(r, 0.75) * std::sqrt(r15 - 3.0 * std::sqrt(r) + 2.0 * a));
}

// Gas in the equatorial plane. Outside the ISCO: circular geodesic. Inside: plunging geodesic with the E and L of the ISCO (Mummery & Balbus 2022; Cunningham 1975 for the Kerr
// plunge). Equatorial Kerr (Sigma = r^2): u^t = [(r^2 + a^2 + 2 a^2/r) E - 2 a L/r] / Delta, u^phi = [(1 - 2/r) L + 2 a E/r] / Delta,
// (u^r)^2 = [(E (r^2 + a^2) - a L)^2 - Delta (r^2 + (L - a E)^2)] / r^4 (u^r < 0).
struct Gas { double E = 1.0, L = 0.0, ut = 1.0, uphi = 0.0, ur = 0.0; };
inline Gas gas(double a, double r, double r_isco) {
  Gas g;
  const double D = r * r - 2.0 * r + a * a;
  auto circ_EL = [&](double rc, double& E, double& L) {
    const double sr = std::sqrt(rc), den = std::pow(rc, 0.75) * std::sqrt(rc * sr - 3.0 * sr + 2.0 * a);
    E = (rc * sr - 2.0 * sr + a) / den; L = (rc * rc - 2.0 * a * sr + a * a) / den;
  };
  if (r >= r_isco) {
    circ_EL(r, g.E, g.L);
    g.ut = u_t_circular(r, a); g.uphi = g.ut / (r * std::sqrt(r) + a);
    return g;
  }
  circ_EL(r_isco, g.E, g.L);
  g.ut = ((r * r + a * a + 2.0 * a * a / r) * g.E - 2.0 * a * g.L / r) / D;
  g.uphi = ((1.0 - 2.0 / r) * g.L + 2.0 * a * g.E / r) / D;
  const double A = g.E * (r * r + a * a) - a * g.L, B = g.L - a * g.E;
  g.ur = -std::sqrt(std::max(0.0, (A * A - D * (r * r + B * B)) / (r * r * r * r)));
  return g;
}
// Energy of a photon with E = 1, L = 0 and Carter constant Q, as the gas measures it at radius r: E_gas = u^t - p_r u^r, p_r = pr_sign sqrt(R) / Delta, R = (r^2 + a^2)^2 - Delta (Q + a^2)
// (pr_sign = sign of dr along the photon's path, +1 outward). The energy the source measured is E_source (below), so g_source->gas = gas_energy / E_source.
inline double gas_energy(double a, double r, double r_isco, double Q, int pr_sign) {
  const Gas g = gas(a, r, r_isco);
  const double D = r * r - 2.0 * r + a * a, R = (r * r + a * a) * (r * r + a * a) - D * (Q + a * a);
  const double pr = pr_sign * std::sqrt(std::max(0.0, R)) / D;
  return g.ut - pr * g.ur;
}

struct Hit {
  bool hit = false;
  double r = 0.0;  // radius of the first equatorial crossing inside [r_in, r_out]
  double g = 0.0;  // E_gas / E_source there (the gas is circular outside the ISCO, plunging inside)
  int pr_sign = 0; // sign of dr along the photon path at the disk (+1 outward)
  int fate = 0;    // when there is no hit: 0 = escaped to infinity, 1 = captured by the hole, 2 = neither (max steps, unsupported)
};

// first crossing of the disk plane (r_in < r < r_out; skip: ignores that many crossings, as kerr::Disk::skip) of the photon leaving the axis at polar angle delta
inline Hit hit(double a, double h, double delta, double r_in, double r_out, int skip = 0) {
  Hit o;
  const double rr = h * h + a * a, Dl = h * h - 2.0 * h + a * a;
  if (!(Dl > 0.0)) return o;
  kerr::Disk disk; disk.r_in = r_in; disk.r_out = r_out; disk.skip = skip;
  kerr_an::Options opt; opt.l_min = 0.0;
  const kerr_an::Result res = kerr_an::trace_consts(a, h, 1.0, 0.0, carter_Q(a, h, delta), rr * std::cos(delta), 0.0, disk, opt);
  if (!res.supported) { o.fate = 2; return o; }
  if (!res.t.disk_hit) { o.fate = res.t.fate == kerr::Fate::Escaped ? 0 : res.t.fate == kerr::Fate::Captured ? 1 : 2; return o; }
  o.hit = true; o.r = res.t.disk_r; o.pr_sign = res.pr_sign;
  o.g = gas_energy(a, o.r, kerr::isco(a), carter_Q(a, h, delta), res.pr_sign) * std::sqrt(Dl / rr);
  return o;
}

struct Table {  // ordered by increasing r
  std::vector<double> r, eps_n, eps_e, g;  // photon-number emissivity (flat space: h/(h^2+r^2)^1.5), eps_e = eps_n <g^(Gamma-1)> (Gamma = 2: the energy emissivity), cell mean of g_source->gas
  double refl_frac = 0.0;                  // geometric reflection fraction N_disk / N_infinity (Dauser et al. 2014) for a disk out to the `r_disk_out` given to emissivity()
};

// eps(r) from the map delta -> r on n samples of x = -cos(delta) in (-1, 1): finite differences of delta(r) between neighbouring hits.
// Pairs with the wrong sign of dr (a non-monotone stretch of the map) are skipped.
// gamma: photon index of the source spectrum N(E) ~ E^-gamma (band model of note 40): eps_e = eps_n <g_sd^(gamma - 1)>; gamma = 2 is the energy emissivity (the bolometric view), 1 the photon number.
// r_disk_out: outer edge of the disk for the geometric reflection fraction N_disk / N_infinity (rays that land beyond it count as not reflected).
inline Table emissivity(double a, double h, double r_in, double r_out, int n = 3000, double gamma = 2.0, double r_disk_out = 1e6) {
  // x = -cos(delta) in (-1, 1): x > 0 photons toward the disk plane, x < 0 emitted upward. In flat space only x > 0 reach the plane; with a strong field the hole bends the nearly
  // horizontal photons down (x ~ 0 lands at finite r, e.g. 45.6 for h = 10, a = 0.9) and the outer disk is lit by photons emitted UPWARD and bent back (kerr_analytic turn = 2).
  std::vector<double> xs;
  for (int i = 0; i < n / 4; ++i) {
    const double x = std::pow(10.0, -5.0 + 5.0 * (i + 0.5) / (n / 4));  // log: reaches r ~ h / x
    xs.push_back(x); xs.push_back(-x);
  }
  for (int i = 0; i < n / 2; ++i) xs.push_back(-1.0 + 2.0 * (i + 0.5) / (n / 2));  // linear: resolves the strong-gravity part
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  struct P { double x, delta, r, g; };
  std::vector<P> p;
  double n_disk = 0.0, n_inf = 0.0;  // measure of x (= of solid angle / 4 pi x 2) that reaches the disk, and that escapes to infinity
  for (size_t i = 0; i < xs.size(); ++i) {
    const double x = xs[i];
    const double w = 0.5 * ((i + 1 < xs.size() ? xs[i + 1] : 1.0) - (i > 0 ? xs[i - 1] : -1.0));  // cell of the sample (the ends reach -1 and 1)
    const double delta = std::acos(-x);  // x > 0: pi/2 < delta < pi (downward)
    const Hit hh = hit(a, h, delta, r_in, std::max(r_out, r_disk_out));
    if (hh.hit) { if (hh.r <= r_out) p.push_back({x, delta, hh.r, hh.g}); (hh.r <= r_disk_out ? n_disk : n_inf) += w; }
    else if (hh.fate == 0) n_inf += w;
  }
  // Photon-number conservation: sin(delta) d(delta) = dx, so eps_N(r) r dr = dx. Each pair of neighbouring rays (a segment of the map x -> r with dr < 0) spreads its dx over
  // [r_{i+1}, r_i] with density ~ r^-2 in r; the segments of ALL branches are summed on a log-spaced grid of radial cells (cell average), so overlapping branches (rays that go round the hole and land at the same
  // radius as direct ones, a <~ 0.5 and low h) add up and the integral of eps_N r dr over the disk equals the measure of x that reaches it, exactly.
  struct Seg { double r0, r1, dx, g, gp; };  // r0 < r1; g = mean g_sd, gp = mean g_sd^(gamma-1)
  std::vector<Seg> sg;
  double rlo = 1e300, rhi = 0.0;
  for (size_t i = 0; i + 1 < p.size(); ++i) {
    const double dr = p[i + 1].r - p[i].r;
    if (!(dr < 0.0)) continue;  // a jump between branches (a gap of rays that miss the disk): not a segment
    sg.push_back({p[i + 1].r, p[i].r, p[i + 1].x - p[i].x, 0.5 * (p[i].g + p[i + 1].g), 0.5 * (std::pow(p[i].g, gamma - 1.0) + std::pow(p[i + 1].g, gamma - 1.0))});
    rlo = std::min(rlo, p[i + 1].r); rhi = std::max(rhi, p[i].r);
  }
  Table s;
  s.refl_frac = n_inf > 0.0 ? n_disk / n_inf : 0.0;
  if (sg.empty()) return s;
  const int nc = std::max(200, n);
  const double l0 = std::log(rlo), dl = (std::log(rhi) - l0) / nc;
  std::vector<double> mass(nc, 0.0), gmass(nc, 0.0), gpmass(nc, 0.0);  // photons (in x) per cell, and the same weighted by g and by g^(gamma-1)
  for (const Seg& e : sg) {
    const double norm = e.dx / (1.0 / e.r0 - 1.0 / e.r1);  // measure of x per unit r is norm / r^2 (the far-field law eps ~ r^-3; exact in the mass, irrelevant for narrow segments)
    int k0 = std::max(0, std::min(nc - 1, int((std::log(e.r0) - l0) / dl))), k1 = std::max(0, std::min(nc - 1, int((std::log(e.r1) - l0) / dl)));
    for (int k = k0; k <= k1; ++k) {
      const double a = std::max(e.r0, std::exp(l0 + dl * k)), b = std::min(e.r1, std::exp(l0 + dl * (k + 1)));
      if (b > a) { const double m = norm * (1.0 / a - 1.0 / b); mass[k] += m; gmass[k] += m * e.g; gpmass[k] += m * e.gp; }
    }
  }
  for (int k = 0; k < nc; ++k) {
    if (!(mass[k] > 0.0)) continue;
    const double ra = std::exp(l0 + dl * k), rb = std::exp(l0 + dl * (k + 1)), area = 0.5 * (rb * rb - ra * ra);
    const double g = gmass[k] / mass[k], en = mass[k] / area;
    s.r.push_back(std::sqrt(ra * rb)); s.eps_n.push_back(en); s.eps_e.push_back(en * gpmass[k] / mass[k]); s.g.push_back(g);
  }
  return s;
}

// Table for the GPU: log10 of the energy emissivity on a uniform ln r grid between the first and last radius of `t`, normalised so that its maximum is 0
// (linear interpolation of ln eps_E in ln r between the nodes, exact for a power law). slope_out = d log eps / d log r of the last segment, used to continue beyond the table.
struct LogTable {
  double lmin = 0.0, lmax = 0.0, slope_out = -3.0, refl_frac = 0.0;
  std::vector<float> log10e;
};

inline LogTable log_table(const Table& t, int n = 2048) {
  LogTable o;
  o.refl_frac = t.refl_frac;
  std::vector<double> r, le;  // strictly increasing r, positive eps_E
  for (size_t i = 0; i < t.r.size(); ++i)
    if (t.eps_e[i] > 0.0 && std::isfinite(t.eps_e[i]) && (r.empty() || t.r[i] > r.back())) { r.push_back(t.r[i]); le.push_back(std::log(t.eps_e[i])); }
  if (r.size() < 2 || n < 2) return o;
  o.lmin = std::log(r.front()); o.lmax = std::log(r.back());
  {  // outer slope: least squares of ln eps against ln r over the last e-fold of r (the last cell is only partly covered, so it is left out); a single segment is too noisy
    const double xe = std::log(r.back());
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = 0;
    for (size_t i = 0; i + 1 < r.size(); ++i) {
      const double x = std::log(r[i]);
      if (x < xe - 1.0) continue;
      sx += x; sy += le[i]; sxx += x * x; sxy += x * le[i]; ++m;
    }
    const double den = m * sxx - sx * sx;
    o.slope_out = (m >= 3 && den > 1e-12) ? (m * sxy - sx * sy) / den : (le.back() - le[le.size() - 2]) / (std::log(r.back()) - std::log(r[r.size() - 2]));
  }
  std::vector<double> v(n);
  size_t k = 0;
  double mx = -1e300;
  for (int i = 0; i < n; ++i) {
    const double x = o.lmin + (o.lmax - o.lmin) * double(i) / double(n - 1);
    while (k + 2 < r.size() && x > std::log(r[k + 1])) ++k;
    const double x0 = std::log(r[k]), x1 = std::log(r[k + 1]), f = std::min(1.0, std::max(0.0, (x - x0) / (x1 - x0)));
    v[i] = (1.0 - f) * le[k] + f * le[k + 1];
    mx = std::max(mx, v[i]);
  }
  o.log10e.resize(n);
  for (int i = 0; i < n; ++i) o.log10e[i] = float((v[i] - mx) / std::log(10.0));
  return o;
}

}  // namespace bhs::lamppost
