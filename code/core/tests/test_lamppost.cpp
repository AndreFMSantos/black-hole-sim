#include <doctest/doctest.h>

#include <bhs/kerr_analytic.hpp>
#include <bhs/lamppost.hpp>
#include <cmath>
#include <vector>

using namespace bhs;
namespace lp = bhs::lamppost;

// A point source on the symmetry axis at BL radius h (height h above the hole), static, isotropic in its own frame; delta = polar angle of the photon's
// direction from +z in the source frame (delta > pi/2: toward the disk plane). E4 of the refinement phase, note 39.

TEST_CASE("lamppost: constants of motion of a photon leaving the axis agree with init_from_zamo just off the axis") {
  for (double a : {0.0, 0.5, 0.9}) {
    for (double h : {3.0, 6.0, 20.0}) {
      for (double delta : {1.9, 2.4, 3.0}) {
        // a ZAMO at theta = 1e-4 is the static observer of the axis to ~1e-6 (omega * varpi -> 0); any transverse azimuth psi gives the same Q and L ~ 0.
        // (not 1e-6: 1 - mu^2 = 1e-12 is computed from mu = cos(1e-6) with a relative error of 2e-4)
        for (double psi : {0.7, 2.1}) {
          const double th = 1e-4;
          const auto in = kerr::init_from_zamo(a, h, th, std::cos(delta), std::sin(delta) * std::cos(psi), std::sin(delta) * std::sin(psi));
          INFO("a = " << a << " h = " << h << " delta = " << delta << " psi = " << psi);
          CHECK(std::fabs(in.L) < 2.0 * th * std::sqrt(h * h + a * a) * in.E_loc);  // L = varpi E_loc n_phi, varpi ~ sqrt(rr) sin(theta)
          CHECK(lp::carter_Q(a, h, delta) == doctest::Approx(in.Q).epsilon(1e-4).scale(1.0));
          CHECK(lp::E_source(a, h) == doctest::Approx(in.E_loc).epsilon(1e-4));
          // the first integral: (dr/ds)^2 = R(h) = rr^2 cos^2(delta)
          const double rr = h * h + a * a, Dl = h * h - 2.0 * h + a * a, K = lp::carter_Q(a, h, delta) + a * a;
          CHECK(rr * rr - Dl * K == doctest::Approx(rr * rr * std::cos(delta) * std::cos(delta)).epsilon(1e-12));
        }
      }
    }
  }
}

namespace {
// Schwarzschild, independent of everything above: a photon leaving the axis with impact parameter b sweeps, in its orbital plane (which contains the axis), the angle
// alpha(r) = int_r^h b dr' / (r'^2 sqrt(1 - b^2 (1 - 2/r') / r'^2)) on its way in (no turning point before r). The disk plane is alpha = pi / 2.
double sweep(double b, double r, double h) {
  // r' = 1/u: alpha = int_{1/h}^{1/r} b du / sqrt(1 - b^2 u^2 (1 - 2u))
  return b * kerr_an::tanh_sinh([&](double u, double, double) { return 1.0 / std::sqrt(std::max(1.0 - b * b * u * u * (1.0 - 2.0 * u), 1e-300)); }, 1.0 / h, 1.0 / r);
}
}  // namespace

TEST_CASE("lamppost: Schwarzschild impact radius r(delta) agrees with a quadrature of the orbit equation") {
  int tested = 0;
  const double bc = 3.0 * std::sqrt(3.0);
  for (double h : {6.0, 8.0, 20.0, 60.0}) {
    // b = h sin(theta') / sqrt(1 - 2/h) with theta' = pi - delta: choose it as a fraction of the critical impact parameter (above it the ray turns instead of crossing)
    for (double f : {0.55, 0.7, 0.85, 0.95, 0.99}) {
      const double theta_p = std::asin(f * bc * std::sqrt(1.0 - 2.0 / h) / h), delta = M_PI - theta_p;
      const double b = std::sqrt(lp::carter_Q(0.0, h, delta));
      REQUIRE(b == doctest::Approx(f * bc).epsilon(1e-12));
      const double rmin = 2.0 + 1e-9;
      if (sweep(b, rmin, h) <= 0.5 * M_PI) continue;  // captured before reaching the plane
      double lo = rmin, hi = h;  // alpha(r) decreases with r
      for (int i = 0; i < 200 && hi - lo > 1e-15 * hi; ++i) { const double mid = 0.5 * (lo + hi); (sweep(b, mid, h) > 0.5 * M_PI ? lo : hi) = mid; }
      const double r_ref = 0.5 * (lo + hi);
      const lp::Hit got = lp::hit(0.0, h, delta, 2.0 + 1e-6, 1e7);
      INFO("h = " << h << " b/b_c = " << f << ": reference r = " << r_ref << ", oracle r = " << got.r);
      REQUIRE(got.hit);
      CHECK(std::fabs(got.r - r_ref) / r_ref < 1e-9);
      ++tested;
    }
  }
  CHECK(tested >= 10);
}

TEST_CASE("lamppost: flat-space limit, emissivity ~ h / (h^2 + r^2)^(3/2), and r^-3 far away") {
  const double h = 500.0;
  const lp::Table t = lp::emissivity(0.0, h, 6.0, 1e7, 6000);
  REQUIRE(t.r.size() > 1000);
  // ratio to the Newtonian lamppost over r in [0.3 h, 8 h]: constant to O(2/h) up to the weak-field deflection
  double lo = 1e300, hi = 0.0;
  int n = 0;
  for (size_t i = 0; i < t.r.size(); ++i) {
    if (t.r[i] < 0.3 * h || t.r[i] > 8.0 * h) continue;
    const double flat = h / std::pow(h * h + t.r[i] * t.r[i], 1.5);
    const double ratio = t.eps_n[i] / flat;
    lo = std::min(lo, ratio); hi = std::max(hi, ratio); ++n;
  }
  INFO("ratio eps / flat over " << n << " points: [" << lo << ", " << hi << "]");
  REQUIRE(n > 100);
  CHECK(hi / lo < 1.03);
  // far-field slope d ln eps / d ln r -> -3 for r >> h
  auto at = [&](double r) { size_t k = 0; for (size_t i = 0; i < t.r.size(); ++i) if (std::fabs(std::log(t.r[i] / r)) < std::fabs(std::log(t.r[k] / r))) k = i; return k; };
  const size_t i1 = at(40.0 * h), i2 = at(160.0 * h);
  const double slope = std::log(t.eps_n[i2] / t.eps_n[i1]) / std::log(t.r[i2] / t.r[i1]);
  INFO("slope between r = " << t.r[i1] << " and " << t.r[i2] << ": " << slope);
  CHECK(slope == doctest::Approx(-3.0).epsilon(0.02));
}

TEST_CASE("lamppost: Kerr emissivity is positive, the map delta -> r is monotone, and rays near the axis are lost to the hole") {
  for (double a : {0.5, 0.9, 0.998}) {
    for (double h : {3.0, 6.0, 15.0}) {
      const double rin = kerr::isco(a);
      const lp::Table t = lp::emissivity(a, h, rin, 1e6, 1500);
      INFO("a = " << a << " h = " << h << ": " << t.r.size() << " table points");
      REQUIRE(t.r.size() > 50);
      for (size_t i = 0; i < t.r.size(); ++i) {
        CHECK(t.eps_n[i] > 0.0);
        CHECK(std::isfinite(t.eps_e[i]));
        CHECK(t.g[i] > 0.0);
        if (i > 0) CHECK(t.r[i] > t.r[i - 1]);  // the table is ordered in r
      }
      // photons with eta < 0 (sin delta < a sqrt(Delta) / (h^2 + a^2)) never reach the equator
      const double dmin = std::asin(a * std::sqrt(h * h - 2.0 * h + a * a) / (h * h + a * a));
      CHECK(!lp::hit(a, h, M_PI - 0.5 * dmin, rin, 1e6).hit);
      CHECK(lp::hit(a, h, M_PI - dmin - 0.3, rin, 1e6).hit == lp::hit(a, h, M_PI - dmin - 0.3, rin, 1e6).hit);  // deterministic
    }
  }
  // a lower lamppost illuminates the inner disk more: the emissivity at r = 2 r_in relative to r = 20 r_in grows when h decreases (Wilkins & Fabian 2012, Fig. 1)
  auto contrast = [&](double h) {
    const lp::Table t = lp::emissivity(0.9, h, kerr::isco(0.9), 1e6, 4000);
    auto val = [&](double r) { size_t k = 0; for (size_t i = 0; i < t.r.size(); ++i) if (std::fabs(t.r[i] - r) < std::fabs(t.r[k] - r)) k = i; return t.eps_n[k]; };
    return val(2.0 * kerr::isco(0.9)) / val(20.0 * kerr::isco(0.9));
  };
  CHECK(contrast(3.0) > contrast(10.0));
}

// The GPU reads eps_E(r) from a table on a uniform ln r grid (like the Novikov-Thorne LUT): log10 of the energy emissivity, normalised so that its maximum is 0.
TEST_CASE("lamppost: log table reproduces a power law exactly, is normalised to a maximum of 0, and brackets the range") {
  lp::Table t;
  for (double r : {2.0, 2.5, 3.1, 4.0, 7.0, 13.0, 40.0, 200.0}) { t.r.push_back(r); t.eps_n.push_back(0.0); t.eps_e.push_back(5.0 * std::pow(r, -3.0)); t.g.push_back(1.0); }
  const lp::LogTable lt = lp::log_table(t, 512);
  REQUIRE(lt.log10e.size() == 512);
  CHECK(lt.lmin == doctest::Approx(std::log(2.0)));
  CHECK(lt.lmax == doctest::Approx(std::log(200.0)));
  CHECK(lt.log10e.front() == doctest::Approx(0.0).epsilon(1e-12));  // the maximum (r = 2) is the first point
  for (size_t i = 0; i < lt.log10e.size(); ++i) {
    const double r = std::exp(lt.lmin + (lt.lmax - lt.lmin) * double(i) / double(lt.log10e.size() - 1));
    CHECK(lt.log10e[i] == doctest::Approx(std::log10(std::pow(r / 2.0, -3.0))).epsilon(1e-5).scale(1.0));  // a power law is linear in (ln r, ln eps)
    CHECK(lt.log10e[i] <= 1e-6f);
  }
  CHECK(lt.slope_out == doctest::Approx(-3.0).epsilon(1e-9));
}

TEST_CASE("lamppost: log table of a real Kerr emissivity is finite, <= 0, and peaks inside the disk") {
  const lp::Table t = lp::emissivity(0.9, 6.0, kerr::isco(0.9), 1e4, 1000);
  const lp::LogTable lt = lp::log_table(t, 2048);
  REQUIRE(lt.log10e.size() == 2048);
  float mx = -1e30f; size_t arg = 0;
  for (size_t i = 0; i < lt.log10e.size(); ++i) { CHECK(std::isfinite(lt.log10e[i])); CHECK(lt.log10e[i] <= 1e-6f); if (lt.log10e[i] > mx) { mx = lt.log10e[i]; arg = i; } }
  CHECK(mx == doctest::Approx(0.0).scale(1.0).epsilon(1e-6));
  const double r_peak = std::exp(lt.lmin + (lt.lmax - lt.lmin) * double(arg) / 2047.0);
  CHECK(r_peak < 6.0 * kerr::isco(0.9));  // the lamppost lights the inner disk most
  CHECK(lt.slope_out < -2.0);
  CHECK(lt.slope_out > -4.5);
}

// Photon-number conservation: eps_N(r) r dr = sin(delta) d(delta) = dx, so the integral of eps_N r dr over the disk is the measure of x = -cos(delta) that reaches it. Before the
// radial histogram (overlapping branches interleaved in r) this was 0.59 at a = 0, h = 3 and 0.77 at a = 0.5, h = 3.
TEST_CASE("lamppost: the table conserves photon number, also where several branches land at the same radius") {
  struct C { double a, h; };
  for (C c : {C{0.0, 3.0}, C{0.0, 6.0}, C{0.5, 3.0}, C{0.5, 6.0}, C{0.9, 3.0}, C{0.9, 15.0}}) {
    const double rin = kerr::isco(c.a);
    const int n = 6000;
    double meas = 0.0;
    for (int i = 0; i < n; ++i) if (lp::hit(c.a, c.h, std::acos(-(-1.0 + 2.0 * (i + 0.5) / n)), rin, 1e6).hit) meas += 2.0 / n;
    const lp::Table t = lp::emissivity(c.a, c.h, rin, 1e6, 4000);
    double I = 0.0;
    for (size_t i = 0; i < t.r.size(); ++i) {  // cells are geometric: integrate eps r dr with the cell edges (sqrt of neighbours' product)
      const double rl = i ? std::sqrt(t.r[i] * t.r[i - 1]) : t.r[i] * std::sqrt(t.r[1] / t.r[0]) > 0 ? t.r[i] / std::sqrt(t.r[1] / t.r[0]) : t.r[i];
      const double rh = i + 1 < t.r.size() ? std::sqrt(t.r[i] * t.r[i + 1]) : t.r[i] * std::sqrt(t.r[i] / t.r[i - 1]);
      I += t.eps_n[i] * 0.5 * (rh * rh - rl * rl);
    }
    INFO("a = " << c.a << " h = " << c.h << " integral " << I << " measure " << meas);
    CHECK(I == doctest::Approx(meas).epsilon(0.02));
  }
}

// ---- S2: gas inside the ISCO. Plunging geodesics with the E and L of the ISCO (Mummery & Balbus 2022); the photon-number emissivity does not depend on the gas motion
// (J^mu n_mu is a scalar on the equatorial hypersurface), only the energy absorbed does: E_gas = -p.u = u^t - L u^phi - p_r u^r.
namespace {
// metric of the equatorial plane (Sigma = r^2): g_tt, g_tphi, g_phiphi, g_rr
struct Eq { double gtt, gtp, gpp, grr, itt, itp, ipp; };
Eq eq_metric(double a, double r) {
  const double D = r * r - 2.0 * r + a * a, A = (r * r + a * a) * (r * r + a * a) - a * a * D;
  Eq m;
  m.gtt = -(1.0 - 2.0 / r); m.gtp = -2.0 * a / r; m.gpp = A / (r * r); m.grr = r * r / D;
  const double det = m.gtt * m.gpp - m.gtp * m.gtp;  // = -D
  m.itt = m.gpp / det; m.itp = -m.gtp / det; m.ipp = m.gtt / det;
  return m;
}
}  // namespace

TEST_CASE("lamppost: gas four-velocity is normalised (u.u = -1), circular outside the ISCO, plunging inside, continuous at the ISCO") {
  for (double a : {0.0, 0.5, 0.9, 0.998, -0.5}) {
    const double ri = kerr::isco(a, a >= 0.0);
    const lp::Gas gi = lp::gas(a, ri * (1.0 + 1e-12), ri);
    const double rh = 1.0 + std::sqrt(1.0 - a * a);
    for (double r : {rh + 0.2 * (ri - rh), rh + 0.5 * (ri - rh), rh + 0.9 * (ri - rh), ri, 1.5 * ri, 4.0 * ri}) {
      if (r <= rh + 1e-3) continue;
      const lp::Gas g = lp::gas(a, r, ri);
      const Eq m = eq_metric(a, r);
      const double norm = m.gtt * g.ut * g.ut + 2.0 * m.gtp * g.ut * g.uphi + m.gpp * g.uphi * g.uphi + m.grr * g.ur * g.ur;
      INFO("a = " << a << " r = " << r << " (isco " << ri << ")");
      CHECK(norm == doctest::Approx(-1.0).epsilon(1e-9));
      if (r >= ri) CHECK(g.ur == 0.0); else CHECK(g.ur < 0.0);
      // independent: u^t, u^phi from the covariant (E, L) through the inverse metric; u^r from the normalisation
      const double ut2 = m.itt * (-g.E) + m.itp * g.L, up2 = m.itp * (-g.E) + m.ipp * g.L;
      CHECK(g.ut == doctest::Approx(ut2).epsilon(1e-9));
      CHECK(g.uphi == doctest::Approx(up2).epsilon(1e-9));
      const double rest = -1.0 - (m.gtt * ut2 * ut2 + 2.0 * m.gtp * ut2 * up2 + m.gpp * up2 * up2);
      CHECK(g.ur * g.ur == doctest::Approx(rest / m.grr).epsilon(1e-6).scale(1e-9));
    }
    CHECK(gi.ur == doctest::Approx(0.0).scale(1.0).epsilon(1e-5));
  }
}

TEST_CASE("lamppost: photon radial momentum from Carter's R satisfies the null condition, and E_gas is positive") {
  const double a = 0.9, h = 6.0, ri = kerr::isco(a);
  for (double delta : {1.8, 2.3, 2.9}) {
    const double Q = lp::carter_Q(a, h, delta);
    for (double r : {1.8, 2.1, ri, 3.0, 10.0}) {
      const double D = r * r - 2.0 * r + a * a, R = (r * r + a * a) * (r * r + a * a) - D * (Q + a * a);
      if (R < 0.0) continue;
      const double pr = std::sqrt(R) / D;
      const Eq m = eq_metric(a, r);
      // g^{mu nu} p_mu p_nu = g^tt + g^rr p_r^2 + p_theta^2 / r^2 with L = 0, E = 1: (the equatorial p_theta^2 = Q)
      const double nul = m.itt + pr * pr * (D / (r * r)) + Q / (r * r);
      INFO("delta = " << delta << " r = " << r);
      CHECK(nul == doctest::Approx(0.0).scale(1.0).epsilon(1e-9));
      for (int sgn : {-1, 1}) CHECK(lp::gas_energy(a, r, ri, Q, sgn) > 0.0);
    }
  }
}

TEST_CASE("lamppost: table extended into the plunging region conserves photon number and adds emission inside the ISCO") {
  const double a = 0.9, h = 3.0, ri = kerr::isco(a), rh = 1.0 + std::sqrt(1.0 - a * a);
  const double r_in = rh + 1e-3;
  const int n = 6000;
  double meas = 0.0;
  for (int i = 0; i < n; ++i) if (lp::hit(a, h, std::acos(-(-1.0 + 2.0 * (i + 0.5) / n)), r_in, 1e6).hit) meas += 2.0 / n;
  const lp::Table t = lp::emissivity(a, h, r_in, 1e6, 4000);
  REQUIRE(t.r.size() > 100);
  double I = 0.0, inside = 0.0;
  for (size_t i = 0; i < t.r.size(); ++i) {
    const double rl = i ? std::sqrt(t.r[i] * t.r[i - 1]) : t.r[i] / std::sqrt(t.r[1] / t.r[0]);
    const double rhh = i + 1 < t.r.size() ? std::sqrt(t.r[i] * t.r[i + 1]) : t.r[i] * std::sqrt(t.r[i] / t.r[i - 1]);
    const double m = t.eps_n[i] * 0.5 * (rhh * rhh - rl * rl);
    I += m; if (t.r[i] < ri) inside += m;
  }
  INFO("integral " << I << " measure " << meas << " inside the ISCO " << inside);
  CHECK(I == doctest::Approx(meas).epsilon(0.02));
  CHECK(t.r.front() < ri);  // the table reaches below the ISCO
  CHECK(inside > 0.05 * I);   // a low lamppost puts a large share of the photons there
  for (size_t i = 0; i < t.r.size(); ++i) CHECK(t.g[i] > 0.0);
}

TEST_CASE("lamppost: the radial direction at the hit is inward for a steep downward photon, and always +-1") {
  const double a = 0.9, h = 6.0, rin = 1.0 + std::sqrt(1.0 - a * a) + 1e-3;  // down to the horizon: the steepest photons land inside the ISCO
  int steep = 0;
  for (double x : {-0.6, -0.3, -0.05, 0.05, 0.3, 0.6, 0.8, 0.9, 0.99}) {
    const lp::Hit hh = lp::hit(a, h, std::acos(-x), rin, 1e6);
    if (!hh.hit) continue;
    CHECK((hh.pr_sign == 1 || hh.pr_sign == -1));
    if (x >= 0.8) { CHECK(hh.pr_sign == -1); ++steep; }  // starts inward steeply and has no periastron before the plane (x = 0.6 turns at r ~ 4.35, lands on the way out)
  }
  CHECK(steep >= 2);  // x = 0.99 is inside the cone that never reaches the plane (Q < 0)
}

// ---- S3: power-law band. Source photon spectrum N(E) ~ E^-Gamma; at the disk it is blueshifted by g_sd: N_d = eps_N g_sd^(Gamma-1) E^-Gamma. eps_E of the table is that weight
// (Gamma = 2 gives eps_N g_sd = the bolometric energy emissivity of the earlier view). Observed: I_E = g^(2+Gamma) E^(1-Gamma) eps(r) (I_E / E^3 invariant), a power law shifts
// self-similarly, so the band limits only rescale the image.
TEST_CASE("lamppost: band weight g_sd^(Gamma-1): Gamma = 2 is the energy emissivity, Gamma = 1 is the photon number, Jensen for Gamma = 3") {
  const double a = 0.9, h = 6.0, rin = kerr::isco(a);
  const lp::Table t2 = lp::emissivity(a, h, rin, 1e4, 1500);              // default Gamma = 2
  const lp::Table t1 = lp::emissivity(a, h, rin, 1e4, 1500, 1.0);
  const lp::Table t3 = lp::emissivity(a, h, rin, 1e4, 1500, 3.0);
  REQUIRE(t2.r.size() == t1.r.size());
  REQUIRE(t2.r.size() == t3.r.size());
  for (size_t i = 0; i < t2.r.size(); ++i) {
    CHECK(t2.eps_e[i] == doctest::Approx(t2.eps_n[i] * t2.g[i]).epsilon(1e-12));
    CHECK(t1.eps_e[i] == doctest::Approx(t1.eps_n[i]).epsilon(1e-12));
    CHECK(t3.eps_e[i] >= t3.eps_n[i] * t3.g[i] * t3.g[i] * (1.0 - 1e-12));  // <g^2> >= <g>^2
    CHECK(t1.eps_n[i] == doctest::Approx(t2.eps_n[i]).epsilon(1e-12));      // the photon number does not depend on the spectrum
  }
}

TEST_CASE("lamppost: geometric reflection fraction N_disk / N_infinity: 1 for a distant source, larger for a low one (light bending), and it does not depend on the spectrum") {
  auto R = [&](double a, double h, double rout) { return lp::emissivity(a, h, kerr::isco(a), 1e4, 1500, 2.0, rout).refl_frac; };
  const double far = R(0.0, 500.0, 1e5);
  INFO("R(h = 500, a = 0) = " << far);
  CHECK(far == doctest::Approx(1.0).epsilon(0.03));
  const double r3 = R(0.9, 3.0, 100.0), r10 = R(0.9, 10.0, 100.0), r30 = R(0.9, 30.0, 100.0);
  INFO("R(h = 3, 10, 30; a = 0.9, disk to 100) = " << r3 << ", " << r10 << ", " << r30);
  CHECK(r3 > r10);
  CHECK(r10 > r30);
  CHECK(r3 > 1.0);   // Dauser et al. 2014 Fig. 1: low sources illuminate the disk more than they send to infinity
  CHECK(R(0.9, 6.0, 100.0) == doctest::Approx(lp::emissivity(0.9, 6.0, kerr::isco(0.9), 1e4, 1500, 3.0, 100.0).refl_frac).epsilon(1e-12));
}
