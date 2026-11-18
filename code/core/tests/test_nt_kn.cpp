#include <doctest/doctest.h>

#include <bhs/kerr.hpp>
#include <bhs/novikov_thorne.hpp>
#include <bhs/novikov_thorne_kn.hpp>
#include <cmath>

using namespace bhs;

// Pugliese, Quevedo & Ruffini 2013 (arXiv:1303.6250), eq. (24)-(28), transcribed from the page images by an agent (M = 1, mu = 1); eq. (28) comes
// with a factor `a` missing in the 1st term of the numerator (a typo in the paper, verified numerically): the corrected form is used.
static void pugliese(double r, double a, double Q, double& Lpro, double& Epro) {
  const double Q2 = Q * Q, M = 1.0;
  const double Psi = 4.0 * a * a * (Q2 - r * M) + std::pow(2.0 * Q2 + (r - 3.0 * M) * r, 2);
  const double Pi = -a * a * (Q2 - r * M) * std::pow(r, 4) * std::pow(a * a + Q2 + (r - 2.0 * M) * r, 2) * std::pow(a * a * (Q2 - M * r) + (2.0 * Q2 - 3.0 * M * r) * r * r, 2);
  const double Sig = r * r * (-(Q2 - M * r) * std::pow(r, 4) * (2.0 * Q2 + (r - 3.0 * M) * r) + std::pow(a, 4) * (Q2 - M * r) * (2.0 * Q2 - (5.0 * M + r) * r) +
                              a * a * (2.0 * std::pow(Q2, 3) + Q2 * Q2 * (r - 11.0 * M) * r - 2.0 * Q2 * (r - 2.0 * M) * r * r * (5.0 * M + r) + 2.0 * M * r * r * r * (r * (3.0 * M + r) - 6.0 * M * M)));
  Lpro = std::sqrt((Sig - 2.0 * M * M * std::sqrt(Pi)) / Psi) / (r * r);  // L- > 0: co-rotating (prograde)
  const double L = Lpro;
  const double num = -a * L * (Q2 - 2.0 * M * r) + std::sqrt(r * r * (a * a + Q2 + (r - 2.0 * M) * r) * (r * r * (L * L + r * r) + a * a * (r * (2.0 * M + r) - Q2)));
  Epro = num / (std::pow(r, 4) + a * a * (r * (2.0 * M + r) - Q2));
}

TEST_CASE("nt-kn: analytic derivatives of the equatorial metric = finite differences; det of the (t,phi) block = -Delta") {
  for (auto [a, q2] : {std::pair{0.0, 0.0}, {0.5, 0.25}, {0.8, 0.3}, {0.3, 0.9}}) {
    for (double r : {2.5, 5.0, 12.0}) {
      const auto m = ntk::eq_metric(r, a, q2);
      auto g = [&](double x) { return ntk::eq_metric(x, a, q2); };
      const double h = 1e-5 * r;
      CHECK(m.dgtt == doctest::Approx((g(r + h).gtt - g(r - h).gtt) / (2 * h)).epsilon(1e-7));
      CHECK(m.dgtp == doctest::Approx((g(r + h).gtp - g(r - h).gtp) / (2 * h)).epsilon(1e-7));
      CHECK(m.dgpp == doctest::Approx((g(r + h).gpp - g(r - h).gpp) / (2 * h)).epsilon(1e-7));
      CHECK(m.gtt * m.gpp - m.gtp * m.gtp == doctest::Approx(-kerr::delta(r, a, q2)).epsilon(1e-12));
    }
  }
}

TEST_CASE("nt-kn: q2 = 0 recovers Kerr (E, L, Omega, ISCO, flux) against the closed formulas of nt:: and kerr::isco") {
  for (double a : {0.0, 0.5, 0.9, 0.998}) {
    for (double r : {6.0, 8.0, 15.0, 40.0}) {
      const auto c = ntk::circ(r, a, 0.0);
      CHECK(c.E == doctest::Approx(nt::energy(r, a)).epsilon(1e-11));
      CHECK(c.L == doctest::Approx(nt::ang_mom(r, a)).epsilon(1e-11));
      CHECK(c.Omega == doctest::Approx(nt::omega(r, a)).epsilon(1e-11));
    }
    CHECK(ntk::isco(a, 0.0) == doctest::Approx(kerr::isco(a)).epsilon(1e-9));
    const double rin = kerr::isco(a);
    for (double r : {1.3, 2.0, 5.0, 20.0}) CHECK(ntk::flux(rin * r, a, 0.0, rin) == doctest::Approx(nt::flux(rin * r, a)).epsilon(2e-6));
  }
}

TEST_CASE("nt-kn: a = 0 recovers Reissner-Nordstrom: ISCO = root of the cubic r^3 - 6 r^2 + 9 q2 r - 4 q2^2 (kerr::rn_isco)") {
  for (double q2 : {0.0, 0.25, 0.64, 0.81, 0.98}) CHECK(ntk::isco(0.0, q2) == doctest::Approx(kerr::rn_isco(q2)).epsilon(1e-8));
}

TEST_CASE("nt-kn: E and L of the circular orbits against eq. (24)-(28) of Pugliese et al. 2013 (independent: another formula, from the paper)") {
  for (auto [a, Q] : {std::pair{0.5, 0.3}, {0.0, 0.5}, {0.8, 0.3}, {0.3, 0.7}, {0.9, 0.2}}) {
    for (double r : {5.0, 7.0, 12.0, 30.0}) {
      double L, E; pugliese(r, a, Q, L, E);
      const auto c = ntk::circ(r, a, Q * Q);
      REQUIRE(c.ok);
      CHECK(c.L == doctest::Approx(L).epsilon(1e-9));
      CHECK(c.E == doctest::Approx(E).epsilon(1e-9));
    }
  }
}

TEST_CASE("nt-kn: energy conservation Int 4 pi r E F dr = 1 - E_isco in Kerr-Newman (independent check of the flux and the ISCO)") {
  for (auto [a, q2] : {std::pair{0.0, 0.64}, {0.5, 0.25}, {0.8, 0.09}, {0.9, 0.18}, {0.3, 0.9}}) {
    const double rin = ntk::isco(a, q2), Eisco = ntk::circ(rin, a, q2).E;
    CHECK(ntk::luminosity(a, q2) == doctest::Approx(1.0 - Eisco).epsilon(1e-5));
  }
}

TEST_CASE("nt-kn: the ISCO decreases with charge (Pugliese p. 13), zero flux at the ISCO and a T profile with a peak outside it; reference values computed separately") {
  double prev = 1e9;
  for (double q2 : {0.0, 0.2, 0.4, 0.6, 0.8}) { const double r = ntk::isco(0.3, q2); CHECK(r < prev); prev = r; }
  // values (a, q2) -> r_isco computed by an agent with another script (1st-order finite differences, ~1e-3): they agree to 3e-3
  CHECK(ntk::isco(0.5, 0.25) == doctest::Approx(3.732).epsilon(3e-3));      // (a = 0.5, Q = 0.5)
  CHECK(ntk::isco(0.8, 0.09) == doctest::Approx(2.66527).epsilon(3e-3));    // (0.8, 0.3)
  CHECK(ntk::isco(0.8, 0.25) == doctest::Approx(2.10828).epsilon(3e-3));    // (0.8, 0.5)
  CHECK(ntk::isco(0.94, 0.09) == doctest::Approx(1.52996).epsilon(3e-3));   // (0.94, 0.3)
  const double rin = ntk::isco(0.5, 0.25);
  CHECK(ntk::flux(rin, 0.5, 0.25, rin) == 0.0);
  ntk::Profile p(0.5, 0.25, 200.0, 512);
  CHECK(p.r_peak() > rin);
  CHECK(p.table().front() == doctest::Approx(0.0).epsilon(1e-3));
}

TEST_CASE("nt-kn: redshift at q2 = 0 = kerr::redshift; in KN the ZAMO observer uses omega = a (2r - q2)/A") {
  for (double a : {0.0, 0.5, 0.9})
    for (double r : {6.0, 10.0})
      for (double L : {-3.0, 0.0, 2.5}) CHECK(ntk::redshift(a, 0.0, r, L, 40.0, 1.3) == doctest::Approx(kerr::redshift(a, r, L, 40.0, 1.3)).epsilon(1e-10));
  // static limit (L = 0, a = 0): g = E_obs/(u^t) with E_obs = 1/alpha = r_obs/sqrt(Delta(r_obs)) (static RN observer)
  const double q2 = 0.5, r = 8.0, ro = 40.0;
  const auto c = ntk::circ(r, 0.0, q2);
  CHECK(ntk::redshift(0.0, q2, r, 0.0, ro, 1.0) == doctest::Approx(ro / std::sqrt(kerr::delta(ro, 0.0, q2)) / c.ut).epsilon(1e-12));
}

TEST_CASE("nt-kn: near the extremal case (a^2 + q2 = 0.99) the ISCO stays outside the horizon and the efficiency rises; a^2 + q2 = 1 is extremal (horizon 0 by rounding: do not use)") {
  const double a = 0.9, q2 = 0.18, rh = kerr::horizon(a, q2), rin = ntk::isco(a, q2);
  CHECK(rin > rh);
  CHECK(rin < ntk::isco(a, 0.0));
  CHECK(1.0 - ntk::circ(rin, a, q2).E > 1.0 - ntk::circ(ntk::isco(a, 0.0), a, 0.0).E);
}
