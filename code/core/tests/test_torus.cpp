#include <doctest/doctest.h>

#include <bhs/torus.hpp>
#include <cmath>
#include <initializer_list>

using namespace bhs;

namespace {
// central derivative
template <class F>
double ddx(F f, double x, double h) { return (f(x + h) - f(x - h)) / (2 * h); }
}  // namespace

TEST_CASE("torus: the constant specific angular momentum is the Keplerian one at r_max (two independent forms)") {
  for (double a : {0.0, 0.5, 0.9})
    for (double r : {8.0, 12.0, 20.0}) {
      const double Om = 1.0 / (std::pow(r, 1.5) + a);  // prograde Keplerian angular velocity (Bardeen)
      const auto m = torus::metric(a, r, M_PI / 2);
      const double l_from_omega = -(m.gtp + m.gpp * Om) / (m.gtt + m.gtp * Om);  // from u_phi / u_t = -l with u^phi = Om u^t
      CHECK(torus::ell_kepler(r, a) == doctest::Approx(l_from_omega).epsilon(1e-12));
    }
}

TEST_CASE("torus: construction validates the parameters") {
  torus::Torus t;
  CHECK(torus::make(0.9, 7.0, 12.0, 4.0 / 3.0, t));
  CHECK_FALSE(torus::make(0.9, 6.0, 12.0, 4.0 / 3.0, t));    // open: |u_t|_in = 1.0041 > 1, no outer surface
  CHECK_FALSE(torus::make(0.9, 14.0, 12.0, 4.0 / 3.0, t));   // r_in > r_max
  CHECK_FALSE(torus::make(0.9, 1.5, 12.0, 4.0 / 3.0, t));    // inside the horizon
  CHECK_FALSE(torus::make(0.9, 6.0, 12.0, 0.9, t));          // Gamma <= 1
}

TEST_CASE("torus: h = 1 at the inner edge, h > 1 inside, pressure maximum (dh/dr = 0) at r_max on the equator") {
  torus::Torus t; REQUIRE(torus::make(0.9, 7.0, 12.0, 4.0 / 3.0, t));
  CHECK(t.enthalpy(7.0, M_PI / 2) == doctest::Approx(1.0).epsilon(1e-12));
  CHECK(t.enthalpy(9.0, M_PI / 2) > 1.0);
  CHECK(t.enthalpy(12.0, M_PI / 2) == doctest::Approx(t.h_max).epsilon(1e-12));
  CHECK(t.h_max > 1.0);
  const double dh = ddx([&](double r) { return t.enthalpy(r, M_PI / 2); }, 12.0, 1e-5);
  CHECK(std::abs(dh) < 1e-6 * t.h_max);
  CHECK(t.enthalpy(11.0, M_PI / 2) < t.h_max);
  CHECK(t.enthalpy(13.0, M_PI / 2) < t.h_max);
}

TEST_CASE("torus: the density is 1 at the pressure maximum, 0 outside the surface (inner edge, far, poles)") {
  torus::Torus t; REQUIRE(torus::make(0.9, 7.0, 12.0, 4.0 / 3.0, t));
  CHECK(t.density(12.0, M_PI / 2) == doctest::Approx(1.0).epsilon(1e-12));
  CHECK(t.density(5.0, M_PI / 2) == 0.0);
  CHECK(t.density(200.0, M_PI / 2) == 0.0);
  CHECK(t.density(12.0, 0.2) == 0.0);
  CHECK(t.density(9.0, M_PI / 2) > 0.0);
  CHECK(t.density(9.0, M_PI / 2) < 1.0);
}

TEST_CASE("torus: u.u = -1 and u_phi / u_t = -l for the flow built from u_t") {
  torus::Torus t; REQUIRE(torus::make(0.9, 7.0, 12.0, 4.0 / 3.0, t));
  for (double r : {7.0, 10.0, 12.0, 15.0})
    for (double th : {M_PI / 2, 1.3, 1.8}) {
      if (t.density(r, th) <= 0) continue;
      const auto u = t.four_velocity(r, th);
      const auto m = torus::metric(0.9, r, th);
      const double uu = m.gtt * u.ut * u.ut + 2 * m.gtp * u.ut * u.uphi + m.gpp * u.uphi * u.uphi;
      CHECK(uu == doctest::Approx(-1.0).epsilon(1e-12));
      const double u_t_low = m.gtt * u.ut + m.gtp * u.uphi, u_p_low = m.gtp * u.ut + m.gpp * u.uphi;
      CHECK(u_p_low / u_t_low == doctest::Approx(-t.ell).epsilon(1e-12));
      CHECK(u.ut > 0);
    }
}

TEST_CASE("torus: relativistic hydrostatic equilibrium d_a ln h = (1/2) u^mu u^nu d_a g_mu nu, from finite differences of the metric (independent)") {
  torus::Torus t; REQUIRE(torus::make(0.9, 7.0, 12.0, 4.0 / 3.0, t));
  int n = 0;
  for (double r : {7.0, 9.0, 12.0, 14.0})
    for (double th : {M_PI / 2, 1.45, 1.7}) {
      if (t.density(r, th) <= 0) continue;
      const auto u = t.four_velocity(r, th);
      const double hs = 1e-5;
      auto G = [&](double rr, double tt) { return torus::metric(0.9, rr, tt); };
      for (int axis = 0; axis < 2; ++axis) {
        auto lnh = [&](double x) { return std::log(axis == 0 ? t.enthalpy(x, th) : t.enthalpy(r, x)); };
        const double lhs = ddx(lnh, axis == 0 ? r : th, hs);
        auto g = [&](double x, int comp) { const auto m = axis == 0 ? G(x, th) : G(r, x); return comp == 0 ? m.gtt : comp == 1 ? m.gtp : m.gpp; };
        const double x0 = axis == 0 ? r : th;
        const double dgtt = ddx([&](double x) { return g(x, 0); }, x0, hs), dgtp = ddx([&](double x) { return g(x, 1); }, x0, hs), dgpp = ddx([&](double x) { return g(x, 2); }, x0, hs);
        const double rhs = 0.5 * (dgtt * u.ut * u.ut + 2 * dgtp * u.ut * u.uphi + dgpp * u.uphi * u.uphi);
        CHECK(lhs == doctest::Approx(rhs).epsilon(1e-5).scale(1e-3));
        ++n;
      }
    }
  CHECK(n >= 12);
}

TEST_CASE("torus: ion temperature P/rho = (Gamma - 1)(h - 1)/Gamma, and the polytropic density exponent 1/(Gamma - 1)") {
  torus::Torus t; REQUIRE(torus::make(0.9, 7.0, 12.0, 4.0 / 3.0, t));
  const double h = t.enthalpy(9.0, M_PI / 2);
  CHECK(t.theta_ion(9.0, M_PI / 2) == doctest::Approx((1.0 / 3.0) * (h - 1.0) / (4.0 / 3.0)).epsilon(1e-12));
  CHECK(t.density(9.0, M_PI / 2) == doctest::Approx(std::pow((h - 1.0) / (t.h_max - 1.0), 3.0)).epsilon(1e-12));  // 1/(Gamma-1) = 3
}
