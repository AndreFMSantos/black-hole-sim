#include <doctest/doctest.h>

#include <bhs/kerr.hpp>
#include <bhs/novikov_thorne.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("nt: circular orbit E, L, Omega agree with the Schwarzschild closed forms at a = 0") {
  for (double r : {7.0, 10.0, 40.0}) {
    const double ut = 1.0 / std::sqrt(1.0 - 3.0 / r);
    CHECK(nt::energy(r, 0.0) == doctest::Approx((1.0 - 2.0 / r) * ut).epsilon(1e-12));
    CHECK(nt::omega(r, 0.0) == doctest::Approx(std::pow(r, -1.5)).epsilon(1e-12));
    CHECK(nt::ang_mom(r, 0.0) == doctest::Approx(r * r * nt::omega(r, 0.0) * ut).epsilon(1e-12));
  }
}

TEST_CASE("nt: flux vanishes at the ISCO and is positive outside") {
  for (double a : {0.0, 0.5, 0.9, 0.998}) {
    const double rin = kerr::isco(a);
    CHECK(nt::flux(rin, a) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(nt::flux(rin * 1.05, a) > 0.0);
    CHECK(nt::flux(rin * 2.0, a) > 0.0);
  }
}

TEST_CASE("nt: far from the hole the flux tends to the Newtonian 3 Mdot/(8 pi r^3) (1 - sqrt(r_in/r))") {
  for (double a : {0.0, 0.9}) {
    const double rin = kerr::isco(a), r = 1e5;
    const double newton = 3.0 / (8.0 * M_PI * r * r * r) * (1.0 - std::sqrt(rin / r));
    CHECK(nt::flux(r, a) == doctest::Approx(newton).epsilon(2e-3));
  }
}

TEST_CASE("nt: total luminosity equals the radiative efficiency 1 - E_isco (energy conservation)") {
  for (double a : {0.0, 0.5, 0.9, 0.998}) {
    const double eff = 1.0 - nt::energy(kerr::isco(a), a);
    CHECK(nt::luminosity(a) == doctest::Approx(eff).epsilon(1e-5));
  }
  CHECK(1.0 - nt::energy(6.0, 0.0) == doctest::Approx(1.0 - std::sqrt(8.0 / 9.0)).epsilon(1e-12));  // 5.72 %
}

TEST_CASE("nt: profile is normalised to a temperature maximum of 1 and zero at the ISCO") {
  const nt::Profile p(0.9);
  CHECK(p.temperature(kerr::isco(0.9)) == doctest::Approx(0.0).epsilon(1e-9));
  double tmax = 0;
  for (double r = kerr::isco(0.9); r < 60.0; r *= 1.01) tmax = std::max(tmax, p.temperature(r));
  CHECK(tmax == doctest::Approx(1.0).epsilon(1e-3));
  CHECK(p.temperature(5.0) < p.temperature(p.r_peak()));
  CHECK(p.temperature(40.0) < p.temperature(p.r_peak()));
}
