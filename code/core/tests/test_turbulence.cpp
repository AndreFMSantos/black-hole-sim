#include <doctest/doctest.h>

#include <bhs/turbulence.hpp>
#include <cmath>
#include <initializer_list>

using namespace bhs;

TEST_CASE("turb: fbm is deterministic, bounded in [-1, 1] and roughly zero-mean with the tabulated variance") {
  double s = 0, s2 = 0;
  const int n = 200000;
  uint32_t h = 777u;
  auto rnd = [&]() { h = h * 1664525u + 1013904223u; return double(h >> 8) / 16777216.0; };
  for (int i = 0; i < n; ++i) {
    const double x = 40.0 * rnd(), y = 40.0 * rnd(), z = 40.0 * rnd();
    const double v = turb::fbm(x, y, z);
    CHECK(v >= -1.0);
    CHECK(v <= 1.0);
    CHECK(v == turb::fbm(x, y, z));
    s += v; s2 += v * v;
  }
  CHECK(std::abs(s / n) < 0.01);
  CHECK(s2 / n == doctest::Approx(turb::kVar).epsilon(0.03));
}

TEST_CASE("turb: amp = 0 gives exactly factor 1 (v0.9b unchanged)") {
  for (double t : {0.0, 123.4}) CHECK(turb::density_factor(9.0, 1.1, 0.02, t, 0.9, 0.05, 0.0, 2.0) == 1.0);
}

TEST_CASE("turb: the pattern rotates rigidly on each ring with Omega(r) (no drift in the corotating frame)") {
  const double a = 0.9, h = 0.06;
  for (double r : {6.5, 9.0, 14.0})
    for (double t : {0.0, 37.0, 250.0}) {
      const double phi = 0.7, mu = 0.03;
      const double f0 = turb::density_factor(r, phi, mu, 0.0, a, h, 0.8, 2.0);
      const double ft = turb::density_factor(r, phi + turb::omega(r, a) * t, mu, t, a, h, 0.8, 2.0);
      CHECK(ft == doctest::Approx(f0).epsilon(1e-9));
    }
}

TEST_CASE("turb: differential rotation: two rings shear by (Omega1 - Omega2) t") {
  const double a = 0.9, h = 0.06, t = 90.0;
  const double r1 = 7.0, r2 = 12.0;
  // the same feature that is at phi = 0.4 at t = 0 in both rings: at t, ring 1 is at 0.4 + Omega1 t, ring 2 at 0.4 + Omega2 t
  const double dphi = (turb::omega(r1, a) - turb::omega(r2, a)) * t;
  CHECK(dphi > 1.0);  // it really rotates by more than 1 rad of difference
  const double f1 = turb::density_factor(r1, 0.4 + turb::omega(r1, a) * t, 0.0, t, a, h, 0.8, 2.0);
  const double f1_0 = turb::density_factor(r1, 0.4, 0.0, 0.0, a, h, 0.8, 2.0);
  const double f2_wrong = turb::density_factor(r2, 0.4 + turb::omega(r1, a) * t, 0.0, t, a, h, 0.8, 2.0);
  const double f2_0 = turb::density_factor(r2, 0.4, 0.0, 0.0, a, h, 0.8, 2.0);
  CHECK(f1 == doctest::Approx(f1_0).epsilon(1e-9));
  CHECK(std::abs(f2_wrong - f2_0) > 1e-3);  // following ring 1 at ring 2 is wrong: the rotation is NOT rigid
}

TEST_CASE("turb: the density factor has mean ~ 1 over a ring (lognormal normalisation)") {
  const double a = 0.9, h = 0.06;
  double s = 0;
  const int n = 60000;
  uint32_t st = 99u;
  auto rnd = [&]() { st = st * 1664525u + 1013904223u; return double(st >> 8) / 16777216.0; };
  for (int i = 0; i < n; ++i) s += turb::density_factor(6.5 + 8.0 * rnd(), 6.2831853 * rnd(), (rnd() - 0.5) * 0.2, 0.0, a, h, 0.6, 2.0);
  CHECK(s / n == doctest::Approx(1.0).epsilon(0.03));
}

TEST_CASE("spot: it returns to the same place after one orbital period 2 pi / Omega(r_h)") {
  turb::Spot sp; sp.r = 9.0; sp.phi0 = 0.8; sp.size = 1.2; sp.amp = 3.0;
  const double a = 0.9, T = 2.0 * M_PI / turb::omega(sp.r, a);
  for (double t : {0.0, 31.0}) {
    CHECK(std::cos(turb::spot_phase(sp, a, t + T)) == doctest::Approx(std::cos(turb::spot_phase(sp, a, t))).epsilon(1e-12));
    CHECK(std::sin(turb::spot_phase(sp, a, t + T)) == doctest::Approx(std::sin(turb::spot_phase(sp, a, t))).epsilon(1e-12));
    const double g0 = turb::spot_T_factor(sp, a, 8.4, 1.1, 0.01, t), g1 = turb::spot_T_factor(sp, a, 8.4, 1.1, 0.01, t + T);
    CHECK(g0 == doctest::Approx(g1).epsilon(1e-12));
  }
  CHECK(T == doctest::Approx(2.0 * M_PI * (std::pow(9.0, 1.5) + 0.9)).epsilon(1e-12));
}

TEST_CASE("spot: the temperature factor raises the bolometric radiance by exactly 1 + amp G; G = 1 at the centre, decays with distance") {
  turb::Spot sp; sp.r = 10.0; sp.phi0 = 0.0; sp.size = 1.5; sp.amp = 2.0;
  const double a = 0.0, t = 0.0;
  CHECK(turb::spot_gauss(sp, a, 10.0, 0.0, 0.0, t) == doctest::Approx(1.0).epsilon(1e-12));
  CHECK(std::pow(turb::spot_T_factor(sp, a, 10.0, 0.0, 0.0, t), 4.0) == doctest::Approx(1.0 + 2.0).epsilon(1e-12));
  CHECK(turb::spot_gauss(sp, a, 10.0 + 1.5, 0.0, 0.0, t) == doctest::Approx(std::exp(-0.5)).epsilon(1e-12));
  CHECK(turb::spot_gauss(sp, a, 10.0, 0.0, 0.15, t) == doctest::Approx(std::exp(-0.5 * 1.5 * 1.5 / (1.5 * 1.5))).epsilon(1e-9));  // z = r mu = 1.5 = 1 sigma
  CHECK(turb::spot_T_factor(sp, a, 30.0, 2.0, 0.0, t) == doctest::Approx(1.0).epsilon(1e-6));
  sp.amp = 0.0;
  CHECK(turb::spot_T_factor(sp, a, 10.0, 0.0, 0.0, t) == 1.0);
}

TEST_CASE("spot: the flare is a Gaussian pulse in time on the amplitude") {
  turb::Spot sp; sp.amp = 1.0; sp.fl_t = 100.0; sp.fl_w = 15.0; sp.fl_boost = 4.0;
  CHECK(turb::spot_amp_t(sp, 100.0) == doctest::Approx(1.0 * (1.0 + 4.0)).epsilon(1e-12));
  CHECK(turb::spot_amp_t(sp, 100.0 + 15.0) == doctest::Approx(1.0 * (1.0 + 4.0 * std::exp(-0.5))).epsilon(1e-12));
  CHECK(turb::spot_amp_t(sp, 1000.0) == doctest::Approx(1.0).epsilon(1e-9));
  sp.fl_boost = 0.0;
  CHECK(turb::spot_amp_t(sp, 100.0) == 1.0);
}
