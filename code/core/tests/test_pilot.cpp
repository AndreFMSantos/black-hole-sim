#include <doctest/doctest.h>

#include <bhs/lorentz.hpp>
#include <bhs/pilot.hpp>
#include <cmath>

using namespace bhs;

namespace {
double energy_of(double a, const pilot::State& s) {  // E = gamma (alpha + omega varpi beta_phi)
  const double g = 1.0 / std::sqrt(1.0 - dot(s.beta, s.beta));
  const auto f = pilot::zamo(a, s.r, s.theta);
  return g * (f.alpha + f.omega * f.varpi * s.beta.z);
}
}  // namespace

TEST_CASE("pilot: state -> constants -> state is the identity") {
  const double a = 0.9;
  for (Vec3 b : {Vec3{0, 0, 0.3}, Vec3{-0.2, 0.1, 0.25}, Vec3{0.05, -0.3, -0.2}}) {
    pilot::Pilot p(a, 12.0, 1.3, 0.4, b);
    const auto s = p.state();
    CHECK(s.beta.x == doctest::Approx(b.x).epsilon(1e-12));
    CHECK(s.beta.y == doctest::Approx(b.y).epsilon(1e-12));
    CHECK(s.beta.z == doctest::Approx(b.z).epsilon(1e-12));
  }
}

TEST_CASE("pilot: a circular Kerr orbit stays circular and keeps E") {
  const double a = 0.9, r0 = 8.0;
  pilot::Pilot p(a, r0, M_PI / 2, 0.0, {0, 0, lorentz::circular_beta(a, r0, M_PI / 2)});
  const double e0 = energy_of(a, p.state());
  for (int i = 0; i < 50; ++i) p.advance(2.0);  // 100 of proper time (several orbits)
  const auto s = p.state();
  CHECK(s.r == doctest::Approx(r0).epsilon(1e-8));
  CHECK(energy_of(a, s) == doctest::Approx(e0).epsilon(1e-9));
  CHECK(s.tau == doctest::Approx(100.0).epsilon(1e-9));
}

TEST_CASE("pilot: azimuthal rate of a Schwarzschild circular orbit, dphi/dtau = r^-3/2 / sqrt(1 - 3/r)") {
  const double r0 = 10.0;
  pilot::Pilot p(0.0, r0, M_PI / 2, 0.0, {0, 0, lorentz::circular_beta(0.0, r0, M_PI / 2)});
  p.advance(10.0);
  const double rate = std::pow(r0, -1.5) / std::sqrt(1.0 - 3.0 / r0);
  CHECK(p.state().phi == doctest::Approx(10.0 * rate).epsilon(1e-8));
}

TEST_CASE("pilot: radial free fall from rest at infinity, r(tau)^{3/2} = r0^{3/2} - 3/2 sqrt(2M) tau") {
  const double r0 = 40.0;
  pilot::Pilot p(0.0, r0, M_PI / 2, 0.0, {-lorentz::infall_beta(r0), 0, 0});
  CHECK(energy_of(0.0, p.state()) == doctest::Approx(1.0).epsilon(1e-12));  // starts from rest at infinity: E = 1
  p.advance(20.0);
  const double expect = std::pow(std::pow(r0, 1.5) - 1.5 * std::sqrt(2.0) * 20.0, 2.0 / 3.0);
  CHECK(p.state().r == doctest::Approx(expect).epsilon(1e-8));
  CHECK(p.state().beta.x < -lorentz::infall_beta(r0));  // accelerates while descending
}

TEST_CASE("pilot: thrust adds rapidity (flat-space limit): beta = tanh(a tau)") {
  const double accel = 0.01, dt = 0.1;
  pilot::Pilot p(0.0, 1e6, M_PI / 2, 0.0, {0, 0, 0});
  for (int i = 0; i < 1000; ++i) {  // tau = 100, a tau = 1
    p.thrust({0, 0, accel}, dt);
    p.advance(dt);
  }
  CHECK(p.state().beta.z == doctest::Approx(std::tanh(1.0)).epsilon(2e-5));
}

TEST_CASE("pilot: an orbit below the ISCO plunges into the horizon") {
  const double r0 = 5.4;  // < 6
  pilot::Pilot p(0.0, r0, M_PI / 2, 0.0, {0, 0, 0.98 * lorentz::circular_beta(0.0, r0, M_PI / 2)});
  for (int i = 0; i < 200 && !p.captured(); ++i) p.advance(5.0);
  CHECK(p.captured());
}

TEST_CASE("pilot: E, L, Q stay constant under free fall in a generic inclined Kerr orbit") {
  const double a = 0.9;
  pilot::Pilot p(a, 15.0, 1.2, 0.0, {0.05, 0.12, 0.28});
  const auto c0 = p.consts();
  const double e0 = energy_of(a, p.state());
  for (int i = 0; i < 100; ++i) p.advance(3.0);
  const auto c1 = p.consts();
  CHECK(c1.E == doctest::Approx(c0.E).epsilon(1e-14));  // the constants only change with thrust
  CHECK(c1.L == doctest::Approx(c0.L).epsilon(1e-14));
  CHECK(c1.Q == doctest::Approx(c0.Q).epsilon(1e-14));
  CHECK(energy_of(a, p.state()) == doctest::Approx(e0).epsilon(1e-7));  // reconstructed from the integrated state
  CHECK_FALSE(p.captured());
}
