
#include <doctest/doctest.h>

#include <algorithm>
#include <bhs/integrators.hpp>
#include <cmath>

using namespace bhs;

namespace {
constexpr double kTwoPi = 6.283185307179586;
auto acc = [](Vec2 p) { return gravity(p); };

struct Result { State final; double max_dE, dL; };

Result run(Method m, double dt, int periods) {
  State s = circular_orbit(1.0);
  const double e0 = energy(s), l0 = angular_momentum(s);
  const int steps = static_cast<int>(periods * kTwoPi / dt);
  double max_dE = 0;
  for (int i = 0; i < steps; ++i) {
    s = step(m, s, dt, acc);
    max_dE = std::max(max_dE, std::abs(energy(s) - e0));
  }
  return {s, max_dE, angular_momentum(s) - l0};
}
}  // namespace

TEST_CASE("gravity points to the origin with magnitude GM/r^2") {
  const Vec2 a = gravity({2, 0});
  CHECK(a.x == doctest::Approx(-0.25));
  CHECK(a.y == doctest::Approx(0));
}

TEST_CASE("circular orbit initial state has E = -GM/(2r) and L = sqrt(GM r)") {
  const State s = circular_orbit(4.0);
  CHECK(energy(s) == doctest::Approx(-1.0 / 8.0));
  CHECK(angular_momentum(s) == doctest::Approx(2.0));
}

TEST_CASE("explicit Euler drifts: orbit spirals outward, energy grows (known defect)") {
  const auto r = run(Method::Euler, 0.01, 10);
  CHECK(norm(r.final.x) > 1.5);   // observed 1.648 after 10 orbits
  CHECK(r.max_dE > 0.1);          // observed 0.203
  CHECK(r.dL > 0.1);              // Euler also pumps angular momentum
}

TEST_CASE("Velocity Verlet keeps the orbit closed and energy bounded") {
  const auto r = run(Method::Verlet, 0.01, 10);
  CHECK(norm(r.final.x) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(r.max_dE < 1e-7);         // observed 1.25e-9, oscillating not drifting
  CHECK(std::abs(r.dL) < 1e-12);  // symplectic + central force: L conserved
}

TEST_CASE("RK4 is far more accurate than Euler at the same step") {
  const auto r = run(Method::RK4, 0.01, 10);
  CHECK(norm(r.final.x) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(r.max_dE < 1e-8);         // observed 8.7e-11
}

TEST_CASE("Verlet is time-reversible") {
  State s = circular_orbit(1.0);
  s.v = 0.8 * s.v;  // eccentric orbit
  const State s0 = s;
  for (int i = 0; i < 500; ++i) s = step_verlet(s, 0.01, acc);
  s.v = -1.0 * s.v;
  for (int i = 0; i < 500; ++i) s = step_verlet(s, 0.01, acc);
  CHECK(s.x.x == doctest::Approx(s0.x.x).epsilon(1e-9));
  CHECK(s.x.y == doctest::Approx(s0.x.y).epsilon(1e-9));
}
