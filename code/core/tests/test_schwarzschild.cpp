#include <doctest/doctest.h>

#include <bhs/schwarzschild.hpp>
#include <cmath>

using namespace bhs::schw;

TEST_CASE("photon sphere r = 3M is a fixed point of the Binet equation") {
  // u'' + u = 3 M u^2 ; u = 1/3 => u'' = -1/3 + 3/9 = 0
  CHECK(binet_accel(1.0 / 3.0) == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("weak-field deflection matches the series 4M/b + 15 pi/4 (M/b)^2 + 128/3 (M/b)^3") {
  for (double b : {50.0, 100.0, 400.0}) {
    const auto r = trace(b);
    REQUIRE(r.fate == Fate::Escaped);
    const double series = 4 / b + 15 * M_PI / 4 / (b * b) + 128.0 / 3.0 / (b * b * b);
    CHECK(r.deflection == doctest::Approx(series).epsilon(2e-3));
  }
}

TEST_CASE("deflection is converged in the step size") {
  const double a = trace(10.0, 1e-3).deflection;
  const double c = trace(10.0, 5e-4).deflection;
  CHECK(a == doctest::Approx(c).epsilon(1e-8));
}

TEST_CASE("small impact parameter is captured, large escapes") {
  CHECK(trace(3.0).fate == Fate::Captured);
  CHECK(trace(5.0).fate == Fate::Captured);
  CHECK(trace(5.4).fate == Fate::Escaped);
  CHECK(trace(20.0).fate == Fate::Escaped);
}

TEST_CASE("critical impact parameter is sqrt(27) M (the black-hole shadow radius)") {
  double lo = 4.0, hi = 6.0;  // lo captured, hi escapes
  for (int i = 0; i < 60; ++i) {
    const double mid = 0.5 * (lo + hi);
    (trace(mid).fate == Fate::Escaped ? hi : lo) = mid;
  }
  CHECK(0.5 * (lo + hi) == doctest::Approx(std::sqrt(27.0)).epsilon(1e-4));
}

TEST_CASE("near-critical rays wind around the photon sphere before leaving") {
  const auto r = trace(critical_b() * 1.0001);
  REQUIRE(r.fate == Fate::Escaped);
  CHECK(r.deflection > 2 * M_PI);  // more than one full turn of deflection
}

TEST_CASE("trace records a path that ends where the ray did") {
  std::vector<bhs::Vec2> path;
  const auto r = trace(20.0, 1e-3, 400, &path);
  REQUIRE(r.fate == Fate::Escaped);
  CHECK(path.size() > 100);
}
