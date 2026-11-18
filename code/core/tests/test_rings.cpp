#include <doctest/doctest.h>

#include <bhs/photon_rings.hpp>
#include <bhs/schwarzschild.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("rings: far rays cross the face-on equatorial plane once (direct image)") {
  const auto c = rings::count_crossings(1e5, 30.0, /*phi_first=*/M_PI / 2);
  CHECK(c.fate == schw::Fate::Escaped);
  CHECK(c.n == 1);
}

TEST_CASE("rings: captured rays inside the shadow cross the plane at most once before the horizon") {
  const auto c = rings::count_crossings(1e5, 3.0, M_PI / 2);
  CHECK(c.fate == schw::Fate::Captured);
  CHECK(c.n <= 1);
}

TEST_CASE("rings: boundaries b_k (total sweep = pi/2 + k pi) separate crossing counts k and k+1") {
  for (int k = 1; k <= 3; ++k) {
    const double bk = rings::ring_boundary(1e5, k);
    CHECK(bk > schw::critical_b());
    const double eps = 1e-3 * (bk - schw::critical_b());
    CHECK(rings::count_crossings(1e5, bk + eps, M_PI / 2).n == k);       // larger b: fewer turns
    CHECK(rings::count_crossings(1e5, bk - eps, M_PI / 2).n == k + 1);   // smaller b: one more crossing
  }
}

TEST_CASE("rings: successive photon rings shrink by exp(-pi) (Lyapunov exponent of the Schwarzschild photon sphere)") {
  const double bc = schw::critical_b();
  double prev = rings::ring_boundary(1e5, 2) - bc;
  for (int k = 3; k <= 5; ++k) {
    const double cur = rings::ring_boundary(1e5, k) - bc;
    CHECK(cur / prev == doctest::Approx(std::exp(-M_PI)).epsilon(0.02));
    prev = cur;
  }
}
