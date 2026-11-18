#include <doctest/doctest.h>

#include <bhs/kerr.hpp>
#include <bhs/lorentz.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("lorentz: zero velocity is the identity") {
  const Vec3 n = normalize(Vec3{0.3, -0.5, 0.8});
  const auto p = lorentz::observer_to_zamo(n, {0, 0, 0});
  CHECK(p.n.x == doctest::Approx(n.x).epsilon(1e-14));
  CHECK(p.n.y == doctest::Approx(n.y).epsilon(1e-14));
  CHECK(p.n.z == doctest::Approx(n.z).epsilon(1e-14));
  CHECK(p.dop == doctest::Approx(1.0).epsilon(1e-14));
}

TEST_CASE("lorentz: aberration of a photon perpendicular to the motion, cos(theta) = beta") {
  const double b = 0.6, gamma = 1.0 / std::sqrt(1.0 - b * b);
  const auto p = lorentz::observer_to_zamo({0, 1, 0}, {b, 0, 0});  // photon perpendicular in the observer's frame
  CHECK(p.n.x == doctest::Approx(b).epsilon(1e-14));               // in the ZAMO frame it tilts forward
  CHECK(p.n.y == doctest::Approx(1.0 / gamma).epsilon(1e-14));
  CHECK(norm(p.n) == doctest::Approx(1.0).epsilon(1e-14));
  CHECK(p.dop == doctest::Approx(1.0 / gamma).epsilon(1e-14));     // E'/E = 1/gamma (Doppler transversal)
}

TEST_CASE("lorentz: head-on and tail-on Doppler factors") {
  const double b = 0.6;
  // observer moving along +x; photon propagating along -x in the observer frame (head-on): blueshift
  const auto head = lorentz::observer_to_zamo({-1, 0, 0}, {b, 0, 0});
  CHECK(head.dop == doctest::Approx(std::sqrt((1 + b) / (1 - b))).epsilon(1e-14));   // E'/E = sqrt((1+b)/(1-b)) = 2
  const auto tail = lorentz::observer_to_zamo({1, 0, 0}, {b, 0, 0});
  CHECK(tail.dop == doctest::Approx(std::sqrt((1 - b) / (1 + b))).epsilon(1e-14));
}

TEST_CASE("lorentz: round trip observer -> zamo -> observer") {
  const Vec3 beta{0.2, -0.3, 0.4};
  for (Vec3 n : {Vec3{1, 0, 0}, Vec3{0, 0, 1}, normalize(Vec3{0.3, 0.7, -0.2}), normalize(Vec3{-0.8, 0.1, 0.5})}) {
    const auto p = lorentz::observer_to_zamo(n, beta);
    const auto back = lorentz::zamo_to_observer(p.n, beta);
    CHECK(back.n.x == doctest::Approx(n.x).epsilon(1e-12));
    CHECK(back.n.y == doctest::Approx(n.y).epsilon(1e-12));
    CHECK(back.n.z == doctest::Approx(n.z).epsilon(1e-12));
    CHECK(back.dop == doctest::Approx(p.dop).epsilon(1e-12));  // same photon: E'/E equal in both directions
  }
}

TEST_CASE("lorentz: velocity of a circular geodesic orbit relative to the ZAMO") {
  // a = 0: the ZAMO is the static observer; v = sqrt(M / (r - 2M))
  for (double r : {8.0, 20.0, 40.0}) {
    CHECK(lorentz::circular_beta(0.0, r, M_PI / 2) == doctest::Approx(std::sqrt(1.0 / (r - 2.0))).epsilon(1e-12));
  }
  // in Kerr the velocity relative to the ZAMO is below 1 even near the ISCO
  const double a = 0.9;
  CHECK(lorentz::circular_beta(a, kerr::isco(a), M_PI / 2) < 1.0);
  CHECK(lorentz::circular_beta(a, kerr::isco(a), M_PI / 2) > 0.5);
}

TEST_CASE("lorentz: free fall from rest at infinity has v = sqrt(2M/r) relative to the static observer") {
  CHECK(lorentz::infall_beta(10.0) == doctest::Approx(std::sqrt(2.0 / 10.0)).epsilon(1e-14));
  CHECK(lorentz::infall_beta(2.0) == doctest::Approx(1.0).epsilon(1e-14));  // at the horizon v -> c
  // exact Kerr infall: a = 0 limit, gamma = 1/alpha (energy conservation, independent of the formula's derivation), beta < 1 outside the horizon, -> 1 at r+
  CHECK(lorentz::infall_beta_kerr(0.0, 10.0, 1.2) == doctest::Approx(std::sqrt(2.0 / 10.0)).epsilon(1e-14));
  for (double a : {0.5, 0.9, 0.998}) for (double th : {0.3, 1.0, M_PI / 2}) for (double r : {1.6, 2.0, 3.0, 10.0, 100.0}) {
    const double rp = 1.0 + std::sqrt(1.0 - a * a); if (r <= rp) continue;
    const double b = lorentz::infall_beta_kerr(a, r, th), s = std::sin(th), mu = std::cos(th), Sigma = r * r + a * a * mu * mu, Delta = kerr::delta(r, a);
    const double A = std::pow(r * r + a * a, 2) - a * a * Delta * s * s, alpha = std::sqrt(Sigma * Delta / A);
    CHECK(b < 1.0);
    CHECK(1.0 / std::sqrt(1.0 - b * b) == doctest::Approx(1.0 / alpha).epsilon(1e-9));
  }
  CHECK(lorentz::infall_beta_kerr(0.9, 1.0 + std::sqrt(1.0 - 0.81) + 1e-9, 1.0) > 0.9999);
}
