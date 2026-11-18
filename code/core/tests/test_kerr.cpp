#include <doctest/doctest.h>

#include <bhs/disk.hpp>
#include <bhs/kerr.hpp>
#include <bhs/schwarzschild.hpp>
#include <bhs/vec3.hpp>
#include <cmath>

using namespace bhs;

namespace {
// specific energy of a prograde equatorial circular orbit in Kerr (M=1)
double circ_energy(double r, double a) {
  return (std::pow(r, 1.5) - 2 * std::sqrt(r) + a) /
         (std::pow(r, 0.75) * std::sqrt(std::pow(r, 1.5) - 3 * std::sqrt(r) + 2 * a));
}
}  // namespace

TEST_CASE("kerr: horizon, ISCO and photon orbit limits") {
  CHECK(kerr::horizon(0.0) == doctest::Approx(2.0));
  CHECK(kerr::horizon(0.6) == doctest::Approx(1.8));
  CHECK(kerr::isco(0.0) == doctest::Approx(6.0));
  CHECK(kerr::isco(1.0) == doctest::Approx(1.0));
  CHECK(kerr::isco(1.0, false) == doctest::Approx(9.0));
  CHECK(kerr::photon_orbit(0.0, true) == doctest::Approx(3.0));
  CHECK(kerr::photon_orbit(1.0, true) == doctest::Approx(1.0));
  CHECK(kerr::photon_orbit(1.0, false) == doctest::Approx(4.0));
}

TEST_CASE("kerr: ISCO is the minimum of the circular-orbit energy (independent check)") {
  for (double a : {0.3, 0.9, 0.998}) {
    const double r = kerr::isco(a);
    const double h = 1e-4;
    const double dE = (circ_energy(r + h, a) - circ_energy(r - h, a)) / (2 * h);
    CHECK(std::abs(dE) < 1e-6);
    CHECK(circ_energy(r + 0.05, a) > circ_energy(r, a));
    CHECK(circ_energy(r - 0.02, a) > circ_energy(r, a));  // inwards the energy rises again
  }
}

TEST_CASE("kerr: equatorial photon orbit is a double root of R(r) (independent check)") {
  for (double a : {0.5, 0.9}) {
    for (bool pro : {true, false}) {
      const double r = kerr::photon_orbit(a, pro);
      const kerr::Consts c{a, kerr::xi_photon(r, a), 0.0};  // Q = 0 in the equatorial plane
      CHECK(kerr::R(r, c) == doctest::Approx(0.0).epsilon(1e-9));
      CHECK(kerr::dR(r, c) == doctest::Approx(0.0).epsilon(1e-9));
    }
  }
}

TEST_CASE("kerr: constants built from the ZAMO frame satisfy the null condition") {
  const double a = 0.9, r0 = 40.0, th0 = 1.4;
  for (auto n : {std::array<double, 3>{0.8, 0.1, -0.5899152}, std::array<double, 3>{0.3, -0.4, 0.8660254},
                 std::array<double, 3>{-0.6, 0.0, 0.8}}) {
    const double nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const auto in = kerr::init_from_zamo(a, r0, th0, n[0] / nn, n[1] / nn, n[2] / nn);
    const kerr::Consts c{a, in.L, in.Q};
    const double Delta = r0 * r0 - 2 * r0 + a * a;
    const double mu = std::cos(th0);
    // (dr/dtau)^2 = R(r), (dmu/dtau)^2 = M(mu); dr/dtau = Delta*p_r ; dmu/dtau = -sin(th) p_theta
    CHECK(kerr::R(r0, c) == doctest::Approx(std::pow(Delta * in.pr, 2)).epsilon(1e-9));
    CHECK(kerr::Mmu(mu, c) == doctest::Approx(std::pow(std::sin(th0) * in.ptheta, 2)).epsilon(1e-9));
  }
}

TEST_CASE("dormand-prince 5(4): harmonic oscillator, error shrinks with tolerance") {
  auto run = [](double tol) {
    std::array<double, 2> y{1.0, 0.0};
    double t = 0, h = 0.1;
    auto f = [](const std::array<double, 2>& s) { return std::array<double, 2>{s[1], -s[0]}; };
    while (t < 10.0) {
      h = std::min(h, 10.0 - t);
      std::array<double, 2> yn, k1, k7;
      const double err = kerr::dp45_step(f, y, h, yn, k1, k7, tol, tol);
      if (err <= 1.0) { y = yn; t += h; }
      h *= kerr::step_factor(err);
    }
    return std::abs(y[0] - std::cos(10.0)) + std::abs(y[1] + std::sin(10.0));
  };
  const double e6 = run(1e-6), e10 = run(1e-10);
  CHECK(e6 < 1e-4);
  CHECK(e10 < 1e-8);
  CHECK(e10 < e6);
}

TEST_CASE("kerr a=0 reproduces the Schwarzschild deflection (equatorial ray)") {
  const double D = 40.0;
  for (double b : {8.0, 12.0, 30.0}) {
    // ray coming from infinity with impact parameter b, static observer = ZAMO if a=0
    const double sin_a = b * std::sqrt(1 - 2.0 / D) / D;
    const double cos_a = std::sqrt(1 - sin_a * sin_a);
    // n = photon (-d); d = (-cos_a e_r, 0, +sin_a e_phi) => n = (cos_a, 0, -sin_a)
    kerr::Options o;
    o.r_far = 2e4;
    const auto t = kerr::trace(0.0, D, M_PI / 2, cos_a, 0.0, -sin_a, {}, o);
    const auto s = schw::trace_from(D, b, true, 1e-3);
    REQUIRE(t.fate == kerr::Fate::Escaped);
    REQUIRE(s.fate == schw::Fate::Escaped);
    CHECK(std::abs(t.phi_end) == doctest::Approx(s.phi_end).epsilon(2e-3));  // the finite r_far leaves ~b/r_far
  }
}

TEST_CASE("kerr a=0: capture threshold is sqrt(27)") {
  const double D = 40.0;
  auto fate = [&](double b) {
    const double sin_a = b * std::sqrt(1 - 2.0 / D) / D;
    const double cos_a = std::sqrt(1 - sin_a * sin_a);
    return kerr::trace(0.0, D, M_PI / 2, cos_a, 0.0, -sin_a).fate;
  };
  CHECK(fate(5.1) == kerr::Fate::Captured);
  CHECK(fate(5.3) == kerr::Fate::Escaped);
}

TEST_CASE("kerr a=0.9: equatorial shadow edges match the circular photon orbits (Bardeen 1973)") {
  // distant observer in the equatorial plane; the ray's L ~ impact parameter. Bisection on the direction.
  const double a = 0.9, D = 5000.0;
  for (bool prograde : {true, false}) {
    const double target = std::abs(kerr::xi_photon(kerr::photon_orbit(a, prograde), a));
    const double sgn = prograde ? 1.0 : -1.0;  // n_phi > 0: photon with L > 0 (co-rotating)
    auto trace_at = [&](double alpha) {
      return kerr::trace(a, D, M_PI / 2, std::cos(alpha), 0.0, sgn * std::sin(alpha));
    };
    double lo = 0.5 / D, hi = 20.0 / D;  // b de ~0.5 a 20
    REQUIRE(trace_at(lo).fate != kerr::Fate::Escaped);
    REQUIRE(trace_at(hi).fate == kerr::Fate::Escaped);
    for (int i = 0; i < 40; ++i) {
      const double mid = 0.5 * (lo + hi);
      (trace_at(mid).fate == kerr::Fate::Escaped ? hi : lo) = mid;
    }
    CHECK(std::abs(trace_at(0.5 * (lo + hi)).L) == doctest::Approx(target).epsilon(2e-3));
  }
}

TEST_CASE("kerr: the first integrals stay conserved along a scattering ray") {
  kerr::Options o;
  const auto t = kerr::trace(0.9, 40.0, 1.2, 0.9, 0.2, -0.38729833462, {}, o);  // normalised n ~ 1
  REQUIRE(t.fate == kerr::Fate::Escaped);
  CHECK(t.max_inv_err < 1e-6);
}

TEST_CASE("kerr a=0 disk crossing agrees with the Schwarzschild trace_with_disk") {
  // v0.4: observer at theta = 80 degrees, D = 40, pixel with direction d = fwd + x*right + y*up
  const double th = 80.0 * M_PI / 180.0, D = 40.0;
  const double x = 0.05, y = -0.04;  // small tangential deviations (tan)
  // in ZAMO coordinates (a=0): fwd = -e_r ; up = -e_theta ; right = +e_phi
  const double nrm = std::sqrt(1 + x * x + y * y);
  const double d_r = -1 / nrm, d_th = -y / nrm, d_ph = x / nrm;
  const auto t = kerr::trace(0.0, D, th, -d_r, -d_th, -d_ph, {6.0, 16.0});

  // Schwarzschild reference (geometry of the v0.4 shade)
  const Vec3 c_hat{std::sin(th), 0, std::cos(th)};
  const Vec3 fwd = -1.0 * c_hat, up0{0, 0, 1};
  Vec3 right = normalize(cross(fwd, up0));
  const Vec3 up = cross(right, fwd);
  const Vec3 d = normalize(fwd + x * right + y * up);
  const double cos_a = dot(d, c_hat);
  const Vec3 perp = d - cos_a * c_hat;
  const double sin_a = norm(perp);
  const Vec3 t_hat = (1.0 / sin_a) * perp;
  const double alpha = std::atan2(sin_a, cos_a);
  const double b = D * std::sin(alpha) / std::sqrt(1 - 2.0 / D);
  double phi0 = std::atan2(-c_hat.z, t_hat.z);
  while (phi0 <= 1e-9) phi0 += M_PI;
  while (phi0 > M_PI) phi0 -= M_PI;
  const auto s = schw::trace_with_disk(D, b, cos_a < 0, phi0, 6.0, 16.0, 1e-3);
  REQUIRE(s.disk.hit);
  REQUIRE(t.disk_hit);
  CHECK(t.disk_r == doctest::Approx(s.disk.r).epsilon(2e-4));
}

TEST_CASE("kerr redshift with a=0 equals the Schwarzschild disk redshift") {
  const double r = 10.0, D = 40.0, L = 3.0;
  CHECK(kerr::redshift(0.0, r, L, D, M_PI / 2) == doctest::Approx(disk::redshift(r, D, L)).epsilon(1e-12));
}

// ---- regression: rotation axis (phi singularity in Boyer-Lindquist) ----
namespace {
Vec3 exit_dir(const kerr::Trace& t) {
  return {std::sin(t.theta_end) * std::cos(t.phi_end), std::sin(t.theta_end) * std::sin(t.phi_end), std::cos(t.theta_end)};
}
kerr::Trace trace_xy(double a, double D, double theta, double x, double y) {
  const double nrm = std::sqrt(1.0 + x * x + y * y);
  kerr::Options o;
  o.rtol = 1e-8;
  o.atol = 1e-9;
  return kerr::trace(a, D, theta, 1.0 / nrm, y / nrm, -x / nrm, {}, o);
}
}  // namespace

TEST_CASE("kerr: rays with L ~ 0 (plane containing the spin axis) cross the pole without a phi jump") {
  // The ray with exact L = 0 crosses the pole: without regularisation phi does not jump by pi (comes out wrong by pi) and
  // with |L| ~ 1e-8 the jump is stiff (1e5 steps). All rays must end in the same direction as their neighbours.
  const double a = 0.9, th = 80.0 * M_PI / 180.0, y = 0.2;
  const auto ref = trace_xy(a, 40.0, th, 1e-4, y);
  REQUIRE(ref.fate == kerr::Fate::Escaped);
  for (double x : {0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-7, -1e-7}) {
    const auto t = trace_xy(a, 40.0, th, x, y);
    REQUIRE(t.fate == kerr::Fate::Escaped);
    CHECK(t.steps < 5000);
    CHECK(norm(exit_dir(t) - exit_dir(ref)) < 1e-2);
  }
}

TEST_CASE("kerr a=0: the deflection of a ray does not depend on the camera polar angle, even at the pole") {
  // In Schwarzschild the ray is planar: the angle between the observer position and the exit direction only depends
  // on (x, y) and not on theta. Near the axis (theta = 0.5 degrees) the Boyer-Lindquist phi is almost singular.
  for (double y : {0.2, 0.25, -0.3})  // outside the shadow (b > sqrt(27) ~ tan 7.5 degrees)
    for (double x : {0.0, 0.1, -0.25}) {
      double ang[3];
      int k = 0;
      for (double thd : {90.0, 0.5, 179.5}) {
        const double th = thd * M_PI / 180.0;
        const auto t = trace_xy(0.0, 40.0, th, x, y);
        REQUIRE(t.fate == kerr::Fate::Escaped);
        CHECK(t.steps < 5000);
        const Vec3 p{std::sin(th), 0.0, std::cos(th)};
        ang[k++] = std::acos(std::clamp(dot(p, exit_dir(t)), -1.0, 1.0));
      }
      CHECK(ang[1] == doctest::Approx(ang[0]).epsilon(1e-5));
      CHECK(ang[2] == doctest::Approx(ang[0]).epsilon(1e-5));
    }
}
