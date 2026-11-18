#include <doctest/doctest.h>

#include <bhs/kerr.hpp>
#include <cmath>
#include <initializer_list>

using namespace bhs;

namespace {
// Schwarzschild, photon with impact parameter b, monotonic radial motion between r_e and r_obs (no turning point):
//   t = int dr / ((1 - 2/r) sqrt(1 - b^2 (1 - 2/r) / r^2))       (M = 1, E = 1)
// Composite Simpson in r (smooth integrand in this interval): an oracle INDEPENDENT of Kerr's RK45 integrator.
double flight_time_schw(double b, double r_e, double r_obs) {
  const int n = 200000;
  const double h = (r_obs - r_e) / n;
  auto f = [&](double r) { return 1.0 / ((1.0 - 2.0 / r) * std::sqrt(1.0 - b * b * (1.0 - 2.0 / r) / (r * r))); };
  double s = f(r_e) + f(r_obs);
  for (int i = 1; i < n; ++i) s += f(r_e + h * i) * (i % 2 ? 4.0 : 2.0);
  return s * h / 3.0;
}
// periastron of a Schwarzschild photon with impact parameter b > sqrt(27): largest root of r^3 = b^2 (r - 2) (Newton from far away)
double periapsis(double b) {
  double r = 4.0 * b;
  for (int i = 0; i < 200; ++i) r -= (r * r * r - b * b * (r - 2.0)) / (3.0 * r * r - b * b);
  return r;
}
}  // namespace

TEST_CASE("slow light: kerr::trace returns the coordinate flight time camera -> disc (Schwarzschild quadrature oracle)") {
  const double a = 0.0, D = 60.0, th = 1.0 * M_PI / 180.0;
  int tested = 0;
  for (double x : {0.095, 0.10, 0.11, 0.12, 0.14}) {
    const double y = 0.0, nrm = std::sqrt(1 + x * x + y * y);
    kerr::Options opt; opt.atol = 1e-11; opt.rtol = 1e-10;
    const auto t = kerr::trace(a, D, th, 1.0 / nrm, y / nrm, -x / nrm, kerr::Disk{3.0, 40.0}, opt);
    REQUIRE(t.disk_hit);
    const double b = std::sqrt(t.Q + t.L * t.L);
    // the oracle assumes monotonic radial motion: the crossing has to be before the periastron (with margin: the integrand
    // has an integrable singularity at r_p and Simpson's rule loses precision close to it; measured: x = 0.22 has r_e - r_p ~ 0.02)
    REQUIRE(t.disk_r > periapsis(b) + 0.1);
    const double expect = flight_time_schw(b, t.disk_r, D);
    CHECK(t.t_disk == doctest::Approx(expect).epsilon(2e-7));
    ++tested;
  }
  CHECK(tested == 5);
}

TEST_CASE("slow light: in flat-space limit the flight time is the Euclidean distance (camera far, large radii)") {
  const double a = 0.0, D = 4000.0, th = 60.0 * M_PI / 180.0, rt = 1500.0;
  const double cx = D * std::sin(th), cz = D * std::cos(th);
  const double dx = rt - cx, dz = -cz, nrm = std::hypot(dx, dz);
  const double nr = -(dx * std::sin(th) + dz * std::cos(th)) / nrm, nth = -(dx * std::cos(th) - dz * std::sin(th)) / nrm;
  kerr::Options opt; opt.atol = 1e-10; opt.rtol = 1e-10;
  const auto t = kerr::trace(a, D, th, nr, nth, 0.0, kerr::Disk{6.0, 3000.0}, opt);
  REQUIRE(t.disk_hit);
  // Euclidean + Shapiro delay ~ 2 ln(...) of order 10: 0.3 % of the distance
  CHECK(t.t_disk == doctest::Approx(nrm).epsilon(5e-3));
  CHECK(t.t_disk > nrm);  // the gravitational delay only increases the time
}

TEST_CASE("slow light: the time-of-flight integrand dt/dtau matches the Kerr formula and reduces to Schwarzschild at a = 0") {
  // a = 0: dt/dtau = r^4 / Delta = r^3 / (r - 2)
  for (double r : {5.0, 12.0, 40.0}) {
    const kerr::Consts c{0.0, 1.7, 9.0};
    CHECK(kerr::dt_dtau(r, 0.3, c) == doctest::Approx(r * r * r / (r - 2.0)).epsilon(1e-12));
  }
  // a != 0, L = 0, equator (mu = 0): dt/dtau = (r^2 + a^2)^2 / Delta - a^2
  const kerr::Consts c{0.9, 0.0, 4.0};
  const double r = 9.0, d = kerr::delta(r, 0.9);
  CHECK(kerr::dt_dtau(r, 0.0, c) == doctest::Approx((r * r + 0.81) * (r * r + 0.81) / d - 0.81).epsilon(1e-12));
}
