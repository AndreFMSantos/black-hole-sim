#include <doctest/doctest.h>

#include <bhs/blackbody.hpp>
#include <bhs/disk.hpp>
#include <bhs/schwarzschild.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("disk temperature: zero at ISCO, peak 1 at 49/36 r_in, ~r^-3/4 far away") {
  CHECK(disk::temperature(6.0, 6.0) == doctest::Approx(0.0).epsilon(1e-12));
  const double r_peak = 49.0 / 36.0 * 6.0;
  CHECK(disk::temperature(r_peak, 6.0) == doctest::Approx(1.0).epsilon(1e-9));
  CHECK(disk::temperature(r_peak * 0.9, 6.0) < 1.0);
  CHECK(disk::temperature(r_peak * 1.1, 6.0) < 1.0);
  const double ratio = disk::temperature(200.0, 6.0) / disk::temperature(100.0, 6.0);
  CHECK(ratio == doctest::Approx(std::pow(2.0, -0.75)).epsilon(0.03));
}

TEST_CASE("redshift without Doppler: gravitational + transverse only") {
  const double r = 10.0, D = 40.0;
  CHECK(disk::redshift(r, D, 0.0) == doctest::Approx(std::sqrt(1 - 3.0 / r) / std::sqrt(1 - 2.0 / D)));
}

TEST_CASE("redshift reduces to the special-relativistic Doppler factor far from the hole") {
  // Nearly flat space: emitter at large r in the disk, photon towards a very distant observer.
  const double r = 1e5, D = 1e9;
  const double omega = std::sqrt(1.0 / (r * r * r));
  for (double ang : {0.0, 1.0, 2.0, 3.0, 4.5}) {  // emitter position in the disk (azimuth)
    const Vec3 P{r * std::cos(ang), r * std::sin(ang), 0};
    const Vec3 phi_hat{-std::sin(ang), std::cos(ang), 0};
    for (Vec3 n : {Vec3{0.6, 0.0, 0.8}, Vec3{-0.3, 0.9539392014169457, 0.0}, Vec3{0.0, -0.6, 0.8}}) {
      // n = direction of the photon emitter->observer (unit)
      const double lz = cross(P, n).z;  // L_z/E of the photon
      const double beta = omega * r;    // = sqrt(M/r)
      const double g_sr = 1.0 / ((1.0 / std::sqrt(1 - beta * beta)) * (1 - dot(n, beta * phi_hat)));
      CHECK(disk::redshift(r, D, lz) == doctest::Approx(g_sr).epsilon(1e-3));
    }
  }
}

TEST_CASE("blackbody colour is white-ish at 6500K, red when cool, blue when hot") {
  const Vec3 w = blackbody::rgb(6500), c = blackbody::rgb(3000), h = blackbody::rgb(15000);
  CHECK(w.x / w.y == doctest::Approx(1.0).epsilon(0.1));
  CHECK(w.z / w.y == doctest::Approx(1.0).epsilon(0.1));
  CHECK(c.x > c.y);
  CHECK(c.y > c.z);
  CHECK(h.z > h.x);
  CHECK(blackbody::rgb(8000).y > blackbody::rgb(4000).y);  // hotter = more luminous
}

TEST_CASE("disk crossing: a nearly-straight ray hits the disk where geometry predicts") {
  // Ray enters from far away (D=1000) with b=50; the disk plane cuts the trajectory at phi = pi/2.
  // In flat space it would be r = b / cos(asin(b/D)) ~ 50.00; Schwarzschild deflects little.
  const auto t = schw::trace_with_disk(1000.0, 50.0, true, M_PI / 2, 40.0, 60.0, 1e-3);
  REQUIRE(t.disk.hit);
  CHECK(t.disk.r == doctest::Approx(50.0).epsilon(0.03));
  CHECK(t.disk.phi == doctest::Approx(M_PI / 2).epsilon(1e-6));
}

TEST_CASE("disk crossing: ray misses a disk that is not at its radius and escapes") {
  const auto t = schw::trace_with_disk(1000.0, 50.0, true, M_PI / 2, 10.0, 20.0, 1e-3);
  CHECK_FALSE(t.disk.hit);
  CHECK(t.fate == schw::Fate::Escaped);
}

TEST_CASE("disk crossing is converged in the step size, also for strongly bent rays") {
  const double b = 6.0;  // just above the shadow: bends a lot, crosses the plane several times
  const auto a = schw::trace_with_disk(1000.0, b, true, 1.1, 2.0, 25.0, 1e-3);
  const auto c = schw::trace_with_disk(1000.0, b, true, 1.1, 2.0, 25.0, 5e-4);
  REQUIRE(a.disk.hit == c.disk.hit);
  if (a.disk.hit) CHECK(a.disk.r == doctest::Approx(c.disk.r).epsilon(1e-4));
}

TEST_CASE("a captured ray reports a disk hit only if it crosses the plane before the horizon") {
  // b = 4 is captured after ~2.05 rad (measured): the plane at 1.0 rad is crossed before, the one at 2.5 rad is not.
  const auto before = schw::trace_with_disk(1000.0, 4.0, true, 1.0, 2.5, 25.0, 1e-3);
  CHECK(before.disk.hit);
  CHECK(before.disk.r < 10.0);  // already close to the hole
  CHECK(before.disk.r > 2.5);
  const auto after = schw::trace_with_disk(1000.0, 4.0, true, 2.5, 2.5, 25.0, 1e-3);
  CHECK_FALSE(after.disk.hit);
  CHECK(after.fate == schw::Fate::Captured);
}
