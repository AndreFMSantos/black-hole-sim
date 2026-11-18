#include <doctest/doctest.h>

#include <bhs/blackbody.hpp>
#include <bhs/camera.hpp>
#include <bhs/lut.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("camera: azimuth 0 reproduces the v0.4 observer in the x-z plane") {
  camera::Orbit c;
  c.theta = 80.0 * M_PI / 180.0;
  Vec3 fwd, right, up;
  c.basis(fwd, right, up);
  const Vec3 c_hat = c.c_hat();
  CHECK(c_hat.y == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(c_hat.x == doctest::Approx(std::sin(c.theta)));
  CHECK(right.y == doctest::Approx(1.0).epsilon(1e-9));  // v0.4: right = +y with the observer at +x
}

TEST_CASE("camera: basis is orthonormal for any angles and rotates rigidly with azimuth") {
  for (double th : {0.3, 1.0, 1.5, 2.4, 3.0})
    for (double ph : {0.0, 0.7, 2.0, -1.3, 4.0}) {
      camera::Orbit c; c.theta = th; c.phi = ph;
      Vec3 f, r, u; c.basis(f, r, u);
      CHECK(norm(c.c_hat()) == doctest::Approx(1.0));
      CHECK(norm(r) == doctest::Approx(1.0)); CHECK(norm(u) == doctest::Approx(1.0));
      CHECK(dot(f, r) == doctest::Approx(0.0).epsilon(1e-9));
      CHECK(dot(f, u) == doctest::Approx(0.0).epsilon(1e-9));
      CHECK(dot(r, u) == doctest::Approx(0.0).epsilon(1e-9));
      CHECK(r.z == doctest::Approx(0.0).epsilon(1e-9));  // screen horizon always parallel to the disk
      CHECK(u.z >= 0.0);                                   // screen "up" points to +z
      // camera with azimuth 0 rotated by ph about z
      camera::Orbit c0; c0.theta = th; Vec3 f0, r0, u0; c0.basis(f0, r0, u0);
      const double cs = std::cos(ph), sn = std::sin(ph);
      CHECK(r.x == doctest::Approx(cs * r0.x - sn * r0.y).epsilon(1e-9));
      CHECK(r.y == doctest::Approx(sn * r0.x + cs * r0.y).epsilon(1e-9));
      CHECK(u.z == doctest::Approx(u0.z).epsilon(1e-9));
    }
}

TEST_CASE("camera: clamp keeps distance >= 10 and avoids the poles") {
  camera::Orbit c; c.dist = 3.0; c.theta = -1.0; c.phi = 9.0;
  c.clamp();
  CHECK(c.dist == doctest::Approx(10.0));
  CHECK(c.theta > 0.0);
  CHECK(std::abs(c.phi) <= M_PI);
}

TEST_CASE("lut: nodes equal blackbody::rgb and lookup interpolates between them") {
  const auto d = lut::build();
  REQUIRE(d.size() == size_t(lut::kN) * 3);
  for (int i : {0, 10, 500, 1023}) {
    const Vec3 ref = blackbody::rgb(lut::temperature_at(i));
    CHECK(d[3 * i] == doctest::Approx(ref.x).epsilon(1e-6));
    CHECK(d[3 * i + 1] == doctest::Approx(ref.y).epsilon(1e-6));
    CHECK(d[3 * i + 2] == doctest::Approx(ref.z).epsilon(1e-6));
  }
  CHECK(norm(lut::lookup(d, 100.0)) == 0.0);  // below T0: black
  // between nodes, small error against direct integration
  const Vec3 a = lut::lookup(d, 4200.0), b = blackbody::rgb(4200.0);
  CHECK(a.x == doctest::Approx(b.x).epsilon(0.01));
  CHECK(a.z == doctest::Approx(b.z).epsilon(0.02));
  // luminance monotonic with T (hotter = brighter)
  CHECK(lut::lookup(d, 6000).y > lut::lookup(d, 3000).y);
}
