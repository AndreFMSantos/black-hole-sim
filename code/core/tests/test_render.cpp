#include <doctest/doctest.h>

#include <bhs/observer.hpp>
#include <bhs/png.hpp>
#include <bhs/schwarzschild.hpp>
#include <bhs/stars.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("crc32 and adler32 match the standard check values") {
  const std::string s = "123456789";
  CHECK(png::crc32(reinterpret_cast<const uint8_t*>(s.data()), s.size()) == 0xCBF43926u);
  const std::string w = "Wikipedia";
  CHECK(png::adler32(reinterpret_cast<const uint8_t*>(w.data()), w.size()) == 0x11E60398u);
}

TEST_CASE("png encoder produces a valid signature, IHDR and IEND") {
  std::vector<uint8_t> rgb(4 * 3 * 3, 128);
  const auto bytes = png::encode_rgb(4, 3, rgb);
  const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  REQUIRE(bytes.size() > 33);
  for (int i = 0; i < 8; ++i) CHECK(bytes[i] == sig[i]);
  CHECK(std::string(bytes.begin() + 12, bytes.begin() + 16) == "IHDR");
  CHECK(std::string(bytes.end() - 8, bytes.end() - 4) == "IEND");
}

TEST_CASE("trace_from an inward start matches trace from infinity up to the missing arc") {
  const double b = 20.0, r0 = 1e4;
  const auto far = schw::trace(b, 1e-3);
  const auto near = schw::trace_from(r0, b, true, 1e-3);
  REQUIRE(near.fate == schw::Fate::Escaped);
  // from r0 the ray skips the first ~asin(b/r0) radians of the sweep
  CHECK(near.phi_end + std::asin(b / r0) == doctest::Approx(far.phi_end).epsilon(1e-4));
}

TEST_CASE("an outward ray far from the hole escapes with a small sweep") {
  // observer at r=30, ray at 0.2 rad from the radial direction, moving away
  const double b = observer::impact_parameter(30.0, 0.2);
  const auto r = schw::trace_from(30.0, b, false, 1e-3);
  REQUIRE(r.fate == schw::Fate::Escaped);
  CHECK(r.phi_end > 0.1);  // ~alpha plus a bit of curvature
  CHECK(r.phi_end < 0.3);
}

TEST_CASE("impact parameter from the viewing angle (static observer)") {
  CHECK(observer::impact_parameter(30.0, 0.0) == doctest::Approx(0.0));
  // b = r sin(alpha)/sqrt(1-2M/r)
  const double a = 0.1;
  CHECK(observer::impact_parameter(30.0, a) ==
        doctest::Approx(30.0 * std::sin(a) / std::sqrt(1 - 2.0 / 30.0)));
}

TEST_CASE("starfield is deterministic, bounded and non-empty") {
  int lit = 0;
  for (int i = 0; i < 2000; ++i) {
    const double az = 0.0031 * i, el = 0.0007 * i - 0.7;
    const Vec3 v{std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
    const Vec3 c1 = stars::sample(v), c2 = stars::sample(v);
    CHECK(c1.x == c2.x);
    CHECK(c1.x >= 0.0);
    if (c1.x + c1.y + c1.z > 0.05) ++lit;
  }
  CHECK(lit > 0);
}

TEST_CASE("apparent shadow edge seen by an observer at r=30 is asin(b_c sqrt(1-2/r)/r)") {
  const double r = 30.0;
  const double alpha_c = std::asin(schw::critical_b() * std::sqrt(1 - 2.0 / r) / r);  // vs direction to the hole
  // alpha measured from the OUTWARD RADIAL direction: pi - alpha_c is the edge; inside the shadow: capture
  auto fate = [&](double a_from_hole) {
    const double b = observer::impact_parameter(r, a_from_hole);
    return schw::trace_from(r, b, true, 1e-3).fate;
  };
  CHECK(fate(alpha_c * 0.999) == schw::Fate::Captured);
  CHECK(fate(alpha_c * 1.001) == schw::Fate::Escaped);
}
