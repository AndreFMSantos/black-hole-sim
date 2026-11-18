#include <doctest/doctest.h>

#include <bhs/exr_loader.hpp>
#include <bhs/skymap.hpp>
#include <cmath>
#include <cstdio>

using namespace bhs;

TEST_CASE("half_to_float: known values") {
  CHECK(sky::half_to_float(0x3C00) == 1.0f);
  CHECK(sky::half_to_float(0xC000) == -2.0f);
  CHECK(sky::half_to_float(0x7BFF) == 65504.0f);
  CHECK(sky::half_to_float(0x0000) == 0.0f);
  CHECK(sky::half_to_float(0x0001) == doctest::Approx(std::ldexp(1.0, -24)));  // smallest subnormal
  CHECK(sky::half_to_float(0x3555) == doctest::Approx(0.33325).epsilon(1e-3));
  CHECK(std::isinf(sky::half_to_float(0x7C00)));
}

TEST_CASE("skymap convention: +x centre, +y at u=0.25 (RA grows to the left), +z top row") {
  double u, v;
  sky::dir_to_uv({1, 0, 0}, u, v);
  CHECK(u == doctest::Approx(0.5)); CHECK(v == doctest::Approx(0.5));
  sky::dir_to_uv({0, 1, 0}, u, v);  // AR 6h
  CHECK(u == doctest::Approx(0.25)); CHECK(v == doctest::Approx(0.5));
  sky::dir_to_uv({0, -1, 0}, u, v);  // AR 18h
  CHECK(u == doctest::Approx(0.75));
  sky::dir_to_uv({-1, 0, 0}, u, v);  // RA 12h: edge (u = 0 or 1, same point with wrap)
  CHECK((u == doctest::Approx(0.0) || u == doctest::Approx(1.0)));
  sky::dir_to_uv({0, 0, 1}, u, v);
  CHECK(v == doctest::Approx(0.0));
  sky::dir_to_uv({0, 0, -1}, u, v);
  CHECK(v == doctest::Approx(1.0));
  sky::dir_to_uv({1, 0, 1}, u, v);  // dec +45
  CHECK(v == doctest::Approx(0.25));
}

TEST_CASE("skymap: direction -> uv -> direction round trip") {
  for (double ra : {0.1, 1.0, 2.5, 3.14, 4.0, 5.9})
    for (double dec : {-1.4, -0.5, 0.0, 0.7, 1.5}) {
      const Vec3 d{std::cos(dec) * std::cos(ra), std::cos(dec) * std::sin(ra), std::sin(dec)};
      double u, v;
      sky::dir_to_uv(d, u, v);
      const Vec3 b = sky::uv_to_dir(u, v);
      CHECK(norm(d - b) < 1e-12);
    }
}

static sky::Skymap make_map(int w, int h) {  // texel (x,y): R = x, G = y (as exact half, < 2048), B = 1
  sky::Skymap m; m.w = w; m.h = h; m.rgb.resize(size_t(w) * h * 3);
  auto to_half = [](int n) -> uint16_t {  // small integers (< 1024) exact in half
    if (n == 0) return 0;
    int e = 0; int t = n; while (t >= 2) { t >>= 1; ++e; }
    const int mant = (n - (1 << e)) << (10 - e);
    return uint16_t(((e + 15) << 10) | mant);
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      m.rgb[(size_t(y) * w + x) * 3 + 0] = to_half(x);
      m.rgb[(size_t(y) * w + x) * 3 + 1] = to_half(y);
      m.rgb[(size_t(y) * w + x) * 3 + 2] = 0x3C00;
    }
  return m;
}

TEST_CASE("skymap sampling: exact at texel centres, linear between, wrap in u, clamp in v") {
  const auto m = make_map(16, 8);
  // texel centre (x=3, y=2): u = (3+0.5)/16, v = (2+0.5)/8
  Vec3 c = sky::sample(m, sky::uv_to_dir(3.5 / 16, 2.5 / 8));
  CHECK(c.x == doctest::Approx(3.0).epsilon(1e-6)); CHECK(c.y == doctest::Approx(2.0).epsilon(1e-6));
  // halfway between x=3 and x=4
  c = sky::sample(m, sky::uv_to_dir(4.0 / 16, 2.5 / 8));
  CHECK(c.x == doctest::Approx(3.5).epsilon(1e-6));
  // wrap: between x=15 and x=0 (at u = 0 == 1) -> mean (15 + 0)/2
  c = sky::sample(m, sky::uv_to_dir(1.0 - 1e-12, 2.5 / 8));
  CHECK(c.x == doctest::Approx(7.5).epsilon(1e-4));
  // clamp: near the north pole, y stays at 0 (top row)
  c = sky::sample(m, Vec3{0.0001, 0, 1});
  CHECK(c.y == doctest::Approx(0.0).epsilon(1e-3));
  CHECK(c.z == doctest::Approx(1.0));
}

TEST_CASE("exr loader: real NASA map (skipped when assets/sky is not fetched)") {
  const char* path = BHS_SKY_EXR;
  FILE* f = std::fopen(path, "rb");
  if (!f) { MESSAGE("sky map not present, skipping (run tools/fetch_sky.sh)"); return; }
  std::fclose(f);
  sky::Skymap m; std::string err;
  REQUIRE_MESSAGE(sky::load_exr_half(path, m, &err), err);
  CHECK(m.w == 8192); CHECK(m.h == 4096);
  CHECK(m.rgb.size() == size_t(8192) * 4096 * 3);
  const Vec3 c = sky::sample(m, {1, 0, 0});
  CHECK(c.x >= 0.0); CHECK(c.x < 65505.0);
}

TEST_CASE("exr writer: exact float32 round trip (linear HDR: values > 1 and very small ones); also validated with the reference OpenEXR library") {
  const int w = 16, h = 8;
  std::vector<float> img(size_t(w) * h * 3);
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
    const size_t i = (size_t(y) * w + x) * 3;
    img[i] = 0.001f * float(x + 1);              // 1e-3 ... 1.6e-2
    img[i + 1] = 0.5f + 100.0f * float(y);       // up to 700.5: well above 1
    img[i + 2] = (x == 3 && y == 2) ? 1500.0f : 0.25f;
  }
  const std::string path = "/tmp/bhs_exr_roundtrip_test.exr";
  std::string err;
  REQUIRE_MESSAGE(sky::save_exr_rgb(path, img.data(), w, h, &err), err);
  std::vector<float> back; int w2 = 0, h2 = 0;
  REQUIRE_MESSAGE(sky::load_exr_float(path, back, w2, h2, &err), err);
  REQUIRE(w2 == w); REQUIRE(h2 == h);
  size_t diff = 0;
  for (size_t i = 0; i < size_t(w) * h; ++i) for (int c = 0; c < 3; ++c) diff += back[4 * i + c] != img[3 * i + c];
  CHECK(diff == 0);                               // lossless float32
  std::remove(path.c_str());
}
