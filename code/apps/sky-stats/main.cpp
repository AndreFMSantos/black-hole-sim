// Luminance distribution of the sky map (to choose --sky-exposure).
// Usage: sky-stats starmap_2020_8k.exr
// Weights each pixel by the solid angle (cos dec). "Milky Way" = galactic latitude |b| < 5 degrees;
// "high" = |b| > 60 degrees (almost only isolated stars and background).
#include <algorithm>
#include <bhs/exr_loader.hpp>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace bhs;

struct Acc {
  void frac(const char* name) {
    double tot = 0; for (auto& p : v) tot += p.second;
    std::printf("  %-20s area with lum*S > 0.5 (disk brightness):", name);
    for (double S : {1.0, 2.0, 3.0, 4.0}) { double a = 0; for (auto& p : v) if (p.first * S > 0.5) a += p.second; std::printf("  S=%g: %.4f%%", S, 100 * a / tot); }
    std::printf("\n");
  }
  std::vector<std::pair<float, float>> v;  // (luminance, weight)
  void add(float l, float w) { v.push_back({l, w}); }
  void report(const char* name) {
    std::sort(v.begin(), v.end());
    double tot = 0; for (auto& p : v) tot += p.second;
    auto pct = [&](double q) { double a = 0; for (auto& p : v) { a += p.second; if (a >= q * tot) return p.first; } return v.back().first; };
    std::printf("%-22s n=%9zu  p50 %.4g  p90 %.4g  p99 %.4g  p99.9 %.4g  p99.99 %.4g  max %.4g\n", name, v.size(), pct(.5), pct(.9), pct(.99),
                pct(.999), pct(.9999), v.back().first);
  }
};

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: sky-stats map.exr\n"); return 1; }
  sky::Skymap m; std::string err;
  if (!sky::load_exr_half(argv[1], m, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  // galactic north pole (J2000): RA 192.85948, dec 27.12825
  const double a0 = 192.85948 * M_PI / 180, d0 = 27.12825 * M_PI / 180;
  const Vec3 gp{std::cos(d0) * std::cos(a0), std::cos(d0) * std::sin(a0), std::sin(d0)};
  Acc all, mw, hi;
  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x) {
      const Vec3 c = sky::texel(m, x, y);
      const float l = float(0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z);
      const Vec3 d = sky::uv_to_dir((x + 0.5) / m.w, (y + 0.5) / m.h);
      const float w = float(std::sqrt(std::max(0.0, 1 - d.z * d.z)));
      const double sb = dot(d, gp);  // sin(b)
      all.add(l, w);
      if (std::abs(sb) < std::sin(5.0 * M_PI / 180)) mw.add(l, w);
      if (std::abs(sb) > std::sin(60.0 * M_PI / 180)) hi.add(l, w);
    }
  std::printf("%dx%d\n", m.w, m.h);
  all.report("whole sky"); mw.report("Milky Way |b|<5"); hi.report("high |b|>60");
  all.frac("whole sky"); mw.frac("Milky Way");
}
