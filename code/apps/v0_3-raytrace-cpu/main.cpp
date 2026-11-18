// v0.3 — CPU ray tracer: the shadow of a Schwarzschild black hole over a star field.
// For each pixel: direction -> impact parameter b -> null geodesic traced BACKWARDS to
// infinity (trace_from) -> celestial direction the light came from -> background colour (stars + grid).
// Usage: v0_3-raytrace-cpu [--out f.png] [--w 1280] [--h 720] [--dist 30] [--fov 40]
//                       [--theta 90] [--ssaa 2] [--nobh]   (--nobh: no black hole, for comparison)
#include <atomic>
#include <bhs/observer.hpp>
#include <bhs/png.hpp>
#include <bhs/schwarzschild.hpp>
#include <bhs/stars.hpp>
#include <bhs/vec3.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace bhs;

struct Opts {
  std::string out = "v03-shadow.png";
  int w = 1280, h = 720, ssaa = 2;
  double dist = 30.0, fov = 40.0, theta = 90.0, phi = 0.0;  // phi: camera azimuth (degrees), as in v0.4+; 0 = v0.3 behaviour
  bool nobh = false;
};

static Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    if (is("--out")) o.out = argv[++i];
    else if (is("--w")) o.w = std::atoi(argv[++i]);
    else if (is("--h")) o.h = std::atoi(argv[++i]);
    else if (is("--ssaa")) o.ssaa = std::atoi(argv[++i]);
    else if (is("--dist")) o.dist = std::atof(argv[++i]);
    else if (is("--fov")) o.fov = std::atof(argv[++i]);
    else if (is("--theta")) o.theta = std::atof(argv[++i]);
    else if (is("--phi")) o.phi = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--nobh") == 0) o.nobh = true;
  }
  return o;
}

// Linear colour of a ray with direction d (unit, in world coordinates).
static Vec3 shade(const Opts& o, Vec3 c_hat, Vec3 d) {
  if (o.nobh) return stars::sample(d);
  const Vec3 f = -1.0 * c_hat;  // direction to the hole
  const double cos_a = dot(d, c_hat);       // >0: moving away (outward radial component)
  const Vec3 perp = d - cos_a * c_hat;       // transverse component
  const double sin_a = norm(perp);
  if (sin_a < 1e-12) {  // radial: inward = captured, outward = sees the opposite direction
    return cos_a < 0 ? Vec3{0, 0, 0} : stars::sample(c_hat);
  }
  const Vec3 t_hat = (1.0 / sin_a) * perp;
  const double alpha = std::atan2(sin_a, cos_a);  // angle of the outward radial direction
  const double b = observer::impact_parameter(o.dist, alpha);
  const auto r = schw::trace_from(o.dist, b, /*inward=*/cos_a < 0);
  if (r.fate != schw::Fate::Escaped) return {0, 0, 0};  // horizon (or endless orbit)
  // asymptotic position at angle phi_end from c_hat, in the (c_hat, t_hat) plane
  const Vec3 v = std::cos(r.phi_end) * c_hat + std::sin(r.phi_end) * t_hat;
  (void)f;
  return stars::sample(v);
}

int main(int argc, char** argv) {
  const Opts o = parse(argc, argv);
  const double th = o.theta * M_PI / 180.0;
  const double ph = o.phi * M_PI / 180.0;
  const Vec3 c_hat{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};  // observer (th = 90, phi = 0: on the equator, at +x)
  const Vec3 fwd = -1.0 * c_hat;
  const Vec3 up0{0, 0, 1};
  Vec3 right = cross(fwd, up0);
  if (norm(right) < 1e-9) right = Vec3{0, 1, 0};
  right = normalize(right);
  const Vec3 up = cross(right, fwd);
  const double tan_half = std::tan(0.5 * o.fov * M_PI / 180.0);
  const double aspect = static_cast<double>(o.w) / o.h;

  std::vector<uint8_t> rgb(static_cast<size_t>(o.w) * o.h * 3);
  std::atomic<int> next_row{0};
  const auto t0 = std::chrono::steady_clock::now();
  const unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < nthreads; ++t) {
    pool.emplace_back([&] {
      for (int j; (j = next_row.fetch_add(1)) < o.h;) {
        for (int i = 0; i < o.w; ++i) {
          Vec3 acc{0, 0, 0};
          for (int sy = 0; sy < o.ssaa; ++sy)
            for (int sx = 0; sx < o.ssaa; ++sx) {
              const double px = (i + (sx + 0.5) / o.ssaa) / o.w, py = (j + (sy + 0.5) / o.ssaa) / o.h;
              const double x = (2 * px - 1) * tan_half * aspect, y = (1 - 2 * py) * tan_half;
              const Vec3 d = normalize(fwd + x * right + y * up);
              acc = acc + shade(o, c_hat, d);
            }
          const double inv = 1.0 / (o.ssaa * o.ssaa);
          const double ch[3] = {acc.x * inv, acc.y * inv, acc.z * inv};
          for (int k = 0; k < 3; ++k) {
            const double v = std::pow(std::min(1.0, std::max(0.0, ch[k])), 1.0 / 2.2);  // gamma
            rgb[(static_cast<size_t>(j) * o.w + i) * 3 + k] = static_cast<uint8_t>(v * 255.0 + 0.5);
          }
        }
      }
    });
  }
  for (auto& th2 : pool) th2.join();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  if (!png::write_file(o.out, png::encode_rgb(o.w, o.h, rgb))) {
    std::fprintf(stderr, "error writing %s\n", o.out.c_str());
    return 1;
  }
  std::printf("%s: %dx%d ssaa=%d, %u threads, %.2f s\n", o.out.c_str(), o.w, o.h, o.ssaa, nthreads, secs);
}
