// v0.7b — Photon rings: map of the images by order (Schwarzschild).
// Each pixel: how many times the ray (traced backwards) crosses the disk plane before escaping or being
// captured. n = 1 direct image, n = 2 the back face (strong lensing), n >= 3 photon rings, each
// ~exp(-pi) = 4.3 % thinner than the previous one. No disk or sky: only the geometry of the images.
// Usage: v0_7b-photon-rings [--out f.png] [--w 1920] [--h 1080] [--fov 20] [--lookx 0] [--theta 0]
//        [--dist 40] [--ssaa 2]
//   --fov  : vertical field in degrees;  --lookx: horizontal offset of the centre, in degrees (to zoom on the edge)
// Prints: pixel count per order and, on the centre line, the measured b boundaries vs ring_boundary().
#include <atomic>
#include <bhs/camera.hpp>
#include <bhs/observer.hpp>
#include <bhs/photon_rings.hpp>
#include <bhs/png.hpp>
#include <bhs/schwarzschild.hpp>
#include <bhs/vec3.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace bhs;

struct Opts {
  std::string out = "v07b-rings.png";
  int w = 1920, h = 1080, ssaa = 2;
  double fov = 20.0, lookx = 0.0, theta = 0.0, dist = 40.0;
};

static Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    if (is("--out")) o.out = argv[++i];
    else if (is("--w")) o.w = std::atoi(argv[++i]);
    else if (is("--h")) o.h = std::atoi(argv[++i]);
    else if (is("--ssaa")) o.ssaa = std::atoi(argv[++i]);
    else if (is("--fov")) o.fov = std::atof(argv[++i]);
    else if (is("--lookx")) o.lookx = std::atof(argv[++i]);
    else if (is("--theta")) o.theta = std::atof(argv[++i]);
    else if (is("--dist")) o.dist = std::atof(argv[++i]);
  }
  return o;
}

struct Pix { int n; bool captured; double b; };

// order of a ray in the direction d = fwd + tx*right + ty*up
static Pix order_of(const Opts& o, const camera::Orbit& cam, double tx, double ty) {
  Vec3 fwd, right, up;
  if (cam.theta == 0.0) { fwd = {0, 0, -1}; right = {1, 0, 0}; up = {0, 1, 0}; }  // exact top-down view
  else cam.basis(fwd, right, up);
  const Vec3 c_hat = cam.c_hat();
  const Vec3 d = normalize(fwd + tx * right + ty * up);
  const double cos_a = dot(d, c_hat);
  const Vec3 perp = d - cos_a * c_hat;
  const double sin_a = norm(perp);
  const double alpha = std::atan2(sin_a, cos_a);
  const double b = observer::impact_parameter(o.dist, alpha);
  if (sin_a < 1e-12) return {0, true, 0};
  const Vec3 t_hat = (1.0 / sin_a) * perp;
  double phi0 = std::atan2(-c_hat.z, t_hat.z);
  while (phi0 <= 1e-9) phi0 += M_PI;
  while (phi0 > M_PI) phi0 -= M_PI;
  const auto c = rings::count_crossings(o.dist, b, phi0, 2e-3, 120.0);
  return {c.n, c.fate != schw::Fate::Escaped, b};
}

static void color_of(int n, bool captured, uint8_t* rgb) {
  // 0 = no crossings (pure shadow/interior only); 1 direct; 2 back face; 3.. photon rings
  static const uint8_t pal[][3] = {{0, 0, 0}, {212, 160, 60}, {70, 150, 215}, {220, 70, 120}, {120, 210, 110},
                                   {240, 240, 90}, {170, 110, 230}, {255, 255, 255}};
  const int i = std::min(n, 7);
  for (int k = 0; k < 3; ++k) rgb[k] = pal[i][k];
  if (captured) for (int k = 0; k < 3; ++k) rgb[k] = uint8_t(rgb[k] * 0.55);  // captured: darker
}

int main(int argc, char** argv) {
  const Opts o = parse(argc, argv);
  camera::Orbit cam;
  cam.theta = o.theta * M_PI / 180.0;
  cam.phi = 0.0;
  cam.dist = o.dist;
  const double tan_half = std::tan(0.5 * o.fov * M_PI / 180.0);
  const double aspect = static_cast<double>(o.w) / o.h;
  const double x0 = std::tan(o.lookx * M_PI / 180.0);

  std::vector<uint8_t> rgb(static_cast<size_t>(o.w) * o.h * 3);
  std::atomic<int> next_row{0};
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t) {
    pool.emplace_back([&] {
      for (int j; (j = next_row.fetch_add(1)) < o.h;)
        for (int i = 0; i < o.w; ++i) {
          double acc[3] = {0, 0, 0};
          for (int sy = 0; sy < o.ssaa; ++sy)
            for (int sx = 0; sx < o.ssaa; ++sx) {
              const double px = (i + (sx + 0.5) / o.ssaa) / o.w, py = (j + (sy + 0.5) / o.ssaa) / o.h;
              const Pix p = order_of(o, cam, x0 + (2 * px - 1) * tan_half * aspect, (1 - 2 * py) * tan_half);
              uint8_t c[3];
              color_of(p.n, p.captured, c);
              for (int k = 0; k < 3; ++k) acc[k] += c[k];
            }
          for (int k = 0; k < 3; ++k) rgb[(static_cast<size_t>(j) * o.w + i) * 3 + k] = uint8_t(acc[k] / (o.ssaa * o.ssaa) + 0.5);
        }
    });
  }
  for (auto& th : pool) th.join();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // boundaries on the centre line (no SSAA), side b > b_c, from right to left
  std::printf("boundaries on the centre line (b_c = %.6f; observer at D = %.0f, so the reference is ring_boundary(D, k)):\n",
              schw::critical_b(), o.dist);
  int prev_n = -1;
  for (int i = o.w - 1; i >= 0; --i) {
    const double tx = x0 + (2 * ((i + 0.5) / o.w) - 1) * tan_half * aspect;
    if (tx < 0) break;  // only the +x semi-axis
    const Pix p = order_of(o, cam, tx, 0.0);
    if (!p.captured && prev_n >= 0 && p.n != prev_n) {
      const int k = std::min(p.n, prev_n);
      std::printf("  n: %d -> %d em b = %.6f   (ring_boundary(D, k=%d) = %.6f)\n", prev_n, p.n, p.b, k,
                  rings::ring_boundary(o.dist, k));
    }
    prev_n = p.captured ? -1 : p.n;
  }
  if (!png::write_file(o.out, png::encode_rgb(o.w, o.h, rgb))) { std::fprintf(stderr, "error writing %s\n", o.out.c_str()); return 1; }
  std::printf("%s: %dx%d ssaa=%d, %.2f s\n", o.out.c_str(), o.w, o.h, o.ssaa, secs);
}
