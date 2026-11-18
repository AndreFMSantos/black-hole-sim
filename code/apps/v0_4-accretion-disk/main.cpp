// v0.4 — Thin accretion disk around a Schwarzschild black hole.
// Like v0.3, but each ray traced backwards looks for the first crossing of the equatorial plane
// with r_in < r < r_out (opaque disk). At the impact point:
//   T_emit = T_peak * profile(r)       (Shakura-Sunyaev, ISCO = 6M)
//   g      = nu_obs/nu_emit           (gravitational + transverse redshift + Doppler)
//   colour = black body at T_obs = g * T_emit  (I_nu/nu^3 invariant)
// Usage: v0_4-accretion-disk [--out f.png] [--w 1280] [--h 720] [--dist 40] [--fov 40] [--theta 80] [--phi 0]
//        [--ssaa 2] [--rin 6] [--rout 16] [--tpeak 4200] [--exposure 1.6] [--nodoppler] [--nodisk]
//        [--sky map.exr [--sky-exposure 2]]   (real NASA HDR sky; without --sky: procedural stars)
#include <atomic>
#include <bhs/blackbody.hpp>
#include <bhs/camera.hpp>
#include <bhs/disk.hpp>
#include <bhs/observer.hpp>
#include <bhs/exr_loader.hpp>
#include <bhs/png.hpp>
#include <bhs/schwarzschild.hpp>
#include <bhs/skymap.hpp>
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
  std::string out = "v04-disk.png";
  int w = 1280, h = 720, ssaa = 2;
  double dist = 40.0, fov = 40.0, theta = 80.0, phi = 0.0;
  double rin = disk::kIsco, rout = 16.0, tpeak = 4200.0, exposure = 1.6;
  bool nodoppler = false, nodisk = false;
  std::string sky;           // EXR path; empty = procedural sky
  double sky_exposure = 2.0;
  sky::Skymap skymap;
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
    else if (is("--phi")) o.phi = std::atof(argv[++i]);  // camera azimuth (degrees), as in v0.5
    else if (is("--rin")) o.rin = std::atof(argv[++i]);
    else if (is("--rout")) o.rout = std::atof(argv[++i]);
    else if (is("--tpeak")) o.tpeak = std::atof(argv[++i]);
    else if (is("--sky")) o.sky = argv[++i];
    else if (is("--sky-exposure")) o.sky_exposure = std::atof(argv[++i]);
    else if (is("--exposure")) o.exposure = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--nodoppler") == 0) o.nodoppler = true;
    else if (std::strcmp(argv[i], "--nodisk") == 0) o.nodisk = true;
  }
  return o;
}

// Black-body colour LUT (absolute radiance), log-spaced in T: avoids integrating Planck per pixel.
struct BlackbodyLut {
  static constexpr int N = 1024;
  static constexpr double T0 = 500.0, T1 = 100000.0;
  std::vector<Vec3> v;
  BlackbodyLut() : v(N) {
    for (int i = 0; i < N; ++i) v[i] = blackbody::rgb(T0 * std::pow(T1 / T0, double(i) / (N - 1)));
  }
  Vec3 operator()(double T) const {
    if (T <= T0) return {0, 0, 0};
    const double x = std::log(std::min(T, T1) / T0) / std::log(T1 / T0) * (N - 1);
    const int i = std::min(N - 2, int(x));
    const double f = x - i;
    return (1 - f) * v[i] + f * v[i + 1];
  }
};

static double lerp(double a, double b, double t) { return a + (b - a) * t; }
// smooth 3D value noise (integer hash); periodic in azimuth because we sample at (cos, sin).
static double vnoise(double x, double y, double z) {
  const int xi = int(std::floor(x)), yi = int(std::floor(y)), zi = int(std::floor(z));
  const double fx = x - xi, fy = y - yi, fz = z - zi;
  auto sm = [](double t) { return t * t * (3 - 2 * t); };
  auto h = [&](int a, int b, int c) { return stars::unit(stars::hash(uint32_t(a), uint32_t(b), uint32_t(c))); };
  double r = 0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx)
        r += h(xi + dx, yi + dy, zi + dz) * (dx ? sm(fx) : 1 - sm(fx)) * (dy ? sm(fy) : 1 - sm(fy)) * (dz ? sm(fz) : 1 - sm(fz));
  return r;
}

static Vec3 sky_color(const Opts& o, Vec3 v) {
  if (o.skymap.empty()) return stars::sample(v, /*grid=*/false);
  return o.sky_exposure * sky::sample(o.skymap, v);
}

static Vec3 shade(const Opts& o, const BlackbodyLut& lut, Vec3 c_hat, Vec3 d) {
  const double cos_a = dot(d, c_hat);
  const Vec3 perp = d - cos_a * c_hat;
  const double sin_a = norm(perp);
  if (sin_a < 1e-12) return cos_a < 0 ? Vec3{0, 0, 0} : sky_color(o, c_hat);
  const Vec3 t_hat = (1.0 / sin_a) * perp;
  const double alpha = std::atan2(sin_a, cos_a);
  const double b = observer::impact_parameter(o.dist, alpha);

  // phi where the disk plane (z = 0) cuts the ray plane: c_z cos(phi) + t_z sin(phi) = 0
  double phi0 = std::atan2(-c_hat.z, t_hat.z);
  while (phi0 <= 1e-9) phi0 += M_PI;
  while (phi0 > M_PI) phi0 -= M_PI;

  const auto tr = o.nodisk ? schw::DiskTrace{schw::trace_from(o.dist, b, cos_a < 0).fate, 0, {}}
                           : schw::trace_with_disk(o.dist, b, cos_a < 0, phi0, o.rin, o.rout);
  if (tr.disk.hit) {
    const double r = tr.disk.r;
    const Vec3 P = r * (std::cos(tr.disk.phi) * c_hat + std::sin(tr.disk.phi) * t_hat);
    // photon L (travelling emitter -> observer) = -b (c_hat x t_hat); disk axis = z
    const double lz = -b * cross(c_hat, t_hat).z;
    const double g = o.nodoppler ? disk::redshift(r, o.dist, 0.0) : disk::redshift(r, o.dist, lz);
    const double T_obs = g * o.tpeak * disk::temperature(r, o.rin);
    // turbulence: noise in (log r, azimuth) with a slight spiral (differential rotation)
    const double psi = std::atan2(P.y, P.x) - 2.2 * std::log(r / o.rin);
    const double lr = std::log(r);
    const double n = 0.55 * vnoise(3 * std::cos(psi), 3 * std::sin(psi), 9 * lr) +
                     0.45 * vnoise(9 * std::cos(psi), 9 * std::sin(psi), 28 * lr);
    return (o.exposure * lerp(0.55, 1.15, n)) * lut(T_obs);
  }
  if (tr.fate != schw::Fate::Escaped) return {0, 0, 0};
  const Vec3 v = std::cos(tr.phi_end) * c_hat + std::sin(tr.phi_end) * t_hat;
  return sky_color(o, v);
}

int main(int argc, char** argv) {
  Opts o = parse(argc, argv);
  if (!o.sky.empty()) {
    std::string err;
    if (!sky::load_exr_half(o.sky, o.skymap, &err)) { std::fprintf(stderr, "sky: %s\n", err.c_str()); return 1; }
    std::printf("sky: %s (%dx%d)\n", o.sky.c_str(), o.skymap.w, o.skymap.h);
  }
  const BlackbodyLut lut;
  // exposure: black-body radiance at T_peak <-> luminance ~0.5 before tone mapping
  const Vec3 ref = lut(o.tpeak);
  o.exposure *= 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z);

  camera::Orbit cam;
  cam.theta = o.theta * M_PI / 180.0;
  cam.phi = o.phi * M_PI / 180.0;
  cam.dist = o.dist;
  const Vec3 c_hat = cam.c_hat();
  Vec3 fwd, right, up;
  cam.basis(fwd, right, up);
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
              // per-sample tone mapping (Reinhard on the luminance): avoids fireflies in the SSAA
              Vec3 c = shade(o, lut, c_hat, d);
              const double L = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
              acc = acc + (1.0 / (1.0 + L)) * c;
            }
          const double inv = 1.0 / (o.ssaa * o.ssaa);
          const double ch[3] = {acc.x * inv, acc.y * inv, acc.z * inv};
          for (int k = 0; k < 3; ++k) {
            const double v = std::pow(std::min(1.0, std::max(0.0, ch[k])), 1.0 / 2.2);
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
