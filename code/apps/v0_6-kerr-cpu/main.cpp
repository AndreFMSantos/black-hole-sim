// v0.6 — Kerr (rotating) black hole, double-precision CPU reference.
// Like v0.4/0.5, but the geodesics are Kerr's (Boyer-Lindquist, Mino time, Carter constants L and Q)
// integrated with adaptive Dormand-Prince 5(4). Rays stop being planar: the rotation
// drags the frames (Lense-Thirring effect) and the shadow is shifted and flattened.
// Camera: ZAMO observer (zero angular momentum) at (dist, theta, phi). Disk: prograde equatorial
// circular orbits, inner edge at the Kerr ISCO (r_isco(a): 6M at a = 0, 2.32M at a = 0.9).
// Usage: v0_6-kerr-cpu [--a 0.9] [--out f.png] [--w 1280] [--h 720] [--dist 40] [--fov 40] [--theta 80]
//        [--phi 0] [--ssaa 2] [--rin <isco>] [--rout 16] [--tpeak 4200] [--exposure 1.6] [--tol 1e-8]
//        [--observer static|orbit|infall | --beta br,bth,bph]  (observer velocity in the ZAMO frame, in c)
//        [--profile nt|newtonian]  (nt = Novikov-Thorne, default)
//        [--nodoppler] [--nodisk] [--sky map.exr [--sky-exposure 2]]
#include <atomic>
#include <bhs/blackbody.hpp>
#include <bhs/camera.hpp>
#include <bhs/disk.hpp>
#include <bhs/observer.hpp>
#include <bhs/exr_loader.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lorentz.hpp>
#include <bhs/novikov_thorne.hpp>
#include <bhs/png.hpp>
#include <bhs/schwarzschild.hpp>
#include <bhs/skymap.hpp>
#include <bhs/stars.hpp>
#include <bhs/vec3.hpp>
#include <chrono>
#include <memory>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace bhs;

struct Opts {
  std::string out = "v06-kerr.png";
  int w = 1280, h = 720, ssaa = 2;
  double dist = 40.0, fov = 40.0, theta = 80.0, phi = 0.0;
  double a = 0.9, tol = 1e-8;
  std::string observer = "static";         // static | orbit | infall
  Vec3 beta{0, 0, 0};                       // observer velocity in the ZAMO frame (r, theta, phi), in c
  bool nt_profile = true;                  // --profile nt|newtonian
  std::shared_ptr<nt::Profile> nt;
  double rin = -1.0 /* -1: ISCO de Kerr */, rout = 16.0, tpeak = 4200.0, exposure = 1.6;
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
    else if (is("--observer")) o.observer = argv[++i];
    else if (is("--beta")) { double br = 0, bt = 0, bp = 0; std::sscanf(argv[++i], "%lf,%lf,%lf", &br, &bt, &bp); o.beta = {br, bt, bp}; o.observer = "custom"; }
    else if (is("--a")) o.a = std::atof(argv[++i]);
    else if (is("--profile")) o.nt_profile = std::strcmp(argv[++i], "newtonian") != 0;
    else if (is("--tol")) o.tol = std::atof(argv[++i]);
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

static Vec3 shade(const Opts& o, const BlackbodyLut& lut, const camera::Orbit& cam, double x, double y) {
  // viewing direction d = fwd + x right + y up, in ZAMO components (e_r, e_theta, e_phi):
  //   fwd = -e_r, up = -e_theta, right = +e_phi  =>  d ~ (-1, -y, x);  photon n = -d ~ (1, y, -x)
  const double nrm = std::sqrt(1.0 + x * x + y * y);
  // moving observer: n' = (1, y, -x)/nrm in the observer frame -> n in the ZAMO (aberration)
  const auto pz = lorentz::observer_to_zamo({1.0 / nrm, y / nrm, -x / nrm}, o.beta);
  kerr::Options ko;
  ko.rtol = o.tol;
  ko.atol = o.tol * 0.1;
  const kerr::Disk disk_cfg = o.nodisk ? kerr::Disk{} : kerr::Disk{o.rin, o.rout};
  const auto tr = kerr::trace(o.a, o.dist, cam.theta, pz.n.x, pz.n.y, pz.n.z, disk_cfg, ko);
  if (tr.disk_hit) {
    const double r = tr.disk_r;
    // g (ZAMO) x E'/E_zamo (observer motion)
    const double g = o.nodoppler ? kerr::redshift(o.a, r, 0.0, o.dist, cam.theta)
                                 : kerr::redshift(o.a, r, tr.L, o.dist, cam.theta) * pz.dop;
    const double Tprof = (o.nt_profile && o.nt) ? o.nt->temperature(r) : disk::temperature(r, o.rin);
    const double T_obs = g * o.tpeak * Tprof;
    const double psi = tr.disk_phi + cam.phi - 2.2 * std::log(r / o.rin);
    const double lr = std::log(r);
    const double n = 0.55 * vnoise(3 * std::cos(psi), 3 * std::sin(psi), 9 * lr) +
                     0.45 * vnoise(9 * std::cos(psi), 9 * std::sin(psi), 28 * lr);
    return (o.exposure * lerp(0.55, 1.15, n)) * lut(T_obs);
  }
  if (tr.fate != kerr::Fate::Escaped) return {0, 0, 0};
  const double ph = tr.phi_end + cam.phi;  // the metric is axisymmetric: the camera azimuth only adds
  const Vec3 v{std::sin(tr.theta_end) * std::cos(ph), std::sin(tr.theta_end) * std::sin(ph), std::cos(tr.theta_end)};
  return sky_color(o, v);
}

int main(int argc, char** argv) {
  Opts o = parse(argc, argv);
  if (!o.sky.empty()) {
    std::string err;
    if (!sky::load_exr_half(o.sky, o.skymap, &err)) { std::fprintf(stderr, "sky: %s\n", err.c_str()); return 1; }
    std::printf("sky: %s (%dx%d)\n", o.sky.c_str(), o.skymap.w, o.skymap.h);
  }
  if (o.rin < 0) o.rin = kerr::isco(o.a, true);
  if (o.observer == "orbit") o.beta = {0, 0, lorentz::circular_beta(o.a, o.dist, o.theta * M_PI / 180.0)};
  else if (o.observer == "infall") o.beta = {-lorentz::infall_beta(o.dist), 0, 0};
  if (norm(o.beta) >= 1.0) { std::fprintf(stderr, "|beta| >= 1\n"); return 1; }
  std::printf("observer: %s, beta (r,theta,phi) = (%.4f, %.4f, %.4f), |beta| = %.4f\n", o.observer.c_str(), o.beta.x, o.beta.y, o.beta.z, norm(o.beta));
  if (o.nt_profile) o.nt = std::make_shared<nt::Profile>(o.a, std::max(2.0 * o.rout, 100.0));
  const BlackbodyLut lut;
  // exposure: black-body radiance at T_peak <-> luminance ~0.5 before tone mapping
  const Vec3 ref = lut(o.tpeak);
  o.exposure *= 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z);

  camera::Orbit cam;
  cam.theta = o.theta * M_PI / 180.0;
  cam.phi = o.phi * M_PI / 180.0;
  cam.dist = o.dist;
  const double tan_half = std::tan(0.5 * o.fov * M_PI / 180.0);
  const double aspect = static_cast<double>(o.w) / o.h;

  std::printf("Kerr a=%.3f: horizon r+=%.4f, ISCO=%.4f, disk [%.3f, %.1f], theta=%.1f, D=%.1f\n", o.a,
              kerr::horizon(o.a), kerr::isco(o.a), o.rin, o.rout, o.theta, o.dist);
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
              // per-sample tone mapping (Reinhard on the luminance): avoids fireflies in the SSAA
              Vec3 c = shade(o, lut, cam, x, y);
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
