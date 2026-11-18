// v0.6b/v0.7d/v0.7e — Kerr on CUDA (RTX 5080). Same options as v0_6-kerr-cpu for direct comparison.
// See --help. Modes: --shot (PNG), --video (mp4 via ffmpeg), --selftest, and (with none of these) the interactive window.
#include "../v0_5-realtime-gpu/gl_loader.hpp"

#include <algorithm>
#include <bhs/camera.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lorentz.hpp>
#include <bhs/novikov_thorne.hpp>
#include <bhs/png.hpp>
#include <bhs/vec3.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "render.hpp"

using namespace bhs;
using Clock = std::chrono::steady_clock;

struct Cli {
  kcuda::Params p;
  std::string shot, sky = BHS_SKY_DEFAULT, observer = "static";
  bool nosky = false, selftest = false, vsync = false, prec_set = false, samples_set = false, cinematic = false, help = false;
  int runs = 20, autoexit = 0;
  // v0.7e: video
  std::string video, clip = "spin";
  int fps = 60, crf = 18;
  double seconds = 0;  // 0 = clip default
};

static void print_help() {
  std::printf(R"(v0_6b-kerr-cuda — Kerr black hole in CUDA (v0.6b) + cinematic render (v0.7d) + video (v0.7e)

Modes:
  --shot out.png      offline render (prints ms/frame per stage, steps per ray)   [--samples N] [--runs R]
  --video out.mp4     H.264 video piped to ffmpeg   --clip spin|orbit|infall [--fps 60] [--seconds S] [--crf 18]
  --selftest          GPU vs CPU (kerr::trace, NT, boost, accumulation); no window
  (nothing)           interactive window: mouse = orbit, wheel = distance, [ ] spin, D Doppler, G disk, +/- exposure,
                      arrows = sky, P double/float, T reinhard/aces, B bloom, O observer, C cinematic, ESC
                      (--autoexit N exits after N frames and prints the fps)

Scene (same as v0_6-kerr-cpu):  --w --h --a 0.9 --theta 80 --phi 0 --dist 40 --fov 40 --rin <isco> --rout 16 --tpeak 4200
                      --exposure 3 --sky-exposure 2 --nodisk --nodoppler --tol 1e-8 --sky map.exr --nosky
Disk:       --profile nt|newtonian        Novikov-Thorne (2048-point LUT) by default; newtonian = v0.4-v0.6
Observer:   --observer static|orbit|infall   |  --beta br,bth,bph  (velocity in c in the ZAMO frame)
            The disk gets aberration + Doppler (g * E'/E_zamo); the sky only aberration (no colour), as on the CPU.
Precision:  --prec double|float|mixed [--tol]       (float: tol >= 2e-5; interactive: float by default)
Image:      --samples N          jittered sub-pixel samples (R2 per quad, seed --seed); HDR mean, tone mapping afterwards
            --accum hdr|ldr      hdr (default) | ldr = Reinhard PER SAMPLE before averaging (same as the CPU SSAA)
            --clamp L            per-sample luminance clamp (fireflies on the blue side of the Doppler; 0 = off)
            --tonemap reinhard|aces      reinhard by default (comparisons with the CPU); aces = Narkowicz
            --bloom s            HDR bloom (6-level Gaussian pyramid, added before tone mapping); 0 = off
            --beaming s          AESTHETIC, NOT PHYSICAL: g -> g^s only in the disk intensity (s < 1 reduces the Doppler asymmetry)
            --cinematic          AESTHETIC PRESET (not physical): aces + bloom 0.12 + beaming 0.5 + clamp 50. The default look is the physical one.
)");
}

static bool parse_beta(const char* s, double b[3]) { return std::sscanf(s, "%lf,%lf,%lf", &b[0], &b[1], &b[2]) == 3; }

static Cli parse(int argc, char** argv) {
  Cli c;
  auto& p = c.p;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    auto flag = [&](const char* k) { return std::strcmp(argv[i], k) == 0; };
    if (is("--shot")) c.shot = argv[++i];
    else if (is("--video")) c.video = argv[++i];
    else if (is("--clip")) c.clip = argv[++i];
    else if (is("--fps")) c.fps = std::atoi(argv[++i]);
    else if (is("--crf")) c.crf = std::atoi(argv[++i]);
    else if (is("--seconds")) c.seconds = std::atof(argv[++i]);
    else if (is("--w")) p.w = std::atoi(argv[++i]);
    else if (is("--h")) p.h = std::atoi(argv[++i]);
    else if (is("--a")) p.a = std::atof(argv[++i]);
    else if (is("--theta")) p.theta_deg = std::atof(argv[++i]);
    else if (is("--phi")) p.phi_deg = std::atof(argv[++i]);
    else if (is("--dist")) p.dist = std::atof(argv[++i]);
    else if (is("--fov")) p.fov_deg = std::atof(argv[++i]);
    else if (is("--rin")) p.rin = std::atof(argv[++i]);
    else if (is("--rout")) p.rout = std::atof(argv[++i]);
    else if (is("--tpeak")) p.tpeak = std::atof(argv[++i]);
    else if (is("--exposure")) p.exposure = std::atof(argv[++i]);
    else if (is("--sky-exposure")) p.sky_exposure = std::atof(argv[++i]);
    else if (is("--tol")) p.tol = std::atof(argv[++i]);
    else if (is("--phi-atol")) p.phi_atol = std::atof(argv[++i]);
    else if (is("--lmin")) p.l_min = std::atof(argv[++i]);
    else if (is("--rfar")) p.r_far = std::atof(argv[++i]);
    else if (is("--sky")) c.sky = argv[++i];
    else if (is("--runs")) c.runs = std::max(1, std::atoi(argv[++i]));
    else if (is("--autoexit")) c.autoexit = std::atoi(argv[++i]);
    else if (is("--block")) p.block = std::atoi(argv[++i]);
    else if (is("--profile")) p.nt = std::strcmp(argv[++i], "newtonian") != 0;
    else if (is("--observer")) c.observer = argv[++i];
    else if (is("--beta")) { if (!parse_beta(argv[++i], p.beta)) { std::fprintf(stderr, "--beta br,bth,bph\n"); std::exit(1); } c.observer = "custom"; }
    else if (is("--samples")) { p.samples = std::max(1, std::atoi(argv[++i])); c.samples_set = true; }
    else if (is("--seed")) p.seed = uint32_t(std::strtoul(argv[++i], nullptr, 10));
    else if (is("--accum")) p.accum_ldr = std::strcmp(argv[++i], "ldr") == 0;
    else if (is("--clamp")) p.clamp_lum = std::atof(argv[++i]);
    else if (is("--tonemap")) p.tonemap = std::strcmp(argv[++i], "aces") == 0 ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
    else if (is("--bloom")) p.bloom = std::atof(argv[++i]);
    else if (is("--beaming")) p.beaming = std::atof(argv[++i]);
    else if (flag("--cinematic")) { c.cinematic = true; p.tonemap = kcuda::Tonemap::Aces; p.bloom = 0.12; p.beaming = 0.5; p.clamp_lum = 50; }
    else if (is("--prec")) {
      const std::string s = argv[++i];
      c.prec_set = true;
      p.prec = s == "float" ? kcuda::Precision::Float : s == "mixed" ? kcuda::Precision::Mixed : kcuda::Precision::Double;
    }
    else if (flag("--nosky")) c.nosky = true;
    else if (flag("--rawsky")) p.sky_raw = true;
    else if (flag("--nodisk")) p.nodisk = true;
    else if (flag("--nodoppler")) p.nodoppler = true;
    else if (flag("--selftest")) c.selftest = true;
    else if (flag("--vsync")) c.vsync = true;
    else if (flag("--help") || flag("-h")) c.help = true;
  }
  return c;
}

// observer velocity in the ZAMO frame from --observer (same convention as v0_6-kerr-cpu)
static void apply_observer(kcuda::Params& p, const std::string& obs) {
  if (obs == "orbit") { p.beta[0] = 0; p.beta[1] = 0; p.beta[2] = lorentz::circular_beta(p.a, p.dist, p.theta_deg * M_PI / 180.0); }
  else if (obs == "infall") { p.beta[0] = -lorentz::infall_beta(p.dist); p.beta[1] = 0; p.beta[2] = 0; }
  else if (obs == "static") { p.beta[0] = p.beta[1] = p.beta[2] = 0; }
}

static const char* prec_name(kcuda::Precision pr) {
  return pr == kcuda::Precision::Double ? "double" : pr == kcuda::Precision::Float ? "float" : "mixed";
}

// ---------------------------------------------------------------- selftest
// Rotation-axis regression (v0.6b bugs found in use): (1) central column: the ray with x = 0 (L = 0) had the
// final phi wrong by pi (vertical line); (2) camera near the pole (theta = 0.5, 179.5): in float 1 - mu^2
// lost all its digits, the step control did not converge (1e5 steps, "orbiting" rays = black streaks, 40x
// slower). Criteria: no orbiting ray, maximum steps < 2000, and the exit direction of the x = 0 ray within
// 0.02 rad of that of the neighbours x = +-1.3e-3 (~1 pixel at 1080p).
static int selftest_axis(const Cli& c) {
  int bad = 0;
  for (auto pr : {kcuda::Precision::Double, kcuda::Precision::Float})
    for (double th : {80.0, 30.0, 5.0, 0.5, 179.5})
      for (double a : {0.0, 0.9}) {
        kcuda::Params p = c.p;
        p.a = a; p.theta_deg = th; p.prec = pr; p.tol = pr == kcuda::Precision::Double ? 1e-8 : 2e-5;
        p.rin = kerr::isco(a, true);
        p.nodisk = true;  // only exit directions on the sky
        const double tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
        std::vector<double> rays;
        auto add = [&](double x, double y) {
          const double nrm = std::sqrt(1 + x * x + y * y);
          rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
        };
        const int nx = 65, ny = 37;  // odd width: the central pixel has x = 0 exactly
        for (int j = 0; j < ny; ++j)
          for (int i = 0; i < nx; ++i) add((2.0 * (i + 0.5) / nx - 1) * tan_half * 16.0 / 9.0, (1 - 2.0 * (j + 0.5) / ny) * tan_half);
        const size_t grid_n = rays.size() / 3;
        std::vector<double> ys;
        for (int j = 0; j < 25; ++j) {
          const double y = (-1.0 + 2.0 * j / 24.0) * tan_half;
          if (std::abs(y) < 0.15) continue;  // inside the shadow
          ys.push_back(y);
          add(0.0, y); add(1.3e-3, y); add(-1.3e-3, y);
        }
        std::vector<kcuda::RayOut> g;
        kcuda::trace_batch(p, rays, g);
        int orbiting = 0, smax = 0;
        for (size_t i = 0; i < grid_n; ++i) { orbiting += g[i].fate == 2; smax = std::max(smax, g[i].steps); }
        double worst = 0;
        for (size_t k = 0; k < ys.size(); ++k) {
          const auto& r0 = g[grid_n + 3 * k];
          for (int s = 1; s <= 2; ++s) {
            const auto& r1 = g[grid_n + 3 * k + s];
            if (r0.fate != 0 || r1.fate != 0) continue;
            auto dir = [](const kcuda::RayOut& r) {
              const double st = std::sqrt(std::max(0.0, 1 - r.mu_end * r.mu_end));
              return Vec3{st * std::cos(r.phi_end), st * std::sin(r.phi_end), r.mu_end};
            };
            const double dd = norm(dir(r0) - dir(r1));
            if (dd > 0.02 && std::getenv("BHS_DEBUG")) std::printf("   y=%.4f s=%d: x=0 mu=%.5f phi=%.5f steps=%d | nb mu=%.5f phi=%.5f steps=%d\n", ys[k], s, r0.mu_end, r0.phi_end, r0.steps, r1.mu_end, r1.phi_end, r1.steps);
            worst = std::max(worst, dd);
          }
        }
        const bool ok = orbiting == 0 && smax < 2000 && worst < 0.02;
        std::printf("axis %-6s a=%.1f th=%5.1f: orbiting %d, max steps %4d, dir(x=0) vs neighbours max %.2e %s\n",
                    pr == kcuda::Precision::Double ? "double" : "float", a, th, orbiting, smax, worst, ok ? "" : "FAILED");
        bad += !ok;
      }
  return bad;
}
// v0.7d: Novikov-Thorne LUT on the GPU vs nt::Profile (CPU), observer boost vs lorentz::observer_to_zamo,
// deterministic accumulation (fixed seed) and coherence of the resolve (bloom, tone mapping).
static int selftest_v07d(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  // 1. NT: GPU (float table, double interpolation) vs nt::Profile::temperature (double)
  for (double a : {0.0, 0.9, 0.998}) {
    kcuda::Params p = c.p; p.a = a; p.nt = true;
    const nt::Profile prof(a, std::max(2.0 * p.rout, 100.0));
    std::vector<double> r;
    for (int i = 0; i < 400; ++i) r.push_back(kerr::isco(a) * std::pow(300.0 / kerr::isco(a), i / 399.0));  // r_isco ... 300 (includes the tail r > 100)
    r.push_back(0.5 * kerr::isco(a));  // inside the ISCO: T = 0
    std::vector<double> gpu;
    kcuda::nt_lookup(p, r, gpu);
    double worst = 0;
    for (size_t i = 0; i < r.size(); ++i) worst = std::max(worst, std::abs(gpu[i] - prof.temperature(r[i])));
    char buf[96]; std::snprintf(buf, sizeof buf, "NT LUT a=%.3f: max |T_gpu - T_cpu|", a);
    check(worst < 2e-6, buf, worst, 2e-6);
  }
  // 2. boost: GPU vs lorentz::observer_to_zamo
  {
    std::vector<double> np;
    std::vector<Vec3> bs = {{0, 0, 0}, {0, 0, 0.3}, {-0.447, 0, 0}, {0.2, -0.3, 0.5}, {0, 0, 0.9}};
    double worst = 0;
    for (const auto& b : bs) {
      np.clear();
      std::vector<Vec3> dirs;
      for (int i = 0; i < 200; ++i) {
        const Vec3 d = normalize(Vec3{std::cos(0.37 * i + 0.1), std::sin(1.3 * i) + 0.2, std::sin(0.71 * i + 0.4)});
        dirs.push_back(d); np.insert(np.end(), {d.x, d.y, d.z});
      }
      double bb[3] = {b.x, b.y, b.z};
      std::vector<double> out;
      kcuda::boost_batch(np, bb, out);
      for (size_t i = 0; i < dirs.size(); ++i) {
        const auto ph = lorentz::observer_to_zamo(dirs[i], b);
        worst = std::max({worst, std::abs(out[4 * i] - ph.n.x), std::abs(out[4 * i + 1] - ph.n.y), std::abs(out[4 * i + 2] - ph.n.z), std::abs(out[4 * i + 3] - ph.dop)});
      }
    }
    check(worst < 1e-12, "boost GPU vs lorentz::observer_to_zamo (5 beta, 200 directions)", worst, 1e-12);
  }
  // 3. deterministic accumulation + resolve coherence (small image, float)
  {
    kcuda::Params p = c.p;
    p.w = 192; p.h = 108; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.a = 0.9; p.theta_deg = 80; p.samples = 6; p.seed = 7;
    p.use_sky = c.p.use_sky;
    std::vector<uint8_t> a1, a2, a3, a4;
    kcuda::render(p); kcuda::download_rgb(a1);
    kcuda::render(p); kcuda::download_rgb(a2);
    size_t diff = 0;
    for (size_t i = 0; i < a1.size(); ++i) diff += a1[i] != a2[i];
    check(diff == 0, "accumulation with a fixed seed: differing bytes between 2 renders", double(diff), 0.5);
    kcuda::Params q = p; q.seed = 8;
    kcuda::render(q); kcuda::download_rgb(a3);
    size_t d2 = 0;
    for (size_t i = 0; i < a1.size(); ++i) d2 += a1[i] != a3[i];
    check(d2 > 0, "another seed changes the jitter (bytes differ)", double(d2), 1.0 - 0.5);
    // N samples with sample 0 centred: samples=1 is the centre sample (does not depend on the seed)
    kcuda::Params s1 = p; s1.samples = 1;
    kcuda::render(s1); kcuda::download_rgb(a3);
    s1.seed = 99; kcuda::render(s1); kcuda::download_rgb(a4);
    size_t d3 = 0;
    for (size_t i = 0; i < a3.size(); ++i) d3 += a3[i] != a4[i];
    check(d3 == 0, "samples=1 (pixel centre) independent of the seed", double(d3), 0.5);
    // bloom adds light: the frame mean rises; aces tone mapping != reinhard
    kcuda::Params b0 = p; b0.samples = 1;
    kcuda::Params b1 = b0; b1.bloom = 0.3;
    kcuda::render(b0); kcuda::download_rgb(a1);
    kcuda::render(b1); kcuda::download_rgb(a2);
    double m0 = 0, m1 = 0;
    for (size_t i = 0; i < a1.size(); ++i) { m0 += a1[i]; m1 += a2[i]; }
    check(m1 > m0, "bloom increases the mean luminance of the frame", (m1 - m0) / a1.size(), 0.0);
    kcuda::Params b2 = b0; b2.tonemap = kcuda::Tonemap::Aces;
    kcuda::render(b2); kcuda::download_rgb(a3);
    size_t d4 = 0;
    for (size_t i = 0; i < a1.size(); ++i) d4 += a1[i] != a3[i];
    check(d4 > 0, "aces tone mapping differs from reinhard", double(d4), 0.5);
  }
  return bad;
}

static int selftest(const Cli& c) {
  int bad = 0;
  for (double a : {0.0, 0.9, 0.998})
    for (double th : {80.0, 86.0}) {
      kcuda::Params p = c.p;
      p.tol = 1e-10;  // equivalence: tight tolerance (the difference to the CPU scales with it, see the notes)
      p.a = a; p.theta_deg = th;
      p.rin = kerr::isco(a, true);
      const int nx = 64, ny = 36;
      const double tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0), aspect = 16.0 / 9.0;
      std::vector<double> rays;
      std::vector<std::pair<double, double>> xy;
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
          const double x = (2 * (i + 0.5) / nx - 1) * tan_half * aspect, y = (1 - 2 * (j + 0.5) / ny) * tan_half;
          const double nrm = std::sqrt(1 + x * x + y * y);
          rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
        }
      std::vector<kcuda::RayOut> g;
      kcuda::trace_batch(p, rays, g);
      kerr::Options ko; ko.rtol = p.tol; ko.atol = p.tol * 0.1;
      int fate_bad = 0, n = nx * ny, both_esc = 0, hit_both = 0;
      std::vector<double> dang, drad;
      for (int i = 0; i < n; ++i) {
        const auto t = kerr::trace(a, p.dist, th * M_PI / 180.0, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], kerr::Disk{p.rin, p.rout}, ko);
        const int cf = t.fate == kerr::Fate::Escaped ? 0 : t.fate == kerr::Fate::Captured ? 1 : 2;
        if (cf != g[i].fate || int(t.disk_hit) != g[i].disk_hit) { ++fate_bad; continue; }
        if (cf == 0) {
          ++both_esc;
          const double dth = std::acos(g[i].mu_end) - t.theta_end;
          double dph = g[i].phi_end - t.phi_end;
          dang.push_back(std::hypot(dth, dph));
        }
        if (t.disk_hit) { ++hit_both; drad.push_back(std::abs(g[i].disk_r - t.disk_r)); }
      }
      std::sort(dang.begin(), dang.end()); std::sort(drad.begin(), drad.end());
      auto q = [](const std::vector<double>& v, double f) { return v.empty() ? 0.0 : v[size_t(f * (v.size() - 1))]; };
      std::printf("a=%.3f th=%2.0f: %d/%d rays with a different fate | escape: n=%d dang p50=%.2e p99=%.2e max=%.2e | disk: n=%d dr p99=%.2e max=%.2e\n",
                  a, th, fate_bad, n, both_esc, q(dang, .5), q(dang, .99), q(dang, 1), hit_both, q(drad, .99), q(drad, 1));
      if (fate_bad > n / 200 || q(dang, .99) > 1e-6 || q(drad, .99) > 1e-6) ++bad;
    }
  bad += selftest_axis(c);
  std::printf("v0.7d:\n");
  bad += selftest_v07d(c);
  std::printf(bad ? "selftest FAILED (%d configurations)\n" : "selftest OK\n", bad);
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------- shot
static int shot(Cli& c) {
  auto& p = c.p;
  if (p.rin < 0) p.rin = kerr::isco(p.a, true);
  apply_observer(p, c.observer);
  kcuda::Stats st;
  float first = kcuda::render(p, &st);
  for (int i = 0; i < 2; ++i) kcuda::render(p);  // warm-up
  double sum = 0, tr = 0, rs = 0, bl = 0;
  for (int i = 0; i < c.runs; ++i) {
    sum += kcuda::render(p);
    const auto t = kcuda::last_timing();
    tr += t.trace_ms; rs += t.resolve_ms; bl += t.bloom_ms;  // trace_ms = last sample only
  }
  std::printf("Kerr a=%.3f theta=%.1f D=%.1f %dx%d tol=%g prec=%s profile=%s observer=%s%s: first %.1f ms; mean of %d: %.3f ms/frame (%.1f fps)\n",
              p.a, p.theta_deg, p.dist, p.w, p.h, p.tol, prec_name(p.prec), p.nt ? "nt" : "newtonian", c.observer.c_str(),
              std::abs(p.beta[0]) + std::abs(p.beta[1]) + std::abs(p.beta[2]) > 0 ? "" : "", first, c.runs, sum / c.runs, 1000.0 * c.runs / sum);
  std::printf("  stages (mean): sample %.3f ms x %d samples | resolve (mean+tone mapping) %.3f ms | bloom %.3f ms | tone mapping %s, bloom %g, beaming %g%s\n",
              tr / c.runs, p.samples, rs / c.runs, bl / c.runs, p.tonemap == kcuda::Tonemap::Aces ? "aces" : "reinhard", p.bloom, p.beaming,
              c.cinematic ? " [cinematic: AESTHETIC, not physical]" : "");
  std::printf("steps per ray: mean %.1f, max %d; warp efficiency (sum/sum of max*32) %.2f\n", st.steps_mean, st.steps_max, st.warp_eff);
  std::vector<uint8_t> rgb;
  kcuda::download_rgb(rgb);
  if (!png::write_file(c.shot, png::encode_rgb(p.w, p.h, rgb))) { std::fprintf(stderr, "error writing %s\n", c.shot.c_str()); return 1; }
  std::printf("wrote %s\n", c.shot.c_str());
  return 0;
}

// ---------------------------------------------------------------- video (v0.7e)
static double smooth(double t) { t = std::min(1.0, std::max(0.0, t)); return t * t * (3.0 - 2.0 * t); }

// parameters of frame i of n for each clip
static void clip_frame(const Cli& c, int i, int n, kcuda::Params& p, std::string& obs) {
  const double t = n > 1 ? double(i) / (n - 1) : 0.0;
  obs = "static";
  if (c.clip == "spin") {  // A: spin sweep 0 -> 0.998, fixed camera (theta 80, D 40)
    p.a = 0.998 * smooth(t); p.theta_deg = 80; p.dist = 40; p.phi_deg = 0;
  } else if (c.clip == "orbit") {  // B: camera orbit: theta 100 -> 60 (from below, from the side, from above) and phi 0 -> 360, a = 0.9
    p.a = 0.9; p.dist = 40; p.theta_deg = 100.0 - 40.0 * smooth(t); p.phi_deg = 360.0 * t;
  } else if (c.clip == "infall") {  // C: radial infall from r = 40 to r = 4, uniform PROPER time: r = (r0^1.5 - 1.5 sqrt(2) tau)^(2/3)
    const double r0 = 40.0, r1 = 4.0;
    const double tau_total = (r0 * std::sqrt(r0) - r1 * std::sqrt(r1)) / (1.5 * std::sqrt(2.0));
    p.a = 0.9; p.theta_deg = 75; p.phi_deg = 0;
    p.dist = std::pow(std::pow(r0, 1.5) - 1.5 * std::sqrt(2.0) * tau_total * t, 2.0 / 3.0);
    obs = "infall";
  }
  apply_observer(p, obs);
}

static int video(Cli& c) {
  auto& p = c.p;
  const double secs = c.seconds > 0 ? c.seconds : (c.clip == "infall" ? 12.0 : 10.0);
  const int n = int(std::lround(secs * c.fps));
  char cmd[1024];
  std::snprintf(cmd, sizeof cmd,
                "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -c:v libx264 -preset medium -crf %d "
                "-pix_fmt yuv420p -movflags +faststart \"%s\"",
                p.w, p.h, c.fps, c.crf, c.video.c_str());
  FILE* ff = popen(cmd, "w");
  if (!ff) { std::fprintf(stderr, "could not launch ffmpeg\n"); return 1; }
  std::printf("video '%s': clip %s, %d frames (%.1f s at %d fps), %dx%d, %s, %d samples/frame, profile %s%s\n", c.video.c_str(), c.clip.c_str(), n, secs,
              c.fps, p.w, p.h, prec_name(p.prec), p.samples, p.nt ? "nt" : "newtonian", c.cinematic ? ", CINEMATIC (aesthetic)" : "");
  const auto t0 = Clock::now();
  std::vector<uint8_t> rgb;
  double gpu_ms = 0;
  for (int i = 0; i < n; ++i) {
    std::string obs;
    clip_frame(c, i, n, p, obs);
    gpu_ms += kcuda::render(p);
    kcuda::download_rgb(rgb);
    if (std::fwrite(rgb.data(), 1, rgb.size(), ff) != rgb.size()) { std::fprintf(stderr, "write to ffmpeg failed (frame %d)\n", i); pclose(ff); return 1; }
    if (i % 60 == 0) { std::printf("  frame %d/%d (a=%.3f theta=%.1f phi=%.0f D=%.2f |beta|=%.3f)\n", i, n, p.a, p.theta_deg, p.phi_deg, p.dist, std::abs(p.beta[0]) + std::abs(p.beta[2])); std::fflush(stdout); }
  }
  const int rc = pclose(ff);
  const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("render+encode: %.1f s de relogio (%.1f ms/frame; GPU %.1f ms/frame), ffmpeg rc=%d\n", wall, 1000.0 * wall / n, gpu_ms / n, rc);
  return rc == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- interactive
struct App {
  Cli c;
  camera::Orbit cam;
  bool dragging = false;
  double lastx = 0, lasty = 0;
};
static void on_scroll(GLFWwindow* w, double, double dy) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  a->cam.dist *= std::exp(-0.08 * dy);
  a->cam.clamp();
}
static void on_button(GLFWwindow* w, int b, int act, int) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (b == GLFW_MOUSE_BUTTON_LEFT) { a->dragging = act == GLFW_PRESS; glfwGetCursorPos(w, &a->lastx, &a->lasty); }
}
static void on_cursor(GLFWwindow* w, double x, double y) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (a->dragging) { a->cam.phi -= (x - a->lastx) * 0.005; a->cam.theta -= (y - a->lasty) * 0.005; a->cam.clamp(); }
  a->lastx = x; a->lasty = y;
}
static void on_key(GLFWwindow* w, int key, int, int act, int) {
  if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  auto& p = a->c.p;
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
  else if (key == GLFW_KEY_D && act == GLFW_PRESS) p.nodoppler = !p.nodoppler;
  else if (key == GLFW_KEY_G && act == GLFW_PRESS) p.nodisk = !p.nodisk;
  else if (key == GLFW_KEY_P && act == GLFW_PRESS) {
    p.prec = p.prec == kcuda::Precision::Double ? kcuda::Precision::Float : kcuda::Precision::Double;
    p.tol = p.prec == kcuda::Precision::Double ? 1e-8 : 1e-5;
  }
  else if (key == GLFW_KEY_LEFT_BRACKET) p.a = std::max(0.0, p.a - 0.01);
  else if (key == GLFW_KEY_RIGHT_BRACKET) p.a = std::min(0.998, p.a + 0.01);
  else if (key == GLFW_KEY_UP) p.sky_exposure *= 1.1;
  else if (key == GLFW_KEY_DOWN) p.sky_exposure /= 1.1;
  else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) p.exposure *= 1.1;
  else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) p.exposure /= 1.1;
  else if (key == GLFW_KEY_T && act == GLFW_PRESS) p.tonemap = p.tonemap == kcuda::Tonemap::Reinhard ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
  else if (key == GLFW_KEY_B && act == GLFW_PRESS) {  // bloom: 0 -> 0.06 -> 0.12 -> 0.25 -> 0.5 -> 0
    const double lv[] = {0, 0.06, 0.12, 0.25, 0.5};
    int k = 0;
    for (int i = 0; i < 5; ++i) if (std::abs(p.bloom - lv[i]) < 1e-9) k = i;
    p.bloom = lv[(k + 1) % 5];
  }
  else if (key == GLFW_KEY_O && act == GLFW_PRESS) a->c.observer = a->c.observer == "static" ? "orbit" : a->c.observer == "orbit" ? "infall" : "static";
  else if (key == GLFW_KEY_C && act == GLFW_PRESS) {  // cinematic (AESTHETIC) preset on/off
    a->c.cinematic = !a->c.cinematic;
    if (a->c.cinematic) { p.tonemap = kcuda::Tonemap::Aces; p.bloom = 0.12; p.beaming = 0.5; p.clamp_lum = 50; }
    else { p.tonemap = kcuda::Tonemap::Reinhard; p.bloom = 0; p.beaming = 1.0; p.clamp_lum = 0; }
  }
}

static int interactive(Cli& c) {
  App a;
  a.c = c;
  if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* win = glfwCreateWindow(c.p.w > 1920 ? 1280 : c.p.w, c.p.w > 1920 ? 720 : c.p.h, "black-hole-sim v0.7d (CUDA)", nullptr, nullptr);
  if (!win) { std::fprintf(stderr, "could not create the OpenGL window\n"); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(c.vsync ? 1 : 0);
  if (!gl::load()) return 1;
  std::printf("GL %s | %s\n", gl::GetString(GL_VERSION), gl::GetString(GL_RENDERER));
  a.cam.theta = c.p.theta_deg * M_PI / 180.0; a.cam.phi = c.p.phi_deg * M_PI / 180.0; a.cam.dist = c.p.dist; a.cam.clamp();
  a.c.p.rin = -1.0;  // ISCO follows the spin

  glfwSetWindowUserPointer(win, &a);
  glfwSetScrollCallback(win, on_scroll);
  glfwSetMouseButtonCallback(win, on_button);
  glfwSetCursorPosCallback(win, on_cursor);
  glfwSetKeyCallback(win, on_key);

  GLuint tex = 0, fbo = 0;
  gl::CreateFramebuffers(1, &fbo);
  int w = 0, h = 0;
  auto make_tex = [&](int nw, int nh) {
    if (tex) { kcuda::gl_unregister(); gl::DeleteTextures(1, &tex); }
    w = nw; h = nh;
    gl::CreateTextures(GL_TEXTURE_2D, 1, &tex);
    gl::TextureStorage2D(tex, 1, GL_RGBA8, w, h);
    gl::NamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, tex, 0);
    std::string err;
    if (!kcuda::gl_register(tex, w, h, err)) { std::fprintf(stderr, "CUDA-GL interop failed: %s\n", err.c_str()); std::exit(1); }
  };
  auto last = Clock::now(), t_start = last;
  int frames = 0, total = 0, k = 0;
  const int max_samples = c.samples_set ? c.p.samples : 128;  // progressive samples when the scene is still
  std::vector<double> key, key_prev;
  float ms = 0;
  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    int fw, fh;
    glfwGetFramebufferSize(win, &fw, &fh);
    if (fw <= 0 || fh <= 0) { glfwWaitEvents(); continue; }
    if (fw != w || fh != h) make_tex(fw, fh);
    auto& p = a.c.p;
    p.w = w; p.h = h; p.dist = a.cam.dist; p.theta_deg = a.cam.theta * 180.0 / M_PI; p.phi_deg = a.cam.phi * 180.0 / M_PI;
    apply_observer(p, a.c.observer);
    // everything that affects the image; any change (mouse, keys, spin) restarts the accumulation
    key = {double(p.w), double(p.h), p.a, p.dist, p.theta_deg, p.phi_deg, p.exposure, p.sky_exposure, double(p.nodisk), double(p.nodoppler),
           double(int(p.prec)), double(p.nt), p.beta[0], p.beta[1], p.beta[2], double(int(p.tonemap)), p.bloom, p.beaming, p.clamp_lum};
    const bool changed = key != key_prev;
    key_prev = key;
    if (changed) { kcuda::accum_reset(); k = 0; }
    const bool converged = k >= max_samples;
    if (!converged) { ms = kcuda::accum_sample(p, k++); kcuda::resolve(p); }
    else glfwWaitEventsTimeout(0.05);  // converged image: do not spend GPU
    kcuda::gl_copy();
    gl::BlitNamedFramebuffer(fbo, 0, 0, 0, w, h, 0, h, w, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glfwSwapBuffers(win);
    ++frames; ++total;
    const double dt = std::chrono::duration<double>(Clock::now() - last).count();
    if (dt >= 0.5) {
      char t[260];
      std::snprintf(t, sizeof t, "black-hole-sim v0.7d CUDA%s | %.0f fps | sample %.2f ms | N=%d | a=%.3f D=%.0f theta=%.0f %s %s %s bloom=%.2f beaming=%.2f obs=%s%s%s",
                    a.c.cinematic ? " [CINEMATIC: aesthetic, not physical]" : "", frames / dt, ms, std::min(k, max_samples), p.a, p.dist, p.theta_deg, prec_name(p.prec),
                    p.nt ? "NT" : "newt", p.tonemap == kcuda::Tonemap::Aces ? "aces" : "reinhard", p.bloom, p.beaming, a.c.observer.c_str(),
                    p.nodoppler ? " | no Doppler" : "", p.nodisk ? " | no disk" : "");
      glfwSetWindowTitle(win, t);
      if (c.autoexit) std::printf("%s\n", t);
      frames = 0; last = Clock::now();
    }
    if (c.autoexit && total >= c.autoexit) {
      const double s = std::chrono::duration<double>(Clock::now() - t_start).count();
      std::printf("autoexit: %d frames in %.2f s = %.1f fps mean (%dx%d)\n", total, s, total / s, w, h);
      break;
    }
  }
  kcuda::gl_unregister();
  glfwTerminate();
  return 0;
}

int main(int argc, char** argv) {
  Cli c = parse(argc, argv);
  if (c.help) { print_help(); return 0; }
  const bool is_interactive = c.shot.empty() && !c.selftest && c.video.empty();
  if (is_interactive && !c.prec_set) { c.p.prec = kcuda::Precision::Float; c.p.tol = 2e-5; }  // real time: float (see the notes)
  if (c.nosky) c.sky.clear();
  else {
    FILE* f = std::fopen(c.sky.c_str(), "rb");
    if (f) std::fclose(f); else { std::printf("sky not found (%s): black background (code/tools/fetch_sky.sh)\n", c.sky.c_str()); c.sky.clear(); }
  }
  c.p.use_sky = !c.sky.empty();
  if (c.p.prec == kcuda::Precision::Float && c.p.tol < 1e-6) {
    std::printf("warning: tol %g is unreachable in float; using 2e-5\n", c.p.tol);
    c.p.tol = 2e-5;
  }
  std::string err;
  if (!kcuda::init(c.sky, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
  if (c.selftest) return selftest(c);
  if (!c.video.empty()) { if (!c.prec_set) { c.p.prec = kcuda::Precision::Float; c.p.tol = 2e-5; } if (c.p.rin < 0) c.p.rin = -1; return video(c); }
  if (!c.shot.empty()) return shot(c);
  return interactive(c);
}
