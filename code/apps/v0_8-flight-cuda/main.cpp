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

#include "denoise.hpp"
#include "flight.hpp"
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
  // v0.8b: voo
  std::string mode = "orbit", start = "orbit";
  double timescale = 5.0, move_scale = 1.0;
  bool hud = true, hud_offline = false;
  std::string hdr_out;  // --hdr out.pfm: HDR average (after the denoiser, before bloom/tone mapping)
  bool selftest_denoise = false, denoise_kernels = false;  // hud: interactive window (default on); hud_offline: --hud in --shot/--video
};

static void print_help() {
  std::printf(R"(v0_6b-kerr-cuda — Kerr black hole in CUDA (v0.6b) + cinematic render (v0.7d) + video (v0.7e)

Modes:
  --shot out.png      offline render (prints ms/frame per stage, steps per ray)   [--samples N] [--runs R]
  --video out.mp4     H.264 video piped to ffmpeg   --clip spin|orbit|infall|flight [--fps 60] [--seconds S] [--crf 18]
  --selftest          GPU vs CPU (kerr::trace, NT, boost, accumulation); no window
  (nothing)           interactive window: mouse = orbit, wheel = distance, [ ] spin, D Doppler, G disk, +/- exposure,
                      arrows = sky, P double/float, T reinhard/aces, B bloom, O observer, C cinematic, ESC
                      (--autoexit N exits after N frames and prints the fps)

Scene (same as v0_6-kerr-cpu):  --w --h --a 0.9 --theta 80 --phi 0 --dist 40 --fov 40 --rin <isco> --rout 16 --tpeak 4200
                      --exposure 3 --sky-exposure 2 --nodisk --nodoppler --tol 1e-8 --sky map.exr --nosky
Disk:       --profile nt|newtonian        Novikov-Thorne (2048-point LUT) by default; newtonian = v0.4-v0.6
Flight (v0.8b): --mode orbit|flight   --start orbit|zamo|infall   --timescale 5 (M of proper time per real second)   --move-scale 0.5   --nohud
            Flight: mouse (drag) look, Q/E roll, W/S forward/back, A/D sideways, Space/Shift up/down, Ctrl brake, +/- acceleration,
            , . time scale, Z pause, H HUD, Backspace reset, F orbit/flight mode (the mouse wheel only works in orbit mode)
Observer:   --observer static|orbit|infall   |  --beta br,bth,bph  (velocity in c in the ZAMO frame)
            The disk gets aberration + Doppler (g * E'/E_zamo); the sky only aberration (no colour), as on the CPU.
Precision:  --prec double|float|mixed [--tol]       (float: tol >= 2e-5; interactive: float by default)
Image:      --samples N          jittered sub-pixel samples (R2 per quad, seed --seed); HDR mean, tone mapping afterwards
            --accum hdr|ldr      hdr (default) | ldr = Reinhard PER SAMPLE before averaging (same as the CPU SSAA)
            --clamp L            per-sample luminance clamp (fireflies on the blue side of the Doppler; 0 = off)
            --tonemap reinhard|aces      reinhard by default (comparisons with the CPU); aces = Narkowicz
            --bloom s            HDR bloom (6-level Gaussian pyramid, added before tone mapping); 0 = off
            --beaming s          AESTHETIC, NOT PHYSICAL: g -> g^s only in the disk intensity (s < 1 reduces the Doppler asymmetry)
            --glare s [--glare-thr 2]   star-shaped glare (6 points) above the threshold, added in HDR      (aesthetic)
            --grain s            luminance-dependent film grain, AFTER tone mapping (fixed seed; 0 = off)
            --vignette s         soft vignette (0..1)                                           (aesthetic)
            --chroma s           radial chromatic aberration (fraction of the radius at the corner)           (aesthetic)
            --denoise none|optix  OptiX denoiser (HDR model, tensor cores according to NVIDIA) between the HDR mean and the bloom; only if compiled
                                 with BHS_OPTIX (code/tools/fetch_optix.sh); default none; key N in the window   [--denoise-albedo: albedo guide]
            --sample-offset K    index of the 1st sample (K > 0: all jittered); --hdr out.pfm writes the HDR mean
            --cinematic          AESTHETIC PRESET (not physical): aces + bloom 0.12 + beaming 0.5 + clamp 50 + glare 0.10 + grain 0.025
                                 + vignette 0.30 + chroma 0.003 (flags after the preset override it). The default look is the physical one.
)");
}

// AESTHETIC preset (not physical): aces + bloom + beaming + clamp (v0.7d) and polish (v0.8d): glare, grain, vignette, chroma
static void apply_cinematic(kcuda::Params& p, bool on) {
  p.tonemap = on ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
  p.bloom = on ? 0.12 : 0; p.beaming = on ? 0.5 : 1.0; p.clamp_lum = on ? 50 : 0;
  p.glare = on ? 0.10 : 0; p.grain = on ? 0.025 : 0; p.vignette = on ? 0.30 : 0; p.chroma = on ? 0.003 : 0;
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
    else if (is("--mode")) c.mode = argv[++i];
    else if (is("--start")) c.start = argv[++i];
    else if (is("--timescale")) c.timescale = std::atof(argv[++i]);
    else if (is("--move-scale")) c.move_scale = std::min(1.0, std::max(0.1, std::atof(argv[++i])));
    else if (is("--denoise")) {
      const std::string d = argv[++i];
      if (d == "optix") p.denoise = true;
      else if (d != "none") { std::fprintf(stderr, "--denoise none|optix\n"); std::exit(1); }
    }
    else if (flag("--denoise-albedo")) p.denoise_albedo = true;
    else if (is("--sample-offset")) p.sample_offset = std::max(0, std::atoi(argv[++i]));
    else if (is("--hdr")) c.hdr_out = argv[++i];
    else if (flag("--selftest-denoise")) c.selftest_denoise = true;
    else if (flag("--denoise-kernels")) c.denoise_kernels = true;
    else if (flag("--nohud")) c.hud = false;
    else if (flag("--hud")) c.hud_offline = true;
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
    else if (is("--glare")) p.glare = std::atof(argv[++i]);
    else if (is("--glare-thr")) p.glare_thr = std::atof(argv[++i]);
    else if (is("--grain")) p.grain = std::atof(argv[++i]);
    else if (is("--vignette")) p.vignette = std::atof(argv[++i]);
    else if (is("--chroma")) p.chroma = std::atof(argv[++i]);
    else if (flag("--cinematic")) { c.cinematic = true; apply_cinematic(p, true); }
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

static flight::Start start_of(const std::string& s) { return s == "zamo" ? flight::Start::Zamo : s == "infall" ? flight::Start::Infall : flight::Start::Orbit; }
static void fill_flight_params(kcuda::Params& p, const flight::Flight& f) {
  const auto s = f.state();
  p.dist = s.r; p.theta_deg = s.theta * 180.0 / M_PI; p.phi_deg = s.phi * 180.0 / M_PI;
  p.beta[0] = s.beta.x; p.beta[1] = s.beta.y; p.beta[2] = s.beta.z;
  const auto& b = f.basis();
  const Vec3* v[3] = {&b.fwd, &b.right, &b.up};
  double* dst[3] = {p.fwd, p.right, p.up};
  for (int k = 0; k < 3; ++k) { dst[k][0] = v[k]->x; dst[k][1] = v[k]->y; dst[k][2] = v[k]->z; }
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
    // v0.8d: polish. Everything at 0 = identical (physical regression); each effect does what it says.
    auto bytes = [&](const kcuda::Params& q, std::vector<uint8_t>& out) { kcuda::render(q); kcuda::download_rgb(out); };
    auto ndiff = [](const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) { size_t d = 0; for (size_t i = 0; i < x.size(); ++i) d += x[i] != y[i]; return d; };
    auto meanv = [](const std::vector<uint8_t>& x) { double m = 0; for (auto v : x) m += v; return m / x.size(); };
    kcuda::Params q0 = b0; q0.tonemap = kcuda::Tonemap::Aces; q0.bloom = 0.12;
    std::vector<uint8_t> r0, r1, r2;
    bytes(q0, r0);
    kcuda::Params q1 = q0; q1.glare = 0; q1.grain = 0; q1.vignette = 0; q1.chroma = 0; bytes(q1, r1);
    check(ndiff(r0, r1) == 0, "polish at 0: differing bytes (must be identical)", double(ndiff(r0, r1)), 0.5);
    kcuda::Params g = q0; g.glare = 0.5; bytes(g, r1);
    check(meanv(r1) > meanv(r0), "glare adds light (mean luminance rises)", meanv(r1) - meanv(r0), 0.0);
    kcuda::Params gr = q0; gr.grain = 0.05; bytes(gr, r1); bytes(gr, r2);
    check(ndiff(r1, r2) == 0 && ndiff(r1, r0) > 0, "grain: reproducible (fixed seed) and changes the image", double(ndiff(r1, r2)), 0.5);
    kcuda::Params gf = gr; gf.frame_index = 5; bytes(gf, r2);
    check(ndiff(r1, r2) > 0, "grain: the frame_index changes the noise", double(ndiff(r1, r2)), 0.5);
    kcuda::Params vg = q0; vg.vignette = 0.6; bytes(vg, r1);
    auto corner = [&](const std::vector<uint8_t>& x) { double m = 0; int n = 0; for (int j = 0; j < 12; ++j) for (int i = 0; i < 12; ++i) { for (int k = 0; k < 3; ++k) m += x[(size_t(j) * p.w + i) * 3 + k]; ++n; } return m / (3.0 * n); };
    check(corner(r1) <= corner(r0), "vignette: the corner does not get brighter", corner(r0) - corner(r1), -1e-9);
    kcuda::Params ch = q0; ch.chroma = 0.02; bytes(ch, r1);
    check(ndiff(r0, r1) > 0, "chroma: changes the image", double(ndiff(r0, r1)), 0.5);
    kcuda::Params b2 = b0; b2.tonemap = kcuda::Tonemap::Aces;
    kcuda::render(b2); kcuda::download_rgb(a3);
    size_t d4 = 0;
    for (size_t i = 0; i < a1.size(); ++i) d4 += a1[i] != a3[i];
    check(d4 > 0, "aces tone mapping differs from reinhard", double(d4), 0.5);
  }
  return bad;
}

// v0.8b: camera basis in the kernel (default = v0.6b bit-for-bit; rotated basis vs the host formula) and scripted flight
static void draw_flight_hud(const flight::Flight& fl, const kcuda::Params& p, int h) {
  const auto s = fl.state();
  const auto cs = flight::center_shift(p.a, s, fl.basis(), kerr::isco(p.a), p.rout);
  kcuda::hud_draw(flight::hud_text("VOO", p.a, s, fl.pilot().consts(), cs, flight::accel_level(fl.level), fl.timescale, fl.captured(), false, true), std::max(1, h / 540));
}

static int selftest_v08b(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-66s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  std::vector<double> xy;
  for (int j = 0; j < 17; ++j) for (int i = 0; i < 31; ++i) xy.insert(xy.end(), {(i - 15) * 0.0243, (j - 8) * 0.0431});
  const size_t n = xy.size() / 2;
  // 1. default basis: bit-for-bit (1, y, -x)/nrm
  {
    kcuda::Params p = c.p; std::vector<double> out;
    kcuda::camera_rays(p, xy, out);
    // the device may contract 1 + x*x + y*y into an FMA (the host does not): equality to 1e-15; the bit-for-bit proof of the kernel is the md5 golden
    // of tools/regress_v07d.sh (ctest), generated before this generalisation
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
      const double x = xy[2 * i], y = xy[2 * i + 1], nrm = std::sqrt(1.0 + x * x + y * y);
      worst = std::max({worst, std::abs(out[3 * i] - 1.0 / nrm), std::abs(out[3 * i + 1] - y / nrm), std::abs(out[3 * i + 2] + x / nrm)});
    }
    check(worst < 1e-15, "default basis: rays vs (1, y, -x)/nrm", worst, 1e-15);
  }
  // 2. rotated basis (yaw 90, pitch 30, roll 20 degrees) vs the host formula n' = -d'/|d'|; |n'| = 1
  {
    flight::Basis b;
    flight::yaw(b, M_PI / 2); flight::pitch(b, 30 * M_PI / 180); flight::roll(b, 20 * M_PI / 180);
    kcuda::Params p = c.p;
    const Vec3* v[3] = {&b.fwd, &b.right, &b.up};
    double* dst[3] = {p.fwd, p.right, p.up};
    for (int k = 0; k < 3; ++k) { dst[k][0] = v[k]->x; dst[k][1] = v[k]->y; dst[k][2] = v[k]->z; }
    std::vector<double> out;
    kcuda::camera_rays(p, xy, out);
    double worst = 0, nerr = 0;
    for (size_t i = 0; i < n; ++i) {
      const Vec3 d = b.fwd + xy[2 * i] * b.right + xy[2 * i + 1] * b.up;
      const Vec3 ref = (-1.0 / norm(d)) * d;
      const Vec3 g{out[3 * i], out[3 * i + 1], out[3 * i + 2]};
      worst = std::max(worst, norm(g - ref)); nerr = std::max(nerr, std::abs(norm(g) - 1.0));
    }
    check(worst < 1e-14 && nerr < 1e-14, "rotated basis (yaw 90, pitch 30, roll 20): GPU vs host", std::max(worst, nerr), 1e-14);
    const double ortho = std::max({std::abs(dot(b.fwd, b.right)), std::abs(dot(b.fwd, b.up)), std::abs(dot(b.right, b.up)), std::abs(norm(b.fwd) - 1.0)});
    check(ortho < 1e-15, "yaw/pitch/roll rotations preserve orthonormality", ortho, 1e-15);
    // yaw 90 degrees: the central ray now comes from the direction +right/-... : n'(0,0) = -fwd
    const double cx = out[3 * (8 * 31 + 15)], cy = out[3 * (8 * 31 + 15) + 1], cz = out[3 * (8 * 31 + 15) + 2];
    const Vec3 center = -1.0 * b.fwd;
    check(norm(Vec3{cx, cy, cz} - center) < 1e-14, "central ray = -fwd", norm(Vec3{cx, cy, cz} - center), 1e-14);
  }
  // 3a. flight controls (host): thruster, brake, look, capture and reset
  {
    flight::Flight f(0.9, 30.0, 80.0, 0.0, flight::Start::Zamo);
    f.timescale = 5.0;
    flight::Controls fw; fw.fwd = 1;
    for (int i = 0; i < 120; ++i) f.step(1.0 / 60, fw, false);  // 2 s of real time = 10 M at 0.02 c/M
    const auto s1 = f.state();
    check(s1.beta.x < -0.15 && std::abs(s1.beta.y) < 1e-3 && std::abs(s1.beta.z) < 1e-3, "W: forward thruster (-e_r) accelerates ~0.02 c/M x 10 M", -s1.beta.x, 0.15);
    flight::Controls br; br.brake = true;
    double bmax_after = 0;
    for (int i = 0; i < 600; ++i) { f.step(1.0 / 60, br, false); }
    bmax_after = norm(f.state().beta);
    check(bmax_after < 0.01, "Ctrl: the brake cancels the velocity relative to the ZAMO (without reversing)", bmax_after, 0.01);
    flight::Basis b0 = f.basis();
    f.look(M_PI / 2, 0, 0);
    check(norm(f.basis().fwd - b0.right) < 1e-12, "look: a 90-degree yaw puts fwd on the old right", norm(f.basis().fwd - b0.right), 1e-12);
    // capture: fall from r = 2.6 at rest
    flight::Flight g(0.9, 2.6, 85.0, 0.0, flight::Start::Zamo);
    g.timescale = 5.0;
    for (int i = 0; i < 600 && !g.captured(); ++i) g.step(1.0 / 60, flight::Controls{}, false);
    const bool cap = g.captured();
    const double r_stop = g.state().r;
    g.step(1.0 / 60, flight::Controls{}, false);  // paused: nothing changes
    check(cap && g.state().r == r_stop, "horizon: captured() and the simulation stops (Backspace resets)", cap ? 0.0 : 1.0, 0.5);
    g.reset();
    check(!g.captured() && std::abs(g.state().r - 2.6) < 1e-12 && g.state().tau == 0.0, "Backspace: reset restores the initial state", std::abs(g.state().r - 2.6), 1e-12);
  }
  // 3. scripted flight (clip): 900 frames at 60 fps, timescale 8.5: constants conserved between burns, never NaN, never captured;
  //    and the kernel never produces non-finite values along the flight
  {
    flight::Flight fl(0.9, 30.0, 80.0, 0.0, flight::Start::Orbit);
    fl.timescale = 8.5;
    kcuda::Params p = c.p; p.w = 96; p.h = 54; p.a = 0.9; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.samples = 1;
    p.nodisk = false;
    bool nan_seen = false, kernel_bad = false;
    double rmin = 1e9, dE = 0, dL = 0, dQ = 0;
    pilot::Consts c0 = fl.pilot().consts(), c1{};
    { const auto s0 = fl.state(); const auto z = pilot::zamo(0.9, s0.r, s0.theta); const double g0 = 1.0 / std::sqrt(1.0 - dot(s0.beta, s0.beta)); c0.E = g0 * (z.alpha + z.omega * z.varpi * s0.beta.z); c0.L = g0 * z.varpi * s0.beta.z; }
    bool have1 = false;
    const auto& scr = flight::flight_script();
    for (int i = 0; i < 900; ++i) {
      const double tau = fl.state().tau;
      if (i > 0) fl.step_accel(1.0 / 60.0, flight::script_accel(tau));
      const auto s = fl.state();
      nan_seen |= !std::isfinite(s.r) || !std::isfinite(s.theta) || !std::isfinite(s.phi) || !std::isfinite(norm(s.beta));
      rmin = std::min(rmin, s.r);
      // coasting (between burns): the constants do not change
      const bool coast_a = s.tau > 0.1 && s.tau < scr[0].t0 - 0.2;           // before the brake
      const bool coast_b = s.tau > scr[1].t1 + 0.3 && s.tau < scr[2].t0 - 0.3;  // free fall
      if (coast_a || coast_b) {
        auto cc = fl.pilot().consts();
        {  // E and L recomputed from the STATE (r, theta, beta), not just the stored constants: they test the integration
          const auto z = pilot::zamo(0.9, s.r, s.theta);
          const double g = 1.0 / std::sqrt(1.0 - dot(s.beta, s.beta));
          cc.E = g * (z.alpha + z.omega * z.varpi * s.beta.z);
          cc.L = g * z.varpi * s.beta.z;
        }
        if (!have1) { c1 = cc; have1 = true; }
        if (coast_b && have1 && s.tau < scr[1].t1 + 0.6) c1 = cc;  // free-fall reference = 1st sample after the burn
        if (coast_a) { dE = std::max(dE, std::abs(cc.E - c0.E)); dL = std::max(dL, std::abs(cc.L - c0.L)); dQ = std::max(dQ, std::abs(cc.Q - c0.Q)); }
        else { dE = std::max(dE, std::abs(cc.E - c1.E)); dL = std::max(dL, std::abs(cc.L - c1.L)); dQ = std::max(dQ, std::abs(cc.Q - c1.Q)); }
      }
      if (i % 3 == 0) {  // 300 frames rendered
        fill_flight_params(p, fl);
        kcuda::render(p);
        kernel_bad |= !kcuda::accum_finite();
      }
    }
    check(!nan_seen && !fl.captured(), "scripted flight (900 frames): no NaN and no capture", nan_seen ? 1.0 : 0.0, 0.5);
    check(rmin > kerr::horizon(0.9) + 2.0 && rmin < 5.0, "clip periastron at r ~ 4.4 (above the horizon)", rmin, 5.0);
    check(dE < 1e-7 && dL < 1e-6 && dQ < 1e-6, "E, L (from the state) and Q conserved between burns (max |dE|,|dL|,|dQ|)", std::max({dE, dL, dQ}), 1e-6);
    check(!kernel_bad, "kernel: accumulator finite throughout the flight (300 frames rendered)", kernel_bad ? 1.0 : 0.0, 0.5);
  }
  return bad;
}

// v0.8c: OptiX denoiser. Without OptiX (stub) the option must fail with a clear message; with OptiX a synthetic image with known noise
// must end up with a smaller error and the pipeline (HDR average -> denoiser -> bloom -> tone mapping) must run.
static int selftest_denoise(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-66s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  kcuda::Params p = c.p;
  p.w = 192; p.h = 108; p.denoise = true;
  std::string err;
  if (!kcuda::denoise_available()) {
    const bool ok = !kcuda::denoise_enable(p, err) && err.find("not available") != std::string::npos;
    std::printf("  without OptiX (stub): \"%s\"\n", err.c_str());
    check(ok, "without OptiX: --denoise optix fails with a clear message", ok ? 0.0 : 1.0, 0.5);
    return bad;
  }
  std::printf("  %s\n", kcuda::denoise_version());
  // 1. synthetic image: smooth HDR (gradient + bright blob) + known multiplicative Gaussian noise (LCG + Box-Muller)
  const int w = 256, h = 144;
  std::vector<float> clean(size_t(w) * h * 4), noisy(clean.size()), out;
  uint64_t st = 0x9E3779B97F4A7C15ull;
  auto u01 = [&] { st = st * 6364136223846793005ull + 1442695040888963407ull; return ((st >> 11) + 0.5) / 9007199254740992.0; };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double g = 1.0 + 3.0 * std::exp(-((x - 128.0) * (x - 128.0) + (y - 72.0) * (y - 72.0)) / (2.0 * 30.0 * 30.0));
      const double base[3] = {(0.2 + 0.8 * x / w) * g, (0.3 + 0.5 * y / h) * g, (0.1 + 0.4 * double(x + y) / (w + h)) * g};
      for (int k = 0; k < 3; ++k) {
        const double z = std::sqrt(-2.0 * std::log(u01())) * std::cos(2.0 * M_PI * u01());
        clean[(size_t(y) * w + x) * 4 + k] = float(base[k]);
        noisy[(size_t(y) * w + x) * 4 + k] = float(base[k] * (1.0 + 0.25 * z));
      }
      clean[(size_t(y) * w + x) * 4 + 3] = noisy[(size_t(y) * w + x) * 4 + 3] = 1.f;
    }
  if (!kcuda::denoise_image(noisy, w, h, out, err)) { check(false, "denoise_image failed", 1.0, 0.5); std::printf("    %s\n", err.c_str()); return bad + 0; }
  double mn = 0, md = 0;
  for (size_t i = 0; i < clean.size(); i += 4) for (int k = 0; k < 3; ++k) { const double a = noisy[i + k] - clean[i + k], b = out[i + k] - clean[i + k]; mn += a * a; md += b * b; }
  check(md < 0.5 * mn, "synthetic image (25 % noise): denoised MSE < 0.5 x noisy MSE", md / mn, 0.5);
  // 2. full pipeline on a small scene: finite, different from the non-denoised one, time measured
  kcuda::Params q = p;
  q.prec = kcuda::Precision::Float; q.tol = 2e-5; q.a = 0.9; q.theta_deg = 80; q.samples = 2; q.sample_offset = 10; q.denoise = true;
  if (!kcuda::denoise_enable(q, err)) { check(false, "denoise_enable failed", 1.0, 0.5); std::printf("    %s\n", err.c_str()); return bad; }
  std::vector<uint8_t> a1, a2;
  kcuda::render(q); kcuda::download_rgb(a1);
  const float dn_ms = kcuda::last_timing().denoise_ms;
  std::vector<float> hdr;
  kcuda::download_hdr(hdr);
  bool finite = true;
  for (float v : hdr) finite &= std::isfinite(v);
  q.denoise = false; kcuda::render(q); kcuda::download_rgb(a2);
  size_t d = 0;
  for (size_t i = 0; i < a1.size(); ++i) d += a1[i] != a2[i];
  check(finite && dn_ms > 0.f, "pipeline with denoiser: finite HDR and denoiser executed (ms > 0)", dn_ms, 0.0);
  check(d > 0, "the denoiser changes the image (bytes differ from --denoise none)", double(d), 0.5);
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
  std::printf("v0.8b:\n");
  bad += selftest_v08b(c);
  std::printf("v0.8c:\n");
  bad += selftest_denoise(c);
  std::printf(bad ? "selftest FAILED (%d configurations)\n" : "selftest OK\n", bad);
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------- shot
static int shot(Cli& c) {
  auto& p = c.p;
  if (p.rin < 0) p.rin = kerr::isco(p.a, true);
  apply_observer(p, c.observer);
  std::unique_ptr<flight::Flight> fl;
  if (c.mode == "flight") {  // initial flight state (--start) at the given (dist, theta, phi) position
    fl = std::make_unique<flight::Flight>(p.a, p.dist, p.theta_deg, p.phi_deg, start_of(c.start));
    fl->timescale = c.timescale;
    fill_flight_params(p, *fl);
  }
  if (p.denoise) {
    std::string derr;
    if (!kcuda::denoise_enable(p, derr)) { std::fprintf(stderr, "--denoise optix: %s\n", derr.c_str()); return 1; }
    std::printf("denoiser: %s\n", kcuda::denoise_version());
  }
  kcuda::Stats st;
  float first = kcuda::render(p, &st);
  for (int i = 0; i < 2; ++i) kcuda::render(p);  // warm-up
  double sum = 0, tr = 0, rs = 0, bl = 0, dn = 0;
  for (int i = 0; i < c.runs; ++i) {
    sum += kcuda::render(p);
    const auto t = kcuda::last_timing();
    tr += t.trace_ms; rs += t.resolve_ms; bl += t.bloom_ms; dn += t.denoise_ms;  // trace_ms = last sample only
  }
  std::printf("Kerr a=%.3f theta=%.1f D=%.1f %dx%d tol=%g prec=%s profile=%s observer=%s%s: first %.1f ms; mean of %d: %.3f ms/frame (%.1f fps)\n",
              p.a, p.theta_deg, p.dist, p.w, p.h, p.tol, prec_name(p.prec), p.nt ? "nt" : "newtonian", c.observer.c_str(),
              std::abs(p.beta[0]) + std::abs(p.beta[1]) + std::abs(p.beta[2]) > 0 ? "" : "", first, c.runs, sum / c.runs, 1000.0 * c.runs / sum);
  std::printf("  stages (mean): sample %.3f ms x %d samples | resolve (mean+tone mapping) %.3f ms | bloom %.3f ms | denoiser %.3f ms | tone mapping %s, bloom %g, beaming %g%s\n",
              tr / c.runs, p.samples, rs / c.runs, bl / c.runs, dn / c.runs, p.tonemap == kcuda::Tonemap::Aces ? "aces" : "reinhard", p.bloom, p.beaming,
              c.cinematic ? " [cinematic: AESTHETIC, not physical]" : "");
  std::printf("steps per ray: mean %.1f, max %d; warp efficiency (sum/sum of max*32) %.2f\n", st.steps_mean, st.steps_max, st.warp_eff);
  if (!c.hdr_out.empty()) {  // RGB float PFM (rows from bottom to top, little-endian)
    std::vector<float> px;
    kcuda::download_hdr(px);
    if (FILE* f = std::fopen(c.hdr_out.c_str(), "wb")) {
      std::fprintf(f, "PF\n%d %d\n-1.0\n", p.w, p.h);
      std::vector<float> row(size_t(p.w) * 3);
      for (int j = p.h - 1; j >= 0; --j) {
        for (int i = 0; i < p.w; ++i) for (int k = 0; k < 3; ++k) row[size_t(i) * 3 + k] = px[(size_t(j) * p.w + i) * 4 + k];
        std::fwrite(row.data(), sizeof(float), row.size(), f);
      }
      std::fclose(f);
      std::printf("wrote %s (HDR)\n", c.hdr_out.c_str());
    }
  }
  if (fl && c.hud_offline) draw_flight_hud(*fl, p, p.h);
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
  const double secs = c.seconds > 0 ? c.seconds : (c.clip == "flight" ? 15.0 : c.clip == "infall" ? 12.0 : 10.0);
  const int n = int(std::lround(secs * c.fps));
  char cmd[1024];
  std::snprintf(cmd, sizeof cmd,
                "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -c:v libx264 -preset medium -crf %d "
                "-pix_fmt yuv420p -movflags +faststart \"%s\"",
                p.w, p.h, c.fps, c.crf, c.video.c_str());
  FILE* ff = popen(cmd, "w");
  if (!ff) { std::fprintf(stderr, "could not launch ffmpeg\n"); return 1; }
  if (c.clip == "flight" && !c.cinematic) std::printf("(warning: the flight clip was designed with --cinematic)\n");
  std::printf("video '%s': clip %s, %d frames (%.1f s at %d fps), %dx%d, %s, %d samples/frame, profile %s%s\n", c.video.c_str(), c.clip.c_str(), n, secs,
              c.fps, p.w, p.h, prec_name(p.prec), p.samples, p.nt ? "nt" : "newtonian", c.cinematic ? ", CINEMATIC (aesthetic)" : "");
  const auto t0 = Clock::now();
  std::vector<uint8_t> rgb;
  double gpu_ms = 0;
  if (p.denoise) {
    std::string derr;
    if (!kcuda::denoise_enable(p, derr)) { std::fprintf(stderr, "--denoise optix: %s\n", derr.c_str()); pclose(ff); return 1; }
  }
  std::unique_ptr<flight::Flight> fl;
  double rmin = 1e9, bmax = 0;
  if (c.clip == "flight") {  // the pilot starts at r = 30 on a circular orbit (a = 0.9, theta = 80); everything written in flight::flight_script()
    p.a = 0.9;
    fl = std::make_unique<flight::Flight>(p.a, 30.0, 80.0, 0.0, flight::Start::Orbit);
    fl->timescale = 8.5;  // 15 s -> 127.5 M of proper time
  }
  for (int i = 0; i < n; ++i) {
    std::string obs;
    if (fl) {
      if (i > 0) fl->step_accel(1.0 / c.fps, flight::script_accel(fl->state().tau));
      fill_flight_params(p, *fl);
      const auto s = fl->state();
      rmin = std::min(rmin, s.r); bmax = std::max(bmax, norm(s.beta));
      if (fl->captured()) { std::fprintf(stderr, "the pilot crossed the horizon at frame %d\n", i); pclose(ff); return 1; }
    } else clip_frame(c, i, n, p, obs);
    p.frame_index = i;
    gpu_ms += kcuda::render(p);
    if (fl && c.hud_offline) draw_flight_hud(*fl, p, p.h);
    kcuda::download_rgb(rgb);
    if (std::fwrite(rgb.data(), 1, rgb.size(), ff) != rgb.size()) { std::fprintf(stderr, "write to ffmpeg failed (frame %d)\n", i); pclose(ff); return 1; }
    if (i % 60 == 0) { std::printf("  frame %d/%d (a=%.3f theta=%.1f phi=%.0f D=%.2f |beta|=%.3f)\n", i, n, p.a, p.theta_deg, p.phi_deg, p.dist, std::abs(p.beta[0]) + std::abs(p.beta[2])); std::fflush(stdout); }
  }
  const int rc = pclose(ff);
  const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
  if (fl) std::printf("flight: r_min = %.3f (horizon %.3f), max |beta| = %.3f, final tau = %.1f, final r = %.2f\n", rmin, kerr::horizon(p.a), bmax, fl->state().tau, fl->state().r);
  std::printf("render+encode: %.1f s de relogio (%.1f ms/frame; GPU %.1f ms/frame), ffmpeg rc=%d\n", wall, 1000.0 * wall / n, gpu_ms / n, rc);
  return rc == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- interactive
struct App {
  Cli c;
  camera::Orbit cam;
  std::unique_ptr<flight::Flight> fl;  // != nullptr <=> flight mode
  bool dragging = false, frozen = false, hud = true, want_reset = false;
  std::string denoise_msg;  // last denoiser error message (shown in the title)
  double lastx = 0, lasty = 0, look_dx = 0, look_dy = 0;  // accumulated mouse movement (px) since the last frame
};
static void enter_flight(App& a) {
  a.fl = std::make_unique<flight::Flight>(a.c.p.a, a.cam.dist, a.cam.theta * 180.0 / M_PI, a.cam.phi * 180.0 / M_PI, start_of(a.c.start));
  a.fl->timescale = a.c.timescale;
}
static void leave_flight(App& a) {
  const auto s = a.fl->state();
  a.cam.dist = s.r; a.cam.theta = s.theta; a.cam.phi = s.phi;
  a.cam.clamp();
  a.fl.reset();
}
static void on_scroll(GLFWwindow* w, double, double dy) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (a->fl) return;  // in flight the wheel does nothing (the distance is a consequence of the trajectory)
  a->cam.dist *= std::exp(-0.08 * dy);
  a->cam.clamp();
}
static void on_button(GLFWwindow* w, int b, int act, int) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (b == GLFW_MOUSE_BUTTON_LEFT) { a->dragging = act == GLFW_PRESS; glfwGetCursorPos(w, &a->lastx, &a->lasty); }
}
static void on_cursor(GLFWwindow* w, double x, double y) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (a->dragging) {
    if (a->fl) { a->look_dx += x - a->lastx; a->look_dy += y - a->lasty; }
    else { a->cam.phi -= (x - a->lastx) * 0.005; a->cam.theta -= (y - a->lasty) * 0.005; a->cam.clamp(); }
  }
  a->lastx = x; a->lasty = y;
}
static void on_key(GLFWwindow* w, int key, int, int act, int) {
  if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  auto& p = a->c.p;
  const bool in_flight = a->fl != nullptr;
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
  else if (key == GLFW_KEY_F && act == GLFW_PRESS) { if (in_flight) leave_flight(*a); else enter_flight(*a); }
  else if (key == GLFW_KEY_H && act == GLFW_PRESS) a->hud = !a->hud;
  else if (key == GLFW_KEY_Z && act == GLFW_PRESS) a->frozen = !a->frozen;
  else if (key == GLFW_KEY_BACKSPACE && act == GLFW_PRESS && in_flight) { a->fl->reset(); a->frozen = false; }
  else if (key == GLFW_KEY_COMMA && in_flight) a->fl->timescale = std::max(0.05, a->fl->timescale / 1.25);
  else if (key == GLFW_KEY_PERIOD && in_flight) a->fl->timescale = std::min(100.0, a->fl->timescale * 1.25);
  else if ((key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) && in_flight) a->fl->level = std::min(6, a->fl->level + 1);
  else if ((key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) && in_flight) a->fl->level = std::max(0, a->fl->level - 1);
  else if (key == GLFW_KEY_PAGE_UP) p.exposure *= 1.1;
  else if (key == GLFW_KEY_PAGE_DOWN) p.exposure /= 1.1;
  else if (key == GLFW_KEY_D && act == GLFW_PRESS && !in_flight) p.nodoppler = !p.nodoppler;
  else if (key == GLFW_KEY_G && act == GLFW_PRESS) p.nodisk = !p.nodisk;
  else if (key == GLFW_KEY_P && act == GLFW_PRESS) {
    p.prec = p.prec == kcuda::Precision::Double ? kcuda::Precision::Float : kcuda::Precision::Double;
    p.tol = p.prec == kcuda::Precision::Double ? 1e-8 : 1e-5;
  }
  else if (key == GLFW_KEY_LEFT_BRACKET && !in_flight) p.a = std::max(0.0, p.a - 0.01);
  else if (key == GLFW_KEY_RIGHT_BRACKET && !in_flight) p.a = std::min(0.998, p.a + 0.01);
  else if (key == GLFW_KEY_UP) p.sky_exposure *= 1.1;
  else if (key == GLFW_KEY_DOWN) p.sky_exposure /= 1.1;
  else if ((key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) && !in_flight) p.exposure *= 1.1;
  else if ((key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) && !in_flight) p.exposure /= 1.1;
  else if (key == GLFW_KEY_N && act == GLFW_PRESS) {  // OptiX denoiser on/off (only if available)
    if (p.denoise) p.denoise = false;
    else {
      std::string derr;
      if (kcuda::denoise_enable(p, derr)) p.denoise = true;
      else { std::fprintf(stderr, "denoiser: %s\n", derr.c_str()); a->denoise_msg = derr; }
    }
  }
  else if (key == GLFW_KEY_T && act == GLFW_PRESS) p.tonemap = p.tonemap == kcuda::Tonemap::Reinhard ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
  else if (key == GLFW_KEY_B && act == GLFW_PRESS) {  // bloom: 0 -> 0.06 -> 0.12 -> 0.25 -> 0.5 -> 0
    const double lv[] = {0, 0.06, 0.12, 0.25, 0.5};
    int k = 0;
    for (int i = 0; i < 5; ++i) if (std::abs(p.bloom - lv[i]) < 1e-9) k = i;
    p.bloom = lv[(k + 1) % 5];
  }
  else if (key == GLFW_KEY_O && act == GLFW_PRESS && !in_flight) a->c.observer = a->c.observer == "static" ? "orbit" : a->c.observer == "orbit" ? "infall" : "static";
  else if (key == GLFW_KEY_C && act == GLFW_PRESS) {  // cinematic (AESTHETIC) preset on/off
    a->c.cinematic = !a->c.cinematic;
    apply_cinematic(p, a->c.cinematic);
  }
}

static int interactive(Cli& c) {
  App a;
  a.c = c;
  a.hud = c.hud;
  if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* win = glfwCreateWindow(c.p.w > 1920 ? 1280 : c.p.w, c.p.w > 1920 ? 720 : c.p.h, "black-hole-sim v0.8c (CUDA)", nullptr, nullptr);
  if (!win) { std::fprintf(stderr, "could not create the OpenGL window\n"); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(c.vsync ? 1 : 0);
  if (!gl::load()) return 1;
  std::printf("GL %s | %s\n", gl::GetString(GL_VERSION), gl::GetString(GL_RENDERER));
  a.cam.theta = c.p.theta_deg * M_PI / 180.0; a.cam.phi = c.p.phi_deg * M_PI / 180.0; a.cam.dist = c.p.dist; a.cam.clamp();
  a.c.p.rin = -1.0;  // ISCO follows the spin
  if (c.mode == "flight") enter_flight(a);

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
  auto last = Clock::now(), t_start = last, t_frame = last;
  int frames = 0, total = 0, k = 0;
  const int max_samples = c.samples_set ? c.p.samples : 128;  // progressive samples when the scene is still
  std::vector<double> key, key_prev;
  float ms = 0;
  bool last_scaled = false;
  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    int fw, fh;
    glfwGetFramebufferSize(win, &fw, &fh);
    if (fw <= 0 || fh <= 0) { glfwWaitEvents(); continue; }
    if (fw != w || fh != h) make_tex(fw, fh);
    auto& p = a.c.p;
    const auto now = Clock::now();
    const double dt = std::min(0.1, std::chrono::duration<double>(now - t_frame).count());  // limits the jump after pauses
    t_frame = now;
    p.w = w; p.h = h;
    const char* mode_name = "ORBITAL";
    if (a.fl) {
      mode_name = "VOO";
      // continuous keys -> thrust and camera rotation
      flight::Controls ct;
      auto down = [&](int k) { return glfwGetKey(win, k) == GLFW_PRESS; };
      ct.fwd = (down(GLFW_KEY_W) ? 1.0 : 0.0) - (down(GLFW_KEY_S) ? 1.0 : 0.0);
      ct.right = (down(GLFW_KEY_D) ? 1.0 : 0.0) - (down(GLFW_KEY_A) ? 1.0 : 0.0);
      ct.up = (down(GLFW_KEY_SPACE) ? 1.0 : 0.0) - (down(GLFW_KEY_LEFT_SHIFT) ? 1.0 : 0.0);
      ct.brake = down(GLFW_KEY_LEFT_CONTROL);
      const double roll = ((down(GLFW_KEY_E) ? 1.0 : 0.0) - (down(GLFW_KEY_Q) ? 1.0 : 0.0)) * 1.2 * dt;
      a.fl->look(a.look_dx * 0.003, -a.look_dy * 0.003, roll);  // drag right = look right; up = look up
      a.look_dx = a.look_dy = 0;
      a.fl->step(dt, ct, a.frozen);
      fill_flight_params(p, *a.fl);
    } else {
      p.dist = a.cam.dist; p.theta_deg = a.cam.theta * 180.0 / M_PI; p.phi_deg = a.cam.phi * 180.0 / M_PI;
      apply_observer(p, a.c.observer);
      for (int i = 0; i < 3; ++i) { p.fwd[i] = i == 0 ? -1 : 0; p.right[i] = i == 2 ? 1 : 0; p.up[i] = i == 1 ? -1 : 0; }
    }
    // everything that affects the image; any change (mouse, keys, spin, flight) restarts the accumulation
    key = {double(p.w), double(p.h), p.a, p.dist, p.theta_deg, p.phi_deg, p.exposure, p.sky_exposure, double(p.nodisk), double(p.nodoppler),
           double(int(p.prec)), double(p.nt), p.beta[0], p.beta[1], p.beta[2], double(int(p.tonemap)), p.bloom, p.beaming, p.clamp_lum,
           p.fwd[0], p.fwd[1], p.fwd[2], p.right[0], p.right[1], p.right[2], p.up[0], p.up[1], p.up[2], double(a.hud), p.glare, p.grain, p.vignette, p.chroma, double(p.denoise)};
    const bool changed = key != key_prev;
    key_prev = key;
    // reduced resolution only while the scene changes (--move-scale); when it stops, it restarts accumulating at full resolution
    const bool scaled_now = changed && c.move_scale < 1.0;
    const bool restart = changed || last_scaled;
    last_scaled = scaled_now;
    kcuda::Params q = p;
    if (scaled_now) { q.w = std::max(16, int(w * c.move_scale)); q.h = std::max(16, int(h * c.move_scale)); }
    if (restart) { kcuda::set_size(q.w, q.h); kcuda::accum_reset(); k = 0; }
    const bool converged = k >= max_samples && !restart;
    if (!converged) {
      ms = kcuda::accum_sample(q, k++);
      kcuda::resolve(q);
      if (a.hud) {
        if (a.fl) {
          const auto s = a.fl->state();
          const auto cs = flight::center_shift(a.c.p.a, s, a.fl->basis(), kerr::isco(a.c.p.a), p.rout);
          kcuda::hud_draw(flight::hud_text(mode_name, a.c.p.a, s, a.fl->pilot().consts(), cs, flight::accel_level(a.fl->level), a.fl->timescale, a.fl->captured(), a.frozen, true), std::max(1, q.h / 540));
        } else {
          const pilot::State s{p.dist, p.theta_deg * M_PI / 180.0, p.phi_deg * M_PI / 180.0, Vec3{p.beta[0], p.beta[1], p.beta[2]}, 0, 0};
          const auto cs = flight::center_shift(p.a, s, flight::Basis{}, kerr::isco(p.a), p.rout);
          kcuda::hud_draw(flight::hud_text(mode_name, p.a, s, pilot::Consts{p.a, 0, 0, 0}, cs, 0, 0, false, false, false), std::max(1, q.h / 540));
        }
      }
    } else glfwWaitEventsTimeout(0.05);  // converged image: do not spend GPU
    kcuda::gl_copy();
    const bool scaled = kcuda::render_w() != w || kcuda::render_h() != h;
    gl::BlitNamedFramebuffer(fbo, 0, 0, 0, kcuda::render_w(), kcuda::render_h(), 0, h, w, 0, GL_COLOR_BUFFER_BIT, scaled ? GL_LINEAR : GL_NEAREST);
    glfwSwapBuffers(win);
    ++frames; ++total;
    const double dtt = std::chrono::duration<double>(Clock::now() - last).count();
    if (dtt >= 0.5) {
      char t[320];
      const char* cap = a.fl && a.fl->captured() ? " | HORIZON: simulation paused" : "";
      std::snprintf(t, sizeof t, "black-hole-sim v0.8c CUDA %s%s | %.0f fps | sample %.2f ms | N=%d | a=%.3f r=%.1f theta=%.0f %s %s %s bloom=%.2f beaming=%.2f%s%s%s",
                    mode_name, a.c.cinematic ? " [CINEMATIC: aesthetic, not physical]" : "", frames / dtt, ms, std::min(k, max_samples), p.a, p.dist, p.theta_deg,
                    prec_name(p.prec), p.nt ? "NT" : "newt", p.tonemap == kcuda::Tonemap::Aces ? "aces" : "reinhard", p.bloom, p.beaming,
                    p.nodoppler ? " | no Doppler" : "", p.nodisk ? " | no disk" : "", cap);
      if (p.denoise) std::snprintf(t + std::strlen(t), sizeof t - std::strlen(t), " | denoise OptiX");
      else if (!a.denoise_msg.empty()) std::snprintf(t + std::strlen(t), sizeof t - std::strlen(t), " | denoiser unavailable (N)");
      glfwSetWindowTitle(win, t);
      if (c.autoexit) std::printf("%s\n", t);
      frames = 0; last = Clock::now();
    }
    if (c.autoexit && total >= c.autoexit) {
      const double s = std::chrono::duration<double>(Clock::now() - t_start).count();
      std::printf("autoexit: %d frames in %.2f s = %.1f fps mean (%dx%d, move-scale %.2f)\n", total, s, total / s, w, h, c.move_scale);
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
  if (c.denoise_kernels) {  // which CUDA kernels does the OptiX denoiser launch? (CUPTI); 1080p, 2 invocations
    if (!kcuda::denoise_available() || !denoise::kernel_trace_available()) { std::fprintf(stderr, "--denoise-kernels needs OptiX and CUPTI compiled in\n"); return 1; }
    if (!kcuda::init("", err)) return 1;
    const int w = 1920, h = 1080;
    std::vector<float> img(size_t(w) * h * 4), out;
    for (size_t i = 0; i < img.size(); ++i) img[i] = (i % 4 == 3) ? 1.f : 0.5f + 0.4f * std::sin(0.01f * float(i / 4 % w)) + 0.2f * float((i * 2654435761u) % 1000) / 1000.f;
    if (!kcuda::denoise_image(img, w, h, out, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }  // warm-up (init + setup)
    denoise::kernel_trace_begin();
    for (int i = 0; i < 2; ++i) kcuda::denoise_image(img, w, h, out, err);
    auto ks = denoise::kernel_trace_end();
    std::sort(ks.begin(), ks.end(), [](const denoise::KernelStat& a, const denoise::KernelStat& b) { return a.total_us > b.total_us; });
    double tot = 0;
    for (const auto& k : ks) { tot += k.total_us; std::printf("%8.1f us x%-3u %s\n", k.total_us, k.count, k.name.c_str()); }
    std::printf("distinct kernels: %zu; total kernel time %.1f us (2 invocations, 1080p)\n", ks.size(), tot);
    std::printf("Note: the names (conv_ampere_*, k_tin3_mlp with __half, fused_convolutions_kernel_sm80) are FP16 convolutions/MLP for Ampere+;\n"
                "consistent with tensor cores, but this does NOT prove it (no hardware counters: no ncu/nsys on this system).\n");
    return 0;
  }
  if (c.selftest_denoise) { c.p.use_sky = false; if (!kcuda::init("", err)) return 1; const int b = selftest_denoise(c); std::printf(b ? "denoiser selftest FAILED\n" : "denoiser selftest OK\n"); return b ? 1 : 0; }
  if (c.selftest) return selftest(c);
  if (!c.video.empty()) { if (!c.prec_set) { c.p.prec = kcuda::Precision::Float; c.p.tol = 2e-5; } if (c.p.rin < 0) c.p.rin = -1; return video(c); }
  if (!c.shot.empty()) return shot(c);
  return interactive(c);
}
