// v0.6b — Kerr on CUDA (RTX 5080). Same options as v0_6-kerr-cpu for direct comparison.
// Usage:  v0_6b-kerr-cuda --shot out.png [--w 1920 --h 1080 --a 0.9 --theta 80 --phi 0 --dist 40 --fov 40
//         --rout 16 --rin <isco> --tpeak 4200 --exposure 3 --sky-exposure 2 --nodisk --nodoppler --tol 1e-8
//         --prec double|float --runs 20 --block 128 --sky map.exr --nosky]
//       v0_6b-kerr-cuda --selftest          (GPU vs the CPU's kerr::trace, no window)
//       v0_6b-kerr-cuda [--a 0.9 ...]       (interactive: mouse orbits, wheel = distance, [ ] spin, D Doppler,
//                                           G disk, +/- exposure, arrows = sky, P precision, ESC quits;
//                                           --autoexit N quits after N frames and prints the fps)
#include "../v0_5-realtime-gpu/gl_loader.hpp"

#include <algorithm>
#include <bhs/camera.hpp>
#include <bhs/kerr.hpp>
#include <bhs/png.hpp>
#include <bhs/vec3.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "render.hpp"

using namespace bhs;
using Clock = std::chrono::steady_clock;

struct Cli {
  kcuda::Params p;
  std::string shot, sky = BHS_SKY_DEFAULT;
  bool nosky = false, selftest = false, vsync = false, prec_set = false;
  int runs = 20, autoexit = 0;
};

static Cli parse(int argc, char** argv) {
  Cli c;
  auto& p = c.p;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    auto flag = [&](const char* k) { return std::strcmp(argv[i], k) == 0; };
    if (is("--shot")) c.shot = argv[++i];
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
  }
  if (p.exposure == 3.0 && c.shot.empty()) p.exposure = 3.0;
  return c;
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
  std::printf(bad ? "selftest FAILED (%d configurations)\n" : "selftest OK\n", bad);
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------- shot
static int shot(Cli& c) {
  auto& p = c.p;
  if (p.rin < 0) p.rin = kerr::isco(p.a, true);
  kcuda::Stats st;
  float first = kcuda::render(p, &st);
  for (int i = 0; i < 3; ++i) kcuda::render(p);  // warm-up
  double sum = 0;
  for (int i = 0; i < c.runs; ++i) sum += kcuda::render(p);
  std::printf("Kerr a=%.3f theta=%.1f D=%.1f %dx%d tol=%g prec=%s: first %.1f ms; mean of %d: %.3f ms/frame (%.1f fps)\n", p.a, p.theta_deg,
              p.dist, p.w, p.h, p.tol, prec_name(p.prec), first, c.runs, sum / c.runs, 1000.0 * c.runs / sum);
  std::printf("steps per ray: mean %.1f, max %d; warp efficiency (sum/sum of max*32) %.2f\n", st.steps_mean, st.steps_max, st.warp_eff);
  std::vector<uint8_t> rgb;
  kcuda::download_rgb(rgb);
  if (!png::write_file(c.shot, png::encode_rgb(p.w, p.h, rgb))) { std::fprintf(stderr, "error writing %s\n", c.shot.c_str()); return 1; }
  std::printf("wrote %s\n", c.shot.c_str());
  return 0;
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
}

static int interactive(Cli& c) {
  App a;
  a.c = c;
  if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* win = glfwCreateWindow(c.p.w > 1920 ? 1280 : c.p.w, c.p.w > 1920 ? 720 : c.p.h, "black-hole-sim v0.6b (CUDA)", nullptr, nullptr);
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
  int frames = 0, total = 0;
  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    int fw, fh;
    glfwGetFramebufferSize(win, &fw, &fh);
    if (fw <= 0 || fh <= 0) { glfwWaitEvents(); continue; }
    if (fw != w || fh != h) make_tex(fw, fh);
    auto& p = a.c.p;
    p.w = w; p.h = h; p.dist = a.cam.dist; p.theta_deg = a.cam.theta * 180.0 / M_PI; p.phi_deg = a.cam.phi * 180.0 / M_PI;
    const float ms = kcuda::render(p);
    kcuda::gl_copy();
    gl::BlitNamedFramebuffer(fbo, 0, 0, 0, w, h, 0, h, w, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glfwSwapBuffers(win);
    ++frames; ++total;
    const double dt = std::chrono::duration<double>(Clock::now() - last).count();
    if (dt >= 0.5) {
      char t[200];
      std::snprintf(t, sizeof t, "black-hole-sim v0.6b CUDA | %.0f fps | kernel %.2f ms | a=%.3f D=%.0f theta=%.0f %s%s%s", frames / dt, ms, p.a, p.dist,
                    p.theta_deg, prec_name(p.prec), p.nodoppler ? " | no Doppler" : "", p.nodisk ? " | no disk" : "");
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
  const bool is_interactive = c.shot.empty() && !c.selftest;
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
  if (!c.shot.empty()) return shot(c);
  return interactive(c);
}
