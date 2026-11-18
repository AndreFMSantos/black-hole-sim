// v1.1 — Recorder of the 2D versions (v0.1 Newton/integrators and v0.2 light in Schwarzschild) for the journey time-lapse.
// The originals are interactive OpenGL windows; this recorder uses the SAME physics (core: integrators.hpp/newton.hpp, schwarzschild.hpp) and the same
// look (colours, dashes, horizon), with a software rasteriser, and accepts the video's global camera C(t): the plane is seen from above
// (theta ~ 0) rotated by the azimuth phi and tilted by theta (orthographic projection: y' = y cos(theta)).
//   v1_1-record-2d --scene newton|light --cams cams.txt --out DIR [--w 1920 --h 1080] [--steps 8] [--first 0]
// cams.txt: one line per frame "theta_deg phi_deg". Writes DIR/frame_NNNNN.png. The numerical results the video shows
// (final Euler energy and radius, critical impact parameter) go to stdout, computed from the simulation.
#include <algorithm>
#include <bhs/integrators.hpp>
#include <bhs/newton.hpp>
#include <bhs/png.hpp>
#include <bhs/schwarzschild.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

using namespace bhs;

struct Canvas {
  int w, h, S;  // S = supersampling linear
  std::vector<float> px;
  Canvas(int w_, int h_, int s) : w(w_), h(h_), S(s), px(size_t(w_) * s * h_ * s * 3) {}
  int W() const { return w * S; }
  int H() const { return h * S; }
  void clear(float r, float g, float b) { for (size_t i = 0; i < px.size(); i += 3) { px[i] = r; px[i + 1] = g; px[i + 2] = b; } }
  void blend(int x, int y, float cr, float cg, float cb, float a) {
    if (x < 0 || y < 0 || x >= W() || y >= H() || a <= 0) return;
    float* p = &px[(size_t(y) * W() + x) * 3];
    p[0] += (cr - p[0]) * a; p[1] += (cg - p[1]) * a; p[2] += (cb - p[2]) * a;
  }
  // segment with thickness (in final pixels) and analytic coverage at the distance to the segment
  void line(double x0, double y0, double x1, double y1, double width, float cr, float cg, float cb, float alpha) {
    const double hw = 0.5 * width * S, ax = x1 - x0, ay = y1 - y0, l2 = ax * ax + ay * ay;
    const int xa = int(std::floor(std::min(x0, x1) - hw - 1)), xb = int(std::ceil(std::max(x0, x1) + hw + 1));
    const int ya = int(std::floor(std::min(y0, y1) - hw - 1)), yb = int(std::ceil(std::max(y0, y1) + hw + 1));
    const double rr = hw + 1.0, alen = std::sqrt(l2);
    for (int y = std::max(0, ya); y <= std::min(H() - 1, yb); ++y) {
      // only the strip of pixels within rr of the segment on this row (the bounding box of a diagonal line is almost the whole screen)
      int xs = std::max(0, xa), xe = std::min(W() - 1, xb);
      if (std::fabs(ay) > 1e-9) {
        const double t = (y + 0.5 - y0) / ay;
        if (t >= 0.0 && t <= 1.0) {
          const double xc = x0 + t * ax, halfw = rr * alen / std::fabs(ay);
          xs = std::max(xs, int(std::floor(xc - halfw - 1))); xe = std::min(xe, int(std::ceil(xc + halfw + 1)));
        }
      }
      for (int x = xs; x <= xe; ++x) {
        const double qx = x + 0.5 - x0, qy = y + 0.5 - y0;
        const double t = l2 > 0 ? std::min(1.0, std::max(0.0, (qx * ax + qy * ay) / l2)) : 0.0;
        const double d = std::hypot(qx - t * ax, qy - t * ay);
        const double cov = std::min(1.0, std::max(0.0, hw + 0.5 - d));
        blend(x, y, cr, cg, cb, float(cov) * alpha);
      }
    }
  }
  void disc(double cx, double cy, double r, float cr, float cg, float cb, float alpha) {
    const double rr = r * S;
    for (int y = std::max(0, int(cy - rr - 1)); y <= std::min(H() - 1, int(cy + rr + 1)); ++y)
      for (int x = std::max(0, int(cx - rr - 1)); x <= std::min(W() - 1, int(cx + rr + 1)); ++x) {
        const double d = std::hypot(x + 0.5 - cx, y + 0.5 - cy);
        blend(x, y, cr, cg, cb, float(std::min(1.0, std::max(0.0, rr + 0.5 - d))) * alpha);
      }
  }
  std::vector<uint8_t> resolve() const {
    std::vector<uint8_t> out(size_t(w) * h * 3);
    const float inv = 1.f / float(S * S);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int c = 0; c < 3; ++c) {
          float s = 0;
          for (int j = 0; j < S; ++j) for (int i = 0; i < S; ++i) s += px[((size_t(y) * S + j) * W() + (size_t(x) * S + i)) * 3 + c];
          out[(size_t(y) * w + x) * 3 + c] = uint8_t(std::min(1.f, std::max(0.f, s * inv)) * 255.f + 0.5f);
        }
    return out;
  }
};

// camera: plane z = 0 seen from above, rotated by phi and tilted by theta
struct Cam {
  double cx, cy, scale, ct, cp, sp;  // scale = supersampled pixels per world unit
  void to_screen(double x, double y, double& sx, double& sy) const {
    const double xr = x * cp - y * sp, yr = x * sp + y * cp;
    sx = cx + scale * xr;
    sy = cy - scale * yr * ct;
  }
};
static Cam make_cam(const Canvas& c, double half_height, double theta_deg, double phi_deg) {
  const double th = theta_deg * M_PI / 180.0, ph = phi_deg * M_PI / 180.0;
  return {0.5 * c.W(), 0.5 * c.H(), 0.5 * c.H() / half_height, std::cos(th), std::cos(ph), std::sin(ph)};
}
static void circle(Canvas& cv, const Cam& cam, double r, double width, float cr, float cg, float cb, float a, int dash = 0) {
  const int segs = 240;
  for (int i = 0; i < segs; ++i) {
    if (dash && ((i / dash) & 1)) continue;
    double x0, y0, x1, y1;
    cam.to_screen(r * std::cos(2 * M_PI * i / segs), r * std::sin(2 * M_PI * i / segs), x0, y0);
    cam.to_screen(r * std::cos(2 * M_PI * (i + 1) / segs), r * std::sin(2 * M_PI * (i + 1) / segs), x1, y1);
    cv.line(x0, y0, x1, y1, width, cr, cg, cb, a);
  }
}

struct Particle { Method method; State s; double e0; std::deque<Vec2> trail; float r, g, b; };

int main(int argc, char** argv) {
  std::string scene = "newton", cams_file, out = ".";
  int w = 1920, h = 1080, steps = 8, first = 0;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    if (is("--scene")) scene = argv[++i];
    else if (is("--cams")) cams_file = argv[++i];
    else if (is("--out")) out = argv[++i];
    else if (is("--w")) w = std::atoi(argv[++i]);
    else if (is("--h")) h = std::atoi(argv[++i]);
    else if (is("--steps")) steps = std::atoi(argv[++i]);
    else if (is("--first")) first = std::atoi(argv[++i]);
  }
  std::vector<std::pair<double, double>> cams;
  { std::ifstream f(cams_file); double t, p; while (f >> t >> p) cams.push_back({t, p}); }
  if (cams.empty()) { std::fprintf(stderr, "uso: --scene newton|light --cams cams.txt --out DIR\n"); return 1; }

  const double bg[3] = {0.02f, 0.02f, 0.05f};
  if (scene == "newton") {
    // v0.1: three particles on the same circular orbit, Euler (red) / Verlet (green) / RK4 (blue); dt = 0.005; view 5 units high
    State s0 = circular_orbit(1.0);
    auto mk = [&](Method m, float r, float g, float b) { return Particle{m, s0, energy(s0), {}, r, g, b}; };
    std::vector<Particle> ps = {mk(Method::Euler, 1, .25f, .25f), mk(Method::Verlet, .25f, 1, .25f), mk(Method::RK4, .35f, .55f, 1)};
    const double dt = 0.005;
    for (size_t f = 0; f < cams.size(); ++f) {
      for (int k = 0; k < steps; ++k)
        for (auto& p : ps) if (norm(p.s.x) > 1e-3) p.s = step(p.method, p.s, dt, [](Vec2 q) { return gravity(q); });
      for (auto& p : ps) { p.trail.push_back(p.s.x); if (p.trail.size() > 4000) p.trail.pop_front(); }
      Canvas cv(w, h, 2);
      cv.clear(float(bg[0]), float(bg[1]), float(bg[2]));
      const Cam cam = make_cam(cv, 2.5, cams[f].first, cams[f].second);
      double sx, sy; cam.to_screen(0, 0, sx, sy);
      cv.disc(sx, sy, 7, 1.f, .9f, .5f, 1.f);
      for (auto& p : ps) {
        for (size_t i = 1; i < p.trail.size(); ++i) {
          double ax, ay, bx, by; cam.to_screen(p.trail[i - 1].x, p.trail[i - 1].y, ax, ay); cam.to_screen(p.trail[i].x, p.trail[i].y, bx, by);
          cv.line(ax, ay, bx, by, 1.5, p.r, p.g, p.b, 1.f);
        }
        cam.to_screen(p.s.x.x, p.s.x.y, sx, sy);
        cv.disc(sx, sy, 4, p.r, p.g, p.b, 1.f);
      }
      char nm[512]; std::snprintf(nm, sizeof nm, "%s/frame_%05d.png", out.c_str(), first + int(f));
      if (!png::write_file(nm, png::encode_rgb(w, h, cv.resolve()))) { std::fprintf(stderr, "error writing %s\n", nm); return 1; }
      if (f + 1 == cams.size()) {
        std::printf("newton: t = %.2f; Euler r = %.4f (r0 = 1), dE/E = %+.3e; Verlet r = %.4f, dE/E = %+.3e; RK4 r = %.4f, dE/E = %+.3e\n", double(steps) * dt * double(cams.size()),
                    norm(ps[0].s.x), (energy(ps[0].s) - ps[0].e0) / std::abs(ps[0].e0), norm(ps[1].s.x), (energy(ps[1].s) - ps[1].e0) / std::abs(ps[1].e0),
                    norm(ps[2].s.x), (energy(ps[2].s) - ps[2].e0) / std::abs(ps[2].e0));
      }
    }
  } else if (scene == "light") {
    // v0.2: parallel beam coming from the left, 81 rays, b in [-14, 14]; red = captured, blue->white = escapes (brighter = more deflected)
    namespace S = bhs::schw;
    const double view = 14.0; const int n = 81;
    struct Beam { double b; S::Fate fate; double deflection; std::vector<Vec2> path; };
    std::vector<Beam> beams;
    int captured = 0;
    for (int i = 0; i < n; ++i) {
      Beam be; be.b = -view + (i + 0.5) * (2 * view / n);
      const auto r = S::trace(std::abs(be.b), 1e-3, 400, &be.path);
      be.fate = r.fate; be.deflection = r.deflection;
      if (be.b < 0) for (auto& p : be.path) p.y = -p.y;
      if (r.fate == S::Fate::Captured) ++captured;
      beams.push_back(std::move(be));
    }
    const double aspect = double(w) / h, xext = view * aspect;
    for (size_t f = 0; f < cams.size(); ++f) {
      Canvas cv(w, h, 2);
      cv.clear(float(bg[0]), float(bg[1]), float(bg[2]));
      const Cam cam = make_cam(cv, view, cams[f].first, cams[f].second);
      auto seg = [&](double x0, double y0, double x1, double y1, double wd, float cr, float cg, float cb, float a) {
        double ax, ay, bx, by; cam.to_screen(x0, y0, ax, ay); cam.to_screen(x1, y1, bx, by); cv.line(ax, ay, bx, by, wd, cr, cg, cb, a);
      };
      for (auto& be : beams) seg(-xext * 1.6, be.b, xext * 1.6, be.b, 1.0, 1, 1, 1, 0.12f);  // light without gravity
      for (auto& be : beams) {
        float cr, cg, cb;
        if (be.fate == S::Fate::Captured) { cr = 1.0f; cg = 0.25f; cb = 0.2f; }
        else if (be.fate == S::Fate::Orbiting) { cr = 1.0f; cg = 0.8f; cb = 0.2f; }
        else { const float t = float(std::min(be.deflection / M_PI, 1.0)); cr = 0.3f + 0.7f * t; cg = 0.6f + 0.4f * t; cb = 1.0f; }
        // captured rays wind near the photon sphere with hundreds of thousands of sub-pixel segments: merges the ones shorter than
        // 0.5 final px (1 supersampled px) on screen, without changing the visible stroke
        size_t last = 0;
        for (size_t i = 1; i < be.path.size(); ++i) {
          double ax, ay, bx, by; cam.to_screen(be.path[last].x, be.path[last].y, ax, ay); cam.to_screen(be.path[i].x, be.path[i].y, bx, by);
          if (i + 1 < be.path.size() && std::hypot(bx - ax, by - ay) < 1.0) continue;
          seg(be.path[last].x, be.path[last].y, be.path[i].x, be.path[i].y, 1.6, cr, cg, cb, 0.95f);
          last = i;
        }
      }
      circle(cv, cam, 3.0, 1.2, 1, 1, 1, 0.35f, 4);                  // photon sphere
      circle(cv, cam, S::critical_b(), 1.2, 1, 1, 1, 0.35f, 2);      // apparent shadow
      double sx, sy; cam.to_screen(0, 0, sx, sy);
      cv.disc(sx, sy, 2.0 * cam.scale / cv.S * 0.999, 0, 0, 0, 1.f);  // horizon (black disk; with tilt it becomes an ellipse via y' = y cos theta: approximated by a circle)
      circle(cv, cam, 2.0, 2.0, 1.f, .9f, .5f, 1.f);
      char nm[512]; std::snprintf(nm, sizeof nm, "%s/frame_%05d.png", out.c_str(), first + int(f));
      if (!png::write_file(nm, png::encode_rgb(w, h, cv.resolve()))) { std::fprintf(stderr, "error writing %s\n", nm); return 1; }
    }
    std::printf("light: %d rays, %d captured, shadow b = %.5f\n", n, captured, S::critical_b());
  } else { std::fprintf(stderr, "--scene newton|light\n"); return 1; }
  return 0;
}
