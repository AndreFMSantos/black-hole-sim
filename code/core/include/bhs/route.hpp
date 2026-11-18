#pragma once
// Offline route (v1.0): JSON file with keyframes (spin, FOV, exposure), ship (orbit/infall/rest + thrusts), view orientation
// (yaw/pitch/roll by keyframes, slerp) and disk parameters; plus the ship trajectory integrated DETERMINISTICALLY.
//
// Conventions:
//  - route time = seconds of VIDEO; ship proper time = t_video * ship.timescale (like flight mode);
//  - thrusts: proper acceleration in c/M in the (fwd, right, up) components of the camera's DEFAULT basis (fwd = -e_r, right = e_phi,
//    up = -e_theta), independent of the view orientation (so looking around does not push the ship); applied as short impulses
//    per grid step (same as Flight::step_accel);
//  - spin: constant (the ship dynamics depend on it); a variable spin is only accepted with a static camera (cinematic effect);
//  - orientation: yaw (rotation from fwd to right), pitch (fwd to up), roll (right to up), applied in this order to the default basis
//    (like flight.hpp); between keyframes, slerp of the quaternions with the same easing as the other quantities.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "json.hpp"
#include "lorentz.hpp"
#include "pilot.hpp"
#include "quat.hpp"
#include "vec3.hpp"

namespace bhs::route {

struct Key { double t = 0, v = 0; };
struct ViewKey { double t = 0, yaw = 0, pitch = 0, roll = 0; };  // degrees
struct Burn { double t0 = 0, t1 = 0; Vec3 acc{0, 0, 0}; };
struct Waypoint { double t = 0, r = 0, theta = 0, phi = 0; };  // t in video seconds; theta, phi in degrees (like ship.theta/phi)
struct LookAt { bool on = false, hole = true; double r = 0, theta = 0, phi = 0, roll = 0; };  // degrees

struct DiskSettings {
  double hr = 0, tau = 10, flare = 0, turb = 0, turb_scale = 2, edge_soft = 0, edge_noise = 0;
  double spot_r = 10, spot_phi = 0, spot_size = 1.5, spot_amp = 0;
  double flare_t = 0, flare_w = 20, flare_boost = 0;
  bool slowlight = false;
  std::string time_mode = "pilot";  // "pilot": t0 + the ship's coordinate time; "rate": t0 + rate * t_video
  double rate = 20, t0 = 0;
};

inline double smooth01(double u) { return u * u * (3.0 - 2.0 * u); }

inline double interp(const std::vector<Key>& k, double t, bool smooth, double def) {
  if (k.empty()) return def;
  if (t <= k.front().t) return k.front().v;
  if (t >= k.back().t) return k.back().v;
  size_t i = 1;
  while (k[i].t < t) ++i;
  double u = (t - k[i - 1].t) / (k[i].t - k[i - 1].t);
  if (smooth) u = smooth01(u);
  return k[i - 1].v + u * (k[i].v - k[i - 1].v);
}

struct Route {
  std::string name = "route";
  int fps = 60;
  double seconds = 10;
  bool ease_smooth = true;
  std::vector<Key> spin, fov, exposure;
  std::string mode = "orbit";  // orbit | infall | zamo | static
  double r0 = 30, theta0 = 80, phi0 = 0, timescale = 5;
  std::vector<Burn> burns;
  std::vector<ViewKey> view;
  DiskSettings disk;
  // v1.0b: waypoints (Hermite spline in pseudo-Cartesian coordinates) followed by thrust; "look at" a target
  std::vector<Waypoint> waypoints;
  double follow_wn = 0;   // controller bandwidth (1/M of proper time); 0 = default 2 / timescale (response ~ 0.5 s of video)
  double max_acc = 0;     // > 0: limit of the proper acceleration (c/M)
  LookAt lookat;

  double spin_value(double t) const { return interp(spin, t, ease_smooth, 0.9); }
  double fov_value(double t) const { return interp(fov, t, ease_smooth, 40.0); }
  double exposure_value(double t) const { return interp(exposure, t, ease_smooth, 3.0); }
  double disk_time(double t_video, double pilot_t) const { return disk.t0 + (disk.time_mode == "rate" ? disk.rate * t_video : pilot_t); }

  static quat::Q view_key_quat(const ViewKey& k) {
    Vec3 e0[3] = {{-1, 0, 0}, {0, 0, 1}, {0, -1, 0}};  // fwd, right, up
    Vec3 f = e0[0], r = e0[1], u = e0[2];
    const double y = k.yaw * M_PI / 180.0, p = k.pitch * M_PI / 180.0, w = k.roll * M_PI / 180.0;
    { const Vec3 f0 = f, r0 = r; f = std::cos(y) * f0 + std::sin(y) * r0; r = std::cos(y) * r0 - std::sin(y) * f0; }
    { const Vec3 f0 = f, u0 = u; f = std::cos(p) * f0 + std::sin(p) * u0; u = std::cos(p) * u0 - std::sin(p) * f0; }
    { const Vec3 r0 = r, u0 = u; r = std::cos(w) * r0 + std::sin(w) * u0; u = std::cos(w) * u0 - std::sin(w) * r0; }
    const Vec3 e1[3] = {f, r, u};
    return quat::from_bases(e0, e1);
  }
  quat::Q view_quat(double t) const {
    if (view.empty()) return quat::Q{};
    if (t <= view.front().t) return view_key_quat(view.front());
    if (t >= view.back().t) return view_key_quat(view.back());
    size_t i = 1;
    while (view[i].t < t) ++i;
    double u = (t - view[i - 1].t) / (view[i].t - view[i - 1].t);
    if (ease_smooth) u = smooth01(u);
    return quat::slerp(view_key_quat(view[i - 1]), view_key_quat(view[i]), u);
  }
  // v1.0b: basis looking at the target (hole = the centre). The direction is the pseudo-Cartesian straight line (r, theta, phi treated as flat spherical),
  // without spacetime curvature: exact for the centre (-e_r), approximate for other targets near the hole. up = -e_theta projected.
  static void lookat_basis(const LookAt& la, double r, double th, double ph, Vec3 out[3]) {
    const double sT = std::sin(th), cT = std::cos(th), sP = std::sin(ph), cP = std::cos(ph);
    Vec3 f{-1, 0, 0};
    if (!la.hole) {
      const double tt = la.theta * M_PI / 180.0, tp = la.phi * M_PI / 180.0;
      const Vec3 T{la.r * std::sin(tt) * std::cos(tp), la.r * std::sin(tt) * std::sin(tp), la.r * std::cos(tt)};
      const Vec3 S{r * sT * cP, r * sT * sP, r * cT};
      const Vec3 d = T - S;
      const Vec3 er{sT * cP, sT * sP, cT}, et{cT * cP, cT * sP, -sT}, ep{-sP, cP, 0};
      const double n = std::sqrt(dot(d, d));
      if (n > 1e-12) f = (1.0 / n) * Vec3{dot(d, er), dot(d, et), dot(d, ep)};
    }
    Vec3 up0{0, -1, 0};
    Vec3 up = up0 - dot(up0, f) * f;
    double un = std::sqrt(dot(up, up));
    if (un < 1e-9) { up0 = Vec3{0, 0, 1}; up = up0 - dot(up0, f) * f; un = std::sqrt(dot(up, up)); }
    up = (1.0 / un) * up;
    Vec3 right{f.y * up.z - f.z * up.y, f.z * up.x - f.x * up.z, f.x * up.y - f.y * up.x};
    const double w = la.roll * M_PI / 180.0;
    const Vec3 r0 = right, u0 = up;
    right = std::cos(w) * r0 + std::sin(w) * u0; up = std::cos(w) * u0 - std::sin(w) * r0;
    out[0] = f; out[1] = right; out[2] = up;
  }
  // with the ship at (r, theta, phi): if there is a "lookat" use it, otherwise the view keyframes
  void view_basis(double t, const pilot::State& s, Vec3 out[3]) const {
    if (lookat.on) { lookat_basis(lookat, s.r, s.theta, s.phi, out); return; }
    view_basis(t, out);
  }
  // camera basis (fwd, right, up) in the (e_r, e_theta, e_phi) components of the local ZAMO
  void view_basis(double t, Vec3 out[3]) const {
    const quat::Q q = view_quat(t);
    const Vec3 e0[3] = {{-1, 0, 0}, {0, 0, 1}, {0, -1, 0}};
    for (int i = 0; i < 3; ++i) out[i] = quat::rotate(q, e0[i]);
  }
};

namespace detail {
inline bool keys_of(const json::Value* v, std::vector<Key>& out, const char* what, std::string& err) {
  out.clear();
  if (!v) return true;
  if (v->type == json::Value::Num) { out.push_back({0.0, v->n}); return true; }
  if (v->type != json::Value::Arr) { err = std::string(what) + ": number or list of {t, v}"; return false; }
  for (const auto& e : v->a) {
    if (e.type != json::Value::Obj || !e.get("t") || !e.get("v")) { err = std::string(what) + ": each keyframe needs t and v"; return false; }
    out.push_back({e.num("t", 0), e.num("v", 0)});
  }
  for (size_t i = 1; i < out.size(); ++i)
    if (!(out[i].t > out[i - 1].t)) { err = std::string(what) + ": keyframe times must increase"; return false; }
  return true;
}
}  // namespace detail

inline bool from_json(const json::Value& v, Route& r, std::string& err) {
  r = Route{};
  if (v.type != json::Value::Obj) { err = "the route must be a JSON object"; return false; }
  r.name = v.str("name", "route");
  r.fps = int(v.num("fps", 60));
  r.seconds = v.num("seconds", 10);
  if (r.fps <= 0 || r.fps > 1000) { err = "fps must be in 1..1000"; return false; }
  if (!(r.seconds > 0)) { err = "seconds must be > 0"; return false; }
  r.ease_smooth = v.str("ease", "smooth") != "linear";
  if (!detail::keys_of(v.get("spin"), r.spin, "spin", err) || !detail::keys_of(v.get("fov"), r.fov, "fov", err) ||
      !detail::keys_of(v.get("exposure"), r.exposure, "exposure", err)) return false;
  for (const auto& k : r.spin) if (k.v < 0 || k.v >= 1) { err = "spin must be in [0, 1)"; return false; }
  for (const auto& k : r.fov) if (k.v <= 1 || k.v >= 170) { err = "fov (degrees) must be in (1, 170)"; return false; }
  if (const json::Value* s = v.get("ship")) {
    r.mode = s->str("mode", "orbit");
    if (r.mode != "orbit" && r.mode != "infall" && r.mode != "zamo" && r.mode != "static") { err = "ship.mode: orbit | infall | zamo | static"; return false; }
    r.r0 = s->num("r", 30); r.theta0 = s->num("theta", 80); r.phi0 = s->num("phi", 0); r.timescale = s->num("timescale", 5);
    if (!(r.r0 > 2.5) || !(r.timescale > 0) || r.theta0 < 0.5 || r.theta0 > 179.5) { err = "ship: r > 2.5, theta em [0.5, 179.5], timescale > 0"; return false; }
    if (const json::Value* b = s->get("burns")) {
      if (b->type != json::Value::Arr) { err = "ship.burns must be a list"; return false; }
      for (const auto& e : b->a) {
        const json::Value* acc = e.get("acc");
        if (!acc || acc->type != json::Value::Arr || acc->a.size() != 3) { err = "burn: acc = [fwd, right, up]"; return false; }
        Burn bu; bu.t0 = e.num("t0", 0); bu.t1 = e.num("t1", 0); bu.acc = {acc->a[0].n, acc->a[1].n, acc->a[2].n};
        if (!(bu.t0 >= 0) || !(bu.t1 > bu.t0)) { err = "burn: 0 <= t0 < t1"; return false; }
        r.burns.push_back(bu);
      }
    }
  }
  if (const json::Value* s = v.get("ship")) {
    if (const json::Value* w = s->get("waypoints")) {
      if (w->type != json::Value::Arr || w->a.size() < 2) { err = "ship.waypoints: list of at least 2 {t, r, theta, phi}"; return false; }
      for (const auto& e : w->a) {
        if (!e.get("t") || !e.get("r")) { err = "waypoint: needs t and r (theta, phi optional)"; return false; }
        Waypoint q; q.t = e.num("t", 0); q.r = e.num("r", 0); q.theta = e.num("theta", 90); q.phi = e.num("phi", 0);
        if (!(q.r > 2.5) || q.theta < 0.5 || q.theta > 179.5) { err = "waypoint: r > 2.5, theta em [0.5, 179.5]"; return false; }
        if (!r.waypoints.empty() && !(q.t > r.waypoints.back().t)) { err = "waypoints: times must increase"; return false; }
        r.waypoints.push_back(q);
      }
      if (r.waypoints.front().t != 0.0) { err = "waypoints: the first one has t = 0 (it is the initial position)"; return false; }
      r.mode = s->boolean("camera_path", false) ? "path" : "follow";
      r.r0 = r.waypoints.front().r; r.theta0 = r.waypoints.front().theta; r.phi0 = r.waypoints.front().phi;
      r.follow_wn = s->num("follow_wn", 0); r.max_acc = s->num("max_acc", 0);
      if (!r.burns.empty()) { err = "waypoints and burns are mutually exclusive (the waypoints generate the thrusts)"; return false; }
    }
  }
  if (const json::Value* la = v.get("lookat")) {
    r.lookat.on = true; r.lookat.roll = la->num("roll", 0);
    const json::Value* t = la->get("target");
    if (t && t->type == json::Value::Arr && t->a.size() == 3) { r.lookat.hole = false; r.lookat.r = t->a[0].n; r.lookat.theta = t->a[1].n; r.lookat.phi = t->a[2].n; }
    else if (t && t->type == json::Value::Str && t->s == "hole") r.lookat.hole = true;
    else { err = "lookat.target: \"hole\" ou [r, theta, phi]"; return false; }
  }
  if (const json::Value* w = v.get("view")) {
    if (w->type != json::Value::Arr) { err = "view must be a list"; return false; }
    for (const auto& e : w->a) r.view.push_back({e.num("t", 0), e.num("yaw", 0), e.num("pitch", 0), e.num("roll", 0)});
    for (size_t i = 1; i < r.view.size(); ++i) if (!(r.view[i].t > r.view[i - 1].t)) { err = "view: times must increase"; return false; }
  }
  if (const json::Value* d = v.get("disk")) {
    auto& k = r.disk;
    k.hr = d->num("hr", 0); k.tau = d->num("tau", 10); k.flare = d->num("flare", 0); k.turb = d->num("turb", 0); k.turb_scale = d->num("turb_scale", 2);
    k.edge_soft = d->num("edge_soft", 0); k.edge_noise = d->num("edge_noise", 0); k.slowlight = d->boolean("slowlight", false);
    if (const json::Value* sp = d->get("spot")) { k.spot_r = sp->num("r", 10); k.spot_phi = sp->num("phi", 0); k.spot_size = sp->num("size", 1.5); k.spot_amp = sp->num("amp", 0); }
    if (const json::Value* fl = d->get("flare_pulse")) { k.flare_t = fl->num("t", 0); k.flare_w = fl->num("w", 20); k.flare_boost = fl->num("boost", 0); }
    if (const json::Value* tm = d->get("time")) { k.time_mode = tm->str("mode", "pilot"); k.rate = tm->num("rate", 20); k.t0 = tm->num("t0", 0); }
    if (k.time_mode != "pilot" && k.time_mode != "rate") { err = "disk.time.mode: pilot | rate"; return false; }
    if (k.hr < 0 || k.tau <= 0) { err = "disk: hr >= 0, tau > 0"; return false; }
  }
  // a variable spin changes the metric: only with a static camera
  if (r.spin.size() > 1 && r.mode != "static") { err = "variable spin is only allowed with ship.mode = \"static\" (the ship dynamics depend on the spin)"; return false; }
  return true;
}

// Instant (s of video) of sample k of N of frame f: stratified inside the shutter (fraction `shutter` of the frame, centred on the middle
// of the frame). Shutter 0 = all samples at the middle of the frame (no motion blur); 1 = the whole frame.
inline double sample_time(int fps, int f, int k, int N, double shutter) {
  return (double(f) + 0.5) / fps + shutter / fps * ((double(k) + 0.5) / N - 0.5);
}

// Ship trajectory on a fine grid (dt = 1 / (fps * sub) s of video), integrated with the Pilot (geodesic + impulses); deterministic.
struct Trajectory {
  double dt = 0, a = 0;
  std::vector<pilot::State> pts;
  std::vector<pilot::Consts> consts;
  bool captured = false;
  double t_capture = 0;
  std::vector<double> acc;   // v1.0b: proper acceleration (c/M) applied at each step ("follow" mode only)
  double acc_max = 0, acc_rms = 0, err_max = 0;  // maximum, rms, and largest position error (M, proper length) relative to the spline

  pilot::State at(double t) const {
    const double x = std::min(std::max(t, 0.0), dt * double(pts.size() - 1)) / dt;
    const size_t i = std::min(size_t(x), pts.size() - 2);
    const double f = x - double(i);
    const auto &p = pts[i], &q = pts[i + 1];
    auto L = [&](double u, double w) { return u + f * (w - u); };
    return {L(p.r, q.r), L(p.theta, q.theta), L(p.phi, q.phi), {L(p.beta.x, q.beta.x), L(p.beta.y, q.beta.y), L(p.beta.z, q.beta.z)}, L(p.tau, q.tau), L(p.t, q.t)};
  }
};


// v1.0b: C1 Hermite spline (tangents by finite differences) through the pseudo-Cartesian positions of the waypoints; returns P(t) and dP/dt (per s of video)
struct Spline {
  std::vector<double> t; std::vector<Vec3> P, m;
  explicit Spline(const std::vector<Waypoint>& w) {
    for (const auto& q : w) {
      const double th = q.theta * M_PI / 180.0, ph = q.phi * M_PI / 180.0;
      t.push_back(q.t); P.push_back({q.r * std::sin(th) * std::cos(ph), q.r * std::sin(th) * std::sin(ph), q.r * std::cos(th)});
    }
    const size_t n = t.size(); m.resize(n);
    if (n == 2) { m[0] = m[1] = (1.0 / (t[1] - t[0])) * (P[1] - P[0]); return; }
    // tangents: derivative of the 2nd-degree Lagrange polynomial through 3 points (exact for parabolas; interior and ends), 2nd order
    auto d3 = [&](size_t i0, size_t i1, size_t i2, size_t at) {  // derivative at t[at] of the quadratic through (i0, i1, i2)
      const double x = t[at], a = t[i0], b = t[i1], c = t[i2];
      const double l0 = ((x - b) + (x - c)) / ((a - b) * (a - c)), l1 = ((x - a) + (x - c)) / ((b - a) * (b - c)), l2 = ((x - a) + (x - b)) / ((c - a) * (c - b));
      return l0 * P[i0] + l1 * P[i1] + l2 * P[i2];
    };
    m[0] = d3(0, 1, 2, 0); m[n - 1] = d3(n - 3, n - 2, n - 1, n - 1);
    for (size_t i = 1; i + 1 < n; ++i) m[i] = d3(i - 1, i, i + 1, i);
  }
  void eval(double tt, Vec3& p, Vec3& v) const {
    tt = std::min(std::max(tt, t.front()), t.back());
    size_t i = 1; while (i + 1 < t.size() && t[i] < tt) ++i;
    const double h = t[i] - t[i - 1], u = (tt - t[i - 1]) / h, u2 = u * u, u3 = u2 * u;
    const double h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    p = h00 * P[i - 1] + (h10 * h) * m[i - 1] + h01 * P[i] + (h11 * h) * m[i];
    const double d00 = (6 * u2 - 6 * u) / h, d10 = 3 * u2 - 4 * u + 1, d01 = (-6 * u2 + 6 * u) / h, d11 = 3 * u2 - 2 * u;
    v = d00 * P[i - 1] + d10 * m[i - 1] + d01 * P[i] + d11 * m[i];
  }
};

// pseudo-Cartesian position/velocity (per unit proper time) -> (r, theta, phi) and local spatial four-velocity (gamma beta) in the ZAMO
// (e_r, e_theta, e_phi): u = (sqrt(Sigma/Delta) rdot, sqrt(Sigma) thetadot, varpi (phidot - omega u^t)), u^t = gamma/alpha, gamma = sqrt(1 + |u|^2) (fixed point)
inline void local_target(double a, const Vec3& P, const Vec3& V, double& r, double& th, double& ph, Vec3& u) {
  r = std::sqrt(dot(P, P)); th = std::acos(std::clamp(P.z / r, -1.0, 1.0)); ph = std::atan2(P.y, P.x);
  const double rd = dot(P, V) / r, st = std::max(1e-9, std::sin(th));
  const double thd = (P.z * rd / r - V.z) / (r * st), phd = (P.x * V.y - P.y * V.x) / std::max(1e-18, P.x * P.x + P.y * P.y);
  const pilot::Zamo z = pilot::zamo(a, r, th);
  u = {std::sqrt(z.Sigma / z.Delta) * rd, std::sqrt(z.Sigma) * thd, 0};
  double g = std::sqrt(1.0 + u.x * u.x + u.y * u.y);
  for (int it = 0; it < 8; ++it) {
    u.z = z.varpi * (phd - z.omega * g / z.alpha);
    g = std::sqrt(1.0 + dot(u, u));
  }
}
// delta p in the pilot frame (boost by -beta) that takes the local four-velocity (gamma beta) to u_new: input of Pilot::thrust
inline Vec3 rest_frame_impulse(const Vec3& beta, const Vec3& u_new) {
  const double b2 = dot(beta, beta), E = std::sqrt(1.0 + dot(u_new, u_new));
  if (b2 < 1e-30) return u_new;
  const double g = 1.0 / std::sqrt(1.0 - b2);
  const double k = (g - 1.0) * dot(u_new, beta) / b2 - g * E;
  return u_new + k * beta;
}

inline Trajectory build_trajectory_follow(const Route& r, int sub) {
  Trajectory tr;
  tr.dt = 1.0 / (double(r.fps) * sub);
  tr.a = r.spin_value(0.0);
  const size_t n = size_t(std::ceil(r.seconds / tr.dt)) + 2;
  const Spline sp(r.waypoints);
  const double dtau = tr.dt * r.timescale;
  const double wn = r.follow_wn > 0 ? r.follow_wn : 2.0 / r.timescale;
  Vec3 P0, V0; sp.eval(0.0, P0, V0);
  double r0, th0, ph0; Vec3 u0;
  local_target(tr.a, P0, (1.0 / r.timescale) * V0, r0, th0, ph0, u0);
  const double g0 = std::sqrt(1.0 + dot(u0, u0));
  pilot::Pilot p(tr.a, r0, th0, ph0, (1.0 / g0) * u0);
  tr.pts.push_back(p.state()); tr.consts.push_back(p.consts()); tr.acc.push_back(0.0);
  double s2 = 0; size_t ns = 0;
  for (size_t i = 1; i < n; ++i) {
    double acc = 0;
    if (!p.captured()) {
      const double tn = double(i) * tr.dt;                      // end of this step
      Vec3 Pt, Vt; sp.eval(tn, Pt, Vt);
      double rt, tht, pht; Vec3 ut; local_target(tr.a, Pt, (1.0 / r.timescale) * Vt, rt, tht, pht, ut);
      pilot::Pilot free_p = p; free_p.advance(dtau);          // what free fall would do in this step
      const pilot::State sf = free_p.state(), sc = p.state();
      const double gf = 1.0 / std::sqrt(1.0 - dot(sf.beta, sf.beta)), gc = 1.0 / std::sqrt(1.0 - dot(sc.beta, sc.beta));
      const Vec3 u_free = gf * sf.beta;
      // predicted position error at the end of the step, in local proper length (BL -> proper: sqrt(Sigma/Delta) dr, sqrt(Sigma) dtheta, varpi dphi)
      const pilot::Zamo z = pilot::zamo(tr.a, sf.r, sf.theta);
      double dph = pht - sf.phi; dph -= 2.0 * M_PI * std::round(dph / (2.0 * M_PI));
      const Vec3 e{std::sqrt(z.Sigma / z.Delta) * (rt - sf.r), std::sqrt(z.Sigma) * (tht - sf.theta), z.varpi * dph};
      tr.err_max = std::max(tr.err_max, std::sqrt(dot(e, e)));
      // desired velocity at the end of the step: the target's plus a correction proportional to the error (1st-order position loop, rate wn); velocity impulse (deadbeat)
      const Vec3 u_des = ut + wn * e;
      Vec3 d = rest_frame_impulse(sc.beta, u_des - u_free + gc * sc.beta);  // impulse that, added to the free fall, gives u_des
      double dn = std::sqrt(dot(d, d));
      if (r.max_acc > 0 && dn / dtau > r.max_acc) { d = (r.max_acc * dtau / dn) * d; dn = r.max_acc * dtau; }
      if (dn > 1e-15) p.thrust((1.0 / dtau) * d, dtau);
      acc = dn / dtau; tr.acc_max = std::max(tr.acc_max, acc); s2 += acc * acc; ++ns;
      p.advance(dtau);
      if (p.captured() && !tr.captured) { tr.captured = true; tr.t_capture = tn; }
    }
    tr.pts.push_back(p.state()); tr.consts.push_back(p.consts()); tr.acc.push_back(acc);
  }
  tr.acc_rms = ns ? std::sqrt(s2 / double(ns)) : 0.0;
  return tr;
}

// v1.1: virtual camera (not a ship): the position (r, theta, phi) follows the waypoint spline exactly, without dynamics; static observer (beta = 0).
// It is what the time-lapse uses for the global camera C(t), continuous across versions.
inline Trajectory build_trajectory_path(const Route& r, int sub) {
  Trajectory tr;
  tr.dt = 1.0 / (double(r.fps) * sub);
  tr.a = r.spin_value(0.0);
  const size_t n = size_t(std::ceil(r.seconds / tr.dt)) + 2;
  const Spline sp(r.waypoints);
  for (size_t i = 0; i < n; ++i) {
    Vec3 P, V; sp.eval(double(i) * tr.dt, P, V);
    const double rr = std::sqrt(dot(P, P));
    const double t = double(i) * tr.dt * r.timescale;
    tr.pts.push_back({rr, std::acos(std::clamp(P.z / rr, -1.0, 1.0)), std::atan2(P.y, P.x), {0, 0, 0}, t, t});
    tr.consts.push_back({tr.a, 1, 0, 0});
  }
  return tr;
}

inline Trajectory build_trajectory(const Route& r, int sub = 16) {
  if (r.mode == "path") return build_trajectory_path(r, sub);
  if (r.mode == "follow") return build_trajectory_follow(r, sub);
  Trajectory tr;
  tr.dt = 1.0 / (double(r.fps) * sub);
  tr.a = r.spin_value(0.0);
  const size_t n = size_t(std::ceil(r.seconds / tr.dt)) + 2;
  const double th = r.theta0 * M_PI / 180.0, ph = r.phi0 * M_PI / 180.0;
  if (r.mode == "static") {
    for (size_t i = 0; i < n; ++i) { const double t = double(i) * tr.dt; tr.pts.push_back({r.r0, th, ph, {0, 0, 0}, t * r.timescale, t * r.timescale}); tr.consts.push_back({tr.a, 1, 0, 0}); }
    return tr;
  }
  Vec3 beta{0, 0, 0};
  if (r.mode == "orbit") beta = {0, 0, lorentz::circular_beta(tr.a, r.r0, th)};
  else if (r.mode == "infall") beta = {-lorentz::infall_beta(r.r0), 0, 0};
  pilot::Pilot p(tr.a, r.r0, th, ph, beta);
  tr.pts.push_back(p.state()); tr.consts.push_back(p.consts());
  const double dtau = tr.dt * r.timescale;
  for (size_t i = 1; i < n; ++i) {
    if (!p.captured()) {
      const double tm = (double(i) - 0.5) * tr.dt;  // mean instant of the step
      for (const auto& b : r.burns)
        if (tm >= b.t0 && tm < b.t1) { p.thrust(Vec3{-b.acc.x, -b.acc.z, b.acc.y}, dtau); break; }  // (fwd, right, up) -> (e_r, e_theta, e_phi)
      p.advance(dtau);
      if (p.captured() && !tr.captured) { tr.captured = true; tr.t_capture = double(i) * tr.dt; }
    }
    tr.pts.push_back(p.state()); tr.consts.push_back(p.consts());
  }
  return tr;
}

}  // namespace bhs::route
