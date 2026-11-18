#pragma once
// v0.8b — host-side flight logic (double precision): pilot (bhs::pilot::Pilot), camera orientation,
// thrust, HUD and the clip script. No CUDA, no GL: testable on the host only.
#include <bhs/kerr.hpp>
#include <bhs/lorentz.hpp>
#include <bhs/pilot.hpp>
#include <bhs/vec3.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace bhs::flight {

// Camera basis in the observer frame's (e_r, e_theta, e_phi) (axes of the boosted ZAMO). Default = v0.6b:
// fwd = -e_r (looking at the hole), right = +e_phi, up = -e_theta. NOT Fermi-Walker transported.
struct Basis {
  Vec3 fwd{-1, 0, 0}, right{0, 0, 1}, up{0, -1, 0};
};
inline Vec3 rotate(Vec3 v, Vec3 axis, double ang) {  // Rodrigues; axis unitario
  const double c = std::cos(ang), s = std::sin(ang);
  return c * v + s * cross(axis, v) + (1.0 - c) * dot(axis, v) * axis;
}
inline void orthonormalize(Basis& b) {
  b.fwd = normalize(b.fwd);
  b.right = normalize(b.right - dot(b.right, b.fwd) * b.fwd);
  b.up = cross(b.fwd, b.right);  // right-handed: right x up = fwd? see note: with the default basis fwd x right = (-1,0,0)x(0,0,1) = (0,1,0) = -up
  b.up = -1.0 * b.up;            // preserves the handedness of the default basis (up = -e_theta)
}
// positive: yaw to the right (fwd -> right), pitch up (fwd -> up), roll in the direction right -> up
inline void yaw(Basis& b, double ang) { const Vec3 f = b.fwd, r = b.right; b.fwd = std::cos(ang) * f + std::sin(ang) * r; b.right = std::cos(ang) * r - std::sin(ang) * f; }
inline void pitch(Basis& b, double ang) { const Vec3 f = b.fwd, u = b.up; b.fwd = std::cos(ang) * f + std::sin(ang) * u; b.up = std::cos(ang) * u - std::sin(ang) * f; }
inline void roll(Basis& b, double ang) { const Vec3 r = b.right, u = b.up; b.right = std::cos(ang) * r + std::sin(ang) * u; b.up = std::cos(ang) * u - std::sin(ang) * r; }

enum class Start { Orbit, Zamo, Infall };

struct Controls {
  double fwd = 0, right = 0, up = 0;  // -1..1 (W/S, D/A, Space/Shift)
  bool brake = false;
};

// proper-acceleration levels (c/M): 0.002 * 10^(i/3), i = 0..6 -> 0.002 ... 0.2; default 0.02 (i = 3)
inline double accel_level(int i) { return 0.002 * std::pow(10.0, std::clamp(i, 0, 6) / 3.0); }

class Flight {
 public:
  Flight(double a, double r, double theta_deg, double phi_deg, Start st) : a_(a), r0_(r), th0_(theta_deg * M_PI / 180.0), ph0_(phi_deg * M_PI / 180.0), start_(st) { reset(); }

  void reset() {
    Vec3 b{0, 0, 0};
    if (start_ == Start::Orbit) b = {0, 0, lorentz::circular_beta(a_, r0_, th0_)};
    else if (start_ == Start::Infall) b = {-lorentz::infall_beta(r0_), 0, 0};
    pilot_ = pilot::Pilot(a_, r0_, th0_, ph0_, b);
    basis_ = Basis{};
    tau_prop_ = 0;
  }

  // dt_real in real seconds; one step = proper time dt_real * timescale
  void step(double dt_real, const Controls& c, bool frozen) {
    if (frozen || pilot_.captured()) return;
    const double dtau = dt_real * timescale;
    Vec3 acc{0, 0, 0};
    const double lvl = accel_level(level);
    if (c.fwd != 0 || c.right != 0 || c.up != 0) {
      Vec3 d = c.fwd * basis_.fwd + c.right * basis_.right + c.up * basis_.up;
      const double n = norm(d);
      acc = (lvl / std::max(1.0, n)) * d;
    }
    if (c.brake) {  // opposes the velocity relative to the ZAMO; magnitude limited so as not to reverse direction in one step
      const Vec3 b = pilot_.state().beta;
      const double bn = norm(b);
      if (bn > 1e-9) acc = acc + (-std::min(2.0 * lvl, bn / dtau) / bn) * b;
    }
    if (norm(acc) > 0) pilot_.thrust(acc, dtau);
    pilot_.advance(dtau);
  }
  // step with a given acceleration in the (fwd, right, up) coordinates of the current basis (used by the scripted clip)
  void step_accel(double dt_real, Vec3 comp) {
    if (pilot_.captured()) return;
    const double dtau = dt_real * timescale;
    const Vec3 acc = comp.x * basis_.fwd + comp.y * basis_.right + comp.z * basis_.up;
    if (norm(acc) > 0) pilot_.thrust(acc, dtau);
    pilot_.advance(dtau);
  }
  // camera rotations (rad)
  void look(double yaw_rad, double pitch_rad, double roll_rad) {
    if (yaw_rad != 0) yaw(basis_, yaw_rad);
    if (pitch_rad != 0) pitch(basis_, pitch_rad);
    if (roll_rad != 0) roll(basis_, roll_rad);
    if (++renorm_ % 64 == 0) orthonormalize(basis_);
  }

  const pilot::Pilot& pilot() const { return pilot_; }
  pilot::State state() const { return pilot_.state(); }
  const Basis& basis() const { return basis_; }
  bool captured() const { return pilot_.captured(); }
  double a() const { return a_; }

  double timescale = 5.0;  // proper time per real second
  int level = 3;           // index of the acceleration level

 private:
  double a_, r0_, th0_, ph0_;
  Start start_;
  pilot::Pilot pilot_{0, 10, M_PI / 2, 0, {0, 0, 0}};
  Basis basis_;
  double tau_prop_ = 0;
  unsigned renorm_ = 0;
};

// Frequency shift of the central ray: sky -> nu_obs/nu_inf = dop * E_loc; disk -> g * dop; horizon -> no value
struct CenterShift {
  int kind = 0;  // 0 sky, 1 disk, 2 horizon
  double factor = 1.0;
};
inline CenterShift center_shift(double a, const pilot::State& s, const Basis& b, double rin, double rout) {
  const Vec3 np = -1.0 * b.fwd;  // photon arriving through the centre: n' = -d' = -fwd
  const auto pz = lorentz::observer_to_zamo(np, s.beta);
  kerr::Options ko; ko.rtol = 1e-8; ko.atol = 1e-9;
  const auto tr = kerr::trace(a, s.r, s.theta, pz.n.x, pz.n.y, pz.n.z, kerr::Disk{rin, rout}, ko);
  CenterShift c;
  if (tr.disk_hit) { c.kind = 1; c.factor = kerr::redshift(a, tr.disk_r, tr.L, s.r, s.theta) * pz.dop; }
  else if (tr.fate == kerr::Fate::Escaped) { c.kind = 0; c.factor = pz.dop * kerr::init_from_zamo(a, s.r, s.theta, pz.n.x, pz.n.y, pz.n.z).E_loc; }
  else c.kind = 2;
  return c;
}

// HUD text (lines separated by \n; those starting with '!' are warnings in red)
inline std::string hud_text(const char* mode, double a, const pilot::State& s, const pilot::Consts& c, const CenterShift& cs, double lvl, double timescale,
                            bool captured, bool frozen, bool has_dynamics) {
  char buf[1024];
  const double b = norm(s.beta), gamma = 1.0 / std::sqrt(1.0 - b * b);
  const double rp = kerr::horizon(a), isco = kerr::isco(a);
  std::string o;
  auto add = [&](const char* fmt, auto... v) { std::snprintf(buf, sizeof buf, fmt, v...); o += buf; o += '\n'; };
  add("MODE %s   a=%.3f   M=1", mode, a);
  add("r=%.3f  th=%.2f  ph=%.1f deg", s.r, s.theta * 180 / M_PI, std::remainder(s.phi * 180 / M_PI, 360.0));
  add("r-r+=%.3f  r-ISCO=%+.3f", s.r - rp, s.r - isco);
  add("|beta|=%.4f  gamma=%.4f", b, gamma);
  if (has_dynamics) {
    add("tau=%.2f  t=%.2f", s.tau, s.t);
    add("E=%.5f  L=%+.4f  Q=%.4f", c.E, c.L, c.Q);
    add("thrust %.4f c/M   time x%.1f M/s%s", lvl, timescale, frozen ? "  [PAUSED]" : "");
  }
  if (cs.kind == 2) add("central ray: horizon");
  else if (cs.kind == 1) add("disk: g = nu_obs/nu_em = %.4f", cs.factor);
  else add("sky ahead: nu/nu_inf = %.4f", cs.factor);
  if (captured) add("!HORIZON: SIMULATION PAUSED  (Backspace resets)");
  if (!o.empty()) o.pop_back();
  return o;
}

// Scripted clip (--clip flight): (instant in PROPER time, acceleration in the (fwd, right, up) coordinates of the default basis).
// Windows obtained by simulating with Pilot (a = 0.9, r = 30, theta = 80, circular orbit, dt = 8.5/60): brake, push
// inwards, free fall (periastron r = 4.4 at tau = 87), outward thrust, and drift to r = 14.6. timescale 8.5 M/s.
struct Burn { double t0, t1; Vec3 acc; };
inline const std::vector<Burn>& flight_script() {
  // Windows obtained by a CONDITIONAL simulation with this same class (a = 0.9, r = 30, theta = 80, circular orbit, 60 fps,
  // timescale 8.5 M/s; thresholds: brake until beta_phi = 0.04, push inwards until beta_r = -0.25, outward thrust when
  // r < 5.5 until beta_r > 0, then 0.06 until beta_r = 0.35, the last burn extended by hand until tau = 108) and placed midway between two frames, so that the list of times
  // reproduces exactly that trajectory (periastron r = 4.41 at tau = 87.1; final r 14.6).
  static const std::vector<Burn> s = {
      {4.04, 7.72, {0, -0.04, 0}},       // brake: removes the angular momentum (-e_phi)
      {7.72, 11.97, {0.06, 0, 0}},       // pushes towards the hole (+fwd = -e_r)
      {82.95, 87.20, {-0.2, 0, 0}},      // strong outward thrust when approaching r ~ 5.5
      {87.20, 108.00, {-0.06, 0, 0}},    // keeps moving away (until beta_r ~ 0.35; then drifts)
  };
  return s;
}
inline Vec3 script_accel(double tau) {
  for (const auto& b : flight_script()) if (tau >= b.t0 && tau < b.t1) return b.acc;
  return {0, 0, 0};
}

}  // namespace bhs::flight
