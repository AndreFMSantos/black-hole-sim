#pragma once
// Pilot: a massive observer falling on a Kerr timelike geodesic (M = 1) that can apply thrust.
//
// Timelike geodesic (mass 1), constants E (energy), L (L_z), Q (Carter). In Mino time
// (d tau_proper = Sigma d lambda) the equations separate, like those of light but with mass terms:
//   (dr/dl)^2  = R(r)  = [E(r^2+a^2) - aL]^2 - Delta [ r^2 + (L - aE)^2 + Q ]
//   (dmu/dl)^2 = M(mu) = (1-mu^2) Q - a^2 (1-E^2) mu^2 (1-mu^2) - L^2 mu^2          mu = cos(theta)
//   dphi/dl    = -aE + L/(1-mu^2) + a [E(r^2+a^2) - aL]/Delta
//   dt/dl      = -a (aE (1-mu^2) - L) + (r^2+a^2) [E(r^2+a^2) - aL]/Delta
// integrated in the 2nd-order form r'' = R'/2, mu'' = M'/2 with reprojection onto the first integrals
// (the same recipe as light in kerr.hpp). The pilot's orientation is that of the local ZAMO frame
// (e_r, e_theta, e_phi) boosted by the velocity: it is NOT Fermi-Walker transported (approximation).
//
// Thrust: in the pilot's rest frame the 4-velocity (1, 0) becomes (sqrt(1+d^2), d), with
// d = a_prop * d_tau; boost back to the ZAMO and recompute E, L, Q. Over a short interval
// this is a constant proper acceleration (addition of rapidities).
#include <algorithm>
#include <array>
#include <cmath>

#include "kerr.hpp"
#include "vec3.hpp"

namespace bhs::pilot {

struct Zamo {
  double alpha, omega, varpi, Sigma, Delta;
};
inline Zamo zamo(double a, double r, double theta) {
  const double s = std::sin(theta), mu = std::cos(theta);
  const double Sigma = r * r + a * a * mu * mu, Delta = kerr::delta(r, a);
  const double A = std::pow(r * r + a * a, 2) - a * a * Delta * s * s;
  return {std::sqrt(Sigma * Delta / A), 2.0 * a * r / A, std::sqrt(A / Sigma) * s, Sigma, Delta};
}

struct Consts {
  double a, E, L, Q;
};

inline double R(double r, const Consts& c) {
  const double A = c.E * (r * r + c.a * c.a) - c.a * c.L;
  return A * A - kerr::delta(r, c.a) * (r * r + (c.L - c.a * c.E) * (c.L - c.a * c.E) + c.Q);
}
inline double dR(double r, const Consts& c) {
  const double A = c.E * (r * r + c.a * c.a) - c.a * c.L;
  const double K = r * r + (c.L - c.a * c.E) * (c.L - c.a * c.E) + c.Q;
  return 4.0 * r * c.E * A - (2.0 * r - 2.0) * K - kerr::delta(r, c.a) * 2.0 * r;
}
inline double Mmu(double mu, const Consts& c) {
  const double s2 = 1.0 - mu * mu;
  return s2 * c.Q - c.a * c.a * (1.0 - c.E * c.E) * mu * mu * s2 - c.L * c.L * mu * mu;
}
inline double dMmu(double mu, const Consts& c) {
  return -2.0 * mu * c.Q - c.a * c.a * (1.0 - c.E * c.E) * (2.0 * mu - 4.0 * mu * mu * mu) - 2.0 * c.L * c.L * mu;
}
inline double dphi_dl(double r, double mu, const Consts& c) {
  const double om = std::max(1e-14, 1.0 - mu * mu);
  return -c.a * c.E + c.L / om + c.a * (c.E * (r * r + c.a * c.a) - c.a * c.L) / kerr::delta(r, c.a);
}
inline double dt_dl(double r, double mu, const Consts& c) {
  const double A = c.E * (r * r + c.a * c.a) - c.a * c.L;
  return -c.a * (c.a * c.E * (1.0 - mu * mu) - c.L) + (r * r + c.a * c.a) * A / kerr::delta(r, c.a);
}

struct State {
  double r, theta, phi;  // position (Boyer-Lindquist)
  Vec3 beta;             // velocity in the local ZAMO, components (e_r, e_theta, e_phi), in c
  double tau, t;         // accumulated proper and coordinate time
};

class Pilot {
 public:
  Pilot(double a, double r, double theta, double phi, Vec3 beta) : a_(a) {
    r_ = r; mu_ = std::cos(theta); phi_ = phi; tau_ = 0; t_ = 0;
    set_velocity(beta);
  }

  bool captured() const { return captured_; }
  const Consts& consts() const { return c_; }

  State state() const {
    const double th = std::acos(std::clamp(mu_, -1.0, 1.0));
    return {r_, th, phi_, beta_now(), tau_, t_};
  }

  // Free fall during dtau of proper time.
  void advance(double dtau, double atol = 1e-11, double rtol = 1e-10) {
    using V = std::array<double, 7>;  // r, vr, mu, vmu, phi, tau, t
    V y{r_, vr_, mu_, vmu_, phi_, tau_, t_};
    const double target = tau_ + dtau, r_plus = kerr::horizon(a_);
    auto f = [&](const V& s) {
      const double Sig = s[0] * s[0] + a_ * a_ * s[2] * s[2];
      return V{s[1], 0.5 * dR(s[0], c_), s[3], 0.5 * dMmu(s[2], c_), dphi_dl(s[0], s[2], c_), Sig, dt_dl(s[0], s[2], c_)};
    };
    double h = h_;
    for (int it = 0; it < 200000 && target - y[5] > 1e-13 * std::max(1.0, std::abs(target)); ++it) {
      const double Sig = y[0] * y[0] + a_ * a_ * y[2] * y[2];
      double hh = std::min(h, (target - y[5]) / Sig);
      V yn, k1, k7;
      const double err = kerr::dp45_step(f, y, hh, yn, k1, k7, atol, rtol);
      if (err > 1.0) { h = hh * kerr::step_factor(err); continue; }
      // reprojection onto the first integrals (keeps the sign of the velocities)
      yn[1] = std::copysign(std::sqrt(std::max(R(yn[0], c_), 0.0)), yn[1]);
      yn[3] = std::copysign(std::sqrt(std::max(Mmu(yn[2], c_), 0.0)), yn[3]);
      y = yn;
      h = std::max(hh * kerr::step_factor(err), 1e-12);
      if (y[0] < r_plus + 1e-3) { captured_ = true; break; }
    }
    r_ = y[0]; vr_ = y[1]; mu_ = y[2]; vmu_ = y[3]; phi_ = y[4]; tau_ = y[5]; t_ = y[6]; h_ = h;
  }

  // Thrust: proper acceleration `accel` (vector, in the pilot frame, boosted ZAMO axes)
  // during dtau; it only changes the velocity (the position stays), like a short impulse.
  void thrust(Vec3 accel, double dtau) {
    const Vec3 b = beta_now();
    const Vec3 d = dtau * accel;
    const double E0 = std::sqrt(1.0 + dot(d, d));  // 4-velocity (E0, d) in the pilot frame
    const double b2 = dot(b, b);
    double Ez, px = d.x, py = d.y, pz = d.z;
    if (b2 < 1e-30) {
      Ez = E0;
    } else {
      const double g = 1.0 / std::sqrt(1.0 - b2);
      const double bd = dot(b, d);
      Ez = g * (E0 + bd);
      const double k = (g - 1.0) * bd / b2 + g * E0;  // p = p' + [(g-1)(p'.b^)/|b| ... + g E0] b
      px += k * b.x; py += k * b.y; pz += k * b.z;
    }
    set_velocity({px / Ez, py / Ez, pz / Ez});
  }

 private:
  Vec3 beta_now() const {
    const auto z = zamo(a_, r_, std::acos(std::clamp(mu_, -1.0, 1.0)));
    const double s = std::sqrt(std::max(1e-28, 1.0 - mu_ * mu_));
    const double pr = vr_ / z.Delta, pth = -vmu_ / s;
    const double gamma = (c_.E - z.omega * c_.L) / z.alpha;
    return {pr * std::sqrt(z.Delta / z.Sigma) / gamma, pth / (std::sqrt(z.Sigma) * gamma), c_.L / (z.varpi * gamma)};
  }

  void set_velocity(Vec3 beta) {
    const double theta = std::acos(std::clamp(mu_, -1.0, 1.0));
    const auto z = zamo(a_, r_, theta);
    const double g = 1.0 / std::sqrt(1.0 - dot(beta, beta));
    c_.a = a_;
    c_.E = g * (z.alpha + z.omega * z.varpi * beta.z);
    c_.L = g * z.varpi * beta.z;
    const double pr = g * beta.x * std::sqrt(z.Sigma / z.Delta), pth = g * beta.y * std::sqrt(z.Sigma);
    c_.Q = pth * pth + mu_ * mu_ * (a_ * a_ * (1.0 - c_.E * c_.E) + c_.L * c_.L / std::max(1e-14, 1.0 - mu_ * mu_));
    vr_ = z.Delta * pr;
    vmu_ = -std::sin(theta) * pth;
    h_ = 0.02 / std::max(1.0, r_);
  }

  double a_, r_, mu_, phi_, tau_, t_, vr_ = 0, vmu_ = 0, h_ = 1e-3;
  Consts c_{0, 1, 0, 0};
  bool captured_ = false;
};

}  // namespace bhs::pilot
