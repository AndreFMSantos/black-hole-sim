#pragma once
// Wormhole (v2.1d): ds^2 = -dt^2 + dl^2 + r(l)^2 dOmega^2 (James, von Tunzelmann, Franklin & Thorne 2015, Am. J. Phys. 83, 486,
// arXiv:1502.03809, eq. 1-7 and appendix; read by a subagent). Spherically symmetric: every geodesic is planar.
//   Ellis (1973): r = sqrt(rho^2 + l^2)  (`ellis = true`, throat of radius rho = b0).
//   Interstellar (three parameters rho, a, M): r = rho for |l| <= a; for |l| > a, r = rho + M [x arctan x - 1/2 ln(1+x^2)], x = 2(|l|-a)/(pi M).
//     Lens width W = 1.42953 M. In the film: rho = 1 km, 2a = 0.01 rho, W = 0.05 rho.
// Null geodesic (appendix A, parameter t, p_t = -1): with B = r sqrt(n_theta^2 + n_phi^2) the total angular momentum (constant),
//   dl/dt = p_l,  dp_l/dt = B^2 r'(l) / r^3,  dpsi/dt = B / r^2,   and H = 0: p_l^2 + B^2/r^2 = 1.
// The 2nd-order form in (l, p_l) is regular at the turning point (the form (dl/dt)^2 = 1 - B^2/r^2 is not). psi is the angle swept in the geodesic's plane. The side
// where the ray ends (sign of l) picks the sky: l < 0 is the "Saturn sky", l > 0 the "Gargantua" one in the film.
#include <algorithm>
#include <cmath>

namespace bhs::wh {

struct Params {
  double rho = 1.0, a = 0.0, M = 0.0;
  bool ellis = false;
};

inline double r_of_l(const Params& p, double l) {
  if (p.ellis) return std::sqrt(p.rho * p.rho + l * l);
  const double al = std::fabs(l);
  if (al <= p.a) return p.rho;
  const double x = 2.0 * (al - p.a) / (M_PI * p.M);
  return p.rho + p.M * (x * std::atan(x) - 0.5 * std::log1p(x * x));
}
// dr/dl (continuous at |l| = a: both expressions give 0)
inline double drdl(const Params& p, double l) {
  if (p.ellis) return l / std::sqrt(p.rho * p.rho + l * l);
  const double al = std::fabs(l);
  if (al <= p.a) return 0.0;
  const double x = 2.0 * (al - p.a) / (M_PI * p.M);
  return std::copysign((2.0 / M_PI) * std::atan(x), l);
}
inline double lensing_width(const Params& p) { return 1.42953 * p.M; }  // W (eq. 7 of the paper; numerical value from the paper)

struct Result {
  int side;        // +1 / -1: sign of l at the end (which sky)
  double psi;      // angle swept in the geodesic plane (rad), including the asymptotic tail
  double B;        // total angular momentum (constant)
  double h_err;    // max |p_l^2 + B^2/r^2 - 1| along the path (conservation of H = 0)
  int steps;
  bool ok;
};

// Ray traced backwards from the camera at rest at l_c: `alpha` is the angle between the viewing direction and +e_l (cos alpha = initial p_l);
// B = r(l_c) sin(alpha). Ends when |l| > l_max (flat tail corrected: psi += asin(B / r)).
inline Result trace(const Params& p, double l_c, double alpha, double l_max = 1e5, int max_steps = 2000000) {
  const double r0 = r_of_l(p, l_c);
  Result res{0, 0.0, r0 * std::sin(alpha), 0.0, 0, false};
  const double B = res.B, B2 = B * B;
  double l = l_c, pl = std::cos(alpha), psi = 0.0;
  auto f = [&](double ll, double ppl, double* d) {
    const double r = r_of_l(p, ll);
    d[0] = ppl; d[1] = B2 * drdl(p, ll) / (r * r * r); d[2] = B / (r * r);
  };
  for (int it = 0; it < max_steps; ++it) {
    if (std::fabs(l) > l_max) {
      const double r = r_of_l(p, l);
      res.side = l > 0 ? 1 : -1;
      // tail: only if moving away (p_l and l with the same sign); otherwise (coming from infinity) there is no useful tail
      res.psi = psi + std::asin(std::min(1.0, B / r));
      res.steps = it; res.ok = true;
      return res;
    }
    const double r = r_of_l(p, l), h = 0.01 * std::min(r, 5.0 * p.rho + 0.2 * std::fabs(l)) + 1e-12;  // step ~ local scale
    double k1[3], k2[3], k3[3], k4[3];
    f(l, pl, k1);
    f(l + 0.5 * h * k1[0], pl + 0.5 * h * k1[1], k2);
    f(l + 0.5 * h * k2[0], pl + 0.5 * h * k2[1], k3);
    f(l + h * k3[0], pl + h * k3[1], k4);
    l += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    pl += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    psi += h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    const double rr = r_of_l(p, l);
    res.h_err = std::max(res.h_err, std::fabs(pl * pl + B2 / (rr * rr) - 1.0));
  }
  res.steps = max_steps;
  return res;
}

}  // namespace bhs::wh
