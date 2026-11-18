#pragma once
// Valid range of the static camera radius for the chosen metric (v3.1 robustness sweep, 2026-11-16).
// The camera must lie where the static frame exists: outside the black-hole horizon and, in Schwarzschild-de Sitter, inside the cosmological one.
// A camera inside the horizon (or beyond r_c) starts the tracer on NaN data and the kernel ran for minutes (driver watchdog -> app killed), so the
// UI limits the slider and the command line refuses; the device code also returns "captured" for such a start as a last defence (kerr_cuda.cuh).
#include <algorithm>
#include <string>

#include "bhs/kerr.hpp"
#include "bhs/sds.hpp"
#include "render.hpp"

namespace bhs::kcuda {

inline constexpr double kCameraMargin = 0.02;  // stay 2 % away from the horizons (static observers there have diverging acceleration)

// rmin/rmax: allowed distance. a, charge, lambda as in Params; wormholes use wh_lc instead (not handled here).
inline void camera_range(const Params& p, double& rmin, double& rmax, const std::string& observer = "static") {
  rmin = 1.5; rmax = 2000.0;                                  // UI slider limits when there is no horizon (naked singularity, wormhole)
  if (p.lambda > 0.0) {
    double rh = 0, rc = 0;
    if (sds::horizons(1.0, p.lambda, rh, rc)) { rmin = rh * (1.0 + kCameraMargin); rmax = rc * (1.0 - kCameraMargin); }
    return;
  }
  const double disc = 1.0 - p.a * p.a - p.charge * p.charge;
  if (disc >= 0.0) rmin = std::max(rmin, (1.0 + std::sqrt(disc)) * (1.0 + kCameraMargin));
  // naked singularity (a^2 + Q^2 >= 1): rays are trapped between the repulsive core and the camera when the camera is close (measured 2026-11-16: mean 10^4 - 5 10^4
  // steps per ray at r < 2.2 for a^2 + Q^2 slightly above 1, i.e. ~10 s for 160x90 -> watchdog); from r = 3 on the cost is ~100 steps per ray
  else rmin = std::max(rmin, 3.0);
  // an observer on a circular geodesic exists only outside the prograde photon orbit r_ph = 2 (1 + cos(2/3 acos(-a))) (Kerr; 'orbit' is only offered there)
  // v4.0: 'infall' uses the exact Kerr velocity (lorentz::infall_beta_kerr), valid down to the horizon: no extra limit (v3_1 needed r >= 2.25)
  if (observer == "orbit" && p.charge == 0.0) rmin = std::max(rmin, 2.0 * (1.0 + std::cos(2.0 / 3.0 * std::acos(-std::min(1.0, p.a)))) * (1.0 + kCameraMargin));
}

inline double clamp_camera(const Params& p, double dist, const std::string& observer = "static") {
  double lo, hi; camera_range(p, lo, hi, observer);
  return std::min(std::max(dist, lo), hi);
}

}  // namespace bhs::kcuda
