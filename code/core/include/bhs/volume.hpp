#pragma once
// Thick disk (v0.9b): density model and radiative transport.
//
// Transport (grey, LTE, source function = Planck at T(r)): along the ray, from the camera to the scene,
//   dI/dtau = B - I   =>   I += trans * B * (1 - e^-dtau),  trans *= e^-dtau       (front-to-back composition)
// with dtau = alpha * ds, ds = length in the GAS FRAME = E_emit * d(lambda) (E_emit = -p.u = u^t (1 - Omega L),
// lambda affine with E = 1 at infinity). B is the already-shifted black-body colour (LUT at T_obs = g T), as in v0.4.
//
// Layer model (Slab, analytic and replaceable: v2.x swaps it for 3D procedural data or GRMHD data): Gaussian in
// z = r mu with height scale H = (H/r)(r) r and TOTAL vertical optical depth tau (integral of alpha dz from -inf to +inf):
//   alpha(r, mu) = tau / (sqrt(2 pi) H) exp(-z^2 / 2 H^2) * w_edge(r),   0 inside the ISCO.
#include <cmath>

#include "disk_emission.hpp"
#include "vec3.hpp"

namespace bhs::volume {

struct Slab {
  double hr = 0.05;       // H/r em r_ref
  double flare = 0.0;     // H/r = hr (r/r_ref)^flare
  double tau = 10.0;      // total vertical optical depth
  double r_ref = 10.0;
  double rin = 6.0, rout = 16.0;
  double edge_soft = 0.0;  // soft outer edge (disk_emission.hpp::edge_weight); 0 = hard cut

  double h(double r) const { return hr * std::pow(r / r_ref, flare); }
  // absorption coefficient (per unit length, M = 1) at the point (r, mu = cos theta)
  double alpha(double r, double mu) const {
    if (r <= rin) return 0.0;
    const double w = emission::edge_weight(r, rout, edge_soft);
    if (w <= 0.0) return 0.0;
    const double H = h(r) * r, z = r * mu;
    return w * tau / (std::sqrt(2.0 * M_PI) * H) * std::exp(-0.5 * z * z / (H * H));
  }
};

struct Acc {
  Vec3 I{0, 0, 0};
  double trans = 1.0;
};
inline void composite(Acc& a, const Vec3& B, double dtau) {
  const double absorbed = -std::expm1(-dtau);
  a.I = a.I + (a.trans * absorbed) * B;
  a.trans *= std::exp(-dtau);
}

}  // namespace bhs::volume
