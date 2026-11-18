#pragma once
#include <cmath>

namespace bhs::observer {

// Static observer at radius r (M = 1). alpha = angle between the light-ray direction
// and the radial direction. Conserved impact parameter (b = L/E):
//   b = r sin(alpha) / sqrt(1 - 2M/r)
inline double impact_parameter(double r, double alpha) {
  return r * std::sin(alpha) / std::sqrt(1.0 - 2.0 / r);
}

}  // namespace bhs::observer
