#pragma once
// Emission and absorption of THERMAL synchrotron (Maxwell-Juttner) in the ultra-relativistic limit (Theta_e >> 1), cgs.
// Source: Dexter 2016, MNRAS 462, 115 (grtrans), Appendix A, eq. A12 and A18 (read in the PDF):
//   j_I(nu) = n_e e^2 nu / (2 sqrt(3) c Theta_e^2) * I_I(x),   x = nu / nu_c,   nu_c = (3/2) nu_B sin(theta_B) Theta_e^2,   nu_B = e B / (2 pi m_e c)
//   I_I(x) = 2.5651 (1 + 1.92 x^(-1/3) + 0.9977 x^(-2/3)) exp(-1.8899 x^(1/3))
// Accuracy (Dexter): ~1 % against the integral that defines I_I; ~20 % against the full emissivities for Theta_e > 3 and nu/nu_c > 1.
// The K_2(1/Theta_e) normalisation does not appear (limit K_2(z) -> 2/z^2); do not use at Theta_e <~ 1 (the Leung et al. 2011 fit, which
// keeps it, was not read at the time). Units: erg s^-1 cm^-3 Hz^-1 sr^-1 (by inspection; the paper does not state them).
// Absorption by Kirchhoff's law (LTE): alpha_nu = j_nu / B_nu(T_e).
#include <cmath>
#include <cstdlib>

namespace bhs::synch {

constexpr double kE = 4.80320471e-10;    // esu
constexpr double kMe = 9.1093837015e-28;  // g
constexpr double kMp = 1.67262192369e-24;  // g
constexpr double kC = 2.99792458e10;     // cm/s
constexpr double kKb = 1.380649e-16;     // erg/K
constexpr double kH = 6.62607015e-27;    // erg s

inline double nu_B(double B) { return kE * B / (2.0 * M_PI * kMe * kC); }

inline double I_I(double x) {
  const double c = std::cbrt(x);
  return 2.5651 * (1.0 + 1.92 / c + 0.9977 / (c * c)) * std::exp(-1.8899 * c);
}

// n_e [cm^-3], Theta_e = kT_e/(m_e c^2), B [G], sin_thB = sin(angle between B and the photon in the fluid frame), nu [Hz]
inline double j_I(double n_e, double Theta_e, double B, double sin_thB, double nu) {
  if (!(B > 0.0) || !(sin_thB > 0.0) || !(Theta_e > 0.0) || !(nu > 0.0)) return 0.0;
  const double nu_c = 1.5 * nu_B(B) * sin_thB * Theta_e * Theta_e, x = nu / nu_c;
  return n_e * kE * kE * nu / (2.0 * std::sqrt(3.0) * kC * Theta_e * Theta_e) * I_I(x);
}

// v2.2b3: fit of Leung, Gammie & Noble 2011 (ApJ 737, 21), eq. 72 (read in the paper's PDF, section 7; matches Pandya et al. 2016 eq. 24,
// except that the Pandya PDF prints K_2(1/Theta^2): a typo, the paper has K_2(1/Theta_e)):
//   j_nu = n_e sqrt(2) pi e^2 nu_s / (3 K_2(1/Theta_e) c) (X^(1/2) + 2^(11/12) X^(1/6))^2 exp(-X^(1/3)),  X = nu/nu_s,
//   nu_s = (2/9) nu_c Theta_e^2 sin(theta),  nu_c = e B/(2 pi m_e c) (eq. 2 of the paper).
// It combines eq. 26 of Petrosian 1981 with the low-frequency limit (eq. 55). Declared validity: Theta_e >~ 0.5 (abstract; below gamma ~ 1 the
// cyclotron streaks appear); the error becomes O(1) for Theta_e ~ (nu/(nu_c sin theta))^(-1/5) (rough estimate from the paper). Absorption: Kirchhoff.
inline double K2(double z) {
#if defined(__cpp_lib_math_special_functions)
  return std::cyl_bessel_k(2.0, z);
#else
  // Abramowitz & Stegun 9.8.5-9.8.8 (fallback when libc++ has no std::cyl_bessel_k)
  double K0, K1;
  if (z <= 2.0) {
    const double lz = std::log(0.5 * z), y = (z / 3.75) * (z / 3.75), w = z * z * 0.25;
    const double i0 = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    const double i1 = z * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
    K0 = -lz * i0 + (-0.57721566 + w * (0.42278420 + w * (0.23069756 + w * (0.03488590 + w * (0.00262698 + w * (0.00010750 + w * 0.0000074))))));
    K1 = lz * i1 + (1.0 / z) * (1.0 + w * (0.15443144 + w * (-0.67278579 + w * (-0.18156897 + w * (-0.01919402 + w * (-0.00110404 + w * -0.00004686))))));
  } else {
    const double u = 2.0 / z, e = std::exp(-z) / std::sqrt(z);
    K0 = e * (1.25331414 + u * (-0.07832358 + u * (0.02189568 + u * (-0.01062446 + u * (0.00587872 + u * (-0.00251540 + u * 0.00053208))))));
    K1 = e * (1.25331414 + u * (0.23498619 + u * (-0.03655620 + u * (0.01504268 + u * (-0.00780353 + u * (0.00325614 + u * -0.00068245))))));
  }
  return K0 + 2.0 * K1 / z;
#endif
}
inline double j_I_leung(double n_e, double Theta_e, double B, double sin_thB, double nu) {
  if (!(B > 0.0) || !(sin_thB > 0.0) || !(Theta_e > 0.0) || !(nu > 0.0)) return 0.0;
  const double nu_s = (2.0 / 9.0) * nu_B(B) * sin_thB * Theta_e * Theta_e, X = nu / nu_s;
  const double t = std::sqrt(X) + std::pow(2.0, 11.0 / 12.0) * std::pow(X, 1.0 / 6.0);
  return n_e * std::sqrt(2.0) * M_PI * kE * kE * nu_s / (3.0 * K2(1.0 / Theta_e) * kC) * t * t * std::exp(-std::cbrt(X));
}

// B_nu(T), cgs (erg s^-1 cm^-2 Hz^-1 sr^-1)
inline double planck(double nu, double T) {
  const double x = kH * nu / (kKb * T);
  if (x > 700.0) return 0.0;
  return 2.0 * kH * nu * nu * nu / (kC * kC) / std::expm1(x);
}

inline double alpha_nu(double n_e, double Theta_e, double B, double sin_thB, double nu, int model = 0) {
  const double j = model == 1 ? j_I_leung(n_e, Theta_e, B, sin_thB, nu) : j_I(n_e, Theta_e, B, sin_thB, nu);
  if (j <= 0.0) return 0.0;
  return j / planck(nu, Theta_e * kMe * kC * kC / kKb);
}

// Rayleigh-Jeans brightness temperature T_b = I c^2 / (2 k nu^2)
inline double brightness_temperature(double I_nu, double nu) { return I_nu * kC * kC / (2.0 * kKb * nu * nu); }

}  // namespace bhs::synch
