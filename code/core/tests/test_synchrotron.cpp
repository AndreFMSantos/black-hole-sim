#include <doctest/doctest.h>

#include <bhs/synchrotron.hpp>
#include <cmath>
#include <initializer_list>

using namespace bhs;

namespace {
// I_I(x) by the integral that defines it (Dexter 2016, eq. A15 and A5): I_I = (1/x) int dz z^2 e^-z F(x/z^2), F(y) = y int_y^inf K_{5/3}(s) ds,
// with K_{5/3} from the standard library (std::cyl_bessel_k): an oracle independent of the fit.
double int_K53(double y) {  // int_y^inf K_{5/3}(s) ds, trapezoid rule in ln s until K ~ 0
  if (y > 300.0) return 0.0;  // K_{5/3}(s) ~ e^-s: the contribution is < 1e-130 (and std::cyl_bessel_k throws for huge s)
  const int n = 4000;
  const double l0 = std::log(y), l1 = std::log(y + 60.0), h = (l1 - l0) / n;
  double s = 0;
  for (int i = 0; i <= n; ++i) {
    const double v = std::exp(l0 + h * i);
    s += (i == 0 || i == n ? 0.5 : 1.0) * std::cyl_bessel_k(5.0 / 3.0, v) * v;
  }
  return s * h;
}
double I_exact(double x) {
  const int n = 800;
  const double l0 = std::log(1e-3), l1 = std::log(60.0), h = (l1 - l0) / n;
  double s = 0;
  for (int i = 0; i <= n; ++i) {
    const double z = std::exp(l0 + h * i), y = x / (z * z);
    s += (i == 0 || i == n ? 0.5 : 1.0) * z * z * std::exp(-z) * (y * int_K53(y)) * z;
  }
  return s * h / x;
}
}  // namespace

TEST_CASE("synchrotron: the fit I_I(x) agrees with its defining integral (Dexter 2016: ~1 %)") {
  for (double x : {0.1, 0.3, 1.0, 3.0, 10.0, 30.0}) {
    const double ex = I_exact(x), fit = synch::I_I(x);
    CHECK(fit == doctest::Approx(ex).epsilon(0.01));  // measured: <= 0.35 % in x = 0.1 ... 30
  }
}

TEST_CASE("synchrotron: ultrarelativistic limit of the total power, (4/3) sigma_T c U_B <gamma^2 beta^2> with <gamma^2> = 12 Theta^2") {
  const double B = 10.0, n = 1.0;
  for (double Th : {100.0, 300.0}) {
    // P/V = int dOmega int j dnu : j depends on theta_B through nu_c ~ sin(theta_B): int dOmega sin^2 = 8 pi / 3
    const double nu_B = synch::nu_B(B), nu_c0 = 1.5 * nu_B * Th * Th;  // nu_c em sin(theta_B) = 1
    double J = 0;  // int x I_I(x) dx (trapezoid rule in ln x)
    const int N = 4000; const double l0 = std::log(1e-9), l1 = std::log(1e3), h = (l1 - l0) / N;
    for (int i = 0; i <= N; ++i) { const double x = std::exp(l0 + h * i); J += (i == 0 || i == N ? 0.5 : 1.0) * x * synch::I_I(x) * x; }
    J *= h;
    const double per_sr_sin2 = n * synch::kE * synch::kE * nu_c0 * nu_c0 / (2.0 * std::sqrt(3.0) * synch::kC * Th * Th) * J;  // por sin^2(theta_B)
    const double P_fit = per_sr_sin2 * 8.0 * M_PI / 3.0;
    const double sigT = 8.0 * M_PI / 3.0 * std::pow(synch::kE * synch::kE / (synch::kMe * synch::kC * synch::kC), 2.0);
    const double P_exact = n * (4.0 / 3.0) * sigT * synch::kC * (B * B / (8.0 * M_PI)) * 12.0 * Th * Th;
    CHECK(P_fit == doctest::Approx(P_exact).epsilon(0.01));  // measured: P_fit / P_exact = 0.9971 (Theta = 100 and 300)
  }
}

TEST_CASE("synchrotron: j scales with n, and j(lambda nu; lambda nu_B) = lambda j(nu; nu_B) (depends on x only)") {
  const double B = 5.0, Th = 20.0, nu = 2.3e11, sth = std::sin(1.0);
  const double j1 = synch::j_I(1e4, Th, B, sth, nu), j2 = synch::j_I(3e4, Th, B, sth, nu);
  CHECK(j2 == doctest::Approx(3.0 * j1).epsilon(1e-12));
  const double lam = 1.7;
  CHECK(synch::j_I(1e4, Th, lam * B, sth, lam * nu) == doctest::Approx(lam * j1).epsilon(1e-12));
}

TEST_CASE("synchrotron: emission vanishes without field and at sin(theta_B) = 0, and the exponential cutoff suppresses large x") {
  CHECK(synch::j_I(1e4, 20.0, 0.0, 1.0, 2.3e11) == 0.0);
  CHECK(synch::j_I(1e4, 20.0, 5.0, 0.0, 2.3e11) == 0.0);
  CHECK(synch::j_I(1e4, 20.0, 5.0, 1.0, 1e18) < 1e-30 * synch::j_I(1e4, 20.0, 5.0, 1.0, 2.3e11));
}

TEST_CASE("synchrotron: Kirchhoff alpha = j / B_nu(T_e) and the Rayleigh-Jeans brightness temperature (radio)") {
  const double Th = 20.0, B = 5.0, nu = 2.3e11, sth = 1.0;
  const double Te = Th * synch::kMe * synch::kC * synch::kC / synch::kKb;
  const double Bn = synch::planck(nu, Te);
  CHECK(synch::alpha_nu(1e4, Th, B, sth, nu) * Bn == doctest::Approx(synch::j_I(1e4, Th, B, sth, nu)).epsilon(1e-12));
  // B_nu(T) em Rayleigh-Jeans: T_b = I c^2 / (2 k nu^2) = T
  CHECK(synch::brightness_temperature(Bn, nu) == doctest::Approx(Te).epsilon(1e-3));
  // I = B_nu(T) => T_b ~ T even at h nu << k T; the Wien correction appears at h nu ~ k T
  CHECK(synch::brightness_temperature(synch::planck(1e9, 1e6), 1e9) == doctest::Approx(1e6).epsilon(1e-4));
}

TEST_CASE("synchrotron: Leung et al. 2011 fit (as quoted by Pandya 2016, eq. 24)") {
  using namespace bhs::synch;
  // K_2(1/Theta) -> 2 Theta^2 for Theta >> 1 (stated in the paper, second-hand)
  CHECK(K2(1.0 / 100.0) / (2.0 * 1e4) == doctest::Approx(1.0).epsilon(2e-4));
  // ultra-relativistic limit: j_leung -> n e^2 sqrt(2) pi nu_s / (6 Theta^2 c) t^2 exp(-X^(1/3)), the same form as the fit with K_2 -> 2 Theta^2
  const double Th = 100.0, B = 10.0, n = 1e5, nu = 230e9, s = 0.8;
  const double nu_s = (2.0 / 9.0) * nu_B(B) * s * Th * Th, X = nu / nu_s;
  const double t = std::sqrt(X) + std::pow(2.0, 11.0 / 12.0) * std::pow(X, 1.0 / 6.0);
  const double lim = n * std::sqrt(2.0) * M_PI * kE * kE * nu_s / (6.0 * Th * Th * kC) * t * t * std::exp(-std::cbrt(X));
  CHECK(j_I_leung(n, Th, B, s, nu) / lim == doctest::Approx(1.0).epsilon(2e-4));
  // agreement between the two fits (different constructions) at Theta_e = 10: ratio j_Leung/j_Dexter between 0.5 and 2 for x = nu/nu_c in [3, 300]
  const double T2 = 10.0;
  for (double x : {3.0, 10.0, 30.0, 100.0, 300.0}) {
    const double nuc = 1.5 * nu_B(B) * s * T2 * T2, nuu = x * nuc;
    const double q = j_I_leung(n, T2, B, s, nuu) / j_I(n, T2, B, s, nuu);
    INFO("x = " << x << " ratio = " << q);
    CHECK(q > 0.5); CHECK(q < 2.0);
  }
  // Kirchhoff with the Leung fit: alpha = j / B_nu(T_e)
  const double al = alpha_nu(n, 3.0, B, s, nu, 1), jj = j_I_leung(n, 3.0, B, s, nu);
  CHECK(al * planck(nu, 3.0 * kMe * kC * kC / kKb) == doctest::Approx(jj).epsilon(1e-12));
}
