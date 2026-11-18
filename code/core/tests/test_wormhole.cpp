#include <doctest/doctest.h>

#include <bhs/wormhole.hpp>
#include <cmath>
#include <functional>

using namespace bhs;

namespace {
// Composite Simpson with many intervals (independent of the ODE integrator; smooth integrands)
double simpson(const std::function<double(double)>& f, double a, double b, double /*tol*/) {
  const int n = 400000; const double h = (b - a) / n;
  double s = f(a) + f(b);
  for (int i = 1; i < n; ++i) s += f(a + i * h) * (i % 2 ? 4.0 : 2.0);
  return s * h / 3.0;
}
}  // namespace

TEST_CASE("wormhole: r(l) of the film and Ellis: values, C1 continuity at |l| = a, symmetry") {
  wh::Params p; p.rho = 1.0; p.a = 0.5; p.M = 0.3;
  CHECK(wh::r_of_l(p, 0.0) == 1.0); CHECK(wh::r_of_l(p, 0.5) == doctest::Approx(1.0).epsilon(1e-15)); CHECK(wh::r_of_l(p, -0.3) == 1.0);
  CHECK(wh::r_of_l(p, 2.0) == doctest::Approx(wh::r_of_l(p, -2.0)).epsilon(1e-15));
  // derivative: dr/dl by central finite differences (independent of the analytic formula)
  for (double l : {0.7, 1.5, 4.0, 12.0, -0.9, -3.0}) {
    const double h = 1e-6 * std::max(1.0, std::fabs(l));
    const double fd = (wh::r_of_l(p, l + h) - wh::r_of_l(p, l - h)) / (2 * h);
    CHECK(wh::drdl(p, l) == doctest::Approx(fd).epsilon(1e-6));
  }
  // C1 at l = a: the derivative tends to 0 from both sides
  CHECK(std::fabs(wh::drdl(p, 0.5 + 1e-9)) < 1e-8); CHECK(wh::drdl(p, 0.4) == 0.0);
  // r(l) grows with |l| and r' -> 1 (flat space); r - |l| varies logarithmically but r' > 0.99 for |l| >> M
  CHECK(wh::drdl(p, 1e4) > 0.9999 - 1e-3);
  // Ellis
  wh::Params e; e.rho = 2.0; e.ellis = true;
  CHECK(wh::r_of_l(e, 0.0) == 2.0); CHECK(wh::r_of_l(e, 1.5) == doctest::Approx(2.5)); CHECK(wh::drdl(e, 1.5) == doctest::Approx(0.6));
  // lens: W = 1.42953 M (value from the paper) and M/rho = 0.5 <-> W/rho = 0.715 (relation quoted in the paper)
  wh::Params q; q.rho = 1.0; q.M = 0.5; CHECK(wh::lensing_width(q) / q.rho == doctest::Approx(0.715).epsilon(1e-3));
}

TEST_CASE("wormhole Ellis: total angle against the independent quadrature, and H = 0 conserved") {
  wh::Params e; e.rho = 1.0; e.ellis = true;
  const double l_c = 1e4;
  for (double B : {0.2, 0.5, 0.8, 0.95}) {                       // B < b0: crosses
    // camera far away, ray directed at the throat: sin(alpha) = B / r(l_c), direction towards -e_l (cos alpha < 0)
    const double alpha = M_PI - std::asin(B / wh::r_of_l(e, l_c));
    const auto r = wh::trace(e, l_c, alpha, 1e6);
    REQUIRE(r.ok);
    CHECK(r.side == -1);                                         // ends up on the other side
    // total psi = integral of B dl / (r sqrt(r^2 - B^2)) from -inf to l_c, in Ellis; tail from l_c to inf: asin(B / l_c) included in the simulation on the opposite side
    auto g = [&](double l) { const double rr = std::sqrt(l * l + 1.0); return B / (rr * std::sqrt(rr * rr - B * B)); };
    // quadrature: [-L, l_c] with large L + flat tail asin(B/L) at both ends
    const double L = 1e4;
    const double ref = simpson(g, -L, l_c, 1e-12) + std::asin(B / std::sqrt(L * L + 1.0)) + std::asin(B / std::sqrt(l_c * l_c + 1.0));
    // the simulation measures from l_c up to |l| > l_max (flat tail already included); the quadrature goes up to +-L and adds the flat tails: the two are compared
    CHECK(r.psi == doctest::Approx(ref).epsilon(2e-4));
    CHECK(r.h_err < 1e-9);
  }
}

TEST_CASE("wormhole: B = rho is the boundary cross/turn back (throat = photon sphere); apparent angular radius of the throat sin(alpha) = rho / r(l_c)") {
  for (int kind = 0; kind < 2; ++kind) {
    wh::Params p; p.rho = 1.0;
    if (kind == 0) p.ellis = true; else { p.a = 0.01; p.M = 0.2; }      // Ellis, and the film's (a/rho = 0.01, W/rho = 0.29)
    for (double l_c : {3.0, 10.0, 50.0}) {
      const double rc = wh::r_of_l(p, l_c), sin_crit = p.rho / rc;
      // look by bisection for the alpha (face-on, towards -e_l) where the final side changes
      double lo = M_PI - std::asin(0.5 * sin_crit), hi = M_PI - std::asin(std::min(0.9999, 1.5 * sin_crit));  // lo crosses, hi turns back
      REQUIRE(wh::trace(p, l_c, lo, 1e5).side == -1);
      REQUIRE(wh::trace(p, l_c, hi, 1e5).side == +1);
      for (int i = 0; i < 40; ++i) { const double m = 0.5 * (lo + hi); (wh::trace(p, l_c, m, 1e5).side == -1 ? lo : hi) = m; }
      const double a_crit = 0.5 * (lo + hi);
      CHECK(std::sin(M_PI - a_crit) == doctest::Approx(sin_crit).epsilon(2e-3));
    }
  }
}

TEST_CASE("wormhole: camera at the throat (l_c = 0) sees half to each side; no deflection in the limit B << rho (straight line only) and plane symmetry") {
  wh::Params p; p.rho = 1.0; p.a = 0.005; p.M = 0.035;                  // film parameters (W = 0.05 rho)
  // l_c = 0: alpha < pi/2 goes to +l, alpha > pi/2 to -l (by symmetry of r(l)); psi equal by symmetry
  for (double alpha : {0.3, 0.9, 1.3}) {
    const auto a = wh::trace(p, 0.0, alpha), b = wh::trace(p, 0.0, M_PI - alpha);
    REQUIRE(a.ok); REQUIRE(b.ok);
    CHECK(a.side == +1); CHECK(b.side == -1);
    CHECK(a.psi == doctest::Approx(b.psi).epsilon(1e-9));
  }
  // camera far away, nearly radial ray (B << rho): crosses with almost no angular deflection (psi ~ B/r_c + asin(B/..)... small)
  const double l_c = 1e3, alpha = M_PI - 1e-4 / wh::r_of_l(p, l_c);
  const auto r = wh::trace(p, l_c, alpha, 1e6);
  CHECK(r.psi < 1e-3);
}
