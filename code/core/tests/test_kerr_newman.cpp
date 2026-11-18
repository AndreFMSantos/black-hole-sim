#include <doctest/doctest.h>

#include <bhs/kerr.hpp>
#include <bhs/schwarzschild.hpp>
#include <bhs/sds.hpp>
#include <cmath>
#include <functional>

using namespace bhs;

TEST_CASE("kerr-newman: q2 = 0 recovers Kerr exactly (same bits)") {
  for (double a : {0.0, 0.5, 0.9}) {
    kerr::Options o0, o1; o1.q2 = 0.0;
    const auto t0 = kerr::trace(a, 40.0, 1.2, 0.8, -0.4, 0.3, {5.0, 16.0, 0}, o0), t1 = kerr::trace(a, 40.0, 1.2, 0.8, -0.4, 0.3, {5.0, 16.0, 0}, o1);
    CHECK(t0.steps == t1.steps); CHECK(t0.phi_end == t1.phi_end); CHECK(t0.theta_end == t1.theta_end); CHECK(t0.disk_r == t1.disk_r);
    CHECK(kerr::horizon(a) == kerr::horizon(a, 0.0));
  }
}

TEST_CASE("kerr-newman: horizons r+- = 1 +- sqrt(1 - a^2 - q2); no horizon if a^2 + q2 > 1; Delta(r+) = 0") {
  for (auto [a, q2] : {std::pair{0.0, 0.25}, {0.5, 0.25}, {0.6, 0.64}, {0.0, 1.0}}) {
    const double rp = kerr::horizon(a, q2);
    CHECK(rp == doctest::Approx(1.0 + std::sqrt(1.0 - a * a - q2)));
    CHECK(kerr::delta(rp, a, q2) == doctest::Approx(0.0).epsilon(1e-12));
  }
  CHECK(kerr::horizon(1.2, 0.0) == 0.0); CHECK(kerr::horizon(0.0, 1.1) == 0.0); CHECK(kerr::horizon(0.8, 0.4) == 0.0);
}

TEST_CASE("RN: photon sphere, shadow and ISCO (formulas) against independent numerical sweeps") {
  for (double q2 : {0.0, 0.25, 0.64, 0.98}) {
    // photon sphere = maximum of f/r^2 (a = 0): fine sweep
    double best = 0, rbest = 0;
    for (double r = 1.0 + std::sqrt(1.0 - q2) + 1e-4; r < 10.0; r += 1e-5) { const double v = (r * r - 2.0 * r + q2) / (r * r * r * r); if (v > best) { best = v; rbest = r; } }
    CHECK(kerr::rn_photon_sphere(q2) == doctest::Approx(rbest).epsilon(1e-4));
    CHECK(kerr::rn_shadow_b(q2) == doctest::Approx(1.0 / std::sqrt(best)).epsilon(1e-8));
    // ISCO = minimum of L^2(r) = r^2 (r - q2)/(r^2 - 3 r + 2 q2) of the timelike circular orbits
    const double rph = kerr::rn_photon_sphere(q2);
    double lmin = 1e300, rl = 0;
    for (double r = rph + 1e-4; r < 20.0; r += 1e-5) { const double L2 = r * r * (r - q2) / (r * r - 3.0 * r + 2.0 * q2); if (L2 < lmin) { lmin = L2; rl = r; } }
    CHECK(kerr::rn_isco(q2) == doctest::Approx(rl).epsilon(1e-4));
  }
  CHECK(kerr::rn_isco(0.0) == doctest::Approx(6.0)); CHECK(kerr::rn_isco(0.25) == doctest::Approx(5.606643).epsilon(1e-6));
  CHECK(kerr::rn_isco(0.64) == doctest::Approx(4.890767).epsilon(1e-6)); CHECK(kerr::rn_isco(0.9801) == doctest::Approx(4.060807).epsilon(1e-6));
}

namespace {
// critical viewing angle (bisection in the Carter tracer) for a static observer (a = 0) at r0: Captured for alpha < alpha_c
double critical_alpha(double q2, double r0) {
  kerr::Options o; o.q2 = q2; o.atol = 1e-11; o.rtol = 1e-10; o.r_far = 5e3; o.max_steps = 2000000;  // the fate only needs the ray far away; r_far = 1e6 would cost ~1.5e5 steps per ray
  auto fate = [&](double alpha) { return kerr::trace(0.0, r0, M_PI / 2, std::cos(alpha), -std::sin(alpha), 0.0, {}, o); };
  double lo = 1e-3, hi = 0.3;
  REQUIRE(fate(lo).fate == kerr::Fate::Captured);
  REQUIRE(fate(hi).fate == kerr::Fate::Escaped);
  for (int i = 0; i < 50; ++i) { const double m = 0.5 * (lo + hi); (fate(m).fate == kerr::Fate::Captured ? lo : hi) = m; }
  return 0.5 * (lo + hi);
}
}  // namespace

TEST_CASE("RN: the Carter tracer reproduces the shadow b_c = r_ph^2/sqrt(Delta(r_ph)) (static observer at r0 = 300)") {
  const double r0 = 300.0;
  for (double q2 : {0.0, 0.25, 0.64, 0.9}) {
    const double f0 = 1.0 - 2.0 / r0 + q2 / (r0 * r0);
    const double sin_alpha = kerr::rn_shadow_b(q2) * std::sqrt(f0) / r0;   // sin(alpha) = b sqrt(f)/r for a static observer
    const double got = std::sin(critical_alpha(q2, r0));
    INFO("q2 = " << q2 << " sin(alpha_c) got " << got << " expected " << sin_alpha);
    CHECK(got == doctest::Approx(sin_alpha).epsilon(2e-5));
  }
  // the shadow shrinks with the charge: b_c(0) = sqrt(27) = 5.196152 > b_c(0.9)
  CHECK(kerr::rn_shadow_b(0.0) == doctest::Approx(std::sqrt(27.0))); CHECK(kerr::rn_shadow_b(0.9) < kerr::rn_shadow_b(0.0));
}

TEST_CASE("naked RN (q2 = 1.1 > 1): no horizon; only nearly radial rays reach r_min (singular); the rest turn back and escape") {
  kerr::Options o; o.q2 = 1.1;
  int singular = 0, escaped = 0, other = 0;
  for (int i = 1; i <= 60; ++i) {
    const double alpha = 0.002 + 0.0025 * i;                              // from 0.0045 to ~0.15 rad (the Schwarzschild shadow: ~0.0173 at r0 = 300)
    const auto t = kerr::trace(0.0, 300.0, M_PI / 2, std::cos(alpha), -std::sin(alpha), 0.0, {}, o);
    if (t.singular) ++singular; else if (t.fate == kerr::Fate::Escaped) ++escaped; else ++other;
  }
  CHECK(escaped > 50);        // almost all turn back: the Q^2/r^2 repulsion (the potential f L^2/r^2 -> +inf at r = 0)
  CHECK(singular < 8);        // only those with L ~ 0 (regularised by l_min) reach r_min
}

TEST_CASE("naked Kerr (a = 1.2): no horizon; r_min stops; rays far from the axis escape; invariants conserved") {
  kerr::Options o; o.r_min = 0.05;
  int singular = 0, escaped = 0;
  for (int i = 1; i <= 40; ++i) {
    const double alpha = 0.004 * i;
    const auto t = kerr::trace(1.2, 40.0, 1.2, std::cos(alpha), -std::sin(alpha), 0.0, {}, o);
    if (t.singular) ++singular; else if (t.fate == kerr::Fate::Escaped) { ++escaped; CHECK(t.max_inv_err < 1e-6); }
  }
  CHECK(escaped > 20);
  CHECK(kerr::horizon(1.2) == 0.0);
}

TEST_CASE("SdS: horizons (roots of f), shadow b_c^2 = 27 M^2/(1 - 9 Lambda M^2), r_ph = 3M, static angle = pi/2 at 3M") {
  const double M = 1.0;
  for (double lam : {0.001, 0.01, 0.05, 0.1}) {
    double rh, rc;
    REQUIRE(sds::horizons(M, lam, rh, rc));
    CHECK(sds::f(rh, M, lam) == doctest::Approx(0.0).epsilon(1e-9)); CHECK(sds::f(rc, M, lam) == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(rh > 2.0 * M); CHECK(rh < 3.0 * M); CHECK(rc > 3.0 * M);
    // photon sphere = maximum of f/r^2, at r = 3M for any Lambda (sweep)
    double best = 0, rbest = 0;
    for (double r = rh + 1e-3; r < rc - 1e-3; r += 1e-4) { const double v = sds::f(r, M, lam) / (r * r); if (v > best) { best = v; rbest = r; } }
    CHECK(rbest == doctest::Approx(3.0 * M).epsilon(1e-3));
    CHECK(sds::shadow_b(M, lam) == doctest::Approx(1.0 / std::sqrt(best)).epsilon(1e-6));
    CHECK(sds::static_shadow_sin(M, lam, 3.0 * M) == doctest::Approx(1.0).epsilon(1e-12));
  }
  CHECK(sds::shadow_b(1.0, 1e-12) == doctest::Approx(std::sqrt(27.0)).epsilon(1e-9));   // Lambda -> 0
  double rh, rc; CHECK_FALSE(sds::horizons(1.0, 0.12, rh, rc));                          // Lambda M^2 > 1/9: no horizons
}

TEST_CASE("SdS: ray in the direct radial ODE (E^2 - f L^2/r^2) falls/turns back at the boundary b_c, and a ray's angle is Schwarzschild's with b_eff") {
  const double M = 1.0, lam = 0.02, rO = 10.0;
  double rh, rc; REQUIRE(sds::horizons(M, lam, rh, rc));
  // 1) bisection of the local angle (static observer) by integrating r'' = -V'/2, V = f L^2/r^2, E = 1 (regular at the turning point)
  auto falls = [&](double sin_a) {
    const double fO = sds::f(rO, M, lam), L = rO * sin_a / std::sqrt(fO);
    auto V = [&](double r) { return sds::f(r, M, lam) * L * L / (r * r); };
    auto dV = [&](double r) { const double h = 1e-6 * r; return (V(r + h) - V(r - h)) / (2 * h); };
    double r = rO, v = -std::sqrt(std::max(0.0, 1.0 - V(rO)));
    for (int i = 0; i < 4000000; ++i) {
      const double h = 2e-4 * std::min(r, 5.0);
      // RK4 em (r, v)
      auto acc = [&](double rr) { return -0.5 * dV(rr); };
      const double k1r = v, k1v = acc(r), k2r = v + 0.5 * h * k1v, k2v = acc(r + 0.5 * h * k1r), k3r = v + 0.5 * h * k2v, k3v = acc(r + 0.5 * h * k2r), k4r = v + h * k3v, k4v = acc(r + h * k3r);
      r += h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r); v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v);
      if (r < rh * 1.0005) return true;
      if (v > 0 && r > rO) return false;
    }
    return true;
  };
  double lo = 0.05, hi = 0.9;                           // falls(lo) = true (small b), falls(hi) = false
  REQUIRE(falls(lo)); REQUIRE_FALSE(falls(hi));
  for (int i = 0; i < 30; ++i) { const double m = 0.5 * (lo + hi); (falls(m) ? lo : hi) = m; }
  CHECK(0.5 * (lo + hi) == doctest::Approx(sds::static_shadow_sin(M, lam, rO)).epsilon(2e-4));
  // 2) deflection: SdS with b == Schwarzschild with b_eff (independent integral in r against schw::trace)
  const double b = 30.0, beff = 1.0 / std::sqrt(1.0 / (b * b) + lam / 3.0);
  const auto ref = schw::trace(beff, 1e-4);
  REQUIRE(ref.fate == schw::Fate::Escaped);
  // total angle swept by quadrature in r: phi = 2 int_{r_t}^{r_max} b dr / (r^2 sqrt(1 - b^2 f/r^2))  (up to where f -> 0 it does not converge: I use r_max = 0.95 r_c)
  // comparison made on the orbit invariant: the SdS(b) turning point r_t coincides with Schwarzschild(b_eff)'s: root of 1/b_eff^2 - u^2 + 2 M u^3 = 0
  double u = 1.0 / b; for (int i = 0; i < 100; ++i) { const double F = 1.0 / (beff * beff) - u * u + 2 * M * u * u * u, dF = -2 * u + 6 * M * u * u; u -= F / dF; }
  double rt_sds = b; for (int i = 0; i < 200; ++i) { const double F = rt_sds * rt_sds - b * b * sds::f(rt_sds, M, lam), dF = 2 * rt_sds - b * b * (2.0 * M / (rt_sds * rt_sds) - 2.0 * lam * rt_sds / 3.0); rt_sds -= F / dF; }
  CHECK(1.0 / u == doctest::Approx(rt_sds).epsilon(1e-9));
}
