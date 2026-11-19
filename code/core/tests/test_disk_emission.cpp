#include <doctest/doctest.h>

#include <bhs/disk_emission.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lut.hpp>
#include <cmath>

using namespace bhs;

namespace {
// Integral of nu over a logarithmic grid (trapezoid rule in ln nu): 1e10 .. 1e19 Hz covers T from 1e3 to 1e6 K.
template <class F>
double integrate_nu(F f) {
  const int n = 40000;
  const double l0 = std::log(1e10), l1 = std::log(1e19), h = (l1 - l0) / n;
  double s = 0;
  for (int i = 0; i <= n; ++i) {
    const double nu = std::exp(l0 + h * i);
    const double w = (i == 0 || i == n) ? 0.5 : 1.0;
    s += w * f(nu) * nu;  // d nu = nu d ln nu
  }
  return s * h;
}
}  // namespace

TEST_CASE("emission: bolometric radiance of a Planck spectrum is sigma T^4 / pi (SI constants)") {
  for (double T : {3e3, 1e4, 1e5}) {
    const double I = integrate_nu([&](double nu) { return emission::planck_nu(nu, T); });
    CHECK(I == doctest::Approx(emission::kSigmaSB * std::pow(T, 4) / M_PI).epsilon(1e-4));
  }
}

TEST_CASE("emission: the hardened spectrum f^-4 B_nu(f T) keeps the bolometric radiance sigma T^4 / pi for any f_col") {
  const double T = 2e4;
  for (double f : {1.0, 1.5, 1.7, 2.0}) {
    const double I = integrate_nu([&](double nu) { return emission::modified_planck_nu(nu, T, f); });
    CHECK(I == doctest::Approx(emission::kSigmaSB * std::pow(T, 4) / M_PI).epsilon(1e-4));
  }
}

TEST_CASE("emission: redshift scales the bolometric radiance as g^4 (I_nu / nu^3 invariant)") {
  const double T = 1e4, f = 1.7;
  for (double g : {0.3, 0.8, 1.4}) {
    // I_obs(nu) = g^3 I_em(nu / g)
    const double Iobs = integrate_nu([&](double nu) { return g * g * g * emission::modified_planck_nu(nu / g, T, f); });
    const double Iem = emission::kSigmaSB * std::pow(T, 4) / M_PI;
    CHECK(Iobs == doctest::Approx(std::pow(g, 4) * Iem).epsilon(1e-4));
    // is the same as a hardened spectrum at T_obs = g T (v0.4 uses this identity)
    CHECK(emission::modified_planck_nu(2e15, g * T, f) ==
          doctest::Approx(g * g * g * emission::modified_planck_nu(2e15 / g, T, f)).epsilon(1e-9));
  }
}

TEST_CASE("emission: rgb_hardened(T, 1) is exactly the v0.8 blackbody colour; f != 1 is f^-4 rgb(f T)") {
  for (double T : {800.0, 4200.0, 30000.0}) {
    const Vec3 a = blackbody::rgb(T), b = emission::rgb_hardened(T, 1.0);
    CHECK(a.x == b.x); CHECK(a.y == b.y); CHECK(a.z == b.z);
    const double f = 1.7;
    const Vec3 c = emission::rgb_hardened(T, f), d = blackbody::rgb(f * T);
    CHECK(c.x == doctest::Approx(d.x / std::pow(f, 4)).epsilon(1e-12));
    CHECK(c.z == doctest::Approx(d.z / std::pow(f, 4)).epsilon(1e-12));
  }
}

TEST_CASE("emission: lut::build(f_col = 1) is unchanged and build(f) matches rgb_hardened") {
  const auto d1 = lut::build(), d1b = lut::build(1.0), d2 = lut::build(1.7);
  REQUIRE(d1.size() == d1b.size());
  for (size_t i = 0; i < d1.size(); ++i) CHECK(d1[i] == d1b[i]);
  const int i = 600;
  const Vec3 c = emission::rgb_hardened(lut::temperature_at(i), 1.7);
  CHECK(d2[3 * i] == doctest::Approx(float(c.x)).epsilon(1e-6));
  CHECK(d2[3 * i + 1] == doctest::Approx(float(c.y)).epsilon(1e-6));
}

TEST_CASE("emission: Chandrasekhar (1+2.06 mu) and grey Eddington (1+1.5 mu) limb darkening conserve flux") {
  const int n = 100000;
  double s_ch = 0, s_ed = 0;
  for (int i = 0; i < n; ++i) {
    const double mu = (i + 0.5) / n;
    s_ch += 2.0 * emission::limb_chandrasekhar(mu) * mu / n;
    s_ed += 2.0 * emission::limb_eddington_grey(mu) * mu / n;
  }
  CHECK(s_ch == doctest::Approx(1.0).epsilon(1e-9));
  CHECK(s_ed == doctest::Approx(1.0).epsilon(1e-9));
  CHECK(emission::limb_chandrasekhar(1.0) / emission::limb_chandrasekhar(0.0) == doctest::Approx(1.0 + 2.06).epsilon(1e-12));
  CHECK(emission::limb_eddington_grey(1.0) / emission::limb_eddington_grey(0.0) == doctest::Approx(2.5).epsilon(1e-12));
  CHECK(emission::limb_eddington(0.4) == emission::limb_chandrasekhar(0.4));
  CHECK(emission::limb_chandrasekhar(0.0) < 1.0);
  CHECK(emission::limb_chandrasekhar(1.0) > 1.0);
}

TEST_CASE("emission: emitter_mu is 0 for Q = 0 and 1/gamma for the L = 0 turning-point photon (Schwarzschild closed form)") {
  CHECK(emission::emitter_mu(0.9, 8.0, 1.3, 0.0) == doctest::Approx(0.0).epsilon(1e-15));
  for (double r : {6.5, 10.0, 40.0}) {
    // photon at radial rest (b^2 = r^3 / (r - 2)) with L = 0: in the static frame it moves only along e_theta (mu_static = 1);
    // in the gas frame (orbital boost of gamma = sqrt((1 - 2/r) / (1 - 3/r)) along e_phi) mu = 1/gamma.
    const double Q = r * r / (1.0 - 2.0 / r);
    CHECK(emission::emitter_mu(0.0, r, 0.0, Q) == doctest::Approx(std::sqrt((1.0 - 3.0 / r) / (1.0 - 2.0 / r))).epsilon(1e-12));
  }
}

TEST_CASE("emission: kerr::trace returns the Carter constant Q of the photon") {
  const double a = 0.9, r0 = 40.0, th = 80.0 * M_PI / 180.0;
  const double nr = -0.9, nth = 0.1, nph = std::sqrt(1.0 - nr * nr - nth * nth);
  const auto in = kerr::init_from_zamo(a, r0, th, nr, nth, nph);
  const auto tr = kerr::trace(a, r0, th, nr, nth, nph, kerr::Disk{kerr::isco(a), 16.0});
  CHECK(tr.Q == doctest::Approx(in.Q).epsilon(1e-12));
}

TEST_CASE("emission: emitter_mu from a traced ray agrees with the flat-space angle to the disc normal far from the hole") {
  // Camera at D = 1e5, theta = 60 degrees, points at the point (r = 1000, phi = 0) of the equatorial plane; there gravity
  // (deflection ~ 4M/b ~ 5 mrad) and the orbital velocity (v ~ 0.03 c) barely move the angle.
  const double a = 0.0, D = 1e5, th = M_PI / 3.0, rt = 1000.0;
  const double cx = D * std::sin(th), cz = D * std::cos(th);
  const double dx = rt - cx, dz = -cz, nrm = std::hypot(dx, dz);
  // trace() takes the PHOTON's propagation direction, opposite to the viewing one d: n = -d (in the local basis e_r, e_theta, e_phi)
  const double nr = -(dx * std::sin(th) + dz * std::cos(th)) / nrm;   // -d . e_r
  const double nth = -(dx * std::cos(th) - dz * std::sin(th)) / nrm;  // -d . e_theta
  const double nph = 0.0;
  kerr::Options opt;
  opt.atol = 1e-9; opt.rtol = 1e-9;
  const auto tr = kerr::trace(a, D, th, nr, nth, nph, kerr::Disk{6.0, 2000.0}, opt);
  REQUIRE(tr.disk_hit);
  CHECK(tr.disk_r == doctest::Approx(rt).epsilon(0.05));
  const double mu_flat = std::abs(dz) / nrm;
  CHECK(emission::emitter_mu(a, tr.disk_r, tr.L, tr.Q) == doctest::Approx(mu_flat).epsilon(0.03));
}

// ---- v0.9a2: soft outer edge ----
TEST_CASE("edge: soft = 0 is the hard cut of v0.8 (weight 1 inside, 0 outside)") {
  CHECK(emission::edge_weight(15.999, 16.0, 0.0) == 1.0);
  CHECK(emission::edge_weight(16.001, 16.0, 0.0) == 0.0);
}

TEST_CASE("edge: soft taper is 1 below r(1-s), 0 above r(1+s), 1/2 at r, monotone and continuous") {
  const double re = 16.0, s = 0.3;
  CHECK(emission::edge_weight(re * (1 - s) - 1e-9, re, s) == 1.0);
  CHECK(emission::edge_weight(re * (1 + s) + 1e-9, re, s) == 0.0);
  CHECK(emission::edge_weight(re, re, s) == doctest::Approx(0.5).epsilon(1e-12));
  double prev = 1.0;
  for (int i = 0; i <= 2000; ++i) {
    const double r = re * (1 - s) + 2 * s * re * i / 2000.0;
    const double w = emission::edge_weight(r, re, s);
    CHECK(w <= prev + 1e-15);
    CHECK(prev - w < 0.01);  // no jumps
    prev = w;
  }
}

TEST_CASE("edge: edge_radius is rout for zero amplitude and spans rout(1 +- amp)") {
  CHECK(emission::edge_radius(16.0, 0.0, 0.2) == 16.0);
  CHECK(emission::edge_radius(16.0, 0.25, 0.0) == doctest::Approx(12.0).epsilon(1e-12));
  CHECK(emission::edge_radius(16.0, 0.25, 1.0) == doctest::Approx(20.0).epsilon(1e-12));
  CHECK(emission::edge_radius(16.0, 0.25, 0.5) == doctest::Approx(16.0).epsilon(1e-12));
  CHECK(emission::edge_trace_limit(16.0, 0.25, 0.3) == doctest::Approx(16.0 * 1.25 * 1.3).epsilon(1e-12));
}

TEST_CASE("trace skip: skipping the first in-disc crossing gives the second crossing, equal to a disc radius cut that excludes the first") {
  const double a = 0.9, D = 40.0, th = 80.0 * M_PI / 180.0;
  int tested = 0;
  const double rin = kerr::isco(a, true);
  for (int i = 0; i < 60 && tested < 8; ++i)
    for (int j = 0; j < 60 && tested < 8; ++j) {
      const double x = -0.22 + 0.44 * (i + 0.5) / 60.0, y = -0.22 + 0.44 * (j + 0.5) / 60.0;
      const double nrm = std::sqrt(1 + x * x + y * y);
      const double nr = 1.0 / nrm, nth = y / nrm, nph = -x / nrm;
      const auto t1 = kerr::trace(a, D, th, nr, nth, nph, kerr::Disk{rin, 60.0, 0});
      const auto t2 = kerr::trace(a, D, th, nr, nth, nph, kerr::Disk{rin, 60.0, 1});
      if (!t1.disk_hit || !t2.disk_hit || std::abs(t1.disk_r - t2.disk_r) < 0.5) continue;
      // ring (r_in', r_out') that excludes the 1st crossing and keeps the 2nd
      const double mid = 0.5 * (t1.disk_r + t2.disk_r);
      const kerr::Disk d = t2.disk_r < t1.disk_r ? kerr::Disk{rin, mid, 0} : kerr::Disk{mid, 60.0, 0};
      const auto t3 = kerr::trace(a, D, th, nr, nth, nph, d);
      REQUIRE(t3.disk_hit);
      CHECK(t3.disk_r == doctest::Approx(t2.disk_r).epsilon(1e-9));
      ++tested;
    }
  CHECK(tested >= 3);
  // skip = 0 is the old behaviour
  const auto t0 = kerr::trace(a, D, th, 0.9, 0.2, 0.1, kerr::Disk{kerr::isco(a, true), 16.0});
  const auto t0b = kerr::trace(a, D, th, 0.9, 0.2, 0.1, kerr::Disk{kerr::isco(a, true), 16.0, 0});
  CHECK(t0.disk_hit == t0b.disk_hit);
  CHECK(t0.disk_r == t0b.disk_r);
}
