#include <doctest/doctest.h>

#include <bhs/volume.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("volume: scale height H/r = hr (r / r_ref)^flare") {
  volume::Slab s; s.hr = 0.05; s.flare = 0.25; s.r_ref = 10.0;
  CHECK(s.h(10.0) == doctest::Approx(0.05).epsilon(1e-12));
  CHECK(s.h(160.0) == doctest::Approx(0.05 * std::pow(16.0, 0.25)).epsilon(1e-12));
  s.flare = 0.0;
  CHECK(s.h(3.0) == doctest::Approx(0.05).epsilon(1e-12));
}

TEST_CASE("volume: the vertical integral of alpha over the whole slab equals tau (any r, flare)") {
  for (double flare : {0.0, 0.3}) {
    volume::Slab s; s.hr = 0.06; s.flare = flare; s.tau = 7.5; s.rin = 6.0; s.rout = 1e9;
    for (double r : {8.0, 12.0, 40.0}) {
      const double H = s.h(r) * r;
      double sum = 0;
      const int n = 40000;
      const double zmax = 12.0 * H;
      for (int i = 0; i < n; ++i) {
        const double z = -zmax + 2.0 * zmax * (i + 0.5) / n;
        sum += s.alpha(r, z / r) * (2.0 * zmax / n);  // alpha is per unit length: integrated in z
      }
      CHECK(sum == doctest::Approx(7.5).epsilon(1e-6));
    }
  }
}

TEST_CASE("volume: alpha is zero inside the ISCO and tapers with the soft edge like edge_weight") {
  volume::Slab s; s.rin = 6.0; s.rout = 16.0; s.edge_soft = 0.0;
  CHECK(s.alpha(5.99, 0.0) == 0.0);
  CHECK(s.alpha(6.01, 0.0) > 0.0);
  CHECK(s.alpha(15.9, 0.0) > 0.0);
  CHECK(s.alpha(16.1, 0.0) == 0.0);  // hard cut
  s.edge_soft = 0.2;
  const double a0 = s.alpha(16.0, 0.0), aw = s.alpha(16.0 * 0.7, 0.0);
  CHECK(a0 > 0.0);
  // at r_e the weight is 1/2: alpha(r_e) = 0.5 alpha_without_edge
  volume::Slab hard = s; hard.edge_soft = 0.0; hard.rout = 1e9;
  CHECK(a0 == doctest::Approx(0.5 * hard.alpha(16.0, 0.0)).epsilon(1e-12));
  CHECK(aw == doctest::Approx(hard.alpha(16.0 * 0.7, 0.0)).epsilon(1e-12));  // below r(1-s): weight 1
}

TEST_CASE("volume: compositing a uniform slab gives B (1 - e^-tau) for any partition of the path") {
  const Vec3 B{2.0, 1.0, 0.5};
  const double tau = 1.7;
  for (int n : {1, 2, 7, 100}) {
    volume::Acc acc;
    for (int i = 0; i < n; ++i) volume::composite(acc, B, tau / n);
    CHECK(acc.I.x == doctest::Approx(2.0 * (1 - std::exp(-tau))).epsilon(1e-12));
    CHECK(acc.I.z == doctest::Approx(0.5 * (1 - std::exp(-tau))).epsilon(1e-12));
    CHECK(acc.trans == doctest::Approx(std::exp(-tau)).epsilon(1e-12));
  }
}

TEST_CASE("volume: thick limit gives B, thin limit gives B tau, two layers compose front to back") {
  const Vec3 B{1, 1, 1};
  volume::Acc thick; volume::composite(thick, B, 60.0);
  CHECK(thick.I.x == doctest::Approx(1.0).epsilon(1e-12));
  CHECK(thick.trans < 1e-20);
  volume::Acc thin; volume::composite(thin, B, 1e-6);
  CHECK(thin.I.x == doctest::Approx(1e-6).epsilon(1e-6));
  // front (B1, t1) and then back (B2, t2): I = B1 (1 - e^-t1) + e^-t1 B2 (1 - e^-t2)
  volume::Acc acc;
  volume::composite(acc, Vec3{3, 0, 0}, 0.4);
  volume::composite(acc, Vec3{1, 0, 0}, 1.1);
  const double expect = 3.0 * (1 - std::exp(-0.4)) + std::exp(-0.4) * 1.0 * (1 - std::exp(-1.1));
  CHECK(acc.I.x == doctest::Approx(expect).epsilon(1e-12));
  CHECK(acc.trans == doctest::Approx(std::exp(-1.5)).epsilon(1e-12));
}

TEST_CASE("volume: face-on path through the slab in the gas frame is tau / mu, so I = B (1 - e^(-tau/mu))") {
  // path in the gas frame per unit of z: ds = dz / mu_c; integrating alpha ds along the vertical gives tau / mu
  volume::Slab s; s.hr = 0.05; s.tau = 3.0; s.rin = 1.0; s.rout = 1e9;
  const double r = 30.0, H = s.h(r) * r, mu_c = 0.8;
  volume::Acc acc;
  const Vec3 B{1, 1, 1};
  const int n = 20000;
  const double zmax = 10.0 * H, dz = 2.0 * zmax / n;
  for (int i = 0; i < n; ++i) {
    const double z = -zmax + dz * (i + 0.5);
    volume::composite(acc, B, s.alpha(r, z / r) * dz / mu_c);
  }
  CHECK(acc.I.x == doctest::Approx(1.0 - std::exp(-3.0 / 0.8)).epsilon(1e-6));
}
