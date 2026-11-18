#include <doctest/doctest.h>

#include <bhs/kerr_analytic.hpp>
#include <cmath>
#include <vector>

using namespace bhs;
namespace an = bhs::kerr_an;

// ---- special functions, each against an independent definition (quadrature of the defining integral, identities, derivatives) ----

TEST_CASE("kerr_analytic: tanh-sinh handles inverse-square-root endpoints") {
  CHECK(an::tanh_sinh([](double x, double, double) { return 1.0 / std::sqrt(x); }, 0.0, 1.0) == doctest::Approx(2.0).epsilon(1e-13));
  CHECK(an::tanh_sinh([](double, double da, double db) { return 1.0 / std::sqrt(da * db); }, 0.0, 1.0) == doctest::Approx(M_PI).epsilon(1e-13));
  CHECK(an::tanh_sinh([](double x, double, double) { return std::sin(x); }, 0.0, 2.0) == doctest::Approx(1.0 - std::cos(2.0)).epsilon(1e-13));
}

TEST_CASE("kerr_analytic: Jacobi functions (m > 0 and m < 0) satisfy their identities and ODEs") {
  for (double m : {0.0, 0.3, 0.9, -0.05, -0.7, -5.0, -40.0}) {
    for (double u : {-3.1, -0.4, 0.2, 1.3, 2.9, 7.7}) {
      double sn, cn, dn;
      an::jacobi(u, m, sn, cn, dn);
      CHECK(sn * sn + cn * cn == doctest::Approx(1.0).epsilon(1e-13));
      CHECK(dn * dn == doctest::Approx(1.0 - m * sn * sn).epsilon(1e-12));
      const double h = 1e-5;
      double sp, cp, dp, sm, cm, dm;
      an::jacobi(u + h, m, sp, cp, dp);
      an::jacobi(u - h, m, sm, cm, dm);
      CHECK((sp - sm) / (2 * h) == doctest::Approx(cn * dn).epsilon(1e-8).scale(1.0));
    }
    // sn(K) = 1, sn(2K) = 0, sn(u + 4K) = sn(u)
    const double K = an::ellip_K(m);
    double sn, cn, dn;
    an::jacobi(K, m, sn, cn, dn);
    CHECK(sn == doctest::Approx(1.0).epsilon(1e-12));
    an::jacobi(2 * K, m, sn, cn, dn);
    CHECK(std::fabs(sn) < 1e-12);
    double s1, c1, d1, s2, c2, d2;
    an::jacobi(0.7, m, s1, c1, d1);
    an::jacobi(0.7 + 4 * K, m, s2, c2, d2);
    CHECK(s2 == doctest::Approx(s1).epsilon(1e-11));
  }
  double sn, cn, dn;
  an::jacobi(0.9, 0.0, sn, cn, dn);
  CHECK(sn == doctest::Approx(std::sin(0.9)).epsilon(1e-14));
}

TEST_CASE("kerr_analytic: Legendre F, K and Pi (Carlson RF, RJ) against quadrature of their definitions") {
  for (double m : {0.4, -0.3, -2.3, -25.0}) {
    for (double phi : {0.2, 0.9, 1.4, 1.5707963267948966}) {
      const double ref = an::tanh_sinh([&](double x, double, double) { const double s = std::sin(x); return 1.0 / std::sqrt(1.0 - m * s * s); }, 0.0, phi);
      CHECK(an::ellip_F(phi, m) == doctest::Approx(ref).epsilon(1e-12));
      for (double n : {0.1, 0.6, 0.95}) {
        const double refp = an::tanh_sinh([&](double x, double, double) { const double s = std::sin(x); return 1.0 / ((1.0 - n * s * s) * std::sqrt(1.0 - m * s * s)); }, 0.0, phi);
        CHECK(an::ellip_Pi(n, phi, m) == doctest::Approx(refp).epsilon(1e-12));
      }
    }
    const double refp = an::tanh_sinh([&](double x, double, double) { const double s = std::sin(x); return 1.0 / ((1.0 - 0.7 * s * s) * std::sqrt(1.0 - m * s * s)); }, 0.0, 1.5707963267948966);
    CHECK(an::ellip_Pi_complete(0.7, m) == doctest::Approx(refp).epsilon(1e-12));
  }
}

// ---- the geodesics: closed-form polar motion + radial Mino integrals against the step-by-step integrator (kerr::trace at tight tolerance) ----
namespace {
double wrap(double x) { return x - 2.0 * M_PI * std::floor(x / (2.0 * M_PI) + 0.5); }

// photon propagation direction in the ZAMO frame (kerr::trace convention: the PHOTON's direction, opposite to the viewing one). `toward` = the camera looks
// `delta` radians off the hole (nr > 0: the traced ray goes in); otherwise it looks away from the hole (nr < 0: the traced ray goes straight out)
void dir(double delta, double psi, bool toward, double& nr, double& nth, double& nph) {
  nr = toward ? std::cos(delta) : -std::cos(delta);
  nth = std::sin(delta) * std::sin(psi);
  nph = std::sin(delta) * std::cos(psi);
}

struct Stats { double worst_sky = 0, worst_disk_r = 0, worst_disk_phi = 0; int n = 0, fate_mismatch = 0, unsupported = 0, esc = 0, cap = 0, disk = 0; };

Stats compare(double a, double r0, double theta0, bool with_disk) {
  Stats st;
  kerr::Options opt;
  opt.atol = 1e-13; opt.rtol = 1e-12; opt.r_far = 1e9; opt.max_steps = 4000000;
  kerr::Disk disk;
  if (with_disk) { disk.r_in = kerr::isco(a); disk.r_out = 20.0; }
  for (double delta : {0.02, 0.06, 0.1, 0.12, 0.135, 0.15, 0.2, 0.35, 0.8}) {
    for (double psi : {0.3, 1.2, 2.5, 3.5, 4.4, 5.6}) for (bool toward : {true, false}) {
      double nr, nth, nph;
      dir(delta, psi, toward, nr, nth, nph);
      const kerr::Trace ref = kerr::trace(a, r0, theta0, nr, nth, nph, disk, opt);
      an::Options ao;  // r_far = 0: the true asymptotic direction (kerr::trace stops at 1e9: O(b / r_far) ~ 1e-8)
      const an::Result got = an::trace(a, r0, theta0, nr, nth, nph, disk, ao);
      ++st.n;
      if (!got.supported) { ++st.unsupported; continue; }
      if (ref.fate == kerr::Fate::Orbiting) continue;
      if (got.t.fate != ref.fate || got.t.disk_hit != ref.disk_hit) { ++st.fate_mismatch; INFO("fate mismatch a=" << a << " delta=" << delta << " psi=" << psi << " ref " << int(ref.fate) << "/" << ref.disk_hit << " got " << int(got.t.fate) << "/" << got.t.disk_hit); CHECK(false); continue; }
      if (ref.disk_hit) {
        ++st.disk;
        st.worst_disk_r = std::max(st.worst_disk_r, std::fabs(got.t.disk_r - ref.disk_r));
        st.worst_disk_phi = std::max(st.worst_disk_phi, std::fabs(wrap(got.t.disk_phi - ref.disk_phi)));
      } else if (ref.fate == kerr::Fate::Escaped) {
        ++st.esc;
        const double dth = std::fabs(got.t.theta_end - ref.theta_end), dph = std::fabs(wrap(got.t.phi_end - ref.phi_end)) * std::sin(ref.theta_end);
        st.worst_sky = std::max({st.worst_sky, dth, dph});
      } else ++st.cap;
    }
  }
  return st;
}
}  // namespace

TEST_CASE("kerr_analytic: sky direction of escaping rays matches the integrator (no disk)") {
  for (double a : {0.0, 0.5, 0.9, 0.998}) {
    const Stats st = compare(a, 40.0, 80.0 * M_PI / 180.0, false);
    INFO("a = " << a << ": " << st.esc << " escaped, " << st.cap << " captured, " << st.unsupported << " unsupported, worst sky error " << st.worst_sky);
    CHECK(st.fate_mismatch == 0);
    CHECK(st.unsupported == 0);  // vortical rays (eta < 0) included
    CHECK(st.esc > 10);
    CHECK(st.worst_sky < 5e-8);  // the integrator's own stop at r = 1e9 is ~1e-8 from the asymptote
  }
}

TEST_CASE("kerr_analytic: first equatorial crossing inside the disk (radius and azimuth) matches the integrator") {
  for (double a : {0.0, 0.5, 0.9}) {
    for (double th : {80.0, 30.0}) {
      const Stats st = compare(a, 40.0, th * M_PI / 180.0, true);
      INFO("a = " << a << " theta_o = " << th << ": " << st.disk << " disk hits, worst r error " << st.worst_disk_r << ", worst phi error " << st.worst_disk_phi);
      CHECK(st.fate_mismatch == 0);
      CHECK(st.unsupported == 0);
      CHECK(st.disk > 5);
      CHECK(st.worst_disk_r < 1e-8);
      CHECK(st.worst_disk_phi < 1e-8);
    }
  }
}

// ---- the cases where an analytic method usually breaks ----
namespace {
struct One { double delta, psi; bool toward; };
// compares one ray set; returns the worst errors (sky, disk radius, disk azimuth) and counts
// Rays with |L| < 1e-3 (meridional: they cross the pole; the l_min floor of both codes applies) are not compared with kerr::trace: its DP45 gives different
// answers at tolerances 1e-12 and 1e-14 there (8e-5 in the disk radius, 0.5 rad in phi; the turning latitude sits ~1e-9 from the pole, the reprojection onto
// sqrt(M) is ill-conditioned, cf. notes/physics/33-dop853.md). They are covered by the RK4 test of the polar motion instead. `degenerate` counts them.
Stats compare_rays(double a, double r0, double theta0, const std::vector<One>& rays, int skip, double r_out = 20.0, int* degenerate = nullptr) {
  Stats st;
  kerr::Options opt;
  opt.atol = 1e-14; opt.rtol = 1e-13; opt.r_far = 1e12; opt.max_steps = 20000000;
  kerr::Disk disk; disk.r_in = kerr::isco(a); disk.r_out = r_out; disk.skip = skip;  // r_out = 0: no disk
  for (const One& o : rays) {
    double nr, nth, nph;
    dir(o.delta, o.psi, o.toward, nr, nth, nph);
    if (std::fabs(kerr::init_from_zamo(a, r0, theta0, nr, nth, nph).L) < 1e-3) { if (degenerate) ++*degenerate; continue; }
    const kerr::Trace ref = kerr::trace(a, r0, theta0, nr, nth, nph, disk, opt);
    const an::Result got = an::trace(a, r0, theta0, nr, nth, nph, disk, {});
    ++st.n;
    if (!got.supported) { ++st.unsupported; continue; }
    if (ref.fate == kerr::Fate::Orbiting) continue;
    if (got.t.fate != ref.fate || got.t.disk_hit != ref.disk_hit) { ++st.fate_mismatch; INFO("a=" << a << " theta=" << theta0 << " delta=" << o.delta << " psi=" << o.psi << " ref " << int(ref.fate) << "/" << ref.disk_hit << " got " << int(got.t.fate) << "/" << got.t.disk_hit); CHECK(false); continue; }
    if (ref.disk_hit) {
      ++st.disk;
      st.worst_disk_r = std::max(st.worst_disk_r, std::fabs(got.t.disk_r - ref.disk_r) / ref.disk_r);
      st.worst_disk_phi = std::max(st.worst_disk_phi, std::fabs(wrap(got.t.disk_phi - ref.disk_phi)));
    } else if (ref.fate == kerr::Fate::Escaped) {
      ++st.esc;
      st.worst_sky = std::max({st.worst_sky, std::fabs(got.t.theta_end - ref.theta_end), std::fabs(wrap(got.t.phi_end - ref.phi_end)) * std::sin(ref.theta_end)});
    } else ++st.cap;
  }
  return st;
}
}  // namespace

TEST_CASE("kerr_analytic: rays close to the critical curve (photon ring), where r3 and r4 almost coincide") {
  // the critical curve of a Schwarzschild observer at r = 40 is at b = 3 sqrt(3) = 5.196: delta ~ asin(b / 40) (in flat space; the exact shift is irrelevant, the sweep crosses it)
  std::vector<One> rays;
  for (double d = 0.1220; d <= 0.1360; d += 0.0004)
    for (double psi : {0.7, 2.0, 4.0}) rays.push_back({d, psi, true});
  for (double a : {0.0, 0.5, 0.9, 0.998}) {
    const Stats st = compare_rays(a, 40.0, 80.0 * M_PI / 180.0, rays, 0, 0.0);  // no disk: the sweep must show escapes, captures and the photon ring
    INFO("a = " << a << ": " << st.esc << " escaped, " << st.cap << " captured, " << st.disk << " disk, unsupported " << st.unsupported << ", worst sky " << st.worst_sky << ", disk r (rel) " << st.worst_disk_r << ", disk phi " << st.worst_disk_phi);
    CHECK(st.fate_mismatch == 0);
    CHECK(st.esc > 10);
    CHECK(st.cap > 10);
    CHECK(st.worst_sky < 5e-9);  // measured ~1e-11: the integrator stops at 1e12, b/r = 3e-11
  }
}

TEST_CASE("kerr_analytic: higher-order images (skip = 1, 2: the ray goes on behind the disk)") {
  std::vector<One> rays;
  for (double d : {0.05, 0.09, 0.12, 0.128, 0.13, 0.131, 0.135, 0.15, 0.3})
    for (double psi : {0.4, 1.0, 2.2, 3.3, 4.1, 5.3}) rays.push_back({d, psi, true});
  for (int skip : {1, 2}) {
    for (double a : {0.0, 0.9}) {
      const Stats st = compare_rays(a, 40.0, 75.0 * M_PI / 180.0, rays, skip);
      INFO("skip = " << skip << " a = " << a << ": " << st.disk << " disk hits, worst r (rel) " << st.worst_disk_r << ", worst phi " << st.worst_disk_phi);
      CHECK(st.fate_mismatch == 0);
      if (skip == 1) CHECK(st.disk > 3);  // third crossings are rare on this grid; the RK4 test below covers skip = 2
      CHECK(st.worst_disk_r < 1e-7);
      CHECK(st.worst_disk_phi < 1e-7);
    }
  }
}

TEST_CASE("kerr_analytic: observer on the equator, near the pole, and rays over the pole (lambda = 0)") {
  std::vector<One> rays;
  for (double d : {0.03, 0.08, 0.125, 0.14, 0.25, 0.6})
    for (double psi : {0.0, 0.5, 1.5707963267948966, 3.141592653589793, 4.7123889803848631, 5.9}) rays.push_back({d, psi, true});
  for (double th : {90.0, 89.0, 5.0, 1.0}) {
    for (double a : {0.5, 0.95}) {
      int degenerate = 0;
      const Stats st = compare_rays(a, 40.0, th * M_PI / 180.0, rays, 0, 20.0, &degenerate);
      INFO("theta_o = " << th << " a = " << a << " (" << degenerate << " meridional rays left to the RK4 test): esc " << st.esc << " cap " << st.cap << " disk " << st.disk << " unsupported " << st.unsupported << ", worst sky " << st.worst_sky << ", disk r (rel) " << st.worst_disk_r << ", disk phi " << st.worst_disk_phi);
      CHECK(st.fate_mismatch == 0);
      CHECK(st.worst_sky < 1e-7);
      CHECK(st.worst_disk_r < 1e-7);
      CHECK(st.worst_disk_phi < 1e-7);
    }
  }
}

// ---- the polar motion against a third, independent reference: RK4 in long double on d2mu/ds2 = M'(mu)/2 and dphi/ds = lambda/(1 - mu^2) ----
namespace {
struct Polar { long double mu, v, ph; };
Polar rk4_polar(long double a, long double lam, long double eta, Polar y, long double S, long n) {
  auto f = [&](const Polar& p) {
    const long double mu = p.mu;
    return Polar{p.v, 0.5L * (2.0L * (a * a - eta - lam * lam) * mu - 4.0L * a * a * mu * mu * mu), lam / (1.0L - mu * mu)};
  };
  const long double h = S / n;
  for (long i = 0; i < n; ++i) {
    const Polar k1 = f(y);
    const Polar k2 = f({y.mu + 0.5L * h * k1.mu, y.v + 0.5L * h * k1.v, y.ph + 0.5L * h * k1.ph});
    const Polar k3 = f({y.mu + 0.5L * h * k2.mu, y.v + 0.5L * h * k2.v, y.ph + 0.5L * h * k2.ph});
    const Polar k4 = f({y.mu + h * k3.mu, y.v + h * k3.v, y.ph + h * k3.ph});
    y = {y.mu + h / 6.0L * (k1.mu + 2 * k2.mu + 2 * k3.mu + k4.mu), y.v + h / 6.0L * (k1.v + 2 * k2.v + 2 * k3.v + k4.v), y.ph + h / 6.0L * (k1.ph + 2 * k2.ph + 2 * k3.ph + k4.ph)};
  }
  return y;
}
}  // namespace

TEST_CASE("kerr_analytic: closed-form polar motion mu(s) and its azimuth against RK4 (ordinary, vortical and meridional rays)") {
  struct Case { double a, lam, eta, mu0, vsign; };
  // eta > 0: ordinary; eta < 0: vortical (needs a^2 cos^2 > |eta|); lam = 1e-4: the meridional rays at the l_min floor (turning latitude within ~1e-9 of the pole)
  const Case cases[] = {{0.9, 0.7, 3.0, 0.17, 1}, {0.9, -2.5, 20.0, -0.4, 1}, {0.5, 4.0, 0.3, 0.05, -1}, {0.998, 1.1, 8.0, 0.9, -1},
                        {0.9, 0.39, -0.08, 0.5, 1}, {0.9, -0.33, -0.04, 0.6, -1}, {0.998, 0.15, -0.09, 0.4, 1},
                        {0.5, 1e-4, 32.5, 0.99, -1}, {0.95, 1e-4, 50.0, 0.2, 1}, {0.0, 3.0, 5.0, 0.3, 1}, {0.3, 6.0, 40.0, 0.2, -1}};
  for (const Case& c : cases) {
    const double Mmu0 = (1.0 - c.mu0 * c.mu0) * (c.eta + c.a * c.a * c.mu0 * c.mu0) - c.lam * c.lam * c.mu0 * c.mu0;
    REQUIRE(Mmu0 > 0.0);  // the case is physical
    const double v0 = c.vsign * std::sqrt(Mmu0);
    an::Angular ang;
    REQUIRE(ang.init(c.a, c.lam, c.eta, c.mu0, v0));
    for (double S : {0.3, 1.7, 4.0, 9.0}) {
      const Polar ref = rk4_polar(c.a, c.lam, c.eta, {c.mu0, v0, 0.0L}, S, 400000);
      const double mu = ang.mu(S);
      INFO("a = " << c.a << " lam = " << c.lam << " eta = " << c.eta << " mu0 = " << c.mu0 << " s = " << S << ": mu " << mu << " vs RK4 " << (double)ref.mu << (ang.vortical ? " (vortical)" : ""));
      CHECK(std::fabs(mu - (double)ref.mu) < 1e-9);
      if (std::fabs(c.lam) > 0.05) CHECK(std::fabs(ang.phi_lam(S) - (double)ref.ph) < 1e-8);
    }
  }
}

TEST_CASE("kerr_analytic: the equatorial crossings are where RK4 finds mu = 0") {
  struct Case { double a, lam, eta, mu0, vsign; };
  const Case cases[] = {{0.9, 0.7, 3.0, 0.17, 1}, {0.9, -2.5, 20.0, -0.4, 1}, {0.5, 4.0, 0.3, 0.05, -1}, {0.998, 1.1, 8.0, 0.9, -1}, {0.5, 1e-4, 32.5, 0.99, -1}, {0.0, 3.0, 5.0, 0.3, 1}};
  for (const Case& c : cases) {
    const double Mmu0 = (1.0 - c.mu0 * c.mu0) * (c.eta + c.a * c.a * c.mu0 * c.mu0) - c.lam * c.lam * c.mu0 * c.mu0;
    const double v0 = c.vsign * std::sqrt(Mmu0);
    an::Angular ang;
    REQUIRE(ang.init(c.a, c.lam, c.eta, c.mu0, v0));
    for (long j = ang.first_cross_index(); j < ang.first_cross_index() + 3; ++j) {
      const double sj = ang.s_cross(j);
      REQUIRE(sj > 0.0);
      const Polar ref = rk4_polar(c.a, c.lam, c.eta, {c.mu0, v0, 0.0L}, sj, 400000);
      INFO("a = " << c.a << " eta = " << c.eta << " crossing " << j << " at s = " << sj << ": RK4 mu there = " << (double)ref.mu);
      CHECK(std::fabs((double)ref.mu) < 1e-9);
    }
  }
}

// ---- the whole ray (radial part included) against RK4 in long double on the Carter equations, up to the disk crossing ----
namespace {
struct Full { long double r, vr, mu, vmu, ph; };
Full rk4_full(long double a, long double lam, long double eta, Full y, long double S, long n) {
  auto f = [&](const Full& p) {
    const long double r = p.r, mu = p.mu, K = eta + (lam - a) * (lam - a), A = r * r + a * a - a * lam, Dl = r * r - 2.0L * r + a * a;
    const long double dR = 4.0L * r * A - (2.0L * r - 2.0L) * K;
    const long double dM = 2.0L * (a * a - eta - lam * lam) * mu - 4.0L * a * a * mu * mu * mu;
    return Full{p.vr, 0.5L * dR, p.vmu, 0.5L * dM, -(lam / (1.0L - mu * mu) - a + a * A / Dl)};  // backward: dphi/ds = -dphi/dtau
  };
  const long double h = S / n;
  for (long i = 0; i < n; ++i) {
    const Full k1 = f(y);
    auto add = [&](const Full& k, long double c) { return Full{y.r + c * k.r, y.vr + c * k.vr, y.mu + c * k.mu, y.vmu + c * k.vmu, y.ph + c * k.ph}; };
    const Full k2 = f(add(k1, 0.5L * h)), k3 = f(add(k2, 0.5L * h)), k4 = f(add(k3, h));
    y = {y.r + h / 6.0L * (k1.r + 2 * k2.r + 2 * k3.r + k4.r), y.vr + h / 6.0L * (k1.vr + 2 * k2.vr + 2 * k3.vr + k4.vr), y.mu + h / 6.0L * (k1.mu + 2 * k2.mu + 2 * k3.mu + k4.mu),
         y.vmu + h / 6.0L * (k1.vmu + 2 * k2.vmu + 2 * k3.vmu + k4.vmu), y.ph + h / 6.0L * (k1.ph + 2 * k2.ph + 2 * k3.ph + k4.ph)};
  }
  return y;
}
}  // namespace

TEST_CASE("kerr_analytic: disk crossings (radius and azimuth) against RK4 in long double, up to the 3rd crossing") {
  int hits = 0, turned = 0, multi = 0;
  double worst_r = 0, worst_phi = 0;
  for (double a : {0.0, 0.9, 0.998}) {
    for (double th : {80.0, 40.0, 5.0}) {
      for (double delta : {0.03, 0.09, 0.12, 0.128, 0.1295, 0.131, 0.14, 0.2, 0.5}) {
        for (double psi : {0.0, 0.4, 1.9, 3.141592653589793, 3.3, 5.0}) {  // psi = 0, pi: nth = 0, the ray starts AT its turning latitude (dmu/ds = 0)
          for (int skip : {0, 1, 2}) {
            double nr, nth, nph;
            dir(delta, psi, true, nr, nth, nph);
            kerr::Disk disk; disk.r_in = kerr::isco(a); disk.r_out = 20.0; disk.skip = skip;
            const an::Result got = an::trace(a, 40.0, th * M_PI / 180.0, nr, nth, nph, disk, {});
            if (!got.supported || !got.t.disk_hit) continue;
            const double thr = th * M_PI / 180.0;
            const auto in = kerr::init_from_zamo(a, 40.0, thr, nr, nth, nph);
            const double lam = got.t.L, eta = got.t.Q;
            // The RK4 integrates the SECOND-order equations, which do not enforce (dr/ds)^2 = R and (dmu/ds)^2 = M: velocities from init_from_zamo (double) are off
            // those first integrals by ~1.7e-16 relative, and that mismatch grows to ~4e-9 in the radius of a third-image crossing (measured: with velocities
            // sqrt(R), sqrt(M) in long double the RK4 converges to the oracle, 9e-11 at 2.4M steps). So the reference starts ON the first integrals, with the data's signs.
            const long double aL = a, lamL = lam, etaL = eta, muL = std::cos(thr), KL = etaL + (lamL - aL) * (lamL - aL), AL = 40.0L * 40.0L + aL * aL - aL * lamL;
            const long double RL = AL * AL - (40.0L * 40.0L - 80.0L + aL * aL) * KL, ML = (1 - muL * muL) * (etaL + aL * aL * muL * muL) - lamL * lamL * muL * muL;
            const double vr0 = -kerr::delta(40.0, a) * in.pr, vmu0d = std::sin(thr) * in.ptheta;
            // a ray that starts AT its turning latitude has dmu/ds = 0 exactly (nth = 0): sqrt(M(mu0)) would be rounding noise (~1e-8), a spurious velocity
            const long double vmuL = std::fabs(vmu0d) < 1e-6 ? (long double)vmu0d : (vmu0d < 0 ? -1.0L : 1.0L) * std::sqrt(ML);
            const Full y0{40.0L, (vr0 < 0 ? -1.0L : 1.0L) * std::sqrt(RL), muL, vmuL, 0.0L};
            const Full ref = rk4_full(a, lam, eta, y0, got.s_hit, 1200000);
            ++hits; turned += got.turned; multi += skip > 0;
            const double er = std::fabs(got.t.disk_r - (double)ref.r) / got.t.disk_r;
            worst_r = std::max(worst_r, er);
            INFO("a = " << a << " theta_o = " << th << " delta = " << delta << " psi = " << psi << " skip = " << skip << " L = " << lam << ": oracle r = " << got.t.disk_r << ", RK4 r = " << (double)ref.r);
            CHECK(er < 2e-9);  // the RK4 reference itself is converged to ~4e-10 at 1.2M steps (error x 1/16 per doubling)
            if (std::fabs(lam) > 0.05) {
              const double ep = std::fabs(wrap(got.t.disk_phi - (double)ref.ph));
              worst_phi = std::max(worst_phi, ep);
              CHECK(ep < 1e-8);
            }
          }
        }
      }
    }
  }
  INFO("hits " << hits << ", with a turning point " << turned << ", skip > 0: " << multi << ", worst r (rel) " << worst_r << ", worst phi " << worst_phi);
  CHECK(hits > 100);
  CHECK(turned > 5);
  CHECK(multi > 20);
}

// ---- outward-moving rays that are bent back by the hole (turn = 2): an observer close to the hole looking away; used by the lamppost (photons emitted upward from a low source) ----
TEST_CASE("kerr_analytic: rays that move out, turn at an outer root and fall back in: disk crossings against RK4 in long double") {
  int found = 0, checked = 0;
  double worst_r = 0, worst_phi = 0;
  for (double a : {0.9, 0.5}) {
    for (double r0 : {2.8, 3.0, 3.5, 4.5}) {
      for (double th : {80.0, 60.0}) {
        for (double delta : {0.5, 0.7, 0.9, 1.1, 1.3, 1.45, 1.55, 1.65}) {
          for (double psi : {0.4, 1.2, 1.9, 2.6, 3.3, 5.0}) {
            const double thr = th * M_PI / 180.0;
            // the photon moves outward (nr < 0 here = the traced ray goes out) with a transverse part large enough to meet the potential barrier
            const double nr = -std::cos(delta), nth = std::sin(delta) * std::sin(psi), nph = std::sin(delta) * std::cos(psi);
            kerr::Disk disk; disk.r_in = kerr::isco(a); disk.r_out = 60.0;
            const an::Result got = an::trace(a, r0, thr, nr, nth, nph, disk, {});
            if (!got.supported || got.turn != 2) continue;
            ++found;
            if (!got.t.disk_hit) continue;
            const auto in = kerr::init_from_zamo(a, r0, thr, nr, nth, nph);
            const double lam = got.t.L, eta = got.t.Q;
            const long double aL = a, lamL = lam, etaL = eta, muL = std::cos(thr), r0L = r0, KL = etaL + (lamL - aL) * (lamL - aL), AL = r0L * r0L + aL * aL - aL * lamL;
            const long double RL = AL * AL - (r0L * r0L - 2 * r0L + aL * aL) * KL, ML = (1 - muL * muL) * (etaL + aL * aL * muL * muL) - lamL * lamL * muL * muL;
            const double vr0 = -kerr::delta(r0, a) * in.pr, vmu0d = std::sin(thr) * in.ptheta;
            const long double vmuL = std::fabs(vmu0d) < 1e-6 ? (long double)vmu0d : (vmu0d < 0 ? -1.0L : 1.0L) * std::sqrt(ML);
            const Full y0{(long double)r0, (vr0 < 0 ? -1.0L : 1.0L) * std::sqrt(RL), muL, vmuL, 0.0L};
            const Full ref = rk4_full(a, lam, eta, y0, got.s_hit, 1200000);
            ++checked;
            const double er = std::fabs(got.t.disk_r - (double)ref.r) / got.t.disk_r;
            worst_r = std::max(worst_r, er);
            INFO("a = " << a << " r0 = " << r0 << " theta_o = " << th << " delta = " << delta << " psi = " << psi << ": oracle r = " << got.t.disk_r << ", RK4 r = " << (double)ref.r);
            CHECK(er < 2e-9);
            if (std::fabs(lam) > 0.05) {
              const double ep = std::fabs(wrap(got.t.disk_phi - (double)ref.ph));
              worst_phi = std::max(worst_phi, ep);
              CHECK(ep < 1e-8);
            }
          }
        }
      }
    }
  }
  INFO(found << " rays with an outer turning point, " << checked << " of them hit the disk; worst r (rel) " << worst_r << ", worst phi " << worst_phi);
  CHECK(found > 30);
  CHECK(checked > 15);
}

// kerr::trace (DP45 in mu) is accurate for meridional rays at ordinary tolerances and says so when the tolerance asks for more than mu = cos(theta) can hold near the pole.
TEST_CASE("kerr::trace meridional rays: accurate at atol >= 1e-10, flagged precision_limited when atol <= 1e-12 (the oracle is the reference there)") {
  int n = 0, flagged = 0;
  for (double a : {0.5, 0.9, 0.998}) {
    const double r0 = 40.0, th0 = 60.0 * M_PI / 180.0;
    kerr::Disk d; d.r_in = kerr::isco(a); d.r_out = 20.0;
    for (double dl : {0.14, 0.3}) {
      for (double cp : {1e-5, 1e-4, 1e-3}) {
        const double nr = std::cos(dl), nth = std::sin(dl) * std::sqrt(1.0 - cp * cp), nph = std::sin(dl) * cp;
        const an::Result g = an::trace(a, r0, th0, nr, nth, nph, d, {});
        if (!g.supported || !g.t.disk_hit) continue;
        kerr::Options o; o.atol = 1e-10; o.rtol = 1e-11; o.r_far = 1e12; o.max_steps = 20000000;
        const kerr::Trace t = kerr::trace(a, r0, th0, nr, nth, nph, d, o);
        INFO("a = " << a << " delta = " << dl << " cp = " << cp << " min_sigma = " << t.min_sigma);
        REQUIRE(t.disk_hit);
        CHECK(std::fabs(t.disk_r - g.t.disk_r) / g.t.disk_r < 1e-6);  // 1.4e-7 measured at min_sigma = 7e-11 (a = 0.998)
        CHECK(!t.precision_limited(1e-10));
        if (t.min_sigma < 1e-6) { CHECK(t.precision_limited(1e-12)); ++flagged; }
        ++n;
      }
    }
  }
  CHECK(n >= 10);
  CHECK(flagged >= 3);  // the test does exercise the near-pole regime
}
