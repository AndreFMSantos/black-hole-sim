#include <doctest/doctest.h>

#include <bhs/kerr_lyapunov.hpp>
#include <cmath>

using namespace bhs;

TEST_CASE("kerr lyapunov: gamma = pi at a = 0 (both senses)") {
  CHECK(kerr::lyapunov_equatorial(0.0, true) == doctest::Approx(M_PI).epsilon(1e-9));
  CHECK(kerr::lyapunov_equatorial(0.0, false) == doctest::Approx(M_PI).epsilon(1e-9));
}

TEST_CASE("kerr lyapunov: prograde photon orbit is less unstable, retrograde more, spin 0.9") {
  const double gp = kerr::lyapunov_equatorial(0.9, true), gr = kerr::lyapunov_equatorial(0.9, false);
  CHECK(gp < M_PI);
  CHECK(gr > 0.0);
  CHECK(gp != doctest::Approx(gr));
}

TEST_CASE("kerr lyapunov: ring ratios measured by integration match exp(-gamma) (equatorial rays)") {
  struct Case { double a; bool pro; };
  for (auto cs : {Case{0.0, true}, Case{0.9, true}, Case{0.9, false}}) {
    const double r_ph = kerr::photon_orbit(cs.a, cs.pro);
    const double Lc = std::abs(cs.a == 0.0 ? std::sqrt(27.0) : kerr::xi_photon(r_ph, cs.a));
    const double gamma = kerr::lyapunov_equatorial(cs.a, cs.pro);
    const double D = 1e5, sgn = cs.pro ? 1.0 : -1.0;
    kerr::Options o;
    o.rtol = 1e-11; o.atol = 1e-13;
    auto sweep = [&](double Labs) {  // total sweep |phi_end| of an equatorial ray with exact |L| = Labs
      const double nph = kerr::nphi_for_L(cs.a, D, M_PI / 2, sgn * Labs) ;
      const double cos_a = std::sqrt(1 - nph * nph);
      const auto t = kerr::trace(cs.a, D, M_PI / 2, cos_a, 0.0, nph, {}, o);
      return t.fate == kerr::Fate::Escaped ? std::abs(t.phi_end) : 1e9;
    };
    auto boundary = [&](double target) {
      double lo = Lc * (1 + 1e-13), hi = Lc * 1.6;
      for (int i = 0; i < 90; ++i) { const double m = 0.5 * (lo + hi); (sweep(m) > target ? lo : hi) = m; }
      return 0.5 * (lo + hi);
    };
    double prev = boundary(4 * M_PI) - Lc;
    for (int k = 5; k <= 7; ++k) {
      const double cur = boundary(k * M_PI) - Lc;
      INFO("a=" << cs.a << " pro=" << cs.pro << " k=" << k);
      CHECK(cur / prev == doctest::Approx(std::exp(-gamma)).epsilon(0.03));
      prev = cur;
    }
  }
}
