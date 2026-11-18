#include <doctest/doctest.h>

#include <bhs/lorentz.hpp>
#include <bhs/route.hpp>
#include <cmath>
#include <initializer_list>

using namespace bhs;

namespace {
route::Route make(const char* text) {
  json::Value v; std::string err;
  REQUIRE(json::parse(text, v, err));
  route::Route r;
  REQUIRE(route::from_json(v, r, err));
  return r;
}
const char* kBase = R"({
  "name": "t", "fps": 60, "seconds": 10,
  "spin": 0.9, "fov": [{"t": 0, "v": 40}, {"t": 10, "v": 20}], "exposure": [{"t": 0, "v": 3}],
  "ship": {"mode": "orbit", "r": 30, "theta": 80, "phi": 0, "timescale": 5,
           "burns": [{"t0": 2, "t1": 3, "acc": [0.02, 0, 0]}]},
  "view": [{"t": 0, "yaw": 0, "pitch": 0, "roll": 0}, {"t": 10, "yaw": 90, "pitch": 0, "roll": 0}],
  "disk": {"hr": 0.05, "tau": 4, "turb": 1.0, "time": {"mode": "rate", "rate": 20, "t0": 5}}
})";
}  // namespace

TEST_CASE("route: parses a route and rejects malformed ones with a message") {
  const auto r = make(kBase);
  CHECK(r.fps == 60);
  CHECK(r.seconds == 10.0);
  CHECK(r.spin_value(0.0) == 0.9);
  CHECK(r.burns.size() == 1);
  CHECK(r.disk.hr == 0.05);
  CHECK(r.disk.time_mode == "rate");
  json::Value v; std::string err; route::Route bad;
  REQUIRE(json::parse(R"({"fps": -5})", v, err));
  CHECK_FALSE(route::from_json(v, bad, err));
  CHECK_FALSE(err.empty());
  REQUIRE(json::parse(R"({"fps": 60, "seconds": 5, "ship": {"mode": "warp"}})", v, err));
  CHECK_FALSE(route::from_json(v, bad, err));
  REQUIRE(json::parse(R"({"fps": 60, "seconds": 5, "ship": {"burns": [{"t0": 3, "t1": 2, "acc": [0,0,0]}]}})", v, err));
  CHECK_FALSE(route::from_json(v, bad, err));
}

TEST_CASE("route: keyframes interpolate with hold before/after and smooth easing hits the endpoints") {
  const auto r = make(kBase);
  CHECK(r.fov_value(-1.0) == 40.0);
  CHECK(r.fov_value(0.0) == 40.0);
  CHECK(r.fov_value(10.0) == doctest::Approx(20.0).epsilon(1e-12));
  CHECK(r.fov_value(99.0) == 20.0);
  CHECK(r.fov_value(5.0) == doctest::Approx(30.0).epsilon(1e-12));   // smoothstep(0.5) = 0.5
  CHECK(r.fov_value(2.5) == doctest::Approx(40.0 - 20.0 * 0.15625).epsilon(1e-12));  // smoothstep(0.25) = 0.15625: slower than linear at the start
  CHECK(r.exposure_value(7.0) == 3.0);                               // a single keyframe: constant
}

TEST_CASE("route: view orientation follows slerp between yaw keyframes at constant angular rate") {
  auto r = make(kBase);
  r.ease_smooth = false;
  const quat::Q q0 = r.view_quat(0.0), q5 = r.view_quat(5.0), q10 = r.view_quat(10.0);
  CHECK(quat::angle_between(q0, q10) == doctest::Approx(M_PI / 2).epsilon(1e-9));
  CHECK(quat::angle_between(q0, q5) == doctest::Approx(M_PI / 4).epsilon(1e-9));
  // yaw 90: the "fwd" axis rotates to "right" (flight.hpp convention)
  const Vec3 fwd = quat::rotate(q10, Vec3{-1, 0, 0}), right = quat::rotate(q10, Vec3{0, 0, 1});
  CHECK(fwd.z == doctest::Approx(1.0).epsilon(1e-9));
  CHECK(right.x == doctest::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("route: a coasting orbit stays a circular orbit and conserves E, L, Q between burns") {
  const auto r = make(R"({"fps": 60, "seconds": 6, "spin": 0.9, "ship": {"mode": "orbit", "r": 20, "theta": 90, "phi": 0, "timescale": 5}})");
  const auto tr = route::build_trajectory(r, 16);
  for (double t : {0.0, 1.0, 3.3, 6.0}) {
    const auto s = tr.at(t);
    CHECK(s.r == doctest::Approx(20.0).epsilon(1e-6));
    CHECK(s.theta == doctest::Approx(M_PI / 2).epsilon(1e-6));
  }
  CHECK(tr.at(6.0).phi > tr.at(1.0).phi);  // advances
  // coordinate and proper time: 6 s of video x timescale 5 = 30 M of proper time; t > tau (dilation)
  CHECK(tr.at(6.0).tau == doctest::Approx(30.0).epsilon(1e-6));
  CHECK(tr.at(6.0).t > tr.at(6.0).tau);
}

TEST_CASE("route: constants of motion are conserved except inside burns (E, L, Q from the pilot at the grid points)") {
  const auto r = make(kBase);
  const auto tr = route::build_trajectory(r, 16);
  REQUIRE(tr.consts.size() == tr.pts.size());
  auto at = [&](double t) { return tr.consts[size_t(std::round(t / tr.dt))]; };
  const auto c0 = at(0.0), c1 = at(1.9), c2 = at(3.2), c3 = at(8.0);  // before, near the end of the 1st segment, after the burn, at the end
  CHECK(c0.E == doctest::Approx(c1.E).epsilon(1e-8));
  CHECK(c0.L == doctest::Approx(c1.L).epsilon(1e-8));
  CHECK(c0.Q == doctest::Approx(c1.Q).epsilon(1e-6));
  CHECK(std::abs(c2.E - c1.E) > 1e-6);                               // the burn changed the constants
  CHECK(c2.E == doctest::Approx(c3.E).epsilon(1e-8));
  CHECK(c2.L == doctest::Approx(c3.L).epsilon(1e-8));
}

TEST_CASE("route: the trajectory is deterministic and its lookup interpolates between grid points") {
  const auto r = make(kBase);
  const auto a = route::build_trajectory(r, 16), b = route::build_trajectory(r, 16);
  REQUIRE(a.pts.size() == b.pts.size());
  for (size_t i = 0; i < a.pts.size(); ++i) { CHECK(a.pts[i].r == b.pts[i].r); CHECK(a.pts[i].phi == b.pts[i].phi); }
  const double t = 4.0 + 0.37 * a.dt;
  const auto s = a.at(t), s0 = a.at(4.0), s1 = a.at(4.0 + a.dt);
  CHECK(s.r == doctest::Approx(s0.r + 0.37 * (s1.r - s0.r)).epsilon(1e-9));
}

TEST_CASE("route: disk time is the video time scaled by the rate (mode rate) or the pilot coordinate time (mode pilot)") {
  const auto r = make(kBase);
  CHECK(r.disk_time(2.0, 123.0) == doctest::Approx(5.0 + 20.0 * 2.0).epsilon(1e-12));
  auto p = make(R"({"fps": 60, "seconds": 5, "disk": {"time": {"mode": "pilot", "t0": 10}}})");
  CHECK(p.disk_time(2.0, 123.0) == doctest::Approx(133.0).epsilon(1e-12));
}

TEST_CASE("route: a varying spin is only allowed for a static camera") {
  json::Value v; std::string err; route::Route r;
  REQUIRE(json::parse(R"({"fps": 60, "seconds": 5, "spin": [{"t": 0, "v": 0}, {"t": 5, "v": 0.9}], "ship": {"mode": "orbit"}})", v, err));
  CHECK_FALSE(route::from_json(v, r, err));
  REQUIRE(json::parse(R"({"fps": 60, "seconds": 5, "spin": [{"t": 0, "v": 0}, {"t": 5, "v": 0.9}], "ship": {"mode": "static", "r": 40}})", v, err));
  CHECK(route::from_json(v, r, err));
}

TEST_CASE("route: sample_time spreads the samples stratified inside the shutter around the frame mid-time (physical motion blur)") {
  const int fps = 60, N = 16;
  for (int f : {0, 7, 599}) {
    const double mid = (f + 0.5) / fps;
    double mean = 0;
    for (int k = 0; k < N; ++k) {
      CHECK(route::sample_time(fps, f, k, N, 0.0) == doctest::Approx(mid).epsilon(1e-14));  // shutter 0: no blur
      const double t = route::sample_time(fps, f, k, N, 0.5);
      CHECK(t >= mid - 0.25 / fps - 1e-12);
      CHECK(t <= mid + 0.25 / fps + 1e-12);
      mean += t / N;
    }
    CHECK(mean == doctest::Approx(mid).epsilon(1e-12));
    // one and only one time per stratum of width shutter/(N fps)
    for (int k = 1; k < N; ++k) CHECK(route::sample_time(fps, f, k, N, 0.5) - route::sample_time(fps, f, k - 1, N, 0.5) == doctest::Approx(0.5 / (N * fps)).epsilon(1e-9));
  }
  // shutter 1 covers the whole frame [f/fps, (f+1)/fps]
  CHECK(route::sample_time(60, 10, 0, 8, 1.0) > 10.0 / 60);
  CHECK(route::sample_time(60, 10, 7, 8, 1.0) < 11.0 / 60);
}

// ---------------------------------------------------------------- v1.0b: waypoints followed by thrust, "look at"
TEST_CASE("route waypoints: hover in Schwarzschild requires the proper acceleration M/(r^2 sqrt(1-2M/r)) (closed formula)") {
  // 3 equal waypoints: the target is rest at r = 10; a = 0 => ZAMO = static observer
  const auto r = make(R"({"fps": 60, "seconds": 8, "spin": 0.0, "fov": 40,
    "ship": {"timescale": 4, "waypoints": [{"t": 0, "r": 10, "theta": 90, "phi": 0}, {"t": 4, "r": 10, "theta": 90, "phi": 0}, {"t": 8, "r": 10, "theta": 90, "phi": 0}]}})");
  CHECK(r.mode == "follow");
  const auto tr = route::build_trajectory(r, 16);
  const double expect = 1.0 / (100.0 * std::sqrt(1.0 - 2.0 / 10.0));
  // last second: steady state
  double sum = 0; int n = 0;
  for (size_t i = tr.acc.size() * 3 / 4; i + 1 < tr.acc.size(); ++i) { sum += tr.acc[i]; ++n; }
  CHECK(sum / n == doctest::Approx(expect).epsilon(2e-3));
  CHECK(tr.err_max < 1e-3);                      // the ship stays at the target (M, proper length)
  const auto s = tr.at(8.0);
  CHECK(s.r == doctest::Approx(10.0).epsilon(1e-4));
}

TEST_CASE("route waypoints: waypoints on a geodesic circular orbit ask for small thrust (< 20 % of the hover)") {
  // circular orbit a = 0.5, r = 12, equatorial plane; omega_K = 1/(r^{3/2} + a); the waypoint positions are those of the geodesic (coordinate time)
  const double a = 0.5, rr = 12.0, om = 1.0 / (std::pow(rr, 1.5) + a);
  // u^t of the circular orbit: (1 + a r^-1.5)/sqrt(1 - 3/r + 2a r^-1.5)  (Bardeen, Press & Teukolsky 1972)
  const double ut = (1.0 + a * std::pow(rr, -1.5)) / std::sqrt(1.0 - 3.0 / rr + 2.0 * a * std::pow(rr, -1.5));
  auto run = [&](double dt_wp) {
    std::string wp;
    for (int k = 0; k * dt_wp <= 8.0 + 1e-9; ++k) {
      const double tv = k * dt_wp, tau = tv * 10.0, phi_deg = om * ut * tau * 180.0 / M_PI;
      char buf[200]; std::snprintf(buf, sizeof buf, "%s{\"t\": %.12g, \"r\": %g, \"theta\": 90, \"phi\": %.12g}", k ? "," : "", tv, rr, phi_deg);
      wp += buf;
    }
    const std::string js = std::string(R"({"fps": 60, "seconds": 8, "spin": 0.5, "fov": 40, "ship": {"timescale": 10, "waypoints": [)") + wp + "]}}";
    return route::build_trajectory(make(js.c_str()), 16);
  };
  const auto t1 = run(0.5), t2 = run(0.25), t3 = run(0.125);
  // references: the hover at r = 12 would need 7.6e-3 c/M; without control the ship would fall with ~ 1/r^2 = 6.9e-3
  CHECK(t2.acc_rms < 1.5e-3);                     // < 20 % do hover
  // measured: rms 9.8e-4 / 7.9e-4 / 7.3e-4 for waypoints at 0.5 / 0.25 / 0.125 s: it does NOT converge to 0; the floor (~7e-4 c/M, ~10 % of the local gravity)
  // comes from the controller (1st-order position loop, linear ramp in time), not from the spline. Convergence is not claimed.
  CHECK(t1.acc_rms < 2e-3); CHECK(t3.acc_rms < 1.5e-3);
  CHECK(t2.err_max < 0.05);
  CHECK(t2.at(8.0).r == doctest::Approx(rr).epsilon(1e-3));
}

TEST_CASE("route waypoints: acceleration limit, capture and parsing errors") {
  const auto r = make(R"({"fps": 30, "seconds": 4, "spin": 0.9, "fov": 40,
    "ship": {"timescale": 3, "max_acc": 0.01, "waypoints": [{"t": 0, "r": 20, "theta": 90, "phi": 0}, {"t": 4, "r": 8, "theta": 60, "phi": 90}]}})");
  const auto tr = route::build_trajectory(r, 16);
  CHECK(tr.acc_max <= 0.01 * (1 + 1e-9));
  json::Value v; std::string err; route::Route bad;
  REQUIRE(json::parse(R"({"fps": 30, "seconds": 4, "ship": {"waypoints": [{"t": 1, "r": 20}, {"t": 2, "r": 10}]}})", v, err));
  CHECK_FALSE(route::from_json(v, bad, err));    // the first waypoint must have t = 0
  REQUIRE(json::parse(R"({"fps": 30, "seconds": 4, "ship": {"waypoints": [{"t": 0, "r": 20}]}})", v, err));
  CHECK_FALSE(route::from_json(v, bad, err));    // at least 2
  REQUIRE(json::parse(R"({"fps": 30, "seconds": 4, "ship": {"waypoints": [{"t": 0, "r": 20}, {"t": 2, "r": 10}], "burns": [{"t0": 0, "t1": 1, "acc": [0.1, 0, 0]}]}})", v, err));
  CHECK_FALSE(route::from_json(v, bad, err));    // mutually exclusive with burns
}

TEST_CASE("route lookat: the centre is -e_r; a target on the camera axis gives fwd = pseudo-Cartesian direction; orthonormal, right-handed basis") {
  route::LookAt la; la.on = true; la.hole = true;
  Vec3 b[3];
  route::Route::lookat_basis(la, 20.0, 1.2, 0.7, b);
  CHECK(b[0].x == doctest::Approx(-1.0)); CHECK(b[0].y == doctest::Approx(0.0)); CHECK(b[0].z == doctest::Approx(0.0));
  CHECK(b[1].z == doctest::Approx(1.0)); CHECK(b[2].y == doctest::Approx(-1.0));  // equal to the default basis (e_phi, -e_theta)
  // target at (r, theta, phi) = (20, 90, 90 deg) seen from the ship at (20, 90, 0): Cartesian S = (20,0,0), T = (0,20,0): d = (-20, 20, 0)/|d|
  la.hole = false; la.r = 20; la.theta = 90; la.phi = 90;
  route::Route::lookat_basis(la, 20.0, M_PI / 2, 0.0, b);
  const double s2 = std::sqrt(0.5);
  // e_r = x, e_theta = -z, e_phi = y na nave: d.e_r = -20, d.e_theta = 0, d.e_phi = 20
  CHECK(b[0].x == doctest::Approx(-s2)); CHECK(b[0].y == doctest::Approx(0.0)); CHECK(b[0].z == doctest::Approx(s2));
  for (int i = 0; i < 3; ++i) CHECK(dot(b[i], b[i]) == doctest::Approx(1.0));
  CHECK(dot(b[0], b[1]) == doctest::Approx(0.0).epsilon(1e-12)); CHECK(dot(b[0], b[2]) == doctest::Approx(0.0).epsilon(1e-12)); CHECK(dot(b[1], b[2]) == doctest::Approx(0.0).epsilon(1e-12));
  // direct: right = fwd x up
  const Vec3 c{b[0].y * b[2].z - b[0].z * b[2].y, b[0].z * b[2].x - b[0].x * b[2].z, b[0].x * b[2].y - b[0].y * b[2].x};
  CHECK(c.x == doctest::Approx(b[1].x)); CHECK(c.y == doctest::Approx(b[1].y)); CHECK(c.z == doctest::Approx(b[1].z));
}

TEST_CASE("route camera_path: the camera follows the spline exactly (no dynamics), static observer") {
  const auto r = make(R"({"fps": 30, "seconds": 4, "spin": 0.9, "fov": 40,
    "ship": {"camera_path": true, "timescale": 1, "waypoints": [{"t": 0, "r": 40, "theta": 10, "phi": 0}, {"t": 2, "r": 40, "theta": 45, "phi": 90}, {"t": 4, "r": 40, "theta": 80, "phi": 180}]}})");
  CHECK(r.mode == "path");
  const auto tr = route::build_trajectory(r, 16);
  CHECK(tr.acc.empty());
  const auto s0 = tr.at(0.0), s2 = tr.at(2.0), s4 = tr.at(4.0);
  CHECK(s0.theta == doctest::Approx(10.0 * M_PI / 180).epsilon(1e-9));
  CHECK(s2.theta == doctest::Approx(45.0 * M_PI / 180).epsilon(1e-6)); CHECK(s2.phi == doctest::Approx(M_PI / 2).epsilon(1e-6)); CHECK(s2.r == doctest::Approx(40.0).epsilon(1e-6));
  CHECK(s4.theta == doctest::Approx(80.0 * M_PI / 180).epsilon(1e-9)); CHECK(s4.phi == doctest::Approx(M_PI).epsilon(1e-9));
  CHECK(dot(s2.beta, s2.beta) == 0.0);
  // continuity: the jump between consecutive grid samples is small
  double worst = 0;
  for (size_t i = 1; i < tr.pts.size(); ++i) worst = std::max(worst, std::abs(tr.pts[i].phi - tr.pts[i - 1].phi));
  CHECK(worst < 0.01);
}
