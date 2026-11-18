#include <doctest/doctest.h>

#include <bhs/json.hpp>
#include <bhs/quat.hpp>
#include <cmath>
#include <initializer_list>

using namespace bhs;

TEST_CASE("json: numbers, strings, bool, null, nested arrays and objects") {
  json::Value v; std::string err;
  REQUIRE(json::parse(R"({"a": -1.5e3, "s": "x\ny\"zA", "t": true, "f": false, "n": null, "arr": [1, [2, 3], {"k": 4}], "o": {"p": 0.25}})", v, err));
  CHECK(v.get("a")->n == -1500.0);
  CHECK(v.get("s")->s == "x\ny\"zA");
  CHECK(v.get("t")->b == true);
  CHECK(v.get("f")->b == false);
  CHECK(v.get("n")->type == json::Value::Null);
  CHECK(v.get("arr")->a.size() == 3);
  CHECK(v.get("arr")->a[1].a[1].n == 3.0);
  CHECK(v.get("arr")->a[2].get("k")->n == 4.0);
  CHECK(v.get("o")->get("p")->n == 0.25);
  CHECK(v.get("missing") == nullptr);
  CHECK(v.num("a", 7.0) == -1500.0);
  CHECK(v.num("missing", 7.0) == 7.0);
}

TEST_CASE("json: errors are reported with a position and never crash") {
  json::Value v; std::string err;
  for (const char* bad : {"{\"a\": 1,}", "[1, 2", "{\"a\" 1}", "tru", "\"abc", "{} extra", "", "[1 2]", "{\"a\": 01x}"}) {
    err.clear();
    CHECK_FALSE(json::parse(bad, v, err));
    CHECK_FALSE(err.empty());
  }
}

TEST_CASE("quat: rotation of a vector by an axis-angle quaternion (right-hand rule)") {
  const Vec3 x{1, 0, 0};
  const quat::Q q = quat::from_axis_angle({0, 0, 1}, M_PI / 2);
  const Vec3 y = quat::rotate(q, x);
  CHECK(y.x == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(y.y == doctest::Approx(1.0).epsilon(1e-12));
  CHECK(y.z == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(quat::norm(q) == doctest::Approx(1.0).epsilon(1e-15));
}

TEST_CASE("quat: slerp hits the endpoints and moves at constant angular speed") {
  const quat::Q a = quat::from_axis_angle({0, 1, 0}, 0.3), b = quat::from_axis_angle({1, 2, 3}, 2.0);
  const quat::Q s0 = quat::slerp(a, b, 0.0), s1 = quat::slerp(a, b, 1.0);
  CHECK(quat::angle_between(s0, a) < 1e-12);
  CHECK(quat::angle_between(s1, b) < 1e-12);
  const double total = quat::angle_between(a, b);
  for (double t : {0.1, 0.25, 0.5, 0.9}) {
    const quat::Q s = quat::slerp(a, b, t);
    CHECK(quat::norm(s) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(quat::angle_between(a, s) == doctest::Approx(t * total).epsilon(1e-9));  // constant angular velocity
  }
}

TEST_CASE("quat: slerp takes the short arc (q and -q are the same rotation)") {
  const quat::Q a = quat::from_axis_angle({0, 0, 1}, 0.1);
  quat::Q b = quat::from_axis_angle({0, 0, 1}, 0.5);
  b = {-b.w, -b.x, -b.y, -b.z};
  const quat::Q m = quat::slerp(a, b, 0.5);
  const Vec3 r = quat::rotate(m, {1, 0, 0});
  CHECK(std::atan2(r.y, r.x) == doctest::Approx(0.3).epsilon(1e-9));
}

TEST_CASE("quat: from_basis recovers the rotation that maps the default basis to a rotated one (round trip)") {
  const Vec3 e0[3] = {{-1, 0, 0}, {0, 0, 1}, {0, -1, 0}};
  const quat::Q q = quat::from_axis_angle({0.3, -0.5, 0.8}, 1.1);
  Vec3 e1[3];
  for (int i = 0; i < 3; ++i) e1[i] = quat::rotate(q, e0[i]);
  const quat::Q r = quat::from_bases(e0, e1);
  CHECK(quat::angle_between(q, r) < 1e-12);
}
