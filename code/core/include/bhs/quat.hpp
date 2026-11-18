#pragma once
// Unit quaternions for the view orientation (slerp). Convention: rotation of a vector v by q: q v q*, right-hand rule.
#include <algorithm>
#include <cmath>

#include "vec3.hpp"

namespace bhs::quat {

struct Q { double w = 1, x = 0, y = 0, z = 0; };

inline double norm(const Q& q) { return std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z); }
inline Q normalize(Q q) { const double n = norm(q); return {q.w / n, q.x / n, q.y / n, q.z / n}; }
inline Q mul(const Q& a, const Q& b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
inline Q conj(const Q& q) { return {q.w, -q.x, -q.y, -q.z}; }
inline double dot(const Q& a, const Q& b) { return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; }

inline Q from_axis_angle(Vec3 axis, double ang) {
  const double n = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z), s = std::sin(0.5 * ang) / n;
  return {std::cos(0.5 * ang), axis.x * s, axis.y * s, axis.z * s};
}
inline Vec3 rotate(const Q& q, Vec3 v) {
  const Q r = mul(mul(q, Q{0, v.x, v.y, v.z}), conj(q));
  return {r.x, r.y, r.z};
}
// angle (rad) of the relative rotation between two orientations; q and -q are the same rotation. Uses atan2(|vector part|, |w|) of the relative
// rotation: acos near 1 loses half the digits (acos(1 - 1e-16) = 1.5e-8)
inline double angle_between(const Q& a, const Q& b) {
  const Q r = mul(conj(a), b);
  return 2.0 * std::atan2(std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z), std::abs(r.w));
}

// spherical interpolation along the short arc, constant angular velocity
inline Q slerp(const Q& a, Q b, double t) {
  double d = dot(a, b);
  if (d < 0.0) { b = {-b.w, -b.x, -b.y, -b.z}; d = -d; }
  if (d > 0.9995) return normalize(Q{a.w + t * (b.w - a.w), a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)});
  const double th = std::acos(d), s = std::sin(th), wa = std::sin((1 - t) * th) / s, wb = std::sin(t * th) / s;
  return {wa * a.w + wb * b.w, wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z};
}

// rotation R that takes the orthonormal basis e0[i] to the basis e1[i] (R = sum_i e1_i e0_i^T); both bases with the same handedness
inline Q from_bases(const Vec3 e0[3], const Vec3 e1[3]) {
  double M[3][3] = {};
  for (int i = 0; i < 3; ++i) {
    const double a[3] = {e0[i].x, e0[i].y, e0[i].z}, b[3] = {e1[i].x, e1[i].y, e1[i].z};
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) M[r][c] += b[r] * a[c];
  }
  const double tr = M[0][0] + M[1][1] + M[2][2];
  Q q;
  if (tr > 0) { const double s = 2.0 * std::sqrt(tr + 1.0); q = {0.25 * s, (M[2][1] - M[1][2]) / s, (M[0][2] - M[2][0]) / s, (M[1][0] - M[0][1]) / s}; }
  else if (M[0][0] > M[1][1] && M[0][0] > M[2][2]) { const double s = 2.0 * std::sqrt(1.0 + M[0][0] - M[1][1] - M[2][2]); q = {(M[2][1] - M[1][2]) / s, 0.25 * s, (M[0][1] + M[1][0]) / s, (M[0][2] + M[2][0]) / s}; }
  else if (M[1][1] > M[2][2]) { const double s = 2.0 * std::sqrt(1.0 + M[1][1] - M[0][0] - M[2][2]); q = {(M[0][2] - M[2][0]) / s, (M[0][1] + M[1][0]) / s, 0.25 * s, (M[1][2] + M[2][1]) / s}; }
  else { const double s = 2.0 * std::sqrt(1.0 + M[2][2] - M[0][0] - M[1][1]); q = {(M[1][0] - M[0][1]) / s, (M[0][2] + M[2][0]) / s, (M[1][2] + M[2][1]) / s, 0.25 * s}; }
  return normalize(q);
}

}  // namespace bhs::quat
