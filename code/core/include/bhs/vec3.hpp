#pragma once
#include <cmath>

namespace bhs {

struct Vec3 {
  double x = 0, y = 0, z = 0;
};
constexpr Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
constexpr double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) { return (1.0 / norm(a)) * a; }

}  // namespace bhs
