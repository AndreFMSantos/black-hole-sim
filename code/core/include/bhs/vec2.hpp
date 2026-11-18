#pragma once
#include <cmath>

namespace bhs {

struct Vec2 {
  double x = 0, y = 0;
};
constexpr Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
constexpr Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
constexpr Vec2 operator*(double s, Vec2 a) { return {s * a.x, s * a.y}; }
constexpr double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
constexpr double cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
inline double norm(Vec2 a) { return std::sqrt(dot(a, a)); }

}  // namespace bhs
