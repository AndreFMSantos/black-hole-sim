#pragma once
#include "newton.hpp"

namespace bhs {

// All of them take an acceleration function acc(pos) -> Vec2 (autonomous).

// Explicit Euler (order 1, NOT symplectic): uses the old state for both.
// Result: the energy always grows, the orbit spirals outwards.
template <class Acc>
State step_euler(const State& s, double dt, Acc acc) {
  return {s.x + dt * s.v, s.v + dt * acc(s.x)};
}

// Velocity Verlet (order 2, symplectic, time-reversible):
// the energy oscillates but does not drift.
template <class Acc>
State step_verlet(const State& s, double dt, Acc acc) {
  const Vec2 a0 = acc(s.x);
  const Vec2 x1 = s.x + dt * s.v + (0.5 * dt * dt) * a0;
  const Vec2 a1 = acc(x1);
  return {x1, s.v + (0.5 * dt) * (a0 + a1)};
}

// Runge-Kutta 4 (order 4, NOT symplectic): very accurate in the short term,
// but the energy drifts slowly over long times.
template <class Acc>
State step_rk4(const State& s, double dt, Acc acc) {
  const Vec2 k1x = s.v, k1v = acc(s.x);
  const Vec2 k2x = s.v + (0.5 * dt) * k1v, k2v = acc(s.x + (0.5 * dt) * k1x);
  const Vec2 k3x = s.v + (0.5 * dt) * k2v, k3v = acc(s.x + (0.5 * dt) * k2x);
  const Vec2 k4x = s.v + dt * k3v, k4v = acc(s.x + dt * k3x);
  return {s.x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
          s.v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)};
}

enum class Method { Euler, Verlet, RK4 };

template <class Acc>
State step(Method m, const State& s, double dt, Acc acc) {
  switch (m) {
    case Method::Euler: return step_euler(s, dt, acc);
    case Method::Verlet: return step_verlet(s, dt, acc);
    case Method::RK4: return step_rk4(s, dt, acc);
  }
  return s;
}

inline const char* name(Method m) {
  switch (m) {
    case Method::Euler: return "Euler";
    case Method::Verlet: return "Verlet";
    case Method::RK4: return "RK4";
  }
  return "?";
}

}  // namespace bhs
