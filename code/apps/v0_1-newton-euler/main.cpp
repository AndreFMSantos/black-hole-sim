// v0.1 — Newton + integrators. Three particles on the SAME initial orbit,
// each with a different integrator: Euler (red), Verlet (green), RK4 (blue).
// Keys: SPACE pause | UP/DOWN speed | E eccentricity | R reset | ESC quit
// Fixed-function OpenGL on purpose (no loader); shaders arrive in v0.2+.
#include <GLFW/glfw3.h>

#include <array>
#include <bhs/integrators.hpp>
#include <cstdio>
#include <deque>

using namespace bhs;

struct Particle {
  Method method;
  State s;
  double e0;
  std::deque<Vec2> trail;
  float r, g, b;
};

static std::array<Particle, 3> make(double ecc_factor) {
  State s0 = circular_orbit(1.0);
  s0.v = ecc_factor * s0.v;
  auto mk = [&](Method m, float r, float g, float b) {
    return Particle{m, s0, energy(s0), {}, r, g, b};
  };
  return {mk(Method::Euler, 1, .25f, .25f), mk(Method::Verlet, .25f, 1, .25f),
          mk(Method::RK4, .35f, .55f, 1)};
}

int main() {
  if (!glfwInit()) return 1;
  GLFWwindow* w = glfwCreateWindow(900, 900, "black-hole-sim v0.1", nullptr, nullptr);
  if (!w) { glfwTerminate(); return 1; }
  glfwMakeContextCurrent(w);
  glfwSwapInterval(1);

  double ecc = 1.0, speed = 1.0;
  bool paused = false;
  auto ps = make(ecc);
  const double dt = 0.005;  // fixed step of the simulation
  double sim_time = 0;
  bool k_space = false, k_r = false, k_e = false;

  while (!glfwWindowShouldClose(w)) {
    glfwPollEvents();
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
    auto edge = [&](int key, bool& was) {
      bool down = glfwGetKey(w, key) == GLFW_PRESS, hit = down && !was;
      was = down;
      return hit;
    };
    if (edge(GLFW_KEY_SPACE, k_space)) paused = !paused;
    if (edge(GLFW_KEY_R, k_r)) { ps = make(ecc); sim_time = 0; }
    if (edge(GLFW_KEY_E, k_e)) { ecc = (ecc > 0.95) ? 0.7 : 1.0; ps = make(ecc); sim_time = 0; }
    if (glfwGetKey(w, GLFW_KEY_UP) == GLFW_PRESS) speed *= 1.02;
    if (glfwGetKey(w, GLFW_KEY_DOWN) == GLFW_PRESS) speed /= 1.02;

    if (!paused) {
      const int n = static_cast<int>(speed * 4);
      for (int i = 0; i < n; ++i) {
        for (auto& p : ps) {
          // Euler can approach the centre on eccentric orbits; guard against 1/0.
          if (norm(p.s.x) > 1e-3) p.s = step(p.method, p.s, dt, [](Vec2 q) { return gravity(q); });
        }
        sim_time += dt;
      }
      for (auto& p : ps) {
        p.trail.push_back(p.s.x);
        if (p.trail.size() > 4000) p.trail.pop_front();
      }
    }

    int fw, fh;
    glfwGetFramebufferSize(w, &fw, &fh);
    glViewport(0, 0, fw, fh);
    glClearColor(0.02f, 0.02f, 0.05f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2.5, 2.5, -2.5, 2.5, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPointSize(14);  // central body
    glColor3f(1, 0.9f, 0.5f);
    glBegin(GL_POINTS); glVertex2d(0, 0); glEnd();

    glLineWidth(1.5f);
    for (auto& p : ps) {
      glColor4f(p.r, p.g, p.b, 1);
      glBegin(GL_LINE_STRIP);
      for (auto& q : p.trail) glVertex2d(q.x, q.y);
      glEnd();
      glPointSize(8);
      glBegin(GL_POINTS); glVertex2d(p.s.x.x, p.s.x.y); glEnd();
    }
    glfwSwapBuffers(w);

    char title[256];
    std::snprintf(title, sizeof title,
                  "v0.1 | t=%.1f | dE/E:  Euler %+.2e  Verlet %+.2e  RK4 %+.2e | x%.1f %s",
                  sim_time,
                  (energy(ps[0].s) - ps[0].e0) / std::abs(ps[0].e0),
                  (energy(ps[1].s) - ps[1].e0) / std::abs(ps[1].e0),
                  (energy(ps[2].s) - ps[2].e0) / std::abs(ps[2].e0), speed,
                  paused ? "[PAUSED]" : "");
    glfwSetWindowTitle(w, title);
  }
  glfwDestroyWindow(w);
  glfwTerminate();
}
