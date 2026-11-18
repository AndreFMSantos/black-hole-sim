// v0.2 — Light bending around a Schwarzschild black hole (2D, equatorial plane).
// A beam of parallel rays comes from the left. Each ray is a null geodesic (Binet equation).
//   red = captured (b < sqrt(27) M) | blue->white = escapes (brighter = more deflected)
//   filled circle = horizon r=2M | dashed = photon sphere r=3M | dotted = shadow b=sqrt(27)M
// Keys: UP/DOWN more/fewer rays | N straight lines (Newtonian light, no curvature) | ESC quit
#include <GLFW/glfw3.h>

#include <algorithm>
#include <bhs/schwarzschild.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace bhs;
using namespace bhs::schw;

struct Beam {
  double b;  // signed: >0 above the axis, <0 below
  Fate fate;
  double deflection;
  std::vector<Vec2> path;
};

static std::vector<Beam> make_beam(int n, double half_height) {
  std::vector<Beam> beams;
  for (int i = 0; i < n; ++i) {
    const double b = -half_height + (i + 0.5) * (2 * half_height / n);  // never b = 0
    Beam be;
    be.b = b;
    const auto r = trace(std::abs(b), 1e-3, 400, &be.path);
    be.fate = r.fate;
    be.deflection = r.deflection;
    if (b < 0) for (auto& p : be.path) p.y = -p.y;  // mirror: symmetry about the x axis
    beams.push_back(std::move(be));
  }
  return beams;
}

static void circle(double r, int segs = 180) {
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < segs; ++i) {
    const double a = 2 * M_PI * i / segs;
    glVertex2d(r * std::cos(a), r * std::sin(a));
  }
  glEnd();
}

int main() {
  if (!glfwInit()) return 1;
  GLFWwindow* w = glfwCreateWindow(1000, 800, "black-hole-sim v0.2", nullptr, nullptr);
  if (!w) { glfwTerminate(); return 1; }
  glfwMakeContextCurrent(w);
  glfwSwapInterval(1);

  const double view = 14.0;  // vertical half-width of the view
  int n = 41;
  auto beams = make_beam(n, view);
  bool show_straight = true, k_n = false, k_up = false, k_dn = false;
  auto edge = [&](int key, bool& was) {
    bool down = glfwGetKey(w, key) == GLFW_PRESS, hit = down && !was;
    was = down;
    return hit;
  };

  while (!glfwWindowShouldClose(w)) {
    glfwPollEvents();
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
    if (edge(GLFW_KEY_N, k_n)) show_straight = !show_straight;
    if (edge(GLFW_KEY_UP, k_up)) { n = std::min(n * 2 + 1, 321); beams = make_beam(n, view); }
    if (edge(GLFW_KEY_DOWN, k_dn)) { n = std::max((n - 1) / 2, 9); beams = make_beam(n, view); }

    int fw, fh;
    glfwGetFramebufferSize(w, &fw, &fh);
    glViewport(0, 0, fw, fh);
    glClearColor(0.02f, 0.02f, 0.05f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const double aspect = static_cast<double>(fw) / fh;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-view * aspect, view * aspect, -view, view, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (show_straight) {  // light without gravity: what we would see without curvature
      glColor4f(1, 1, 1, 0.12f);
      glLineWidth(1);
      glBegin(GL_LINES);
      for (auto& be : beams) { glVertex2d(-view * aspect, be.b); glVertex2d(view * aspect, be.b); }
      glEnd();
    }

    int captured = 0, escaped = 0, orbiting = 0;
    glLineWidth(1.6f);
    for (auto& be : beams) {
      switch (be.fate) {
        case Fate::Captured: ++captured; glColor4f(1.0f, 0.25f, 0.2f, 0.9f); break;
        case Fate::Orbiting: ++orbiting; glColor4f(1.0f, 0.8f, 0.2f, 0.9f); break;
        case Fate::Escaped: {
          ++escaped;
          const float t = static_cast<float>(std::min(be.deflection / M_PI, 1.0));  // 0..1
          glColor4f(0.3f + 0.7f * t, 0.6f + 0.4f * t, 1.0f, 0.95f);
        }
      }
      glBegin(GL_LINE_STRIP);
      for (auto& p : be.path) glVertex2d(p.x, p.y);
      glEnd();
    }

    // shadow (b = sqrt27), photon sphere (r=3) and horizon (r=2)
    glEnable(GL_LINE_STIPPLE);
    glColor4f(1, 1, 1, 0.35f);
    glLineStipple(1, 0x0F0F); circle(3.0);                // photon sphere
    glLineStipple(2, 0xAAAA); circle(critical_b());       // apparent shadow
    glDisable(GL_LINE_STIPPLE);
    glColor3f(0, 0, 0);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2d(0, 0);
    for (int i = 0; i <= 120; ++i) glVertex2d(2 * std::cos(2 * M_PI * i / 120), 2 * std::sin(2 * M_PI * i / 120));
    glEnd();
    glColor3f(1, 0.9f, 0.5f);
    glLineWidth(2);
    circle(2.0);

    glfwSwapBuffers(w);
    char title[256];
    std::snprintf(title, sizeof title,
                  "v0.2 | %d rays: %d captured, %d escape%s | shadow b=%.4f | N: straight %s | UP/DOWN: density",
                  n, captured, escaped, orbiting ? " (+orbiting)" : "", critical_b(), show_straight ? "on" : "off");
    glfwSetWindowTitle(w, title);
  }
  glfwDestroyWindow(w);
  glfwTerminate();
}
