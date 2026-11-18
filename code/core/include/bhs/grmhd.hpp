#pragma once
// Reader of GRMHD dumps (v2.2f): FMKS map (native coordinates X1, X2, X3 <-> Kerr-Schild r, theta, phi), inverse KS metric and loader of the .bgm format
// (tools/convert_grmhd.py). pyharm conventions (commit fda4277, see notes/references.md):
//   r = exp(X1), phi = X3, theta = theta_g + exp(mks_smooth (startx1 - X1)) (theta_j - theta_g),
//   theta_g = pi X2 + ((1 - hslope)/2) sin(2 pi X2), y = 2 X2 - 1, theta_j = poly_norm y (1 + (y/poly_xt)^poly_alpha/(poly_alpha + 1)) + pi/2,
//   poly_norm = (pi/2) / (1 + 1/(poly_alpha + 1)/poly_xt^poly_alpha);  cell centre X = startx + (i + 1/2) dx.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace bhs::grmhd {

struct Fmks {
  double hslope = 0.3, mks_smooth = 0.5, poly_alpha = 14.0, poly_xt = 0.82, startx1 = 0.0;

  double poly_norm() const { return 0.5 * M_PI / (1.0 + 1.0 / (poly_alpha + 1.0) / std::pow(poly_xt, poly_alpha)); }
  double theta_g(double X2) const { return M_PI * X2 + 0.5 * (1.0 - hslope) * std::sin(2.0 * M_PI * X2); }
  double theta_j(double X2) const {
    const double y = 2.0 * X2 - 1.0;
    return poly_norm() * y * (1.0 + std::pow(y / poly_xt, poly_alpha) / (poly_alpha + 1.0)) + 0.5 * M_PI;
  }
  double weight(double X1) const { return std::exp(mks_smooth * (startx1 - X1)); }
  double theta(double X1, double X2) const { const double tg = theta_g(X2); return tg + weight(X1) * (theta_j(X2) - tg); }
  double dtheta_dX1(double X1, double X2) const { return -mks_smooth * weight(X1) * (theta_j(X2) - theta_g(X2)); }
  double dtheta_dX2(double X1, double X2) const {
    const double y = 2.0 * X2 - 1.0, dg = M_PI + (1.0 - hslope) * M_PI * std::cos(2.0 * M_PI * X2);
    const double dj = 2.0 * poly_norm() * (1.0 + std::pow(y / poly_xt, poly_alpha));
    return dg + weight(X1) * (dj - dg);
  }
  // X2 in [0, 1] such that theta(X1, X2) = th (theta is strictly increasing in X2): Newton with bisection as a safety net
  bool inv_theta(double X1, double th, double& X2) const {
    if (!(th >= 0.0) || !(th <= M_PI)) return false;
    double lo = 0.0, hi = 1.0, x = th / M_PI;
    for (int it = 0; it < 60; ++it) {
      const double f = theta(X1, x) - th;
      if (std::abs(f) < 1e-15) break;
      (f > 0 ? hi : lo) = x;
      double xn = x - f / dtheta_dX2(X1, x);
      if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
      x = xn;
    }
    X2 = x;
    return true;
  }
};

// inverse of the Kerr-Schild metric (t, r, theta, phi), M = 1: g^tt = -(1 + 2r/S), g^tr = 2r/S, g^rr = D/S, g^rphi = a/S, g^thth = 1/S, g^phiphi = 1/(S sin^2 theta)
inline void ks_inverse(double a, double r, double th, double g[4][4]) {
  const double s2 = std::sin(th) * std::sin(th), S = r * r + a * a * std::cos(th) * std::cos(th), D = r * r - 2.0 * r + a * a;
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) g[i][j] = 0.0;
  g[0][0] = -(1.0 + 2.0 * r / S);
  g[0][1] = g[1][0] = 2.0 * r / S;
  g[1][1] = D / S;
  g[1][3] = g[3][1] = a / S;
  g[2][2] = 1.0 / S;
  g[3][3] = 1.0 / (S * s2);
}

constexpr int kNF = 12;  // rho, theta_i, |b|, beta, u^mu (4), bhat^mu (4), em Kerr-Schild

struct Header {
  uint32_t n1 = 0, n2 = 0, n3 = 0, nf = 0;
  double startx1 = 0, dx1 = 0, dx2 = 0, dx3 = 0, a = 0, hslope = 0.3, mks_smooth = 0.5, poly_alpha = 14.0, poly_xt = 0.82, gam = 4.0 / 3.0, t = 0;
  Fmks fmks() const { Fmks m; m.hslope = hslope; m.mks_smooth = mks_smooth; m.poly_alpha = poly_alpha; m.poly_xt = poly_xt; m.startx1 = startx1; return m; }
};

struct Field {
  Header h;
  std::vector<float> data;  // [n1][n2][n3][nf]
  float at(size_t i, size_t j, size_t k, size_t f) const { return data[((i * h.n2 + j) * h.n3 + k) * h.nf + f]; }
};

inline bool load_bgm(const std::string& path, Field& f, std::string& err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) { err = "could not open " + path; return false; }
  char hdr[256];
  if (!in.read(hdr, 256) || std::memcmp(hdr, "BGM1", 4) != 0) { err = "invalid .bgm header (magic BGM1)"; return false; }
  uint32_t d[5]; std::memcpy(d, hdr + 4, sizeof d);
  double v[16]; std::memcpy(v, hdr + 24, sizeof v);
  Header& h = f.h;
  h.n1 = d[0]; h.n2 = d[1]; h.n3 = d[2]; h.nf = d[3];
  h.startx1 = v[0]; h.dx1 = v[1]; h.dx2 = v[2]; h.dx3 = v[3]; h.a = v[4]; h.hslope = v[5]; h.mks_smooth = v[6]; h.poly_alpha = v[7]; h.poly_xt = v[8]; h.gam = v[9]; h.t = v[10];
  if (h.nf != uint32_t(kNF) || h.n1 == 0 || h.n2 == 0 || h.n3 == 0) { err = ".bgm with invalid dimensions"; return false; }
  f.data.resize(size_t(h.n1) * h.n2 * h.n3 * h.nf);
  if (!in.read(reinterpret_cast<char*>(f.data.data()), std::streamsize(f.data.size() * sizeof(float)))) { err = "truncated .bgm"; return false; }
  return true;
}

}  // namespace bhs::grmhd
