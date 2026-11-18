#pragma once
// v4.0 interfaces of the render kernel: Observer / Metric / DiskModel as small static structs, composed by kernel<PREC, MODEL, METRIC> (render.cu).
// This header is included by render.cu AFTER the file-local helpers and types it uses (KP, camera_ray, observer_to_zamo, wh_trace, VolAccT, trace_dispatch,
// nt_temp, disk_temp, redshift, lut_lookup, vnoise, ...): it is not self-contained on purpose (they are device functions of one translation unit).
// Metric / Observer / DiskModel, each with ONE job; the kernel only composes them (all static, chosen at compile time).
//  Observer : pixel -> ray direction in the local orthonormal frame of the metric (+ Doppler factor of the observer's motion)
//  Metric   : ray -> where it ends: a disk hit (Kerr family, bk::trace) or a point of a sky (SkyHit; the wormhole has two skies, sky_id says which)
//  DiskModel: how the gas along the ray becomes colour (thin surface with soft edges, or volume transport: layer / torus / GRMHD)
constexpr int kMetricKerrFamily = 0;  // Kerr, Kerr-Newman, Schwarzschild-de Sitter (bk::trace, chosen per ray by the Carter tags)
constexpr int kMetricWormhole = 3;    // Ellis / Interstellar wormhole (planar tracer, two skies)

struct SkyHit { float u, v; int sky_id; };  // sky texture coordinates and which sky (0: the main one, 1: the 2nd sky behind the throat)

struct KerrZamoObserver {  // camera basis -> aberrated direction n in the ZAMO frame (e_r, e_theta, e_phi)
  __device__ static void ray(const KP& p, double x, double y, double3& n, double& dop) { observer_to_zamo(camera_ray(p, x, y), p.bx, p.by, p.bz, n, dop); }
};

struct WormholeObserver {  // static or radially moving observer at l_c: VIEWING direction v in the frame (e_l, e_theta, e_phi)
  __device__ static void ray(const KP& p, double x, double y, double3& v, double& dop) {
    // l_c < 0 (the camera went through the throat): its radial axis points away from the throat (along -e_l), so the radial velocity (given along +e_l) changes sign in the camera frame,
    // and the throat lies in the +l direction, so the radial component of the viewing direction flips; the transverse ones do not (the metric is symmetric under l -> -l)
    double3 nz;
    observer_to_zamo(camera_ray(p, x, y), p.wh_lc < 0.0 ? -p.bx : p.bx, p.by, p.bz, nz, dop);
    v = make_double3(p.wh_lc < 0.0 ? nz.x : -nz.x, -nz.y, -nz.z);
  }
};

struct WormholeMetric {  // the ray ends on a sky; the side where it ends (sign of l) is physical, so the 2nd sky is behind the camera once it has crossed the throat
  __device__ static SkyHit sky(const KP& p, const double3& vdir) {
    const double vl = vdir.x, vt = vdir.y, vp = vdir.z, vperp = sqrt(vt * vt + vp * vp);
    float psi; int sd;
    wh_trace(p, float(fmax(-1.0, fmin(1.0, vl))), float(fmin(1.0, vperp)), sd, psi);
    // static camera at l_c, angular position (theta, phi) = (p.theta, p.phi)
    const double st = sin(p.theta), ct = cos(p.theta), sp = sin(p.phi), cp = cos(p.phi);
    const double3 P0 = make_double3(st * cp, st * sp, ct), th_hat = make_double3(ct * cp, ct * sp, -st), ph_hat = make_double3(-sp, cp, 0.0);
    const double tx = vperp > 1e-12 ? vt / vperp : 1.0, ty = vperp > 1e-12 ? vp / vperp : 0.0;
    const double cps = cos(double(psi)), sps = sin(double(psi));
    const double3 D = make_double3(cps * P0.x + sps * (tx * th_hat.x + ty * ph_hat.x), cps * P0.y + sps * (tx * th_hat.y + ty * ph_hat.y), cps * P0.z + sps * (tx * th_hat.z + ty * ph_hat.z));
    double uu = 0.5 - atan2(D.y, D.x) / (2.0 * M_PI);
    uu -= floor(uu);
    return SkyHit{float(uu), float(acos(fmax(-1.0, fmin(1.0, D.z))) / M_PI), sd < 0 ? 1 : 0};
  }
};

// thin disk: surface at the equatorial plane, soft / irregular edge (layers behind the fringe), Novikov-Thorne or Newtonian temperature, filaments, arms, rings, limb darkening
template <int PREC>
struct ThinDisk {
  __device__ static void shade(const KP& p, const double3& nz, double dop, double rout, double atol, bk::Trace& tr, double3& base, double& trans, double3& col, int& steps) {
    // v0.9a2: soft/irregular edge. The disk becomes semi-transparent in the fringe: the ray continues behind it
    // (skip = layers already shaded) and the layers add up with transmittance Tr. No fringe = the v0.8 path.
    const bool soft = p.edge_soft > 0.0 || p.edge_noise > 0.0;
    const double rtrace = (soft && rout > 0.0) ? rout * (1.0 + p.edge_noise) * (1.0 + p.edge_soft) : rout;
    trace_dispatch<PREC>(p, nz.x, nz.y, nz.z, rtrace, atol, tr);
    steps = tr.steps;
  for (int layer = 0; tr.disk_hit; ++layer) {
    const double r = tr.disk_r;
    double wgt = 1.0;
    if (soft) {
      const double phe = tr.disk_phi + p.phi;
      const double ne = 0.6 * vnoise(3 * cos(phe), 3 * sin(phe), 1.7) + 0.4 * vnoise(8 * cos(phe), 8 * sin(phe), 5.3);
      const double re = rout * (1.0 + p.edge_noise * (2.0 * ne - 1.0));
      wgt = p.edge_soft > 0.0 ? 1.0 - ss(re * (1.0 - p.edge_soft), re * (1.0 + p.edge_soft), r) : (r < re ? 1.0 : 0.0);
    }
    double g = p.noshift ? 1.0 : (p.lam > 0.0 || (p.q2 > 0.0 && p.a * p.a + p.q2 >= 1.0)) ? 1.0 : p.nodoppler ? redshift(p.a, r, 0.0, p.dist, p.theta, p.q2, p.zamo) : redshift(p.a, r, tr.L, p.dist, p.theta, p.q2, p.zamo) * dop;  // SdS: no frequency shift; KN: ntk (Kerr-Newman)
    if (p.beaming != 1.0) g = pow(g, p.beaming);  // --beaming: aesthetic (not physical) choice, see the notes
    double k;
    double3 c;
    if (p.lp_n > 0) {  // E4 X-ray lamppost: bolometric reflected intensity I_obs = g^4 eps_E(r), false colour, no surface pattern
      k = 1.0;
      if (p.lp_plunge && r < p.lp_risco && !p.noshift && !p.nodoppler) g = redshift_plunge(p, r, tr.L, tr.Q, tr.vr) * dop;  // gas inside the ISCO falls in
      c = xray_rgb((2.0 + p.lp_gamma) * log10(fmax(g, 1e-30)) + lp_log10e(p, r) + p.lp_lscale, p.lp_lo, p.lp_hi);
    } else {
    const double Tprof = p.uniform_disk ? 1.0 : p.nt ? nt_temp(p, r) : disk_temp(r, p.rin, p.fnorm);
    const double T_obs = g * p.tpeak * Tprof;
    const double lr = log(r);
    const double sfac = (p.arm_amp > 0.0 || p.ring_amp > 0.0) ? double(disk_struct_f(p, float(r), float(tr.disk_phi + p.phi), float(p.disk_time))) : 1.0;
    if (p.tex_amp > 0.0) {  // v2.4: filaments: fbm of tex_oct octaves, radial frequency tex_fr >> azimuthal tex_fphi (stretched in azimuth); phase sheared by Omega(r) t
      const double psi = tr.disk_phi + p.phi - 2.2 * log(r / p.rin) - p.disk_time / (r * sqrt(r) + p.a);
      double s = 0.0, ws = 0.0, w = 1.0, f = 1.0;
      for (int o = 0; o < p.tex_oct; ++o) {
        s += w * (2.0 * vnoise(f * p.tex_fphi * cos(psi) + 17.3 * o, f * p.tex_fphi * sin(psi) + 5.7 * o, f * p.tex_fr * lr + 11.1 * o) - 1.0);
        ws += w; w *= 0.62; f *= 2.0;
      }
      k = p.exposure * exp(p.tex_amp * s / ws);
    } else {
      // the surface pattern turns with the Keplerian Omega(r) = 1/(r^1.5 + a) (differential rotation, as the filaments/turbulence do): without it a smooth axisymmetric disk
      // shows no motion at all, so the disk-time animation only worked for turbulence / hot spot (a = 0 time term is exactly zero: disk_time = 0 reproduces the old image)
      const double psi = tr.disk_phi + p.phi - 2.2 * log(r / p.rin) - p.disk_time / (r * sqrt(r) + p.a);
      const double n = 0.55 * vnoise(3 * cos(psi), 3 * sin(psi), 9 * lr) + 0.45 * vnoise(9 * cos(psi), 9 * sin(psi), 28 * lr);
      k = p.exposure * (0.55 + (1.15 - 0.55) * n);
    }
    if (p.smooth_disk) k = p.exposure * 0.85;  // v4.0: no surface pattern (0.85 = mean of the pattern)
    k *= sfac;
    if (p.limb) k *= limb_eddington_d(emitter_mu_d(p.a, r, tr.L, tr.Q));
    c = lut_lookup(p.lut, T_obs);
    }
    if (!soft) { col = make_double3(k * c.x, k * c.y, k * c.z); break; }
    base.x += trans * wgt * k * c.x; base.y += trans * wgt * k * c.y; base.z += trans * wgt * k * c.z;
    trans *= 1.0 - wgt;
    if (trans < 1e-3 || layer >= 2) { trans = 0.0; break; }
    trace_dispatch<PREC>(p, nz.x, nz.y, nz.z, rtrace, atol, tr, layer + 1);  // continues behind the gas
    steps += tr.steps;
  }
    if (soft) col = base;
  }
};

// volume disks (the host only launches MODEL >= 0 with p.vol && rout > 0): 0 = Gaussian layer, 1 = Fishbone-Moncrief torus, 2 = GRMHD snapshot; radiative transport along the ray
template <int PREC, int MODEL>
struct VolumeDisk {
  __device__ static void shade(const KP& p, const double3& nz, double dop, double rout, double atol, bk::Trace& tr, double3& base, double& trans, double3& col, int& steps) {
    VolAccT<MODEL> vol(p, dop);
    trace_dispatch_vol<PREC>(p, nz.x, nz.y, nz.z, rout, atol, tr, vol);
    base = make_double3(vol.I.x, vol.I.y, vol.I.z); trans = vol.opaque() ? 0.0 : double(vol.trans); col = base;
    if (MODEL >= 1) { base = MODEL == 4 || MODEL == 5 ? sph_rgb(p, vol.Irad, MODEL == 5) : radio_rgb(p, vol.Irad); col = base; trans = 0.0; }  // radio / spherical: false colour, no sky (like the EHT images)  // radio mode: false colour, no sky (like the EHT images)
    steps = tr.steps;
  }
};

