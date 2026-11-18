#pragma once
// Structures shared between the Objective-C++ host (render_metal.mm) and the Metal shader (kerr_metal.metal).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using i32 = int;
using u32 = uint;
using f32 = float;
using f32_2 = float2;
#else
#include <cstdint>
using i32 = int32_t;
using u32 = uint32_t;
using f32 = float;
struct f32_2 { float x, y; };
#endif

struct MetalKP {
  i32 w, h, use_sky, sky_raw, nodisk, nodoppler, max_steps, prec;
  f32 phi_atol, l_min;
  f32 a, dist, theta, phi, tan_half, aspect, rin, rout, exposure, sky_exposure, tol, r_far;
  f32 tpeak, fnorm;
  i32 nt, nt_n, ldr, sample_k, limb;
  u32 seed;
  f32 nt_rin, nt_rout, bx, by, bz, clamp_lum, beaming;
  f32 turb_amp, turb_scale, disk_time, spot_r, spot_ph, spot_size, spot_amp, spot_om, fl_t, fl_w, fl_boost;
  i32 slow, rcheck, has_theta_lut, has_grmhd;
  i32 g_n1, g_n2, g_n3, el_model;
  f32 g_startx1, g_dx1, g_hslope, g_ms, g_palpha, g_pxt, g_pnorm, g_rho_unit, g_b_unit, g_rhigh, g_ce, g_rcap, g_gam;
  i32 model;
  f32 t_ell, t_utin, t_hmax, t_gamma, t_rin, t_rmax, ne0, pbeta, ti_te, nu_obs, mcm, tb_lo, tb_hi, th_e_max, t_step, g_step, the_min;
  i32 emis, vol, v_sub;
  f32 v_hr, v_flare, v_tau, edge_soft, edge_noise;
  f32 fwd[3], right[3], up[3];
  i32 has_stats, has_alb;
  // Double-float pairs (hi, lo) to keep input precision in mixed/double mode without hardware FP64
  f32_2 a_df, dist_df, theta_df, theta_sin_df, theta_cos_df, rin_df, rout_df, r_far_df, tol_df, phi_atol_df, l_min_df;
  f32_2 fwd_df[3], right_df[3], up_df[3], beta_df[3];
};

struct MetalTrace {
  i32 fate, disk_hit, steps, _pad;
  f32 mu_end, phi_end, disk_r, disk_phi, L, Q;
};

struct MetalTorusPoint {
  f32 rho, thE, B, sinB, j, alpha, g, e_emit;
};

struct MetalPolish {
  f32 glare, grain, vignette, chroma;
  i32 frame;
  u32 seed;
  i32 has_g1, gw, gh;
  f32 bloom;
  i32 levels, ldr, tonemap, w, h;
  i32 lw[6], lh[6];
};
