#pragma once
// Host interface of the CUDA Kerr renderer (implemented in render.cu). Plain types only: main.cpp
// is compiled with the normal C++ compiler.
#include <cstdint>
#include <string>
#include <vector>

namespace bhs::kcuda {

enum class Precision { Double = 0, Float = 1, Mixed = 2 };
enum class Tonemap { Reinhard = 0, Aces = 1 };

struct Params {
  int w = 1920, h = 1080;
  double a = 0.9, dist = 40.0, theta_deg = 80.0, phi_deg = 0.0, fov_deg = 40.0;
  double rin = -1.0 /* -1: ISCO */, rout = 16.0, tpeak = 4200.0, exposure = 3.0, sky_exposure = 2.0;
  double tol = 1e-8, r_far = 1e6;
  int integ = 1;         // O3: integrator of the pure-double thin-disk/sky path: 0 = Dormand-Prince 5(4), 1 = DOP853 (order 8, default; run at tol/10, calibrated so
                         // that every error quantile is no worse than DP45 at the same --tol: notes/physics/33-dop853.md)
  double r_tail = 1000;  // O1: beyond this radius an outgoing ray finishes with the analytic escape tail (Kerr/KN); 0 = off, integrate to r_far.
                         // The kernel raises it to 10x the outermost emitter (disk rout, torus 4 rmax, RIAF rout, GRMHD rcap, spherical/cone rout) so no emission is skipped
  bool nodisk = false, nodoppler = false;
  bool noshift = false, uniform_disk = false, smooth_disk = false;  // v4.0 (Interstellar preset): no frequency shift at all (g = 1); uniform disk temperature = tpeak; no surface pattern
  bool use_sky = true;
  bool sky_raw = false;  // no mipmaps/anisotropy: bilinear at level 0 only (same as the CPU reference, to validate the geometry)
  int max_steps = 200000;
  double phi_atol = 0;   // absolute tolerance of phi in the step control (rad); 0 = same as atol (v0.6b)
  double l_min = -1;     // minimum |L| (regularises the plane that contains the axis): -1 = automatic (1e-4, as kerr::Options::l_min); 0 = off
  Precision prec = Precision::Double;
  int block = 128;  // threads per block (multiple of 32)
  // v0.7d
  bool nt = true;                  // Novikov-Thorne temperature profile (LUT); false = Newtonian (v0.4-v0.6)
  double fwd[3] = {-1, 0, 0}, right[3] = {0, 0, 1}, up[3] = {0, -1, 0};  // camera basis in (e_r, e_theta, e_phi); default = v0.6b's
  double beta[3] = {0, 0, 0};      // observer velocity in the ZAMO frame (e_r, e_theta, e_phi), in c
  int samples = 1;                 // sub-pixel samples (offline mode: render())
  uint32_t seed = 1;               // jitter seed (determinism)
  bool accum_ldr = false;          // true: Reinhard PER SAMPLE before accumulating (= CPU SSAA); false: HDR average
  double clamp_lum = 0;            // per-sample luminance clamp (fireflies); 0 = off
  Tonemap tonemap = Tonemap::Reinhard;
  double bloom = 0;                // bloom strength (0 = off)
  // v0.8d: cinematic polish (only with --cinematic or explicit flags; 0 = off = v0.7d byte-identical)
  double glare = 0;                // fraction of the energy above the threshold redistributed into 6-point streaks
  // v2.6 aesthetic disk structure: arms (m, pitch in degrees, amplitude, pattern radius), rings with gaps, variable thickness (layer only)
  double arm_amp = 0.0, arm_pitch_deg = 20.0, arm_rpat = 14.0; int arm_m = 2;
  double ring_amp = 0.0, ring_freq = 8.0, ring_gap = 0.0, hv_amp = 0.0, hv_scale = 3.0;
  // v2.1d wormhole (Ellis or the 3-parameter Interstellar one): camera at rest at l = wh_lc; side l < 0 = 2nd sky (load_sky2); traced in float
  bool sky2_off = false;  // wormhole: force the 1st sky on both sides (--no-sky2)
  bool wormhole = false, wh_ellis = false; double wh_rho = 1.0, wh_a = 0.005, wh_M = 0.03498, wh_lc = 8.0, wh_lmax = 100.0;
  double lambda = 0.0;             // v2.1c: cosmological constant (Schwarzschild-de Sitter, a = 0 and Q = 0, 0 < Lambda < 1/9 in units M = 1)
  double charge = 0.0;             // v2.1a: black-hole charge Q (Kerr-Newman, Q^2 + a^2 <= 1 has a horizon; > 1 naked singularity); geometry only + thin disk without frequency shift
  bool hdr = false; double hdr_nits = 120.0, hdr_peak = 1000.0;  // v2.5: HDR10 output (PQ, BT.2020): 1 unit of linear light = hdr_nits cd/m2, soft shoulder up to hdr_peak
  double glare_cap = 1.5;          // sources with luminance above cap leave the glare (smooth up to 2 cap; 0 = no limit): stars up to ~1; the photon ring and the disk core, much brighter, no longer trigger it (intermittent blue streaks inside the shadow)
  double glare_thr = 0.5;          // glare threshold: excess of the pixel's HDR luminance over the coarse level (1/8) of the pyramid
  double glare_len = 16.0;         // decay length (px at half resolution)
  double grain = 0;                // film-grain amplitude (fraction of the scale; after tone mapping)
  double vignette = 0;             // 0..1
  double chroma = 0;               // radial chromatic aberration (fraction of the radius at the corner)
  int frame_index = 0;
  // v0.8c: OptiX denoiser (optional; between the HDR average and the bloom/tone mapping)
  bool denoise = false;            // --denoise optix
  bool denoise_albedo = false;     // albedo guide (normalised surface colour) for the denoiser
  int sample_offset = 0;           // index of the 1st sample (>0: all jittered; for denoiser tests)             // varies the grain between frames (reproducible)
  double beaming = 1.0;            // g -> g^beaming in the disk intensity: 1 = physical, < 1 = AESTHETIC (cinematic)
  // v0.9a: disk spectrum and colour (default = v0.8 byte-identical)
  double fcol = 1.0;               // hardening factor: I_nu = f^-4 B_nu(f T) (bolometric unchanged); physical, ~1.5-2 in stellar disks
  bool limb = false;               // Eddington limb darkening I(mu) ~ 1 + 2.06 mu (mu in the gas frame; flux conserved)
  // v0.9a2: outer edge (default = v0.8 hard cut, byte-identical)
  double edge_soft = 0.0;          // relative half-width of the emission decay (smoothstep in r_e(1 +- s)); 0 = hard cut
  double edge_noise = 0.0;         // edge irregularity: r_e = rout (1 + amp (2n - 1)), n = azimuth noise
  // v0.9b: thick disk (hr = 0: the thin disk of the earlier versions, byte-identical)
  double hr = 0.0;                 // H/r at r = 10 (> 0 turns on the volume: Gaussian layer in z, radiative transport along the ray)
  double flare = 0.0;              // H/r = hr (r/10)^flare
  double tau = 10.0;               // total vertical optical depth of the layer
  int vol_sub = 4;                 // transport sub-samples per accepted step (Hermite)
  // v0.9c: 3D turbulence, hot spot and flare (only in volume mode, --hr > 0); everything at 0 = v0.9b (equal to ~1e-6; only the thin disk --hr 0 is byte-identical)
  double turb = 0.0;               // lognormal amplitude of the turbulence (density * exp(a n - a^2 var / 2)); 0 = off
  // v2.4 thin-disk texture (Interstellar): multiscale filaments stretched in azimuth, sheared by Omega(r) with the disk time; 0 = original texture
  double tex_amp = 0.0, tex_fr = 60.0, tex_fphi = 6.0; int tex_oct = 5;
  double turb_scale = 2.0;         // cell size in units of the scale height H(r)
  double spot_r = 10.0, spot_phi_deg = 0.0, spot_size = 1.5, spot_amp = 0.0;  // hot spot: orbits at r with Omega(r); amp > 0 turns it on
  double flare_t = 0.0, flare_w = 20.0, flare_boost = 0.0;                    // flare: Gaussian pulse in time on top of the spot amplitude
  bool slowlight = false;          // v0.9c2: propagation delay (slow light): the disk is seen at the emission time t_obs - flight time, not at t_obs
  double disk_time = 0.0;          // disk time (M): the turbulence and the spot rotate with the Kerr Omega(r); instantaneous (no propagation delay)
  // v2.2b: radio mode (Fishbone-Moncrief torus + thermal synchrotron), only with --hr > 0 turned on as a volume; brightness-temperature image
  bool theta_lut = true;           // v2.2h: GRMHD: theta inverse by LUT + 2 Newton steps (false = exact 40-iteration bisection, reference)
  int model = 0;                   // 0 = Gaussian layer (v0.9b/c), 1 = torus with synchrotron
  double torus_rin = 7.0, torus_rmax = 12.0, torus_gamma = 4.0 / 3.0;
  // v4.0 parametric RIAF (--disk-model riaf): n_e = ne0 (r/6)^-pn exp(-z^2 / 2H^2), H = hr r, Theta_e = theta_e_max (r/6)^-pt, B from the plasma beta, gas on circular orbits at xi Omega_K;
  // r from the ISCO (rin) to riaf_rout. ALL the exponents and amplitudes are free model parameters (there is no verified source for default values in this project: model-dependent)
  double riaf_pn = 1.1, riaf_pt = 0.84, riaf_hr = 0.3, riaf_xi = 0.5, riaf_rout = 40.0;
  // E1 spherical inflow (--disk-model spherical, model 4): Narayan, Johnson & Gammie 2019 (ApJL 885 L33, arXiv:1910.02957), optically thin and BOLOMETRIC:
  // I_obs = int g^4 j ds (ds = proper length in the gas frame), j = gas-frame emission coefficient. Flow: free fall from infinity (u_t = -1, u_phi = u_theta = 0;
  // Kerr generalisation as in Papoutsis et al. 2023) or gas at rest (static observers, only outside the ergoregion). Emissivity: the paper's exact
  // P = 1/[(1-2/r)^1/2 4 pi r^4], j = P/4pi (a = 0 only), or j = r^-alpha/(16 pi^2). Kerr only (the volume tracer is Kerr-only). Display: log10 I in [sph_lo, sph_hi].
  // E2 jet walls, dual-cone model (--disk-model cones, model 5): Papoutsis, Bauboeck, Chang & Gammie 2023 (ApJ, arXiv:2212.06281). Optically thin,
  // frequency-independent j = r^-alpha [exp(-((theta - theta_c)/w)^2) + exp(-((theta - (pi - theta_c))/w)^2)] (alpha = 5, w = 1/128 rad in the paper),
  // gas in free fall (u_t = -1, u_theta = u_phi = 0), I = int g^3 j ds (J = j/nu^2 invariant). Kerr only. Display: log10 I in [cone_lo, cone_hi].
  double cone_theta_deg = 30.0, cone_w = 1.0 / 128.0, cone_alpha = 5.0, cone_rout = 60.0, cone_lo = -6.0, cone_hi = -1.0;
  // E4 X-ray lamppost (--disk-model lamppost, thin disk, Kerr only): a static point source on the axis at BL radius lp_h illuminates the disk; the disk re-emits the reflected light
  // isotropically, bolometric: I_obs = g^4 eps_E(r) (albedo / pi folded into the normalisation, max of eps_E = 1). eps_E(r) from the analytic oracle (core/include/bhs/lamppost.hpp).
  // lp_h > 0 turns it on. Display: false colour of log10 I_obs in [lp_lo, lp_hi]. Notes: notes/physics/39-lamppost-corona.md, 40-lamppost-xray-view.md.
  double lp_h = 0.0, lp_lo = -4.0, lp_hi = 1.0;
  double lp_gamma = 2.0;      // photon index of the corona spectrum N(E) ~ E^-Gamma: observed band intensity ~ g^(2+Gamma) eps(r) with eps weighted by g_sd^(Gamma-1); 2 = the bolometric view
  double lp_refl = 0.0;       // reflection fraction R_f = N_disk / N_inf asked for: the disk brightness scales by R_f / R_geometric (0 = the geometric value; Dauser et al. 2014)
  bool lp_plunge = false;     // light the plunging region too: the disk extends to the horizon, gas inside the ISCO falls on the geodesics with the ISCO's E and L (notes/physics/40)
  bool lp_async = false;      // interactive: a new table is built in the background and the previous one is drawn meanwhile (offline: always synchronous)
  int lp_samples = 2000;      // photons traced to build the table (~0.8 ms each, once per (a, h, r_in))
  int sph_flow = 0;           // 0 = free fall from infinity, 1 = gas at rest
  bool sph_narayan = false;   // the paper's exact emissivity (a = 0, r > 2)
  double sph_alpha = 4.0, sph_rin = 0.0 /* 0 = horizon (2 with the Narayan profile) */, sph_rout = 200.0, sph_lo = -7.0, sph_hi = -3.0;
  double ne0 = 1e5;                // electron density at the pressure maximum, cm^-3
  double plasma_beta = 10.0;       // plasma beta = P_gas / P_mag (toroidal field)
  double torus_step = 1.0;         // maximum path (M) per geodesic step inside the torus gas
  double ti_te = 10.0;             // T_i / T_e (used if theta_e_max = 0)
  double theta_e_max = 0.0;        // > 0: Theta_e = theta_e_max (h-1)/(h_max-1), i.e. T_e proportional to the torus T_i, with this maximum (the analytic torus is cold:
                                   // Theta_i,max ~ 0.006; simulations heat the gas, here it is a model parameter); 0 = Theta_e = Theta_i (m_p/m_e)/ti_te
  double nu_ghz = 230.0;           // observing frequency
  double mbh = 6.5e9;              // black-hole mass in solar masses (length scale GM/c^2)
  double tb_lo = 8.0, tb_hi = 11.0;  // log10(T_b/K) range of the false colour
  // v2.2f: GRMHD mode (model = 2): a .bgm snapshot from tools/convert_grmhd.py sampled by 3D textures; see notes/physics/22f-grmhd-snapshot.md
  double munit = 1e26;             // M_unit (g): rho_unit = M_unit / L^3, L = G M_BH / c^2 (--mbh); n_e = rho rho_unit/(m_p + m_e), B_G = |b| c sqrt(4 pi rho_unit)
  int emission = 0;                // v2.2b3: 0 Dexter 2016 (ultra-relativistic, Theta_e > 3), 1 Leung et al. 2011 (K_2(1/Theta_e), Theta_e >~ 0.5)
  double theta_e_min = 0.3;        // no emission below this Theta_e
  int electrons = 0;               // 0: Theta_e = Theta (m_p/m_e)/(1 + Ti/Te) (pyharm); 1: R_high/R_low, EHT convention (R = (R_high beta^2 + 1)/(1 + beta^2), see eht_theta_e_coef); 2: Theta_e = th_e_ratio Theta; 3: R_high with the old single-temperature split Theta (m_p/m_e)/(1 + R)
  double rhigh = 20.0, th_e_ratio = 1.0;
  double grmhd_sigma_cut = 0.0;   // E3: no emission where sigma = b^2/rho > cut (0 = off). EHT M87 Papers V/VIII use 1 (VIII also tested 10); Chael et al. 2019 used 25.
                                   // In the funnel rho is the numerical floor, so its emission is not physical; the UI sets 1 for GRMHD
  double grmhd_rcap = 60.0;        // v2.2f-d: radius beyond which the GRMHD ray marching skips (empty space; the emission ends at r ~ 30); measured in notes/physics/22f-b
  double grmhd_step = 0.05;        // GRMHD mode: maximum step along the ray = grmhd_step * r (M), the FMKS radial cell has r dX1 = 0.022 r; <= 2 M
  bool debug_null = false;         // test: check the photon null condition in KS in GRMHD mode
  bool expose_physical = false;    // fixed exposure (reference: black body f = 1 at tpeak) instead of renormalised to each LUT
};

struct Stats {
  double steps_mean = 0;
  int steps_max = 0;
  int sanitized = 0;     // pixels whose value was non-finite and were set to black (v2.7; must be 0)
  double warp_eff = 0;  // mean(steps)/mean(max steps per warp): 1 = no divergence
  int stuck_rays = 0, stuck_pixel = -1;  // rays that hit max_steps in the last frame and the (py * w + px) of one of them (v4.0 diagnostic)
};

struct Timing {
  float trace_ms = 0;    // last sample
  float resolve_ms = 0;  // average + tone mapping (without the bloom)
  float bloom_ms = 0;    // bloom pyramid
  float denoise_ms = 0;  // optixDenoiserInvoke (0 if off)
};

bool init(const std::string& sky_path, std::string& err);  // empty sky_path = no sky (black)
bool has_sky();
bool load_sky2(const std::string& path, std::string& err);  // 2nd sky (see Params::wormhole)
void ntkn_batch(const Params& p, const std::vector<double>& rL, std::vector<double>& out);  // test: per (r, L) -> (normalised T_NT, g = redshift) with the disk/observer of Params (Kerr or Kerr-Newman)
void wh_batch(const Params& p, const std::vector<double>& alphas, std::vector<double>& psi, std::vector<int>& side);  // test: wh_trace in float per alpha
bool denoise_available();                       // compiled with OptiX?
bool denoise_enable(const Params& p, std::string& err);  // creates the denoiser (once); false + err if unavailable
void download_hdr(std::vector<float>& rgba);    // w*h*4: HDR average (after the denoiser, if active), before bloom/tone mapping
const char* denoise_version();
bool denoise_image(const std::vector<float>& rgba_in, int w, int h, std::vector<float>& rgba_out, std::string& err);  // test: direct denoise of a host RGBA32F image
void set_size(int w, int h);
// Renders into the internal RGBA8 buffer; returns GPU ms (cudaEvent). Optional stats (costs a copy).
float render(const Params& p, Stats* stats = nullptr);  // = accum_reset + p.samples x accum_sample + resolve
// progressive accumulation (interactive mode): restarts with accum_reset() when something changes
void accum_reset();
int accum_count();
int render_w();  // logical size of the last render (may be < the window with --move-scale)
int render_h();
float accum_sample(const Params& p, int k, Stats* stats = nullptr);  // adds sample k (k = 0: pixel centre)
float resolve(const Params& p);                                       // media -> bloom -> tone mapping -> RGBA8
Timing last_timing();
void hud_draw(const std::string& text, int scale, bool top_right = false);  // draws text (lines separated by \n; lines starting with ! in red) on the final frame
void download_rgb16(std::vector<uint16_t>& rgb);  // w*h*3, 16 bit per channel (v1.0), the same final frame before quantising to 8 bit
void download_rgb(std::vector<uint8_t>& rgb);  // w*h*3, row 0 at the top

// CUDA-OpenGL interop: registers a GL texture (RGBA8) and copies the buffer into it.
bool gl_register(unsigned tex, int w, int h, std::string& err);
void gl_copy();
void gl_unregister();

// Pure tracing of a batch of rays (for the selftest). rays = (nr, nth, nph) of the photon.
struct RayOut {
  int fate, disk_hit, steps;
  double mu_end, phi_end, disk_r, disk_phi, L, Q;
  int rejects = 0;
};
// tests: NT temperature (GPU LUT) and observer boost
void camera_rays(const Params& p, const std::vector<double>& xy, std::vector<double>& out);  // n' per pixel (x, y) with p's basis
bool accum_finite();  // are all the accumulated values finite?
void nt_lookup(const Params& p, const std::vector<double>& r, std::vector<double>& out);
void boost_batch(const std::vector<double>& np, const double beta[3], std::vector<double>& out);  // out = (n.xyz, dop) per ray
void trace_batch(const Params& p, const std::vector<double>& rays, std::vector<RayOut>& out, int skip = 0);
void vol_flight_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out);  // test (slow light): per ray, the accumulated flight time until the transmittance drops to 1/2 (-1 if never)
struct TorusPoint { double rho, thE, B, sinB, j, alpha, g, e_emit; };
void lamppost_plunge_g_batch(const Params& p, const std::vector<double>& rLQvr, std::vector<double>& g);  // test: g = E_obs/E_gas of camera photons (r, L, Q, vr) on the plunging gas (needs p.lp_plunge)
void lamppost_batch(const Params& p, const std::vector<double>& r, std::vector<double>& log10e);  // test: log10 eps_E(r) as the kernel reads it (builds the table for p.a, p.lp_h, p.rin)
void torus_batch(const Params& p, const std::vector<double>& rmuL, std::vector<TorusPoint>& out, bool fast = false);  // fast: the same code in float (production)  // (r, mu, L) -> gas state and emission (test); g = 1/e_emit (fictitious camera)
bool grmhd_load(const std::string& path, std::string& err);  // loads a .bgm into 3D textures (trilinear, periodic phi)
bool grmhd_loaded();
bool grmhd_load_next(const std::string& path, std::string& err);  // v4.0: second snapshot (t + dt, same grid) for the time interpolation
void grmhd_prefetch(const std::string& path);                      // parse the snapshot after the next one on a worker thread (used by the next grmhd_load_next)
void grmhd_advance();                                              // the second snapshot becomes the first (swap, no upload)
void grmhd_set_blend(double tau, double dt_M, double pattern = 0.2);  // tau in [0,1] between the two snapshots (dt_M apart); tau < 0 = static first snapshot; pattern = pattern speed / Keplerian
bool grmhd_has_next();
void grmhd_set_half(bool on);  // v2.2f-d: store the snapshot as 16-bit float textures (half the VRAM / bandwidth); applies to the next grmhd_load
double grmhd_spin();
// R_high electron temperature in the EHT convention: Theta_e = Theta_i * eht_theta_e_coef(gam) / (2 + R), with Theta_i = (gam - 1) u / rho of the snapshot
// (from Theta_e = (m_p/m_e)(ge-1)(gp-1)/((gp-1) + (ge-1) R) u/rho, ge = 4/3, gp = 5/3; ipole and the EHT GRRT formula list)
inline double eht_theta_e_coef(double gam) { return (1.67262192369e-24 / 9.1093837015e-28) * (2.0 / 3.0) / (gam - 1.0); }
void grmhd_sample_batch(const Params& p, const std::vector<double>& rmuphi, std::vector<float>& out);  // (r, mu, phi_BL) -> 12 interpolated fields (test)
void grmhd_map_batch(const Params& p, const std::vector<double>& rmu, std::vector<float>& out);  // test: (r, mu) -> (X1, X2) through the device's inverse FMKS map
void grmhd_null_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out);   // test: max |g^{mu nu} p_mu p_nu| / p_t^2 (KS) along each ray
void radio_image(const Params& p, int ss, std::vector<double>& I, std::vector<double>& ab);  // v2.2b3: w*h radio image (cgs intensity) and (alpha,beta) in M per pixel
void radio_intensity_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out);  // test: specific intensity (cgs) per ray in torus mode
void turb_batch(const Params& p, const std::vector<double>& rpmt, std::vector<double>& out);  // (r, phi, mu) per entry -> (density factor, spot T factor) with p's time (test)
void emitter_mu_batch(double a, const std::vector<double>& rLQ, std::vector<double>& out);  // (r, L, Q) per entry -> mu in the gas frame (test)

}  // namespace bhs::kcuda
