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
  bool nodisk = false, nodoppler = false;
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
  double turb_scale = 2.0;         // cell size in units of the scale height H(r)
  double spot_r = 10.0, spot_phi_deg = 0.0, spot_size = 1.5, spot_amp = 0.0;  // hot spot: orbits at r with Omega(r); amp > 0 turns it on
  double flare_t = 0.0, flare_w = 20.0, flare_boost = 0.0;                    // flare: Gaussian pulse in time on top of the spot amplitude
  bool slowlight = false;          // v0.9c2: propagation delay (slow light): the disk is seen at the emission time t_obs - flight time, not at t_obs
  double disk_time = 0.0;          // disk time (M): the turbulence and the spot rotate with the Kerr Omega(r); instantaneous (no propagation delay)
  bool expose_physical = false;    // fixed exposure (reference: black body f = 1 at tpeak) instead of renormalised to each LUT
};

struct Stats {
  double steps_mean = 0;
  int steps_max = 0;
  double warp_eff = 0;  // mean(steps)/mean(max steps per warp): 1 = no divergence
};

struct Timing {
  float trace_ms = 0;    // last sample
  float resolve_ms = 0;  // average + tone mapping (without the bloom)
  float bloom_ms = 0;    // bloom pyramid
  float denoise_ms = 0;  // optixDenoiserInvoke (0 if off)
};

bool init(const std::string& sky_path, std::string& err);  // empty sky_path = no sky (black)
bool has_sky();
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
void hud_draw(const std::string& text, int scale);  // draws text (lines separated by \n; lines starting with ! in red) on the final frame
void download_rgb(std::vector<uint8_t>& rgb);  // w*h*3, row 0 at the top

// CUDA-OpenGL interop: registers a GL texture (RGBA8) and copies the buffer into it.
bool gl_register(unsigned tex, int w, int h, std::string& err);
void gl_copy();
void gl_unregister();

// Pure tracing of a batch of rays (for the selftest). rays = (nr, nth, nph) of the photon.
struct RayOut {
  int fate, disk_hit, steps;
  double mu_end, phi_end, disk_r, disk_phi, L, Q;
};
// tests: NT temperature (GPU LUT) and observer boost
void camera_rays(const Params& p, const std::vector<double>& xy, std::vector<double>& out);  // n' per pixel (x, y) with p's basis
bool accum_finite();  // are all the accumulated values finite?
void nt_lookup(const Params& p, const std::vector<double>& r, std::vector<double>& out);
void boost_batch(const std::vector<double>& np, const double beta[3], std::vector<double>& out);  // out = (n.xyz, dop) per ray
void trace_batch(const Params& p, const std::vector<double>& rays, std::vector<RayOut>& out, int skip = 0);
void vol_flight_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out);  // test (slow light): per ray, the accumulated flight time until the transmittance drops to 1/2 (-1 if never)
void turb_batch(const Params& p, const std::vector<double>& rpmt, std::vector<double>& out);  // (r, phi, mu) per entry -> (density factor, spot T factor) with p's time (test)
void emitter_mu_batch(double a, const std::vector<double>& rLQ, std::vector<double>& out);  // (r, L, Q) per entry -> mu in the gas frame (test)

}  // namespace bhs::kcuda
