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
  double beta[3] = {0, 0, 0};      // observer velocity in the ZAMO frame (e_r, e_theta, e_phi), in c
  int samples = 1;                 // sub-pixel samples (offline mode: render())
  uint32_t seed = 1;               // jitter seed (determinism)
  bool accum_ldr = false;          // true: Reinhard PER SAMPLE before accumulating (= CPU SSAA); false: HDR average
  double clamp_lum = 0;            // per-sample luminance clamp (fireflies); 0 = off
  Tonemap tonemap = Tonemap::Reinhard;
  double bloom = 0;                // bloom strength (0 = off)
  double beaming = 1.0;            // g -> g^beaming in the disk intensity: 1 = physical, < 1 = AESTHETIC (cinematic)
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
};

bool init(const std::string& sky_path, std::string& err);  // empty sky_path = no sky (black)
bool has_sky();
void set_size(int w, int h);
// Renders into the internal RGBA8 buffer; returns GPU ms (cudaEvent). Optional stats (costs a copy).
float render(const Params& p, Stats* stats = nullptr);  // = accum_reset + p.samples x accum_sample + resolve
// progressive accumulation (interactive mode): restarts with accum_reset() when something changes
void accum_reset();
int accum_count();
float accum_sample(const Params& p, int k, Stats* stats = nullptr);  // adds sample k (k = 0: pixel centre)
float resolve(const Params& p);                                       // media -> bloom -> tone mapping -> RGBA8
Timing last_timing();
void download_rgb(std::vector<uint8_t>& rgb);  // w*h*3, row 0 at the top

// CUDA-OpenGL interop: registers a GL texture (RGBA8) and copies the buffer into it.
bool gl_register(unsigned tex, int w, int h, std::string& err);
void gl_copy();
void gl_unregister();

// Pure tracing of a batch of rays (for the selftest). rays = (nr, nth, nph) of the photon.
struct RayOut {
  int fate, disk_hit, steps;
  double mu_end, phi_end, disk_r, disk_phi, L;
};
// tests: NT temperature (GPU LUT) and observer boost
void nt_lookup(const Params& p, const std::vector<double>& r, std::vector<double>& out);
void boost_batch(const std::vector<double>& np, const double beta[3], std::vector<double>& out);  // out = (n.xyz, dop) per ray
void trace_batch(const Params& p, const std::vector<double>& rays, std::vector<RayOut>& out);

}  // namespace bhs::kcuda
