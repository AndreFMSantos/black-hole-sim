// v0.6b — Kerr render in CUDA. One thread per pixel runs exactly the shade() of v0.6a (CPU).
// Ver notes/physics/06b-kerr-cuda.md.
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bhs/exr_loader.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lut.hpp>
#include <bhs/novikov_thorne.hpp>
#include <bhs/skymap.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kerr_cuda.cuh"
#include "render.hpp"

namespace bhs::kcuda {
namespace {

#define CK(x)                                                                      \
  do {                                                                             \
    cudaError_t e_ = (x);                                                          \
    if (e_ != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                \
    }                                                                              \
  } while (0)

// parameters in kernel units
struct KP {
  int w, h, use_sky, sky_raw, nodisk, nodoppler, max_steps, prec;
  double phi_atol, l_min;
  double a, dist, theta, phi, tan_half, aspect, rin, rout, exposure, sky_exposure, tol, r_far;
  double tpeak, fnorm;
  // v0.7d
  int nt, nt_n, ldr, sample_k;
  uint32_t seed;
  double nt_rin, nt_rout, bx, by, bz, clamp_lum, beaming;
  const float* nt_tab;
  float4* accum;
  const float* lut;
  cudaTextureObject_t sky;
  unsigned long long* stats;  // [0] sum of steps, [1] sum of per-warp maxima * 32, [2] maximum
};

constexpr double kT0 = lut::kT0, kT1 = lut::kT1;

__device__ double3 lut_lookup(const float* d, double T) {
  if (T <= kT0) return make_double3(0, 0, 0);
  const double x = log(fmin(T, kT1) / kT0) / log(kT1 / kT0) * (lut::kN - 1);
  const int i = min(lut::kN - 2, int(x));
  const double f = x - i;
  return make_double3((1 - f) * d[3 * i] + f * d[3 * i + 3], (1 - f) * d[3 * i + 1] + f * d[3 * i + 4],
                      (1 - f) * d[3 * i + 2] + f * d[3 * i + 5]);
}

__device__ uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ b * 0x85EBCA6Bu ^ c * 0xC2B2AE35u;
  h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
  return h;
}
__device__ double vnoise(double x, double y, double z) {
  const int xi = int(floor(x)), yi = int(floor(y)), zi = int(floor(z));
  const double fx = x - xi, fy = y - yi, fz = z - zi;
  auto sm = [](double t) { return t * t * (3 - 2 * t); };
  double r = 0;
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        const double hv = double(hash3(uint32_t(xi + dx), uint32_t(yi + dy), uint32_t(zi + dz)) & 0xFFFFFF) / 16777216.0;
        r += hv * (dx ? sm(fx) : 1 - sm(fx)) * (dy ? sm(fy) : 1 - sm(fy)) * (dz ? sm(fz) : 1 - sm(fz));
      }
  return r;
}

// kerr::redshift
__device__ double redshift(double a, double r, double L, double r_obs, double th_obs) {
  const double s = sin(th_obs), mu = cos(th_obs);
  const double Sigma = r_obs * r_obs + a * a * mu * mu, Delta = bk::delta_d(r_obs, a);
  const double rr = r_obs * r_obs + a * a;
  const double A = rr * rr - a * a * Delta * s * s;
  const double alpha = sqrt(Sigma * Delta / A), omega = 2.0 * a * r_obs / A;
  const double E_obs = (1.0 - omega * L) / alpha;
  const double ut = (pow(r, 1.5) + a) / (pow(r, 0.75) * sqrt(pow(r, 1.5) - 3.0 * sqrt(r) + 2.0 * a));
  const double Omega = 1.0 / (pow(r, 1.5) + a);
  return E_obs / (ut * (1.0 - Omega * L));
}

// disk::temperature
__device__ double disk_temp(double r, double rin, double fnorm) {
  if (r <= rin) return 0.0;
  const double f = pow(r, -3.0) * (1.0 - sqrt(rin / r));
  return pow(f / fnorm, 0.25);
}

// nt::Profile::temperature: table in ln r between r_in (ISCO) and r_out, linear interpolation; r^-3/4 tail
__device__ double nt_temp(const KP& p, double r) {
  if (r <= p.nt_rin) return 0.0;
  if (r >= p.nt_rout) return double(p.nt_tab[p.nt_n - 1]) * pow(r / p.nt_rout, -0.75);
  const double lmin = log(p.nt_rin), lmax = log(p.nt_rout);
  const double x = (log(r) - lmin) / (lmax - lmin) * (p.nt_n - 1);
  const int i = min(p.nt_n - 2, int(x));
  const double f = x - i;
  return (1 - f) * double(p.nt_tab[i]) + f * double(p.nt_tab[i + 1]);
}

// lorentz::observer_to_zamo: n' (observer) -> n (ZAMO); dop = E'/E_zamo. beta = observer velocity in the ZAMO frame.
__device__ void observer_to_zamo(double3 np, double bx, double by, double bz, double3& n, double& dop) {
  const double b2 = bx * bx + by * by + bz * bz;
  if (b2 < 1e-30) { n = np; dop = 1.0; return; }
  const double gamma = 1.0 / sqrt(1.0 - b2);
  const double bn = bx * np.x + by * np.y + bz * np.z;
  const double E = gamma * (1.0 + bn);
  const double k = ((gamma - 1.0) * bn / b2) + gamma;
  double3 q = make_double3(np.x + k * bx, np.y + k * by, np.z + k * bz);
  const double inv = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  n = make_double3(q.x * inv, q.y * inv, q.z * inv);
  dop = 1.0 / E;
}

__device__ __forceinline__ void trace_dispatch(const KP& p, double nr, double nth, double nph, double rout, double atol, bk::Trace& tr) {
  const double ap = p.phi_atol > 0 ? p.phi_atol : atol;
  if (p.prec == 0) bk::trace<double, double>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr);
  else if (p.prec == 1) bk::trace<float, float>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr);
  else bk::trace<double, float>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr);
}

// Each thread = 1 pixel and 1 sub-pixel sample (R2 jitter per quad, sample index sample_k; sample 0 is the
// pixel centre). Adds the HDR colour into p.accum (RGBA32F); the resolve (average, bloom, tone mapping) is separate.
// 1D blocks; 4 consecutive threads form a 2x2 quad (horizontal neighbour = lane^1, vertical = lane^2) to
// derive the sky-uv gradient via __shfl_sync, like v0.5b's subgroup_quad. The jitter is the same in the 4
// threads of the quad (quad hash), so that the uv difference between neighbours remains the pixel's derivative.
// One warp (32) = 8 quads = an 8x4-pixel tile; a block of 128 = 4 warps = a 16x8 tile.
__global__ void kernel(const KP p) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lq = lane & 3, lx = lq & 1, ly = lq >> 1, quad = lane >> 2;
  const int tilesx = (p.w + 15) / 16;
  const int bx = blockIdx.x % tilesx, by = blockIdx.x / tilesx;
  const int px = bx * 16 + (warp & 1) * 8 + (quad & 3) * 2 + lx;
  const int py = by * 8 + (warp >> 1) * 4 + (quad >> 2) * 2 + ly;
  const bool inside = px < p.w && py < p.h;

  double jx = 0, jy = 0;
  if (p.sample_k > 0) {
    const uint32_t hq = hash3(uint32_t(px >> 1), uint32_t(py >> 1), p.seed);
    const double kx = p.sample_k * 0.7548776662466927 + double(hq & 0xFFFF) / 65536.0;
    const double ky = p.sample_k * 0.5698402909980532 + double(hq >> 16) / 65536.0;
    jx = kx - floor(kx) - 0.5;
    jy = ky - floor(ky) - 0.5;
  }

  double3 col = make_double3(0, 0, 0);
  float u = 0.f, v = 0.f;
  bool sky_ok = false;
  int steps = 0;
  if (inside) {
    const double x = (2.0 * ((px + 0.5 + jx) / p.w) - 1.0) * p.tan_half * p.aspect;
    const double y = (1.0 - 2.0 * ((py + 0.5 + jy) / p.h)) * p.tan_half;
    const double nrm = sqrt(1.0 + x * x + y * y);
    const double rout = p.nodisk ? 0.0 : p.rout;
    const double atol = p.tol * 0.1;
    // moving observer: n' = (1, y, -x)/nrm in the observer frame -> n in the ZAMO (aberration)
    double3 nz; double dop;
    observer_to_zamo(make_double3(1.0 / nrm, y / nrm, -x / nrm), p.bx, p.by, p.bz, nz, dop);
    bk::Trace tr;
    trace_dispatch(p, nz.x, nz.y, nz.z, rout, atol, tr);
    steps = tr.steps;
    if (tr.disk_hit) {
      const double r = tr.disk_r;
      double g = p.nodoppler ? redshift(p.a, r, 0.0, p.dist, p.theta) : redshift(p.a, r, tr.L, p.dist, p.theta) * dop;
      if (p.beaming != 1.0) g = pow(g, p.beaming);  // --beaming: aesthetic (not physical) choice, see the notes
      const double Tprof = p.nt ? nt_temp(p, r) : disk_temp(r, p.rin, p.fnorm);
      const double T_obs = g * p.tpeak * Tprof;
      const double psi = tr.disk_phi + p.phi - 2.2 * log(r / p.rin);
      const double lr = log(r);
      const double n = 0.55 * vnoise(3 * cos(psi), 3 * sin(psi), 9 * lr) + 0.45 * vnoise(9 * cos(psi), 9 * sin(psi), 28 * lr);
      const double k = p.exposure * (0.55 + (1.15 - 0.55) * n);
      const double3 c = lut_lookup(p.lut, T_obs);
      col = make_double3(k * c.x, k * c.y, k * c.z);
    } else if (tr.fate == 0 && p.use_sky) {
      const double ph = tr.phi_end + p.phi;  // axisymmetry: the camera azimuth only adds
      double uu = 0.5 - ph / (2.0 * M_PI);
      uu -= floor(uu);
      u = float(uu);
      v = float(acos(tr.mu_end) / M_PI);
      sky_ok = true;
    }
  }
  __syncwarp();
  if (p.use_sky) {  // uniform flow: all threads of the warp execute the shuffles
    const unsigned full = 0xffffffffu;
    const float uH = __shfl_xor_sync(full, u, 1), vH = __shfl_xor_sync(full, v, 1);
    const float uV = __shfl_xor_sync(full, u, 2), vV = __shfl_xor_sync(full, v, 2);
    const int okH = __shfl_xor_sync(full, int(sky_ok), 1), okV = __shfl_xor_sync(full, int(sky_ok), 2);
    if (sky_ok) {
      const float sx = lx ? -1.f : 1.f, sy = ly ? -1.f : 1.f;  // the neighbour is at +1 if lx == 0, at -1 if lx == 1
      auto wrap = [](float d) { return d - rintf(d); };
      float2 gx = okH ? make_float2(sx * wrap(uH - u), sx * (vH - v)) : make_float2(0.f, 0.f);
      float2 gy = okV ? make_float2(sy * wrap(uV - u), sy * (vV - v)) : make_float2(0.f, 0.f);
      const float4 s = p.sky_raw ? tex2DLod<float4>(p.sky, u, v, 0.f) : tex2DGrad<float4>(p.sky, u, v, gx, gy);
      col = make_double3(p.sky_exposure * s.x, p.sky_exposure * s.y, p.sky_exposure * s.z);
    }
  }
  if (p.stats) {
    const int wm = __reduce_max_sync(0xffffffffu, steps);
    const int ws = __reduce_add_sync(0xffffffffu, steps);
    if (lane == 0) {
      atomicAdd(&p.stats[0], (unsigned long long)ws);
      atomicAdd(&p.stats[1], (unsigned long long)wm * 32ull);
      atomicMax(&p.stats[2], (unsigned long long)wm);
    }
  }
  if (!inside) return;
  double L = 0.2126 * col.x + 0.7152 * col.y + 0.0722 * col.z;
  // per-sample luminance clamp (fireflies on the blue side of the Doppler); 0 = off
  if (p.clamp_lum > 0 && L > p.clamp_lum) { const double s = p.clamp_lum / L; col.x *= s; col.y *= s; col.z *= s; L = p.clamp_lum; }
  // "ldr" mode: Reinhard per sample before accumulating (= v0.6a/b and the CPU SSAA); otherwise accumulates HDR
  if (p.ldr) { const double t = 1.0 / (1.0 + L); col.x *= t; col.y *= t; col.z *= t; }
  float4& acc = p.accum[size_t(py) * p.w + px];
  acc.x += float(col.x); acc.y += float(col.y); acc.z += float(col.z);
}

// ---------------------------------------------------------------- resolve: media, bloom, tone mapping
constexpr int kBloomLevels = 6;

__global__ void k_avg(const float4* acc, float4* hdr, int n, float inv) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float4 a = acc[i];
  hdr[i] = make_float4(a.x * inv, a.y * inv, a.z * inv, 1.f);
}
__device__ __forceinline__ float4 ld(const float4* t, int w, int h, int x, int y) {
  return t[size_t(min(max(y, 0), h - 1)) * w + min(max(x, 0), w - 1)];
}
// 2x2 (box) reduction of the previous level
__global__ void k_down(const float4* src, int sw, int sh, float4* dst, int dw, int dh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  float4 a = ld(src, sw, sh, 2 * x, 2 * y), b = ld(src, sw, sh, 2 * x + 1, 2 * y);
  float4 c = ld(src, sw, sh, 2 * x, 2 * y + 1), d = ld(src, sw, sh, 2 * x + 1, 2 * y + 1);
  dst[size_t(y) * dw + x] = make_float4(0.25f * (a.x + b.x + c.x + d.x), 0.25f * (a.y + b.y + c.y + d.y), 0.25f * (a.z + b.z + c.z + d.z), 1.f);
}
// separable Gaussian [1 4 6 4 1]/16
__global__ void k_blur(const float4* src, float4* dst, int w, int h, int dx, int dy) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float wt[5] = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  float3 s = make_float3(0, 0, 0);
  for (int k = -2; k <= 2; ++k) {
    const float4 t = ld(src, w, h, x + k * dx, y + k * dy);
    s.x += wt[k + 2] * t.x; s.y += wt[k + 2] * t.y; s.z += wt[k + 2] * t.z;
  }
  dst[size_t(y) * w + x] = make_float4(s.x, s.y, s.z, 1.f);
}
struct BloomPtrs {
  const float4* lv[kBloomLevels];
  int lw[kBloomLevels], lh[kBloomLevels];
};
__device__ float3 bilerp(const float4* t, int w, int h, float fx, float fy) {  // fx, fy in texels (centres at integers)
  const float x0 = floorf(fx), y0 = floorf(fy), tx = fx - x0, ty = fy - y0;
  const int ix = int(x0), iy = int(y0);
  const float4 a = ld(t, w, h, ix, iy), b = ld(t, w, h, ix + 1, iy), c = ld(t, w, h, ix, iy + 1), d = ld(t, w, h, ix + 1, iy + 1);
  return make_float3((1 - ty) * ((1 - tx) * a.x + tx * b.x) + ty * ((1 - tx) * c.x + tx * d.x),
                     (1 - ty) * ((1 - tx) * a.y + tx * b.y) + ty * ((1 - tx) * c.y + tx * d.y),
                     (1 - ty) * ((1 - tx) * a.z + tx * b.z) + ty * ((1 - tx) * c.z + tx * d.z));
}
// ACES, Narkowicz fit (2016); input already multiplied by the exposure
__device__ float aces(float x) { return fminf(1.f, fmaxf(0.f, (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f))); }

__global__ void k_final(const float4* hdr, uchar4* out, int w, int h, BloomPtrs bp, int levels, float bloom, int ldr, int tonemap) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float4 c4 = hdr[size_t(y) * w + x];
  float3 c = make_float3(c4.x, c4.y, c4.z);
  if (bloom > 0.f && levels > 0) {
    float3 sum = make_float3(0, 0, 0);
    for (int l = 0; l < levels; ++l) {
      const float fx = (x + 0.5f) * bp.lw[l] / w - 0.5f, fy = (y + 0.5f) * bp.lh[l] / h - 0.5f;
      const float3 b = bilerp(bp.lv[l], bp.lw[l], bp.lh[l], fx, fy);
      sum.x += b.x; sum.y += b.y; sum.z += b.z;
    }
    const float k = bloom / levels;
    c.x += k * sum.x; c.y += k * sum.y; c.z += k * sum.z;
  }
  if (!ldr) {
    if (tonemap == 0) {  // Reinhard on the luminance (as in v0.6a/b, but after the average)
      const float L = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z, t = 1.f / (1.f + L);
      c.x *= t; c.y *= t; c.z *= t;
    } else {  // per-channel ACES, exposure 0.6 (Narkowicz fit convention)
      c.x = aces(0.6f * c.x); c.y = aces(0.6f * c.y); c.z = aces(0.6f * c.z);
    }
  }
  auto enc = [](float v) { return (unsigned char)(powf(fminf(1.f, fmaxf(0.f, v)), 1.f / 2.2f) * 255.f + 0.5f); };
  out[size_t(y) * w + x] = make_uchar4(enc(c.x), enc(c.y), enc(c.z), 255);
}

// --- test tools
__global__ void nt_kernel(KP p, const double* r, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = nt_temp(p, r[i]);
}
__global__ void boost_kernel(const double* np, double bx, double by, double bz, int n, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double3 nz; double dop;
  observer_to_zamo(make_double3(np[3 * i], np[3 * i + 1], np[3 * i + 2]), bx, by, bz, nz, dop);
  out[4 * i] = nz.x; out[4 * i + 1] = nz.y; out[4 * i + 2] = nz.z; out[4 * i + 3] = dop;
}

__global__ void batch_kernel(KP p, const double* rays, int n, bk::Trace* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double rout = p.nodisk ? 0.0 : p.rout;
  trace_dispatch(p, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], rout, p.tol * 0.1, out[i]);
}

// ------------------------------------------------------------------ host
struct State {
  float* lut = nullptr;
  std::vector<float> lutd;
  cudaMipmappedArray_t mm = nullptr;
  cudaTextureObject_t sky = 0;
  bool has_sky = false;
  uchar4* out = nullptr;
  float4* accum = nullptr;  // sum of the samples (RGBA32F)
  float4* hdr = nullptr;    // media
  float4* lv[kBloomLevels] = {};  // bloom pyramid
  float4* tmp = nullptr;
  int lw[kBloomLevels] = {}, lh[kBloomLevels] = {};
  int samples = 0;  // accumulated samples
  float* nt_tab = nullptr;
  int nt_n = 0;
  double nt_a = -1, nt_rin = 0, nt_rout = 0;
  unsigned long long* stats = nullptr;
  int w = 0, h = 0;
  cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr;
  Timing last{};
  cudaGraphicsResource_t gres = nullptr;
  int gw = 0, gh = 0;
} S;

uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

void ensure_nt(const Params& p, KP& k) {
  // Novikov-Thorne T(r) table in ln r (2048 points, nt::Profile::table()); rebuilt when a (or rout) changes
  const double rout_prof = std::max(2.0 * p.rout, 100.0);
  if (!S.nt_tab || std::abs(p.a - S.nt_a) > 1e-9 || std::abs(rout_prof - S.nt_rout) > 1e-9) {
    const nt::Profile prof(p.a, rout_prof);
    std::vector<float> tab(prof.table().begin(), prof.table().end());
    if (!S.nt_tab) CK(cudaMalloc(&S.nt_tab, 2048 * sizeof(float)));
    CK(cudaMemcpy(S.nt_tab, tab.data(), tab.size() * sizeof(float), cudaMemcpyHostToDevice));
    S.nt_n = int(tab.size()); S.nt_a = p.a; S.nt_rout = rout_prof; S.nt_rin = prof.r_in();
  }
  k.nt_tab = S.nt_tab; k.nt_n = S.nt_n; k.nt_rin = S.nt_rin; k.nt_rout = S.nt_rout;
}

KP make_kp(const Params& p) {
  KP k{};
  k.w = p.w; k.h = p.h; k.use_sky = p.use_sky && S.has_sky; k.sky_raw = p.sky_raw; k.nodisk = p.nodisk; k.nodoppler = p.nodoppler;
  k.max_steps = p.max_steps; k.prec = int(p.prec);
  k.a = p.a; k.dist = p.dist; k.theta = p.theta_deg * M_PI / 180.0; k.phi = p.phi_deg * M_PI / 180.0;
  k.tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
  k.aspect = double(p.w) / p.h;
  k.rin = p.rin < 0 ? kerr::isco(p.a, true) : p.rin;
  k.rout = p.rout;
  // exposure: black-body luminance at T_peak <-> 0.5 before tone mapping (as in v0.6a)
  const Vec3 ref = lut::lookup(S.lutd, p.tpeak);
  k.exposure = p.exposure * 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z);
  k.sky_exposure = p.sky_exposure;
  k.phi_atol = p.phi_atol; k.l_min = p.l_min >= 0 ? p.l_min : 1e-4;
  k.tol = p.tol; k.r_far = p.r_far; k.tpeak = p.tpeak;
  auto f = [&](double x) { return std::pow(x, -3.0) * (1.0 - std::sqrt(k.rin / x)); };
  k.fnorm = f(49.0 / 36.0 * k.rin);
  k.lut = S.lut; k.sky = S.sky;
  k.nt = p.nt; k.ldr = p.accum_ldr; k.seed = p.seed;
  k.bx = p.beta[0]; k.by = p.beta[1]; k.bz = p.beta[2];
  k.clamp_lum = p.clamp_lum; k.beaming = p.beaming;
  if (p.nt) ensure_nt(p, k);
  k.accum = S.accum;
  return k;
}

}  // namespace

bool init(const std::string& sky_path, std::string& err) {
  CK(cudaFree(nullptr));
  S.lutd = lut::build();
  const std::vector<float>& lutd = S.lutd;
  CK(cudaMalloc(&S.lut, lutd.size() * sizeof(float)));
  CK(cudaMemcpy(S.lut, lutd.data(), lutd.size() * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.stats, 3 * sizeof(unsigned long long)));
  CK(cudaEventCreate(&S.e0));
  CK(cudaEventCreate(&S.e1));
  CK(cudaEventCreate(&S.e2));
  CK(cudaEventCreate(&S.e3));
  if (sky_path.empty()) return true;
  sky::Skymap m;
  if (!sky::load_exr_half(sky_path, m, &err)) return false;
  int levels = 1;
  for (int n = std::max(m.w, m.h); n > 1; n >>= 1) ++levels;
  cudaChannelFormatDesc cd = cudaCreateChannelDescHalf4();
  CK(cudaMallocMipmappedArray(&S.mm, &cd, make_cudaExtent(m.w, m.h, 0), levels));
  // level 0: RGB half -> RGBA half; following levels: 2x2 mean in float (linear HDR values)
  std::vector<uint16_t> cur(size_t(m.w) * m.h * 4), nxt;
  for (size_t i = 0; i < size_t(m.w) * m.h; ++i) {
    cur[4 * i] = m.rgb[3 * i]; cur[4 * i + 1] = m.rgb[3 * i + 1]; cur[4 * i + 2] = m.rgb[3 * i + 2]; cur[4 * i + 3] = f2h(1.f);
  }
  m.rgb.clear(); m.rgb.shrink_to_fit();
  int cw = m.w, ch = m.h;
  for (int l = 0; l < levels; ++l) {
    cudaArray_t arr;
    CK(cudaGetMipmappedArrayLevel(&arr, S.mm, l));
    CK(cudaMemcpy2DToArray(arr, 0, 0, cur.data(), size_t(cw) * 8, size_t(cw) * 8, ch, cudaMemcpyHostToDevice));
    if (l + 1 == levels) break;
    const int nw = std::max(1, cw / 2), nh = std::max(1, ch / 2);
    nxt.assign(size_t(nw) * nh * 4, 0);
    for (int y = 0; y < nh; ++y)
      for (int x = 0; x < nw; ++x)
        for (int c = 0; c < 4; ++c) {
          float s = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) s += h2f(cur[(size_t(std::min(ch - 1, 2 * y + dy)) * cw + std::min(cw - 1, 2 * x + dx)) * 4 + c]);
          nxt[(size_t(y) * nw + x) * 4 + c] = f2h(0.25f * s);
        }
    cur.swap(nxt); cw = nw; ch = nh;
  }
  cudaResourceDesc rd{};
  rd.resType = cudaResourceTypeMipmappedArray;
  rd.res.mipmap.mipmap = S.mm;
  cudaTextureDesc td{};
  td.addressMode[0] = cudaAddressModeWrap;   // u: right ascension, wraps around
  td.addressMode[1] = cudaAddressModeClamp;  // v: polos
  td.filterMode = cudaFilterModeLinear;
  td.mipmapFilterMode = cudaFilterModeLinear;
  td.readMode = cudaReadModeElementType;
  td.normalizedCoords = 1;
  td.maxAnisotropy = 16;
  td.minMipmapLevelClamp = 0.f;
  td.maxMipmapLevelClamp = float(levels - 1);
  CK(cudaCreateTextureObject(&S.sky, &rd, &td, nullptr));
  S.has_sky = true;
  std::printf("sky: %s (%dx%d, %d levels, aniso 16x)\n", sky_path.c_str(), m.w, m.h, levels);
  return true;
}

bool has_sky() { return S.has_sky; }

void set_size(int w, int h) {
  if (w == S.w && h == S.h && S.accum) return;
  if (S.out) { CK(cudaFree(S.out)); CK(cudaFree(S.accum)); CK(cudaFree(S.hdr)); CK(cudaFree(S.tmp)); for (auto* l : S.lv) CK(cudaFree(l)); }
  S.w = w; S.h = h;
  const size_t n = size_t(w) * h;
  CK(cudaMalloc(&S.out, n * sizeof(uchar4)));
  CK(cudaMemset(S.out, 0, n * sizeof(uchar4)));
  CK(cudaMalloc(&S.accum, n * sizeof(float4)));
  CK(cudaMalloc(&S.hdr, n * sizeof(float4)));
  int lw = w, lh = h;
  size_t tmpn = 0;
  for (int l = 0; l < kBloomLevels; ++l) {
    lw = std::max(1, (lw + 1) / 2); lh = std::max(1, (lh + 1) / 2);
    S.lw[l] = lw; S.lh[l] = lh;
    CK(cudaMalloc(&S.lv[l], size_t(lw) * lh * sizeof(float4)));
    tmpn = std::max(tmpn, size_t(lw) * lh);
  }
  CK(cudaMalloc(&S.tmp, tmpn * sizeof(float4)));
  S.samples = 0;
  CK(cudaMemset(S.accum, 0, n * sizeof(float4)));
}

void accum_reset() {
  if (S.accum) CK(cudaMemset(S.accum, 0, size_t(S.w) * S.h * sizeof(float4)));
  S.samples = 0;
}
int accum_count() { return S.samples; }

float accum_sample(const Params& p, int k, Stats* stats) {
  set_size(p.w, p.h);
  KP kp = make_kp(p);
  kp.sample_k = k;
  kp.stats = nullptr;
  if (stats) {
    CK(cudaMemset(S.stats, 0, 3 * sizeof(unsigned long long)));
    kp.stats = S.stats;
  }
  const int tilesx = (p.w + 15) / 16, tilesy = (p.h + 7) / 8;
  CK(cudaEventRecord(S.e0));
  kernel<<<tilesx * tilesy, 128>>>(kp);
  CK(cudaEventRecord(S.e1));
  CK(cudaEventSynchronize(S.e1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, S.e0, S.e1));
  ++S.samples;
  if (stats) {
    unsigned long long h[3];
    CK(cudaMemcpy(h, S.stats, sizeof h, cudaMemcpyDeviceToHost));
    stats->steps_mean = double(h[0]) / (double(p.w) * p.h);
    stats->steps_max = int(h[2]);
    stats->warp_eff = h[1] ? double(h[0]) / double(h[1]) : 1.0;
  }
  S.last.trace_ms = ms;
  return ms;
}

// average of the samples -> bloom (Gaussian pyramid) -> tone mapping -> RGBA8
float resolve(const Params& p) {
  set_size(p.w, p.h);
  const int w = S.w, h = S.h, n = w * h;
  CK(cudaEventRecord(S.e0));
  k_avg<<<(n + 255) / 256, 256>>>(S.accum, S.hdr, n, 1.f / float(std::max(1, S.samples)));
  CK(cudaEventRecord(S.e1));
  const bool bloom = p.bloom > 0;
  if (bloom) {
    const dim3 blk(16, 16);
    const float4* src = S.hdr;
    int sw = w, sh = h;
    for (int l = 0; l < kBloomLevels; ++l) {
      const int dw = S.lw[l], dh = S.lh[l];
      const dim3 grid((dw + 15) / 16, (dh + 15) / 16);
      k_down<<<grid, blk>>>(src, sw, sh, S.lv[l], dw, dh);
      k_blur<<<grid, blk>>>(S.lv[l], S.tmp, dw, dh, 1, 0);
      k_blur<<<grid, blk>>>(S.tmp, S.lv[l], dw, dh, 0, 1);
      src = S.lv[l]; sw = dw; sh = dh;
    }
  }
  CK(cudaEventRecord(S.e2));
  BloomPtrs bp{};
  for (int l = 0; l < kBloomLevels; ++l) { bp.lv[l] = S.lv[l]; bp.lw[l] = S.lw[l]; bp.lh[l] = S.lh[l]; }
  const dim3 blk(16, 16), grid((w + 15) / 16, (h + 15) / 16);
  k_final<<<grid, blk>>>(S.hdr, S.out, w, h, bp, bloom ? kBloomLevels : 0, float(p.bloom), p.accum_ldr, int(p.tonemap));
  CK(cudaEventRecord(S.e3));
  CK(cudaEventSynchronize(S.e3));
  CK(cudaGetLastError());
  float a = 0, b = 0, c = 0;
  CK(cudaEventElapsedTime(&a, S.e0, S.e1));
  CK(cudaEventElapsedTime(&b, S.e1, S.e2));
  CK(cudaEventElapsedTime(&c, S.e2, S.e3));
  S.last.bloom_ms = b;
  S.last.resolve_ms = a + c;
  return a + b + c;
}

float render(const Params& p, Stats* stats) {
  set_size(p.w, p.h);
  accum_reset();
  float ms = 0;
  for (int k = 0; k < std::max(1, p.samples); ++k) ms += accum_sample(p, k, k == 0 ? stats : nullptr);
  ms += resolve(p);
  return ms;
}

Timing last_timing() { return S.last; }

void download_rgb(std::vector<uint8_t>& rgb) {
  std::vector<uchar4> px(size_t(S.w) * S.h);
  CK(cudaMemcpy(px.data(), S.out, px.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));
  rgb.resize(px.size() * 3);
  for (size_t i = 0; i < px.size(); ++i) { rgb[3 * i] = px[i].x; rgb[3 * i + 1] = px[i].y; rgb[3 * i + 2] = px[i].z; }
}

bool gl_register(unsigned tex, int w, int h, std::string& err) {
  gl_unregister();
  cudaError_t e = cudaGraphicsGLRegisterImage(&S.gres, tex, 0x0DE1 /*GL_TEXTURE_2D*/, cudaGraphicsRegisterFlagsWriteDiscard);
  if (e != cudaSuccess) { err = cudaGetErrorString(e); S.gres = nullptr; return false; }
  S.gw = w; S.gh = h;
  return true;
}
void gl_copy() {
  if (!S.gres) return;
  CK(cudaGraphicsMapResources(1, &S.gres));
  cudaArray_t arr;
  CK(cudaGraphicsSubResourceGetMappedArray(&arr, S.gres, 0, 0));
  CK(cudaMemcpy2DToArray(arr, 0, 0, S.out, size_t(S.w) * 4, size_t(S.w) * 4, S.h, cudaMemcpyDeviceToDevice));
  CK(cudaGraphicsUnmapResources(1, &S.gres));
}
void gl_unregister() {
  if (S.gres) { cudaGraphicsUnregisterResource(S.gres); S.gres = nullptr; }
}

void trace_batch(const Params& p, const std::vector<double>& rays, std::vector<RayOut>& out) {
  const int n = int(rays.size() / 3);
  KP k = make_kp(p);
  double* dr; bk::Trace* dt;
  CK(cudaMalloc(&dr, rays.size() * sizeof(double)));
  CK(cudaMalloc(&dt, n * sizeof(bk::Trace)));
  CK(cudaMemcpy(dr, rays.data(), rays.size() * sizeof(double), cudaMemcpyHostToDevice));
  batch_kernel<<<(n + 63) / 64, 64>>>(k, dr, n, dt);
  CK(cudaGetLastError());
  std::vector<bk::Trace> h(n);
  CK(cudaMemcpy(h.data(), dt, n * sizeof(bk::Trace), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dt));
  out.resize(n);
  for (int i = 0; i < n; ++i)
    out[i] = RayOut{h[i].fate, h[i].disk_hit, h[i].steps, h[i].mu_end, h[i].phi_end, h[i].disk_r, h[i].disk_phi, h[i].L};
}

void nt_lookup(const Params& p, const std::vector<double>& r, std::vector<double>& out) {
  KP kp = make_kp(p);
  Params q = p; q.nt = true;
  kp = make_kp(q);
  double *dr, *dout;
  CK(cudaMalloc(&dr, r.size() * sizeof(double)));
  CK(cudaMalloc(&dout, r.size() * sizeof(double)));
  CK(cudaMemcpy(dr, r.data(), r.size() * sizeof(double), cudaMemcpyHostToDevice));
  nt_kernel<<<(int(r.size()) + 63) / 64, 64>>>(kp, dr, int(r.size()), dout);
  out.resize(r.size());
  CK(cudaMemcpy(out.data(), dout, r.size() * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dr)); CK(cudaFree(dout));
}

void boost_batch(const std::vector<double>& np, const double beta[3], std::vector<double>& out) {
  const int n = int(np.size() / 3);
  double *dn, *dout;
  CK(cudaMalloc(&dn, np.size() * sizeof(double)));
  CK(cudaMalloc(&dout, size_t(n) * 4 * sizeof(double)));
  CK(cudaMemcpy(dn, np.data(), np.size() * sizeof(double), cudaMemcpyHostToDevice));
  boost_kernel<<<(n + 63) / 64, 64>>>(dn, beta[0], beta[1], beta[2], n, dout);
  out.resize(size_t(n) * 4);
  CK(cudaMemcpy(out.data(), dout, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaFree(dn)); CK(cudaFree(dout));
}

}  // namespace bhs::kcuda
