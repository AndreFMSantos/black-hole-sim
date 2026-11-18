// v0.6b — Kerr render in CUDA. One thread per pixel runs exactly the shade() of v0.6a (CPU).
// Ver notes/physics/06b-kerr-cuda.md.
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bhs/exr_loader.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lut.hpp>
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
  const float* lut;
  cudaTextureObject_t sky;
  uchar4* out;
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

__device__ __forceinline__ void trace_dispatch(const KP& p, double nr, double nth, double nph, double rout, double atol, bk::Trace& tr) {
  const double ap = p.phi_atol > 0 ? p.phi_atol : atol;
  if (p.prec == 0) bk::trace<double, double>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr);
  else if (p.prec == 1) bk::trace<float, float>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr);
  else bk::trace<double, float>(p.a, p.dist, p.theta, nr, nth, nph, p.rin, rout, p.tol, atol, ap, p.l_min, p.r_far, p.max_steps, tr);
}

// Each thread = 1 pixel. 1D blocks; 4 consecutive threads form a 2x2 quad (horizontal neighbour = lane^1,
// vertical = lane^2) to derive the sky-uv gradient via __shfl_sync, like v0.5b's subgroup_quad.
// One warp (32) = 8 quads = an 8x4-pixel tile; a block of 128 = 4 warps = a 16x8 tile.
__global__ void kernel(const KP p) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lq = lane & 3, lx = lq & 1, ly = lq >> 1, quad = lane >> 2;
  const int tilesx = (p.w + 15) / 16;
  const int bx = blockIdx.x % tilesx, by = blockIdx.x / tilesx;
  const int px = bx * 16 + (warp & 1) * 8 + (quad & 3) * 2 + lx;
  const int py = by * 8 + (warp >> 1) * 4 + (quad >> 2) * 2 + ly;
  const bool inside = px < p.w && py < p.h;

  double3 col = make_double3(0, 0, 0);
  float u = 0.f, v = 0.f;
  bool sky_ok = false;
  int steps = 0;
  if (inside) {
    const double x = (2.0 * ((px + 0.5) / p.w) - 1.0) * p.tan_half * p.aspect;
    const double y = (1.0 - 2.0 * ((py + 0.5) / p.h)) * p.tan_half;
    const double nrm = sqrt(1.0 + x * x + y * y);
    const double rout = p.nodisk ? 0.0 : p.rout;
    const double atol = p.tol * 0.1;
    bk::Trace tr;
    trace_dispatch(p, 1.0 / nrm, y / nrm, -x / nrm, rout, atol, tr);
    steps = tr.steps;
    if (tr.disk_hit) {
      const double r = tr.disk_r;
      const double g = p.nodoppler ? redshift(p.a, r, 0.0, p.dist, p.theta) : redshift(p.a, r, tr.L, p.dist, p.theta);
      const double T_obs = g * p.tpeak * disk_temp(r, p.rin, p.fnorm);
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
  const double L = 0.2126 * col.x + 0.7152 * col.y + 0.0722 * col.z;
  const double t = 1.0 / (1.0 + L);
  auto enc = [&](double c) {
    return (unsigned char)(pow(fmin(1.0, fmax(0.0, c * t)), 1.0 / 2.2) * 255.0 + 0.5);
  };
  p.out[size_t(py) * p.w + px] = make_uchar4(enc(col.x), enc(col.y), enc(col.z), 255);
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
  cudaMipmappedArray_t mm = nullptr;
  cudaTextureObject_t sky = 0;
  bool has_sky = false;
  uchar4* out = nullptr;
  unsigned long long* stats = nullptr;
  int w = 0, h = 0;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  cudaGraphicsResource_t gres = nullptr;
  int gw = 0, gh = 0;
} S;

uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

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
  const std::vector<float> lutd = lut::build();
  const Vec3 ref = lut::lookup(lutd, p.tpeak);
  k.exposure = p.exposure * 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z);
  k.sky_exposure = p.sky_exposure;
  k.phi_atol = p.phi_atol; k.l_min = p.l_min >= 0 ? p.l_min : 1e-4;
  k.tol = p.tol; k.r_far = p.r_far; k.tpeak = p.tpeak;
  auto f = [&](double x) { return std::pow(x, -3.0) * (1.0 - std::sqrt(k.rin / x)); };
  k.fnorm = f(49.0 / 36.0 * k.rin);
  k.lut = S.lut; k.sky = S.sky; k.out = S.out;
  return k;
}

}  // namespace

bool init(const std::string& sky_path, std::string& err) {
  CK(cudaFree(nullptr));
  const std::vector<float> lutd = lut::build();
  CK(cudaMalloc(&S.lut, lutd.size() * sizeof(float)));
  CK(cudaMemcpy(S.lut, lutd.data(), lutd.size() * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.stats, 3 * sizeof(unsigned long long)));
  CK(cudaEventCreate(&S.e0));
  CK(cudaEventCreate(&S.e1));
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
  if (w == S.w && h == S.h) return;
  if (S.out) CK(cudaFree(S.out));
  S.w = w; S.h = h;
  CK(cudaMalloc(&S.out, size_t(w) * h * sizeof(uchar4)));
  CK(cudaMemset(S.out, 0, size_t(w) * h * sizeof(uchar4)));
}

float render(const Params& p, Stats* stats) {
  set_size(p.w, p.h);
  KP k = make_kp(p);
  k.stats = nullptr;
  if (stats) {
    CK(cudaMemset(S.stats, 0, 3 * sizeof(unsigned long long)));
    k.stats = S.stats;
  }
  const int tilesx = (p.w + 15) / 16, tilesy = (p.h + 7) / 8;
  CK(cudaEventRecord(S.e0));
  kernel<<<tilesx * tilesy, 128>>>(k);
  CK(cudaEventRecord(S.e1));
  CK(cudaEventSynchronize(S.e1));
  CK(cudaGetLastError());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, S.e0, S.e1));
  if (stats) {
    unsigned long long h[3];
    CK(cudaMemcpy(h, S.stats, sizeof h, cudaMemcpyDeviceToHost));
    stats->steps_mean = double(h[0]) / (double(p.w) * p.h);
    stats->steps_max = int(h[2]);
    stats->warp_eff = h[1] ? double(h[0]) / double(h[1]) : 1.0;
  }
  return ms;
}

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

}  // namespace bhs::kcuda
