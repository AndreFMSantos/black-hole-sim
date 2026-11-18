// OptiX denoiser (HDR model). The context, the denoiser, the state and the scratch are created ONCE (init); each frame only calls
// optixDenoiserInvoke. The runtime (libnvoptix) ships with the NVIDIA driver; the headers are NVIDIA/optix-dev's (outside Git).
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>  // defines the function table: in exactly one translation unit
#include <optix_stubs.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#ifdef BHS_HAVE_CUPTI
#include <cupti.h>
#endif

#include "denoise.hpp"

namespace bhs::denoise {
namespace {
struct State {
  OptixDeviceContext ctx = nullptr;
  OptixDenoiser den = nullptr;
  CUdeviceptr state = 0, scratch = 0;
  size_t state_bytes = 0, scratch_bytes = 0;
  int max_w = 0, max_h = 0, set_w = 0, set_h = 0;
  bool albedo = false, inited = false;
} S;

void log_cb(unsigned int level, const char* tag, const char* msg, void*) { std::fprintf(stderr, "[OptiX %u][%s] %s\n", level, tag, msg); }

#define OPTIX(call, what)                                                                  \
  do {                                                                                     \
    OptixResult r_ = (call);                                                               \
    if (r_ != OPTIX_SUCCESS) { err = std::string(what) + ": " + optixGetErrorName(r_); return false; } \
  } while (0)

void release() {
  if (S.den) optixDenoiserDestroy(S.den);
  if (S.state) cudaFree(reinterpret_cast<void*>(S.state));
  if (S.scratch) cudaFree(reinterpret_cast<void*>(S.scratch));
  S.den = nullptr; S.state = S.scratch = 0; S.set_w = S.set_h = 0;
}
}  // namespace

bool compiled() { return true; }
const char* version() {
  static char b[96];
  std::snprintf(b, sizeof b, "OptiX %d.%d.%d (headers; ABI %d; runtime = libnvoptix do driver)", OPTIX_VERSION / 10000, (OPTIX_VERSION % 10000) / 100, OPTIX_VERSION % 100, OPTIX_ABI_VERSION);
  return b;
}

bool init(int w, int h, bool albedo, std::string& err) {
  if (S.inited && S.albedo == albedo && w <= S.max_w && h <= S.max_h) return true;
  if (cudaFree(nullptr) != cudaSuccess) { err = "CUDA context unavailable"; return false; }
  if (!S.ctx) {
    OPTIX(optixInit(), "optixInit (libnvoptix do driver)");
    OptixDeviceContextOptions o{};
    o.logCallbackFunction = &log_cb;
    o.logCallbackLevel = 2;
    OPTIX(optixDeviceContextCreate(nullptr, &o, &S.ctx), "optixDeviceContextCreate");
  }
  release();
  OptixDenoiserOptions opt{};
  opt.guideAlbedo = albedo ? 1 : 0;
  opt.guideNormal = 0;
  opt.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
  OPTIX(optixDenoiserCreate(S.ctx, OPTIX_DENOISER_MODEL_KIND_HDR, &opt, &S.den), "optixDenoiserCreate");
  S.max_w = std::max(w, S.max_w); S.max_h = std::max(h, S.max_h);
  OptixDenoiserSizes sz{};
  OPTIX(optixDenoiserComputeMemoryResources(S.den, S.max_w, S.max_h, &sz), "optixDenoiserComputeMemoryResources");
  S.state_bytes = sz.stateSizeInBytes;
  S.scratch_bytes = std::max(sz.withoutOverlapScratchSizeInBytes, std::max(sz.computeIntensitySizeInBytes, sz.computeAverageColorSizeInBytes));
  if (cudaMalloc(reinterpret_cast<void**>(&S.state), S.state_bytes) != cudaSuccess || cudaMalloc(reinterpret_cast<void**>(&S.scratch), S.scratch_bytes) != cudaSuccess) {
    err = "cudaMalloc of the denoiser state/scratch failed";
    return false;
  }
  S.albedo = albedo;
  S.inited = true;
  return true;
}

static OptixImage2D img(const void* p, int w, int h) {
  OptixImage2D i{};
  i.data = reinterpret_cast<CUdeviceptr>(p);
  i.width = w; i.height = h;
  i.rowStrideInBytes = w * 16; i.pixelStrideInBytes = 16;
  i.format = OPTIX_PIXEL_FORMAT_FLOAT4;
  return i;
}

bool run(const void* in, const void* albedo, void* out, int w, int h, void* stream, std::string& err) {
  if (!S.inited) { err = "denoiser not initialised"; return false; }
  CUstream st = reinterpret_cast<CUstream>(stream);
  if (S.set_w != w || S.set_h != h) {  // only when the size changes (--move-scale, resizing)
    OPTIX(optixDenoiserSetup(S.den, st, w, h, S.state, S.state_bytes, S.scratch, S.scratch_bytes), "optixDenoiserSetup");
    S.set_w = w; S.set_h = h;
  }
  OptixDenoiserParams params{};  // hdrIntensity = nullptr: auto-exposure computed by OptiX; blendFactor = 0: 100 % denoised
  OptixDenoiserGuideLayer guide{};
  if (S.albedo && albedo) guide.albedo = img(albedo, w, h);
  OptixDenoiserLayer layer{};
  layer.input = img(in, w, h);
  layer.output = img(out, w, h);
  OPTIX(optixDenoiserInvoke(S.den, st, &params, S.state, S.state_bytes, &guide, &layer, 1, 0, 0, S.scratch, S.scratch_bytes), "optixDenoiserInvoke");
  return true;
}

#ifdef BHS_HAVE_CUPTI
namespace {
std::vector<KernelStat> g_kernels;
void CUPTIAPI buf_req(uint8_t** b, size_t* size, size_t* max_records) { *size = 1 << 20; *b = static_cast<uint8_t*>(std::malloc(*size)); *max_records = 0; }
void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t* b, size_t, size_t valid) {
  CUpti_Activity* rec = nullptr;
  while (cuptiActivityGetNextRecord(b, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
    const auto* k = reinterpret_cast<const CUpti_ActivityKernel13*>(rec);
    const std::string nm = k->name ? k->name : "?";
    auto it = std::find_if(g_kernels.begin(), g_kernels.end(), [&](const KernelStat& s) { return s.name == nm; });
    if (it == g_kernels.end()) { g_kernels.push_back({nm, 0, 0}); it = g_kernels.end() - 1; }
    ++it->count; it->total_us += double(k->end - k->start) / 1000.0;
  }
  std::free(b);
}
}  // namespace
bool kernel_trace_available() { return true; }
void kernel_trace_begin() { g_kernels.clear(); cuptiActivityRegisterCallbacks(buf_req, buf_done); cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL); }
std::vector<KernelStat> kernel_trace_end() { cudaDeviceSynchronize(); cuptiActivityFlushAll(1); cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL); return g_kernels; }
#else
bool kernel_trace_available() { return false; }
void kernel_trace_begin() {}
std::vector<KernelStat> kernel_trace_end() { return {}; }
#endif

void shutdown() {
  release();
  if (S.ctx) optixDeviceContextDestroy(S.ctx);
  S = State{};
}

}  // namespace bhs::denoise
