#pragma once
// v0.8c — OPTIONAL OptiX denoiser. Two layers: at build time (BHS_OPTIX, only if the headers are in
// code/third_party/optix-dev, see tools/fetch_optix.sh) and at run time (--denoise none|optix). Without OptiX this module
// compiles into a stub that says it is not available.
#include <string>
#include <vector>

namespace bhs::denoise {

bool compiled();  // was it compiled with BHS_OPTIX?
// Creates the context + the denoiser (HDR model) for images up to w x h; optional albedo guide. Idempotent.
bool init(int w, int h, bool albedo, std::string& err);
// Denoise an RGBA32F (float4) w x h image in device memory (no input/output overlap).
// albedo: float4 RGBA32F or nullptr (only if init was done with albedo = true). stream: cudaStream_t (0 = default).
bool run(const void* in, const void* albedo, void* out, int w, int h, void* stream, std::string& err);
const char* version();  // "OptiX 9.1.0 (ABI 118)" or "unavailable"
void shutdown();

// List of the CUDA kernels launched (name, number of launches, total time) between begin and end, via CUPTI (if available): used to
// SEE which kernels OptiX uses (names like "xmma"/"cutlass"/"tensorop" indicate tensor-core kernels).
struct KernelStat { std::string name; unsigned count = 0; double total_us = 0; };
bool kernel_trace_available();
void kernel_trace_begin();
std::vector<KernelStat> kernel_trace_end();

}  // namespace bhs::denoise
