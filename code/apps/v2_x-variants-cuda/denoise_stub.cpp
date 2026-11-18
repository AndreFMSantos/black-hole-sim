// Without OptiX: the denoiser is not available (the rest of the renderer does not depend on it).
#include "denoise.hpp"

namespace bhs::denoise {
bool compiled() { return false; }
bool init(int, int, bool, std::string& err) { err = "OptiX not available: built without BHS_OPTIX (run code/tools/fetch_optix.sh and rebuild)"; return false; }
bool run(const void*, const void*, void*, int, int, void*, std::string& err) { err = "OptiX not available"; return false; }
const char* version() { return "unavailable (built without BHS_OPTIX)"; }
void shutdown() {}
bool kernel_trace_available() { return false; }
void kernel_trace_begin() {}
std::vector<KernelStat> kernel_trace_end() { return {}; }
}  // namespace bhs::denoise
