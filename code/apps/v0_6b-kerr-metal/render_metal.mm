// Metal backend (Apple Silicon) for the bhs::kcuda interface (render.hpp).
// Compiles kerr_metal.metal at run time via Metal.framework (without depending on the external MetalToolchain)
// and uses Unified Memory (MTLResourceStorageModeShared) for zero-copy access between CPU and GPU.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <OpenGL/gl3.h>

#include <algorithm>
#include <bhs/exr_loader.hpp>
#include <bhs/grmhd.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lut.hpp>
#include <bhs/novikov_thorne.hpp>
#include <bhs/skymap.hpp>
#include <bhs/synchrotron.hpp>
#include <bhs/torus.hpp>
#include <bhs/turbulence.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "denoise.hpp"
#include "hud_font.hpp"
#include "metal_shared.hpp"
#include "render.hpp"

namespace bhs::kcuda {
namespace {

constexpr int kBloomLevels = 6;

inline uint16_t f2h(float f) {
  __fp16 h = static_cast<__fp16>(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}
inline float h2f(uint16_t u) {
  __fp16 h;
  std::memcpy(&h, &u, 2);
  return static_cast<float>(h);
}
inline f32_2 df_from_d(double d) {
  float hi = static_cast<float>(d);
  float lo = static_cast<float>(d - static_cast<double>(hi));
  return f32_2{hi, lo};
}

struct MetalState {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pso_cache;
  id<MTLComputePipelineState> trace_pso[3][4] = {};  // [prec 0..2][model 0..3 (-1->0, 0->1, 1->2, 2->3)]

  id<MTLBuffer> lut = nil;
  std::vector<float> lutd, lut1;
  double lut_fcol = 1.0;

  id<MTLBuffer> nt_tab = nil;
  int nt_n = 0;
  double nt_a = -1, nt_rin = 0, nt_rout = 0;

  id<MTLTexture> sky = nil;
  id<MTLTexture> dummy2d = nil;
  id<MTLTexture> dummy3d = nil;
  bool has_sky = false;

  id<MTLTexture> grm_tex[3] = {nil, nil, nil};
  id<MTLTexture> grm_theta_tex = nil;
  grmhd::Header grm_h;
  bool grm_loaded = false;

  id<MTLBuffer> out = nil;
  id<MTLBuffer> out16 = nil;
  id<MTLBuffer> accum = nil;
  id<MTLBuffer> hdr = nil;
  id<MTLBuffer> alb = nil;
  id<MTLBuffer> alb_avg = nil;
  id<MTLBuffer> lv[kBloomLevels] = {};
  id<MTLBuffer> tmp = nil;
  id<MTLBuffer> g0 = nil;
  id<MTLBuffer> g1 = nil;
  id<MTLBuffer> stats = nil;
  id<MTLBuffer> hud_buf = nil;
  id<MTLBuffer> font_buf = nil;

  int lw[kBloomLevels] = {}, lh[kBloomLevels] = {};
  int glw = 0, glh = 0;
  int samples = 0;
  int w = 0, h = 0, cw = 0, ch = 0;
  int hud_cap = 0;
  Timing last{};

  unsigned gl_tex = 0;
  int gw = 0, gh = 0;
};

MetalState S;

id<MTLBuffer> make_buf(size_t bytes) {
  return [S.dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
}

id<MTLComputePipelineState> get_pso(const std::string& name) {
  auto it = S.pso_cache.find(name);
  if (it != S.pso_cache.end()) return it->second;
  NSError* err = nil;
  MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
  int32_t pval = 1, mval = -1;
  [cv setConstantValue:&pval type:MTLDataTypeInt atIndex:0];
  [cv setConstantValue:&mval type:MTLDataTypeInt atIndex:1];
  id<MTLFunction> fn = [S.lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()] constantValues:cv error:&err];
  if (!fn) {
    std::fprintf(stderr, "Metal: function '%s' not found: %s\n", name.c_str(), err ? [[err localizedDescription] UTF8String] : "");
    std::exit(2);
  }
  id<MTLComputePipelineState> pso = [S.dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    std::fprintf(stderr, "Metal PSO %s: %s\n", name.c_str(), [[err localizedDescription] UTF8String]);
    std::exit(2);
  }
  S.pso_cache[name] = pso;
  return pso;
}

id<MTLComputePipelineState> get_trace_pso(int prec, int model) {
  int mi = model + 1;  // -1->0, 0->1, 1->2, 2->3
  if (S.trace_pso[prec][mi]) return S.trace_pso[prec][mi];
  MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
  int32_t pval = prec, mval = model;
  [cv setConstantValue:&pval type:MTLDataTypeInt atIndex:0];
  [cv setConstantValue:&mval type:MTLDataTypeInt atIndex:1];
  NSError* err = nil;
  id<MTLFunction> fn = [S.lib newFunctionWithName:@"kernel_trace" constantValues:cv error:&err];
  if (!fn) {
    std::fprintf(stderr, "Metal fn kernel_trace(%d,%d): %s\n", prec, model, [[err localizedDescription] UTF8String]);
    std::exit(2);
  }
  id<MTLComputePipelineState> pso = [S.dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    std::fprintf(stderr, "Metal PSO kernel_trace(%d,%d): %s\n", prec, model, [[err localizedDescription] UTF8String]);
    std::exit(2);
  }
  S.trace_pso[prec][mi] = pso;
  return pso;
}

void ensure_nt(const Params& p, MetalKP& k) {
  const double rout_prof = std::max(2.0 * p.rout, 100.0);
  if (!S.nt_tab || std::abs(p.a - S.nt_a) > 1e-9 || std::abs(rout_prof - S.nt_rout) > 1e-9) {
    const nt::Profile prof(p.a, rout_prof);
    std::vector<float> tab(prof.table().begin(), prof.table().end());
    if (!S.nt_tab) S.nt_tab = make_buf(2048 * sizeof(float));
    std::memcpy([S.nt_tab contents], tab.data(), tab.size() * sizeof(float));
    S.nt_n = int(tab.size());
    S.nt_a = p.a;
    S.nt_rout = rout_prof;
    S.nt_rin = prof.r_in();
  }
  k.nt_n = S.nt_n;
  k.nt_rin = float(S.nt_rin);
  k.nt_rout = float(S.nt_rout);
}

MetalKP make_kp(const Params& p) {
  MetalKP k{};
  k.w = p.w; k.h = p.h;
  k.use_sky = p.use_sky && S.has_sky ? 1 : 0;
  k.sky_raw = p.sky_raw ? 1 : 0;
  k.nodisk = p.nodisk ? 1 : 0;
  k.nodoppler = p.nodoppler ? 1 : 0;
  k.max_steps = p.max_steps;
  k.prec = int(p.prec);
  k.a = float(p.a);
  k.dist = float(p.dist);
  const double th_rad = p.theta_deg * M_PI / 180.0;
  const double ph_rad = p.phi_deg * M_PI / 180.0;
  k.theta = float(th_rad);
  k.phi = float(ph_rad);
  k.tan_half = float(std::tan(0.5 * p.fov_deg * M_PI / 180.0));
  k.aspect = float(double(p.w) / p.h);
  const double rin_v = p.rin < 0 ? kerr::isco(p.a, true) : p.rin;
  k.rin = float(rin_v);
  k.rout = float(p.rout);

  if (p.fcol != S.lut_fcol) {
    S.lutd = lut::build(p.fcol);
    S.lut_fcol = p.fcol;
    std::memcpy([S.lut contents], S.lutd.data(), S.lutd.size() * sizeof(float));
  }
  const Vec3 ref = lut::lookup(p.expose_physical ? S.lut1 : S.lutd, p.tpeak);
  k.exposure = float(p.exposure * 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z));
  k.sky_exposure = float(p.sky_exposure);
  k.phi_atol = float(p.phi_atol);
  const double lmin_v = p.l_min >= 0 ? p.l_min : 1e-4;
  k.l_min = float(lmin_v);
  k.tol = float(p.tol);
  k.r_far = float(p.r_far);
  k.tpeak = float(p.tpeak);
  auto f = [&](double x) { return std::pow(x, -3.0) * (1.0 - std::sqrt(rin_v / x)); };
  k.fnorm = float(f(49.0 / 36.0 * rin_v));
  k.nt = p.nt ? 1 : 0;
  k.ldr = p.accum_ldr ? 1 : 0;
  k.seed = p.seed;
  k.bx = float(p.beta[0]); k.by = float(p.beta[1]); k.bz = float(p.beta[2]);
  k.clamp_lum = float(p.clamp_lum);
  k.beaming = float(p.beaming);
  k.limb = p.limb ? 1 : 0;
  k.edge_soft = float(p.edge_soft);
  k.edge_noise = float(p.edge_noise);
  {
    turb::Spot sp;
    sp.r = p.spot_r; sp.phi0 = p.spot_phi_deg * M_PI / 180.0; sp.amp = p.spot_amp;
    sp.fl_t = p.flare_t; sp.fl_w = p.flare_w; sp.fl_boost = p.flare_boost;
    k.spot_ph = float(std::fmod(turb::spot_phase(sp, p.a, p.disk_time), 2.0 * M_PI));
    k.spot_amp = float(p.spot_amp);
    k.spot_om = float(turb::omega(p.spot_r, p.a));
    k.fl_t = float(p.flare_t); k.fl_w = float(p.flare_w); k.fl_boost = float(p.flare_boost);
    k.slow = p.slowlight ? 1 : 0;
    k.spot_r = float(p.spot_r); k.spot_size = float(p.spot_size);
    k.turb_amp = float(p.turb); k.turb_scale = float(p.turb_scale); k.disk_time = float(p.disk_time);
  }
  k.emis = p.emission;
  k.the_min = float(p.theta_e_min);
  k.model = p.model;
  k.nu_obs = float(p.nu_ghz * 1e9);
  const double mcm_v = p.mbh * 1.476625e5;
  k.mcm = float(mcm_v);
  k.tb_lo = float(p.tb_lo); k.tb_hi = float(p.tb_hi);
  k.ne0 = float(p.ne0); k.pbeta = float(p.plasma_beta); k.ti_te = float(p.ti_te);
  k.th_e_max = float(p.theta_e_max); k.t_step = float(p.torus_step);
  k.t_gamma = float(p.torus_gamma); k.t_rin = float(p.torus_rin);
  if (p.model == 1) {
    torus::Torus tt;
    if (torus::make(p.a, p.torus_rin, p.torus_rmax, p.torus_gamma, tt)) {
      k.t_ell = float(tt.ell); k.t_utin = float(tt.ut_in); k.t_hmax = float(tt.h_max);
    } else {
      k.t_ell = 0; k.t_utin = 1; k.t_hmax = 1.0000001f;
    }
  }
  k.t_rmax = float(p.torus_rmax);
  k.rcheck = p.debug_null ? 1 : 0;
  k.has_grmhd = (p.model == 2 && S.grm_loaded) ? 1 : 0;
  k.has_theta_lut = (p.theta_lut && S.grm_theta_tex != nil) ? 1 : 0;
  double eff_a = p.a;
  if (p.model == 2 && S.grm_loaded) {
    const auto& h = S.grm_h;
    eff_a = h.a;
    k.a = float(h.a);
    k.g_n1 = int(h.n1); k.g_n2 = int(h.n2); k.g_n3 = int(h.n3);
    k.g_startx1 = float(h.startx1); k.g_dx1 = float(h.dx1);
    k.g_hslope = float(h.hslope); k.g_ms = float(h.mks_smooth);
    k.g_palpha = float(h.poly_alpha); k.g_pxt = float(h.poly_xt);
    k.g_pnorm = float(h.fmks().poly_norm()); k.g_gam = float(h.gam);
    double rho_unit = p.munit / (mcm_v * mcm_v * mcm_v);
    k.g_rho_unit = float(rho_unit);
    k.g_b_unit = float(synch::kC * std::sqrt(4.0 * M_PI * rho_unit));
    k.g_step = float(p.grmhd_step); k.el_model = p.electrons;
    k.g_rhigh = float(p.rhigh); k.g_ce = float(p.th_e_ratio); k.g_rcap = 60.0f;
  }
  k.vol = (p.hr > 0.0 || p.model >= 1) ? 1 : 0;
  k.v_hr = float(p.hr); k.v_flare = float(p.flare); k.v_tau = float(p.tau);
  k.v_sub = std::max(1, p.vol_sub);
  for (int i = 0; i < 3; ++i) {
    k.fwd[i] = float(p.fwd[i]); k.right[i] = float(p.right[i]); k.up[i] = float(p.up[i]);
    k.fwd_df[i] = df_from_d(p.fwd[i]); k.right_df[i] = df_from_d(p.right[i]); k.up_df[i] = df_from_d(p.up[i]);
    k.beta_df[i] = df_from_d(p.beta[i]);
  }
  k.a_df = df_from_d(eff_a);
  k.dist_df = df_from_d(p.dist);
  k.theta_df = df_from_d(th_rad);
  k.theta_sin_df = df_from_d(std::sin(th_rad));
  k.theta_cos_df = df_from_d(std::cos(th_rad));
  k.rin_df = df_from_d(rin_v);
  k.rout_df = df_from_d(p.rout);
  k.r_far_df = df_from_d(p.r_far);
  k.tol_df = df_from_d(p.tol);
  k.phi_atol_df = df_from_d(p.phi_atol);
  k.l_min_df = df_from_d(lmin_v);

  if (p.nt) ensure_nt(p, k);
  return k;
}

std::string load_metal_source(std::string& err) {
  auto read_str = [&](const std::string& path, std::string& out) -> bool {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
  };
  std::string hdr, src;
  const std::string dir = BHS_METAL_SHADER_DIR;
  if (!read_str(dir + "/metal_shared.hpp", hdr)) { err = "could not open metal_shared.hpp"; return ""; }
  if (!read_str(dir + "/kerr_metal.metal", src)) { err = "could not open kerr_metal.metal"; return ""; }
  const std::string inc = "#include \"metal_shared.hpp\"";
  size_t pos = src.find(inc);
  if (pos != std::string::npos) src.replace(pos, inc.size(), hdr);
  return src;
}

}  // namespace

bool init(const std::string& sky_path, std::string& err) {
  if (!S.dev) {
    S.dev = MTLCreateSystemDefaultDevice();
    if (!S.dev) { err = "Metal device unavailable"; return false; }
    S.queue = [S.dev newCommandQueue];
    std::string src = load_metal_source(err);
    if (src.empty()) return false;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = NO;
    NSError* ns_err = nil;
    S.lib = [S.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opts error:&ns_err];
    if (!S.lib) {
      err = ns_err ? [[ns_err localizedDescription] UTF8String] : "error compiling kerr_metal.metal";
      return false;
    }

    S.lutd = lut::build();
    S.lut1 = S.lutd;
    S.lut_fcol = 1.0;
    S.lut = make_buf(S.lutd.size() * sizeof(float));
    std::memcpy([S.lut contents], S.lutd.data(), S.lutd.size() * sizeof(float));
    S.nt_tab = make_buf(2048 * sizeof(float));
    S.stats = make_buf(3 * sizeof(uint32_t));
    {
      uint8_t f[95 * 13];
      for (int i = 0; i < 95; ++i)
        for (int j = 0; j < 13; ++j) f[i * 13 + j] = hud::kFont[i][j];
      S.font_buf = make_buf(sizeof(f));
      std::memcpy([S.font_buf contents], f, sizeof(f));
    }

    MTLTextureDescriptor* td2 = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:1 height:1 mipmapped:NO];
    S.dummy2d = [S.dev newTextureWithDescriptor:td2];
    MTLTextureDescriptor* td3 = [[MTLTextureDescriptor alloc] init];
    td3.textureType = MTLTextureType3D;
    td3.pixelFormat = MTLPixelFormatRGBA32Float;
    td3.width = 1; td3.height = 1; td3.depth = 1;
    S.dummy3d = [S.dev newTextureWithDescriptor:td3];
  }

  if (sky_path.empty()) return true;
  sky::Skymap m;
  if (!sky::load_exr_half(sky_path, m, &err)) return false;
  int levels = 1;
  for (int n = std::max(m.w, m.h); n > 1; n >>= 1) ++levels;

  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:m.w height:m.h mipmapped:YES];
  td.mipmapLevelCount = levels;
  td.usage = MTLTextureUsageShaderRead;
  S.sky = [S.dev newTextureWithDescriptor:td];

  std::vector<uint16_t> cur(size_t(m.w) * m.h * 4), nxt;
  for (size_t i = 0; i < size_t(m.w) * m.h; ++i) {
    cur[4 * i] = m.rgb[3 * i];
    cur[4 * i + 1] = m.rgb[3 * i + 1];
    cur[4 * i + 2] = m.rgb[3 * i + 2];
    cur[4 * i + 3] = f2h(1.0f);
  }
  m.rgb.clear(); m.rgb.shrink_to_fit();

  int cw = m.w, ch = m.h;
  for (int l = 0; l < levels; ++l) {
    [S.sky replaceRegion:MTLRegionMake2D(0, 0, cw, ch) mipmapLevel:l withBytes:cur.data() bytesPerRow:size_t(cw) * 8];
    if (l + 1 == levels) break;
    const int nw = std::max(1, cw / 2), nh = std::max(1, ch / 2);
    nxt.assign(size_t(nw) * nh * 4, 0);
    for (int y = 0; y < nh; ++y)
      for (int x = 0; x < nw; ++x)
        for (int c = 0; c < 4; ++c) {
          float s = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
              s += h2f(cur[(size_t(std::min(ch - 1, 2 * y + dy)) * cw + std::min(cw - 1, 2 * x + dx)) * 4 + c]);
          nxt[(size_t(y) * nw + x) * 4 + c] = f2h(0.25f * s);
        }
    cur.swap(nxt); cw = nw; ch = nh;
  }
  S.has_sky = true;
  std::printf("sky (Metal): %s (%dx%d, %d levels, aniso 16x)\n", sky_path.c_str(), m.w, m.h, levels);
  return true;
}

bool has_sky() { return S.has_sky; }
bool denoise_available() { return denoise::compiled(); }
const char* denoise_version() { return denoise::version(); }
bool denoise_enable(const Params&, std::string& err) {
  err = "OptiX not available on the Metal backend (macOS)";
  return false;
}
bool denoise_image(const std::vector<float>&, int, int, std::vector<float>&, std::string& err) {
  err = "OptiX not available on the Metal backend (macOS)";
  return false;
}

void download_hdr(std::vector<float>& rgba) {
  rgba.resize(size_t(S.w) * S.h * 4);
  std::memcpy(rgba.data(), [S.hdr contents], rgba.size() * sizeof(float));
}

void set_size(int w, int h) {
  auto dims = [&] {
    S.w = w; S.h = h;
    int lw = w, lh = h;
    for (int l = 0; l < kBloomLevels; ++l) {
      lw = std::max(1, (lw + 1) / 2); lh = std::max(1, (lh + 1) / 2);
      S.lw[l] = lw; S.lh[l] = lh;
    }
    S.glw = std::max(1, (w + 1) / 2); S.glh = std::max(1, (h + 1) / 2);
  };
  if (S.accum && w <= S.cw && h <= S.ch) { dims(); return; }
  S.cw = std::max(w, S.cw); S.ch = std::max(h, S.ch);
  const size_t n = size_t(S.cw) * S.ch;
  S.out = make_buf(n * 4);
  S.out16 = make_buf(n * 8);
  S.accum = make_buf(n * 16);
  S.hdr = make_buf(n * 16);
  S.alb = make_buf(n * 16);
  S.alb_avg = make_buf(n * 16);
  std::memset([S.out contents], 0, n * 4);
  std::memset([S.out16 contents], 0, n * 8);
  std::memset([S.accum contents], 0, n * 16);
  std::memset([S.alb contents], 0, n * 16);

  int lw = S.cw, lh = S.ch;
  size_t tmpn = 0;
  for (int l = 0; l < kBloomLevels; ++l) {
    lw = std::max(1, (lw + 1) / 2); lh = std::max(1, (lh + 1) / 2);
    S.lv[l] = make_buf(size_t(lw) * lh * 16);
    tmpn = std::max(tmpn, size_t(lw) * lh);
  }
  S.tmp = make_buf(tmpn * 16);
  const size_t gn = size_t(std::max(1, (S.cw + 1) / 2)) * std::max(1, (S.ch + 1) / 2);
  S.g0 = make_buf(gn * 16);
  S.g1 = make_buf(gn * 16);
  dims();
  S.samples = 0;
}

void accum_reset() {
  if (S.accum) std::memset([S.accum contents], 0, size_t(S.w) * S.h * 16);
  if (S.alb) std::memset([S.alb contents], 0, size_t(S.w) * S.h * 16);
  S.samples = 0;
}
int accum_count() { return S.samples; }
int render_w() { return S.w; }
int render_h() { return S.h; }

float accum_sample(const Params& p, int k, Stats* stats) {
  set_size(p.w, p.h);
  MetalKP kp = make_kp(p);
  kp.sample_k = k + p.sample_offset;
  kp.has_alb = 0;
  kp.has_stats = stats ? 1 : 0;
  if (stats) std::memset([S.stats contents], 0, 3 * sizeof(uint32_t));

  const bool vol = kp.vol && !kp.nodisk && kp.rout > 0.0f;
  const int model = vol ? kp.model : -1;
  id<MTLComputePipelineState> pso = get_trace_pso(kp.prec, model);

  const int tilesx = (p.w + 15) / 16, tilesy = (p.h + 7) / 8;
  const int blocks = tilesx * tilesy;

  const auto t0 = std::chrono::steady_clock::now();
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:S.accum offset:0 atIndex:1];
    [enc setBuffer:S.alb offset:0 atIndex:2];
    [enc setBuffer:S.lut offset:0 atIndex:3];
    [enc setBuffer:S.nt_tab offset:0 atIndex:4];
    [enc setBuffer:S.stats offset:0 atIndex:5];
    [enc setTexture:(S.sky ? S.sky : S.dummy2d) atIndex:0];
    [enc setTexture:(S.grm_tex[0] ? S.grm_tex[0] : S.dummy3d) atIndex:1];
    [enc setTexture:(S.grm_tex[1] ? S.grm_tex[1] : S.dummy3d) atIndex:2];
    [enc setTexture:(S.grm_tex[2] ? S.grm_tex[2] : S.dummy3d) atIndex:3];
    [enc setTexture:(S.grm_theta_tex ? S.grm_theta_tex : S.dummy2d) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
  const float ms = float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  ++S.samples;
  if (stats) {
    const uint32_t* h = static_cast<const uint32_t*>([S.stats contents]);
    stats->steps_mean = double(h[0]) / (double(p.w) * p.h);
    stats->steps_max = int(h[2]);
    stats->warp_eff = h[1] ? double(h[0]) / double(h[1]) : 1.0;
  }
  S.last.trace_ms = ms;
  return ms;
}

float resolve(const Params& p) {
  set_size(p.w, p.h);
  const int w = S.w, h = S.h, n = w * h;
  const auto t0 = std::chrono::steady_clock::now();
  double bloom_ms = 0.0;

  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    {
      struct { int n; float inv; } avg{n, 1.0f / float(std::max(1, S.samples))};
      [enc setComputePipelineState:get_pso("k_avg")];
      [enc setBuffer:S.accum offset:0 atIndex:0];
      [enc setBuffer:S.hdr offset:0 atIndex:1];
      [enc setBytes:&avg length:sizeof(avg) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    const bool bloom = p.bloom > 0;
    if (bloom || p.glare > 0) {
      id<MTLBuffer> src = S.hdr;
      int sw = w, sh = h;
      for (int l = 0; l < kBloomLevels; ++l) {
        const int dw = S.lw[l], dh = S.lh[l];
        struct { int sw, sh, dw, dh; } da{sw, sh, dw, dh};
        [enc setComputePipelineState:get_pso("k_down")];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:S.lv[l] offset:0 atIndex:1];
        [enc setBytes:&da length:sizeof(da) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(dw, dh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

        struct { int w, h, dx, dy; } ba1{dw, dh, 1, 0}, ba2{dw, dh, 0, 1};
        [enc setComputePipelineState:get_pso("k_blur")];
        [enc setBuffer:S.lv[l] offset:0 atIndex:0];
        [enc setBuffer:S.tmp offset:0 atIndex:1];
        [enc setBytes:&ba1 length:sizeof(ba1) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(dw, dh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

        [enc setBuffer:S.tmp offset:0 atIndex:0];
        [enc setBuffer:S.lv[l] offset:0 atIndex:1];
        [enc setBytes:&ba2 length:sizeof(ba2) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(dw, dh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        src = S.lv[l]; sw = dw; sh = dh;
      }
    }

    if (p.glare > 0) {
      struct { int w, h, gw, gh, cw, ch; float thr; } br{w, h, S.glw, S.glh, S.lw[2], S.lh[2], float(p.glare_thr)};
      [enc setComputePipelineState:get_pso("k_bright")];
      [enc setBuffer:S.hdr offset:0 atIndex:0];
      [enc setBuffer:S.lv[2] offset:0 atIndex:1];
      [enc setBuffer:S.g0 offset:0 atIndex:2];
      [enc setBytes:&br length:sizeof(br) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(S.glw, S.glh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

      struct { int gw, gh, taps; float tau; } st{S.glw, S.glh, int(3 * p.glare_len), float(p.glare_len)};
      [enc setComputePipelineState:get_pso("k_streaks")];
      [enc setBuffer:S.g0 offset:0 atIndex:0];
      [enc setBuffer:S.g1 offset:0 atIndex:1];
      [enc setBytes:&st length:sizeof(st) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(S.glw, S.glh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }

    MetalPolish po{};
    po.glare = float(p.glare); po.grain = float(p.grain); po.vignette = float(p.vignette); po.chroma = float(p.chroma);
    po.frame = p.frame_index; po.seed = 0xA5F1u; po.has_g1 = p.glare > 0 ? 1 : 0;
    po.gw = S.glw; po.gh = S.glh; po.bloom = float(p.bloom);
    po.levels = bloom ? kBloomLevels : 0;
    po.ldr = p.accum_ldr ? 1 : 0; po.tonemap = int(p.tonemap); po.w = w; po.h = h;
    for (int l = 0; l < kBloomLevels; ++l) { po.lw[l] = S.lw[l]; po.lh[l] = S.lh[l]; }

    [enc setComputePipelineState:get_pso("k_final")];
    [enc setBuffer:S.hdr offset:0 atIndex:0];
    [enc setBuffer:S.out offset:0 atIndex:1];
    [enc setBuffer:S.out16 offset:0 atIndex:2];
    [enc setBytes:&po length:sizeof(po) atIndex:3];
    for (int l = 0; l < kBloomLevels; ++l) [enc setBuffer:S.lv[l] offset:0 atIndex:4 + l];
    [enc setBuffer:S.g1 offset:0 atIndex:10];
    [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }

  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  S.last.denoise_ms = 0.0f;
  S.last.bloom_ms = float(bloom_ms);
  S.last.resolve_ms = float(ms);
  return float(ms);
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

void hud_draw(const std::string& text, int scale) {
  if (!S.out || text.empty()) return;
  std::vector<std::string> lines;
  size_t pos = 0, cols = 1;
  while (pos <= text.size()) {
    const size_t nl = text.find('\n', pos);
    std::string ln = text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    cols = std::max(cols, ln.size());
    lines.push_back(ln);
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }
  const int rows = int(lines.size());
  std::string grid(size_t(cols) * rows, ' ');
  for (int r = 0; r < rows; ++r)
    for (size_t c = 0; c < lines[r].size(); ++c) grid[size_t(r) * cols + c] = lines[r][c];
  const int need = int(grid.size());
  if (need > S.hud_cap) { S.hud_buf = make_buf(need); S.hud_cap = need; }
  std::memcpy([S.hud_buf contents], grid.data(), need);
  scale = std::max(1, scale);
  const int pad = 4 * scale, rw = int(cols) * hud::kGlyphW * scale + 2 * pad, rh = rows * hud::kGlyphH * scale + 2 * pad;
  struct { int w, h, cols, rows, scale, x0, y0, pad, rw, rh; } ha{S.w, S.h, int(cols), rows, scale, 12 * scale, 12 * scale, pad, rw, rh};
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("k_hud")];
    [enc setBuffer:S.out offset:0 atIndex:0];
    [enc setBuffer:S.hud_buf offset:0 atIndex:1];
    [enc setBuffer:S.font_buf offset:0 atIndex:2];
    [enc setBytes:&ha length:sizeof(ha) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(rw, rh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
}

void download_rgb16(std::vector<uint16_t>& rgb) {
  const size_t n = size_t(S.w) * S.h;
  const uint16_t* px = static_cast<const uint16_t*>([S.out16 contents]);
  rgb.resize(n * 3);
  for (size_t i = 0; i < n; ++i) {
    rgb[3 * i] = px[4 * i];
    rgb[3 * i + 1] = px[4 * i + 1];
    rgb[3 * i + 2] = px[4 * i + 2];
  }
}

void download_rgb(std::vector<uint8_t>& rgb) {
  const size_t n = size_t(S.w) * S.h;
  const uint8_t* px = static_cast<const uint8_t*>([S.out contents]);
  rgb.resize(n * 3);
  for (size_t i = 0; i < n; ++i) {
    rgb[3 * i] = px[4 * i];
    rgb[3 * i + 1] = px[4 * i + 1];
    rgb[3 * i + 2] = px[4 * i + 2];
  }
}

bool gl_register(unsigned tex, int w, int h, std::string&) {
  S.gl_tex = tex; S.gw = w; S.gh = h;
  return true;
}
void gl_copy() {
  if (!S.gl_tex || !S.out) return;
  glBindTexture(GL_TEXTURE_2D, S.gl_tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, S.w, S.h, GL_RGBA, GL_UNSIGNED_BYTE, [S.out contents]);
}
void gl_unregister() { S.gl_tex = 0; }

// ------------------------------------------------------------------ Test / batch hooks
void camera_rays(const Params& p, const std::vector<double>& xy, std::vector<double>& out) {
  const int n = int(xy.size() / 2);
  MetalKP kp = make_kp(p);
  std::vector<float> h_xy(xy.size());
  for (size_t i = 0; i < xy.size(); ++i) h_xy[i] = float(xy[i]);
  id<MTLBuffer> bin = make_buf(h_xy.size() * sizeof(float)), bout = make_buf(n * 3 * sizeof(float));
  std::memcpy([bin contents], h_xy.data(), h_xy.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("cam_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(size_t(n) * 3);
  for (size_t i = 0; i < out.size(); ++i) out[i] = ho[i];
}

bool accum_finite() {
  const size_t n = size_t(S.w) * S.h;
  const float* a = static_cast<const float*>([S.accum contents]);
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(a[4 * i]) || !std::isfinite(a[4 * i + 1]) || !std::isfinite(a[4 * i + 2])) return false;
  return true;
}

void nt_lookup(const Params& p, const std::vector<double>& r, std::vector<double>& out) {
  Params q = p; q.nt = true;
  MetalKP kp = make_kp(q);
  const int n = int(r.size());
  std::vector<float> hr(n);
  for (int i = 0; i < n; ++i) hr[i] = float(r[i]);
  id<MTLBuffer> bin = make_buf(n * sizeof(float)), bout = make_buf(n * sizeof(float));
  std::memcpy([bin contents], hr.data(), n * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("nt_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:S.nt_tab offset:0 atIndex:2];
    [enc setBuffer:bout offset:0 atIndex:3];
    [enc setBytes:&n length:sizeof(n) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = ho[i];
}

void boost_batch(const std::vector<double>& np, const double beta[3], std::vector<double>& out) {
  const int n = int(np.size() / 3);
  std::vector<float> hnp(np.size());
  for (size_t i = 0; i < np.size(); ++i) hnp[i] = float(np[i]);
  float b3[4] = {float(beta[0]), float(beta[1]), float(beta[2]), 0.0f};
  id<MTLBuffer> bin = make_buf(hnp.size() * sizeof(float)), bout = make_buf(n * 4 * sizeof(float));
  std::memcpy([bin contents], hnp.data(), hnp.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("boost_kernel")];
    [enc setBuffer:bin offset:0 atIndex:0];
    [enc setBytes:b3 length:sizeof(b3) atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(size_t(n) * 4);
  for (size_t i = 0; i < out.size(); ++i) out[i] = ho[i];
}

void trace_batch(const Params& p, const std::vector<double>& rays, std::vector<RayOut>& out, int skip) {
  const int n = int(rays.size() / 3);
  MetalKP kp = make_kp(p);
  std::vector<float> hr(rays.size());
  for (size_t i = 0; i < rays.size(); ++i) hr[i] = float(rays[i]);
  id<MTLBuffer> bin = make_buf(hr.size() * sizeof(float)), bout = make_buf(n * sizeof(MetalTrace));
  std::memcpy([bin contents], hr.data(), hr.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("batch_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&skip length:sizeof(skip) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const MetalTrace* h = static_cast<const MetalTrace*>([bout contents]);
  out.resize(n);
  for (int i = 0; i < n; ++i)
    out[i] = RayOut{h[i].fate, h[i].disk_hit, h[i].steps, h[i].mu_end, h[i].phi_end, h[i].disk_r, h[i].disk_phi, h[i].L, h[i].Q};
}

void vol_flight_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out) {
  const int n = int(rays.size() / 3);
  MetalKP kp = make_kp(p); kp.slow = 1;
  std::vector<float> hr(rays.size());
  for (size_t i = 0; i < rays.size(); ++i) hr[i] = float(rays[i]);
  id<MTLBuffer> bin = make_buf(hr.size() * sizeof(float)), bout = make_buf(n * sizeof(float));
  std::memcpy([bin contents], hr.data(), hr.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("vol_flight_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:S.lut offset:0 atIndex:2];
    [enc setBuffer:S.nt_tab offset:0 atIndex:3];
    [enc setBuffer:bout offset:0 atIndex:4];
    [enc setBytes:&n length:sizeof(n) atIndex:5];
    [enc setTexture:(S.grm_tex[0] ? S.grm_tex[0] : S.dummy3d) atIndex:0];
    [enc setTexture:(S.grm_tex[1] ? S.grm_tex[1] : S.dummy3d) atIndex:1];
    [enc setTexture:(S.grm_tex[2] ? S.grm_tex[2] : S.dummy3d) atIndex:2];
    [enc setTexture:(S.grm_theta_tex ? S.grm_theta_tex : S.dummy2d) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = ho[i];
}

void torus_batch(const Params& p, const std::vector<double>& rmuL, std::vector<TorusPoint>& out, bool) {
  const int n = int(rmuL.size() / 3);
  MetalKP kp = make_kp(p);
  std::vector<float> hin(rmuL.size());
  for (size_t i = 0; i < rmuL.size(); ++i) hin[i] = float(rmuL[i]);
  id<MTLBuffer> bin = make_buf(hin.size() * sizeof(float)), bout = make_buf(n * sizeof(MetalTorusPoint));
  std::memcpy([bin contents], hin.data(), hin.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("torus_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const MetalTorusPoint* ho = static_cast<const MetalTorusPoint*>([bout contents]);
  out.resize(n);
  for (int i = 0; i < n; ++i)
    out[i] = TorusPoint{ho[i].rho, ho[i].thE, ho[i].B, ho[i].sinB, ho[i].j, ho[i].alpha, ho[i].g, ho[i].e_emit};
}

void turb_batch(const Params& p, const std::vector<double>& rpmt, std::vector<double>& out) {
  const int n = int(rpmt.size() / 3);
  MetalKP kp = make_kp(p);
  std::vector<float> hin(rpmt.size());
  for (size_t i = 0; i < rpmt.size(); ++i) hin[i] = float(rpmt[i]);
  id<MTLBuffer> bin = make_buf(hin.size() * sizeof(float)), bout = make_buf(n * 2 * sizeof(float));
  std::memcpy([bin contents], hin.data(), hin.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("turb_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(size_t(n) * 2);
  for (size_t i = 0; i < out.size(); ++i) out[i] = ho[i];
}

void emitter_mu_batch(double a, const std::vector<double>& rLQ, std::vector<double>& out) {
  const int n = int(rLQ.size() / 3);
  float af = float(a);
  std::vector<float> hin(rLQ.size());
  for (size_t i = 0; i < rLQ.size(); ++i) hin[i] = float(rLQ[i]);
  id<MTLBuffer> bin = make_buf(hin.size() * sizeof(float)), bout = make_buf(n * sizeof(float));
  std::memcpy([bin contents], hin.data(), hin.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("mu_kernel")];
    [enc setBytes:&af length:sizeof(af) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = ho[i];
}

bool grmhd_load(const std::string& path, std::string& err) {
  grmhd::Field f;
  if (!grmhd::load_bgm(path, f, err)) return false;
  const auto& h = f.h;
  std::vector<float> stage(size_t(h.n1) * h.n2 * h.n3 * 4);
  for (int t = 0; t < 3; ++t) {
    for (size_t c = 0; c < size_t(h.n1) * h.n2 * h.n3; ++c) {
      const float* q = &f.data[c * grmhd::kNF + 4 * t];
      stage[4 * c] = q[0]; stage[4 * c + 1] = q[1]; stage[4 * c + 2] = q[2]; stage[4 * c + 3] = q[3];
    }
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType3D;
    td.pixelFormat = MTLPixelFormatRGBA32Float;
    td.width = h.n3; td.height = h.n2; td.depth = h.n1;
    td.usage = MTLTextureUsageShaderRead;
    S.grm_tex[t] = [S.dev newTextureWithDescriptor:td];
    [S.grm_tex[t] replaceRegion:MTLRegionMake3D(0, 0, 0, h.n3, h.n2, h.n1) mipmapLevel:0 slice:0
                       withBytes:stage.data() bytesPerRow:h.n3 * 16 bytesPerImage:size_t(h.n3) * h.n2 * 16];
  }
  const int ln1 = 256, ln2 = 512;
  std::vector<float> ltab(size_t(ln1) * ln2);
  const auto fm = h.fmks();
  for (int i = 0; i < ln1; ++i) {
    const double X1 = h.startx1 + (double(i) + 0.5) / ln1 * h.n1 * h.dx1;
    for (int j = 0; j < ln2; ++j) {
      double X2 = 0;
      fm.inv_theta(X1, (double(j) + 0.5) / ln2 * M_PI, X2);
      ltab[size_t(j) * ln1 + i] = float(X2);
    }
  }
  MTLTextureDescriptor* ltd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:ln1 height:ln2 mipmapped:NO];
  ltd.usage = MTLTextureUsageShaderRead;
  S.grm_theta_tex = [S.dev newTextureWithDescriptor:ltd];
  [S.grm_theta_tex replaceRegion:MTLRegionMake2D(0, 0, ln1, ln2) mipmapLevel:0 withBytes:ltab.data() bytesPerRow:ln1 * 4];
  S.grm_h = h;
  S.grm_loaded = true;
  return true;
}

bool grmhd_loaded() { return S.grm_loaded; }
double grmhd_spin() { return S.grm_loaded ? S.grm_h.a : 0.0; }

void grmhd_sample_batch(const Params& p, const std::vector<double>& rmuphi, std::vector<float>& out) {
  const int n = int(rmuphi.size() / 3);
  MetalKP kp = make_kp(p);
  std::vector<float> hin(rmuphi.size());
  for (size_t i = 0; i < rmuphi.size(); ++i) hin[i] = float(rmuphi[i]);
  id<MTLBuffer> bin = make_buf(hin.size() * sizeof(float)), bout = make_buf(n * 12 * sizeof(float));
  std::memcpy([bin contents], hin.data(), hin.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("grm_sample_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setTexture:(S.grm_tex[0] ? S.grm_tex[0] : S.dummy3d) atIndex:0];
    [enc setTexture:(S.grm_tex[1] ? S.grm_tex[1] : S.dummy3d) atIndex:1];
    [enc setTexture:(S.grm_tex[2] ? S.grm_tex[2] : S.dummy3d) atIndex:2];
    [enc setTexture:(S.grm_theta_tex ? S.grm_theta_tex : S.dummy2d) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  out.resize(size_t(n) * 12);
  std::memcpy(out.data(), [bout contents], out.size() * sizeof(float));
}

void grmhd_map_batch(const Params& p, const std::vector<double>& rmu, std::vector<float>& out) {
  const int n = int(rmu.size() / 2);
  MetalKP kp = make_kp(p);
  std::vector<float> hin(rmu.size());
  for (size_t i = 0; i < rmu.size(); ++i) hin[i] = float(rmu[i]);
  id<MTLBuffer> bin = make_buf(hin.size() * sizeof(float)), bout = make_buf(n * 2 * sizeof(float));
  std::memcpy([bin contents], hin.data(), hin.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("grm_map_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setTexture:(S.grm_theta_tex ? S.grm_theta_tex : S.dummy2d) atIndex:0];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  out.resize(size_t(n) * 2);
  std::memcpy(out.data(), [bout contents], out.size() * sizeof(float));
}

void grmhd_null_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out) {
  const int n = int(rays.size() / 3);
  MetalKP kp = make_kp(p);
  std::vector<float> hr(rays.size());
  for (size_t i = 0; i < rays.size(); ++i) hr[i] = float(rays[i]);
  id<MTLBuffer> bin = make_buf(hr.size() * sizeof(float)), bout = make_buf(n * sizeof(float));
  std::memcpy([bin contents], hr.data(), hr.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("grm_null_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:S.lut offset:0 atIndex:2];
    [enc setBuffer:S.nt_tab offset:0 atIndex:3];
    [enc setBuffer:bout offset:0 atIndex:4];
    [enc setBytes:&n length:sizeof(n) atIndex:5];
    [enc setTexture:(S.grm_tex[0] ? S.grm_tex[0] : S.dummy3d) atIndex:0];
    [enc setTexture:(S.grm_tex[1] ? S.grm_tex[1] : S.dummy3d) atIndex:1];
    [enc setTexture:(S.grm_tex[2] ? S.grm_tex[2] : S.dummy3d) atIndex:2];
    [enc setTexture:(S.grm_theta_tex ? S.grm_theta_tex : S.dummy2d) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = ho[i];
}

void radio_image(const Params& p, int ss, std::vector<double>& I, std::vector<double>& ab) {
  MetalKP kp = make_kp(p);
  const int n = p.w * p.h;
  id<MTLBuffer> bI = make_buf(n * sizeof(float)), bab = make_buf(n * 2 * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("radio_image_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:S.lut offset:0 atIndex:1];
    [enc setBuffer:S.nt_tab offset:0 atIndex:2];
    [enc setBuffer:bI offset:0 atIndex:3];
    [enc setBuffer:bab offset:0 atIndex:4];
    [enc setBytes:&ss length:sizeof(ss) atIndex:5];
    [enc setTexture:(S.grm_tex[0] ? S.grm_tex[0] : S.dummy3d) atIndex:0];
    [enc setTexture:(S.grm_tex[1] ? S.grm_tex[1] : S.dummy3d) atIndex:1];
    [enc setTexture:(S.grm_tex[2] ? S.grm_tex[2] : S.dummy3d) atIndex:2];
    [enc setTexture:(S.grm_theta_tex ? S.grm_theta_tex : S.dummy2d) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(p.w, p.h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* hi = static_cast<const float*>([bI contents]);
  const float* ha = static_cast<const float*>([bab contents]);
  I.resize(n); ab.resize(size_t(n) * 2);
  for (int i = 0; i < n; ++i) I[i] = hi[i];
  for (size_t i = 0; i < ab.size(); ++i) ab[i] = ha[i];
}

void radio_intensity_batch(const Params& p, const std::vector<double>& rays, std::vector<double>& out) {
  const int n = int(rays.size() / 3);
  MetalKP kp = make_kp(p);
  std::vector<float> hr(rays.size());
  for (size_t i = 0; i < rays.size(); ++i) hr[i] = float(rays[i]);
  id<MTLBuffer> bin = make_buf(hr.size() * sizeof(float)), bout = make_buf(n * sizeof(float));
  std::memcpy([bin contents], hr.data(), hr.size() * sizeof(float));
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [S.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:get_pso("radio_kernel")];
    [enc setBytes:&kp length:sizeof(kp) atIndex:0];
    [enc setBuffer:bin offset:0 atIndex:1];
    [enc setBuffer:S.lut offset:0 atIndex:2];
    [enc setBuffer:S.nt_tab offset:0 atIndex:3];
    [enc setBuffer:bout offset:0 atIndex:4];
    [enc setBytes:&n length:sizeof(n) atIndex:5];
    [enc setTexture:(S.grm_tex[0] ? S.grm_tex[0] : S.dummy3d) atIndex:0];
    [enc setTexture:(S.grm_tex[1] ? S.grm_tex[1] : S.dummy3d) atIndex:1];
    [enc setTexture:(S.grm_tex[2] ? S.grm_tex[2] : S.dummy3d) atIndex:2];
    [enc setTexture:(S.grm_theta_tex ? S.grm_theta_tex : S.dummy2d) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  }
  const float* ho = static_cast<const float*>([bout contents]);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = ho[i];
}

}  // namespace bhs::kcuda
