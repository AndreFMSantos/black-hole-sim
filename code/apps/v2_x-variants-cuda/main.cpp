// v0.6b/v0.7d/v0.7e — Kerr on CUDA (RTX 5080). Same options as v0_6-kerr-cpu for direct comparison.
// See --help. Modes: --shot (PNG), --video (mp4 via ffmpeg), --selftest, and (with none of these) the interactive window.
#include "../v0_5-realtime-gpu/gl_loader.hpp"

#include <algorithm>
#include <array>
#include <bhs/camera.hpp>
#include <bhs/disk_emission.hpp>
#include <bhs/grmhd.hpp>
#include <bhs/sds.hpp>
#include <bhs/novikov_thorne_kn.hpp>
#include <bhs/wormhole.hpp>
#include <bhs/kerr.hpp>
#include <bhs/lorentz.hpp>
#include <bhs/lut.hpp>
#include <bhs/novikov_thorne.hpp>
#include <bhs/png.hpp>
#include <bhs/route.hpp>
#include <bhs/exr_loader.hpp>
#include <bhs/synchrotron.hpp>
#include <bhs/torus.hpp>
#include <bhs/turbulence.hpp>
#include <bhs/vec3.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <map>

#include "denoise.hpp"
#include "flight.hpp"
#include "render.hpp"

using namespace bhs;
using Clock = std::chrono::steady_clock;

struct Cli {
  kcuda::Params p;
  std::string sky2;
  std::string shot, sky = BHS_SKY_DEFAULT, observer = "static";
  bool nosky = false, selftest = false, vsync = false, prec_set = false, samples_set = false, cinematic = false, help = false;
  int runs = 20, autoexit = 0;
  // v0.7e: video
  std::string video, clip = "spin";
  int fps = 60, crf = 18;
  double seconds = 0;  // 0 = clip default
  double disk_rate = 0;  // v0.9c: M of disk time per second of video (0 = stopped)
  // v1.0: offline routes
  std::string route_file, route_out, workdir, codec = "h265", route_shot, grmhd_file, route_exr;
  int chunk = 60, first = 0, last = -1, route_frame = 0, crf16 = 16;
  double shutter = 0.5;
  bool resume = false;
  // v0.8b: voo
  std::string mode = "orbit", start = "orbit";
  double timescale = 5.0, move_scale = 1.0;
  bool hud = true, hud_offline = false;
  std::string hdr_out;  // --hdr out.pfm: HDR average (after the denoiser, before bloom/tone mapping)
  bool radio_cal = false, fit_ne0 = false;  // v2.2b3: radio-mode calibration report (flux in Jy, ring in uas); --fit-ne0 fits ne0 to the target flux
  double d_pc = 0, flux_target = 0, ring_ref = 0, ring_win = 8.0;  // ring_ref: observed diameter (uas); ring_win: maximum radius (M) of the ring search         // source distance (pc) and target total flux (Jy)
  std::string radio_out;                    // --radio-out f.pfm: image in Jy/uas^2
  int audit_n = 0; uint32_t audit_seed = 12345;  // --audit-float N [--audit-seed S]: float robustness audit (v2.7)
  bool selftest_denoise = false, denoise_kernels = false;  // hud: interactive window (default on); hud_offline: --hud in --shot/--video
};

struct RadioStats { double flux_jy = 0, peak_jy_uas2 = 0, peak_tb = 0, ring_diam_uas = 0, ring_diam_M = 0, uas_per_M = 0, cx = 0, cy = 0; };
static RadioStats radio_measure(const kcuda::Params& p, int ss, double D_cm, std::vector<float>* pfm = nullptr);

static void print_help() {
  std::printf(R"(v0_6b-kerr-cuda — Kerr black hole in CUDA (v0.6b) + cinematic render (v0.7d) + video (v0.7e)

Modes:
  --shot out.png      offline render (prints ms/frame per stage, steps per ray)   [--samples N] [--runs R]
  --video out.mp4     H.264 video piped to ffmpeg   --clip spin|orbit|infall|flight [--fps 60] [--seconds S] [--crf 18]
  --selftest          GPU vs CPU (kerr::trace, NT, boost, accumulation); no window
  (nothing)           interactive window: mouse = orbit, wheel = distance, [ ] spin, D Doppler, G disk, +/- exposure,
                      arrows = sky, P double/float, T reinhard/aces, B bloom, O observer, C cinematic, ESC
                      (--autoexit N exits after N frames and prints the fps)

Scene (same as v0_6-kerr-cpu):  --w --h --a 0.9 --theta 80 --phi 0 --dist 40 --fov 40 --rin <isco> --rout 16 --tpeak 4200
                      --exposure 3 --sky-exposure 2 --nodisk --nodoppler --tol 1e-8 --sky map.exr --nosky
Disk:       --profile nt|newtonian        Novikov-Thorne (2048-point LUT) by default; newtonian = v0.4-v0.6
Flight (v0.8b): --mode orbit|flight   --start orbit|zamo|infall   --timescale 5 (M of proper time per real second)   --move-scale 0.5   --nohud
            Flight: mouse (drag) look, Q/E roll, W/S forward/back, A/D sideways, Space/Shift up/down, Ctrl brake, +/- acceleration,
            , . time scale, Z pause, H HUD, Backspace reset, F orbit/flight mode (the mouse wheel only works in orbit mode)
Observer:   --observer static|orbit|infall   |  --beta br,bth,bph  (velocity in c in the ZAMO frame)
            The disk gets aberration + Doppler (g * E'/E_zamo); the sky only aberration (no colour), as on the CPU.
Precision:  --prec double|float|mixed [--tol]       (float: tol >= 2e-5; interactive: float by default)
Radio (v2.2b3): --emission dexter|leung  --theta-e-min T (default 0.3); --radio-cal --d-pc D [--flux-target Jy --fit-ne0] [--radio-out f.pfm]  total flux (Jy), peak I (Jy/uas^2), T_b and ring diameter (uas); with --fit-ne0 fits ne0 to the flux
Image:      --samples N          jittered sub-pixel samples (R2 per quad, seed --seed); HDR mean, tone mapping afterwards
            --accum hdr|ldr      hdr (default) | ldr = Reinhard PER SAMPLE before averaging (same as the CPU SSAA)
            --clamp L            per-sample luminance clamp (fireflies on the blue side of the Doppler; 0 = off)
            --tonemap reinhard|aces      reinhard by default (comparisons with the CPU); aces = Narkowicz
            --bloom s            HDR bloom (6-level Gaussian pyramid, added before tone mapping); 0 = off
            --turb a [--turb-scale 2] --spot r,phi_deg,size,amp --spot-flare t,w,boost --time T   (v0.9c, only with --hr) 3D lognormal turbulence rotating with the Kerr Omega(r) (differential rotation),
                                 hot spot orbiting at r (T -> T (1 + amp G)^(1/4)), flare = Gaussian pulse in time on top of amp; T = disk time in M (snapshot, no propagation delay);
                                 --disk-rate R advances time R M per video second; --clip disk
            --disk-model grmhd --grmhd dump.bgm [--munit 1e26 --mbh 6.5e9 --electrons const|rhigh|prop --ti-te 3 --rhigh 20 --th-e-ratio 1 --nu-ghz 230 --tb-range 8,11]
                                 (v2.2f) GRMHD MODE: a real snapshot (tools/convert_grmhd.py) in 3D textures; n_e = rho rho_unit/(m_p + m_e), B = |b| B_unit; the spin comes from the dump;
                                 electrons const: Theta_e = Theta (m_p/m_e)/(1 + Ti/Te); rhigh: Ti/Te = (R_high beta^2 + 1)/(1 + beta^2); prop: Theta_e = th-e-ratio Theta
            --disk-model slab|torus [--torus rin,rmax[,gamma] --ne0 1e5 --plasma-beta 10 --ti-te 10 --theta-e-max 0 --nu-ghz 230 --mbh 6.5e9 --tb-range 8,11]
                                 (v2.2b) RADIO MODE: Fishbone-Moncrief torus (rin < rmax; bound torus) with thermal synchrotron (Dexter 2016, valid for Theta_e > ~3) and a
                                 toroidal field of constant beta; image in brightness temperature (afmhot false colour, log10 T_b in --tb-range), no sky; ne0 in cm^-3 at the
                                 pressure maximum; --theta-e-max T sets Theta_e = T (h-1)/(h_max-1) (the analytic torus is cold: electron heating is a model parameter); the camera must stay outside the gas (--dist >= 80); frequency --nu-ghz; the gas is not an optical black body
            --route r.json --route-out out.mp4 [--shutter 0.5 --chunk 60 --resume --first F --last L --workdir D --codec h265|prores --crf16 16]
            --route r.json --route-exr DIR [--first F --last L]   linear float32 HDR frames (frame_NNNNN.exr), resumes by skipping existing ones (v1.0b)
                                 (v1.0) [--route-exr DIR: linear float32 EXR frames] OFFLINE ROUTE: ship + view + disk from a JSON file, N samples per frame (--samples, default 64) with time stratified inside the
                                 shutter (physical motion blur; --shutter 0 turns it off), 16-bit output -> 10-bit H.265 (or ProRes), in blocks of --chunk frames with
                                 resume (--resume skips finished blocks) and final concatenation without recompression; same file => same frames (fixed seed)
            --route r.json --route-frame N --route-shot f.png   a single route frame to an 8-bit PNG (preview)
            --slowlight          (v0.9c2) slow light: the disk is seen at the emission time t_obs - (flight time - T_ref) instead of t_obs; only with --hr and time (turbulence/spot/flare)
            --hr h [--tau 10 --flare 0 --vol-sub 4]  (v0.9b) THICK disk: Gaussian layer in z with H/r = h at r = 10 (H/r = h (r/10)^flare),
                                 total vertical optical depth tau; radiative transfer along the ray (without the aesthetic noise); 0 = thin disk (default)
            --edge-soft s        (v0.9a2) soft outer edge: emission decays (smoothstep) between rout(1-s) and rout(1+s); the gas becomes semi-transparent (the ray continues behind it); 0 = hard cut (default)
            --edge-noise a       (v0.9a2) irregular edge: edge radius rout(1 + a(2n-1)), n = azimuth noise (0..0.5); 0 = circle
            --fcol f             (v0.9a, physical) spectral hardening factor: I = f^-4 B(f T), same bolometric flux; 1 = black body (default), 1.5-2 = stellar disk
            --limb none|eddington  (v0.9a, physical) limb darkening I ~ 1 + 2.06 mu (mu in the gas frame); flux conserved
            --expose relative|physical  relative (default): brightness renormalised to the peak of each LUT; physical: fixed exposure (ref. f = 1), --fcol/--limb change the brightness
            --beaming s          AESTHETIC, NOT PHYSICAL: g -> g^s only in the disk intensity (s < 1 reduces the Doppler asymmetry)
            --disk-tex amp[,fr,fphi,oct]   (v2.4) filamentary texture of the THIN disk: amp = contrast (lognormal, ~0.6-1.2), fr/fphi = radial (in ln r) and azimuthal frequencies (fr >> fphi stretches in azimuth), oct octaves; with --time the pattern is sheared by Omega(r)
            --wormhole ellis[,rho] | interstellar | rho,a,M   (v2.1d) wormhole: static camera at l = --wh-l (default 8 rho) and angular position --theta/--phi; rays that end at l < 0 see the 2nd sky (--sky2 EXR; without it the 1st is repeated); no disk, sky only. --wh-lmax (100 rho): where tracing stops and the flat tail begins
            --audit-float N [--audit-seed S]   (v2.7) FLOAT robustness audit: N random scenarios (Kerr, KN, naked KN, SdS, wormhole, radio torus; camera from near the horizon to 2000 M, poles, FOV 1-170, relativistic observer, layer/turbulence/slow light/bloom/glare/HDR10) at 96x54; fails on NaN/inf/negative in the HDR
            --charge Q   (v2.1a) Kerr-Newman: charge Q (in M); Delta = r^2 - 2r + a^2 + Q^2; Q^2 + a^2 > 1 = naked singularity (rays reaching r = 0.05 come out black). Thin disk and static observer; a^2 + Q^2 < 1: Kerr-Newman Novikov-Thorne/Page-Thorne disk (own ISCO, T(r) and redshift, v2.1e); naked Kerr-Newman: Newtonian profile and --rin
            --disk-arms m,pitch,amp[,r]   (v2.6) logarithmic spiral arms (m arms, pitch in degrees, amplitude 0-1) rotating as a rigid pattern with Omega(reference r); --disk-rings amp[,freq,gap]   radial rings with gaps (1D noise in ln r); --hr-var amp[,scale]   variable thickness (>= H(r)), only in layer mode (--hr). Aesthetic
            --lambda L   (v2.1c) Schwarzschild-de Sitter: cosmological constant L (M = 1, 0 < L < 1/9; a = 0, Q = 0); rays escape at r_c (cosmological horizon); the camera must stay between the horizons; --rin required with a disk; same restrictions as --charge
            --hdr10 peak [--hdr-nits 120]   (v2.5) HDR10 output: PQ + BT.2020 at 16 bit (frames, --route-out prores/h265 with metadata), no display tone mapping; 1 unit of light = hdr-nits cd/m2, soft shoulder up to peak cd/m2
            --glare s [--glare-thr 2 --glare-cap 1.5]   star-shaped glare (6 points) above the threshold (sources above --glare-cap are excluded from the glare, 0 = no limit), added in HDR   (aesthetic)
            --grain s            luminance-dependent film grain, AFTER tone mapping (fixed seed; 0 = off)
            --vignette s         soft vignette (0..1)                                           (aesthetic)
            --chroma s           radial chromatic aberration (fraction of the radius at the corner)           (aesthetic)
            --denoise none|optix  OptiX denoiser (HDR model, tensor cores according to NVIDIA) between the HDR mean and the bloom; only if compiled
                                 with BHS_OPTIX (code/tools/fetch_optix.sh); default none; key N in the window   [--denoise-albedo: albedo guide]
            --sample-offset K    index of the 1st sample (K > 0: all jittered); --hdr out.pfm writes the HDR mean
            --cinematic          AESTHETIC PRESET (not physical): aces + bloom 0.12 + beaming 0.5 + clamp 50 + glare 0.10 + grain 0.025
                                 + vignette 0.30 + chroma 0.003 (flags after the preset override it). The default look is the physical one.
)");
}

// AESTHETIC preset (not physical): aces + bloom + beaming + clamp (v0.7d) and polish (v0.8d): glare, grain, vignette, chroma
static void apply_cinematic(kcuda::Params& p, bool on) {
  p.tonemap = on ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
  p.bloom = on ? 0.12 : 0; p.beaming = on ? 0.5 : 1.0; p.clamp_lum = on ? 50 : 0;
  p.glare = on ? 0.10 : 0; p.grain = on ? 0.025 : 0; p.vignette = on ? 0.30 : 0; p.chroma = on ? 0.003 : 0;
}

static bool parse_beta(const char* s, double b[3]) { return std::sscanf(s, "%lf,%lf,%lf", &b[0], &b[1], &b[2]) == 3; }

static Cli parse(int argc, char** argv) {
  Cli c;
  auto& p = c.p;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    auto flag = [&](const char* k) { return std::strcmp(argv[i], k) == 0; };
    if (is("--shot")) c.shot = argv[++i];
    else if (is("--video")) c.video = argv[++i];
    else if (is("--clip")) c.clip = argv[++i];
    else if (is("--fps")) c.fps = std::atoi(argv[++i]);
    else if (is("--crf")) c.crf = std::atoi(argv[++i]);
    else if (is("--seconds")) c.seconds = std::atof(argv[++i]);
    else if (is("--mode")) c.mode = argv[++i];
    else if (is("--start")) c.start = argv[++i];
    else if (is("--timescale")) c.timescale = std::atof(argv[++i]);
    else if (is("--move-scale")) c.move_scale = std::min(1.0, std::max(0.1, std::atof(argv[++i])));
    else if (is("--denoise")) {
      const std::string d = argv[++i];
      if (d == "optix") p.denoise = true;
      else if (d != "none") { std::fprintf(stderr, "--denoise none|optix\n"); std::exit(1); }
    }
    else if (flag("--denoise-albedo")) p.denoise_albedo = true;
    else if (is("--sample-offset")) p.sample_offset = std::max(0, std::atoi(argv[++i]));
    else if (is("--hdr")) c.hdr_out = argv[++i];
    else if (flag("--selftest-denoise")) c.selftest_denoise = true;
    else if (flag("--denoise-kernels")) c.denoise_kernels = true;
    else if (flag("--nohud")) c.hud = false;
    else if (flag("--hud")) c.hud_offline = true;
    else if (is("--w")) p.w = std::atoi(argv[++i]);
    else if (is("--h")) p.h = std::atoi(argv[++i]);
    else if (is("--a")) p.a = std::min(0.9999, std::max(0.0, std::atof(argv[++i])));  // limit 0.9999 (Thorne's is 0.998): as a -> 1 the ISCO sticks to the horizon, degenerate disk (NaN in extreme corners; v2.7 audit)
    else if (is("--theta")) p.theta_deg = std::atof(argv[++i]);
    else if (is("--phi")) p.phi_deg = std::atof(argv[++i]);
    else if (is("--dist")) p.dist = std::atof(argv[++i]);
    else if (is("--fov")) p.fov_deg = std::atof(argv[++i]);
    else if (is("--rin")) p.rin = std::atof(argv[++i]);
    else if (is("--rout")) p.rout = std::atof(argv[++i]);
    else if (is("--tpeak")) p.tpeak = std::atof(argv[++i]);
    else if (is("--exposure")) p.exposure = std::atof(argv[++i]);
    else if (is("--sky-exposure")) p.sky_exposure = std::atof(argv[++i]);
    else if (is("--tol")) p.tol = std::atof(argv[++i]);
    else if (is("--phi-atol")) p.phi_atol = std::atof(argv[++i]);
    else if (is("--lmin")) p.l_min = std::atof(argv[++i]);
    else if (is("--rfar")) p.r_far = std::atof(argv[++i]);
    else if (is("--sky")) c.sky = argv[++i];
    else if (is("--runs")) c.runs = std::max(1, std::atoi(argv[++i]));
    else if (is("--autoexit")) c.autoexit = std::atoi(argv[++i]);
    else if (is("--block")) p.block = std::atoi(argv[++i]);
    else if (is("--profile")) p.nt = std::strcmp(argv[++i], "newtonian") != 0;
    else if (is("--observer")) c.observer = argv[++i];
    else if (is("--beta")) { if (!parse_beta(argv[++i], p.beta)) { std::fprintf(stderr, "--beta br,bth,bph\n"); std::exit(1); } c.observer = "custom"; }
    else if (is("--samples")) { p.samples = std::max(1, std::atoi(argv[++i])); c.samples_set = true; }
    else if (is("--seed")) p.seed = uint32_t(std::strtoul(argv[++i], nullptr, 10));
    else if (is("--accum")) p.accum_ldr = std::strcmp(argv[++i], "ldr") == 0;
    else if (is("--clamp")) p.clamp_lum = std::atof(argv[++i]);
    else if (is("--tonemap")) p.tonemap = std::strcmp(argv[++i], "aces") == 0 ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
    else if (is("--bloom")) p.bloom = std::atof(argv[++i]);
    else if (is("--beaming")) p.beaming = std::atof(argv[++i]);
    else if (is("--disk-model")) { const char* m = argv[++i]; p.model = std::strcmp(m, "torus") == 0 ? 1 : std::strcmp(m, "grmhd") == 0 ? 2 : 0; }
    else if (is("--grmhd")) c.grmhd_file = argv[++i];
    else if (is("--route-exr")) c.route_exr = argv[++i];
    else if (is("--emission")) { const std::string m = argv[++i]; if (m == "dexter") p.emission = 0; else if (m == "leung") p.emission = 1; else { std::fprintf(stderr, "--emission dexter|leung\n"); std::exit(1); } }
    else if (is("--theta-e-min")) p.theta_e_min = std::atof(argv[++i]);
    else if (is("--source")) {  // v2.2b3: mass, distance, 230 GHz flux and ring observed by the EHT (sources in notes/references.md)
      const std::string m = argv[++i];
      if (m == "m87") { p.mbh = 6.5e9; c.d_pc = 16.8e6; c.flux_target = 0.5; c.ring_ref = 42.0; p.nu_ghz = 230.0; }       // Paper VI (M, D), Paper V (0.5 Jy), Paper I (42 uas)
      else if (m == "sgra") { p.mbh = 4.14e6; c.d_pc = 8127.0; c.flux_target = 2.4; c.ring_ref = 51.8; p.nu_ghz = 230.0; }  // Paper V (M, D, 2.4 Jy), Paper I (51.8 uas)
      else { std::fprintf(stderr, "--source m87|sgra\n"); std::exit(1); }
    }
    else if (is("--ring-win")) c.ring_win = std::atof(argv[++i]);
    else if (flag("--radio-cal")) c.radio_cal = true;
    else if (flag("--fit-ne0")) c.fit_ne0 = true;
    else if (is("--d-pc")) c.d_pc = std::atof(argv[++i]);
    else if (is("--flux-target")) c.flux_target = std::atof(argv[++i]);
    else if (is("--radio-out")) c.radio_out = argv[++i];
    else if (is("--munit")) p.munit = std::max(1.0, std::atof(argv[++i]));
    else if (is("--electrons")) { const char* m = argv[++i]; p.electrons = std::strcmp(m, "rhigh") == 0 ? 1 : std::strcmp(m, "prop") == 0 ? 2 : std::strcmp(m, "rhigh-single") == 0 ? 3 : 0; }
    else if (is("--rhigh")) p.rhigh = std::max(1.0, std::atof(argv[++i]));
    else if (is("--th-e-ratio")) p.th_e_ratio = std::max(0.0, std::atof(argv[++i]));
    else if (is("--torus")) { if (std::sscanf(argv[++i], "%lf,%lf,%lf", &p.torus_rin, &p.torus_rmax, &p.torus_gamma) < 2) { std::fprintf(stderr, "--torus rin,rmax[,gamma]\n"); std::exit(1); } }
    else if (is("--ne0")) p.ne0 = std::max(0.0, std::atof(argv[++i]));
    else if (is("--plasma-beta")) p.plasma_beta = std::max(1e-3, std::atof(argv[++i]));
    else if (is("--ti-te")) p.ti_te = std::max(0.1, std::atof(argv[++i]));
    else if (is("--grmhd-step")) p.grmhd_step = std::max(0.005, std::atof(argv[++i]));
    else if (is("--torus-step")) p.torus_step = std::max(0.05, std::atof(argv[++i]));
    else if (is("--theta-e-max")) p.theta_e_max = std::max(0.0, std::atof(argv[++i]));
    else if (is("--nu-ghz")) p.nu_ghz = std::max(1e-3, std::atof(argv[++i]));
    else if (is("--mbh")) p.mbh = std::max(1.0, std::atof(argv[++i]));
    else if (is("--tb-range")) { if (std::sscanf(argv[++i], "%lf,%lf", &p.tb_lo, &p.tb_hi) != 2) { std::fprintf(stderr, "--tb-range lo,hi (log10 K)\n"); std::exit(1); } }
    else if (is("--route")) c.route_file = argv[++i];
    else if (is("--route-out")) c.route_out = argv[++i];
    else if (is("--route-frame")) c.route_frame = std::atoi(argv[++i]);
    else if (is("--route-shot")) c.route_shot = argv[++i];
    else if (is("--shutter")) c.shutter = std::min(1.0, std::max(0.0, std::atof(argv[++i])));
    else if (is("--chunk")) c.chunk = std::max(1, std::atoi(argv[++i]));
    else if (is("--first")) c.first = std::max(0, std::atoi(argv[++i]));
    else if (is("--last")) c.last = std::atoi(argv[++i]);
    else if (is("--workdir")) c.workdir = argv[++i];
    else if (is("--codec")) c.codec = argv[++i];
    else if (is("--crf16")) c.crf16 = std::atoi(argv[++i]);
    else if (flag("--resume")) c.resume = true;
    else if (is("--turb")) p.turb = std::max(0.0, std::atof(argv[++i]));
    else if (is("--disk-tex")) { double v[4] = {0, 60, 6, 5}; const int nn = std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3]); if (nn < 1) { std::fprintf(stderr, "--disk-tex amp[,fr[,fphi[,oct]]]\n"); std::exit(1); } p.tex_amp = v[0]; p.tex_fr = v[1]; p.tex_fphi = v[2]; p.tex_oct = std::max(1, int(v[3])); }
    else if (is("--turb-scale")) p.turb_scale = std::max(0.2, std::atof(argv[++i]));
    else if (is("--spot")) { if (std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &p.spot_r, &p.spot_phi_deg, &p.spot_size, &p.spot_amp) != 4) { std::fprintf(stderr, "--spot r,phi_deg,size,amp\n"); std::exit(1); } }
    else if (is("--spot-flare")) { if (std::sscanf(argv[++i], "%lf,%lf,%lf", &p.flare_t, &p.flare_w, &p.flare_boost) != 3) { std::fprintf(stderr, "--spot-flare t,w,boost\n"); std::exit(1); } }
    else if (is("--time")) p.disk_time = std::atof(argv[++i]);
    else if (flag("--slowlight")) p.slowlight = true;
    else if (is("--disk-rate")) c.disk_rate = std::atof(argv[++i]);
    else if (is("--hr")) p.hr = std::max(0.0, std::atof(argv[++i]));
    else if (is("--flare")) p.flare = std::atof(argv[++i]);
    else if (is("--tau")) p.tau = std::max(1e-6, std::atof(argv[++i]));
    else if (is("--vol-sub")) p.vol_sub = std::max(1, std::atoi(argv[++i]));
    else if (is("--edge-soft")) p.edge_soft = std::max(0.0, std::min(0.9, std::atof(argv[++i])));
    else if (is("--edge-noise")) p.edge_noise = std::max(0.0, std::min(0.5, std::atof(argv[++i])));
    else if (is("--fcol")) p.fcol = std::max(0.5, std::atof(argv[++i]));
    else if (is("--limb")) p.limb = std::strcmp(argv[++i], "eddington") == 0;
    else if (is("--expose")) p.expose_physical = std::strcmp(argv[++i], "physical") == 0;
    else if (is("--glare")) p.glare = std::atof(argv[++i]);
    else if (is("--glare-thr")) p.glare_thr = std::atof(argv[++i]);
    else if (is("--wormhole")) {  // ellis,rho  |  rho,a,M  (Interstellar: rho = 1, 2a = 0.01 rho, W = 0.05 rho  =>  a = 0.005, M = W/1.42953 = 0.03498)
      p.wormhole = true; const std::string s = argv[++i];
      if (s.rfind("ellis", 0) == 0) { p.wh_ellis = true; if (s.size() > 6) p.wh_rho = std::atof(s.c_str() + 6); }
      else if (s == "interstellar") { p.wh_ellis = false; p.wh_rho = 1.0; p.wh_a = 0.005; p.wh_M = 0.05 / 1.42953; }
      else { double v[3] = {1, 0.005, 0.03498}; std::sscanf(s.c_str(), "%lf,%lf,%lf", &v[0], &v[1], &v[2]); p.wh_ellis = false; p.wh_rho = v[0]; p.wh_a = v[1]; p.wh_M = v[2]; }
    }
    else if (is("--wh-l")) { p.wh_lc = std::atof(argv[++i]); }
    else if (is("--wh-lmax")) { p.wh_lmax = std::atof(argv[++i]); }
    else if (is("--sky2")) c.sky2 = argv[++i];
    else if (is("--disk-arms")) { double v[4] = {2, 20, 0.5, 14}; const int nn = std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3]); if (nn < 3) { std::fprintf(stderr, "--disk-arms m,pitch_graus,amp[,r_padrao]\n"); std::exit(1); } p.arm_m = std::max(1, int(v[0])); p.arm_pitch_deg = v[1]; p.arm_amp = std::min(0.98, std::max(0.0, v[2])); p.arm_rpat = v[3]; }
    else if (is("--disk-rings")) { double v[3] = {0.8, 8, 0}; const int nn = std::sscanf(argv[++i], "%lf,%lf,%lf", &v[0], &v[1], &v[2]); if (nn < 1) { std::fprintf(stderr, "--disk-rings amp[,freq[,gap]]\n"); std::exit(1); } p.ring_amp = v[0]; p.ring_freq = v[1]; p.ring_gap = v[2]; }
    else if (is("--hr-var")) { double v[2] = {0.7, 3}; const int nn = std::sscanf(argv[++i], "%lf,%lf", &v[0], &v[1]); if (nn < 1) { std::fprintf(stderr, "--hr-var amp[,scale]\n"); std::exit(1); } p.hv_amp = v[0]; p.hv_scale = v[1]; }
    else if (is("--lambda")) p.lambda = std::max(0.0, std::atof(argv[++i]));
    else if (is("--charge")) p.charge = std::max(0.0, std::atof(argv[++i]));
    else if (is("--hdr10")) { p.hdr = true; const double v = std::atof(argv[++i]); if (v > 0) p.hdr_peak = v; }
    else if (is("--hdr-nits")) p.hdr_nits = std::atof(argv[++i]);
    else if (is("--glare-cap")) p.glare_cap = std::atof(argv[++i]);
    else if (is("--grain")) p.grain = std::atof(argv[++i]);
    else if (is("--vignette")) p.vignette = std::atof(argv[++i]);
    else if (is("--chroma")) p.chroma = std::atof(argv[++i]);
    else if (flag("--cinematic")) { c.cinematic = true; apply_cinematic(p, true); }
    else if (is("--prec")) {
      const std::string s = argv[++i];
      c.prec_set = true;
      p.prec = s == "float" ? kcuda::Precision::Float : s == "mixed" ? kcuda::Precision::Mixed : kcuda::Precision::Double;
    }
    else if (flag("--nosky")) c.nosky = true;
    else if (flag("--rawsky")) p.sky_raw = true;
    else if (flag("--nodisk")) p.nodisk = true;
    else if (flag("--no-theta-lut")) p.theta_lut = false;
    else if (flag("--nodoppler")) p.nodoppler = true;
    else if (is("--audit-float")) { c.audit_n = std::max(1, std::atoi(argv[++i])); }
    else if (is("--audit-seed")) { c.audit_seed = uint32_t(std::strtoul(argv[++i], nullptr, 10)); }
    else if (flag("--selftest")) c.selftest = true;
    else if (flag("--vsync")) c.vsync = true;
    else if (flag("--help") || flag("-h")) c.help = true;
  }
  if (p.wormhole) { p.nodisk = true; p.nt = false; if (p.charge > 0.0 || p.lambda > 0.0 || p.model != 0 || p.hr > 0.0) { std::fprintf(stderr, "--wormhole: sky only (no disk, charge, lambda, volume)\n"); std::exit(1); } p.dist = p.wh_lc; }
  if (p.lambda > 0.0 && (p.a != 0.0 || p.charge > 0.0 || p.lambda >= 1.0 / 9.0 || p.dist >= 0.99 * 1e9)) { std::fprintf(stderr, "--lambda: so Schwarzschild-de Sitter (a = 0, Q = 0, 0 < Lambda < 1/9)\n"); std::exit(1); }
  if (p.charge > 0.0 || p.lambda > 0.0) {  // v2.1a/c/e: Kerr-Newman and SdS: only what is validated
    // KN (a^2 + Q^2 < 1): KN Novikov-Thorne/Page-Thorne thin disk with its own ISCO and redshift (ntk::); SdS or naked KN: Newtonian profile, no frequency shift and --rin
    const bool kn_ok = p.lambda == 0.0 && p.a * p.a + p.charge * p.charge < 1.0;
    if (p.model != 0 || p.hr > 0.0 || c.observer != "static" || (!kn_ok && p.rin < 0.0 && !p.nodisk)) {
      std::fprintf(stderr, "--charge/--lambda: thin disk and static observer only; SdS and naked Kerr-Newman (a^2 + Q^2 >= 1) need --rin (see --help)\n"); std::exit(1);
    }
    if (!kn_ok) p.nt = false;
    p.limb = false;  // limb darkening uses the Kerr gas frame
  }
  return c;
}

// observer velocity in the ZAMO frame from --observer (same convention as v0_6-kerr-cpu)
static void apply_observer(kcuda::Params& p, const std::string& obs) {
  if (obs == "orbit") { p.beta[0] = 0; p.beta[1] = 0; p.beta[2] = lorentz::circular_beta(p.a, p.dist, p.theta_deg * M_PI / 180.0); }
  else if (obs == "infall") { p.beta[0] = -lorentz::infall_beta(p.dist); p.beta[1] = 0; p.beta[2] = 0; }
  else if (obs == "static") { p.beta[0] = p.beta[1] = p.beta[2] = 0; }
}

static const char* prec_name(kcuda::Precision pr) {
  return pr == kcuda::Precision::Double ? "double" : pr == kcuda::Precision::Float ? "float" : "mixed";
}

static flight::Start start_of(const std::string& s) { return s == "zamo" ? flight::Start::Zamo : s == "infall" ? flight::Start::Infall : flight::Start::Orbit; }
static void fill_flight_params(kcuda::Params& p, const flight::Flight& f) {
  const auto s = f.state();
  p.dist = s.r; p.theta_deg = s.theta * 180.0 / M_PI; p.phi_deg = s.phi * 180.0 / M_PI;
  p.beta[0] = s.beta.x; p.beta[1] = s.beta.y; p.beta[2] = s.beta.z;
  const auto& b = f.basis();
  const Vec3* v[3] = {&b.fwd, &b.right, &b.up};
  double* dst[3] = {p.fwd, p.right, p.up};
  for (int k = 0; k < 3; ++k) { dst[k][0] = v[k]->x; dst[k][1] = v[k]->y; dst[k][2] = v[k]->z; }
}

// ---------------------------------------------------------------- selftest
// Rotation-axis regression (v0.6b bugs found in use): (1) central column: the ray with x = 0 (L = 0) had the
// final phi wrong by pi (vertical line); (2) camera near the pole (theta = 0.5, 179.5): in float 1 - mu^2
// lost all its digits, the step control did not converge (1e5 steps, "orbiting" rays = black streaks, 40x
// slower). Criteria: no orbiting ray, maximum steps < 2000, and the exit direction of the x = 0 ray within
// 0.02 rad of that of the neighbours x = +-1.3e-3 (~1 pixel at 1080p).
static int selftest_axis(const Cli& c) {
  int bad = 0;
  for (auto pr : {kcuda::Precision::Double, kcuda::Precision::Float})
    for (double th : {80.0, 30.0, 5.0, 0.5, 179.5})
      for (double a : {0.0, 0.9}) {
        kcuda::Params p = c.p;
        p.a = a; p.theta_deg = th; p.prec = pr; p.tol = pr == kcuda::Precision::Double ? 1e-8 : 2e-5;
        p.rin = kerr::isco(a, true);
        p.nodisk = true;  // only exit directions on the sky
        const double tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
        std::vector<double> rays;
        auto add = [&](double x, double y) {
          const double nrm = std::sqrt(1 + x * x + y * y);
          rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
        };
        const int nx = 65, ny = 37;  // odd width: the central pixel has x = 0 exactly
        for (int j = 0; j < ny; ++j)
          for (int i = 0; i < nx; ++i) add((2.0 * (i + 0.5) / nx - 1) * tan_half * 16.0 / 9.0, (1 - 2.0 * (j + 0.5) / ny) * tan_half);
        const size_t grid_n = rays.size() / 3;
        std::vector<double> ys;
        for (int j = 0; j < 25; ++j) {
          const double y = (-1.0 + 2.0 * j / 24.0) * tan_half;
          if (std::abs(y) < 0.15) continue;  // inside the shadow
          ys.push_back(y);
          add(0.0, y); add(1.3e-3, y); add(-1.3e-3, y);
        }
        std::vector<kcuda::RayOut> g;
        kcuda::trace_batch(p, rays, g);
        int orbiting = 0, smax = 0;
        for (size_t i = 0; i < grid_n; ++i) { orbiting += g[i].fate == 2; smax = std::max(smax, g[i].steps); }
        double worst = 0;
        for (size_t k = 0; k < ys.size(); ++k) {
          const auto& r0 = g[grid_n + 3 * k];
          for (int s = 1; s <= 2; ++s) {
            const auto& r1 = g[grid_n + 3 * k + s];
            if (r0.fate != 0 || r1.fate != 0) continue;
            auto dir = [](const kcuda::RayOut& r) {
              const double st = std::sqrt(std::max(0.0, 1 - r.mu_end * r.mu_end));
              return Vec3{st * std::cos(r.phi_end), st * std::sin(r.phi_end), r.mu_end};
            };
            const double dd = norm(dir(r0) - dir(r1));
            if (dd > 0.02 && std::getenv("BHS_DEBUG")) std::printf("   y=%.4f s=%d: x=0 mu=%.5f phi=%.5f steps=%d | nb mu=%.5f phi=%.5f steps=%d\n", ys[k], s, r0.mu_end, r0.phi_end, r0.steps, r1.mu_end, r1.phi_end, r1.steps);
            worst = std::max(worst, dd);
          }
        }
        const bool ok = orbiting == 0 && smax < 2000 && worst < 0.02;
        std::printf("axis %-6s a=%.1f th=%5.1f: orbiting %d, max steps %4d, dir(x=0) vs neighbours max %.2e %s\n",
                    pr == kcuda::Precision::Double ? "double" : "float", a, th, orbiting, smax, worst, ok ? "" : "FAILED");
        bad += !ok;
      }
  return bad;
}
// v0.7d: Novikov-Thorne LUT on the GPU vs nt::Profile (CPU), observer boost vs lorentz::observer_to_zamo,
// deterministic accumulation (fixed seed) and coherence of the resolve (bloom, tone mapping).
static int selftest_v07d(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  // 1. NT: GPU (float table, double interpolation) vs nt::Profile::temperature (double)
  for (double a : {0.0, 0.9, 0.998}) {
    kcuda::Params p = c.p; p.a = a; p.nt = true;
    const nt::Profile prof(a, std::max(2.0 * p.rout, 100.0));
    std::vector<double> r;
    for (int i = 0; i < 400; ++i) r.push_back(kerr::isco(a) * std::pow(300.0 / kerr::isco(a), i / 399.0));  // r_isco ... 300 (includes the tail r > 100)
    r.push_back(0.5 * kerr::isco(a));  // inside the ISCO: T = 0
    std::vector<double> gpu;
    kcuda::nt_lookup(p, r, gpu);
    double worst = 0;
    for (size_t i = 0; i < r.size(); ++i) worst = std::max(worst, std::abs(gpu[i] - prof.temperature(r[i])));
    char buf[96]; std::snprintf(buf, sizeof buf, "NT LUT a=%.3f: max |T_gpu - T_cpu|", a);
    check(worst < 2e-6, buf, worst, 2e-6);
  }
  // 2. boost: GPU vs lorentz::observer_to_zamo
  {
    std::vector<double> np;
    std::vector<Vec3> bs = {{0, 0, 0}, {0, 0, 0.3}, {-0.447, 0, 0}, {0.2, -0.3, 0.5}, {0, 0, 0.9}};
    double worst = 0;
    for (const auto& b : bs) {
      np.clear();
      std::vector<Vec3> dirs;
      for (int i = 0; i < 200; ++i) {
        const Vec3 d = normalize(Vec3{std::cos(0.37 * i + 0.1), std::sin(1.3 * i) + 0.2, std::sin(0.71 * i + 0.4)});
        dirs.push_back(d); np.insert(np.end(), {d.x, d.y, d.z});
      }
      double bb[3] = {b.x, b.y, b.z};
      std::vector<double> out;
      kcuda::boost_batch(np, bb, out);
      for (size_t i = 0; i < dirs.size(); ++i) {
        const auto ph = lorentz::observer_to_zamo(dirs[i], b);
        worst = std::max({worst, std::abs(out[4 * i] - ph.n.x), std::abs(out[4 * i + 1] - ph.n.y), std::abs(out[4 * i + 2] - ph.n.z), std::abs(out[4 * i + 3] - ph.dop)});
      }
    }
    check(worst < 1e-12, "boost GPU vs lorentz::observer_to_zamo (5 beta, 200 directions)", worst, 1e-12);
  }
  // 3. deterministic accumulation + resolve coherence (small image, float)
  {
    kcuda::Params p = c.p;
    p.w = 192; p.h = 108; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.a = 0.9; p.theta_deg = 80; p.samples = 6; p.seed = 7;
    p.use_sky = c.p.use_sky;
    std::vector<uint8_t> a1, a2, a3, a4;
    kcuda::render(p); kcuda::download_rgb(a1);
    kcuda::render(p); kcuda::download_rgb(a2);
    size_t diff = 0;
    for (size_t i = 0; i < a1.size(); ++i) diff += a1[i] != a2[i];
    check(diff == 0, "accumulation with a fixed seed: differing bytes between 2 renders", double(diff), 0.5);
    kcuda::Params q = p; q.seed = 8;
    kcuda::render(q); kcuda::download_rgb(a3);
    size_t d2 = 0;
    for (size_t i = 0; i < a1.size(); ++i) d2 += a1[i] != a3[i];
    check(d2 > 0, "another seed changes the jitter (bytes differ)", double(d2), 1.0 - 0.5);
    // N samples with sample 0 centred: samples=1 is the centre sample (does not depend on the seed)
    kcuda::Params s1 = p; s1.samples = 1;
    kcuda::render(s1); kcuda::download_rgb(a3);
    s1.seed = 99; kcuda::render(s1); kcuda::download_rgb(a4);
    size_t d3 = 0;
    for (size_t i = 0; i < a3.size(); ++i) d3 += a3[i] != a4[i];
    check(d3 == 0, "samples=1 (pixel centre) independent of the seed", double(d3), 0.5);
    // bloom adds light: the frame mean rises; aces tone mapping != reinhard
    kcuda::Params b0 = p; b0.samples = 1;
    kcuda::Params b1 = b0; b1.bloom = 0.3;
    kcuda::render(b0); kcuda::download_rgb(a1);
    kcuda::render(b1); kcuda::download_rgb(a2);
    double m0 = 0, m1 = 0;
    for (size_t i = 0; i < a1.size(); ++i) { m0 += a1[i]; m1 += a2[i]; }
    check(m1 > m0, "bloom increases the mean luminance of the frame", (m1 - m0) / a1.size(), 0.0);
    // v0.8d: polish. Everything at 0 = identical (physical regression); each effect does what it says.
    auto bytes = [&](const kcuda::Params& q, std::vector<uint8_t>& out) { kcuda::render(q); kcuda::download_rgb(out); };
    auto ndiff = [](const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) { size_t d = 0; for (size_t i = 0; i < x.size(); ++i) d += x[i] != y[i]; return d; };
    auto meanv = [](const std::vector<uint8_t>& x) { double m = 0; for (auto v : x) m += v; return m / x.size(); };
    kcuda::Params q0 = b0; q0.tonemap = kcuda::Tonemap::Aces; q0.bloom = 0.12;
    std::vector<uint8_t> r0, r1, r2;
    bytes(q0, r0);
    kcuda::Params q1 = q0; q1.glare = 0; q1.grain = 0; q1.vignette = 0; q1.chroma = 0; bytes(q1, r1);
    check(ndiff(r0, r1) == 0, "polish at 0: differing bytes (must be identical)", double(ndiff(r0, r1)), 0.5);
    kcuda::Params g = q0; g.glare = 0.5; bytes(g, r1);
    check(meanv(r1) > meanv(r0), "glare adds light (mean luminance rises)", meanv(r1) - meanv(r0), 0.0);
    kcuda::Params gr = q0; gr.grain = 0.05; bytes(gr, r1); bytes(gr, r2);
    check(ndiff(r1, r2) == 0 && ndiff(r1, r0) > 0, "grain: reproducible (fixed seed) and changes the image", double(ndiff(r1, r2)), 0.5);
    kcuda::Params gf = gr; gf.frame_index = 5; bytes(gf, r2);
    check(ndiff(r1, r2) > 0, "grain: the frame_index changes the noise", double(ndiff(r1, r2)), 0.5);
    kcuda::Params vg = q0; vg.vignette = 0.6; bytes(vg, r1);
    auto corner = [&](const std::vector<uint8_t>& x) { double m = 0; int n = 0; for (int j = 0; j < 12; ++j) for (int i = 0; i < 12; ++i) { for (int k = 0; k < 3; ++k) m += x[(size_t(j) * p.w + i) * 3 + k]; ++n; } return m / (3.0 * n); };
    check(corner(r1) <= corner(r0), "vignette: the corner does not get brighter", corner(r0) - corner(r1), -1e-9);
    kcuda::Params ch = q0; ch.chroma = 0.02; bytes(ch, r1);
    check(ndiff(r0, r1) > 0, "chroma: changes the image", double(ndiff(r0, r1)), 0.5);
    kcuda::Params b2 = b0; b2.tonemap = kcuda::Tonemap::Aces;
    kcuda::render(b2); kcuda::download_rgb(a3);
    size_t d4 = 0;
    for (size_t i = 0; i < a1.size(); ++i) d4 += a1[i] != a3[i];
    check(d4 > 0, "aces tone mapping differs from reinhard", double(d4), 0.5);
  }
  return bad;
}

// v0.8b: camera basis in the kernel (default = v0.6b bit-for-bit; rotated basis vs the host formula) and scripted flight
static void draw_flight_hud(const flight::Flight& fl, const kcuda::Params& p, int h) {
  const auto s = fl.state();
  const auto cs = flight::center_shift(p.a, s, fl.basis(), kerr::isco(p.a), p.rout);
  kcuda::hud_draw(flight::hud_text("VOO", p.a, s, fl.pilot().consts(), cs, flight::accel_level(fl.level), fl.timescale, fl.captured(), false, true), std::max(1, h / 540));
}

// v0.9a: photon Q and mu in the gas frame (GPU vs core), and the effect of limb darkening / f_col on the images.
static int selftest_v09a(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  // 1. mu(r, L, Q) GPU vs emission::emitter_mu
  {
    std::vector<double> rLQ;
    uint32_t h = 12345u;
    auto rnd = [&]() { h = h * 1664525u + 1013904223u; return double(h >> 8) / 16777216.0; };
    for (int i = 0; i < 2000; ++i) rLQ.insert(rLQ.end(), {3.0 + 47.0 * rnd(), -6.0 + 12.0 * rnd(), 120.0 * rnd()});
    for (double a : {0.0, 0.9, 0.998}) {
      std::vector<double> g;
      kcuda::emitter_mu_batch(a, rLQ, g);
      double worst = 0;
      for (size_t i = 0; i < g.size(); ++i) {
        const double cpu = emission::emitter_mu(a, rLQ[3 * i], rLQ[3 * i + 1], rLQ[3 * i + 2]);
        worst = std::max(worst, std::abs(g[i] - cpu));
      }
      check(worst < 1e-12, a == 0.0 ? "mu GPU vs emission::emitter_mu (a=0, 2000 cases)" : a == 0.9 ? "mu GPU vs emission::emitter_mu (a=0.9)" : "mu GPU vs emission::emitter_mu (a=0.998)", worst, 1e-12);
    }
  }
  // 2. photon Q: GPU (trace) vs kerr::trace
  {
    kcuda::Params p = c.p; p.a = 0.9; p.theta_deg = 75; p.prec = kcuda::Precision::Double;
    const int n = 400;
    std::vector<double> rays;
    for (int i = 0; i < n; ++i) {
      const double x = (-1.0 + 2.0 * (i + 0.5) / n) * 0.5, y = 0.1 + 0.3 * std::sin(i * 0.37);
      const double nrm = std::sqrt(1 + x * x + y * y);
      rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
    }
    std::vector<kcuda::RayOut> g;
    kcuda::trace_batch(p, rays, g);
    double worst = 0; int hits = 0;
    for (int i = 0; i < n; ++i) {
      const auto t = kerr::trace(p.a, p.dist, p.theta_deg * M_PI / 180.0, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], kerr::Disk{kerr::isco(p.a, true), p.rout});
      hits += g[i].disk_hit;
      worst = std::max(worst, std::abs(g[i].Q - t.Q) / (1.0 + std::abs(t.Q)));
    }
    check(worst < 1e-12 && hits > 0, "GPU Q vs kerr::trace (400 rays)", worst, 1e-12);
  }
  // 3. image: limb darkening (flux conserved: face-on > 1, edge-on < 1), f_col = 1 equal to the default, f_col != 1 changes it
  {
    kcuda::Params p = c.p;
    p.w = 256; p.h = 144; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.a = 0.9; p.use_sky = false; p.samples = 1; p.tonemap = kcuda::Tonemap::Reinhard;
    p.fov_deg = 30; p.dist = 40;
    auto mean = [&](const kcuda::Params& q) {
      kcuda::render(q);
      std::vector<float> hdr; kcuda::download_hdr(hdr);
      double s = 0;
      for (size_t i = 0; i < hdr.size(); i += 4) s += 0.2126 * hdr[i] + 0.7152 * hdr[i + 1] + 0.0722 * hdr[i + 2];
      return s / (hdr.size() / 4);
    };
    for (double th : {5.0, 85.0}) {
      p.theta_deg = th;
      kcuda::Params q = p; q.limb = true;
      const double r = mean(q) / mean(p);
      if (th < 45) check(r > 1.10 && r < 1.30, "limb darkening face-on (theta = 5): mean luminance x1.10..1.30", r, 1.3);
      else check(r > 0.60 && r < 0.98, "limb darkening edge-on (theta = 85): mean luminance < 1", r, 0.98);
    }
    p.theta_deg = 60;
    kcuda::Params q = p; q.fcol = 1.0;
    check(std::abs(mean(q) - mean(p)) == 0.0, "f_col = 1 equal to the default (identical mean luminance)", 0.0, 0.0);
    q.fcol = 1.7; q.expose_physical = true;
    const double hard = mean(q) / mean(p);
    q.expose_physical = false;
    const double hard_rel = mean(q) / mean(p);
    check(hard != 1.0 && hard_rel != 1.0, "f_col = 1.7 changes the image (physical and relative)", hard, 1.0);
    std::printf("    f_col 1.7: mean luminance x%.3f with physical exposure, x%.3f renormalised to the peak\n", hard, hard_rel);
    q.fcol = 1.0;  // restore the LUT for the rest of the selftest
    mean(q);
  }
  return bad;
}

// v0.9a2: soft edge. GPU vs CPU skip, default intact, soft edge without jumps and letting you see what is behind it.
// v2.1a: Kerr-Newman on the GPU vs kerr::trace (CPU, q2 in kerr::Options): fate of each ray (escaped / absorbed), exit direction and first disk crossing
static int selftest_v21(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  struct Case { const char* nm; double a, q; };
  for (const Case cs : {Case{"Kerr-Newman a = 0.5, Q = 0.5 (horizon)", 0.5, 0.5}, Case{"Reissner-Nordstrom Q = 0.8", 0.0, 0.8}, Case{"naked singularity a = 0.5, Q = 0.95", 0.5, 0.95}, Case{"Kerr control a = 0.5, Q = 0", 0.5, 0.0}}) {
    kcuda::Params p = c.p; p.a = cs.a; p.charge = cs.q; p.theta_deg = 80; p.dist = 40; p.prec = kcuda::Precision::Double; p.tol = 1e-9; p.r_far = 5e3; p.max_steps = 2000000;
    p.rin = 3.0; p.rout = 20.0;
    std::vector<double> rays;
    for (int i = 0; i < 40; ++i)
      for (int j = 0; j < 40; ++j) {
        const double x = -0.22 + 0.44 * (i + 0.5) / 40, y = -0.22 + 0.44 * (j + 0.5) / 40, nrm = std::sqrt(1 + x * x + y * y);
        rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
      }
    std::vector<kcuda::RayOut> g;
    kcuda::trace_batch(p, rays, g);
    kerr::Options ko; ko.rtol = p.tol; ko.atol = p.tol * 0.1; ko.q2 = cs.q * cs.q; ko.r_far = p.r_far; ko.max_steps = p.max_steps;
    int mism = 0, esc = 0, cap = 0, hit = 0; double worst_dir = 0, worst_r = 0, worst_phi = 0; std::map<int, int> mm;
    for (size_t i = 0; i < g.size(); ++i) {
      const auto t = kerr::trace(p.a, p.dist, p.theta_deg * M_PI / 180.0, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], kerr::Disk{p.rin, p.rout}, ko);
      const bool cesc = t.fate == kerr::Fate::Escaped && !t.disk_hit, gesc = g[i].fate == 0 && !g[i].disk_hit;
      if (cesc != gesc || int(t.disk_hit) != g[i].disk_hit) { ++mism; ++mm[int(t.fate) * 10 + g[i].fate + (t.disk_hit ? 100 : 0) + (g[i].disk_hit ? 1000 : 0)]; continue; }
      if (cesc) {
        ++esc; worst_dir = std::max(worst_dir, std::abs(std::cos(t.theta_end) - g[i].mu_end));
        double dphi = std::fmod(std::abs(t.phi_end - g[i].phi_end), 2.0 * M_PI); dphi = std::min(dphi, 2.0 * M_PI - dphi);  // v2.7: the azimuth too (the phi equation uses the KN Delta)
        worst_phi = std::max(worst_phi, dphi);
      }
      else if (t.disk_hit) { ++hit; worst_r = std::max(worst_r, std::abs(t.disk_r - g[i].disk_r)); } else ++cap;
    }
    std::printf("  %s: %d escaped, %d absorbed, %d on the disk, %d different fates\n", cs.nm, esc, cap, hit, mism);
    for (const auto& kv : mm) std::printf("    different fate [cpu fate*10 + gpu fate + 100 cpu disk + 1000 gpu disk] = %d : %d\n", kv.first, kv.second);
    // v2.7: strict criterion also on the naked singularity: the earlier 139/1600 different fates were the dphif defect (q2 was missing in Delta), not ill-conditioned rays
    check(mism <= 2 && esc > 100, "GPU fates = CPU (<= 2 of 1600)", double(mism), 2.0);
    // 4e-5 also in the Kerr control (Q = 0): it comes from the last step before r_far = 5e3 (d mu/dr ~ b/r^2 times a step), not from the charge
    check(worst_dir < 1e-4, "exit direction: max |dmu| GPU vs CPU (last step at r_far)", worst_dir, 1e-4);
    check(worst_r < 1e-4, "first disk crossing: max |dr| GPU vs CPU", worst_r, 1e-4);
    check(worst_phi < 5e-4, "exit azimuth: max |dphi| GPU vs CPU (mod 2 pi)", worst_phi, 5e-4);
  }
  return bad;
}

// v2.1c: Schwarzschild-de Sitter on the GPU: angular radius of the shadow vs sds::static_shadow_sin (analytic, verified in the CPU core against the radial ODE and the Binet identity)
static int selftest_v21c(const Cli& c) {
  int bad = 0;
  struct Case { double lam, rO; };
  for (const Case cs : {Case{0.002, 10.0}, Case{0.01, 8.0}, Case{0.03, 5.0}}) {
    double rh = 0, rc = 0;
    if (!sds::horizons(1.0, cs.lam, rh, rc) || !(cs.rO > rh && cs.rO < rc)) { std::printf("  invalid case (Lambda %.3f, r_O %.1f)\n", cs.lam, cs.rO); ++bad; continue; }
    kcuda::Params p = c.p; p.a = 0.0; p.charge = 0.0; p.lambda = cs.lam; p.theta_deg = 80; p.dist = cs.rO; p.prec = kcuda::Precision::Double; p.tol = 1e-10; p.rout = 0.0; p.max_steps = 2000000;
    auto captured_at = [&](const std::vector<double>& xs) {
      std::vector<double> rays;
      for (double x : xs) { const double n = std::sqrt(1 + x * x); rays.insert(rays.end(), {1.0 / n, 0.0, -x / n}); }
      std::vector<kcuda::RayOut> g; kcuda::trace_batch(p, rays, g);
      std::vector<int> cap; for (const auto& o : g) cap.push_back(o.fate == 1 && !o.disk_hit);
      return cap;
    };
    std::vector<double> xs; for (int k = 0; k < 2000; ++k) xs.push_back(0.001 * (k + 1));
    auto cap = captured_at(xs);
    int k0 = 0; while (k0 < 1999 && cap[k0]) ++k0;
    std::vector<double> xs2; for (int k = 0; k < 1001; ++k) xs2.push_back(xs[k0 - 1] + 1e-6 * k);
    auto cap2 = captured_at(xs2);
    int k1 = 0; while (k1 < 1000 && cap2[k1]) ++k1;
    const double x_edge = xs2[k1] - 0.5e-6;
    const double s_an = sds::static_shadow_sin(1.0, cs.lam, cs.rO), t_an = s_an / std::sqrt(1 - s_an * s_an);
    const double err = std::abs(x_edge - t_an) / t_an;
    std::printf("  SdS Lambda = %.3f, r_O = %.1f (horizons %.3f, %.3f): tan(alpha_c) GPU %.6f, analytic %.6f\n", cs.lam, cs.rO, rh, rc, x_edge, t_an);
    const bool ok = err < 2e-5; std::printf("  %-62s %.3e (limit %.1e) %s\n", "angular radius of the shadow: relative error", err, 2e-5, ok ? "" : "FAILED"); bad += !ok;
  }
  return bad;
}

// v2.1d: wormhole on the GPU (float) vs bhs::wh::trace (CPU, double) with the same l_max: swept angle psi and final side; and the flat tail (l_max = 100 vs 1e5)
static int selftest_v21d(const Cli& c) {
  int bad = 0;
  struct Case { const char* nm; bool ellis; double rho, a, M; };
  for (const Case cs : {Case{"Ellis rho = 1", true, 1.0, 0.0, 0.0}, Case{"Interstellar (rho = 1, a = 0.005, M = 0.03498)", false, 1.0, 0.005, 0.03498}}) {
    kcuda::Params p = c.p; p.wormhole = true; p.wh_ellis = cs.ellis; p.wh_rho = cs.rho; p.wh_a = cs.a; p.wh_M = cs.M; p.wh_lc = 8.0; p.wh_lmax = 100.0;
    wh::Params wp; wp.rho = cs.rho; wp.a = cs.a; wp.M = cs.M; wp.ellis = cs.ellis;
    const double r0 = wh::r_of_l(wp, p.wh_lc), sa_c = cs.rho / r0;
    std::vector<double> al; for (int k = 0; k < 600; ++k) al.push_back(M_PI * (k + 0.5) / 600.0);
    std::vector<double> psi; std::vector<int> side;
    kcuda::wh_batch(p, al, psi, side);
    double worst = 0, worst_tail = 0; int mism = 0, n = 0;
    for (size_t k = 0; k < al.size(); ++k) {
      if (std::abs(std::sin(al[k]) - sa_c) < 2e-3) continue;   // neighbourhood of the critical ray: the side is very sensitive
      const auto t = wh::trace(wp, p.wh_lc, al[k], p.wh_lmax), t2 = wh::trace(wp, p.wh_lc, al[k], 1e5);
      ++n;
      if (t.side != side[k] || !t.ok) { ++mism; continue; }
      worst = std::max(worst, std::abs(t.psi - psi[k]));
      if (t2.ok && t2.side == t.side) worst_tail = std::max(worst_tail, std::abs(t2.psi - t.psi));
    }
    std::printf("  %s: %d rays, different side in %d\n", cs.nm, n, mism);
    auto check = [&](bool ok, const char* what, double v, double lim) { std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED"); bad += !ok; };
    check(mism == 0, "sides GPU = CPU", double(mism), 0.0);
    check(worst < 2e-4, "psi: max |GPU (float) - CPU (double)| (same l_max)", worst, 2e-4);
    check(worst_tail < 2e-3, "flat tail: |psi(l_max = 100) - psi(l_max = 1e5)| (CPU)", worst_tail, 2e-3);
  }
  return bad;
}

// v2.1e: Kerr-Newman disk on the GPU: T(r) (float table) and redshift (double) vs ntk:: (CPU)
static int selftest_v21e(const Cli& c) {
  int bad = 0;
  for (const auto cs : {std::pair<double, double>{0.5, 0.5}, {0.0, 0.8}, {0.8, 0.3}, {0.0, 0.0}}) {
    kcuda::Params p = c.p; p.a = cs.first; p.charge = cs.second; p.theta_deg = 75; p.dist = 60; p.rout = 20.0; p.nt = true;
    const double q2 = cs.second * cs.second;
    const double rin = q2 > 0 ? ntk::isco(p.a, q2) : kerr::isco(p.a);
    std::vector<double> in; std::vector<double> rs;
    for (int k = 0; k < 200; ++k) { const double r = rin * std::pow(30.0 / rin, (k + 0.5) / 200.0); rs.push_back(r); in.insert(in.end(), {r, -3.0 + 0.03 * k}); }
    std::vector<double> out; kcuda::ntkn_batch(p, in, out);
    double wT = 0, wg = 0;
    // T table: the same ntk::Profile/nt::Profile (float on the GPU); compared with the ln r interpolation done here
    const double rout_prof = std::max(2.0 * p.rout, 100.0);
    std::vector<double> tab; double rin_t;
    if (q2 > 0) { ntk::Profile pr(p.a, q2, rout_prof); tab = pr.table(); rin_t = pr.r_in(); } else { nt::Profile pr(p.a, rout_prof); tab = pr.table(); rin_t = pr.r_in(); }
    auto Tcpu = [&](double r) { const double x = (std::log(r) - std::log(rin_t)) / (std::log(rout_prof) - std::log(rin_t)) * (tab.size() - 1); const int i = std::min<int>(int(tab.size()) - 2, int(x)); const double f = x - i; return (1 - f) * tab[i] + f * tab[i + 1]; };
    for (size_t k = 0; k < rs.size(); ++k) {
      wT = std::max(wT, std::abs(out[2 * k] - Tcpu(rs[k])));
      const double gc = q2 > 0 ? ntk::redshift(p.a, q2, rs[k], in[2 * k + 1], p.dist, p.theta_deg * M_PI / 180.0) : kerr::redshift(p.a, rs[k], in[2 * k + 1], p.dist, p.theta_deg * M_PI / 180.0);
      wg = std::max(wg, std::abs(out[2 * k + 1] - gc) / gc);
    }
    std::printf("  KN disk GPU vs CPU (a = %.2f, Q = %.2f, ISCO %.4f): max |dT| %.2e, max rel |dg| %.2e\n", p.a, cs.second, rin, wT, wg);
    const bool ok = wT < 1e-5 && wg < 1e-10; if (!ok) std::printf("  FAILED\n"); bad += !ok;
  }
  return bad;
}

// v2.7: FLOAT robustness audit (the interactive app runs in float): N random scenarios (fixed seed) and fixed corners, at 96x54 with 2 samples; fails if the HDR
// has NaN/inf/negatives, warns if a ray does not terminate (steps = max_steps) or if a frame takes > 3 s. Respects the restrictions of each variant.
static int audit_float(const Cli& c, int n, uint32_t seed) {
  std::mt19937 rng(seed);
  auto U = [&](double a, double b) { return a + (b - a) * std::uniform_real_distribution<double>(0.0, 1.0)(rng); };
  auto pick = [&](std::initializer_list<double> v) { return *(v.begin() + std::uniform_int_distribution<int>(0, int(v.size()) - 1)(rng)); };
  int bad = 0, warn_steps = 0, warn_slow = 0, runs = 0;
  std::map<std::string, int> per_class, fail_class;
  for (int s = 0; s < n; ++s) {
    kcuda::Params p = c.p; p.w = 96; p.h = 54; p.samples = 2; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.use_sky = c.p.use_sky;
    const double u = U(0, 1);
    const char* cls;
    // geometry and camera
    p.a = pick({0.0, 0.1, 0.5, 0.9, 0.99, 0.998, 0.9999, U(0, 0.9999)});
    double rh = kerr::horizon(p.a);
    if (u < 0.55) cls = "kerr";
    else if (u < 0.70) { cls = "kn"; p.charge = U(0.05, 0.999) * std::sqrt(1.0 - p.a * p.a); rh = kerr::horizon(p.a, p.charge * p.charge); }
    else if (u < 0.75) { cls = "kn-nu"; p.charge = std::sqrt(std::max(0.0, 1.0 - p.a * p.a)) + U(0.05, 0.6); p.nt = false; p.rin = U(1.5, 6.0); rh = 0.05; }
    else if (u < 0.85) { cls = "sds"; p.a = 0; p.lambda = U(1e-4, 0.11); p.nt = false; p.rin = 6.0; p.nodisk = U(0, 1) < 0.5; sds::horizons(1.0, p.lambda, rh, p.wh_lc); }
    else if (u < 0.93) { cls = "wormhole"; p.wormhole = true; p.nodisk = true; p.wh_ellis = U(0, 1) < 0.5; if (!p.wh_ellis) { p.wh_rho = 1; p.wh_a = 0.005; p.wh_M = 0.03498; } p.wh_lc = std::exp(U(std::log(1.05), std::log(80.0))); rh = 0; }
    else { cls = "torus-radio"; p.model = 1; p.torus_rin = 7; p.torus_rmax = 12; p.ne0 = 1e5; p.plasma_beta = 10; p.theta_e_max = 20; p.mbh = 6.5e9; p.nu_ghz = 230; p.use_sky = false; }
    // distance: from r+ (with margin) to 2000 M; SdS between the horizons; torus >= 80
    double dmin = std::max(rh * 1.02 + 0.01, 1.05), dmax = 2000.0;
    if (!std::strcmp(cls, "sds")) { dmax = 0.97 * p.wh_lc; dmin = std::max(dmin, 1.05 * rh); }
    if (!std::strcmp(cls, "torus-radio")) dmin = 100.0;
    p.dist = std::exp(U(std::log(dmin), std::log(std::max(dmax, dmin * 1.01))));
    if (!std::strcmp(cls, "wormhole")) p.dist = p.wh_lc;
    if (!std::strcmp(cls, "sds")) p.wh_lc = 8.0;  // (the wh field is not used in SdS)
    p.theta_deg = pick({0.0, 0.001, 0.5, 5.0, 45.0, 89.99, 90.0, 90.01, 120.0, 179.5, 180.0, U(0, 180)});
    p.phi_deg = U(0, 360);
    p.fov_deg = pick({1.0, 5.0, 30.0, 60.0, 90.0, 120.0, 150.0, 170.0, U(2, 140)});
    p.rout = U(8, 60);
    p.exposure = std::exp(U(std::log(0.1), std::log(20.0)));
    // disk and effects (only Kerr/KN with a horizon: layer, turbulence, spot, slow light, observer motion)
    if (!std::strcmp(cls, "kerr") || !std::strcmp(cls, "kn")) {
      /* layer/volume only in Kerr (parse refuses --charge with --hr) */ if (!std::strcmp(cls, "kerr") && U(0, 1) < 0.3) { p.hr = U(0.01, 0.15); p.tau = U(0.5, 8); p.flare = U(0, 0.4); if (U(0, 1) < 0.5) { p.turb = U(0.2, 2); p.turb_scale = U(0.6, 3); } if (U(0, 1) < 0.3) { p.spot_r = U(7, 12); p.spot_phi_deg = U(0, 360); p.spot_size = U(0.5, 2); p.spot_amp = U(1, 6); } p.slowlight = U(0, 1) < 0.3; p.disk_time = U(0, 500); p.edge_soft = U(0, 1) < 0.3 ? U(0.05, 0.3) : 0.0; }
      if (!std::strcmp(cls, "kerr") && U(0, 1) < 0.25) { const double b = U(0, 0.99); const double th = U(0, M_PI), ph = U(0, 2 * M_PI); p.beta[0] = b * std::sin(th) * std::cos(ph); p.beta[1] = b * std::sin(th) * std::sin(ph); p.beta[2] = b * std::cos(th); }
      p.nodoppler = U(0, 1) < 0.1; p.limb = !std::strcmp(cls, "kerr") && U(0, 1) < 0.2;
    }
    if (U(0, 1) < 0.35) { p.bloom = U(0.05, 0.6); p.tonemap = kcuda::Tonemap::Aces; }
    if (U(0, 1) < 0.25) { p.glare = U(0.05, 0.3); p.grain = 0.02; p.vignette = 0.3; p.chroma = 0.003; }
    if (U(0, 1) < 0.2) { p.hdr = true; }
    ++runs; ++per_class[cls];
    kcuda::Stats st;
    const auto t0 = Clock::now();
    kcuda::render(p, &st);
    std::vector<float> h; kcuda::download_hdr(h);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    size_t nonfinite = 0, negative = 0;
    for (size_t i = 0; i < h.size(); ++i) { if (i % 4 == 3) continue; if (!std::isfinite(h[i])) ++nonfinite; else if (h[i] < -1e-6f) ++negative; }
    const bool noterm = st.steps_max >= p.max_steps;
    char desc[512];
    std::snprintf(desc, sizeof desc, "[%s] a=%.5f Q=%.4f L=%.4f d=%.4f th=%.4f phi=%.1f fov=%.1f rout=%.1f hr=%.3f beta=(%.2f,%.2f,%.2f) wh=%d", cls, p.a, p.charge, p.lambda, p.dist, p.theta_deg, p.phi_deg, p.fov_deg, p.rout, p.hr, p.beta[0], p.beta[1], p.beta[2], int(p.wormhole));
    if (st.sanitized > 0 && std::getenv("AUDIT_DEBUG")) {  // diagnostic: where does the NaN come from? (tracing or shading)
      std::vector<double> rays; const double tf = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
      for (int i = 0; i < 60; ++i) for (int j = 0; j < 60; ++j) { const double x = (2.0 * (i + 0.5) / 60 - 1) * tf, y = (2.0 * (j + 0.5) / 60 - 1) * tf, nrm = std::sqrt(1 + x * x + y * y); rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm}); }
      std::vector<kcuda::RayOut> go; kcuda::trace_batch(p, rays, go);
      int nm = 0, np = 0, nd = 0, nl = 0, f0 = 0, f1 = 0, f2 = 0, hit = 0;
      for (auto& o : go) { nm += !std::isfinite(o.mu_end); np += !std::isfinite(o.phi_end); nd += !std::isfinite(o.disk_r); nl += !std::isfinite(o.L); f0 += o.fate == 0; f1 += o.fate == 1; f2 += o.fate == 2; hit += o.disk_hit; }
      std::printf("    diag trace: non-finite mu_end %d phi_end %d disk_r %d L %d | fate0 %d fate1 %d fate2 %d disk %d of %zu\n", nm, np, nd, nl, f0, f1, f2, hit, go.size());
    }
    if (st.sanitized > 0) { ++bad; ++fail_class[cls]; std::printf("  FAILED: %d non-finite pixels sanitised: %s\n", st.sanitized, desc); }
    if (nonfinite || negative) { ++bad; ++fail_class[cls]; std::printf("  FAILED: %zu non-finite, %zu negative: %s\n", nonfinite, negative, desc); }
    if (noterm) { ++warn_steps; std::printf("  WARNING (ray did not terminate, max steps %d): %s\n", st.steps_max, desc); }
    if (ms > 3000.0) { ++warn_slow; std::printf("  WARNING (slow, %.0f ms): %s\n", ms, desc); }
  }
  std::printf("float audit: %d scenarios (seed %u):", runs, seed);
  for (auto& kv : per_class) std::printf(" %s=%d", kv.first.c_str(), kv.second);
  std::printf("\n  failures (NaN/inf/negative): %d | warnings: %d did not terminate, %d slow\n", bad, warn_steps, warn_slow);
  return bad;
}

static int selftest_v09a2(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  {  // 1. second crossing: kcuda::trace_batch(skip) vs kerr::trace(Disk::skip)
    kcuda::Params p = c.p; p.a = 0.9; p.theta_deg = 80; p.prec = kcuda::Precision::Double; p.rout = 60.0;
    std::vector<double> rays;
    for (int i = 0; i < 40; ++i)
      for (int j = 0; j < 40; ++j) {
        const double x = -0.22 + 0.44 * (i + 0.5) / 40, y = -0.22 + 0.44 * (j + 0.5) / 40, nrm = std::sqrt(1 + x * x + y * y);
        rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
      }
    std::vector<kcuda::RayOut> g;
    kcuda::trace_batch(p, rays, g, 1);
    // same tolerances on both sides (as in the 1st-crossing selftest); higher-order images amplify the error, hence p99
    kerr::Options ko; ko.rtol = p.tol; ko.atol = p.tol * 0.1;
    std::vector<double> dr; int both = 0, mism = 0;
    for (size_t i = 0; i < g.size(); ++i) {
      const auto t = kerr::trace(p.a, p.dist, p.theta_deg * M_PI / 180.0, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], kerr::Disk{kerr::isco(p.a, true), p.rout, 1}, ko);
      if (int(t.disk_hit) != g[i].disk_hit) { ++mism; continue; }
      if (t.disk_hit) { ++both; dr.push_back(std::abs(t.disk_r - g[i].disk_r)); }
    }
    std::sort(dr.begin(), dr.end());
    const double worst = dr.empty() ? 0.0 : dr[size_t(0.99 * (dr.size() - 1))];
    std::printf("    dr: p50 %.2e p99 %.2e max %.2e\n", dr.empty() ? 0.0 : dr[dr.size() / 2], worst, dr.empty() ? 0.0 : dr.back());
    // measured: p99(|dr|) = 1.9e-4, 5.7e-6, 1.2e-7 for tol = 1e-6, 1e-8, 1e-10 (scales with the tolerance: higher-order images, not a systematic error)
    check(both > 20 && mism <= int(g.size()) / 200 && worst < 2000.0 * p.tol, "2nd crossing (skip = 1): GPU vs kerr::trace, p99 of |dr| < 2000 tol", worst, 2000.0 * p.tol);
    std::printf("    %d rays with a 2nd crossing, %d with a different fate\n", both, mism);
  }
  {  // 2. image: soft edge (soft 0.3) keeps the mean brightness, only changes the fringe; noise does not blow up
    kcuda::Params p = c.p;
    p.w = 256; p.h = 144; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.a = 0.9; p.use_sky = false; p.samples = 1; p.theta_deg = 70; p.rout = 16;
    auto mean = [&](const kcuda::Params& q, std::vector<float>* keep = nullptr) {
      kcuda::render(q);
      std::vector<float> hdr; kcuda::download_hdr(hdr);
      double s = 0;
      for (size_t i = 0; i < hdr.size(); i += 4) s += 0.2126 * hdr[i] + 0.7152 * hdr[i + 1] + 0.0722 * hdr[i + 2];
      if (keep) *keep = hdr;
      return s / (hdr.size() / 4);
    };
    std::vector<float> h0, h1;
    const double m0 = mean(p, &h0);
    kcuda::Params q = p; q.edge_soft = 0.0; q.edge_noise = 0.0;
    check(mean(q) == m0, "edge-soft = 0, edge-noise = 0 equal to the default", 0.0, 0.0);
    q.edge_soft = 0.3;
    const double m1 = mean(q, &h1);
    check(m1 < m0 && m1 > 0.8 * m0, "soft edge (0.3): mean brightness lower but > 80 % of the hard one", m1 / m0, 0.8);
    size_t same = 0, n = h0.size() / 4;  // pixels far from the edge stay equal
    for (size_t i = 0; i < h0.size(); i += 4) same += h0[i + 1] == h1[i + 1];
    check(same > n / 2, "more than half of the pixels (far from the edge) unchanged", double(same) / n, 0.5);
    q.edge_noise = 0.2;
    std::vector<float> h2; mean(q, &h2);
    bool finite = true; for (float v : h2) finite &= std::isfinite(v);
    check(finite, "soft edge + noise 0.2: finite HDR", 0.0, 0.0);
  }
  return bad;
}

// v0.9b: thick disk. Analytic face-on oracle (I = B (1 - e^(-tau/mu_gas))), limits and convergences.
static int selftest_v09b(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  auto hdr_of = [&](const kcuda::Params& q) { kcuda::render(q); std::vector<float> h; kcuda::download_hdr(h); return h; };
  auto mean_of = [&](const std::vector<float>& h) {
    double s = 0;
    for (size_t i = 0; i < h.size(); i += 4) s += 0.2126 * h[i] + 0.7152 * h[i + 1] + 0.0722 * h[i + 2];
    return s / (h.size() / 4);
  };
  // 1. face-on (theta = 1 degree, D = 1000, a = 0): each disk pixel = exposure * LUT(g T) * (1 - exp(-tau / mu_gas)),
  //    with r, L, Q from the thin-disk crossing (kerr::trace) and mu_gas = emission::emitter_mu (v0.9a)
  {
    kcuda::Params p = c.p;
    p.w = 64; p.h = 64; p.prec = kcuda::Precision::Double; p.tol = 1e-8; p.a = 0.0; p.dist = 1000.0; p.theta_deg = 1.0; p.fov_deg = 8.0;
    p.rout = 100.0; p.use_sky = false; p.samples = 1; p.nt = true; p.hr = 0.004; p.vol_sub = 8; p.tonemap = kcuda::Tonemap::Reinhard;
    const double rin = kerr::isco(p.a, true);
    const nt::Profile prof(p.a, std::max(2.0 * p.rout, 100.0));
    const std::vector<float> lutd = lut::build();
    const Vec3 ref = lut::lookup(lutd, p.tpeak);
    const double k0 = p.exposure * 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z);
    const double tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
    for (double tau : {0.5, 2.0}) {
      p.tau = tau;
      const auto h = hdr_of(p);
      double worst = 0; int n = 0;
      for (int j = 8; j < 64; j += 7)
        for (int i = 8; i < 64; i += 7) {
          const double x = (2.0 * ((i + 0.5) / p.w) - 1.0) * tan_half, y = (1.0 - 2.0 * ((j + 0.5) / p.h)) * tan_half;
          const double nrm = std::sqrt(1 + x * x + y * y);
          const auto t = kerr::trace(p.a, p.dist, p.theta_deg * M_PI / 180.0, 1.0 / nrm, y / nrm, -x / nrm, kerr::Disk{rin, p.rout}, kerr::Options{1e-10, 1e-9});
          if (!t.disk_hit || t.disk_r < 1.2 * rin) continue;
          const double g = kerr::redshift(p.a, t.disk_r, t.L, p.dist, p.theta_deg * M_PI / 180.0);
          const double mu = emission::emitter_mu(p.a, t.disk_r, t.L, t.Q);
          const Vec3 col = lut::lookup(lutd, g * p.tpeak * prof.temperature(t.disk_r));
          const double expect = k0 * col.y * (1.0 - std::exp(-tau / mu));
          const double got = h[4 * (size_t(j) * p.w + i) + 1];
          if (expect < 1e-4) continue;
          worst = std::max(worst, std::abs(got - expect) / expect);
          ++n;
        }
      char what[128];
      std::snprintf(what, sizeof what, "face-on, tau = %.1f: I = B(1 - e^(-tau/mu_gas)), max relative error (%d pixels)", tau, n);
      check(n > 20 && worst < 0.03, what, worst, 0.03);
    }
  }
  // 2. inclined image: limits and convergences
  {
    kcuda::Params p = c.p;
    p.w = 192; p.h = 108; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.a = 0.9; p.theta_deg = 75; p.use_sky = false; p.samples = 1; p.fov_deg = 30;
    p.hr = 0.05; p.tau = 10.0; p.vol_sub = 4;
    const auto base = hdr_of(p);
    // tau -> 0: optically thin regime, luminance linear in tau
    kcuda::Params a1 = p, a2 = p; a1.tau = 0.01; a2.tau = 0.02;
    const double thin_ratio = mean_of(hdr_of(a1)) / mean_of(hdr_of(a2));
    check(std::abs(thin_ratio - 0.5) < 0.03, "tau -> 0: L(0.01) / L(0.02) = 1/2 (thin regime)", thin_ratio, 0.5);
    // sub-sample refinement (Hermite per step)
    kcuda::Params s16 = p; s16.vol_sub = 16;
    const double msub = mean_of(hdr_of(s16)) / mean_of(base);
    check(std::abs(msub - 1.0) < 0.01, "vol_sub 4 vs 16: mean luminance within 1 %", msub, 1.0);
    // float vs double
    kcuda::Params dd = p; dd.prec = kcuda::Precision::Double; dd.tol = 1e-8;
    const double mdbl = mean_of(hdr_of(dd)) / mean_of(base);
    check(std::abs(mdbl - 1.0) < 0.01, "float vs double: mean luminance within 1 %", mdbl, 1.0);
    // H/r -> 0 with large tau converges (the thin disk is the limit)
    kcuda::Params h1 = p, h2 = p; h1.hr = 0.004; h1.tau = 300; h2.hr = 0.002; h2.tau = 300;
    const double mh = mean_of(hdr_of(h1)) / mean_of(hdr_of(h2));
    check(std::abs(mh - 1.0) < 0.02, "H/r 0.004 vs 0.002 (tau = 300): converges to the thin disk", mh, 1.0);
    // thicker: slightly more visible area => larger luminance (does not decrease); and finite
    kcuda::Params th = p; th.hr = 0.15;
    const auto ht = hdr_of(th);
    bool finite = true; for (float v : ht) finite &= std::isfinite(v);
    check(finite && mean_of(ht) > 0.0, "H/r = 0.15: finite and positive HDR", mean_of(ht), 0.0);
  }
  return bad;
}

// v0.9c: turbulence, hot spot and flare. Device vs core, spot period, rotation direction consistent with the Doppler.
static int selftest_v09c(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  auto hdr_of = [&](const kcuda::Params& q) { kcuda::render(q); std::vector<float> h; kcuda::download_hdr(h); return h; };
  auto lum = [&](const std::vector<float>& h, size_t i) { return 0.2126 * h[4 * i] + 0.7152 * h[4 * i + 1] + 0.0722 * h[4 * i + 2]; };
  kcuda::Params base = c.p;
  base.w = 192; base.h = 108; base.prec = kcuda::Precision::Float; base.tol = 2e-5; base.a = 0.9; base.theta_deg = 75; base.use_sky = false; base.samples = 1;
  base.fov_deg = 30; base.hr = 0.06; base.tau = 6.0; base.rout = 16;
  // 1. device (float) vs core (double): density factor and spot T factor
  {
    kcuda::Params p = base; p.turb = 0.8; p.turb_scale = 2.0; p.disk_time = 137.0;
    p.spot_r = 9.0; p.spot_phi_deg = 40.0; p.spot_size = 1.3; p.spot_amp = 5.0; p.flare_t = 150.0; p.flare_w = 30.0; p.flare_boost = 2.0;
    std::vector<double> in;
    uint32_t h = 4242u;
    auto rnd = [&]() { h = h * 1664525u + 1013904223u; return double(h >> 8) / 16777216.0; };
    for (int i = 0; i < 3000; ++i) in.insert(in.end(), {6.5 + 9.0 * rnd(), 6.283185307 * rnd(), (rnd() - 0.5) * 0.3});
    std::vector<double> g;
    kcuda::turb_batch(p, in, g);
    turb::Spot sp; sp.r = p.spot_r; sp.phi0 = p.spot_phi_deg * M_PI / 180.0; sp.size = p.spot_size; sp.amp = p.spot_amp; sp.fl_t = p.flare_t; sp.fl_w = p.flare_w; sp.fl_boost = p.flare_boost;
    double wd = 0, ws = 0;
    for (size_t i = 0; i < g.size() / 2; ++i) {
      const double r = in[3 * i], ph = in[3 * i + 1], mu = in[3 * i + 2];
      const double hh = p.hr;  // flare = 0: H/r = hr
      const double d = turb::density_factor(r, ph, mu, p.disk_time, p.a, hh, p.turb, p.turb_scale);
      const double s = turb::spot_T_factor(sp, p.a, r, ph, mu, p.disk_time);
      wd = std::max(wd, std::abs(g[2 * i] - d) / d);
      ws = std::max(ws, std::abs(g[2 * i + 1] - s) / s);
    }
    check(wd < 5e-3, "GPU (float) density factor vs core (double), t = 137, 3000 points", wd, 5e-3);
    check(ws < 1e-4, "GPU spot T factor vs core (with flare)", ws, 1e-4);
  }
  // 2. spot period: at t and t + 2 pi / Omega(r_h) the image is the same; at half a period it is different
  {
    kcuda::Params p = base; p.spot_r = 9.0; p.spot_phi_deg = 100.0; p.spot_size = 1.3; p.spot_amp = 6.0;
    const double T = 2.0 * M_PI / turb::omega(p.spot_r, p.a);
    p.disk_time = 10.0;
    const auto i0 = hdr_of(p);
    p.disk_time = 10.0 + T;
    const auto i1 = hdr_of(p);
    p.disk_time = 10.0 + 0.5 * T;
    const auto i2 = hdr_of(p);
    double d1 = 0, d2 = 0, sc = 0;
    for (size_t i = 0; i < i0.size() / 4; ++i) { d1 += std::abs(lum(i0, i) - lum(i1, i)); d2 += std::abs(lum(i0, i) - lum(i2, i)); sc += lum(i0, i); }
    check(d1 / sc < 1e-4 && d2 / sc > 5e-3, "spot: image(t) = image(t + 2 pi / Omega(r_h)) and != at half a period", d1 / sc, 1e-4);
    std::printf("    period %.2f M; |diff|/sum: full period %.2e, half period %.2e\n", T, d1 / sc, d2 / sc);
  }
  // 3. rotation direction: the spot (starting from the far side) moves towards the approaching (brighter, Doppler) side
  {
    kcuda::Params p = base; p.spot_r = 9.0; p.spot_phi_deg = 180.0; p.spot_size = 1.2; p.spot_amp = 8.0;
    kcuda::Params q = p; q.spot_amp = 0.0;
    const auto b0 = hdr_of(q);
    double xb = 0, sb = 0;
    for (int j = 0; j < p.h; ++j) for (int i = 0; i < p.w; ++i) { const double L = lum(b0, size_t(j) * p.w + i); xb += L * (i + 0.5 - 0.5 * p.w); sb += L; }
    const double side = xb / sb;  // < 0: the left side (approaching) is the brighter one
    auto centroid = [&](double t) {
      p.disk_time = t;
      const auto s = hdr_of(p);
      double x = 0, w = 0;
      for (int j = 0; j < p.h; ++j) for (int i = 0; i < p.w; ++i) { const double d = std::max(0.0, lum(s, size_t(j) * p.w + i) - lum(b0, size_t(j) * p.w + i)); x += d * (i + 0.5 - 0.5 * p.w); w += d; }
      return x / w;
    };
    const double dt = 0.5 / turb::omega(p.spot_r, p.a);  // 0.5 rad
    const double c0 = centroid(0.0), c1 = centroid(dt);
    check((c1 - c0) * side > 0.0 && std::abs(c1 - c0) > 2.0, "spot: moves towards the approaching (brighter, Doppler) side", c1 - c0, 0.0);
    std::printf("    bright side (image centroid) %.1f px; spot centroid %.1f -> %.1f px in 0.5 rad\n", side, c0, c1);
  }
  // 4. turbulence: preserves the mean luminance (lognormal normalisation) but changes the image; evolves in time; finite
  {
    kcuda::Params p = base; p.turb = 0.7; p.disk_time = 0.0;
    const auto t0 = hdr_of(p);
    p.disk_time = 60.0;
    const auto t1 = hdr_of(p);
    kcuda::Params q = base;
    const auto s0 = hdr_of(q);
    double m0 = 0, ms = 0, dt = 0; bool finite = true;
    for (size_t i = 0; i < t0.size() / 4; ++i) { m0 += lum(t0, i); ms += lum(s0, i); dt += std::abs(lum(t0, i) - lum(t1, i)); }
    for (float v : t1) finite &= std::isfinite(v);
    check(std::abs(m0 / ms - 1.0) < 0.10 && finite, "turbulence: mean luminance = smooth +- 10 % (and finite)", m0 / ms, 0.10);
    check(dt / m0 > 1e-3, "turbulence: the image evolves with the disk time (t = 0 vs 60)", dt / m0, 1e-3);
  }
  return bad;
}

static int selftest_v10(const Cli& c);
static int selftest_v22b(const Cli& c);
static int selftest_v22f(const Cli& c);
static double render_route_frame(const route::Route& r, const route::Trajectory& tr, kcuda::Params p, int f, int N, double shutter);
// v0.9c2: slow light. Flight time accumulated on the GPU vs kerr::trace::t_disk (Schwarzschild validated by quadrature in the core), and the
// end-to-end oracle: a uniform flare peaks at each pixel at t_f + t_flight(pixel) - T_ref.
static int selftest_v09c2(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  auto tref_of = [](double D) { return (D + 2.0 * std::log(D / 2.0 - 1.0)) - (10.0 + 2.0 * std::log(4.0)); };
  // 1. GPU flight time (T where the transmittance drops to 1/2, thin opaque layer) vs the CPU's t_disk
  for (double a : {0.0, 0.9}) {
    kcuda::Params p = c.p; p.a = a; p.dist = 60.0; p.theta_deg = a == 0.0 ? 1.0 : 70.0; p.fov_deg = a == 0.0 ? 40.0 : 20.0; p.prec = kcuda::Precision::Double; p.tol = 1e-8;
    p.rout = 40.0; p.hr = 0.0005; p.tau = 100.0; p.vol_sub = 64; p.slowlight = true;
    // T_half is the time where the transmittance drops to 1/2, ~2.5 H ABOVE the plane: the offset ~ 2.5 H / cos(ray inclination) is systematic (measured with
    // hr = 0.004: 0.14 M at a = 0, 0.53 M at a = 0.9 and theta = 70). With hr = 0.0005 it is 8x smaller
    const double rin = kerr::isco(a, true), tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0);
    std::vector<double> rays;
    for (int j = 0; j < 14; ++j)
      for (int i = 0; i < 14; ++i) {
        const double x = (2.0 * (i + 0.5) / 14 - 1.0) * tan_half, y = (2.0 * (j + 0.5) / 14 - 1.0) * tan_half, nrm = std::sqrt(1 + x * x + y * y);
        rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
      }
    std::vector<double> g;
    kcuda::vol_flight_batch(p, rays, g);
    kerr::Options ko; ko.atol = 1e-10; ko.rtol = 1e-9;
    double worst = 0; int n = 0;
    for (size_t k = 0; k < g.size(); ++k) {
      const auto t = kerr::trace(a, p.dist, p.theta_deg * M_PI / 180.0, rays[3 * k], rays[3 * k + 1], rays[3 * k + 2], kerr::Disk{rin, p.rout}, ko);
      if (!t.disk_hit || t.disk_r < 1.15 * rin || g[k] < 0) continue;
      worst = std::max(worst, std::abs(g[k] - t.t_disk) / t.t_disk);
      ++n;
    }
    char what[128];
    std::snprintf(what, sizeof what, "GPU flight time vs kerr::trace t_disk, a = %.1f (%d rays), max relative error", a, n);
    check(n > 15 && worst < 3e-3, what, worst, 3e-3);
  }
  // 2. uniform flare (giant spot): the peak of each pixel is at t_f + t_flight - T_ref (slow light) or at t_f (instantaneous light)
  {
    kcuda::Params p = c.p;
    p.w = 24; p.h = 14; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.a = 0.0; p.dist = 60.0; p.theta_deg = 60.0; p.fov_deg = 16.0; p.use_sky = false; p.samples = 1;
    p.rout = 16; p.hr = 0.01; p.tau = 50.0; p.vol_sub = 4;
    p.spot_r = 10.0; p.spot_phi_deg = 0.0; p.spot_size = 1e4; p.spot_amp = 1.0; p.flare_t = 100.0; p.flare_w = 12.0; p.flare_boost = 20.0;
    const double tref = tref_of(p.dist), tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0), aspect = double(p.w) / p.h, rin = kerr::isco(p.a, true);
    struct Px { int i, j; double shift; };
    std::vector<Px> px;
    for (int j = 1; j < p.h; j += 2)
      for (int i = 1; i < p.w; i += 3) {
        const double x = (2.0 * ((i + 0.5) / p.w) - 1.0) * tan_half * aspect, y = (1.0 - 2.0 * ((j + 0.5) / p.h)) * tan_half, nrm = std::sqrt(1 + x * x + y * y);
        kerr::Options ko; ko.atol = 1e-10; ko.rtol = 1e-9;
        const auto t = kerr::trace(p.a, p.dist, p.theta_deg * M_PI / 180.0, 1.0 / nrm, y / nrm, -x / nrm, kerr::Disk{rin, p.rout}, ko);
        if (t.disk_hit && t.disk_r > 7.0 && t.disk_r < 14.0 && std::abs(t.t_disk - tref) > 3.0) px.push_back({i, j, t.t_disk - tref});
      }
    // time sweep: peak of each pixel by a parabolic fit to the logarithm around the maximum
    const int nt = 41;
    const double t0 = 70.0, dt = 2.0;
    auto peaks = [&](bool slow) {
      kcuda::Params q = p; q.slowlight = slow;
      std::vector<std::vector<double>> v(px.size(), std::vector<double>(nt));
      for (int k = 0; k < nt; ++k) {
        q.disk_time = t0 + dt * k;
        kcuda::render(q);
        std::vector<float> h; kcuda::download_hdr(h);
        for (size_t m = 0; m < px.size(); ++m) { const size_t o = 4 * (size_t(px[m].j) * p.w + px[m].i); v[m][k] = std::log(std::max(1e-12, double(h[o + 1]))); }
      }
      std::vector<double> pk;
      for (size_t m = 0; m < px.size(); ++m) {
        int kb = 1; for (int k = 1; k < nt - 1; ++k) if (v[m][k] > v[m][kb]) kb = k;
        const double y0 = v[m][kb - 1], y1 = v[m][kb], y2 = v[m][kb + 1], den = y0 - 2 * y1 + y2;
        pk.push_back(t0 + dt * (kb + (den != 0 ? 0.5 * (y0 - y2) / den : 0.0)));
      }
      return pk;
    };
    const auto pk_slow = peaks(true), pk_fast = peaks(false);
    double worst_slow = 0, worst_fast = 0, min_shift = 1e9;
    for (size_t m = 0; m < px.size(); ++m) {
      worst_slow = std::max(worst_slow, std::abs(pk_slow[m] - (p.flare_t + px[m].shift)));
      worst_fast = std::max(worst_fast, std::abs(pk_fast[m] - p.flare_t));
      min_shift = std::min(min_shift, std::abs(px[m].shift));
    }
    char what[128];
    std::snprintf(what, sizeof what, "flare: per-pixel peak = t_f + t_flight - T_ref (%zu pixels, deviation |t| > 3 M)", px.size());
    check(px.size() >= 6 && worst_slow < 1.0, what, worst_slow, 1.0);
    check(worst_fast < 1.0, "flare without slow light: the peak is at t_f in all pixels", worst_fast, 1.0);
    std::printf("    measured time offset in the pixels: min |t_flight - T_ref| = %.1f M, flare at t_f = %.0f\n", min_shift, p.flare_t);
  }
  // 3. without time dependence (turb = 0, no spot) slow light does not change the image
  {
    kcuda::Params p = c.p;
    p.w = 128; p.h = 72; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.a = 0.9; p.theta_deg = 75; p.use_sky = false; p.samples = 1; p.hr = 0.06; p.tau = 6.0; p.rout = 16; p.fov_deg = 30;
    kcuda::render(p); std::vector<float> h0; kcuda::download_hdr(h0);
    p.slowlight = true; kcuda::render(p); std::vector<float> h1; kcuda::download_hdr(h1);
    double d = 0, s = 0;
    for (size_t i = 0; i < h0.size(); ++i) { d += std::abs(h0[i] - h1[i]); s += std::abs(h0[i]); }
    check(d / s < 1e-6, "slow light without turbulence or spot: image unchanged", d / s, 1e-6);
  }
  return bad;
}

static int selftest_v08b(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-66s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  std::vector<double> xy;
  for (int j = 0; j < 17; ++j) for (int i = 0; i < 31; ++i) xy.insert(xy.end(), {(i - 15) * 0.0243, (j - 8) * 0.0431});
  const size_t n = xy.size() / 2;
  // 1. default basis: bit-for-bit (1, y, -x)/nrm
  {
    kcuda::Params p = c.p; std::vector<double> out;
    kcuda::camera_rays(p, xy, out);
    // the device may contract 1 + x*x + y*y into an FMA (the host does not): equality to 1e-15; the bit-for-bit proof of the kernel is the md5 golden
    // of tools/regress_v07d.sh (ctest), generated before this generalisation
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
      const double x = xy[2 * i], y = xy[2 * i + 1], nrm = std::sqrt(1.0 + x * x + y * y);
      worst = std::max({worst, std::abs(out[3 * i] - 1.0 / nrm), std::abs(out[3 * i + 1] - y / nrm), std::abs(out[3 * i + 2] + x / nrm)});
    }
    check(worst < 1e-15, "default basis: rays vs (1, y, -x)/nrm", worst, 1e-15);
  }
  // 2. rotated basis (yaw 90, pitch 30, roll 20 degrees) vs the host formula n' = -d'/|d'|; |n'| = 1
  {
    flight::Basis b;
    flight::yaw(b, M_PI / 2); flight::pitch(b, 30 * M_PI / 180); flight::roll(b, 20 * M_PI / 180);
    kcuda::Params p = c.p;
    const Vec3* v[3] = {&b.fwd, &b.right, &b.up};
    double* dst[3] = {p.fwd, p.right, p.up};
    for (int k = 0; k < 3; ++k) { dst[k][0] = v[k]->x; dst[k][1] = v[k]->y; dst[k][2] = v[k]->z; }
    std::vector<double> out;
    kcuda::camera_rays(p, xy, out);
    double worst = 0, nerr = 0;
    for (size_t i = 0; i < n; ++i) {
      const Vec3 d = b.fwd + xy[2 * i] * b.right + xy[2 * i + 1] * b.up;
      const Vec3 ref = (-1.0 / norm(d)) * d;
      const Vec3 g{out[3 * i], out[3 * i + 1], out[3 * i + 2]};
      worst = std::max(worst, norm(g - ref)); nerr = std::max(nerr, std::abs(norm(g) - 1.0));
    }
    check(worst < 1e-14 && nerr < 1e-14, "rotated basis (yaw 90, pitch 30, roll 20): GPU vs host", std::max(worst, nerr), 1e-14);
    const double ortho = std::max({std::abs(dot(b.fwd, b.right)), std::abs(dot(b.fwd, b.up)), std::abs(dot(b.right, b.up)), std::abs(norm(b.fwd) - 1.0)});
    check(ortho < 1e-15, "yaw/pitch/roll rotations preserve orthonormality", ortho, 1e-15);
    // yaw 90 degrees: the central ray now comes from the direction +right/-... : n'(0,0) = -fwd
    const double cx = out[3 * (8 * 31 + 15)], cy = out[3 * (8 * 31 + 15) + 1], cz = out[3 * (8 * 31 + 15) + 2];
    const Vec3 center = -1.0 * b.fwd;
    check(norm(Vec3{cx, cy, cz} - center) < 1e-14, "central ray = -fwd", norm(Vec3{cx, cy, cz} - center), 1e-14);
  }
  // 3a. flight controls (host): thruster, brake, look, capture and reset
  {
    flight::Flight f(0.9, 30.0, 80.0, 0.0, flight::Start::Zamo);
    f.timescale = 5.0;
    flight::Controls fw; fw.fwd = 1;
    for (int i = 0; i < 120; ++i) f.step(1.0 / 60, fw, false);  // 2 s of real time = 10 M at 0.02 c/M
    const auto s1 = f.state();
    check(s1.beta.x < -0.15 && std::abs(s1.beta.y) < 1e-3 && std::abs(s1.beta.z) < 1e-3, "W: forward thruster (-e_r) accelerates ~0.02 c/M x 10 M", -s1.beta.x, 0.15);
    flight::Controls br; br.brake = true;
    double bmax_after = 0;
    for (int i = 0; i < 600; ++i) { f.step(1.0 / 60, br, false); }
    bmax_after = norm(f.state().beta);
    check(bmax_after < 0.01, "Ctrl: the brake cancels the velocity relative to the ZAMO (without reversing)", bmax_after, 0.01);
    flight::Basis b0 = f.basis();
    f.look(M_PI / 2, 0, 0);
    check(norm(f.basis().fwd - b0.right) < 1e-12, "look: a 90-degree yaw puts fwd on the old right", norm(f.basis().fwd - b0.right), 1e-12);
    // capture: fall from r = 2.6 at rest
    flight::Flight g(0.9, 2.6, 85.0, 0.0, flight::Start::Zamo);
    g.timescale = 5.0;
    for (int i = 0; i < 600 && !g.captured(); ++i) g.step(1.0 / 60, flight::Controls{}, false);
    const bool cap = g.captured();
    const double r_stop = g.state().r;
    g.step(1.0 / 60, flight::Controls{}, false);  // paused: nothing changes
    check(cap && g.state().r == r_stop, "horizon: captured() and the simulation stops (Backspace resets)", cap ? 0.0 : 1.0, 0.5);
    g.reset();
    check(!g.captured() && std::abs(g.state().r - 2.6) < 1e-12 && g.state().tau == 0.0, "Backspace: reset restores the initial state", std::abs(g.state().r - 2.6), 1e-12);
  }
  // 3. scripted flight (clip): 900 frames at 60 fps, timescale 8.5: constants conserved between burns, never NaN, never captured;
  //    and the kernel never produces non-finite values along the flight
  {
    flight::Flight fl(0.9, 30.0, 80.0, 0.0, flight::Start::Orbit);
    fl.timescale = 8.5;
    kcuda::Params p = c.p; p.w = 96; p.h = 54; p.a = 0.9; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.samples = 1;
    p.nodisk = false;
    bool nan_seen = false, kernel_bad = false;
    double rmin = 1e9, dE = 0, dL = 0, dQ = 0;
    pilot::Consts c0 = fl.pilot().consts(), c1{};
    { const auto s0 = fl.state(); const auto z = pilot::zamo(0.9, s0.r, s0.theta); const double g0 = 1.0 / std::sqrt(1.0 - dot(s0.beta, s0.beta)); c0.E = g0 * (z.alpha + z.omega * z.varpi * s0.beta.z); c0.L = g0 * z.varpi * s0.beta.z; }
    bool have1 = false;
    const auto& scr = flight::flight_script();
    for (int i = 0; i < 900; ++i) {
      const double tau = fl.state().tau;
      if (i > 0) fl.step_accel(1.0 / 60.0, flight::script_accel(tau));
      const auto s = fl.state();
      nan_seen |= !std::isfinite(s.r) || !std::isfinite(s.theta) || !std::isfinite(s.phi) || !std::isfinite(norm(s.beta));
      rmin = std::min(rmin, s.r);
      // coasting (between burns): the constants do not change
      const bool coast_a = s.tau > 0.1 && s.tau < scr[0].t0 - 0.2;           // before the brake
      const bool coast_b = s.tau > scr[1].t1 + 0.3 && s.tau < scr[2].t0 - 0.3;  // free fall
      if (coast_a || coast_b) {
        auto cc = fl.pilot().consts();
        {  // E and L recomputed from the STATE (r, theta, beta), not just the stored constants: they test the integration
          const auto z = pilot::zamo(0.9, s.r, s.theta);
          const double g = 1.0 / std::sqrt(1.0 - dot(s.beta, s.beta));
          cc.E = g * (z.alpha + z.omega * z.varpi * s.beta.z);
          cc.L = g * z.varpi * s.beta.z;
        }
        if (!have1) { c1 = cc; have1 = true; }
        if (coast_b && have1 && s.tau < scr[1].t1 + 0.6) c1 = cc;  // free-fall reference = 1st sample after the burn
        if (coast_a) { dE = std::max(dE, std::abs(cc.E - c0.E)); dL = std::max(dL, std::abs(cc.L - c0.L)); dQ = std::max(dQ, std::abs(cc.Q - c0.Q)); }
        else { dE = std::max(dE, std::abs(cc.E - c1.E)); dL = std::max(dL, std::abs(cc.L - c1.L)); dQ = std::max(dQ, std::abs(cc.Q - c1.Q)); }
      }
      if (i % 3 == 0) {  // 300 frames rendered
        fill_flight_params(p, fl);
        kcuda::render(p);
        kernel_bad |= !kcuda::accum_finite();
      }
    }
    check(!nan_seen && !fl.captured(), "scripted flight (900 frames): no NaN and no capture", nan_seen ? 1.0 : 0.0, 0.5);
    check(rmin > kerr::horizon(0.9) + 2.0 && rmin < 5.0, "clip periastron at r ~ 4.4 (above the horizon)", rmin, 5.0);
    check(dE < 1e-7 && dL < 1e-6 && dQ < 1e-6, "E, L (from the state) and Q conserved between burns (max |dE|,|dL|,|dQ|)", std::max({dE, dL, dQ}), 1e-6);
    check(!kernel_bad, "kernel: accumulator finite throughout the flight (300 frames rendered)", kernel_bad ? 1.0 : 0.0, 0.5);
  }
  return bad;
}

// v0.8c: OptiX denoiser. Without OptiX (stub) the option must fail with a clear message; with OptiX a synthetic image with known noise
// must end up with a smaller error and the pipeline (HDR average -> denoiser -> bloom -> tone mapping) must run.
static int selftest_denoise(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-66s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  kcuda::Params p = c.p;
  p.w = 192; p.h = 108; p.denoise = true;
  std::string err;
  if (!kcuda::denoise_available()) {
    const bool ok = !kcuda::denoise_enable(p, err) && err.find("not available") != std::string::npos;
    std::printf("  without OptiX (stub): \"%s\"\n", err.c_str());
    check(ok, "without OptiX: --denoise optix fails with a clear message", ok ? 0.0 : 1.0, 0.5);
    return bad;
  }
  std::printf("  %s\n", kcuda::denoise_version());
  // 1. synthetic image: smooth HDR (gradient + bright blob) + known multiplicative Gaussian noise (LCG + Box-Muller)
  const int w = 256, h = 144;
  std::vector<float> clean(size_t(w) * h * 4), noisy(clean.size()), out;
  uint64_t st = 0x9E3779B97F4A7C15ull;
  auto u01 = [&] { st = st * 6364136223846793005ull + 1442695040888963407ull; return ((st >> 11) + 0.5) / 9007199254740992.0; };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double g = 1.0 + 3.0 * std::exp(-((x - 128.0) * (x - 128.0) + (y - 72.0) * (y - 72.0)) / (2.0 * 30.0 * 30.0));
      const double base[3] = {(0.2 + 0.8 * x / w) * g, (0.3 + 0.5 * y / h) * g, (0.1 + 0.4 * double(x + y) / (w + h)) * g};
      for (int k = 0; k < 3; ++k) {
        const double z = std::sqrt(-2.0 * std::log(u01())) * std::cos(2.0 * M_PI * u01());
        clean[(size_t(y) * w + x) * 4 + k] = float(base[k]);
        noisy[(size_t(y) * w + x) * 4 + k] = float(base[k] * (1.0 + 0.25 * z));
      }
      clean[(size_t(y) * w + x) * 4 + 3] = noisy[(size_t(y) * w + x) * 4 + 3] = 1.f;
    }
  if (!kcuda::denoise_image(noisy, w, h, out, err)) { check(false, "denoise_image failed", 1.0, 0.5); std::printf("    %s\n", err.c_str()); return bad + 0; }
  double mn = 0, md = 0;
  for (size_t i = 0; i < clean.size(); i += 4) for (int k = 0; k < 3; ++k) { const double a = noisy[i + k] - clean[i + k], b = out[i + k] - clean[i + k]; mn += a * a; md += b * b; }
  check(md < 0.5 * mn, "synthetic image (25 % noise): denoised MSE < 0.5 x noisy MSE", md / mn, 0.5);
  // 2. full pipeline on a small scene: finite, different from the non-denoised one, time measured
  kcuda::Params q = p;
  q.prec = kcuda::Precision::Float; q.tol = 2e-5; q.a = 0.9; q.theta_deg = 80; q.samples = 2; q.sample_offset = 10; q.denoise = true;
  if (!kcuda::denoise_enable(q, err)) { check(false, "denoise_enable failed", 1.0, 0.5); std::printf("    %s\n", err.c_str()); return bad; }
  std::vector<uint8_t> a1, a2;
  kcuda::render(q); kcuda::download_rgb(a1);
  const float dn_ms = kcuda::last_timing().denoise_ms;
  std::vector<float> hdr;
  kcuda::download_hdr(hdr);
  bool finite = true;
  for (float v : hdr) finite &= std::isfinite(v);
  q.denoise = false; kcuda::render(q); kcuda::download_rgb(a2);
  size_t d = 0;
  for (size_t i = 0; i < a1.size(); ++i) d += a1[i] != a2[i];
  check(finite && dn_ms > 0.f, "pipeline with denoiser: finite HDR and denoiser executed (ms > 0)", dn_ms, 0.0);
  check(d > 0, "the denoiser changes the image (bytes differ from --denoise none)", double(d), 0.5);
  return bad;
}

static int selftest(const Cli& c) {
  int bad = 0;
  for (double a : {0.0, 0.9, 0.998})
    for (double th : {80.0, 86.0}) {
      kcuda::Params p = c.p;
      p.tol = 1e-10;  // equivalence: tight tolerance (the difference to the CPU scales with it, see the notes)
      p.a = a; p.theta_deg = th;
      p.rin = kerr::isco(a, true);
      const int nx = 64, ny = 36;
      const double tan_half = std::tan(0.5 * p.fov_deg * M_PI / 180.0), aspect = 16.0 / 9.0;
      std::vector<double> rays;
      std::vector<std::pair<double, double>> xy;
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
          const double x = (2 * (i + 0.5) / nx - 1) * tan_half * aspect, y = (1 - 2 * (j + 0.5) / ny) * tan_half;
          const double nrm = std::sqrt(1 + x * x + y * y);
          rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
        }
      std::vector<kcuda::RayOut> g;
      kcuda::trace_batch(p, rays, g);
      kerr::Options ko; ko.rtol = p.tol; ko.atol = p.tol * 0.1;
      int fate_bad = 0, n = nx * ny, both_esc = 0, hit_both = 0;
      std::vector<double> dang, drad;
      for (int i = 0; i < n; ++i) {
        const auto t = kerr::trace(a, p.dist, th * M_PI / 180.0, rays[3 * i], rays[3 * i + 1], rays[3 * i + 2], kerr::Disk{p.rin, p.rout}, ko);
        const int cf = t.fate == kerr::Fate::Escaped ? 0 : t.fate == kerr::Fate::Captured ? 1 : 2;
        if (cf != g[i].fate || int(t.disk_hit) != g[i].disk_hit) { ++fate_bad; continue; }
        if (cf == 0) {
          ++both_esc;
          const double dth = std::acos(g[i].mu_end) - t.theta_end;
          double dph = g[i].phi_end - t.phi_end;
          dang.push_back(std::hypot(dth, dph));
        }
        if (t.disk_hit) { ++hit_both; drad.push_back(std::abs(g[i].disk_r - t.disk_r)); }
      }
      std::sort(dang.begin(), dang.end()); std::sort(drad.begin(), drad.end());
      auto q = [](const std::vector<double>& v, double f) { return v.empty() ? 0.0 : v[size_t(f * (v.size() - 1))]; };
      std::printf("a=%.3f th=%2.0f: %d/%d rays with a different fate | escape: n=%d dang p50=%.2e p99=%.2e max=%.2e | disk: n=%d dr p99=%.2e max=%.2e\n",
                  a, th, fate_bad, n, both_esc, q(dang, .5), q(dang, .99), q(dang, 1), hit_both, q(drad, .99), q(drad, 1));
      if (fate_bad > n / 200 || q(dang, .99) > 1e-6 || q(drad, .99) > 1e-6) ++bad;
    }
  bad += selftest_axis(c);
  std::printf("v0.7d:\n");
  bad += selftest_v07d(c);
  std::printf("v0.8b:\n");
  bad += selftest_v08b(c);
  std::printf("v0.9a:\n");
  bad += selftest_v09a(c);
  std::printf("v0.9a2:\n");
  bad += selftest_v09a2(c);
  bad += selftest_v21(c);
  bad += selftest_v21c(c);
  bad += selftest_v21e(c);
  std::printf("float robustness audit (120 scenarios, seed 777)\n"); bad += audit_float(c, 120, 777);
  bad += selftest_v21d(c);
  std::printf("v0.9b:\n");
  bad += selftest_v09b(c);
  std::printf("v0.9c:\n");
  bad += selftest_v09c(c);
  std::printf("v0.9c2:\n");
  bad += selftest_v09c2(c);
  std::printf("v1.0:\n");
  bad += selftest_v10(c);
  std::printf("v2.2b:\n");
  bad += selftest_v22b(c);
  std::printf("v2.2f:\n");
  bad += selftest_v22f(c);
  std::printf("v0.8c:\n");
  bad += selftest_denoise(c);
  std::printf(bad ? "selftest FAILED (%d configurations)\n" : "selftest OK\n", bad);
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------- shot
static int shot(Cli& c) {
  auto& p = c.p;
  if (p.rin < 0) p.rin = kerr::isco(p.a, true);
  apply_observer(p, c.observer);
  std::unique_ptr<flight::Flight> fl;
  if (c.mode == "flight") {  // initial flight state (--start) at the given (dist, theta, phi) position
    fl = std::make_unique<flight::Flight>(p.a, p.dist, p.theta_deg, p.phi_deg, start_of(c.start));
    fl->timescale = c.timescale;
    fill_flight_params(p, *fl);
  }
  if (p.denoise) {
    std::string derr;
    if (!kcuda::denoise_enable(p, derr)) { std::fprintf(stderr, "--denoise optix: %s\n", derr.c_str()); return 1; }
    std::printf("denoiser: %s\n", kcuda::denoise_version());
  }
  kcuda::Stats st;
  float first = kcuda::render(p, &st);
  for (int i = 0; i < 2; ++i) kcuda::render(p);  // warm-up
  double sum = 0, tr = 0, rs = 0, bl = 0, dn = 0;
  for (int i = 0; i < c.runs; ++i) {
    sum += kcuda::render(p);
    const auto t = kcuda::last_timing();
    tr += t.trace_ms; rs += t.resolve_ms; bl += t.bloom_ms; dn += t.denoise_ms;  // trace_ms = last sample only
  }
  std::printf("Kerr a=%.3f theta=%.1f D=%.1f %dx%d tol=%g prec=%s profile=%s observer=%s%s: first %.1f ms; mean of %d: %.3f ms/frame (%.1f fps)\n",
              p.a, p.theta_deg, p.dist, p.w, p.h, p.tol, prec_name(p.prec), p.nt ? "nt" : "newtonian", c.observer.c_str(),
              std::abs(p.beta[0]) + std::abs(p.beta[1]) + std::abs(p.beta[2]) > 0 ? "" : "", first, c.runs, sum / c.runs, 1000.0 * c.runs / sum);
  std::printf("  stages (mean): sample %.3f ms x %d samples | resolve (mean+tone mapping) %.3f ms | bloom %.3f ms | denoiser %.3f ms | tone mapping %s, bloom %g, beaming %g%s\n",
              tr / c.runs, p.samples, rs / c.runs, bl / c.runs, dn / c.runs, p.tonemap == kcuda::Tonemap::Aces ? "aces" : "reinhard", p.bloom, p.beaming,
              c.cinematic ? " [cinematic: AESTHETIC, not physical]" : "");
  std::printf("steps per ray: mean %.1f, max %d; warp efficiency (sum/sum of max*32) %.2f\n", st.steps_mean, st.steps_max, st.warp_eff);
  if (!c.hdr_out.empty()) {  // RGB float PFM (rows from bottom to top, little-endian)
    std::vector<float> px;
    kcuda::download_hdr(px);
    if (FILE* f = std::fopen(c.hdr_out.c_str(), "wb")) {
      std::fprintf(f, "PF\n%d %d\n-1.0\n", p.w, p.h);
      std::vector<float> row(size_t(p.w) * 3);
      for (int j = p.h - 1; j >= 0; --j) {
        for (int i = 0; i < p.w; ++i) for (int k = 0; k < 3; ++k) row[size_t(i) * 3 + k] = px[(size_t(j) * p.w + i) * 4 + k];
        std::fwrite(row.data(), sizeof(float), row.size(), f);
      }
      std::fclose(f);
      std::printf("wrote %s (HDR)\n", c.hdr_out.c_str());
    }
  }
  if (fl && c.hud_offline) draw_flight_hud(*fl, p, p.h);
  std::vector<uint8_t> rgb;
  kcuda::download_rgb(rgb);
  if (!png::write_file(c.shot, png::encode_rgb(p.w, p.h, rgb))) { std::fprintf(stderr, "error writing %s\n", c.shot.c_str()); return 1; }
  std::printf("wrote %s\n", c.shot.c_str());
  return 0;
}

// ---------------------------------------------------------------- video (v0.7e)
static double smooth(double t) { t = std::min(1.0, std::max(0.0, t)); return t * t * (3.0 - 2.0 * t); }

// parameters of frame i of n for each clip
static void clip_frame(const Cli& c, int i, int n, kcuda::Params& p, std::string& obs) {
  const double t = n > 1 ? double(i) / (n - 1) : 0.0;
  obs = "static";
  if (c.clip == "spin") {  // A: spin sweep 0 -> 0.998, fixed camera (theta 80, D 40)
    p.a = 0.998 * smooth(t); p.theta_deg = 80; p.dist = 40; p.phi_deg = 0;
  } else if (c.clip == "orbit") {  // B: camera orbit: theta 100 -> 60 (from below, from the side, from above) and phi 0 -> 360, a = 0.9
    p.a = 0.9; p.dist = 40; p.theta_deg = 100.0 - 40.0 * smooth(t); p.phi_deg = 360.0 * t;
  } else if (c.clip == "disk") {  // D (v0.9c): fixed camera, the disk time runs (turbulence, spot, flare); volumetric-disk defaults if the user did not give --hr
    p.a = 0.9; p.theta_deg = 75; p.dist = 40; p.phi_deg = 0;
  } else if (c.clip == "infall") {  // C: radial infall from r = 40 to r = 4, uniform PROPER time: r = (r0^1.5 - 1.5 sqrt(2) tau)^(2/3)
    const double r0 = 40.0, r1 = 4.0;
    const double tau_total = (r0 * std::sqrt(r0) - r1 * std::sqrt(r1)) / (1.5 * std::sqrt(2.0));
    p.a = 0.9; p.theta_deg = 75; p.phi_deg = 0;
    p.dist = std::pow(std::pow(r0, 1.5) - 1.5 * std::sqrt(2.0) * tau_total * t, 2.0 / 3.0);
    obs = "infall";
  }
  apply_observer(p, obs);
}

static int video(Cli& c) {
  auto& p = c.p;
  const double secs = c.seconds > 0 ? c.seconds : (c.clip == "flight" ? 15.0 : c.clip == "infall" ? 12.0 : 10.0);
  const int n = int(std::lround(secs * c.fps));
  const double disk_t0 = p.disk_time;
  if (c.clip == "disk") {  // disk-clip defaults: volume with turbulence and time running at 20 M/s (period at r = 10: ~10 s)
    if (p.hr <= 0.0) { p.hr = 0.08; p.flare = 0.15; p.tau = 4.0; p.edge_soft = 0.12; if (p.turb <= 0.0) { p.turb = 1.2; p.turb_scale = 1.2; } }
    if (c.disk_rate <= 0.0) c.disk_rate = 20.0;
  }
  char cmd[1024];
  std::snprintf(cmd, sizeof cmd,
                "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -c:v libx264 -preset medium -crf %d "
                "-pix_fmt yuv420p -movflags +faststart \"%s\"",
                p.w, p.h, c.fps, c.crf, c.video.c_str());
  FILE* ff = popen(cmd, "w");
  if (!ff) { std::fprintf(stderr, "could not launch ffmpeg\n"); return 1; }
  if (c.clip == "flight" && !c.cinematic) std::printf("(warning: the flight clip was designed with --cinematic)\n");
  std::printf("video '%s': clip %s, %d frames (%.1f s at %d fps), %dx%d, %s, %d samples/frame, profile %s%s\n", c.video.c_str(), c.clip.c_str(), n, secs,
              c.fps, p.w, p.h, prec_name(p.prec), p.samples, p.nt ? "nt" : "newtonian", c.cinematic ? ", CINEMATIC (aesthetic)" : "");
  const auto t0 = Clock::now();
  std::vector<uint8_t> rgb;
  double gpu_ms = 0;
  if (p.denoise) {
    std::string derr;
    if (!kcuda::denoise_enable(p, derr)) { std::fprintf(stderr, "--denoise optix: %s\n", derr.c_str()); pclose(ff); return 1; }
  }
  std::unique_ptr<flight::Flight> fl;
  double rmin = 1e9, bmax = 0;
  if (c.clip == "flight") {  // the pilot starts at r = 30 on a circular orbit (a = 0.9, theta = 80); everything written in flight::flight_script()
    p.a = 0.9;
    fl = std::make_unique<flight::Flight>(p.a, 30.0, 80.0, 0.0, flight::Start::Orbit);
    fl->timescale = 8.5;  // 15 s -> 127.5 M of proper time
  }
  for (int i = 0; i < n; ++i) {
    std::string obs;
    if (fl) {
      if (i > 0) fl->step_accel(1.0 / c.fps, flight::script_accel(fl->state().tau));
      fill_flight_params(p, *fl);
      const auto s = fl->state();
      rmin = std::min(rmin, s.r); bmax = std::max(bmax, norm(s.beta));
      if (fl->captured()) { std::fprintf(stderr, "the pilot crossed the horizon at frame %d\n", i); pclose(ff); return 1; }
    } else clip_frame(c, i, n, p, obs);
    if (c.disk_rate > 0) p.disk_time = disk_t0 + (double(i) / c.fps) * c.disk_rate;
    p.frame_index = i;
    gpu_ms += kcuda::render(p);
    if (fl && c.hud_offline) draw_flight_hud(*fl, p, p.h);
    kcuda::download_rgb(rgb);
    if (std::fwrite(rgb.data(), 1, rgb.size(), ff) != rgb.size()) { std::fprintf(stderr, "write to ffmpeg failed (frame %d)\n", i); pclose(ff); return 1; }
    if (i % 60 == 0) { std::printf("  frame %d/%d (a=%.3f theta=%.1f phi=%.0f D=%.2f |beta|=%.3f)\n", i, n, p.a, p.theta_deg, p.phi_deg, p.dist, std::abs(p.beta[0]) + std::abs(p.beta[2])); std::fflush(stdout); }
  }
  const int rc = pclose(ff);
  const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
  if (fl) std::printf("flight: r_min = %.3f (horizon %.3f), max |beta| = %.3f, final tau = %.1f, final r = %.2f\n", rmin, kerr::horizon(p.a), bmax, fl->state().tau, fl->state().r);
  std::printf("render+encode: %.1f s de relogio (%.1f ms/frame; GPU %.1f ms/frame), ffmpeg rc=%d\n", wall, 1000.0 * wall / n, gpu_ms / n, rc);
  return rc == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- interactive
struct App {
  Cli c;
  camera::Orbit cam;
  std::unique_ptr<flight::Flight> fl;  // != nullptr <=> flight mode
  bool dragging = false, frozen = false, hud = true, want_reset = false;
  std::string denoise_msg;  // last denoiser error message (shown in the title)
  double lastx = 0, lasty = 0, look_dx = 0, look_dy = 0;  // accumulated mouse movement (px) since the last frame
};
static void enter_flight(App& a) {
  a.fl = std::make_unique<flight::Flight>(a.c.p.a, a.cam.dist, a.cam.theta * 180.0 / M_PI, a.cam.phi * 180.0 / M_PI, start_of(a.c.start));
  a.fl->timescale = a.c.timescale;
}
static void leave_flight(App& a) {
  const auto s = a.fl->state();
  a.cam.dist = s.r; a.cam.theta = s.theta; a.cam.phi = s.phi;
  a.cam.clamp();
  a.fl.reset();
}
static void on_scroll(GLFWwindow* w, double, double dy) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (a->fl) return;  // in flight the wheel does nothing (the distance is a consequence of the trajectory)
  a->cam.dist *= std::exp(-0.08 * dy);
  a->cam.clamp();
}
static void on_button(GLFWwindow* w, int b, int act, int) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (b == GLFW_MOUSE_BUTTON_LEFT) { a->dragging = act == GLFW_PRESS; glfwGetCursorPos(w, &a->lastx, &a->lasty); }
}
static void on_cursor(GLFWwindow* w, double x, double y) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  if (a->dragging) {
    if (a->fl) { a->look_dx += x - a->lastx; a->look_dy += y - a->lasty; }
    else { a->cam.phi -= (x - a->lastx) * 0.005; a->cam.theta -= (y - a->lasty) * 0.005; a->cam.clamp(); }
  }
  a->lastx = x; a->lasty = y;
}
static void on_key(GLFWwindow* w, int key, int, int act, int) {
  if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(w));
  auto& p = a->c.p;
  const bool in_flight = a->fl != nullptr;
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
  else if (key == GLFW_KEY_F && act == GLFW_PRESS) { if (in_flight) leave_flight(*a); else enter_flight(*a); }
  else if (key == GLFW_KEY_H && act == GLFW_PRESS) a->hud = !a->hud;
  else if (key == GLFW_KEY_Z && act == GLFW_PRESS) a->frozen = !a->frozen;
  else if (key == GLFW_KEY_BACKSPACE && act == GLFW_PRESS && in_flight) { a->fl->reset(); a->frozen = false; }
  else if (key == GLFW_KEY_COMMA && in_flight) a->fl->timescale = std::max(0.05, a->fl->timescale / 1.25);
  else if (key == GLFW_KEY_PERIOD && in_flight) a->fl->timescale = std::min(100.0, a->fl->timescale * 1.25);
  else if ((key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) && in_flight) a->fl->level = std::min(6, a->fl->level + 1);
  else if ((key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) && in_flight) a->fl->level = std::max(0, a->fl->level - 1);
  else if (key == GLFW_KEY_PAGE_UP) p.exposure *= 1.1;
  else if (key == GLFW_KEY_PAGE_DOWN) p.exposure /= 1.1;
  else if (key == GLFW_KEY_D && act == GLFW_PRESS && !in_flight) p.nodoppler = !p.nodoppler;
  else if (key == GLFW_KEY_G && act == GLFW_PRESS) p.nodisk = !p.nodisk;
  else if (key == GLFW_KEY_P && act == GLFW_PRESS) {
    p.prec = p.prec == kcuda::Precision::Double ? kcuda::Precision::Float : kcuda::Precision::Double;
    p.tol = p.prec == kcuda::Precision::Double ? 1e-8 : 1e-5;
  }
  else if (key == GLFW_KEY_LEFT_BRACKET && !in_flight) p.a = std::max(0.0, p.a - 0.01);
  else if (key == GLFW_KEY_RIGHT_BRACKET && !in_flight) p.a = std::min(0.998, p.a + 0.01);
  else if (key == GLFW_KEY_UP) p.sky_exposure *= 1.1;
  else if (key == GLFW_KEY_DOWN) p.sky_exposure /= 1.1;
  else if ((key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) && !in_flight) p.exposure *= 1.1;
  else if ((key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) && !in_flight) p.exposure /= 1.1;
  else if (key == GLFW_KEY_N && act == GLFW_PRESS) {  // OptiX denoiser on/off (only if available)
    if (p.denoise) p.denoise = false;
    else {
      std::string derr;
      if (kcuda::denoise_enable(p, derr)) p.denoise = true;
      else { std::fprintf(stderr, "denoiser: %s\n", derr.c_str()); a->denoise_msg = derr; }
    }
  }
  else if (key == GLFW_KEY_T && act == GLFW_PRESS) p.tonemap = p.tonemap == kcuda::Tonemap::Reinhard ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
  else if (key == GLFW_KEY_B && act == GLFW_PRESS) {  // bloom: 0 -> 0.06 -> 0.12 -> 0.25 -> 0.5 -> 0
    const double lv[] = {0, 0.06, 0.12, 0.25, 0.5};
    int k = 0;
    for (int i = 0; i < 5; ++i) if (std::abs(p.bloom - lv[i]) < 1e-9) k = i;
    p.bloom = lv[(k + 1) % 5];
  }
  else if (key == GLFW_KEY_O && act == GLFW_PRESS && !in_flight) a->c.observer = a->c.observer == "static" ? "orbit" : a->c.observer == "orbit" ? "infall" : "static";
  else if (key == GLFW_KEY_C && act == GLFW_PRESS) {  // cinematic (AESTHETIC) preset on/off
    a->c.cinematic = !a->c.cinematic;
    apply_cinematic(p, a->c.cinematic);
  }
}

static int interactive(Cli& c) {
  App a;
  a.c = c;
  a.hud = c.hud;
  if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* win = glfwCreateWindow(c.p.w > 1920 ? 1280 : c.p.w, c.p.w > 1920 ? 720 : c.p.h, "black-hole-sim v0.8c (CUDA)", nullptr, nullptr);
  if (!win) { std::fprintf(stderr, "could not create the OpenGL window\n"); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(c.vsync ? 1 : 0);
  if (!gl::load()) return 1;
  std::printf("GL %s | %s\n", gl::GetString(GL_VERSION), gl::GetString(GL_RENDERER));
  a.cam.theta = c.p.theta_deg * M_PI / 180.0; a.cam.phi = c.p.phi_deg * M_PI / 180.0; a.cam.dist = c.p.dist; a.cam.clamp();
  a.c.p.rin = -1.0;  // ISCO follows the spin
  if (c.mode == "flight") enter_flight(a);

  glfwSetWindowUserPointer(win, &a);
  glfwSetScrollCallback(win, on_scroll);
  glfwSetMouseButtonCallback(win, on_button);
  glfwSetCursorPosCallback(win, on_cursor);
  glfwSetKeyCallback(win, on_key);

  GLuint tex = 0, fbo = 0;
  gl::CreateFramebuffers(1, &fbo);
  int w = 0, h = 0;
  auto make_tex = [&](int nw, int nh) {
    if (tex) { kcuda::gl_unregister(); gl::DeleteTextures(1, &tex); }
    w = nw; h = nh;
    gl::CreateTextures(GL_TEXTURE_2D, 1, &tex);
    gl::TextureStorage2D(tex, 1, GL_RGBA8, w, h);
    gl::NamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, tex, 0);
    std::string err;
    if (!kcuda::gl_register(tex, w, h, err)) { std::fprintf(stderr, "CUDA-GL interop failed: %s\n", err.c_str()); std::exit(1); }
  };
  auto last = Clock::now(), t_start = last, t_frame = last;
  int frames = 0, total = 0, k = 0;
  const int max_samples = c.samples_set ? c.p.samples : 128;  // progressive samples when the scene is still
  std::vector<double> key, key_prev;
  float ms = 0;
  bool last_scaled = false;
  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    int fw, fh;
    glfwGetFramebufferSize(win, &fw, &fh);
    if (fw <= 0 || fh <= 0) { glfwWaitEvents(); continue; }
    if (fw != w || fh != h) make_tex(fw, fh);
    auto& p = a.c.p;
    const auto now = Clock::now();
    const double dt = std::min(0.1, std::chrono::duration<double>(now - t_frame).count());  // limits the jump after pauses
    t_frame = now;
    p.w = w; p.h = h;
    const char* mode_name = "ORBITAL";
    if (a.fl) {
      mode_name = "VOO";
      // continuous keys -> thrust and camera rotation
      flight::Controls ct;
      auto down = [&](int k) { return glfwGetKey(win, k) == GLFW_PRESS; };
      ct.fwd = (down(GLFW_KEY_W) ? 1.0 : 0.0) - (down(GLFW_KEY_S) ? 1.0 : 0.0);
      ct.right = (down(GLFW_KEY_D) ? 1.0 : 0.0) - (down(GLFW_KEY_A) ? 1.0 : 0.0);
      ct.up = (down(GLFW_KEY_SPACE) ? 1.0 : 0.0) - (down(GLFW_KEY_LEFT_SHIFT) ? 1.0 : 0.0);
      ct.brake = down(GLFW_KEY_LEFT_CONTROL);
      const double roll = ((down(GLFW_KEY_E) ? 1.0 : 0.0) - (down(GLFW_KEY_Q) ? 1.0 : 0.0)) * 1.2 * dt;
      a.fl->look(a.look_dx * 0.003, -a.look_dy * 0.003, roll);  // drag right = look right; up = look up
      a.look_dx = a.look_dy = 0;
      a.fl->step(dt, ct, a.frozen);
      fill_flight_params(p, *a.fl);
    } else {
      p.dist = a.cam.dist; p.theta_deg = a.cam.theta * 180.0 / M_PI; p.phi_deg = a.cam.phi * 180.0 / M_PI;
      apply_observer(p, a.c.observer);
      for (int i = 0; i < 3; ++i) { p.fwd[i] = i == 0 ? -1 : 0; p.right[i] = i == 2 ? 1 : 0; p.up[i] = i == 1 ? -1 : 0; }
    }
    // everything that affects the image; any change (mouse, keys, spin, flight) restarts the accumulation
    key = {double(p.w), double(p.h), p.a, p.dist, p.theta_deg, p.phi_deg, p.exposure, p.sky_exposure, double(p.nodisk), double(p.nodoppler),
           double(int(p.prec)), double(p.nt), p.beta[0], p.beta[1], p.beta[2], double(int(p.tonemap)), p.bloom, p.beaming, p.clamp_lum, p.fcol, double(p.limb), double(p.expose_physical), p.edge_soft, p.edge_noise, p.hr, p.tau, p.flare,
           p.fwd[0], p.fwd[1], p.fwd[2], p.right[0], p.right[1], p.right[2], p.up[0], p.up[1], p.up[2], double(a.hud), p.glare, p.grain, p.vignette, p.chroma, double(p.denoise)};
    const bool changed = key != key_prev;
    key_prev = key;
    // reduced resolution only while the scene changes (--move-scale); when it stops, it restarts accumulating at full resolution
    const bool scaled_now = changed && c.move_scale < 1.0;
    const bool restart = changed || last_scaled;
    last_scaled = scaled_now;
    kcuda::Params q = p;
    if (scaled_now) { q.w = std::max(16, int(w * c.move_scale)); q.h = std::max(16, int(h * c.move_scale)); }
    if (restart) { kcuda::set_size(q.w, q.h); kcuda::accum_reset(); k = 0; }
    const bool converged = k >= max_samples && !restart;
    if (!converged) {
      ms = kcuda::accum_sample(q, k++);
      kcuda::resolve(q);
      if (a.hud) {
        if (a.fl) {
          const auto s = a.fl->state();
          const auto cs = flight::center_shift(a.c.p.a, s, a.fl->basis(), kerr::isco(a.c.p.a), p.rout);
          kcuda::hud_draw(flight::hud_text(mode_name, a.c.p.a, s, a.fl->pilot().consts(), cs, flight::accel_level(a.fl->level), a.fl->timescale, a.fl->captured(), a.frozen, true), std::max(1, q.h / 540));
        } else {
          const pilot::State s{p.dist, p.theta_deg * M_PI / 180.0, p.phi_deg * M_PI / 180.0, Vec3{p.beta[0], p.beta[1], p.beta[2]}, 0, 0};
          const auto cs = flight::center_shift(p.a, s, flight::Basis{}, kerr::isco(p.a), p.rout);
          kcuda::hud_draw(flight::hud_text(mode_name, p.a, s, pilot::Consts{p.a, 0, 0, 0}, cs, 0, 0, false, false, false), std::max(1, q.h / 540));
        }
      }
    } else glfwWaitEventsTimeout(0.05);  // converged image: do not spend GPU
    kcuda::gl_copy();
    const bool scaled = kcuda::render_w() != w || kcuda::render_h() != h;
    gl::BlitNamedFramebuffer(fbo, 0, 0, 0, kcuda::render_w(), kcuda::render_h(), 0, h, w, 0, GL_COLOR_BUFFER_BIT, scaled ? GL_LINEAR : GL_NEAREST);
    glfwSwapBuffers(win);
    ++frames; ++total;
    const double dtt = std::chrono::duration<double>(Clock::now() - last).count();
    if (dtt >= 0.5) {
      char t[320];
      const char* cap = a.fl && a.fl->captured() ? " | HORIZON: simulation paused" : "";
      std::snprintf(t, sizeof t, "black-hole-sim v0.8c CUDA %s%s | %.0f fps | sample %.2f ms | N=%d | a=%.3f r=%.1f theta=%.0f %s %s %s bloom=%.2f beaming=%.2f%s%s%s",
                    mode_name, a.c.cinematic ? " [CINEMATIC: aesthetic, not physical]" : "", frames / dtt, ms, std::min(k, max_samples), p.a, p.dist, p.theta_deg,
                    prec_name(p.prec), p.nt ? "NT" : "newt", p.tonemap == kcuda::Tonemap::Aces ? "aces" : "reinhard", p.bloom, p.beaming,
                    p.nodoppler ? " | no Doppler" : "", p.nodisk ? " | no disk" : "", cap);
      if (p.denoise) std::snprintf(t + std::strlen(t), sizeof t - std::strlen(t), " | denoise OptiX");
      else if (!a.denoise_msg.empty()) std::snprintf(t + std::strlen(t), sizeof t - std::strlen(t), " | denoiser unavailable (N)");
      glfwSetWindowTitle(win, t);
      if (c.autoexit) std::printf("%s\n", t);
      frames = 0; last = Clock::now();
    }
    if (c.autoexit && total >= c.autoexit) {
      const double s = std::chrono::duration<double>(Clock::now() - t_start).count();
      std::printf("autoexit: %d frames in %.2f s = %.1f fps mean (%dx%d, move-scale %.2f)\n", total, s, total / s, w, h, c.move_scale);
      break;
    }
  }
  kcuda::gl_unregister();
  glfwTerminate();
  return 0;
}

// ---------------------------------------------------------------- v1.0: offline routes
static bool load_route(const std::string& path, route::Route& r, std::string& err) {
  std::ifstream in(path);
  if (!in) { err = "could not open " + path; return false; }
  std::stringstream ss; ss << in.rdbuf();
  json::Value v;
  if (!json::parse(ss.str(), v, err)) return false;
  return route::from_json(v, r, err);
}

// parameters that are constant along the frame (spin, FOV, exposure, disk) from the route; t = middle of the frame
static void route_frame_params(const route::Route& r, int f, kcuda::Params& p) {
  const double t = (double(f) + 0.5) / r.fps;
  p.a = r.spin_value(t); p.fov_deg = r.fov_value(t); p.exposure = r.exposure_value(t);
  p.frame_index = f;
  const auto& d = r.disk;
  p.hr = d.hr; p.tau = d.tau; p.flare = d.flare; p.turb = d.turb; p.turb_scale = d.turb_scale; p.edge_soft = d.edge_soft; p.edge_noise = d.edge_noise;
  p.spot_r = d.spot_r; p.spot_phi_deg = d.spot_phi; p.spot_size = d.spot_size; p.spot_amp = d.spot_amp;
  p.flare_t = d.flare_t; p.flare_w = d.flare_w; p.flare_boost = d.flare_boost; p.slowlight = d.slowlight;
  p.rin = kerr::isco(p.a, true);
}

// parameters that vary inside the shutter: ship position and velocity, view orientation, disk time
static void route_time_params(const route::Route& r, const route::Trajectory& tr, double t, kcuda::Params& p) {
  const auto s = tr.at(t);
  p.dist = s.r; p.theta_deg = s.theta * 180.0 / M_PI; p.phi_deg = s.phi * 180.0 / M_PI;
  p.beta[0] = s.beta.x; p.beta[1] = s.beta.y; p.beta[2] = s.beta.z;
  Vec3 b[3];
  r.view_basis(t, s, b);  // v1.0b: with "lookat" the basis depends on the ship position
  double* dst[3] = {p.fwd, p.right, p.up};
  for (int k = 0; k < 3; ++k) { dst[k][0] = b[k].x; dst[k][1] = b[k].y; dst[k][2] = b[k].z; }
  p.disk_time = r.disk_time(t, s.t);
}

// one route frame: N jittered sub-pixel samples, each at its own instant inside the shutter; ends up in S.out/out16 (resolve)
static double render_route_frame(const route::Route& r, const route::Trajectory& tr, kcuda::Params p, int f, int N, double shutter) {
  route_frame_params(r, f, p);
  kcuda::set_size(p.w, p.h);  // before the reset: accum_reset only clears the area of the previous size (defect caught by the determinism selftest)
  kcuda::accum_reset();
  double ms = 0;
  for (int k = 0; k < N; ++k) {
    route_time_params(r, tr, route::sample_time(r.fps, f, k, N, shutter), p);
    ms += kcuda::accum_sample(p, k);
  }
  return ms + kcuda::resolve(p);
}

static std::string chunk_name(const std::string& dir, int ci, const std::string& codec) {
  char b[64]; std::snprintf(b, sizeof b, "chunk_%04d.%s", ci, codec == "prores" ? "mov" : "mp4");
  return dir + "/" + b;
}

static int route_main(Cli& c) {
  route::Route r; std::string err;
  if (!load_route(c.route_file, r, err)) { std::fprintf(stderr, "invalid route: %s\n", err.c_str()); return 1; }
  auto& p = c.p;
  if (!c.prec_set) { p.prec = kcuda::Precision::Float; p.tol = 2e-5; }
  if (!c.samples_set) p.samples = 64;
  const int N = p.samples, F = int(std::lround(r.seconds * r.fps));
  const auto traj = route::build_trajectory(r, 16);
  if (traj.captured) { std::fprintf(stderr, "the ship crosses the horizon at %.2f s of video: impossible route\n", traj.t_capture); return 1; }
  kcuda::set_size(p.w, p.h);
  if (!c.route_shot.empty()) {  // preview of one frame
    const int f = std::min(std::max(c.route_frame, 0), F - 1);
    const double ms = render_route_frame(r, traj, p, f, N, c.shutter);
    std::vector<uint8_t> rgb; kcuda::download_rgb(rgb);
    if (!png::write_file(c.route_shot, png::encode_rgb(p.w, p.h, rgb))) { std::fprintf(stderr, "error writing %s\n", c.route_shot.c_str()); return 1; }
    std::printf("route '%s': frame %d/%d, %d samples, shutter %.2f, %.1f ms -> %s\n", r.name.c_str(), f, F, N, c.shutter, ms, c.route_shot.c_str());
    return 0;
  }
  if (traj.acc_max > 0)  // v1.0b: report of the thrusts generated by the waypoints
  {
    double bmax = 0; for (const auto& q : traj.pts) bmax = std::max(bmax, std::sqrt(dot(q.beta, q.beta)));
    std::printf("waypoints: %zu; max proper thrust %.4g c/M, rms %.4g c/M; max position error %.3g M (proper length); max velocity beta = %.3f (gamma %.2f)\n", r.waypoints.size(), traj.acc_max, traj.acc_rms, traj.err_max, bmax, 1.0 / std::sqrt(1.0 - bmax * bmax));
    if (bmax > 0.7) std::printf("WARNING: relativistic speed (beta > 0.7): strong aberration and Doppler, stars turn into streaks; raise ship.timescale or space the waypoints further apart in time\n");
  }
  if (!c.route_exr.empty()) {  // v1.0b: linear HDR frames in EXR (average of the samples, before bloom/tone mapping); resumes by skipping the existing ones
    std::filesystem::create_directories(c.route_exr);
    const int l = c.last < 0 ? F - 1 : std::min(c.last, F - 1);
    int written = 0;
    for (int f = c.first; f <= l; ++f) {
      char nm[64]; std::snprintf(nm, sizeof nm, "frame_%05d.exr", f);
      const std::string path = c.route_exr + "/" + nm;
      if (std::filesystem::exists(path)) continue;
      render_route_frame(r, traj, p, f, N, c.shutter);
      std::vector<float> h4; kcuda::download_hdr(h4);
      std::vector<float> rgbf(size_t(p.w) * p.h * 3);
      for (size_t i = 0; i < size_t(p.w) * p.h; ++i) { rgbf[3 * i] = h4[4 * i]; rgbf[3 * i + 1] = h4[4 * i + 1]; rgbf[3 * i + 2] = h4[4 * i + 2]; }
      if (!sky::save_exr_rgb(path + ".tmp", rgbf.data(), p.w, p.h, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
      std::filesystem::rename(path + ".tmp", path); ++written;
    }
    std::printf("route '%s': %d new EXR frames (float32, linear) in %s (%d..%d)\n", r.name.c_str(), written, c.route_exr.c_str(), c.first, l);
    if (c.route_out.empty()) return 0;
  }
  if (c.route_out.empty()) { std::fprintf(stderr, "missing --route-out out.mp4 (or --route-frame N --route-shot f.png, or --route-exr DIR)\n"); return 1; }
  const std::string dir = c.workdir.empty() ? c.route_out + ".chunks" : c.workdir;
  std::filesystem::create_directories(dir);
  const int last = c.last < 0 ? F - 1 : std::min(c.last, F - 1);
  const int c0 = c.first / c.chunk, c1 = last / c.chunk;
  std::printf("route '%s': %d frames (%.1f s at %d fps), %dx%d, %d samples/frame, shutter %.2f, %s, chunks of %d, folder %s\n", r.name.c_str(), F, r.seconds, r.fps, p.w, p.h, N,
              c.shutter, c.codec.c_str(), c.chunk, dir.c_str());
  const auto t0 = Clock::now();
  int done_frames = 0;
  std::vector<uint16_t> rgb16;
  for (int ci = c0; ci <= c1; ++ci) {
    const int fa = ci * c.chunk, fb = std::min((ci + 1) * c.chunk, F) - 1;
    const std::string file = chunk_name(dir, ci, c.codec), marker = file + ".done";
    if (c.resume && std::filesystem::exists(marker)) { std::printf("  chunk %d (frames %d-%d): already done, skipped\n", ci, fa, fb); continue; }
    char cmd[1536];
    if (c.codec == "prores" && p.hdr)  // v2.5: PQ/BT.2020 values in 16-bit RGB -> BT.2020nc 10-bit YCbCr, limited range, with colour tags
      std::snprintf(cmd, sizeof cmd, "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb48le -s %dx%d -r %d -i - -vf scale=in_range=full:out_range=tv:out_color_matrix=bt2020nc,format=yuv422p10le "
                    "-c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le -colorspace bt2020nc -color_primaries bt2020 -color_trc smpte2084 -color_range tv \"%s\"", p.w, p.h, r.fps, file.c_str());
    else if (c.codec == "prores")
      std::snprintf(cmd, sizeof cmd, "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb48le -s %dx%d -r %d -i - -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \"%s\"", p.w, p.h, r.fps, file.c_str());
    else
      std::snprintf(cmd, sizeof cmd, "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb48le -s %dx%d -r %d -i - -c:v libx265 -preset slow -crf %d -pix_fmt yuv420p10le -tag:v hvc1 "
                    "-x265-params log-level=error \"%s\"", p.w, p.h, r.fps, c.crf16, file.c_str());
    FILE* ff = popen(cmd, "w");
    if (!ff) { std::fprintf(stderr, "could not launch ffmpeg\n"); return 1; }
    for (int f = fa; f <= fb; ++f) {
      render_route_frame(r, traj, p, f, N, c.shutter);
      kcuda::download_rgb16(rgb16);
      if (std::fwrite(rgb16.data(), 2, rgb16.size(), ff) != rgb16.size()) { std::fprintf(stderr, "write to ffmpeg failed (frame %d)\n", f); pclose(ff); return 1; }
      ++done_frames;
      const double el = std::chrono::duration<double>(Clock::now() - t0).count();
      if (done_frames % 10 == 0 || f == fb) { std::printf("  frame %d/%d  (%.1f s, %.2f s/frame)\n", f + 1, F, el, el / done_frames); std::fflush(stdout); }
    }
    if (pclose(ff) != 0) { std::fprintf(stderr, "ffmpeg failed on chunk %d\n", ci); return 1; }
    std::ofstream(marker) << "ok\n";
  }
  if (c.first == 0 && last == F - 1) {  // concatenates without recompression
    const std::string list = dir + "/list.txt";
    { std::ofstream l(list); for (int ci = 0; ci <= (F - 1) / c.chunk; ++ci) l << "file '" << std::filesystem::absolute(chunk_name(dir, ci, c.codec)).string() << "'\n"; }
    char cmd[1536];
    std::snprintf(cmd, sizeof cmd, "ffmpeg -y -loglevel error -f concat -safe 0 -i \"%s\" -c copy \"%s\"", list.c_str(), c.route_out.c_str());
    if (std::system(cmd) != 0) { std::fprintf(stderr, "the concatenation failed\n"); return 1; }
    std::printf("video final: %s\n", c.route_out.c_str());
  }
  const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("render+encode: %.1f s (%d frames rendered%s, %.2f s/frame)\n", wall, done_frames, c.resume ? " in this session" : "", done_frames ? wall / done_frames : 0.0);
  return 0;
}

// v1.0: offline routes. Determinism, equivalence with the interactive render, 16-bit output and motion blur.
static int selftest_v10(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  auto mk = [&](const char* text) {
    json::Value v; std::string err; route::Route r;
    if (!json::parse(text, v, err) || !route::from_json(v, r, err)) { std::printf("  invalid test route: %s\n", err.c_str()); ++bad; }
    return r;
  };
  kcuda::Params base = c.p;
  base.w = 160; base.h = 90; base.prec = kcuda::Precision::Float; base.tol = 2e-5; base.use_sky = false; base.samples = 4;
  // 1. determinism: the same frame of the same route, twice, bit for bit; and equal to the interactive render with the same parameters
  {
    const auto r = mk(R"({"fps": 30, "seconds": 2, "spin": 0.9, "fov": 30, "exposure": 3, "ship": {"mode": "static", "r": 40, "theta": 75, "phi": 0}})");
    const auto tr = route::build_trajectory(r, 16);
    std::vector<uint16_t> a, b, d;
    render_route_frame(r, tr, base, 5, 4, 0.0); kcuda::download_rgb16(a);
    render_route_frame(r, tr, base, 5, 4, 0.0); kcuda::download_rgb16(b);
    size_t diff = 0; for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
    check(diff == 0, "same frame, same route, 2 renders: differing 16-bit values", double(diff), 0.5);
    kcuda::Params q = base; q.a = 0.9; q.fov_deg = 30; q.exposure = 3; q.dist = 40; q.theta_deg = 75; q.phi_deg = 0; q.samples = 4; q.frame_index = 5; q.rin = kerr::isco(0.9, true);
    kcuda::render(q); kcuda::download_rgb16(d);
    size_t d2 = 0; for (size_t i = 0; i < a.size(); ++i) d2 += a[i] != d[i];
    check(d2 == 0, "offline frame (shutter 0) = interactive render with the same parameters", double(d2), 0.5);
    // 2. 16 bit consistent with 8 bit (same value before quantising)
    std::vector<uint8_t> e;
    kcuda::download_rgb(e);
    size_t worst = 0;
    for (size_t i = 0; i < d.size(); ++i) { const long v8 = long(std::lround(double(d[i]) / 257.0)); worst = std::max<size_t>(worst, size_t(std::labs(v8 - long(e[i])))); }
    check(worst <= 1, "16 bit consistent with 8 bit (|round(v16/257) - v8| <= 1)", double(worst), 1.0);
  }
  // 3. motion blur: a fast orbit blurs with the shutter open; mean luminance preserved
  {
    const auto r = mk(R"({"fps": 30, "seconds": 2, "spin": 0.9, "fov": 30, "exposure": 3, "ship": {"mode": "orbit", "r": 10, "theta": 80, "phi": 0, "timescale": 300}})");
    const auto tr = route::build_trajectory(r, 16);
    auto mean_hdr = [&](double shutter, std::vector<float>* keep) {
      render_route_frame(r, tr, base, 10, 8, shutter);
      std::vector<float> h; kcuda::download_hdr(h);
      double s = 0; for (size_t i = 0; i < h.size(); i += 4) s += 0.2126 * h[i] + 0.7152 * h[i + 1] + 0.0722 * h[i + 2];
      if (keep) *keep = h;
      return s / (h.size() / 4);
    };
    std::vector<float> h0, h1;
    const double m0 = mean_hdr(0.0, &h0), m1 = mean_hdr(1.0, &h1);
    double dsum = 0, ssum = 0;
    for (size_t i = 0; i < h0.size(); ++i) { dsum += std::abs(h0[i] - h1[i]); ssum += std::abs(h0[i]); }
    check(dsum / ssum > 5e-3, "motion blur: shutter 1 differs from shutter 0 on a fast orbit", dsum / ssum, 5e-3);
    check(std::abs(m1 / m0 - 1.0) < 0.05, "motion blur preserves the mean luminance (+- 5 %)", m1 / m0, 0.05);
  }
  return bad;
}

// v2.2b: radio mode. Gas state and emission GPU vs core (torus + synchrotron), field angle, I ~ M scaling in the thin limit, Planck limit in the
// thick one and axisymmetry.
static int selftest_v22b(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  kcuda::Params base = c.p;
  base.a = 0.9; base.model = 1; base.torus_rin = 7.0; base.torus_rmax = 12.0; base.torus_gamma = 4.0 / 3.0; base.ne0 = 1e5; base.plasma_beta = 3.0;
  base.theta_e_max = 20.0; base.nu_ghz = 230.0; base.mbh = 6.5e9; base.prec = kcuda::Precision::Float; base.tol = 2e-5; base.use_sky = false;
  torus::Torus tt;
  if (!torus::make(base.a, base.torus_rin, base.torus_rmax, base.torus_gamma, tt)) { check(false, "torus::make failed", 1.0, 0.5); return bad; }
  // 1. gas state and emission: GPU (double) vs core
  {
    std::vector<double> in;
    uint32_t h = 777u;
    auto rnd = [&]() { h = h * 1664525u + 1013904223u; return double(h >> 8) / 16777216.0; };
    int n = 0;
    while (n < 1500) {
      const double r = 8.0 + 20.0 * rnd(), mu = (rnd() - 0.5) * 0.7, L = (rnd() - 0.5) * 12.0;
      if (tt.density(r, std::acos(mu)) < 1e-4) continue;
      in.insert(in.end(), {r, mu, L}); ++n;
    }
    std::vector<kcuda::TorusPoint> g;
    kcuda::torus_batch(base, in, g);
    double wrho = 0, wB = 0, wj = 0, wa = 0, wth = 0; int used = 0;
    for (size_t i = 0; i < g.size(); ++i) {
      if (g[i].rho <= 0) continue;  // GPU refused (thE < 0.3)
      const double r = in[3 * i], th = std::acos(in[3 * i + 1]);
      const double hh = tt.enthalpy(r, th), rho = tt.density(r, th), thI = tt.theta_ion(r, th);
      const double thE = base.theta_e_max * (hh - 1.0) / (tt.h_max - 1.0);
      const double ne = base.ne0 * rho, P = ne * synch::kC * synch::kC * (synch::kMp * thI + synch::kMe * thE), B = std::sqrt(8.0 * M_PI * P / base.plasma_beta);
      const double nu_em = base.nu_ghz * 1e9 / g[i].g;
      const double j = synch::j_I(ne, thE, B, g[i].sinB, nu_em), al = synch::alpha_nu(ne, thE, B, g[i].sinB, nu_em);
      wrho = std::max(wrho, std::abs(g[i].rho - rho) / rho); wth = std::max(wth, std::abs(g[i].thE - thE) / thE); wB = std::max(wB, std::abs(g[i].B - B) / B);
      if (j > 1e-300) { wj = std::max(wj, std::abs(g[i].j - j) / j); wa = std::max(wa, std::abs(g[i].alpha - al) / al); }
      ++used;
    }
    check(used > 200 && wrho < 1e-9 && wth < 1e-9, "density and Theta_e GPU vs core (double)", std::max(wrho, wth), 1e-9);
    check(wB < 1e-9, "magnetic field B GPU vs core", wB, 1e-9);
    check(wj < 1e-6 && wa < 1e-6, "emissivity j and absorption alpha GPU vs core (same g and sinB)", std::max(wj, wa), 1e-6);
    {  // v2.2b3: the same with the Leung fit (K_2 by A&S polynomial in double vs std::cyl_bessel_k)
      kcuda::Params bl = base; bl.emission = 1; bl.theta_e_min = 0.5;
      std::vector<kcuda::TorusPoint> gl; kcuda::torus_batch(bl, in, gl);
      double wjl = 0, wal = 0, jmx = 0; int ul = 0;
      for (auto& q : gl) jmx = std::max(jmx, q.j);
      for (size_t i = 0; i < gl.size(); ++i) {
        if (gl[i].rho <= 0) continue;
        const double r = in[3 * i], th = std::acos(in[3 * i + 1]), hh = tt.enthalpy(r, th), rho = tt.density(r, th), thI = tt.theta_ion(r, th);
        const double thE = base.theta_e_max * (hh - 1.0) / (tt.h_max - 1.0);
        const double ne = base.ne0 * rho, P = ne * synch::kC * synch::kC * (synch::kMp * thI + synch::kMe * thE), B = std::sqrt(8.0 * M_PI * P / base.plasma_beta);
        const double nu_em = base.nu_ghz * 1e9 / gl[i].g;
        const double j = synch::j_I_leung(ne, thE, B, gl[i].sinB, nu_em), al = synch::alpha_nu(ne, thE, B, gl[i].sinB, nu_em, 1);
        if (j > 1e-12 * jmx) { wjl = std::max(wjl, std::abs(gl[i].j - j) / j); wal = std::max(wal, std::abs(gl[i].alpha - al) / al); ++ul; }
      }
      check(ul > 100 && wjl < 2e-6 && wal < 2e-6, "Leung: j and alpha GPU (polynomial K_2) vs core (cyl_bessel_k)", std::max(wjl, wal), 2e-6);
    }
    // the production path (float) vs the reference one (double), at the same points
    std::vector<kcuda::TorusPoint> gf;
    kcuda::torus_batch(base, in, gf, true);
    double frho = 0, fB = 0, fj = 0, fa = 0, jmax_d = 0; int nf = 0, miss = 0;
    for (auto& q : g) jmax_d = std::max(jmax_d, q.j);
    for (size_t i = 0; i < g.size(); ++i) {
      if (g[i].rho <= 0) continue;
      if (gf[i].rho <= 0) { ++miss; continue; }
      frho = std::max(frho, std::abs(gf[i].rho - g[i].rho) / g[i].rho); fB = std::max(fB, std::abs(gf[i].B - g[i].B) / g[i].B);
      if (g[i].j > 1e-12 * jmax_d) {  // below this the float underflows (j ~ 1e-71 ... 1e-166, ~1e-53 of the maximum) and nothing affects an image
        fj = std::max(fj, std::abs(gf[i].j - g[i].j) / g[i].j); fa = std::max(fa, std::abs(gf[i].alpha - g[i].alpha) / g[i].alpha);
      }
      ++nf;
    }
    std::printf("    float vs double: %d points (%d refused only in float); max error: rho %.1e, B %.1e, j %.1e, alpha %.1e\n", nf, miss, frho, fB, fj, fa);
    check(nf > 200 && miss <= nf / 100 && frho < 1e-3 && fB < 1e-3 && fj < 5e-3 && fa < 5e-3, "float path (production) vs double (reference)", std::max({frho, fB, fj, fa}), 5e-3);
  }
  // 2. field angle: for a photon with L = 0 the ZAMO momentum has no phi component, so in the gas (which moves at gamma along phi)
  //    cos(theta_B) = -v and sin(theta_B) = 1/gamma, gamma = alpha u^t
  {
    std::vector<double> in;
    for (double r : {9.0, 12.0, 16.0}) in.insert(in.end(), {r, 0.0, 0.0});
    std::vector<kcuda::TorusPoint> g;
    kcuda::torus_batch(base, in, g);
    double worst = 0;
    for (size_t i = 0; i < g.size(); ++i) {
      const double r = in[3 * i], a = base.a, Sig = r * r, Del = r * r - 2 * r + a * a, A = (r * r + a * a) * (r * r + a * a) - a * a * Del;
      const auto u = tt.four_velocity(r, M_PI / 2);
      const double gamma = std::sqrt(Sig * Del / A) * u.ut;
      worst = std::max(worst, std::abs(g[i].sinB * gamma - 1.0));
    }
    check(worst < 1e-9, "L = 0: sin(theta_B) = 1/gamma (aberration of the toroidal field)", worst, 1e-9);
  }
  {  // v2.2b3: angular scale and flux in Jy. a = 0: |b| = r sin(alpha_loc)/sqrt(1-2/r) exact (Schwarzschild, static observer)
    kcuda::Params q = base; q.a = 0.0; q.dist = 150.0; q.theta_deg = 40.0; q.fov_deg = 12.0; q.w = 32; q.h = 32; q.rout = 16.0; q.prec = kcuda::Precision::Float; q.ne0 = 1e2; q.samples = 1;
    std::vector<double> I, ab; kcuda::radio_image(q, 1, I, ab);
    const double th = std::tan(0.5 * q.fov_deg * M_PI / 180.0); double worst = 0;
    for (int j = 0; j < q.h; ++j) for (int i = 0; i < q.w; ++i) {
      const double x = (2.0 * (i + 0.5) / q.w - 1.0) * th, y = (1.0 - 2.0 * (j + 0.5) / q.h) * th;
      const double sa = std::sqrt((x * x + y * y) / (1 + x * x + y * y)), bex = q.dist * sa / std::sqrt(1.0 - 2.0 / q.dist);
      const double bgot = std::hypot(ab[2 * (size_t(j) * q.w + i)], ab[2 * (size_t(j) * q.w + i) + 1]);
      worst = std::max(worst, std::fabs(bgot - bex) / std::max(bex, 1e-9));
    }
    check(worst < 1e-6, "radio-cal: |(alpha,beta)| = r sin(a_loc)/sqrt(1-2/r) (a = 0)", worst, 1e-6);
  }
  {  // flux in the thin limit: F ~ M^3 / D^2 (I ~ M by the path, dOmega ~ M^2/D^2)
    kcuda::Params q = base; q.dist = 150.0; q.theta_deg = 17.0; q.fov_deg = 12.0; q.w = 48; q.h = 48; q.rout = 16.0; q.prec = kcuda::Precision::Float; q.ne0 = 1e2; q.samples = 1;
    auto flux = [&](double mbh, double D_cm) { kcuda::Params r = q; r.mbh = mbh; return radio_measure(r, 1, D_cm).flux_jy; };
    const double F1 = flux(1e9, 3e24), F2 = flux(2e9, 3e24), F3 = flux(1e9, 6e24);
    check(std::fabs(F2 / F1 / 8.0 - 1.0) < 2e-3, "radio-cal: F(2M)/F(M) = 8 (thin limit)", F2 / F1, 8.0);
    check(std::fabs(F3 / F1 / 0.25 - 1.0) < 1e-9, "radio-cal: F(2D)/F(D) = 1/4", F3 / F1, 0.25);
  }
  // 3. image: rays from a distant camera
  kcuda::Params ip = base;
  ip.dist = 150.0; ip.theta_deg = 17.0; ip.fov_deg = 12.0; ip.rout = 16.0; ip.prec = kcuda::Precision::Float;
  const double tan_half = std::tan(0.5 * ip.fov_deg * M_PI / 180.0);
  std::vector<double> rays;
  for (int j = 0; j < 24; ++j) for (int i = 0; i < 24; ++i) {
    const double x = (2.0 * (i + 0.5) / 24 - 1.0) * tan_half, y = (1.0 - 2.0 * (j + 0.5) / 24) * tan_half, nrm = std::sqrt(1 + x * x + y * y);
    rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
  }
  auto total = [&](const kcuda::Params& q, std::vector<double>* keep = nullptr) {
    std::vector<double> I; kcuda::radio_intensity_batch(q, rays, I);
    double s = 0; for (double v : I) s += v;
    if (keep) *keep = I;
    return s;
  };
  {  // thin limit: I ~ M (path length in cm ~ GM/c^2) without absorption
    kcuda::Params a1 = ip, a2 = ip; a1.ne0 = 1e2; a2.ne0 = 1e2; a1.mbh = 1e9; a2.mbh = 2e9;
    const double r = total(a2) / total(a1);
    check(total(a1) > 0 && std::abs(r - 2.0) < 0.02, "thin limit (ne0 = 1e2): I(2 M) / I(M) = 2 (path length)", r, 2.0);
  }
  {  // thick limit: T_b = I c^2 / (2 k nu^2) <= g T_e,max with g <~ 2 (black body by Kirchhoff)
    kcuda::Params th = ip; th.ne0 = 1e9;
    std::vector<double> I; total(th, &I);
    double tbmax = 0; for (double v : I) tbmax = std::max(tbmax, synch::brightness_temperature(v, th.nu_ghz * 1e9));
    const double Te_max = th.theta_e_max * synch::kMe * synch::kC * synch::kC / synch::kKb;
    check(tbmax > 0.1 * Te_max && tbmax < 2.0 * Te_max, "thick limit (ne0 = 1e9): 0.1 T_e,max < T_b,max < 2 T_e,max (Planck)", tbmax / Te_max, 2.0);
  }
  {  // axisymmetry: rotating the camera in azimuth does not change the image
    kcuda::Params b1 = ip, b2 = ip; b2.phi_deg = 100.0;
    std::vector<double> I1, I2; total(b1, &I1); total(b2, &I2);
    double d = 0, s = 0; for (size_t i = 0; i < I1.size(); ++i) { d += std::abs(I1[i] - I2[i]); s += I1[i]; }
    check(s > 0 && d / s < 2e-3, "axisymmetry: camera at phi = 0 and 100 degrees, equal images", d / s, 2e-3);
  }
  {  // Doppler beaming: in the thin limit the approaching side (left, x < 0) is brighter
    kcuda::Params dp = ip; dp.ne0 = 1e2;
    std::vector<double> I; total(dp, &I);
    double L = 0, R = 0;
    for (int j = 0; j < 24; ++j) for (int i = 0; i < 24; ++i) (i < 12 ? L : R) += I[size_t(j) * 24 + i];
    check(L > 1.2 * R, "Doppler beaming: approaching side brighter (> 1.2x)", L / R, 1.2);
  }
  return bad;
}

// v2.2f: GRMHD reader. Needs a real .bgm (BHS_GRMHD_TEST or $BHS_DATA/bgm/MAD_a0_5500.bgm); skipped without it.
static int selftest_v22f(const Cli& c) {
  int bad = 0;
  auto check = [&](bool ok, const char* what, double v, double lim) {
    std::printf("  %-62s %.3e (limit %.1e) %s\n", what, v, lim, ok ? "" : "FAILED");
    bad += !ok;
  };
  const char* env = std::getenv("BHS_GRMHD_TEST");
  const std::string path = env ? env : BHS_DATA_DEFAULT "/bgm/MAD_a0_5500.bgm";
  grmhd::Field f; std::string err;
  if (!grmhd::load_bgm(path, f, err)) { std::printf("  (skipped: %s)\n", err.c_str()); return 0; }
  std::string e2;
  if (!kcuda::grmhd_loaded() && !kcuda::grmhd_load(path, e2)) { std::printf("  (skipped: %s)\n", e2.c_str()); return 0; }
  kcuda::Params p = c.p;
  p.model = 2; p.a = f.h.a; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.mbh = 6.5e9; p.nu_ghz = 230.0; p.use_sky = false;
  const auto fm = f.h.fmks();
  // 0. inverse FMKS map on the device: (r, mu = cos theta(X1, X2)) -> X2, error in cell fractions
  {
    std::vector<double> in, X2e; std::vector<double> X1e;
    for (int i = 4; i < int(f.h.n1) - 4; i += 9) for (int j = 3; j < int(f.h.n2) - 3; j += 5) {
      const double X1 = f.h.startx1 + (i + 0.5) * f.h.dx1, X2 = (j + 0.5) * f.h.dx2;
      in.insert(in.end(), {std::exp(X1), std::cos(fm.theta(X1, X2))}); X1e.push_back(X1); X2e.push_back(X2);
    }
    std::vector<float> g;
    kcuda::grmhd_map_batch(p, in, g);
    double w1 = 0, w2 = 0, worst_theta = 0;
    for (size_t n = 0; n < X2e.size(); ++n) { w1 = std::max(w1, std::abs(g[2 * n] - X1e[n]) / f.h.dx1); w2 = std::max(w2, std::abs(g[2 * n + 1] - X2e[n]) / f.h.dx2); }
    (void)worst_theta;
    check(w1 < 1e-2 && w2 < 1e-2, "inverse FMKS map on the device: error in X1 and X2 (in cell fractions)", std::max(w1, w2), 1e-2);
  }
  // 0b. (v2.2h) LUT + 2 Newton vs exact 40-iteration bisection, with dense points next to the poles (X2 in [0, 0.02] and [0.98, 1]) and over the whole r range
  {
    std::vector<double> in; std::vector<double> X1e;
    uint32_t hh = 2024u;
    auto u01 = [&]() { hh = hh * 1664525u + 1013904223u; return ((hh >> 8) & 0xFFFFFF) / double(0x1000000); };
    for (int n = 0; n < 6000; ++n) {
      const double X1 = f.h.startx1 + (0.01 + 0.98 * u01()) * f.h.n1 * f.h.dx1;
      const double w = u01(), X2 = n % 3 == 0 ? 0.02 * w : n % 3 == 1 ? 1.0 - 0.02 * w : 0.02 + 0.96 * w;
      in.insert(in.end(), {std::exp(X1), std::cos(fm.theta(X1, X2))});
    }
    std::vector<float> gl, ge;
    kcuda::Params pe = p; pe.theta_lut = false;
    kcuda::grmhd_map_batch(p, in, gl);
    kcuda::grmhd_map_batch(pe, in, ge);
    double w2 = 0;
    for (size_t n = 0; n < in.size() / 2; ++n) w2 = std::max(w2, std::abs(double(gl[2 * n + 1]) - ge[2 * n + 1]) / f.h.dx2);
    check(w2 < 2e-2, "LUT + 2 Newton vs exact bisection (X2, with points near the poles; cell fraction)", w2, 2e-2);
  }
  // 1. at the cell centres the texture returns the stored values (direct FMKS map on the host, inverse on the device)
  {
    std::vector<double> in; std::vector<std::array<int, 3>> cells;
    uint32_t h = 31337u;
    auto rnd = [&](uint32_t n) { h = h * 1664525u + 1013904223u; return (h >> 8) % n; };
    for (int n = 0; n < 2000; ++n) {
      const int i = 2 + int(rnd(f.h.n1 - 4)), j = 2 + int(rnd(f.h.n2 - 4)), k = int(rnd(f.h.n3));
      const double X1 = f.h.startx1 + (i + 0.5) * f.h.dx1, X2 = (j + 0.5) * f.h.dx2, r = std::exp(X1), th = fm.theta(X1, X2), ph = (k + 0.5) * f.h.dx3;
      // mu = cos(theta) in float loses precision near the poles (acosf: d(theta) ~ sqrt(2 eps) ~ 4e-4 rad ~ 0.04 cell): outside the strict test |cos theta| > 0.995
      if (std::abs(std::cos(th)) > 0.995) { --n; continue; }
      in.insert(in.end(), {r, std::cos(th), ph}); cells.push_back({i, j, k});
    }
    std::vector<float> g;
    kcuda::grmhd_sample_batch(p, in, g);
    double wr = 0, wb = 0, wu = 0;
    for (size_t n = 0; n < cells.size(); ++n) {
      const auto [i, j, k] = cells[n];
      for (int v = 0; v < 4; ++v) { const double s = f.at(i, j, k, v), t = g[12 * n + v]; wr = std::max(wr, std::abs(t - s) / (std::abs(s) + 1e-30)); }
      for (int v = 4; v < 8; ++v) wu = std::max(wu, double(std::abs(g[12 * n + v] - f.at(i, j, k, v))));
      for (int v = 8; v < 12; ++v) wb = std::max(wb, double(std::abs(g[12 * n + v] - f.at(i, j, k, v))));
    }
    check(wr < 1e-4, "cell centres (hardware filter): rho, theta_i, |b|, beta = stored values (rel.)", wr, 1e-4);
    check(wu < 1e-5 && wb < 1e-5, "cell centres: u^mu and bhat^mu = stored values (abs.)", std::max(wu, wb), 1e-5);
    std::printf("    (2000 cells outside |cos theta| > 0.995; the inversion of theta(X2) needs ~40 iterations: with 14 there was 0.017 cells of error near the pole)\n");
  }
  // 2. invariants at arbitrary (interpolated) points: u.u = -1, b.u = 0, b.b = 1 with the KS metric
  {
    std::vector<double> in;
    uint32_t h = 4242u;
    auto rr = [&]() { h = h * 1664525u + 1013904223u; return double(h >> 8) / 16777216.0; };
    for (int n = 0; n < 3000; ++n) in.insert(in.end(), {2.2 + 40.0 * rr(), 1.8 * rr() - 0.9, 6.2831853 * rr()});
    std::vector<float> g;
    kcuda::grmhd_sample_batch(p, in, g);
    double wuu = 0, wbu = 0, wbb = 0; int used = 0;
    for (size_t n = 0; n < in.size() / 3; ++n) {
      const float* q = &g[12 * n];
      if (!std::isfinite(q[0])) continue;
      const double r = in[3 * n], mu = in[3 * n + 1], a = f.h.a, s2 = 1 - mu * mu, S = r * r + a * a * mu * mu, z = 2 * r / S;
      double G[4][4] = {};
      G[0][0] = -1 + z; G[0][1] = G[1][0] = z; G[0][3] = G[3][0] = -a * s2 * z; G[1][1] = 1 + z; G[1][3] = G[3][1] = -a * s2 * (1 + z); G[2][2] = S; G[3][3] = s2 * (S + a * a * s2 * (1 + z));
      auto dotg = [&](const float* x, const float* y) { double s = 0; for (int i2 = 0; i2 < 4; ++i2) for (int j2 = 0; j2 < 4; ++j2) s += G[i2][j2] * double(x[i2]) * double(y[j2]); return s; };
      wuu = std::max(wuu, std::abs(dotg(q + 4, q + 4) + 1.0)); wbu = std::max(wbu, std::abs(dotg(q + 8, q + 4))); wbb = std::max(wbb, std::abs(dotg(q + 8, q + 8) - 1.0));
      ++used;
    }
    check(used > 1000 && wuu < 1e-4 && wbu < 1e-4 && wbb < 1e-4, "interpolated: u.u = -1, b.u = 0, b.b = 1 (KS metric, max error)", std::max({wuu, wbu, wbb}), 1e-4);
    std::printf("    (%d points inside the grid; u.u %.1e, b.u %.1e, b.b %.1e)\n", used, wuu, wbu, wbb);
  }
  // 3. photon null condition in Kerr-Schild along real rays: g^{mu nu} p_mu p_nu = 0 (validates p_r^KS, p_theta, Mino derivatives and the sign of tau')
  {
    kcuda::Params ip = p; ip.dist = 100.0; ip.theta_deg = 60.0; ip.fov_deg = 10.0; ip.rout = 16.0;
    const double tan_half = std::tan(0.5 * ip.fov_deg * M_PI / 180.0);
    std::vector<double> rays;
    for (int j = 0; j < 16; ++j) for (int i = 0; i < 16; ++i) {
      const double x = (2.0 * (i + 0.5) / 16 - 1.0) * tan_half, y = (1.0 - 2.0 * (j + 0.5) / 16) * tan_half, nrm = std::sqrt(1 + x * x + y * y);
      rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
    }
    std::vector<double> ne;
    kcuda::grmhd_null_batch(ip, rays, ne);
    double worst = 0; int nz = 0;
    for (double v : ne) { worst = std::max(worst, v); nz += v > 0; }
    check(nz > 100 && worst < 5e-3, "photon null condition in KS along 256 rays (max |g^{mn} p_m p_n| / p_t^2)", worst, 5e-3);
  }
  // 4. END-TO-END ORACLE: the analytic torus (--disk-model torus, Boyer-Lindquist, v2.2b) against the SAME torus written by tools/make_torus_dump.py
  //    into a .bgm (FMKS, Kerr-Schild) and read by the GRMHD reader. Apart from the discretisation, it has to give the same specific intensity per ray.
  {
    const char* envs = std::getenv("BHS_GRMHD_SYNTH");
    const std::string spath = envs ? envs : BHS_DATA_DEFAULT "/bgm/torus_a0_synth.bgm";
    grmhd::Field fs; std::string es;
    if (!grmhd::load_bgm(spath, fs, es)) { std::printf("  (torus oracle: skipped, %s is missing: uv run --with numpy python -I tools/make_torus_dump.py %s 7 12 20 3)\n", spath.c_str(), spath.c_str()); return bad; }
    std::string e4;
    if (!kcuda::grmhd_load(spath, e4)) { std::printf("  (skipped: %s)\n", e4.c_str()); return bad; }
    torus::Torus tt;
    if (!torus::make(0.0, 7.0, 12.0, 4.0 / 3.0, tt)) { check(false, "torus::make", 1.0, 0.5); return bad; }
    const double ne0 = 3e4, L = 1.476625e5 * 6.5e9, thI_max = (4.0 / 3.0 - 1.0) / (4.0 / 3.0) * (tt.h_max - 1.0);
    kcuda::Params base2 = c.p;
    base2.a = 0.0; base2.prec = kcuda::Precision::Float; base2.tol = 2e-5; base2.use_sky = false; base2.dist = 150.0; base2.theta_deg = 17.0; base2.fov_deg = 14.0; base2.rout = 16.0;
    base2.mbh = 6.5e9; base2.nu_ghz = 230.0;
    kcuda::Params pt = base2, pg = base2;
    pt.model = 1; pt.torus_rin = 7.0; pt.torus_rmax = 12.0; pt.ne0 = ne0; pt.plasma_beta = 3.0; pt.theta_e_max = 20.0; pt.torus_step = 0.3;
    pg.model = 2; pg.munit = ne0 * (synch::kMp + synch::kMe) * L * L * L; pg.electrons = 2; pg.th_e_ratio = 20.0 / thI_max; pg.grmhd_step = 0.02;
    const double tan_half = std::tan(0.5 * base2.fov_deg * M_PI / 180.0);
    std::vector<double> rays;
    for (int j = 0; j < 32; ++j) for (int i = 0; i < 32; ++i) {
      const double x = (2.0 * (i + 0.5) / 32 - 1.0) * tan_half, y = (1.0 - 2.0 * (j + 0.5) / 32) * tan_half, nrm = std::sqrt(1 + x * x + y * y);
      rays.insert(rays.end(), {1.0 / nrm, y / nrm, -x / nrm});
    }
    std::vector<double> It, Ig;
    kcuda::radio_intensity_batch(pt, rays, It); kcuda::radio_intensity_batch(pg, rays, Ig);
    double st = 0, sg = 0, imax = 0;
    for (size_t i = 0; i < It.size(); ++i) { st += It[i]; sg += Ig[i]; imax = std::max(imax, It[i]); }
    std::vector<double> rel;
    for (size_t i = 0; i < It.size(); ++i) if (It[i] > 1e-2 * imax) rel.push_back(std::abs(Ig[i] - It[i]) / It[i]);
    std::sort(rel.begin(), rel.end());
    const double med = rel.empty() ? 1.0 : rel[rel.size() / 2], p95 = rel.empty() ? 1.0 : rel[size_t(0.95 * (rel.size() - 1))];
    check(st > 0 && std::abs(sg / st - 1.0) < 0.01, "torus: total intensity, GRMHD reader (KS, grid) / analytic (BL), ratio - 1", std::abs(sg / st - 1.0), 0.01);
    check(med < 0.02 && p95 < 0.10, "torus: relative error per ray, median < 2 % and p95 < 10 %", med, 0.02);
    std::printf("    %zu rays with I > 1 %% of the maximum: per-ray error median %.2e, p95 %.2e; ratio of totals %.5f\n", rel.size(), med, p95, sg / st);
  }
  return bad;
}

// ---------------------------------------------------------------- v2.2b3: radio-mode calibration
// Total flux F = sum I_pix dOmega_pix (Jy), with dOmega = |d(alpha,beta)/d(px,py)| (GM/c^2 / D)^2; (alpha,beta) = celestial coordinates
// of the ray of each pixel centre (there is no flat-field approximation: they are the ray's constants L, p_theta).
static double g_ring_win = 8.0;  // maximum radius (M) of the ring search
static RadioStats radio_measure(const kcuda::Params& p, int ss, double D_cm, std::vector<float>* pfm) {
  std::vector<double> I, ab;
  kcuda::radio_image(p, ss, I, ab);
  const int w = p.w, h = p.h;
  const double GM = p.mbh * 1.476625e5;                 // cm
  const double M_rad = GM / D_cm;                       // rad por M
  const double uas_per_M = M_rad * (180.0 / M_PI) * 3600.0 * 1e6;
  auto A = [&](int i, int j, int k) { return ab[2 * (size_t(j) * w + i) + k]; };
  std::vector<double> area(size_t(w) * h);              // em M^2
  for (int j = 0; j < h; ++j) for (int i = 0; i < w; ++i) {
    const int i0 = std::max(0, i - 1), i1 = std::min(w - 1, i + 1), j0 = std::max(0, j - 1), j1 = std::min(h - 1, j + 1);
    const double dax = (A(i1, j, 0) - A(i0, j, 0)) / (i1 - i0), dbx = (A(i1, j, 1) - A(i0, j, 1)) / (i1 - i0);
    const double day = (A(i, j1, 0) - A(i, j0, 0)) / (j1 - j0), dby = (A(i, j1, 1) - A(i, j0, 1)) / (j1 - j0);
    area[size_t(j) * w + i] = std::fabs(dax * dby - day * dbx);
  }
  RadioStats st; st.uas_per_M = uas_per_M;
  const double nu = p.nu_ghz * 1e9, kB = 1.380649e-16, cL = 2.99792458e10;
  double fx = 0, fy = 0;
  for (size_t q = 0; q < I.size(); ++q) {
    const double dom = area[q] * M_rad * M_rad;          // sr
    const double F = I[q] * dom * 1e23;                   // Jy
    st.flux_jy += F;
    fx += F * ab[2 * q]; fy += F * ab[2 * q + 1];
    const double iu = I[q] * 1e23 * std::pow(M_PI / 180.0 / 3600.0 / 1e6, 2);  // Jy/uas^2
    if (iu > st.peak_jy_uas2) { st.peak_jy_uas2 = iu; st.peak_tb = I[q] * cL * cL / (2.0 * kB * nu * nu); }
    if (pfm) (*pfm)[q] = float(iu);
  }
  if (st.flux_jy > 0) { st.cx = fx / st.flux_jy; st.cy = fy / st.flux_jy; }
  // ring diameter in the spirit of REx (EHT Paper IV): for each candidate centre (grid +-3 M, step 0.2 M) the azimuth-averaged radial profile (0.1 M bins);
  // the ring is the peak of that profile (r >= 2.5 M, so as not to catch the base of the jet) and the chosen centre is the one that maximises the peak. The previous version (peak around the flux
  // centroid) gave 1-6 M depending on the snapshot; this one gives 9.70 M = 37.0 uas in 5 of 6 dumps (Kerr critical curve a = 0.9375, i = 17: 9.79 M)
  const int nb = 400; double bestpk = 0, bestr = 0, bcx = 0, bcy = 0;
  std::vector<double> sum(nb), cnt(nb);
  for (double oy = -3.0; oy <= 3.0 + 1e-9; oy += 0.2)
    for (double ox = -3.0; ox <= 3.0 + 1e-9; ox += 0.2) {
      std::fill(sum.begin(), sum.end(), 0.0); std::fill(cnt.begin(), cnt.end(), 0.0);
      for (size_t q = 0; q < I.size(); ++q) {
        const double r = std::hypot(ab[2 * q] - ox, ab[2 * q + 1] - oy);
        const int b = int(r / 0.1);
        if (b < nb) { sum[b] += I[q]; cnt[b] += 1; }
      }
      for (int b = 25; b < std::min(nb - 2, int(g_ring_win / 0.1)); ++b) if (cnt[b] > 3) {
        const double m = (sum[b - 1] + sum[b] + sum[b + 1]) / std::max(1.0, cnt[b - 1] + cnt[b] + cnt[b + 1]);
        if (m > bestpk) { bestpk = m; bestr = (b + 0.5) * 0.1; bcx = ox; bcy = oy; }
      }
    }
  st.cx = bcx; st.cy = bcy;
  st.ring_diam_M = 2.0 * bestr; st.ring_diam_uas = st.ring_diam_M * uas_per_M;
  return st;
}

static int radio_cal(Cli& c) {
  auto& p = c.p;
  if (p.model < 1) { std::fprintf(stderr, "--radio-cal needs --disk-model torus or grmhd\n"); return 1; }
  if (c.d_pc <= 0) { std::fprintf(stderr, "--radio-cal needs --d-pc (distance in pc)\n"); return 1; }
  p.rin = -1; p.nodisk = false; g_ring_win = c.ring_win;
  const double D_cm = c.d_pc * 3.0856775814913673e18;
  const int ss = std::max(1, p.samples);
  if (c.fit_ne0 && c.flux_target > 0) {  // secante em log-log: F ~ ne0^k, k em [1, 2]
    double& knob = p.model == 2 ? p.munit : p.ne0;  // GRMHD: the mass scale M_unit (g); torus: ne0 (cm^-3)
    double x0 = std::log(knob), f0 = std::log(radio_measure(p, ss, D_cm).flux_jy), slope = 1.5;
    const double tgt = std::log(c.flux_target);
    for (int it = 0; it < 12 && std::fabs(f0 - tgt) > 2e-3; ++it) {
      double x1 = x0 + (tgt - f0) / slope;
      x1 = std::max(x0 - 3.0, std::min(x0 + 3.0, x1));
      knob = std::exp(x1);
      const double f1 = std::log(radio_measure(p, ss, D_cm).flux_jy);
      if (std::fabs(x1 - x0) > 1e-9 && std::fabs(f1 - f0) > 1e-12) slope = std::max(0.2, std::min(3.0, (f1 - f0) / (x1 - x0)));
      std::printf("  fit %2d: %s = %.4g  F = %.4g Jy  (target %.4g; slope %.2f)\n", it, p.model == 2 ? "M_unit [g]" : "ne0 [cm^-3]", knob, std::exp(f1), c.flux_target, slope);
      x0 = x1; f0 = f1;
    }
  }
  std::vector<float> pfm(size_t(p.w) * p.h);
  const RadioStats st = radio_measure(p, ss, D_cm, c.radio_out.empty() ? nullptr : &pfm);
  std::printf("radio-cal: a=%g incl=%g deg  M=%.3g Msol  D=%.4g pc  nu=%g GHz  ne0=%.4g cm^-3  beta=%g  theta_e_max=%g\n", p.a, p.theta_deg, p.mbh, c.d_pc, p.nu_ghz, p.ne0, p.plasma_beta, p.theta_e_max);
  if (p.model == 2) std::printf("  M_unit = %.4g g  (rho_unit = %.4g g/cm^3)\n", p.munit, p.munit / std::pow(p.mbh * 1.476625e5, 3.0));
  std::printf("  scale: GM/c^2/D = %.4f uas;  field of view %.1f uas\n", st.uas_per_M, st.uas_per_M * p.dist * 2.0 * std::tan(0.5 * p.fov_deg * M_PI / 180.0));
  std::printf("  total flux    = %.4g Jy%s\n", st.flux_jy, c.flux_target > 0 ? (std::string("   (target ") + std::to_string(c.flux_target) + ")").c_str() : "");
  std::printf("  peak I        = %.4g Jy/uas^2  (peak T_b = %.3g K)\n", st.peak_jy_uas2, st.peak_tb);
  std::printf("  ring (r <= %g M): diameter = %.2f M = %.1f uas%s (ring centre at (%.2f, %.2f) M)\n", c.ring_win, st.ring_diam_M, st.ring_diam_uas, c.ring_ref > 0 ? (std::string("  [EHT: ") + std::to_string(c.ring_ref).substr(0, 4) + " uas]").c_str() : "", st.cx, st.cy);
  if (!c.radio_out.empty()) {
    FILE* f = std::fopen(c.radio_out.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "error writing %s\n", c.radio_out.c_str()); return 1; }
    std::fprintf(f, "Pf\n%d %d\n-1.0\n", p.w, p.h);
    for (int j = p.h - 1; j >= 0; --j) std::fwrite(&pfm[size_t(j) * p.w], 4, p.w, f);
    std::fclose(f);
    std::printf("  wrote %s (Jy/uas^2, PFM)\n", c.radio_out.c_str());
  }
  return 0;
}

int main(int argc, char** argv) {
  Cli c = parse(argc, argv);
  if (c.help) { print_help(); return 0; }
  const bool is_interactive = c.shot.empty() && !c.selftest && c.video.empty() && !c.radio_cal;
  if (is_interactive && !c.prec_set) { c.p.prec = kcuda::Precision::Float; c.p.tol = 2e-5; }  // real time: float (see the notes)
  if (c.nosky) c.sky.clear();
  else {
    FILE* f = std::fopen(c.sky.c_str(), "rb");
    if (f) std::fclose(f); else { std::printf("sky not found (%s): black background (code/tools/fetch_sky.sh)\n", c.sky.c_str()); c.sky.clear(); }
  }
  c.p.use_sky = !c.sky.empty();
  if (c.p.prec == kcuda::Precision::Float && c.p.tol < 1e-6) {
    std::printf("warning: tol %g is unreachable in float; using 2e-5\n", c.p.tol);
    c.p.tol = 2e-5;
  }
  std::string err;
  if (!kcuda::init(c.sky, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
  if (!c.sky2.empty() && !kcuda::load_sky2(c.sky2, err)) { std::fprintf(stderr, "--sky2: %s\n", err.c_str()); return 1; }
  if (!c.grmhd_file.empty()) {
    const auto tg0 = Clock::now();
    if (!kcuda::grmhd_load(c.grmhd_file, err)) { std::fprintf(stderr, "--grmhd: %s\n", err.c_str()); return 1; }
    c.p.a = kcuda::grmhd_spin();  // the integrator has to use the dump's spin
    std::printf("GRMHD: %s loaded (3D textures) in %.0f ms, spin a = %g\n", c.grmhd_file.c_str(), 1000.0 * std::chrono::duration<double>(Clock::now() - tg0).count(), c.p.a);
  }
  if (c.denoise_kernels) {  // which CUDA kernels does the OptiX denoiser launch? (CUPTI); 1080p, 2 invocations
    if (!kcuda::denoise_available() || !denoise::kernel_trace_available()) { std::fprintf(stderr, "--denoise-kernels needs OptiX and CUPTI compiled in\n"); return 1; }
    if (!kcuda::init("", err)) return 1;
    const int w = 1920, h = 1080;
    std::vector<float> img(size_t(w) * h * 4), out;
    for (size_t i = 0; i < img.size(); ++i) img[i] = (i % 4 == 3) ? 1.f : 0.5f + 0.4f * std::sin(0.01f * float(i / 4 % w)) + 0.2f * float((i * 2654435761u) % 1000) / 1000.f;
    if (!kcuda::denoise_image(img, w, h, out, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }  // warm-up (init + setup)
    denoise::kernel_trace_begin();
    for (int i = 0; i < 2; ++i) kcuda::denoise_image(img, w, h, out, err);
    auto ks = denoise::kernel_trace_end();
    std::sort(ks.begin(), ks.end(), [](const denoise::KernelStat& a, const denoise::KernelStat& b) { return a.total_us > b.total_us; });
    double tot = 0;
    for (const auto& k : ks) { tot += k.total_us; std::printf("%8.1f us x%-3u %s\n", k.total_us, k.count, k.name.c_str()); }
    std::printf("distinct kernels: %zu; total kernel time %.1f us (2 invocations, 1080p)\n", ks.size(), tot);
    std::printf("Note: the names (conv_ampere_*, k_tin3_mlp with __half, fused_convolutions_kernel_sm80) are FP16 convolutions/MLP for Ampere+;\n"
                "consistent with tensor cores, but this does NOT prove it (no hardware counters: no ncu/nsys on this system).\n");
    return 0;
  }
  if (c.selftest_denoise) { c.p.use_sky = false; if (!kcuda::init("", err)) return 1; const int b = selftest_denoise(c); std::printf(b ? "denoiser selftest FAILED\n" : "denoiser selftest OK\n"); return b ? 1 : 0; }
  if (c.audit_n > 0) return audit_float(c, c.audit_n, c.audit_seed) ? 1 : 0;
  if (c.selftest) return selftest(c);
  if (c.radio_cal) return radio_cal(c);
  if (!c.route_file.empty()) return route_main(c);
  if (!c.video.empty()) { if (!c.prec_set) { c.p.prec = kcuda::Precision::Float; c.p.tol = 2e-5; } if (c.p.rin < 0) c.p.rin = -1; return video(c); }
  if (!c.shot.empty()) return shot(c);
  return interactive(c);
}
