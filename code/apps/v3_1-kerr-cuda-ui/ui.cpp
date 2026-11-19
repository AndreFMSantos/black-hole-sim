// Interactive UI (v3.1): Dear ImGui panel. See ui.hpp.
#include "ui.hpp"

#include "camera_limits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

namespace ui {

const char* badge_name(Badge b) {
  switch (b) {
    case Badge::Observed: return "OBSERVED / TESTED";
    case Badge::ModelDependent: return "MODEL-DEPENDENT";
    case Badge::Hypothetical: return "HYPOTHETICAL";
    case Badge::Aesthetic: return "AESTHETIC";
  }
  return "";
}
static ImVec4 badge_colour(Badge b) {
  switch (b) {
    case Badge::Observed: return ImVec4(0.35f, 0.80f, 0.45f, 1);
    case Badge::ModelDependent: return ImVec4(0.95f, 0.80f, 0.30f, 1);
    case Badge::Hypothetical: return ImVec4(0.95f, 0.45f, 0.35f, 1);
    case Badge::Aesthetic: return ImVec4(0.65f, 0.55f, 0.95f, 1);
  }
  return ImVec4(1, 1, 1, 1);
}

// ---------------------------------------------------------------- scenarios (one entry per version/variant)
struct Scenario {
  const char* title; const char* group; Badge badge; const char* note;
  void (*apply)(Setup&);
};
static void radio_common(Setup& s) {
  auto& p = s.p;
  p.nodisk = false; p.nt = false; p.limb = false; p.a = 0.9375; p.dist = 150; p.fov_deg = 16; p.mbh = 6.5e9; p.nu_ghz = 230; p.tb_lo = 8.5; p.tb_hi = 11;
  p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.exposure = 3; p.grmhd_rcap = 30;  // v2.2f-d: skip ray marching beyond r = 30 (flux change 4e-5, 22 % faster at 230 GHz)
  s.cam = {17, 0, 150};
}
static const Scenario kScenarios[] = {
    {"Schwarzschild shadow (no disk)", "v0.3 - v0.5", Badge::Observed, "Shadow and gravitational lensing: b = sqrt(27) M; consistent with the EHT (M87* and Sgr A* rings).",
     [](Setup& s) { s.p.a = 0; s.p.nodisk = true; s.cam = {80, 0, 40}; }},
    {"Thin accretion disk (Newtonian profile)", "v0.3 - v0.5", Badge::ModelDependent, "Simplified T(r) profile, illustrative only. Doppler shift and redshift are physical.",
     [](Setup& s) { s.p.a = 0; s.p.nt = false; s.cam = {80, 0, 40}; }},
    {"Kerr a = 0.9 (Newtonian disk)", "v0.6", Badge::ModelDependent, "Exact Carter geodesics; the disk is simplified.",
     [](Setup& s) { s.p.a = 0.9; s.p.nt = false; s.cam = {80, 0, 40}; }},
    {"Novikov-Thorne and photon rings (a = 0.9)", "v0.7", Badge::ModelDependent, "Thin Novikov-Thorne disk; NOT valid for accretion rates < 1% of Eddington (Shimura & Takahara 1995). Photon rings are a prediction not yet resolved.",
     [](Setup& s) { s.p.a = 0.9; s.p.nt = true; s.cam = {80, 0, 40}; }},
    {"Orbiting observer (r = 12)", "v0.7c", Badge::ModelDependent, "Aberration and Doppler shift for an observer on a circular orbit.",
     [](Setup& s) { s.p.a = 0.9; s.p.nt = true; s.observer = "orbit"; s.cam = {80, 0, 12}; }},
    {"Cinematic finish", "v0.8d", Badge::Aesthetic, "ACES, bloom, glare, grain, vignette, chromatic aberration: camera effects, not physics.",
     [](Setup& s) { s.p.a = 0.9; s.p.nt = true; s.cinematic = true; s.cam = {80, 0, 40}; }},
    {"Thick disk (simple model)", "v0.9b", Badge::ModelDependent, "Semi-transparent Gaussian layer: a real optically thick disk shows a sharp surface; this looks like a cloud (known limitation).",
     [](Setup& s) { s.p.a = 0.9; s.p.nt = true; s.p.hr = 0.06; s.p.tau = 4; s.p.flare = 0.15; s.cam = {80, 0, 40}; }},
    {"Turbulence, hot spot and slow light", "v0.9c", Badge::ModelDependent, "Lognormal turbulence rotated by the Kerr Omega(r); illustrative hot spot and flare.",
     [](Setup& s) {
       auto& p = s.p; p.a = 0.9; p.nt = true; p.hr = 0.06; p.tau = 4; p.flare = 0.15; p.turb = 1.2; p.turb_scale = 1.2; p.edge_soft = 0.12;
       p.spot_r = 9; p.spot_phi_deg = 200; p.spot_size = 1.2; p.spot_amp = 5; p.slowlight = true; p.disk_time = 100; s.cam = {75, 0, 40};
     }},
    {"Reissner-Nordstrom Q = 0.6", "v2.1a", Badge::Hypothetical, "Charged black hole: astrophysical charge is neutralised quickly in plasma. Generalised Page-Thorne disk.",
     [](Setup& s) { s.p.a = 0; s.p.charge = 0.6; s.p.nt = true; s.cam = {78, 0, 30}; }},
    {"Kerr-Newman a = 0.5, Q = 0.5", "v2.1e", Badge::Hypothetical, "Novikov-Thorne/Page-Thorne disk in Kerr-Newman (own ISCO, T(r) and redshift; validated against independent oracles).",
     [](Setup& s) { s.p.a = 0.5; s.p.charge = 0.5; s.p.nt = true; s.cam = {78, 0, 40}; }},
    {"Naked singularity (Q = 1.1)", "v2.1a", Badge::Hypothetical, "No horizon: cosmic censorship forbids it. The repulsive core barrier creates nested images of the disk.",
     [](Setup& s) { s.p.a = 0; s.p.charge = 1.1; s.p.nt = false; s.p.rin = 6; s.cam = {78, 0, 30}; }},
    {"Schwarzschild-de Sitter (Lambda = 0.01)", "v2.1c", Badge::Hypothetical, "Real Lambda is ~1e-30 (in units of M^-2): 0.01 is an exaggeration so it can be seen. The 'sky' at the cosmological horizon is our own convention.",
     [](Setup& s) { s.p.a = 0; s.p.lambda = 0.01; s.p.nodisk = true; s.cam = {90, 0, 8}; s.p.fov_deg = 70; }},
    {"Ellis wormhole", "v2.1d", Badge::Hypothetical, "No observational evidence; requires exotic matter. The far side of the throat shows a second sky (WISE infrared map, assets/sky/wise).",
     [](Setup& s) { s.p.wormhole = true; s.p.wh_ellis = true; s.p.wh_rho = 1; s.p.wh_lc = 4; s.p.nodisk = true; s.p.fov_deg = 50; s.cam = {90, 0, 4}; }},
    {"Interstellar wormhole", "v2.1d", Badge::Hypothetical, "Metric of James et al. 2015 (rho = 1, 2a = 0.01 rho, W = 0.05 rho).",
     [](Setup& s) { s.p.wormhole = true; s.p.wh_ellis = false; s.p.wh_rho = 1; s.p.wh_a = 0.005; s.p.wh_M = 0.05 / 1.42953; s.p.wh_lc = 4; s.p.nodisk = true; s.p.fov_deg = 50; s.cam = {90, 0, 4}; }},
    {"Radio: Fishbone-Moncrief torus", "v2.2b", Badge::ModelDependent, "Analytic torus + thermal synchrotron; electron temperature and density are model parameters.",
     [](Setup& s) {
       radio_common(s); auto& p = s.p; p.a = 0.9; p.model = 1; p.torus_rin = 7; p.torus_rmax = 12; p.ne0 = 3e4; p.plasma_beta = 3; p.theta_e_max = 20; p.emission = 0; p.tonemap = kcuda::Tonemap::Aces;
     }},
    {"Own GRMHD (M87*, MAD a = 0.9375)", "v2.2g", Badge::ModelDependent, "Our own AthenaK simulation, calibrated to 0.5 Jy (model scale of EHT Paper V; measured ~0.66 Jy). Ring = Kerr critical curve.",
     [](Setup& s) {
       radio_common(s); auto& p = s.p; p.model = 2; p.electrons = 1; p.rhigh = 20; p.munit = 8.589e24; p.grmhd_sigma_cut = 1.0; p.emission = 1;
       s.grmhd = BHS_DATA_DEFAULT "/athenak/run_d/bgm_hi/mad_00350.bgm";
     }},
    {"Own GRMHD as Sgr A* (EHT best guess)", "v2.2g", Badge::ModelDependent, "MAD, R_high = 160, i = 30 degrees, M = 4.14e6 Msun at 8127 pc (values from the EHT Paper V simulations, not the measurement), calibrated to 2.4 Jy. Ring 49.8 uas vs 51.8 +- 2.3 from the EHT.",
     [](Setup& s) {
       radio_common(s); auto& p = s.p; p.model = 2; p.electrons = 1; p.rhigh = 160; p.munit = 5.003e17; p.grmhd_sigma_cut = 1.0; p.emission = 1; p.mbh = 4.14e6; s.cam = {30, 0, 150};
       s.grmhd = BHS_DATA_DEFAULT "/athenak/run_d/bgm_hi/mad_00350.bgm";
     }},
};
static constexpr int kN = int(sizeof(kScenarios) / sizeof(kScenarios[0]));
int scenario_count() { return kN; }
const char* scenario_title(int i) { return kScenarios[i].title; }

Setup make_setup(int i, const kcuda::Params& cur) {
  Setup s;
  s.p = kcuda::Params{};
  s.p.w = cur.w; s.p.h = cur.h; s.p.use_sky = cur.use_sky; s.p.prec = cur.prec; s.p.tol = cur.tol; s.p.samples = cur.samples; s.p.max_steps = cur.max_steps;
  s.p.exposure = cur.exposure; s.p.sky_exposure = cur.sky_exposure;
  s.p.rin = -1;
  kScenarios[i].apply(s);
  if (s.cinematic) { /* the preset parameters are applied in the main loop (apply_cinematic) */ }
  return s;
}

Badge classify(const kcuda::Params& p, bool cinematic, std::string& why) {
  if (p.wormhole) { why = "Wormholes have no evidence and require exotic matter."; return Badge::Hypothetical; }
  if (p.charge > 0) { why = p.a * p.a + p.charge * p.charge >= 1 ? "Naked singularity: forbidden by cosmic censorship." : "Charge: neutralised in plasma; never observed."; return Badge::Hypothetical; }
  if (p.lambda > 0) { why = "Real Lambda is ~1e-30 in units of M^-2: this value is exaggerated so it can be seen."; return Badge::Hypothetical; }
  if (cinematic) { why = "Camera effects (bloom, glare, grain): a visual choice."; return Badge::Aesthetic; }
  if (p.model == 2) { why = "GRMHD: physical dynamics, but electron T (R_high), M_unit and calibration are model choices."; return Badge::ModelDependent; }
  if (p.model == 1) { why = "Analytic torus: free parameters (density, beta, electron T)."; return Badge::ModelDependent; }
  if (p.nodisk) { why = "Shadow and lensing: a prediction of general relativity, consistent with the EHT."; return Badge::Observed; }
  if (p.hr > 0) { why = "Simple semi-transparent layer: not the surface of a real thick disk."; return Badge::ModelDependent; }
  why = "Thin Novikov-Thorne disk: not valid at accretion rates < 1% of Eddington (Shimura & Takahara 1995)."; return Badge::ModelDependent;
}

// ---------------------------------------------------------------- ImGui
bool init(GLFWwindow* win) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;  // no layout file
  ImGui::StyleColorsDark();
  ImGui::GetStyle().WindowRounding = 4.f; ImGui::GetStyle().Alpha = 0.95f;
  if (!ImGui_ImplGlfw_InitForOpenGL(win, true)) return false;
  return ImGui_ImplOpenGL3_Init("#version 460");
}
void shutdown() { ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext(); }
bool wants_mouse() { return ImGui::GetIO().WantCaptureMouse; }
bool wants_keyboard() { return ImGui::GetIO().WantCaptureKeyboard; }
void new_frame() { ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame(); }
void render() { ImGui::Render(); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); }

static void scan_dumps(State& st) {
  namespace fs = std::filesystem;
  st.dump_dir_ok = false; int lo = 1 << 30, hi = -1;
  std::error_code ec;
  if (!st.dump_dir.empty() && fs::is_directory(st.dump_dir, ec))
    for (const auto& e : fs::directory_iterator(st.dump_dir, ec)) {
      const std::string n = e.path().filename().string();
      int k = -1;
      if (std::sscanf(n.c_str(), "mad_%d.bgm", &k) == 1 && k >= 0) { lo = std::min(lo, k); hi = std::max(hi, k); }
    }
  if (hi >= 0) { st.dump_first = lo; st.dump_last = hi; st.dump_dir_ok = true; st.dump_idx = std::min(std::max(st.dump_idx, lo), hi); }
}

Request draw(State& st, kcuda::Params& p, Camera& cam, std::string& observer, bool cinematic, const Perf& perf, double dt) {
  Request rq;
  if (!st.show) return rq;
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(400, 720), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("black-hole-sim (F1 show/hide)", &st.show)) { ImGui::End(); return rq; }
  ImGui::Text("%.0f fps | %.1f ms/sample | sample %d", perf.fps, perf.ms, perf.samples);
  std::string why; const Badge b = classify(p, cinematic, why);
  ImGui::TextColored(badge_colour(b), "[%s]", badge_name(b)); ImGui::SameLine(); ImGui::TextWrapped("%s", why.c_str());
  ImGui::Separator();

  if (ImGui::CollapsingHeader("Versions and variants", ImGuiTreeNodeFlags_DefaultOpen)) {
    const char* last = "";
    for (int i = 0; i < kN; ++i) {
      if (std::string(kScenarios[i].group) != last) { last = kScenarios[i].group; ImGui::TextDisabled("%s", last); }
      ImGui::PushStyleColor(ImGuiCol_Text, badge_colour(kScenarios[i].badge));
      char lab[160]; std::snprintf(lab, sizeof lab, "%s##sc%d", kScenarios[i].title, i);
      const bool sel = ImGui::Selectable(lab, st.scenario == i);
      ImGui::PopStyleColor();
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n[%s]", kScenarios[i].note, badge_name(kScenarios[i].badge));
      if (sel) { st.scenario = i; rq.apply_setup = true; rq.setup = make_setup(i, p); }
    }
  }

  if (p.wormhole && !st.sky2_note.empty()) ImGui::TextColored(ImVec4(1.f, 0.45f, 0.35f, 1.f), "%s", st.sky2_note.c_str());

  if (ImGui::CollapsingHeader("Black hole", ImGuiTreeNodeFlags_DefaultOpen)) {
    int kind = p.wormhole ? 3 : p.lambda > 0 ? 2 : p.charge > 0 ? 1 : 0;
    const char* kinds[] = {"Kerr", "Kerr-Newman / Reissner-Nordstrom", "Schwarzschild-de Sitter", "Wormhole"};
    int nk = kind;
    if (ImGui::Combo("Metric", &nk, kinds, 4) && nk != kind) {
      p.wormhole = nk == 3; p.charge = nk == 1 ? 0.5 : 0; p.lambda = nk == 2 ? 0.01 : 0;
      if (nk == 3) { p.nodisk = true; p.wh_ellis = false; p.wh_rho = 1; p.wh_a = 0.005; p.wh_M = 0.05 / 1.42953; p.wh_lc = 4; p.fov_deg = 50; cam.dist = 4; }
      if (nk == 2) { p.a = 0; p.nodisk = true; cam.dist = std::min(cam.dist, 8.0); }
      p.rin = -1; p.nt = true; p.hr = 0; p.model = 0;
    }
    float a = float(p.a);
    if (!p.wormhole && ImGui::SliderFloat("spin a", &a, 0.f, 0.9999f, "%.4f")) { p.a = a; if (p.lambda > 0) p.a = 0; }
    if (p.charge > 0 || nk == 1) {
      float q = float(p.charge);
      if (ImGui::SliderFloat("charge Q", &q, 0.01f, 1.5f, "%.3f")) p.charge = q;
      const bool naked = p.a * p.a + p.charge * p.charge >= 1.0;
      if (naked) { p.nt = false; if (p.rin < 0) p.rin = 6; ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1), "a^2 + Q^2 >= 1: naked singularity (Newtonian disk, r_in = %.1f)", p.rin); }
      else p.rin = -1;
    }
    if (p.lambda > 0) {
      float l = float(p.lambda);
      if (ImGui::SliderFloat("Lambda", &l, 1e-4f, 0.11f, "%.4f", ImGuiSliderFlags_Logarithmic)) p.lambda = l;
      p.nt = false; if (p.rin < 0) p.rin = 6;
    }
    if (p.wormhole) {
      bool ell = p.wh_ellis; if (ImGui::Checkbox("Ellis (otherwise: Interstellar)", &ell)) p.wh_ellis = ell;
      float lc = float(p.wh_lc); if (ImGui::SliderFloat("distance l (rho)", &lc, 1.05f, 60.f, "%.2f", ImGuiSliderFlags_Logarithmic)) { p.wh_lc = lc; cam.dist = lc; }
    }
  }

  if (ImGui::CollapsingHeader("Disk")) {
    const bool restricted = p.wormhole || p.charge > 0 || p.lambda > 0;
    int dk = p.nodisk ? 0 : p.model == 2 ? 5 : p.model == 1 ? 4 : p.hr > 0 ? 3 : p.nt ? 1 : 2;
    const char* dks[] = {"None", "Thin Novikov-Thorne", "Thin Newtonian", "Thick (layer)", "Radio: torus", "Radio: GRMHD"};
    int nd = dk;
    if (restricted) ImGui::TextDisabled("This metric only supports a thin disk (or none).");
    if (ImGui::Combo("Type", &nd, dks, restricted ? 3 : 6) && nd != dk) {
      p.nodisk = nd == 0; p.model = nd == 4 ? 1 : nd == 5 ? 2 : 0; p.nt = nd == 1; p.hr = nd == 3 ? 0.06 : 0; if (nd == 3) { p.tau = 4; p.flare = 0.15; p.nt = true; }
      if (nd == 4 || nd == 5) { p.nodisk = false; p.fov_deg = 16; cam.dist = std::max(cam.dist, 150.0); p.mbh = 6.5e9; p.nu_ghz = 230; p.tb_lo = 8.5; p.tb_hi = 11; p.prec = kcuda::Precision::Float; p.tol = 2e-5; p.nt = false;
        if (nd == 4) { p.torus_rin = 7; p.torus_rmax = 12; p.ne0 = 3e4; p.plasma_beta = 3; p.theta_e_max = 20; } else { p.electrons = 1; p.emission = 1; p.munit = 8.589e24; p.grmhd_rcap = 30; p.grmhd_sigma_cut = 1.0; } }
    }
    if (!p.nodisk && p.model == 0) { float ro = float(p.rout); if (ImGui::SliderFloat("outer radius", &ro, 8.f, 80.f)) p.rout = ro; }
    if (p.hr > 0) { float h = float(p.hr), t = float(p.tau), f = float(p.flare); ImGui::SliderFloat("H/r", &h, 0.01f, 0.2f); ImGui::SliderFloat("tau", &t, 0.5f, 10.f); ImGui::SliderFloat("flare", &f, 0.f, 0.5f); p.hr = h; p.tau = t; p.flare = f; }
    if (p.model == 1) { float n = float(std::log10(p.ne0)), be = float(p.plasma_beta), te = float(p.theta_e_max); if (ImGui::SliderFloat("log10 n_e0", &n, 3.f, 7.f)) p.ne0 = std::pow(10.0, n); ImGui::SliderFloat("beta", &be, 0.5f, 30.f); ImGui::SliderFloat("Theta_e max", &te, 5.f, 40.f); p.plasma_beta = be; p.theta_e_max = te; }
    if (p.model == 2) {
      float rh = float(p.rhigh), mu = float(std::log10(p.munit)), sc = float(p.grmhd_sigma_cut), tl = float(p.tb_lo), th = float(p.tb_hi);
      if (ImGui::SliderFloat("R_high", &rh, 1.f, 160.f, "%.0f", ImGuiSliderFlags_Logarithmic)) p.rhigh = rh;
      if (ImGui::SliderFloat("log10 M_unit [g]", &mu, 23.f, 27.f)) p.munit = std::pow(10.0, mu);
      if (ImGui::SliderFloat("sigma cut (0=off)", &sc, 0.f, 30.f, "%.1f")) p.grmhd_sigma_cut = sc;
      ImGui::SliderFloat("T_b min (log)", &tl, 6.f, 11.f); ImGui::SliderFloat("T_b max (log)", &th, 8.f, 12.f); p.tb_lo = tl; p.tb_hi = std::max<double>(th, tl + 0.5);
      char buf[512]; std::snprintf(buf, sizeof buf, "%s", st.dump_dir.c_str());
      if (ImGui::InputText("dump folder", buf, sizeof buf)) { st.dump_dir = buf; scan_dumps(st); }
      if (st.dump_dir.empty()) { st.dump_dir = BHS_DATA_DEFAULT "/athenak/run_d/bgm_hi"; scan_dumps(st); }
      if (st.dump_dir_ok) {
        int idx = st.dump_idx; bool ch = ImGui::SliderInt("snapshot", &idx, st.dump_first, st.dump_last);
        ImGui::Checkbox("play", &st.playing); ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::SliderFloat("dumps/s", &st.play_rate, 1.f, 30.f);
        if (st.playing) { st.play_acc += dt * st.play_rate; while (st.play_acc >= 1.0) { st.play_acc -= 1.0; idx = idx >= st.dump_last ? st.dump_first : idx + 1; ch = true; } }
        if (ch) { st.dump_idx = idx; char path[1024]; std::snprintf(path, sizeof path, "%s/mad_%05d.bgm", st.dump_dir.c_str(), idx); rq.load_dump = true; rq.dump_path = path; }
        ImGui::TextDisabled("2 M between snapshots (T600_d); t = %.0f M", 10000.0 + 2.0 * (idx - 200.0));
      } else ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1), "no mad_*.bgm dumps in this folder");
    }
    if (p.model == 0 && !p.nodisk && (p.turb > 0 || p.spot_amp > 0)) {  // time-dependent structure: every ring turns with the Kerr Omega(r) (inner rings faster)
      ImGui::Checkbox("play disk time", &st.disk_play); ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::SliderFloat("M/s", &st.disk_rate, 1.f, 100.f, "%.0f", ImGuiSliderFlags_Logarithmic);
      float t = float(p.disk_time); if (ImGui::SliderFloat("disk time (M)", &t, 0.f, 2000.f)) p.disk_time = t;
      if (st.disk_play) p.disk_time += double(st.disk_rate) * dt;
    }
    bool nd2 = p.nodoppler; if (ImGui::Checkbox("no Doppler", &nd2)) p.nodoppler = nd2;
  }

  if (ImGui::CollapsingHeader("Camera and observer", ImGuiTreeNodeFlags_DefaultOpen)) {
    float d = float(cam.dist), th = float(cam.theta_deg), ph = float(cam.phi_deg), fov = float(p.fov_deg);
    double dmin = 1.5, dmax = 2000.0;
    if (!p.wormhole) { kcuda::camera_range(p, dmin, dmax, observer); d = float(std::min(std::max(double(d), dmin), dmax)); cam.dist = d; }  // never inside the horizon / beyond r_c
    if (ImGui::SliderFloat("distance", &d, float(dmin), float(dmax), "%.2f", ImGuiSliderFlags_Logarithmic)) { cam.dist = d; if (p.wormhole) p.wh_lc = d; }
    if (ImGui::SliderFloat("theta (inclination)", &th, 0.1f, 179.9f)) cam.theta_deg = th;
    if (ImGui::SliderFloat("phi (azimuth)", &ph, 0.f, 360.f)) cam.phi_deg = ph;
    if (ImGui::SliderFloat("FOV", &fov, 2.f, 150.f)) p.fov_deg = fov;
    if (!p.wormhole && p.charge == 0 && p.lambda == 0) {
      const char* obs[] = {"static", "orbit", "infall"}; int oi = observer == "orbit" ? 1 : observer == "infall" ? 2 : 0;
      if (ImGui::Combo("observer", &oi, obs, 3)) observer = obs[oi];
    }
  }

  if (ImGui::CollapsingHeader("Image")) {
    float ex = float(std::log10(p.exposure)), sx = float(std::log10(p.sky_exposure)), bl = float(p.bloom);
    if (ImGui::SliderFloat("exposure (log10)", &ex, -1.5f, 1.5f)) p.exposure = std::pow(10.0, ex);
    if (ImGui::SliderFloat("sky exposure (log10)", &sx, -1.f, 1.5f)) p.sky_exposure = std::pow(10.0, sx);
    if (ImGui::SliderFloat("bloom", &bl, 0.f, 0.6f)) p.bloom = bl;
    int tm = int(p.tonemap); if (ImGui::Combo("tone mapping", &tm, "Reinhard\0ACES\0")) p.tonemap = tm ? kcuda::Tonemap::Aces : kcuda::Tonemap::Reinhard;
    bool cin = cinematic; if (ImGui::Checkbox("cinematic finish (aesthetic)", &cin)) rq.toggle_cinematic = true;
    int pr = p.prec == kcuda::Precision::Double ? 1 : 0;
    if (ImGui::Combo("precision", &pr, "float (real time)\0double\0")) { p.prec = pr ? kcuda::Precision::Double : kcuda::Precision::Float; p.tol = pr ? 1e-8 : 2e-5; }
  }

  if (ImGui::CollapsingHeader("Help")) {
    ImGui::TextWrapped("Mouse: drag to orbit the camera; wheel = distance. Keys: [ ] spin, PgUp/PgDn exposure, T tone mapping, B bloom, C cinematic, O observer, G disk, D Doppler, P precision, F1 hides this panel, F12 saves a snapshot (snapshots/bhs_NNN.png + .txt with the command line that reproduces it), Esc quits.");
    ImGui::TextWrapped("The coloured label says what is observed, what is model-dependent, what is hypothetical and what is aesthetic (notes/physics/26-realistic-naked-eye-views.md).");
  }
  ImGui::End();
  return rq;
}

}  // namespace ui
