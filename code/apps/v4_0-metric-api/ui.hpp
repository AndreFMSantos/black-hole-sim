#pragma once
// Interactive UI (v3.1): Dear ImGui panel to choose version/variant/options in real time. It only edits kcuda::Params and the camera; main.cpp's loop
// detects the change (vector `key`) and restarts the accumulation. Compiled only with BHS_IMGUI (code/third_party/imgui, tools/fetch_imgui.sh).
#include <string>

#include "render.hpp"

namespace kcuda = bhs::kcuda;  // the renderer types live in bhs::kcuda

struct GLFWwindow;

namespace ui {

// Epistemic status of each configuration (notes/physics/26-realistic-naked-eye-views.md)
enum class Badge { Observed, ModelDependent, Hypothetical, Aesthetic };
const char* badge_name(Badge b);

struct Camera { double theta_deg = 80, phi_deg = 0, dist = 40; };
struct Setup {  // everything a scenario fixes
  kcuda::Params p;
  Camera cam;
  std::string observer = "static";
  bool cinematic = false;
  std::string grmhd;  // .bgm file (GRMHD modes); empty = none
};

struct Perf { double fps = 0, ms = 0; int samples = 0; };

// requests the panel makes to the main loop (which is what applies cinematic, the observer and loads dumps)
struct Request {
  bool apply_setup = false; Setup setup;
  bool toggle_cinematic = false;
  bool load_dump = false; std::string dump_path;
  bool blend = false; int blend_idx = 0; double blend_tau = 0;  // time interpolation between snapshot blend_idx and blend_idx + 1 (the loop loads / swaps them)
  bool want_screenshot = false;
  bool toggle_flight = false;  // the user asked for the ship (flight mode) or to leave it
};

struct State {
  int scenario = -1;
  bool show = true;
  std::string dump_dir, status;
  int dump_idx = 0, dump_first = 0, dump_last = 0; bool playing = false; float play_rate = 8.f; double play_acc = 0;
  float assist_r = 12.f;  // target radius of the orbit-assist slider
  bool in_flight = false;  // set by the loop: flight mode shows the small flight panel instead of the full one
  bool dump_dir_ok = false; std::string dump_scanned;  // dump_scanned: the folder last scanned (the app sets dump_dir when a scenario loads a snapshot, so the panel must rescan)
  bool wh_travel = false; float wh_speed = 0.5f;  // wormhole: fly through the throat (wh_speed = beta, the velocity in c measured by the static frame; l decreases at 5 beta rho per second and the observer's aberration is applied; the loop restarts on the far side)
  bool dump_blend = true;  // while playing: interpolate in time between consecutive snapshots (smooth motion at any rate) instead of jumping
  bool orbit_play = false; float orbit_rate = 15.f;  // auto-orbit: the camera circles the hole at orbit_rate degrees per second (every variant)
  bool disk_play = true; float disk_rate = 10.f;  // turbulence / hot spot: advance the disk time (M of disk time per real second) to see the differential rotation
  std::string sky2_note;  // non-empty: the 2nd sky (wormhole) could not be loaded
};

bool init(GLFWwindow* win);
void shutdown();
bool wants_mouse();
bool wants_keyboard();
void new_frame();
// draws the panel; edits p, cam, observer, cinematic in place; returns the requests
struct FlightInfo {  // what the loop tells the flight panel
  bool paused = false, hud = true, view_free = false, orbital_controls = true, assisting = false;
  int level = 3; double timescale = 5.0;
  double r = 0, speed = 0, v_radial = 0, v_circ = 0, r_isco = 0;
  const char* assist = "off";
};
struct FlightReq {
  bool leave = false, reset = false, face_hole = false, toggle_lock = false, toggle_pause = false, toggle_hud = false, toggle_orbital = false;
  bool circularize = false, go_to = false, cancel_assist = false; double go_to_r = 0;
  int level_delta = 0; double timescale_mul = 1.0;
};
FlightReq draw_flight(State& st, double fps, const FlightInfo& fi);  // flight mode panel: what the user asked for
Request draw(State& st, kcuda::Params& p, Camera& cam, std::string& observer, bool cinematic, const Perf& perf, double dt);
void render();
// applies a scenario (index into the list) on top of the current state (keeps size, sky, precision)
int scenario_count();
const char* scenario_title(int i);
Setup make_setup(int i, const kcuda::Params& current);
Badge classify(const kcuda::Params& p, bool cinematic, std::string& why);

}  // namespace ui
