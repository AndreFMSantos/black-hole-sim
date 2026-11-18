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
  bool want_screenshot = false;
};

struct State {
  int scenario = -1;
  bool show = true;
  std::string dump_dir, status;
  int dump_idx = 0, dump_first = 0, dump_last = 0; bool playing = false; float play_rate = 8.f; double play_acc = 0;
  bool dump_dir_ok = false;
  bool disk_play = false; float disk_rate = 10.f;  // turbulence / hot spot: advance the disk time (M of disk time per real second) to see the differential rotation
  std::string sky2_note;  // non-empty: the 2nd sky (wormhole) could not be loaded
};

bool init(GLFWwindow* win);
void shutdown();
bool wants_mouse();
bool wants_keyboard();
void new_frame();
// draws the panel; edits p, cam, observer, cinematic in place; returns the requests
Request draw(State& st, kcuda::Params& p, Camera& cam, std::string& observer, bool cinematic, const Perf& perf, double dt);
void render();
// applies a scenario (index into the list) on top of the current state (keeps size, sky, precision)
int scenario_count();
const char* scenario_title(int i);
Setup make_setup(int i, const kcuda::Params& current);
Badge classify(const kcuda::Params& p, bool cinematic, std::string& why);

}  // namespace ui
