// v0.5 — the v0.4 render in real time in a GLSL 4.6 compute shader (OpenGL), with a free camera.
// Mouse: drag = orbit (theta, azimuth); wheel = distance (>= 10). Keys: D Doppler, G disk,
// +/- exposure, ESC quit.
// Usage: v0_5-realtime-gpu [--w 1280] [--h 720] [--theta 80] [--phi 0] [--dist 40] [--fov 40]
//        [--rin 6] [--rout 16] [--tpeak 4200] [--exposure 1.6] [--step 0.015] [--maxphi 100]
//        [--nodoppler] [--nodisk] [--vsync]
//        [--sky map.exr] [--nosky] [--sky-exposure S]  (real NASA HDR sky; by default uses
//        assets/sky/starmap_2020_8k.exr if it exists, otherwise procedural stars; --nosky forces the procedural ones)
// Extra keys: UP/DOWN ARROW = sky exposure (independent of the disk's)
//        [--shot out.png [--frames N]]   (no interaction: writes a PNG and prints ms/frame)
#include "gl_loader.hpp"

#include <bhs/blackbody.hpp>
#include <bhs/camera.hpp>
#include <bhs/exr_loader.hpp>
#include <bhs/lut.hpp>
#include <bhs/png.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace bhs;
using Clock = std::chrono::steady_clock;

struct Opts {
  int w = 1280, h = 720, frames = 1;
  double theta = 80.0, phi = 0.0, dist = 40.0, fov = 40.0;
  double rin = 6.0, rout = 16.0, tpeak = 4200.0, exposure = 1.6, step = 0.015, maxphi = 100.0;
  bool doppler = true, disk = true, vsync = false;
  std::string shot;
  std::string sky = BHS_SKY_DEFAULT;
  bool nosky = false;
  int skymode = 0;  // 0 quad, 1 extra rays, 2 no filter
  double sky_exposure = 2.0;
};

static Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
    if (is("--w")) o.w = std::atoi(argv[++i]);
    else if (is("--h")) o.h = std::atoi(argv[++i]);
    else if (is("--frames")) o.frames = std::max(1, std::atoi(argv[++i]));
    else if (is("--theta")) o.theta = std::atof(argv[++i]);
    else if (is("--phi")) o.phi = std::atof(argv[++i]);
    else if (is("--dist")) o.dist = std::atof(argv[++i]);
    else if (is("--fov")) o.fov = std::atof(argv[++i]);
    else if (is("--rin")) o.rin = std::atof(argv[++i]);
    else if (is("--rout")) o.rout = std::atof(argv[++i]);
    else if (is("--tpeak")) o.tpeak = std::atof(argv[++i]);
    else if (is("--exposure")) o.exposure = std::atof(argv[++i]);
    else if (is("--step")) o.step = std::atof(argv[++i]);
    else if (is("--maxphi")) o.maxphi = std::atof(argv[++i]);
    else if (is("--sky")) o.sky = argv[++i];
    else if (is("--sky-exposure")) o.sky_exposure = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--nosky")) o.nosky = true;
    else if (is("--sky-mode")) o.skymode = std::atoi(argv[++i]);  // 0 quad (default), 1 extra rays, 2 no filter (aliasing)
    else if (is("--shot")) o.shot = argv[++i];
    else if (!std::strcmp(argv[i], "--nodoppler")) o.doppler = false;
    else if (!std::strcmp(argv[i], "--nodisk")) o.disk = false;
    else if (!std::strcmp(argv[i], "--vsync")) o.vsync = true;
  }
  return o;
}

struct App {
  Opts o;
  camera::Orbit cam;
  double exposure_scale = 1.0;  // normalisation: black-body luminance at T_peak -> 0.5
  double exposure = 1.0;
  double sky_exposure = 1.0;
  bool use_sky = false;
  GLuint skytex = 0;
  GLuint prog = 0, lut = 0, img = 0, fbo = 0, vao = 0;
  int w = 0, h = 0;
  bool dragging = false, dirty_exit = false;
  double lastx = 0, lasty = 0;
};

static GLuint compile_program(const std::string& path) {
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "could not open %s\n", path.c_str()); return 0; }
  std::stringstream ss; ss << f.rdbuf();
  const std::string src = ss.str();
  const char* p = src.c_str();
  GLuint sh = gl::CreateShader(GL_COMPUTE_SHADER);
  gl::ShaderSource(sh, 1, &p, nullptr);
  gl::CompileShader(sh);
  GLint ok = 0;
  gl::GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[4096]; gl::GetShaderInfoLog(sh, sizeof log, nullptr, log);
    std::fprintf(stderr, "shader error:\n%s\n", log);
    return 0;
  }
  GLuint pr = gl::CreateProgram();
  gl::AttachShader(pr, sh);
  gl::LinkProgram(pr);
  gl::GetProgramiv(pr, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[4096]; gl::GetProgramInfoLog(pr, sizeof log, nullptr, log);
    std::fprintf(stderr, "link error:\n%s\n", log);
    return 0;
  }
  gl::DeleteShader(sh);
  return pr;
}

static void make_image(App& a, int w, int h) {
  if (a.img) gl::DeleteTextures(1, &a.img);
  a.w = w; a.h = h;
  gl::CreateTextures(GL_TEXTURE_2D, 1, &a.img);
  gl::TextureStorage2D(a.img, 1, GL_RGBA32F, w, h);
  gl::NamedFramebufferTexture(a.fbo, GL_COLOR_ATTACHMENT0, a.img, 0);
  gl::BindImageTexture(0, a.img, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
}

static void render(App& a) {
  const Opts& o = a.o;
  Vec3 fwd, right, up;
  a.cam.basis(fwd, right, up);
  const Vec3 c = a.cam.c_hat();
  const GLuint p = a.prog;
  gl::UseProgram(p);
  auto U = [&](const char* n) { return gl::GetUniformLocation(p, n); };
  gl::ProgramUniform2i(p, U("size"), a.w, a.h);
  gl::ProgramUniform3f(p, U("cHat"), float(c.x), float(c.y), float(c.z));
  gl::ProgramUniform3f(p, U("fwd"), float(fwd.x), float(fwd.y), float(fwd.z));
  gl::ProgramUniform3f(p, U("rightV"), float(right.x), float(right.y), float(right.z));
  gl::ProgramUniform3f(p, U("upV"), float(up.x), float(up.y), float(up.z));
  gl::ProgramUniform1f(p, U("dist"), float(a.cam.dist));
  gl::ProgramUniform1f(p, U("tanHalf"), float(std::tan(0.5 * o.fov * M_PI / 180.0)));
  gl::ProgramUniform1f(p, U("rin"), float(o.rin));
  gl::ProgramUniform1f(p, U("rout"), float(o.rout));
  gl::ProgramUniform1f(p, U("tpeak"), float(o.tpeak));
  gl::ProgramUniform1f(p, U("exposure"), float(a.exposure * a.exposure_scale));
  gl::ProgramUniform1f(p, U("dphi"), float(o.step));
  gl::ProgramUniform1f(p, U("maxPhi"), float(o.maxphi));
  gl::ProgramUniform1i(p, U("doppler"), o.doppler);
  gl::ProgramUniform1i(p, U("showDisk"), o.disk);
  gl::ProgramUniform1i(p, U("useSky"), a.use_sky);
  gl::ProgramUniform1i(p, U("skyMode"), o.skymode);
  gl::ProgramUniform1f(p, U("skyExposure"), float(a.sky_exposure));
  gl::BindTextureUnit(0, a.lut);
  if (a.use_sky) gl::BindTextureUnit(1, a.skytex);
  gl::BindImageTexture(0, a.img, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
  gl::DispatchCompute((a.w + 7) / 8, (a.h + 7) / 8, 1);
  gl::MemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

static void on_scroll(GLFWwindow* win, double, double dy) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(win));
  a->cam.dist *= std::exp(-0.08 * dy);
  a->cam.clamp();
}
static void on_button(GLFWwindow* win, int button, int action, int) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(win));
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    a->dragging = action == GLFW_PRESS;
    glfwGetCursorPos(win, &a->lastx, &a->lasty);
  }
}
static void on_cursor(GLFWwindow* win, double x, double y) {
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(win));
  if (a->dragging) {
    a->cam.phi -= (x - a->lastx) * 0.005;
    a->cam.theta -= (y - a->lasty) * 0.005;
    a->cam.clamp();
  }
  a->lastx = x; a->lasty = y;
}
static void on_key(GLFWwindow* win, int key, int, int action, int) {
  if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
  auto* a = static_cast<App*>(glfwGetWindowUserPointer(win));
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(win, GLFW_TRUE);
  else if (key == GLFW_KEY_D && action == GLFW_PRESS) a->o.doppler = !a->o.doppler;
  else if (key == GLFW_KEY_G && action == GLFW_PRESS) a->o.disk = !a->o.disk;
  else if (key == GLFW_KEY_UP) a->sky_exposure *= 1.1;
  else if (key == GLFW_KEY_DOWN) a->sky_exposure /= 1.1;
  else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) a->exposure *= 1.1;
  else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) a->exposure /= 1.1;
}

int main(int argc, char** argv) {
  App a;
  a.o = parse(argc, argv);
  const bool shot = !a.o.shot.empty();
  if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  if (shot) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow* win = glfwCreateWindow(a.o.w, a.o.h, "black-hole-sim v0.5", nullptr, nullptr);
  if (!win) { std::fprintf(stderr, "could not create an OpenGL 4.6 context\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(a.o.vsync ? 1 : 0);
  if (!gl::load()) return 1;
  std::printf("GL %s | %s\n", gl::GetString(GL_VERSION), gl::GetString(GL_RENDERER));

  a.prog = compile_program(std::string(BHS_SHADER_DIR) + "/blackhole.comp");
  if (!a.prog) return 1;

  const std::vector<float> lutd = lut::build();
  gl::CreateTextures(GL_TEXTURE_1D, 1, &a.lut);
  gl::TextureStorage1D(a.lut, 1, GL_RGB32F, lut::kN);
  gl::TextureSubImage1D(a.lut, 0, 0, lut::kN, GL_RGB, GL_FLOAT, lutd.data());
  gl::TextureParameteri(a.lut, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl::TextureParameteri(a.lut, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl::TextureParameteri(a.lut, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl::CreateFramebuffers(1, &a.fbo);
  gl::CreateVertexArrays(1, &a.vao);

  if (!a.o.nosky) {
    FILE* sf = std::fopen(a.o.sky.c_str(), "rb");
    if (sf) {
      std::fclose(sf);
      const auto t0 = Clock::now();
      sky::Skymap m; std::string err;
      if (!sky::load_exr_half(a.o.sky, m, &err)) { std::fprintf(stderr, "sky: %s\n", err.c_str()); return 1; }
      int levels = 1; for (int n = std::max(m.w, m.h); n > 1; n >>= 1) ++levels;
      gl::CreateTextures(GL_TEXTURE_2D, 1, &a.skytex);
      gl::TextureStorage2D(a.skytex, levels, GL_RGB16F, m.w, m.h);
      gl::TextureSubImage2D(a.skytex, 0, 0, 0, m.w, m.h, GL_RGB, GL_HALF_FLOAT, m.rgb.data());
      gl::GenerateTextureMipmap(a.skytex);
      gl::TextureParameteri(a.skytex, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
      gl::TextureParameteri(a.skytex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      gl::TextureParameteri(a.skytex, GL_TEXTURE_WRAP_S, GL_REPEAT);          // u: right ascension, wraps around
      gl::TextureParameteri(a.skytex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);   // v: polos
      float max_aniso = 1.0f;
      gl::GetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &max_aniso);
      gl::TextureParameterf(a.skytex, GL_TEXTURE_MAX_ANISOTROPY, max_aniso);
      a.use_sky = true;
      std::printf("sky: %s (%dx%d, %d levels, aniso %gx) loaded in %.1f s\n", a.o.sky.c_str(), m.w, m.h, levels, max_aniso,
                  std::chrono::duration<double>(Clock::now() - t0).count());
    } else {
      std::printf("real sky not found (%s): procedural stars (see code/tools/fetch_sky.sh)\n", a.o.sky.c_str());
    }
  }
  a.sky_exposure = a.o.sky_exposure;

  // exposure: black-body radiance at T_peak <-> luminance ~0.5 before tone mapping (as in v0.4)
  const Vec3 ref = lut::lookup(lutd, a.o.tpeak);
  a.exposure_scale = 0.5 / (0.2126 * ref.x + 0.7152 * ref.y + 0.0722 * ref.z);
  a.exposure = a.o.exposure;
  a.cam.theta = a.o.theta * M_PI / 180.0;
  a.cam.phi = a.o.phi * M_PI / 180.0;
  a.cam.dist = a.o.dist;
  a.cam.clamp();

  if (shot) {
    make_image(a, a.o.w, a.o.h);
    std::vector<double> ms;
    for (int f = 0; f < a.o.frames; ++f) {
      const auto t0 = Clock::now();
      render(a);
      gl::Finish();
      ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    double sum = 0; int n = 0;
    for (size_t i = a.o.frames > 1 ? 1 : 0; i < ms.size(); ++i) { sum += ms[i]; ++n; }
    std::printf("%dx%d: first frame %.2f ms (includes JIT compilation); mean of %d frame(s) %.3f ms = %.1f fps\n",
                a.w, a.h, ms[0], n, sum / n, 1000.0 * n / sum);
    std::vector<float> px(size_t(a.w) * a.h * 4);
    gl::GetTextureImage(a.img, 0, GL_RGBA, GL_FLOAT, GLsizei(px.size() * sizeof(float)), px.data());
    std::vector<uint8_t> rgb(size_t(a.w) * a.h * 3);
    for (int j = 0; j < a.h; ++j)
      for (int i = 0; i < a.w; ++i)
        for (int k = 0; k < 3; ++k) {
          const float v = px[(size_t(a.h - 1 - j) * a.w + i) * 4 + k];  // GL: row 0 at the bottom
          rgb[(size_t(j) * a.w + i) * 3 + k] = uint8_t(std::min(1.0f, std::max(0.0f, v)) * 255.0f + 0.5f);
        }
    if (!png::write_file(a.o.shot, png::encode_rgb(a.w, a.h, rgb))) { std::fprintf(stderr, "error writing %s\n", a.o.shot.c_str()); return 1; }
    std::printf("wrote %s\n", a.o.shot.c_str());
    glfwTerminate();
    return 0;
  }

  glfwSetWindowUserPointer(win, &a);
  glfwSetScrollCallback(win, on_scroll);
  glfwSetMouseButtonCallback(win, on_button);
  glfwSetCursorPosCallback(win, on_cursor);
  glfwSetKeyCallback(win, on_key);
  int fw, fh;
  glfwGetFramebufferSize(win, &fw, &fh);
  make_image(a, fw, fh);

  auto last_title = Clock::now();
  int frames = 0;
  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    glfwGetFramebufferSize(win, &fw, &fh);
    if (fw <= 0 || fh <= 0) { glfwWaitEvents(); continue; }
    if (fw != a.w || fh != a.h) make_image(a, fw, fh);
    render(a);
    gl::BlitNamedFramebuffer(a.fbo, 0, 0, 0, a.w, a.h, 0, 0, fw, fh, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glfwSwapBuffers(win);
    ++frames;
    const double dt = std::chrono::duration<double>(Clock::now() - last_title).count();
    if (dt >= 0.5) {
      char t[160];
      std::snprintf(t, sizeof t, "black-hole-sim v0.5 | %.0f fps | %.2f ms/frame | D=%.0f theta=%.0f sky=%.2f%s%s", frames / dt,
                    1000.0 * dt / frames, a.cam.dist, a.cam.theta * 180.0 / M_PI, a.sky_exposure, a.o.doppler ? "" : " | no Doppler",
                    a.o.disk ? "" : " | no disk");
      glfwSetWindowTitle(win, t);
      frames = 0; last_title = Clock::now();
    }
  }
  glfwTerminate();
  return 0;
}
