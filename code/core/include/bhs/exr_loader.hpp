#pragma once
// Reader of half-RGB EXR for bhs::sky::Skymap (tinyexr + miniz, compiled into bhs_exr).
#include <string>
#include <vector>

#include "skymap.hpp"

namespace bhs::sky {
bool load_exr_half(const std::string& path, Skymap& out, std::string* err = nullptr);
// v1.0b: writes a linear RGB EXR in float32 (tinyexr ZIP compression); `rgb` = w*h*3 floats, rows from top to bottom
// float32 RGBA reader (tinyexr LoadEXR): for HDR frames written by save_exr_rgb and for tests
bool load_exr_float(const std::string& path, std::vector<float>& rgba, int& w, int& h, std::string* err = nullptr);
bool save_exr_rgb(const std::string& path, const float* rgb, int w, int h, std::string* err = nullptr);
}
