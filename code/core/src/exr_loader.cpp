#include <bhs/exr_loader.hpp>

#include <tinyexr.h>

namespace bhs::sky {

bool load_exr_half(const std::string& path, Skymap& out, std::string* err) {
  auto fail = [&](const char* msg, const char* detail = nullptr) {
    if (err) *err = std::string(msg) + (detail ? std::string(": ") + detail : "");
    return false;
  };
  EXRVersion ver;
  if (ParseEXRVersionFromFile(&ver, path.c_str()) != TINYEXR_SUCCESS) return fail("not a valid EXR", path.c_str());
  if (ver.multipart || ver.non_image) return fail("EXR multipart/deep not supported");
  EXRHeader hdr;
  InitEXRHeader(&hdr);
  const char* msg = nullptr;
  if (ParseEXRHeaderFromFile(&hdr, &ver, path.c_str(), &msg) != TINYEXR_SUCCESS) {
    std::string m = msg ? msg : "";
    FreeEXRErrorMessage(msg);
    return fail("EXR header", m.c_str());
  }
  int ir = -1, ig = -1, ib = -1;
  for (int i = 0; i < hdr.num_channels; ++i) {
    hdr.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;  // do not expand to float: 4x memory
    const std::string n = hdr.channels[i].name;
    if (n == "R") ir = i; else if (n == "G") ig = i; else if (n == "B") ib = i;
  }
  if (ir < 0 || ig < 0 || ib < 0) { FreeEXRHeader(&hdr); return fail("missing R/G/B channels"); }
  EXRImage img;
  InitEXRImage(&img);
  if (LoadEXRImageFromFile(&img, &hdr, path.c_str(), &msg) != TINYEXR_SUCCESS) {
    std::string m = msg ? msg : "";
    FreeEXRErrorMessage(msg);
    FreeEXRHeader(&hdr);
    return fail("EXR read", m.c_str());
  }
  if (img.tiles || !img.images) { FreeEXRImage(&img); FreeEXRHeader(&hdr); return fail("tiled EXR not supported"); }
  out.w = img.width; out.h = img.height;
  out.rgb.assign(size_t(out.w) * out.h * 3, 0);
  const int idx[3] = {ir, ig, ib};
  for (int c = 0; c < 3; ++c) {
    const uint16_t* src = reinterpret_cast<const uint16_t*>(img.images[idx[c]]);
    for (size_t i = 0, n = size_t(out.w) * out.h; i < n; ++i) out.rgb[3 * i + c] = src[i];
  }
  FreeEXRImage(&img);
  FreeEXRHeader(&hdr);
  return true;
}

bool save_exr_rgb(const std::string& path, const float* rgb, int w, int h, std::string* err) {
  const char* msg = nullptr;
  if (SaveEXR(rgb, w, h, 3, 0 /* float32 */, path.c_str(), &msg) != TINYEXR_SUCCESS) {
    if (err) *err = std::string("SaveEXR: ") + (msg ? msg : "error");
    if (msg) FreeEXRErrorMessage(msg);
    return false;
  }
  return true;
}

bool load_exr_float(const std::string& path, std::vector<float>& rgba, int& w, int& h, std::string* err) {
  float* out = nullptr; const char* msg = nullptr;
  if (LoadEXR(&out, &w, &h, path.c_str(), &msg) != TINYEXR_SUCCESS) {
    if (err) *err = std::string("LoadEXR: ") + (msg ? msg : "error");
    if (msg) FreeEXRErrorMessage(msg);
    return false;
  }
  rgba.assign(out, out + size_t(w) * h * 4);
  free(out);
  return true;
}

}  // namespace bhs::sky
