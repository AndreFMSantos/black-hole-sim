#pragma once
// Where the data lives (GRMHD .bgm snapshots, timelapse work): $BHS_DATA if set, else the folder CMake chose (BHS_DATA_DIR, by default <repo>/data).
#include <cstdlib>
#include <string>

inline std::string bhs_data(const std::string& rel = "") {
  const char* e = std::getenv("BHS_DATA");
  const std::string base = (e && *e) ? std::string(e) : std::string(BHS_DATA_DEFAULT);
  return rel.empty() ? base : base + rel;
}
