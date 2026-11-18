add_library(bhs_core INTERFACE)
target_include_directories(bhs_core INTERFACE include)
target_compile_features(bhs_core INTERFACE cxx_std_20)

# EXR reader (heavy dependency stays out of the INTERFACE bhs_core; only whoever needs to read EXR links bhs_exr)
add_library(bhs_exr STATIC src/exr_loader.cpp ${tinyexr_SOURCE_DIR}/tinyexr.cc ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c)
target_include_directories(bhs_exr PUBLIC include PRIVATE ${tinyexr_SOURCE_DIR} ${tinyexr_SOURCE_DIR}/deps/miniz)
target_compile_features(bhs_exr PUBLIC cxx_std_20)
target_compile_options(bhs_exr PRIVATE -w)  # third-party code

add_executable(bhs_core_tests tests/main.cpp tests/test_integrators.cpp tests/test_schwarzschild.cpp tests/test_render.cpp tests/test_disk.cpp tests/test_kerr.cpp tests/test_nt.cpp tests/test_rings.cpp tests/test_boost.cpp tests/test_kerr_rings.cpp tests/test_pilot.cpp tests/test_realtime.cpp tests/test_skymap.cpp tests/test_disk_emission.cpp tests/test_volume.cpp tests/test_turbulence.cpp tests/test_slowlight.cpp tests/test_json_quat.cpp tests/test_route.cpp tests/test_torus.cpp tests/test_synchrotron.cpp tests/test_grmhd.cpp tests/test_wormhole.cpp tests/test_kerr_newman.cpp tests/test_nt_kn.cpp tests/test_kerr_analytic.cpp tests/test_lamppost.cpp)
target_link_libraries(bhs_core_tests PRIVATE bhs_core bhs_exr doctest::doctest)
target_compile_definitions(bhs_core_tests PRIVATE BHS_SKY_EXR="${CMAKE_CURRENT_SOURCE_DIR}/../../assets/sky/starmap_2020_8k.exr")
target_compile_options(bhs_core_tests PRIVATE -Wall -Wextra)
add_test(NAME core COMMAND bhs_core_tests)
