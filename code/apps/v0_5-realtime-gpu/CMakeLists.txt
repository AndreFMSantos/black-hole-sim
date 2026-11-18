# v0.5 is Linux only (OpenGL 4.6 + compute shaders; macOS stays at 4.1 without compute).
add_executable(v0_5-realtime-gpu main.cpp)
target_link_libraries(v0_5-realtime-gpu PRIVATE bhs_core bhs_exr glfw)
target_compile_options(v0_5-realtime-gpu PRIVATE -Wall -Wextra)
target_compile_definitions(v0_5-realtime-gpu PRIVATE BHS_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
target_compile_definitions(v0_5-realtime-gpu PRIVATE BHS_SKY_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/sky/starmap_2020_8k.exr")
