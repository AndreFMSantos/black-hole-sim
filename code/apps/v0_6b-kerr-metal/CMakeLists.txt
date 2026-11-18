# v0.6b-metal: Kerr in Metal (Apple Silicon), sharing the render.hpp interface of v0.6b-kerr-cuda.
add_executable(v0_6b-kerr-metal
  main.cpp
  render_metal.mm
  ../v3_1-kerr-cuda-ui/denoise_stub.cpp
)
target_include_directories(v0_6b-kerr-metal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../v3_1-kerr-cuda-ui")
target_link_libraries(v0_6b-kerr-metal PRIVATE
  bhs_core
  bhs_exr
  glfw
  "-framework Metal"
  "-framework Foundation"
  "-framework OpenGL"
)
target_compile_options(v0_6b-kerr-metal PRIVATE -Wall -Wextra -Wno-deprecated-declarations)
set_source_files_properties(render_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_compile_definitions(v0_6b-kerr-metal PRIVATE
  BHS_SKY_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/sky/starmap_2020_8k.exr"
  BHS_METAL_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME v0_6b_metal_selftest COMMAND v0_6b-kerr-metal --selftest --nosky)
add_test(NAME v08c_metal_no_optix COMMAND v0_6b-kerr-metal --selftest-denoise)
