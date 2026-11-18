# v0.6b: Kerr on CUDA. Linux with nvcc only (see the top-level CMakeLists.txt); links OpenGL via glfw for the window.
add_executable(v0_7-cinematic-cuda main.cpp render.cu)
set_target_properties(v0_7-cinematic-cuda PROPERTIES CUDA_ARCHITECTURES "${BHS_CUDA_ARCH}" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(v0_7-cinematic-cuda PRIVATE bhs_core bhs_exr glfw CUDA::cudart)
target_compile_options(v0_7-cinematic-cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
                                              $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
target_compile_definitions(v0_7-cinematic-cuda PRIVATE BHS_SKY_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/sky/starmap_2020_8k.exr")
add_test(NAME v0_7_selftest COMMAND v0_7-cinematic-cuda --selftest --nosky)
