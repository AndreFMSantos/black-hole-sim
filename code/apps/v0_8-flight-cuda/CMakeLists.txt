# v0.6b: Kerr on CUDA. Linux with nvcc only (see the top-level CMakeLists.txt); links OpenGL via glfw for the window.
# v0.8c: OPTIONAL OptiX denoiser at build time. Only if the headers exist (code/third_party/optix-dev, outside Git, see
# tools/fetch_optix.sh); otherwise a stub is compiled and the rest stays the same. -DBHS_OPTIX=OFF turns it off even with the headers.
option(BHS_OPTIX "optional OptiX denoiser (needs code/third_party/optix-dev)" ON)
set(BHS_OPTIX_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/optix-dev/include")
if(BHS_OPTIX AND EXISTS "${BHS_OPTIX_INCLUDE}/optix.h")
  set(BHS_DENOISE_SRC denoise_optix.cpp)
  set(BHS_HAVE_OPTIX ON)
  message(STATUS "v0_8-flight-cuda: denoiser OptiX ACTIVADO (${BHS_OPTIX_INCLUDE})")
else()
  set(BHS_DENOISE_SRC denoise_stub.cpp)
  set(BHS_HAVE_OPTIX OFF)
  message(STATUS "v0_8-flight-cuda: no OptiX (BHS_OPTIX=${BHS_OPTIX}; headers ${BHS_OPTIX_INCLUDE}/optix.h not found or turned off)")
endif()
add_executable(v0_8-flight-cuda main.cpp render.cu ${BHS_DENOISE_SRC})
if(BHS_HAVE_OPTIX)
  target_include_directories(v0_8-flight-cuda PRIVATE "${BHS_OPTIX_INCLUDE}")
  target_compile_definitions(v0_8-flight-cuda PRIVATE BHS_HAVE_OPTIX=1)
  target_link_libraries(v0_8-flight-cuda PRIVATE ${CMAKE_DL_LIBS})
  if(TARGET CUDA::cupti)  # optional: --denoise-kernels lists the denoiser's CUDA kernels (CUPTI)
    target_link_libraries(v0_8-flight-cuda PRIVATE CUDA::cupti)
    target_compile_definitions(v0_8-flight-cuda PRIVATE BHS_HAVE_CUPTI=1)
  endif()
endif()
set_target_properties(v0_8-flight-cuda PROPERTIES CUDA_ARCHITECTURES "${BHS_CUDA_ARCH}" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(v0_8-flight-cuda PRIVATE bhs_core bhs_exr glfw CUDA::cudart)
target_compile_options(v0_8-flight-cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
                                              $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
target_compile_definitions(v0_8-flight-cuda PRIVATE BHS_SKY_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/sky/starmap_2020_8k.exr")
add_test(NAME v0_8_selftest COMMAND v0_8-flight-cuda --selftest --nosky)
# regression of the physical look (byte-identical to v0.7d): md5 of 3 renders
add_test(NAME v0_8_v07d_regression COMMAND ${CMAKE_COMMAND} -E env BHS_BIN=$<TARGET_FILE:v0_8-flight-cuda>
         ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/regress_v07d.sh)
# selftest of the "no OptiX" path (the stub): --denoise optix must fail with a clear message; only when built without OptiX
if(NOT BHS_HAVE_OPTIX)
  add_test(NAME v0_8_no_optix COMMAND v0_8-flight-cuda --selftest-denoise)
endif()
