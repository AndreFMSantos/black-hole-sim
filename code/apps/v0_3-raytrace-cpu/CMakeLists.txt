find_package(Threads REQUIRED)
add_executable(v0_3-raytrace-cpu main.cpp)
target_link_libraries(v0_3-raytrace-cpu PRIVATE bhs_core Threads::Threads)
target_compile_options(v0_3-raytrace-cpu PRIVATE -Wall -Wextra)
