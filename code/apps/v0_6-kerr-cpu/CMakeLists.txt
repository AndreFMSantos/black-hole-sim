find_package(Threads REQUIRED)
add_executable(v0_6-kerr-cpu main.cpp)
target_link_libraries(v0_6-kerr-cpu PRIVATE bhs_core bhs_exr Threads::Threads)
target_compile_options(v0_6-kerr-cpu PRIVATE -Wall -Wextra)
