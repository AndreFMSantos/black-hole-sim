find_package(Threads REQUIRED)
add_executable(v0_4-accretion-disk main.cpp)
target_link_libraries(v0_4-accretion-disk PRIVATE bhs_core bhs_exr Threads::Threads)
target_compile_options(v0_4-accretion-disk PRIVATE -Wall -Wextra)
