add_executable(sky-stats main.cpp)
target_link_libraries(sky-stats PRIVATE bhs_core bhs_exr)
target_compile_options(sky-stats PRIVATE -Wall -Wextra)
