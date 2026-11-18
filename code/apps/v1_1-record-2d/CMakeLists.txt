add_executable(v1_1-record-2d main.cpp)
target_link_libraries(v1_1-record-2d PRIVATE bhs_core)
target_compile_options(v1_1-record-2d PRIVATE -Wall -Wextra)
