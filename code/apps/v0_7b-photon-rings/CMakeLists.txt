find_package(Threads REQUIRED)
add_executable(v0_7b-photon-rings main.cpp)
target_link_libraries(v0_7b-photon-rings PRIVATE bhs_core Threads::Threads)
target_compile_options(v0_7b-photon-rings PRIVATE -Wall -Wextra)
