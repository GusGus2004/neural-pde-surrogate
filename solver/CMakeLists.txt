cmake_minimum_required(VERSION 3.16)
project(euler_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(euler src/main.cpp)
target_compile_options(euler PRIVATE -Wall -Wextra -O3)


add_executable(hdf5_check src/hdf5_check.cpp)
target_include_directories(hdf5_check PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(hdf5_check PRIVATE ${HDF5_LIBRARIES})
target_compile_options(hdf5_check PRIVATE -Wall -Wextra -O3)