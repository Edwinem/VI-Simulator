cmake_minimum_required(VERSION 3.15)
project(vi_simulator)

set(CMAKE_CXX_STANDARD 11)



option(WITH_PANGOLIN "Build with Pangolin Visualization" OFF)

if(${WITH_PANGOLIN})
    find_package(PkgConfig)
    pkg_search_module(Eigen3 REQUIRED eigen3)
    include_directories(${Eigen_INCLUDE_DIR})
endif()



find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(src)

add_library(vi_simulator
        src/b3_spline.h
        src/b3_spline.cpp
        src/common_types.cpp src/common_types.h src/common_ops.h src/vio_sim.cpp src/vio_sim.h src/opencv_reimplementations.h)




add_executable(main main.cpp)

target_link_libraries(main vi_simulator)
