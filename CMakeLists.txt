cmake_minimum_required(VERSION 3.10)
project(vi_simulator)

set(CMAKE_CXX_STANDARD 11)



option(WITH_PANGOLIN "Build with Pangolin Visualization" OFF)

if(${WITH_PANGOLIN})


# Hack to disable CPack in Pangolin.
macro(include)
    if(NOT ${ARGV0} STREQUAL "CPack")
        _include(${ARGN})
    endif()
endmacro()
add_subdirectory(Pangolin EXCLUDE_FROM_ALL)


# fix pangolin: gcc
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(pangolin PRIVATE "-Wno-implicit-fallthrough")
endif()

# fix pangolin: macOS
if(APPLE)
    target_compile_options(pangolin PRIVATE "-Wno-objc-missing-super-calls")
endif()

# fix pangolin: macOS >= 10.14 Mojave
if(APPLE AND CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL 18.0.0)
    target_compile_options(pangolin PRIVATE "-Wno-deprecated-declarations")
endif()

# fix pangolin: clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(pangolin PRIVATE "-Wno-null-pointer-arithmetic")
endif()

# fix pangolin: clang >= 8.0
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0.0)
    target_compile_options(pangolin PRIVATE "-Wno-defaulted-function-deleted")
endif()

# check here the directories for the pangolin and opengv targets, and
# confirm that the eigen-related include dirs match.
foreach(_target pangolin)
    get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_dir IN LISTS _include_dirs)
        if("${_dir}" MATCHES ".*/eigen3?(/unsupported)?$")
            if(NOT _dir MATCHES "^${EIGEN3_INCLUDE_DIR}.*")
            endif()
        endif()
    endforeach()
endforeach()

# opengv and pangolin assume that eigen is found outside the source
# directory and thus include it in INTERFACE_INCLUDE_DIRECTORIES,
# which makes cmake complain that that property contains paths in the
# source folder. Thus, we update the property to only include these
# eigen paths in the "BUILD_INTERFACE" (not "INSTALL").
if (EIGEN3_INCLUDE_DIR MATCHES "^${CMAKE_SOURCE_DIR}.*")
    foreach(_target pangolin)
        get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
        set(_include_dirs_new "")
        foreach(_dir IN LISTS _include_dirs)
            if(_dir MATCHES ".*/eigen(/unsupported)?$")
                string(REGEX REPLACE "(^${CMAKE_SOURCE_DIR}.*$)" "$<BUILD_INTERFACE:\\1>" _dir "${_dir}")
            endif()
            list(APPEND _include_dirs_new "${_dir}")
        endforeach()
        set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs_new}")
    endforeach()
endif()

endif()



find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(src)

add_library(vi_simulator
        src/b3_spline.h
        src/b3_spline.cpp
        src/common_types.cpp src/common_types.h src/common_ops.h src/vio_sim.cpp src/vio_sim.h src/opencv_reimplementations.h)

add_executable(main examples/main.cpp)
add_executable(profile examples/profile.cpp)

target_link_libraries(main vi_simulator)
target_link_libraries(profile vi_simulator)

if(${WITH_PANGOLIN})
    add_executable(visualization examples/visualized_sim.cpp)
    target_link_libraries(visualization vi_simulator pangolin)
endif()