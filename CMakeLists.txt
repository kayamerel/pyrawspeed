cmake_minimum_required(VERSION 3.22)
project(pyrawspeed CXX)

find_package(Python 3.10 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# Disable rawspeed's test/bench/fuzz targets — we only need the library
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKING OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(RAWSPEED_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

add_subdirectory(rawspeed)

# STABLE_ABI builds an abi3 module on Python >= 3.12 and is a no-op below
nanobind_add_module(_pyrawspeed STABLE_ABI src/pyrawspeed.cpp)

target_link_libraries(_pyrawspeed PRIVATE rawspeed)

install(TARGETS _pyrawspeed DESTINATION pyrawspeed)

# Bundle the camera database from the pinned submodule into the package
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rawspeed/data/cameras.xml"
        DESTINATION pyrawspeed/data)
