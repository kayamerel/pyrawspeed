cmake_minimum_required(VERSION 3.22)
project(pyrawspeed CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Disable rawspeed's test/bench/fuzz targets — we only need the library
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKING OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(RAWSPEED_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

add_subdirectory(rawspeed)

nanobind_add_module(_pyrawspeed src/pyrawspeed.cpp)

target_link_libraries(_pyrawspeed PRIVATE rawspeed)

install(TARGETS _pyrawspeed DESTINATION pyrawspeed)
