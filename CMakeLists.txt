cmake_minimum_required(VERSION 3.12.0)
project(shallow_waters LANGUAGES CXX C)

option(TEST_MANUFACTURED_H "Enable manufactured solution for height" ON)
if(TEST_MANUFACTURED_H)
  add_compile_definitions(TEST_MANUFACTURED_H=1)
  add_compile_definitions(TEST_MANUFACTURED=1)
endif()

option(TEST_MANUFACTURED_U "Enable manufactured solution for velocity" OFF)
if(TEST_MANUFACTURED_U)
  add_compile_definitions(TEST_MANUFACTURED_U=1)
  add_compile_definitions(TEST_MANUFACTURED=1)
endif()

include(cmake-common.cmake)

add_executable(shallow_waters src/Shallow_waters.cpp src/main.cpp)
deal_ii_setup_target(shallow_waters)
