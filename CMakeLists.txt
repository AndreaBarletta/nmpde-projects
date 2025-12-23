cmake_minimum_required(VERSION 3.12.0)
project(shallow_waters LANGUAGES CXX C)

include(cmake-common.cmake)

add_executable(shallow_waters src/Shallow_waters.cpp src/main.cpp)
deal_ii_setup_target(shallow_waters)

option(TEST_MANUFACTURED_H "Enable manufactured solution for height" OFF)
if(TEST_MANUFACTURED_H)
  target_compile_definitions(shallow_waters PRIVATE TEST_MANUFACTURED_H=1)
  message(STATUS "TEST_MANUFACTURED_H enabled")
endif()

option(TEST_MANUFACTURED_U "Enable manufactured solution for velocity" OFF)
if(TEST_MANUFACTURED_U)
  target_compile_definitions(shallow_waters PRIVATE TEST_MANUFACTURED_U=1)
  message(STATUS "TEST_MANUFACTURED_U enabled")
endif()
