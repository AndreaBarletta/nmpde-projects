cmake_minimum_required(VERSION 3.12.0)
project(shallow_waters LANGUAGES CXX C)

include(cmake-common.cmake)

add_executable(shallow_waters src/main.cpp)
deal_ii_setup_target(shallow_waters)
