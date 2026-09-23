cmake_minimum_required(VERSION 3.16)

project(Lab1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCES ./src/main.cpp ./src/formulas.cpp ./src/analysis.cpp)

include_directories(./include)

add_executable(Lab1 ${SOURCES})