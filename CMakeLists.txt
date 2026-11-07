cmake_minimum_required(VERSION 3.18)
project(jacobi2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

add_compile_options(
	"$<$<COMPILE_LANGUAGE:CXX>:-march=native;-ffp-contract=off;-Wall;-Wextra>"
)

add_executable(jacobi_seq src/seq/main.cpp)
target_include_directories(jacobi_seq PRIVATE include)
