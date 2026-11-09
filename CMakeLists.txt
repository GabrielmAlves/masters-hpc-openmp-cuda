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

add_library(jacobi_common STATIC
	src/common/config.cpp
	src/common/grid.cpp
	src/common/output.cpp
	src/common/timing.cpp
)
target_include_directories(jacobi_common PUBLIC include)

add_executable(jacobi_seq
	src/seq/main.cpp
	src/seq/solver.cpp
)
target_link_libraries(jacobi_seq PRIVATE jacobi_common)

add_executable(compare tools/compare.cpp)
target_link_libraries(compare PRIVATE jacobi_common)
