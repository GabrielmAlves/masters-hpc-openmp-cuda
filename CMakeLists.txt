cmake_minimum_required(VERSION 3.18)
project(jacobi2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

# Flags de arquitetura da CPU. O padrão usa todas as extensões da máquina
# (AVX2 no i5-13400F). Para o estudo de vetorização:
#   -DJACOBI_CPU_FLAGS="-march=x86-64 -mtune=native"            (só SSE2)
#   -DJACOBI_CPU_FLAGS="-march=native -fno-tree-vectorize"      (sem vetorizar)
set(JACOBI_CPU_FLAGS "-march=native" CACHE STRING "Flags de arquitetura da CPU")
separate_arguments(JACOBI_CPU_FLAGS_LIST UNIX_COMMAND "${JACOBI_CPU_FLAGS}")

add_compile_options(
	"$<$<COMPILE_LANGUAGE:CXX>:${JACOBI_CPU_FLAGS_LIST};-ffp-contract=off;-Wall;-Wextra>"
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

find_package(OpenMP REQUIRED)

add_executable(jacobi_omp
	src/omp/main.cpp
	src/omp/solver.cpp
	src/omp/variants.cpp
)
target_link_libraries(jacobi_omp PRIVATE jacobi_common OpenMP::OpenMP_CXX)

add_executable(compare tools/compare.cpp)
target_link_libraries(compare PRIVATE jacobi_common)

# Referência de banda de memória da CPU (cópia), para o modelo de desempenho.
add_executable(bandwidth_cpu tools/bandwidth_cpu.cpp)
target_link_libraries(bandwidth_cpu PRIVATE OpenMP::OpenMP_CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3060
	endif()
	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)

	add_executable(jacobi_cuda
		src/cuda/main.cu
		src/cuda/solver.cu
	)
	target_link_libraries(jacobi_cuda PRIVATE jacobi_common)
	target_compile_options(jacobi_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")

	# Referência de banda de memória da GPU (cudaMemcpy e kernel de cópia).
	add_executable(bandwidth_gpu tools/bandwidth_gpu.cu)
else()
	message(STATUS "CUDA não encontrado: jacobi_cuda não será compilado")
endif()
