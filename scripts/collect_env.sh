#!/usr/bin/env bash
# Registra o hardware e o software usados nos experimentos em results/env.txt.
cd "$(dirname "$0")/.."
mkdir -p results
{
	echo "# Ambiente de execução ($(date -Iseconds))"
	echo
	echo "## CPU"
	lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|L1d|L2|L3|Flags" | sed 's/Flags:.*\(avx2\).*/Flags: ... avx2 .../'
	echo
	echo "## Memória"
	free -h
	echo
	echo "## GPU"
	nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,clocks.max.sm,clocks.max.mem --format=csv 2>/dev/null || echo "sem GPU NVIDIA"
	echo
	echo "## Sistema"
	grep PRETTY_NAME /etc/os-release
	uname -r
	echo
	echo "## Compiladores e ferramentas"
	gcc --version | head -1
	(nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | tail -2
	cmake --version | head -1
	echo
	echo "## Flags"
	grep -E "JACOBI_CPU_FLAGS|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES" build/CMakeCache.txt 2>/dev/null
} > results/env.txt
cat results/env.txt
