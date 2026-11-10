// Banda de memória da CPU: cópia b[i] = a[i] em vetores de 256 MB (fora da cache).
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main() {
	const size_t n = 4096ull * 4096ull * 2;  // 256 MB por vetor
	double* a = static_cast<double*>(std::aligned_alloc(64, n * 8));
	double* b = static_cast<double*>(std::aligned_alloc(64, n * 8));
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < n; i++) { a[i] = 1.0 + i % 7; b[i] = 0.0; }
	for (int r = 0; r < 2; r++) {
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n; i++) b[i] = a[i];
	}
	const int reps = 10;
	auto t0 = std::chrono::steady_clock::now();
	for (int r = 0; r < reps; r++) {
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n; i++) b[i] = a[i];
	}
	double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	// 16 B por elemento (leitura + escrita); com write-allocate, o tráfego real é 24 B.
	std::printf("%2d threads: cópia %.1f GB/s (16 B/elem), %.1f GB/s (24 B/elem, com write-allocate) [b0=%g]\n",
				omp_get_max_threads(), 16.0 * n * reps / s / 1e9, 24.0 * n * reps / s / 1e9, b[1]);
	return 0;
}
