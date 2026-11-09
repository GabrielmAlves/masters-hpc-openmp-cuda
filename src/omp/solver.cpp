#include "solver.hpp"

#include <utility>

namespace {
	void sweep(const double* __restrict__ in, double* __restrict__ out, std::size_t n) {
		#pragma omp parallel for schedule(static)
			for (std::size_t i = 1; i < n - 1; i++) {
				const double* up = in + (i - 1)*n;
				const double* middle = in +i*n;
				const double* down = in + (i + 1)*n;
				double* dst = out + i*n;

				for (std::size_t j = 1; j < n - 1; j++) {
					dst[j] = 0.25*(up[j] + down[j] + middle[j - 1] + middle[j + 1]);
				}
			}
	}
} // namespace

const double* solve(double* current, double* next, std::size_t n, int iters) {
	for (int k = 0; k < iters; k++) {
		sweep(current, next, n);
		std::swap(current, next);
	}
	
	return current;
}
