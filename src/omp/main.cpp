#include <omp.h>

#include "config.hpp"
#include "grid.hpp"
#include "output.hpp"
#include "solver.hpp"
#include "timing.hpp"

namespace {
	// Inicializa as linhas internas com a mesma divisão do sweep (first touch):
	// cada thread toca primeiro as linhas que vai calcular depois.
	void init_parallel(double* a, std::size_t n, InitKind kind) {
		init_rows(a, n, kind, 0, 1);
		init_rows(a, n, kind, n - 1, n);

		#pragma omp parallel for schedule(static)
			for (std::size_t i = 1; i < n - 1; i++) {
				init_rows(a, n, kind, i, i + 1);
			}
	}
} // namespace

int main(int argc, char** argv) {
	const Options o = parse_args(argc, argv);
	const int threads = omp_get_max_threads();

	if (o.header) {
		print_csv_header();
	}

	GridPtr a = alloc_grid(o.n);
	GridPtr b = alloc_grid(o.n);
	const double* result = nullptr;

	for (int rep = 0; rep < o.reps; rep++) {
		init_parallel(a.get(), o.n, o.init);
		init_parallel(b.get(), o.n, o.init);

		const double start = now_seconds();
		result = solve(a.get(), b.get(), o.n, o.iters);
		const double elapsed = now_seconds() - start;

		print_csv_row({"omp", o.n, o.iters, threads, rep, elapsed, 0.0});
	}

	if (!o.dump.empty()) {
		write_grid(o.dump, result, o.n, o.iters);
	}

	return 0;
}
