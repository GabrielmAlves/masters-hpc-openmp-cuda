#include <cstdio>

#include "config.hpp"
#include "grid.hpp"
#include "output.hpp"
#include "solver.hpp"
#include "timing.hpp"

int main(int argc, char** argv) {
	const Options o = parse_args(argc, argv);
	if (!o.variant.empty() && o.variant != "base") {
		std::fprintf(stderr, "erro: a versão sequencial só tem a variante base\n");
		return 2;
	}
	if (o.use_float) {
		std::fprintf(stderr, "erro: a versão sequencial só usa double\n");
		return 2;
	}

	if (o.header) {
		print_csv_header();
	}

	GridPtr a = alloc_grid(o.n);
	GridPtr b = alloc_grid(o.n);
	const double* result = nullptr;

	for (int rep = 0; rep < o.reps; rep++) {
		init_rows(a.get(), o.n, o.init, 0, o.n);
		init_rows(b.get(), o.n, o.init, 0, o.n);

		const double start = now_seconds();
		result = solve(a.get(), b.get(), o.n, o.iters);
		const double elapsed = now_seconds() - start;

		print_csv_row({"seq", "base", "double", o.n, o.iters, 1, 0, 0, rep, elapsed, 0.0});
	}

	if (!o.dump.empty()) {
		write_grid(o.dump, result, o.n, o.iters);
	}

	return 0;
}
