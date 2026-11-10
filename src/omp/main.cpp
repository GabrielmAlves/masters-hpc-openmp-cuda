#include <omp.h>

#include <cstdio>
#include <string>

#include "config.hpp"
#include "grid.hpp"
#include "output.hpp"
#include "solver.hpp"
#include "timing.hpp"
#include "variants.hpp"

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

	enum class Variant { Basic, Single, Temporal };
} // namespace

int main(int argc, char** argv) {
	const Options o = parse_args(argc, argv);
	const int threads = omp_get_max_threads();

	if (o.use_float) {
		std::fprintf(stderr, "erro: a versão OpenMP só usa double\n");
		return 2;
	}

	Variant variant = Variant::Basic;
	std::string label = "basic";
	if (o.variant.empty() || o.variant == "basic") {
		variant = Variant::Basic;
	} else if (o.variant == "single") {
		variant = Variant::Single;
		label = "single";
	} else if (o.variant == "temporal") {
		variant = Variant::Temporal;
		label = "temporal_b" + std::to_string(o.tile) + "_t" + std::to_string(o.tsteps);
	} else {
		std::fprintf(stderr, "erro: variante OpenMP desconhecida: %s (basic, single, temporal)\n",
					 o.variant.c_str());
		return 2;
	}

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
		switch (variant) {
		case Variant::Basic:
			result = solve(a.get(), b.get(), o.n, o.iters);
			break;
		case Variant::Single:
			result = solve_single(a.get(), b.get(), o.n, o.iters);
			break;
		case Variant::Temporal:
			result = solve_temporal(a.get(), b.get(), o.n, o.iters, o.tile, o.tsteps);
			break;
		}
		const double elapsed = now_seconds() - start;

		print_csv_row({"omp", label, "double", o.n, o.iters, threads, 0, 0, rep, elapsed, 0.0});
	}

	if (!o.dump.empty()) {
		write_grid(o.dump, result, o.n, o.iters);
	}

	return 0;
}
