#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "grid.hpp"
#include "output.hpp"

namespace {

[[noreturn]] void fail(const std::string& message) {
	std::fprintf(stderr, "erro: %s\nuso: compare <referencia.bin> <teste.bin> [--tol valor]\n",
				 message.c_str());
	std::exit(2);
}

double parse_tolerance(const std::string& text) {
	double value = 0.0;
	std::size_t used = 0;
	try {
		value = std::stod(text, &used);
	} catch (const std::exception&) {
		fail("--tol precisa ser um número: " + text);
	}
	if (used != text.size() || !std::isfinite(value) || value < 0.0) {
		fail("--tol precisa ser um número finito e não negativo: " + text);
	}
	return value;
}

struct Worst {
	double error = 0.0;
	std::size_t i = 0;
	std::size_t j = 0;
	double ref = 0.0;
	double test = 0.0;
	bool found = false;

	void update(double e, std::size_t row, std::size_t col, double r, double t) {
		if (!found || e > error) {
			error = e;
			i = row;
			j = col;
			ref = r;
			test = t;
			found = true;
		}
	}

	void print(const char* label) const {
		if (!found) {
			std::printf("%-15s -\n", label);
			return;
		}
		std::printf("%-15s %.6e em (%zu, %zu): ref = %.17g, teste = %.17g\n", label, error, i, j,
					ref, test);
	}
};

} // namespace

int main(int argc, char** argv) {
	std::string paths[2];
	int path_count = 0;
	double tol = 1e-12;

	for (int a = 1; a < argc; a++) {
		const std::string arg = argv[a];
		if (arg == "--tol") {
			if (a + 1 >= argc) {
				fail("falta o valor de --tol");
			}
			tol = parse_tolerance(argv[++a]);
		} else if (arg.rfind("--", 0) == 0) {
			fail("argumento desconhecido: " + arg);
		} else if (path_count < 2) {
			paths[path_count++] = arg;
		} else {
			fail("arquivos demais: " + arg);
		}
	}
	if (path_count != 2) {
		fail("informe dois arquivos");
	}

	std::size_t n_ref = 0, n_test = 0;
	int iters_ref = 0, iters_test = 0;
	const GridPtr ref = read_grid(paths[0], n_ref, iters_ref);
	const GridPtr test = read_grid(paths[1], n_test, iters_test);

	if (n_ref != n_test) {
		fail("N diferente: " + std::to_string(n_ref) + " e " + std::to_string(n_test));
	}
	if (iters_ref != iters_test) {
		fail("iters diferente: " + std::to_string(iters_ref) + " e " + std::to_string(iters_test));
	}
	const std::size_t n = n_ref;

	const double zero_limit = std::numeric_limits<double>::min();

	Worst abs_err, rel_err, abs_err_at_zero;
	std::size_t zero_count = 0;
	std::size_t nonfinite_count = 0;
	Worst first_nonfinite;

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			const double r = ref[i * n + j];
			const double t = test[i * n + j];

			if (!std::isfinite(r) || !std::isfinite(t)) {
				if (nonfinite_count++ == 0) {
					first_nonfinite.update(0.0, i, j, r, t);
				}
				continue;
			}

			const double e = std::fabs(t - r);
			abs_err.update(e, i, j, r, t);

			if (std::fabs(r) < zero_limit) {
				zero_count++;
				abs_err_at_zero.update(e, i, j, r, t);
			} else {
				rel_err.update(e / std::fabs(r), i, j, r, t);
			}
		}
	}

	const bool pass =
		nonfinite_count == 0 && rel_err.error <= tol && abs_err_at_zero.error <= tol;

	std::printf("ref:            %s\n", paths[0].c_str());
	std::printf("teste:          %s\n", paths[1].c_str());
	std::printf("N = %zu, iters = %d, tolerância = %.3e\n", n, iters_ref, tol);
	abs_err.print("erro_abs máx:");
	rel_err.print("erro_rel máx:");
	std::printf("ref zero:       %zu pontos\n", zero_count);
	if (zero_count > 0) {
		abs_err_at_zero.print("erro_abs ref 0:");
	}
	if (nonfinite_count > 0) {
		std::printf("não finitos:    %zu pontos, o primeiro em (%zu, %zu): ref = %g, teste = %g\n",
					nonfinite_count, first_nonfinite.i, first_nonfinite.j, first_nonfinite.ref,
					first_nonfinite.test);
	}
	std::printf("resultado:      %s\n", pass ? "OK" : "FALHOU");

	return pass ? 0 : 1;
}
