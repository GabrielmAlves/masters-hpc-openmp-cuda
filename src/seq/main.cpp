#include "config.hpp"
#include "grid.hpp"
#include "output.hpp"

// Provisório: testa o módulo output. O método de Jacobi ainda não está aqui.
int main(int argc, char** argv) {
	const Options o = parse_args(argc, argv);

	print_csv_header();
	print_csv_row({"seq", 1002, 1000, 1, 0, 2.0, 0.0});

	GridPtr a = alloc_grid(o.n);
	init_rows(a.get(), o.n, o.init, 0, o.n);
	if (!o.dump.empty()) {
		write_grid(o.dump, a.get(), o.n, o.iters);
	}

	return 0;
}
