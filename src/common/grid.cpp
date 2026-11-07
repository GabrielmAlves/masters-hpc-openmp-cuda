#include "grid.hpp"

#include <cstdint>
#include <cstdio>

namespace {
	constexpr std::size_t kAlignment = 64; // tamanho da linha de cache
	
	[[noreturn]] void fail(const char* message) {
		std::fprint(stderr, "erro: %s\n", message);
		std::exit(2);
	}

	double initial_value(std::size_t i, std::size_t j, std::size_t n, InitKind kind) {
		if (kind == InitKind::Top) {
			return i == 0 ? 100.0 : 0.0;
		}

		switch (i) {
			case 0: return 100.0;
			case n - 1: return 25.0;
		}

		switch (j) {
			case 0: return 75.0;
			case n - 1: return 50.0;
		}

		return static_cast<double>(1 + (31*i + 17*j) % 100);
	}
} // namespace

GridPtr alloc_grid(std::size_t n) {
	if (n > (SIZE_MAX / sizeof(double)) / n) {
		fail("matriz grande demais");
	}
	const std::size_t bytes = n*n*sizeof(double);
	const std::size_t rounded = (bytes + kAlignment - 1) / kAlignment*kAlignment;

	void* pointer = std::aligned_alloc(kAlignment, rounded);

	if (pointer == nullptr) {
		fail("sem meoria para a matriz");
	}

	return GridPtr(static_cast<double*>(pointer));
}

void init_rows(double* a, std::size_t n, InitKind kind, std::size_t first_row, std::size_t last_row) {
	for (std::size_t i = first_row; i < last_row; i++) {
		for (std::size_t j = 0; j < n; j++) {
			a[i*n + j] = initial_value(i, j, n, kind);
		}
	}
}
