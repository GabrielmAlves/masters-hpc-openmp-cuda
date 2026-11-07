#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>

#include "config.hpp"

struct FreeDeleter {
	void operator()(double* pointer) const { std::free(pointer); }
};

using GridPtr = std::unique_ptr<double[], FreeDeleter>;

GridPtr alloc_grid(std::size_t n);

void init_rows(double* a, std::size_t n, InitKind kind, std::size_t first_row, std::size_t last_row);
