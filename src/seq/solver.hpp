#pragma once

#include <cstddef>

// Executa `iters` iterações de Jacobi alternando os dois buffers.
// Os dois precisam chegar com as bordas inicializadas, pois trocam de papel.
// Devolve o buffer que contém o resultado final (cur ou next).
const double* solve(double* cur, double* next, std::size_t n, int iters);
