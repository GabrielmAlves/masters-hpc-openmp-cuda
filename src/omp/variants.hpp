#pragma once

#include <cstddef>

// Variantes otimizadas da versão OpenMP. A básica é o solve() de solver.hpp.
// Todas devolvem o ponteiro do buffer com o resultado e dão o mesmo resultado,
// bit a bit, que a versão sequencial.

// Uma única região paralela para todas as iterações (sem abrir e fechar o
// grupo de threads a cada iteração).
const double* solve_single(double* current, double* next, std::size_t n, int iters);

// Blocagem temporal com sobreposição: cada bloco de tile x tile células avança
// tsteps iterações em buffers locais da thread (que cabem na cache) antes de
// voltar para a matriz global. Lê e grava a matriz global uma vez a cada
// tsteps iterações, ao custo de recalcular uma faixa de tsteps células em
// volta de cada bloco.
const double* solve_temporal(double* current, double* next, std::size_t n, int iters, int tile,
							 int tsteps);
