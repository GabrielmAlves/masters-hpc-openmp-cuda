#pragma once

#include <cstddef>
#include <string>

enum class InitKind { Pattern, Top };

struct Options {
	std::size_t n = 1024;              // dimensão da matriz, com as bordas
	int iters = 1000;                  // número fixo de iterações
	int reps = 1;                      // repetições medidas
	InitKind init = InitKind::Pattern; // matriz inicial
	std::string dump;                  // arquivo da matriz final (vazio: não grava)
	bool header = false;               // imprime o cabeçalho do CSV

	// Variante do algoritmo. Vazio: a variante padrão de cada versão.
	std::string variant;

	// OpenMP, variante temporal: lado do bloco e passos de tempo por bloco.
	int tile = 64;
	int tsteps = 4;

	// CUDA: dimensões do bloco, linhas por thread (variante coarse),
	// precisão, memória pinned no host e CUDA Graphs.
	int bx = 16;
	int by = 16;
	int coarsen = 4;
	bool use_float = false;
	bool pinned = false;
	bool graph = false;
};

// Lê os argumentos. Em caso de erro, escreve no stderr e encerra com código 2.
Options parse_args(int argc, char** argv);
