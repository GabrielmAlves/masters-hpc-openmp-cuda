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
};

// Lê os argumentos. Em caso de erro, escreve no stderr e encerra com código 2.
Options parse_args(int argc, char** argv);
