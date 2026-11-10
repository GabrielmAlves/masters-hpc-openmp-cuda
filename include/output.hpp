#pragma once

#include <cstddef>
#include <string>

#include "grid.hpp"

// Dados de uma execução medida, uma linha do CSV.
struct RunRecord {
	const char* version; // "seq", "omp" ou "cuda"
	std::string variant; // variante do algoritmo, com os parâmetros relevantes
	const char* prec;    // "double" ou "float"
	std::size_t n;
	int iters;
	int threads; // threads da CPU; no CUDA, threads por bloco
	int bx;      // CUDA: dimensões do bloco (0 nas versões de CPU)
	int by;
	int rep;
	double t_loop;     // segundos no laço principal
	double t_transfer; // segundos em transferências (0 nas versões de CPU)
};

void print_csv_header();
void print_csv_row(const RunRecord& r);

// Grava a matriz final: cabeçalho seguido de n*n doubles.
// Em caso de erro, escreve no stderr e encerra com código 2.
void write_grid(const std::string& path, const double* a, std::size_t n, int iters);

GridPtr read_grid(const std::string& path, std::size_t& n, int& iters);
