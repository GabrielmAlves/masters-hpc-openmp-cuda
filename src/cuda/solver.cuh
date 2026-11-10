#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Encerra o programa com uma mensagem se uma chamada CUDA falhar.
#define CUDA_CHECK(call)                                                        \
	do {                                                                        \
		const cudaError_t err_ = (call);                                        \
		if (err_ != cudaSuccess) {                                              \
			std::fprintf(stderr, "erro CUDA: %s (%s:%d)\n",                     \
						 cudaGetErrorString(err_), __FILE__, __LINE__);         \
			std::exit(2);                                                       \
		}                                                                       \
	} while (0)

// Kernels disponíveis. Todos usam a mesma expressão e a mesma ordem das somas.
enum class Kernel {
	Basic,       // uma thread por célula, índices size_t (versão inicial)
	Int32,       // igual à Basic, com índices int
	Uncoalesced, // x -> linha, y -> coluna: acessos não coalescidos (contraexemplo)
	Shared,      // bloco + halo em shared memory
	Coarse,      // cada thread calcula `coarsen` linhas seguidas da mesma coluna
	Temporal,    // blocagem temporal: `tsteps` iterações por lançamento em shared memory
};

struct LaunchConfig {
	Kernel kernel = Kernel::Basic;
	dim3 block{16, 16};
	int coarsen = 4;
	int tsteps = 4;
	cudaStream_t stream = nullptr;
};

// Iterações que um lançamento avança (tsteps no Temporal, 1 nos outros).
int steps_per_launch(const LaunchConfig& c);

// Lança um kernel (assíncrono) que avança `steps` iterações de in para out.
template <typename T>
void launch_sweep(const T* in, T* out, std::size_t n, const LaunchConfig& c, int steps);

// Enfileira `iters` iterações, trocando os ponteiros a cada lançamento. Não sincroniza.
template <typename T>
void enqueue(T*& d_current, T*& d_next, std::size_t n, int iters, const LaunchConfig& c);

// Executa `iters` iterações com lançamentos individuais e espera a GPU.
// d_current e d_next estão na GPU. Devolve o ponteiro (na GPU) com o resultado.
template <typename T>
const T* solve_device(T* d_current, T* d_next, std::size_t n, int iters, const LaunchConfig& c);

// CUDA Graphs: grava um número par de lançamentos num grafo, que é lançado
// várias vezes. Reduz o custo de lançar milhares de kernels pequenos.
struct SweepGraph {
	cudaGraphExec_t exec = nullptr;
	int chunk_iters = 0; // iterações avançadas por lançamento do grafo
};

template <typename T>
SweepGraph build_graph(T* d_current, T* d_next, std::size_t n, int launches, const LaunchConfig& c);

template <typename T>
const T* solve_graph(T* d_current, T* d_next, std::size_t n, int iters, const LaunchConfig& c,
					 const SweepGraph& g);

void destroy_graph(SweepGraph& g);
