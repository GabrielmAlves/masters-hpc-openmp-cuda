#include "solver.cuh"

#include <algorithm>
#include <utility>

namespace {
	// Uma thread por célula: x -> coluna (j), y -> linha (i). O +1 pula a borda.
	// Idx = std::size_t na versão inicial, int na variante Int32.
	template <typename T, typename Idx>
	__global__ void sweep_global(const T* __restrict__ in, T* __restrict__ out, Idx n) {
		const Idx j = static_cast<Idx>(blockIdx.x)*blockDim.x + threadIdx.x + 1;
		const Idx i = static_cast<Idx>(blockIdx.y)*blockDim.y + threadIdx.y + 1;

		if (i < n - 1 && j < n - 1) {
			out[i*n + j] = T(0.25)*(in[(i - 1)*n + j] + in[(i + 1)*n + j] + in[i*n + j - 1] + in[i*n + j + 1]);
		}
	}

	// Contraexemplo: x -> linha. Threads vizinhas de um warp acessam linhas
	// diferentes, a n posições de distância, e os acessos não coalescem.
	template <typename T>
	__global__ void sweep_uncoalesced(const T* __restrict__ in, T* __restrict__ out, int n) {
		const int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
		const int j = blockIdx.y*blockDim.y + threadIdx.y + 1;

		if (i < n - 1 && j < n - 1) {
			out[i*n + j] = T(0.25)*(in[(i - 1)*n + j] + in[(i + 1)*n + j] + in[i*n + j - 1] + in[i*n + j + 1]);
		}
	}

	// O bloco carrega a sua região mais um halo de uma célula em shared memory,
	// e cada célula da matriz é lida uma vez da memória global por bloco.
	template <typename T>
	__global__ void sweep_shared(const T* __restrict__ in, T* __restrict__ out, int n) {
		extern __shared__ __align__(16) unsigned char smem_raw[];
		T* s = reinterpret_cast<T*>(smem_raw);

		const int bw = blockDim.x;
		const int bh = blockDim.y;
		const int sw = bw + 2; // largura da região em shared, com o halo
		const int tx = threadIdx.x;
		const int ty = threadIdx.y;
		const int j = blockIdx.x*bw + tx + 1;
		const int i = blockIdx.y*bh + ty + 1;

		// Carrega as células que existem na matriz (inclusive a borda).
		// Halo de cima e da esquerda: sempre existe (i-1 >= 0, j-1 >= 0).
		// Halo de baixo e da direita: só a última linha/coluna do bloco carrega;
		// se o bloco passa do fim da matriz, a borda já entrou como célula central.
		if (i < n && j < n) {
			s[(ty + 1)*sw + tx + 1] = in[i*n + j];
			if (ty == 0) s[tx + 1] = in[(i - 1)*n + j];
			if (ty == bh - 1 && i + 1 < n) s[(ty + 2)*sw + tx + 1] = in[(i + 1)*n + j];
			if (tx == 0) s[(ty + 1)*sw] = in[i*n + j - 1];
			if (tx == bw - 1 && j + 1 < n) s[(ty + 1)*sw + tx + 2] = in[i*n + j + 1];
		}
		__syncthreads();

		if (i < n - 1 && j < n - 1) {
			out[i*n + j] = T(0.25)*(s[ty*sw + tx + 1] + s[(ty + 2)*sw + tx + 1] + s[(ty + 1)*sw + tx] +
									s[(ty + 1)*sw + tx + 2]);
		}
	}

	// Cada thread calcula `coarsen` linhas seguidas da mesma coluna e reaproveita
	// em registradores as linhas de cima e do meio (3 leituras por célula em vez de 4).
	template <typename T>
	__global__ void sweep_coarse(const T* __restrict__ in, T* __restrict__ out, int n, int coarsen) {
		const int j = blockIdx.x*blockDim.x + threadIdx.x + 1;
		const int i0 = (blockIdx.y*blockDim.y + threadIdx.y)*coarsen + 1;
		if (j >= n - 1 || i0 >= n - 1) {
			return;
		}
		const int i_end = min(i0 + coarsen, n - 1);

		T up = in[(i0 - 1)*n + j];
		T middle = in[i0*n + j];
		for (int i = i0; i < i_end; i++) {
			const T down = in[(i + 1)*n + j];
			out[i*n + j] = T(0.25)*(up + down + in[i*n + j - 1] + in[i*n + j + 1]);
			up = middle;
			middle = down;
		}
	}

	// Blocagem temporal: o bloco de bw x bh threads carrega uma região de
	// bw x bh células (com halo de s células de cada lado) em shared memory,
	// avança s iterações sem voltar à memória global e grava só o miolo de
	// (bw - 2s) x (bh - 2s) células. A cada passo, a região válida encolhe uma
	// célula de cada lado, exceto na borda global, que é constante.
	template <typename T>
	__global__ void sweep_temporal(const T* __restrict__ in, T* __restrict__ out, int n, int s) {
		extern __shared__ __align__(16) unsigned char smem_raw[];
		const int bw = blockDim.x;
		const int bh = blockDim.y;
		T* a = reinterpret_cast<T*>(smem_raw);
		T* b = a + bw*bh;

		const int tx = threadIdx.x;
		const int ty = threadIdx.y;
		const int j = blockIdx.x*(bw - 2*s) + tx + 1 - s;
		const int i = blockIdx.y*(bh - 2*s) + ty + 1 - s;
		const int idx = ty*bw + tx;
		const bool in_matrix = i >= 0 && i < n && j >= 0 && j < n;
		const bool interior = i >= 1 && i < n - 1 && j >= 1 && j < n - 1;

		const T value = in_matrix ? in[i*n + j] : T(0);
		a[idx] = value;
		b[idx] = value; // a borda global precisa existir nos dois buffers
		__syncthreads();

		for (int t = 1; t <= s; t++) {
			if (interior && tx >= t && tx < bw - t && ty >= t && ty < bh - t) {
				b[idx] = T(0.25)*(a[idx - bw] + a[idx + bw] + a[idx - 1] + a[idx + 1]);
			}
			__syncthreads();
			T* tmp = a;
			a = b;
			b = tmp;
		}

		if (interior && tx >= s && tx < bw - s && ty >= s && ty < bh - s) {
			out[i*n + j] = a[idx];
		}
	}

	unsigned blocks_for(std::size_t length, std::size_t per_block) {
		return static_cast<unsigned>((length + per_block - 1)/per_block);
	}
} // namespace

int steps_per_launch(const LaunchConfig& c) {
	return c.kernel == Kernel::Temporal ? c.tsteps : 1;
}

template <typename T>
void launch_sweep(const T* in, T* out, std::size_t n, const LaunchConfig& c, int steps) {
	const std::size_t interior = n - 2;
	const dim3 b = c.block;
	const dim3 grid(blocks_for(interior, b.x), blocks_for(interior, b.y));
	const int ni = static_cast<int>(n);

	switch (c.kernel) {
	case Kernel::Basic:
		sweep_global<T, std::size_t><<<grid, b, 0, c.stream>>>(in, out, n);
		break;
	case Kernel::Int32:
		sweep_global<T, int><<<grid, b, 0, c.stream>>>(in, out, ni);
		break;
	case Kernel::Uncoalesced:
		sweep_uncoalesced<T><<<grid, b, 0, c.stream>>>(in, out, ni);
		break;
	case Kernel::Shared: {
		const std::size_t smem = (b.x + 2)*(b.y + 2)*sizeof(T);
		sweep_shared<T><<<grid, b, smem, c.stream>>>(in, out, ni);
		break;
	}
	case Kernel::Coarse: {
		const dim3 coarse_grid(blocks_for(interior, b.x),
							   blocks_for(interior, static_cast<std::size_t>(b.y)*c.coarsen));
		sweep_coarse<T><<<coarse_grid, b, 0, c.stream>>>(in, out, ni, c.coarsen);
		break;
	}
	case Kernel::Temporal: {
		// Cada bloco grava (bx - 2s) x (by - 2s) células.
		const unsigned s = static_cast<unsigned>(steps);
		const dim3 temporal_grid(blocks_for(interior, b.x - 2*s), blocks_for(interior, b.y - 2*s));
		const std::size_t smem = 2*static_cast<std::size_t>(b.x)*b.y*sizeof(T);
		sweep_temporal<T><<<temporal_grid, b, smem, c.stream>>>(in, out, ni, steps);
		break;
	}
	}
}

template <typename T>
void enqueue(T*& d_current, T*& d_next, std::size_t n, int iters, const LaunchConfig& c) {
	// Kernels no mesmo stream executam em ordem: não precisa sincronizar entre
	// as iterações. O lançamento é assíncrono; o laço só enfileira os kernels.
	const int per_launch = steps_per_launch(c);
	for (int done = 0; done < iters;) {
		const int steps = std::min(per_launch, iters - done);
		launch_sweep<T>(d_current, d_next, n, c, steps);
		std::swap(d_current, d_next);
		done += steps;
	}
}

template <typename T>
const T* solve_device(T* d_current, T* d_next, std::size_t n, int iters, const LaunchConfig& c) {
	enqueue<T>(d_current, d_next, n, iters, c);
	CUDA_CHECK(cudaGetLastError());               // erro de lançamento
	CUDA_CHECK(cudaStreamSynchronize(c.stream)); // espera o último kernel

	return d_current;
}

template <typename T>
SweepGraph build_graph(T* d_current, T* d_next, std::size_t n, int launches, const LaunchConfig& c) {
	SweepGraph g;
	g.chunk_iters = launches*steps_per_launch(c);

	cudaGraph_t graph = nullptr;
	CUDA_CHECK(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
	enqueue<T>(d_current, d_next, n, g.chunk_iters, c);
	CUDA_CHECK(cudaStreamEndCapture(c.stream, &graph));
	CUDA_CHECK(cudaGraphInstantiate(&g.exec, graph, 0));
	CUDA_CHECK(cudaGraphDestroy(graph));

	return g;
}

template <typename T>
const T* solve_graph(T* d_current, T* d_next, std::size_t n, int iters, const LaunchConfig& c,
					 const SweepGraph& g) {
	// O grafo foi gravado com (d_current, d_next) nessa ordem e com um número par
	// de lançamentos: depois de cada lançamento do grafo, os papéis voltam ao início.
	const int full = iters/g.chunk_iters;
	const int rest = iters%g.chunk_iters;

	for (int k = 0; k < full; k++) {
		CUDA_CHECK(cudaGraphLaunch(g.exec, c.stream));
	}
	enqueue<T>(d_current, d_next, n, rest, c);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(c.stream));

	return d_current;
}

void destroy_graph(SweepGraph& g) {
	if (g.exec != nullptr) {
		CUDA_CHECK(cudaGraphExecDestroy(g.exec));
		g.exec = nullptr;
	}
}

template void launch_sweep<double>(const double*, double*, std::size_t, const LaunchConfig&, int);
template void launch_sweep<float>(const float*, float*, std::size_t, const LaunchConfig&, int);
template void enqueue<double>(double*&, double*&, std::size_t, int, const LaunchConfig&);
template void enqueue<float>(float*&, float*&, std::size_t, int, const LaunchConfig&);
template const double* solve_device<double>(double*, double*, std::size_t, int, const LaunchConfig&);
template const float* solve_device<float>(float*, float*, std::size_t, int, const LaunchConfig&);
template SweepGraph build_graph<double>(double*, double*, std::size_t, int, const LaunchConfig&);
template SweepGraph build_graph<float>(float*, float*, std::size_t, int, const LaunchConfig&);
template const double* solve_graph<double>(double*, double*, std::size_t, int, const LaunchConfig&,
										   const SweepGraph&);
template const float* solve_graph<float>(float*, float*, std::size_t, int, const LaunchConfig&,
										 const SweepGraph&);
