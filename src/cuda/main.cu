#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "config.hpp"
#include "grid.hpp"
#include "output.hpp"
#include "solver.cuh"
#include "timing.hpp"

namespace {
	[[noreturn]] void fail(const std::string& message) {
		std::fprintf(stderr, "erro: %s\n", message.c_str());
		std::exit(2);
	}

	// Memória do host: paginável (malloc) ou pinned (cudaMallocHost).
	template <typename T>
	class HostBuffer {
	public:
		HostBuffer(std::size_t count, bool pinned) : pinned_(pinned) {
			if (pinned_) {
				CUDA_CHECK(cudaMallocHost(&data_, count*sizeof(T)));
			} else {
				data_ = static_cast<T*>(std::malloc(count*sizeof(T)));
				if (data_ == nullptr) {
					fail("sem memória no host");
				}
			}
		}
		~HostBuffer() {
			if (pinned_) {
				cudaFreeHost(data_);
			} else {
				std::free(data_);
			}
		}
		HostBuffer(const HostBuffer&) = delete;
		HostBuffer& operator=(const HostBuffer&) = delete;

		T* get() const { return data_; }

	private:
		T* data_ = nullptr;
		bool pinned_;
	};

	template <typename T>
	void run(const Options& o, Kernel kernel, const std::string& kernel_label) {
		const std::size_t n = o.n;
		const std::size_t count = n*n;
		const std::size_t bytes = count*sizeof(T);
		const char* prec = sizeof(T) == sizeof(float) ? "float" : "double";

		LaunchConfig c;
		c.kernel = kernel;
		c.block = dim3(static_cast<unsigned>(o.bx), static_cast<unsigned>(o.by));
		c.coarsen = o.coarsen;
		c.tsteps = o.tsteps;
		// Stream bloqueante: sincroniza com os cudaMemcpy do stream padrão.
		CUDA_CHECK(cudaStreamCreate(&c.stream));

		// Host: matriz inicial (não muda) e matriz do resultado, separadas.
		// A inicialização é feita em double e convertida para T.
		HostBuffer<T> initial(count, o.pinned);
		HostBuffer<T> result(count, o.pinned);
		{
			GridPtr reference = alloc_grid(n);
			init_rows(reference.get(), n, o.init, 0, n);
			for (std::size_t k = 0; k < count; k++) {
				initial.get()[k] = static_cast<T>(reference[k]);
			}
		}

		// GPU: os dois buffers, com bordas.
		T* d_a = nullptr;
		T* d_b = nullptr;
		CUDA_CHECK(cudaMalloc(&d_a, bytes));
		CUDA_CHECK(cudaMalloc(&d_b, bytes));

		const auto upload = [&]() {
			CUDA_CHECK(cudaMemcpy(d_a, initial.get(), bytes, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_b, initial.get(), bytes, cudaMemcpyHostToDevice));
			// Com memória paginável, o cudaMemcpy pode voltar antes do fim do DMA.
			CUDA_CHECK(cudaDeviceSynchronize());
		};

		// Aquecimento: cria o contexto e carrega o kernel antes de medir.
		upload();
		solve_device<T>(d_a, d_b, n, 1, c);

		// CUDA Graphs: o grafo é montado uma vez, fora da medição.
		// O grafo grava um número par de lançamentos (no máximo 100).
		SweepGraph graph;
		const int full_launches = o.iters/steps_per_launch(c);
		const int launches = std::min(full_launches - full_launches%2, 100);
		const bool use_graph = o.graph && launches >= 2;
		if (use_graph) {
			graph = build_graph<T>(d_a, d_b, n, launches, c);
			CUDA_CHECK(cudaGraphLaunch(graph.exec, c.stream));
			CUDA_CHECK(cudaStreamSynchronize(c.stream));
		}

		std::string label = kernel_label;
		if (use_graph) label += "+graph";
		if (o.pinned) label += "+pinned";
		const int block_threads = o.bx*o.by;

		for (int rep = 0; rep < o.reps; rep++) {
			const double t0 = now_seconds();
			upload();

			const double t1 = now_seconds();
			const T* d_result = use_graph ? solve_graph<T>(d_a, d_b, n, o.iters, c, graph)
										  : solve_device<T>(d_a, d_b, n, o.iters, c);
			const double t2 = now_seconds();

			CUDA_CHECK(cudaMemcpy(result.get(), d_result, bytes, cudaMemcpyDeviceToHost));
			const double t3 = now_seconds();

			print_csv_row({"cuda", label, prec, n, o.iters, block_threads, o.bx, o.by, rep, t2 - t1,
						   (t1 - t0) + (t3 - t2)});
		}

		if (!o.dump.empty()) {
			GridPtr out = alloc_grid(n);
			for (std::size_t k = 0; k < count; k++) {
				out[k] = static_cast<double>(result.get()[k]);
			}
			write_grid(o.dump, out.get(), n, o.iters);
		}

		destroy_graph(graph);
		CUDA_CHECK(cudaFree(d_a));
		CUDA_CHECK(cudaFree(d_b));
		CUDA_CHECK(cudaStreamDestroy(c.stream));
	}
} // namespace

int main(int argc, char** argv) {
	const Options o = parse_args(argc, argv);

	Kernel kernel = Kernel::Basic;
	std::string label = "basic";
	if (o.variant.empty() || o.variant == "basic") {
		kernel = Kernel::Basic;
	} else if (o.variant == "int32") {
		kernel = Kernel::Int32;
		label = "int32";
	} else if (o.variant == "uncoalesced") {
		kernel = Kernel::Uncoalesced;
		label = "uncoalesced";
	} else if (o.variant == "shared") {
		kernel = Kernel::Shared;
		label = "shared";
	} else if (o.variant == "coarse") {
		kernel = Kernel::Coarse;
		label = "coarse_c" + std::to_string(o.coarsen);
	} else if (o.variant == "temporal") {
		kernel = Kernel::Temporal;
		label = "temporal_t" + std::to_string(o.tsteps);
	} else {
		fail("variante CUDA desconhecida: " + o.variant +
			 " (basic, int32, uncoalesced, shared, coarse, temporal)");
	}

	if (o.bx*o.by > 1024) {
		fail("o bloco tem mais de 1024 threads");
	}
	if (kernel != Kernel::Basic && o.n > 46340) {
		fail("com índices int, N precisa ser no máximo 46340 (N*N cabe em int)");
	}
	if (kernel == Kernel::Temporal && (o.bx <= 2*o.tsteps || o.by <= 2*o.tsteps)) {
		fail("na variante temporal, --bx e --by precisam ser maiores que 2*tsteps");
	}
	std::size_t rows_per_block = static_cast<std::size_t>(o.by);
	if (kernel == Kernel::Coarse) rows_per_block *= o.coarsen;
	if (kernel == Kernel::Temporal) rows_per_block -= 2*o.tsteps;
	if ((o.n - 2 + rows_per_block - 1)/rows_per_block > 65535) {
		fail("a grade passa de 65535 blocos em y: aumente --by");
	}

	if (o.header) {
		print_csv_header();
	}

	if (o.use_float) {
		run<float>(o, kernel, label);
	} else {
		run<double>(o, kernel, label);
	}

	return 0;
}
