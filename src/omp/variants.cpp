#include "variants.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace {
	// Atualiza as colunas [j0, j1) de uma linha. Mesma ordem das somas da versão
	// sequencial, para o resultado ser idêntico bit a bit.
	inline void update_row(const double* __restrict__ up, const double* __restrict__ middle,
						   const double* __restrict__ down, double* __restrict__ dst,
						   std::size_t j0, std::size_t j1) {
		for (std::size_t j = j0; j < j1; j++) {
			dst[j] = 0.25*(up[j] + down[j] + middle[j - 1] + middle[j + 1]);
		}
	}

	// max(value - amount, low), sem estourar o unsigned.
	inline std::size_t sub_clamp(std::size_t value, std::size_t amount, std::size_t low) {
		return value > low + amount ? value - amount : low;
	}

	// Avança s iterações no bloco [i0, i1) x [j0, j1) e grava o resultado em out.
	// la e lb são buffers locais com pelo menos (tile + 2s)^2 posições.
	void advance_tile(const double* __restrict__ in, double* __restrict__ out, std::size_t n,
					  std::size_t i0, std::size_t j0, std::size_t tile, std::size_t s,
					  double* la, double* lb) {
		const std::size_t i1 = std::min(i0 + tile, n - 1);
		const std::size_t j1 = std::min(j0 + tile, n - 1);

		// Região local: o bloco mais um halo de s células, limitado à matriz.
		const std::size_t li0 = sub_clamp(i0, s, 0);
		const std::size_t lj0 = sub_clamp(j0, s, 0);
		const std::size_t li1 = std::min(n, i1 + s);
		const std::size_t lj1 = std::min(n, j1 + s);
		const std::size_t h = li1 - li0;
		const std::size_t w = lj1 - lj0;

		for (std::size_t r = 0; r < h; r++) {
			std::memcpy(la + r*w, in + (li0 + r)*n + lj0, w*sizeof(double));
		}
		// As bordas globais nunca são recalculadas: precisam existir nos dois buffers.
		std::memcpy(lb, la, h*w*sizeof(double));

		double* src = la;
		double* dst = lb;
		for (std::size_t t = 1; t <= s; t++) {
			// Depois deste passo ainda falta avançar m vezes: a região válida
			// precisa de m células de halo. Ela encolhe uma célula por passo,
			// exceto onde encosta na borda global, que é constante.
			const std::size_t m = s - t;
			const std::size_t ri0 = sub_clamp(i0, m, 1);
			const std::size_t ri1 = std::min(n - 1, i1 + m);
			const std::size_t c0 = sub_clamp(j0, m, 1) - lj0;
			const std::size_t c1 = std::min(n - 1, j1 + m) - lj0;

			for (std::size_t i = ri0; i < ri1; i++) {
				const std::size_t r = i - li0;
				update_row(src + (r - 1)*w, src + r*w, src + (r + 1)*w, dst + r*w, c0, c1);
			}
			std::swap(src, dst);
		}

		for (std::size_t i = i0; i < i1; i++) {
			std::memcpy(out + i*n + j0, src + (i - li0)*w + (j0 - lj0), (j1 - j0)*sizeof(double));
		}
	}
} // namespace

const double* solve_single(double* current, double* next, std::size_t n, int iters) {
	#pragma omp parallel firstprivate(current, next)
	{
		for (int k = 0; k < iters; k++) {
			// A barreira implícita no fim do for separa as iterações.
			#pragma omp for schedule(static)
				for (std::size_t i = 1; i < n - 1; i++) {
					update_row(current + (i - 1)*n, current + i*n, current + (i + 1)*n, next + i*n,
							   1, n - 1);
				}
			// Cada thread troca a sua cópia dos ponteiros; todas fazem a mesma troca.
			std::swap(current, next);
		}
	}

	// As cópias privadas somem no fim da região: o resultado depende da paridade.
	return iters % 2 == 0 ? current : next;
}

const double* solve_temporal(double* current, double* next, std::size_t n, int iters, int tile,
							 int tsteps) {
	const std::size_t b = static_cast<std::size_t>(tile);
	const std::size_t t_max = static_cast<std::size_t>(tsteps);
	const std::size_t tiles = (n - 2 + b - 1)/b;
	const std::size_t local = (b + 2*t_max)*(b + 2*t_max);

	#pragma omp parallel
	{
		std::vector<double> la(local);
		std::vector<double> lb(local);
		double* cur = current;
		double* nxt = next;

		for (int done = 0; done < iters;) {
			const std::size_t s = std::min<std::size_t>(t_max, static_cast<std::size_t>(iters - done));

			#pragma omp for collapse(2) schedule(static)
				for (std::size_t ti = 0; ti < tiles; ti++) {
					for (std::size_t tj = 0; tj < tiles; tj++) {
						advance_tile(cur, nxt, n, 1 + ti*b, 1 + tj*b, b, s, la.data(), lb.data());
					}
				}
			std::swap(cur, nxt);
			done += static_cast<int>(s);
		}
	}

	const int blocks = (iters + tsteps - 1)/tsteps;
	return blocks % 2 == 0 ? current : next;
}
