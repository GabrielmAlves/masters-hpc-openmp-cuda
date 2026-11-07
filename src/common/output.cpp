#include "output.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Cabeçalho do arquivo binário da matriz final.
struct GridFileHeader {
	char magic[8]; // "JACOBI2D", sem '\0'
	std::uint64_t n;
	std::uint64_t iters;
};
static_assert(sizeof(GridFileHeader) == 24, "o cabeçalho não pode ter preenchimento");

[[noreturn]] void fail(const std::string& message) {
	std::fprintf(stderr, "erro: %s\n", message.c_str());
	std::exit(2);
}

} // namespace

void print_csv_header() {
	std::puts("version,N,iters,threads,rep,t_loop,t_transfer,mlups");
}

void print_csv_row(const RunRecord& r) {
	const double inner = static_cast<double>(r.n - 2);
	const double mlups = r.t_loop > 0.0 ? inner * inner * r.iters / (r.t_loop * 1e6) : 0.0;

	std::printf("%s,%zu,%d,%d,%d,%.6f,%.6f,%.1f\n", r.version, r.n, r.iters, r.threads, r.rep,
				r.t_loop, r.t_transfer, mlups);

	// Não perde linhas já medidas se o job for interrompido.
	std::fflush(stdout);
}

void write_grid(const std::string& path, const double* a, std::size_t n, int iters) {
	GridFileHeader h{};
	std::memcpy(h.magic, "JACOBI2D", sizeof h.magic);
	h.n = n;
	h.iters = static_cast<std::uint64_t>(iters);

	std::FILE* f = std::fopen(path.c_str(), "wb");
	if (f == nullptr) {
		fail("não foi possível criar " + path);
	}

	const bool written = std::fwrite(&h, sizeof h, 1, f) == 1 &&
						 std::fwrite(a, sizeof(double), n * n, f) == n * n;

	// O fclose também pode falhar: parte dos dados só vai para o disco nele.
	const bool closed = std::fclose(f) == 0;

	if (!written || !closed) {
		fail("falha ao gravar " + path);
	}
}
