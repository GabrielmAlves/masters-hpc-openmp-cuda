#include <cstdio>
#include <cuda_runtime.h>

__global__ void copy_kernel(const double* __restrict__ in, double* __restrict__ out, size_t n) {
	size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
	if (i < n) out[i] = in[i];
}

int main() {
	const size_t n = 4096ull * 4096ull;  // 128 MB, como N=4096
	const size_t bytes = n * sizeof(double);
	double *a, *b;
	cudaMalloc(&a, bytes);
	cudaMalloc(&b, bytes);
	cudaMemset(a, 0, bytes);
	cudaEvent_t s, e;
	cudaEventCreate(&s);
	cudaEventCreate(&e);
	const int reps = 50;

	for (int w = 0; w < 3; w++) cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice);
	cudaEventRecord(s);
	for (int r = 0; r < reps; r++) cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice);
	cudaEventRecord(e);
	cudaEventSynchronize(e);
	float ms;
	cudaEventElapsedTime(&ms, s, e);
	printf("cudaMemcpy D2D: %.1f GB/s (leitura + escrita)\n", 2.0 * bytes * reps / (ms / 1e3) / 1e9);

	copy_kernel<<<(n + 255) / 256, 256>>>(a, b, n);
	cudaEventRecord(s);
	for (int r = 0; r < reps; r++) copy_kernel<<<(n + 255) / 256, 256>>>(a, b, n);
	cudaEventRecord(e);
	cudaEventSynchronize(e);
	cudaEventElapsedTime(&ms, s, e);
	printf("kernel de cópia: %.1f GB/s (leitura + escrita)\n", 2.0 * bytes * reps / (ms / 1e3) / 1e9);

	int clk, bus;
	cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, 0);
	cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
	printf("pico teórico: %.1f GB/s (clock de memória %d kHz, barramento %d bits)\n",
		   2.0 * clk * 1e3 * bus / 8 / 1e9, clk, bus);
	int l2;
	cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
	printf("L2: %d KB\n", l2 / 1024);
	return 0;
}
