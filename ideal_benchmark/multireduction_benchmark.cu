#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
using namespace std::chrono;

#define _warpSize (32)
#define intsPerVector (32)
#define vectorsPerLoop (32)
#define warpsPerBlock (8)
#define loopsPerWarp (64)
#define vectorsPerWarp (vectorsPerLoop * loopsPerWarp)
#define vectorsPerBlock (vectorsPerWarp * warpsPerBlock)
#define blocksPerSM (8)

__launch_bounds__(_warpSize * warpsPerBlock, blocksPerSM)
__global__ void
add32i_naive(const int *g_V, int *g_S)
{
	int v[vectorsPerLoop];
	int readOffset = (blockIdx.x  * intsPerVector * vectorsPerBlock)
				   + (threadIdx.y * intsPerVector * vectorsPerWarp)
				   + (threadIdx.x);
	int writeOffset = (blockIdx.x  * vectorsPerBlock)
	                + (threadIdx.y * vectorsPerWarp)
					+ (threadIdx.x);
	for (int loop = 0; loop < loopsPerWarp; loop++, writeOffset += vectorsPerLoop) {
		#pragma unroll
		for (int i = 0; i < vectorsPerLoop; i++, readOffset += _warpSize)
			v[i] = g_V[readOffset];
		#pragma unroll
		for (int j = 1; j < _warpSize; j <<= 1)
			#pragma unroll
			for (int i = 0; i < vectorsPerLoop; i++)
				v[i] += __shfl_xor(v[i], j);
		if (threadIdx.x < vectorsPerLoop)
			g_S[writeOffset] = v[threadIdx.x];
	}
}

__launch_bounds__(_warpSize * warpsPerBlock, blocksPerSM)
__global__ void
add32i_multi(const int *g_V, int *g_S)
{
	int v[vectorsPerLoop];
	int readOffset = (blockIdx.x  * intsPerVector * vectorsPerBlock)
				   + (threadIdx.y * intsPerVector * vectorsPerWarp)
				   + (threadIdx.x);
	int writeOffset = (blockIdx.x  * vectorsPerBlock)
	                + (threadIdx.y * vectorsPerWarp)
					+ (threadIdx.x);
	for (int loop = 0; loop < loopsPerWarp; loop++, writeOffset += vectorsPerLoop) {
		#pragma unroll
		for (int i = 0; i < vectorsPerLoop; i++, readOffset += _warpSize) v[i] = g_V[readOffset];
		// This blob of code can be emitted with the printCode() function.
		// Attempting to write the below code with a series of loops causes the kernel
		//   to die in a fire on my machine (Ubuntu 15.10, GTX 970M, CUDA 7.5).
		// I am told the concise approach might be fixed, or even preferable, on CUDA 8.
		v[0] += __shfl_xor(v[0], 1);
		v[1] += __shfl_xor(v[1], 1);
		v[2] += __shfl_xor(v[2], 1);
		v[3] += __shfl_xor(v[3], 1);
		v[4] += __shfl_xor(v[4], 1);
		v[5] += __shfl_xor(v[5], 1);
		v[6] += __shfl_xor(v[6], 1);
		v[7] += __shfl_xor(v[7], 1);
		v[8] += __shfl_xor(v[8], 1);
		v[9] += __shfl_xor(v[9], 1);
		v[10] += __shfl_xor(v[10], 1);
		v[11] += __shfl_xor(v[11], 1);
		v[12] += __shfl_xor(v[12], 1);
		v[13] += __shfl_xor(v[13], 1);
		v[14] += __shfl_xor(v[14], 1);
		v[15] += __shfl_xor(v[15], 1);
		v[16] += __shfl_xor(v[16], 1);
		v[17] += __shfl_xor(v[17], 1);
		v[18] += __shfl_xor(v[18], 1);
		v[19] += __shfl_xor(v[19], 1);
		v[20] += __shfl_xor(v[20], 1);
		v[21] += __shfl_xor(v[21], 1);
		v[22] += __shfl_xor(v[22], 1);
		v[23] += __shfl_xor(v[23], 1);
		v[24] += __shfl_xor(v[24], 1);
		v[25] += __shfl_xor(v[25], 1);
		v[26] += __shfl_xor(v[26], 1);
		v[27] += __shfl_xor(v[27], 1);
		v[28] += __shfl_xor(v[28], 1);
		v[29] += __shfl_xor(v[29], 1);
		v[30] += __shfl_xor(v[30], 1);
		v[31] += __shfl_xor(v[31], 1);
		if (threadIdx.x & 1) {
			v[0] = v[1];
			v[2] = v[3];
			v[4] = v[5];
			v[6] = v[7];
			v[8] = v[9];
			v[10] = v[11];
			v[12] = v[13];
			v[14] = v[15];
			v[16] = v[17];
			v[18] = v[19];
			v[20] = v[21];
			v[22] = v[23];
			v[24] = v[25];
			v[26] = v[27];
			v[28] = v[29];
			v[30] = v[31];
		}
		v[0] += __shfl_xor(v[0], 2);
		v[2] += __shfl_xor(v[2], 2);
		v[4] += __shfl_xor(v[4], 2);
		v[6] += __shfl_xor(v[6], 2);
		v[8] += __shfl_xor(v[8], 2);
		v[10] += __shfl_xor(v[10], 2);
		v[12] += __shfl_xor(v[12], 2);
		v[14] += __shfl_xor(v[14], 2);
		v[16] += __shfl_xor(v[16], 2);
		v[18] += __shfl_xor(v[18], 2);
		v[20] += __shfl_xor(v[20], 2);
		v[22] += __shfl_xor(v[22], 2);
		v[24] += __shfl_xor(v[24], 2);
		v[26] += __shfl_xor(v[26], 2);
		v[28] += __shfl_xor(v[28], 2);
		v[30] += __shfl_xor(v[30], 2);
		if (threadIdx.x & 2) {
			v[0] = v[2];
			v[4] = v[6];
			v[8] = v[10];
			v[12] = v[14];
			v[16] = v[18];
			v[20] = v[22];
			v[24] = v[26];
			v[28] = v[30];
		}
		v[0] += __shfl_xor(v[0], 4);
		v[4] += __shfl_xor(v[4], 4);
		v[8] += __shfl_xor(v[8], 4);
		v[12] += __shfl_xor(v[12], 4);
		v[16] += __shfl_xor(v[16], 4);
		v[20] += __shfl_xor(v[20], 4);
		v[24] += __shfl_xor(v[24], 4);
		v[28] += __shfl_xor(v[28], 4);
		if (threadIdx.x & 4) {
			v[0] = v[4];
			v[8] = v[12];
			v[16] = v[20];
			v[24] = v[28];
		}
		v[0] += __shfl_xor(v[0], 8);
		v[8] += __shfl_xor(v[8], 8);
		v[16] += __shfl_xor(v[16], 8);
		v[24] += __shfl_xor(v[24], 8);
		if (threadIdx.x & 8) {
			v[0] = v[8];
			v[16] = v[24];
		}
		v[0] += __shfl_xor(v[0], 16);
		v[16] += __shfl_xor(v[16], 16);
		if (threadIdx.x & 16) {
			v[0] = v[16];
		}
		// End generated code.
		if (threadIdx.x < vectorsPerLoop) {
			g_S[writeOffset] = v[0];
		}
	}
}

void printCode(void) {
	for (int k=1; k<vectorsPerLoop; k<<=1) {
		for (int i=0; i<vectorsPerLoop; i+=k) {
			printf("v[%d] += __shfl_xor(v[%d], %d);\n", i,i,k);
		}
		printf("if (threadIdx.x & %d) { \n", k);
		for (int i=0; i<vectorsPerLoop; i+=(k<<1)) printf("    v[%d] = v[%d];\n", i, i+k);
		printf("}\n");
	}
	for (int k=vectorsPerLoop; k<_warpSize; k<<=1) {
		for (int i=0; i<vectorsPerLoop; i+=k) printf("v[%d] += __shfl_xor(v[%d], %d);\n",i,i,k);
	}
}

int main(int argc, char* argv[])
{
	int const device = (argc >= 2) ? atoi(argv[1]) : 0;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);
	printf("%s (%2d SMs)\n",props.name, props.multiProcessorCount);
	cudaSetDevice(device);
	cudaError_t err = cudaSuccess;

	constexpr int warmups = 16;
	constexpr int runs = 64;
	const int SMs = props.multiProcessorCount;
	const int N = vectorsPerLoop * loopsPerWarp * warpsPerBlock * blocksPerSM * SMs; // number of vectors
	const int k = intsPerVector;
	size_t size = N * k * sizeof(int);
	printf("Using %d vectors of length %d each.\n", N, k);
	printf("Total GPU memory usage: %d MB\n", (int)((double)(size + size/k) / (1024*1024) ) );

	int *h_V = (int *)malloc(size);
	int *h_S = (int *)malloc(size/k);
	for (int i = 0; i < N*k; ++i) {
		h_V[i] = rand()/(RAND_MAX>>16);
	}

	int *d_V = NULL;
	int *d_S = NULL;
	cudaMalloc((void **)&d_V, size);
	cudaMalloc((void **)&d_S, size/k);
	cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

	high_resolution_clock::time_point start, end;
	dim3 threadsPerBlock(_warpSize, warpsPerBlock);
	int blocksPerGrid =(N + vectorsPerBlock - 1) / vectorsPerBlock;

	for (int i=0; i< warmups; i++)
		add32i_multi<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
	cudaDeviceSynchronize();
	start = high_resolution_clock::now();
	for (int i=0; i<runs; i++)
		add32i_multi<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
	cudaDeviceSynchronize();
	end = high_resolution_clock::now();
	double t1 = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	printf("Multireduction (ms):         %.4f\n", t1*1000);

	for (int i=0; i< warmups; i++)
		add32i_naive<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
	cudaDeviceSynchronize();
	start = high_resolution_clock::now();
	for (int i=0; i<runs; i++)
		add32i_naive<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
	cudaDeviceSynchronize();
	end = high_resolution_clock::now();
	double t2 = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	printf("Previous best practice (ms): %.4f\n", t2*1000);
	printf("Speedup:                     %.4f\n", t2/t1);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(h_S, d_S, size/k, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy sums from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaFree(d_S);
	cudaFree(d_V);
	free(h_S);
	free(h_V);
	cudaDeviceReset();
	return 0;
}
