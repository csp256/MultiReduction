#include "match.h"
#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std::chrono;

int main(int argc, char* argv[]) {
	int const device = (argc >= 2) ? atoi(argv[1]) : 0;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);
	printf("%s (%2d)\n",props.name, props.multiProcessorCount);
	cudaSetDevice(device);
	int size = 256*6*props.multiProcessorCount *5;

	constexpr int warmups = 4;
	constexpr int runs = 10;
	constexpr int brute_warmups = 0;
	constexpr int brute_runs = 1;
	constexpr int threshold = 10;
	constexpr int max_twiddles = 2;


	static_cast<void>(argc);
	static_cast<void>(argv);

	//std::ifstream inf("ref.txt", std::ios::binary);
	void* qvecs = malloc(64 * size);
	//inf.read(reinterpret_cast<char*>(qvecs), 64 * size);
	void* tvecs = malloc(64 * size);
	//inf.read(reinterpret_cast<char*>(tvecs), 64 * size);
	//inf.close();

	srand(36);
	for (int i = 0; i < 64*size; ++i) {
		reinterpret_cast<uint8_t*>(qvecs)[i] = rand();
		reinterpret_cast<uint8_t*>(tvecs)[i] = rand();
	}

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	void* d_qvecs;
	cudaMalloc(&d_qvecs, 64 * size);
	cudaMemcpy(d_qvecs, qvecs, 64 * size, cudaMemcpyHostToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_qvecs;
	resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.linear.desc.x = 32;
	resDesc.res.linear.desc.y = 32;
	resDesc.res.linear.sizeInBytes = 64 * size;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	cudaTextureObject_t tex_q = 0;
	cudaCreateTextureObject(&tex_q, &resDesc, &texDesc, nullptr);

	void* d_tvecs;
	cudaMalloc(&d_tvecs, 64 * size);
	cudaMemcpy(d_tvecs, tvecs, 64 * size, cudaMemcpyHostToDevice);

	int* d_matches;
	cudaMalloc(&d_matches, 4 * size);

	for (int i = 0; i < warmups; ++i) CUDABRUTE(d_tvecs, size, tex_q, size, d_matches, threshold);

	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CUDABRUTE(d_tvecs, size, tex_q, size, d_matches, threshold);
	high_resolution_clock::time_point end = high_resolution_clock::now();

	int* h_matches = reinterpret_cast<int*>(malloc(4 * size));
	cudaMemcpy(h_matches, d_matches, 4 * size, cudaMemcpyDeviceToHost);
	// cudaDeviceReset();

	std::vector<Match> matches;
	for (int i = 0; i < size; ++i) {
		if (h_matches[i] != -1) {
			matches.emplace_back(i, h_matches[i]);
		}
	}

	double sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	std::cout << "CUDA multireduction: found " << matches.size() << " matches in " << sec * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(size)*static_cast<double>(size) / sec * 1e-9 << " billion comparisons/second." << std::endl << std::endl;

	matches.clear();
	for (int i = 0; i < warmups; ++i) CUDABRUTE_naive(d_tvecs, size, tex_q, size, d_matches, threshold);

	high_resolution_clock::time_point start2 = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CUDABRUTE_naive(d_tvecs, size, tex_q, size, d_matches, threshold);
	high_resolution_clock::time_point end2 = high_resolution_clock::now();

	//h_matches = reinterpret_cast<int*>(malloc(4 * size));
	cudaMemcpy(h_matches, d_matches, 4 * size, cudaMemcpyDeviceToHost);
	cudaDeviceReset();

	for (int i = 0; i < size; ++i) {
		if (h_matches[i] != -1) {
			matches.emplace_back(i, h_matches[i]);
		}
	}

	double sec2 = static_cast<double>(duration_cast<nanoseconds>(end2 - start2).count()) * 1e-9 / static_cast<double>(runs);
	std::cout << "CUDA naive: found " << matches.size() << " matches in " << sec2 * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(size)*static_cast<double>(size) / sec2 * 1e-9 << " billion comparisons/second." << std::endl << std::endl;

	Matcher<false> m;
	m.update(tvecs, size, qvecs, size, threshold, max_twiddles);

	for (int i = 0; i < brute_warmups; ++i) m.bruteMatch();

	start = high_resolution_clock::now();
	for (int i = 0; i < brute_runs; ++i) m.bruteMatch();
	end = high_resolution_clock::now();

	sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(brute_runs);
	std::cout << "Brute: found " << m.matches.size() << " matches in " << sec * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(size)*static_cast<double>(size) / sec * 1e-9 << " billion comparisons/second." << std::endl << std::endl;

	if (matches.size() != m.matches.size()) {
		std::cout << "MISMATCH!" << std::endl;
		return EXIT_FAILURE;
	}

	struct SP {
		bool operator()(const Match& buf, const Match& b) {
			return buf.q < b.q;
		}
	};

	std::sort(m.matches.begin(), m.matches.end(), SP());

	for (size_t i = 0; i < matches.size(); ++i) {
		if (matches[i].q != m.matches[i].q || matches[i].t != m.matches[i].t) {
			std::cout << "MISMATCH on #" << i << "! Expected (" << m.matches[i].q << ", " << m.matches[i].t << "), got (" << matches[i].q << ", " << matches[i].t << ")" << std::endl;
			return EXIT_FAILURE;
		}
	}

}
