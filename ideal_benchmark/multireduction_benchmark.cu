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
#define scale (32 / vectorsPerLoop * 8 / blocksPerSM)

template<typename T>
__launch_bounds__(_warpSize * warpsPerBlock, blocksPerSM)
__global__ void
add32_naive(const T *g_V, T *g_S)
{
	T v[vectorsPerLoop];
	int readOffset = (blockIdx.x  * vectorsPerBlock)
				   + (threadIdx.y * vectorsPerWarp)
				   + (threadIdx.x);
	int writeOffset = (blockIdx.x  * vectorsPerBlock)
	                + (threadIdx.y * vectorsPerWarp)
					+ (threadIdx.x);
	for (int loop = 0; loop < loopsPerWarp; loop++, writeOffset += vectorsPerLoop) {
		#pragma unroll
		for (int i = 0; i < vectorsPerLoop; i++, readOffset += 1/*_warpSize*/) {
			// if (i == 1 && threadIdx.y == 0 && blockIdx.x == 0 && loop == 0) printf("^^ %d, %d\n", threadIdx.x, readOffset+i);
			v[i] = g_V[readOffset/**//**/];
		}
		#pragma unroll
		for (int j = 1; j < _warpSize; j <<= 1)
			#pragma unroll
			for (int i = 0; i < vectorsPerLoop; i++)
				v[i] += __shfl_xor(v[i], j);
		{
			if (threadIdx.x == 1) v[0] = v[1];
			if (threadIdx.x == 2) v[0] = v[2];
			if (threadIdx.x == 3) v[0] = v[3];
			if (threadIdx.x == 4) v[0] = v[4];
			if (threadIdx.x == 5) v[0] = v[5];
			if (threadIdx.x == 6) v[0] = v[6];
			if (threadIdx.x == 7) v[0] = v[7];
			if (threadIdx.x == 8) v[0] = v[8];
			if (threadIdx.x == 9) v[0] = v[9];
			if (threadIdx.x == 10) v[0] = v[10];
			if (threadIdx.x == 11) v[0] = v[11];
			if (threadIdx.x == 12) v[0] = v[12];
			if (threadIdx.x == 13) v[0] = v[13];
			if (threadIdx.x == 14) v[0] = v[14];
			if (threadIdx.x == 15) v[0] = v[15];

			#if (vectorsPerLoop == 32)
			if (threadIdx.x == 16) v[0] = v[16];
			if (threadIdx.x == 17) v[0] = v[17];
			if (threadIdx.x == 18) v[0] = v[18];
			if (threadIdx.x == 19) v[0] = v[19];
			if (threadIdx.x == 20) v[0] = v[20];
			if (threadIdx.x == 21) v[0] = v[21];
			if (threadIdx.x == 22) v[0] = v[22];
			if (threadIdx.x == 23) v[0] = v[23];
			if (threadIdx.x == 24) v[0] = v[24];
			if (threadIdx.x == 25) v[0] = v[25];
			if (threadIdx.x == 26) v[0] = v[26];
			if (threadIdx.x == 27) v[0] = v[27];
			if (threadIdx.x == 28) v[0] = v[28];
			if (threadIdx.x == 29) v[0] = v[29];
			if (threadIdx.x == 30) v[0] = v[30];
			if (threadIdx.x == 31) v[0] = v[31];
			#endif
		}

		// if (threadIdx.x <2 && blockIdx.x ==0 && threadIdx.y ==0) {
		if (threadIdx.x < 32)
			// if (loop < 100) printf("loop %d: %d %d\n", loop, writeOffset, v[0]);
			// if (threadIdx.x == 1) printf("loop %d: %d %d\n", loop, writeOffset, v[0]);
			g_S[writeOffset] = v[0];
		// }
	}
}

template<typename T>
__launch_bounds__(_warpSize * warpsPerBlock, blocksPerSM)
__global__ void
add32_multi(const T *g_V, T *g_S)
{
	// T v[vectorsPerLoop];
	int readOffset = (blockIdx.x  * vectorsPerBlock)
				   + (threadIdx.y * vectorsPerWarp)
				   + (threadIdx.x);
	int writeOffset = (blockIdx.x  * vectorsPerBlock)
	                + (threadIdx.y * vectorsPerWarp)
					+ (threadIdx.x * 2);
		#pragma unroll
		// for (int i = 0; i < vectorsPerLoop; i++, readOffset += _warpSize) v[i] = g_V[readOffset];
		for (int loop = 0; loop < loopsPerWarp; loop+=2, writeOffset += 2*vectorsPerLoop/**/, readOffset += 2*_warpSize/*(_warpSize*vectorsPerLoop) & ((1<<11)-1)*/) {
		// This blob of code can be emitted with the printMultiCode() function.
		// Attempting to write the below code with a series of loops causes the kernel
		//   to die in a fire on my machine (Ubuntu 15.10, GTX 970M, CUDA 7.5).
		// I am told the concise approach might be fixed, or even preferable, on CUDA 8.
		#if vectorsPerLoop == -32
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
		#endif
		#if vectorsPerLoop == 16
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
			if (threadIdx.x & 1) {
			    v[0] = v[1];
			    v[2] = v[3];
			    v[4] = v[5];
			    v[6] = v[7];
			    v[8] = v[9];
			    v[10] = v[11];
			    v[12] = v[13];
			    v[14] = v[15];
			}
			v[0] += __shfl_xor(v[0], 2);
			v[2] += __shfl_xor(v[2], 2);
			v[4] += __shfl_xor(v[4], 2);
			v[6] += __shfl_xor(v[6], 2);
			v[8] += __shfl_xor(v[8], 2);
			v[10] += __shfl_xor(v[10], 2);
			v[12] += __shfl_xor(v[12], 2);
			v[14] += __shfl_xor(v[14], 2);
			if (threadIdx.x & 2) {
			    v[0] = v[2];
			    v[4] = v[6];
			    v[8] = v[10];
			    v[12] = v[14];
			}
			v[0] += __shfl_xor(v[0], 4);
			v[4] += __shfl_xor(v[4], 4);
			v[8] += __shfl_xor(v[8], 4);
			v[12] += __shfl_xor(v[12], 4);
			if (threadIdx.x & 4) {
			    v[0] = v[4];
			    v[8] = v[12];
			}
			v[0] += __shfl_xor(v[0], 8);
			v[8] += __shfl_xor(v[8], 8);
			if (threadIdx.x & 8) {
			    v[0] = v[8];
			}
			v[0] += __shfl_xor(v[0], 16);
		#endif
		// End generated code.

		/*
			// ITERATIVE MULTIREDUCTION
			T r[6];
			{
			// 0
			r[0] = g_V[readOffset + 0];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 1
			r[0] = g_V[readOffset + 32];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 2
			r[0] = g_V[readOffset + 64];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 3
			r[0] = g_V[readOffset + 96];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			r[3] = r[2];
			// 4
			r[0] = g_V[readOffset + 128];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 5
			r[0] = g_V[readOffset + 160];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 6
			r[0] = g_V[readOffset + 192];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 7
			r[0] = g_V[readOffset + 224];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			r[3] += __shfl_xor(r[3], 8);
			r[4] = r[3];
			// 8
			r[0] = g_V[readOffset + 256];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 9
			r[0] = g_V[readOffset + 288];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 10
			r[0] = g_V[readOffset + 320];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 11
			r[0] = g_V[readOffset + 352];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			r[3] = r[2];
			// 12
			r[0] = g_V[readOffset + 384];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 13
			r[0] = g_V[readOffset + 416];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 14
			r[0] = g_V[readOffset + 448];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 15
			r[0] = g_V[readOffset + 480];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			r[3] += __shfl_xor(r[3], 8);
			if (threadIdx.x & 8) r[4] = r[3];
			r[4] += __shfl_xor(r[4], 16);
			r[5] = r[4];
			// 16
			r[0] = g_V[readOffset + 512];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 17
			r[0] = g_V[readOffset + 544];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 18
			r[0] = g_V[readOffset + 576];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 19
			r[0] = g_V[readOffset + 608];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			r[3] = r[2];
			// 20
			r[0] = g_V[readOffset + 640];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 21
			r[0] = g_V[readOffset + 672];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 22
			r[0] = g_V[readOffset + 704];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 23
			r[0] = g_V[readOffset + 736];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			r[3] += __shfl_xor(r[3], 8);
			r[4] = r[3];
			// 24
			r[0] = g_V[readOffset + 768];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 25
			r[0] = g_V[readOffset + 800];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 26
			r[0] = g_V[readOffset + 832];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 27
			r[0] = g_V[readOffset + 864];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			r[3] = r[2];
			// 28
			r[0] = g_V[readOffset + 896];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 29
			r[0] = g_V[readOffset + 928];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			r[2] = r[1];
			// 30
			r[0] = g_V[readOffset + 960];
			r[0] += __shfl_xor(r[0], 1);
			r[1] = r[0];
			// 31
			r[0] = g_V[readOffset + 992];
			r[0] += __shfl_xor(r[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			r[1] += __shfl_xor(r[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			r[2] += __shfl_xor(r[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			r[3] += __shfl_xor(r[3], 8);
			if (threadIdx.x & 8) r[4] = r[3];
			r[4] += __shfl_xor(r[4], 16);
			if (threadIdx.x & 16) r[5] = r[4];
			}
		*/

		// Double iterative Multireduction
//		/*
		T r[6];
		T s[6];
		{
			/*
				I met a traveller from an antique land
				Who said: Two vast and trunkless legs of stone
				Stand in the desert. Near them, on the sand,
				Half sunk, a shattered visage lies, whose frown,
				And wrinkled lip, and sneer of cold command,
				Tell that its sculptor well those passions read
				Which yet survive, stamped on these lifeless things,
				The hand that mocked them and the heart that fed:
				And on the pedestal these words appear:
				'My name is Ozymandias, king of kings:
				Look on my works, ye Mighty, and despair!'
				Nothing beside remains. Round the decay
				Of that colossal wreck, boundless and bare
				The lone and level sands stretch far away.
						Percey Shelley's "Ozymandias"
			*/
			// 0
			r[0] = g_V[readOffset + 0];
			s[0] = g_V[readOffset + 1];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 1
			r[0] = g_V[readOffset + 2];
			s[0] = g_V[readOffset + 3];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 2
			r[0] = g_V[readOffset + 4];
			s[0] = g_V[readOffset + 5];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 3
			r[0] = g_V[readOffset + 6];
			s[0] = g_V[readOffset + 7];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			r[3] = r[2];
			s[3] = s[2];
			// 4
			r[0] = g_V[readOffset + 8];
			s[0] = g_V[readOffset + 9];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 5
			r[0] = g_V[readOffset + 10];
			s[0] = g_V[readOffset + 11];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 6
			r[0] = g_V[readOffset + 12];
			s[0] = g_V[readOffset + 13];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 7
			r[0] = g_V[readOffset + 14];
			s[0] = g_V[readOffset + 15];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			if (threadIdx.x & 4) s[3] = s[2];
			r[3] += __shfl_xor(r[3], 8);
			s[3] += __shfl_xor(s[3], 8);
			r[4] = r[3];
			s[4] = s[3];
			// 8
			r[0] = g_V[readOffset + 16];
			s[0] = g_V[readOffset + 17];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 9
			r[0] = g_V[readOffset + 18];
			s[0] = g_V[readOffset + 19];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 10
			r[0] = g_V[readOffset + 20];
			s[0] = g_V[readOffset + 21];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 11
			r[0] = g_V[readOffset + 22];
			s[0] = g_V[readOffset + 23];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			r[3] = r[2];
			s[3] = s[2];
			// 12
			r[0] = g_V[readOffset + 24];
			s[0] = g_V[readOffset + 25];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 13
			r[0] = g_V[readOffset + 26];
			s[0] = g_V[readOffset + 27];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 14
			r[0] = g_V[readOffset + 28];
			s[0] = g_V[readOffset + 29];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 15
			r[0] = g_V[readOffset + 30];
			s[0] = g_V[readOffset + 31];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			if (threadIdx.x & 4) s[3] = s[2];
			r[3] += __shfl_xor(r[3], 8);
			s[3] += __shfl_xor(s[3], 8);
			if (threadIdx.x & 8) r[4] = r[3];
			if (threadIdx.x & 8) s[4] = s[3];
			r[4] += __shfl_xor(r[4], 16);
			s[4] += __shfl_xor(s[4], 16);
			r[5] = r[4];
			s[5] = s[4];
			// 16
			r[0] = g_V[readOffset + 32];
			s[0] = g_V[readOffset + 33];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 17
			r[0] = g_V[readOffset + 34];
			s[0] = g_V[readOffset + 35];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 18
			r[0] = g_V[readOffset + 36];
			s[0] = g_V[readOffset + 37];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 19
			r[0] = g_V[readOffset + 38];
			s[0] = g_V[readOffset + 39];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			r[3] = r[2];
			s[3] = s[2];
			// 20
			r[0] = g_V[readOffset + 40];
			s[0] = g_V[readOffset + 41];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 21
			r[0] = g_V[readOffset + 42];
			s[0] = g_V[readOffset + 43];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 22
			r[0] = g_V[readOffset + 44];
			s[0] = g_V[readOffset + 45];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 23
			r[0] = g_V[readOffset + 46];
			s[0] = g_V[readOffset + 47];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			if (threadIdx.x & 4) s[3] = s[2];
			r[3] += __shfl_xor(r[3], 8);
			s[3] += __shfl_xor(s[3], 8);
			r[4] = r[3];
			s[4] = s[3];
			// 24
			r[0] = g_V[readOffset + 48];
			s[0] = g_V[readOffset + 49];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 25
			r[0] = g_V[readOffset + 50];
			s[0] = g_V[readOffset + 51];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 26
			r[0] = g_V[readOffset + 52];
			s[0] = g_V[readOffset + 53];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 27
			r[0] = g_V[readOffset + 54];
			s[0] = g_V[readOffset + 55];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			r[3] = r[2];
			s[3] = s[2];
			// 28
			r[0] = g_V[readOffset + 56];
			s[0] = g_V[readOffset + 57];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 29
			r[0] = g_V[readOffset + 58];
			s[0] = g_V[readOffset + 59];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			r[2] = r[1];
			s[2] = s[1];
			// 30
			r[0] = g_V[readOffset + 60];
			s[0] = g_V[readOffset + 61];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			r[1] = r[0];
			s[1] = s[0];
			// 31
			r[0] = g_V[readOffset + 62];
			s[0] = g_V[readOffset + 63];
			r[0] += __shfl_xor(r[0], 1);
			s[0] += __shfl_xor(s[0], 1);
			if (threadIdx.x & 1) r[1] = r[0];
			if (threadIdx.x & 1) s[1] = s[0];
			r[1] += __shfl_xor(r[1], 2);
			s[1] += __shfl_xor(s[1], 2);
			if (threadIdx.x & 2) r[2] = r[1];
			if (threadIdx.x & 2) s[2] = s[1];
			r[2] += __shfl_xor(r[2], 4);
			s[2] += __shfl_xor(s[2], 4);
			if (threadIdx.x & 4) r[3] = r[2];
			if (threadIdx.x & 4) s[3] = s[2];
			r[3] += __shfl_xor(r[3], 8);
			s[3] += __shfl_xor(s[3], 8);
			if (threadIdx.x & 8) r[4] = r[3];
			if (threadIdx.x & 8) s[4] = s[3];
			r[4] += __shfl_xor(r[4], 16);
			s[4] += __shfl_xor(s[4], 16);
			if (threadIdx.x & 16) r[5] = r[4];
			if (threadIdx.x & 16) s[5] = s[4];
		}
//		*/

		// if (threadIdx.x < 16) {
			g_S[writeOffset    ] = r[5];
			g_S[writeOffset + 1] = s[5];
			// g_S[writeOffset] = v[0];
		// }
	}
}

void printMultiCode(void) {
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

void printNaiveCode(void) {
	// for (int i = 0; i < vectorsPerLoop; i++)
	// 	printf("v[%d] = g_V[readOffset+%d];\n", i, i*_warpSize);
	for (int j = 1; j < _warpSize; j <<= 1)
		for (int i = 0; i < vectorsPerLoop; i++)
			printf("v[%d] += __shfl_xor(v[%d], %d);\n", i, i, j);

}

template <typename T>
void check(const T * v, const T * s, const int size, const bool debug) {
	int good = 0;
	int bad = 0;
	for (int i=0, j=0; i < size; j++) {
		T t = 0;
		for (int k = 0; k < _warpSize; i++, k++) {
			t += v[i];
		}
		if (s[j] != t) {
			bad++;
			if (debug) {
				if (j < 100) {
					if (j%32==0) printf("\n\n");
					printf("%d: %d %d \n", j, (int)s[j], (int)t);
				}
			}
		} else {
			if (debug && good < 100) printf("                         %d\n", j);
			good++;
		}
	}
	if (bad != 0) printf("Good %d\nBad  %d\n", good, bad);
}

template <typename T>
void check2(const T * v, const T * s, int size, bool debug) {
	size /= _warpSize;
	debug = false;
	int good = 0;
	int bad = 0;
	T t = 0;
	for (int i=0; i < _warpSize; i++) t += v[i];
	for (int i=0, j=_warpSize; i < size; i++, j++) {
		if (s[i] != t) {
			bad++;
			if (debug) {
				if (i < 100) {
					// if (i%32==0) printf("\n\n");
					printf("%d: %d %d \n", i, (int)s[i], (int)t);
				}
			}
		} else {
			if (debug && i < 100) printf("                         %d: %d\n", i, (int)t);
			good++;
		}
		t = t - v[i] + v[j];
	}
	if (bad != 0) printf("Good %d\nBad  %d\n", good, bad);
	// for (int i=0; i<100; i++) printf(".. %d, %d\n", i, v[i]);
}

template<typename T>
int test(int argc, char* argv[])
{
	int const device = (argc >= 2) ? atoi(argv[1]) : 0;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);
	cudaSetDevice(device);
	cudaError_t err = cudaSuccess;

	constexpr int warmups = 16;
	constexpr int runs = 64;
	const int SMs = props.multiProcessorCount;
	const int N = scale * vectorsPerLoop * loopsPerWarp * warpsPerBlock * blocksPerSM * SMs; // number of vectors
	const int k = intsPerVector;
	size_t size = N * k * sizeof(T); // This is MUCH LARGER than it needs to be & corresponds to an earlier version.
	printf("Total GPU memory usage: %d MB\n", (int)((double)(size + size/k) / (1024*1024) ) ); 	// But it hurts nothing.

	T *h_V = (T *)malloc(size);
	T *h_S = (T *)malloc(size/k);
	for (int i = 0; i < N*k; ++i) {
		h_V[i] = (T)(rand() & ((1 << 8)-1)); // Should have enough bits of precision for all types to get exact answers
	}
	for (int i = 0; i < N; i++) {
		h_S[i] = 0;
	}

	T *d_V = NULL;
	T *d_S = NULL;
	cudaMalloc((void **)&d_V, size);
	cudaMalloc((void **)&d_S, size/k);
	cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S, h_S, size/k, cudaMemcpyHostToDevice);

	high_resolution_clock::time_point start, end;
	dim3 threadsPerBlock(_warpSize, warpsPerBlock);
	int blocksPerGrid =(N + vectorsPerBlock - 1) / vectorsPerBlock;

	for (int i=0; i< warmups; i++)
		add32_multi<T><<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
	cudaDeviceSynchronize();
	start = high_resolution_clock::now();
	for (int i=0; i<runs; i++)
		add32_multi<T><<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
	cudaDeviceSynchronize();
	end = high_resolution_clock::now();
	double t1 = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	printf("Multireduction (ms):         %.4f\n", t1*1000);
	cudaMemcpy(h_S, d_S, size/k, cudaMemcpyDeviceToHost);
	check2<T>(h_V, h_S, N*k, true);

	// Call me paranoid and inefficient.
	for (int i = 0; i < N; i++) {
		h_S[i] = 0;
	}
	cudaMemcpy(d_S, h_S, size/k, cudaMemcpyHostToDevice);

	for (int i=0; i< warmups; i++)
		add32_naive<T><<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
	cudaDeviceSynchronize();
	start = high_resolution_clock::now();
	for (int i=0; i<runs; i++)
		add32_naive<T><<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S);
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
	check2<T>(h_V, h_S, N*k, true);

	cudaFree(d_S);
	cudaFree(d_V);
	free(h_S);
	free(h_V);
	cudaDeviceReset();
	return 0;
}

int main(int argc, char* argv[]) {
	int const device = (argc >= 2) ? atoi(argv[1]) : 0;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,device);
	printf("%s (%2d SMs)\n",props.name, props.multiProcessorCount);
	cudaSetDevice(device);

	const int SMs = props.multiProcessorCount;
	const int N = scale * vectorsPerLoop * loopsPerWarp * warpsPerBlock * blocksPerSM * SMs; // number of vectors
	const int k = intsPerVector;
	printf("Using %d vectors of length %d each.\n", N, k);
	printf("Each SM is assigned %d / 8 blocks.\n", blocksPerSM);
	printf("Vectors per loop: %d\n", vectorsPerLoop);

	int ret = 0;
	printf("\n-- Int\n");
	ret |= test<int>(argc, argv);
	printf("\n-- Float\n");
	ret |= test<float>(argc, argv);
	printf("\n-- Double\n");
	ret |= test<double>(argc, argv);
	return ret;
}
