#pragma once

//#include "twiddle_table.h"

#include <future>
#include <nmmintrin.h>
#include <vector>

//#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#include <iostream>
#endif

// Reports the successful match of buf query vector,
// at index 'q' in the original list of query vectors,
// with the training vector at index 't' in the original
// list of training vectors.
struct Match {
	int q;
	int t;

	Match() {}
	Match(const int _q, const int _t) : q(_q), t(_t) {}
};

// Holds an in-progress match that requires at least
// one more twiddle (or buf brute-force) to resolve.
struct Partial {
	int q;
	int best_i;
	int16_t best_v;
	int16_t second_v;

	Partial() {}
	Partial(const int _q, const int _best_i, const int16_t _best_v, const int16_t _second_v) : q(_q), best_i(_best_i), best_v(_best_v), second_v(_second_v) {}
};

struct uninitialized_int {
	int x;
	uninitialized_int() {}
};

// Supports 8- and 16-bit hash tables, templated
// for performance.
template <const bool eightbit>
class Matcher {
public:
	// Vector of successful matches.
	std::vector<Match> matches;

	// Vector of indices of best-matching training vector for each query vector.
	std::vector<uninitialized_int> match_idxs;

	// Vector of in-progress matches requiring more twiddles
	// (or brute force) to resolve.
	std::vector<Partial> remainder;

private:
	// Multi-index hash table (MIHT)
	std::vector<int>* const __restrict raw_table;

	// Contiguous MIHT
	std::vector<int> compact_table;

	// Array of end indices in compact_table for each bin
	int* const __restrict ends;

	// Incoming 512-bit training vectors
	const void* __restrict tset;

	// Incoming 512-bit query vectors
	const void* __restrict qset;

	// Number of training vectors
	int tcount;

	// Number of query vectors
	int qcount;

	// Threshold by which the best match
	// must exceed the second-best match
	// to be considered buf match
	int threshold;

	// Max twiddle passes before the system
	// switches to the brute-force solver
	// on the remaining query vectors
	int max_twiddles;

	int hw_concur;

	std::future<void>* const __restrict fut;

	std::vector<Partial>* const __restrict rems;

public:
	Matcher() :
		raw_table(new std::vector<int>[eightbit ? 256 * 64 : 65536 * 32]),
		ends(new int[(eightbit ? 256 * 64 : 65536 * 32) + 1]),
		hw_concur(static_cast<int>(std::thread::hardware_concurrency())),
		fut(new std::future<void>[hw_concur]),
		rems(new std::vector<Partial>[hw_concur]) {}

	~Matcher() {
		delete[] raw_table;
		delete[] ends;
		delete[] fut;
	}

	// Step through the fixed-size 'match_idxs' and add
	// all valid matches to the 'matches' vector
	void addMatches() {
		for (int q = 0; q < qcount; ++q) {
			if (match_idxs[q].x != -1) {
				matches.emplace_back(q, match_idxs[q].x);
			}
		}
	}

	// Simple but optimized brute force (n^2) matcher.
	void bruteMatch() {
		match_idxs.resize(qcount);
		const int stride = (qcount - 1) / hw_concur + 1;
		int i = 0;
		int start = 0;
		for (; i < std::min(qcount - 1, hw_concur - 1); ++i, start += stride) {
			fut[i] = std::async(std::launch::async, &Matcher::_bruteMatch, this, start, stride);
		}
		fut[i] = std::async(std::launch::async, &Matcher::_bruteMatch, this, start, qcount - start);
		for (int j = 0; j <= i; ++j) fut[j].wait();
		matches.clear();
		addMatches();
	}

	void update(const void* const __restrict _tset, const int _tcount, const void* const __restrict _qset, const int _qcount, const int _threshold, const int _max_twiddles) {
		tset = _tset;
		tcount = _tcount;
		qset = _qset;
		qcount = _qcount;
		threshold = _threshold;
		max_twiddles = _max_twiddles;
	};

	private:
		void _bruteMatch(const int start, const int count) {
			const uint64_t* const __restrict q64 = reinterpret_cast<const uint64_t* const __restrict>(qset);
			const uint64_t* const __restrict t64 = reinterpret_cast<const uint64_t* const __restrict>(tset);

			for (int q = start; q < start + count; ++q) {
				const uint64_t qp = q << 3;
				int best_i = -1;
				int16_t best_v = 10000;
				int16_t second_v = 20000;

				const register uint64_t qa = q64[qp];
				const register uint64_t qb = q64[qp + 1];
				const register uint64_t qc = q64[qp + 2];
				const register uint64_t qd = q64[qp + 3];
				const register uint64_t qe = q64[qp + 4];
				const register uint64_t qf = q64[qp + 5];
				const register uint64_t qg = q64[qp + 6];
				const register uint64_t qh = q64[qp + 7];

				for (int t = 0, tp = 0; t < tcount; ++t, tp += 8) {
					const int16_t score = static_cast<int16_t>(
						_mm_popcnt_u64(qa ^ t64[tp])
						+ _mm_popcnt_u64(qb ^ t64[tp + 1])
						+ _mm_popcnt_u64(qc ^ t64[tp + 2])
						+ _mm_popcnt_u64(qd ^ t64[tp + 3])
						+ _mm_popcnt_u64(qe ^ t64[tp + 4])
						+ _mm_popcnt_u64(qf ^ t64[tp + 5])
						+ _mm_popcnt_u64(qg ^ t64[tp + 6])
						+ _mm_popcnt_u64(qh ^ t64[tp + 7]));
					if (score < second_v) second_v = score;
					if (score < best_v) {
						second_v = best_v;
						best_v = score;
						best_i = t;
					}
				}

				if (second_v - best_v <= threshold) best_i = -1;
				match_idxs[q].x = best_i;
			}
		}
};
