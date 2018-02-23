# MultiReduction

A design pattern for accelerating many independent simultaneous reduction operations on many-core architectures. 

On NVIDIA graphics cards you can usually expect a 1.5x to 2.5x speedup compared to NVIDIA's recommended solution.

Please refer to the paper (PDF within this repo) for more details. 

Reference code is provided, as are benchmarks. 

Note: Figure one's annotation contains an error. It should read: "Reduction a is available on threads
0 and 4, reduction b is on threads 1 and 5, etc."
