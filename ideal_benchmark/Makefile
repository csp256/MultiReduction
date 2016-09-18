all:
	@nvcc multireduction_benchmark.cu -lineinfo -std=c++11 -o multireduction_benchmark -arch=sm_30

clean:
	-@rm multireduction_benchmark 2> /dev/null || true

run: all
	@./multireduction_benchmark 0

test: clean run

nvvp:
	@nvvp ./multireduction_benchmark 0 &
