HIP_PATH?=$(shell hipconfig --path)
HIPCC=$(HIP_PATH)/bin/hipcc

all: cache.hip.out

cache.cuda.out : cache.cu
	nvcc cache.cu -o $@

#hipify cache.cu > cache.cpp
# Then review & finish port in cache.cpp

cache.hip.out: cache.cpp
	$(HIPCC) -O3 cache.cpp -o $@



clean:
	rm -f *.o *.out	
