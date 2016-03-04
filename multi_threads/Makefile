HIP_PATH?=$(shell hipconfig --path)
HIPCC=$(HIP_PATH)/bin/hipcc

all: multi_threads.hip.out

multi_threads.cuda.out : multi_threads.cu
	nvcc multi_threads.cu -o $@

#hipify multi_threads.cu > multi_threads.cpp
# Then review & finish port in multi_threads.cpp

multi_threads.hip.out: multi_threads.cpp
	$(HIPCC) -fopenmp multi_threads.cpp -o $@



clean:
	rm -f *.o *.out	
