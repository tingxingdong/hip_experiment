HIP_PATH?=$(shell hipconfig --path)
HIPCC=$(HIP_PATH)/bin/hipcc

all: reduction.hip.out

reduction.cuda.out : reduction.cu
	nvcc reduction.cu -o $@

#hipify reduction.cu > reduction.cpp
# Then review & finish port in reduction.cpp

reduction.hip.out: reduction.cpp
	$(HIPCC) reduction.cpp -o $@



clean:
	rm -f *.*~ *.o *.out	
