HIP_PATH?=$(shell hipconfig --path)

HIPCC= $(HIP_PATH)/bin/hipcc


all: complex.hip.out

complex.cuda.out : complex.cu
	nvcc complex.cu -o $@

#hipify complex.cu > complex.cpp
# Then review & finish port in complex.cpp

complex.hip.out: complex.cpp
	$(HIPCC) complex.cpp -o $@



clean:
	rm -f *.o *.out	
