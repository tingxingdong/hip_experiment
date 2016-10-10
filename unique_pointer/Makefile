HIP_PATH?=$(shell hipconfig --path)
HIPCC=$(HIP_PATH)/bin/hipcc

all: pointer_offset.hip.out

pointer_offset.cuda.out : pointer_offset.cu
	nvcc pointer_offset.cu -o $@

#hipify pointer_offset.cu > pointer_offset.cpp
# Then review & finish port in pointer_offset.cpp

pointer_offset.hip.out: pointer_offset.cpp
	$(HIPCC) -std=c++11 pointer_offset.cpp -g -o $@



clean:
	rm -f *.o *.out	
