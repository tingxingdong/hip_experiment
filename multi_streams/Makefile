HIP_PATH?=$(shell hipconfig --path)
HIPCC=$(HIP_PATH)/bin/hipcc

all: multi_stream.hip.out

multi_stream.cuda.out : multi_stream.cu
	nvcc multi_stream.cu -o $@

#hipify multi_stream.cu > multi_stream.cpp
# Then review & finish port in multi_stream.cpp

multi_stream.hip.out: multi_stream.cpp
	$(HIPCC) multi_stream.cpp -o $@



clean:
	rm -f *.o *.out	
