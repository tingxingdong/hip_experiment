HIP_PATH?=$(shell hipconfig --path)

HIPCC= $(HIP_PATH)/bin/hipcc


all: call.hip.out

call.cuda.out : call.cu
	nvcc call.cu -o $@

#hipify call.cu > call.cpp
# Then review & finish port in call.cpp

call.hip.out: call.cpp
	$(HIPCC) call.cpp -o $@ ../generate_lib/libkernel.a



clean:
	rm -f *.o *.out *~	
