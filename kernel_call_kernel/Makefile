HIP_PATH?=$(shell hipconfig --path)
HIPCC=$(HIP_PATH)/bin/hipcc

all: kernel_call_kernel.hip.out

kernel_call_kernel.cuda.out : kernel_call_kernel.cu
	nvcc kernel_call_kernel.cu -o $@

#hipify kernel_call_kernel.cu > kernel_call_kernel.cpp
# Then review & finish port in kernel_call_kernel.cpp

kernel_call_kernel.hip.out: kernel_call_kernel.cpp
	$(HIPCC) kernel_call_kernel.cpp -o $@



clean:
	rm -f *.o *.out	
