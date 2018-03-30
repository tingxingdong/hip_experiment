/*
 *  README:  perform reducton sum in hip kernel using device function
 *
 */


#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <typeinfo>
#include "kernel.hpp"

//using namespace std;


#define CHECK(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}



extern "C"
int goo(int argc, char *argv[])
{
	void *A_d, *C_d;
	float *A_h, *C_h;
	size_t N = NB_X;
	size_t Nbytes = N * sizeof(float);

	hipDeviceProp_t props;
	CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
	printf ("info: running on device %s\n", props.name);

	printf ("info: allocate host mem (%6.2f KB)\n", 2*Nbytes/1024.0);
	A_h = (float*)malloc(Nbytes);
	CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );

    C_h = (float*)malloc(sizeof(float));
	CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
	// Fill with Phi + i

    for (size_t i=0; i<N; i++)
	{
		A_h[i]= 1.618f ;
	}

	printf ("info: allocate device mem (%6.2f KB)\n", 2*Nbytes/1024.0);
	CHECK(hipMalloc(&A_d, Nbytes));
	CHECK(hipMalloc(&C_d, sizeof(float)));

	printf ("info: copy Host2Device\n");
    CHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

	const unsigned blocks = 1;
	const unsigned threadsPerBlock = NB_X;

	printf ("info: launch 'rocblas_sum_kernel' kernel\n");
	hipLaunchKernel(HIP_KERNEL_NAME(rocblas_sum_kernel), dim3(blocks), dim3(threadsPerBlock), 0, 0, (float*)C_d, (float*)A_d, N);

	printf ("info: copy Device2Host\n");
    CHECK ( hipMemcpy(C_h, C_d, sizeof(float), hipMemcpyDeviceToHost));

	printf ("info: check result\n");

    //
    float result = 0; //both real, imag compoent will be initilaized with 0

    //the +=, = operators are overloaed already
    for (size_t i=0; i<N; i++)  {
        result += A_h[i];
    }

    printf("CPU result=%f, GPU result=%f\n", result, C_h[0]);


    if (C_h[0] != result) {
		CHECK(hipErrorUnknown);
	}

	printf ("PASSED!\n");

    hipFree(C_d);
    hipFree(A_d);
}



