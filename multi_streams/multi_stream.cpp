/*
  README:  Test multiple streams in one host thread

*/
#include <stdio.h>
#include <hip_runtime.h>

#define NUM_STREAMS 4

#define CHECK(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}


/* 
 * Square each element in the array A and write to array C.
 */

//hipLaunchParm lp,
template <typename T>
__global__ void
vector_square(hipLaunchParm lp,T *C_d, T *A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
		C_d[i] = A_d[i] * A_d[i];
	}
}


int main(int argc, char *argv[])
{
	float *A_d, *C_d;
	float *A_h, *C_h;
	size_t N = 1000000;
	size_t Nbytes = NUM_STREAMS * N * sizeof(float);

	hipDeviceProp_t props;
	CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
	printf ("info: running on device %s\n", props.name);

    hipStream_t  streams[NUM_STREAMS];

	printf ("info: num of streams = %d\n", NUM_STREAMS);

	printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
	A_h = (float*)malloc(Nbytes);
	CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
	C_h = (float*)malloc(Nbytes);
	CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
	// Fill with Phi + i
    for (size_t i=0; i< NUM_STREAMS * N; i++) 
	{
		A_h[i] = 1.618f + i; 
	}

	printf ("info: allocate device mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
	CHECK(hipMalloc(&A_d, Nbytes));
	CHECK(hipMalloc(&C_d, Nbytes));


	printf ("info: copy Host2Device\n");
    CHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

	const unsigned blocks = 512;
	const unsigned threadsPerBlock = 256;

	printf ("info: launch 'vector_square' kernel\n");
    
    for(int i=0; i<NUM_STREAMS; i++){
        hipStreamCreate(&streams[i]); 
    }

    for(int i=0; i<NUM_STREAMS; i++){
	    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_square), dim3(blocks), dim3(threadsPerBlock), 0, streams[i], C_d+i*N, A_d+i*N, N);
    }

    for(int i=0; i<NUM_STREAMS; i++){
        hipStreamSynchronize(streams[i]);
    }

    for (int i=0; i<NUM_STREAMS; i++){
        hipStreamDestroy(streams[i]);    
    }
	printf ("info: copy Device2Host\n");
    CHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

	printf ("info: check result\n");
    for (size_t i=0; i<NUM_STREAMS * N; i++)  {
		if (C_h[i] != A_h[i] * A_h[i]) {
			CHECK(hipErrorUnknown);
		}
	}
	printf ("PASSED!\n");
}
