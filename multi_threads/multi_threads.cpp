/*
  README:  Test multiple threads
           each omp thread creates a private stream and launch a kernel 
           
  Notice:  OpenMP is not by default in HCC compiler. 
           Install it by https://bitbucket.org/multicoreware/hcc/wiki/OpenMP
*/

#include <omp.h>
#include <stdio.h>
#include <hip_runtime.h>

#define NUM_THREADS 4

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
	size_t N = 100000;

    omp_set_num_threads(NUM_THREADS);

	size_t Nbytes = NUM_THREADS * N * sizeof(float);

    int thread_id;

	hipDeviceProp_t props;
	CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
	printf ("info: running on device %s\n", props.name);

    hipStream_t  streams[NUM_THREADS];

	//printf ("info: num of streams = %d\n", NUM_THREADS);

	printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
	A_h = (float*)malloc(Nbytes);
	CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
	C_h = (float*)malloc(Nbytes);
	CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
	// Fill with Phi + i
    for (size_t i=0; i< NUM_THREADS * N; i++) 
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

/*============================================================================================ */

    //spawn openmp threads 
    #pragma omp parallel private(thread_id)
    {

        /* Obtain thread number */
        int thread_id = omp_get_thread_num();
    
        if(thread_id == 0)  printf("Using %d threads, %d streams \n", omp_get_num_threads( ), NUM_THREADS);

    	printf ("info: launch 'vector_square' kernel in thread %d\n", thread_id);

        hipStreamCreate(&streams[thread_id]); 


	    hipLaunchKernelGGL(HIP_KERNEL_NAME(vector_square), dim3(blocks), dim3(threadsPerBlock), 0, streams[thread_id], 
                        C_d+thread_id*N, A_d+thread_id*N, N);


        hipStreamSynchronize(streams[thread_id]);        

        hipStreamDestroy(streams[thread_id]);    

	    printf ("info: copy Device2Host in thread %d\n", thread_id);

    }

/*============================================================================================ */
    //verify result in main thread 
    CHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));


	printf ("info: check result\n");
    for (int i=0; i<NUM_THREADS * N; i++)  {
		if (C_h[i] != A_h[i] * A_h[i]) {
            printf("C_h[%d]=%f \n", i, C_h[i]);
			CHECK(hipErrorUnknown);
		}
	}
	printf ("PASSED!\n");
}
