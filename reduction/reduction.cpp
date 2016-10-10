/*
 *  README:  perform reducton sum in hip kernel using device function
 *
 */


#include <stdio.h>
#include <hip_runtime.h>

#define CHECK(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}



/*! \brief This device function is used in various BLAS routines demanding reduction sum.


    \details

    @param[in]
    n         rocblas_int. assume n <= 1024 and a mutiple of 2;
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is assumed stored in shared memory;

              x[0] store the final result.

              This device function is verified by Tim Dong on Feb 2016.
    ********************************************************************/

template< int n, typename T >
__device__ void
rocblas_sum_reduce( int i, T* x )
{
    __syncthreads();
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }

    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce


/*
 * Square each element in the array A and write to array C.
 */

#define NB_X  512

template <typename T>
__global__ void
rocblas_sum_kernel(hipLaunchParm lp, T *res, const T *A, size_t N)
{
    if(N == 0) {
        res[0] = 0.0;
        return;
    }

    int tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);

    int tx = hipThreadIdx_x;

    __shared__ T shared_A[NB_X];

    shared_A[tx] = A[tid];

    rocblas_sum_reduce<NB_X, T>(tx, shared_A);

    if(tx == 0) res[0] = shared_A[0] ;
}


int main(int argc, char *argv[])
{
	float *A_d, *C_d;
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
		A_h[i] = 1.618f + i;
	}

	printf ("info: allocate device mem (%6.2f KB)\n", 2*Nbytes/1024.0);
	CHECK(hipMalloc(&A_d, Nbytes));
	CHECK(hipMalloc(&C_d, sizeof(float)));


	printf ("info: copy Host2Device\n");
    CHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

	const unsigned blocks = 1;
	const unsigned threadsPerBlock = NB_X;

	printf ("info: launch 'rocblas_sum_kernel' kernel\n");
	hipLaunchKernel(HIP_KERNEL_NAME(rocblas_sum_kernel), dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

	printf ("info: copy Device2Host\n");
    CHECK ( hipMemcpy(C_h, C_d, sizeof(float), hipMemcpyDeviceToHost));

	printf ("info: check result\n");
    float result = 0.0;

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
