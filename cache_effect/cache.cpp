/*
 *  README:  perform reducton sum in hip kernel using device function
 *
 */


#include <stdio.h>
#include <hip_runtime.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <sys/time.h> 

#define CHECK(error) \
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}


// CPU Timer(in millisecond): 
    double rocblas_wtime( void ){
        hipDeviceSynchronize();
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (tv.tv_sec * 1000) + tv.tv_usec /1000;
    };

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


/*
 * Square each element in the array A and write to array C.
 */

#define NB_X  256
#define NUM_COLUMN  256

template <typename T>
__global__ void
gemv_kernel(hipLaunchParm lp, T *A, T *X, T *Y, size_t NUM_ROW)
{
    if(NUM_ROW == 0) {
        return;
    }

    int tid = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    
    int tx = hipThreadIdx_x;

    T res = 0.0;

    __shared__ T sX[NUM_COLUMN];
    
    for(int i=tx; tx<NUM_COLUMN; tx+=hipBlockDim_x)
    {
        sX[i] = X[i];
    }

    __syncthreads();

    if(tid < NUM_ROW)
    {
        for(int i=0; i<NUM_COLUMN;i++)
        {
           res += A[tid + i * NUM_ROW] * sX[i] ;
        }

        Y[tid] = res;
    }
}


void no_cache(float *A_h, float *A_d, float *X_h, float *X_d, float *Y_h, float *Y_d, size_t NUM_ROW, int p=0)
{


	if(p) printf ("info: allocate host mem (%6.2f KB)\n", NUM_COLUMN*NUM_ROW*sizeof(float)/1024.0);

	A_h = (float*)malloc(NUM_ROW * NUM_COLUMN * sizeof(float));
	CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );

    X_h = (float*)malloc(NUM_COLUMN * sizeof(float));
	CHECK(X_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    
    Y_h = (float*)malloc(NUM_ROW * sizeof(float));
	CHECK(Y_h == 0 ? hipErrorMemoryAllocation : hipSuccess );

	// Fill with Phi + i
    for (size_t i=0; i<NUM_ROW * NUM_COLUMN; i++)
	{
		 A_h[i] = 1.618f + (i % NB_X);
	}

    for (size_t i=0; i< NUM_COLUMN; i++)
	{
		 X_h[i] = 1.618f + i;
	}

    
	if(p) printf ("info: allocate device mem (%6.2f KB)\n", NUM_ROW * NUM_COLUMN * sizeof(float)/1024.0);
	CHECK(hipMalloc(&A_d, NUM_ROW * NUM_COLUMN * sizeof(float)));
	CHECK(hipMalloc(&X_d, NUM_COLUMN * sizeof(float)));
	CHECK(hipMalloc(&Y_d, NUM_ROW * sizeof(float)));

	if(p) printf ("info: copy Host2Device\n");
    CHECK ( hipMemcpy(A_d, A_h, NUM_ROW * NUM_COLUMN * sizeof(float), hipMemcpyHostToDevice));
    CHECK ( hipMemcpy(X_d, X_h, NUM_COLUMN * sizeof(float), hipMemcpyHostToDevice));

	const unsigned blocks = (NUM_ROW -1)/NB_X + 1;
	const unsigned threadsPerBlock = NB_X;

	if(p) printf ("info: launch 'gemv_kernel' kernel\n");


    for(int i=1 ; i < 1e3; i*=2)
    {
        size_t num_row = NB_X * i;
        clock_t t;
        t = clock();
        
        double time; 
        time = rocblas_wtime();

	    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_kernel), dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, X_d, Y_d, num_row);
    
        //time = rocblas_wtime() - time;

        hipDeviceSynchronize();
        t = clock() - t;
        time = ((double)t)/CLOCKS_PER_SEC*1000 ;

        if(p) printf ("It took me %d clicks (%f miliseconds).\n",t, time);
    
        
        printf ("Row = %d, It took me (%f milliseconds), Gflops=%f\n",time, num_row, 2*num_row*NUM_COLUMN/(time)/10e6);
    }
/*
	if(p) printf ("info: copy Device2Host\n");
    CHECK ( hipMemcpy(Y_h, Y_d, NUM_ROW * sizeof(float), hipMemcpyDeviceToHost));

	if(p) printf ("info: check result\n");


    for (size_t i=0; i<NUM_ROW; i++)  {
        float res = 0;
        for(int j=0; j<NUM_COLUMN; j++){
            res += A_h[i + j * NUM_ROW] * X_h[j];
        }
        if (Y_h[i] != res) 
        {
            printf("i=%d, CPU result=%f, GPU result=%f\n", i, res, Y_h[i]);
		    //CHECK(hipErrorUnknown);
        }
    }

*/
	if(p) printf ("PASSED!\n");

    hipFree(A_d);
    hipFree(Y_d);
    hipFree(X_d);

    free(A_h);
    free(Y_h);
    free(X_h);

}



int main(int argc, char *argv[])
{
	float *A_h, *A_d;
	float *Y_h, *Y_d;
	float *X_h, *X_d;

	hipDeviceProp_t props;
	CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
	printf ("info: running on device %s\n", props.name);

//bug will appear if num_row is too big 3125*128
    //for(int i=1 ; i < 1e3; i*=2)
    int i = 131072/NB_X;
    { 
        size_t num_row = NB_X * i;
        
        no_cache(A_h, A_d, X_h, X_d, Y_h, Y_d, num_row);
    }
}



