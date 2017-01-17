/*
 *  README:  Test Constant memory API in HIP
 *  As of ROCM 1.4, the constant memory is not supported either in HIP or HCC. The porting guide documentation is wrong.
 */


#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>

#define HIP_ASSERT(status) \
    assert(status == hipSuccess)

#define LEN 512
#define SIZE 2048

__constant__ int Value[LEN];

__global__ void Get(hipLaunchParm lp, int *Ad)
{
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Ad[tid] = Value[tid];
}

int main()
{
    int *A, *B, *Ad;
    A = new int[LEN];
    B = new int[LEN];
    for(unsigned i=0;i<LEN;i++)
    {
        A[i] = -1*i;
        B[i] = 0;
    }

    printf(" 1-------------- ");

    HIP_ASSERT(hipMalloc((void**)&Ad, SIZE));

    //HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));

    printf(" 2-------------- ");

#if 1

    hipLaunchKernel(Get, dim3(1,1,1), dim3(LEN,1,1), 0, 0, Ad);

    printf(" 3-------------- ");

    HIP_ASSERT(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
#endif

    printf(" 4-------------- ");
    for(unsigned i=0;i<LEN;i++)
    {
        assert(A[i] == B[i]);
    }
    std::cout<<"Passed"<<std::endl;
}

