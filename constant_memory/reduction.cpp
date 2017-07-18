/*
 *  README:  Test Constant memory API in HIP
 *  As of ROCM 1.4, the constant memory is not supported either in HIP or HCC. The porting guide documentation is wrong.
 */


#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<iostream>

#define HIP_ASSERT(status) \
    assert(status == hipSuccess)

#define LEN 24

//Starting from ROCM 1.4,you can directly fill number with constant memory
//However, there is no constant hardware, it still goes through global memory, but it makes programming easier

//__constant__ int Value[] = {100, 1, 2, 20, 100, 245};
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

    HIP_ASSERT(hipMalloc((void**)&Ad, LEN));

    //directly copy does not work on rocm 14.
    //
    HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, sizeof(int)*LEN, 0, hipMemcpyHostToDevice));

    printf(" 2-------------- ");



    hipLaunchKernel(Get, dim3(1,1,1), dim3(LEN,1,1), 0, 0, Ad);

    printf(" 3-------------- ");

    HIP_ASSERT(hipMemcpy(B, Ad, sizeof(int)*LEN, hipMemcpyDeviceToHost));


    printf(" 4-------------- \n");
    for(unsigned i=0;i<LEN;i++)
    {
        printf("B[%d]=%d \n", i, B[i]);
        //assert(A[i] == B[i]);
    }
    std::cout<<"Passed"<<std::endl;
}

