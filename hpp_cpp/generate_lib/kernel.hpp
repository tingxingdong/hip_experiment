/*
 *  README:  perform reducton sum in hip kernel using device function
 *
 */


/*
 * Square each element in the array A and write to array C.
 */


#ifndef KERNEL_HPP
#define KERNEL_HPP

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

    res[0] = shared_A[0] ;

}

#endif


