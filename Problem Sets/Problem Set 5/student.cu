/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

// // __global__
// // void yourHisto(const unsigned int* const vals, //INPUT
// //                unsigned int* const histo,      //OUPUT
// //                int numVals)
// // {
// //    int idx = blockIdx.x * blockDim.x + threadIdx.x;

// //     if (idx < numVals)
// //     {
// //         unsigned int bin = vals[idx];

// //         atomicAdd(&histo[bin], 1);
// //     }
// //   //TODO fill in this kernel to calculate the histogram
// //   //as quickly as possible

// //   //Although we provide only one kernel skeleton,
// //   //feel free to use more if it will help you
// //   //write faster code
// // }

// // void computeHistogram(const unsigned int* const d_vals, //INPUT
// //                       unsigned int* const d_histo,      //OUTPUT
// //                       const unsigned int numBins,
// //                       const unsigned int numElems)
// // { 
// //    const int threadsPerBlock = 256;

// //     int blocksPerGrid =
// //         (numElems + threadsPerBlock - 1) / threadsPerBlock;

// //     yourHisto<<<blocksPerGrid, threadsPerBlock>>>(
// //         d_vals,
// //         d_histo,
// //         numElems
// //     );

// //     checkCudaErrors(cudaGetLastError());
// //     checkCudaErrors(cudaDeviceSynchronize());
// //   //TODO Launch the yourHisto kernel

// //   //if you want to use/launch more than one kernel,
// //   //feel free

// //   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
// // }

// ///////////////////////////////version2
// __global__
// void yourHisto(const unsigned int* const vals,
//                unsigned int* const histo,
//                int numVals)
// {
//     constexpr int NUM_BINS = 1024;
//     constexpr int WARP_SIZE = 32;
//     constexpr int NUM_WARPS = 8;   // 256 threads / 32

//     // 一个 block:
//     // 8 个 warp × 1024 bins
//     __shared__ unsigned int warpHisto[NUM_WARPS][NUM_BINS];

//     int tid = threadIdx.x;

//     // 当前 thread 属于哪个 warp
//     int warpId = tid / WARP_SIZE;

//     // 当前 thread 在 warp 内的位置
//     int laneId = tid % WARP_SIZE;

//     // --------------------------------------------------
//     // 1. 初始化每个 warp 自己的 histogram
//     // --------------------------------------------------

//     // 一个 warp 有 32 个线程。
//     // 让一个 warp 的 32 个线程合作清零 1024 个 bin。
//     for (int bin = laneId; bin < NUM_BINS; bin += WARP_SIZE)
//     {
//         warpHisto[warpId][bin] = 0;
//     }

//     __syncthreads();

//     // --------------------------------------------------
//     // 2. 计算 histogram
//     // --------------------------------------------------

//     int idx = blockIdx.x * blockDim.x + tid;
//     int stride = blockDim.x * gridDim.x;

//     while (idx < numVals)
//     {
//         unsigned int bin = vals[idx];

//         // 当前 warp 只更新自己的 histogram
//         atomicAdd(&warpHisto[warpId][bin], 1);

//         idx += stride;
//     }

//     __syncthreads();

//     // --------------------------------------------------
//     // 3. 把 8 个 warp 的 histogram 合并
//     // --------------------------------------------------

//     for (int bin = tid; bin < NUM_BINS; bin += blockDim.x)
//     {
//         unsigned int count = 0;

//         for (int w = 0; w < NUM_WARPS; ++w)
//         {
//             count += warpHisto[w][bin];
//         }

//         if (count > 0)
//         {
//             atomicAdd(&histo[bin], count);
//         }
//     }
// }


// void computeHistogram(const unsigned int* const d_vals,
//                       unsigned int* const d_histo,
//                       const unsigned int numBins,
//                       const unsigned int numElems)
// {
//     const int threadsPerBlock = 256;

//     int blocksPerGrid =
//         (numElems + threadsPerBlock - 1) / threadsPerBlock;

//     yourHisto<<<blocksPerGrid, threadsPerBlock>>>(
//         d_vals,
//         d_histo,
//         numElems
//     );

//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());

//     (void)numBins;
// }
////////////////////////////////////////////////////////////////

__global__
void yourHisto(const unsigned int* const vals,
               unsigned int* const histo,
               int numVals)
{
    // 每个 block 都建立自己的 histogram
    __shared__ unsigned int localHisto[1024];

    int tid = threadIdx.x;

    // 1. 把 shared histogram 清零
    for (int i = tid; i < 1024; i += blockDim.x)
    {
        localHisto[i] = 0;
    }

    // 确保整个 block 都完成清零
    __syncthreads();

    // 2. 每个 thread 处理一个或多个输入元素
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < numVals)
    {
        unsigned int bin = vals[idx];

        // 先更新 block 自己的 histogram
        atomicAdd(&localHisto[bin], 1);

        idx += stride;
    }

    // 确保所有 thread 完成 local histogram
    __syncthreads();

    // 3. 把每个 block 的 local histogram 加到 global histogram
    for (int i = tid; i < 1024; i += blockDim.x)
    {
        unsigned int count = localHisto[i];

        if (count > 0)
        {
            atomicAdd(&histo[i], count);
        }
    }
}

void computeHistogram(const unsigned int* const d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems)
{
    const int threadsPerBlock = 256;

    int blocksPerGrid =
        (numElems + threadsPerBlock - 1) / threadsPerBlock;

    yourHisto<<<blocksPerGrid, threadsPerBlock>>>(
        d_vals,
        d_histo,
        numElems
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    (void)numBins;
}

///////////////////////////////////////////////version3





// //./HW5
// 473
// Your code ran in: 4.539808 msecs.


// 474
// Your code ran in: 1.903296 msecs.


// 474
// Your code ran in: 0.907392 msecs.

// 539
// Your code ran in: 0.897088 msecs.
// CPU reference calculation: 8.75028 msecs.
// GPU result: Correct!
