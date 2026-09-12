// Udacity HW 4
// Parallel Radix Sort

#include "utils.h"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

#include <cstdio>
#include <utility>



// 1. Extract current bit


__global__
void bitFlagKernel(const unsigned int* input,
                   unsigned int* flags,
                   size_t numElems,
                   int bit)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems)
    {
        flags[idx] = (input[idx] >> bit) & 1;
    }
}



// 2. Stable scatter

//
// flags:
//     0 0 1 1 0 0 1
//
// prefix:
//     0 0 0 1 2 2 2
//
// prefix[idx] = number of 1s before idx
//
// If digit == 0:
//
//     number of zeros before idx
//     = idx - number of ones before idx
//
//     = idx - prefix[idx]
//
// If digit == 1:
//
//     totalZeros + number of ones before idx
//
//     = totalZeros + prefix[idx]
//


__global__
void stableScatterKernel(
    const unsigned int* inputVals,
    const unsigned int* inputPos,

    const unsigned int* flags,
    const unsigned int* prefix,

    unsigned int* outputVals,
    unsigned int* outputPos,

    size_t numElems,
    unsigned int totalZeros)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElems)
    {
        unsigned int digit = flags[idx];

        unsigned int outputIndex;

        if (digit == 0)
        {
            // Number of zeros before idx
            outputIndex = static_cast<unsigned int>(idx)
                        - prefix[idx];
        }
        else
        {
            // All zeros come first
            // Then the ones
            outputIndex = totalZeros + prefix[idx];
        }

        outputVals[outputIndex] = inputVals[idx];
        outputPos[outputIndex]  = inputPos[idx];
    }
}



// Radix Sort


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const int numBits = 32;

    const int threadsPerBlock = 256;

    const int numBlocks =static_cast<int>((numElems + threadsPerBlock - 1)/ threadsPerBlock);


    
    // Temporary arrays
    

    thrust::device_vector<unsigned int> d_flags(numElems);

    thrust::device_vector<unsigned int> d_prefix(numElems);


    unsigned int* flags =
        thrust::raw_pointer_cast(d_flags.data());

    unsigned int* prefix =
        thrust::raw_pointer_cast(d_prefix.data());


    
    // Ping-pong buffers
    

    unsigned int* valsIn  = d_inputVals;
    unsigned int* posIn   = d_inputPos;

    unsigned int* valsOut = d_outputVals;
    unsigned int* posOut  = d_outputPos;


    
    // Process every bit
    

    for (int bit = 0; bit < numBits; ++bit)
    {
       
        // Step 1:
        // Extract current bit
       

        bitFlagKernel<<<numBlocks, threadsPerBlock>>>(valsIn,flags,numElems,bit);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());


       
        // Step 2:
        // Exclusive prefix sum
        //
        // flags:
        //
        // 0 0 1 1 0 0 1
        //
        // prefix:
        //
        // 0 0 0 1 2 2 2
       

        thrust::exclusive_scan( d_flags.begin(),d_flags.end(),d_prefix.begin());


       
        // Step 3:
        // Count total number of 1s
       

        unsigned int totalOnes =thrust::reduce( d_flags.begin(),d_flags.end(),0u,thrust::plus<unsigned int>());


        unsigned int totalZeros =static_cast<unsigned int>(numElems)- totalOnes;


       
        // Step 4:
        // Stable scatter
       

        stableScatterKernel<<<
        numBlocks, threadsPerBlock
        >>>(
            valsIn,posIn,flags,prefix,valsOut,posOut,numElems,totalZeros
        );

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());


       
        // Step 5:
        // Ping-pong
       

        std::swap(valsIn, valsOut);
        std::swap(posIn, posOut);
    }


    
    // After 32 passes
    //
    // 32 is even.
    //
    // Therefore:
    //
    // valsIn  == d_inputVals
    // posIn   == d_inputPos
    //
    // But the assignment expects the result in:
    //
    // d_outputVals
    // d_outputPos
    //
    // So copy the final result.
    

    checkCudaErrors(cudaMemcpy(d_outputVals, valsIn,numElems * sizeof(unsigned int),cudaMemcpyDeviceToDevice));

    checkCudaErrors( cudaMemcpy(d_outputPos,posIn,numElems * sizeof(unsigned int),cudaMemcpyDeviceToDevice) );
}