/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

// #include "utils.h"
// #include <cuda_runtime.h>
// #include <cfloat>
// #include <cstddef>


// void your_histogram_and_prefixsum(const float* const d_logLuminance,
//                                   unsigned int* const d_cdf,
//                                   float &min_logLum,
//                                   float &max_logLum,
//                                   const size_t numRows,
//                                   const size_t numCols,
//                                   const size_t numBins)
// {
//   //TODO
//   /*Here are the steps you need to implement
//     1) find the minimum and maximum value in the input logLuminance channel
//        store in min_logLum and max_logLum
//     2) subtract them to find the range
//     3) generate a histogram of all the values in the logLuminance channel using
//        the formula: bin = (lum[i] - lumMin) / lumRange * numBins
//     4) Perform an exclusive scan (prefix sum) on the histogram to get
//        the cumulative distribution of luminance values (this should go in the
//        incoming d_cdf pointer which already has been allocated for you)       */
// const size_t numPixels = numRows * numCols;

//     if (numPixels == 0 || numBins == 0)
//     {
//         min_logLum = 0.0f;
//         max_logLum = 0.0f;
//         return;
//     }


//     
//     // STEP 1
//     // Find minimum and maximum log luminance
//     

//     float* d_min;
//     float* d_max;

//     checkCudaErrors(
//         cudaMalloc(
//             reinterpret_cast<void**>(&d_min),
//             sizeof(float)
//         )
//     );

//     checkCudaErrors(
//         cudaMalloc(
//             reinterpret_cast<void**>(&d_max),
//             sizeof(float)
//         )
//     );


//     // Initialize device min/max.
//     //
//     // Since log luminance should be finite, these are safe
//     // initial values.

//     const float initialMin = FLT_MAX;
//     const float initialMax = -FLT_MAX;

//     checkCudaErrors(
//         cudaMemcpy(
//             d_min,
//             &initialMin,
//             sizeof(float),
//             cudaMemcpyHostToDevice
//         )
//     );

//     checkCudaErrors(
//         cudaMemcpy(
//             d_max,
//             &initialMax,
//             sizeof(float),
//             cudaMemcpyHostToDevice
//         )
//     );


//     // Launch min/max kernel.

//     const int threadsPerBlock = 256;

//     const int blocksPerGrid =
//         static_cast<int>(
//             (numPixels + threadsPerBlock - 1)
//             / threadsPerBlock
//         );


//     find_min_max_kernel<<<
//         blocksPerGrid,
//         threadsPerBlock
//     >>>(
//         d_logLuminance,
//         d_min,
//         d_max,
//         numPixels
//     );

//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());


//     // Copy min/max back to CPU.

//     checkCudaErrors(
//         cudaMemcpy(
//             &min_logLum,
//             d_min,
//             sizeof(float),
//             cudaMemcpyDeviceToHost
//         )
//     );

//     checkCudaErrors(
//         cudaMemcpy(
//             &max_logLum,
//             d_max,
//             sizeof(float),
//             cudaMemcpyDeviceToHost
//         )
//     );


//     
//     // STEP 2
//     // Calculate luminance range
//     

//     const float logLumRange =
//         max_logLum - min_logLum;


//     
//     // Special case:
//     // If every pixel has exactly the same luminance,
//     // logLumRange == 0.
//     //
//     // The reference implementation would divide by zero.
//     // We handle it safely here.
//     

//     if (logLumRange == 0.0f)
//     {
//         checkCudaErrors(
//             cudaMemset(
//                 d_cdf,
//                 0,
//                 sizeof(unsigned int) * numBins
//             )
//         );

//         checkCudaErrors(cudaFree(d_min));
//         checkCudaErrors(cudaFree(d_max));

//         return;
//     }


//     
//     // STEP 3
//     // Generate histogram
//     

//     // d_cdf is already allocated by preProcess().
//     //
//     // At this point d_cdf is used as the histogram.
//     //
//     // After the prefix sum it will become the CDF.

//     checkCudaErrors(
//         cudaMemset(
//             d_cdf,
//             0,
//             sizeof(unsigned int) * numBins
//         )
//     );


//     histogram_kernel<<<
//         blocksPerGrid,
//         threadsPerBlock
//     >>>(
//         d_logLuminance,
//         d_cdf,
//         min_logLum,
//         logLumRange,
//         numPixels,
//         numBins
//     );

//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());


//     
//     // STEP 4
//     // Exclusive prefix sum
//     

//     //
//     // numBins = 1024 in HW3.
//     //
//     // 1024 threads/block is allowed on your RTX 2060.
//     //
//     // Each thread handles one histogram bin.
//     //

//     const int scanThreads =
//         static_cast<int>(numBins);


//     const size_t sharedMemorySize =
//         sizeof(unsigned int) * numBins;


//     exclusive_scan_kernel<<<
//         1,
//         scanThreads,
//         sharedMemorySize
//     >>>(
//         d_cdf,
//         numBins
//     );

//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());


//     
//     // Cleanup temporary min/max memory
//     

//     checkCudaErrors(cudaFree(d_min));
//     checkCudaErrors(cudaFree(d_max));

// }
// /* Udacity Homework 3
//    HDR Tone-mapping

//    Student implementation:
//    1. Find minimum and maximum log luminance
//    2. Build histogram
//    3. Perform exclusive prefix sum
// */

#include "utils.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>



// Utility: atomic minimum for float


__device__
float atomicMinFloat(float* address, float value)
{
    int* address_as_int = reinterpret_cast<int*>(address);

    int old = *address_as_int;
    int assumed;

    while (value < __int_as_float(old))
    {
        assumed = old;

        old = atomicCAS(
            address_as_int,
            assumed,
            __float_as_int(value)
        );

        if (assumed == old)
            break;
    }

    return __int_as_float(old);
}



// Utility: atomic maximum for float


__device__
float atomicMaxFloat(float* address, float value)
{
    int* address_as_int = reinterpret_cast<int*>(address);

    int old = *address_as_int;
    int assumed;

    while (value > __int_as_float(old))
    {
        assumed = old;

        old = atomicCAS(
            address_as_int,
            assumed,
            __float_as_int(value)
        );

        if (assumed == old)
            break;
    }

    return __int_as_float(old);
}



// Kernel 1: Find minimum and maximum
//
// Every thread reads one luminance value and atomically
// updates the global minimum and maximum.


__global__
void find_min_max_kernel(
    const float* d_luminance,
    float* d_min,
    float* d_max,
    size_t numPixels)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (idx < numPixels)
    {
        float value = d_luminance[idx];

        atomicMinFloat(d_min, value);
        atomicMaxFloat(d_max, value);
    }
}



// Kernel 2: Build histogram
//
// Each thread processes one luminance value.
//
// bin =
//   (lum - minLum) / range * numBins
//
// Multiple threads may update the same bin, so atomicAdd()
// is required.


__global__
void histogram_kernel(
    const float* d_luminance,
    unsigned int* d_histogram,
    float minLum,
    float lumRange,
    size_t numPixels,
    size_t numBins)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (idx < numPixels)
    {
        float lum = d_luminance[idx];

        unsigned int bin =
            static_cast<unsigned int>(
                (lum - minLum)
                / lumRange
                * static_cast<float>(numBins)
            );

        // The maximum value can produce bin == numBins.
        // Valid bins are [0, numBins - 1].
        if (bin >= numBins)
        {
            bin = static_cast<unsigned int>(numBins - 1);
        }

        atomicAdd(&d_histogram[bin], 1);
    }
}



// Kernel 3: Exclusive scan
//
// This kernel performs an exclusive prefix sum on the
// histogram.
//
// Example:
//
// input:
// [4, 7, 3]
//
// output:
// [0, 4, 11]
//
// Since numBins = 1024, one CUDA block is enough.
//
// We use shared memory and the Hillis-Steele style scan.


__global__
void exclusive_scan_kernel(
    unsigned int* d_data,
    size_t numBins)
{
    extern __shared__ unsigned int shared[];

    unsigned int tid = threadIdx.x;

    // Load histogram into shared memory.
    if (tid < numBins)
    {
        shared[tid] = d_data[tid];
    }
    else
    {
        shared[tid] = 0;
    }

    __syncthreads();


    
    // Hillis-Steele inclusive scan
    //
    // [4 7 3 2]
    //
    // step = 1
    // [4 11 10 5]
    //
    // step = 2
    // [4 11 14 16]
    //
    // step = 4 ...
    

    for (unsigned int offset = 1;
         offset < numBins;
         offset *= 2)
    {
        unsigned int value = 0;

        if (tid >= offset)
        {
            value = shared[tid - offset];
        }

        __syncthreads();

        if (tid < numBins)
        {
            shared[tid] += value;
        }

        __syncthreads();
    }


    
    // Convert inclusive scan to exclusive scan.
    //
    // inclusive:
    // [4 11 14]
    //
    // exclusive:
    // [0 4 11]
    

    if (tid < numBins)
    {
        unsigned int inclusive = shared[tid];

        if (tid == 0)
        {
            d_data[tid] = 0;
        }
        else
        {
            d_data[tid] = shared[tid - 1];
        }
    }
}



// Main function required by Udacity HW3


void your_histogram_and_prefixsum(
    const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float &min_logLum,
    float &max_logLum,
    const size_t numRows,
    const size_t numCols,
    const size_t numBins)
{
    
    // Basic information
    

    const size_t numPixels = numRows * numCols;

    if (numPixels == 0 || numBins == 0)
    {
        min_logLum = 0.0f;
        max_logLum = 0.0f;
        return;
    }


    
    // STEP 1
    // Find minimum and maximum log luminance
    

    float* d_min;
    float* d_max;

    checkCudaErrors(
        cudaMalloc(
            reinterpret_cast<void**>(&d_min),
            sizeof(float)
        )
    );

    checkCudaErrors(
        cudaMalloc(
            reinterpret_cast<void**>(&d_max),
            sizeof(float)
        )
    );


    // Initialize device min/max.
    //
    // Since log luminance should be finite, these are safe
    // initial values.

    const float initialMin = FLT_MAX;
    const float initialMax = -FLT_MAX;

    checkCudaErrors(
        cudaMemcpy(
            d_min,
            &initialMin,
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    checkCudaErrors(
        cudaMemcpy(
            d_max,
            &initialMax,
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // Launch min/max kernel.

    const int threadsPerBlock = 256;

    const int blocksPerGrid =static_cast<int>((numPixels + threadsPerBlock - 1)/ threadsPerBlock);


    find_min_max_kernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_logLuminance,
        d_min,
        d_max,
        numPixels
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Copy min/max back to CPU.

    checkCudaErrors(cudaMemcpy(&min_logLum,d_min,sizeof(float),cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(&max_logLum,d_max,sizeof(float),cudaMemcpyDeviceToHost));


    
    // STEP 2
    // Calculate luminance range
    

    const float logLumRange =max_logLum - min_logLum;


    
    // Special case:
    // If every pixel has exactly the same luminance,
    // logLumRange == 0.
    //
    // The reference implementation would divide by zero.
    // We handle it safely here.
    

    if (logLumRange == 0.0f)
    {
        checkCudaErrors(cudaMemset( d_cdf,0,sizeof(unsigned int) * numBins));

        checkCudaErrors(cudaFree(d_min));
        checkCudaErrors(cudaFree(d_max));

        return;
    }


    
    // STEP 3
    // Generate histogram
    

    // d_cdf is already allocated by preProcess().
    //
    // At this point d_cdf is used as the histogram.
    //
    // After the prefix sum it will become the CDF.

    checkCudaErrors(cudaMemset( d_cdf,0,sizeof(unsigned int) * numBins));


    histogram_kernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_logLuminance,
        d_cdf,
        min_logLum,
        logLumRange,
        numPixels,
        numBins
    );

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    
    // STEP 4
    // Exclusive prefix sum
    

    //
    // numBins = 1024 in HW3.
    //
    // 1024 threads/block is allowed on your RTX 2060.
    //
    // Each thread handles one histogram bin.
    //

    const int scanThreads =static_cast<int>(numBins);


    const size_t sharedMemorySize =sizeof(unsigned int) * numBins;


    exclusive_scan_kernel<<<
        1,
        scanThreads,
        sharedMemorySize
    >>>(d_cdf,numBins);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    
    // Cleanup temporary min/max memory
    

    checkCudaErrors(cudaFree(d_min));
    checkCudaErrors(cudaFree(d_max));
}