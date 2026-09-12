// //Udacity HW 6
// //Poisson Blending

// /* Background
//    ==

//    The goal for this assignment is to take one image (the source) and
//    paste it into another image (the destination) attempting to match the
//    two images so that the pasting is non-obvious. This is
//    known as a "seamless clone".

//    The basic ideas are as follows:

//    1) Figure out the interior and border of the source image
//    2) Use the values of the border pixels in the destination image 
//       as boundary conditions for solving a Poisson equation that tells
//       us how to blend the images.
   
//       No pixels from the destination except pixels on the border
//       are used to compute the match.

//    Solving the Poisson Equation
//    

//    There are multiple ways to solve this equation - we choose an iterative
//    method - specifically the Jacobi method. Iterative methods start with
//    a guess of the solution and then iterate to try and improve the guess
//    until it stops changing.  If the problem was well-suited for the method
//    then it will stop and where it stops will be the solution.

//    The Jacobi method is the simplest iterative method and converges slowly - 
//    that is we need a lot of iterations to get to the answer, but it is the
//    easiest method to write.

//    Jacobi Iterations
//    =

//    Our initial guess is going to be the source image itself.  This is a pretty
//    good guess for what the blended image will look like and it means that
//    we won't have to do as many iterations compared to if we had started far
//    from the final solution.

//    ImageGuess_prev (Floating point)
//    ImageGuess_next (Floating point)

//    DestinationImg
//    SourceImg

//    Follow these steps to implement one iteration:

//    1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
//       Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
//              else if the neighbor in on the border then += DestinationImg[neighbor]

//       Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

//    2) Calculate the new pixel value:
//       float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
//       ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


//     In this assignment we will do 800 iterations.
//    */



// #include "utils.h"
// #include <thrust/host_vector.h>

// void your_blend(const uchar4* const h_sourceImg,  //IN
//                 const size_t numRowsSource, const size_t numColsSource,
//                 const uchar4* const h_destImg, //IN
//                 uchar4* const h_blendedImg) //OUT
// {

//   /* To Recap here are the steps you need to implement
  
//      1) Compute a mask of the pixels from the source image to be copied
//         The pixels that shouldn't be copied are completely white, they
//         have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

//      2) Compute the interior and border regions of the mask.  An interior
//         pixel has all 4 neighbors also inside the mask.  A border pixel is
//         in the mask itself, but has at least one neighbor that isn't.

//      3) Separate out the incoming image into three separate channels

//      4) Create two float(!) buffers for each color channel that will
//         act as our guesses.  Initialize them to the respective color
//         channel of the source image since that will act as our intial guess.

//      5) For each color channel perform the Jacobi iteration described 
//         above 800 times.

//      6) Create the output image by replacing all the interior pixels
//         in the destination image with the result of the Jacobi iterations.
//         Just cast the floating point values to unsigned chars since we have
//         already made sure to clamp them to the correct range.

//       Since this is final assignment we provide little boilerplate code to
//       help you.  Notice that all the input/output pointers are HOST pointers.

//       You will have to allocate all of your own GPU memory and perform your own
//       memcopies to get data in and out of the GPU memory.

//       Remember to wrap all of your calls with checkCudaErrors() to catch any
//       thing that might go wrong.  After each kernel call do:

//       cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

//       to catch any errors that happened while executing the kernel.
//   */
// }

// Udacity HW 6
// Poisson Blending

#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>


// 
// 1. Compute mask
// 
//
// Source image:
//   white pixel: R=255,G=255,B=255 -> do NOT copy
//   other pixel                    -> copy/blend
//
// uchar4:
//   .x = R
//   .y = G
//   .z = B
//   .w = A
//
__global__
void computeMaskKernel(const uchar4* sourceImg,
                       unsigned char* mask,
                       size_t numPixels)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPixels)
        return;

    uchar4 pixel = sourceImg[idx];

    mask[idx] = (pixel.x + pixel.y + pixel.z < 3 * 255) ? 1 : 0;
}


// 
// 2. Compute strict interior pixels
// 
//
// A pixel is strict interior iff:
//
//   mask[p]          == 1
//   mask[left]       == 1
//   mask[right]      == 1
//   mask[up]         == 1
//   mask[down]       == 1
//
// We use 0/1 unsigned char instead of a list.
// This is simpler and still fully parallel.
//
// Pixels on the outer image boundary are never considered
// interior.
//
__global__
void computeInteriorKernel(const unsigned char* mask,
                           unsigned char* strictInterior,
                           size_t numRows,
                           size_t numCols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= numRows || col >= numCols)
        return;

    size_t idx = row * numCols + col;

    // Boundary pixels cannot be strict interior.
    if (row == 0 || row == numRows - 1 ||
        col == 0 || col == numCols - 1)
    {
        strictInterior[idx] = 0;
        return;
    }

    if (mask[idx] &&
        mask[idx - 1] &&
        mask[idx + 1] &&
        mask[idx - numCols] &&
        mask[idx + numCols])
    {
        strictInterior[idx] = 1;
    }
    else
    {
        strictInterior[idx] = 0;
    }
}


// 
// 3. Initialize channels and compute g
// 
//
// g[p] = 4 * source[p]
//        - source[left]
//        - source[right]
//        - source[up]
//        - source[down]
//
// Only strict interior pixels need a meaningful g value.
// Other values are initialized to 0.
//
__global__
void initializeChannelKernel(const uchar4* sourceImg,
                             const unsigned char* strictInterior,
                             float* sourceChannel,
                             float* guess1,
                             float* guess2,
                             float* g,
                             size_t numRows,
                             size_t numCols,
                             int channel)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= numRows || col >= numCols)
        return;

    size_t idx = row * numCols + col;

    uchar4 pixel = sourceImg[idx];

    float value;

    if (channel == 0)
        value = static_cast<float>(pixel.x);   // R
    else if (channel == 1)
        value = static_cast<float>(pixel.y);   // G
    else
        value = static_cast<float>(pixel.z);   // B

    // Initial guess = source image
    sourceChannel[idx] = value;
    guess1[idx] = value;
    guess2[idx] = value;

    if (!strictInterior[idx])
    {
        g[idx] = 0.0f;
        return;
    }

    float left;
    float right;
    float up;
    float down;

    if (channel == 0)
    {
        left  = static_cast<float>(sourceImg[idx - 1].x);
        right = static_cast<float>(sourceImg[idx + 1].x);
        up    = static_cast<float>(sourceImg[idx - numCols].x);
        down  = static_cast<float>(sourceImg[idx + numCols].x);
    }
    else if (channel == 1)
    {
        left  = static_cast<float>(sourceImg[idx - 1].y);
        right = static_cast<float>(sourceImg[idx + 1].y);
        up    = static_cast<float>(sourceImg[idx - numCols].y);
        down  = static_cast<float>(sourceImg[idx + numCols].y);
    }
    else
    {
        left  = static_cast<float>(sourceImg[idx - 1].z);
        right = static_cast<float>(sourceImg[idx + 1].z);
        up    = static_cast<float>(sourceImg[idx - numCols].z);
        down  = static_cast<float>(sourceImg[idx + numCols].z);
    }

    g[idx] = 4.0f * value
           - left
           - right
           - up
           - down;
}


// 
// 4. One Jacobi iteration
// 
//
// For every strict interior pixel:
//
//     Sum1 = neighboring previous guesses
//          + destination boundary values
//
//     Sum2 = g[p]
//
//     next[p] = (Sum1 + g[p]) / 4
//
// If a neighbor is strict interior:
//     use previous guess
//
// Otherwise:
//     use destination image
//
// This matches reference_calc.cpp.
// 

__global__
void jacobiIterationKernel(const uchar4* destImg,
                           const unsigned char* strictInterior,
                           const float* previous,
                           const float* g,
                           float* next,
                           size_t numRows,
                           size_t numCols,
                           int channel)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= numRows || col >= numCols)
        return;

    size_t idx = row * numCols + col;

    // Only strict interior pixels are updated.
    if (!strictInterior[idx])
        return;

    float sum = 0.0f;

    
    // Left
    
    if (strictInterior[idx - 1])
    {
        sum += previous[idx - 1];
    }
    else
    {
        if (channel == 0)
            sum += static_cast<float>(destImg[idx - 1].x);
        else if (channel == 1)
            sum += static_cast<float>(destImg[idx - 1].y);
        else
            sum += static_cast<float>(destImg[idx - 1].z);
    }

    
    // Right
    
    if (strictInterior[idx + 1])
    {
        sum += previous[idx + 1];
    }
    else
    {
        if (channel == 0)
            sum += static_cast<float>(destImg[idx + 1].x);
        else if (channel == 1)
            sum += static_cast<float>(destImg[idx + 1].y);
        else
            sum += static_cast<float>(destImg[idx + 1].z);
    }

    
    // Up
    
    if (strictInterior[idx - numCols])
    {
        sum += previous[idx - numCols];
    }
    else
    {
        if (channel == 0)
            sum += static_cast<float>(destImg[idx - numCols].x);
        else if (channel == 1)
            sum += static_cast<float>(destImg[idx - numCols].y);
        else
            sum += static_cast<float>(destImg[idx - numCols].z);
    }

    
    // Down
    
    if (strictInterior[idx + numCols])
    {
        sum += previous[idx + numCols];
    }
    else
    {
        if (channel == 0)
            sum += static_cast<float>(destImg[idx + numCols].x);
        else if (channel == 1)
            sum += static_cast<float>(destImg[idx + numCols].y);
        else
            sum += static_cast<float>(destImg[idx + numCols].z);
    }

    // Jacobi update
    float newValue = (sum + g[idx]) / 4.0f;

    // Clamp to [0,255]
    newValue = fminf(255.0f, fmaxf(0.0f, newValue));

    next[idx] = newValue;
}


// 
// 5. Build output image
// 
//
// First copy destination image to output.
//
// Then replace only strict interior pixels with the computed
// Poisson-blended result.
//
__global__
void assembleOutputKernel(const uchar4* destImg,
                          const unsigned char* strictInterior,
                          const float* red,
                          const float* green,
                          const float* blue,
                          uchar4* blendedImg,
                          size_t numPixels)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPixels)
        return;

    // Start with destination image.
    blendedImg[idx] = destImg[idx];

    // Only replace strict interior pixels.
    if (strictInterior[idx])
    {
        uchar4 pixel = destImg[idx];

        pixel.x = static_cast<unsigned char>(red[idx]);
        pixel.y = static_cast<unsigned char>(green[idx]);
        pixel.z = static_cast<unsigned char>(blue[idx]);

        blendedImg[idx] = pixel;
    }
}


// 
// Host function
// 

void your_blend(const uchar4* const h_sourceImg,
                const size_t numRowsSource,
                const size_t numColsSource,
                const uchar4* const h_destImg,
                uchar4* const h_blendedImg)
{
    const size_t numPixels = numRowsSource * numColsSource;
    const size_t imageBytes = numPixels * sizeof(uchar4);
    const size_t maskBytes = numPixels * sizeof(unsigned char);
    const size_t floatBytes = numPixels * sizeof(float);

    
    // CUDA launch configuration
    

    const int block1D = 256;

    dim3 block2D(16, 16);

    dim3 grid1D(static_cast<unsigned int>((numPixels + block1D - 1) / block1D));

    dim3 grid2D(static_cast<unsigned int>((numColsSource + block2D.x - 1) / block2D.x),
    static_cast<unsigned int>((numRowsSource + block2D.y - 1) / block2D.y));


    
    // Device memory
    

    uchar4* d_sourceImg = nullptr;
    uchar4* d_destImg = nullptr;
    uchar4* d_blendedImg = nullptr;

    unsigned char* d_mask = nullptr;
    unsigned char* d_strictInterior = nullptr;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_sourceImg),imageBytes));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_destImg),imageBytes));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_blendedImg), imageBytes));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_mask), maskBytes));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_strictInterior),maskBytes));


    
    // Copy source and destination images to GPU
    

    checkCudaErrors(cudaMemcpy(d_sourceImg,h_sourceImg,imageBytes,cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_destImg,h_destImg,imageBytes,cudaMemcpyHostToDevice));


    // 
    // STEP 1: Compute mask
    // 

    computeMaskKernel<<<grid1D, block1D>>>(d_sourceImg,d_mask,numPixels);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    // 
    // STEP 2: Compute strict interior pixels
    // 

    computeInteriorKernel<<<grid2D, block2D>>>(
        d_mask,
        d_strictInterior,
        numRowsSource,
        numColsSource);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    // 
    // Allocate channel buffers
    // 

    float* d_sourceRed = nullptr;
    float* d_sourceGreen = nullptr;
    float* d_sourceBlue = nullptr;

    float* d_red1 = nullptr;
    float* d_red2 = nullptr;

    float* d_green1 = nullptr;
    float* d_green2 = nullptr;

    float* d_blue1 = nullptr;
    float* d_blue2 = nullptr;

    float* d_gRed = nullptr;
    float* d_gGreen = nullptr;
    float* d_gBlue = nullptr;


    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_sourceRed),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_sourceGreen),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_sourceBlue),floatBytes));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_red1),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_red2),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_green1),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_green2), floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_blue1),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_blue2),floatBytes));


    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_gRed),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_gGreen),floatBytes));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_gBlue),floatBytes));


    // 
    // STEP 3 + STEP 4
    //
    // Separate RGB channels
    // Initial guess = source image
    // Compute g
    // 

    initializeChannelKernel<<<grid2D, block2D>>>(
        d_sourceImg,
        d_strictInterior,
        d_sourceRed,
        d_red1,
        d_red2,
        d_gRed,
        numRowsSource,
        numColsSource,
        0);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    initializeChannelKernel<<<grid2D, block2D>>>(
        d_sourceImg,
        d_strictInterior,
        d_sourceGreen,
        d_green1,
        d_green2,
        d_gGreen,
        numRowsSource,
        numColsSource,
        1);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    initializeChannelKernel<<<grid2D, block2D>>>(
        d_sourceImg,
        d_strictInterior,
        d_sourceBlue,
        d_blue1,
        d_blue2,
        d_gBlue,
        numRowsSource,
        numColsSource,
        2);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    // 
    // STEP 5: 800 Jacobi iterations
    // 

    const size_t numIterations = 800;


    
    // RED
    

    for (size_t i = 0; i < numIterations; ++i)
    {
        jacobiIterationKernel<<<grid2D, block2D>>>(
            d_destImg,
            d_strictInterior,
            d_red1,
            d_gRed,
            d_red2,
            numRowsSource,
            numColsSource,
            0);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        std::swap(d_red1, d_red2);
    }


    
    // GREEN
    

    for (size_t i = 0; i < numIterations; ++i)
    {
        jacobiIterationKernel<<<grid2D, block2D>>>(
            d_destImg,
            d_strictInterior,
            d_green1,
            d_gGreen,
            d_green2,
            numRowsSource,
            numColsSource,
            1);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        std::swap(d_green1, d_green2);
    }


    
    // BLUE
    

    for (size_t i = 0; i < numIterations; ++i)
    {
        jacobiIterationKernel<<<grid2D, block2D>>>(
            d_destImg,
            d_strictInterior,
            d_blue1,
            d_gBlue,
            d_blue2,
            numRowsSource,
            numColsSource,
            2);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        std::swap(d_blue1, d_blue2);
    }


    // 
    // STEP 6: Assemble final image
    // 

    assembleOutputKernel<<<grid1D, block1D>>>(
        d_destImg,
        d_strictInterior,
        d_red1,
        d_green1,
        d_blue1,
        d_blendedImg,
        numPixels);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    // 
    // Copy result back to host
    // 

    checkCudaErrors(cudaMemcpy(
        h_blendedImg,
        d_blendedImg,
        imageBytes,
        cudaMemcpyDeviceToHost));


    // 
    // Free GPU memory
    // 

    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_destImg));
    checkCudaErrors(cudaFree(d_blendedImg));

    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_strictInterior));

    checkCudaErrors(cudaFree(d_sourceRed));
    checkCudaErrors(cudaFree(d_sourceGreen));
    checkCudaErrors(cudaFree(d_sourceBlue));

    checkCudaErrors(cudaFree(d_red1));
    checkCudaErrors(cudaFree(d_red2));

    checkCudaErrors(cudaFree(d_green1));
    checkCudaErrors(cudaFree(d_green2));

    checkCudaErrors(cudaFree(d_blue1));
    checkCudaErrors(cudaFree(d_blue2));

    checkCudaErrors(cudaFree(d_gRed));
    checkCudaErrors(cudaFree(d_gGreen));
    checkCudaErrors(cudaFree(d_gBlue));
}

