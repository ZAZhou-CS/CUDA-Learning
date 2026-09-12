#include "utils.h"


// 1. Gaussian blur kernel

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows,
                   int numCols,
                   const float* const filter,
                   const int filterWidth)
{
    
    // 每个 thread 负责一个 pixel
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 防止 thread 跑到图像范围之外
    if (x >= numCols || y >= numRows)
    {
        return;
    }

    // 当前 pixel 在一维数组中的位置
    const int pixelIndex = y * numCols + x;

    // filter 的半径
    const int filterRadius = filterWidth / 2;

    // 中间计算必须使用 float
    float result = 0.0f;

    
    // 遍历 filter
    
    for (int filter_r = -filterRadius;
         filter_r <= filterRadius;
         ++filter_r)
    {
        for (int filter_c = -filterRadius;
             filter_c <= filterRadius;
             ++filter_c)
        {
            // 当前 filter 元素对应的 image 坐标
            int image_r = y + filter_r;
            int image_c = x + filter_c;

            // ------------------------------------------------
            // clamp 到合法的 image 范围
            // ------------------------------------------------

            if (image_r < 0)
            {
                image_r = 0;
            }
            else if (image_r >= numRows)
            {
                image_r = numRows - 1;
            }

            if (image_c < 0)
            {
                image_c = 0;
            }
            else if (image_c >= numCols)
            {
                image_c = numCols - 1;
            }

            // image 中邻居 pixel 的一维位置
            const int imageIndex = image_r * numCols + image_c;

            // filter 中当前元素的一维位置
            const int filterIndex =(filter_r + filterRadius) * filterWidth + (filter_c + filterRadius);

            // 读取 image pixel
            const float imageValue =static_cast<float>(inputChannel[imageIndex]);

            // 读取 filter value
            const float filterValue =filter[filterIndex];

            // weighted sum
            result += imageValue * filterValue;
        }
    }

    
    // 最后转换成 unsigned char
    // 和 reference implementation 保持一致
    
    outputChannel[pixelIndex] =static_cast<unsigned char>(result);
}


// 2. Separate RGBA image into R / G / B channels

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
    
    // 2D thread -> 2D image pixel
    

    const int x =blockIdx.x * blockDim.x + threadIdx.x;

    const int y =blockIdx.y * blockDim.y + threadIdx.y;

    // 越界 thread 不处理
    if (x >= numCols || y >= numRows)
    {
        return;
    }

    // 2D -> 1D
    const int pixelIndex =y * numCols + x;

    // 读取一个 RGBA pixel
    uchar4 pixel =inputImageRGBA[pixelIndex];

    
    // AoS -> SoA
    //
    // RGBA RGBA RGBA ...
    //
    //     ↓
    //
    // R R R R ...
    // G G G G ...
    // B B B B ...
    

    redChannel[pixelIndex]   = pixel.x;
    greenChannel[pixelIndex] = pixel.y;
    blueChannel[pixelIndex]  = pixel.z;
}


// 3. Recombine R / G / B channels

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
    const int2 thread_2D_pos =make_int2(
            blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos =thread_2D_pos.y * numCols + thread_2D_pos.x;

    // 防止越界
    if (thread_2D_pos.x >= numCols ||thread_2D_pos.y >= numRows)
    {
        return;
    }

    unsigned char red =redChannel[thread_1D_pos];

    unsigned char green =greenChannel[thread_1D_pos];

    unsigned char blue =blueChannel[thread_1D_pos];

    // Alpha = 255
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] =outputPixel;
}


// Device memory

unsigned char *d_red;
unsigned char *d_green;
unsigned char *d_blue;

float *d_filter;


// 4. Allocate GPU memory and copy filter

void allocateMemoryAndCopyToGPU(
    const size_t numRowsImage,
    const size_t numColsImage,
    const float* const h_filter,
    const size_t filterWidth)
{
    const size_t numPixels =numRowsImage * numColsImage;

    
    // Allocate R / G / B channels
    

    checkCudaErrors(cudaMalloc(&d_red,sizeof(unsigned char) * numPixels));

    checkCudaErrors(cudaMalloc(&d_green,sizeof(unsigned char) * numPixels));

    checkCudaErrors(cudaMalloc(&d_blue,sizeof(unsigned char) * numPixels));

    
    // Allocate filter on GPU
    

    const size_t filterSize =filterWidth * filterWidth;

    checkCudaErrors(cudaMalloc(&d_filter,sizeof(float) * filterSize));

    
    // Copy filter:
    //
    // Host h_filter
    //       ↓
    // Device d_filter
    

    checkCudaErrors(cudaMemcpy(d_filter,h_filter,sizeof(float) * filterSize,cudaMemcpyHostToDevice));
}


// 5. Main Gaussian blur function

void your_gaussian_blur(
    const uchar4 * const h_inputImageRGBA,
    uchar4 * const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA,
    const size_t numRows,
    const size_t numCols,
    unsigned char *d_redBlurred,
    unsigned char *d_greenBlurred,
    unsigned char *d_blueBlurred,
    const int filterWidth)
{
    
    // Block size
    //
    // 16 x 16 = 256 threads
    

    const dim3 blockSize(16, 16);

    
    // Grid size
    //
    // ceil(numCols / 16)
    // ceil(numRows / 16)
    

    const dim3 gridSize(
        (numCols + blockSize.x - 1) / blockSize.x,
        (numRows + blockSize.y - 1) / blockSize.y
    );


    
    // Step 1:
    // RGBA -> R / G / B
    

    separateChannels<<<gridSize, blockSize>>>( d_inputImageRGBA,
        numRows,
        numCols,
        d_red,
        d_green,
        d_blue
    );

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    
    // Step 2:
    // Gaussian blur R channel
    

    gaussian_blur<<<gridSize, blockSize>>>(
        d_red,
        d_redBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth
    );


    
    // Step 3:
    // Gaussian blur G channel
    

    gaussian_blur<<<gridSize, blockSize>>>(
        d_green,
        d_greenBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth
    );


    
    // Step 4:
    // Gaussian blur B channel
    

    gaussian_blur<<<gridSize, blockSize>>>(
        d_blue,
        d_blueBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth
    );


    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());


    
    // Step 5:
    // R / G / B -> RGBA
    

    recombineChannels<<<gridSize, blockSize>>>(
        d_redBlurred,
        d_greenBlurred,
        d_blueBlurred,
        d_outputImageRGBA,
        numRows,
        numCols
    );

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}


// 6. Cleanup

void cleanup()
{
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));

    checkCudaErrors(cudaFree(d_filter));
}