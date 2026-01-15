/**
 * \file
 * \brief Unified Convolution Implementation for Subspot Processing
 * 
 * This file combines subspot-to-CPB convolution and GPU convolution algorithms
 */

#include "../include/algorithms/convolution.h"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>
#include <chrono>

// ============================================================================
// Subspot Data Reading and Processing
// ============================================================================

// 从纹理读取subspot数据并存储到SubspotInfo数组
__global__ void readSubspotDataKernel(
    cudaTextureObject_t subspotData,    // 输入：subspot数据纹理
    SubspotInfo* subspotInfoArray,      // 输出：subspot信息数组
    int nsubspot,                       // subspot数量
    int layerIdx,                       // 能量层索引
    vec3f beamDirection,                // 束流主方向
    vec3f bmXDirection,                 // 束流X方向
    vec3f bmYDirection,                 // 束流Y方向
    vec3f sourcePosition,                // 源点位置
    float sad,                          // Source-to-axis distance
    float refPlaneZ                     // 参考平面Z坐标
) {
    int subspotIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subspotIdx >= nsubspot) return;
    
    // 调试输出
    if (subspotIdx == 0 && layerIdx == 0) {
        printf("GPU Kernel: readSubspotDataKernel called with nsubspot=%d, layerIdx=%d\n", nsubspot, layerIdx);
    }
    
    // 使用SubspotInfo的纹理构造函数
    subspotInfoArray[subspotIdx] = SubspotInfo(
        subspotData, subspotIdx, layerIdx,
        beamDirection, bmXDirection, bmYDirection,
        sourcePosition, sad, refPlaneZ
    );
}

// 计算subspot影响范围
__global__ void calculateSubspotRangeKernel(
    const SubspotInfo* subspotInfoArray,
    int nsubspot,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    int* subspotRanges  // [nsubspot * 4] -> (minX, maxX, minY, maxY)
) {
    int subspotIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subspotIdx >= nsubspot) return;
    
    const SubspotInfo& subspot = subspotInfoArray[subspotIdx];
    if (!subspot.isValid) {
        subspotRanges[subspotIdx * 4 + 0] = 0;
        subspotRanges[subspotIdx * 4 + 1] = 0;
        subspotRanges[subspotIdx * 4 + 2] = 0;
        subspotRanges[subspotIdx * 4 + 3] = 0;
        return;
    }
    
    // 计算3-sigma截断范围
    float sigmaX = subspot.sigmaX;
    float sigmaY = subspot.sigmaY;
    float cutoff = SIGMA_CUTOFF;
    
    vec3f pos = subspot.position;
    float minX = pos.x - cutoff * sigmaX;
    float maxX = pos.x + cutoff * sigmaX;
    float minY = pos.y - cutoff * sigmaY;
    float maxY = pos.y + cutoff * sigmaY;
    
    // 转换为CPB网格索引
    int cpbMinX = max(0, (int)floorf((minX - cpbCorner.x) / cpbResolution.x));
    int cpbMaxX = min(cpbDims.x - 1, (int)ceilf((maxX - cpbCorner.x) / cpbResolution.x));
    int cpbMinY = max(0, (int)floorf((minY - cpbCorner.y) / cpbResolution.y));
    int cpbMaxY = min(cpbDims.y - 1, (int)ceilf((maxY - cpbCorner.y) / cpbResolution.y));
    
    // Debug output removed for performance
    
    subspotRanges[subspotIdx * 4 + 0] = cpbMinX;
    subspotRanges[subspotIdx * 4 + 1] = cpbMaxX;
    subspotRanges[subspotIdx * 4 + 2] = cpbMinY;
    subspotRanges[subspotIdx * 4 + 3] = cpbMaxY;
}

// ============================================================================
// Subspot to CPB Convolution
// ============================================================================

// 优化的subspot到CPB卷积kernel
__global__ void subspotToCPBConvolutionOptimizedKernel(
    const SubspotInfo* subspotInfoArray,
    const int* subspotRanges,
    int nsubspot,
    int layerIdx,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    float* cpbWeights
) {
    int subspotIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subspotIdx >= nsubspot) return;
    
    const SubspotInfo& subspot = subspotInfoArray[subspotIdx];
    if (!subspot.isValid) return;
    
    // 获取影响范围
    int minX = subspotRanges[subspotIdx * 4 + 0];
    int maxX = subspotRanges[subspotIdx * 4 + 1];
    int minY = subspotRanges[subspotIdx * 4 + 2];
    int maxY = subspotRanges[subspotIdx * 4 + 3];
    
    // Debug output removed for performance
    
    int processedPoints = 0;
    for (int cpbY = minY; cpbY <= maxY; cpbY++) {
        for (int cpbX = minX; cpbX <= maxX; cpbX++) {
            // Debug output removed for performance
            // 计算CPB网格点位置
            vec3f cpbPos = vec3f(
                cpbCorner.x + (cpbX + 0.5f) * cpbResolution.x,
                cpbCorner.y + (cpbY + 0.5f) * cpbResolution.y,
                cpbCorner.z
            );
            
            // 计算高斯权重
            float dx = cpbPos.x - subspot.position.x;
            float dy = cpbPos.y - subspot.position.y;
            float sigmaX = subspot.sigmaX;
            float sigmaY = subspot.sigmaY;
            
            // 使用误差函数计算精确的高斯积分
            float erfX1 = erf((dx - 0.5f * cpbResolution.x) / (1.41421356f * sigmaX));
            float erfX2 = erf((dx + 0.5f * cpbResolution.x) / (1.41421356f * sigmaX));
            float erfY1 = erf((dy - 0.5f * cpbResolution.y) / (1.41421356f * sigmaY));
            float erfY2 = erf((dy + 0.5f * cpbResolution.y) / (1.41421356f * sigmaY));
            
            float weight = 0.25f * (erfX2 - erfX1) * (erfY2 - erfY1) * subspot.weight;
            
            // 边界检查：确保索引在有效范围内
            if (cpbX >= 0 && cpbX < cpbDims.x && cpbY >= 0 && cpbY < cpbDims.y) {
                // 原子累加到CPB权重
                int cpbIdx = layerIdx * cpbDims.x * cpbDims.y + cpbY * cpbDims.x + cpbX;
                atomicAdd(&cpbWeights[cpbIdx], weight);
            }
        }
    }
}

// GPU 2D Convolution

__global__ void gpuConvolution2DKernel(
    float* input,           // 输入数据
    float* output,          // 输出数据
    int width,              // 输入宽度
    int height,             // 输入高度
    float* kernel,          // 卷积核
    int kernelSize,         // 卷积核大小
    int padding             // 填充大小
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int halfKernel = kernelSize / 2;
    
    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            
            // 边界处理
            if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                if (padding == 0) continue; // 零填充
                // 可以添加其他填充模式
            }
            
            int inputIdx = iy * width + ix;
            int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
            
            sum += input[inputIdx] * kernel[kernelIdx];
        }
    }
    
    output[y * width + x] = sum;
}

// ============================================================================
// Ray Weight Initialization
// ============================================================================

// GPU kernel: 将CPB权重映射到ray权重（只处理指定层）
__global__ void mapCPBWeightsToRayWeightsKernel(
    float* cpbWeights,               // input：CPB权重
    int cpbDimsX, int cpbDimsY, int cpbDimsZ,  // CPB维度（展开结构体）
    float cpbCornerX, float cpbCornerY, float cpbCornerZ,  // CPB corner（展开结构体）
    float cpbResolutionX, float cpbResolutionY, float cpbResolutionZ,  // CPB resolution（展开结构体）
    float* rayWeights,               // output
    int rayDimsX, int rayDimsY,     // ray网格维度（展开结构体）
    int layerIdx,                    // 当前能量层索引（只处理这一层）
    float beamDirX, float beamDirY, float beamDirZ,  // beam direction（展开结构体）
    float bmXDirX, float bmXDirY, float bmXDirZ,    // bmX direction（展开结构体）
    float bmYDirX, float bmYDirY, float bmYDirZ,    // bmY direction（展开结构体）
    float sourcePosX, float sourcePosY, float sourcePosZ,  // source position（展开结构体）
    float sad,                       // Source-to-axis distance
    float refPlaneZ                  // 参考平面Z
) {
    int rayY = blockIdx.y * blockDim.y + threadIdx.y;
    int rayX = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (rayY >= rayDimsY || rayX >= rayDimsX) return;
    
    int rayIdx = rayY * rayDimsX + rayX;
    rayWeights[rayIdx] = 0.0f;
    
    // ray在参考平面上的位置
    // 将ray网格映射到CPB网格区域
    float rayXPos = cpbCornerX + rayX * cpbResolutionX;
    float rayYPos = cpbCornerY + rayY * cpbResolutionY;
    
    // 计算在CPB网格中的索引（2D ray grid映射到3D CPB grid）
    int cpbX = (int)floorf((rayXPos - cpbCornerX) / cpbResolutionX);
    int cpbY = (int)floorf((rayYPos - cpbCornerY) / cpbResolutionY);
    
    if (cpbX >= 0 && cpbX < cpbDimsX && cpbY >= 0 && cpbY < cpbDimsY && layerIdx >= 0 && layerIdx < cpbDimsZ) {
        // 只处理当前能量层（不是所有层求和！）
        int cpbIdx = layerIdx * cpbDimsX * cpbDimsY + cpbY * cpbDimsX + cpbX;
        rayWeights[rayIdx] = cpbWeights[cpbIdx];
    }
}


void performSubspotToCPBConvolution(
    cudaTextureObject_t subspotData,
    int numLayers,
    int maxSubspotsPerLayer,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    float* cpbWeights,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
) {
    GPU_TIMER_START();
    
    SubspotInfo* d_subspotInfoArray;
    int* d_subspotRanges;
    float* d_cpbWeights;
    
    size_t subspotInfoSize = maxSubspotsPerLayer * sizeof(SubspotInfo);
    size_t subspotRangesSize = maxSubspotsPerLayer * 4 * sizeof(int);
    size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
    
    checkCudaErrors(cudaMalloc(&d_subspotInfoArray, subspotInfoSize));
    checkCudaErrors(cudaMalloc(&d_subspotRanges, subspotRangesSize));
    checkCudaErrors(cudaMalloc(&d_cpbWeights, cpbWeightsSize));
    
    // initialize
    checkCudaErrors(cudaMemset(d_cpbWeights, 0, cpbWeightsSize));

    for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {

        dim3 blockSize(256);
        dim3 gridSize((maxSubspotsPerLayer + blockSize.x - 1) / blockSize.x);
        
        printf("Host: Launching readSubspotDataKernel with gridSize=%d, blockSize=%d\n", gridSize.x, blockSize.x);
        
        readSubspotDataKernel<<<gridSize, blockSize>>>(
            subspotData, d_subspotInfoArray, maxSubspotsPerLayer, layerIdx,
            beamDirection, bmXDirection, bmYDirection,
            sourcePosition, sad, refPlaneZ
        );
        checkCudaErrors(cudaDeviceSynchronize());
        
        printf("Host: readSubspotDataKernel completed for layer %d\n", layerIdx);

        cudaDeviceSynchronize();

        calculateSubspotRangeKernel<<<gridSize, blockSize>>>(
            d_subspotInfoArray, maxSubspotsPerLayer,
            cpbCorner, cpbResolution, cpbDims,
            d_subspotRanges
        );
        checkCudaErrors(cudaDeviceSynchronize());
        
        std::vector<int> hostRanges(maxSubspotsPerLayer * 4);
        cudaMemcpy(hostRanges.data(), d_subspotRanges, maxSubspotsPerLayer * 4 * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Host: subspotRanges for layer %d:\n", layerIdx);
        for (int i = 0; i < maxSubspotsPerLayer; i++) {
            printf("  Subspot %d: minX=%d, maxX=%d, minY=%d, maxY=%d\n", 
                   i, hostRanges[i*4+0], hostRanges[i*4+1], hostRanges[i*4+2], hostRanges[i*4+3]);
        }
        
        subspotToCPBConvolutionOptimizedKernel<<<gridSize, blockSize>>>(
            d_subspotInfoArray, d_subspotRanges, maxSubspotsPerLayer, layerIdx,
            cpbCorner, cpbResolution, cpbDims, d_cpbWeights
        );
        checkCudaErrors(cudaDeviceSynchronize());
    }
    
    checkCudaErrors(cudaMemcpy(cpbWeights, d_cpbWeights, cpbWeightsSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_subspotInfoArray));
    checkCudaErrors(cudaFree(d_subspotRanges));
    checkCudaErrors(cudaFree(d_cpbWeights));
    
    GPU_TIMER_END("Subspot to CPB Convolution");
}


void performGPUConvolution2D(
    float* input,
    float* output,
    int width,
    int height,
    float* kernel,
    int kernelSize,
    int padding
) {
    GPU_TIMER_START();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    gpuConvolution2DKernel<<<gridSize, blockSize>>>(
        input, output, width, height, kernel, kernelSize, padding
    );
    checkCudaErrors(cudaDeviceSynchronize());
    
    GPU_TIMER_END("GPU 2D Convolution");
}

void performCPBToRayWeightMapping(
    float* cpbWeights,
    vec3i cpbDims,
    vec3f cpbCorner,
    vec3f cpbResolution,
    float* rayWeights,
    vec3i rayDims,
    int layerIdx,  // Current energy layer index
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
) {
    GPU_TIMER_START();

    // Validate input parameters
    if (cpbWeights == nullptr || rayWeights == nullptr) {
        fprintf(stderr, "CUDA error: Null pointer in performCPBToRayWeightMapping\n");
        exit(1);
    }
    
    if (cpbDims.x <= 0 || cpbDims.y <= 0 || cpbDims.z <= 0) {
        fprintf(stderr, "CUDA error: Invalid CPB dimensions (%d, %d, %d)\n", 
                cpbDims.x, cpbDims.y, cpbDims.z);
        exit(1);
    }
    
    if (rayDims.x <= 0 || rayDims.y <= 0) {
        fprintf(stderr, "CUDA error: Invalid ray dimensions (%d, %d)\n", 
                rayDims.x, rayDims.y);
        exit(1);
    }

    float* d_cpbWeights;
    float* d_rayWeights;
    
    size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
    size_t rayWeightsSize = rayDims.x * rayDims.y * sizeof(float); // 2D ray grid
    
    checkCudaErrors(cudaMalloc(&d_cpbWeights, cpbWeightsSize));
    checkCudaErrors(cudaMalloc(&d_rayWeights, rayWeightsSize));
    
    // Copy CPB weights to device; handle both host and device input pointers
    cudaPointerAttributes cpbAttr;
    cudaError_t attrErr = cudaPointerGetAttributes(&cpbAttr, cpbWeights);
    cudaMemcpyKind cpbCopyKind = cudaMemcpyDeviceToDevice; // default assume device pointer (common case)
    bool copyKindDetermined = false;
    
    if (attrErr == cudaSuccess) {
#if CUDART_VERSION >= 10000
        if (cpbAttr.type == cudaMemoryTypeHost || cpbAttr.type == cudaMemoryTypeUnregistered) {
            cpbCopyKind = cudaMemcpyHostToDevice;
        }
#else
        if (cpbAttr.memoryType == cudaMemoryTypeHost) {
            cpbCopyKind = cudaMemcpyHostToDevice;
        }
#endif
        copyKindDetermined = true;
    } else {
        // If attributes query fails, clear the error and we'll try DeviceToDevice first
        cudaGetLastError(); // clear the error from pointer query
    }
    
    // Perform the copy with error handling
    cudaError_t copyErr = cudaMemcpy(d_cpbWeights, cpbWeights, cpbWeightsSize, cpbCopyKind);
    if (copyErr != cudaSuccess && !copyKindDetermined) {
        // If copy failed and we didn't determine the copy kind from attributes,
        // try the alternative (HostToDevice)
        cudaGetLastError(); // clear the error
        cpbCopyKind = cudaMemcpyHostToDevice;
        copyErr = cudaMemcpy(d_cpbWeights, cpbWeights, cpbWeightsSize, cpbCopyKind);
    }
    
    if (copyErr != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMemcpy for cpbWeights: %s\n", cudaGetErrorString(copyErr));
        fprintf(stderr, "  Tried copy kind: %d (0=HostToDevice, 1=DeviceToHost, 2=DeviceToDevice)\n", (int)cpbCopyKind);
        fprintf(stderr, "  Size: %zu bytes\n", cpbWeightsSize);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    checkCudaErrors(cudaMemset(d_rayWeights, 0, rayWeightsSize));
    
    // Validate kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((rayDims.x + blockSize.x - 1) / blockSize.x,
                  (rayDims.y + blockSize.y - 1) / blockSize.y);
    
    // Check for valid grid and block sizes
    if (gridSize.x == 0 || gridSize.y == 0) {
        fprintf(stderr, "CUDA error: Invalid grid size (%d, %d) for rayDims (%d, %d)\n",
                gridSize.x, gridSize.y, rayDims.x, rayDims.y);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Check maximum grid size limits (typically 65535 for x and y)
    if (gridSize.x > 65535 || gridSize.y > 65535) {
        fprintf(stderr, "CUDA error: Grid size (%d, %d) exceeds maximum (65535, 65535)\n",
                gridSize.x, gridSize.y);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Validate kernel parameters before launch
    if (d_cpbWeights == nullptr || d_rayWeights == nullptr) {
        fprintf(stderr, "CUDA error: Null device pointer before kernel launch\n");
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Validate floating point parameters
    if (std::isnan(sad) || std::isinf(sad) || sad <= 0.0f || sad > 10000.0f) {
        fprintf(stderr, "CUDA error: Invalid SAD value: %f\n", sad);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    if (std::isnan(refPlaneZ) || std::isinf(refPlaneZ)) {
        fprintf(stderr, "CUDA error: Invalid refPlaneZ value: %f\n", refPlaneZ);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Debug output removed for performance
    
    // Launch kernel with explicit error checking
    // Clear any previous errors before launch
    cudaGetLastError();
    
    // Expand struct parameters to avoid kernel launch issues
    mapCPBWeightsToRayWeightsKernel<<<gridSize, blockSize>>>(
        d_cpbWeights,
        cpbDims.x, cpbDims.y, cpbDims.z,
        cpbCorner.x, cpbCorner.y, cpbCorner.z,
        cpbResolution.x, cpbResolution.y, cpbResolution.z,
        d_rayWeights,
        rayDims.x, rayDims.y,
        layerIdx,  // 只处理当前层
        beamDirection.x, beamDirection.y, beamDirection.z,
        bmXDirection.x, bmXDirection.y, bmXDirection.z,
        bmYDirection.x, bmYDirection.y, bmYDirection.z,
        sourcePosition.x, sourcePosition.y, sourcePosition.z,
        sad, refPlaneZ
    );
    
    // Check for kernel launch errors immediately
    cudaError_t launchErr = cudaPeekAtLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in mapCPBWeightsToRayWeightsKernel: %s\n",
                cudaGetErrorString(launchErr));
        fprintf(stderr, "  Parameters: gridSize=(%d,%d), blockSize=(%d,%d)\n",
                gridSize.x, gridSize.y, blockSize.x, blockSize.y);
        fprintf(stderr, "  rayDims=(%d,%d), cpbDims=(%d,%d,%d)\n",
                rayDims.x, rayDims.y, cpbDims.x, cpbDims.y, cpbDims.z);
        fprintf(stderr, "  beamDirection=(%.3f,%.3f,%.3f), sourcePosition=(%.3f,%.3f,%.3f)\n",
                beamDirection.x, beamDirection.y, beamDirection.z,
                sourcePosition.x, sourcePosition.y, sourcePosition.z);
        fprintf(stderr, "  sad=%.3f, refPlaneZ=%.3f\n", sad, refPlaneZ);
        
        // Check for invalid float values
        if (std::isnan(sad) || std::isinf(sad)) {
            fprintf(stderr, "  ERROR: Invalid SAD value detected!\n");
        }
        if (std::isnan(refPlaneZ) || std::isinf(refPlaneZ)) {
            fprintf(stderr, "  ERROR: Invalid refPlaneZ value detected!\n");
        }
        
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        // Don't exit - return error code instead to allow program to continue
        GPU_TIMER_END("CPB to Ray Weight Mapping");
        return;
    }
    
    // Synchronize and check for execution errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution error in mapCPBWeightsToRayWeightsKernel: %s\n",
                cudaGetErrorString(syncErr));
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Copy results back (check if rayWeights is host or device pointer)
    cudaPointerAttributes rayAttr;
    cudaError_t rayAttrErr = cudaPointerGetAttributes(&rayAttr, rayWeights);
    cudaMemcpyKind rayCopyKind = cudaMemcpyDeviceToDevice; // default assume device pointer (common case)
    bool rayCopyKindDetermined = false;
    
    if (rayAttrErr == cudaSuccess) {
#if CUDART_VERSION >= 10000
        if (rayAttr.type == cudaMemoryTypeHost || rayAttr.type == cudaMemoryTypeUnregistered) {
            rayCopyKind = cudaMemcpyDeviceToHost;
        }
#else
        if (rayAttr.memoryType == cudaMemoryTypeHost) {
            rayCopyKind = cudaMemcpyDeviceToHost;
        }
#endif
        rayCopyKindDetermined = true;
    } else {
        // If attributes query fails, clear the error and we'll try DeviceToDevice first
        cudaGetLastError(); // clear the error from pointer query
    }
    
    // Validate destination pointer before copy
    if (rayWeights == nullptr) {
        fprintf(stderr, "CUDA error: Null rayWeights pointer before copy\n");
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Perform the copy with error checking
    cudaError_t rayCopyErr = cudaMemcpy(rayWeights, d_rayWeights, rayWeightsSize, rayCopyKind);
    if (rayCopyErr != cudaSuccess && !rayCopyKindDetermined) {
        // If copy failed and we didn't determine the copy kind from attributes,
        // try the alternative (DeviceToHost)
        cudaGetLastError(); // clear the error
        rayCopyKind = cudaMemcpyDeviceToHost;
        rayCopyErr = cudaMemcpy(rayWeights, d_rayWeights, rayWeightsSize, rayCopyKind);
    }
    
    if (rayCopyErr != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMemcpy for rayWeights: %s\n", cudaGetErrorString(rayCopyErr));
        fprintf(stderr, "  Tried copy kind: %d (0=HostToDevice, 1=DeviceToHost, 2=DeviceToDevice)\n", (int)rayCopyKind);
        fprintf(stderr, "  Size: %zu bytes\n", rayWeightsSize);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    checkCudaErrors(cudaFree(d_cpbWeights));
    checkCudaErrors(cudaFree(d_rayWeights));
    
    GPU_TIMER_END("CPB to Ray Weight Mapping");
}
