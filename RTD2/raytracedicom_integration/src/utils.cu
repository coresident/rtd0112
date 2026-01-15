/**
 * \file
 * \brief Utility Functions Implementation for RayTraceDicom Integration
 */

#include "utils.h"
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <iostream>

// Texture creation utilities
cudaTextureObject_t create3DTexture(const float* data, const int3& dims, 
                                   cudaTextureFilterMode filterMode,
                                   cudaTextureAddressMode addressMode) {
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* devArray;
    cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
    cudaMalloc3DArray(&devArray, &floatChannelDesc, extent);
    
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data, extent.width*sizeof(float), extent.width, extent.height);
    copyParams.dstArray = devArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = filterMode;
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.addressMode[2] = addressMode;
    
    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    return texObj;
}

cudaTextureObject_t create2DTexture(const float* data, const int2& dims,
                                   cudaTextureFilterMode filterMode,
                                   cudaTextureAddressMode addressMode) {
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* devArray;
    cudaMallocArray(&devArray, &floatChannelDesc, dims.x, dims.y);
    
    // Use cudaMemcpy3D instead of deprecated cudaMemcpyToArray
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data, dims.x * sizeof(float), dims.x, dims.y);
    copyParams.dstArray = devArray;
    copyParams.extent = make_cudaExtent(dims.x, dims.y, 1);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = filterMode;
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    
    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    return texObj;
}

cudaTextureObject_t create1DTexture(const float* data, int size,
                                   cudaTextureFilterMode filterMode,
                                   cudaTextureAddressMode addressMode) {
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* devArray;
    cudaMallocArray(&devArray, &floatChannelDesc, size);
    
    // Use cudaMemcpy3D instead of deprecated cudaMemcpyToArray
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data, size * sizeof(float), size, 1);
    copyParams.dstArray = devArray;
    copyParams.extent = make_cudaExtent(size, 1, 1);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = filterMode;
    texDesc.addressMode[0] = addressMode;
    
    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    return texObj;
}

// Memory management utilities
void* allocateDeviceMemory(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void freeDeviceMemory(void* ptr) {
    cudaFree(ptr);
}

void copyToDevice(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void copyFromDevice(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

// Grid and block size calculation utilities
dim3 calculateGridSize(const dim3& blockSize, const int3& problemSize) {
    return dim3((problemSize.x + blockSize.x - 1) / blockSize.x,
                (problemSize.y + blockSize.y - 1) / blockSize.y,
                (problemSize.z + blockSize.z - 1) / blockSize.z);
}

dim3 calculateGridSize(const dim3& blockSize, const int2& problemSize) {
    return dim3((problemSize.x + blockSize.x - 1) / blockSize.x,
                (problemSize.y + blockSize.y - 1) / blockSize.y);
}

// Debug utilities
void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "Number of devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
}

void printMemoryInfo() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    std::cout << "GPU Memory Information:" << std::endl;
    std::cout << "  Total: " << total / (1024*1024) << " MB" << std::endl;
    std::cout << "  Free: " << free / (1024*1024) << " MB" << std::endl;
    std::cout << "  Used: " << (total - free) / (1024*1024) << " MB" << std::endl;
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
    }
}
