/**
 * \file
 * \brief Superposition Kernel Components for RayTraceDicom Integration
 * 
 * Includes template-based superposition kernels and convolution utilities
 */

#ifndef SUPERPOSITION_KERNELS_H
#define SUPERPOSITION_KERNELS_H

#include <cuda_runtime.h>
#include <vector>

// Template-based superposition kernel declarations
template<int RADIUS>
__global__ void kernelSuperposition(
    float const* __restrict__ inDose, 
    float const* __restrict__ inRSigmaEff, 
    float* const outDose, 
    const int inDosePitch, 
    const int rayDimsX,
    const int rayDimsY);

// Primary dose transformation kernel
__global__ void primTransfDivKernel(
    float* const result, 
    const int3 startIdx, 
    const int maxZ, 
    const uint3 doseDims,
    cudaTextureObject_t bevPrimDoseTex);

// Superposition parameters
struct SuperpositionParams {
    int maxRadius;
    int tileSize;
    int batchSize;
    float gaussianThreshold;
    
    SuperpositionParams() : maxRadius(32), tileSize(256), batchSize(1), gaussianThreshold(1e6f) {}
};

// Helper functions for superposition
void launchSuperpositionKernels(float* devRayIdd, float* devRayRSigmaEff, float* devBevPrimDose,
                                int rayDimsX, int rayDimsY, const SuperpositionParams& params);

#endif // SUPERPOSITION_KERNELS_H
