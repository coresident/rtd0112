/**
 * \file
 * \brief Superposition Kernels Implementation for RayTraceDicom Integration
 */

#include "raytracedicom_integration.h"
#include "superposition_kernels.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

// Template-based superposition kernel implementation
template<int RADIUS>
__global__ void kernelSuperposition(
    float const* __restrict__ inDose, 
    float const* __restrict__ inRSigmaEff, 
    float* const outDose, 
    const int inDosePitch, 
    const int rayDimsX,
    const int rayDimsY) {
    
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int x = tid % rayDimsX;
    const unsigned int y = tid / rayDimsX;
    
    if (x < rayDimsX && y < rayDimsY) {
        float dose = inDose[y * rayDimsX + x];
        float rSigmaEff = inRSigmaEff[y * rayDimsX + x];
        
        if (rSigmaEff < 1e6f) { // Not infinity
            float sigma = 1.0f / (rSigmaEff * 1.41421356f); // sqrt(2)
            
            // Apply Gaussian convolution with radius RADIUS
            float totalWeight = 0.0f;
            float weightedDose = 0.0f;
            
            for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < rayDimsX && ny >= 0 && ny < rayDimsY) {
                        float distSq = dx * dx + dy * dy;
                        if (distSq <= RADIUS * RADIUS) {
                            float gaussianWeight = expf(-0.5f * distSq / (sigma * sigma));
                            weightedDose += gaussianWeight * dose;
                            totalWeight += gaussianWeight;
                        }
                    }
                }
            }
            
            if (totalWeight > 0.0f) {
                dose = weightedDose / totalWeight;
            }
        }
        
        atomicAdd(&outDose[y * rayDimsX + x], dose);
    }
}

// Primary dose transformation kernel implementation
__global__ void primTransfDivKernel(
    float* const result, 
    const int3 startIdx, 
    const int maxZ, 
    const uint3 doseDims,
    cudaTextureObject_t bevPrimDoseTex) {
    
    unsigned int x = startIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y) {
        float *res = result + startIdx.z * doseDims.x * doseDims.y + y * doseDims.x + x;
        for (int z = startIdx.z; z <= maxZ; ++z) {
            float3 pos = make_float3(x + HALF, y + HALF, z + HALF);
            float tmp = tex3D<float>(bevPrimDoseTex, pos.x, pos.y, pos.z);
            if (tmp > 0.0f) {
                *res += tmp;
            }
            res += doseDims.x * doseDims.y;
        }
    }
}

// Explicit template instantiations for common radii
template __global__ void kernelSuperposition<0>(float const* __restrict__, float const* __restrict__, 
                                               float* const, const int, const int, const int);
template __global__ void kernelSuperposition<1>(float const* __restrict__, float const* __restrict__, 
                                               float* const, const int, const int, const int);
template __global__ void kernelSuperposition<2>(float const* __restrict__, float const* __restrict__, 
                                               float* const, const int, const int, const int);
template __global__ void kernelSuperposition<4>(float const* __restrict__, float const* __restrict__, 
                                               float* const, const int, const int, const int);
template __global__ void kernelSuperposition<8>(float const* __restrict__, float const* __restrict__, 
                                               float* const, const int, const int, const int);
template __global__ void kernelSuperposition<16>(float const* __restrict__, float const* __restrict__, 
                                                float* const, const int, const int, const int);
template __global__ void kernelSuperposition<32>(float const* __restrict__, float const* __restrict__, 
                                                float* const, const int, const int, const int);

// Helper function to launch superposition kernels
void launchSuperpositionKernels(float* devRayIdd, float* devRayRSigmaEff, float* devBevPrimDose,
                                int rayDimsX, int rayDimsY, const SuperpositionParams& params) {
    dim3 superpBlock(params.tileSize);
    dim3 superpGrid((rayDimsX * rayDimsY + superpBlock.x - 1) / superpBlock.x, 1);
    
    // Launch superposition kernels for different radii
    if (params.maxRadius >= 0) kernelSuperposition<0><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, rayDimsX, rayDimsY);
    if (params.maxRadius >= 1) kernelSuperposition<1><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, rayDimsX, rayDimsY);
    if (params.maxRadius >= 2) kernelSuperposition<2><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, rayDimsX, rayDimsY);
    if (params.maxRadius >= 4) kernelSuperposition<4><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, rayDimsX, rayDimsY);
    if (params.maxRadius >= 8) kernelSuperposition<8><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, rayDimsX, rayDimsY);
    if (params.maxRadius >= 16) kernelSuperposition<16><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, rayDimsX, rayDimsY);
    if (params.maxRadius >= 32) kernelSuperposition<32><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, rayDimsX, rayDimsY);
}
