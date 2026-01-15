/**
 * \file
 * \brief Unified Convolution Algorithm Headers
 * 
 * This file provides unified headers for all convolution-related algorithms
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../core/common.cuh"
#include "../core/Macro.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Kernel Declarations
// ============================================================================

// Subspot data reading kernel
__global__ void readSubspotDataKernel(
    cudaTextureObject_t subspotData,
    SubspotInfo* subspotInfoArray,
    int nsubspot,
    int layerIdx,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
);

// Subspot range calculation kernel
__global__ void calculateSubspotRangeKernel(
    const SubspotInfo* subspotInfoArray,
    int nsubspot,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    int* subspotRanges
);

// Subspot to CPB convolution kernel
__global__ void subspotToCPBConvolutionOptimizedKernel(
    const SubspotInfo* subspotInfoArray,
    const int* subspotRanges,
    int nsubspot,
    int layerIdx,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    float* cpbWeights
);

// GPU 2D convolution kernel
__global__ void gpuConvolution2DKernel(
    float* input,
    float* output,
    int width,
    int height,
    float* kernel,
    int kernelSize,
    int padding
);

// CPB to ray weight mapping kernel
__global__ void mapCPBWeightsToRayWeightsKernel(
    float* cpbWeights,
    vec3i cpbDims,
    vec3f cpbCorner,
    vec3f cpbResolution,
    float* rayWeights,
    vec3i rayDims,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
);

// ============================================================================
// Function Declarations
// ============================================================================

// Subspot to CPB Convolution
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
);

// GPU 2D Convolution
void performGPUConvolution2D(
    float* input,
    float* output,
    int width,
    int height,
    float* kernel,
    int kernelSize,
    int padding
);

// CPB to Ray Weight Mapping
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
);

#endif // CONVOLUTION_H
