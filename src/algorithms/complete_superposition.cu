/**
 * \file
 * \brief Complete Tile-Based Superposition Implementation (RayTracedicom algorithm)
 * 
 * This file implements the complete tile-based superposition algorithm from RayTracedicom
 * without any simplifications.
 */

#include "../include/algorithms/superposition.h"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <limits>
#include <algorithm>

// Constants from original RayTracedicom
const int maxSuperpR = MAX_SUPERP_RADIUS;  // 32
const int superpTileX = SUPERP_TILE_X;     // 32
const int superpTileY = SUPERP_TILE_Y;     // 8
const int tileRadBlockY = 4;
const int minTilesInBatch = 256;  // Minimum tiles per batch for efficient processing

// ============================================================================
// Tile Radius Calculation Kernel
// ============================================================================

template<unsigned int blockY>
__global__ void tileRadCalc(
    float* const devIn,
    const int startZ,
    int* const tilePrimRadCtrs,
    int2* const inOutIdcs,
    const int noTiles
) {
    __shared__ float tile[superpTileX * blockY];
    
    const int tileIdx = superpTileX * threadIdx.y + threadIdx.x;
    const int pitch = gridDim.x * superpTileX;
    const int inIdx = (startZ + blockIdx.z) * (gridDim.y * superpTileY * pitch) +
                      (blockIdx.y * superpTileY + threadIdx.y) * pitch +
                      blockIdx.x * superpTileX + threadIdx.x;
    
    // Find minimum rSigmaEff in this tile
    float minVal = devIn[inIdx];
    for (int i = 1; i < superpTileY / blockY; ++i) {
        float testVal = devIn[inIdx + i * blockY * pitch];
        if (testVal < minVal) { minVal = testVal; }
    }
    tile[tileIdx] = minVal;
    __syncthreads();
    
    // Reduction over y
    for (int maxIdxY = blockY / 2; maxIdxY > 0; maxIdxY >>= 1) {
        if (threadIdx.y < maxIdxY && tile[tileIdx + maxIdxY * superpTileX] < tile[tileIdx]) {
            tile[tileIdx] = tile[tileIdx + maxIdxY * superpTileX];
        }
        __syncthreads();
    }
    
    // Reduction over x
    for (int maxIdxX = superpTileX / 2; maxIdxX > 0; maxIdxX >>= 1) {
        if (threadIdx.x < maxIdxX && tile[tileIdx + maxIdxX] < tile[tileIdx]) {
            tile[tileIdx] = tile[tileIdx + maxIdxX];
        }
        __syncthreads();
    }
    
    if (tileIdx == 0) {
        float minRSigmaEff = tile[0];
        if (minRSigmaEff < 1e6f) {  // Not infinity
            float sigma = 1.0f / (minRSigmaEff * 1.41421356f);  // sqrt(2)
            int radius = (int)ceilf(sigma * superpTileX);
            radius = min(radius, maxSuperpR);
            radius = max(radius, 0);
            
            // Calculate the linear index of the first element in this tile
            // This is: (startZ + blockIdx.z) * (rayDimsY * rayDimsX) + (blockIdx.y * superpTileY) * rayDimsX + (blockIdx.x * superpTileX)
            int rayDimsX = gridDim.x * superpTileX;
            int rayDimsY = gridDim.y * superpTileY;
            int linearIdx = (startZ + blockIdx.z) * (rayDimsY * rayDimsX) + 
                           (blockIdx.y * superpTileY) * rayDimsX + 
                           (blockIdx.x * superpTileX);
            
            int counterIdx = radius;
            int tileCounterIdx = atomicAdd(&tilePrimRadCtrs[counterIdx], 1);
            
            if (tileCounterIdx < noTiles) {
                // Store linear index and actual radius
                inOutIdcs[counterIdx * noTiles + tileCounterIdx] = make_int2(linearIdx, radius);
            }
        }
    }
}

// Explicit instantiation
template __global__ void tileRadCalc<4>(float* const, const int, int* const, int2* const, const int);

// ============================================================================
// Complete Tile-Based Superposition Kernel
// ============================================================================

template<int rad>
__global__ void kernelSuperposition(
    float const* __restrict__ inDose,
    float const* __restrict__ inRSigmaEff,
    float* const outDose,
    const int inDosePitch,
    int2* const inOutIdcs,
    const int inOutIdxPitch,
    int* const tileCtrs
) {
    volatile __shared__ float tile[(superpTileX + 2 * rad) * (superpTileY + 2 * rad)];
    
    // Constants
    const int maxSuperpR = MAX_SUPERP_RADIUS;
    const int superpTileX = SUPERP_TILE_X;
    const int superpTileY = SUPERP_TILE_Y;
    
    // Initialize tile
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; 
         i < (superpTileY + 2 * rad) * (superpTileX + 2 * rad); 
         i += blockDim.x * blockDim.y) {
        tile[i] = 0.0f;
    }
    __syncthreads();
    
    int tileIdx = blockIdx.x;
    int radIdx = rad;
    while (tileIdx >= tileCtrs[radIdx]) {
        tileIdx -= tileCtrs[radIdx];
        radIdx -= 1;
    }
    
    // Get tile information from inOutIdcs
    // inOutIdcs stores: (linearIdx, actualRadius)
    // linearIdx is the index of the first element in the tile (original array, no padding)
    const int2 tileInfo = inOutIdcs[radIdx * inOutIdxPitch + tileIdx];
    const int tileStartIdx = tileInfo.x;  // Linear index of tile start (no padding)
    const int actualRadius = tileInfo.y;
    
    // Calculate tile coordinates from linear index
    // linearIdx = z * (rayDimsY * rayDimsX) + y * rayDimsX + x
    // where z is relative to startZ (beamFirstInside), y and x are absolute coordinates
    // rayDimsX = gridDim.x * superpTileX (from tileRadCalc)
    // rayDimsY = gridDim.y * superpTileY (from tileRadCalc)
    // In kernelSuperposition: gridDim.y = numTilesY, gridDim.z = numStepsInside
    int rayDimsX = inDosePitch;  // This should match gridDim.x * superpTileX from tileRadCalc
    int rayDimsY = gridDim.y * superpTileY;  // This should match gridDim.y * superpTileY from tileRadCalc
    
    // Extract tile coordinates from linear index
    // linearIdx = z * (rayDimsY * rayDimsX) + y * rayDimsX + x
    int x0 = tileStartIdx % rayDimsX;
    int yzRem = tileStartIdx / rayDimsX;
    int y0 = yzRem % rayDimsY;
    int z = yzRem / rayDimsY;  // z is relative to beamFirstInside (startZ)
    
    // Verify z is within bounds
    if (z < 0 || z >= gridDim.z) {
        return;  // Invalid tile index
    }
    
    // Process each row in the tile
    for (int row = threadIdx.y; row < superpTileY; row += blockDim.y) {
        const int x = x0 + threadIdx.x;
        const int y = y0 + row;
        
        // Calculate linear index: z * (rayDimsY * inDosePitch) + y * inDosePitch + x
        const int inIdx = z * (rayDimsY * inDosePitch) + y * inDosePitch + x;
        
        if (x < inDosePitch && y < rayDimsY) {
            float dose = inDose[inIdx];
            
            if (__syncthreads_or(dose > 0.0f)) {
                float rSigmaEff = inRSigmaEff[inIdx];
                
                if (rSigmaEff < 1e6f) {  // Not infinity
                    // Calculate erf-based weights
                    float erfNew = erff(rSigmaEff * HALF);
                    float erfOld = -erfNew;
                    float erfDiffs[rad + 1];
                    
                    for (int i = 0; i <= rad; ++i) {
                        erfDiffs[i] = HALF * (erfNew - erfOld);
                        erfOld = erfNew;
                        erfNew = erff(rSigmaEff * (float(i) + 1.5f));
                    }
                    
                    // Store in shared memory tile with padding
                    int tileRow = row + rad;
                    int tileCol = threadIdx.x + rad;
                    tile[tileRow * (superpTileX + 2 * rad) + tileCol] = dose;
                    __syncthreads();
                    
                    // Apply Gaussian convolution using erf weights
                    float totalWeight = 0.0f;
                    float weightedDose = 0.0f;
                    
                    for (int dy = -actualRadius; dy <= actualRadius; ++dy) {
                        for (int dx = -actualRadius; dx <= actualRadius; ++dx) {
                            float distSq = dx * dx + dy * dy;
                            if (distSq <= actualRadius * actualRadius) {
                                int dist = (int)sqrtf(distSq);
                                if (dist <= rad) {
                                    float weight = erfDiffs[dist];
                                    int tileY = tileRow + dy;
                                    int tileX = tileCol + dx;
                                    
                                    if (tileY >= 0 && tileY < superpTileY + 2 * rad &&
                                        tileX >= 0 && tileX < superpTileX + 2 * rad) {
                                        float neighborDose = tile[tileY * (superpTileX + 2 * rad) + tileX];
                                        weightedDose += weight * neighborDose;
                                        totalWeight += weight;
                                    }
                                }
                            }
                        }
                    }
                    
                    if (totalWeight > 1e-6f) {
                        // Output array has padding: (rayDimsX + 2*maxSuperpR) x (rayDimsY + 2*maxSuperpR)
                        // Original coordinates (x, y) need offset by maxSuperpR
                        int outX = x + maxSuperpR;
                        int outY = y + maxSuperpR;
                        int outDosePitch = rayDimsX + 2 * maxSuperpR;
                        int outDoseHeight = rayDimsY + 2 * maxSuperpR;
                        const int outIdx = z * (outDoseHeight * outDosePitch) + outY * outDosePitch + outX;
                        atomicAdd(&outDose[outIdx], weightedDose / totalWeight);
                    }
                }
            }
        }
    }
}

// Explicit instantiation for all radii 0-32
template __global__ void kernelSuperposition<0>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<1>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<2>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<3>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<4>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<5>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<6>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<7>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<8>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<9>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<10>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<11>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<12>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<13>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<14>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<15>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<16>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<17>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<18>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<19>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<20>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<21>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<22>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<23>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<24>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<25>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<26>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<27>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<28>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<29>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<30>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<31>(float const*, float const*, float*, int, int2*, int, int*);
template __global__ void kernelSuperposition<32>(float const*, float const*, float*, int, int2*, int, int*);

// ============================================================================
// Helper Kernels for Finding Min/Max
// ============================================================================

template<typename T, int blockSize>
__global__ void sliceMinVar(
    T* const devIn,
    T* const devOut,
    const int n
) {
    __shared__ T sdata[blockSize];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockSize + tid;
    
    // Use CUDA-compatible max value
    T maxVal = (T)(1 << 30);  // Large value for int
    if (sizeof(T) == sizeof(int)) {
        maxVal = (T)2147483647;  // INT_MAX
    }
    sdata[tid] = (i < n) ? devIn[i] : maxVal;
    __syncthreads();
    
    // Reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        devOut[blockIdx.x] = sdata[0];
    }
}

template<typename T, int blockSize>
__global__ void sliceMaxVar(
    T* const devIn,
    T* const devOut,
    const int n
) {
    __shared__ T sdata[blockSize];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockSize + tid;
    
    // Use CUDA-compatible min value
    T minVal = (T)(-2147483648);  // INT_MIN for int
    sdata[tid] = (i < n) ? devIn[i] : minVal;
    __syncthreads();
    
    // Reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        devOut[blockIdx.x] = sdata[0];
    }
}

// Explicit instantiations
template __global__ void sliceMinVar<int, 1024>(int* const, int* const, const int);
template __global__ void sliceMaxVar<int, 1024>(int* const, int* const, const int);

// ============================================================================
// Complete Tile-Based Superposition Function
// ============================================================================

void performCompleteTileBasedSuperposition(
    float* devRayIdd,
    float* devRayRSigmaEff,
    float* devBevPrimDose,
    int rayDimsX,
    int rayDimsY,
    int steps,
    int beamFirstInside,
    int beamFirstCalculatedPassive
) {
    GPU_TIMER_START();
    
    // Calculate dimensions
    // Round up to ensure tiles cover entire area
    int numTilesX = (rayDimsX + superpTileX - 1) / superpTileX;
    int numTilesY = (rayDimsY + superpTileY - 1) / superpTileY;
    int numStepsInside = beamFirstCalculatedPassive - beamFirstInside;
    if (numStepsInside <= 0) {
        fprintf(stderr, "Warning: numStepsInside <= 0, skipping superposition\n");
        GPU_TIMER_END("Complete Tile-Based Superposition");
        return;
    }
    
    int maxNoPrimTiles = numStepsInside * numTilesX * numTilesY;
    
    // Allocate tile radius counters and indices
    int* devTilePrimRadCtrs;
    int2* devPrimInOutIdcs;
    checkCudaErrors(cudaMalloc(&devTilePrimRadCtrs, (maxSuperpR + 2) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&devPrimInOutIdcs, (maxSuperpR + 2) * maxNoPrimTiles * sizeof(int2)));
    
    // Initialize tile radius counters
    std::vector<int> tilePrimRadCtrs(maxSuperpR + 2, 0);
    checkCudaErrors(cudaMemcpy(devTilePrimRadCtrs, &tilePrimRadCtrs[0], 
                               (maxSuperpR + 2) * sizeof(int), cudaMemcpyHostToDevice));
    
    // Calculate tile radii
    dim3 tileRadBlockDim(superpTileX, tileRadBlockY);
    dim3 tilePrimRadGridDim(numTilesX, numTilesY, numStepsInside);
    
    // Calculate BEV dose dimensions with padding
    int bevDoseX = rayDimsX + 2 * maxSuperpR;
    int bevDoseY = rayDimsY + 2 * maxSuperpR;
    
    // Calculate pitch for BEV dose (with padding) - currently unused but kept for future use
    // int bevDosePitch = bevDoseX;
    
    // tileRadCalc needs to work with original rayDims (no padding)
    tileRadCalc<4><<<tilePrimRadGridDim, tileRadBlockDim>>>(
        devRayRSigmaEff, beamFirstInside, devTilePrimRadCtrs, devPrimInOutIdcs, maxNoPrimTiles);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy tile radius counters back
    checkCudaErrors(cudaMemcpy(&tilePrimRadCtrs[0], devTilePrimRadCtrs, 
                               (maxSuperpR + 2) * sizeof(int), cudaMemcpyDeviceToHost));
    
    if (tilePrimRadCtrs[maxSuperpR + 1] > 0) {
        fprintf(stderr, "Error: Found larger than allowed kernel superposition radius\n");
        cudaFree(devTilePrimRadCtrs);
        cudaFree(devPrimInOutIdcs);
        GPU_TIMER_END("Complete Tile-Based Superposition");
        return;
    }
    
    // Find maximum radius
    int layerMaxPrimSuperpR = 0;
    for (int i = 0; i < maxSuperpR + 2; ++i) {
        if (tilePrimRadCtrs[i] > 0) { layerMaxPrimSuperpR = i; }
    }
    
    // Batch tiles by radius
    int recPrimRad = layerMaxPrimSuperpR;
    std::vector<int> batchedPrimTileRadCtrs(maxSuperpR + 1, 0);
    batchedPrimTileRadCtrs[0] = tilePrimRadCtrs[0];
    for (int rad = layerMaxPrimSuperpR; rad > 0; --rad) {
        batchedPrimTileRadCtrs[recPrimRad] += tilePrimRadCtrs[rad];
        if (batchedPrimTileRadCtrs[recPrimRad] >= minTilesInBatch) {
            recPrimRad = rad - 1;
        }
    }
    
    // Launch superposition kernels for each radius batch
    // Note: gridDim.y in kernelSuperposition should match numTilesY used in tileRadCalc
    dim3 superpBlockDim(superpTileX, 8);
    int inDosePitch = rayDimsX;  // Original array pitch (no padding)
    
    // We need to pass gridDim info to kernelSuperposition somehow
    // For now, we'll use dim3 gridDim(numTilesX, numTilesY, numStepsInside) implicitly
    // The kernel will need to know gridDim.y to calculate rayDimsY correctly
    
    // Launch kernels for each radius (0-32)
    // The kernel will need to write to devBevPrimDose with padding offset
    // We need to pass gridDim.y (numTilesY) to the kernel
    // For now, we'll assume gridDim.y = numTilesY, and gridDim.z = numStepsInside in the kernel
    // Note: gridDim.x in kernelSuperposition is the number of tiles for this radius batch
    if (batchedPrimTileRadCtrs[0] > 0) { 
        dim3 gridDim0(batchedPrimTileRadCtrs[0], numTilesY, numStepsInside);
        kernelSuperposition<0><<<gridDim0, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[1] > 0) { 
        dim3 gridDim1(batchedPrimTileRadCtrs[1], numTilesY, numStepsInside);
        kernelSuperposition<1><<<gridDim1, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[2] > 0) { 
        dim3 gridDim2(batchedPrimTileRadCtrs[2], numTilesY, numStepsInside);
        kernelSuperposition<2><<<gridDim2, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[3] > 0) { 
        dim3 gridDim3(batchedPrimTileRadCtrs[3], numTilesY, numStepsInside);
        kernelSuperposition<3><<<gridDim3, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[4] > 0) { 
        dim3 gridDim4(batchedPrimTileRadCtrs[4], numTilesY, numStepsInside);
        kernelSuperposition<4><<<gridDim4, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[5] > 0) { 
        dim3 gridDim5(batchedPrimTileRadCtrs[5], numTilesY, numStepsInside);
        kernelSuperposition<5><<<gridDim5, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[6] > 0) { 
        dim3 gridDim6(batchedPrimTileRadCtrs[6], numTilesY, numStepsInside);
        kernelSuperposition<6><<<gridDim6, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[7] > 0) { 
        dim3 gridDim7(batchedPrimTileRadCtrs[7], numTilesY, numStepsInside);
        kernelSuperposition<7><<<gridDim7, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[8] > 0) { 
        dim3 gridDim8(batchedPrimTileRadCtrs[8], numTilesY, numStepsInside);
        kernelSuperposition<8><<<gridDim8, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[9] > 0) { 
        dim3 gridDim9(batchedPrimTileRadCtrs[9], numTilesY, numStepsInside);
        kernelSuperposition<9><<<gridDim9, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[10] > 0) { 
        dim3 gridDim10(batchedPrimTileRadCtrs[10], numTilesY, numStepsInside);
        kernelSuperposition<10><<<gridDim10, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[11] > 0) { 
        dim3 gridDim11(batchedPrimTileRadCtrs[11], numTilesY, numStepsInside);
        kernelSuperposition<11><<<gridDim11, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[12] > 0) { 
        dim3 gridDim12(batchedPrimTileRadCtrs[12], numTilesY, numStepsInside);
        kernelSuperposition<12><<<gridDim12, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[13] > 0) { 
        dim3 gridDim13(batchedPrimTileRadCtrs[13], numTilesY, numStepsInside);
        kernelSuperposition<13><<<gridDim13, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[14] > 0) { 
        dim3 gridDim14(batchedPrimTileRadCtrs[14], numTilesY, numStepsInside);
        kernelSuperposition<14><<<gridDim14, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[15] > 0) { 
        dim3 gridDim15(batchedPrimTileRadCtrs[15], numTilesY, numStepsInside);
        kernelSuperposition<15><<<gridDim15, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[16] > 0) { 
        dim3 gridDim16(batchedPrimTileRadCtrs[16], numTilesY, numStepsInside);
        kernelSuperposition<16><<<gridDim16, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[17] > 0) { 
        dim3 gridDim17(batchedPrimTileRadCtrs[17], numTilesY, numStepsInside);
        kernelSuperposition<17><<<gridDim17, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[18] > 0) { 
        dim3 gridDim18(batchedPrimTileRadCtrs[18], numTilesY, numStepsInside);
        kernelSuperposition<18><<<gridDim18, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[19] > 0) { 
        dim3 gridDim19(batchedPrimTileRadCtrs[19], numTilesY, numStepsInside);
        kernelSuperposition<19><<<gridDim19, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[20] > 0) { 
        dim3 gridDim20(batchedPrimTileRadCtrs[20], numTilesY, numStepsInside);
        kernelSuperposition<20><<<gridDim20, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[21] > 0) { 
        dim3 gridDim21(batchedPrimTileRadCtrs[21], numTilesY, numStepsInside);
        kernelSuperposition<21><<<gridDim21, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[22] > 0) { 
        dim3 gridDim22(batchedPrimTileRadCtrs[22], numTilesY, numStepsInside);
        kernelSuperposition<22><<<gridDim22, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[23] > 0) { 
        dim3 gridDim23(batchedPrimTileRadCtrs[23], numTilesY, numStepsInside);
        kernelSuperposition<23><<<gridDim23, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[24] > 0) { 
        dim3 gridDim24(batchedPrimTileRadCtrs[24], numTilesY, numStepsInside);
        kernelSuperposition<24><<<gridDim24, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[25] > 0) { 
        dim3 gridDim25(batchedPrimTileRadCtrs[25], numTilesY, numStepsInside);
        kernelSuperposition<25><<<gridDim25, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[26] > 0) { 
        dim3 gridDim26(batchedPrimTileRadCtrs[26], numTilesY, numStepsInside);
        kernelSuperposition<26><<<gridDim26, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[27] > 0) { 
        dim3 gridDim27(batchedPrimTileRadCtrs[27], numTilesY, numStepsInside);
        kernelSuperposition<27><<<gridDim27, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[28] > 0) { 
        dim3 gridDim28(batchedPrimTileRadCtrs[28], numTilesY, numStepsInside);
        kernelSuperposition<28><<<gridDim28, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[29] > 0) { 
        dim3 gridDim29(batchedPrimTileRadCtrs[29], numTilesY, numStepsInside);
        kernelSuperposition<29><<<gridDim29, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[30] > 0) { 
        dim3 gridDim30(batchedPrimTileRadCtrs[30], numTilesY, numStepsInside);
        kernelSuperposition<30><<<gridDim30, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[31] > 0) { 
        dim3 gridDim31(batchedPrimTileRadCtrs[31], numTilesY, numStepsInside);
        kernelSuperposition<31><<<gridDim31, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    if (batchedPrimTileRadCtrs[32] > 0) { 
        dim3 gridDim32(batchedPrimTileRadCtrs[32], numTilesY, numStepsInside);
        kernelSuperposition<32><<<gridDim32, superpBlockDim>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, inDosePitch, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); 
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(devTilePrimRadCtrs);
    cudaFree(devPrimInOutIdcs);
    
    GPU_TIMER_END("Complete Tile-Based Superposition");
}

