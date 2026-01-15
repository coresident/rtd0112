/**
 * \file
 * \brief Unified Superposition Algorithm Headers
 * 
 * This file provides unified headers for all superposition-related algorithms
 */

#ifndef SUPERPOSITION_H
#define SUPERPOSITION_H

#include "../core/common.cuh"
#include "../core/Macro.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Function Declarations
// ============================================================================

// Enhanced Superposition Algorithm
void performEnhancedSuperposition(
    float* inDose,
    float* inRSigmaEff,
    float* outDose,
    int inDosePitch,
    int rayDimsX,
    int rayDimsY,
    int numLayers,
    int startZ
);

// Kernel Superposition Algorithm
void performKernelSuperposition(
    float* inDose,
    float* inRSigmaEff,
    float* outDose,
    int inDosePitch,
    int rayDimsX,
    int rayDimsY,
    int radius
);

// Complete Tile-Based Superposition Algorithm (RayTracedicom)
void performCompleteTileBasedSuperposition(
    float* devRayIdd,
    float* devRayRSigmaEff,
    float* devBevPrimDose,
    int rayDimsX,
    int rayDimsY,
    int steps,
    int beamFirstInside,
    int beamFirstCalculatedPassive
);

// Helper kernels for reduction
template<typename T, int blockSize>
__global__ void sliceMinVar(T* const devIn, T* const devOut, const int n);

template<typename T, int blockSize>
__global__ void sliceMaxVar(T* const devIn, T* const devOut, const int n);

#endif // SUPERPOSITION_H
