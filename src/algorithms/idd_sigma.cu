/**
 * \file
 * \brief Simplified IDD and Sigma Calculation Implementation
 * 
 * This file provides simplified implementations for IDD and sigma calculation
 */

#include "../include/algorithms/idd_sigma.h"
#include "../include/algorithms/fill_idd_and_sigma_params.cuh"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

// ============================================================================
// Simplified IDD and Sigma Calculation
// ============================================================================

// Correct fillIddAndSigma kernel following RayTracedicom algorithm
// Uses energyIdx to query IDD lookup table (not density!)
__global__ void fillIddAndSigmaKernel(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int* firstPassive,
    FillIddAndSigmaParams params,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex
) {
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int memStep = rayDimsY * rayDimsX;
    unsigned int idx = y * rayDimsX + x;
    
    if (x >= rayDimsX || y >= rayDimsY) return;
    
    bool beamLive = true;
    unsigned int afterLast = min(firstOutside[idx], static_cast<unsigned int>(params.getAfterLastStep()));
    const float rayWeight = rayWeights[idx];
    
    // Debug output removed for performance
    // if ((x < 3 && y < 3) || (x == 200 && y == 200)) {
    //     printf("fillIddAndSigma: ray(%d,%d) rayWeight=%.6f, ...\n", ...);
    // }
    
    if (rayWeight < 1e-6f || afterLast < params.getFirstStep()) {
        beamLive = false;
        afterLast = 0;
    }
    
    float cumulSp;
    float cumulSpOld = 0.0f;
    float cumulDose;
    float cumulDoseOld = 0.0f;
    
    // Initialize step and air division parameters
    params.initStepAndAirDiv();
    
    // Constants for sigma calculation
    const float pInv = 0.5649718f; // 1/p, p=1.77
    const float eCoef = 8.639415f; // (10*alpha)^(-1/p), alpha=2.2e-3
    const float sqrt2 = 1.41421356f; // sqrt(2.0f)
    const float eRefSq = 198.81f; // 14.1^2, E_s^2
    const float sigmaDelta = 0.21f;
    const float BP_DEPTH_CUTOFF = 1.5f;
    
    float incScat = 0.0f;
    float incincScat = 0.0f;
    float incDiv = params.getSigmaSqAirLin() + (2.0f * float(params.getFirstStep()) - 1.0f) * params.getSigmaSqAirQuad();
    float sigmaSq = -incDiv; // Compensate for first addition
    
    idx += params.getFirstStep() * memStep;
    
    for (unsigned int stepNo = params.getFirstStep(); stepNo < params.getAfterLastStep() && stepNo < afterLast; ++stepNo) {
        if (beamLive) {
            cumulSp = bevCumulSp[idx];
            
            // CRITICAL: Use energyIdx (not density!) to query IDD lookup table
            // Texture dimensions: width=nEnergySamples (depth), height=nEnergies (energy)
            // Matrix storage: row-major [energy][sample] = ciddMatrix[e * nEnergySamples + s]
            // Texture query: X-axis (width) = depth index, Y-axis (height) = energy index
            float depthIdx = cumulSp * params.getEnergyScaleFact() + HALF;
            float energyIdx = params.getEnergyIdx() + HALF;
            
            // Debug output removed for performance
            // if ((x < 2 && y < 2 && stepNo == params.getFirstStep()) || (x == 200 && y == 200 && stepNo == params.getFirstStep())) {
            //     printf("fillIddAndSigma: ray(%d,%d) step=%d, cumulSp=%.3f, ...\n", ...);
            // }
            
            cumulDose = tex2D<float>(cumulIddTex, depthIdx, energyIdx);
            
            float density = bevDensity[idx];
            
            // Debug output removed for performance
            // if ((x < 2 && y < 2 && stepNo == params.getFirstStep()) || (x == 200 && y == 200 && stepNo == params.getFirstStep())) {
            //     printf("fillIddAndSigma: ray(%d,%d) step=%d, cumulDose=%.6f, ...\n", ...);
            // }
            
            // Calculate sigma using Molière scattering theory
            if (cumulSp < params.getPeakDepth()) {
                float resE = eCoef * __powf(params.getPeakDepth() - HALF * (cumulSp + cumulSpOld), pInv);
                float betaP = resE + 938.3f - 938.3f * 938.3f / (resE + 938.3f);
                float rRl = density * tex1D<float>(rRadiationLengthTex, density * params.getRRlScale() + HALF);
                float thetaSq = eRefSq / (betaP * betaP) * params.getStepLength() * rRl;
                
                sigmaSq += incScat + incDiv;
                incincScat += 2.0f * thetaSq * params.getStepLength() * params.getStepLength();
                incScat += incincScat;
                incDiv += 2.0f * params.getSigmaSqAirQuad();
            } else {
                sigmaSq -= 1.5f * (incScat + incDiv) * density; // Empirical solution after BP
            }
            
            float rSigmaEff = HALF * (params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / 
                             (sqrt2 * (sqrtf(sigmaSq) + sigmaDelta));
            
            // Calculate dose increment
            float mass = density * params.stepVol(stepNo);
            if (mass > 1e-6f) {  // Use smaller threshold (1e-6 instead of 1e-2)
                float doseIncrement = rayWeight * (cumulDose - cumulDoseOld) / mass;
                bevIdd[idx] = doseIncrement;
            } else {
                bevIdd[idx] = 0.0f;
            }
            
            bevRSigmaEff[idx] = rSigmaEff;
            
            // Check if beam should stop
            if (cumulSp > params.getPeakDepth() * BP_DEPTH_CUTOFF || stepNo == afterLast) {
                beamLive = false;
                afterLast = stepNo;
            }
            
            cumulSpOld = cumulSp;
            cumulDoseOld = cumulDose;
        }
        
        idx += memStep;
    }
}

// Legacy simplified kernel (kept for compatibility, but should not be used)
__global__ void simpleIddCalculationKernel(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex
) {
    // This kernel is deprecated - use fillIddAndSigmaKernel instead
    // Kept only for backward compatibility
}

// ============================================================================
// Simplified Sigma Texture Calculation
// ============================================================================

// Simplified sigma texture calculation kernel
__global__ void simpleSigmaTextureKernel(
    cudaTextureObject_t subspotData,
    float* sigmaXTexture,
    float* sigmaYTexture,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    int layerIdx,
    int layerSize
) {
    int cpbIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCPBPoints = cpbDims.x * cpbDims.y;
    
    if (cpbIdx >= totalCPBPoints) return;
    
    int cpbX = cpbIdx % cpbDims.x;
    int cpbY = cpbIdx / cpbDims.x;
    
    // Calculate CPB grid point position
    vec3f cpbPos = vec3f(
        cpbCorner.x + (cpbX + 0.5f) * cpbResolution.x,
        cpbCorner.y + (cpbY + 0.5f) * cpbResolution.y,
        cpbCorner.z
    );
    
    float totalWeight = 0.0f;
    float weightedSigmaX = 0.0f;
    float weightedSigmaY = 0.0f;
    
    // Calculate weighted sigma from subspots
    for (int subspotIdx = 0; subspotIdx < layerSize; subspotIdx++) {
        float deltaX = tex3D<float>(subspotData, 0.0f, float(subspotIdx), float(layerIdx));
        float deltaY = tex3D<float>(subspotData, 1.0f, float(subspotIdx), float(layerIdx));
        float weight = tex3D<float>(subspotData, 2.0f, float(subspotIdx), float(layerIdx));
        float sigmaX = tex3D<float>(subspotData, 3.0f, float(subspotIdx), float(layerIdx));
        float sigmaY = tex3D<float>(subspotData, 4.0f, float(subspotIdx), float(layerIdx));
        
        if (weight < 0.001f) continue; // Simple cutoff
        
        // Simple distance-based weighting
        float dx = cpbPos.x - deltaX;
        float dy = cpbPos.y - deltaY;
        float distance = sqrtf(dx * dx + dy * dy);
        float distanceWeight = expf(-distance * distance / 100.0f); // Simplified weighting
        
        float totalSubspotWeight = weight * distanceWeight;
        totalWeight += totalSubspotWeight;
        weightedSigmaX += totalSubspotWeight * sigmaX;
        weightedSigmaY += totalSubspotWeight * sigmaY;
    }
    
    // Calculate weighted average sigma
    if (totalWeight > 0.001f) {
        sigmaXTexture[cpbIdx] = weightedSigmaX / totalWeight;
        sigmaYTexture[cpbIdx] = weightedSigmaY / totalWeight;
    } else {
        sigmaXTexture[cpbIdx] = 0.0f;
        sigmaYTexture[cpbIdx] = 0.0f;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

// Simplified IDD and sigma calculation
void performIddAndSigmaCalculation(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int* firstPassive,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex
) {
    GPU_TIMER_START();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((rayDimsX + blockSize.x - 1) / blockSize.x,
                  (rayDimsY + blockSize.y - 1) / blockSize.y);
    
    simpleIddCalculationKernel<<<gridSize, blockSize>>>(
        bevDensity, bevCumulSp, bevIdd, bevRSigmaEff, rayWeights,
        firstInside, firstOutside, rayDimsX, rayDimsY, steps,
        cumulIddTex, rRadiationLengthTex
    );
    checkCudaErrors(cudaDeviceSynchronize());
    
    GPU_TIMER_END("Simplified IDD and Sigma Calculation");
}

// Simplified sigma texture calculation
void performSigmaTextureCalculation(
    cudaTextureObject_t subspotData,
    float* sigmaXTexture,
    float* sigmaYTexture,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ,
    int numLayers,
    int maxSubspotsPerLayer
) {
    GPU_TIMER_START();
    
    dim3 blockSize(256);
    int totalCPBPoints = cpbDims.x * cpbDims.y;
    dim3 gridSize((totalCPBPoints + blockSize.x - 1) / blockSize.x);
    
    // Calculate sigma texture for each energy layer
    for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        simpleSigmaTextureKernel<<<gridSize, blockSize>>>(
            subspotData, sigmaXTexture, sigmaYTexture,
            cpbCorner, cpbResolution, cpbDims,
            layerIdx, maxSubspotsPerLayer
        );
        checkCudaErrors(cudaDeviceSynchronize());
    }
    
    GPU_TIMER_END("Simplified Sigma Texture Calculation");
}