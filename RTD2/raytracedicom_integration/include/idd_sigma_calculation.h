/**
 * \file
 * \brief IDD and Sigma Calculation Components for RayTraceDicom Integration
 * 
 * Includes IDD calculation, sigma computation, and related parameters
 */

#ifndef IDD_SIGMA_CALCULATION_H
#define IDD_SIGMA_CALCULATION_H

#include <cuda_runtime.h>
#include <vector>

// IDD and Sigma calculation parameters
struct FillIddAndSigmaParams {
    int energyIdx;
    float peakDepth;
    float scaleFact;
    float energyScaleFact;
    float rRlScale;
    float stepLength;
    float sigmaSqAirLin;
    float sigmaSqAirQuad;
    int firstStep;
    int afterLastStep;
    float spotDist;
    float entrySigmaSq;
    
    FillIddAndSigmaParams() : energyIdx(0), peakDepth(0), scaleFact(1.0f), energyScaleFact(1.0f),
                              rRlScale(1.0f), stepLength(1.0f), sigmaSqAirLin(0.0f), sigmaSqAirQuad(0.0f),
                              firstStep(0), afterLastStep(100), spotDist(1.0f), entrySigmaSq(0.0f) {}
    
    __device__ __host__ void initStepAndAirDiv() {}
    __device__ __host__ int getFirstStep() const { return firstStep; }
    __device__ __host__ int getAfterLastStep() const { return afterLastStep; }
    __device__ __host__ float getPeakDepth() const { return peakDepth; }
    __device__ __host__ float getEnergyScaleFact() const { return energyScaleFact; }
    __device__ __host__ float getRRlScale() const { return rRlScale; }
    __device__ __host__ float getStepLength() const { return stepLength; }
    __device__ __host__ float getSigmaSqAirLin() const { return sigmaSqAirLin; }
    __device__ __host__ float getSigmaSqAirQuad() const { return sigmaSqAirQuad; }
    __device__ __host__ float getSpotDist() const { return spotDist; }
    __device__ __host__ float getEntrySigmaSq() const { return entrySigmaSq; }
    
    __device__ __host__ float2 voxelWidth(int step) const {
        return make_float2(1.0f, 1.0f); // Simplified
    }
    
    __device__ __host__ float stepVol(int step) const {
        return 1.0f; // Simplified
    }
};

// IDD and Sigma calculation kernel declarations
__global__ void fillIddAndSigmaKernel(
    float* const bevDensity, 
    float* const bevCumulSp, 
    float* const bevIdd, 
    float* const bevRSigmaEff, 
    float* const rayWeights, 
    int* const firstInside, 
    int* const firstOutside, 
    int* const firstPassive, 
    const FillIddAndSigmaParams params,
    cudaTextureObject_t cumulIddTex, 
    cudaTextureObject_t rRadiationLengthTex);

// Helper functions for IDD and Sigma calculation
FillIddAndSigmaParams createIddParams(int energyIdx, float peakDepth, float scaleFact,
                                     float energyScaleFact, float rRlScale, float stepLength,
                                     int firstStep, int afterLastStep, float spotDist, float entrySigmaSq);

// Physical constants for IDD and Sigma calculation
namespace PhysicsConstants {
    const float pInv = 0.5649718f; // 1/p, p=1.77
    const float eCoef = 8.639415f; // (10*alpha)^(-1/p), alpha=2.2e-3
    const float sqrt2 = 1.41421356f; // sqrt(2.0f)
    const float eRefSq = 198.81f; // 14.1^2, E_s^2
    const float sigmaDelta = 0.21f;
    const float RAY_WEIGHT_CUTOFF = 1e-6f;
    const float BP_DEPTH_CUTOFF = 0.95f;
}

#endif // IDD_SIGMA_CALCULATION_H
