/**
 * \file
 * \brief IDD and Sigma Calculation Implementation for RayTraceDicom Integration
 */

#include "raytracedicom_integration.h"
#include "idd_sigma_calculation.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

// IDD and Sigma calculation kernel implementation
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
    cudaTextureObject_t rRadiationLengthTex) {
    
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y * blockDim.y * gridDim.x * blockDim.x;
    unsigned int idx = y * gridDim.x * blockDim.x + x;

    bool beamLive = true;
    const int firstIn = firstInside[idx];
    unsigned int afterLast = min(firstOutside[idx], static_cast<int>(params.getAfterLastStep()));
    const float rayWeight = rayWeights[idx];
    
    if (rayWeight < PhysicsConstants::RAY_WEIGHT_CUTOFF || afterLast < params.getFirstStep()) {
        beamLive = false;
        afterLast = 0;
    }

    float res = 0.0f;
    float rSigmaEff;
    float cumulSp;
    float cumulSpOld = 0.0f;
    float cumulDose;
    float cumulDoseOld = 0.0f;

    float incScat = 0.0f;
    float incincScat = 0.0f;
    float incDiv = params.getSigmaSqAirLin() + (2.0f*float(params.getFirstStep()) - 1.0f) * params.getSigmaSqAirQuad();
    float sigmaSq = -incDiv; // Compensate for first addition of incDiv

    idx += params.getFirstStep() * memStep;
    for (unsigned int stepNo = params.getFirstStep(); stepNo < params.getAfterLastStep(); ++stepNo) {
        if (beamLive) {
            cumulSp = bevCumulSp[idx];
            cumulDose = tex2D<float>(cumulIddTex, cumulSp * params.getEnergyScaleFact() + HALF, params.energyIdx + HALF);

            float density = bevDensity[idx];

            // Sigma peaks 1 - 2 mm before the BP
            if (cumulSp < params.getPeakDepth()) {
                float resE = PhysicsConstants::eCoef * __powf(params.getPeakDepth() - HALF*(cumulSp+cumulSpOld), PhysicsConstants::pInv);
                float betaP = resE + 938.3f - 938.3f*938.3f / (resE+938.3f);
                float rRl = density * tex1D<float>(rRadiationLengthTex, density * params.getRRlScale() + HALF);
                float thetaSq = PhysicsConstants::eRefSq/(betaP*betaP) * params.getStepLength() * rRl;

                sigmaSq += incScat + incDiv;
                incincScat += 2.0f * thetaSq * params.getStepLength() * params.getStepLength();
                incScat += incincScat;
                incDiv += 2.0f * params.getSigmaSqAirQuad();
            } else {
                sigmaSq -= 1.5f * (incScat + incDiv) * density; // Empirical solution to dip in sigma after BP
            }

            rSigmaEff = HALF*(params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / (PhysicsConstants::sqrt2 * (sqrtf(sigmaSq) + PhysicsConstants::sigmaDelta));
            
            if (cumulSp > params.getPeakDepth() * PhysicsConstants::BP_DEPTH_CUTOFF || stepNo == afterLast) {
                beamLive = false;
                afterLast = stepNo;
            }

            float mass = density * params.stepVol(stepNo);
            
            if (mass > 1e-2f) {
                res = rayWeight * (cumulDose-cumulDoseOld) / mass;
            }

            cumulSpOld = cumulSp;
            cumulDoseOld = cumulDose;
        }
        
        if (!beamLive || static_cast<int>(stepNo) < (firstIn-1)) {
            res = 0.0f;
            rSigmaEff = __int_as_float(0x7f800000); // inf, equals sigma = 0
        }
        
        bevIdd[idx] = res;
        bevRSigmaEff[idx] = rSigmaEff;
        idx += memStep;
    }
    firstPassive[y * gridDim.x * blockDim.x + x] = afterLast;
}

// Helper functions for IDD and Sigma calculation
FillIddAndSigmaParams createIddParams(int energyIdx, float peakDepth, float scaleFact,
                                     float energyScaleFact, float rRlScale, float stepLength,
                                     int firstStep, int afterLastStep, float spotDist, float entrySigmaSq) {
    FillIddAndSigmaParams params;
    params.energyIdx = energyIdx;
    params.peakDepth = peakDepth;
    params.scaleFact = scaleFact;
    params.energyScaleFact = energyScaleFact;
    params.rRlScale = rRlScale;
    params.stepLength = stepLength;
    params.firstStep = firstStep;
    params.afterLastStep = afterLastStep;
    params.spotDist = spotDist;
    params.entrySigmaSq = entrySigmaSq;
    return params;
}
