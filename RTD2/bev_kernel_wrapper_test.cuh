#ifndef BEV_KERNEL_WRAPPER_TEST_CUH
#define BEV_KERNEL_WRAPPER_TEST_CUH

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include "bev_transforms_test.cuh"

// Constants from RayTraceDicom
#define maxSuperpR 32
#define superpTileX 32
#define superpTileY 32
#define minTilesInBatch 100
#define BP_DEPTH_CUTOFF 0.8f
#define HALF 0.5f

// Helper function to round up to nearest multiple
template<typename T>
__host__ __device__ inline T roundTo(T val, T multiple) {
    return ((val + multiple - 1) / multiple) * multiple;
}

// Forward declarations for RayTraceDicom kernels
template<int R>
__global__ void kernelSuperposition(
    float* const idd, float* const rSigmaEff, float* const bevDose,
    const unsigned int rayDimsX, int* const inOutIdcs, const unsigned int maxNoTiles,
    int* const tileRadCtrs);

__global__ void tileRadCalc(
    float* const rSigmaEff, const int beamFirstInside, int* const tileRadCtrs,
    int* const inOutIdcs, const unsigned int maxNoTiles);

__global__ void sliceMaxVar(
    int* const input, int* const output, const unsigned int n);

// FillIddAndSigmaParams structure for kernel parameters
struct FillIddAndSigmaParams {
    int energyIdx;
    float energyScaleFact;
    float peakDepth;
    float entrySigmaSq;
    float rRlScaleFact;
    float spotDistInRays;
    int beamFirstInside;
    unsigned int afterLastStep;
    Float3ToBevTransform_test rayIdxToImIdx;
    
    __host__ __device__ FillIddAndSigmaParams() : energyIdx(0), energyScaleFact(1.0f), peakDepth(0.0f), 
        entrySigmaSq(0.0f), rRlScaleFact(1.0f), spotDistInRays(1.0f), beamFirstInside(0), afterLastStep(0) {}
    
    __host__ __device__ FillIddAndSigmaParams(int eIdx, float eScale, float pDepth, float eSigmaSq, 
        float rRlScale, float spotDist, int bFirstInside, unsigned int aLastStep, const Float3ToBevTransform_test& rayToIm)
        : energyIdx(eIdx), energyScaleFact(eScale), peakDepth(pDepth), entrySigmaSq(eSigmaSq),
          rRlScaleFact(rRlScale), spotDistInRays(spotDist), beamFirstInside(bFirstInside), 
          afterLastStep(aLastStep), rayIdxToImIdx(rayToIm) {}
};

// TransferParamStructDiv3 for BEV to dose space transformation
struct TransferParamStructDiv3 {
    float3_affine_test transform;
    
    __host__ __device__ TransferParamStructDiv3() {}
    __host__ __device__ TransferParamStructDiv3(const float3_affine_test& t) : transform(t) {}
};

// Forward declarations for transformation kernels
__global__ void primTransfDiv(
    float* const doseBox, const TransferParamStructDiv3 params, 
    const int3 minIdx, const int maxZ, const int3 doseDims
#if CUDART_VERSION >= 12000
    , cudaTextureObject_t bevDoseTex
#endif
);

__global__ void nucTransfDiv(
    float* const doseBox, const TransferParamStructDiv3 params, 
    const int3 minIdx, const int maxZ, const int3 doseDims
#if CUDART_VERSION >= 12000
    , cudaTextureObject_t bevDoseTex
#endif
);

// Enhanced beam settings structure
struct EnhancedBeamSettings {
    std::vector<float> energies;           // Energy for each layer
    std::vector<float2> spotSigmas;        // (sigmax, sigmay) at iso in air for each energy layer
    float2 raySpacing;                     // Spacing between adjacent raytracing rays at iso
    unsigned int steps;                     // Number of raytracing steps
    float2 sourceDist;                     // Source to iso distance in x and y
    float3 spotOffset;                     // Spot offset in gantry coordinates
    float3 spotDelta;                      // Spot spacing in gantry coordinates
    float3 gantryToImOffset;              // Gantry to image transform offset
    float3 gantryToImMatrix;              // Gantry to image transform matrix (3x3 linear part)
    float3 gantryToDoseOffset;            // Gantry to dose transform offset
    float3 gantryToDoseMatrix;            // Gantry to dose transform matrix (3x3 linear part)
    
    // Additional parameters for RayTraceDicom integration
    float3_affine_test spotIdxToGantry;   // Spot index to gantry transform
    float3IdxTransform_test gantryToDoseIdx; // Gantry to dose index transform
    bool enableNuclearCorrection;         // Enable nuclear correction
    bool enableFineTiming;                // Enable fine-grained timing
};

// Enhanced energy structure
struct EnhancedEnergyStruct {
    int nEnergySamples;                   // Number of energy bins
    int nEnergies;                        // Number of energies
    std::vector<float> energiesPerU;      // Energy per bin
    std::vector<float> peakDepths;        // Proton penetration depth per bin
    std::vector<float> scaleFacts;        // Scaling factor per bin
    std::vector<float> ciddMatrix;        // 2D matrix of cumulative integral dose
    
    int nDensitySamples;                  // Number of density bins
    float densityScaleFact;               // Density scaling factor
    std::vector<float> densityVector;     // Densities for each HU
    
    int nSpSamples;                       // Number of stopping power bins
    float spScaleFact;                    // Scaling factor for stopping power
    std::vector<float> spVector;          // Stopping power for each HU
    
    int nRRlSamples;                      // Number of radiation length bins
    float rRlScaleFact;                   // Radiation length scaling factor
    std::vector<float> rRlVector;         // Radiation length for each HU
    
    // Additional parameters for RayTraceDicom
    std::vector<float> entrySigmas;       // Entry sigma for each energy
    std::vector<int> energyIdcs;          // Energy indices for each layer
};

// Main wrapper function declaration
extern "C" void enhancedProtonsWrapper(
    float* imVolData, int3 imVolDims, float3 imVolSpacing, float3 imVolOrigin,
    float* doseVolData, int3 doseVolDims, float3 doseVolSpacing, float3 doseVolOrigin,
    const EnhancedBeamSettings* beamSettings, int numBeams,
    const EnhancedEnergyStruct* energyData,
    int gpuId = 0, bool enableNuclearCorrection = false, bool enableFineTiming = false
);

// Helper functions for creating and destroying enhanced structures
extern "C" EnhancedBeamSettings* createEnhancedBeamSettings(
    float* energies, int numEnergies,
    float* spotSigmas, int numSpotSigmas,
    float2 raySpacing, unsigned int steps, float2 sourceDist,
    float3 spotOffset, float3 spotDelta, float3 gantryToImOffset, float3 gantryToImMatrix,
    float3 gantryToDoseOffset, float3 gantryToDoseMatrix, bool enableNuclearCorrection, bool enableFineTiming
);

extern "C" void destroyEnhancedBeamSettings(EnhancedBeamSettings* beam);

extern "C" EnhancedEnergyStruct* createEnhancedEnergyStruct(
    int nEnergySamples, int nEnergies, float* energiesPerU, float* peakDepths, float* scaleFacts,
    float* ciddMatrix, int nDensitySamples, float densityScaleFact, float* densityVector,
    int nSpSamples, float spScaleFact, float* spVector, int nRRlSamples, float rRlScaleFact,
    float* rRlVector, float* entrySigmas, int* energyIdcs
);

extern "C" void destroyEnhancedEnergyStruct(EnhancedEnergyStruct* energy);

#endif // BEV_KERNEL_WRAPPER_TEST_CUH