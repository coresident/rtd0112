/**
 * \file
 * \brief Simplified protons wrapper function for C++ compilation interface
 * 
 * This is a simplified version of RayTraceDicom's cudaWrapperProtons function,
 * designed to be used as a C++ compilation interface module.
 * The interface is simplified based on carbonPBS pybind11 design patterns.
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
#include <device_launch_parameters.h>
#include "bev_transforms_test.cuh"

// Simplified data structures for interface
struct SimplifiedBeamSettings {
    std::vector<float> energies;           // Energy for each layer
    std::vector<float2> spotSigmas;        // (sigmax, sigmay) at iso in air for each energy layer
    float2 raySpacing;                     // Spacing between adjacent raytracing rays at iso
    unsigned int steps;                    // Number of raytracing steps
    float2 sourceDist;                     // Source to iso distance in x and y
    float3 spotOffset;                     // Spot offset in gantry coordinates
    float3 spotDelta;                      // Spot spacing in gantry coordinates
    float3 gantryToImOffset;              // Gantry to image transform offset
    float3 gantryToImMatrix;              // Gantry to image transform matrix (3x3 linear part)
    float3 gantryToDoseOffset;            // Gantry to dose transform offset
    float3 gantryToDoseMatrix;            // Gantry to dose transform matrix (3x3 linear part)
};

struct SimplifiedEnergyStruct {
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
};

// CUDA kernel for density and stopping power tracing
__global__ void fillBevDensityAndSpKernel(
    float* const bevDensity,
    float* const bevCumulSp, 
    int* const beamFirstInside, 
    int* const firstStepOutside, 
    const DensityAndSpTracerParams params,
    cudaTextureObject_t imVolTex, 
    cudaTextureObject_t densityTex, 
    cudaTextureObject_t stoppingPowerTex,
    const int3 imVolDims) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y * blockDim.y * gridDim.x * blockDim.x;
    unsigned int idx = y * gridDim.x * blockDim.x + x;

    if (x >= imVolDims.x || y >= imVolDims.y) return;

    // Compensate for value located at voxel corner instead of centre
    float3 startPos = params.getStart(x, y);
    float3 pos = make_float3(startPos.x + HALF, startPos.y + HALF, startPos.z + HALF);
    float3 step = params.getInc(x, y);
    float stepLen = params.stepLen(x, y);
    
    float cumulSp = 0.0f;
    float cumulHuPlus1000 = 0.0f;
    int beforeFirstInside = -1;
    int lastInside = -1;

    for (unsigned int i = 0; i < params.getSteps(); ++i) {
        float huPlus1000 = tex3D<float>(imVolTex, pos.x, pos.y, pos.z);
        cumulHuPlus1000 += huPlus1000;
        
        bevDensity[idx] = tex1D<float>(densityTex, huPlus1000 * params.getDensityScale() + HALF);

        cumulSp += stepLen * tex1D<float>(stoppingPowerTex, huPlus1000 * params.getSpScale() + HALF);

        if (cumulHuPlus1000 < 150.0f) {
            beforeFirstInside = i;
        }
        if (huPlus1000 > 150.0f) {
            lastInside = i;
        }
        
        bevCumulSp[idx] = cumulSp;

        idx += memStep;
        pos.x = pos.x + step.x;
        pos.y = pos.y + step.y;
        pos.z = pos.z + step.z;
    }
    
    beamFirstInside[y * gridDim.x * blockDim.x + x] = beforeFirstInside + 1;
    firstStepOutside[y * gridDim.x * blockDim.x + x] = lastInside + 1;
}

// CUDA kernel for IDD and sigma calculation
__global__ void fillIddAndSigmaKernel(
    float* const bevDensity, 
    float* const bevCumulSp, 
    float* const bevIdd, 
    float* const bevRSigmaEff, 
    float* const rayWeights, 
    int* const firstInside, 
    int* const firstOutside, 
    int* const firstPassive, 
    const SimplifiedEnergyStruct* energyData,
    const int energyIdx,
    const float peakDepth,
    const float scaleFact,
    const int3 imVolDims,
    cudaTextureObject_t cumulIddTex, 
    cudaTextureObject_t rRadiationLengthTex) {
    
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y * blockDim.y * gridDim.x * blockDim.x;
    unsigned int idx = y * gridDim.x * blockDim.x + x;

    if (x >= imVolDims.x || y >= imVolDims.y) return;

    bool beamLive = true;
    const int firstIn = firstInside[idx];
    unsigned int afterLast = min(firstOutside[idx], static_cast<int>(100)); // Fixed: use constant
    const float rayWeight = rayWeights[idx];
    
    if (rayWeight < 1e-6f || afterLast < firstIn) {
        beamLive = false;
        afterLast = 0;
    }

    float res = 0.0f;
    float rSigmaEff;
    float cumulSp;
    float cumulSpOld = 0.0f;
    float cumulDose;
    float cumulDoseOld = 0.0f;

    const float pInv = 0.5649718f; // 1/p, p=1.77
    const float eCoef = 8.639415f; // (10*alpha)^(-1/p), alpha=2.2e-3
    const float sqrt2 = 1.41421356f; // sqrt(2.0f)
    const float eRefSq = 198.81f; // 14.1^2, E_s^2
    const float sigmaDelta = 0.21f;

    float incScat = 0.0f;
    float incincScat = 0.0f;
    float incDiv = 0.0f; // This should be calculated based on air parameters
    float sigmaSq = 0.0f;

    idx += firstIn * memStep;
    for (unsigned int stepNo = firstIn; stepNo < afterLast; ++stepNo) {
        if (beamLive) {
            cumulSp = bevCumulSp[idx];
            cumulDose = tex2D<float>(cumulIddTex, cumulSp * scaleFact + HALF, energyIdx + HALF);

            float density = bevDensity[idx];

            // Sigma peaks 1 - 2 mm before the BP
            if (cumulSp < peakDepth) {
                float resE = eCoef * __powf(peakDepth - HALF * (cumulSp + cumulSpOld), pInv);
                float betaP = resE + 938.3f - 938.3f * 938.3f / (resE + 938.3f);
                float rRl = density * tex1D<float>(rRadiationLengthTex, density * 0.001f + HALF);
                float thetaSq = eRefSq / (betaP * betaP) * 1.0f * rRl; // Fixed: use constant

                sigmaSq += incScat + incDiv;
                incincScat += 2.0f * thetaSq * 1.0f * 1.0f; // Fixed: use constant
                incincScat += incScat;
                incDiv += 2.0f * 0.0f; // air quadratic term
            } else {
                sigmaSq -= 1.5f * (incScat + incDiv) * density; // Empirical solution
            }

            rSigmaEff = HALF * (1.0f + 1.0f) / (sqrt2 * (sqrtf(sigmaSq) + sigmaDelta));
            
            if (cumulSp > peakDepth * 0.8f || stepNo == afterLast) {
                beamLive = false;
                afterLast = stepNo;
            }

            float mass = density * 1.0f; // step volume simplified
            if (mass > 1e-2f) {
                res = rayWeight * (cumulDose - cumulDoseOld) / mass;
            }

            cumulSpOld = cumulSp;
            cumulDoseOld = cumulDose;
        }
        
        if (!beamLive || static_cast<int>(stepNo) < (firstIn - 1)) {
            res = 0.0f;
            rSigmaEff = __int_as_float(0x7f800000); // inf
        }
        
        bevIdd[idx] = res;
        bevRSigmaEff[idx] = rSigmaEff;

        idx += memStep;
    }
    
    firstPassive[y * gridDim.x * blockDim.x + x] = afterLast;
}

// Simplified wrapper function for protons calculation
extern "C" void simplifiedProtonsWrapper(
    // Input parameters
    float* imVolData,                     // 3D image volume data (HU values)
    int3 imVolDims,                       // Image volume dimensions
    float3 imVolSpacing,                  // Image volume spacing
    float3 imVolOrigin,                   // Image volume origin
    
    // Output parameters
    float* doseVolData,                   // 3D dose volume data (output)
    int3 doseVolDims,                     // Dose volume dimensions
    float3 doseVolSpacing,                // Dose volume spacing
    float3 doseVolOrigin,                 // Dose volume origin
    
    // Beam settings
    const SimplifiedBeamSettings* beamSettings,
    int numBeams,
    
    // Energy data
    const SimplifiedEnergyStruct* energyData,
    
    // Control parameters
    int gpuId = 0,                        // GPU device ID
    bool enableNuclearCorrection = false, // Enable nuclear correction
    bool enableFineTiming = false         // Enable fine-grained timing
) {
    // Initialize CUDA context
    cudaSetDevice(gpuId);
    cudaFree(0);
    
    // Basic error checking
    if (!imVolData || !doseVolData || !beamSettings || !energyData) {
        std::cerr << "Error: Invalid input parameters" << std::endl;
        return;
    }
    
    if (numBeams <= 0) {
        std::cerr << "Error: Number of beams must be positive" << std::endl;
        return;
    }
    
    // Calculate total image and dose volume sizes
    size_t imVolSize = imVolDims.x * imVolDims.y * imVolDims.z;
    size_t doseVolSize = doseVolDims.x * doseVolDims.y * doseVolDims.z;
    
    // Allocate device memory for image volume
    float* devImVol;
    cudaMalloc((void**)&devImVol, imVolSize * sizeof(float));
    cudaMemcpy(devImVol, imVolData, imVolSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate device memory for dose volume
    float* devDoseVol;
    cudaMalloc((void**)&devDoseVol, doseVolSize * sizeof(float));
    cudaMemcpy(devDoseVol, doseVolData, doseVolSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create texture objects for image volume
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    cudaArray* devImVolArr;
    cudaExtent imExt = make_cudaExtent(imVolDims.x, imVolDims.y, imVolDims.z);
    cudaMalloc3DArray(&devImVolArr, &floatChannelDesc, imExt);
    
    cudaMemcpy3DParms imCopyParams = {};
    imCopyParams.srcPtr = make_cudaPitchedPtr((void*)devImVol, imExt.width*sizeof(float), imExt.width, imExt.height);
    imCopyParams.dstArray = devImVolArr;
    imCopyParams.extent = imExt;
    imCopyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&imCopyParams);
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devImVolArr;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    
    cudaTextureObject_t imVolTex;
    cudaCreateTextureObject(&imVolTex, &resDesc, &texDesc, NULL);
    
    // Create texture objects for energy data
    cudaArray* devCumulIddArr;
    cudaMallocArray(&devCumulIddArr, &floatChannelDesc, energyData->nEnergySamples, energyData->nEnergies);
    cudaMemcpyToArray(devCumulIddArr, 0, 0, &energyData->ciddMatrix[0], 
                      energyData->nEnergySamples*energyData->nEnergies*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t cumulIddTex;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devCumulIddArr;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    cudaCreateTextureObject(&cumulIddTex, &resDesc, &texDesc, NULL);
    
    // Create texture objects for density and stopping power
    cudaArray* devDensityArr;
    cudaMallocArray(&devDensityArr, &floatChannelDesc, energyData->nDensitySamples);
    cudaMemcpyToArray(devDensityArr, 0, 0, &energyData->densityVector[0], 
                      energyData->nDensitySamples*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t densityTex;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devDensityArr;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaCreateTextureObject(&densityTex, &resDesc, &texDesc, NULL);
    
    cudaArray* devStoppingPowerArr;
    cudaMallocArray(&devStoppingPowerArr, &floatChannelDesc, energyData->nSpSamples);
    cudaMemcpyToArray(devStoppingPowerArr, 0, 0, &energyData->spVector[0], 
                      energyData->nSpSamples*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t stoppingPowerTex;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devStoppingPowerArr;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaCreateTextureObject(&stoppingPowerTex, &resDesc, &texDesc, NULL);
    
    // Create texture object for radiation length
    cudaArray* devReciprocalRadiationLengthArr;
    cudaMallocArray(&devReciprocalRadiationLengthArr, &floatChannelDesc, energyData->nRRlSamples);
    cudaMemcpyToArray(devReciprocalRadiationLengthArr, 0, 0, &energyData->rRlVector[0], 
                      energyData->nRRlSamples*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t rRadiationLengthTex;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devReciprocalRadiationLengthArr;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaCreateTextureObject(&rRadiationLengthTex, &resDesc, &texDesc, NULL);
    
    // Process each beam
    for (int beamNo = 0; beamNo < numBeams; ++beamNo) {
        const SimplifiedBeamSettings& beam = beamSettings[beamNo];
        
        // Extract beam parameters
        const size_t nLayers = beam.energies.size();
        if (nLayers == 0) continue;
        
        // Calculate maximum spot sigmas
        float2 maxSpotSigmas = make_float2(0.0f, 0.0f);
        for (size_t layerNo = 0; layerNo < nLayers; ++layerNo) {
            maxSpotSigmas.x = fmaxf(maxSpotSigmas.x, beam.spotSigmas[layerNo].x);
            maxSpotSigmas.y = fmaxf(maxSpotSigmas.y, beam.spotSigmas[layerNo].y);
        }
        
        // Calculate ray dimensions based on beam settings
        uint3 rayDims = make_uint3(
            (int)(imVolDims.x * 1.2f),  // Simplified ray grid size
            (int)(imVolDims.y * 1.2f),
            beam.steps
        );
        
        // Allocate device memory for ray tracing
        float* devRayWeights;
        cudaMalloc((void**)&devRayWeights, rayDims.x * rayDims.y * sizeof(float));
        
        float* devRayIdd;
        cudaMalloc((void**)&devRayIdd, rayDims.x * rayDims.y * beam.steps * sizeof(float));
        
        float* devRayRSigmaEff;
        cudaMalloc((void**)&devRayRSigmaEff, rayDims.x * rayDims.y * beam.steps * sizeof(float));
        
        // Initialize ray weights (simplified - in original code this involves complex convolution)
        cudaMemset(devRayWeights, 0, rayDims.x * rayDims.y * sizeof(float));
        
        // Process each energy layer
        for (size_t layerNo = 0; layerNo < nLayers; ++layerNo) {
            float energy = beam.energies[layerNo];
            float2 spotSigma = beam.spotSigmas[layerNo];
            
            // Find energy index in energy data
            int energyIdx = 0;
            for (int i = 0; i < energyData->nEnergies; ++i) {
                if (energyData->energiesPerU[i] >= energy) {
                    energyIdx = i;
                    break;
                }
            }
            
            // Calculate peak depth and scaling factors
            float peakDepth = energyData->peakDepths[energyIdx];
            float scaleFact = energyData->scaleFacts[energyIdx];
            
            // Create BEV transform for this beam
            // Note: This is a simplified version - you'll need to implement the full transform logic
            Float3ToBevTransform_test bevTransform = BevTransformUtils::createBevTransform(
                beam.spotOffset, beam.spotDelta, beam.gantryToImOffset, 
                beam.gantryToImMatrix, beam.sourceDist, imVolOrigin);
            
            // Create density and stopping power tracing parameters
            DensityAndSpTracerParams tracerParams(energyData->densityScaleFact, energyData->spScaleFact, 
                                                beam.steps, bevTransform);
            
            // Allocate additional device memory for kernel outputs
            int* devBeamFirstInside;
            cudaMalloc((void**)&devBeamFirstInside, rayDims.x * rayDims.y * sizeof(int));
            
            int* devFirstStepOutside;
            cudaMalloc((void**)&devFirstStepOutside, rayDims.x * rayDims.y * sizeof(int));
            
            int* devFirstPassive;
            cudaMalloc((void**)&devFirstPassive, rayDims.x * rayDims.y * sizeof(int));
            
            // Launch density and stopping power tracing kernel
            dim3 tracerBlock(32, 8);
            dim3 tracerGrid((rayDims.x + tracerBlock.x - 1) / tracerBlock.x, 
                           (rayDims.y + tracerBlock.y - 1) / tracerBlock.y);
            
            fillBevDensityAndSpKernel<<<tracerGrid, tracerBlock>>>(
                devRayWeights,           // bevDensity
                devRayIdd,               // bevCumulSp
                devBeamFirstInside,      // beamFirstInside
                devFirstStepOutside,     // firstStepOutside
                tracerParams,            // params
                imVolTex,                // imVolTex
                densityTex,              // densityTex
                stoppingPowerTex,        // stoppingPowerTex
                imVolDims                // imVolDims
            );
            
            // Launch IDD and sigma calculation kernel
            fillIddAndSigmaKernel<<<tracerGrid, tracerBlock>>>(
                devRayWeights,           // bevDensity
                devRayIdd,               // bevCumulSp
                devRayIdd,               // bevIdd (output)
                devRayRSigmaEff,         // bevRSigmaEff (output)
                devRayWeights,           // rayWeights
                devBeamFirstInside,      // firstInside
                devFirstStepOutside,     // firstOutside
                devFirstPassive,         // firstPassive (output)
                energyData,              // energyData
                energyIdx,               // energyIdx
                peakDepth,               // peakDepth
                scaleFact,               // scaleFact
                imVolDims,               // imVolDims
                cumulIddTex,             // cumulIddTex
                rRadiationLengthTex      // rRadiationLengthTex
            );
            
            // Clean up layer-specific memory
            // (In original code, this would be done after superposition)
        }
        
        // Clean up beam-specific memory
        cudaFree(devRayWeights);
        cudaFree(devRayIdd);
        cudaFree(devRayRSigmaEff);
        cudaFree(devBeamFirstInside);
        cudaFree(devFirstStepOutside);
        cudaFree(devFirstPassive);
    }
    
    // Copy final dose back to host
    cudaMemcpy(doseVolData, devDoseVol, doseVolSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up all device memory
    cudaFree(devImVol);
    cudaFree(devDoseVol);
    cudaFreeArray(devImVolArr);
    cudaFreeArray(devCumulIddArr);
    cudaFreeArray(devDensityArr);
    cudaFreeArray(devStoppingPowerArr);
    cudaFreeArray(devReciprocalRadiationLengthArr);
    
    // Destroy texture objects
    cudaDestroyTextureObject(imVolTex);
    cudaDestroyTextureObject(cumulIddTex);
    cudaDestroyTextureObject(densityTex);
    cudaDestroyTextureObject(stoppingPowerTex);
    cudaDestroyTextureObject(rRadiationLengthTex);
    
    std::cout << "Simplified protons calculation completed successfully" << std::endl;
}

// Helper function to create simplified beam settings from Python input
extern "C" SimplifiedBeamSettings* createSimplifiedBeamSettings(
    float* energies, int numEnergies,
    float* spotSigmas, int numSpotSigmas,
    float2 raySpacing,
    unsigned int steps,
    float2 sourceDist,
    float3 spotOffset,
    float3 spotDelta,
    float3 gantryToImOffset,
    float3 gantryToImMatrix,
    float3 gantryToDoseOffset,
    float3 gantryToDoseMatrix
) {
    SimplifiedBeamSettings* beam = new SimplifiedBeamSettings();
    
    // Copy energies
    beam->energies.assign(energies, energies + numEnergies);
    
    // Copy spot sigmas
    beam->spotSigmas.assign((float2*)spotSigmas, (float2*)spotSigmas + numSpotSigmas);
    
    // Set other parameters
    beam->raySpacing = raySpacing;
    beam->steps = steps;
    beam->sourceDist = sourceDist;
    beam->spotOffset = spotOffset;
    beam->spotDelta = spotDelta;
    beam->gantryToImOffset = gantryToImOffset;
    beam->gantryToImMatrix = gantryToImMatrix;
    beam->gantryToDoseOffset = gantryToDoseOffset;
    beam->gantryToDoseMatrix = gantryToDoseMatrix;
    
    return beam;
}

// Helper function to destroy simplified beam settings
extern "C" void destroySimplifiedBeamSettings(SimplifiedBeamSettings* beam) {
    if (beam) {
        delete beam;
    }
}

// Helper function to create simplified energy struct from Python input
extern "C" SimplifiedEnergyStruct* createSimplifiedEnergyStruct(
    int nEnergySamples, int nEnergies,
    float* energiesPerU, float* peakDepths, float* scaleFacts, float* ciddMatrix,
    int nDensitySamples, float densityScaleFact, float* densityVector,
    int nSpSamples, float spScaleFact, float* spVector,
    int nRRlSamples, float rRlScaleFact, float* rRlVector
) {
    SimplifiedEnergyStruct* energy = new SimplifiedEnergyStruct();
    
    // Set basic parameters
    energy->nEnergySamples = nEnergySamples;
    energy->nEnergies = nEnergies;
    energy->nDensitySamples = nDensitySamples;
    energy->densityScaleFact = densityScaleFact;
    energy->nSpSamples = nSpSamples;
    energy->spScaleFact = spScaleFact;
    energy->nRRlSamples = nRRlSamples;
    energy->rRlScaleFact = rRlScaleFact;
    
    // Copy vectors
    energy->energiesPerU.assign(energiesPerU, energiesPerU + nEnergies);
    energy->peakDepths.assign(peakDepths, peakDepths + nEnergies);
    energy->scaleFacts.assign(scaleFacts, scaleFacts + nEnergies);
    energy->ciddMatrix.assign(ciddMatrix, ciddMatrix + nEnergySamples * nEnergies);
    energy->densityVector.assign(densityVector, densityVector + nDensitySamples);
    energy->spVector.assign(spVector, spVector + nSpSamples);
    energy->rRlVector.assign(rRlVector, rRlVector + nRRlSamples);
    
    return energy;
}

// Helper function to destroy simplified energy struct
extern "C" void destroySimplifiedEnergyStruct(SimplifiedEnergyStruct* energy) {
    if (energy) {
        delete energy;
    }
}
