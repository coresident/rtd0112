/**
 * \file
 * \brief Main RayTraceDicom Wrapper Implementation
 */

#include "raytracedicom_integration.h"
#include "ray_tracing.h"
#include "idd_sigma_calculation.h"
#include "superposition_kernels.h"
#include "utils.h"
#include <iostream>
#include <vector>

// Main wrapper function implementation
void raytraceDicomWrapper(
    const float* imVolData, const int3& imVolDims, const float3& imVolSpacing, const float3& imVolOrigin,
    float* doseVolData, const int3& doseVolDims, const float3& doseVolSpacing, const float3& doseVolOrigin,
    const RayTraceDicomBeamSettings* beamSettings, size_t numBeams,
    const RayTraceDicomEnergyStruct* energyData,
    int gpuId, bool nuclearCorrection, bool fineTiming
) {
    std::cout << "Starting RayTraceDicom wrapper with complete kernel integration..." << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(gpuId);
    cudaFree(0);
    
    if (fineTiming) {
        printDeviceInfo();
        printMemoryInfo();
    }
    
    // Create texture objects for lookup tables
    cudaTextureObject_t imVolTex = create3DTexture(imVolData, imVolDims, cudaFilterModeLinear, cudaAddressModeBorder);
    
    cudaTextureObject_t cumulIddTex = create2DTexture(&energyData->ciddMatrix[0], 
                                                     make_int2(energyData->nEnergySamples, energyData->nEnergies),
                                                     cudaFilterModeLinear, cudaAddressModeClamp);
    
    cudaTextureObject_t densityTex = create1DTexture(&energyData->densityVector[0], 
                                                    energyData->nDensitySamples,
                                                    cudaFilterModeLinear, cudaAddressModeClamp);
    
    cudaTextureObject_t stoppingPowerTex = create1DTexture(&energyData->spVector[0], 
                                                          energyData->nSpSamples,
                                                          cudaFilterModeLinear, cudaAddressModeClamp);
    
    cudaTextureObject_t rRadiationLengthTex = create1DTexture(&energyData->rRlVector[0], 
                                                             energyData->nRRlSamples,
                                                             cudaFilterModeLinear, cudaAddressModeClamp);
    
    // Allocate dose volume
    size_t doseSize = doseVolDims.x * doseVolDims.y * doseVolDims.z;
    float* devDoseVol = (float*)allocateDeviceMemory(doseSize * sizeof(float));
    copyToDevice(devDoseVol, doseVolData, doseSize * sizeof(float));
    
    // Process each beam
    for (size_t beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
        const RayTraceDicomBeamSettings& beam = beamSettings[beamIdx];
        std::cout << "Processing beam " << beamIdx << " with " << beam.energies.size() << " energy layers" << std::endl;
        
        // Process each energy layer
        for (size_t layerIdx = 0; layerIdx < beam.energies.size(); ++layerIdx) {
            float energy = beam.energies[layerIdx];
            std::cout << "  Layer " << layerIdx << ": " << energy << " MeV" << std::endl;
            
            // Find energy index
            int energyIdx = 0;
            for (int i = 0; i < energyData->nEnergies; ++i) {
                if (energyData->energiesPerU[i] >= energy) {
                    energyIdx = i;
                    break;
                }
            }
            
            // Calculate ray dimensions
            int2 rayDims = make_int2(imVolDims.x, imVolDims.y);
            
            // Allocate ray tracing memory
            size_t raySize = rayDims.x * rayDims.y * beam.steps;
            float* devRayWeights = (float*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(float));
            float* devRayIdd = (float*)allocateDeviceMemory(raySize * sizeof(float));
            float* devRayRSigmaEff = (float*)allocateDeviceMemory(raySize * sizeof(float));
            int* devBeamFirstInside = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
            int* devFirstStepOutside = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
            int* devFirstPassive = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
            
            // Initialize ray weights
            std::vector<float> rayWeights(rayDims.x * rayDims.y, 1.0f);
            copyToDevice(devRayWeights, rayWeights.data(), rayDims.x * rayDims.y * sizeof(float));
            
            // Create BEV transform
            Float3ToBevTransform bevTransform = createBevTransform(beam.spotOffset, beam.spotDelta, 
                                                                 beam.gantryToImOffset, beam.gantryToImMatrix,
                                                                 beam.sourceDist, imVolOrigin);
            
            // Create tracer parameters
            DensityAndSpTracerParams tracerParams = createTracerParams(energyData->densityScaleFact, 
                                                                      energyData->spScaleFact, beam.steps, bevTransform,
                                                                      beam.raySpacing, beam.spotOffset, 
                                                                      make_float3(0, 0, 1.0f/beam.steps), 1.0f/beam.steps);
            
            // Launch density and stopping power tracing kernel
            dim3 tracerBlock(32, 8);
            dim3 tracerGrid = calculateGridSize(tracerBlock, rayDims);
            
            fillBevDensityAndSpKernel<<<tracerGrid, tracerBlock>>>(
                devRayWeights, devRayIdd, devBeamFirstInside, devFirstStepOutside,
                tracerParams, imVolTex, densityTex, stoppingPowerTex, imVolDims);
            
            // Create IDD and sigma parameters
            FillIddAndSigmaParams iddParams = createIddParams(energyIdx, energyData->peakDepths[energyIdx], 
                                                            energyData->scaleFacts[energyIdx], 1.0f, 
                                                            energyData->rRlScaleFact, 1.0f/beam.steps, 0, beam.steps, 
                                                            1.0f, beam.spotSigmas[layerIdx].x * beam.spotSigmas[layerIdx].x);
            
            // Launch IDD and sigma calculation kernel
            fillIddAndSigmaKernel<<<tracerGrid, tracerBlock>>>(
                devRayWeights, devRayIdd, devRayIdd, devRayRSigmaEff, devRayWeights,
                devBeamFirstInside, devFirstStepOutside, devFirstPassive,
                iddParams, cumulIddTex, rRadiationLengthTex);
            
            // Allocate BEV dose volume for superposition
            float* devBevPrimDose = (float*)allocateDeviceMemory(rayDims.x * rayDims.y * beam.steps * sizeof(float));
            cudaMemset(devBevPrimDose, 0, rayDims.x * rayDims.y * beam.steps * sizeof(float));
            
            // Launch superposition kernels
            SuperpositionParams superpParams;
            superpParams.maxRadius = 32;
            superpParams.tileSize = 256;
            
            launchSuperpositionKernels(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDims.x, rayDims.y, superpParams);
            
            // Create BEV dose texture
            cudaTextureObject_t bevPrimDoseTex = create3DTexture(devBevPrimDose, 
                                                                make_int3(rayDims.x, rayDims.y, beam.steps),
                                                                cudaFilterModeLinear, cudaAddressModeBorder);
            
            // Launch dose transformation kernel
            dim3 transformBlock(32, 8);
            dim3 transformGrid = calculateGridSize(transformBlock, make_int3(doseVolDims.x, doseVolDims.y, doseVolDims.z));
            
            primTransfDivKernel<<<transformGrid, transformBlock>>>(
                devDoseVol, make_int3(0, 0, 0), doseVolDims.z - 1, make_uint3(doseVolDims.x, doseVolDims.y, doseVolDims.z),
                bevPrimDoseTex);
            
            // Clean up layer-specific memory
            freeDeviceMemory(devRayWeights);
            freeDeviceMemory(devRayIdd);
            freeDeviceMemory(devRayRSigmaEff);
            freeDeviceMemory(devBeamFirstInside);
            freeDeviceMemory(devFirstStepOutside);
            freeDeviceMemory(devFirstPassive);
            freeDeviceMemory(devBevPrimDose);
            cudaDestroyTextureObject(bevPrimDoseTex);
        }
    }
    
    // Copy final dose back to host
    copyFromDevice(doseVolData, devDoseVol, doseSize * sizeof(float));
    
    // Clean up
    freeDeviceMemory(devDoseVol);
    cudaDestroyTextureObject(imVolTex);
    cudaDestroyTextureObject(cumulIddTex);
    cudaDestroyTextureObject(densityTex);
    cudaDestroyTextureObject(stoppingPowerTex);
    cudaDestroyTextureObject(rRadiationLengthTex);
    
    std::cout << "RayTraceDicom wrapper completed!" << std::endl;
}

// Helper functions for creating test data
RayTraceDicomBeamSettings* createRayTraceDicomBeamSettings() {
    RayTraceDicomBeamSettings* beam = new RayTraceDicomBeamSettings();
    beam->energies = {150.0f, 140.0f, 130.0f, 120.0f, 110.0f};
    beam->spotSigmas = {make_float2(2.0f, 2.0f), make_float2(2.0f, 2.0f), make_float2(2.0f, 2.0f), 
                       make_float2(2.0f, 2.0f), make_float2(2.0f, 2.0f)};
    beam->raySpacing = make_float2(1.0f, 1.0f);
    beam->steps = 100;
    beam->sourceDist = make_float2(1000.0f, 1000.0f);
    beam->spotOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam->spotDelta = make_float3(5.0f, 5.0f, 0.0f);
    beam->gantryToImOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam->gantryToImMatrix = make_float3(1.0f, 0.0f, 0.0f);
    beam->gantryToDoseOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam->gantryToDoseMatrix = make_float3(1.0f, 0.0f, 0.0f);
    return beam;
}

void destroyRayTraceDicomBeamSettings(RayTraceDicomBeamSettings* beam) {
    delete beam;
}

RayTraceDicomEnergyStruct* createRayTraceDicomEnergyStruct() {
    RayTraceDicomEnergyStruct* energy = new RayTraceDicomEnergyStruct();
    energy->nEnergySamples = 100;
    energy->nEnergies = 5;
    energy->energiesPerU = {150.0f, 140.0f, 130.0f, 120.0f, 110.0f};
    energy->peakDepths = {15.0f, 14.0f, 13.0f, 12.0f, 11.0f};
    energy->scaleFacts = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    energy->ciddMatrix.resize(energy->nEnergySamples * energy->nEnergies, 1.0f);
    energy->nDensitySamples = 100;
    energy->densityScaleFact = 1.0f;
    energy->densityVector.resize(energy->nDensitySamples, 1.0f);
    energy->nSpSamples = 100;
    energy->spScaleFact = 1.0f;
    energy->spVector.resize(energy->nSpSamples, 1.0f);
    energy->nRRlSamples = 100;
    energy->rRlScaleFact = 1.0f;
    energy->rRlVector.resize(energy->nRRlSamples, 1.0f);
    return energy;
}

void destroyRayTraceDicomEnergyStruct(RayTraceDicomEnergyStruct* energy) {
    delete energy;
}
