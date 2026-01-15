#include "../include/core/raytracedicom_integration.h"
#include "../include/core/ray_tracing.h"
#include "../include/core/bev_ray_tracing.h"
#include "../include/algorithms/idd_sigma.h"
#include "../include/algorithms/superposition.h"
#include "../include/algorithms/fill_idd_and_sigma_params.cuh"
#include "../include/core/forward_declarations.h"
#include "../include/utils/utils.h"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include "../include/algorithms/convolution.h"
#include "../include/algorithms/prim_transf_kernel.h"
#include "../include/algorithms/transfer_param_struct_div3.cuh"
#include "../include/algorithms/transfer_param_helper.h"
#include <iostream>
#include <vector>
#include <random>

// Main wrapper function implementation
void subsecondWrapper(
    const float* imVolData, // ctdata
    const int3& imVolDims,  // self.doseGrid.dims(x*y*z)
    const float3& imVolSpacing, // self.doseGrid.resolution
    const float3& imVolOrigin, // self.doseGrid.corner
    float* doseVolData, // finalDose, pybind11::array out_finalDose
    const int3& doseVolDims, 
    const float3& doseVolSpacing, 
    const float3& doseVolOrigin,
    const RTDBeamSettings* beamSettings, size_t numBeams,
    const RTDEnergyStruct* energyData,
    int gpuId, bool nuclearCorrection, bool fineTiming
) {
    CPU_TIMER_START();
    std::cout << "Starting RTD wrapper with simplified implementation" << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(gpuId);
    cudaFree(0);
    
    if (fineTiming) {
        printDeviceInfo();
        printMemoryInfo();
    }
    
    
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
    
    size_t doseSize = doseVolDims.x * doseVolDims.y * doseVolDims.z;
    float* devDoseVol = (float*)allocateDeviceMemory(doseSize * sizeof(float));
    
    // Debug: Check if devDoseVol allocation was successful
    if (devDoseVol == nullptr) {
        std::cout << "Error: Failed to allocate devDoseVol memory!" << std::endl;
        return;
    }
    
    copyToDevice(devDoseVol, doseVolData, doseSize * sizeof(float));
    
    for (size_t beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
        const RTDBeamSettings& beam = beamSettings[beamIdx];
        std::cout << "Processing beam " << beamIdx << " with " << beam.energies.size() << " energy layers" << std::endl;
        
        // ============================================================================
        // Pre-compute CPB convolution for all layers (energy layer  is executed inside)
        // ============================================================================
        
        int numLayers = beam.energies.size();
        int maxSubspotsPerLayer = 2; // This should come from beam settings
        
        // Generate test subspot data (in real implementation, this should come from beam settings)
        std::vector<float> subspotData(numLayers * maxSubspotsPerLayer * 5);
        std::default_random_engine generator(42);
        std::uniform_real_distribution<float> deltaDist(-5.0f, 5.0f);
        std::uniform_real_distribution<float> weightDist(0.5f, 2.0f);
        std::uniform_real_distribution<float> sigmaDist(0.8f, 1.5f);
        

        //  subspotdata from carbonPBS
        for (int l = 0; l < numLayers; ++l) {
            for (int i = 0; i < maxSubspotsPerLayer; ++i) {
                int baseIdx = (l * maxSubspotsPerLayer + i) * 5;
                subspotData[baseIdx + 0] = deltaDist(generator); // deltaX
                subspotData[baseIdx + 1] = deltaDist(generator); // deltaY
                subspotData[baseIdx + 2] = weightDist(generator); // weight
                subspotData[baseIdx + 3] = sigmaDist(generator); // sigmaX
                subspotData[baseIdx + 4] = sigmaDist(generator); // sigmaY
            }
        }
        
        // Create subspot texture
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaArray* subspotArray;
        cudaExtent subspotExtent = make_cudaExtent(5, maxSubspotsPerLayer, numLayers);
        cudaMalloc3DArray(&subspotArray, &channelDesc, subspotExtent);
        
        cudaMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_cudaPitchedPtr(subspotData.data(), 5 * sizeof(float), 5, maxSubspotsPerLayer);
        copyParams.dstArray = subspotArray;
        copyParams.extent = subspotExtent;
        copyParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = subspotArray;
        
        cudaTextureDesc texDesc = {};
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        
        cudaTextureObject_t subspotTexture = 0;
        cudaCreateTextureObject(&subspotTexture, &resDesc, &texDesc, nullptr);
        
        // Define CPB grid parameters
        vec3f cpbCorner = make_vec3f(-10.0f, -10.0f, doseVolOrigin.z - ROI_MARGIN_Z );
        vec3f cpbResolution = make_vec3f(doseVolSpacing.x * 0.5f, 
                                       doseVolSpacing.y * 0.5f, 
                                       doseVolSpacing.z);
        vec3i cpbDims = make_vec3i(
            (int)(20.0f / cpbResolution.x), // 20cm coverage
            (int)(20.0f / cpbResolution.y), // 20cm coverage
            numLayers
        );
        
        int2 rayDims = make_int2(cpbDims.x, cpbDims.y);
        
        // Allocate CPB weights (all layers)
        float* d_cpbWeights;
        size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
        cudaMalloc(&d_cpbWeights, cpbWeightsSize);
        cudaMemset(d_cpbWeights, 0, cpbWeightsSize);
        
        // Beam geometry parameters
        vec3f beamDirection = make_vec3f(0.0f, 0.0f, 1.0f);
        vec3f bmXDirection = make_vec3f(1.0f, 0.0f, 0.0f);
        vec3f bmYDirection = make_vec3f(0.0f, 1.0f, 0.0f);
        vec3f sourcePosition = make_vec3f(0.0f, 0.0f, -beam.sourceDist.x);
        float sad = beam.sourceDist.x;
        float refPlaneZ = doseVolOrigin.z;
        
        // Perform Subspot to CPB Convolution for all layers
        performSubspotToCPBConvolution(subspotTexture, numLayers, maxSubspotsPerLayer,
                                      cpbCorner, cpbResolution, cpbDims, d_cpbWeights,
                                      beamDirection, bmXDirection, bmYDirection,
                                      sourcePosition, sad, refPlaneZ);
        
        // Allocate rayWeights for all layers (numLayers * rayDims.x * rayDims.y)
        size_t rayWeightsSize = numLayers * rayDims.x * rayDims.y * sizeof(float);
        float* devRayWeightsAllLayers = (float*)allocateDeviceMemory(rayWeightsSize);
        cudaMemset(devRayWeightsAllLayers, 0, rayWeightsSize);
        
        // Map CPB weights to ray weights for each layer
        vec3i rayDimsVec = make_vec3i(rayDims.x, rayDims.y, 1);
        for (int l = 0; l < numLayers; ++l) {
            float* devRayWeightsLayer = devRayWeightsAllLayers + l * rayDims.x * rayDims.y;
            performCPBToRayWeightMapping(d_cpbWeights, cpbDims, cpbCorner, cpbResolution,
                                        devRayWeightsLayer, rayDimsVec,
                                        l,  // Current layer index
                                        beamDirection, bmXDirection, bmYDirection,
                                        sourcePosition, sad, refPlaneZ);
        }
        
        // Cleanup CPB weights and subspot texture (keep rayWeightsAllLayers)
        cudaFree(d_cpbWeights);
        cudaDestroyTextureObject(subspotTexture);
        cudaFreeArray(subspotArray);
        
        // process for each energy layer
        
        for (size_t layerIdx = 0; layerIdx < beam.energies.size(); ++layerIdx) {
            float energy = beam.energies[layerIdx];
            
            // Get rayWeights for current layer
            float* devRayWeights = devRayWeightsAllLayers + layerIdx * rayDims.x * rayDims.y;
            
            // Find energy index using findDecimalOrdered (returns float index for interpolation)
            float energyIdx = 0.0f;
            if (energy >= energyData->energiesPerU.back()) {
                energyIdx = float(energyData->nEnergies - 1);
            } else if (energy < energyData->energiesPerU.front()) {
                energyIdx = 0.0f;
            } else {
                // Find floor index - energiesPerU should be sorted (descending or ascending)
                // Check if sorted ascending or descending
                bool ascending = energyData->energiesPerU[0] < energyData->energiesPerU[energyData->nEnergies - 1];
                int floorIdx = 0;
                if (ascending) {
                    // Ascending order
                    for (int i = 0; i < energyData->nEnergies - 1; ++i) {
                        if (energyData->energiesPerU[i] <= energy && energy < energyData->energiesPerU[i + 1]) {
                            floorIdx = i;
                            break;
                        }
                    }
                } else {
                    // Descending order (typical for proton energies)
                    for (int i = 0; i < energyData->nEnergies - 1; ++i) {
                        if (energyData->energiesPerU[i] >= energy && energy > energyData->energiesPerU[i + 1]) {
                            floorIdx = i;
                    break;
                        }
                    }
                }
                // Calculate fractional part for interpolation
                float corr = (energy - energyData->energiesPerU[floorIdx]) / 
                            (energyData->energiesPerU[floorIdx + 1] - energyData->energiesPerU[floorIdx]);
                energyIdx = float(floorIdx) + corr;
            }
            
            // Calculate energyScaleFact using vectorInterpolate
            float energyScaleFact = 1.0f;
            if (energyIdx <= 0.0f) {
                energyScaleFact = energyData->scaleFacts[0];
            } else if (energyIdx >= float(energyData->nEnergies - 1)) {
                energyScaleFact = energyData->scaleFacts[energyData->nEnergies - 1];
            } else {
                float intPart;
                float decimals = std::modf(energyIdx, &intPart);
                int floorIdx = static_cast<int>(intPart);
                energyScaleFact = energyData->scaleFacts[floorIdx] + 
                                 (energyData->scaleFacts[floorIdx + 1] - energyData->scaleFacts[floorIdx]) * decimals;
            }
            
            // Calculate peakDepth using vectorInterpolate
            float peakDepth = 10.0f;
            if (energyIdx <= 0.0f) {
                peakDepth = energyData->peakDepths[0];
            } else if (energyIdx >= float(energyData->nEnergies - 1)) {
                peakDepth = energyData->peakDepths[energyData->nEnergies - 1];
            } else {
                float intPart;
                float decimals = std::modf(energyIdx, &intPart);
                int floorIdx = static_cast<int>(intPart);
                peakDepth = energyData->peakDepths[floorIdx] + 
                           (energyData->peakDepths[floorIdx + 1] - energyData->peakDepths[floorIdx]) * decimals;
            }
            
            // Allocate ray tracing memory for current layer
            size_t raySize = rayDims.x * rayDims.y * beam.steps;
            float* devRayIdd = (float*)allocateDeviceMemory(raySize * sizeof(float));
            float* devRayRSigmaEff = (float*)allocateDeviceMemory(raySize * sizeof(float));
            int* devBeamFirstInside = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
            int* devFirstStepOutside = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
            int* devFirstPassive = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
            
            // ============================================================================
            // Step 3: Ray Tracing (BEV Density and Stopping Power Calculation)
            // ============================================================================
            
            // Perform ray tracing
            dim3 tracerBlock(16, 16);
            dim3 tracerGrid((rayDims.x + tracerBlock.x - 1) / tracerBlock.x,
                           (rayDims.y + tracerBlock.y - 1) / tracerBlock.y);
            
            // ============================================================================
            // Step 3: Ray Tracing - Density and Stopping Power Calculation
            // ============================================================================
            
            // Create proper tracer parameters for RayTracedicom algorithm
            // Ray grid should align with CPB grid at reference plane
            // startPos should be the first ray position in image volume coordinates
            Float3ToBevTransform bevTransform;
            bevTransform.origin = make_float3(0.0f, 0.0f, 0.0f);
            bevTransform.xAxis = make_float3(1.0f, 0.0f, 0.0f);
            bevTransform.yAxis = make_float3(0.0f, 1.0f, 0.0f);
            bevTransform.zAxis = make_float3(0.0f, 0.0f, 1.0f);
            
            DensityAndSpTracerParams densityTracerParams(1.0f, 1.0f, beam.steps, bevTransform);
            
            // Ray spacing should match CPB resolution
            densityTracerParams.raySpacing = make_float2(cpbResolution.x, cpbResolution.y);
            
            // Start position: first ray position in image volume coordinates
            // Map CPB corner to image volume coordinates
            // CPB corner is in world coordinates (cm), need to convert to image volume indices
            // Ensure startPos is within imagoe volume bounds
            float startX_world = cpbCorner.x;
            float startY_world = cpbCorner.y;
            float startZ_world = refPlaneZ;
            
            // Convert to image volume indices
            float startX_idx = (startX_world - imVolOrigin.x) / imVolSpacing.x;
            float startY_idx = (startY_world - imVolOrigin.y) / imVolSpacing.y;
            float startZ_idx = (startZ_world - imVolOrigin.z) / imVolSpacing.z;
            
            // Clamp to valid range [0, dims-1]
            startX_idx = std::max(0.0f, std::min(startX_idx, float(imVolDims.x - 1)));
            startY_idx = std::max(0.0f, std::min(startY_idx, float(imVolDims.y - 1)));
            startZ_idx = std::max(0.0f, std::min(startZ_idx, float(imVolDims.z - 1)));
            
            densityTracerParams.startPos = make_float3(startX_idx, startY_idx, startZ_idx);
            
            // Step increment: direction from source through ray position
            // For simplified case: step along beam direction (Z axis)
            // Convert ray spacing to image volume index spacing
            float stepSizeX = cpbResolution.x / imVolSpacing.x; // Step size in image volume indices
            float stepSizeY = cpbResolution.y / imVolSpacing.y;
            float stepSizeZ = imVolSpacing.z / imVolSpacing.z; // 1.0 (step by one voxel in Z)
            densityTracerParams.stepInc = make_float3(0.0f, 0.0f, stepSizeZ);
            densityTracerParams.stepLength = imVolSpacing.z; // Physical step length in cm
            
            // Debug: Print tracer parameters
            std::cout << "  DensityAndSpTracerParams:" << std::endl;
            std::cout << "    steps: " << densityTracerParams.getSteps() << std::endl;
            std::cout << "    densityScale: " << densityTracerParams.getDensityScale() << std::endl;
            std::cout << "    spScale: " << densityTracerParams.getSpScale() << std::endl;
            std::cout << "    startPos: (" << densityTracerParams.startPos.x << ", " << densityTracerParams.startPos.y << ", " << densityTracerParams.startPos.z << ")" << std::endl;
            std::cout << "    raySpacing: (" << densityTracerParams.raySpacing.x << ", " << densityTracerParams.raySpacing.y << ")" << std::endl;
            std::cout << "    stepInc: (" << densityTracerParams.stepInc.x << ", " << densityTracerParams.stepInc.y << ", " << densityTracerParams.stepInc.z << ")" << std::endl;
            std::cout << "    stepLength: " << densityTracerParams.stepLength << std::endl;
            
            // Create IDD parameters
            FillIddAndSigmaParams iddParams;
            iddParams.energyIdx = energyIdx;  // Float index for texture interpolation
            iddParams.energyScaleFact = energyScaleFact;  // Interpolated scale factor
            iddParams.peakDepth = peakDepth;  // Interpolated peak depth
            iddParams.rRlScale = 1.0f;
            iddParams.spotDist = 0.0f;
            iddParams.nucMemStep = rayDims.x * rayDims.y;
            iddParams.first = 0;
            iddParams.afterLast = beam.steps - 1;
            iddParams.entrySigmaSq = 0.1f; // Initial sigma squared (non-zero)
            iddParams.stepLength = 0.1f;
            iddParams.sigmaSqAirLin = 0.01f; // Air scattering linear term (non-zero)
            iddParams.sigmaSqAirQuad = 0.001f; // Air scattering quadratic term (non-zero)
            iddParams.dist = make_vec3f(100.0f, 100.0f, 100.0f);
            iddParams.corner = make_vec3f(-50.0f, -50.0f, 0.0f);
            iddParams.delta = make_vec3f(0.1f, 0.1f, 0.1f);
            
            // Calculate volConst, volLin, volSq for stepVol calculation
            // Based on RayTracedicom formula: volPerDist(k) = volConst + k*volLin + k*k*volSq
            float deltaX = iddParams.delta.x;
            float deltaY = iddParams.delta.y;
            float deltaZ = iddParams.delta.z;
            float cornerZ = iddParams.corner.z;
            float distX = iddParams.dist.x;
            float distY = iddParams.dist.y;
            float deltaXYZ = fabsf(deltaX * deltaY * deltaZ);
            
            iddParams.volConst = deltaXYZ * (1.0f - cornerZ/distX - cornerZ/distY + 
                                             (cornerZ*cornerZ + deltaZ*deltaZ/12.0f)/(distX*distY));
            iddParams.volLin = deltaXYZ * deltaZ * (-1.0f/distX - 1.0f/distY + 2.0f*cornerZ/(distX*distY));
            iddParams.volSq = deltaXYZ * deltaZ * deltaZ / (distX * distY);
            
            // Initialize step and air division parameters
            iddParams.initStepAndAirDiv();
            
            // Allocate BEV arrays for proper ray tracing (3D arrays: steps * rayDims.x * rayDims.y)
            size_t bevArraySize = beam.steps * rayDims.x * rayDims.y;
            float* devBevDensity = (float*)allocateDeviceMemory(bevArraySize * sizeof(float));
            float* devBevCumulSp = (float*)allocateDeviceMemory(bevArraySize * sizeof(float));
            
            // Initialize arrays
            cudaMemset(devBevDensity, 0, bevArraySize * sizeof(float));
            cudaMemset(devBevCumulSp, 0, bevArraySize * sizeof(float));
            
            // Create layer energy array
            std::vector<float> layerEnergyData(numLayers);
            for (int i = 0; i < numLayers; i++) {
                layerEnergyData[i] = beam.energies[i];
            }
            float* devLayerEnergy = (float*)allocateDeviceMemory(numLayers * sizeof(float));
            copyToDevice(devLayerEnergy, layerEnergyData.data(), numLayers * sizeof(float));
            
            // Launch proper BEV ray tracing kernel
            rayTracingBEVKernel<<<tracerGrid, tracerBlock>>>(
                devBevDensity, devBevCumulSp, devRayIdd, devRayRSigmaEff,
                devRayWeights, devBeamFirstInside, devFirstStepOutside, devFirstPassive,
                densityTracerParams, iddParams, devLayerEnergy,
                imVolTex, densityTex, stoppingPowerTex, cumulIddTex, rRadiationLengthTex,
                imVolDims
            );
            checkCudaErrors(cudaDeviceSynchronize());
            
            // ============================================================================
            // Step 3b: IDD and Sigma Calculation using proper RayTracedicom algorithm
            // ============================================================================
            
            // Launch fillIddAndSigmaKernel - uses energyIdx to query IDD lookup table
            fillIddAndSigmaKernel<<<tracerGrid, tracerBlock>>>(
                devBevDensity, devBevCumulSp, devRayIdd, devRayRSigmaEff,
                devRayWeights, devBeamFirstInside, devFirstStepOutside, devFirstPassive,
                iddParams,
                rayDims.x, rayDims.y, beam.steps,
                cumulIddTex, rRadiationLengthTex
            );
            checkCudaErrors(cudaDeviceSynchronize());
            
            // Clean up layer energy
            freeDeviceMemory(devLayerEnergy);
            
            // Clean up BEV arrays
            freeDeviceMemory(devBevDensity);
            freeDeviceMemory(devBevCumulSp);
            
            // Debug: Check ray tracing results
            std::vector<float> rayIddData(raySize);
            std::vector<float> raySigmaData(raySize);
            copyToHost(rayIddData.data(), devRayIdd, raySize * sizeof(float));
            copyToHost(raySigmaData.data(), devRayRSigmaEff, raySize * sizeof(float));
            
            float totalRayIdd = 0.0f;
            float maxRayIdd = 0.0f;
            int nonZeroRayIdd = 0;
            float totalRaySigma = 0.0f;
            float maxRaySigma = 0.0f;
            int nonZeroRaySigma = 0;
            
            for (int i = 0; i < raySize; i++) {
                totalRayIdd += rayIddData[i];
                if (rayIddData[i] > maxRayIdd) maxRayIdd = rayIddData[i];
                if (rayIddData[i] > 1e-6f) nonZeroRayIdd++;
                
                totalRaySigma += raySigmaData[i];
                if (raySigmaData[i] > maxRaySigma) maxRaySigma = raySigmaData[i];
                if (raySigmaData[i] > 1e-6f) nonZeroRaySigma++;
            }
            
            std::cout << "  Ray Tracing Results Analysis:" << std::endl;
            std::cout << "    Ray IDD - Total: " << totalRayIdd << ", Max: " << maxRayIdd << ", Non-zero: " << nonZeroRayIdd << "/" << raySize << std::endl;
            std::cout << "    Ray Sigma - Total: " << totalRaySigma << ", Max: " << maxRaySigma << ", Non-zero: " << nonZeroRaySigma << "/" << raySize << std::endl;
            
            // ============================================================================
            // Step 3c: Calculate beamFirstInside and beamFirstCalculatedPassive
            // ============================================================================
            
            // Calculate beamFirstInside (minimum first inside step across all rays)
            int* devBeamFirstInsideMin;
            checkCudaErrors(cudaMalloc(&devBeamFirstInsideMin, sizeof(int)));
            sliceMinVar<int, 1024><<<1, 1024, 1024*sizeof(int)>>>(
                devBeamFirstInside, devBeamFirstInsideMin, rayDims.x * rayDims.y);
            checkCudaErrors(cudaDeviceSynchronize());
            
            int beamFirstInside;
            checkCudaErrors(cudaMemcpy(&beamFirstInside, devBeamFirstInsideMin, sizeof(int), cudaMemcpyDeviceToHost));
            beamFirstInside = beamFirstInside - 1;  // Convert from 1-based to 0-based
            
            // Calculate beamFirstOutside (maximum first outside step)
            int* devBeamFirstOutsideMax;
            checkCudaErrors(cudaMalloc(&devBeamFirstOutsideMax, sizeof(int)));
            sliceMaxVar<int, 1024><<<1, 1024, 1024*sizeof(int)>>>(
                devFirstStepOutside, devBeamFirstOutsideMax, rayDims.x * rayDims.y);
            checkCudaErrors(cudaDeviceSynchronize());
            
            int beamFirstOutside;
            checkCudaErrors(cudaMemcpy(&beamFirstOutside, devBeamFirstOutsideMax, sizeof(int), cudaMemcpyDeviceToHost));
            beamFirstOutside = beamFirstOutside - 1;
            
            // Calculate beamFirstCalculatedPassive (maximum passive step)
            int* devBeamFirstPassiveMax;
            checkCudaErrors(cudaMalloc(&devBeamFirstPassiveMax, sizeof(int)));
            sliceMaxVar<int, 1024><<<1, 1024, 1024*sizeof(int)>>>(
                devFirstPassive, devBeamFirstPassiveMax, rayDims.x * rayDims.y);
            checkCudaErrors(cudaDeviceSynchronize());
            
            int beamFirstPassive;
            checkCudaErrors(cudaMemcpy(&beamFirstPassive, devBeamFirstPassiveMax, sizeof(int), cudaMemcpyDeviceToHost));
            beamFirstPassive = beamFirstPassive - 1;
            
            // Calculate beamFirstCalculatedPassive (minimum of firstOutside and firstPassive)
            // If firstPassive is -1 (not found), use firstOutside
            // If both are -1, use beamFirstInside (at least calculate from entry point)
            int beamFirstCalculatedPassive;
            if (beamFirstPassive >= 0 && beamFirstOutside >= 0) {
                beamFirstCalculatedPassive = std::min(beamFirstOutside, beamFirstPassive);
            } else if (beamFirstPassive >= 0) {
                beamFirstCalculatedPassive = beamFirstPassive;
            } else if (beamFirstOutside >= 0) {
                beamFirstCalculatedPassive = beamFirstOutside;
            } else {
                // Both are -1, use beamFirstInside as fallback
                beamFirstCalculatedPassive = beamFirstInside;
            }
            
            // Ensure beamFirstCalculatedPassive is at least beamFirstInside
            if (beamFirstCalculatedPassive < beamFirstInside) {
                beamFirstCalculatedPassive = beamFirstInside;
            }
            
            // Ensure beamFirstCalculatedPassive is valid (positive and within steps)
            if (beamFirstCalculatedPassive < 0) {
                beamFirstCalculatedPassive = beamFirstInside;
            }
            if (beamFirstCalculatedPassive >= beam.steps) {
                beamFirstCalculatedPassive = beam.steps - 1;
            }
            
            std::cout << "  Beam Entry/Exit Analysis:" << std::endl;
            std::cout << "    beamFirstInside: " << beamFirstInside << std::endl;
            std::cout << "    beamFirstOutside: " << beamFirstOutside << std::endl;
            std::cout << "    beamFirstPassive: " << beamFirstPassive << std::endl;
            std::cout << "    beamFirstCalculatedPassive: " << beamFirstCalculatedPassive << std::endl;
            
            // Cleanup temporary arrays
            cudaFree(devBeamFirstInsideMin);
            cudaFree(devBeamFirstOutsideMax);
            cudaFree(devBeamFirstPassiveMax);
            
            // ============================================================================
            // Step 4: Complete Tile-Based Superposition (RayTracedicom algorithm)
            // ============================================================================
            
            // Allocate 3D BEV dose array with padding for superposition
            // Dimensions: (rayDims.x + 2*maxSuperpR) x (rayDims.y + 2*maxSuperpR) x beamFirstCalculatedPassive
            const int maxSuperpR = MAX_SUPERP_RADIUS;
            
            // Check if beamFirstCalculatedPassive is valid before allocating
            // It should be >= beamFirstInside and < beam.steps
            if (beamFirstCalculatedPassive < beamFirstInside || beamFirstCalculatedPassive > beam.steps || beamFirstCalculatedPassive < 0) {
                std::cout << "  Warning: Invalid beamFirstCalculatedPassive (" << beamFirstCalculatedPassive 
                         << ") or beamFirstInside (" << beamFirstInside << "), skipping superposition" << std::endl;
                std::cout << "    beamFirstOutside: " << beamFirstOutside << ", beamFirstPassive: " << beamFirstPassive << std::endl;
                // Clean up and continue to next layer
                // Note: devRayWeights is part of devRayWeightsAllLayers, don't free here
                freeDeviceMemory(devRayIdd);
                freeDeviceMemory(devRayRSigmaEff);
                freeDeviceMemory(devBeamFirstInside);
                freeDeviceMemory(devFirstStepOutside);
                freeDeviceMemory(devFirstPassive);
                continue; // Skip to next energy layer
            }
            
            const int bevDoseX = rayDims.x + 2 * maxSuperpR;
            const int bevDoseY = rayDims.y + 2 * maxSuperpR;
            const int bevDoseZ = beamFirstCalculatedPassive;
            
            size_t bevDoseSize = bevDoseX * bevDoseY * bevDoseZ;
            float* devBevPrimDose = (float*)allocateDeviceMemory(bevDoseSize * sizeof(float));
            cudaMemset(devBevPrimDose, 0, bevDoseSize * sizeof(float));
            
            std::cout << "  Allocating BEV dose array with padding:" << std::endl;
            std::cout << "    BEV dimensions: (" << bevDoseX << ", " << bevDoseY << ", " << bevDoseZ << ")" << std::endl;
            std::cout << "    Padding: " << maxSuperpR << " voxels on each side" << std::endl;
            
            // Adjust ray IDD and sigma arrays to account for padding offset
            // We need to copy data with offset to account for padding
            // For now, we'll use the original arrays and adjust indices in superposition
            
            // Perform complete tile-based superposition
            // This uses the full RayTracedicom algorithm with tile-based radius calculation
            performCompleteTileBasedSuperposition(
                devRayIdd,
                devRayRSigmaEff,
                devBevPrimDose,
                rayDims.x,
                rayDims.y,
                beam.steps,
                beamFirstInside,
                beamFirstCalculatedPassive
            );
            
            // Debug: Check BEV dose array after superposition
            std::vector<float> bevDoseHost(bevDoseSize);
            copyToHost(bevDoseHost.data(), devBevPrimDose, bevDoseSize * sizeof(float));
            
            float totalBevDose = 0.0f;
            float maxBevDose = 0.0f;
            int nonZeroBevDose = 0;
            for (size_t i = 0; i < bevDoseSize; i++) {
                totalBevDose += bevDoseHost[i];
                if (bevDoseHost[i] > maxBevDose) maxBevDose = bevDoseHost[i];
                if (bevDoseHost[i] > 1e-6f) nonZeroBevDose++;
            }
            std::cout << "  BEV Dose Analysis after Superposition:" << std::endl;
            std::cout << "    Total BEV dose: " << totalBevDose << ", Max: " << maxBevDose << ", Non-zero: " << nonZeroBevDose << "/" << bevDoseSize << std::endl;
            
            // ============================================================================
            // Step 5: Dose Transformation and Accumulation (3D BEV to Dose Grid)
            // ============================================================================
            
            // Create BEV dose 3D texture
            cudaTextureObject_t bevPrimDoseTex = create3DTexture(
                devBevPrimDose,
                make_int3(bevDoseX, bevDoseY, bevDoseZ),
                cudaFilterModeLinear,
                cudaAddressModeBorder
            );
            
            // Create TransferParamStructDiv3 from beam parameters
            // Ray spacing at reference plane should match CPB resolution (not dose volume spacing)
            // The ray grid is defined by CPB grid, so ray spacing should be CPB resolution
            vec3f raySpacing = make_vec3f(
                cpbResolution.x,  // Use CPB resolution for ray spacing
                cpbResolution.y,
                doseVolSpacing.z  // Z spacing matches dose volume
            );
            
            // Ray offset: CPB corner position in world coordinates
            vec3f rayOffset = make_vec3f(
                cpbCorner.x,
                cpbCorner.y,
                refPlaneZ
            );
            
            // Global offset accounts for padding and beamFirstInside
            vec3f globalOffset = make_vec3f(
                float(maxSuperpR),  // X padding offset
                float(maxSuperpR),  // Y padding offset
                float(beamFirstInside)  // Z offset (beamFirstInside)
            );
            
            // Create transfer parameters
            vec2f sourceDistVec = make_vec2f(beam.sourceDist.x, beam.sourceDist.y);
            TransferParamStructDiv3 transferParams = createTransferParamFromBeamParams(
                raySpacing,
                rayOffset,
                sourceDistVec,
                make_vec3f(doseVolSpacing.x, doseVolSpacing.y, doseVolSpacing.z),
                make_vec3f(doseVolOrigin.x, doseVolOrigin.y, doseVolOrigin.z),
                globalOffset
            );
            
            // Launch primTransfDiv kernel to transform BEV coordinates to dose grid
            dim3 transfBlockDim(16, 16);
            dim3 transfGridDim(
                (doseVolDims.x + transfBlockDim.x - 1) / transfBlockDim.x,
                (doseVolDims.y + transfBlockDim.y - 1) / transfBlockDim.y
            );
            
            vec3i startIdx = make_vec3i(0, 0, beamFirstInside);
            int maxZIdx = beamFirstCalculatedPassive - 1;
            
            std::cout << "  Launching primTransfDiv kernel:" << std::endl;
            std::cout << "    Grid: (" << transfGridDim.x << ", " << transfGridDim.y << ")" << std::endl;
            std::cout << "    Block: (" << transfBlockDim.x << ", " << transfBlockDim.y << ")" << std::endl;
            std::cout << "    startIdx: (" << startIdx.x << ", " << startIdx.y << ", " << startIdx.z << ")" << std::endl;
            std::cout << "    maxZ: " << maxZIdx << std::endl;
            
            primTransfDiv<<<transfGridDim, transfBlockDim>>>(
                devDoseVol,
                transferParams,
                startIdx,
                maxZIdx,
                make_vec3i(doseVolDims.x, doseVolDims.y, doseVolDims.z),
                bevPrimDoseTex
            );
            checkCudaErrors(cudaDeviceSynchronize());
            
            // Cleanup texture
            cudaDestroyTextureObject(bevPrimDoseTex);
            
            std::cout << "  Dose transformation completed using primTransfDiv kernel" << std::endl;
            
            std::cout << "  Dose accumulation completed" << std::endl;
            
            // Clean up memory (devRayWeights is part of devRayWeightsAllLayers, cleaned up after all layers)
            freeDeviceMemory(devRayIdd);
            freeDeviceMemory(devRayRSigmaEff);
            freeDeviceMemory(devBeamFirstInside);
            freeDeviceMemory(devFirstStepOutside);
            freeDeviceMemory(devFirstPassive);
            freeDeviceMemory(devBevPrimDose);
        }
    }
    
    // Copy accumulated dose from device to host
    copyToHost(doseVolData, devDoseVol, doseSize * sizeof(float));
    
    // Clean up device memory
    freeDeviceMemory(devDoseVol);
    
    // Clean up textures
    cudaDestroyTextureObject(imVolTex);
    cudaDestroyTextureObject(cumulIddTex);
    cudaDestroyTextureObject(densityTex);
    cudaDestroyTextureObject(stoppingPowerTex);
    cudaDestroyTextureObject(rRadiationLengthTex);
    
    CPU_TIMER_END("RTD Wrapper");
}

// Helper functions
extern "C" RTDBeamSettings* createRTDBeamSettings() {
    return new RTDBeamSettings();
}

extern "C" void destroyRTDBeamSettings(RTDBeamSettings* beam) {
    delete beam;
}

extern "C" RTDEnergyStruct* createRTDEnergyStruct() {
    return new RTDEnergyStruct();
}

extern "C" void destroyRTDEnergyStruct(RTDEnergyStruct* energy) {
    delete energy;
}