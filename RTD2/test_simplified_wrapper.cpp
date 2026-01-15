/**
 * \file
 * \brief Test file for simplified protons wrapper
 * 
 * This file demonstrates how to use the simplified protons wrapper interface.
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "simplified_protons_wrapper.cuh"

// Helper function to create test data
void createTestData(
    std::vector<float>& imVolData, int3& imVolDims,
    std::vector<float>& doseVolData, int3& doseVolDims,
    std::vector<SimplifiedBeamSettings>& beamSettings,
    SimplifiedEnergyStruct& energyData
) {
    // Set volume dimensions
    imVolDims = make_int3(128, 128, 64);
    doseVolDims = make_int3(128, 128, 64);
    
    // Create test image volume (simple water phantom)
    size_t imVolSize = imVolDims.x * imVolDims.y * imVolDims.z;
    imVolData.resize(imVolSize);
    
    for (size_t i = 0; i < imVolSize; ++i) {
        // Set to water equivalent (HU = 0)
        imVolData[i] = 0.0f;
    }
    
    // Create test dose volume
    size_t doseVolSize = doseVolDims.x * doseVolDims.y * doseVolDims.z;
    doseVolData.resize(doseVolSize, 0.0f);
    
    // Create test beam settings
    beamSettings.resize(1); // Single beam
    
    SimplifiedBeamSettings& beam = beamSettings[0];
    
    // Set energy layers
    beam.energies = {150.0f, 180.0f, 200.0f}; // MeV
    
    // Set spot sigmas
    beam.spotSigmas.resize(3);
    beam.spotSigmas[0] = make_float2(3.0f, 3.0f); // mm
    beam.spotSigmas[1] = make_float2(3.5f, 3.5f);
    beam.spotSigmas[2] = make_float2(4.0f, 4.0f);
    
    // Set other beam parameters
    beam.raySpacing = make_float2(2.0f, 2.0f); // mm
    beam.steps = 100;
    beam.sourceDist = make_float2(2000.0f, 2000.0f); // mm
    beam.spotOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam.spotDelta = make_float3(5.0f, 5.0f, 2.0f);
    
    // Set transform matrices (simplified)
    beam.gantryToImOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam.gantryToImMatrix = make_float3(1.0f, 0.0f, 0.0f); // Identity matrix (simplified)
    beam.gantryToDoseOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam.gantryToDoseMatrix = make_float3(1.0f, 0.0f, 0.0f);
    
    // Create test energy data
    energyData.nEnergySamples = 100;
    energyData.nEnergies = 20;
    energyData.nDensitySamples = 2000;
    energyData.nSpSamples = 2000;
    energyData.nRRlSamples = 2000;
    
    // Set scaling factors
    energyData.densityScaleFact = 0.001f;
    energyData.spScaleFact = 0.001f;
    energyData.rRlScaleFact = 0.001f;
    
    // Create energy vectors
    energyData.energiesPerU.resize(energyData.nEnergies);
    energyData.peakDepths.resize(energyData.nEnergies);
    energyData.scaleFacts.resize(energyData.nEnergies);
    
    for (int i = 0; i < energyData.nEnergies; ++i) {
        energyData.energiesPerU[i] = 100.0f + i * 10.0f; // 100-290 MeV
        energyData.peakDepths[i] = 50.0f + i * 2.0f;      // 50-88 mm
        energyData.scaleFacts[i] = 1.0f;                   // No scaling
    }
    
    // Create IDD matrix
    energyData.ciddMatrix.resize(energyData.nEnergySamples * energyData.nEnergies);
    for (size_t i = 0; i < energyData.ciddMatrix.size(); ++i) {
        energyData.ciddMatrix[i] = 1.0f; // Simplified constant value
    }
    
    // Create density vector (HU to density conversion)
    energyData.densityVector.resize(energyData.nDensitySamples);
    for (int i = 0; i < energyData.nDensitySamples; ++i) {
        float hu = (float)(i - 1000); // HU range: -1000 to 999
        if (hu < -900) {
            energyData.densityVector[i] = 0.001f; // Air
        } else if (hu < 100) {
            energyData.densityVector[i] = 1.0f;   // Water
        } else {
            energyData.densityVector[i] = 1.5f;   // Bone
        }
    }
    
    // Create stopping power vector
    energyData.spVector.resize(energyData.nSpSamples);
    for (int i = 0; i < energyData.nSpSamples; ++i) {
        float hu = (float)(i - 1000);
        if (hu < -900) {
            energyData.spVector[i] = 0.001f; // Air
        } else if (hu < 100) {
            energyData.spVector[i] = 1.0f;   // Water
        } else {
            energyData.spVector[i] = 1.5f;   // Bone
        }
    }
    
    // Create radiation length vector
    energyData.rRlVector.resize(energyData.nRRlSamples);
    for (int i = 0; i < energyData.nRRlSamples; ++i) {
        float hu = (float)(i - 1000);
        if (hu < -900) {
            energyData.rRlVector[i] = 0.001f; // Air
        } else if (hu < 100) {
            energyData.rRlVector[i] = 1.0f;   // Water
        } else {
            energyData.rRlVector[i] = 1.5f;   // Bone
        }
    }
}

// Helper function to print volume statistics
void printVolumeStats(const std::vector<float>& data, const int3& dims, const std::string& name) {
    if (data.empty()) {
        std::cout << name << ": Empty data" << std::endl;
        return;
    }
    
    float minVal = data[0];
    float maxVal = data[0];
    float sum = 0.0f;
    
    for (float val : data) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
        sum += val;
    }
    
    float mean = sum / data.size();
    
    std::cout << name << " statistics:" << std::endl;
    std::cout << "  Dimensions: " << dims.x << " x " << dims.y << " x " << dims.z << std::endl;
    std::cout << "  Total voxels: " << data.size() << std::endl;
    std::cout << "  Min value: " << minVal << std::endl;
    std::cout << "  Max value: " << maxVal << std::endl;
    std::cout << "  Mean value: " << mean << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "Testing Simplified Protons Wrapper" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }
    
    // Get CUDA device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Create test data
    std::vector<float> imVolData;
    int3 imVolDims;
    std::vector<float> doseVolData;
    int3 doseVolDims;
    std::vector<SimplifiedBeamSettings> beamSettings;
    SimplifiedEnergyStruct energyData;
    
    createTestData(imVolData, imVolDims, doseVolData, doseVolDims, beamSettings, energyData);
    
    // Print test data information
    printVolumeStats(imVolData, imVolDims, "Image Volume");
    printVolumeStats(doseVolData, doseVolDims, "Dose Volume");
    
    std::cout << "Beam Settings:" << std::endl;
    std::cout << "  Number of beams: " << beamSettings.size() << std::endl;
    if (!beamSettings.empty()) {
        std::cout << "  Energy layers: " << beamSettings[0].energies.size() << std::endl;
        std::cout << "  Ray tracing steps: " << beamSettings[0].steps << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Energy Data:" << std::endl;
    std::cout << "  Energy samples: " << energyData.nEnergySamples << std::endl;
    std::cout << "  Number of energies: " << energyData.nEnergies << std::endl;
    std::cout << "  Density samples: " << energyData.nDensitySamples << std::endl;
    std::cout << "  Stopping power samples: " << energyData.nSpSamples << std::endl;
    std::cout << "  Radiation length samples: " << energyData.nRRlSamples << std::endl;
    std::cout << std::endl;
    
    // Set volume spacing and origin (simplified)
    float3 imVolSpacing = make_float3(2.0f, 2.0f, 2.0f); // mm
    float3 imVolOrigin = make_float3(-128.0f, -128.0f, -64.0f); // mm
    float3 doseVolSpacing = make_float3(2.0f, 2.0f, 2.0f); // mm
    float3 doseVolOrigin = make_float3(-128.0f, -128.0f, -64.0f); // mm
    
    std::cout << "Calling simplified protons wrapper..." << std::endl;
    
    try {
        // Call the wrapper function
        simplifiedProtonsWrapper(
            imVolData.data(), imVolDims, imVolSpacing, imVolOrigin,
            doseVolData.data(), doseVolDims, doseVolSpacing, doseVolOrigin,
            beamSettings.data(), beamSettings.size(),
            &energyData,
            0,  // GPU ID
            false, // Nuclear correction
            false  // Fine timing
        );
        
        std::cout << "Wrapper function completed successfully!" << std::endl;
        
        // Print results
        printVolumeStats(doseVolData, doseVolDims, "Final Dose Volume");
        
    } catch (const std::exception& e) {
        std::cerr << "Error in wrapper function: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
