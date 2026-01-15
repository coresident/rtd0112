/**
 * \file
 * \brief Test program for integrated wrapper with subspot to CPB convolution
 */

#include "../include/core/raytracedicom_integration.h"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

// Test data generation
std::vector<float> createTestCTData(int3 dims) {
    std::vector<float> data(dims.x * dims.y * dims.z);
    std::default_random_engine generator(42);
    
    // Generate HU values (HU+1000 format for RayTracedicom)
    // HU values: -1000 (air) to 3000 (bone)
    // HU+1000: 0 (air) to 4000 (bone)
    // We'll use mostly water-like tissue (HU ~ 0, so HU+1000 ~ 1000)
    std::uniform_real_distribution<float> huDist(800.0f, 1200.0f); // Water-like tissue (HU ~ 0, HU+1000 ~ 1000)
    
    // Create a simple phantom: central region with tissue
    int centerX = dims.x / 2;
    int centerY = dims.y / 2;
    int centerZ = dims.z / 2;
    int radius = std::min({dims.x, dims.y, dims.z}) / 3;
    
    for (int z = 0; z < dims.z; ++z) {
        for (int y = 0; y < dims.y; ++y) {
            for (int x = 0; x < dims.x; ++x) {
                int idx = z * dims.x * dims.y + y * dims.x + x;
                
                // Calculate distance from center
                float dx = x - centerX;
                float dy = y - centerY;
                float dz = z - centerZ;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                
                if (dist < radius) {
                    // Inside phantom: tissue (HU+1000 ~ 1000)
                    data[idx] = huDist(generator);
                } else {
                    // Outside phantom: air (HU+1000 ~ 0)
                    data[idx] = 50.0f + huDist(generator) * 0.1f; // Small value for air
                }
            }
        }
    }
    
    return data;
}

RTDBeamSettings createTestBeamSettings() {
    RTDBeamSettings beam;
    
    // Add energy layers
    beam.energies = {120.0f, 140.0f, 160.0f};
    
    // Add spot sigmas
    beam.spotSigmas = {{1.0f, 1.0f}, {1.2f, 1.2f}, {1.4f, 1.4f}};
    
    // Ray spacing
    beam.raySpacing = {0.1f, 0.1f};
    
    // Steps
    beam.steps = 100;
    
    // Source distance
    beam.sourceDist = {100.0f, 100.0f};
    
    // Spot offset and delta
    beam.spotOffset = {0.0f, 0.0f, 0.0f};
    beam.spotDelta = {0.5f, 0.5f, 0.0f};
    
    // Transform matrices (identity for now)
    beam.gantryToImOffset = {0.0f, 0.0f, 0.0f};
    beam.gantryToImMatrix = {1.0f, 0.0f, 0.0f};
    beam.gantryToDoseOffset = {0.0f, 0.0f, 0.0f};
    beam.gantryToDoseMatrix = {1.0f, 0.0f, 0.0f};
    
    return beam;
}

RTDEnergyStruct createTestEnergyData() {
    RTDEnergyStruct energy;
    
    energy.nEnergySamples = 50;  // Number of depth samples per energy
    energy.nEnergies = 3;         // Number of energy layers
    
    // Energy list (nEnergies elements, not nEnergySamples!)
    energy.energiesPerU.resize(energy.nEnergies);
    energy.energiesPerU[0] = 120.0f;  // Match test beam energies
    energy.energiesPerU[1] = 140.0f;
    energy.energiesPerU[2] = 160.0f;
    
    // Peak depths (one per energy)
    energy.peakDepths.resize(energy.nEnergies);
    energy.peakDepths[0] = 10.0f;
    energy.peakDepths[1] = 12.0f;
    energy.peakDepths[2] = 14.0f;
    
    // Scale factors (one per energy)
    energy.scaleFacts.resize(energy.nEnergies);
    energy.scaleFacts[0] = 1.0f;
    energy.scaleFacts[1] = 1.0f;
    energy.scaleFacts[2] = 1.0f;
    
    // CIDD matrix: [energy][sample] = ciddMatrix[e * nEnergySamples + s]
    // Texture: width=nEnergySamples, height=nEnergies
    energy.ciddMatrix.resize(energy.nEnergySamples * energy.nEnergies);
    for (int e = 0; e < energy.nEnergies; ++e) {
        for (int s = 0; s < energy.nEnergySamples; ++s) {
            // Create a simple Gaussian-like dose distribution
            float depth = (float)s / energy.nEnergySamples * 30.0f;  // Max depth 30cm
            float peakDepth = energy.peakDepths[e];
            float value = exp(-(depth - peakDepth) * (depth - peakDepth) / (2.0f * 4.0f));
            energy.ciddMatrix[e * energy.nEnergySamples + s] = value * 10.0f;  // Scale to reasonable values
        }
    }
    
    // Density vector
    energy.nDensitySamples = 20;
    energy.densityVector.resize(energy.nDensitySamples);
    for (int i = 0; i < energy.nDensitySamples; ++i) {
        energy.densityVector[i] = 0.5f + i * 0.1f;
    }
    
    // Stopping power vector
    energy.nSpSamples = 20;
    energy.spVector.resize(energy.nSpSamples);
    for (int i = 0; i < energy.nSpSamples; ++i) {
        energy.spVector[i] = 1.0f + i * 0.05f;
    }
    
    // Radiation length vector
    energy.nRRlSamples = 20;
    energy.rRlVector.resize(energy.nRRlSamples);
    for (int i = 0; i < energy.nRRlSamples; ++i) {
        energy.rRlVector[i] = 0.1f + i * 0.01f;
    }
    
    return energy;
}

int main() {
    std::cout << "=== Wrapper Integration Test with Subspot to CPB Convolution ===" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Test parameters
    int3 imVolDims = {64, 64, 32};
    float3 imVolSpacing = {0.1f, 0.1f, 0.1f};
    float3 imVolOrigin = {-3.2f, -3.2f, 0.0f};
    
    int3 doseVolDims = {400, 400, 32};  // Match rayDims for proper dose accumulation
    float3 doseVolSpacing = {0.1f, 0.1f, 0.1f};
    float3 doseVolOrigin = {-20.0f, -20.0f, 0.0f};  // Match CPB grid coverage
    
    // Create test data
    std::cout << "Creating test data..." << std::endl;
    std::vector<float> imVolData = createTestCTData(imVolDims);
    std::vector<float> doseVolData(doseVolDims.x * doseVolDims.y * doseVolDims.z, 0.0f);
    
    RTDBeamSettings beam = createTestBeamSettings();
    RTDEnergyStruct energy = createTestEnergyData();
    
    std::cout << "Test data created:" << std::endl;
    std::cout << "  Image volume: " << imVolDims.x << "x" << imVolDims.y << "x" << imVolDims.z << std::endl;
    std::cout << "  Dose volume: " << doseVolDims.x << "x" << doseVolDims.y << "x" << doseVolDims.z << std::endl;
    std::cout << "  Energy layers: " << beam.energies.size() << std::endl;
    std::cout << "  Ray steps: " << beam.steps << std::endl;
    
    // Call the wrapper function
    std::cout << "\nCalling subsecondWrapper..." << std::endl;
    
    try {
        subsecondWrapper(
            imVolData.data(), imVolDims, imVolSpacing, imVolOrigin,
            doseVolData.data(), doseVolDims, doseVolSpacing, doseVolOrigin,
            &beam, 1, &energy,
            0, false, true  // gpuId=0, nuclearCorrection=false, fineTiming=true
        );
        
        std::cout << "Wrapper completed successfully!" << std::endl;
        
        // Analyze results
        float totalDose = 0.0f;
        float maxDose = 0.0f;
        int nonZeroCount = 0;
        
        // Debug: Check first few dose values
        std::cout << "\nDebug: First 10 dose values: ";
        for (int i = 0; i < 10 && i < doseVolData.size(); i++) {
            std::cout << doseVolData[i] << " ";
        }
        std::cout << std::endl;
        
        for (float dose : doseVolData) {
            if (dose > 1e-15f) {  // Even lower threshold to detect very small dose values
                nonZeroCount++;
                totalDose += dose;
                maxDose = std::max(maxDose, dose);
            }
        }
        
        std::cout << "\nDose Analysis:" << std::endl;
        std::cout << "  Total dose: " << totalDose << std::endl;
        std::cout << "  Max dose: " << maxDose << std::endl;
        std::cout << "  Non-zero voxels: " << nonZeroCount << "/" << doseVolData.size() << std::endl;
        
        if (nonZeroCount > 0) {
            std::cout << "  Average dose: " << totalDose / nonZeroCount << std::endl;
        }
        
        // Create output directory
        std::string outputDir = "output";
        mkdir(outputDir.c_str(), 0755);
        
        // Generate subspot data (same as in raytracedicom_wrapper.cu)
        int numLayers = beam.energies.size();
        int maxSubspotsPerLayer = 2;
        std::vector<float> subspotData(numLayers * maxSubspotsPerLayer * 5);
        std::default_random_engine generator(42);  // Same seed as wrapper
        std::uniform_real_distribution<float> deltaDist(-5.0f, 5.0f);
        std::uniform_real_distribution<float> weightDist(0.5f, 2.0f);
        std::uniform_real_distribution<float> sigmaDist(0.8f, 1.5f);
        
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
        
        // Beam parameters for visualization (simplified)
        float sad = beam.sourceDist.x;  // Use sourceDist.x as SAD
        float3 sourcePos = make_float3(0.0f, 0.0f, -sad);
        float3 beamDir = make_float3(0.0f, 0.0f, 1.0f);
        float3 bmXDir = make_float3(1.0f, 0.0f, 0.0f);
        float3 bmYDir = make_float3(0.0f, 1.0f, 0.0f);
        float refPlaneZ = doseVolOrigin.z;
        
        // Calculate subspot positions
        std::vector<std::vector<std::pair<float, float>>> subspotPositions(numLayers);
        for (int l = 0; l < numLayers; ++l) {
            for (int i = 0; i < maxSubspotsPerLayer; ++i) {
                int baseIdx = (l * maxSubspotsPerLayer + i) * 5;
                float deltaX = subspotData[baseIdx + 0];
                float deltaY = subspotData[baseIdx + 1];
                
                // Calculate reference plane center
                float3 refPlaneCenter = make_float3(
                    sourcePos.x + beamDir.x * sad,
                    sourcePos.y + beamDir.y * sad,
                    sourcePos.z + beamDir.z * sad
                );
                
                // Calculate subspot position
                float3 position = make_float3(
                    refPlaneCenter.x + bmXDir.x * deltaX + bmYDir.x * deltaY,
                    refPlaneCenter.y + bmXDir.y * deltaX + bmYDir.y * deltaY,
                    refPlaneCenter.z + bmXDir.z * deltaX + bmYDir.z * deltaY
                );
                
                subspotPositions[l].push_back({position.x, position.y});
            }
        }
        
        // Write configuration file
        std::string configFile = outputDir + "/test_output_config.txt";
        std::ofstream configStream(configFile);
        if (configStream.is_open()) {
            configStream << std::fixed << std::setprecision(6);
            configStream << "# Test Output Configuration\n";
            configStream << "# Generated by wrapper_integration_test\n\n";
            
            // Dose volume parameters
            configStream << "DOSE_VOL_DIMS = (" << doseVolDims.x << ", " << doseVolDims.y << ", " << doseVolDims.z << ")\n";
            configStream << "DOSE_VOL_SPACING = (" << doseVolSpacing.x << ", " << doseVolSpacing.y << ", " << doseVolSpacing.z << ")  # cm\n";
            configStream << "DOSE_VOL_ORIGIN = (" << doseVolOrigin.x << ", " << doseVolOrigin.y << ", " << doseVolOrigin.z << ")  # cm\n\n";
            
            // Energy layer parameters
            configStream << "NUM_LAYERS = " << numLayers << "\n";
            configStream << "MAX_SUBSPOTS_PER_LAYER = " << maxSubspotsPerLayer << "\n";
            configStream << "ENERGIES = [";
            for (size_t i = 0; i < beam.energies.size(); ++i) {
                configStream << beam.energies[i];
                if (i < beam.energies.size() - 1) configStream << ", ";
            }
            configStream << "]  # MeV\n\n";
            
            // Beam parameters
            configStream << "# Beam parameters\n";
            configStream << "SAD = " << sad << "  # Source-to-axis distance (cm)\n";
            configStream << "SOURCE_POS = [" << sourcePos.x << ", " << sourcePos.y << ", " << sourcePos.z << "]  # Simplified source position\n";
            configStream << "BEAM_DIR = [" << beamDir.x << ", " << beamDir.y << ", " << beamDir.z << "]  # Beam direction (along +Z)\n";
            configStream << "BM_X_DIR = [" << bmXDir.x << ", " << bmXDir.y << ", " << bmXDir.z << "]  # Beam X direction\n";
            configStream << "BM_Y_DIR = [" << bmYDir.x << ", " << bmYDir.y << ", " << bmYDir.z << "]  # Beam Y direction\n";
            configStream << "REF_PLANE_Z = " << refPlaneZ << "  # Reference plane Z coordinate\n\n";
            
            // Subspot positions
            configStream << "# Subspot positions (layer_idx, subspot_idx, x, y)\n";
            configStream << "SUBSPOT_POSITIONS = [\n";
            for (int l = 0; l < numLayers; ++l) {
                for (int i = 0; i < maxSubspotsPerLayer; ++i) {
                    configStream << "    (" << l << ", " << i << ", " 
                                << subspotPositions[l][i].first << ", " 
                                << subspotPositions[l][i].second << "),\n";
                }
            }
            configStream << "]\n";
            
            configStream.close();
            std::cout << "  Configuration saved to: " << configFile << std::endl;
        } else {
            std::cerr << "  Warning: Could not open " << configFile << " for writing" << std::endl;
        }
        
        // Output dose distribution to binary file for analysis
        std::cout << "\nWriting dose distribution to file..." << std::endl;
        std::string doseFile = outputDir + "/dose_distribution.bin";
        std::ofstream doseStream(doseFile, std::ios::binary);
        if (doseStream.is_open()) {
            // Write header: dimensions, spacing, origin
            doseStream.write(reinterpret_cast<const char*>(&doseVolDims.x), sizeof(int));
            doseStream.write(reinterpret_cast<const char*>(&doseVolDims.y), sizeof(int));
            doseStream.write(reinterpret_cast<const char*>(&doseVolDims.z), sizeof(int));
            doseStream.write(reinterpret_cast<const char*>(&doseVolSpacing.x), sizeof(float));
            doseStream.write(reinterpret_cast<const char*>(&doseVolSpacing.y), sizeof(float));
            doseStream.write(reinterpret_cast<const char*>(&doseVolSpacing.z), sizeof(float));
            doseStream.write(reinterpret_cast<const char*>(&doseVolOrigin.x), sizeof(float));
            doseStream.write(reinterpret_cast<const char*>(&doseVolOrigin.y), sizeof(float));
            doseStream.write(reinterpret_cast<const char*>(&doseVolOrigin.z), sizeof(float));
            
            // Write dose data (stored as x, y, z order in memory)
            doseStream.write(reinterpret_cast<const char*>(doseVolData.data()), 
                          doseVolData.size() * sizeof(float));
            doseStream.close();
            std::cout << "  Dose distribution saved to: " << doseFile << std::endl;
            std::cout << "  Dimensions: " << doseVolDims.x << " x " << doseVolDims.y << " x " << doseVolDims.z << std::endl;
            std::cout << "  Spacing: (" << doseVolSpacing.x << ", " << doseVolSpacing.y << ", " << doseVolSpacing.z << ") cm" << std::endl;
            std::cout << "  Origin: (" << doseVolOrigin.x << ", " << doseVolOrigin.y << ", " << doseVolOrigin.z << ") cm" << std::endl;
        } else {
            std::cerr << "  Warning: Could not open " << doseFile << " for writing" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\nTotal execution time: " << duration.count() << " μs" << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;
    
    return 0;
}


