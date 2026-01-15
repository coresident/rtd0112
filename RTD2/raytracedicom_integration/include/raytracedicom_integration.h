/**
 * \file
 * \brief RayTraceDicom Integration - Main Header
 * 
 * Complete integration of RayTraceDicom project with all core components
 */

#ifndef RAYTRACEDICOM_INTEGRATION_H
#define RAYTRACEDICOM_INTEGRATION_H

#include <vector>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
#include <device_launch_parameters.h>

// Forward declarations
struct float2;
struct float3;
struct int3;
struct uint3;

// Constants from RayTraceDicom
#define HALF 0.5f
#define RAY_WEIGHT_CUTOFF 1e-6f
#define BP_DEPTH_CUTOFF 0.95f

// Main data structures
struct RayTraceDicomBeamSettings {
    std::vector<float> energies;           // Energy for each layer
    std::vector<float2> spotSigmas;        // (sigmax, sigmay) at iso in air for each energy layer
    float2 raySpacing;                     // Spacing between adjacent raytracing rays at iso
    unsigned int steps;                    // Number of raytracing steps
    float2 sourceDist;                     // Source to iso distance in x and y
    float3 spotOffset;                     // Spot offset in gantry coordinates
    float3 spotDelta;                      // Spot spacing in gantry coordinates
    float3 gantryToImOffset;              // Gantry to image transform offset
    float3 gantryToImMatrix;              // Gantry to image transform matrix (3x3 linear part)
    float3 gantryToDoseOffset;             // Gantry to dose transform offset
    float3 gantryToDoseMatrix;             // Gantry to dose transform matrix (3x3 linear part)
};

struct RayTraceDicomEnergyStruct {
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

// Main wrapper function declaration
extern "C" void raytraceDicomWrapper(
    const float* imVolData, const int3& imVolDims, const float3& imVolSpacing, const float3& imVolOrigin,
    float* doseVolData, const int3& doseVolDims, const float3& doseVolSpacing, const float3& doseVolOrigin,
    const RayTraceDicomBeamSettings* beamSettings, size_t numBeams,
    const RayTraceDicomEnergyStruct* energyData,
    int gpuId = 0, bool nuclearCorrection = false, bool fineTiming = false
);

// Helper functions
extern "C" RayTraceDicomBeamSettings* createRayTraceDicomBeamSettings();
extern "C" void destroyRayTraceDicomBeamSettings(RayTraceDicomBeamSettings* beam);
extern "C" RayTraceDicomEnergyStruct* createRayTraceDicomEnergyStruct();
extern "C" void destroyRayTraceDicomEnergyStruct(RayTraceDicomEnergyStruct* energy);

#endif // RAYTRACEDICOM_INTEGRATION_H
