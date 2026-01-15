/**
 * \file
 * \brief Simplified protons wrapper header file for C++ compilation interface
 * 
 * This header defines the simplified interface for protons calculation,
 * designed to be used as a C++ compilation interface module.
 */

#ifndef SIMPLIFIED_PROTONS_WRAPPER_CUH
#define SIMPLIFIED_PROTONS_WRAPPER_CUH

#include <vector>
#include <cuda_runtime.h>

// Forward declarations for CUDA types
struct float2;
struct float3;
struct int3;

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

// Main wrapper function declaration
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
);

// Helper function declarations for creating and destroying data structures
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
);

extern "C" void destroySimplifiedBeamSettings(SimplifiedBeamSettings* beam);

extern "C" SimplifiedEnergyStruct* createSimplifiedEnergyStruct(
    int nEnergySamples, int nEnergies,
    float* energiesPerU, float* peakDepths, float* scaleFacts, float* ciddMatrix,
    int nDensitySamples, float densityScaleFact, float* densityVector,
    int nSpSamples, float spScaleFact, float* spVector,
    int nRRlSamples, float rRlScaleFact, float* rRlVector
);

extern "C" void destroySimplifiedEnergyStruct(SimplifiedEnergyStruct* energy);

// Utility functions for data conversion
namespace SimplifiedProtonsUtils {
    
    /**
     * \brief Convert 3D array from row-major to column-major format
     * \param input Input array in row-major format
     * \param output Output array in column-major format
     * \param dims Dimensions of the 3D array
     */
    void convertRowMajorToColumnMajor(const float* input, float* output, const int3& dims);
    
    /**
     * \brief Convert 3D array from column-major to row-major format
     * \param input Input array in column-major format
     * \param output Output array in row-major format
     * \param dims Dimensions of the 3D array
     */
    void convertColumnMajorToRowMajor(const float* input, float* output, const int3& dims);
    
    /**
     * \brief Validate input parameters for the wrapper function
     * \param imVolData Image volume data pointer
     * \param imVolDims Image volume dimensions
     * \param doseVolData Dose volume data pointer
     * \param doseVolDims Dose volume dimensions
     * \param beamSettings Beam settings pointer
     * \param numBeams Number of beams
     * \param energyData Energy data pointer
     * \return true if parameters are valid, false otherwise
     */
    bool validateInputParameters(
        const float* imVolData, const int3& imVolDims,
        const float* doseVolData, const int3& doseVolDims,
        const SimplifiedBeamSettings* beamSettings, int numBeams,
        const SimplifiedEnergyStruct* energyData
    );
    
    /**
     * \brief Calculate optimal CUDA grid and block dimensions
     * \param volumeDims Volume dimensions
     * \param blockSize Preferred block size
     * \return Optimal grid dimensions
     */
    dim3 calculateOptimalGrid(const int3& volumeDims, const dim3& blockSize);
    
    /**
     * \brief Get CUDA device information
     * \param deviceId GPU device ID
     * \return Device name as string
     */
    std::string getCudaDeviceInfo(int deviceId);
}

#endif // SIMPLIFIED_PROTONS_WRAPPER_CUH

