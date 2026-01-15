/**
 * \file
 * \brief Test Application for RayTraceDicom Integration
 */

#include "raytracedicom_integration.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing RayTraceDicom Integration with Complete Kernel Functions" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    // Create test data
    int3 imVolDims = make_int3(64, 64, 64);
    float3 imVolSpacing = make_float3(1.0f, 1.0f, 1.0f);
    float3 imVolOrigin = make_float3(-32.0f, -32.0f, -32.0f);
    
    int3 doseVolDims = make_int3(64, 64, 64);
    float3 doseVolSpacing = make_float3(1.0f, 1.0f, 1.0f);
    float3 doseVolOrigin = make_float3(-32.0f, -32.0f, -32.0f);
    
    // Create image volume data (water phantom)
    size_t imVolSize = imVolDims.x * imVolDims.y * imVolDims.z;
    std::vector<float> imVolData(imVolSize, 0.0f); // Water (HU = 0)
    
    // Create dose volume data
    size_t doseVolSize = doseVolDims.x * doseVolDims.y * doseVolDims.z;
    std::vector<float> doseVolData(doseVolSize, 0.0f);
    
    // Create beam settings
    std::vector<RayTraceDicomBeamSettings> beamSettings = {*createRayTraceDicomBeamSettings()};
    
    // Create energy data
    RayTraceDicomEnergyStruct* energyData = createRayTraceDicomEnergyStruct();
    
    // Run RayTraceDicom wrapper
    raytraceDicomWrapper(imVolData.data(), imVolDims, imVolSpacing, imVolOrigin,
                        doseVolData.data(), doseVolDims, doseVolSpacing, doseVolOrigin,
                        beamSettings.data(), beamSettings.size(), energyData,
                        0, false, true);
    
    // Calculate and display final dose statistics
    float maxDose = 0.0f;
    float totalDose = 0.0f;
    for (size_t i = 0; i < doseVolSize; ++i) {
        maxDose = std::max(maxDose, doseVolData[i]);
        totalDose += doseVolData[i];
    }
    
    std::cout << std::endl;
    std::cout << "Final Dose Statistics:" << std::endl;
    std::cout << "  Maximum dose: " << maxDose << " Gy" << std::endl;
    std::cout << "  Total dose: " << totalDose << " Gy" << std::endl;
    std::cout << "  Average dose: " << totalDose / doseVolSize << " Gy" << std::endl;
    
    std::cout << std::endl;
    std::cout << "RayTraceDicom integration test completed successfully!" << std::endl;
    
    // Clean up
    destroyRayTraceDicomEnergyStruct(energyData);
    
    return 0;
}
