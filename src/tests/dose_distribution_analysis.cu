#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>

// Include necessary headers
#include "../include/core/common.cuh"
#include "../include/core/forward_declarations.h"
#include "../include/core/raytracedicom_integration.h"

// Simple CSV writer for Excel compatibility
class CSVWriter {
private:
    std::ofstream file;
    
public:
    CSVWriter(const std::string& filename) : file(filename) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }
    
    ~CSVWriter() {
        if (file.is_open()) {
            file.close();
        }
    }
    
    template<typename T>
    CSVWriter& operator<<(const T& data) {
        file << data;
        return *this;
    }
    
    CSVWriter& operator<<(const char* str) {
        file << str;
        return *this;
    }
    
    CSVWriter& operator<<(const std::string& str) {
        file << str;
        return *this;
    }
    
    CSVWriter& endl() {
        file << "\n";
        return *this;
    }
    
    CSVWriter& comma() {
        file << ",";
        return *this;
    }
};

// Gaussian fitting functions
struct GaussianFit {
    float amplitude;
    float center_x;
    float center_y;
    float sigma_x;
    float sigma_y;
    float correlation;
    float residual_sum_squares;
    float r_squared;
};

// Simple Gaussian fitting (2D)
GaussianFit fitGaussian2D(const std::vector<float>& data, int width, int height, 
                         float x_min, float x_max, float y_min, float y_max) {
    GaussianFit fit = {0};
    
    // Calculate center of mass
    float total_dose = 0.0f;
    float weighted_x = 0.0f;
    float weighted_y = 0.0f;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float dose = data[idx];
            total_dose += dose;
            
            float world_x = x_min + (x + 0.5f) * (x_max - x_min) / width;
            float world_y = y_min + (y + 0.5f) * (y_max - y_min) / height;
            
            weighted_x += dose * world_x;
            weighted_y += dose * world_y;
        }
    }
    
    if (total_dose > 0) {
        fit.center_x = weighted_x / total_dose;
        fit.center_y = weighted_y / total_dose;
        fit.amplitude = total_dose;
        
        // Calculate sigma (standard deviation)
        float sum_x2 = 0.0f, sum_y2 = 0.0f, sum_xy = 0.0f;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float dose = data[idx];
                
                float world_x = x_min + (x + 0.5f) * (x_max - x_min) / width;
                float world_y = y_min + (y + 0.5f) * (y_max - y_min) / height;
                
                float dx = world_x - fit.center_x;
                float dy = world_y - fit.center_y;
                
                sum_x2 += dose * dx * dx;
                sum_y2 += dose * dy * dy;
                sum_xy += dose * dx * dy;
            }
        }
        
        fit.sigma_x = sqrtf(sum_x2 / total_dose);
        fit.sigma_y = sqrtf(sum_y2 / total_dose);
        fit.correlation = sum_xy / sqrtf(sum_x2 * sum_y2);
        
        // Calculate R-squared
        float total_variance = 0.0f;
        float residual_variance = 0.0f;
        float mean_dose = total_dose / (width * height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float dose = data[idx];
                
                float world_x = x_min + (x + 0.5f) * (x_max - x_min) / width;
                float world_y = y_min + (y + 0.5f) * (y_max - y_min) / height;
                
                // Gaussian model prediction
                float dx = world_x - fit.center_x;
                float dy = world_y - fit.center_y;
                float predicted = fit.amplitude * expf(-(dx*dx/(2*fit.sigma_x*fit.sigma_x) + 
                                                       dy*dy/(2*fit.sigma_y*fit.sigma_y)));
                
                total_variance += (dose - mean_dose) * (dose - mean_dose);
                residual_variance += (dose - predicted) * (dose - predicted);
            }
        }
        
        fit.r_squared = 1.0f - (residual_variance / total_variance);
        fit.residual_sum_squares = residual_variance;
    }
    
    return fit;
}

// Extract dose slice from 3D dose volume
std::vector<float> extractDoseSlice(const std::vector<float>& doseVol, 
                                   int3 doseVolDims, int slice_z) {
    std::vector<float> slice(doseVolDims.x * doseVolDims.y, 0.0f);
    
    for (int y = 0; y < doseVolDims.y; y++) {
        for (int x = 0; x < doseVolDims.x; x++) {
            int slice_idx = y * doseVolDims.x + x;
            int vol_idx = slice_z * doseVolDims.x * doseVolDims.y + slice_idx;
            slice[slice_idx] = doseVol[vol_idx];
        }
    }
    
    return slice;
}

// Find slice with maximum dose
int findMaxDoseSlice(const std::vector<float>& doseVol, int3 doseVolDims) {
    float max_dose = 0.0f;
    int max_slice = 0;
    
    for (int z = 0; z < doseVolDims.z; z++) {
        float slice_max = 0.0f;
        for (int y = 0; y < doseVolDims.y; y++) {
            for (int x = 0; x < doseVolDims.x; x++) {
                int idx = z * doseVolDims.x * doseVolDims.y + y * doseVolDims.x + x;
                slice_max = std::max(slice_max, doseVol[idx]);
            }
        }
        
        if (slice_max > max_dose) {
            max_dose = slice_max;
            max_slice = z;
        }
    }
    
    return max_slice;
}

// Create test data similar to wrapper_integration_test
std::vector<float> createTestCTData(int3 dims) {
    std::vector<float> data(dims.x * dims.y * dims.z);
    
    for (int z = 0; z < dims.z; z++) {
        for (int y = 0; y < dims.y; y++) {
            for (int x = 0; x < dims.x; x++) {
                int idx = z * dims.x * dims.y + y * dims.x + x;
                
                // Create a simple phantom: water-like material
                float center_x = dims.x / 2.0f;
                float center_y = dims.y / 2.0f;
                float center_z = dims.z / 2.0f;
                
                float dist = sqrtf((x - center_x) * (x - center_x) + 
                                 (y - center_y) * (y - center_y) + 
                                 (z - center_z) * (z - center_z));
                
                if (dist < dims.x / 4.0f) {
                    data[idx] = 1000.0f; // Water (HU = 0, so HU+1000 = 1000)
                } else {
                    data[idx] = 0.0f; // Air
                }
            }
        }
    }
    
    return data;
}

// Create test beam settings (similar to wrapper_integration_test)
RTDBeamSettings createTestBeamSettings() {
    RTDBeamSettings beam;
    
    // Basic beam parameters
    beam.sourceDist = make_float3(0.0f, 0.0f, -100.0f);
    beam.steps = 100;
    
    // Energy layers
    beam.energies.resize(1);
    beam.energies[0] = 150.0f; // 150 MeV
    
    // Spot positions (5x2 grid)
    beam.spotPositions.resize(10);
    int spot_idx = 0;
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 5; x++) {
            beam.spotPositions[spot_idx] = make_vec3f(
                (x - 2.0f) * 0.5f,  // -1.0, -0.5, 0.0, 0.5, 1.0
                (y - 0.5f) * 0.5f,  // -0.25, 0.25
                0.0f
            );
            spot_idx++;
        }
    }
    
    return beam;
}

// Create test energy data (similar to wrapper_integration_test)
RTDEnergyStruct createTestEnergyData() {
    RTDEnergyStruct energy;
    
    // IDD samples
    energy.nIddSamples = 100;
    energy.iddVector.resize(energy.nIddSamples);
    for (int i = 0; i < energy.nIddSamples; ++i) {
        energy.iddVector[i] = 1.0f - i * 0.01f; // Decreasing IDD
    }
    
    // Stopping power samples
    energy.nSpSamples = 20;
    energy.spVector.resize(energy.nSpSamples);
    for (int i = 0; i < energy.nSpSamples; ++i) {
        energy.spVector[i] = 1.0f + i * 0.05f;
    }
    
    // Radiation length samples
    energy.nRRlSamples = 20;
    energy.rRlVector.resize(energy.nRRlSamples);
    for (int i = 0; i < energy.nRRlSamples; ++i) {
        energy.rRlVector[i] = 0.1f + i * 0.01f;
    }
    
    return energy;
}

int main() {
    std::cout << "=== Dose Distribution Analysis ===" << std::endl;
    
    try {
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Create test CT data
        int3 ctDims = {64, 64, 32};
        std::vector<float> ctData = createTestCTData(ctDims);
        
        // Create test beam settings
        RTDBeamSettings beam = createTestBeamSettings();
        
        // Create test energy structure
        RTDEnergyStruct energy = createTestEnergyData();
        
        // Create test dose volume
        int3 doseVolDims = {400, 400, 32};
        float3 doseVolSpacing = {0.1f, 0.1f, 0.1f};
        float3 doseVolOrigin = {-20.0f, -20.0f, 0.0f};
        std::vector<float> doseVolData(doseVolDims.x * doseVolDims.y * doseVolDims.z, 0.0f);
        
        std::cout << "Running dose calculation..." << std::endl;
        
        // Run dose calculation using the same parameters as wrapper_integration_test
        int3 imVolDims = ctDims;
        float3 imVolSpacing = {0.1f, 0.1f, 0.1f};
        float3 imVolOrigin = {-3.2f, -3.2f, 0.0f};
        
        subsecondWrapper(
            ctData.data(), imVolDims, imVolSpacing, imVolOrigin,
            doseVolData.data(), doseVolDims, doseVolSpacing, doseVolOrigin,
            &beam, 1, &energy,
            0, false, true  // gpuId=0, nuclearCorrection=false, fineTiming=true
        );
        
        std::cout << "Dose calculation completed successfully!" << std::endl;
        
        // Find slice with maximum dose
        int maxSlice = findMaxDoseSlice(doseVolData, doseVolDims);
        std::cout << "Maximum dose slice: " << maxSlice << std::endl;
        
        // Extract dose slice
        std::vector<float> doseSlice = extractDoseSlice(doseVolData, doseVolDims, maxSlice);
        
        // Calculate slice statistics
        float totalDose = 0.0f, maxDose = 0.0f, minDose = 1e6f;
        int nonZeroCount = 0;
        
        for (float dose : doseSlice) {
            if (dose > 1e-10f) {
                totalDose += dose;
                maxDose = std::max(maxDose, dose);
                minDose = std::min(minDose, dose);
                nonZeroCount++;
            }
        }
        
        std::cout << "Slice " << maxSlice << " Statistics:" << std::endl;
        std::cout << "  Total dose: " << totalDose << std::endl;
        std::cout << "  Max dose: " << maxDose << std::endl;
        std::cout << "  Min dose: " << minDose << std::endl;
        std::cout << "  Non-zero voxels: " << nonZeroCount << "/" << doseSlice.size() << std::endl;
        
        // Fit Gaussian to dose distribution
        float x_min = doseVolOrigin.x;
        float x_max = doseVolOrigin.x + doseVolDims.x * doseVolSpacing.x;
        float y_min = doseVolOrigin.y;
        float y_max = doseVolOrigin.y + doseVolDims.y * doseVolSpacing.y;
        
        GaussianFit fit = fitGaussian2D(doseSlice, doseVolDims.x, doseVolDims.y, 
                                      x_min, x_max, y_min, y_max);
        
        std::cout << "\nGaussian Fit Results:" << std::endl;
        std::cout << "  Amplitude: " << fit.amplitude << std::endl;
        std::cout << "  Center X: " << fit.center_x << " cm" << std::endl;
        std::cout << "  Center Y: " << fit.center_y << " cm" << std::endl;
        std::cout << "  Sigma X: " << fit.sigma_x << " cm" << std::endl;
        std::cout << "  Sigma Y: " << fit.sigma_y << " cm" << std::endl;
        std::cout << "  Correlation: " << fit.correlation << std::endl;
        std::cout << "  R-squared: " << fit.r_squared << std::endl;
        std::cout << "  Residual Sum Squares: " << fit.residual_sum_squares << std::endl;
        
        // Determine if distribution is Gaussian-like
        bool isGaussian = (fit.r_squared > 0.8f) && (fit.sigma_x > 0.1f) && (fit.sigma_y > 0.1f);
        std::cout << "\nGaussian Distribution Assessment: " << (isGaussian ? "YES" : "NO") << std::endl;
        
        // Output to CSV file (Excel compatible)
        std::string filename = "dose_distribution_analysis.csv";
        CSVWriter csv(filename);
        
        // Write header
        csv << "Dose Distribution Analysis Results" << csv.endl();
        csv << "Slice Number" << csv.comma() << maxSlice << csv.endl();
        csv << "Total Dose" << csv.comma() << totalDose << csv.endl();
        csv << "Max Dose" << csv.comma() << maxDose << csv.endl();
        csv << "Min Dose" << csv.comma() << minDose << csv.endl();
        csv << "Non-zero Voxels" << csv.comma() << nonZeroCount << csv.endl();
        csv << csv.endl();
        
        csv << "Gaussian Fit Parameters" << csv.endl();
        csv << "Parameter" << csv.comma() << "Value" << csv.comma() << "Units" << csv.endl();
        csv << "Amplitude" << csv.comma() << fit.amplitude << csv.comma() << "dose units" << csv.endl();
        csv << "Center X" << csv.comma() << fit.center_x << csv.comma() << "cm" << csv.endl();
        csv << "Center Y" << csv.comma() << fit.center_y << csv.comma() << "cm" << csv.endl();
        csv << "Sigma X" << csv.comma() << fit.sigma_x << csv.comma() << "cm" << csv.endl();
        csv << "Sigma Y" << csv.comma() << fit.sigma_y << csv.comma() << "cm" << csv.endl();
        csv << "Correlation" << csv.comma() << fit.correlation << csv.comma() << "dimensionless" << csv.endl();
        csv << "R-squared" << csv.comma() << fit.r_squared << csv.comma() << "dimensionless" << csv.endl();
        csv << "Residual Sum Squares" << csv.comma() << fit.residual_sum_squares << csv.comma() << "dose units^2" << csv.endl();
        csv << "Is Gaussian" << csv.comma() << (isGaussian ? "YES" : "NO") << csv.comma() << "boolean" << csv.endl();
        csv << csv.endl();
        
        // Write dose distribution data
        csv << "Dose Distribution Data (X, Y, Dose)" << csv.endl();
        csv << "X (cm)" << csv.comma() << "Y (cm)" << csv.comma() << "Dose" << csv.endl();
        
        for (int y = 0; y < doseVolDims.y; y++) {
            for (int x = 0; x < doseVolDims.x; x++) {
                int idx = y * doseVolDims.x + x;
                float dose = doseSlice[idx];
                
                if (dose > 1e-10f) { // Only write non-zero doses
                    float world_x = x_min + (x + 0.5f) * doseVolSpacing.x;
                    float world_y = y_min + (y + 0.5f) * doseVolSpacing.y;
                    
                    csv << world_x << csv.comma() << world_y << csv.comma() << dose << csv.endl();
                }
            }
        }
        
        std::cout << "\nResults saved to: " << filename << std::endl;
        std::cout << "You can open this file in Excel for further analysis." << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}