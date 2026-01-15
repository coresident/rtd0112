/**
 * \file
 * \brief BEV发散坐标系下的射线追踪实现
 */

#include "../include/core/common.cuh"
#include "../include/core/ray_tracing.h"
#include "../include/algorithms/superposition.h"
#include "../include/algorithms/convolution.h"
#include "../include/algorithms/idd_sigma.h"
#include "../include/algorithms/fill_idd_and_sigma_params.cuh"
#include "../include/algorithms/transfer_param_struct_div3.cuh"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

// BEV发散坐标系下的射线追踪kernel
__global__ void rayTracingBEVKernel(
    float* bevDensity,           // 输出：BEV密度数组
    float* bevCumulSp,           // 输出：BEV累积停止功率
    float* bevIdd,               // 输出：BEV积分深度剂量
    float* bevRSigmaEff,         // 输出：BEV有效sigma倒数
    float* rayWeights,           // 输入：射线权重
    int* beamFirstInside,        // 输出：首次进入体模的步数
    int* firstStepOutside,       // 输出：首次离开体模的步数
    int* firstPassive,           // 输出：首次被动散射的步数
    DensityAndSpTracerParams params,
    FillIddAndSigmaParams iddParams,
    float* layerEnergy,          // 输入：层能量数组
    cudaTextureObject_t imVolTex,
    cudaTextureObject_t densityTex,
    cudaTextureObject_t stoppingPowerTex,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex,
    int3 imVolDims
) {
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Debug output removed for performance
    // if (x < 5 && y < 5) {
    //     printf("Debug: rayTracingBEVKernel called for thread (%d,%d)\n", x, y);
    // }
    
    // Calculate ray dimensions from grid and block dimensions
    const unsigned int rayDimsX = gridDim.x * blockDim.x;
    const unsigned int rayDimsY = gridDim.y * blockDim.y;
    const unsigned int rayDimsZ = params.getSteps();
    
    // Calculate memory step for 3D indexing: step * rayDimsY * rayDimsX + y * rayDimsX + x
    const unsigned int memStep = rayDimsY * rayDimsX;
    unsigned int idx = y * rayDimsX + x;

    if (x >= rayDimsX || y >= rayDimsY) return;

    // 补偿体素值位于体素中心而不是角点
    vec3f startPos = vec3f(params.getStart(x, y).x, params.getStart(x, y).y, params.getStart(x, y).z);
    vec3f pos = vec3f(startPos.x + HALF, startPos.y + HALF, startPos.z + HALF);
    vec3f step = vec3f(params.getInc(x, y).x, params.getInc(x, y).y, params.getInc(x, y).z);
    float stepLen = params.stepLen(x, y);
    
    float cumulSp = 0.0f;
    float cumulHuPlus1000 = 0.0f;
    int beforeFirstInside = -1;
    int lastInside = -1;
    int firstPassiveStep = -1;

    // 射线追踪循环
    for (unsigned int i = 0; i < params.getSteps(); ++i) {
        // 边界检查：确保纹理坐标在有效范围内
        if (pos.x < 0.0f || pos.x >= imVolDims.x || 
            pos.y < 0.0f || pos.y >= imVolDims.y || 
            pos.z < 0.0f || pos.z >= imVolDims.z) {
            break; // 超出边界，停止射线追踪
        }
        
        float huPlus1000 = tex3D<float>(imVolTex, pos.x, pos.y, pos.z);
        cumulHuPlus1000 += huPlus1000;
        
        // density&SP
        float density = tex1D<float>(densityTex, huPlus1000 * params.getDensityScale() + HALF);
        float stoppingPower = tex1D<float>(stoppingPowerTex, huPlus1000 * params.getSpScale() + HALF);
        
        bevDensity[idx] = density;
        cumulSp += stepLen * stoppingPower;
        bevCumulSp[idx] = cumulSp;

        // 跟踪进入/离开体模的步数
        // RayTracedicom算法：使用HU+1000来判断if still inside
        // beforeFirstInside: 第一个HU+1000 > 150之前的步数；即进入体模之前的最后一个step
        // lastInside: HU+1000 > 150的最后一个步数（即离开体模之前的最后一个步数）
        if (beforeFirstInside == -1 && huPlus1000 > 150.0f) {
            // Found first inside voxel, beforeFirstInside is the step before this
            beforeFirstInside = (i > 0) ? (i - 1) : 0;
        }
        if (huPlus1000 > 150.0f) {
            lastInside = i;
        }
        
        // Note: IDD and sigma calculation is done separately in fillIddAndSigma kernel
        // This kernel only calculates density and cumulative stopping power
        
        idx += memStep;
        pos.x += step.x;
        pos.y += step.y;
        pos.z += step.z;
    }
    
    beamFirstInside[y * rayDimsX + x] = beforeFirstInside + 1;
    firstStepOutside[y * rayDimsX + x] = lastInside + 1;
    firstPassive[y * rayDimsX + x] = firstPassiveStep + 1;
}

// BEV到剂量网格的转换kernel
__global__ void bevToDoseGridKernel(
    float* doseGrid,             // final dose grid
    float* bevDose,              // input BEV dose array 
    TransferParamStructDiv3 params,
    int3 startIdx,
    int maxZ,
    uint3 doseDims,
    cudaTextureObject_t bevDoseTex
) {
    unsigned int x = startIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y) {
        params.init(x, y);
        float *res = doseGrid + startIdx.z * doseDims.x * doseDims.y + y * doseDims.x + x;
        
        for (int z = startIdx.z; z <= maxZ; ++z) {
            vec3f pos = params.getFanIdx(z) + vec3f(HALF, HALF, HALF);
            float dose = tex3D<float>(bevDoseTex, pos.x, pos.y, pos.z);
            
            if (dose > 0.0f) {
                *res += dose;
            }
            res += doseDims.x * doseDims.y;
        }
    }
}

// 主机函数：执行完整的BEV射线追踪
int performBEVRayTracing(
    float* d_bevDensity,
    float* d_bevCumulSp,
    float* d_bevIdd,
    float* d_bevRSigmaEff,
    float* d_rayWeights,
    int* d_beamFirstInside,
    int* d_firstStepOutside,
    int* d_firstPassive,
    DensityAndSpTracerParams densityParams,
    FillIddAndSigmaParams iddParams,
    float* layerEnergy,
    cudaTextureObject_t imVolTex,
    cudaTextureObject_t densityTex,
    cudaTextureObject_t stoppingPowerTex,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex,
    int3 imVolDims,
    int gpuId
) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaSetDevice(gpuId);
    
    // 设置网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((imVolDims.x + blockSize.x - 1) / blockSize.x,
                  (imVolDims.y + blockSize.y - 1) / blockSize.y);
    
    // 执行射线追踪
    rayTracingBEVKernel<<<gridSize, blockSize>>>(
        d_bevDensity,
        d_bevCumulSp,
        d_bevIdd,
        d_bevRSigmaEff,
        d_rayWeights,
        d_beamFirstInside,
        d_firstStepOutside,
        d_firstPassive,
        densityParams,
        iddParams,
        layerEnergy,
        imVolTex,
        densityTex,
        stoppingPowerTex,
        cumulIddTex,
        rRadiationLengthTex,
        imVolDims
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error in BEV ray tracing: %s\n", cudaGetErrorString(error));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("[TIMING] performBEVRayTracing_ERROR: %ld μs\n", duration.count());
        return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("[TIMING] performBEVRayTracing: %ld μs\n", duration.count());
    return 1;
}

// 主机函数：执行BEV到剂量网格的转换
int performBEVToDoseGridTransfer(
    float* d_doseGrid,
    float* d_bevDose,
    TransferParamStructDiv3 transferParams,
    int3 startIdx,
    int maxZ,
    uint3 doseDims,
    cudaTextureObject_t bevDoseTex,
    int gpuId
) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaSetDevice(gpuId);
    
    // 设置网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((doseDims.x + blockSize.x - 1) / blockSize.x,
                  (doseDims.y + blockSize.y - 1) / blockSize.y);
    
    // 执行转换
    bevToDoseGridKernel<<<gridSize, blockSize>>>(
        d_doseGrid,
        d_bevDose,
        transferParams,
        startIdx,
        maxZ,
        doseDims,
        bevDoseTex
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error in BEV to dose grid transfer: %s\n", cudaGetErrorString(error));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("[TIMING] performBEVToDoseGridTransfer_ERROR: %ld μs\n", duration.count());
        return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("[TIMING] performBEVToDoseGridTransfer: %ld μs\n", duration.count());
    return 1;
}
