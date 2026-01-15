/**
 * \file
 * \brief Ray Tracing Implementation for RTD Integration
 */

#include "../include/core/raytracedicom_integration.h"
#include "../include/core/ray_tracing.h"
#include "../include/utils/utils.h"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

// Ray tracing kernel implementation
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
        pos.x += step.x;
        pos.y += step.y;
        pos.z += step.z;
    }
    
    beamFirstInside[y * gridDim.x * blockDim.x + x] = beforeFirstInside + 1;
    firstStepOutside[y * gridDim.x * blockDim.x + x] = lastInside + 1;
}

// Helper functions for ray tracing
Float3ToBevTransform createBevTransform(const float3& spotOffset, const float3& spotDelta, 
                                       const float3& gantryToImOffset, const float3& gantryToImMatrix,
                                       const float2& sourceDist, const float3& imVolOrigin) {
    Float3ToBevTransform transform;
    transform.origin = spotOffset;
    transform.xAxis = make_float3(1, 0, 0);
    transform.yAxis = make_float3(0, 1, 0);
    transform.zAxis = make_float3(0, 0, 1);
    return transform;
}

DensityAndSpTracerParams createTracerParams(float densityScale, float spScale, unsigned int steps,
                                           const Float3ToBevTransform& transform, const float2& raySpacing,
                                           const float3& startPos, const float3& stepInc, float stepLength) {
    DensityAndSpTracerParams params(densityScale, spScale, steps, transform);
    params.raySpacing = raySpacing;
    params.startPos = startPos;
    params.stepInc = stepInc;
    params.stepLength = stepLength;
    return params;
}
