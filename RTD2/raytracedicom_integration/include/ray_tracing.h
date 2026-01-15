/**
 * \file
 * \brief Ray Tracing Components for RayTraceDicom Integration
 * 
 * Includes BEV transforms, density and stopping power tracing
 */

#ifndef RAY_TRACING_H
#define RAY_TRACING_H

#include <cuda_runtime.h>
#include <vector>

// BEV transform structures
struct Float3ToBevTransform {
    float3 origin;
    float3 xAxis;
    float3 yAxis;
    float3 zAxis;
    
    Float3ToBevTransform() : origin(make_float3(0,0,0)), xAxis(make_float3(1,0,0)), 
                            yAxis(make_float3(0,1,0)), zAxis(make_float3(0,0,1)) {}
    
    __device__ __host__ float3 transform(const float3& point) const {
        float3 relative = make_float3(point.x - origin.x, point.y - origin.y, point.z - origin.z);
        return make_float3(
            relative.x * xAxis.x + relative.y * xAxis.y + relative.z * xAxis.z,
            relative.x * yAxis.x + relative.y * yAxis.y + relative.z * yAxis.z,
            relative.x * zAxis.x + relative.y * zAxis.y + relative.z * zAxis.z
        );
    }
};

struct DensityAndSpTracerParams {
    float densityScale;
    float spScale;
    unsigned int steps;
    Float3ToBevTransform transform;
    float2 raySpacing;
    float3 startPos;
    float3 stepInc;
    float stepLength;
    
    DensityAndSpTracerParams(float dScale, float sScale, unsigned int s, const Float3ToBevTransform& t) 
        : densityScale(dScale), spScale(sScale), steps(s), transform(t) {}
    
    __device__ __host__ float3 getStart(unsigned int x, unsigned int y) const {
        return make_float3(
            startPos.x + x * raySpacing.x,
            startPos.y + y * raySpacing.y,
            startPos.z
        );
    }
    
    __device__ __host__ float3 getInc(unsigned int x, unsigned int y) const {
        return stepInc;
    }
    
    __device__ __host__ float stepLen(unsigned int x, unsigned int y) const {
        return stepLength;
    }
    
    __device__ __host__ unsigned int getSteps() const { return steps; }
    __device__ __host__ float getDensityScale() const { return densityScale; }
    __device__ __host__ float getSpScale() const { return spScale; }
};

// Ray tracing kernel declarations
__global__ void fillBevDensityAndSpKernel(
    float* const bevDensity,
    float* const bevCumulSp, 
    int* const beamFirstInside, 
    int* const firstStepOutside, 
    const DensityAndSpTracerParams params,
    cudaTextureObject_t imVolTex, 
    cudaTextureObject_t densityTex, 
    cudaTextureObject_t stoppingPowerTex,
    const int3 imVolDims);

// Helper functions for ray tracing
Float3ToBevTransform createBevTransform(const float3& spotOffset, const float3& spotDelta, 
                                       const float3& gantryToImOffset, const float3& gantryToImMatrix,
                                       const float2& sourceDist, const float3& imVolOrigin);

DensityAndSpTracerParams createTracerParams(float densityScale, float spScale, unsigned int steps,
                                           const Float3ToBevTransform& transform, const float2& raySpacing,
                                           const float3& startPos, const float3& stepInc, float stepLength);

#endif // RAY_TRACING_H
