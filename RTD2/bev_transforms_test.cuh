#pragma once

// BEV transforms (adapted from RayTraceDicom fan/divergent transforms)
// Note: We mirror the mathematical approach from RayTraceDicom but use naming consistent with cudaCalDose/deviceCalDose:
// - "BEV" equals the original "fan/divergent" concept
// - Variable names follow our codebase style
// - Detailed English comments explain the mapping

#include <cuda_runtime.h>
#include <math.h>

// Define CUDA_CALLABLE_MEMBER macro if not already defined
#ifndef CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER __device__ __host__
#endif

// Forward declarations for CUDA types
struct float2;
struct float3;
struct int3;

// Define basic CUDA types if not already defined
#ifndef __CUDA_ARCH__
struct float2 {
    float x, y;
    __host__ __device__ float2() : x(0), y(0) {}
    __host__ __device__ float2(float x_, float y_) : x(x_), y(y_) {}
};

struct float3 {
    float x, y, z;
    __host__ __device__ float3() : x(0), y(0), z(0) {}
    __host__ __device__ float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    __host__ __device__ float3 operator+(const float3& other) const {
        return float3(x + other.x, y + other.y, z + other.z);
    }
    
    __host__ __device__ float3 operator-(const float3& other) const {
        return float3(x - other.x, y - other.y, z - other.z);
    }
    
    __host__ __device__ float3 operator*(float scalar) const {
        return float3(x * scalar, y * scalar, z * scalar);
    }
};

struct int3 {
    int x, y, z;
    __host__ __device__ int3() : x(0), y(0), z(0) {}
    __host__ __device__ int3(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};
#endif

// Simple Matrix3x3 class for 3x3 matrix operations
class Matrix3x3 {
public:
    float m[3][3];
    
    CUDA_CALLABLE_MEMBER Matrix3x3() {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    CUDA_CALLABLE_MEMBER Matrix3x3(float val) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m[i][j] = val;
            }
        }
    }
    
    CUDA_CALLABLE_MEMBER Matrix3x3 inverse() const {
        // Simple inverse for 3x3 matrix (assuming it's invertible)
        Matrix3x3 result;
        float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        
        if (fabs(det) < 1e-6f) {
            // Return identity matrix if determinant is too small
            return Matrix3x3();
        }
        
		float invDet = 1.0f / det;
        
        result.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
        result.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
        result.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
        result.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet;
        result.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
        result.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet;
        result.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
        result.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet;
        result.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;
        
        return result;
    }
    
    CUDA_CALLABLE_MEMBER float3 operator*(const float3& v) const {
        return make_float3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }
};

// Forward declarations
struct float3;
struct float2;
struct int3;

// Additional type definitions
struct vec3f {
    float x, y, z;
    __host__ __device__ vec3f() : x(0), y(0), z(0) {}
    __host__ __device__ vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct vec3i {
    int x, y, z;
    __host__ __device__ vec3i() : x(0), y(0), z(0) {}
    __host__ __device__ vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};

struct Grid {
    float3 resolution;
    float3 corner;
    __host__ __device__ Grid() : resolution(make_float3(1.0f, 1.0f, 1.0f)), corner(make_float3(0.0f, 0.0f, 0.0f)) {}
};

struct DoseGrid {
    float3 resolution;
    float3 corner;
    __host__ __device__ DoseGrid() : resolution(make_float3(1.0f, 1.0f, 1.0f)), corner(make_float3(0.0f, 0.0f, 0.0f)) {}
};

struct deviceBevCoordSystem {
    float3 origin;
    float3 direction;
    float3 up;
    float3 right;
    __host__ __device__ deviceBevCoordSystem() : origin(make_float3(0.0f, 0.0f, 0.0f)), 
                                                 direction(make_float3(0.0f, 0.0f, 1.0f)), 
                                                 up(make_float3(0.0f, 1.0f, 0.0f)), 
                                                 right(make_float3(1.0f, 0.0f, 0.0f)) {}
};

struct spTracerParams {
    unsigned int steps;
    float stepLength;
    float densityThreshold;
    
    __host__ __device__ spTracerParams() : steps(100), stepLength(1.0f), densityThreshold(150.0f) {}
    __host__ __device__ spTracerParams(unsigned int s, float sl, float dt) : steps(s), stepLength(sl), densityThreshold(dt) {}
};

class BeamTracer {
public:
    __host__ __device__ BeamTracer() {}
    __host__ __device__ ~BeamTracer() {}
};

// Density and stopping power tracing parameters
struct DensityAndSpTracerParams {
    float densityScale;
    float spScale;
    unsigned int steps;
    Float3ToBevTransform_test bevTransform;
    
    __host__ __device__ DensityAndSpTracerParams() : densityScale(1.0f), spScale(1.0f), steps(100) {}
    __host__ __device__ DensityAndSpTracerParams(float ds, float ss, unsigned int s, const Float3ToBevTransform_test& bt) 
        : densityScale(ds), spScale(ss), steps(s), bevTransform(bt) {}
    
    __host__ __device__ float3 getStart(int x, int y) const {
        // Simplified start position calculation
        return make_float3(static_cast<float>(x), static_cast<float>(y), 0.0f);
    }
    
    __host__ __device__ float3 getInc(int x, int y) const {
        // Simplified increment calculation
        return make_float3(0.0f, 0.0f, 1.0f);
    }
    
    __host__ __device__ float stepLen(int x, int y) const {
        // Simplified step length
        return 1.0f;
    }
    
    __host__ __device__ float getDensityScale() const { return densityScale; }
    __host__ __device__ float getSpScale() const { return spScale; }
    __host__ __device__ unsigned int getSteps() const { return steps; }
};





// Constants
#define HALF 0.5f
#define DENSITY_SCALE 0.001f

class float3_affine_test {
public:
    CUDA_CALLABLE_MEMBER float3_affine_test() : matrix3x3(1.0f), offset(make_float3(0.0f, 0.0f, 0.0f)) {}
    
    CUDA_CALLABLE_MEMBER float3_affine_test(const Matrix3x3& mat, const float3& ofst) : matrix3x3(mat), offset(ofst) {}

    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3& p) const {
        // Apply the 3x3 matrix transformation, then add the offset
        float3 transformed = matrix3x3 * p;
        return make_float3(transformed.x + offset.x, transformed.y + offset.y, transformed.z + offset.z);
    }

    CUDA_CALLABLE_MEMBER float3_affine_test inverse() const {
        // Compute inverse of the 3x3 matrix
        Matrix3x3 invMatrix = matrix3x3.inverse();
        
        // Compute the inverse offset (-R^-1 * t)
        float3 negOffset = make_float3(-offset.x, -offset.y, -offset.z);
        float3 invOffset = invMatrix * negOffset;
        
        return float3_affine_test(invMatrix, invOffset);
    }

    CUDA_CALLABLE_MEMBER Matrix3x3 getMatrix() const { return matrix3x3; }
    CUDA_CALLABLE_MEMBER float3 getOffset() const { return offset; }

private:
    Matrix3x3 matrix3x3;
    float3 offset;
};

class float3IdxTransform_test {
	// Index-to-gantry transform (RayTraceDicom Float3IdxTransform)
	// p_out = offset + delta * p_in
public:
    CUDA_CALLABLE_MEMBER float3IdxTransform_test() : delta(make_float3(1.0f, 1.0f, 1.0f)), offset(make_float3(0.0f, 0.0f, 0.0f)) {}
    
    CUDA_CALLABLE_MEMBER float3IdxTransform_test(const float3& d, const float3& o) : delta(d), offset(o) {}

    CUDA_CALLABLE_MEMBER float3 transformPoint(const float3& idx) const {
        return make_float3(offset.x + delta.x * idx.x,
                          offset.y + delta.y * idx.y,
                          offset.z + delta.z * idx.z);
	}

	CUDA_CALLABLE_MEMBER float3IdxTransform_test inverse() const {
        float3IdxTransform_test inv;
        inv.delta = make_float3(1.0f / delta.x, 1.0f / delta.y, 1.0f / delta.z);
        inv.offset = make_float3(-offset.x * inv.delta.x,
                                 -offset.y * inv.delta.y,
                                 -offset.z * inv.delta.z);
		return inv;
	}

    CUDA_CALLABLE_MEMBER float3IdxTransform_test shiftOffset(const float3& s) const {
		float3IdxTransform_test r = *this;
		r.offset.x += s.x;
		r.offset.y += s.y;
		r.offset.z += s.z;
		return r;
	}

    CUDA_CALLABLE_MEMBER float3 getDelta() const { return delta; }
    CUDA_CALLABLE_MEMBER float3 getOffset() const { return offset; }

private: 
    float3 delta;   // spacing in each dimension
    float3 offset;  // origin shift in gantry coordinates
};

// Float3ToBevTransform: image-index -> gantry -> perspective to BEV -> BEV-index
struct Float3ToBevTransform_test {
    float3_affine_test imIdxToGantry;        // image-index to gantry coordinates
    float2 sourceDist;                        // source to isocenter distance
    float3IdxTransform_test bevToBevIdx;     // BEV coordinates to BEV index
    float3 imgCorner;                         // image corner for coordinate system
    
    __host__ __device__ Float3ToBevTransform_test() : sourceDist(make_float2(1000.0f, 1000.0f), imgCorner(make_float3(0.0f, 0.0f, 0.0f)) {}
    
    __host__ __device__ Float3ToBevTransform_test(const float3_affine_test& imgToGantry, const float2& srcDist, const float3& corner) 
        : imIdxToGantry(imgToGantry), sourceDist(srcDist), imgCorner(corner) {}
    
    __host__ __device__ float3 transformPoint(const float3& imIdx) const {
        // Transform image index to gantry coordinates
        float3 gantryPos = imIdxToGantry.transformPoint(imIdx);
        
        // Transform gantry coordinates to BEV coordinates (perspective projection)
        float3 bevPos;
        if (sourceDist.x > 0.0f && sourceDist.y > 0.0f) {
            bevPos.x = gantryPos.x * sourceDist.x / (sourceDist.x - gantryPos.z);
            bevPos.y = gantryPos.y * sourceDist.y / (sourceDist.y - gantryPos.z);
            bevPos.z = gantryPos.z;
        } else {
            bevPos = gantryPos;
        }
        
        // Transform BEV coordinates to BEV index
        return bevToBevIdx.transformPoint(bevPos);
    }
    
    __host__ __device__ imgFromBevTransform_test inverse() const;
};

// Implementation of inverse functions
inline imgFromBevTransform_test Float3ToBevTransform_test::inverse() const {
    return imgFromBevTransform_test(bevToBevIdx.inverse(), sourceDist, imIdxToGantry.inverse(), imgCorner);
}

inline Float3ToBevTransform_test imgFromBevTransform_test::inverse() const {
    return Float3ToBevTransform_test(gantryToImIdx.inverse(), sourceDist, bevIdxToBev.inverse(), imgCorner);
}

// Float3FromBevTransform: BEV-index -> BEV -> inverse perspective -> gantry -> image-index
struct imgFromBevTransform_test {
    float3IdxTransform_test bevIdxToBev;     // BEV index to BEV coordinates
    float2 sourceDist;                        // source to isocenter distance
    float3_affine_test gantryToImIdx;        // gantry coordinates to image index
    float3 imgCorner;                         // image corner for coordinate system
    
    __host__ __device__ imgFromBevTransform_test() : sourceDist(make_float2(1000.0f, 1000.0f), imgCorner(make_float3(0.0f, 0.0f, 0.0f)) {}
    
    __host__ __device__ imgFromBevTransform_test(const float3IdxTransform_test& bevToBev, const float2& srcDist, const float3_affine_test& gantryToImg, const float3& corner) 
        : bevIdxToBev(bevToBev), sourceDist(srcDist), gantryToImIdx(gantryToImg), imgCorner(corner) {}
    
    __host__ __device__ float3 transformPoint(const float3& bevIdx) const {
        // Transform BEV index to BEV coordinates
        float3 bevPos = bevIdxToBev.transformPoint(bevIdx);
        
        // Transform BEV coordinates to gantry coordinates (inverse perspective)
        float3 gantryPos;
        if (sourceDist.x > 0.0f && sourceDist.y > 0.0f) {
            gantryPos.x = bevPos.x * (sourceDist.x - bevPos.z) / sourceDist.x;
            gantryPos.y = bevPos.y * (sourceDist.y - bevPos.z) / sourceDist.y;
            gantryPos.z = bevPos.z;
        } else {
            gantryPos = bevPos;
        }
        
        // Transform gantry coordinates to image index
        return gantryToImIdx.transformPoint(gantryPos);
    }
    
    __host__ __device__ Float3ToBevTransform_test inverse() const;
};

// Helper function for world to BEV transformation
__host__ __device__ inline float3 worldToBevDevice(const float3& worldPos, const deviceBevCoordSystem& coordSys) {
    // Simplified transformation - you may need to implement the full logic
    float3 localPos = worldPos - coordSys.origin;
    float3 bevPos;
    bevPos.x = localPos.x * coordSys.right.x + localPos.y * coordSys.right.y + localPos.z * coordSys.right.z;
    bevPos.y = localPos.x * coordSys.up.x + localPos.y * coordSys.up.y + localPos.z * coordSys.up.z;
    bevPos.z = localPos.x * coordSys.direction.x + localPos.y * coordSys.direction.y + localPos.z * coordSys.direction.z;
    return bevPos;
}

// BevTransformUtils namespace for helper functions
namespace BevTransformUtils {
    __host__ __device__ inline Float3ToBevTransform_test createBevTransform(
        const float3& spotOffset, const float3& spotDelta, 
        const float3& gantryToImOffset, const float3& gantryToImMatrix,
        const float2& sourceDist, const float3& imVolOrigin) {
        
        // Create simplified BEV transform
        Float3ToBevTransform_test transform;
        transform.sourceDist = sourceDist;
        transform.imgCorner = imVolOrigin;
        
        // Set up basic transforms (simplified)
        transform.imIdxToGantry = float3_affine_test();
        transform.bevToBevIdx = float3IdxTransform_test();
        
        return transform;
    }
    
    __host__ __device__ inline float3 imageToBev(const float3& imgPos, const Float3ToBevTransform_test& transform) {
        return transform.transformPoint(imgPos);
    }
    
    __host__ __device__ inline float3 bevToImage(const float3& bevPos, const imgFromBevTransform_test& transform) {
        return transform.transformPoint(bevPos);
    }
}



// Texture interpolation kernel for BEV dose calculation
__global__ void textureInterpolationKernel(float* bevDoseTexture,
                                          const float* doseInDivergentCoord,
                                          const deviceBevCoordSystem* coordSystems,
                                          const int* beamGroupIds,
                                          vec3i* roiIndex,
                                          int nRoi,
                                          int nBeam,
                                          Grid doseGrid,
                                          int texWidth,
                                          int texHeight,
                                          int texDepth) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    
    while (tid < nRoi * nBeam) {
        int roiIdx = tid / nBeam;
        int beamIdx = tid % nBeam;
        
        if (roiIdx >= nRoi || beamIdx >= nBeam) {
            tid += totalThreads;
            continue;
        }
        
        // Get world coordinates
        vec3f worldPos = vec3f(roiIndex[roiIdx].x * doseGrid.resolution.x + doseGrid.corner.x,
                              roiIndex[roiIdx].y * doseGrid.resolution.y + doseGrid.corner.y,
                              roiIndex[roiIdx].z * doseGrid.resolution.z + doseGrid.corner.z);
        
        // Transform to BEV divergent coordinate system
        int groupId = beamGroupIds[beamIdx];
        deviceBevCoordSystem coordSys = coordSystems[groupId];
        vec3f bevPos = worldToBevDevice(worldPos, coordSys);
        
        // Calculate texture coordinates
        float texX = (bevPos.x + texWidth/2.0f) / texWidth * texWidth;
        float texY = (bevPos.y + texHeight/2.0f) / texHeight * texHeight;
        float texZ = (bevPos.z + texDepth/2.0f) / texDepth * texDepth;
        
        // Ensure within texture bounds
        texX = fmaxf(0.0f, fminf(texWidth - 1.0f, texX));
        texY = fmaxf(0.0f, fminf(texHeight - 1.0f, texY));
        texZ = fmaxf(0.0f, fminf(texDepth - 1.0f, texZ));
        
        // Store to BEV texture
        int texIdx = int(texZ) * texWidth * texHeight + int(texY) * texWidth + int(texX);
        int doseIdx = roiIdx * nBeam + beamIdx;
        bevDoseTexture[texIdx] = doseInDivergentCoord[doseIdx];
        
        tid += totalThreads;
    }
}

// Density and stopping power tracing kernel
__global__ void densityAndSpTracer(
    float* const devDensity,
    float* const devCumulSp,
    float* const devCumulWepl,
    int* const devFirstInside,
    int* const devFirstOutside,
    float3* const devBeamDirection,
    float* const devIdBeamXY, // from idbeamxy
    int3* const devRoiIdx,
    struct DoseGrid doseGrid,
    spTracerParams params,
    cudaTextureObject_t ctDataTex,
    cudaTextureObject_t densityTex,
    cudaTextureObject_t stoppingPowerTex,
    int nBeams,
    int nRoi = 1) {
    
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y * blockDim.y * gridDim.x * blockDim.x;
    unsigned int beamIdx = y * gridDim.x * blockDim.x + x; // named after 'beam' but from initial to trace the following steps
    
    if (beamIdx >= nBeams) return;
    
    float ix = devIdBeamXY[beamIdx * 2];
    float iy = devIdBeamXY[beamIdx * 2 + 1];
  
    // Initialize tracer (you'll need to implement BeamTracer class)
    // BeamTracer tracer(devBeamDirection[beamIdx],
    //                   ix, iy,
    //                   make_float3(0, 0, 0),  // sourcePos
    //                   make_float3(1, 0, 0),  // bmxDir
    //                   make_float3(0, 1, 0),  // bmyDir
    //                   doseGrid,
    //                   ctDataTex,
    //                   densityTex,
    //                   stoppingPowerTex);
    
    // For now, using simplified approach
    float3 pos = make_float3(ix, iy, 0.0f);
    float3 step = make_float3(0.0f, 0.0f, 1.0f);
    float stepLen = 1.0f;

    float cumulWepl = 0.0f;
    float cumulSp = 0.0f;
    float cumulHuPlus1000 = 0.0f; // sliding variable for boundary detection
    int beforeFirstInside = -1, lastInside = -1;
    
    for (unsigned int z = 0; z < params.steps; ++z) {
        float stoppingPower, wepl;
        
        // Get voxel info
        float huPlus1000 = tex3D<float>(ctDataTex, pos.x, pos.y, pos.z);
        cumulHuPlus1000 += huPlus1000;

        devDensity[beamIdx] = tex1D<float>(densityTex, huPlus1000 * DENSITY_SCALE + HALF);

        // Calculate WEPL (Water Equivalent Path Length)
        wepl = stepLen * tex1D<float>(stoppingPowerTex, huPlus1000 * DENSITY_SCALE + HALF) * 
                devDensity[beamIdx] / tex1D<float>(stoppingPowerTex, 1000.0f * DENSITY_SCALE + HALF);

        cumulWepl += wepl;
        cumulSp += stepLen * tex1D<float>(stoppingPowerTex, huPlus1000 * DENSITY_SCALE + HALF);

        // Boundary detection
        if (cumulHuPlus1000 < 150.0f) {
            beforeFirstInside = z;
        }
        if (huPlus1000 > 150.0f) {
            lastInside = z;
        }

        devCumulSp[beamIdx] = cumulSp;
        devCumulWepl[beamIdx] = cumulWepl;

        beamIdx += memStep;
        pos.x += step.x;
        pos.y += step.y;
        pos.z += step.z;
    }
    
    devFirstInside[y * gridDim.x * blockDim.x + x] = beforeFirstInside + 1;
    devFirstOutside[y * gridDim.x * blockDim.x + x] = lastInside + 1;
}

// Implementation of inverse function
inline Float3ToBevTransform_test Float3ToBevTransform_test::inverse() const {
    return imgFromBevTransform_test(bevToBevIdx.inverse(), sourceDist, imIdxToGantry.inverse(), imgCorner);
}

// Additional utility functions for BEV transforms
namespace BevTransformUtils {
    
    // Create BEV transform from simplified beam settings
    CUDA_CALLABLE_MEMBER Float3ToBevTransform_test createBevTransform(
        const float3& spotOffset,
        const float3& spotDelta,
        const float3& gantryToImOffset,
        const float3& gantryToImMatrix,
        const float2& sourceDist,
        const float3& imgCorner) {
        
        // Create index transform for BEV coordinates
        float3IdxTransform_test bevToBevIdx(spotDelta, spotOffset);
        
        // Create affine transform for image to gantry
        Matrix3x3 matrix;
        // You'll need to implement this based on your Matrix3x3 class
        // matrix.setFromFloat3(gantryToImMatrix);
        
        float3_affine_test imIdxToGantry(matrix, gantryToImOffset);
        
        return Float3ToBevTransform_test(imIdxToGantry, sourceDist, bevToBevIdx, imgCorner);
    }
    
    // Transform point from image coordinates to BEV coordinates
    CUDA_CALLABLE_MEMBER float3 imageToBev(const float3& imgPos, const Float3ToBevTransform_test& transform) {
        return transform.transformPoint(imgPos);
    }
    
    // Transform point from BEV coordinates to image coordinates
    CUDA_CALLABLE_MEMBER float3 bevToImage(const float3& bevPos, const imgFromBevTransform_test& transform) {
        return transform.transformPoint(bevPos);
    }
}