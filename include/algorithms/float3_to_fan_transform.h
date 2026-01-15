/**
 * \file
 * \brief Float3ToFanTransform class declaration
 */

#ifndef FLOAT3_TO_FAN_TRANSFORM_H
#define FLOAT3_TO_FAN_TRANSFORM_H

#include "common.cuh"
#include "Macro.cuh"

// Forward declaration
struct Float3FromFanTransform;
struct Float3IdxTransform;
struct Float3AffineTransform;

/**
 * \brief Transform to fan class
 */
struct Float3ToFanTransform {
    Float3IdxTransform fTFI;    ///< fanToFanIdx
    Float3AffineTransform iITG;  ///< imIdxToGantry
    vec2f dist;                  ///< 2D source distance
    
    __host__ __device__ Float3ToFanTransform() {}
    
    __host__ __device__ Float3ToFanTransform(
        const Float3AffineTransform imIdxToGantry,
        const vec2f sourceDist,
        const Float3IdxTransform fanToFanIdx
    ) : fTFI(fanToFanIdx), iITG(imIdxToGantry), dist(sourceDist) {}
    
    /**
     * \brief Return the point resulting from applying the transform to an input point
     * \param imIdx the input point
     * \return the transformed point as vec3f
     */
    __host__ __device__ vec3f transformPoint(const vec3f imIdx) const;
    
    /**
     * \brief Calculate inverse of this instance
     * \return The inverse transform
     */
    __host__ __device__ Float3FromFanTransform inverse() const;
    
    /**
     * \brief Get fanToFanIdx
     * \return the transform
     */
    __host__ __device__ Float3IdxTransform getFanToFanIdx() const { return fTFI; }
    
    /**
     * \brief Get imIdxToGantry
     * \return the affine transform
     */
    __host__ __device__ Float3AffineTransform getImIdxToGantry() const { return iITG; }
    
    /**
     * \brief Get the source distance x,y
     * \return 2D distance
     */
    __host__ __device__ vec2f getSourceDist() const { return dist; }
};

#endif // FLOAT3_TO_FAN_TRANSFORM_H

