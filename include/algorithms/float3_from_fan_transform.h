/**
 * \file
 * \brief Float3FromFanTransform class declaration
 */

#ifndef FLOAT3_FROM_FAN_TRANSFORM_H
#define FLOAT3_FROM_FAN_TRANSFORM_H

#include "common.cuh"
#include "Macro.cuh"

// Forward declaration
struct Float3ToFanTransform;
struct Float3IdxTransform;
struct Float3AffineTransform;

/**
 * \brief Transform from fan coordinate system
 */
struct Float3FromFanTransform {
    Float3IdxTransform fITF;    ///< fanIdxToFan
    Float3AffineTransform gTII;  ///< gantryToImIdx
    vec2f dist;                 ///< source distance x,y
    
    __host__ __device__ Float3FromFanTransform() {}
    
    __host__ __device__ Float3FromFanTransform(
        const Float3IdxTransform fanIdxToFan,
        const vec2f sourceDist,
        const Float3AffineTransform gantryToImIdx
    ) : fITF(fanIdxToFan), gTII(gantryToImIdx), dist(sourceDist) {}
    
    /**
     * \brief Transform a 3D point according to internal matrix
     * \param fanIdx the 3D point
     * \return the transformed point
     */
    __host__ __device__ vec3f transformPoint(const vec3f fanIdx) const;
    
    /**
     * \brief Calculate inverse of this instance
     * \return The inverse transform
     */
    __host__ __device__ Float3ToFanTransform inverse() const;
    
    /**
     * \brief Invert matrix and shift in 3D
     * \param shift a 3D shift
     * \return the modified transform
     */
    __host__ __device__ Float3ToFanTransform invertAndShift(const vec3f shift) const;
};

#endif // FLOAT3_FROM_FAN_TRANSFORM_H

