/**
 * \file
 * \brief primTransfDiv kernel implementation
 */

#include "prim_transf_kernel.h"
#include "transfer_param_struct_div3.cuh"
#include "common.cuh"
#include "Macro.cuh"

__global__ void primTransfDiv(
    float* const result,
    TransferParamStructDiv3 params,
    const vec3i startIdx,
    const int maxZ,
    const vec3i doseDims,
    cudaTextureObject_t bevPrimDoseTex
) {
    unsigned int x = startIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y) {
        params.init(x, y); // Initialize object with current index position
        float* res = result + startIdx.z * doseDims.x * doseDims.y + y * doseDims.x + x;
        
        // Debug: Print first few coordinates and tex3D results
        if (x < 3 && y < 3 && threadIdx.x == 0 && threadIdx.y == 0) {
            vec3f pos0 = params.getFanIdx(startIdx.z) + make_vec3f(HALF, HALF, HALF);
            float tmp0 = tex3D<float>(bevPrimDoseTex, pos0.x, pos0.y, pos0.z);
            printf("GPU Debug primTransfDiv: (x,y)=(%d,%d), z=%d: pos0=(%.3f,%.3f,%.3f), tmp0=%.6f\n",
                   x, y, startIdx.z, pos0.x, pos0.y, pos0.z, tmp0);
        }
        
        for (int z = startIdx.z; z <= maxZ; ++z) {
            vec3f pos = params.getFanIdx(z) + make_vec3f(HALF, HALF, HALF); // Compensate for voxel value sitting at centre of voxel
            
            // Check bounds
            if (pos.x < 0.0f || pos.y < 0.0f || pos.z < 0.0f) {
                continue; // Skip invalid coordinates
            }
            
            float tmp = tex3D<float>(bevPrimDoseTex, pos.x, pos.y, pos.z);
            
            if (tmp > 0.0f) { // Only write to global memory if non-zero
                *res += tmp;
            }
            res += doseDims.x * doseDims.y;
        }
    }
}

