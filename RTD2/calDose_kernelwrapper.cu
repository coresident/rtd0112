#include "common/common.cuh"
#include <cuda_runtime.h>
#include "cooperative_groups.h"
#include "thrust/device_vector.h"
#include "thrust/reduce.h"
#include "thrust/functional.h"
#include "thrust/execution_policy.h"
#include "thrust/sort.h"
#include "thrust/functional.h"
#include <thrust/for_each.h>
#include <vector>
#include <algorithm>

#include "omp.h"
namespace cg = cooperative_groups;

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    abort(); } } while(0)
#endif



// if beam (group or subspot) direction is given: 
__device__ float3 matrixMultiplyShared(const float3& point, const float* matrix)
 {
    // use shared memory to store the transform matrix(3*4, linear)
    __shared__ float sharedMatrix[12];

    int tid = threadIdx.x;
    
    if (tid < 12) {
        sharedMatrix[tid] = matrix[tid];
    }
    __syncthreads(); 

    float3 result;
    result.x = sharedMatrix[0] * point.x + sharedMatrix[1] * point.y + sharedMatrix[2] * point.z + sharedMatrix[3];
    result.y = sharedMatrix[4] * point.x + sharedMatrix[5] * point.y + sharedMatrix[6] * point.z + sharedMatrix[7];
    result.z = sharedMatrix[8] * point.x + sharedMatrix[9] * point.y + sharedMatrix[10] * point.z + sharedMatrix[11];
    return result;
}

__global__ void bevIdxToImg(float3* inputPoints, float3* outputPoints, 
                                                const float* bevIdxToBevMatrix, const float* gantryToImIdxMatrix, 
                                                float2 sourceDist, int numPoints) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 每个线程计算一个点
    if (idx < numPoints) {
        float3 point = inputPoints[idx];

        // 1) BEV 索引 -> BEV 坐标
        float3 bev = matrixMultiplyShared(point, bevIdxToBevMatrix);

        // 2) 逆透视变换
        bev.x *= (1.0f - bev.z / sourceDist.x);
        bev.y *= (1.0f - bev.z / sourceDist.y);

        // 3) Gantry -> 图像坐标
        outputPoints[idx] = matrixMultiplyShared(bev, gantryToImIdxMatrix);
    }
}

__global__ void bevIdxToImgSinglePoint(float3& bevIdx, float3& imgCoord, 
                                                const float* bevIdxToBevMatrix, const float* gantryToImIdxMatrix, 
                                                float2 sourceDist, int numPoints) {

	// 1) BEV 索引 -> BEV 坐标
	float3 bev = matrixMultiplyShared(bevIdx, bevIdxToBevMatrix);

	// 2) 逆透视变换
	bev.x *= (1.0f - bev.z / sourceDist.x);
	bev.y *= (1.0f - bev.z / sourceDist.y);

	// 3) Gantry -> 图像坐标
	imgCoord = matrixMultiplyShared(bev, gantryToImIdxMatrix);

}





__global__ void calFinalDoseKernel(float *dose,
									vec3f *d_beamDirect,
									int   num_beam,
									vec3f source,
									vec3f bmxdir,
									vec3f bmydir,
									vec3i *roiIndex,
									int   num_roi,
									Grid  doseGrid,
									cudaTextureObject_t rayweqData,
									cudaTextureObject_t iddData,
									cudaTextureObject_t profileData,
									cudaTextureObject_t subspotData,
									vec3f rayweqSetting,
									vec3f iddDepth,
									vec3f profileDepth,
									float *d_idbeamxy,
									int   *d_numParticles,
									int   nsubspot,
									int   nGauss,
									int   eneIdx,
									int   beamOffset,
									float beamParaPos,
									float longitudalCutoff,
									float transCutoff,
									float rtheta,
									float theta2,
									float r22,
									float sad) {

  cg::grid_group g   = cg::this_grid();
  cg::thread_block b = cg::this_thread_block();
  dim3 grid_index    = b.group_index();
  dim3 block_dim     = b.group_dim();
  dim3 thread_index  = b.thread_index();

  unsigned int gtx = thread_index.x + block_dim.x * grid_index.x;
  while (gtx < num_beam) {
//	int beamid = gtx + beamOffset;   

	bool beamLive = true;
	if(d_numParticles[gtx] <= BEAM_PARTICLE_CUTOFF){
		beamLive = false;
		
	}


	vec3f beamDirect = d_beamDirect[gtx];
	unsigned int gty = thread_index.y + block_dim.y * grid_index.y;

	float ix = d_idbeamxy[gtx * 2];
	float iy = d_idbeamxy[gtx * 2 + 1];

	while (gty < num_roi) {

	  vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
						roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
						roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

	  int absId = roiIndex[gty].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[gty].y * doseGrid.dims.z + roiIndex[gty].z;

//	  int nnz   = 0;
//	  bool increaseNNZFlag = true;
	  for (int isubspot = 0; isubspot < nsubspot; isubspot++) {
		float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
		float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

		float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
		if (subspotweight < 0.001) continue;
		float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
		float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
		float r2     = sigmax * sigmax + sigmay * sigmay;
		vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
		subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
		float projectedLength = dot(subspotDirection, pos - source);
		float idx       = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
		vec3f target    = source + subspotDirection * projectedLength;
		float crossDis2 = dot(pos - target, pos - target);

		float weqDepth  = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
		float phyDepth  = projectedLength - (sad - beamParaPos);

	  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d longitudal:%f\n", weqDepth, eneIdx, longitudalCutoff);
	  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d ", weqDepth, eneIdx);

		if (weqDepth < longitudalCutoff) {
		  float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
		  float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
		  float gaussianWeight = 0;
		  float sigma;
		  for (int j = 0; j < nGauss; j++) {
			float w = tex3D<float>(profileData, float(j + 0.5), profileDepthIdx, float(eneIdx + 0.5));
			sigma = tex3D<float>(profileData, float(j + nGauss + 0.5), profileDepthIdx, float(eneIdx + 0.5));
			sigma = sigma * sigma + initR2; // 计算点位置处的展宽
			// = 水带来的展宽+空气里的
			gaussianWeight += calGaussianR(crossDis2, sigma) * w;
		  }
		  // if (gtx == 0 && gty == 127422) printf("profileDepthId:%f sigma:%f initR2:%f gaussianweight:%e\n", profileDepthIdx, sigma, initR2,
		  //        gaussianWeight);
		  // if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
		  if (gaussianWeight > transCutoff) {
			float iddDepthIdx   = (weqDepth - iddDepth.x) / iddDepth.y;
			float idd           = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
			float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
		  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
		  	// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
		  	// if (gtx == 0 && gty == 127422) printf("weq:%f idddepthidx:%f eneIdx:%d idd:%e\n", weqDepth, iddDepthIdx, eneIdx, idd);
		  	// if (gtx == 0 && gty == 127422) printf("%e %e %e %e %d gty:%d\n", idd, gaussianWeight, overallWeight, subspotweight, d_numParticles[gtx], gty);
			atomicAdd(dose + absId, idd * gaussianWeight * overallWeight * subspotweight * d_numParticles[gtx]);
		  }
		}
	  }
	  gty += g.group_dim().y * b.group_dim().y;
	}
	gtx += g.group_dim().x * b.group_dim().x;
  }
}

__global__ void cuCalFinalDoseAndRBEKernel(
										float* dose,
										float* rbe2,
										float* counter,
										float* dalpha,
										float* dbeta,
#ifdef coalescing_optimize
										float* d_beamDirect_x,
										float* d_beamDirect_y,
										float* d_beamDirect_z,
#else
										vec3f* d_beamDirect,
#endif
										int num_beam,
										vec3f source,
										vec3f bmxdir,
										vec3f bmydir,
										vec3i* roiIndex,
										int num_roi,
										Grid doseGrid,
										cudaTextureObject_t rayweqData,
										cudaTextureObject_t iddData,
										cudaTextureObject_t profileData,
										cudaTextureObject_t subspotData,
										cudaTextureObject_t letData,
										cudaTextureObject_t lqData,
										vec3f rayweqSetting,
										vec3f iddDepth,
										vec3f profileDepth,
										float* d_idbeamxy,
										int* d_numParticles,
										int nsubspot,
										int nGauss,
										int eneIdx,
										int fieldId,
										float beamParaPos,
										float longitudalCutoff,
										float transCutoff,
										float rtheta,
										float theta2,
										float estDimension,
										float sad,
										float rs)
{
	cg::grid_group g = cg::this_grid();


	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;

#ifdef coalescing_optimize
		vec3f beamDirect = vec3f(d_beamDirect_x[gtx], d_beamDirect_y[gtx], d_beamDirect_z[gtx]);
		float ix = reinterpret_cast<float2*>(&d_idbeamxy[gtx * 2])[0].x;
		float iy = reinterpret_cast<float2*>(&d_idbeamxy[gtx * 2])[0].y;
#else
		vec3f beamDirect = d_beamDirect[gtx];
		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];
#endif

		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int absId = roiIndex[gty].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[gty].y * doseGrid.dims.z +
			roiIndex[g
		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);

			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);

			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0;
				float let = 0;
				float let2 = 0;
				float sigma;
				for (int j = 0; j < nGauss; j++)
				{
#ifdef tex_optimize
					float w = tex3D<float>(profileData, 
                                 float(j) + 0.5f,     
                                 float(profileDepthIdx) + 0.5f, 
                                 float(eneIdx) + 0.5f);

          sigma = tex3D<float>(profileData, 
                                     float(j + nGauss) + 0.5f, 
                                     float(profileDepthIdx) + 0.5f, 
                                     float(eneIdx) + 0.5f);

#else
          float w = tex3D<float>(profileData, j + 0.5, profileDepthIdx, eneIdx + 0.5);
					sigma = tex3D<float>(profileData, j + nGauss + 0.5, profileDepthIdx, eneIdx + 0.5);
#endif
					sigma = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;
					float tmplet = tex2D<float>(letData, (weqDepth + rs + (j - 2.0f) * estDimension / 5.0f) * 2.0f,
					                            eneIdx);
					let2 = let2 + tmplet * tmplet;
					let = let + tmplet;
				}
				let = let2 / let;

				float alpha, beta;
				alpha = tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 0.5f, 0.5f);
				beta = sqrtf(tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 1.5f, 0.5f));

				if (gaussianWeight > transCutoff)
				{
					float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
					float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
					float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));

					float tmpDose = idd * gaussianWeight * overallWeight * subspotweight * d_numParticles[gtx];
					atomicAdd(dose + absId + fieldId * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z, tmpDose);
					atomicAdd(rbe2 + absId, let*tmpDose);
//					atomicAdd(counter + absId, 1.0f);
					atomicAdd(dalpha + absId, tmpDose * alpha);
					atomicAdd(dbeta + absId, tmpDose * beta);
				}
			}
		}
		gid += g.size();
	}
}