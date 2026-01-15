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


struct BeamGroup {
    std::vector<int> originalBeamIdx;    
    vec3f avgDir;                  
    vec3f sourcePosition;               
    int groupId;                        
};

template<typename T>
struct sqrt_op
{
	__device__ 
	T operator()(T x)
	{
		return sqrtf(x);
	}
};

void calSqrt(float *first, float *last, float *result)
{
	thrust::transform(thrust::device, first, last, result, sqrt_op<float>());
}


// #define ptx_optimize    //whether use ptx optimization or not
// #define tex_optimize    //whether use tex optimization or not
// #define coalescing_optimize //whether use coalesce optimization or not


#ifdef ptx_optimize
__device__ float calGaussianR(float r2, float sigma2) {
    float result;
    asm volatile (
        "{\n"
        "   .reg .f32             tmp, denom;\n"
        "   add.f32               denom, %1, %3;\n"                                  // denom = sigma2 + 1e-5
        "   neg.f32               tmp, %2;\n"                                        // tmp = -r2
        "   div.rn.f32            tmp, tmp, denom;\n"                                // tmp = -r2/denom
				"   mul.f32               tmp, tmp, %5;\n"                                   // tmp = -r2/denom * log(2)
        "   ex2.approx.ftz.f32    tmp, tmp;\n"                                       // tmp = exp(tmp)
        "   mul.f32               denom, denom, %4;\n"                               // π*denom
        "   rcp.approx.ftz.f32    denom, denom;\n"                                   // 1/(π*denom)
        "   mul.f32               %0, tmp, denom;\n" 
        "}\n"
        : "=f"(result)
        : "f"(sigma2), "f"(r2),"f"(0.00001f),"f"(3.1415926f),"f"(1.442695f)
    );
    return result;
}
#else
__device__ float
calGaussianR(float r2, float sigma2) {
    return expf(-r2 / (sigma2 + 0.00001f)) / 3.1415926f / (sigma2 + 0.00001f);
}
#endif


__global__ void calFluenceMapAlphaBetaKernel(float* dalpha,
                                             float* dbeta,
                                             float* ddosecube,
                                             float* ddose,
                                             float* weights,
                                             float* rbe2,
                                             float* counter,
                                             vec3i* roiIndex,
                                             vec3f* beamDirect,
                                             vec3f* dsource,
                                             int* drowIndex,
                                             int* colPtr,
                                             int* deneId,
                                             vec3f rayweqSetting,
                                             int nSpots,
                                             Grid doseGrid,
                                             float rs,
                                             float estDimension,
                                             cudaTextureObject_t rayweqData,
                                             cudaTextureObject_t letData,
                                             cudaTextureObject_t lqData)
{
	cg::grid_group g = cg::this_grid();
	int gid          = g.thread_rank();
//    printf("i am in kernel!\n");
	while (gid < nSpots)
	{
		int nVoxels = colPtr[gid+1] - colPtr[gid];
		for (int i = 0; i < nVoxels; i++)
		{
			int   index     = colPtr[gid] + i;
			int   rowIndex  = drowIndex[index];

			int   eneId     = deneId[gid];
			vec3f source    = dsource[gid];
			vec3f direction = beamDirect[gid];
			float weight    = weights[gid];

			vec3f pos = vec3f(roiIndex[rowIndex].x * doseGrid.resolution.x + doseGrid.corner.x,
			                  roiIndex[rowIndex].y * doseGrid.resolution.y + doseGrid.corner.y,
			                  roiIndex[rowIndex].z * doseGrid.resolution.z + doseGrid.corner.z);

			int absId = roiIndex[rowIndex].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[rowIndex].y * doseGrid.dims.z + roiIndex[rowIndex].z;

			float projectedLength = dot(direction, pos - source);
			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			float weqDepth = tex2D<float>(rayweqData, idx + 0.5f, float(gid) + 0.5f);

			float let  = 0;
			float let2 = 0;

			for (int j = 0; j < 5; j++)
			{
				float tmplet = tex2D<float>(letData, (weqDepth + rs + (float(j) - 2.0f) * estDimension / 5.0f) * 2.0f, float(eneId));
				let2 = let2 + tmplet * tmplet;
				let  = let + tmplet;
			}

			let = let2 / let;
			float alpha, beta, tmpDose;
			alpha = tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 0.5f, 0.5f);
			beta  = sqrtf(tex3D<float>(lqData, log10f(let) * 100.0f + 0.5f, 1.5f, 0.5f));
			tmpDose = ddose[index] * weight;
			atomicAdd(dalpha + absId, tmpDose * alpha);
			atomicAdd(dbeta + absId, tmpDose * beta);
			atomicAdd(ddosecube + absId, tmpDose);
			atomicAdd(rbe2 + absId, let*tmpDose);
//			atomicAdd(counter + absId, 1.0f);
//            printf("let:%f alpha:%f beta:%f dose:%f\n", let, alpha, beta, tmpDose);
		}
		gid += g.size();
	}
}

__global__ void calDoseSumSquareKernel(float  *d_dose,
							  vec3f  *d_beamDirect,
							  int    num_beam,
							  vec3f  source,
							  vec3f  bmxdir,
							  vec3f  bmydir,
							  vec3i  *roiIndex,
							  int    num_roi,
							  Grid   doseGrid,
							  cudaTextureObject_t rayweqData,
							  cudaTextureObject_t iddData,
							  cudaTextureObject_t profileData,
							  cudaTextureObject_t subspotData,
							  vec3f rayweqSetting,
							  vec3f iddDepth,
							  vec3f profileDepth,
							  float *d_idbeamxy,
							  int   nsubspot,
							  int   nGauss,
							  int   eneIdx,
							  int   beamOffset,
							  float beamParaPos,
							  float longitudalCutoff,
							  float transCutoff,
							  float rtheta,
							  float theta2,
							  float radius_cutoff,
							  float sad) 
{
	cg::grid_group g = cg::this_grid();
	cg::thread_block b = cg::this_thread_block();

	dim3 grid_index    = b.group_index();
	dim3 block_dim     = b.group_dim();
	dim3 thread_index  = b.thread_index();

	uint64_t voxelIndex = thread_index.x + block_dim.x * grid_index.x;
	uint64_t roiThreadNum = g.group_dim().x * b.group_dim().x;
	uint64_t beamThreadNum = g.group_dim().y * b.group_dim().y;

	while (voxelIndex < num_roi) {
		vec3f pos = vec3f(roiIndex[voxelIndex].x * doseGrid.resolution.x + doseGrid.corner.x,
			roiIndex[voxelIndex].y * doseGrid.resolution.y + doseGrid.corner.y,
			roiIndex[voxelIndex].z * doseGrid.resolution.z + doseGrid.corner.z);
		uint64_t beamIndex = thread_index.y + block_dim.y * grid_index.y;
		while (beamIndex < num_beam) {
			vec3f beamDirect = d_beamDirect[beamIndex];
			float ix = d_idbeamxy[beamIndex * 2];
			float iy = d_idbeamxy[beamIndex * 2 + 1];
			float dose = 0;
			for (int isubspot = 0; isubspot < nsubspot; isubspot++) {
				float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
				float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));
	
				float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
				if (subspotweight < 0.001f) continue;
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
				
	
				if (weqDepth < longitudalCutoff) {
					float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
					float initR2 = r2 + 2.f * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
					float gaussianWeight = 0.f;
	
					for (int j = 0; j < nGauss; j++)
					{
						float w = tex3D<float>(profileData, float(j) + 0.5, profileDepthIdx, float(eneIdx + 0.5));
						float sigma = tex3D<float>(profileData, float(j + nGauss) + 0.5, profileDepthIdx, float(eneIdx + 0.5));
						sigma = sigma * sigma + initR2; // 计算点位置处的展宽
						// = 水带来的展宽+空气里的
						gaussianWeight += calGaussianR(crossDis2, sigma) * w;
					}
	
					if (gaussianWeight > transCutoff)
					{
						float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
						float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
						float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
						dose += idd * gaussianWeight * overallWeight * subspotweight;
					}
				}
			}
			atomicAdd(d_dose + voxelIndex, dose * dose);
			beamIndex += beamThreadNum;
		}
		voxelIndex += roiThreadNum;
	}
}

__global__ void calDoseKernel(float  *dose,
							int    *cscIndices,
							int    *colindices,
							int    *cscIndptr,
							int    totalnnz,
							int    *sumNNZ,
							vec3f  *d_beamDirect,
							int    num_beam,
							vec3f  source,
							vec3f  bmxdir,
							vec3f  bmydir,
							vec3i  *roiIndex,
							int    num_roi,
							Grid   doseGrid,
							cudaTextureObject_t rayweqData,
							cudaTextureObject_t iddData,
							cudaTextureObject_t profileData,
							cudaTextureObject_t subspotData,
							vec3f rayweqSetting,
							vec3f iddDepth,
							vec3f profileDepth,
							float *d_idbeamxy,
							int   nsubspot,
							int   nGauss,
							int   eneIdx,
							int   beamOffset,
							float beamParaPos,
							float longitudalCutoff,
							float transCutoff,
							float rtheta,
							float theta2,
							float radius_cutoff,
							float sad) {
	cg::grid_group g = cg::this_grid();

	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(
		num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;
		//	int beamid       = gtx + beamOffset;
		vec3f beamDirect = d_beamDirect[gtx];

		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];


		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int nnz = 0;
		bool increaseNNZFlag = true;
		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001f) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);
			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			//		printf("source:%f %f %f sad:%f beampos:%f\n", source.x, source.y, source.z, sad, beamParaPos);
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);
			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2.f * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0.f;
				for (int j = 0; j < nGauss; j++)
				{
#ifdef tex_optimize
					float w = tex3D<float>(profileData, 
                                 float(j) + 0.5f,     
                                 float(profileDepthIdx) + 0.5f, 
                                 float(eneIdx) + 0.5f);

          float sigma = tex3D<float>(profileData, 
                                     float(j + nGauss) + 0.5f, 
                                     float(profileDepthIdx) + 0.5f, 
                                     float(eneIdx) + 0.5f);

#else
          float w = tex3D<float>(profileData, j + 0.5, profileDepthIdx, eneIdx + 0.5);
					float sigma = tex3D<float>(profileData, j + nGauss + 0.5, profileDepthIdx, eneIdx + 0.5);
#endif
					
          sigma = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;	
				}
				// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
				if (gaussianWeight > transCutoff)
				{
					if (increaseNNZFlag)
					{
						nnz = atomicAdd(sumNNZ, 1);
						atomicAdd(&(cscIndptr[gtx + 1]), 1);
						increaseNNZFlag = false;
					}
					if (nnz < totalnnz)
					{
						cscIndices[nnz] = gty;
						colindices[nnz] = gtx + beamOffset;
						float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
						float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
						float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
						dose[nnz] += idd * gaussianWeight * overallWeight * subspotweight;
					}
				}
			}
		}
		gid += g.size();
	}
}

__global__ void calDoseKernel_(float  *dose,
							int    *cscIndices,
							int    *colindices,
							int    *cscIndptr,
							float  *d_idd_per_depth,//suplementary
							int    totalnnz,
							int    *sumNNZ,
							vec3f  *d_beamDirect,
							int    num_beam,
							vec3f  source,
							vec3f  bmxdir,
							vec3f  bmydir,
							vec3i  *roiIndex,
							int    num_roi,
							Grid   doseGrid,
							cudaTextureObject_t rayweqData,
							cudaTextureObject_t iddData,
							cudaTextureObject_t profileData,
							cudaTextureObject_t subspotData,
							vec3f rayweqSetting,
							vec3f iddDepth,
							vec3f profileDepth,
							float *d_idbeamxy,
							int   nsubspot,
							int   nGauss,
							int   eneIdx,
							int   beamOffset,
							float beamParaPos,
							float longitudalCutoff,
							float transCutoff,
							float rtheta,
							float theta2,
							float radius_cutoff,
							float sad) 
	{
	cg::grid_group g = cg::this_grid();

	unsigned long long gid = g.thread_rank();
	unsigned long long total_size = static_cast<unsigned long long>(num_beam) * static_cast<unsigned long long>(
		num_roi);

	while (gid < total_size)
	{
		unsigned long long gtx = gid % num_beam;
		unsigned long long gty = gid / num_beam;
		//	int beamid       = gtx + beamOffset;
		vec3f beamDirect = d_beamDirect[gtx];

		float ix = d_idbeamxy[gtx * 2];
		float iy = d_idbeamxy[gtx * 2 + 1];


		vec3f pos = vec3f(roiIndex[gty].x * doseGrid.resolution.x + doseGrid.corner.x,
		                  roiIndex[gty].y * doseGrid.resolution.y + doseGrid.corner.y,
		                  roiIndex[gty].z * doseGrid.resolution.z + doseGrid.corner.z);

		int nnz = 0;
		bool increaseNNZFlag = true;
		for (int isubspot = 0; isubspot < nsubspot; isubspot++)
		{
			float deltax = tex3D<float>(subspotData, 0.f, float(isubspot), float(eneIdx));
			float deltay = tex3D<float>(subspotData, 1.f, float(isubspot), float(eneIdx));

			float subspotweight = tex3D<float>(subspotData, 2.f, float(isubspot), float(eneIdx));
			if (subspotweight < 0.001f) continue;
			float sigmax = tex3D<float>(subspotData, 3.f, float(isubspot), float(eneIdx));
			float sigmay = tex3D<float>(subspotData, 4.f, float(isubspot), float(eneIdx));
			float r2 = sigmax * sigmax + sigmay * sigmay;
			vec3f subspotDirection = beamDirect * sad + deltax * bmxdir + deltay * bmydir;
			subspotDirection /= sqrtf(dot(subspotDirection, subspotDirection));
			float projectedLength = dot(subspotDirection, pos - source);
			float idx = (projectedLength - rayweqSetting.x) / rayweqSetting.y;
			vec3f target = source + subspotDirection * projectedLength;
			//		printf("source:%f %f %f sad:%f beampos:%f\n", source.x, source.y, source.z, sad, beamParaPos);
			float crossDis2 = dot(pos - target, pos - target);

			float weqDepth = tex3D<float>(rayweqData, idx, iy + deltay, ix + deltax);
			float phyDepth = projectedLength - (sad - beamParaPos);
			if (weqDepth < longitudalCutoff)
			{
				float profileDepthIdx = (weqDepth - profileDepth.x) / profileDepth.y;
				float initR2 = r2 + 2 * rtheta * phyDepth + theta2 * phyDepth * phyDepth;
				float gaussianWeight = 0;
				for (int j = 0; j < nGauss; j++)
				{
					float w = tex3D<float>(profileData, float(j + 0.5), profileDepthIdx, float(eneIdx + 0.5));
					float sigma = tex3D<float>(profileData, float(j + nGauss + 0.5), profileDepthIdx, float(eneIdx + 0.5));
					sigma = sigma * sigma + initR2; // 计算点位置处的展宽
					// = 水带来的展宽+空气里的
					gaussianWeight += calGaussianR(crossDis2, sigma) * w;
				}
				// if (gtx == 0 && gty == 127422) printf("weq:%f eneIdx:%d\n", weqDepth, eneIdx);
				if (gaussianWeight > transCutoff)
				{
					if (increaseNNZFlag)
					{
						nnz = atomicAdd(sumNNZ, 1);
						atomicAdd(&(cscIndptr[gtx + 1]), 1);
						increaseNNZFlag = false;
					}
					if (nnz < totalnnz)
					{
						cscIndices[nnz] = gty;
						colindices[nnz] = gtx + beamOffset;
						float iddDepthIdx = (weqDepth - iddDepth.x) / iddDepth.y;
						float idd = tex2D<float>(iddData, iddDepthIdx, float(eneIdx + 0.5));
						float overallWeight = tex3D<float>(profileData, 10.5f, profileDepthIdx, float(eneIdx + 0.5));
						dose[nnz] += idd * gaussianWeight * overallWeight * subspotweight;

            atomicAdd(d_idd_per_depth + int(iddDepthIdx), idd * gaussianWeight * overallWeight * subspotweight);  //suplementary

					}
				}
			}
		}
		gid += g.size();
	}
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
			roiIndex[gty].z;

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

__global__ void LQRBEKernel(int num, Grid doseGrid, vec3i* roiIndex, float* alphamap, float* betamap, float* rbemap,
                            float* rbe2, float* counter, float* phyDose, int nField)
{// alpha as input is alpha for each voxel with dose as weight
	// alpha as output is RBE
	// int id = blockDim.x * blockIdx.x + threadIdx.x;
	cg::grid_group g = cg::this_grid();
	int id = g.thread_rank();
	// printf("i am in kernel!\n");
	while(id < num)
	{
		int absId  = roiIndex[id].x * doseGrid.dims.y * doseGrid.dims.z + roiIndex[id].y * doseGrid.dims.z + roiIndex[id].z;
		float dose = 0.0f;
#pragma unroll
		for(int i = 0; i < nField; i++)
			dose += phyDose[i * doseGrid.dims.x * doseGrid.dims.y * doseGrid.dims.z + absId];
		// float dose = phyDose[absId];
		if(dose==0 || alphamap[absId]==0 || betamap[absId]==0)
		{
			rbemap[absId]=1;
			// printf("dose:%e alpha:%f beta:%f\n", dose, alphamap[absId], betamap[absId]);
		}
		else{

			float alpha = alphamap[absId]/dose;
			float beta  = betamap[absId]/dose;
			beta       *= beta; // average sqrt(beta)

			beta = (-alpha+sqrtf(alpha*alpha-4*-2.302585f*beta))/beta*0.5f;// Note, beta reused!

			rbemap[absId] = 3.988565f/beta;
		}
		if (dose > 0.0f)
			rbe2[absId] /= dose;
		id += g.size();
	}
}



__global__ void rotate3DArrayKernel(Grid new_g, float *d_new_array, cudaTextureObject_t oldArrayObj, vec4f *d_backRot) {
    cg::grid_group g = cg::this_grid();
    int id = g.thread_rank();
    int size = new_g.dims.x * new_g.dims.y * new_g.dims.z;
    while (id < size) {
        int z = id % new_g.dims.z;
        int y = id / new_g.dims.z % new_g.dims.y;
        int x = id / (new_g.dims.z * new_g.dims.y);
        vec3f center = new_g.corner + vec3f(x, y, z) * new_g.resolution;
        vec3f ori_pos = vec3f(owl::common::dot(d_backRot[0], vec4f(center.x, center.y, center.z, 1.0f)),
                              owl::common::dot(d_backRot[1], vec4f(center.x, center.y, center.z, 1.0f)),
                              owl::common::dot(d_backRot[2], vec4f(center.x, center.y, center.z, 1.0f)));
        d_new_array[id] = tex3D<float>(oldArrayObj, ori_pos.z, ori_pos.y, ori_pos.x);
        id += g.size();
    }
}

__global__ void setROIKernel(vec2i *d_roiFlag, vec3i *d_roiInd, vec3f *d_center, Grid grid, int size) {
    cg::grid_group g = cg::this_grid();
    int id = g.thread_rank();
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (id < size) {
        int absId = d_roiInd[id].x * grid.dims.y * grid.dims.z + d_roiInd[id].y * grid.dims.z + d_roiInd[id].z;
        d_roiFlag[absId].x = 1;
        d_roiFlag[absId].y = id;
        d_center[id] = grid.corner + vec3f(d_roiInd[id]) * grid.resolution + 0.5f * grid.resolution;
        id += g.size();
    }
}

// based on average direction, mark and group beams
__global__ void markSimilarBeamsKernel(vec3f* bmdir,
									int nBeam,
									vec3f leaderDirNorm,
									const int* assigned,
									unsigned char* __restrict__ membership,
									float cosThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int j = idx; j < nBeam; j += stride) {
        if (assigned[j]) { membership[j] = 0; continue; }
        vec3f dir = vec3f(bmdir[j], bmdir[nBeam + j], bmdir[2 * nBeam + j]);
        float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) + 1e-20f;
        dir.x /= len; dir.y /= len; dir.z /= len;
        float d = dir.x * leaderDirNorm.x + dir.y * leaderDirNorm.y + dir.z * leaderDirNorm.z;
        membership[j] = (d >= cosThreshold) ? 1 : 0;
    }
}

// 新增：对membership为1的beam做方向向量求和与计数（原地全局原子加）
__global__ void reduceSumAndCountKernel(const float* __restrict__ bmdir,
									int nBeam,
									const unsigned char* __restrict__ membership,
									float* __restrict__ sumX,
									float* __restrict__ sumY,
									float* __restrict__ sumZ,
									int* __restrict__ count)
{
    extern __shared__ unsigned char smem[]; // 复用共享内存存放局部和
    float* sSumX = (float*)smem;
    float* sSumY = (float*)&sSumX[blockDim.x];
    float* sSumZ = (float*)&sSumY[blockDim.x];
    int* sCnt    = (int*)&sSumZ[blockDim.x];

    int tid = threadIdx.x;
    sSumX[tid] = 0.0f; sSumY[tid] = 0.0f; sSumZ[tid] = 0.0f; sCnt[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float lSumX = 0.0f, lSumY = 0.0f, lSumZ = 0.0f; int lCnt = 0;
    for (int j = idx; j < nBeam; j += stride) {
        if (membership[j]) {
            lSumX += bmdir[j];
            lSumY += bmdir[nBeam + j];
            lSumZ += bmdir[2 * nBeam + j];
            lCnt  += 1;
        }
    }

    sSumX[tid] = lSumX; sSumY[tid] = lSumY; sSumZ[tid] = lSumZ; sCnt[tid] = lCnt;
    __syncthreads();

    // 归约到线程0
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sSumX[tid] += sSumX[tid + offset];
            sSumY[tid] += sSumY[tid + offset];
            sSumZ[tid] += sSumZ[tid + offset];
            sCnt[tid]  += sCnt[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sumX, sSumX[0]);
        atomicAdd(sumY, sSumY[0]);
        atomicAdd(sumZ, sSumZ[0]);
        atomicAdd(count, sCnt[0]);
    }
}

// 新增：host launcher 封装，供 .cpp 调用
void launchMarkSimilarBeamsKernel(const float* bmdir,
								int nBeam,
								vec3f leaderDirNorm,
								const int* assigned,
								unsigned char* membership,
								float cosThreshold,
								int grid,
								int block)
{
    markSimilarBeamsKernel<<<grid, block>>>(bmdir, nBeam, leaderDirNorm, assigned, membership, cosThreshold);
}

void launchReduceSumAndCountKernel(const float* bmdir,
								int nBeam,
								const unsigned char* membership,
								float* sumX,
								float* sumY,
								float* sumZ,
								int* count,
								int grid,
								int block,
								size_t sharedBytes)
{
    reduceSumAndCountKernel<<<grid, block, sharedBytes>>>(bmdir, nBeam, membership, sumX, sumY, sumZ, count);
}

// main function to wrapper kernels
void calOptimizedDoseWithBeamGrouping(float* finalDose,
									vec3f* d_beamDirect,
									vec3f* bmxdir, 
									vec3f float* bmydir,
									vec3f float* source,
									vec3i* roiIndex,
									int nBeam,
									int nRoi,
									Grid doseGrid,
									cudaTextureObject_t densityTex,
									cudaTextureObject_t spTex,
									cudaTextureObject_t iddTex,
									float divergenceThreshold = 0.1f,
									float cutoffSigma,
									int gpuId)
{
    printf("Starting optimized dose calculation with beam grouping...\n");
    
	cudaErrchk(cudaFree(0)); // Initialise CUDA context
	// adapted from RTD timing
	#ifdef FINE_GRAINED_TIMING
		float timeCopyAndBind = 0.0f;
		float timeAllocateAndSetup = 0.0f;
		float timeRaytracing = 0.0f;
		float timePrepareEnergyLoop = 0.0f;
		float timeFillIddSigma;
		float timePrepareSuperp;
		float timeSuperp;
		float timeCopyingToTexture = 0.0f;
		float timeTransforming = 0.0f;
		float timeCopyBack = 0.0f;
		float timeFreeMem = 0.0f;
		float timeTotal = 0.0f;
		cudaEvent_t start, stop;
		float elapsedTime;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

	#else // FINE_GRAINED_TIMING
		cudaEvent_t globalStart, globalStop;
		float globalTime;
		cudaEventCreate(&globalStart);
		cudaEventCreate(&globalStop);
		cudaEventRecord(globalStart, 0);

	#endif // FINE_GRAINED_TIMING 

	//beam grouping
	
    std::vector<int> beamToGroupMap;
    std::vector<BeamGroup> groups;
    beamToGroupMap.resize(nBeam);
	groupBeamsByAngle(BeamGroup* groups, vec3f* d_beamDirect, vec3f* source, int nBeam, float divergenceThreshold, std::vector<int>& beamToGroupMap);
	int nGroups = groups.size();

    
    // set BEV system for beam groups
    std::vector<deviceBevCoordSystem> coordSystems(nGroups);
    for (int g = 0; g < nGroups; g++) { //TODO: write a kernel
        // calculate the average direction
        vec3f avgDir = vec3f(0, 0, -1); // 简化版本
        vec3f avgSource = vec3f(source[0], source[nBeam], source[2*nBeam]); // 使用第一个beam的源位置
        
        coordSystems[g].sourcePos = avgSource;
        coordSystems[g].mainAxis = -avgDir; // z轴负方向
        
        // 建立正交基
        vec3f up = vec3f(0.0f, 0.0f, 1.0f);
        if (abs(dot(coordSystems[g].mainAxis, up)) > 0.9f) {
            up = vec3f(1.0f, 0.0f, 0.0f);
        }
        coordSystems[g].xAxis = normalize(cross(up, coordSystems[g].mainAxis));
        coordSystems[g].yAxis = normalize(cross(coordSystems[g].mainAxis, coordSystems[g].xAxis));
        coordSystems[g].srcToIsoDis = 100.0f;
    }
    
    // 3. 分配GPU内存
    deviceBevCoordSystem* d_coordSystems;
    int* d_beamGroupIds;
    int* d_groupToOriginalBeamMap;
    float* d_bevWepl;
    float* d_bevDensity;
    float* d_bevIdd;
    float* d_bevSigma;
    float* d_energies;
    float* d_bevDoseTexture;
    
    size_t coordSysSize = nGroups * sizeof(deviceBevCoordSystem);
    size_t beamMapSize = nBeam * sizeof(int);
    size_t bevDataSize = nRoi * nBeam * sizeof(float);
    size_t energySize = nBeam * sizeof(float);
    
    checkCudaErrors(cudaMalloc(&d_coordSystems, coordSysSize));
    checkCudaErrors(cudaMalloc(&d_beamGroupIds, beamMapSize));
    checkCudaErrors(cudaMalloc(&d_groupToOriginalBeamMap, beamMapSize));
    checkCudaErrors(cudaMalloc(&d_bevWepl, bevDataSize));
    checkCudaErrors(cudaMalloc(&d_bevDensity, bevDataSize));
    checkCudaErrors(cudaMalloc(&d_bevIdd, bevDataSize));
    checkCudaErrors(cudaMalloc(&d_bevSigma, bevDataSize));
    checkCudaErrors(cudaMalloc(&d_energies, energySize));
    
    // 纹理大小（BEV坐标系）
    int texWidth = 512, texHeight = 512, texDepth = 256;
    checkCudaErrors(cudaMalloc(&d_bevDoseTexture, texWidth * texHeight * texDepth * sizeof(float)));
    
    // 4. 复制数据到GPU
    checkCudaErrors(cudaMemcpy(d_coordSystems, coordSystems.data(), coordSysSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_beamGroupIds, beamToGroupMap.data(), beamMapSize, cudaMemcpyHostToDevice));
    
    // 创建组到原始beam映射
    std::vector<int> groupToOriginalBeamMap(nBeam);
    for (int i = 0; i < nBeam; i++) {
        groupToOriginalBeamMap[i] = i;
    }
    checkCudaErrors(cudaMemcpy(d_groupToOriginalBeamMap, groupToOriginalBeamMap.data(), beamMapSize, cudaMemcpyHostToDevice));
    
    // 创建能量数组（简化版本）
    std::vector<float> energies(nBeam);
    for (int i = 0; i < nBeam; i++) {
        energies[i] = 150.0f; // 假设所有beam都是150 MeV
    }
    checkCudaErrors(cudaMemcpy(d_energies, energies.data(), energySize, cudaMemcpyHostToDevice));
    
    //launch kernel
    int blockSize = 256;
    int gridSize = (nRoi * nBeam + blockSize - 1) / blockSize;
    
    printf("Launching kernel for ray tracing: gridSize=%d, blockSize=%d\n", gridSize, blockSize);
    
    // ray tracing
    int nSteps = 1000;
    float stepSize = 0.1f; // TODO: consider about this(z?)
    calWaterEquivalentPathKernel<<<gridSize, blockSize>>>(d_bevWepl, d_bevDensity, d_coordSystems, d_beamGroupIds, d_groupToOriginalBeamMap,
		roiIndex, nRoi, nBeam, doseGrid, densityTex, spTex, nSteps, stepSize);
    
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Ray tracing completed\n");
    
    // IDD & Sigma
    calIddAndSigmaKernel<<<gridSize, blockSize>>>(d_bevIdd, d_bevSigma, d_bevWepl, d_bevDensity, d_coordSystems, d_beamGroupIds,
		 roiIndex, nRoi, nBeam, iddTex, d_energies, cutoffSigma);
    
    checkCudaErrors(cudaDeviceSynchronize());
    printf("IDD and Sigma calculation completed\n");
    
    // dose calculation with RTD approaches
    dim3 doseBlockSize(32, 8);
    dim3 doseGridSize((nRoi + 255) / 256, (nBeam + doseBlockSize.x - 1) / doseBlockSize.x);
    
    calOptimizedDoseKernel<<<doseGridSize, doseBlockSize>>>(finalDose, d_bevIdd, d_bevSigma, d_coordSystems, d_beamGroupIds, d_groupToOriginalBeamMap,
        roiIndex, nRoi, nBeam, doseGrid, cutoffSigma);
    
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Optimized dose calculation completed\n");
    
    // interp
    textureInterpolationKernel<<<gridSize, blockSize>>>(d_bevDoseTexture, finalDose, d_coordSystems, d_beamGroupIds, 
		roiIndex, nRoi, nBeam, doseGrid, texWidth, texHeight, texDepth);
    
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Texture interpolation completed\n");
    
    
    checkCudaErrors(cudaFree(d_coordSystems));
    checkCudaErrors(cudaFree(d_beamGroupIds));
    checkCudaErrors(cudaFree(d_groupToOriginalBeamMap));
    checkCudaErrors(cudaFree(d_bevWepl));
    checkCudaErrors(cudaFree(d_bevDensity));
    checkCudaErrors(cudaFree(d_bevIdd));
    checkCudaErrors(cudaFree(d_bevSigma));
    checkCudaErrors(cudaFree(d_energies));
    checkCudaErrors(cudaFree(d_bevDoseTexture));
    
    printf("Optimized dose calculation with beam grouping completed successfully!\n");
}

void calFluenceMapAlphaBeta(float* dalpha,
							float* dbeta,
							float* ddosecube,
							float* ddose,
							float* weights,
							float* rbe2,
							float* counter,
							vec3i* roiIndex,
							vec3f* beamDirect,
							vec3f* dsource,
							int* drowIndex,
							int* colPtr,
							int* deneId,
							vec3f rayweqSetting,
							int nSpots,
							Grid doseGrid,
							float rs,
							float estDimension,
							cudaTextureObject_t rayweqData,
							cudaTextureObject_t letData,
							cudaTextureObject_t lqData,
							int gpuid)
{
	void* args[] = {
		(void*)&dalpha, (void*)&dbeta, (void*)&ddosecube, (void*)&ddose, (void*)&weights, (void *)&rbe2, (void *)&counter, (void*)&roiIndex,
		(void*)&beamDirect, (void*)&dsource, (void*)&drowIndex, (void*)&colPtr, (void*)&deneId, (void*)&rayweqSetting,
		(void*)&nSpots, (void*)&doseGrid, (void*)&rs, (void*)&estDimension, (void*)&rayweqData, (void*)&letData, (void*)&lqData
	};
	int batchSize = queryBatchSize(calFluenceMapAlphaBetaKernel, gpuid);
	launchCudaKernel(calFluenceMapAlphaBetaKernel, batchSize, args);
}


void calFinalDose(float *dose,
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
				float r2,
				float sad,
				int   gpuid) 
{
	void *args[] = {(void *)&dose, (void *)&d_beamDirect, (void *)&num_beam, (void *)&source, (void *)&bmxdir,
					(void *)&bmydir, (void *)&roiIndex, (void *)&num_roi, (void *)&doseGrid, (void *)&rayweqData,
					(void *)&iddData, (void *)&profileData, (void *)&subspotData, (void *)&rayweqSetting, (void *)&iddDepth,
					(void *)&profileDepth, (void *)&d_idbeamxy, (void *)&d_numParticles, (void *)&nsubspot,
					(void *)&nGauss, (void *)&eneIdx, (void *)&beamOffset, (void *)&beamParaPos,
					(void *)&longitudalCutoff, (void *)&transCutoff,
					(void *)&rtheta, (void *)&theta2, (void *)&r2, (void *)&sad};
	int batchSize = queryBatchSize(calFinalDoseKernel, gpuid);
	launchCudaKernel2D(calFinalDoseKernel, batchSize, args);
}

void calFinalDoseAndRBE(float* dose,
						float* rbe2,
						float* counter,
						float* dalpha,
						float* dbeta,
                        vec3f* d_beamDirect,
                        int    num_beam,
                        vec3f  source,
                        vec3f  bmxdir,
                        vec3f  bmydir,
                        vec3i* roiIndex,
                        int    num_roi,
                        Grid   doseGrid,
                        cudaTextureObject_t rayweqData,
                        cudaTextureObject_t iddData,
                        cudaTextureObject_t profileData,
                        cudaTextureObject_t subspotData,
                        cudaTextureObject_t letData,
                        cudaTextureObject_t lqData,
                        vec3f  rayweqSetting,
                        vec3f  iddDepth,
                        vec3f  profileDepth,
                        float* d_idbeamxy,
                        int*   d_numParticles,
                        int    nsubspot,
                        int    nGauss,
                        int    eneIdx,
                        int    fieldId,
                        float  beamParaPos,
                        float  longitudalCutoff,
                        float  transCutoff,
                        float  rtheta,
                        float  theta2,
                        float  r2,
                        float  estDimension,
                        float  sad,
                        float  rs,
                        int    gpuid)
{

#ifdef coalescing_optimize
    thrust::device_vector<float> d_beamDirect_x(num_beam);
	thrust::device_vector<float> d_beamDirect_y(num_beam);		
	thrust::device_vector<float> d_beamDirect_z(num_beam);
	auto d_beamDirect_xptr = d_beamDirect_x.data().get();
	auto d_beamDirect_yptr = d_beamDirect_y.data().get();
	auto d_beamDirect_zptr = d_beamDirect_z.data().get();
	thrust::for_each(
		thrust::device,
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(num_beam),
		[d_beamDirect_xptr,
			d_beamDirect_yptr,
			d_beamDirect_zptr,
			d_beamDirect] __device__ (int i) {
			d_beamDirect_xptr[i] = d_beamDirect[i].x;
			d_beamDirect_yptr[i] = d_beamDirect[i].y;
			d_beamDirect_zptr[i] = d_beamDirect[i].z;
		}
	);

#endif
	
	void* args[] = {
		(void*)&dose, (void*)&rbe2, (void*)&counter, (void*)&dalpha, (void*)&dbeta, 
#ifdef coalescing_optimize
		(void*)&(d_beamDirect_xptr), (void*)&(d_beamDirect_yptr), (void*)&(d_beamDirect_zptr),
#else
		(void*)&d_beamDirect, 
#endif
		(void*)&num_beam,
		(void*)&source, (void*)&bmxdir, (void*)&bmydir, (void*)&roiIndex, (void*)&num_roi, (void*)&doseGrid, (void*)&rayweqData,
		(void*)&iddData, (void*)&profileData, (void*)&subspotData, (void*)&letData, (void*)&lqData, (void*)&rayweqSetting,
		(void*)&iddDepth, (void*)&profileDepth, (void*)&d_idbeamxy, (void*)&d_numParticles, (void*)&nsubspot,
		(void*)&nGauss, (void*)&eneIdx, (void *)&fieldId, (void*)&beamParaPos,(void*)&longitudalCutoff, (void*)&transCutoff,
		(void*)&rtheta, (void*)&theta2, (void*)&estDimension,(void*)&sad, (void*)&rs,
	};
	int batchSize = queryBatchSize(cuCalFinalDoseAndRBEKernel, gpuid);
	cudaFuncSetCacheConfig(cuCalFinalDoseAndRBEKernel, cudaFuncCachePreferL1);
	launchCudaKernel(cuCalFinalDoseAndRBEKernel, batchSize, args);
}

void calFinalDoseAndMKM(float* dose,
                        float* Z1Dmix,
                        vec3f* d_beamDirect,
                        int    num_beam,
                        vec3f  source,
                        vec3f  bmxdir,
                        vec3f  bmydir,
                        vec3i* roiIndex,
                        int    num_roi,
                        Grid   doseGrid,
                        cudaTextureObject_t rayweqData,
                        cudaTextureObject_t iddData,
                        cudaTextureObject_t profileData,
                        cudaTextureObject_t subspotData,
                        cudaTextureObject_t Z1DijData,
                        vec3f  rayweqSetting,
                        vec3f  iddDepth,
                        vec3f  profileDepth,
                        float* d_idbeamxy,
                        int*   d_numParticles,
                        int    nsubspot,
                        int    nGauss,
                        int    eneIdx,
                        int    fieldId,
                        float  beamParaPos,
                        float  longitudalCutoff,
                        float  transCutoff,
                        float  rtheta,
                        float  theta2,
                        float  r2,
                        float  estDimension,
                        float  sad,
                        float  rs,
                        float  energy,
                        int    gpuid)
{
	void* args[] = {
		(void*)&dose, (void*)&Z1Dmix, (void*)&d_beamDirect, (void*)&num_beam, (void*)&source,
		(void*)&bmxdir, (void*)&bmydir, (void*)&roiIndex, (void*)&num_roi, (void*)&doseGrid, (void*)&rayweqData,
		(void*)&iddData, (void*)&profileData, (void*)&subspotData, (void*)&Z1DijData, (void*)&rayweqSetting,
		(void*)&iddDepth, (void*)&profileDepth, (void*)&d_idbeamxy, (void*)&d_numParticles, (void*)&nsubspot,
		(void*)&nGauss, (void*)&eneIdx, (void *)&fieldId, (void*)&beamParaPos,(void*)&longitudalCutoff, (void*)&transCutoff,
		(void*)&rtheta, (void*)&theta2, (void*)&estDimension,(void*)&sad, (void*)&rs, (void*)&energy
	};
	int batchSize = queryBatchSize(cuCalFinalDoseAndMKMKernel, gpuid);
	launchCudaKernel(cuCalFinalDoseAndMKMKernel, batchSize, args);
}


void calDoseSumSquares(float *d_doseNorm,
	vec3f *d_beamDirect,
	int num_beam,
	vec3f source,
	vec3f bmxdir,
	vec3f bmydir,
	vec3i *roiIndex,
	int num_roi,
	Grid doseGrid,
	cudaTextureObject_t rayweqData,
	cudaTextureObject_t iddData,
	cudaTextureObject_t profileData,
	cudaTextureObject_t subspotData,
	vec3f rayweqSetting,
	vec3f iddDepth,
	vec3f profileDepth,
	float *d_idbeamxy,
	int nsubspot,
	int nGauss,
	int eneIdx,
	int beamOffset,
	float beamParaPos,
	float longitudalCutoff,
	float transCutoff,
	float rtheta,
	float theta2,
	float r2,
	float sad,
	int gpuid)
{
	void *args[] = {(void *)&d_doseNorm,
		(void *)&d_beamDirect, (void *)&num_beam, (void *)&source, (void *)&bmxdir, (void *)&bmydir,
		(void *)&roiIndex,
		(void *)&num_roi, (void *)&doseGrid, (void *)&rayweqData, (void *)&iddData, (void *)&profileData,
		(void *)&subspotData, (void *)&rayweqSetting, (void *)&iddDepth,
		(void *)&profileDepth, (void *)&d_idbeamxy, (void *)&nsubspot,
		(void *)&nGauss, (void *)&eneIdx, (void *)&beamOffset, (void *)&beamParaPos,
		(void *)&longitudalCutoff, (void *)&transCutoff,
		(void *)&rtheta, (void *)&theta2, (void *)&r2, (void *)&sad};
	
	int batchSize = queryBatchSize(calDoseSumSquareKernel, gpuid);
	launchCudaKernel2D(calDoseSumSquareKernel, batchSize, args);
}

int calOneLayerDoseOld(float *dose,
					   int *cscIndices,
					   int *colindices,
					   int *cscIndptr,
					   int totalnnz,
					   int *sumNNZ,
					   vec3f *d_beamDirect,
					   int num_beam,
					   vec3f source,
					   vec3f bmxdir,
					   vec3f bmydir,
					   vec3i *roiIndex,
					   int num_roi,
					   Grid doseGrid,
					   cudaTextureObject_t rayweqData,
					   cudaTextureObject_t iddData,
					   cudaTextureObject_t profileData,
					   cudaTextureObject_t subspotData,
					   vec3f rayweqSetting,
					   vec3f iddDepth,
					   vec3f profileDepth,
					   float *d_idbeamxy,
					   int nsubspot,
					   int nGauss,
					   int eneIdx,
					   int beamOffset,
					   float beamParaPos,
					   float longitudalCutoff,
					   float transCutoff,
					   float rtheta,
					   float theta2,
					   float r2,
					   float sad,
					   int gpuid) {

  void *args[] = {(void *)&dose, (void *)&cscIndices,
				  (void *)&colindices,
				  (void *)&cscIndptr, (void *)&totalnnz, (void *)&sumNNZ,
				  (void *)&d_beamDirect, (void *)&num_beam, (void *)&source, (void *)&bmxdir, (void *)&bmydir,
				  (void *)&roiIndex,
				  (void *)&num_roi, (void *)&doseGrid, (void *)&rayweqData, (void *)&iddData, (void *)&profileData,
				  (void *)&subspotData, (void *)&rayweqSetting, (void *)&iddDepth,
				  (void *)&profileDepth, (void *)&d_idbeamxy, (void *)&nsubspot,
				  (void *)&nGauss, (void *)&eneIdx, (void *)&beamOffset, (void *)&beamParaPos,
				  (void *)&longitudalCutoff, (void *)&transCutoff,
				  (void *)&rtheta, (void *)&theta2, (void *)&r2, (void *)&sad};

  int batchSize = queryBatchSize(calDoseKernel, gpuid);
  launchCudaKernel(calDoseKernel, batchSize, args);
  int *d_colInd1;
  int h_NNZ;
  checkCudaErrors(cudaMemcpy(&h_NNZ, sumNNZ, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMalloc((void **)&d_colInd1, sizeof(int) * h_NNZ));
  checkCudaErrors(cudaMemcpy(d_colInd1, colindices, sizeof(int) * h_NNZ, cudaMemcpyDeviceToDevice));

  thrust::sort_by_key(thrust::device, colindices, colindices + h_NNZ, cscIndices);
  thrust::sort_by_key(thrust::device, d_colInd1, d_colInd1 + h_NNZ, dose);

  std::vector<int> h_cscIndptr(num_beam);
  checkCudaErrors(cudaMemcpy(h_cscIndptr.data(), cscIndptr+1, sizeof(int) * num_beam, cudaMemcpyDeviceToHost));
  int start = 0;
  int end   = 0;

  std::vector<std::thread> threads;
	threads.reserve(num_beam);
	for (int i = 0; i < num_beam; ++i)
	{
			end += h_cscIndptr[i];
			if (end > start)
			{
					threads.emplace_back([=]() {
							thrust::sort_by_key(thrust::device,
																	cscIndices + start,
																	cscIndices + end,
																	dose + start);
					});
			}
			start += h_cscIndptr[i];
	}

	for (auto &t : threads)
	{
			t.join();
	}

  checkCudaErrors(cudaFree(d_colInd1));
  return h_NNZ;
}


void RBEMap(int num, Grid doseGrid, vec3i* roiIndex, float* alphamap, float* betamap, float* rbemap, float* rbe2,
            float* counter, float* phyDose, int nField, int gpuid)
{
	void* args[] = {
		(void*)&num, (void*)&doseGrid, (void*)&roiIndex, (void*)&alphamap, (void*)&betamap, (void*)&rbemap, (void*)&rbe2,
		(void*)&counter, (void*)&phyDose, (void*)&nField
	};
	int batchSize = queryBatchSize(LQRBEKernel, gpuid);
	launchCudaKernel(LQRBEKernel, batchSize, args);
}

void mkmRBEMap(int num, Grid doseGrid, vec3i* roiIndex, float* rbemap1, float* rbemap2, float* Z1Dmix,
               float* phyDose,
               vec2f alpha0AndBeta0, int nField, int gpuid)
{
	void* args[] = {
		(void*)&num, (void*)&doseGrid, (void*)&roiIndex, (void*)&rbemap1, (void*)&rbemap2, (void*)&Z1Dmix,
		(void*)&phyDose, (void*)&alpha0AndBeta0, (void*)&nField,
	};
	int batchSize = queryBatchSize(mkmRBEMapKernel, gpuid);
	launchCudaKernel(mkmRBEMapKernel, batchSize, args);
}

void setROI(vec2i *d_roiFlag, vec3i *d_roiInd, vec3f *d_center, Grid grid, int size) {
    int batchSize = queryBatchSize(setROIKernel, 0);
    void *args[] = {(void *)&d_roiFlag, (void *)&d_roiInd, (void *)&d_center, (void *)&grid, (void *)&size};
    launchCudaKernel(setROIKernel, batchSize, args);
}

int rotate3DArray(Grid &new_g,
                  const Grid &old_g,
                  float *h_Array_new,
                  float *h_Array_old,
                  float *d_rot_forward,
                  int type,
                  int gpuId) {
    auto *h_rot_forward = new vec4f[4];
    auto *h_rot_backward = new vec4f[4];

    vec4f *d_rot_backward;
    float *d_array_new;

    cudaArray_t old3DArray;
    cudaTextureObject_t old3DArrayObj;

    weq::create3DTexture(h_Array_old,
                         &old3DArray,
                         &old3DArrayObj,
                         weq::LINEAR,
                         weq::R,
                         old_g.dims.z,
                         old_g.dims.y,
                         old_g.dims.x,
                         type,
                         gpuId);

    h_rot_forward[0] = vec4f(d_rot_forward[0], d_rot_forward[1], d_rot_forward[2], d_rot_forward[3]);
    h_rot_forward[1] = vec4f(d_rot_forward[4], d_rot_forward[5], d_rot_forward[6], d_rot_forward[7]);
    h_rot_forward[2] = vec4f(d_rot_forward[8], d_rot_forward[9], d_rot_forward[10], d_rot_forward[11]);
    h_rot_forward[3] = vec4f(d_rot_forward[12], d_rot_forward[13], d_rot_forward[14], d_rot_forward[15]);

    h_rot_backward[0] = vec4f(d_rot_forward[0], d_rot_forward[4], d_rot_forward[8], -d_rot_forward[3]);
    h_rot_backward[1] = vec4f(d_rot_forward[1], d_rot_forward[5], d_rot_forward[9], -d_rot_forward[7]);
    h_rot_backward[2] = vec4f(d_rot_forward[2], d_rot_forward[6], d_rot_forward[10], -d_rot_forward[11]);
    h_rot_backward[3] = vec4f(d_rot_forward[12], d_rot_forward[13], d_rot_forward[14], d_rot_forward[15]);

    vec3f corner_1 = old_g.corner;
    vec3f corner_2 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x, old_g.corner.y, old_g.corner.z);
    vec3f corner_3 = vec3f(old_g.corner.x, old_g.corner.y + old_g.resolution.y * old_g.dims.y, old_g.corner.z);
    vec3f corner_4 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x,
                           old_g.corner.y + old_g.resolution.y * old_g.dims.y,
                           old_g.corner.z);

    vec3f corner_5 = vec3f(old_g.corner.x, old_g.corner.y, old_g.corner.z + old_g.resolution.z * old_g.dims.z);
    vec3f corner_6 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x,
                           old_g.corner.y,
                           old_g.corner.z + old_g.resolution.z * old_g.dims.z);
    vec3f corner_7 = vec3f(old_g.corner.x,
                           old_g.corner.y + old_g.resolution.y * old_g.dims.y,
                           old_g.corner.z + old_g.resolution.z * old_g.dims.z);
    vec3f corner_8 = vec3f(old_g.corner.x + old_g.resolution.x * old_g.dims.x,
                           old_g.corner.y + old_g.resolution.y * old_g.dims.y,
                           old_g.corner.z + old_g.resolution.z * old_g.dims.z);

    vec3f new_corner_1 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_1.x, corner_1.y, corner_1.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_1.x, corner_1.y, corner_1.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_1.x, corner_1.y, corner_1.z, 1.f)));
    vec3f new_corner_2 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_2.x, corner_2.y, corner_2.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_2.x, corner_2.y, corner_2.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_2.x, corner_2.y, corner_2.z, 1.f)));
    vec3f new_corner_3 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_3.x, corner_3.y, corner_3.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_3.x, corner_3.y, corner_3.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_3.x, corner_3.y, corner_3.z, 1.f)));
    vec3f new_corner_4 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_4.x, corner_4.y, corner_4.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_4.x, corner_4.y, corner_4.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_4.x, corner_4.y, corner_4.z, 1.f)));

    vec3f new_corner_5 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_5.x, corner_5.y, corner_5.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_5.x, corner_5.y, corner_5.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_5.x, corner_5.y, corner_5.z, 1.f)));
    vec3f new_corner_6 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_6.x, corner_6.y, corner_6.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_6.x, corner_6.y, corner_6.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_6.x, corner_6.y, corner_6.z, 1.f)));
    vec3f new_corner_7 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_7.x, corner_7.y, corner_7.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_7.x, corner_7.y, corner_7.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_7.x, corner_7.y, corner_7.z, 1.f)));
    vec3f new_corner_8 = vec3f(owl::common::dot(h_rot_forward[0], vec4f(corner_8.x, corner_8.y, corner_8.z, 1.f)),
                               owl::common::dot(h_rot_forward[1], vec4f(corner_8.x, corner_8.y, corner_8.z, 1.f)),
                               owl::common::dot(h_rot_forward[2], vec4f(corner_8.x, corner_8.y, corner_8.z, 1.f)));

    float
            new_corner_x[8] = {new_corner_1.x, new_corner_2.x, new_corner_3.x, new_corner_4.x, new_corner_5.x, new_corner_6.x,
                               new_corner_7.x, new_corner_8.x};
    float
            new_corner_y[8] = {new_corner_1.y, new_corner_2.y, new_corner_3.y, new_corner_4.y, new_corner_5.y, new_corner_6.y,
                               new_corner_7.y, new_corner_8.y};
    float
            new_corner_z[8] = {new_corner_1.z, new_corner_2.z, new_corner_3.z, new_corner_4.z, new_corner_5.z, new_corner_6.z,
                               new_corner_7.z, new_corner_8.z};

    float min_x = *std::min_element(new_corner_x, new_corner_x + 8);
    float min_y = *std::min_element(new_corner_y, new_corner_y + 8);
    float min_z = *std::min_element(new_corner_z, new_corner_z + 8);

    float max_x = *std::max_element(new_corner_x, new_corner_x + 8);
    float max_y = *std::max_element(new_corner_y, new_corner_y + 8);
    float max_z = *std::max_element(new_corner_z, new_corner_z + 8);

    new_g.corner = vec3f(min_x, min_y, min_z);
    new_g.upperCorner = vec3f(max_x, max_y, max_z);

    vec3f dims = (new_g.upperCorner - new_g.corner) / old_g.resolution + 1.0f;

    new_g.dims = vec3i(int(dims.x), int(dims.y), int(dims.z));
    new_g.resolution = old_g.resolution;
    checkCudaErrors(cudaMalloc((void **)&d_rot_backward, sizeof(vec4f) * 4));
    checkCudaErrors(cudaMalloc((void **)&d_array_new, sizeof(float) * new_g.dims.x * new_g.dims.y * new_g.dims.z));
    checkCudaErrors(cudaMemcpy(d_rot_backward, h_rot_backward, sizeof(vec4f) * 4, cudaMemcpyHostToDevice));

    int batchSize = queryBatchSize(rotate3DArrayKernel, gpuId);
    void *args[] = {(void *)&new_g, (void *)&d_array_new, (void *)&old3DArrayObj, (void *)&d_rot_backward};

    launchCudaKernel(rotate3DArrayKernel, batchSize, args);

    checkCudaErrors(cudaMemcpy(h_Array_new,
                               d_array_new,
                               sizeof(float) * new_g.dims.x * new_g.dims.y * new_g.dims.z,
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_rot_backward));
    checkCudaErrors(cudaFree(d_array_new));
    checkCudaErrors(cudaFreeArray(old3DArray));
    checkCudaErrors(cudaDestroyTextureObject(old3DArrayObj));

    delete[] h_rot_forward;
    delete[] h_rot_backward;
    return new_g.dims.x * new_g.dims.y * new_g.dims.z;
}

// ======================== 新的优化内核实现 ========================


struct deviceBevCoordSystem {
    vec3f sourcePos;
    vec3f mainAxis;     // z轴负方向
    vec3f xAxis;
    vec3f yAxis;
    float srcToIsoDis;
};


__device__ vec3f worldToBevDevice(const vec3f& worldPos, const deviceBevCoordSystem& coordSys) {
    vec3f relativePos = worldPos - coordSys.sourcePos;
    float x = dot(relativePos, coordSys.xAxis);
    float y = dot(relativePos, coordSys.yAxis);
    float z = dot(relativePos, coordSys.mainAxis);
    return vec3f(x, y, z);
}

// 发散坐标系到世界坐标转换（设备端）
__device__ vec3f bevToWorldDevice(const vec3f& divPos, const deviceBevCoordSystem& coordSys) {
    return coordSys.sourcePos + 
           divPos.x * coordSys.xAxis + 
           divPos.y * coordSys.yAxis + 
           divPos.z * coordSys.mainAxis;
}

// ray tracing kernel for WEPL
// 应该是一个beam有一个自己的方向，对这个方向查找每一点也就是bev的z轴上每一点的信息
__global__ void calWaterEquivalentPathKernel(float* bevWepl,
                                            float* bevDensity,
                                            const deviceBevCoordSystem* coordSystems,
                                            const int* beamGroupIds,
                                            const int* groupToOriginalBeamMap,
                                            vec3i* roiIndex,
                                            int nRoi,
                                            int nBeam,
                                            Grid doseGrid,
                                            cudaTextureObject_t densityTex,
                                            cudaTextureObject_t spTex,
                                            int nSteps,
                                            float stepSize) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    
    while (tid < nRoi * nBeam) {
        int roiIdx = tid / nBeam;
        int beamIdx = tid % nBeam;
        
        
        //roi的3d直角坐标
        vec3f roiPos = vec3f(roiIndex[roiIdx].x * doseGrid.resolution.x + doseGrid.corner.x,
                            roiIndex[roiIdx].y * doseGrid.resolution.y + doseGrid.corner.y,
                            roiIndex[roiIdx].z * doseGrid.resolution.z + doseGrid.corner.z);
        
        // get group information for bev system
        int groupId = beamGroupIds[beamIdx];
        deviceBevCoordSystem coordSys = coordSystems[groupId];
        
        // transform the roiPos to bev coordinates
        vec3f divPos = worldToBevDevice(roiPos, coordSys);
        
        // ray tracing
        vec3f rayStart = coordSys.sourcePos;
        vec3f rayDir = normalize(roiPos - rayStart);
        
        float totalWepl = 0.0f;
        float totalDensity = 0.0f;
        
        for (int step = 0; step < nSteps; step++) {
            float t = step * stepSize;
            vec3f currentPos = rayStart + t * rayDir;
            
            // make sure the pos is inside the roi TODO: check before?
            if (currentPos.x < doseGrid.corner.x || currentPos.y < doseGrid.corner.y || currentPos.z < doseGrid.corner.z ||
                currentPos.x > (doseGrid.corner.x + doseGrid.dims.x * doseGrid.resolution.x) || currentPos.y > (doseGrid.corner.y + doseGrid.dims.y * doseGrid.resolution.y) || currentPos.z > (doseGrid.corner.z + doseGrid.dims.z * doseGrid.resolution.z)) break; 

            // to voxel idx
            float voxelX = (currentPos.x - doseGrid.corner.x) / doseGrid.resolution.x;
            float voxelY = (currentPos.y - doseGrid.corner.y) / doseGrid.resolution.y;
            float voxelZ = (currentPos.z - doseGrid.corner.z) / doseGrid.resolution.z;
            
            // density and stopping power from texture
            float density = tex3D<float>(densityTex, voxelX + 0.5f, voxelY + 0.5f, voxelZ + 0.5f);
            float sp = tex3D<float>(spTex, voxelX + 0.5f, voxelY + 0.5f, voxelZ + 0.5f);
            
            totalWepl += sp * stepSize;
            totalDensity += density;
            
            // avoid 边界问题
            if (length(currentPos - roiPos) < stepSize) {
                break;
            }
        }
        
        int outputIdx = roiIdx * nBeam + beamIdx;
        bevWepl[outputIdx] = totalWepl;
        bevDensity[outputIdx] = totalDensity / nSteps;
        
        tid += totalThreads;
    }
}

__global__ void fillBevDensityAndSp(float* const bevDensity,
									float* const bevCumulSp, 
									int* const beamFirstInside, 
									int* const firstStepOutside, 
									const DensityAndSpTracerParams params, 
									cudaTextureObject_t imVolTex, 
									cudaTextureObject_t densityTex, 
									cudaTextureObject_t stoppingPowerTex) 
	{

    const unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y*blockDim.y*gridDim.x*blockDim.x;
    unsigned int idx = y*gridDim.x*blockDim.x + x;

    // Compensate for value located at voxel corner instead of centre
    float3 pos = params.getStart(x, y) + make_float3(HALF, HALF, HALF);
    float3 step = params.getInc(x, y);
    float stepLen = params.stepLen(x, y);
    //float huPlus1000;
    float cumulSp = 0.0f;
    float cumulHuPlus1000 = 0.0f;
    int beforeFirstInside = -1;
    int lastInside = -1;

    for (unsigned int i=0; i<params.getSteps(); ++i) {
        //huPlus1000 = tex3D(imVolTex, pos.x, pos.y, pos.z) + 1000.0f;
        float huPlus1000 = 
        #if CUDART_VERSION < 12000
            tex3D(imVolTex, pos.x, pos.y, pos.z);
        #else
            tex3D<float>(imVolTex, pos.x, pos.y, pos.z);
        #endif
        cumulHuPlus1000 += huPlus1000;
        bevDensity[idx] = 
        #if CUDART_VERSION < 12000
            tex1D(densityTex, huPlus1000*params.getDensityScale() + HALF);
        #else
            tex1D<float>(densityTex, huPlus1000*params.getDensityScale() + HALF);
        #endif

        cumulSp += 
        #if CUDART_VERSION < 12000
            stepLen * tex1D(stoppingPowerTex, huPlus1000*params.getSpScale() + HALF);
        #else
            stepLen * tex1D<float>(stoppingPowerTex, huPlus1000*params.getSpScale() + HALF);
        #endif

        if (cumulHuPlus1000 < 150.0f) {
            beforeFirstInside = i;
        }
        if (huPlus1000 > 150.0f) {
            lastInside = i;
        }
        bevCumulSp[idx] = cumulSp;

        idx += memStep;
        pos += step;
    }
    beamFirstInside[y*gridDim.x*blockDim.x + x] = beforeFirstInside+1;
    firstStepOutside[y*gridDim.x*blockDim.x + x] = lastInside+1;
}


// IDD和Sigma计算内核
__global__ void calIddAndSigmaKernel(float* bevIdd,
                                    float* bevSigma,
                                    const float* bevWepl,
                                    const float* bevDensity,
                                    const deviceBevCoordSystem* coordSystems,
                                    const int* beamGroupIds,
                                    vec3i* roiIndex,
                                    int nRoi,
                                    int nBeam,
                                    cudaTextureObject_t iddTex,
                                    const float* energies,
                                    float cutoffSigma) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    
    while (tid < nRoi * nBeam) {
        int roiIdx = tid / nBeam;
        int beamIdx = tid % nBeam;
        
        if (roiIdx >= nRoi || beamIdx >= nBeam) {
            tid += totalThreads;
            continue;
        }
        
        int outputIdx = roiIdx * nBeam + beamIdx;
        float wepl = bevWepl[outputIdx];
        float density = bevDensity[outputIdx];
        float energy = energies[beamIdx];
        
        // 从IDD查找表获取剂量
        float energyIdx = (energy - 80.0f) / 2.5f; // 假设能量范围80-200 MeV，步长2.5
        float idd = tex2D<float>(iddTex, wepl + 0.5f, energyIdx + 0.5f);
        
        // 计算sigma（基于多重散射理论）
        float sigma = 2.3f + 290.0f / (wepl + 15.0f); // 经验公式
        sigma *= sqrtf(density); // 密度修正
        
        // 存储结果
        bevIdd[outputIdx] = idd;
        bevSigma[outputIdx] = sigma;
        
        tid += totalThreads;
    }
}

//#ifdef NUCLEAR_CORR
//__global__ void fillIddAndSigma(float* const bevDensity, float* const bevCumulSp, float* const bevIdd, float* const bevRSigmaEff, float* const rayWeights, float* const bevNucIdd, float* const bevNucRSigmaEff, float* const nucRayWeights, int* const nucIdcs, int* const firstInside, int* const firstOutside, int* const firstPassive, FillIddAndSigmaParams params
//#else
__global__ void fillIddAndSigma(float* const bevDensity, 
								float* const bevCumulSp,
								float* const bevIdd, 
								float* const bevRSigmaEff, 
								float* const rayWeights, 
								int* const firstInside, 
								int* const firstOutside, 
								int* const firstPassive, 
								FillIddAndSigmaParams params, 
								cudaTextureObject_t cumulIddTex, 
								cudaTextureObject_t rRadiationLengthTex
//#ifdef NUCLEAR_CORR
//, cudaTextureObject_t nucWeightTex, cudaTextureObject_t nucSqSigmaTex
//#endif
//#endif
) {
    const unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y*blockDim.y*gridDim.x*blockDim.x;
    unsigned int idx = y*gridDim.x*blockDim.x + x;

    bool beamLive = true;
    const int firstIn = firstInside[idx];
    unsigned int afterLast = min(firstOutside[idx], static_cast<int>(params.getAfterLastStep())); // In case doesn't get changed
    const float rayWeight = rayWeights[idx];
    if (rayWeight < RAY_WEIGHT_CUTOFF || afterLast < params.getFirstStep()) {
        beamLive = false;
        afterLast = 0;
    }

    float res = 0.0f;
    float rSigmaEff;
    //float sigma;
    float cumulSp;
    float cumulSpOld = 0.0f;
    float cumulDose;
    float cumulDoseOld = 0.0f;
    //float stepLength = params.stepLen(x,y);
    params.initStepAndAirDiv(/*x,y*/);

    const float pInv = 0.5649718f; // 1/p, p=1.77
    const float eCoef = 8.639415f; // (10*alpha)^(-1/p), alpha=2.2e-3
    const float sqrt2 = 1.41421356f; // sqrt(2.0f)
/*
#ifdef NUCLEAR_CORR

    // CORRECT ALL THESE

#if NUCLEAR_CORR == SOUKUP
    const float eRefSq = 190.44f; // 13.8^2, E_s^2
    const float sigmaDelta = 0.0f;
#elif NUCLEAR_CORR == FLUKA
    const float eRefSq = 216.09f; // 14.7^2, E_s^2
    const float sigmaDelta = 0.08f;
#elif NUCLEAR_CORR == GAUSS_FIT
    const float eRefSq = 169.00; // 13.0^2, E_s^2
    const float sigmaDelta = 0.06f;
#endif
*/
//#else // NUCLEAR_CORR
    const float eRefSq = 198.81f; // 14.1^2, E_s^2
    const float sigmaDelta = 0.21f;
//#endif // NUCLEAR_CORR


    float incScat = 0.0f;
    float incincScat = 0.0f;
    // Value of increment when getting to params.getFirstStep()
    float incDiv = params.getSigmaSqAirLin() + (2.0f*float(params.getFirstStep()) - 1.0f) * params.getSigmaSqAirQuad();
    float sigmaSq = -incDiv; // Compensate for first addition of incDiv

/*
#ifdef NUCLEAR_CORR
    float nucRes = 0.0f;
    float nucRSigmaEff;
    int nucIdx = nucIdcs[idx];
    float nucRayWeight;
    if (nucIdx >= 0)
    {
        nucRayWeight = nucRayWeights[nucIdx];
    }
    nucIdx += params.getFirstStep()*params.getNucMemStep();
#endif // NUCLEAR_CORR
*/

    idx += params.getFirstStep()*memStep; // Compensate for first layer not 0
    for (unsigned int stepNo=params.getFirstStep(); stepNo<params.getAfterLastStep(); ++stepNo) {
        if (beamLive) {
            cumulSp = bevCumulSp[idx];
            cumulDose =
            #if CUDART_VERSION < 12000
                tex2D(cumulIddTex, cumulSp*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
            #else
                tex2D<float>(cumulIddTex, cumulSp*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
            #endif

            float density = bevDensity[idx]; // Consistently used throughout?
            //float peakDepth = params.getPeakDepth();

            // Sigma peaks 1 - 2 mm before the BP
            if (cumulSp < (params.getPeakDepth()))
            {
                float resE = eCoef * __powf(params.getPeakDepth() - HALF*(cumulSp+cumulSpOld), pInv); // 7.1 / 16.5 ms for __powf / powf 128x128, 512 steps on laptop
                // See Rossi et al. 1941 p. 242 for expressions for calculationg beta*p
                float betaP = resE + 938.3f - 938.3f*938.3f / (resE+938.3f); // 2.1 ms for 128x128, 512 steps on laptop
                float rRl =
                #if CUDART_VERSION < 12000
                    density * tex1D(rRadiationLengthTex, density*params.getRRlScale() + HALF);
                #else
                    density * tex1D<float>(rRadiationLengthTex, density*params.getRRlScale() + HALF);
                #endif
                float thetaSq = eRefSq/(betaP*betaP) * params.getStepLength() * rRl;

                sigmaSq += incScat + incDiv; // Adding 0.25f * thetaSq * params.getStepLength() * params.getStepLength() makes no difference
                incincScat += 2.0f * thetaSq * params.getStepLength() * params.getStepLength();
                incScat += incincScat;
                incDiv += 2.0f * params.getSigmaSqAirQuad();
            }
            else
            {
#ifndef(NUCLEAR_CORR) || NUCLEAR_CORR != GAUSS_FIT
                sigmaSq -= 1.5f * (incScat + incDiv) * density; // Empirical solution to dip in sigma after BP
#endif // !defined(NUCLEAR_CORR) || NUCLEAR_CORR != GAUSS_FIT

            }

            // Todo: Change to account for different divergence in x and y?
            rSigmaEff = HALF*(params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / (sqrt2 * (sqrtf(sigmaSq) + sigmaDelta)); // Empirical widening of beam
            //sigma = sqrtf(sigmaSq) + sigmaDelta;
            if (cumulSp > params.getPeakDepth()*BP_DEPTH_CUTOFF || stepNo == afterLast) {
                beamLive = false;
                afterLast = stepNo;
            }

#ifdef DOSE_TO_WATER
            float mass = (cumulSp-cumulSpOld) * params.stepVol(stepNo);
#else // DOSE_TO_WATER
            float mass = density * params.stepVol(stepNo);
#endif // DOSE_TO_WATER

/*
#ifdef NUCLEAR_CORR
            if (mass > 1e-2f) // Avoid 0/0 and rippling effect in low density materials
            {
                float nucWeight =
                #if CUDART_VERSION < 12000
                tex2D(nucWeightTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #else
                tex2D<float>(nucWeightTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #endif
                res = (1.0f - nucWeight) * rayWeight * (cumulDose-cumulDoseOld) / mass;
                nucRes = nucWeight * nucRayWeight * (cumulDose-cumulDoseOld) / (mass*params.getSpotDist()*params.getSpotDist());
            }
            if (nucIdx >= 0)
            {
                float nucSqSigma =
                #if CUDART_VERSION < 12000
                tex2D(nucSqSigmaTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #else
                tex2D<float>(nucSqSigmaTex, HALF*(cumulSp+cumulSpOld)*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
                #endif
                nucRSigmaEff = HALF * params.getSpotDist() *(params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / (sqrt2 * sqrtf(sigmaSq + nucSqSigma + params.getEntrySigmaSq()));
            }
#else // NUCLEAR_CORR 
*/
            if (mass > 1e-2f) // Avoid 0/0 and ripling effect in low density materials
            {
                res = rayWeight * (cumulDose-cumulDoseOld) / mass;
            }
//#endif // NUCLEAR_CORR

            cumulSpOld = cumulSp;
            cumulDoseOld = cumulDose;
        }
        if (!beamLive || static_cast<int>(stepNo)<(firstIn-1)) {
            res = 0.0f;
            rSigmaEff = __int_as_float(0x7f800000); // inf, equals sigma = 0
/*
#ifdef NUCLEAR_CORR
            nucRes = 0.0f;
            nucRSigmaEff = __int_as_float(0x7f800000); // inf, equals sigma = 0
#endif // NUCLEAR_CORR
*/
            //sigma = 0.0f;
        }
        bevIdd[idx] = res;
        bevRSigmaEff[idx] = rSigmaEff;
        //bevRSigmaEff[idx] = HALF*(params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) / (sqrt2 * sigma);
        //bevRSigmaEff[idx] = 1.0f / (sqrt2 * sigma);
        //bevRSigmaEff[idx] = params.voxelWidth(stepNo).x / (sqrt2 * sigma);
/*
#ifdef NUCLEAR_CORR
        if (nucIdx >= 0)
        {
            bevNucIdd[nucIdx] = nucRes;
            bevNucRSigmaEff[nucIdx] = nucRSigmaEff;
        }
        nucIdx += params.getNucMemStep();
#endif // NUCLEAR_CORR 
*/

        idx += memStep;
    }
    firstPassive[y*gridDim.x*blockDim.x + x] = afterLast;
}


// 高效剂量计算内核（基于scatter和tile batch方法）
__global__ void calOptimizedDoseKernel(float* finalDose,
                                      const float* bevIdd,
                                      const float* bevSigma,
                                      const deviceBevCoordSystem* coordSystems,
                                      const int* beamGroupIds,
                                      const int* groupToOriginalBeamMap,
                                      vec3i* roiIndex,
                                      int nRoi,
                                      int nBeam,
                                      Grid doseGrid,
                                      float cutoffSigma) {
    
    // 使用共享内存进行tile batch处理
    __shared__ float tileData[32][32];
    __shared__ vec3f roiPositions[256]; // 缓存ROI位置
    
    int blockRoiIdx = blockIdx.x;
    int threadIdx1D = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 每个block处理一个ROI tile
    if (blockRoiIdx >= nRoi) return;
    
    // 加载ROI位置到共享内存
    if (threadIdx1D < min(256, nRoi - blockRoiIdx * 256)) {
        int globalRoiIdx = blockRoiIdx * 256 + threadIdx1D;
        if (globalRoiIdx < nRoi) {
            roiPositions[threadIdx1D] = vec3f(
                roiIndex[globalRoiIdx].x * doseGrid.resolution.x + doseGrid.corner.x,
                roiIndex[globalRoiIdx].y * doseGrid.resolution.y + doseGrid.corner.y,
                roiIndex[globalRoiIdx].z * doseGrid.resolution.z + doseGrid.corner.z
            );
        }
    }
    __syncthreads();
    
    // 每个线程处理一个beam
    int beamIdx = threadIdx.x + blockIdx.y * blockDim.x;
    if (beamIdx >= nBeam) return;
    
    int groupId = beamGroupIds[beamIdx];
    deviceBevCoordSystem coordSys = coordSystems[groupId];
    
    // 处理当前ROI和beam的组合
    for (int localRoiIdx = 0; localRoiIdx < min(256, nRoi - blockRoiIdx * 256); localRoiIdx++) {
        int globalRoiIdx = blockRoiIdx * 256 + localRoiIdx;
        if (globalRoiIdx >= nRoi) break;
        
        vec3f roiPos = roiPositions[localRoiIdx];
        
        // 转换到发散坐标系
        vec3f divPos = worldToBevDevice(roiPos, coordSys);
        
        // 获取IDD和sigma
        int dataIdx = globalRoiIdx * nBeam + beamIdx;
        float idd = bevIdd[dataIdx];
        float sigma = bevSigma[dataIdx];
        
        // 计算3sigma范围内的剂量贡献
        float dose = 0.0f;
        float cutoffRadius = cutoffSigma * sigma;
        
        // 高斯核卷积 - 优化版本
        for (int dy = -int(cutoffRadius) - 1; dy <= int(cutoffRadius) + 1; dy++) {
            for (int dx = -int(cutoffRadius) - 1; dx <= int(cutoffRadius) + 1; dx++) {
                float r2 = dx * dx + dy * dy;
                if (r2 <= cutoffRadius * cutoffRadius) {
                    // 高斯权重
                    float weight = expf(-r2 / (2.0f * sigma * sigma)) / (2.0f * M_PI * sigma * sigma);
                    dose += idd * weight;
                }
            }
        }
        
        // 原子写入最终结果
        atomicAdd(&finalDose[globalRoiIdx * nBeam + beamIdx], dose);
    }
}

// 纹理插值和坐标转换内核
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
        
        // 获取世界坐标
        vec3f worldPos = vec3f(roiIndex[roiIdx].x * doseGrid.resolution.x + doseGrid.corner.x,
                              roiIndex[roiIdx].y * doseGrid.resolution.y + doseGrid.corner.y,
                              roiIndex[roiIdx].z * doseGrid.resolution.z + doseGrid.corner.z);
        
        // 转换到BEV发散坐标系
        int groupId = beamGroupIds[beamIdx];
        deviceBevCoordSystem coordSys = coordSystems[groupId];
        vec3f bevPos = worldToBevDevice(worldPos, coordSys);
        
        // 纹理坐标计算
        float texX = (bevPos.x + texWidth/2.0f) / texWidth * texWidth;
        float texY = (bevPos.y + texHeight/2.0f) / texHeight * texHeight;
        float texZ = (bevPos.z + texDepth/2.0f) / texDepth * texDepth;
        
        // 确保在纹理范围内
        texX = fmaxf(0.0f, fminf(texWidth - 1.0f, texX));
        texY = fmaxf(0.0f, fminf(texHeight - 1.0f, texY));
        texZ = fmaxf(0.0f, fminf(texDepth - 1.0f, texZ));
        
        // 存储到BEV纹理中
        int texIdx = int(texZ) * texWidth * texHeight + int(texY) * texWidth + int(texX);
        int doseIdx = roiIdx * nBeam + beamIdx;
        bevDoseTexture[texIdx] = doseInDivergentCoord[doseIdx];
        
        tid += totalThreads;
    }
}

// 新增：Host launchers for BEV ray tracing and IDD/Sigma steps
void launchCalWaterEquivalentPathKernel(float* bevWepl,
                                        float* bevDensity,
                                        const deviceBevCoordSystem* coordSystems,
                                        const int* beamGroupIds,
                                        const int* groupToOriginalBeamMap,
                                        vec3i* roiIndex,
                                        int nRoi,
                                        int nBeam,
                                        Grid doseGrid,
                                        cudaTextureObject_t densityTex,
                                        cudaTextureObject_t spTex,
                                        int nSteps,
                                        float stepSize,
                                        int gridSize,
                                        int blockSize)
{
    calWaterEquivalentPathKernel<<<gridSize, blockSize>>>(bevWepl, bevDensity, coordSystems, beamGroupIds, groupToOriginalBeamMap,
                                                         roiIndex, nRoi, nBeam, doseGrid, densityTex, spTex, nSteps, stepSize);
}

void launchCalIddAndSigmaKernel(float* bevIdd,
                                float* bevSigma,
                                const float* bevWepl,
                                const float* bevDensity,
                                const deviceBevCoordSystem* coordSystems,
                                const int* beamGroupIds,
                                vec3i* roiIndex,
                                int nRoi,
                                int nBeam,
                                cudaTextureObject_t iddTex,
                                const float* energies,
                                float cutoffSigma,
                                int gridSize,
                                int blockSize)
{
    calIddAndSigmaKernel<<<gridSize, blockSize>>>(bevIdd, bevSigma, bevWepl, bevDensity, coordSystems, beamGroupIds, roiIndex, nRoi, nBeam, iddTex, energies, cutoffSigma);
}




// Core function: perform convolution for a single energy layer in a BEV group
// - devPrimIdd: ray dose before superposition (rX*rY*steps)
// - devPrimRSigmaEff: reciprocal sigma per ray step in BEV (rX*rY*steps)
// - devPrimInOutIdcs: input->output index mapping buffer for superposition
// - devTilePrimRadCtrs: per-radius tile counters (size maxSuperpR+2)
// - devBevPrimDose: output dose buffer in BEV (with 2*maxSuperpR padding in x/y)
// - primRayDims: (rX, rY, steps)
// - beamFirstInside: first step index inside patient for this beam
// - layerFirstPassive: last step (exclusive) where rays are live for this layer
// - maxNoPrimTiles: capacity of inOutIdcs per radius list
// The function reproduces: tileRadCalc -> build batches -> kernelSuperposition for rad 0..maxSuperpR
static void bev_run_convolution_layer_0814(float* devPrimIdd,
                                           float* devPrimRSigmaEff,
                                           int2* devPrimInOutIdcs,
                                           int* devTilePrimRadCtrs,
                                           float* devBevPrimDose,
                                           const uint3 primRayDims,
                                           const int beamFirstInside,
                                           const int layerFirstPassive,
                                           const int maxNoPrimTiles)
{
    // Grid/block for tile radial classification
    const int tileRadBlockY = 4; // matches RTD
    dim3 tileRadBlockDim(superpTileX, tileRadBlockY);

    // one tile per (superpTileX x superpTileY) region; z over steps in [beamFirstInside, layerFirstPassive)
    dim3 tilePrimRadGridDim(primRayDims.x/superpTileX, primRayDims.y/superpTileY, layerFirstPassive - beamFirstInside);

    // Zero counters
    CUDA_CHECK(cudaMemset(devTilePrimRadCtrs, 0, (maxSuperpR+2)*sizeof(int)));

    // Classify tiles by required superposition radius
    tileRadCalc<tileRadBlockY><<<tilePrimRadGridDim, tileRadBlockDim>>>(
        devPrimRSigmaEff,
        beamFirstInside,
        devTilePrimRadCtrs,
        devPrimInOutIdcs,
        maxNoPrimTiles);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy counters to host
    std::vector<int> tilePrimRadCtrs(maxSuperpR+2, 0);
    CUDA_CHECK(cudaMemcpy(tilePrimRadCtrs.data(), devTilePrimRadCtrs, (maxSuperpR+2)*sizeof(int), cudaMemcpyDeviceToHost));

    // Sanity: no tiles with radius > maxSuperpR
    if (tilePrimRadCtrs[maxSuperpR+1] > 0) {
        throw std::runtime_error("Found larger than allowed kernel superposition radius");
    }

    // Build batched tile counts from largest radius downward
    int layerMaxPrimSuperpR = 0;
    for (int i = 0; i < int(maxSuperpR+2); ++i) { if (tilePrimRadCtrs[i] > 0) layerMaxPrimSuperpR = i; }

    int recPrimRad = layerMaxPrimSuperpR;
    std::vector<int> batchedPrimTileRadCtrs(maxSuperpR+1, 0);
    batchedPrimTileRadCtrs[0] = tilePrimRadCtrs[0];
    for (int rad = layerMaxPrimSuperpR; rad > 0; --rad) {
        batchedPrimTileRadCtrs[recPrimRad] += tilePrimRadCtrs[rad];
        if (batchedPrimTileRadCtrs[recPrimRad] >= minTilesInBatch) {
            recPrimRad = rad - 1;
        }
    }

    // Launch kernel superposition for all radii with non-zero batches
    dim3 superpBlockDim(superpTileX, 8);
    // inDose pitch is primRayDims.x
    if (batchedPrimTileRadCtrs[0]  > 0) kernelSuperposition<0>  <<<batchedPrimTileRadCtrs[0],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[1]  > 0) kernelSuperposition<1>  <<<batchedPrimTileRadCtrs[1],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[2]  > 0) kernelSuperposition<2>  <<<batchedPrimTileRadCtrs[2],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[3]  > 0) kernelSuperposition<3>  <<<batchedPrimTileRadCtrs[3],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[4]  > 0) kernelSuperposition<4>  <<<batchedPrimTileRadCtrs[4],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[5]  > 0) kernelSuperposition<5>  <<<batchedPrimTileRadCtrs[5],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[6]  > 0) kernelSuperposition<6>  <<<batchedPrimTileRadCtrs[6],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[7]  > 0) kernelSuperposition<7>  <<<batchedPrimTileRadCtrs[7],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[8]  > 0) kernelSuperposition<8>  <<<batchedPrimTileRadCtrs[8],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[9]  > 0) kernelSuperposition<9>  <<<batchedPrimTileRadCtrs[9],  superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[10] > 0) kernelSuperposition<10> <<<batchedPrimTileRadCtrs[10], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[11] > 0) kernelSuperposition<11> <<<batchedPrimTileRadCtrs[11], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[12] > 0) kernelSuperposition<12> <<<batchedPrimTileRadCtrs[12], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[13] > 0) kernelSuperposition<13> <<<batchedPrimTileRadCtrs[13], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[14] > 0) kernelSuperposition<14> <<<batchedPrimTileRadCtrs[14], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[15] > 0) kernelSuperposition<15> <<<batchedPrimTileRadCtrs[15], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[16] > 0) kernelSuperposition<16> <<<batchedPrimTileRadCtrs[16], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[17] > 0) kernelSuperposition<17> <<<batchedPrimTileRadCtrs[17], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[18] > 0) kernelSuperposition<18> <<<batchedPrimTileRadCtrs[18], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[19] > 0) kernelSuperposition<19> <<<batchedPrimTileRadCtrs[19], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[20] > 0) kernelSuperposition<20> <<<batchedPrimTileRadCtrs[20], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[21] > 0) kernelSuperposition<21> <<<batchedPrimTileRadCtrs[21], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[22] > 0) kernelSuperposition<22> <<<batchedPrimTileRadCtrs[22], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[23] > 0) kernelSuperposition<23> <<<batchedPrimTileRadCtrs[23], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[24] > 0) kernelSuperposition<24> <<<batchedPrimTileRadCtrs[24], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[25] > 0) kernelSuperposition<25> <<<batchedPrimTileRadCtrs[25], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[26] > 0) kernelSuperposition<26> <<<batchedPrimTileRadCtrs[26], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[27] > 0) kernelSuperposition<27> <<<batchedPrimTileRadCtrs[27], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[28] > 0) kernelSuperposition<28> <<<batchedPrimTileRadCtrs[28], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[29] > 0) kernelSuperposition<29> <<<batchedPrimTileRadCtrs[29], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[30] > 0) kernelSuperposition<30> <<<batchedPrimTileRadCtrs[30], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[31] > 0) kernelSuperposition<31> <<<batchedPrimTileRadCtrs[31], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    if (batchedPrimTileRadCtrs[32] > 0) kernelSuperposition<32> <<<batchedPrimTileRadCtrs[32], superpBlockDim>>>(devPrimIdd, devPrimRSigmaEff, devBevPrimDose, primRayDims.x, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Public entry for testing convolution across energy layers, adapted to our BEV structures.
// For brevity, this function assumes that ray tracing (bevDensity/bevWepl) and IDD/sigma filling
// have been performed prior to the call, and that devPrimRSigmaEff and devPrimIdd are ready.
void bev_kernel_wrapper_test_run_layers_0814(const BevGroupParams_0814& group,
                                             const uint3 primRayDims,
                                             const int beamFirstInside,
                                             const std::vector<int>& layerFirstPassiveVec,
                                             float* devPrimIdd,
                                             float* devPrimRSigmaEff,
                                             int2* devPrimInOutIdcs,
                                             float* devBevPrimDose,
                                             const int maxNoPrimTiles)
{
    // Allocate per-radius tile counters on device
    int* devTilePrimRadCtrs = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devTilePrimRadCtrs, (maxSuperpR+2)*sizeof(int)));

    // Iterate energy layers and perform convolution
    for (size_t layerNo = 0; layerNo < layerFirstPassiveVec.size(); ++layerNo) {
        const int layerFirstPassive = layerFirstPassiveVec[layerNo];
        bev_run_convolution_layer_0814(devPrimIdd,
                                       devPrimRSigmaEff,
                                       devPrimInOutIdcs,
                                       devTilePrimRadCtrs,
                                       devBevPrimDose,
                                       primRayDims,
                                       beamFirstInside,
                                       layerFirstPassive,
                                       maxNoPrimTiles);
    }

    CUDA_CHECK(cudaFree(devTilePrimRadCtrs));
}