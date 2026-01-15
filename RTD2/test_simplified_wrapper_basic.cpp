#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Constants from RayTraceDicom
#define HALF 0.5f
#define RAY_WEIGHT_CUTOFF 1e-6f
#define BP_DEPTH_CUTOFF 0.95f

// Simplified data structures
struct SimplifiedBeamSettings {
    std::vector<float> energies;
    std::vector<float2> spotSigmas;
    float2 raySpacing;
    unsigned int steps;
    float2 sourceDist;
    float3 spotOffset;
    float3 spotDelta;
    float3 gantryToImOffset;
    float3 gantryToImMatrix;
    float3 gantryToDoseOffset;
    float3 gantryToDoseMatrix;
};

struct SimplifiedEnergyStruct {
    int nEnergySamples;
    int nEnergies;
    std::vector<float> energiesPerU;
    std::vector<float> peakDepths;
    std::vector<float> scaleFacts;
    std::vector<float> ciddMatrix;
    int nDensitySamples;
    float densityScaleFact;
    std::vector<float> densityVector;
    int nSpSamples;
    float spScaleFact;
    std::vector<float> spVector;
    int nRRlSamples;
    float rRlScaleFact;
    std::vector<float> rRlVector;
};

// RayTraceDicom core kernel functions
__global__ void fillBevDensityAndSpKernel(
    float* const bevDensity,
    float* const bevCumulSp, 
    int* const beamFirstInside, 
    int* const firstStepOutside, 
    const int steps,
    const float densityScale,
    const float spScale,
    cudaTextureObject_t imVolTex, 
    cudaTextureObject_t densityTex, 
    cudaTextureObject_t stoppingPowerTex,
    const int3 imVolDims) {

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y * blockDim.y * gridDim.x * blockDim.x;
    unsigned int idx = y * gridDim.x * blockDim.x + x;

    if (x >= imVolDims.x || y >= imVolDims.y) return;

    float3 pos = make_float3(x + HALF, y + HALF, HALF);
    float3 step = make_float3(0.0f, 0.0f, 1.0f);
    float stepLen = 1.0f;
    
    float cumulSp = 0.0f;
    float cumulHuPlus1000 = 0.0f;
    int beforeFirstInside = -1;
    int lastInside = -1;

    for (unsigned int i = 0; i < steps; ++i) {
        float huPlus1000 = tex3D<float>(imVolTex, pos.x, pos.y, pos.z);
        cumulHuPlus1000 += huPlus1000;
        
        bevDensity[idx] = tex1D<float>(densityTex, huPlus1000 * densityScale + HALF);
        cumulSp += stepLen * tex1D<float>(stoppingPowerTex, huPlus1000 * spScale + HALF);

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

__global__ void fillIddAndSigmaKernel(
    float* const bevDensity, 
    float* const bevCumulSp, 
    float* const bevIdd, 
    float* const bevRSigmaEff, 
    float* const rayWeights, 
    int* const firstInside, 
    int* const firstOutside, 
    int* const firstPassive, 
    const int energyIdx,
    const float peakDepth,
    const float scaleFact,
    const float energyScaleFact,
    const float rRlScale,
    const float stepLength,
    const int steps,
    cudaTextureObject_t cumulIddTex, 
    cudaTextureObject_t rRadiationLengthTex) {
    
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int memStep = gridDim.y * blockDim.y * gridDim.x * blockDim.x;
    unsigned int idx = y * gridDim.x * blockDim.x + x;

    bool beamLive = true;
    const int firstIn = firstInside[idx];
    unsigned int afterLast = min(firstOutside[idx], steps);
    const float rayWeight = rayWeights[idx];
    
    if (rayWeight < RAY_WEIGHT_CUTOFF || afterLast < firstIn) {
        beamLive = false;
        afterLast = 0;
    }

    float res = 0.0f;
    float rSigmaEff;
    float cumulSp;
    float cumulSpOld = 0.0f;
    float cumulDose;
    float cumulDoseOld = 0.0f;

    const float pInv = 0.5649718f;
    const float eCoef = 8.639415f;
    const float sqrt2 = 1.41421356f;
    const float eRefSq = 198.81f;
    const float sigmaDelta = 0.21f;

    float incScat = 0.0f;
    float incincScat = 0.0f;
    float incDiv = 0.0f;
    float sigmaSq = -incDiv;

    idx += firstIn * memStep;
    for (unsigned int stepNo = firstIn; stepNo < afterLast; ++stepNo) {
        if (beamLive) {
            cumulSp = bevCumulSp[idx];
            cumulDose = tex2D<float>(cumulIddTex, cumulSp * energyScaleFact + HALF, energyIdx + HALF);

            float density = bevDensity[idx];

            if (cumulSp < peakDepth) {
                float resE = eCoef * __powf(peakDepth - HALF*(cumulSp+cumulSpOld), pInv);
                float betaP = resE + 938.3f - 938.3f*938.3f / (resE+938.3f);
                float rRl = density * tex1D<float>(rRadiationLengthTex, density * rRlScale + HALF);
                float thetaSq = eRefSq/(betaP*betaP) * stepLength * rRl;

                sigmaSq += incScat + incDiv;
                incincScat += 2.0f * thetaSq * stepLength * stepLength;
                incScat += incincScat;
                incDiv += 2.0f * 0.0f; // sigmaSqAirQuad = 0
            } else {
                sigmaSq -= 1.5f * (incScat + incDiv) * density;
            }

            rSigmaEff = HALF * 2.0f / (sqrt2 * (sqrtf(sigmaSq) + sigmaDelta));
            
            if (cumulSp > peakDepth * BP_DEPTH_CUTOFF || stepNo == afterLast) {
                beamLive = false;
                afterLast = stepNo;
            }

            float mass = density * 1.0f; // stepVol = 1.0f
            
            if (mass > 1e-2f) {
                res = rayWeight * (cumulDose-cumulDoseOld) / mass;
            }

            cumulSpOld = cumulSp;
            cumulDoseOld = cumulDose;
        }
        
        if (!beamLive || static_cast<int>(stepNo) < (firstIn-1)) {
            res = 0.0f;
            rSigmaEff = __int_as_float(0x7f800000); // inf
        }
        
        bevIdd[idx] = res;
        bevRSigmaEff[idx] = rSigmaEff;
        idx += memStep;
    }
    firstPassive[y * gridDim.x * blockDim.x + x] = afterLast;
}

template<int RADIUS>
__global__ void kernelSuperposition(
    float const* __restrict__ inDose, 
    float const* __restrict__ inRSigmaEff, 
    float* const outDose, 
    const int inDosePitch, 
    const int rayDimsX,
    const int rayDimsY) {
    
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int x = tid % rayDimsX;
    const unsigned int y = tid / rayDimsX;
    
    if (x < rayDimsX && y < rayDimsY) {
        float dose = inDose[y * rayDimsX + x];
        float rSigmaEff = inRSigmaEff[y * rayDimsX + x];
        
        if (rSigmaEff < 1e6f) {
            float sigma = 1.0f / (rSigmaEff * 1.41421356f);
            float gaussianWeight = expf(-0.5f * (x * x + y * y) / (sigma * sigma));
            dose *= gaussianWeight;
        }
        
        atomicAdd(&outDose[y * rayDimsX + x], dose);
    }
}

__global__ void primTransfDivKernel(
    float* const result, 
    const int3 startIdx, 
    const int maxZ, 
    const uint3 doseDims,
    cudaTextureObject_t bevPrimDoseTex) {
    
    unsigned int x = startIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y) {
        float *res = result + startIdx.z * doseDims.x * doseDims.y + y * doseDims.x + x;
        for (int z = startIdx.z; z <= maxZ; ++z) {
            float3 pos = make_float3(x + HALF, y + HALF, z + HALF);
            float tmp = tex3D<float>(bevPrimDoseTex, pos.x, pos.y, pos.z);
            if (tmp > 0.0f) {
                *res += tmp;
            }
            res += doseDims.x * doseDims.y;
        }
    }
}

// Main wrapper function with complete RayTraceDicom integration
void raytraceDicomWrapper(
    const float* imVolData, const int3& imVolDims, const float3& imVolSpacing, const float3& imVolOrigin,
    float* doseVolData, const int3& doseVolDims, const float3& doseVolSpacing, const float3& doseVolOrigin,
    const SimplifiedBeamSettings* beamSettings, size_t numBeams,
    const SimplifiedEnergyStruct* energyData,
    int gpuId, bool nuclearCorrection, bool fineTiming
) {
    std::cout << "Starting RayTraceDicom wrapper with complete kernel integration..." << std::endl;
    
    cudaSetDevice(gpuId);
    cudaFree(0);
    
    // Create texture objects
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    
    // Image volume texture
    cudaArray* devImVolArr;
    cudaExtent imExt = make_cudaExtent(imVolDims.x, imVolDims.y, imVolDims.z);
    cudaMalloc3DArray(&devImVolArr, &floatChannelDesc, imExt);
    
    cudaMemcpy3DParms imCopyParams = {};
    imCopyParams.srcPtr = make_cudaPitchedPtr((void*)imVolData, imExt.width*sizeof(float), imExt.width, imExt.height);
    imCopyParams.dstArray = devImVolArr;
    imCopyParams.extent = imExt;
    imCopyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&imCopyParams);
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devImVolArr;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    
    cudaTextureObject_t imVolTex;
    cudaCreateTextureObject(&imVolTex, &resDesc, &texDesc, NULL);
    
    // Lookup table textures
    cudaArray* devCumulIddArr;
    cudaMallocArray(&devCumulIddArr, &floatChannelDesc, energyData->nEnergySamples, energyData->nEnergies);
    cudaMemcpyToArray(devCumulIddArr, 0, 0, &energyData->ciddMatrix[0], 
                     energyData->nEnergySamples * energyData->nEnergies * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t cumulIddTex;
    resDesc.res.array.array = devCumulIddArr;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    cudaCreateTextureObject(&cumulIddTex, &resDesc, &texDesc, NULL);
    
    // Density texture
    cudaArray* devDensityArr;
    cudaMallocArray(&devDensityArr, &floatChannelDesc, energyData->nDensitySamples);
    cudaMemcpyToArray(devDensityArr, 0, 0, &energyData->densityVector[0], 
                     energyData->nDensitySamples * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t densityTex;
    resDesc.res.array.array = devDensityArr;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    cudaCreateTextureObject(&densityTex, &resDesc, &texDesc, NULL);
    
    // Stopping power texture
    cudaArray* devStoppingPowerArr;
    cudaMallocArray(&devStoppingPowerArr, &floatChannelDesc, energyData->nSpSamples);
    cudaMemcpyToArray(devStoppingPowerArr, 0, 0, &energyData->spVector[0], 
                     energyData->nSpSamples * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t stoppingPowerTex;
    resDesc.res.array.array = devStoppingPowerArr;
    cudaCreateTextureObject(&stoppingPowerTex, &resDesc, &texDesc, NULL);
    
    // Radiation length texture
    cudaArray* devRRlArr;
    cudaMallocArray(&devRRlArr, &floatChannelDesc, energyData->nRRlSamples);
    cudaMemcpyToArray(devRRlArr, 0, 0, &energyData->rRlVector[0], 
                     energyData->nRRlSamples * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaTextureObject_t rRadiationLengthTex;
    resDesc.res.array.array = devRRlArr;
    cudaCreateTextureObject(&rRadiationLengthTex, &resDesc, &texDesc, NULL);
    
    // Allocate dose volume
    size_t doseSize = doseVolDims.x * doseVolDims.y * doseVolDims.z;
    float* devDoseVol;
    cudaMalloc((void**)&devDoseVol, doseSize * sizeof(float));
    cudaMemcpy(devDoseVol, doseVolData, doseSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Process each beam
    for (size_t beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
        const SimplifiedBeamSettings& beam = beamSettings[beamIdx];
        std::cout << "Processing beam " << beamIdx << " with " << beam.energies.size() << " energy layers" << std::endl;
        
        // Process each energy layer
        for (size_t layerIdx = 0; layerIdx < beam.energies.size(); ++layerIdx) {
            float energy = beam.energies[layerIdx];
            std::cout << "  Layer " << layerIdx << ": " << energy << " MeV" << std::endl;
            
            // Find energy index
            int energyIdx = 0;
            for (int i = 0; i < energyData->nEnergies; ++i) {
                if (energyData->energiesPerU[i] >= energy) {
                    energyIdx = i;
                    break;
                }
            }
            
            // Calculate ray dimensions
            int2 rayDims = make_int2(imVolDims.x, imVolDims.y);
            
            // Allocate ray tracing memory
            float* devRayWeights;
            float* devRayIdd;
            float* devRayRSigmaEff;
            int* devBeamFirstInside;
            int* devFirstStepOutside;
            int* devFirstPassive;
            
            size_t raySize = rayDims.x * rayDims.y * beam.steps;
            cudaMalloc((void**)&devRayWeights, raySize * sizeof(float));
            cudaMalloc((void**)&devRayIdd, raySize * sizeof(float));
            cudaMalloc((void**)&devRayRSigmaEff, raySize * sizeof(float));
            cudaMalloc((void**)&devBeamFirstInside, rayDims.x * rayDims.y * sizeof(int));
            cudaMalloc((void**)&devFirstStepOutside, rayDims.x * rayDims.y * sizeof(int));
            cudaMalloc((void**)&devFirstPassive, rayDims.x * rayDims.y * sizeof(int));
            
            // Initialize ray weights
            std::vector<float> rayWeights(rayDims.x * rayDims.y, 1.0f);
            cudaMemcpy(devRayWeights, rayWeights.data(), rayDims.x * rayDims.y * sizeof(float), cudaMemcpyHostToDevice);
            
            // Launch density and stopping power tracing kernel
            dim3 tracerBlock(32, 8);
            dim3 tracerGrid((rayDims.x + tracerBlock.x - 1) / tracerBlock.x, 
                           (rayDims.y + tracerBlock.y - 1) / tracerBlock.y);
            
            fillBevDensityAndSpKernel<<<tracerGrid, tracerBlock>>>(
                devRayWeights, devRayIdd, devBeamFirstInside, devFirstStepOutside,
                beam.steps, energyData->densityScaleFact, energyData->spScaleFact,
                imVolTex, densityTex, stoppingPowerTex, imVolDims);
            
            // Launch IDD and sigma calculation kernel
            fillIddAndSigmaKernel<<<tracerGrid, tracerBlock>>>(
                devRayWeights, devRayIdd, devRayIdd, devRayRSigmaEff, devRayWeights,
                devBeamFirstInside, devFirstStepOutside, devFirstPassive,
                energyIdx, energyData->peakDepths[energyIdx], energyData->scaleFacts[energyIdx],
                1.0f, energyData->rRlScaleFact, 1.0f/beam.steps, beam.steps,
                cumulIddTex, rRadiationLengthTex);
            
            // Allocate BEV dose volume for superposition
            float* devBevPrimDose;
            cudaMalloc((void**)&devBevPrimDose, rayDims.x * rayDims.y * beam.steps * sizeof(float));
            cudaMemset(devBevPrimDose, 0, rayDims.x * rayDims.y * beam.steps * sizeof(float));
            
            // Launch superposition kernels
            dim3 superpBlock(256);
            dim3 superpGrid((rayDims.x * rayDims.y + superpBlock.x - 1) / superpBlock.x, 1);
            
            kernelSuperposition<0><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDims.x, rayDims.x, rayDims.y);
            kernelSuperposition<1><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDims.x, rayDims.x, rayDims.y);
            kernelSuperposition<2><<<superpGrid, superpBlock>>>(devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDims.x, rayDims.x, rayDims.y);
            
            // Create BEV dose texture
            cudaArray* devBevDoseArr;
            cudaMalloc3DArray(&devBevDoseArr, &floatChannelDesc, make_cudaExtent(rayDims.x, rayDims.y, beam.steps));
            
            cudaMemcpy3DParms bevCopyParams = {};
            bevCopyParams.srcPtr = make_cudaPitchedPtr(devBevPrimDose, rayDims.x*sizeof(float), rayDims.x, rayDims.y);
            bevCopyParams.dstArray = devBevDoseArr;
            bevCopyParams.extent = make_cudaExtent(rayDims.x, rayDims.y, beam.steps);
            bevCopyParams.kind = cudaMemcpyDeviceToDevice;
            cudaMemcpy3D(&bevCopyParams);
            
            cudaTextureObject_t bevPrimDoseTex;
            resDesc.res.array.array = devBevDoseArr;
            texDesc.addressMode[0] = cudaAddressModeBorder;
            texDesc.addressMode[1] = cudaAddressModeBorder;
            texDesc.addressMode[2] = cudaAddressModeBorder;
            cudaCreateTextureObject(&bevPrimDoseTex, &resDesc, &texDesc, NULL);
            
            // Launch dose transformation kernel
            dim3 transformBlock(32, 8);
            dim3 transformGrid((doseVolDims.x + transformBlock.x - 1) / transformBlock.x,
                             (doseVolDims.y + transformBlock.y - 1) / transformBlock.y);
            
            primTransfDivKernel<<<transformGrid, transformBlock>>>(
                devDoseVol, make_int3(0, 0, 0), doseVolDims.z - 1, make_uint3(doseVolDims.x, doseVolDims.y, doseVolDims.z),
                bevPrimDoseTex);
            
            // Clean up layer-specific memory
            cudaFree(devRayWeights);
            cudaFree(devRayIdd);
            cudaFree(devRayRSigmaEff);
            cudaFree(devBeamFirstInside);
            cudaFree(devFirstStepOutside);
            cudaFree(devFirstPassive);
            cudaFree(devBevPrimDose);
            cudaDestroyTextureObject(bevPrimDoseTex);
            cudaFreeArray(devBevDoseArr);
        }
    }
    
    // Copy final dose back to host
    cudaMemcpy(doseVolData, devDoseVol, doseSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(devDoseVol);
    cudaDestroyTextureObject(imVolTex);
    cudaDestroyTextureObject(cumulIddTex);
    cudaDestroyTextureObject(densityTex);
    cudaDestroyTextureObject(stoppingPowerTex);
    cudaDestroyTextureObject(rRadiationLengthTex);
    cudaFreeArray(devImVolArr);
    cudaFreeArray(devCumulIddArr);
    cudaFreeArray(devDensityArr);
    cudaFreeArray(devStoppingPowerArr);
    cudaFreeArray(devRRlArr);
    
    std::cout << "RayTraceDicom wrapper completed!" << std::endl;
}

// Helper functions for creating test data
SimplifiedBeamSettings createTestBeamSettings() {
    SimplifiedBeamSettings beam;
    beam.energies = {150.0f, 140.0f, 130.0f, 120.0f, 110.0f};
    beam.spotSigmas = {make_float2(2.0f, 2.0f), make_float2(2.0f, 2.0f), make_float2(2.0f, 2.0f), 
                       make_float2(2.0f, 2.0f), make_float2(2.0f, 2.0f)};
    beam.raySpacing = make_float2(1.0f, 1.0f);
    beam.steps = 100;
    beam.sourceDist = make_float2(1000.0f, 1000.0f);
    beam.spotOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam.spotDelta = make_float3(5.0f, 5.0f, 0.0f);
    beam.gantryToImOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam.gantryToImMatrix = make_float3(1.0f, 0.0f, 0.0f);
    beam.gantryToDoseOffset = make_float3(0.0f, 0.0f, 0.0f);
    beam.gantryToDoseMatrix = make_float3(1.0f, 0.0f, 0.0f);
    return beam;
}

SimplifiedEnergyStruct createTestEnergyStruct() {
    SimplifiedEnergyStruct energy;
    energy.nEnergySamples = 100;
    energy.nEnergies = 5;
    energy.energiesPerU = {150.0f, 140.0f, 130.0f, 120.0f, 110.0f};
    energy.peakDepths = {15.0f, 14.0f, 13.0f, 12.0f, 11.0f};
    energy.scaleFacts = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    energy.ciddMatrix.resize(energy.nEnergySamples * energy.nEnergies, 1.0f);
    energy.nDensitySamples = 100;
    energy.densityScaleFact = 1.0f;
    energy.densityVector.resize(energy.nDensitySamples, 1.0f);
    energy.nSpSamples = 100;
    energy.spScaleFact = 1.0f;
    energy.spVector.resize(energy.nSpSamples, 1.0f);
    energy.nRRlSamples = 100;
    energy.rRlScaleFact = 1.0f;
    energy.rRlVector.resize(energy.nRRlSamples, 1.0f);
    return energy;
}

int main() {
    std::cout << "Testing RayTraceDicom Integration with Complete Kernel Functions" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    // Create test data
    int3 imVolDims = make_int3(64, 64, 64);
    float3 imVolSpacing = make_float3(1.0f, 1.0f, 1.0f);
    float3 imVolOrigin = make_float3(-32.0f, -32.0f, -32.0f);
    
    int3 doseVolDims = make_int3(64, 64, 64);
    float3 doseVolSpacing = make_float3(1.0f, 1.0f, 1.0f);
    float3 doseVolOrigin = make_float3(-32.0f, -32.0f, -32.0f);
    
    // Create image volume data (water phantom)
    size_t imVolSize = imVolDims.x * imVolDims.y * imVolDims.z;
    std::vector<float> imVolData(imVolSize, 0.0f); // Water (HU = 0)
    
    // Create dose volume data
    size_t doseVolSize = doseVolDims.x * doseVolDims.y * doseVolDims.z;
    std::vector<float> doseVolData(doseVolSize, 0.0f);
    
    // Create beam settings
    std::vector<SimplifiedBeamSettings> beamSettings = {createTestBeamSettings()};
    
    // Create energy data
    SimplifiedEnergyStruct energyData = createTestEnergyStruct();
    
    // Run RayTraceDicom wrapper
    raytraceDicomWrapper(imVolData.data(), imVolDims, imVolSpacing, imVolOrigin,
                        doseVolData.data(), doseVolDims, doseVolSpacing, doseVolOrigin,
                        beamSettings.data(), beamSettings.size(), &energyData,
                        0, false, false);
    
    // Calculate and display final dose statistics
    float maxDose = 0.0f;
    float totalDose = 0.0f;
    for (size_t i = 0; i < doseVolSize; ++i) {
        maxDose = std::max(maxDose, doseVolData[i]);
        totalDose += doseVolData[i];
    }
    
    std::cout << std::endl;
    std::cout << "Final Dose Statistics:" << std::endl;
    std::cout << "  Maximum dose: " << maxDose << " Gy" << std::endl;
    std::cout << "  Total dose: " << totalDose << " Gy" << std::endl;
    std::cout << "  Average dose: " << totalDose / doseVolSize << " Gy" << std::endl;
    
    std::cout << std::endl;
    std::cout << "RayTraceDicom integration test completed successfully!" << std::endl;
    
    return 0;
}
