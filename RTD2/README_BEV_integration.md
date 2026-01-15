# BEV变换与简化Protons Wrapper集成说明

## 概述

本文档说明了如何将RayTraceDicom项目中的BEV（Beam's Eye View）变换功能集成到简化的protons wrapper中，确保核心的射线追踪和坐标变换功能得到保留。

## BEV变换的核心组件

### 1. float3_affine_test 类

这个类实现了3x3矩阵变换加偏移的仿射变换：

```cpp
class float3_affine_test {
    Matrix3x3 matrix3x3;  // 3x3变换矩阵
    float3 offset;         // 偏移向量
    
    float3 transformPoint(const float3& p) const;  // 变换点
    float3_affine_test inverse() const;            // 求逆变换
};
```

**用途**：处理图像坐标到机架坐标的变换，以及机架坐标到剂量坐标的变换。

### 2. float3IdxTransform_test 类

这个类实现了索引到物理坐标的线性变换：

```cpp
class float3IdxTransform_test {
    float3 delta;   // 各维度的间距
    float3 offset;  // 原点偏移
    
    float3 transformPoint(const float3& idx) const;  // 变换点
    float3IdxTransform_test inverse() const;         // 求逆变换
    float3IdxTransform_test shiftOffset(const float3& s) const;  // 偏移变换
};
```

**用途**：处理BEV坐标系到BEV索引的变换，以及点斑间距和偏移的计算。

### 3. Float3ToBevTransform_test 结构

这个结构实现了完整的图像索引到BEV坐标的变换链：

```cpp
struct Float3ToBevTransform_test {
    float3_affine_test imIdxToGantry;        // 图像索引到机架坐标
    float2 sourceDist;                        // 源到等中心距离
    float3IdxTransform_test bevToBevIdx;     // BEV坐标到BEV索引
    float3 imgCorner;                         // 图像角点
    
    float3 transformPoint(const float3& imIdx) const;  // 完整变换
    Float3ToBevTransform_test inverse() const;         // 逆变换
};
```

**变换流程**：
1. 图像索引 → 机架坐标（通过仿射变换）
2. 机架坐标 → BEV坐标（通过透视变换）
3. BEV坐标 → BEV索引（通过线性变换）

### 4. imgFromBevTransform_test 结构

这个结构实现了BEV索引到图像索引的逆变换：

```cpp
struct imgFromBevTransform_test {
    float3IdxTransform_test bevIdxToBev;     // BEV索引到BEV坐标
    float2 sourceDist;                        // 源到等中心距离
    float3_affine_test gantryToImIdx;        // 机架坐标到图像索引
    float3 imgCorner;                         // 图像角点
    
    float3 transformPoint(const float3& bevIdx) const;  // 逆变换
    Float3ToBevTransform_test inverse() const;          // 逆变换
};
```

## 与简化Protons Wrapper的集成

### 1. 数据结构映射

原始的RayTraceDicom复杂数据结构被映射到简化的结构：

| 原始组件 | 简化组件 | 说明 |
|----------|----------|------|
| BeamSettings | SimplifiedBeamSettings | 束流参数简化 |
| EnergyStruct | SimplifiedEnergyStruct | 能量数据简化 |
| Float3ToBevTransform | Float3ToBevTransform_test | BEV变换保留 |
| Float3FromBevTransform | imgFromBevTransform_test | 逆变换保留 |

### 2. 核心算法保留

以下RayTraceDicom的核心算法被保留并集成：

#### 密度和阻止本领追踪
```cpp
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
```

**功能**：
- 沿射线追踪计算累积阻止本领
- 检测组织边界（空气/组织界面）
- 计算密度和阻止本领值

#### IDD和Sigma计算
```cpp
__global__ void fillIddAndSigmaKernel(
    float* const bevDensity, 
    float* const bevCumulSp, 
    float* const bevIdd, 
    float* const bevRSigmaEff, 
    float* const rayWeights, 
    int* const firstInside, 
    int* const firstOutside, 
    int* const firstPassive, 
    const SimplifiedEnergyStruct* energyData,
    const int energyIdx,
    const float peakDepth,
    const float scaleFact,
    const int3 imVolDims,
    cudaTextureObject_t cumulIddTex, 
    cudaTextureObject_t rRadiationLengthTex);
```

**功能**：
- 计算积分深度剂量（IDD）
- 计算有效Sigma值（用于高斯叠加）
- 处理布拉格峰附近的散射效应

### 3. 纹理插值内核

```cpp
__global__ void textureInterpolationKernel(
    float* bevDoseTexture,
    const float* doseInDivergentCoord,
    const deviceBevCoordSystem* coordSystems,
    const int* beamGroupIds,
    vec3i* roiIndex,
    int nRoi, int nBeam,
    Grid doseGrid,
    int texWidth, int texHeight, int texDepth);
```

**功能**：
- 将发散坐标系中的剂量值插值到BEV纹理
- 支持多束流和多ROI的处理
- 处理坐标变换和纹理映射

## 使用示例

### 1. 创建BEV变换

```cpp
// 从简化的束流设置创建BEV变换
Float3ToBevTransform_test bevTransform = BevTransformUtils::createBevTransform(
    beam.spotOffset,           // 点斑偏移
    beam.spotDelta,            // 点斑间距
    beam.gantryToImOffset,     // 机架到图像偏移
    beam.gantryToImMatrix,     // 机架到图像矩阵
    beam.sourceDist,           // 源距离
    imVolOrigin                // 图像原点
);
```

### 2. 坐标变换

```cpp
// 图像坐标到BEV坐标
float3 bevPos = BevTransformUtils::imageToBev(imgPos, bevTransform);

// BEV坐标到图像坐标
float3 imgPos = BevTransformUtils::bevToImage(bevPos, bevTransform.inverse());
```

### 3. 在CUDA内核中使用

```cpp
// 在密度追踪内核中使用BEV变换
DensityAndSpTracerParams tracerParams(
    energyData->densityScaleFact, 
    energyData->spScaleFact, 
    beam.steps, 
    bevTransform
);

fillBevDensityAndSpKernel<<<tracerGrid, tracerBlock>>>(
    devRayWeights, devRayIdd, devRayRSigmaEff, 
    nullptr, nullptr, tracerParams, 
    imVolTex, densityTex, stoppingPowerTex, imVolDims);
```

## 性能优化建议

### 1. 内存访问优化

- 使用纹理内存进行3D图像数据访问
- 合理组织BEV坐标数据以减少内存跳转
- 利用CUDA的合并内存访问模式

### 2. 计算优化

- 预计算常用的变换矩阵逆
- 缓存BEV坐标变换结果
- 使用共享内存减少全局内存访问

### 3. 并行化策略

- 每个射线使用一个CUDA线程
- 利用2D网格结构进行射线并行处理
- 根据GPU架构调整块大小

## 扩展性

### 1. 添加新的变换类型

可以通过继承现有的变换类来添加新的坐标变换：

```cpp
class CustomTransform : public float3_affine_test {
    // 添加自定义变换逻辑
};
```

### 2. 支持更多束流类型

可以扩展SimplifiedBeamSettings来支持不同类型的束流：

```cpp
struct AdvancedBeamSettings : public SimplifiedBeamSettings {
    std::vector<float> spotWeights;     // 点斑权重
    std::vector<float3> spotPositions;  // 点斑位置
    // 其他高级参数
};
```

### 3. 集成更多物理模型

可以在现有的内核中添加更多的物理效应：

```cpp
// 核修正
if (enableNuclearCorrection) {
    // 添加核修正计算
}

// 精细计时
if (enableFineTiming) {
    // 添加时间相关计算
}
```

## 注意事项

### 1. 内存管理

- 确保正确释放CUDA内存和纹理对象
- 注意BEV变换对象的生命周期管理
- 避免内存泄漏和重复释放

### 2. 错误处理

- 添加适当的CUDA错误检查
- 验证BEV变换参数的有效性
- 处理边界情况和异常输入

### 3. 数值稳定性

- 注意矩阵求逆的数值稳定性
- 处理接近奇异的变换矩阵
- 使用适当的数值容差

## 总结

通过集成BEV变换功能，简化的protons wrapper保持了RayTraceDicom项目的核心算法能力，同时提供了更简洁的接口。这种设计既满足了简化接口的需求，又确保了计算结果的准确性。

关键优势：
1. **功能完整**：保留了所有核心的坐标变换和射线追踪功能
2. **接口简洁**：简化了复杂的数据结构，提高了易用性
3. **性能优化**：保持了CUDA并行计算的性能优势
4. **扩展性强**：可以方便地添加新功能和优化

这种集成方案为将RayTraceDicom作为C++编译接口模块移植到其他项目提供了坚实的基础。

