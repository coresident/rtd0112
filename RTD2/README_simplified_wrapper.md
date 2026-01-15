# Simplified Protons Wrapper

## 概述

这是一个基于RayTraceDicom项目的简化protons wrapper函数，设计用于作为C++编译接口模块移植到其他项目。该接口参考了carbonPBS的pybind11设计模式，在保持核心功能的同时简化了复杂的接口结构。

## 设计思路

### 原始RayTraceDicom项目分析

原始的`cudaWrapperProtons`函数具有以下特点：
- 复杂的BeamSettings类结构，包含多个变换矩阵和参数
- 复杂的EnergyStruct数据结构，包含多个查找表
- 大量的CUDA内存管理和纹理对象创建
- 复杂的核函数调用链（fillBevDensityAndSp, fillIddAndSigma, kernelSuperposition等）
- 支持核修正（nuclear correction）等高级功能

### 简化策略

基于carbonPBS的pybind11接口设计，我们采用了以下简化策略：

1. **数据结构简化**：
   - 将复杂的BeamSettings类简化为`SimplifiedBeamSettings`结构体
   - 将复杂的EnergyStruct简化为`SimplifiedEnergyStruct`结构体
   - 使用简单的向量和基本类型替代复杂的类层次结构

2. **接口参数简化**：
   - 直接使用原始数据指针而不是复杂的包装类
   - 简化变换矩阵的表示（使用3x3矩阵的线性部分）
   - 减少可选参数的数量

3. **功能模块化**：
   - 保留核心的质子剂量计算流程
   - 简化复杂的核函数调用
   - 提供可选的核修正和精细计时功能

## 文件结构

```
RTD2/
├── simplified_protons_wrapper.cu      # 主要的CUDA实现文件
├── simplified_protons_wrapper.cuh     # 头文件，定义接口和数据结构
├── test_simplified_wrapper.cpp        # 测试文件，演示接口使用方法
├── CMakeLists.txt                     # CMake构建文件
└── README_simplified_wrapper.md       # 本文档
```

## 核心数据结构

### SimplifiedBeamSettings

```cpp
struct SimplifiedBeamSettings {
    std::vector<float> energies;           // 每个能量层的能量值
    std::vector<float2> spotSigmas;        // 每个能量层的点斑大小
    float2 raySpacing;                     // 射线间距
    unsigned int steps;                    // 射线追踪步数
    float2 sourceDist;                     // 源到等中心距离
    float3 spotOffset;                     // 点斑偏移
    float3 spotDelta;                      // 点斑间距
    float3 gantryToImOffset;              // 机架到图像变换偏移
    float3 gantryToImMatrix;              // 机架到图像变换矩阵
    float3 gantryToDoseOffset;            // 机架到剂量变换偏移
    float3 gantryToDoseMatrix;            // 机架到剂量变换矩阵
};
```

### SimplifiedEnergyStruct

```cpp
struct SimplifiedEnergyStruct {
    int nEnergySamples;                   // 能量样本数
    int nEnergies;                        // 能量层数
    std::vector<float> energiesPerU;      // 每个能量层的能量值
    std::vector<float> peakDepths;        // 每个能量层的峰值深度
    std::vector<float> scaleFacts;        // 缩放因子
    std::vector<float> ciddMatrix;        // 累积积分剂量矩阵
    
    int nDensitySamples;                  // 密度样本数
    float densityScaleFact;               // 密度缩放因子
    std::vector<float> densityVector;     // 密度向量（HU到密度转换）
    
    int nSpSamples;                       // 阻止本领样本数
    float spScaleFact;                    // 阻止本领缩放因子
    std::vector<float> spVector;          // 阻止本领向量
    
    int nRRlSamples;                      // 辐射长度样本数
    float rRlScaleFact;                   // 辐射长度缩放因子
    std::vector<float> rRlVector;         // 辐射长度向量
};
```

## 主要接口函数

### simplifiedProtonsWrapper

主要的包装函数，执行质子剂量计算：

```cpp
extern "C" void simplifiedProtonsWrapper(
    // 输入参数
    float* imVolData,                     // 3D图像体积数据（HU值）
    int3 imVolDims,                       // 图像体积尺寸
    float3 imVolSpacing,                  // 图像体积间距
    float3 imVolOrigin,                   // 图像体积原点
    
    // 输出参数
    float* doseVolData,                   // 3D剂量体积数据（输出）
    int3 doseVolDims,                     // 剂量体积尺寸
    float3 doseVolSpacing,                // 剂量体积间距
    float3 doseVolOrigin,                 // 剂量体积原点
    
    // 束流设置
    const SimplifiedBeamSettings* beamSettings,
    int numBeams,
    
    // 能量数据
    const SimplifiedEnergyStruct* energyData,
    
    // 控制参数
    int gpuId = 0,                        // GPU设备ID
    bool enableNuclearCorrection = false, // 启用核修正
    bool enableFineTiming = false         // 启用精细计时
);
```

### 辅助函数

```cpp
// 创建和销毁束流设置
SimplifiedBeamSettings* createSimplifiedBeamSettings(...);
void destroySimplifiedBeamSettings(SimplifiedBeamSettings* beam);

// 创建和销毁能量结构
SimplifiedEnergyStruct* createSimplifiedEnergyStruct(...);
void destroySimplifiedEnergyStruct(SimplifiedEnergyStruct* energy);
```

## 使用方法

### 1. 编译

```bash
mkdir build
cd build
cmake ..
make
```

### 2. 基本使用

```cpp
#include "simplified_protons_wrapper.cuh"

// 创建测试数据
std::vector<float> imVolData = /* 图像数据 */;
std::vector<float> doseVolData = /* 剂量数据 */;
std::vector<SimplifiedBeamSettings> beamSettings = /* 束流设置 */;
SimplifiedEnergyStruct energyData = /* 能量数据 */;

// 调用包装函数
simplifiedProtonsWrapper(
    imVolData.data(), imVolDims, imVolSpacing, imVolOrigin,
    doseVolData.data(), doseVolDims, doseVolSpacing, doseVolOrigin,
    beamSettings.data(), beamSettings.size(),
    &energyData,
    0, false, false
);
```

### 3. 测试

```bash
./test_simplified_wrapper
```

## 与原始RayTraceDicom的对比

| 特性 | 原始版本 | 简化版本 |
|------|----------|----------|
| 数据结构 | 复杂的类层次结构 | 简单的结构体 |
| 接口参数 | 多个复杂对象 | 基本类型和指针 |
| 内存管理 | 复杂的CUDA内存管理 | 简化的内存分配 |
| 核函数 | 完整的核函数链 | 简化的计算流程 |
| 核修正 | 完整支持 | 可选支持 |
| 精细计时 | 完整支持 | 可选支持 |
| 编译复杂度 | 高 | 低 |
| 接口易用性 | 复杂 | 简单 |

## 移植建议

### 1. 数据准备

在移植到其他项目时，需要确保：
- 图像数据格式兼容（HU值）
- 束流参数的正确转换
- 能量数据的格式匹配

### 2. 依赖管理

确保目标项目具有：
- CUDA运行时支持
- 兼容的C++编译器
- 必要的CUDA头文件

### 3. 性能优化

根据具体需求，可以：
- 调整CUDA网格和块大小
- 优化内存访问模式
- 添加更多的并行化

## 扩展性

该简化wrapper设计为可扩展的：
- 可以添加更多的控制参数
- 可以集成更多的核函数
- 可以支持更多的数据格式
- 可以添加Python绑定

## 注意事项

1. **内存管理**：确保正确释放CUDA内存和纹理对象
2. **错误处理**：添加适当的错误检查和异常处理
3. **性能监控**：在生产环境中监控GPU内存使用和计算性能
4. **数据验证**：验证输入数据的有效性和范围

## 联系信息

如有问题或建议，请参考原始RayTraceDicom项目和carbonPBS项目的文档。

