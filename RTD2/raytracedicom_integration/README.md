# RayTraceDicom Integration Project

## 概述

这是一个完整的RayTraceDicom项目集成，包含了原始项目的所有核心组件，重新组织为模块化的C++编译接口。

## 项目结构

```
raytracedicom_integration/
├── include/                          # 头文件目录
│   ├── raytracedicom_integration.h   # 主头文件
│   ├── ray_tracing.h                 # 射线追踪组件
│   ├── idd_sigma_calculation.h       # IDD和Sigma计算组件
│   ├── superposition_kernels.h      # 叠加核函数组件
│   └── utils.h                       # 工具函数
├── src/                              # 源代码目录
│   ├── ray_tracing.cu                # 射线追踪实现
│   ├── idd_sigma_calculation.cu      # IDD和Sigma计算实现
│   ├── superposition_kernels.cu     # 叠加核函数实现
│   ├── utils.cu                      # 工具函数实现
│   ├── raytracedicom_wrapper.cu      # 主包装函数实现
│   └── test_raytracedicom.cpp        # 测试应用程序
├── build/                            # 构建目录（自动创建）
├── bin/                              # 可执行文件目录（自动创建）
├── CMakeLists.txt                    # CMake构建文件
├── Makefile                          # Make构建文件
└── README.md                         # 本文档
```

## 核心组件

### 1. 射线追踪 (Ray Tracing)
- **文件**: `include/ray_tracing.h`, `src/ray_tracing.cu`
- **功能**: BEV变换、密度和阻止本领追踪
- **核函数**: `fillBevDensityAndSpKernel`

### 2. IDD和Sigma计算
- **文件**: `include/idd_sigma_calculation.h`, `src/idd_sigma_calculation.cu`
- **功能**: 积分深度剂量计算、有效半径Sigma计算
- **核函数**: `fillIddAndSigmaKernel`

### 3. 叠加核函数
- **文件**: `include/superposition_kernels.h`, `src/superposition_kernels.cu`
- **功能**: 模板化高斯卷积叠加、剂量变换
- **核函数**: `kernelSuperposition<RADIUS>`, `primTransfDivKernel`

### 4. 工具函数
- **文件**: `include/utils.h`, `src/utils.cu`
- **功能**: 纹理创建、内存管理、错误检查、调试工具

## 编译和运行

### 方法1: 使用编译脚本 (推荐)

#### Windows (需要Visual Studio或WSL)
```bash
# 使用编译脚本
.\compile_all.bat

# 运行程序
.\bin\test_raytracedicom.exe
```

#### Linux/CentOS
```bash
# 给脚本执行权限
chmod +x compile_all.sh

# 使用编译脚本
./compile_all.sh

# 运行程序
./bin/test_raytracedicom
```

### 方法2: 使用Makefile

```bash
# 安装依赖 (CentOS)
make install-deps

# 编译项目
make

# 运行测试
make test

# 清理构建文件
make clean

# 查看帮助
make help
```

### 方法3: 使用CMake

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake ..

# 编译项目
make

# 运行测试
./bin/test_raytracedicom
```

### 方法4: 手动编译

#### Windows (需要Visual Studio)
```bash
# 创建目录
mkdir build bin

# 编译CUDA文件
nvcc -std=c++14 -O2 -I./include -c src/ray_tracing.cu -o build/ray_tracing.o
nvcc -std=c++14 -O2 -I./include -c src/idd_sigma_calculation.cu -o build/idd_sigma_calculation.o
nvcc -std=c++14 -O2 -I./include -c src/superposition_kernels.cu -o build/superposition_kernels.o
nvcc -std=c++14 -O2 -I./include -c src/utils.cu -o build/utils.o
nvcc -std=c++14 -O2 -I./include -c src/raytracedicom_wrapper.cu -o build/raytracedicom_wrapper.o

# 编译C++文件
g++ -std=c++14 -O2 -I./include -c src/test_raytracedicom.cpp -o build/test_raytracedicom.o

# 链接
nvcc build/*.o -o bin/test_raytracedicom -lcudart
```

#### Linux/CentOS
```bash
# 创建目录
mkdir -p build bin

# 编译CUDA文件
nvcc -std=c++14 -O2 -I./include -c src/ray_tracing.cu -o build/ray_tracing.o
nvcc -std=c++14 -O2 -I./include -c src/idd_sigma_calculation.cu -o build/idd_sigma_calculation.o
nvcc -std=c++14 -O2 -I./include -c src/superposition_kernels.cu -o build/superposition_kernels.o
nvcc -std=c++14 -O2 -I./include -c src/utils.cu -o build/utils.o
nvcc -std=c++14 -O2 -I./include -c src/raytracedicom_wrapper.cu -o build/raytracedicom_wrapper.o

# 编译C++文件
g++ -std=c++14 -O2 -I./include -c src/test_raytracedicom.cpp -o build/test_raytracedicom.o

# 链接
nvcc build/*.o -o bin/test_raytracedicom -lcudart
```

## 系统要求

### 软件要求
- CUDA Toolkit 10.0+
- GCC 7.0+ 或兼容编译器
- CMake 3.10+ (可选)

### Windows环境解决方案

#### 方案1: 安装Visual Studio Build Tools
```bash
# 下载并安装 Visual Studio Build Tools
# https://visualstudio.microsoft.com/zh-hans/downloads/#build-tools-for-visual-studio-2022

# 或者使用Chocolatey安装
choco install visualstudio2022buildtools
```

#### 方案2: 使用WSL (Windows Subsystem for Linux)
```bash
# 启用WSL
wsl --install

# 在WSL中安装CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# 在WSL中编译项目
cd /mnt/c/Users/18000/Documents/RTD2/raytracedicom_integration
chmod +x compile_all.sh
./compile_all.sh
```

#### 方案3: 使用Docker
```bash
# 创建Dockerfile
cat > Dockerfile << EOF
FROM nvidia/cuda:11.8-devel-ubuntu20.04
RUN apt-get update && apt-get install -y g++ make
WORKDIR /app
COPY . .
RUN chmod +x compile_all.sh && ./compile_all.sh
CMD ["./bin/test_raytracedicom"]
EOF

# 构建和运行
docker build -t raytracedicom .
docker run --gpus all raytracedicom
```

### 硬件要求
- NVIDIA GPU with Compute Capability 3.0+
- 至少4GB GPU内存（推荐8GB+）

### CentOS系统依赖
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cuda-toolkit
```

## 使用方法

### 基本使用

```cpp
#include "raytracedicom_integration.h"

// 创建测试数据
int3 imVolDims = make_int3(64, 64, 64);
std::vector<float> imVolData(imVolDims.x * imVolDims.y * imVolDims.z, 0.0f);
std::vector<float> doseVolData(imVolDims.x * imVolDims.y * imVolDims.z, 0.0f);

// 创建束流设置
std::vector<RayTraceDicomBeamSettings> beamSettings = {*createRayTraceDicomBeamSettings()};

// 创建能量数据
RayTraceDicomEnergyStruct* energyData = createRayTraceDicomEnergyStruct();

// 运行RayTraceDicom计算
raytraceDicomWrapper(imVolData.data(), imVolDims, imVolSpacing, imVolOrigin,
                     doseVolData.data(), doseVolDims, doseVolSpacing, doseVolOrigin,
                     beamSettings.data(), beamSettings.size(), energyData,
                     0, false, true);

// 清理
destroyRayTraceDicomEnergyStruct(energyData);
```

## 输出示例

```
Testing RayTraceDicom Integration with Complete Kernel Functions
================================================================
CUDA Device Information:
Number of devices: 1
Device 0: NVIDIA GeForce RTX 3080
  Compute capability: 8.6
  Global memory: 10240 MB
  Multiprocessors: 68

GPU Memory Information:
  Total: 10240 MB
  Free: 10180 MB
  Used: 60 MB

Starting RayTraceDicom wrapper with complete kernel integration...
Processing beam 0 with 5 energy layers
  Layer 0: 150 MeV
  Layer 1: 140 MeV
  Layer 2: 130 MeV
  Layer 3: 120 MeV
  Layer 4: 110 MeV
RayTraceDicom wrapper completed!

Final Dose Statistics:
  Maximum dose: 0.123 Gy
  Total dose: 45.67 Gy
  Average dose: 0.017 Gy

RayTraceDicom integration test completed successfully!
```

## 许可证

本项目基于原始RayTraceDicom项目，遵循GPL-3.0许可证。
