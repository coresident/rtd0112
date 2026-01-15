# RayTraceDicom Integration Project - CentOS/Linux 部署指南

## 项目概述

本项目成功集成了RayTraceDicom算法到carbonPBS项目中，提供了完整的质子束剂量计算功能。所有文件已针对CentOS/Linux系统进行了优化。

## 文件清单

### 核心文件
- `Makefile` - 主构建文件（CentOS优化）
- `configure.sh` - 自动配置脚本（CentOS优化）
- `test_environment.sh` - 环境测试脚本
- `test_compilation.sh` - 编译测试脚本
- `BUILD_INSTRUCTIONS.md` - 详细构建说明

### 源代码文件
- `test_simplified_wrapper_optimizesuperposition.cpp` - 完整版本（包含pybind11）
- `test_simplified_wrapper_basic.cpp` - 基本版本（不包含pybind11）
- `bev_transforms_test.cuh` - BEV变换头文件
- `bev_kernel_wrapper_test.cuh` - 内核包装器头文件
- `bev_kernel_wrapper_test.cu` - 内核包装器实现

## 快速部署步骤

### 1. 环境检查
```bash
# 给脚本添加执行权限
chmod +x configure.sh test_environment.sh test_compilation.sh

# 运行环境测试
./test_environment.sh
```

### 2. 自动配置
```bash
# 运行配置脚本
./configure.sh
```

### 3. 安装依赖
```bash
# 安装所有必需的依赖包
make install-deps
```

### 4. 编译测试
```bash
# 先编译基本版本（推荐）
make test-basic

# 或者使用编译测试脚本
./test_compilation.sh

# 编译完整版本
make all
```

## 主要特性

### ✅ 已解决的问题
1. **头文件定位问题** - 通过自动配置脚本解决
2. **Python.h找不到** - 支持多种Python安装路径
3. **pybind11路径问题** - 自动检测和配置
4. **CUDA路径问题** - 支持多个CUDA版本
5. **编译错误** - 所有语法错误已修复
6. **CUDA类型重复定义** - 移除重复的float2/float3/int3/uint3定义
7. **cooperative_groups命名空间** - 修复命名空间使用问题

### ✅ 集成的功能
1. **完整的RayTraceDicom算法** - 包含叠加算法和卷积过程
2. **BEV变换** - 完整的坐标变换功能
3. **碳离子PBS接口兼容性** - 保持与carbonPBS的兼容
4. **剂量输出** - 在main函数中输出finaldose
5. **双版本支持** - 基本版本和完整版本

### ✅ CentOS优化
1. **yum包管理器支持** - 自动安装依赖
2. **Python路径适配** - 适配CentOS的Python安装路径
3. **CUDA工具包支持** - 支持CentOS的CUDA安装
4. **开发工具组** - 自动安装Development Tools
5. **64位库路径** - 适配CentOS的64位库路径

## 编译状态

### ✅ 编译就绪
- 所有语法错误已修复
- 头文件依赖已解决
- 类型定义完整
- 宏定义正确
- 函数声明完整
- **修复了CUDA类型重复定义问题**
- **修复了cooperative_groups命名空间问题**

### ✅ 测试就绪
- 基本版本可独立编译（不依赖pybind11）
- 完整版本包含所有功能
- 包含完整的测试数据生成
- 包含详细的输出信息
- **添加了编译测试脚本**

## 使用说明

### 基本版本测试
```bash
make raytrace_dicom_basic_test
./raytrace_dicom_basic_test
```

### 完整版本测试
```bash
make raytrace_dicom_test
./raytrace_dicom_test
```

### 调试模式
```bash
make debug
```

### 优化模式
```bash
make release
```

## 故障排除

### 常见问题
1. **Python.h找不到** → 运行 `make install-deps`
2. **pybind11找不到** → 运行 `pip3 install pybind11`
3. **CUDA找不到** → 运行 `sudo yum install cuda-toolkit`
4. **编译错误** → 运行 `make info` 检查配置

### 调试命令
```bash
make info          # 显示编译配置
make check-python  # 检查Python安装
./test_environment.sh  # 全面环境检查
./test_compilation.sh  # 编译测试
```

## 技术规格

### 系统要求
- CentOS 7/8/9 或 RHEL 7/8/9
- Python 3.6+ (推荐3.8+)
- CUDA 10.0+ (推荐11.x)
- GCC 7.0+
- 至少4GB RAM
- NVIDIA GPU with CUDA support

### 支持的CUDA架构
- 默认: sm_60
- 可配置: 通过修改Makefile中的CUDA_ARCH

### 编译标志
- C++14标准
- O2优化级别
- Wall警告级别
- fPIC位置无关代码

## 项目状态

### ✅ 完成状态
- [x] RayTraceDicom算法集成
- [x] BEV变换集成
- [x] carbonPBS接口兼容
- [x] CentOS/Linux优化
- [x] 自动配置脚本
- [x] 依赖管理
- [x] 编译系统
- [x] 测试框架
- [x] 文档完善

### 🎯 项目目标达成
1. **自洽性** ✅ - 所有组件正确集成
2. **准确性** ✅ - 算法逻辑保持正确
3. **编译成功** ✅ - 所有语法错误已修复
4. **CentOS兼容** ✅ - 专门针对CentOS优化
5. **易于部署** ✅ - 自动化配置和安装

## 总结

本项目已成功解决了所有编译问题，完全适配CentOS/Linux环境，并提供了完整的RayTraceDicom算法集成。项目现在可以：

1. **自动检测和配置**所有依赖路径
2. **一键安装**所有必需的包
3. **成功编译**基本版本和完整版本
4. **输出剂量结果**在main函数中
5. **提供完整的测试框架**

项目已准备好在CentOS系统中部署和使用。
