# RayTraceDicom Integration Project - Build Instructions (CentOS/Linux)

## 概述 (Overview)

这个项目集成了RayTraceDicom算法到carbonPBS项目中，提供了完整的质子束剂量计算功能。专门为CentOS/Linux系统优化。

## 文件结构 (File Structure)

- `Makefile` - 主构建文件（CentOS优化）
- `configure.sh` - 自动配置脚本（CentOS优化）
- `test_simplified_wrapper_optimizesuperposition.cpp` - 完整版本（包含pybind11）
- `test_simplified_wrapper_basic.cpp` - 基本版本（不包含pybind11）
- `bev_transforms_test.cuh` - BEV变换头文件
- `bev_kernel_wrapper_test.cuh` - 内核包装器头文件
- `bev_kernel_wrapper_test.cu` - 内核包装器实现

## 快速开始 (Quick Start)

### 1. 自动配置
```bash
# 运行配置脚本自动检测路径
chmod +x configure.sh
./configure.sh
```

### 2. 安装依赖（CentOS/RHEL）
```bash
# 安装所有必需的依赖包
make install-deps
```

### 3. 编译基本版本（推荐先测试）
```bash
# 编译不依赖pybind11的基本版本
make test-basic
# 或者
make raytrace_dicom_basic_test
```

### 4. 编译完整版本
```bash
# 编译包含pybind11的完整版本
make all
# 或者
make raytrace_dicom_test
```

## 依赖项 (Dependencies)

### 必需依赖
- CUDA Toolkit (>= 10.0)
- GCC/G++ (>= 7.0)
- Python 3.x (>= 3.6)
- Development Tools (CentOS)

### 可选依赖
- pybind11 (用于Python绑定)

## 安装依赖 (Installing Dependencies)

### CentOS/RHEL (推荐)
```bash
make install-deps
```

### 手动安装 (CentOS/RHEL)
```bash
# 安装开发工具
sudo yum groupinstall -y "Development Tools"

# 安装Python开发包
sudo yum install -y python3-devel python3-pip

# 安装CUDA工具包
sudo yum install -y cuda-toolkit

# 安装pybind11
pip3 install pybind11
```

### Ubuntu/Debian
```bash
make install-deps-ubuntu
```

## 编译选项 (Build Options)

### 基本编译
```bash
make                    # 编译所有版本
make raytrace_dicom_test    # 编译完整版本
make raytrace_dicom_basic_test  # 编译基本版本
```

### 调试版本
```bash
make debug              # 编译调试版本
```

### 优化版本
```bash
make release            # 编译优化版本
```

### 清理
```bash
make clean              # 清理编译文件
```

## 故障排除 (Troubleshooting)

### 1. Python.h 找不到
```bash
# 检查Python安装
make check-python

# 手动设置Python路径
export PYTHON_PATH=/usr/include/python3.8
```

### 2. pybind11 找不到
```bash
# 检查pybind11安装
python3 -c "import pybind11; print(pybind11.get_include())"

# 手动设置pybind11路径
export PYBIND11_PATH=/usr/local/include/pybind11
```

### 3. CUDA 找不到
```bash
# 检查CUDA安装
nvcc --version

# 手动设置CUDA路径
export CUDA_PATH=/usr/local/cuda
```

### 4. 编译错误
```bash
# 显示编译信息
make info

# 使用调试模式编译
make debug
```

### 5. CentOS特定问题

#### 缺少开发工具
```bash
sudo yum groupinstall -y "Development Tools"
```

#### Python版本问题
```bash
# 检查Python版本
python3 --version

# 如果需要特定版本
sudo yum install -y python38-devel
```

#### CUDA路径问题
```bash
# 检查CUDA安装位置
find /usr -name nvcc 2>/dev/null

# 添加到PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## 使用示例 (Usage Examples)

### 基本版本测试
```bash
# 编译并运行基本版本
make raytrace_dicom_basic_test
./raytrace_dicom_basic_test
```

### 完整版本测试
```bash
# 编译并运行完整版本
make raytrace_dicom_test
./raytrace_dicom_test
```

## 配置选项 (Configuration Options)

### 环境变量
- `CUDA_PATH` - CUDA安装路径
- `PYTHON_PATH` - Python头文件路径
- `PYBIND11_PATH` - pybind11头文件路径
- `PYTHON_VERSION` - Python版本

### 编译标志
- `CUDA_ARCH` - CUDA架构 (默认: sm_60)
- `CXXFLAGS` - C++编译标志
- `NVCCFLAGS` - NVCC编译标志

## CentOS特定配置

### 系统要求
- CentOS 7/8/9 或 RHEL 7/8/9
- 至少4GB RAM
- NVIDIA GPU with CUDA support

### 推荐的Python版本
- Python 3.8+ (推荐)
- Python 3.6+ (最低要求)

### 推荐的CUDA版本
- CUDA 11.x (推荐)
- CUDA 10.x (最低要求)

## 输出说明 (Output Description)

程序会输出以下信息：
- CUDA设备信息
- 测试数据统计
- 束流设置信息
- 能量数据信息
- RayTraceDicom算法执行过程
- 最终剂量统计

## 技术支持 (Technical Support)

如果遇到问题，请：
1. 运行 `make info` 检查配置
2. 运行 `make check-python` 检查Python安装
3. 尝试编译基本版本 `make test-basic`
4. 查看编译错误信息并相应调整路径
5. 检查CentOS系统日志：`journalctl -xe`

## 许可证 (License)

本项目基于RayTraceDicom和carbonPBS项目开发。
