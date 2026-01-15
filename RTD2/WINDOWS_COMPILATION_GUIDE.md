# Windows 编译和执行指南

## 当前状态
你的Windows环境缺少Visual Studio编译器（cl.exe），这是CUDA编译所必需的。

## 解决方案

### 方案1：安装Visual Studio Build Tools（推荐）
1. 下载并安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. 选择 "C++ build tools" 工作负载
3. 安装完成后，重新打开命令提示符
4. 运行编译脚本：
   ```cmd
   compile_windows.bat
   ```

### 方案2：使用WSL（Windows Subsystem for Linux）
1. 安装WSL2：
   ```cmd
   wsl --install
   ```
2. 在WSL中安装CUDA和编译工具：
   ```bash
   sudo apt update
   sudo apt install build-essential
   sudo apt install nvidia-cuda-toolkit
   ```
3. 在WSL中编译：
   ```bash
   chmod +x configure.sh test_compilation.sh
   ./test_compilation.sh
   ```

### 方案3：使用Docker
1. 安装Docker Desktop
2. 运行CUDA容器：
   ```cmd
   docker run --gpus all -it nvidia/cuda:11.8-devel-ubuntu20.04
   ```
3. 在容器中编译项目

## 快速测试

### 检查环境
```cmd
REM 检查nvcc
nvcc --version

REM 检查Visual Studio
where cl

REM 检查CUDA路径
echo %CUDA_PATH%
```

### 手动编译（如果环境配置正确）
```cmd
nvcc -std=c++14 -O2 -I. -I./include -o raytrace_dicom_basic_test.exe test_simplified_wrapper_basic.cpp -lcudart
```

### 执行程序
```cmd
raytrace_dicom_basic_test.exe
```

## 预期输出
如果编译成功，程序会输出：
- CUDA设备信息
- 测试数据统计
- RayTraceDicom算法执行过程
- 最终剂量统计

## 故障排除

### 常见问题
1. **nvcc fatal: Cannot find compiler 'cl.exe'**
   - 安装Visual Studio Build Tools
   - 或使用WSL/Docker

2. **CUDA not found**
   - 安装CUDA Toolkit
   - 设置环境变量

3. **Permission denied**
   - 以管理员身份运行命令提示符

### 环境变量设置
```cmd
REM 设置CUDA路径（根据你的安装位置调整）
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PATH=%CUDA_PATH%\bin;%PATH%
set LD_LIBRARY_PATH=%CUDA_PATH%\lib\x64;%LD_LIBRARY_PATH%
```

## 推荐方案
对于Windows环境，推荐使用 **方案1（Visual Studio Build Tools）** 或 **方案2（WSL）**，这样可以获得最佳的编译体验。
