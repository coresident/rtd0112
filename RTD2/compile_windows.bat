@echo off
REM Windows compilation script for RayTraceDicom Integration Project

echo ========================================
echo RayTraceDicom Integration Project
echo Windows Compilation Script
echo ========================================

REM Check if nvcc is available
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: nvcc not found in PATH
    echo Please install CUDA Toolkit and add it to PATH
    pause
    exit /b 1
)

echo Found nvcc compiler

REM Check if Visual Studio is available
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Visual Studio compiler (cl.exe) not found
    echo Trying to use nvcc with default settings...
    
    REM Try to compile basic version
    echo Compiling basic version...
    nvcc -std=c++14 -O2 -I. -I./include -o raytrace_dicom_basic_test.exe test_simplified_wrapper_basic.cpp -lcudart
    
    if %errorlevel% equ 0 (
        echo ✓ Basic version compiled successfully!
        echo.
        echo To run the program:
        echo   raytrace_dicom_basic_test.exe
        echo.
    ) else (
        echo ✗ Compilation failed
        echo.
        echo Possible solutions:
        echo 1. Install Visual Studio Build Tools
        echo 2. Install CUDA Toolkit
        echo 3. Set up environment variables
        echo.
    )
) else (
    echo Found Visual Studio compiler
    
    REM Compile with Visual Studio
    echo Compiling basic version with Visual Studio...
    nvcc -std=c++14 -O2 -I. -I./include -o raytrace_dicom_basic_test.exe test_simplified_wrapper_basic.cpp -lcudart
    
    if %errorlevel% equ 0 (
        echo ✓ Basic version compiled successfully!
        echo.
        echo To run the program:
        echo   raytrace_dicom_basic_test.exe
        echo.
    ) else (
        echo ✗ Compilation failed
        echo.
        echo Please check the error messages above
        echo.
    )
)

echo ========================================
echo Compilation script completed
echo ========================================
pause
