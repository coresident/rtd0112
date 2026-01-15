@echo off
REM Quick recompile script for Windows

echo ========================================
echo Quick Recompile Script (Windows)
echo ========================================

REM Check if source file exists
if not exist test_simplified_wrapper_basic.cpp (
    echo Error: test_simplified_wrapper_basic.cpp not found!
    pause
    exit /b 1
)

REM Clean previous builds
echo Cleaning previous builds...
del *.o raytrace_dicom_basic_test.exe raytrace_dicom_test.exe 2>nul

REM Compile basic version
echo Compiling basic version...
nvcc -std=c++14 -O2 -I. -I./include -o raytrace_dicom_basic_test.exe test_simplified_wrapper_basic.cpp -lcudart

if %errorlevel% equ 0 (
    echo ✓ Basic version compiled successfully!
    
    REM Ask if user wants to run the program
    echo.
    set /p choice="Do you want to run the program now? (y/n): "
    if /i "%choice%"=="y" (
        echo Running raytrace_dicom_basic_test.exe...
        echo ========================================
        raytrace_dicom_basic_test.exe
    ) else (
        echo To run the program later, use: raytrace_dicom_basic_test.exe
    )
) else (
    echo ✗ Compilation failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo ========================================
echo Recompile completed!
echo ========================================
pause
