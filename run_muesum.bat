@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: MUESUM — AI Historical Artwork Restoration Suite Launcher
:: ============================================================================

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONIOENCODING=utf-8"

:: Verify Python installation
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found on your system PATH.
    echo Please install Python 3.8+ or add it to your PATH environment variable.
    pause
    exit /b 1
)

:: Parse Command Argument
set "CMD=%1"

if "%CMD%"=="" goto launch_demo
if /i "%CMD%"=="demo" goto launch_demo
if /i "%CMD%"=="app" goto launch_demo
if /i "%CMD%"=="test" goto run_tests
if /i "%CMD%"=="tests" goto run_tests
if /i "%CMD%"=="restore" goto run_restore
if /i "%CMD%"=="help" goto show_help
if /i "%CMD%"=="-h" goto show_help
if /i "%CMD%"=="--help" goto show_help

echo [ERROR] Unknown command: %CMD%
echo.
goto show_help

:: ----------------------------------------------------------------------------
:: Launch Gradio Restoration Studio
:: ----------------------------------------------------------------------------
:launch_demo
echo ============================================================================
echo Starting MUESUM Interactive Restoration Studio...
echo ============================================================================
python app/demo.py
goto end

:: ----------------------------------------------------------------------------
:: Run Automated QA & Regression Test Suite
:: ----------------------------------------------------------------------------
:run_tests
echo ============================================================================
echo Executing MUESUM Automated QA & Regression Test Suite...
echo ============================================================================
python tests/run_all_tests.py
goto end

:: ----------------------------------------------------------------------------
:: Headless CLI Inpainting Restoration
:: ----------------------------------------------------------------------------
:run_restore
set "IMG=%2"
set "MSK=%3"
set "OUT=%4"

if "%IMG%"=="" set "IMG=release/sample_artwork.jpg"
if "%MSK%"=="" set "MSK=release/sample_mask.png"
if "%OUT%"=="" set "OUT=restored_output.png"

echo ============================================================================
echo Executing MUESUM Headless Inpainting Pipeline
echo Image:  %IMG%
echo Mask:   %MSK%
echo Output: %OUT%
echo ============================================================================

python src/inference.py --image "%IMG%" --mask "%MSK%" --output "%OUT%" --blend_mode mask_guided --feather_radius 3
goto end

:: ----------------------------------------------------------------------------
:: Show Help / Usage Menu
:: ----------------------------------------------------------------------------
:show_help
echo ============================================================================
echo MUESUM — AI Historical Artwork Restoration Suite
echo ============================================================================
echo Usage:
echo   run_muesum.bat [command] [options]
echo.
echo Commands:
echo   demo                   Launch the interactive Gradio Restoration Studio (default)
echo   test                   Run all 29 automated QA and unit regression tests
echo   restore [img] [msk] [out]  Restore a damaged artwork using mask-guided inpainting
echo   help                   Display this usage guide
echo.
echo Examples:
echo   run_muesum.bat
echo   run_muesum.bat demo
echo   run_muesum.bat test
echo   run_muesum.bat restore release/sample_artwork.jpg release/sample_mask.png restored.png
echo ============================================================================
goto end

:end
endlocal
