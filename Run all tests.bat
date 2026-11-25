@echo off
rem Re-runs every GeoHarmonic test without reinstalling anything.
rem Results: test_summary.txt here + test_results.txt in each tool folder.
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "SUM=%ROOT%\test_summary.txt"
set "PYQTGRAPH_QT_LIB=PySide6"
echo GeoHarmonic test summary - %DATE% %TIME% > "%SUM%"
echo Running tests - up to about 30 minutes. Please wait...
echo Progress is written to test_results.txt in each tool folder as it goes.
call :tool "Well Correlation Tool" well_correlation 10
call :tool "Seismic Interpretation Tool" seismic_tool 20
call :mistie
echo.
type "%SUM%"
echo.
echo Saved to test_summary.txt
pause
exit /b 0

:tool
set "TDIR=%ROOT%\tools\%~1"
set "VPY=%TDIR%\.venv\Scripts\python.exe"
set "RES=%TDIR%\test_results.txt"
if not exist "%VPY%" goto tool_noinst
pushd "%TDIR%"
echo ===== SELFTEST ===== > "%RES%"
"%VPY%" -u -m %~2 --selftest >> "%RES%" 2>&1
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" echo %~1 self-test: PASSED >> "%SUM%"
if not "%RC%"=="0" echo %~1 self-test: FAILED - see tools\%~1\test_results.txt >> "%SUM%"
echo. >> "%RES%"
echo ===== GUI TEST ===== >> "%RES%"
"%VPY%" -u "%ROOT%\tools\guitest_runner.py" %~2 %~3 >> "%RES%" 2>&1
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" echo %~1 GUI test: PASSED >> "%SUM%"
if not "%RC%"=="0" echo %~1 GUI test: FAILED - see tools\%~1\test_results.txt >> "%SUM%"
popd
exit /b 0
:tool_noinst
echo %~1: not installed - run Install GeoHarmonic.bat first >> "%SUM%"
exit /b 0

:mistie
set "MDIR=%ROOT%\tools\Mistie Tool"
set "PY=%MDIR%\.venv\Scripts\python.exe"
if exist "%PY%" goto mistie_run
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3" >nul 2>&1
set "PY=python"
:mistie_run
pushd "%MDIR%"
echo ===== SELFTEST ===== > "%MDIR%\test_results.txt"
"%PY%" -u -m mistie_tool --selftest >> "%MDIR%\test_results.txt" 2>&1
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" echo Mistie Tool self-test: PASSED >> "%SUM%"
if not "%RC%"=="0" echo Mistie Tool self-test: FAILED - see tools\Mistie Tool\test_results.txt >> "%SUM%"
popd
exit /b 0
