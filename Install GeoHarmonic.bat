@echo off
rem ======================================================================
rem  GeoHarmonic one-click installer
rem  Double-click this file. It sets up the Well Correlation Tool and the
rem  Seismic Interpretation Tool, runs every test, and puts shortcuts on
rem  your desktop. Safe to run again at any time (it repairs / updates).
rem  Written without ( ) blocks on purpose: a folder name containing ")"
rem  would otherwise break the script.
rem ======================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "LOG=%ROOT%\install_log.txt"
set "FAILS=0"
set "FAILLIST="
set "PYQTGRAPH_QT_LIB=PySide6"
title GeoHarmonic installer

echo GeoHarmonic install log - %DATE% %TIME% > "%LOG%"
echo Folder: %ROOT% >> "%LOG%"
echo.
echo  ============================================================
echo   GeoHarmonic installer
echo   Folder: %ROOT%
echo   Takes 5-15 minutes the first time. Leave this window open.
echo   Everything is also written to install_log.txt
echo  ============================================================
echo.

if not exist "%ROOT%\tools\Well Correlation Tool\well_correlation" goto wrongplace

echo [1/6] Unblocking downloaded files...
echo Step 1: unblocking >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -LiteralPath $env:ROOT -Recurse -File | Unblock-File" >nul 2>&1

echo [2/6] Looking for Python...
echo Step 2: looking for Python >> "%LOG%"
set "PY="
call :trypy "%USERPROFILE%\anaconda3\python.exe"
if not defined PY call :trypy "%USERPROFILE%\miniconda3\python.exe"
if not defined PY call :trypy "%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY call :trypy "%ProgramData%\anaconda3\python.exe"
if not defined PY call :trypylauncher
if not defined PY call :trypy python
if not defined PY goto nopython
"%PY%" -c "import sys;print(sys.version.split()[0])" > "%TEMP%\gh_pyver.txt" 2>nul
set /p PYVER=<"%TEMP%\gh_pyver.txt"
echo       Using Python %PYVER% at %PY%
echo Python %PYVER%: %PY% >> "%LOG%"
"%PY%" -c "import sys;sys.exit(0 if sys.version_info[:2]<=(3,11) else 1)" || echo       WARNING: PySide6 6.5.3 needs Python 3.11 or older - the package step may fail.

echo [3/6] Well Correlation Tool
set "GUIMIN=10"
call :tool "Well Correlation Tool" well_correlation
echo [4/6] Seismic Interpretation Tool
set "GUIMIN=20"
call :tool "Seismic Interpretation Tool" seismic_tool

echo [5/6] Mistie Tool test and desktop shortcuts
call :mistie
echo [6/6] Illustrated tutorial with screenshots from this PC, and desktop shortcuts
call :tutorial
call :shortcuts

echo. >> "%LOG%"
echo Failures: %FAILS% %FAILLIST% >> "%LOG%"
echo.
echo  ============================================================
if "%FAILS%"=="0" goto allgood
echo   Finished with %FAILS% problem(s):%FAILLIST%
echo   Tell Claude, it will read install_log.txt and the
echo   test_results.txt file in each tool folder.
echo  ============================================================
goto done
:allgood
echo   ALL GOOD. Both tools are installed and every test passed.
echo   Start them from the desktop shortcuts:
echo     GeoHarmonic - Well Correlation
echo     GeoHarmonic - Seismic Interpretation
echo   Tutorial: desktop shortcut "GeoHarmonic - Tutorial"
echo  ============================================================
:done
echo.
pause
exit /b 0

rem ----------------------------------------------------------------------
:trypy
rem %1 = a python.exe path or command. Sets PY to the real executable.
set "CAND="
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
"%~1" -c "import sys;print(sys.executable)" > "%TEMP%\gh_py.txt" 2>nul
set /p CAND=<"%TEMP%\gh_py.txt"
if not defined CAND exit /b 0
echo %CAND% | find /i "WindowsApps" >nul
if not errorlevel 1 exit /b 0
set "PY=%CAND%"
exit /b 0

:trypylauncher
py -3 -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
py -3 -c "import sys;print(sys.executable)" > "%TEMP%\gh_py.txt" 2>nul
set /p PY=<"%TEMP%\gh_py.txt"
exit /b 0

rem ----------------------------------------------------------------------
:tool
rem %1 = tool folder name, %2 = python package
set "TNAME=%~1"
set "TDIR=%ROOT%\tools\%~1"
set "PKG=%~2"
set "VPY=%TDIR%\.venv\Scripts\python.exe"
set "RES=%TDIR%\test_results.txt"
echo. >> "%LOG%"
echo ===== %TNAME% ===== >> "%LOG%"
if exist "%VPY%" goto tool_havevenv
echo       Creating its private Python environment (.venv)...
"%PY%" -m venv "%TDIR%\.venv" >> "%LOG%" 2>&1
if not exist "%VPY%" goto tool_venvfail
:tool_havevenv
echo       Installing numpy 1.x, matplotlib, PySide6 6.5.3, pyqtgraph 0.13.7...
"%VPY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%VPY%" -m pip install "numpy<2" matplotlib "PySide6==6.5.3" "pyqtgraph==0.13.7" >> "%LOG%" 2>&1
if errorlevel 1 goto tool_pipfail
"%VPY%" -c "from PySide6 import QtWidgets; import pyqtgraph; import PySide6; import numpy; print('Qt OK - PySide6', PySide6.__version__, '- numpy', numpy.__version__)" >> "%LOG%" 2>&1
if errorlevel 1 goto tool_qtfail
echo       Qt OK
pushd "%TDIR%"
echo       Running self-test...
echo ===== SELFTEST ===== > "%RES%"
"%VPY%" -u -m %PKG% --selftest >> "%RES%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" call :fail "%TNAME% self-test"
if "%RC%"=="0" echo       Self-test passed
echo. >> "%RES%"
echo       Running GUI test - real Qt, no window opens, stops itself after %GUIMIN% min...
echo ===== GUI TEST ===== >> "%RES%"
"%VPY%" -u "%ROOT%\tools\guitest_runner.py" %PKG% %GUIMIN% >> "%RES%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" call :fail "%TNAME% GUI test"
if "%RC%"=="0" echo       GUI test passed
popd
echo Results in %RES% >> "%LOG%"
exit /b 0
:tool_venvfail
echo       Could not create the environment. See install_log.txt
call :fail "%TNAME% environment"
exit /b 0
:tool_pipfail
echo       Package install failed (internet connection?). See install_log.txt
call :fail "%TNAME% packages"
exit /b 0
:tool_qtfail
echo       Qt would not load. See install_log.txt
call :fail "%TNAME% Qt"
exit /b 0

rem ----------------------------------------------------------------------
:mistie
rem The Mistie Tool gets its own environment too (numpy, scipy, matplotlib), so a mismatch in
rem Anaconda's base packages cannot break it. tkinter comes from the base Python.
set "MDIR=%ROOT%\tools\Mistie Tool"
set "MRES=%MDIR%\test_results.txt"
set "MPY=%MDIR%\.venv\Scripts\python.exe"
if not exist "%MDIR%\mistie_tool" exit /b 0
echo. >> "%LOG%"
echo ===== Mistie Tool ===== >> "%LOG%"
if exist "%MPY%" goto mistie_havevenv
echo       Creating the Mistie Tool environment...
"%PY%" -m venv "%MDIR%\.venv" >> "%LOG%" 2>&1
if not exist "%MPY%" goto mistie_venvfail
:mistie_havevenv
echo       Installing numpy, scipy, matplotlib for the Mistie Tool...
"%MPY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%MPY%" -m pip install numpy scipy matplotlib >> "%LOG%" 2>&1
if errorlevel 1 goto mistie_pipfail
if exist "%USERPROFILE%\anaconda3\Library\lib\tcl8.6" set "TCL_LIBRARY=%USERPROFILE%\anaconda3\Library\lib\tcl8.6"
if exist "%USERPROFILE%\anaconda3\Library\lib\tk8.6" set "TK_LIBRARY=%USERPROFILE%\anaconda3\Library\lib\tk8.6"
"%MPY%" -c "import tkinter; tkinter.Tcl(); print('tkinter OK')" >> "%LOG%" 2>&1
if errorlevel 1 echo       Note: tkinter did not load - the Mistie window may not open. See install_log.txt
pushd "%MDIR%"
echo ===== SELFTEST ===== > "%MRES%"
"%MPY%" -u -m mistie_tool --selftest >> "%MRES%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" call :fail "Mistie self-test"
if "%RC%"=="0" echo       Mistie self-test passed
popd
exit /b 0
:mistie_venvfail
call :fail "Mistie environment"
exit /b 0
:mistie_pipfail
call :fail "Mistie packages"
exit /b 0

rem ----------------------------------------------------------------------
:tutorial
rem Rebuilds the tutorial with real window screenshots into tutorial_local (not tracked by git).
rem A problem here is only a note: the tools still work and docs\TUTORIAL.html is always there.
set "TPY=%ROOT%\tools\Seismic Interpretation Tool\.venv\Scripts\python.exe"
set "TUT=%ROOT%\docs\TUTORIAL.html"
if not exist "%TPY%" exit /b 0
if not exist "%ROOT%\dev\build_tutorial.py" exit /b 0
echo       Building the tutorial: worked example on Documents\F3 Netherlands when it exists, else on a
echo       synthetic F3-like project; then the tool chapters - a few minutes, one line per chapter...
echo ===== Tutorial ===== >> "%LOG%"
if exist "%ROOT%\tutorial_local\TUTORIAL.html" del /q "%ROOT%\tutorial_local\TUTORIAL.html" >nul 2>&1
rem progress lines on screen; errors in install_log.txt; full log in tutorial_local\tutorial_build_log.txt
"%TPY%" -u "%ROOT%\dev\build_tutorial.py" --pc 2>> "%LOG%"
if exist "%ROOT%\tutorial_local\TUTORIAL.html" set "TUT=%ROOT%\tutorial_local\TUTORIAL.html"
if exist "%ROOT%\tutorial_local\TUTORIAL.html" echo       Tutorial built: tutorial_local\TUTORIAL.html
if not exist "%ROOT%\tutorial_local\TUTORIAL.html" echo       Note: tutorial not rebuilt - the shortcut opens docs\TUTORIAL.html. See install_log.txt
exit /b 0

rem ----------------------------------------------------------------------
:shortcuts
set "SC_WC=%ROOT%\tools\Well Correlation Tool\Well Correlation Tool.bat"
set "SC_SE=%ROOT%\tools\Seismic Interpretation Tool\Seismic Interpretation Tool.bat"
set "SC_MI=%ROOT%\tools\Mistie Tool\Mistie Tool.bat"
if not defined TUT set "TUT=%ROOT%\docs\TUTORIAL.html"
set "SC_TU=%TUT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$d=[Environment]::GetFolderPath('Desktop'); $w=New-Object -ComObject WScript.Shell;" ^
 "foreach($p in @(@('GeoHarmonic - Well Correlation',$env:SC_WC),@('GeoHarmonic - Seismic Interpretation',$env:SC_SE),@('GeoHarmonic - Mistie',$env:SC_MI),@('GeoHarmonic - Tutorial',$env:SC_TU))){" ^
 " if(Test-Path -LiteralPath $p[1]){ $s=$w.CreateShortcut((Join-Path $d ($p[0]+'.lnk'))); $s.TargetPath=$p[1];" ^
 " $s.WorkingDirectory=(Split-Path -LiteralPath $p[1]); $s.IconLocation='imageres.dll,109'; $s.Save() } }" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo       Could not make desktop shortcuts - start the tools from their folders instead.
if "%RC%"=="0" echo       Desktop shortcuts created
exit /b 0

rem ----------------------------------------------------------------------
:fail
set /a FAILS+=1
set "FAILLIST=%FAILLIST% [%~1]"
echo       PROBLEM: %~1
echo PROBLEM: %~1 >> "%LOG%"
exit /b 0

:wrongplace
echo This file must sit in the main GeoHarmonic folder (the one that holds
echo the "tools" and "data" folders). Move it there and run it again.
pause
exit /b 1

:nopython
echo No Python found. Install Anaconda (Individual Edition, "Just Me") from
echo https://www.anaconda.com/download and run this installer again.
echo No Python found >> "%LOG%"
pause
exit /b 1
