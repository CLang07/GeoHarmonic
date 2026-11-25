@echo off
rem Builds Documents\F3 Netherlands from the F3 download (Downloads\F3_Demo_2023).
rem Needs about 3 GB of free disk space and 5-20 minutes. Safe to run again.
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
set "VPY=%ROOT%\tools\Seismic Interpretation Tool\.venv\Scripts\python.exe"
title GeoHarmonic - F3 setup
if not exist "%VPY%" goto noinst
echo Building the F3 Netherlands project. Leave this window open.
echo.
"%VPY%" -u "%ROOT%\tools\setup_f3.py" %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo Done. Open "Documents\F3 Netherlands\F3 Netherlands.seisproj.json" in the Seismic Interpretation Tool.
if not "%RC%"=="0" echo Something went wrong - tell Claude, it will read "Documents\F3 Netherlands\setup_log.txt".
pause
exit /b %RC%
:noinst
echo The Seismic Interpretation Tool is not installed yet. Run "Install GeoHarmonic.bat" first.
pause
exit /b 1
