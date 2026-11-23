@echo off
rem Seismic Interpretation Tool - double-click, or drag a .seisproj.json project onto it
cd /d "%~dp0"
set PYQTGRAPH_QT_LIB=PySide6
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m seismic_tool %*
) else (
  echo Run "Install (run once).bat" first.
  pause
  exit /b 1
)
if errorlevel 1 pause
