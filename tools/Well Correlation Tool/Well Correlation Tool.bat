@echo off
rem Well Correlation Tool - double-click, or drag a .wcproj.json project onto it
cd /d "%~dp0"
set PYQTGRAPH_QT_LIB=PySide6
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m well_correlation %*
) else (
  echo Run "Install (run once).bat" first.
  pause
  exit /b 1
)
if errorlevel 1 pause
