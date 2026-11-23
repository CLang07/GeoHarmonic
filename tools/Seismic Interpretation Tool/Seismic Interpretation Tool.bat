@echo off
rem Seismic Interpretation Tool - double-click, or drag a .seisproj.json project onto it
cd /d "%~dp0"
set PYQTGRAPH_QT_LIB=PySide6
if not exist ".venv\Scripts\python.exe" goto noenv
rem launch outside any ( ) block: a dragged path containing ")" would end the block
".venv\Scripts\python.exe" -m seismic_tool %*
if errorlevel 1 pause
exit /b
:noenv
echo Run "Install (run once).bat" first.
pause
exit /b 1
