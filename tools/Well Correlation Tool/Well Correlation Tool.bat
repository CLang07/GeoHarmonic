@echo off
rem Well Correlation Tool - double-click, or drag a .wcproj.json project onto it
cd /d "%~dp0"
set PYQTGRAPH_QT_LIB=PySide6
if not exist ".venv\Scripts\python.exe" goto noenv
rem launch outside any ( ) block: a dragged path containing ")" would end the block
".venv\Scripts\python.exe" -m well_correlation %*
if errorlevel 1 pause
exit /b
:noenv
echo Run "Install (run once).bat" first.
pause
exit /b 1
