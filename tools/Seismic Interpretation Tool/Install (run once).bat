@echo off
rem One-time setup. Creates a private Python environment (.venv) for this tool and installs
rem PySide6 6.5.3 + pyqtgraph 0.13.7 (same versions that work for the Well Correlation Tool).
cd /d "%~dp0"
set "PY=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
if not exist ".venv\Scripts\python.exe" "%PY%" -m venv .venv
if not exist ".venv\Scripts\python.exe" goto novenv
echo Installing packages - a few minutes the first time...
".venv\Scripts\python.exe" -m pip install numpy matplotlib "PySide6==6.5.3" "pyqtgraph==0.13.7"
echo.
echo Checking Qt...
".venv\Scripts\python.exe" -c "from PySide6 import QtWidgets; import pyqtgraph; print('Qt OK - PySide6', __import__('PySide6').__version__)"
echo.
echo Running self-test...
".venv\Scripts\python.exe" -m seismic_tool --selftest
echo.
echo Done. If you saw "Qt OK" and "SELFTEST PASSED" above, double-click "Seismic Interpretation Tool.bat".
pause
exit /b 0
:novenv
echo Could not create the environment with %PY%
pause
