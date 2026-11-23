@echo off
rem Double-click to test the Seismic Interpretation Tool. Results are saved to test_results.txt in this folder.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run "Install (run once).bat" first.
  pause
  exit /b 1
)
set PYQTGRAPH_QT_LIB=PySide6
echo Running tests - this takes a few minutes. Please wait...
echo ===== SELFTEST ===== > test_results.txt
".venv\Scripts\python.exe" -m seismic_tool --selftest >> test_results.txt 2>&1
echo. >> test_results.txt
echo ===== GUI TEST ===== >> test_results.txt
".venv\Scripts\python.exe" -m seismic_tool --guitest >> test_results.txt 2>&1
echo.
findstr /b /c:"FAIL" /c:"SELFTEST" /c:"GUI TEST" /c:"Traceback" test_results.txt
echo.
echo Full results saved in test_results.txt (opening it now).
start "" notepad test_results.txt
pause
