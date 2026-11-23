@echo off
rem Optional: installs xgboost, scikit-learn and shap into this tool's .venv (the tool works without them).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run "Install (run once).bat" first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install xgboost scikit-learn shap
.venv\Scripts\python.exe -m seismic_tool --check-ml
pause
