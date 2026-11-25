@echo off
rem Mistie Tool self-test (synthetic lines). Uses .venv if Install GeoHarmonic.bat made one.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto venv
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
python -m mistie_tool --selftest
pause
exit /b
:venv
".venv\Scripts\python.exe" -m mistie_tool --selftest
pause
