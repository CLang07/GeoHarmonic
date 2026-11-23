@echo off
rem Mistie Tool launcher - double-click, or drag a project file onto it
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
python -m mistie_tool %*
if errorlevel 1 pause
