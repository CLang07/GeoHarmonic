@echo off
rem Runs the built-in synthetic self-test
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
python -m mistie_tool --selftest
pause
