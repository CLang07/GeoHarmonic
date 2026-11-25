@echo off
rem Mistie Tool launcher - double-click, or drag a project file onto it.
rem Uses its own environment (.venv, made by Install GeoHarmonic.bat); falls back to Anaconda.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto venv
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
python -m mistie_tool %*
if errorlevel 1 pause
exit /b
:venv
rem tkinter in a venv made from Anaconda needs to be told where Tcl/Tk live
if exist "%USERPROFILE%\anaconda3\Library\lib\tcl8.6" set "TCL_LIBRARY=%USERPROFILE%\anaconda3\Library\lib\tcl8.6"
if exist "%USERPROFILE%\anaconda3\Library\lib\tk8.6" set "TK_LIBRARY=%USERPROFILE%\anaconda3\Library\lib\tk8.6"
".venv\Scripts\python.exe" -m mistie_tool %*
if errorlevel 1 pause
