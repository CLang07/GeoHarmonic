@echo off
rem Removes what Install GeoHarmonic.bat added: the two .venv folders and the
rem desktop shortcuts. Your data, projects, tops and interpretations are NOT touched.
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%CD%"
echo This removes the GeoHarmonic Python environments (.venv) and desktop shortcuts.
echo Your data and project files stay where they are.
choice /c YN /m "Continue"
if errorlevel 2 exit /b 0
if exist "%ROOT%\tools\Well Correlation Tool\.venv" rmdir /s /q "%ROOT%\tools\Well Correlation Tool\.venv"
if exist "%ROOT%\tools\Seismic Interpretation Tool\.venv" rmdir /s /q "%ROOT%\tools\Seismic Interpretation Tool\.venv"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); Get-ChildItem -LiteralPath $d -Filter 'GeoHarmonic - *.lnk' | Remove-Item" >nul 2>&1
echo Done. Run Install GeoHarmonic.bat to put it back.
pause
