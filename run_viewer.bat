@echo off
REM Launch the pipeline viewer on Windows without activating anything.
REM
REM `streamlit` is not on PATH unless the venv is activated, and cmd.exe does
REM not change drive with a bare `cd`, so the two usual failures are "not
REM recognized as an internal or external command" and silently staying on C:.
REM %~dp0 is this file's own folder, so both go away: double-click this file,
REM or run it from any directory on any drive.

setlocal
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    echo   No virtual environment found at:
    echo     %VENV_PY%
    echo.
    echo   Create it and install the dependencies:
    echo     python -m venv "%~dp0.venv"
    echo     "%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
    echo.
    exit /b 1
)

echo Starting the viewer. Press Ctrl+C in this window to stop it.
echo.
"%VENV_PY%" -m streamlit run "%~dp0app.py" %*
