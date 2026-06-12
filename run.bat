@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    set "PYTHON_ARGS="
    goto check_dependencies
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3.11"
        goto check_dependencies
    )

    py -3.10 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3.10"
        goto check_dependencies
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    set "PYTHON_ARGS="
    goto check_dependencies
)

echo Python was not found.
echo Please install Python 3.10 or 3.11, then run:
echo     py -3.11 -m pip install -r requirements.txt
pause
exit /b 1

:check_dependencies
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import cv2, mediapipe as mp, pyautogui; assert hasattr(mp, 'solutions')" >nul 2>nul
if errorlevel 1 (
    echo Required dependencies are missing or incompatible.
    echo Please run:
    echo     "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% main.py
if errorlevel 1 (
    echo.
    echo The program exited with an error.
    pause
)

endlocal
