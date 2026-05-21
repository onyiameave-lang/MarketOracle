@echo off
echo ===================================================
echo MarketOracle: Environment Setup
echo ===================================================

echo [1/3] Detecting Python installation...

:: Try the Python Launcher 'py' first
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto :create_venv
)

:: Try 'python' command and verify it's not just the MS Store stub
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :create_venv
)

echo ===================================================
echo [ERROR] Python was not found or is not working correctly.
echo 1. Download Python from: https://www.python.org/downloads/
echo 2. Run the installer and select "Modify" or "Install Now".
echo 3. IMPORTANT: Check the box "Add Python to PATH".
echo ===================================================
pause
exit /b 1

:create_venv
echo Creating virtual environment using %PYTHON_CMD%...
%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 ( echo Failed to create venv. && pause && exit /b 1 )

:: 2. Install modules using the internal pip
echo [2/3] Installing modules from requirements.txt...
venv\Scripts\python.exe -m pip install --upgrade pip --default-timeout=1000 --retries 10
venv\Scripts\python.exe -m pip install -r requirements.txt --default-timeout=1000 --retries 10
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

:: 3. Organize project folders
echo [3/3] Organizing project directories...
venv\Scripts\python.exe -c "from experts.db_handler import ensure_dirs; ensure_dirs()"

echo ===================================================
echo SETUP COMPLETE!
echo To start, run: venv\Scripts\activate
echo ===================================================
pause