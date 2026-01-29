@echo off
echo Starting Virtual Robotic Arm Controller...
echo Using Python 3.12...

py -3.12 pybullet_sim/main.py

if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to start the application.
    echo Please make sure you have installed the dependencies:
    echo py -3.12 -m pip install -r requirements_windows.txt
    pause
)
pause
