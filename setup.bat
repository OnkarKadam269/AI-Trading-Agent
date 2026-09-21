@echo off
echo =========================================
echo KRONOS HYPER AGENT - INITIAL SETUP
echo =========================================
cd backend
echo Creating Virtual Environment...
python -m venv venv
echo Activating and Installing Dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo Setup Complete! You can now click run.bat
pause
