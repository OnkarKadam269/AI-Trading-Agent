@echo off
echo =========================================
echo LAUNCHING KRONOS HYPER AGENT
echo =========================================
cd backend
call venv\Scripts\activate.bat
python live_agent\kronos_hyper_trader.py
pause
