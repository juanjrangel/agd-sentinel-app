@echo off
title Iniciando AGD-Sentinel
echo Iniciando el motor predictivo, por favor espere...
cd /d "%~dp0"
call motor_python\scripts\env.bat
streamlit run app.py --server.port 8505