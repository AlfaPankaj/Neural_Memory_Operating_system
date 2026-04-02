@echo off
SET "PYTHON_EXE=C:\Users\PANKAJ\AppData\Local\Programs\Python\Python310\python.exe"
SET "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

REM Add CUDA bins to PATH
SET "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

REM -------------------------------------------------------------------
REM NMOS MODE CONFIG
REM Local stable mode (recommended for this machine):
REM   SET "NMOS_ORACLE_MODE=local"
REM   SET "NMOS_ENABLE_SPECULATIVE=0"
REM   SET "NMOS_LOCAL_GPU_LAYERS=10"
REM
REM Remote mandatory-72B mode:
REM   SET "NMOS_ORACLE_MODE=remote"
REM   SET "NMOS_REMOTE_URL=http://127.0.0.1:8000"
REM   SET "NMOS_REMOTE_MODEL=Qwen2.5-72B-Instruct"
REM   SET "NMOS_REMOTE_API_KEY=your_key_if_needed"
REM -------------------------------------------------------------------
IF NOT DEFINED NMOS_ORACLE_MODE SET "NMOS_ORACLE_MODE=local"
IF NOT DEFINED NMOS_ENABLE_SPECULATIVE SET "NMOS_ENABLE_SPECULATIVE=0"
IF NOT DEFINED NMOS_LOCAL_GPU_LAYERS SET "NMOS_LOCAL_GPU_LAYERS=10"

echo [NMOS] System Booting...
echo [NMOS] Environment: CUDA v13.2 | Device: RTX 2050
echo [NMOS] Oracle Mode: %NMOS_ORACLE_MODE%
echo.

"%PYTHON_EXE%" NMOS_SHELL.py

pause
