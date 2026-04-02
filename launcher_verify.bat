@echo off
SET "PYTHON_EXE=C:\Users\PANKAJ\AppData\Local\Programs\Python\Python310\python.exe"
SET "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

REM Add CUDA bins to PATH
SET "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

echo [NMOS] Launcher Active.
echo [NMOS] Verifying CUDA with Python 3.10...

"%PYTHON_EXE%" Extra/verify_cuda.py

pause
