@echo off
SET "PYTHON_EXE=C:\Users\PANKAJ\AppData\Local\Programs\Python\Python310\python.exe"
SET "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
SET "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
SET "CMAKE_ARGS=-DGGML_CUDA=on"

echo [NMOS] Configuring CUDA v13.2 Environment for Python 3.10...
echo [NMOS] Installing llama-cpp-python...

"%PYTHON_EXE%" -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

echo.
echo [NMOS] Done. Please run the verification using Python 3.10.
pause
