where nvidia-smi >nul 2>&1 || (echo nvidia-smi not on PATH; exit /b 0)
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > "%IRIS_OUT%\cuda-device-inventory.csv"
type "%IRIS_OUT%\cuda-device-inventory.csv"
