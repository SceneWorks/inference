where nvidia-smi >nul 2>&1 || (echo nvidia-smi not on PATH; exit /b 0)
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > "%QWEN_IMAGE_2_1_RENDER_OUT%\cuda-device-inventory.csv"
type "%QWEN_IMAGE_2_1_RENDER_OUT%\cuda-device-inventory.csv"
