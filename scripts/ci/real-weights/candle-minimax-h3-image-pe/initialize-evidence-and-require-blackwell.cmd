if "%CANDLE_GEN_MODELS_ROOT%"=="" (
  echo Missing repository variable CANDLE_GEN_MODELS_ROOT for the Windows CUDA runner.
  exit /b 1
)
if exist "%RUNNER_TEMP%\minimax-pe-cuda-evidence" rmdir /s /q "%RUNNER_TEMP%\minimax-pe-cuda-evidence"
mkdir "%RUNNER_TEMP%\minimax-pe-cuda-evidence" || exit /b 1
echo MINIMAX_PE_OUT=%RUNNER_TEMP%\minimax-pe-cuda-evidence>>"%GITHUB_ENV%"
rem The converted tiers of the pinned revision; owned by this lane and rewritten by every run.
echo MINIMAX_PE_TIERS=%MINIMAX_PE_TIERS_ROOT%\35603d4b63949afa12cb098227fffefedf081afa>>"%GITHUB_ENV%"
rem NVFP4 runs natively only on compute capability 12.x (sm_120). Record the inventory and refuse
rem to produce a "native NVFP4" reference on anything else.
where nvidia-smi >nul 2>&1 || (echo ::error::nvidia-smi is not on PATH & exit /b 1)
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used --format=csv > "%MINIMAX_PE_OUT%\cuda-device-inventory.csv" || exit /b 1
type "%MINIMAX_PE_OUT%\cuda-device-inventory.csv"
for /f "skip=1 tokens=3 delims=," %%C in ('nvidia-smi --query-gpu=index^,name^,compute_cap --format=csv -i 0') do set "CAP=%%C"
set "CAP=%CAP: =%"
if not "%CAP:~0,3%"=="12." (echo ::error::cuda:0 is compute capability %CAP%, not Blackwell sm_120 & exit /b 1)
echo cuda:0 compute capability %CAP%
