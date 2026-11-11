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
rem Redirection leads each echo: a value ending in a digit (GPU index 0) before ">>" is read as a handle number and drops the value.
rem Another tenant can hold a large share of one or both cards, so the job does not take cuda:0 by
rem default: it picks the card with the MOST free memory now, in PCI bus order (the order
rem nvidia-smi indexes by), and pins every later step to it through CUDA_VISIBLE_DEVICES.
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
echo CUDA_DEVICE_ORDER=PCI_BUS_ID>>"%GITHUB_ENV%"
set "SEL_GPU="
set "SEL_FREE=-1"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do call :consider_card %%A %%B
if not defined SEL_GPU (echo ::error::nvidia-smi listed no GPU with a free-memory figure & exit /b 1)
for /f "tokens=1,2 delims=," %%A in ('nvidia-smi -i %SEL_GPU% --query-gpu^=compute_cap^,name --format^=csv^,noheader') do (set "CAP=%%A" & set "SEL_NAME=%%B")
set "CAP=%CAP: =%"
>>"%GITHUB_ENV%" echo CUDA_VISIBLE_DEVICES=%SEL_GPU%
>>"%GITHUB_ENV%" echo MINIMAX_PE_GPU=%SEL_GPU%
>"%MINIMAX_PE_OUT%\selected-gpu.txt" echo selected_gpu_index=%SEL_GPU%
>>"%MINIMAX_PE_OUT%\selected-gpu.txt" echo selected_gpu_name=%SEL_NAME%
>>"%MINIMAX_PE_OUT%\selected-gpu.txt" echo selected_gpu_free_mib_at_selection=%SEL_FREE%
>>"%MINIMAX_PE_OUT%\selected-gpu.txt" echo selected_gpu_compute_capability=%CAP%
type "%MINIMAX_PE_OUT%\selected-gpu.txt"
rem Blackwell is required of the SELECTED card, not of index 0.
if not "%CAP:~0,3%"=="12." (echo ::error::selected GPU %SEL_GPU% is compute capability %CAP%, not Blackwell sm_120 & exit /b 1)
exit /b 0

rem %1 = index, %2 = free MiB: keep the card with the most free memory.
:consider_card
if %~2 GTR %SEL_FREE% set "SEL_GPU=%~1"
if %~2 GTR %SEL_FREE% set "SEL_FREE=%~2"
exit /b 0
