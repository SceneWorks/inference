rem SW-7089 - MiniMax-H3-Image 2-D VAE decoder on real weights (candle/CUDA), profile minimax-h3-image.
rem CANDLE_MINIMAX_H3_IMAGE_VAE_REFERENCE is the same operator-provisioned file the MLX lane reads
rem (crates/media/mlx-gen/tools/dump_minimax_h3_image_vae_real.py); copy it onto this runner.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%MINIMAX_H3_IMAGE_SNAPSHOT%"=="" (
  echo Missing repository variable CANDLE_MINIMAX_H3_IMAGE_SNAPSHOT for the Windows CUDA runner.
  exit /b 1
)
where nvidia-smi >nul 2>&1 && nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
if "%MINIMAX_H3_IMAGE_VAE_REFERENCE%"=="" (
  echo Repository variable CANDLE_MINIMAX_H3_IMAGE_VAE_REFERENCE is unset; the parity gate refuses to report green without it.
  exit /b 1
)
if not exist "%MINIMAX_H3_IMAGE_VAE_REFERENCE%" (
  echo CANDLE_MINIMAX_H3_IMAGE_VAE_REFERENCE names no file on this runner: %MINIMAX_H3_IMAGE_VAE_REFERENCE%
  exit /b 1
)
call "%VCVARS%"
call :run_one image_vae_decode_matches_the_official_diffusers_vae || exit /b 1
call :run_one image_vae_tiled_decode_lowers_the_16mp_peak || exit /b 1
exit /b 0

:run_one
set "log=%RUNNER_TEMP%\minimax-h3-image-%~1.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
