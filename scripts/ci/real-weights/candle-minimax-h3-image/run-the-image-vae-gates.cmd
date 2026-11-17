rem SW-7089 - MiniMax-H3-Image 2-D VAE decoder on real weights (candle/CUDA). Materializes only
rem vae_diffusers/ of the pinned snapshot under CANDLE_GEN_MODELS_ROOT, then runs both gates against
rem the reference the macOS job produced with the hash-pinned producer (downloaded artifact).
rem Each gate is built with the registered cuda,model-inference-proof features and run with
rem MODEL_INFERENCE_CASE_FQN set; verify_model_inference_log.py then applies the proof runner's own
rem check - one passing summary, one BODY marker and the candle-cuda device set.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MINIMAX_H3_IMAGE_VAE_REFERENCE=%RUNNER_TEMP%\minimax-h3-image-vae-reference\reference.safetensors"
if "%CANDLE_GEN_MODELS_ROOT%"=="" (
  echo Missing repository variable CANDLE_GEN_MODELS_ROOT for the Windows CUDA runner.
  exit /b 1
)
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
if not exist "%MINIMAX_H3_IMAGE_VAE_REFERENCE%" (
  echo The macOS job's image VAE reference artifact is missing: %MINIMAX_H3_IMAGE_VAE_REFERENCE%
  exit /b 1
)
where nvidia-smi >nul 2>&1 && nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
call "%VCVARS%"
call :run_one image_vae_decode_matches_the_official_diffusers_vae || exit /b 1
call :run_one image_vae_tiled_decode_lowers_the_16mp_peak || exit /b 1
exit /b 0

:run_one
set "log=%RUNNER_TEMP%\minimax-h3-image-%~1.log"
set "MODEL_INFERENCE_CASE_FQN=real_weights::%~1"
cargo test --locked -p candle-gen-minimax-h3 --features cuda,model-inference-proof --release --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1
set "H3I_STATUS=%ERRORLEVEL%"
set "MODEL_INFERENCE_CASE_FQN="
type "%log%"
if not "%H3I_STATUS%"=="0" goto :run_one_failed
"%REVIEWED_PYTHON%" -I scripts/ci/verify_model_inference_log.py --package candle-gen-minimax-h3 --backend candle-cuda --fqn real_weights::%~1 --log "%log%" || goto :run_one_failed
exit /b 0

:run_one_failed
echo ::error::%~1 failed or lacks its registered proof evidence - exit %H3I_STATUS%
exit /b 1
