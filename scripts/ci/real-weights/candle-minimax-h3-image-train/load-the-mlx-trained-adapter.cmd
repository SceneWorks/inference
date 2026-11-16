set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
set "MINIMAX_H3_IMAGE_TRAINED_LORA=%RUNNER_TEMP%\minimax-h3-image-trained-lora\h3_image_lora.safetensors"
if not exist "%MINIMAX_H3_IMAGE_TRAINED_LORA%" exit /b 1
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
set "RUSTC_WRAPPER="
call "%VCVARS%"
set "MODEL_INFERENCE_CASE_FQN=h3_image_train_lora_load_real::h3_image_mlx_trained_lora_installs_on_the_real_dit_cuda"
cargo test --locked --release -p candle-gen-minimax-h3 --features cuda,model-inference-proof --test integration "%MODEL_INFERENCE_CASE_FQN%" -- --exact --ignored --nocapture > "%RUNNER_TEMP%\candle-load.log" 2>&1
set "STATUS=%ERRORLEVEL%"
type "%RUNNER_TEMP%\candle-load.log"
if not "%STATUS%"=="0" exit /b 1
"%REVIEWED_PYTHON%" -I scripts/ci/verify_model_inference_log.py --package candle-gen-minimax-h3 --backend candle-cuda --fqn "%MODEL_INFERENCE_CASE_FQN%" --log "%RUNNER_TEMP%\candle-load.log" || exit /b 1
