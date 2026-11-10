rem SW-7091 - MiniMax-H3-Image DiT parity and the 2048x2048 text-to-image render on candle/CUDA.
rem Materializes the minimax-h3-image-dit row (transformer_diffusers/, tokenizer/) into the image
rem snapshot and the minimax-h3-text-encoder row (MiniMax-H3's byte-identical text_encoder/ shards
rem 1-12) into the H3 snapshot, with the hub lock the image VAE gate step installed, then runs both
rem gates. The DiT reference is the macOS job's (dit-reference.safetensors in the downloaded
rem minimax-h3-image-vae-reference artifact). The render is written beside the logs as evidence.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MINIMAX_H3_IMAGE_DIT_REFERENCE=%RUNNER_TEMP%\minimax-h3-image-vae-reference\dit-reference.safetensors"
set "MINIMAX_H3_IMAGE_RENDER_OUT=%RUNNER_TEMP%\minimax-h3-image-evidence\t2i-2048.png"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
if not exist "%MINIMAX_H3_IMAGE_DIT_REFERENCE%" (
  echo The macOS job's DiT reference is missing from its artifact: %MINIMAX_H3_IMAGE_DIT_REFERENCE%
  exit /b 1
)
if not exist "%RUNNER_TEMP%\minimax-h3-image-evidence" mkdir "%RUNNER_TEMP%\minimax-h3-image-evidence"
where nvidia-smi >nul 2>&1 && nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
call "%VCVARS%"
call :run_one image_dit_first_step_matches_the_official_diffusers_transformer || exit /b 1
call :run_one image_text_to_image_renders_2048_on_cuda || exit /b 1
rem SW-7096: the three LoRA layouts fold identically on the real image DiT; ten stack with their scales.
call :run_lora h3_image_lora_layouts_fold_identically_on_the_real_dit_cuda || exit /b 1
call :run_lora h3_image_adaln_and_output_head_loras_reach_the_denoiser_cuda || exit /b 1
exit /b 0

:run_one
set "log=%RUNNER_TEMP%\minimax-h3-image-evidence\%~1.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0

:run_lora
set "log=%RUNNER_TEMP%\minimax-h3-image-evidence\%~1.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration h3_image_lora_real::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
