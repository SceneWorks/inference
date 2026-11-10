rem SW-7091 - MiniMax-H3-Image DiT parity and the 2048x2048 text-to-image render on candle/CUDA.
rem Materializes the minimax-h3-image-dit row (transformer_diffusers/, tokenizer/) into the image
rem snapshot and the minimax-h3-text-encoder row (MiniMax-H3's byte-identical text_encoder/ shards
rem 1-12) into the H3 snapshot, with the hub lock the image VAE gate step installed, then runs both
rem gates. The DiT reference is the macOS job's (dit-reference.safetensors in the downloaded
rem minimax-h3-image-vae-reference artifact). The render is written beside the logs as evidence.
rem SW-7092 - the edit gates against the macOS job's edit-reference.safetensors (same artifact),
rem and a 3-reference edit render, which also needs MiniMax-H3's vision tower (text-encoder shard 14,
rem the minimax-h3-vision-tower row).
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MINIMAX_H3_IMAGE_DIT_REFERENCE=%RUNNER_TEMP%\minimax-h3-image-vae-reference\dit-reference.safetensors"
set "MINIMAX_H3_IMAGE_RENDER_OUT=%RUNNER_TEMP%\minimax-h3-image-evidence\t2i-2048.png"
set "MINIMAX_H3_IMAGE_EDIT_REFERENCE=%RUNNER_TEMP%\minimax-h3-image-vae-reference\edit-reference.safetensors"
set "MINIMAX_H3_IMAGE_EDIT_RENDER_OUT=%RUNNER_TEMP%\minimax-h3-image-evidence\edit-1024x768.png"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
if not exist "%MINIMAX_H3_IMAGE_DIT_REFERENCE%" (
  echo The macOS job's DiT reference is missing from its artifact: %MINIMAX_H3_IMAGE_DIT_REFERENCE%
  exit /b 1
)
if not exist "%MINIMAX_H3_IMAGE_EDIT_REFERENCE%" (
  echo The macOS job's edit reference is missing from its artifact: %MINIMAX_H3_IMAGE_EDIT_REFERENCE%
  exit /b 1
)
if not exist "%RUNNER_TEMP%\minimax-h3-image-evidence" mkdir "%RUNNER_TEMP%\minimax-h3-image-evidence"
where nvidia-smi >nul 2>&1 && nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-vision-tower --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
call "%VCVARS%"
call :run_one image_dit_first_step_matches_the_official_diffusers_transformer || exit /b 1
call :run_one image_text_to_image_renders_2048_on_cuda || exit /b 1
call :run_one image_edit_presentation_matches_the_processor_template || exit /b 1
call :run_one image_edit_reference_latents_match_the_official_diffusers_vae || exit /b 1
call :run_one image_edit_first_step_matches_the_official_diffusers_transformer || exit /b 1
call :run_one image_edit_renders_three_references_on_cuda || exit /b 1
rem SW-7093 - then the q8 / q4 tiers: convert, audit and render each.
call :tiers || exit /b 1
exit /b 0

:run_one
set "log=%RUNNER_TEMP%\minimax-h3-image-evidence\%~1.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0

:tiers
rem SW-7093 - MiniMax-H3-Image q8 / q4 DiT tiers on candle/CUDA: convert from the dense snapshot with the
rem candle converter (the same MLX-affine artifact the MLX lane writes), audit it, and render 2048x2048
rem through the registered generator at each tier. Called at the end of this script, which has set
rem the snapshot variables, the hub lock, the CUDA/MSVC environment and REVIEWED_PYTHON. The converted tiers go under the runner's models root
rem (never to Hugging Face) and are removed after each tier's render. The text encoder at each tier is
rem MiniMax-H3's existing SceneWorks/minimax-h3-mlx artifact. The card is shared: a render that runs out
rem of CUDA memory fails loudly here.
set "MINIMAX_H3_TIER_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--SceneWorks--minimax-h3-mlx\snapshots\137ce668c55a20bc0935fd1cf2a3de8448abb7f4"
set "MINIMAX_H3_IMAGE_TIERS_ROOT=%CANDLE_GEN_MODELS_ROOT%\sceneworks-minimax-h3-image-tiers"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
for %%T in (q8 q4) do "%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-mlx-te-%%T --snapshot "%MINIMAX_H3_TIER_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
if exist "%MINIMAX_H3_IMAGE_TIERS_ROOT%" rmdir /s /q "%MINIMAX_H3_IMAGE_TIERS_ROOT%"
mkdir "%MINIMAX_H3_IMAGE_TIERS_ROOT%" || exit /b 1
set "MINIMAX_H3_IMAGE_RENDER_OUT=%RUNNER_TEMP%\minimax-h3-image-evidence\t2i-2048"
for %%T in (q8 q4) do (
  call :tier %%T || exit /b 1
)
exit /b 0

:tier
set "MINIMAX_H3_IMAGE_SRC=%MINIMAX_H3_IMAGE_SNAPSHOT%"
set "MINIMAX_H3_IMAGE_OUT=%MINIMAX_H3_IMAGE_TIERS_ROOT%\%~1"
set "MINIMAX_H3_TIER=%~1"
set "log=%RUNNER_TEMP%\minimax-h3-image-evidence\convert-%~1.log"
cargo run --locked --release -p candle-gen-minimax-h3 --example minimax_h3_image_prequant > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
call :run_one image_t2i_renders_2048_%~1_on_cuda || exit /b 1
rmdir /s /q "%MINIMAX_H3_IMAGE_OUT%"
exit /b 0
