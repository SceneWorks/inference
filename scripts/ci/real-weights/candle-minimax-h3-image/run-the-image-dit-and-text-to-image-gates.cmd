rem SW-7091 - MiniMax-H3-Image DiT parity and the 2048x2048 text-to-image render on candle/CUDA.
rem Materializes the minimax-h3-image-dit row (transformer_diffusers/, tokenizer/) into the image
rem snapshot and the minimax-h3-text-encoder row (MiniMax-H3's byte-identical text_encoder/ shards
rem 1-12) into the H3 snapshot, with the hub lock the image VAE gate step installed, then runs both
rem gates. The DiT reference is the macOS job's (dit-reference.safetensors in the downloaded
rem minimax-h3-image-vae-reference artifact). The render is written beside the logs as evidence.
rem SW-7092 - the edit gates against the macOS job's edit-reference.safetensors (same artifact),
rem and a 3-reference edit render, which also needs MiniMax-H3's vision tower (text-encoder shard 14,
rem the minimax-h3-vision-tower row). SW-7095 adds a 16 MP streamed render, SW-7093 the q8 / q4
rem tiers and SW-7096 the LoRA gates.
rem
rem The card is the one with the MOST free memory, in PCI bus order, selected with
rem CUDA_VISIBLE_DEVICES - another tenant may hold most of a card - and recorded in
rem cuda-device-selection.txt. The gates that keep the bf16 DiT resident read the selected card's
rem free memory again first and are REFUSED, loudly, when it is below the DiT's shard bytes plus
rem H3I_HEADROOM_MIB: a card that cannot hold the weights would spill them to system memory instead
rem of failing. The VAE-only, streamed and quantized-tier gates need less than that and run
rem unchecked. Every gate runs and reports even when an earlier one fails or is refused; the step
rem fails at the end if any did.
rem
rem No command here sits inside a parenthesized block and no echoed text carries a parenthesis:
rem an unescaped closing parenthesis in an echo ends the enclosing block early.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MINIMAX_H3_IMAGE_DIT_REFERENCE=%RUNNER_TEMP%\minimax-h3-image-vae-reference\dit-reference.safetensors"
set "MINIMAX_H3_IMAGE_EDIT_REFERENCE=%RUNNER_TEMP%\minimax-h3-image-vae-reference\edit-reference.safetensors"
set "H3I_EVIDENCE=%RUNNER_TEMP%\minimax-h3-image-evidence"
set "MINIMAX_H3_IMAGE_RENDER_OUT=%H3I_EVIDENCE%\t2i-2048.png"
set "MINIMAX_H3_IMAGE_EDIT_RENDER_OUT=%H3I_EVIDENCE%\edit-1024x768.png"
set "MINIMAX_H3_IMAGE_STREAMED_RENDER_OUT=%H3I_EVIDENCE%\t2i-4096-streamed.png"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
rem An allowance over the DiT's weights for the CUDA context and activations - not a measured peak.
set "H3I_HEADROOM_MIB=4096"
if not exist "%MINIMAX_H3_IMAGE_DIT_REFERENCE%" goto :no_reference
if not exist "%MINIMAX_H3_IMAGE_EDIT_REFERENCE%" goto :no_edit_reference
if not exist "%H3I_EVIDENCE%" mkdir "%H3I_EVIDENCE%"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-vision-tower --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
"%REVIEWED_PYTHON%" -I -c "import pathlib, sys; print(-(-sum(p.stat().st_size for p in pathlib.Path(sys.argv[1]).glob('*.safetensors')) // 1048576))" "%MINIMAX_H3_IMAGE_SNAPSHOT%\transformer_diffusers" > "%H3I_EVIDENCE%\dit-weight-mib.txt" || exit /b 1
set /p H3I_DIT_MIB=<"%H3I_EVIDENCE%\dit-weight-mib.txt"
set /a H3I_REQUIRED_MIB=H3I_DIT_MIB + H3I_HEADROOM_MIB

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > "%H3I_EVIDENCE%\cuda-device-inventory.csv" || goto :no_card
type "%H3I_EVIDENCE%\cuda-device-inventory.csv"
set "H3I_CARD="
set "H3I_FREE=0"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do call :consider_card %%A %%B
if not defined H3I_CARD goto :no_card
for /f "tokens=* delims=" %%N in ('nvidia-smi --id^=%H3I_CARD% --query-gpu^=name --format^=csv^,noheader') do set "H3I_CARD_NAME=%%N"
set "CUDA_VISIBLE_DEVICES=%H3I_CARD%"
> "%H3I_EVIDENCE%\cuda-device-selection.txt" echo CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%H3I_CARD% name=%H3I_CARD_NAME% free_mib=%H3I_FREE% dit_weight_mib=%H3I_DIT_MIB% required_mib=%H3I_REQUIRED_MIB%
type "%H3I_EVIDENCE%\cuda-device-selection.txt"

call "%VCVARS%"
set "H3I_FAILED="
call :run_one image_dit_first_step_matches_the_official_diffusers_transformer
call :run_one image_text_to_image_renders_2048_on_cuda
call :run_unchecked integration real_weights::image_edit_presentation_matches_the_processor_template
call :run_unchecked integration real_weights::image_edit_reference_latents_match_the_official_diffusers_vae
call :run_one image_edit_first_step_matches_the_official_diffusers_transformer
call :run_one image_edit_renders_three_references_on_cuda
rem SW-7095 - a 16 MP bf16 render admitted against the card's free memory and run through the
rem streamed route: the one-block window over the text encoder and the DiT.
call :run_unchecked integration real_weights::image_t2i_renders_16mp_streamed_on_cuda
rem SW-7093 - then the q8 / q4 tiers: convert, audit and render each.
call :tiers
rem SW-7096: the three LoRA layouts fold identically on the real image DiT; ten stack with their scales;
rem blocks streamed through the one-block window carry the LoRA like resident ones.
call :run_lora h3_image_lora_layouts_fold_identically_on_the_real_dit_cuda
call :run_lora h3_image_adaln_and_output_head_loras_reach_the_denoiser_cuda
call :run_lora h3_image_streamed_blocks_carry_the_lora_like_resident_ones_cuda
if defined H3I_FAILED goto :failed
exit /b 0

:failed
echo ::error::failed or refused gates:%H3I_FAILED%
exit /b 1

:no_reference
echo The macOS job's DiT reference is missing from its artifact: %MINIMAX_H3_IMAGE_DIT_REFERENCE%
exit /b 1

:no_edit_reference
echo The macOS job's edit reference is missing from its artifact: %MINIMAX_H3_IMAGE_EDIT_REFERENCE%
exit /b 1

:no_card
echo ::error::no CUDA card found
exit /b 1

rem %1 card index, %2 its free MiB: keep the card with the most free memory
:consider_card
if %~2 GTR %H3I_FREE% set "H3I_CARD=%~1"
if %~2 GTR %H3I_FREE% set "H3I_FREE=%~2"
exit /b 0

rem %1 a real_weights test that needs the bf16 DiT resident on the selected card
:run_one
set "log=%H3I_EVIDENCE%\%~1.log"
set "H3I_NOW_FREE="
for /f "tokens=1" %%F in ('nvidia-smi --id^=%H3I_CARD% --query-gpu^=memory.free --format^=csv^,noheader^,nounits') do set "H3I_NOW_FREE=%%F"
if not defined H3I_NOW_FREE set "H3I_NOW_FREE=0"
>> "%H3I_EVIDENCE%\cuda-device-selection.txt" echo %~1: card %H3I_CARD% free_mib=%H3I_NOW_FREE% required_mib=%H3I_REQUIRED_MIB%
if %H3I_NOW_FREE% LSS %H3I_REQUIRED_MIB% goto :refuse_one
call :run_unchecked integration real_weights::%~1
exit /b 0

:refuse_one
echo ::error::%~1 REFUSED: card %H3I_CARD% has %H3I_NOW_FREE% MiB free, below the %H3I_REQUIRED_MIB% MiB a resident bf16 DiT needs - %H3I_DIT_MIB% MiB of weights plus %H3I_HEADROOM_MIB% MiB headroom
>> "%log%" echo REFUSED: card %H3I_CARD% free_mib=%H3I_NOW_FREE% required_mib=%H3I_REQUIRED_MIB%
set "H3I_FAILED=%H3I_FAILED% %~1"
exit /b 0

rem %1 the test binary, %2 the test path; runs it without the resident-DiT memory check
:run_unchecked
set "H3I_NAME=%~2"
set "H3I_NAME=%H3I_NAME:*::=%"
set "log=%H3I_EVIDENCE%\%H3I_NAME%.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test %~1 %~2 -- --ignored --exact --nocapture > "%log%" 2>&1
set "H3I_STATUS=%ERRORLEVEL%"
type "%log%"
if not "%H3I_STATUS%"=="0" goto :fail_one
findstr /C:"test result: ok. 1 passed" "%log%" >nul || goto :fail_one
exit /b 0

:fail_one
echo ::error::%H3I_NAME% failed or did not run exactly one passing test - exit %H3I_STATUS%; a rename would make this step vacuously green
set "H3I_FAILED=%H3I_FAILED% %H3I_NAME%"
exit /b 0

rem %1 the h3_image_lora_real test
:run_lora
call :run_unchecked integration h3_image_lora_real::%~1
exit /b 0

:tiers
rem SW-7093 - MiniMax-H3-Image q8 / q4 DiT tiers on candle/CUDA: convert from the dense snapshot with the
rem candle converter - the same MLX-affine artifact the MLX lane writes - audit it, and render 2048x2048
rem through the registered generator at each tier. The converted tiers go under the runner's models
rem root, never to Hugging Face, and are removed after each tier's render. The text encoder at each
rem tier is MiniMax-H3's existing SceneWorks/minimax-h3-mlx artifact.
set "MINIMAX_H3_TIER_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--SceneWorks--minimax-h3-mlx\snapshots\137ce668c55a20bc0935fd1cf2a3de8448abb7f4"
set "MINIMAX_H3_IMAGE_TIERS_ROOT=%CANDLE_GEN_MODELS_ROOT%\sceneworks-minimax-h3-image-tiers"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-mlx-te-q8 --snapshot "%MINIMAX_H3_TIER_SNAPSHOT%" || goto :tiers_unavailable
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-mlx-te-q4 --snapshot "%MINIMAX_H3_TIER_SNAPSHOT%" || goto :tiers_unavailable
set "PYTHONPATH="
if exist "%MINIMAX_H3_IMAGE_TIERS_ROOT%" rmdir /s /q "%MINIMAX_H3_IMAGE_TIERS_ROOT%"
mkdir "%MINIMAX_H3_IMAGE_TIERS_ROOT%" || goto :tiers_unavailable
set "MINIMAX_H3_IMAGE_RENDER_OUT=%H3I_EVIDENCE%\t2i-2048"
call :tier q8
call :tier q4
exit /b 0

:tiers_unavailable
set "PYTHONPATH="
echo ::error::the q8 / q4 tier inputs could not be prepared
set "H3I_FAILED=%H3I_FAILED% tiers"
exit /b 0

rem %1 the tier: q8 or q4
:tier
set "MINIMAX_H3_IMAGE_SRC=%MINIMAX_H3_IMAGE_SNAPSHOT%"
set "MINIMAX_H3_IMAGE_OUT=%MINIMAX_H3_IMAGE_TIERS_ROOT%\%~1"
set "MINIMAX_H3_TIER=%~1"
set "log=%H3I_EVIDENCE%\convert-%~1.log"
cargo run --locked --release -p candle-gen-minimax-h3 --example minimax_h3_image_prequant > "%log%" 2>&1
set "H3I_STATUS=%ERRORLEVEL%"
type "%log%"
if not "%H3I_STATUS%"=="0" goto :tier_convert_failed
call :run_unchecked integration real_weights::image_t2i_renders_2048_%~1_on_cuda
if exist "%MINIMAX_H3_IMAGE_OUT%" rmdir /s /q "%MINIMAX_H3_IMAGE_OUT%"
exit /b 0

:tier_convert_failed
echo ::error::the %~1 tier conversion failed - exit %H3I_STATUS%
set "H3I_FAILED=%H3I_FAILED% convert-%~1"
if exist "%MINIMAX_H3_IMAGE_OUT%" rmdir /s /q "%MINIMAX_H3_IMAGE_OUT%"
exit /b 0
