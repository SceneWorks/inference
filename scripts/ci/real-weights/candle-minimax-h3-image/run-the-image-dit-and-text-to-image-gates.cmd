rem SW-7091 - MiniMax-H3-Image DiT parity and the 2048x2048 text-to-image render on candle/CUDA.
rem Materializes the minimax-h3-image-dit row (transformer_diffusers/, tokenizer/) into the image
rem snapshot and the minimax-h3-text-encoder row (MiniMax-H3's byte-identical text_encoder/ shards
rem 1-12) into the H3 snapshot, with the hub lock the image VAE gate step installed, then runs both
rem gates. The DiT reference is the macOS job's (dit-reference.safetensors in the downloaded
rem minimax-h3-image-vae-reference artifact). The render is written beside the logs as evidence.
rem
rem The card is the one with the MOST free memory, in PCI bus order, selected with
rem CUDA_VISIBLE_DEVICES - another tenant may hold most of a card - and recorded in
rem cuda-device-selection.txt. Both gates keep the bf16 DiT resident, so before each one the
rem selected card's free memory is read again and the gate is REFUSED, loudly, when it is below
rem the DiT's shard bytes plus H3I_HEADROOM_MIB: a card that cannot hold the weights would spill
rem them to system memory instead of failing. Every gate runs and reports even when an earlier
rem one fails or is refused; the step fails at the end if any did.
rem
rem No command here sits inside a parenthesized block and no echoed text carries a parenthesis:
rem an unescaped closing parenthesis in an echo ends the enclosing block early.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MINIMAX_H3_IMAGE_DIT_REFERENCE=%RUNNER_TEMP%\minimax-h3-image-vae-reference\dit-reference.safetensors"
set "H3I_EVIDENCE=%RUNNER_TEMP%\minimax-h3-image-evidence"
set "MINIMAX_H3_IMAGE_RENDER_OUT=%H3I_EVIDENCE%\t2i-2048.png"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
rem An allowance over the DiT's weights for the CUDA context and activations - not a measured peak.
set "H3I_HEADROOM_MIB=4096"
if not exist "%MINIMAX_H3_IMAGE_DIT_REFERENCE%" goto :no_reference
if not exist "%H3I_EVIDENCE%" mkdir "%H3I_EVIDENCE%"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
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
if defined H3I_FAILED goto :failed
exit /b 0

:failed
echo ::error::failed or refused gates:%H3I_FAILED%
exit /b 1

:no_reference
echo The macOS job's DiT reference is missing from its artifact: %MINIMAX_H3_IMAGE_DIT_REFERENCE%
exit /b 1

:no_card
echo ::error::no CUDA card found
exit /b 1

rem %1 card index, %2 its free MiB: keep the card with the most free memory
:consider_card
if %~2 GTR %H3I_FREE% set "H3I_CARD=%~1"
if %~2 GTR %H3I_FREE% set "H3I_FREE=%~2"
exit /b 0

rem %1 the test; both gates need the bf16 DiT resident on the selected card
:run_one
set "log=%H3I_EVIDENCE%\%~1.log"
set "H3I_NOW_FREE="
for /f "tokens=1" %%F in ('nvidia-smi --id^=%H3I_CARD% --query-gpu^=memory.free --format^=csv^,noheader^,nounits') do set "H3I_NOW_FREE=%%F"
if not defined H3I_NOW_FREE set "H3I_NOW_FREE=0"
>> "%H3I_EVIDENCE%\cuda-device-selection.txt" echo %~1: card %H3I_CARD% free_mib=%H3I_NOW_FREE% required_mib=%H3I_REQUIRED_MIB%
if %H3I_NOW_FREE% LSS %H3I_REQUIRED_MIB% goto :refuse_one
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1
set "H3I_STATUS=%ERRORLEVEL%"
type "%log%"
if not "%H3I_STATUS%"=="0" goto :fail_one
findstr /C:"test result: ok. 1 passed" "%log%" >nul || goto :fail_one
exit /b 0

:refuse_one
echo ::error::%~1 REFUSED: card %H3I_CARD% has %H3I_NOW_FREE% MiB free, below the %H3I_REQUIRED_MIB% MiB a resident bf16 DiT needs - %H3I_DIT_MIB% MiB of weights plus %H3I_HEADROOM_MIB% MiB headroom
>> "%log%" echo REFUSED: card %H3I_CARD% free_mib=%H3I_NOW_FREE% required_mib=%H3I_REQUIRED_MIB%
set "H3I_FAILED=%H3I_FAILED% %~1"
exit /b 0

:fail_one
echo ::error::%~1 failed or did not run exactly one passing test - exit %H3I_STATUS%; a rename would make this step vacuously green
set "H3I_FAILED=%H3I_FAILED% %~1"
exit /b 0
