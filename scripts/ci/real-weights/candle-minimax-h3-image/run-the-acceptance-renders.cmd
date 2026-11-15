rem SW-7104 - feature SW-F363's acceptance renders on candle/CUDA - tests/h3_image_acceptance_real.rs:
rem AT2 - 2048x2048, 4096x4096 and 6272x2688 at the admitted strategy, one fresh process each; AT3 -
rem a 3-reference edit that must draw on each reference; AT6 - the refusals; AT8 - 16 MP with two
rem community LoRAs through bounded transformer residency and bounded decode; AT11 - every
rem community LoRA applies. Installs the pinned huggingface-hub and materializes - a verified
rem snapshot is a no-op, so after the earlier gate step it reads nothing - the image DiT, VAE and
rem tokenizer, MiniMax-H3's text encoder and vision tower, and the community LoRAs - the
rem minimax-h3-image-community-lora-* rows - under CANDLE_GEN_MODELS_ROOT, so the
rem minimax-h3-image-acceptance profile can run this step without the earlier gates.
rem
rem The card is the one with the MOST free memory, in PCI bus order, selected with
rem CUDA_VISIBLE_DEVICES and recorded in acceptance-device-selection.txt, as the gate step does.
rem Each gate is built with the registered cuda,model-inference-proof features, run with
rem MODEL_INFERENCE_CASE_FQN set and checked by verify_model_inference_log.py. A render admitted
rem on bounded transformer residency reads the streamed text encoder's token table on the host, so
rem its proof configuration is admitted_streamed_cuda - candle-cpu and candle-cuda - and every other
rem rung's is admitted_resident_cuda; the run record line in the log names the admitted rung. Every
rem gate runs and reports even when an earlier one fails; the step fails at the end if any did. The
rem PNGs and logs go to minimax-h3-image-acceptance, uploaded as minimax-h3-image-acceptance-cuda.
rem
rem No command here sits inside a parenthesized block and no echoed text carries a parenthesis.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "H3I_OUT=%RUNNER_TEMP%\minimax-h3-image-acceptance"
set "MINIMAX_H3_IMAGE_ACCEPTANCE_OUT=%H3I_OUT%"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
set "MINIMAX_H3_IMAGE_LORA_TANAAMI_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--syntaxdiffusion--tanaami-style-h3-image-lora\snapshots\3804b381526a3e537c7961942ba4861a831b2d39"
set "MINIMAX_H3_IMAGE_LORA_DOMOKUN_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--SimpleTuner--minimax-h3-image-domokun-2048\snapshots\57214626f457eac87077c7035bb1b64d5264ebf3"
set "MINIMAX_H3_IMAGE_LORA_FIZGIG_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--ShootTheSound--Fizgig-MiniMax-H3-Image-Training-Adapter\snapshots\f1a4547596faeb98bae568dd28da296c49fbe0a1"
set "MINIMAX_H3_IMAGE_LORA_PDMD_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--linoyts--MiniMax-H3-Image-pDMD-2steps\snapshots\c07d96e5635311ed007d44654c2f98cc8d6e0a1e"
set "MINIMAX_H3_IMAGE_LORA_CHARACTER_SWAP_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--akhaliq--MiniMax-H3-Image-Character-Swap\snapshots\0b7f918092f9e23c01ca62b63d7c17e4fce6faa9"
if not exist "%H3I_OUT%" mkdir "%H3I_OUT%"
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "HF_HUB_DISABLE_XET=1"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-vision-tower --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
call :materialize tanaami "%MINIMAX_H3_IMAGE_LORA_TANAAMI_SNAPSHOT%" || exit /b 1
call :materialize domokun "%MINIMAX_H3_IMAGE_LORA_DOMOKUN_SNAPSHOT%" || exit /b 1
call :materialize fizgig "%MINIMAX_H3_IMAGE_LORA_FIZGIG_SNAPSHOT%" || exit /b 1
call :materialize pdmd "%MINIMAX_H3_IMAGE_LORA_PDMD_SNAPSHOT%" || exit /b 1
call :materialize character-swap "%MINIMAX_H3_IMAGE_LORA_CHARACTER_SWAP_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
set "MINIMAX_H3_IMAGE_COMMUNITY_LORAS=tanaami=%MINIMAX_H3_IMAGE_LORA_TANAAMI_SNAPSHOT%\tanaami_h3_image_lora_2000.safetensors;domokun=%MINIMAX_H3_IMAGE_LORA_DOMOKUN_SNAPSHOT%\pytorch_lora_weights.safetensors;fizgig=%MINIMAX_H3_IMAGE_LORA_FIZGIG_SNAPSHOT%\fizgig_minimax_h3_image_training_adapter.safetensors;pdmd=%MINIMAX_H3_IMAGE_LORA_PDMD_SNAPSHOT%\pytorch_lora_weights.safetensors;character-swap=%MINIMAX_H3_IMAGE_LORA_CHARACTER_SWAP_SNAPSHOT%\lora\h3_image_character_swap_diffusers.safetensors"
set "MINIMAX_H3_IMAGE_AT8_LORAS=tanaami,domokun"

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > "%H3I_OUT%\cuda-device-inventory.csv" || goto :no_card
type "%H3I_OUT%\cuda-device-inventory.csv"
set "H3I_CARD="
set "H3I_FREE=0"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do call :consider_card %%A %%B
if not defined H3I_CARD goto :no_card
for /f "tokens=* delims=" %%N in ('nvidia-smi --id^=%H3I_CARD% --query-gpu^=name --format^=csv^,noheader') do set "H3I_CARD_NAME=%%N"
set "CUDA_VISIBLE_DEVICES=%H3I_CARD%"
> "%H3I_OUT%\acceptance-device-selection.txt" echo CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%H3I_CARD% name=%H3I_CARD_NAME% free_mib=%H3I_FREE%
type "%H3I_OUT%\acceptance-device-selection.txt"

call "%VCVARS%"
set "H3I_FAILED="
call :run_gate h3_image_t2i_2048_at_the_admitted_strategy_cuda admitted
call :run_gate h3_image_t2i_4096_at_the_admitted_strategy_cuda admitted
call :run_gate h3_image_t2i_6272x2688_at_the_admitted_strategy_cuda admitted
call :run_gate h3_image_edit_draws_on_each_of_three_references_cuda admitted
call :run_gate h3_image_refuses_the_unsupported_requests_cuda unlearned
call :run_gate h3_image_16mp_with_two_loras_streams_and_tiles_cuda fixed
call :run_gate h3_image_community_loras_apply_cuda admitted
findstr /B /C:"RUN RECORD" /C:"AT2" /C:"AT3" /C:"AT6" /C:"AT8" /C:"AT11" "%H3I_OUT%\*.log" > "%H3I_OUT%\run-records.txt"
type "%H3I_OUT%\run-records.txt"
if defined H3I_FAILED goto :failed
exit /b 0

:failed
echo ::error::failed acceptance gates:%H3I_FAILED%
exit /b 1

:no_card
echo ::error::no CUDA card found
exit /b 1

rem %1 the row suffix, %2 its snapshot
:materialize
for /l %%T in (1,1,3) do "%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-community-lora-%~1 --snapshot "%~2" && exit /b 0
echo ::error::minimax-h3-image-community-lora-%~1 could not be materialized
exit /b 1

rem %1 card index, %2 its free MiB: keep the card with the most free memory
:consider_card
if %~2 GTR %H3I_FREE% set "H3I_CARD=%~1"
if %~2 GTR %H3I_FREE% set "H3I_FREE=%~2"
exit /b 0

rem %1 the h3_image_acceptance_real test; %2 admitted, fixed or unlearned - how its proof is checked
:run_gate
set "H3I_NAME=%~1"
set "log=%H3I_OUT%\%H3I_NAME%.log"
set "H3I_FQN=h3_image_acceptance_real::%H3I_NAME%"
set "MODEL_INFERENCE_CASE_FQN=%H3I_FQN%"
cargo test --locked -p candle-gen-minimax-h3 --features cuda,model-inference-proof --release --test integration %H3I_FQN% -- --ignored --exact --nocapture > "%log%" 2>&1
set "H3I_STATUS=%ERRORLEVEL%"
set "MODEL_INFERENCE_CASE_FQN="
type "%log%"
if not "%H3I_STATUS%"=="0" goto :fail_gate
set "H3I_PROOF=--backend candle-cuda"
if "%~2"=="unlearned" set "H3I_PROOF=--backend candle-cuda --no-learned-work"
if "%~2"=="admitted" set "H3I_PROOF=--configuration admitted_resident_cuda"
if "%~2"=="admitted" findstr /C:"admitted BoundedTransformerResidency" "%log%" >nul && set "H3I_PROOF=--configuration admitted_streamed_cuda"
echo proof check: %H3I_PROOF%
"%REVIEWED_PYTHON%" -I scripts/ci/verify_model_inference_log.py --package candle-gen-minimax-h3 %H3I_PROOF% --fqn %H3I_FQN% --log "%log%" || goto :fail_gate
exit /b 0

:fail_gate
echo ::error::%H3I_NAME% failed or lacks its registered proof evidence - exit %H3I_STATUS%; a rename would make this step vacuously green
set "H3I_FAILED=%H3I_FAILED% %H3I_NAME%"
exit /b 0
