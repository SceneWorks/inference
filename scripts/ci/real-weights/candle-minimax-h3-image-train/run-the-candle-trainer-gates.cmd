rem SW-7098 - the MiniMax-H3-Image LoRA trainer on candle/CUDA. Materializes the minimax-h3-image row
rem and the minimax-h3-image-dit row into the image snapshot and MiniMax-H3's text encoder and vision
rem tower into the H3 snapshot, converts the q4 DiT tier with the candle converter under the runner's
rem models root, then runs the three trainer gates: gradient correctness - the streamed backward
rem against the resident one and a finite difference - t2i train/reproduce/resume at 2048, and edit
rem train/reproduce. The t2i adapter lands in MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA_OUT for the MLX
rem load gate. The converted tier is removed afterwards.
rem
rem The card is the one with the MOST free memory, in PCI bus order, selected with
rem CUDA_VISIBLE_DEVICES and recorded in cuda-device-selection.txt.
rem
rem Every gate is built with the registered cuda,model-inference-proof features and run with
rem MODEL_INFERENCE_CASE_FQN set; verify_model_inference_log.py then applies the proof runner's own
rem check. Every gate runs and reports even when an earlier one fails; the step fails at the end if
rem any did.
rem
rem No command here sits inside a parenthesized block and no echoed text carries a parenthesis.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "H3T_LOGS=%RUNNER_TEMP%\minimax-h3-image-train-candle-logs"
set "MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA_OUT=%RUNNER_TEMP%\minimax-h3-image-candle-trained-lora"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
set "MINIMAX_H3_IMAGE_TIERS_ROOT=%CANDLE_GEN_MODELS_ROOT%\sceneworks-minimax-h3-image-train-tiers"
if not exist "%H3T_LOGS%" mkdir "%H3T_LOGS%"
if not exist "%MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA_OUT%" mkdir "%MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA_OUT%"
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-vision-tower --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "H3T_CARD="
set "H3T_FREE=0"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do call :consider_card %%A %%B
if not defined H3T_CARD goto :no_card
set "CUDA_VISIBLE_DEVICES=%H3T_CARD%"
> "%H3T_LOGS%\cuda-device-selection.txt" echo CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%H3T_CARD% free_mib=%H3T_FREE%
type "%H3T_LOGS%\cuda-device-selection.txt"

set "RUSTC_WRAPPER="
call "%VCVARS%"
set "H3T_FAILED="
if exist "%MINIMAX_H3_IMAGE_TIERS_ROOT%" rmdir /s /q "%MINIMAX_H3_IMAGE_TIERS_ROOT%"
mkdir "%MINIMAX_H3_IMAGE_TIERS_ROOT%" || exit /b 1
set "MINIMAX_H3_IMAGE_SRC=%MINIMAX_H3_IMAGE_SNAPSHOT%"
set "MINIMAX_H3_IMAGE_OUT=%MINIMAX_H3_IMAGE_TIERS_ROOT%\q4"
set "MINIMAX_H3_TIER=q4"
cargo run --locked --release -p candle-gen-minimax-h3 --example minimax_h3_image_prequant > "%H3T_LOGS%\convert-q4.log" 2>&1
set "H3T_STATUS=%ERRORLEVEL%"
type "%H3T_LOGS%\convert-q4.log"
if not "%H3T_STATUS%"=="0" goto :convert_failed
call :run_gate --lib training::real_tests::streamed_backward_matches_resident_and_finite_differences_q4_cuda
call :run_gate "--test integration" h3_image_train_real::h3_image_train_t2i_saves_reproduces_and_resumes_q4_cuda
call :run_gate "--test integration" h3_image_train_real::h3_image_train_edit_saves_and_reproduces_q4_cuda
if exist "%MINIMAX_H3_IMAGE_TIERS_ROOT%" rmdir /s /q "%MINIMAX_H3_IMAGE_TIERS_ROOT%"
if defined H3T_FAILED goto :failed
exit /b 0

:failed
echo ::error::failed gates:%H3T_FAILED%
exit /b 1

:convert_failed
echo ::error::the q4 tier conversion failed - exit %H3T_STATUS%
if exist "%MINIMAX_H3_IMAGE_TIERS_ROOT%" rmdir /s /q "%MINIMAX_H3_IMAGE_TIERS_ROOT%"
exit /b 1

:no_card
echo ::error::no CUDA card found
exit /b 1

rem %1 card index, %2 its free MiB: keep the card with the most free memory
:consider_card
if %~2 GTR %H3T_FREE% set "H3T_CARD=%~1"
if %~2 GTR %H3T_FREE% set "H3T_FREE=%~2"
exit /b 0

rem %1 the cargo test target selector, %2 the test path inside it
:run_gate
set "H3T_NAME=%~2"
set "H3T_NAME=%H3T_NAME:*::=%"
set "H3T_NAME=%H3T_NAME:*::=%"
set "log=%H3T_LOGS%\%H3T_NAME%.log"
set "MODEL_INFERENCE_CASE_FQN=%~2"
cargo test --locked -p candle-gen-minimax-h3 --features cuda,model-inference-proof --release %~1 %~2 -- --ignored --exact --nocapture > "%log%" 2>&1
set "H3T_STATUS=%ERRORLEVEL%"
set "MODEL_INFERENCE_CASE_FQN="
type "%log%"
if not "%H3T_STATUS%"=="0" goto :fail_gate
"%REVIEWED_PYTHON%" -I scripts/ci/verify_model_inference_log.py --package candle-gen-minimax-h3 --backend candle-cuda --fqn %~2 --log "%log%" || goto :fail_gate
exit /b 0

:fail_gate
echo ::error::%H3T_NAME% failed or lacks its registered proof evidence - exit %H3T_STATUS%
set "H3T_FAILED=%H3T_FAILED% %H3T_NAME%"
exit /b 0
