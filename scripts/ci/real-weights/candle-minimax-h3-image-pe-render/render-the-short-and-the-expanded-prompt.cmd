rem SW-7104 - AT9 on candle/CUDA: render a short t2i prompt and the PE's expansion of it at one seed -
rem h3_image_acceptance_real::h3_image_pe_expanded_prompt_renders_more_detail_cuda. The expansion is
rem case t2i-0 of the candle-minimax-h3-image-pe job's native NVFP4 run - nvfp4.json, downloaded
rem beside this job; its short prompt is core-llm's minimax_h3_image_pe::quality::T2I_PROMPTS[0].
rem Materializes the image DiT, VAE, tokenizer and MiniMax-H3's text encoder shards 1-12 under
rem CANDLE_GEN_MODELS_ROOT as the minimax-h3-image job does. The card is the one with the most free
rem memory, in PCI bus order. The gate is built with the registered cuda,model-inference-proof
rem features, run with MODEL_INFERENCE_CASE_FQN set and checked by verify_model_inference_log.py at
rem the configuration of the rung admission selected - admitted_streamed_cuda when the run record
rem names bounded transformer residency, else admitted_resident_cuda.
rem
rem No command here sits inside a parenthesized block and no echoed text carries a parenthesis.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "H3I_OUT=%RUNNER_TEMP%\minimax-h3-image-acceptance"
set "MINIMAX_H3_IMAGE_ACCEPTANCE_OUT=%H3I_OUT%"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
set "H3I_EVIDENCE=%RUNNER_TEMP%\minimax-pe-cuda-evidence\nvfp4.json"
if not exist "%H3I_OUT%" mkdir "%H3I_OUT%"
if not exist "%H3I_EVIDENCE%" goto :no_evidence
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "HF_HUB_DISABLE_XET=1"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
set "MINIMAX_H3_IMAGE_PE_SHORT_PROMPT=a red bicycle leaning against a brick wall"
set "MINIMAX_H3_IMAGE_PE_EXPANDED_FILE=%H3I_OUT%\pe-expanded-t2i-0.txt"
set "MINIMAX_H3_IMAGE_PE_LABEL="
"%REVIEWED_PYTHON%" -I scripts/ci/h3_image_pe_expansion.py "%H3I_EVIDENCE%" t2i-0 "%MINIMAX_H3_IMAGE_PE_EXPANDED_FILE%" > "%H3I_OUT%\pe-label.txt" || goto :no_expansion
set /p MINIMAX_H3_IMAGE_PE_LABEL=<"%H3I_OUT%\pe-label.txt"
if not defined MINIMAX_H3_IMAGE_PE_LABEL goto :no_expansion

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > "%H3I_OUT%\cuda-device-inventory.csv" || goto :no_card
type "%H3I_OUT%\cuda-device-inventory.csv"
set "H3I_CARD="
set "H3I_FREE=0"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do call :consider_card %%A %%B
if not defined H3I_CARD goto :no_card
set "CUDA_VISIBLE_DEVICES=%H3I_CARD%"
> "%H3I_OUT%\pe-render-device-selection.txt" echo CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%H3I_CARD% free_mib=%H3I_FREE%
type "%H3I_OUT%\pe-render-device-selection.txt"

call "%VCVARS%"
set "H3I_FQN=h3_image_acceptance_real::h3_image_pe_expanded_prompt_renders_more_detail_cuda"
set "log=%H3I_OUT%\h3_image_pe_expanded_prompt_renders_more_detail_cuda.log"
set "MODEL_INFERENCE_CASE_FQN=%H3I_FQN%"
cargo test --locked -p candle-gen-minimax-h3 --features cuda,model-inference-proof --release --test integration %H3I_FQN% -- --ignored --exact --nocapture > "%log%" 2>&1
set "H3I_STATUS=%ERRORLEVEL%"
set "MODEL_INFERENCE_CASE_FQN="
type "%log%"
if not "%H3I_STATUS%"=="0" goto :failed
set "H3I_PROOF=admitted_resident_cuda"
findstr /C:"admitted BoundedTransformerResidency" "%log%" >nul && set "H3I_PROOF=admitted_streamed_cuda"
echo proof configuration: %H3I_PROOF%
"%REVIEWED_PYTHON%" -I scripts/ci/verify_model_inference_log.py --package candle-gen-minimax-h3 --configuration %H3I_PROOF% --fqn %H3I_FQN% --log "%log%" || goto :failed
exit /b 0

:failed
echo ::error::the PE-expansion render failed or lacks its registered proof evidence - exit %H3I_STATUS%
exit /b 1

:no_evidence
echo ::error::the PE job's nvfp4.json is missing: %H3I_EVIDENCE%
exit /b 1

:no_expansion
echo ::error::nvfp4.json has no t2i-0 expansion
exit /b 1

:no_card
echo ::error::no CUDA card found
exit /b 1

rem %1 card index, %2 its free MiB: keep the card with the most free memory
:consider_card
if %~2 GTR %H3I_FREE% set "H3I_CARD=%~1"
if %~2 GTR %H3I_FREE% set "H3I_FREE=%~2"
exit /b 0
