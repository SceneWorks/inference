rem SW-7106 - the H3-Image sampler A/B on candle/CUDA: the registered generator renders two
rem prompts at two seeds, 2048x2048 bf16, with the released velocity Euler and with ComfyUI's
rem stochastic LCM, s_noise 1.1 to 0.9. The gate asserts every render is non-degenerate - pixel
rem std above 10 - and that each LCM render differs from its Euler twin; the eight PNGs and the
rem log are uploaded as minimax-h3-image-sampler-ab-candle. The bf16 DiT stays resident, so the
rem card with the most free memory is selected, as the gates step does, and the gate is REFUSED
rem when that card has less free memory than the DiT's shard bytes plus H3I_HEADROOM_MIB.
rem No command here sits inside a parenthesized block and no echoed text carries a parenthesis.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MINIMAX_H3_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765"
set "MINIMAX_H3_IMAGE_SNAPSHOT=%CANDLE_GEN_MODELS_ROOT%\models--MiniMaxAI--MiniMax-H3-Image\snapshots\492fd8c4eca666f1085f57e30738c1e0d86b3439"
set "MINIMAX_H3_IMAGE_SAMPLER_AB_OUT=%RUNNER_TEMP%\minimax-h3-image-sampler-ab"
set "H3I_HEADROOM_MIB=4096"
if not exist "%MINIMAX_H3_IMAGE_SAMPLER_AB_OUT%" mkdir "%MINIMAX_H3_IMAGE_SAMPLER_AB_OUT%"
set "log=%MINIMAX_H3_IMAGE_SAMPLER_AB_OUT%\sampler-ab.log"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "%MINIMAX_H3_IMAGE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-text-encoder --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
set "PYTHONPATH="
"%REVIEWED_PYTHON%" -I -c "import pathlib, sys; print(-(-sum(p.stat().st_size for p in pathlib.Path(sys.argv[1]).glob('*.safetensors')) // 1048576))" "%MINIMAX_H3_IMAGE_SNAPSHOT%\transformer_diffusers" > "%MINIMAX_H3_IMAGE_SAMPLER_AB_OUT%\dit-weight-mib.txt" || exit /b 1
set /p H3I_DIT_MIB=<"%MINIMAX_H3_IMAGE_SAMPLER_AB_OUT%\dit-weight-mib.txt"
set /a H3I_REQUIRED_MIB=H3I_DIT_MIB + H3I_HEADROOM_MIB
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "H3I_CARD="
set "H3I_FREE=0"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do call :consider_card %%A %%B
if not defined H3I_CARD goto :no_card
set "CUDA_VISIBLE_DEVICES=%H3I_CARD%"
echo CUDA_VISIBLE_DEVICES=%H3I_CARD% free_mib=%H3I_FREE% required_mib=%H3I_REQUIRED_MIB%
if %H3I_FREE% LSS %H3I_REQUIRED_MIB% goto :refused
call "%VCVARS%"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration h3_image_sampler_real::image_sampler_ab_renders_2048_on_cuda -- --ignored --exact --nocapture > "%log%" 2>&1
set "H3I_STATUS=%ERRORLEVEL%"
type "%log%"
if not "%H3I_STATUS%"=="0" goto :failed
findstr /C:"test result: ok. 1 passed" "%log%" >nul || goto :failed
exit /b 0

:failed
echo ::error::the sampler A/B failed or did not run exactly one passing test - exit %H3I_STATUS%
exit /b 1

:refused
echo ::error::the sampler A/B is REFUSED: card %H3I_CARD% has %H3I_FREE% MiB free, below the %H3I_REQUIRED_MIB% MiB a resident bf16 DiT needs
exit /b 1

:no_card
echo ::error::no CUDA card found
exit /b 1

rem %1 card index, %2 its free MiB: keep the card with the most free memory
:consider_card
if %~2 GTR %H3I_FREE% set "H3I_CARD=%~1"
if %~2 GTR %H3I_FREE% set "H3I_FREE=%~2"
exit /b 0
