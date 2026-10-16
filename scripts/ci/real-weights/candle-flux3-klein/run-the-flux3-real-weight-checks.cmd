rem The story-local real-weight evidence (sc-26235, the Candle twin of the mlx-flux3-klein lane): the
rem pinned package through the Candle port on the CUDA card, one case per process. The component
rem cases compare against the COMMITTED upstream CPU reference `flux3_real_golden_64.safetensors`
rem (the same oracle the MLX port is held to); the render case is the ordinary catalog route
rem (`provider_registry().load("flux3_klein")` -> one text-to-image render at FLUX3_REAL_SIZE /
rem FLUX3_REAL_STEPS) and keeps the PNG + its JSON record (with the report-only `nvidia-smi` VRAM peak)
rem in the evidence artifact. No number gates anything else; the job is red when a case fails or
rem does not run exactly one passing test.
call "%VCVARS%"
set "FLUX3_FAILED=0"
call :run_one real_package_config_is_the_release real_weights::real_package_config_is_the_release || set "FLUX3_FAILED=1"
call :run_one real_text_encoder_matches_upstream real_weights::real_text_encoder_matches_upstream || set "FLUX3_FAILED=1"
call :run_one real_autoencoder_decode_matches_upstream real_weights::real_autoencoder_decode_matches_upstream || set "FLUX3_FAILED=1"
call :run_one real_dit_forward_matches_upstream real_weights::real_dit_forward_matches_upstream || set "FLUX3_FAILED=1"
call :run_one real_e2e_64_matches_upstream real_weights::real_e2e_64_matches_upstream || set "FLUX3_FAILED=1"
call :run_one real_distilled_text_to_image_renders real_weights::real_distilled_text_to_image_renders || set "FLUX3_FAILED=1"
dir "%FLUX3_OUT%"
if not "%FLUX3_FAILED%"=="0" exit /b 1
exit /b 0

rem %1 = evidence log name, %2 = test path in the integration binary (every case is #[ignore]d).
:run_one
set "log=%FLUX3_OUT%\%~1.log"
cargo test --locked --release -p candle-gen-flux3 --features cuda --test integration %~2 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~2 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~2 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
