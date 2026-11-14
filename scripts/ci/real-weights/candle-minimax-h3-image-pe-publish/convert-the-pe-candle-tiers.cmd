rem NVFP4 -> bf16 on the CPU, then candle-llm GGML Q8_0 / Q4_K (SW-7099); the bf16 intermediate is
rem scratch and is removed.
call "%VCVARS%"
cargo build --locked --release -p candle-llm --example modelopt_nvfp4_to_bf16 --example prepare_snapshot || exit /b 1
"%CARGO_TARGET_DIR%\release\examples\modelopt_nvfp4_to_bf16.exe" "%MINIMAX_PE_SNAPSHOT%" "%PUBLISH_SCRATCH%\bf16" || exit /b 1
"%CARGO_TARGET_DIR%\release\examples\prepare_snapshot.exe" "%PUBLISH_SCRATCH%\bf16" "%PUBLISH_SCRATCH%\candle-q8" q8 || exit /b 1
"%CARGO_TARGET_DIR%\release\examples\prepare_snapshot.exe" "%PUBLISH_SCRATCH%\bf16" "%PUBLISH_SCRATCH%\candle-q4" q4 || exit /b 1
rmdir /s /q "%PUBLISH_SCRATCH%\bf16"
exit /b 0
