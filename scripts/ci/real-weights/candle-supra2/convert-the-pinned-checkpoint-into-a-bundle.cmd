rem The install-time converter: verifies the pinned checkpoint and both component snapshots
rem (sizes + SHA-256), then writes the native bundle the provider loads.
if exist "%SUPRA2_BUNDLE%" rmdir /s /q "%SUPRA2_BUNDLE%"
cargo run --locked --release -p sceneworks-gen-core --example supra2_convert -- "%SUPRA2_SOURCE_SNAPSHOT%\model_final_ema.pt" "%SUPRA2_BUNDLE%" "%SUPRA2_T5_SNAPSHOT%" "%SUPRA2_VAE_SNAPSHOT%" > "%SUPRA2_OUT%\convert.log" 2>&1 || (type "%SUPRA2_OUT%\convert.log" & exit /b 1)
type "%SUPRA2_OUT%\convert.log"
copy /y "%SUPRA2_BUNDLE%\supra2_bundle.json" "%SUPRA2_OUT%\supra2_bundle.json" >nul || exit /b 1
