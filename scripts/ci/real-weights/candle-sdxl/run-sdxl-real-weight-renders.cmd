call "%VCVARS%"
cargo test --locked -p candle-gen-sdxl --features cuda --release --test integration conformance::sdxl_conformance -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-gen-sdxl --features cuda --release --test integration conformance::sdxl_cfg_off_guidance_one_render -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-gen-sdxl --features cuda --release --lib edit_validate::real_weight_edit -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-gen-sdxl --features cuda --release --lib ip_validate::real_weight_ip_adapter -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-gen-sdxl --features cuda --release --test integration trainer_e2e::sdxl_trainer_lora_trains_reloads_and_renders -- --ignored --nocapture || exit /b 1
