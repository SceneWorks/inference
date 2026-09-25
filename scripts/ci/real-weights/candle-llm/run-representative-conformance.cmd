call "%VCVARS%"
cargo test --locked -p candle-llm --features cuda --test conformance real_model_passes_core_llm_conformance -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-llm --features cuda --test conformance qwen3_passes_core_llm_conformance -- --ignored --nocapture || exit /b 1
