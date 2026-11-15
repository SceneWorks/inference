cargo test --locked -p mlx-llm --test integration conformance::real_model_passes_core_llm_conformance -- --ignored --nocapture
cargo test --locked -p mlx-llm --test integration conformance::real_qwen3_passes_core_llm_conformance -- --ignored --nocapture
