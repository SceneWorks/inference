//! One explicitly selected real-weight comparison process; missing environment is an error.

#[test]
#[ignore = "requires a frozen model and complete BONSAI_COMPARISON_* evidence environment"]
fn native_comparison() {
    core_llm_testkit::comparison::run_environment(
        |spec| {
            let spec = match std::env::var("BONSAI_COMPARISON_PROJECTOR") {
                Ok(path) if !path.trim().is_empty() => spec.clone().with_projector(path),
                _ => spec.clone(),
            };
            candle_llm::load_for_model(&spec)
        },
        || {
            serde_json::json!({
                "backend":"candle",
                "native_allocator_counters_available":false,
                "memory_evidence":"external process RSS and per-process CUDA samples required",
            })
        },
    )
    .expect("complete native comparison evidence");
}
