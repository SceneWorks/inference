//! One explicitly selected real-weight comparison process; missing environment is an error.

#[test]
#[ignore = "requires a frozen model and complete BONSAI_COMPARISON_* evidence environment"]
fn native_comparison() {
    mlx_rs::memory::reset_peak_memory();
    core_llm_testkit::comparison::run_environment(
        |spec| {
            let spec = match std::env::var("BONSAI_COMPARISON_PROJECTOR") {
                Ok(path) if !path.trim().is_empty() => spec.clone().with_projector(path),
                _ => spec.clone(),
            };
            mlx_llm::load_for_model(&spec)
        },
        || {
            serde_json::json!({
                "backend":"mlx",
                "device":"unified",
                "active_bytes":mlx_rs::memory::get_active_memory(),
                "cache_bytes":mlx_rs::memory::get_cache_memory(),
                "peak_active_bytes":mlx_rs::memory::get_peak_memory(),
            })
        },
    )
    .expect("complete native comparison evidence");
}
