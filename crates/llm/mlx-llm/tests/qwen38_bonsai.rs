//! One explicitly selected real-weight comparison process; missing environment is an error.

#[test]
#[ignore = "requires a frozen model and complete BONSAI_COMPARISON_* evidence environment"]
fn native_comparison() {
    mlx_rs::memory::reset_peak_memory();
    core_llm_testkit::comparison::run_environment(mlx_llm::load_for_model, || {
        serde_json::json!({
            "backend":"mlx",
            "active_bytes":mlx_rs::memory::get_active_memory(),
            "cache_bytes":mlx_rs::memory::get_cache_memory(),
            "peak_active_bytes":mlx_rs::memory::get_peak_memory(),
        })
    })
    .expect("complete native comparison evidence");
}
