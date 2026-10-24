//! One explicitly selected real-weight comparison process; missing environment is an error.

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_core::Tensor, ()>::Err(()));
};

#[test]
#[ignore = "requires a frozen model and complete BONSAI_COMPARISON_* evidence environment"]
fn native_comparison() {
    let native_device = {
        let selected = candle_llm::select_device().expect("select the native comparison device");
        if selected.is_cpu() {
            "cpu"
        } else if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(feature = "metal") {
            "metal"
        } else {
            panic!("non-CPU Candle comparison device has no compiled backend")
        }
    };
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
                "device":native_device,
                "peak_active_bytes":null,
                "native_allocator_counters_available":false,
                "peak_unavailable_reason":"Candle exposes no portable native active-allocator peak counter",
                "memory_evidence":"external process RSS and per-process CUDA samples required",
            })
        },
        || false,
    )
    .expect("complete native comparison evidence");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED native_comparison");
}
