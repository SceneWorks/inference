//! Opt-in model-inference device readback (a6 shared contract, in-scope copy).
//!
//! Emits the actual selected-stream device kind for the registered
//! `MODEL_INFERENCE_CASE_FQN`; a no-op when the env is absent (ordinary CI
//! unchanged). Reads the SELECTED stream via `Stream::task_local_or_default()`
//! → `get_device()` → `Device::get_type()`; never creates an extra GPU stream,
//! never evaluates, never unions the default device with the used device.

pub fn record_model_inference_device() {
    let Some(fqn) = std::env::var("MODEL_INFERENCE_CASE_FQN").ok() else {
        return;
    };
    let kind = match mlx_rs::Stream::task_local_or_default()
        .get_device()
        .and_then(|d| d.get_type())
    {
        Ok(mlx_rs::DeviceType::Cpu) => "mlx-cpu",
        Ok(mlx_rs::DeviceType::Gpu) => "mlx-metal",
        Err(_) => "mlx-unknown",
    };
    println!("\nMODEL_INFERENCE_DEVICE {fqn} {kind}");
}

/// A missing required fixture/matrix arm must not count as a completed model proof.
pub fn record_model_inference_prerequisite_missing(reason: &str) {
    if let Ok(fqn) = std::env::var("MODEL_INFERENCE_CASE_FQN") {
        eprintln!("\nMODEL_INFERENCE_PREREQUISITE_MISSING {fqn} {reason}");
    }
}
