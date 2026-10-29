#![cfg(test)]

use super::projected_dense_peak_gb;

/// The empirical fit must reproduce the measured first-step peaks (the basis of the pre-flight OOM
/// guard) and stay monotonic. Measured by `first_step_repro::first_step_attn_ckpt_sweep` (128 GB
/// Mac, f32, attention-segment checkpointing on): s = (edge/32)² → 512/768/1024 = 256/576/1024
/// tokens → 23.3/31.5/42.6 GB.
#[test]
fn projection_matches_measured_curve() {
    for (s, measured) in [(256.0, 23.3), (576.0, 31.5), (1024.0, 42.6)] {
        let p = projected_dense_peak_gb(s);
        assert!(
            (p - measured).abs() < 1.5,
            "f32 projection at s={s} = {p:.1} GB, expected ≈{measured} GB"
        );
    }
    // Monotonic increasing in token count.
    assert!(projected_dense_peak_gb(256.0) < projected_dense_peak_gb(576.0));
    assert!(projected_dense_peak_gb(576.0) < projected_dense_peak_gb(1024.0));
    // 1024 still fits a 64 GB video tier (budget ≈ 54 GB) without block-checkpointing.
    assert!(projected_dense_peak_gb(1024.0) < 54.0);
}
