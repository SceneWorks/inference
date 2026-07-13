//! Real-weight gen-core **contract conformance** for `z_image_turbo` (epic 3720, sc-4481).
//!
//! This is the "one real family" the testkit AC pins to the macos-mlx lane: it drives the actual
//! MLX engine through the backend-neutral checks (capability honesty, progress monotonicity, typed
//! cancellation, seed determinism, registry round-trip) — the guarantees a candle provider will be
//! held to identically. `#[ignore]` because it needs the real `Tongyi-MAI/Z-Image-Turbo` weights
//! (set `ZIMAGE_SNAPSHOT` or populate the HF hub cache); run it on the self-hosted Apple-Silicon
//! runner or a populated dev box.

mod common;

// Force-link the provider so its `inventory::submit!` registration survives (otherwise the linker
// dead-strips it — this test references no other z-image symbol — and `mlx_gen::load` reports "no
// generator registered"). The worker does the same `as _` import per model crate.
use mlx_gen_z_image as _;

use gen_core_testkit::Profile;
use mlx_gen::{LoadSpec, WeightsSource};

#[test]
#[ignore = "needs real Z-Image-Turbo weights (ZIMAGE_SNAPSHOT or HF cache); macos-mlx / dev box only"]
fn z_image_turbo_satisfies_gen_core_contract() {
    let snap = common::snapshot();
    gen_core_testkit::conformance(
        || {
            // Dense bf16 — the cheapest load; the suite exercises the contract, not quantization.
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            mlx_gen::load("z_image_turbo", &spec).expect("load z_image_turbo")
        },
        // 256² / 2 steps — the minimum valid Z-Image config (min_size 256, multiple-of-16).
        &Profile::cheap(),
    );
}
