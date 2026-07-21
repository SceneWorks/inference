//! Real-weight gen-core **contract conformance** for `krea_2_turbo` (epic 3720, sc-4481 standard).
//!
//! Drives the assembled MLX engine (tokenizer + Qwen3-VL-4B TE + single-stream DiT + Qwen-Image VAE)
//! through the backend-neutral checks — capability honesty, progress monotonicity, typed cancellation
//! ([`mlx_gen::Error::Canceled`] round-tripping to `gen_core::Error::Canceled`), seed determinism,
//! and seed determinism — the same guarantees a candle provider will be held to (sc-7580). `#[ignore]`
//! because it needs the real `krea/Krea-2-Turbo` weights; run on the macos-mlx lane / a dev box:
//! ```sh
//! KREA_TURBO_DIR=/path/to/models--krea--Krea-2-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-krea --release --test conformance -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use gen_core_testkit::Profile;
use mlx_gen::{LoadSpec, WeightsSource};

/// The `krea/Krea-2-Turbo` snapshot: `KREA_TURBO_DIR` if set, else the first snapshot under the HF hub
/// cache. Panics with a clear message when absent (the `#[ignore]` gate needs real weights to run).
fn snapshot() -> PathBuf {
    let p = std::env::var("KREA_TURBO_DIR").unwrap_or_else(|_| panic!("set KREA_TURBO_DIR to the required snapshot dir; inference never self-fetches or derives a cache location (epic 13657)"));
    PathBuf::from(p)
}

#[test]
#[ignore = "needs real Krea 2 Turbo weights (KREA_TURBO_DIR or HF cache); macos-mlx / dev box only"]
fn krea_2_turbo_satisfies_gen_core_contract() {
    let snap = snapshot();
    gen_core_testkit::conformance(
        || {
            // Dense bf16 — the cheapest load; the suite exercises the contract, not quantization.
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            mlx_gen_krea::provider_registry()
                .unwrap()
                .load("krea_2_turbo", &spec)
                .expect("load krea_2_turbo")
        },
        // 256² / few-step — the minimum valid Krea config (min_size 256, multiple-of-16).
        &Profile::cheap(),
    );
}
