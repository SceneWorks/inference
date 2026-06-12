//! gen-core contract conformance for the candle SDXL provider (sc-4481, epic 3720).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism, registry round-trip — against the real candle generator.
//! This is the suite whose **seed-determinism** check is the regression guard for the spike's
//! repro defect (sc-3498) that sc-3673 fixed.
//!
//! It drives a real `generate`, so it needs the CUDA backend + a local SDXL snapshot and is
//! `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set SDXL_SNAPSHOT=C:\Users\…\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\<hash>
//! cargo test -p candle-gen-sdxl --features cuda --release --test conformance -- --ignored
//! ```
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs SDXL_SNAPSHOT (a diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn sdxl_conformance() {
    let snap = std::env::var("SDXL_SNAPSHOT")
        .expect("set SDXL_SNAPSHOT to a stabilityai/stable-diffusion-xl-base-1.0 snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 512² (≥ the descriptor's min_size 512) at a small step count keeps the suite's ~4 generate()
    // calls cheap — it verifies contract behavior, not image quality. `steps` must equal what the
    // model resolves req.steps to (check_progress asserts Step.total == profile.steps); the pipeline
    // uses req.steps verbatim, so 4 → 4.
    let profile = Profile {
        width: 512,
        height: 512,
        steps: 4,
        ..Profile::cheap()
    };

    // Resolve through THIS crate's `load` (its inventory registration is linked into the test binary,
    // so the suite's registry round-trip check also passes). Panics with aggregated failures.
    conformance(|| candle_gen_sdxl::load(&spec).unwrap(), &profile);
}
