//! gen-core contract conformance for the candle Chroma provider (sc-4481, epic 3692 / sc-5484).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism — against the real candle generator.
//! The **seed-determinism** check is the regression guard for the deterministic CPU-seeded noise
//! (sc-3673 parity) the pipeline relies on.
//!
//! Each drives a real `generate`, so it needs the CUDA backend + a local Chroma snapshot and is
//! `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set CHROMA_HD_SNAPSHOT=C:\Users\…\models--lodestones--Chroma1-HD\snapshots\<hash>
//! set CHROMA_FLASH_SNAPSHOT=C:\Users\…\models--lodestones--Chroma1-Flash\snapshots\<hash>
//! cargo test -p candle-gen-chroma --features cuda --release --test conformance -- --ignored
//! ```
//!
//! As with the SDXL/FLUX/Z-Image slices: same-backend determinism only; cross-backend pixel equality
//! vs `mlx-gen-chroma` is NOT a goal (RNG algorithms differ).
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use gen_core_testkit::{conformance, Profile};

#[test]
#[ignore = "needs CHROMA_HD_SNAPSHOT (a lodestones/Chroma1-HD snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn chroma_hd_conformance() {
    let snap = std::env::var("CHROMA_HD_SNAPSHOT")
        .expect("set CHROMA_HD_SNAPSHOT to a lodestones/Chroma1-HD snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // 256² (== the descriptor's min_size) and a tiny step count keep the suite's ~4 generate() calls
    // cheap — it verifies contract behavior, not image quality. true_cfg defaults to the variant's 4.0,
    // exercising the dual-branch CFG path. Both dims are multiples of the /16 alignment.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 4,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_chroma::load_hd(&spec).unwrap(), &profile);
}

#[test]
#[ignore = "needs CHROMA_FLASH_SNAPSHOT (a lodestones/Chroma1-Flash snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn chroma_flash_conformance() {
    let snap = std::env::var("CHROMA_FLASH_SNAPSHOT")
        .expect("set CHROMA_FLASH_SNAPSHOT to a lodestones/Chroma1-Flash snapshot dir");
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap)));

    // Flash is distilled (true_cfg 1.0 → single forward), so the negative branch is skipped — the
    // cheaper conformance run, exercising the no-CFG path.
    let profile = Profile {
        width: 256,
        height: 256,
        steps: 4,
        ..Profile::cheap()
    };
    conformance(|| candle_gen_chroma::load_flash(&spec).unwrap(), &profile);
}
