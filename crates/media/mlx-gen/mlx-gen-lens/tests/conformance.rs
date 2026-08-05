//! Real-weight gen-core **contract conformance** for `lens_turbo` / `lens` (epic 3720, sc-4481
//! standard) — capability honesty, progress monotonicity, typed cancellation, seed determinism, and
//! the CFG-off render arm.
//!
//! Added by sc-17616. sc-17418 put `check_cfg_off_render` into the shared suite and recorded that it
//! "covers `mlx-gen-lens` behaviourally on the nightly" — but **mlx-gen-lens was never wired to the
//! testkit at all**: no `[dev-dependencies]`, no `tests/conformance.rs`, and no entry in
//! `real-weights.yml`. Only boogu, krea and z-image ran the suite. So the arm covering the very axis
//! lens is most sensitive to had never executed against lens. This file closes that gap; without it
//! sc-17616's "confirm it still passes the arm" acceptance item is unsatisfiable.
//!
//! `lens_turbo` is the pointed case: its default guidance IS `1.0`, so the CFG-off arm exercises the
//! **production default path**, not an edge case. The arm asserts two renders differing only in their
//! negative prompt are byte-identical — at `g = 1.0` the combine reduces to `cond`, so the
//! unconditional branch cannot reach the output under any correct implementation. That is what makes
//! it catch a narrow (or now, a gate) to the *wrong* row, which a liveness-only check cannot see.
//!
//! `#[ignore]`d — needs the real `SceneWorks/Lens-Turbo` snapshot (~21 GiB on disk) and Metal:
//! ```sh
//! LENS_SNAPSHOT=/path/to/models--SceneWorks--Lens-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-lens --release --test conformance -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use gen_core_testkit::Profile;
use mlx_gen::{LoadSpec, WeightsSource};

/// The Lens snapshot directory. Explicit passed-in path only — inference never self-fetches or
/// derives a cache location (epic 13657).
fn snapshot() -> PathBuf {
    let p = std::env::var("LENS_SNAPSHOT").unwrap_or_else(|_| {
        panic!(
            "set LENS_SNAPSHOT to the required Lens snapshot dir (tokenizer/ text_encoder/ \
             transformer/ vae/); inference never self-fetches or derives a cache location \
             (epic 13657)"
        )
    });
    PathBuf::from(p)
}

/// Drive one registered Lens variant through the whole backend-neutral suite.
fn run_conformance(model_id: &str) {
    let snap = snapshot();
    gen_core_testkit::conformance(
        || {
            // Dense bf16 — the cheapest load; the suite exercises the contract, not quantization.
            let spec = LoadSpec::new(WeightsSource::Dir(snap.clone()));
            mlx_gen_lens::provider_registry()
                .unwrap()
                .load(model_id, &spec)
                .unwrap_or_else(|e| panic!("load {model_id}: {e}"))
        },
        // 256² / 2 steps — lens `min_size` is 256 and both edges must be divisible by the 16×
        // VAE·patch alignment, so the cheap profile is already valid here.
        &Profile::cheap(),
    );
}

/// The turbo variant — **default guidance 1.0**, i.e. the CFG-off arm runs the production default.
#[test]
#[ignore = "needs the real SceneWorks/Lens-Turbo snapshot (LENS_SNAPSHOT) + a Metal-capable macOS host"]
fn lens_turbo_satisfies_gen_core_contract() {
    run_conformance(mlx_gen_lens::registry::MODEL_ID_TURBO);
}

/// The base variant — same architecture and snapshot layout, but a **guided** default (guidance > 1),
/// so this arm covers the joint `[pos; neg]` path the turbo id never takes. Both ids resolve from one
/// snapshot, so running both is one extra model load, not a second download.
#[test]
#[ignore = "needs the real SceneWorks/Lens snapshot (LENS_SNAPSHOT) + a Metal-capable macOS host"]
fn lens_base_satisfies_gen_core_contract() {
    run_conformance(mlx_gen_lens::registry::MODEL_ID_BASE);
}
