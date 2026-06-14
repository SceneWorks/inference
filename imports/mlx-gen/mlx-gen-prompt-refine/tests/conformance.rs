//! Real-weight gen-core **TextLlm contract** conformance for `prompt_refine` (epic 3720, sc-5552).
//!
//! The MLX half of the "one real provider per contract" AC (the candle backend, sc-5500, is held to
//! the same checks): it drives the actual Llama-3.2-3B MLX engine through the backend-neutral checks
//! (validate honesty, progress monotonicity, typed pre-inference cancellation, registry round-trip).
//! `#[ignore]` because it needs a real Llama-3.2-3B-Instruct snapshot; run on the self-hosted
//! Apple-Silicon runner or a populated dev box:
//!   cargo test -p mlx-gen-prompt-refine --test conformance -- --ignored --nocapture

use std::path::PathBuf;

// Force-link the provider so its `inventory::submit!` registration survives the linker (this test
// references no other prompt-refine symbol) — the registry round-trip check would otherwise fail.
use mlx_gen_prompt_refine as _;

use gen_core_testkit::TextLlmProfile;
use mlx_gen::{LoadSpec, WeightsSource};

#[test]
#[ignore = "needs a Llama-3.2-3B-Instruct snapshot; set PROMPT_REFINE_MODEL_DIR (macos-mlx / dev box only)"]
fn prompt_refine_satisfies_gen_core_contract() {
    let root = PathBuf::from(std::env::var("PROMPT_REFINE_MODEL_DIR").expect(
        "set PROMPT_REFINE_MODEL_DIR to a Llama-3.2-3B-Instruct snapshot directory \
             (config.json, tokenizer.json, model-*.safetensors)",
    ));
    let id = mlx_gen_prompt_refine::prompt::PROMPT_REFINE_ID;
    gen_core_testkit::textllm_conformance(
        || {
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            mlx_gen::registry::load_textllm(id, &spec).expect("load prompt_refine")
        },
        // Short prompt / 16 greedy tokens — the cheapest valid generation (greedy is seed-free, fast).
        &TextLlmProfile::cheap(),
    );
}
