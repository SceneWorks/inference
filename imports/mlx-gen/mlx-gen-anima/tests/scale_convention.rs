//! LoRA scale-convention tests for mlx-gen-anima (sc-10521), in their **own integration-test binary**.
//!
//! These are `#[ignore]`d + real-weights-gated (they need the `circlestone-labs/Anima` base snapshot
//! and, for the DiT legs, the `Anima-Official-LoRAs` snapshot, plus Metal). CI runs NONE of them. Run
//! this binary alone with:
//!   cargo test -p mlx-gen-anima --release --test scale_convention -- --ignored --nocapture
//! It is also covered by the full documented invocation `cargo test -p mlx-gen-anima --release --
//! --ignored`, which runs every `tests/*.rs` binary.
//!
//! ## Why a separate binary AND a single test function
//! Each leg captures a target's base weight, runs the heavy 508-target injection, then re-reads that
//! target's forward. That capture → heavy-inject → re-read pattern strands an array on mlx-rs's Metal
//! default stream, which is **thread-local**: running the original `scale_convention_…` test **6th** in
//! the seven-test `lora_injection` binary panicked in `eval` with "There is no Stream(gpu, N) **in
//! current thread**", though it passed in isolation. The libtest harness runs every `#[test]` on its
//! own worker thread (even single-threaded via `RUST_TEST_THREADS=1`), so an array loaded on one test's
//! thread fails to `eval` once a later test on a different thread churns the per-thread stream state.
//! So two things are needed, both mirroring sc-10515's `real_weights`/`velocity_convention` handling:
//!   1. a **separate binary** (fresh process), and
//!   2. **one `#[test]` function** running every leg sequentially on a single thread, with
//!      `mlx_rs::memory::clear_cache()` between legs — exactly how `real_weights.rs::
//!      generate_all_three_variants_1024` does three back-to-back model loads in one test without
//!      tripping the stream. Splitting the legs into separate `#[test]`s reintroduces the failure.
//!
//! ## The scale convention itself: α = rank ⇒ scale 1.0, NO alpha/rank fold
//! The shipped Anima LoRAs are ComfyUI-format PEFT files that carry **no** scaling metadata — zero
//! per-target `.alpha` tensors in the file (measured: 0 `.alpha` keys), and `__metadata__ ==
//! {"format":"pt"}` (no `lora_alpha`, no `alpha_pattern`, no `rank_pattern`). With neither an `.alpha`
//! tensor nor a `lora_adapter_metadata` blob, the core loader takes its **no-fold** branch
//! (`src/adapters/loader.rs`: the `if let Some(alpha) = parts.alpha.or(cfg_alpha)` guard is skipped),
//! so `B` is installed unmodified and the effective residual is `spec.scale · B·A` with
//! `spec.scale == 1.0`. That is the α = rank ⇒ α/rank = 1.0 default, and it matches ComfyUI's own
//! missing-`alpha` behaviour for ComfyUI-format LoRAs (a file without an alpha is treated as α = rank,
//! i.e. unit scale) — cited, NOT vendored (ComfyUI is GPL-3.0; mlx-gen is Apache-2.0). The tests below
//! anchor that the *applied* residual is exactly `base + B·A` at scale 1.0 (and `base + 0.5·B·A` at
//! scale 0.5), on both a DiT and a **non-zero** conditioner target.

mod common;

use mlx_rs::Dtype;

use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;

use mlx_gen_anima::apply_anima_adapters;

use common::{
    base_weight, injected_rel_err, l2, load_base, lora_spec, synth_conditioner_lora, turbo_lora,
    COND_Q, DIT_ADALN, DIT_Q,
};

/// The non-zero conditioner `lora_B` key (raw file spelling), for the vacuity guard.
const COND_B_KEY: &str = "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_B.weight";

/// All four scale legs in ONE test (one thread — see the module doc's thread-local-stream note). Each
/// leg does one model load + one injection, then `clear_cache()`s before the next, exactly like
/// `real_weights.rs::generate_all_three_variants_1024`'s back-to-back loads.
#[test]
#[ignore = "needs the circlestone-labs/Anima + Anima-Official-LoRAs snapshots"]
fn scale_convention_dit_and_conditioner() {
    // ---------------------------------------------------------------------------------------------
    // LEG 1 — DiT + adaLN at scale 1.0 (real turbo LoRA, its 448 DiT `lora_B` are non-zero).
    // Proves the injected forward == base + B·A at scale 1.0 the way we compute B·A (self-consistency);
    // the α = rank ⇒ 1.0 choice is anchored externally by the zero `.alpha` tensors + `{"format":"pt"}`
    // metadata + the loader's no-fold branch (see the module doc). Also confirms scale-sensitivity: the
    // same scale-1.0 forward must NOT match base + 0.5·B·A.
    // ---------------------------------------------------------------------------------------------
    {
        let mut c = load_base();
        let wb_dit = base_weight(&mut c, &DIT_Q);
        let wb_adaln = base_weight(&mut c, &DIT_ADALN);
        let lw = Weights::from_file(turbo_lora()).unwrap();
        apply_anima_adapters(
            &mut c.dit,
            &mut c.conditioner,
            &[lora_spec(turbo_lora(), 1.0)],
        )
        .unwrap();

        let e_dit = injected_rel_err(&mut c, &DIT_Q, &lw, &wb_dit, 1.0);
        let e_adaln = injected_rel_err(&mut c, &DIT_ADALN, &lw, &wb_adaln, 1.0);
        let e_dit_half = injected_rel_err(&mut c, &DIT_Q, &lw, &wb_dit, 0.5);
        println!("[DiT scale=1.0] rel_err  DiT={e_dit:.2e}  adaLN={e_adaln:.2e}  (vs 0.5·B·A: {e_dit_half:.2e})");
        for (name, e) in [("DiT", e_dit), ("adaLN", e_adaln)] {
            assert!(
                e < 2e-2,
                "{name}: injected weight != base + B·A at scale 1.0 (rel_err {e:.3e})"
            );
        }
        assert!(
            e_dit_half > 1e-3,
            "DiT scale-1.0 forward also matches 0.5·B·A — the scale check is insensitive"
        );
    }
    mlx_rs::memory::clear_cache();

    // ---------------------------------------------------------------------------------------------
    // LEG 2 — DiT halve-scale mutation: inject the turbo LoRA at scale 0.5. The forward must MATCH
    // base + 0.5·B·A and DIVERGE from base + 1.0·B·A (scale enters linearly; a wrong scale FAILS).
    // ---------------------------------------------------------------------------------------------
    {
        let mut c = load_base();
        let wb = base_weight(&mut c, &DIT_Q);
        let lw = Weights::from_file(turbo_lora()).unwrap();
        apply_anima_adapters(
            &mut c.dit,
            &mut c.conditioner,
            &[lora_spec(turbo_lora(), 0.5)],
        )
        .unwrap();

        let e_half_ref = injected_rel_err(&mut c, &DIT_Q, &lw, &wb, 0.5);
        let e_vs_scale1 = injected_rel_err(&mut c, &DIT_Q, &lw, &wb, 1.0);
        println!(
            "[DiT scale=0.5] rel_err vs 0.5·B·A={e_half_ref:.3e}  vs 1.0·B·A={e_vs_scale1:.3e}"
        );
        assert!(
            e_half_ref < 2e-2,
            "scale enters linearly (base + 0.5·B·A), rel_err {e_half_ref:.3e}"
        );
        assert!(
            e_vs_scale1 > 1e-3,
            "halving the scale did NOT change the forward — the scale test is insensitive!"
        );
    }
    mlx_rs::memory::clear_cache();

    // ---------------------------------------------------------------------------------------------
    // LEG 3 — conditioner at scale 1.0, via a synthesized NON-ZERO LoRA (Issue 3a). The shipped turbo
    // LoRA's 60 conditioner `lora_B` are all zero-initialized, so `B·A ≡ 0` there and `want == got ==
    // base_out` for ANY scale — a conditioner scale assertion against the turbo file is vacuous. This
    // fixture drives a real non-zero delta so the scale convention is genuinely exercised on a
    // `llm_adapter.*` target. Proves forward == base + B·A at scale 1.0 AND diverges from base + 0.5·B·A.
    // ---------------------------------------------------------------------------------------------
    {
        let synth = synth_conditioner_lora();
        let lw = Weights::from_file(&synth).unwrap();
        // Vacuity guard: the synth B is non-zero, so base + B·A genuinely differs from base and the
        // scale assertion below can actually fail (the turbo conditioner leg could not).
        let nb = l2(&lw
            .require(COND_B_KEY)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap());
        assert!(
            nb > 1e-2,
            "synth conditioner LoRA B must be non-zero or the scale check is vacuous: |B|={nb:.3e}"
        );

        let mut c = load_base();
        let wb = base_weight(&mut c, &COND_Q);
        apply_anima_adapters(
            &mut c.dit,
            &mut c.conditioner,
            &[AdapterSpec::new(synth, 1.0, AdapterKind::Lora)],
        )
        .unwrap();

        let e1 = injected_rel_err(&mut c, &COND_Q, &lw, &wb, 1.0);
        let e_half = injected_rel_err(&mut c, &COND_Q, &lw, &wb, 0.5);
        println!(
            "[conditioner scale=1.0] rel_err vs 1.0·B·A={e1:.3e}  vs 0.5·B·A={e_half:.3e}  |B|={nb:.3e}"
        );
        assert!(
            e1 < 2e-2,
            "conditioner: injected forward != base + B·A at scale 1.0 (rel_err {e1:.3e})"
        );
        assert!(
            e_half > 3e-2,
            "conditioner scale assertion is vacuous — 0.5·B·A also matches (rel_err {e_half:.3e})"
        );
    }
    mlx_rs::memory::clear_cache();

    // ---------------------------------------------------------------------------------------------
    // LEG 4 — conditioner halve-scale: inject the same synth LoRA at scale 0.5. The forward must MATCH
    // base + 0.5·B·A and DIVERGE from base + 1.0·B·A — the non-zero conditioner analogue of LEG 2.
    // ---------------------------------------------------------------------------------------------
    {
        let synth = synth_conditioner_lora();
        let lw = Weights::from_file(&synth).unwrap();

        let mut c = load_base();
        let wb = base_weight(&mut c, &COND_Q);
        apply_anima_adapters(
            &mut c.dit,
            &mut c.conditioner,
            &[AdapterSpec::new(synth, 0.5, AdapterKind::Lora)],
        )
        .unwrap();

        let e_half = injected_rel_err(&mut c, &COND_Q, &lw, &wb, 0.5);
        let e_full = injected_rel_err(&mut c, &COND_Q, &lw, &wb, 1.0);
        println!(
            "[conditioner scale=0.5] rel_err vs 0.5·B·A={e_half:.3e}  vs 1.0·B·A={e_full:.3e}"
        );
        assert!(
            e_half < 2e-2,
            "conditioner: injected forward != base + 0.5·B·A at scale 0.5 (rel_err {e_half:.3e})"
        );
        assert!(
            e_full > 3e-2,
            "halving the conditioner scale did NOT change the forward — the scale test is insensitive!"
        );
    }
    mlx_rs::memory::clear_cache();
}
