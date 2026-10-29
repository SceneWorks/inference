//! sc-18724 — the lightx2v turbo LoRA folds at the **measured** strength, on the **published** key
//! space, over the committed tiny DiT.
//!
//! # What this file is defending
//!
//! `lightx2v/Minimax-h3-Turbo` stamps its LoRA alpha as a bare top-level `__metadata__` string and
//! ships **no** `rank` key, no per-target `.alpha` tensor and no `lora_adapter_metadata` blob. Both
//! pre-existing in-tree alpha paths mishandle that with no error at all — the PEFT one folds an
//! `alpha=8, rank=128` file **16× too strong**, the `parse_rank_alpha` one **128×**. Neither failure
//! has a shape, a checksum or a key-coverage proof that can see it: the render simply comes out
//! wrong. See [`mlx_gen_minimax_h3::adapters`].
//!
//! # Why all three alpha cases are here, not just one
//!
//! The set disagrees *within one repo* — `alpha: "128"` on the 4-step 768p file, `alpha: "8"` on the
//! 8-step and the ref2v ones, and **nothing at all** on `4step_v0.1`. A test covering only the
//! `alpha=128` file would pass against an implementation that hardcodes `1.0`, and a test covering
//! only an `alpha=8` file would pass against one that hardcodes `0.0625`. Every constant-returning
//! implementation fails at least one arm below.
//!
//! The absolute arm is checked against an **independently computed** `x·Aᵀ·Bᵀ` rather than against a
//! second call into the code under test, and the relative arms (`×16`, and absent == declared-8) are
//! ratios, so a shared error in the fold formula cannot cancel out of both.
//!
//! The `#[ignore]`d gate at the bottom is the same claim against the real published files:
//!
//! ```text
//! MINIMAX_H3_TURBO_LORA=<dir of lightx2v/Minimax-h3-Turbo> \
//!   cargo test -p mlx-gen-minimax-h3 --test integration turbo_lora:: -- --ignored --nocapture
//! ```

use crate::common;

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::ops::{concatenate_axis, matmul, subtract};
use mlx_rs::{Array, Device, DeviceType, Dtype};
use sha2::{Digest, Sha256};

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::adapters::{
    adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters, convert_comfyui_key_space,
    convert_minimax_h3_trainer_key_space, is_comfyui_key_space, resolve_alpha, resolve_rank,
    unflatten_minimax_h3_trainer_tensors, DEFAULT_LORA_ALPHA,
};
use mlx_gen_minimax_h3::{MiniMaxH3Dit, MiniMaxH3DitConfig, SMALLEST_LEGAL_FRAMES};

use common::{dit_fixture_config, DIT_FIXTURE};
use mlx_gen::testing::max_abs;

// sc-27342: the fold fixtures, the ComfyUI-namespace conversion gates, the real-weight render
// harness and the SC-21028 trainer-artifact support live in `turbo_lora/`. This file keeps the
// `#[ignore]`d gates, whose `turbo_lora::<name>` paths the proof registry and the synthetic-W4A8
// lane address. It is `#[path]`-loaded by `tests/main.rs`, so its submodules name their files.
#[path = "turbo_lora/comfyui_twin.rs"]
mod comfyui_twin;
#[path = "turbo_lora/fixtures.rs"]
mod fixtures;
#[path = "turbo_lora/real_weights.rs"]
mod real_weights;
#[path = "turbo_lora/trainer_receipt.rs"]
mod trainer_receipt;
use comfyui_twin::*;
use fixtures::*;
use real_weights::*;
use trainer_receipt::*;

/// **The alpha gate.** All three published spellings, over the committed fixture, at strength 1.0:
/// `alpha=128` folds at exactly 1.0, `alpha=8` at 0.0625, and an absent alpha at 0.0625 — the last
/// through [`DEFAULT_LORA_ALPHA`] and *not* through the rank.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_three_published_alphas_fold_at_one_and_one_sixteenth() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let x = tensor(&[3, cfg.hidden_size], 0.7);

    let a128 = write_lora(dir.path(), "alpha128.safetensors", &cfg, Some("128"));
    let a8 = write_lora(dir.path(), "alpha8.safetensors", &cfg, Some("8"));
    let none = write_lora(dir.path(), "noalpha.safetensors", &cfg, None);

    // The alpha=128 file must fold at exactly 1.0 — checked against an INDEPENDENTLY computed
    // `x·Aᵀ·Bᵀ` read straight off the fixture file, not against a second call into the loader.
    let w = Weights::from_file(&a128).expect("read back");
    let a = w
        .require(&format!("{PROBE}.lora_A.default.weight"))
        .unwrap()
        .clone();
    let b = w
        .require(&format!("{PROBE}.lora_B.default.weight"))
        .unwrap()
        .clone();
    assert_eq!(a.shape(), [PUBLISHED_RANK, cfg.hidden_size]);
    assert_eq!(resolve_rank(PROBE, &a, None).unwrap(), 128.0);
    let unscaled = matmul(matmul(&x, a.t()).unwrap(), b.t()).unwrap();

    let r128 = probe_residual(&cfg, &a128, 1.0, &x);
    let scale_ref = max_abs(&unscaled);
    assert!(scale_ref > 1e-3, "the probe residual must be non-trivial");
    let dev = max_abs(&subtract(&r128, &unscaled).unwrap()) / scale_ref;
    println!("[alpha=128] fold 1.0: rel-max-abs vs x·Aᵀ·Bᵀ = {dev:.3e}");
    assert!(
        dev < 1e-6,
        "alpha=128 at rank 128 must fold at exactly 1.0, got rel dev {dev:.3e}"
    );

    // The alpha=8 file must fold at 0.0625 — i.e. exactly one sixteenth of the alpha=128 one. Stated
    // as a RATIO so it cannot pass by sharing a wrong constant with the arm above.
    let r8 = probe_residual(&cfg, &a8, 1.0, &x);
    let sixteenth = max_abs(&r8) / max_abs(&r128);
    println!("[alpha=8]   |residual| / |alpha=128 residual| = {sixteenth:.6}");
    assert!(
        (sixteenth - 0.0625).abs() < 1e-5,
        "alpha=8 at rank 128 must fold 16x weaker than alpha=128, got {sixteenth}"
    );
    let want8 = max_abs(&unscaled) * alpha_rank_fold(8.0, 128.0);
    assert!(
        (max_abs(&r8) - want8).abs() / want8 < 1e-5,
        "alpha=8 must fold at 0.0625 absolutely, not merely relatively"
    );

    // And the file with NO alpha must be byte-identical to the one declaring 8 — the whole point of
    // `DEFAULT_LORA_ALPHA`. Falling back to the rank instead would make this arm equal to r128.
    let rnone = probe_residual(&cfg, &none, 1.0, &x);
    let drift = max_abs(&subtract(&rnone, &r8).unwrap()) / max_abs(&r8);
    println!("[no alpha]  rel-max-abs vs alpha=8 = {drift:.3e}");
    assert!(
        drift < 1e-6,
        "an absent alpha must fall back to DEFAULT_LORA_ALPHA = {DEFAULT_LORA_ALPHA}, not to the \
         rank; got rel dev {drift:.3e} against the declared-8 file"
    );
    let vs_rank_fallback = max_abs(&rnone) / max_abs(&r128);
    assert!(
        (vs_rank_fallback - 0.0625).abs() < 1e-5,
        "…and it must NOT equal the alpha = rank fallback (which would give 1.0), got \
         {vs_rank_fallback}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED turbo_lora::the_three_published_alphas_fold_at_one_and_one_sixteenth");
}

/// **The fold scalar is dtype-matched — and only an assertion on the INSTALLED adapter can see it.**
///
/// `apply_one_lora` folds `alpha/rank` into `b` with `scalar(fold).as_dtype(up.dtype())`. Dropping
/// the `as_dtype` is numerically **invisible**: every published fold is an exact power of two, so the
/// bf16 and f32 products carry identical values, and every numeric arm above would stay green. What
/// changes is the dtype of `b` — and with it, the dtype the low-rank `(x·A)·B` matmul runs in: f32
/// where the reference runs bf16, a silent fork-parity deviation. (It is *not* a host-widening
/// question: `AdaptableLinear::apply_adapters` narrows every residual to `out.dtype()` before the
/// add, sc-15265.)
///
/// So this test asserts the dtype of the array the loader actually installed, reached through
/// `apply_minimax_h3_adapters` — it never recomputes the multiply, because a test that reimplements
/// the thing under test validates only the reimplementation.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_installed_fold_keeps_the_bf16_factor_dtype() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(dir.path(), "bf16.safetensors", &cfg, Some("8"));

    // The fixture must really carry the published dtype, or the assertion below is vacuous — this is
    // exactly the hole the f32 fixtures left.
    let w = Weights::from_file(&lora).expect("read back");
    for suffix in ["lora_A", "lora_B"] {
        let f = w
            .require(&format!("{PROBE}.{suffix}.default.weight"))
            .unwrap();
        assert_eq!(
            f.dtype(),
            Dtype::Bfloat16,
            "{suffix}: the fixture must be bf16 like every published turbo tensor"
        );
    }

    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(lora, 1.0)]).expect("install");
    assert_eq!(report.applied, adapter_target_paths(&cfg).len());
    let segs: Vec<&str> = PROBE.split('.').collect();
    let installed = dit.adaptable_mut(&segs).expect("probe module").adapters();
    assert_eq!(installed.len(), 1, "exactly one residual per module");
    let Adapter::Lora { a, b, scale } = &installed[0] else {
        panic!("the diffusers path must install a LoRA residual, not a LoKr one");
    };
    assert_eq!(*scale, 1.0);
    assert_eq!(
        a.dtype(),
        Dtype::Bfloat16,
        "the down factor is installed untouched, so it keeps its loaded dtype"
    );
    assert_eq!(
        b.dtype(),
        Dtype::Bfloat16,
        "the alpha/rank fold must NOT promote the up factor to f32 — a bare `scalar(fold)` would, \
         running the low-rank matmul in f32 where the reference runs it in bf16"
    );
}

/// The user's `AdapterSpec::scale` multiplies the `alpha/rank` fold rather than replacing it — so a
/// strength-0.5 install of the 8-step file lands at 0.03125, and a strength-0 one is inert.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_user_strength_multiplies_the_alpha_fold() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let a8 = write_lora(dir.path(), "alpha8.safetensors", &cfg, Some("8"));

    let full = max_abs(&probe_residual(&cfg, &a8, 1.0, &x));
    let half = max_abs(&probe_residual(&cfg, &a8, 0.5, &x));
    assert!(
        (half / full - 0.5).abs() < 1e-5,
        "strength must scale the fold, got {}",
        half / full
    );
    let off = max_abs(&probe_residual(&cfg, &a8, 0.0, &x));
    assert_eq!(off, 0.0, "a scale-0 adapter must be exactly inert");
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED turbo_lora::the_user_strength_multiplies_the_alpha_fold"
    );
}

/// **Coverage.** Every module of a published-shaped file resolves, on both stacks — including the
/// token refiner, which 24 of the 624 published tensors target and which therefore cannot be a stub.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn every_target_resolves_across_both_stacks() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let lora = write_lora(dir.path(), "cover.safetensors", &cfg, Some("8"));

    let targets = adapter_target_paths(&cfg);
    assert_eq!(
        targets.len(),
        (cfg.num_layers + cfg.num_refiner_layers) as usize * 6
    );
    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(lora, 1.0)]).expect("install");
    assert_eq!(report.applied, targets.len());
    assert!(report.unmatched_paths.is_empty());

    // Every enumerated path really resolves through the host — the list and the module tree cannot
    // drift apart while this holds.
    for t in &targets {
        let segs: Vec<&str> = t.split('.').collect();
        assert!(dit.adaptable_mut(&segs).is_some(), "unreachable target {t}");
    }
    assert_eq!(
        targets
            .iter()
            .filter(|t| t.starts_with("token_refiner."))
            .count(),
        (cfg.num_refiner_layers * 6) as usize
    );

    // Modules that are deliberately NOT on the surface: the AdaLN projection (evicted mid-render),
    // the norms, and the 17 mixed-precision heads.
    for absent in [
        "transformer_blocks.0.adaln_proj.linear",
        "transformer_blocks.0.norm1",
        "transformer_blocks.0.attn.norm_q",
        "proj_in",
        "context_embedder",
        "proj_out",
        "token_refiner.final_norm",
    ] {
        let segs: Vec<&str> = absent.split('.').collect();
        assert!(
            dit.adaptable_mut(&segs).is_none(),
            "{absent} must not be an adapter target"
        );
    }
}

/// A key that resolves to no module is an **error**, not a quiet partial fold.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn an_unmatched_target_is_loud() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("stray.safetensors");
    let a = tensor(&[8, cfg.hidden_size], 1.0);
    let b = tensor(&[cfg.hidden_size, 8], 2.0);
    // One real target so the install is not rejected for matching nothing, plus one stray.
    let entries: Vec<(&str, &Array)> = vec![
        ("transformer_blocks.0.attn.to_q.lora_A.default.weight", &a),
        ("transformer_blocks.0.attn.to_q.lora_B.default.weight", &b),
        (
            "transformer_blocks.0.adaln_proj.linear.lora_A.default.weight",
            &a,
        ),
        (
            "transformer_blocks.0.adaln_proj.linear.lora_B.default.weight",
            &b,
        ),
    ];
    Array::save_safetensors(entries, None, &path).expect("write");

    let mut dit = tiny_dit(&cfg);
    let err = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("an unmatched target must fail the install");
    let msg = err.to_string();
    // The offending stem is surfaced by name (actionability); the classification itself is the
    // typed contract — asserted directly, never by pinning message wording (sc-26688).
    assert!(msg.contains("adaln_proj"), "got {msg}");
    use mlx_gen_minimax_h3::{classify_unmatched_target, AdaLnSurface, UnmatchedTargetClass};
    assert_eq!(
        classify_unmatched_target(
            "transformer_blocks.0.adaln_proj.linear",
            AdaLnSurface::FullTimestepMlp
        ),
        UnmatchedTargetClass::DeliberatelyExcluded,
        "the stray AdaLN projection is a real module the surface deliberately excludes"
    );
}

/// `supports_lokr` is **reachable**, not merely declared: a genuine LoKr file (`networkType=lokr`
/// plus `lokr_w1`/`lokr_w2` factors) routes to the shared LyCORIS seam on this same host and
/// installs. Routing is by the FILE, which is what keeps the turbo LoRAs — an `alpha` string with no
/// `networkType` — off `parse_rank_alpha` and its 128x fold.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn a_genuine_lokr_routes_to_the_shared_seam_and_the_turbo_files_do_not() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("lokr.safetensors");
    // `kron(w1 [a,c], w2 [b,d])` reshapes to the base's `[96, 64]`: a·b = 8·12, c·d = 8·8.
    let w1 = tensor(&[8, 8], 1.0);
    let w2 = tensor(&[12, 8], 2.0);
    let meta: HashMap<String, String> = [
        ("networkType".to_string(), "lokr".to_string()),
        ("rank".to_string(), "8".to_string()),
        ("alpha".to_string(), "8".to_string()),
    ]
    .into_iter()
    .collect();
    Array::save_safetensors(
        vec![
            ("transformer_blocks.0.attn.to_q.lokr_w1", &w1),
            ("transformer_blocks.0.attn.to_q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .expect("write the LoKr");

    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)]).expect("install the LoKr");
    assert_eq!(report.applied, 1);
    assert!(report.unmatched_paths.is_empty());
    let segs: Vec<&str> = PROBE.split('.').collect();
    assert_eq!(dit.adaptable_mut(&segs).unwrap().adapters().len(), 1);

    // …and every published turbo file must NOT classify as LoKr: a `networkType` stamp or a
    // `lokr_*` key would route it to `parse_rank_alpha`, where its missing `rank` defaults to 1.0.
    let turbo = write_lora(dir.path(), "turbo.safetensors", &cfg, Some("8"));
    let w = Weights::from_file(&turbo).unwrap();
    assert!(
        w.metadata("networkType").is_none(),
        "a turbo LoRA carries no networkType stamp"
    );
    assert!(
        !w.keys().any(|k| k.contains(".lokr_w")),
        "a turbo LoRA carries no LoKr factors"
    );
    assert!(
        w.metadata("rank").is_none(),
        "…and no `rank` key, which is exactly why parse_rank_alpha would default it to 1.0"
    );
}

/// An exact H3 trainer namespace claim owns the whole file: a valid LoKr factor must not be allowed
/// to route around trainer validation and make the malformed mixed file look successfully applied.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn h3_trainer_metadata_cannot_route_mixed_lokr_factors_around_validation() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("mixed-h3-trainer-lokr.safetensors");
    let w1 = tensor(&[8, 8], 1.0);
    let w2 = tensor(&[12, 8], 2.0);
    let meta: HashMap<String, String> = [
        (
            "ss_network_module".to_string(),
            "networks.lora_minimax_h3".to_string(),
        ),
        ("ss_h3_lora_token_refiner".to_string(), "False".to_string()),
        ("rank".to_string(), "8".to_string()),
        ("alpha".to_string(), "8".to_string()),
    ]
    .into_iter()
    .collect();
    Array::save_safetensors(
        vec![
            ("transformer_blocks.0.attn.to_q.lokr_w1", &w1),
            ("transformer_blocks.0.attn.to_q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .expect("write the mixed adapter");

    let mut dit = tiny_dit(&cfg);
    let err = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0)])
        .expect_err("mixed H3 trainer and LoKr namespaces must fail before LoKr application");
    let msg = err.to_string();
    assert!(msg.contains("networks.lora_minimax_h3"), "got {msg}");
    assert!(msg.contains("LoKr"), "got {msg}");
    assert!(msg.contains("mixed"), "got {msg}");
}

/// A file that matches nothing at all names the key space it expected — **per file**, including when
/// it rides alongside a file that matched everything.
///
/// The zero-match check must be per-spec, not an aggregate `report.applied == 0` over the whole
/// list: with an aggregate, `[good, junk]` returns `Ok` and the junk file is silently ignored. The
/// hole is specifically files carrying **no recognized LoRA suffix at all** — a merged LoRA, a base
/// checkpoint, a textual inversion. A wrong-model file whose keys *do* end in `.lora_A.weight` is
/// caught either way, by the unmatched-target guard.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn a_file_matching_nothing_names_the_expected_key_space() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let path = dir.path().join("empty.safetensors");
    let t = tensor(&[2, 2], 1.0);
    // No recognized LoRA suffix at all, so this contributes neither an `applied` nor an
    // `unmatched_path` — only a per-spec count can see it.
    Array::save_safetensors(vec![("some.base.weight", &t)], None, &path).expect("write");
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(path.clone(), 1.0)])
        .expect_err("must fail")
        .to_string();
    assert!(msg.contains("no target modules matched"), "got {msg}");
    assert!(msg.contains("lora_A.default.weight"), "got {msg}");
    assert!(msg.contains("empty.safetensors"), "got {msg}");

    // Two specs, junk SECOND: the good file folds 24 modules, and an aggregate check would return
    // Ok(applied = 24) right here.
    let good = write_lora(dir.path(), "good.safetensors", &cfg, Some("8"));
    assert_eq!(adapter_target_paths(&cfg).len(), 24);
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(
        &mut dit,
        &[spec(good.clone(), 1.0), spec(path.clone(), 1.0)],
    )
    .expect_err("a junk file riding alongside a good one must still fail")
    .to_string();
    assert!(msg.contains("no target modules matched"), "got {msg}");
    assert!(
        msg.contains("empty.safetensors") && !msg.contains("good.safetensors"),
        "the error must name the OFFENDING file, not the list; got {msg}"
    );

    // …and junk FIRST, so the check cannot be an end-of-loop artifact of spec order.
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(path, 1.0), spec(good.clone(), 1.0)])
        .expect_err("junk first must fail too")
        .to_string();
    assert!(msg.contains("empty.safetensors"), "got {msg}");

    // The control: two GOOD files stack without error, so the per-spec check is not simply rejecting
    // every multi-spec install.
    let good2 = write_lora(dir.path(), "good2.safetensors", &cfg, Some("128"));
    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(good, 1.0), spec(good2, 1.0)])
        .expect("two good files must stack");
    assert_eq!(report.applied, 2 * adapter_target_paths(&cfg).len());
    let segs: Vec<&str> = PROBE.split('.').collect();
    assert_eq!(dit.adaptable_mut(&segs).unwrap().adapters().len(), 2);
}

/// **sc-23402 — every adapter file is forced and GPU-verified at its own load boundary.**
///
/// `Weights::from_file` leaves an adapter's factors lazy `Load` buffers on the CPU stream, and
/// their first Metal consumer is the residual inside denoise step 1: the sc-22414 seam, in its
/// silent shape. A stale all-zero `a`/`b` folds a **zero** residual, so the render comes out at base
/// strength with no refusal, no degeneracy trip and no unmatched-target error. Adapters are
/// installed in `MiniMaxH3::load_task_dit` for both denoise paths, `ref2va` included, so this is the
/// same route every other load sc-23402 verified.
///
/// The count is the **return value of** the verification (`verify_adapter_file` returns what it
/// verified), not a separate tally, and the verification runs before the file is classified,
/// converted or folded — hence the ComfyUI arm, whose *source* file is what must be verified.
///
/// MUTATION: delete `report.verified_source_tensors += verify_adapter_file(&w)?;` from
/// `apply_minimax_h3_adapters` — every arm below goes to 0 and reds.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn every_adapter_file_is_forced_and_gpu_verified_at_its_load_boundary() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let per_file = 2 * adapter_target_paths(&cfg).len();
    assert_eq!(per_file, 48, "24 modules, an `a`/`b` pair each");

    let one = write_lora(dir.path(), "verify_one.safetensors", &cfg, Some("8"));
    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(one.clone(), 1.0)]).expect("install");
    assert_eq!(
        report.verified_source_tensors, per_file,
        "the WHOLE adapter file is the read set, not only the factors that folded"
    );

    // Summed per spec: the second file is verified at its own boundary rather than skipped because
    // the first one passed.
    let two = write_lora(dir.path(), "verify_two.safetensors", &cfg, Some("128"));
    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(one, 1.0), spec(two, 1.0)])
        .expect("install both");
    assert_eq!(report.verified_source_tensors, 2 * per_file);

    // A file that goes through a key-space conversion is verified as it was READ — the converted
    // factors are graph nodes over these buffers, so the source is where the cold `Load` is.
    let (qkv, fc1) = twin_factors(&cfg);
    let comfy = write_comfyui_twin(
        dir.path(),
        "verify_comfy.safetensors",
        &cfg,
        FusedQkvSpec::at(24.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    let source_tensors = Weights::from_file(&comfy).expect("read back").len();
    let mut dit = tiny_dit(&cfg);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(comfy, 1.0)]).expect("install");
    assert_eq!(report.converted_from_comfyui, 1, "the conversion arm ran");
    assert_eq!(report.verified_source_tensors, source_tensors);
}

/// A PEFT `lora_adapter_metadata` blob whose `r` disagrees with the factor shapes is a **hard
/// error**, exactly like a disagreeing `__metadata__["rank"]` string — never a silent override.
///
/// The shared loader takes `cfg_rank.unwrap_or(factor_rank)`, so a `{"r": 8}` blob over rank-128
/// factors folds at `8/8 = 1.0` instead of `8/128 = 0.0625`: the same 16× overshoot this module
/// exists to close, arriving through a different door. PEFT writes `r` equal to the factor rank, so
/// the consistent arm below shows the check rejects nothing legitimate.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn a_peft_blob_rank_disagreeing_with_the_factors_is_refused() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");

    let bad = write_lora_with_meta(
        dir.path(),
        "blob_r8.safetensors",
        &cfg,
        None,
        &[(
            "lora_adapter_metadata",
            r#"{"r": 8, "lora_alpha": 8, "peft_type": "LORA"}"#,
        )],
    );
    let mut dit = tiny_dit(&cfg);
    let msg = apply_minimax_h3_adapters(&mut dit, &[spec(bad, 1.0)])
        .expect_err("a blob rank that disagrees with the factors must be refused")
        .to_string();
    assert!(msg.contains("lora_adapter_metadata"), "got {msg}");
    assert!(msg.contains("declares rank 8"), "got {msg}");
    assert!(msg.contains("rank 128"), "got {msg}");
    assert!(msg.contains("shapes are authoritative"), "got {msg}");

    // A CONSISTENT blob is still honored, and still supplies the alpha. `lora_alpha` is 128 here,
    // NOT 8, so the arm cannot pass by falling through to `DEFAULT_LORA_ALPHA`: it must equal the
    // top-level `alpha = "128"` sibling (fold 1.0) and differ 16x from the `alpha = "8"` one.
    // Compared against those sibling files rather than against a recomputed scalar.
    let x = tensor(&[3, cfg.hidden_size], 0.7);
    let blob = write_lora_with_meta(
        dir.path(),
        "blob_r128.safetensors",
        &cfg,
        None,
        &[("lora_adapter_metadata", r#"{"r": 128, "lora_alpha": 128}"#)],
    );
    let top = write_lora(dir.path(), "top_alpha128.safetensors", &cfg, Some("128"));
    let r_blob = probe_residual(&cfg, &blob, 1.0, &x);
    let r_top = probe_residual(&cfg, &top, 1.0, &x);
    let drift = max_abs(&subtract(&r_blob, &r_top).unwrap()) / max_abs(&r_top);
    println!("[peft blob] rel-max-abs vs top-level alpha=128 = {drift:.3e}");
    assert!(
        drift < 1e-6,
        "a consistent PEFT blob must fold at alpha 128 / rank 128 like the top-level stamp; got \
         {drift:.3e}"
    );
    let default_fallback = write_lora(dir.path(), "no_alpha.safetensors", &cfg, None);
    let ratio = max_abs(&probe_residual(&cfg, &default_fallback, 1.0, &x)) / max_abs(&r_blob);
    assert!(
        (ratio - 0.0625).abs() < 1e-5,
        "…and the blob alpha must WIN over DEFAULT_LORA_ALPHA, which would have given 1.0; got \
         {ratio}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED turbo_lora::a_peft_blob_rank_disagreeing_with_the_factors_is_refused");
}

/// The conversion renames every ComfyUI module, the trunk container included, and the report says it
/// ran — a diffusers file must NOT be counted.
#[test]
#[ignore = "complete synthetic model/block-stack execution; dispatched only by the manual H3 native suite"]
fn the_conversion_leaves_no_comfyui_module_behind() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);
    let comfy = write_comfyui_twin(
        dir.path(),
        "c.safetensors",
        &cfg,
        FusedQkvSpec::at(24.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );

    let w = Weights::from_file(&comfy).unwrap();
    assert!(is_comfyui_key_space(&w));
    let converted = convert_comfyui_key_space(&w).unwrap();
    assert!(
        !is_comfyui_key_space(&converted),
        "the converted map must carry no ComfyUI module name at all"
    );
    assert!(converted
        .keys()
        .all(|k| k.starts_with("transformer_blocks.") || k.starts_with("token_refiner.")));

    let mut dit = tiny_dit(&cfg);
    let r = apply_minimax_h3_adapters(&mut dit, &[spec(comfy, 1.0)]).expect("install");
    assert_eq!(r.converted_from_comfyui, 1);

    let plain = write_lora(dir.path(), "plain.safetensors", &cfg, Some("8"));
    let mut dit = tiny_dit(&cfg);
    let r = apply_minimax_h3_adapters(&mut dit, &[spec(plain, 1.0)]).expect("install");
    assert_eq!(
        r.converted_from_comfyui, 0,
        "a diffusers file must not go through the conversion"
    );
}

/// **The real-weight gate**: every published turbo file's 624 tensors collapse to 312 modules, all
/// of which are MiniMax-H3 DiT targets — 0 unmatched — and each file's alpha resolves to the value
/// it actually ships.
///
/// Proves it RAN rather than skipped: the file count is asserted, so an empty or wrong directory
/// fails instead of passing in 0.00s.
#[test]
#[ignore = "needs the published lightx2v/Minimax-h3-Turbo files (MINIMAX_H3_TURBO_LORA)"]
fn published_turbo_files_resolve_to_dit_targets_at_the_measured_alphas() {
    let dir = turbo_dir();
    let cfg = MiniMaxH3DitConfig::default();
    let targets: std::collections::BTreeSet<String> =
        adapter_target_paths(&cfg).into_iter().collect();
    assert_eq!(targets.len(), 312);

    // file stem → (expected alpha, whether the file DECLARES it). Measured off the published
    // headers.
    //
    // **The `declared` flag is what makes three of these four arms mean anything.** Three published
    // files resolve to `8.0`, which *is* `DEFAULT_LORA_ALPHA` — so `alpha == 8.0` passes whether
    // resolution read the header or fell straight through to the fallback, and the arms would be
    // vacuous on their own. The published alphas cannot be changed to non-default values without
    // making the fixture a lie about real files, so provenance is asserted instead: a `declared`
    // file must carry a parseable `__metadata__["alpha"]` that equals the resolved value, and the
    // undeclared one must carry no such key at all. Between the two claims, an implementation that
    // ignored the header and always returned the default fails every `declared` arm.
    let expected: [(&str, f32, bool); 4] = [
        ("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16", 128.0, true),
        ("minimax_h3_fl2v_turbo_8step_v1.0_bf16", 8.0, true),
        ("minimax_h3_ref2v_turbo_4step_v0.1_bf16", 8.0, true),
        // Ships NO alpha — resolves through DEFAULT_LORA_ALPHA.
        (
            "minimax_h3_fl2v_turbo_4step_v0.1",
            DEFAULT_LORA_ALPHA,
            false,
        ),
    ];
    let comfy = [
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16",
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16",
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16",
    ];

    let mut examined = 0usize;
    for (stem, want_alpha, declared) in expected {
        let path = dir.join(format!("{stem}.safetensors"));
        assert!(path.is_file(), "missing {}", path.display());
        let w = Weights::from_file(&path).unwrap_or_else(|e| panic!("read {stem}: {e}"));
        assert!(
            !is_comfyui_key_space(&w),
            "{stem} must be the diffusers key space"
        );
        assert_eq!(w.len(), 624, "{stem} tensor count");

        let alpha = resolve_alpha(&w).unwrap_or_else(|e| panic!("{stem} alpha: {e}"));
        assert_eq!(alpha, want_alpha, "{stem} alpha");
        // Provenance, so the three `8.0` arms are not just re-asserting `DEFAULT_LORA_ALPHA`.
        let raw = w.metadata("alpha");
        assert_eq!(
            raw.is_some(),
            declared,
            "{stem}: expected __metadata__[\"alpha\"] to be {}, got {raw:?}",
            if declared { "present" } else { "absent" }
        );
        match raw {
            Some(s) => assert_eq!(
                s.trim().parse::<f32>().ok(),
                Some(want_alpha),
                "{stem}: the resolved alpha must come FROM the header, not from the fallback"
            ),
            None => assert_eq!(
                alpha, DEFAULT_LORA_ALPHA,
                "{stem} declares no alpha, so it must resolve through DEFAULT_LORA_ALPHA"
            ),
        }
        // No published file carries a per-target `.alpha` tensor or a PEFT blob — the two higher
        // links of the chain — which is exactly why the `__metadata__` route has to work.
        assert!(
            !w.keys().any(|k| k.ends_with(".alpha")),
            "{stem}: no published file carries an in-band .alpha"
        );
        assert!(
            w.metadata("lora_adapter_metadata").is_none(),
            "{stem}: no published file carries a PEFT blob"
        );
        assert!(
            w.metadata(mlx_gen_minimax_h3::adapters::RANK_METADATA_KEY)
                .is_none(),
            "{stem}: no published file carries a rank key, so the rank cross-check cannot be \
             relied on to catch an alpha error"
        );

        // Collapse the 624 factor keys to module paths through the real suffix table, and check each
        // against the DiT's target surface.
        let mut modules = std::collections::BTreeSet::new();
        let mut ranks = std::collections::BTreeSet::new();
        for key in w.keys() {
            let Some(stem_path) = key
                .strip_suffix(".lora_A.default.weight")
                .or_else(|| key.strip_suffix(".lora_B.default.weight"))
            else {
                panic!("{stem}: unexpected key spelling {key}");
            };
            if key.ends_with(".lora_A.default.weight") {
                let a = w.require(key).unwrap();
                ranks.insert(resolve_rank(stem_path, a, None).unwrap() as i32);
            }
            modules.insert(mlx_gen_minimax_h3::adapters::normalize_minimax_h3_key(
                stem_path,
            ));
        }
        assert_eq!(modules.len(), 312, "{stem} module count");
        assert_eq!(
            ranks,
            std::collections::BTreeSet::from([128]),
            "{stem} rank, derived from the lora_A shapes"
        );
        let unmatched: Vec<&String> = modules.difference(&targets).collect();
        let fold = alpha_rank_fold(alpha, 128.0);
        println!(
            "[turbo lora] {stem}: 624 tensors -> {} modules, {} unmatched, alpha {alpha} / rank 128 \
             => fold {fold}",
            modules.len(),
            unmatched.len()
        );
        assert!(
            unmatched.is_empty(),
            "{stem}: {} module(s) would fold onto nothing: {unmatched:?}",
            unmatched.len()
        );
        assert_eq!(
            modules
                .iter()
                .filter(|m| m.starts_with("token_refiner."))
                .count(),
            12,
            "{stem}: the refiner's 24 tensors must resolve"
        );
        examined += 1;
    }

    for stem in comfy {
        let path = dir.join(format!("{stem}.safetensors"));
        assert!(path.is_file(), "missing {}", path.display());
        let w = Weights::from_file(&path).unwrap_or_else(|e| panic!("read {stem}: {e}"));
        assert!(
            is_comfyui_key_space(&w),
            "{stem} must be detected as the comfyui key space and converted"
        );
        // sc-19443 converts these rather than refusing them. The alpha lives in `__metadata__` on
        // the real files too, so this is the exact route the review found unguarded.
        let converted = convert_comfyui_key_space(&w).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            !is_comfyui_key_space(&converted),
            "{stem}: the conversion must leave no ComfyUI module behind"
        );
        let fused_alpha = resolve_alpha(&w).unwrap_or_else(|e| panic!("{stem} alpha: {e}"));
        let per_target = converted
            .require("transformer_blocks.0.attn.to_q.alpha")
            .unwrap_or_else(|_| panic!("{stem}: no per-target alpha emitted"))
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()[0];
        println!(
            "[turbo lora] {stem}: comfyui key space — converted, fused alpha {fused_alpha} => \
             per-target {per_target}"
        );
        assert_eq!(
            per_target,
            fused_alpha / 3.0,
            "{stem}: these files are block-diagonal, so the resolved alpha must divide by three"
        );
        examined += 1;
    }

    println!("[turbo lora] EXAMINED {examined} published file(s)");
    assert_eq!(examined, 7, "the published set is 4 diffusers + 3 comfyui");
}

/// **The end-to-end gate**: the real 8-step turbo LoRA folds onto the real DiT partition — all 312
/// modules matched, at the measured 0.0625 — and the adapted forward differs from the base one.
///
/// `MINIMAX_H3_TURBO_DIT` points at a `transformer/` directory (any tier: the fold is
/// tier-independent). Budget the disk and the load before running it — **measured**, the `q4`
/// `transformer/` is **~17.5 GiB (18.8 GB)** across 14 shards (1.28–1.49 GB each) against bf16's
/// **~61.7 GiB (66.3 GB)**.
#[test]
#[ignore = "needs a real MiniMax-H3 transformer/ (MINIMAX_H3_TURBO_DIT) + the turbo LoRA + Metal"]
fn the_real_turbo_lora_folds_onto_the_real_dit() {
    let lora = turbo_dir().join("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors");
    assert!(lora.is_file(), "missing {}", lora.display());
    let raw = std::env::var("MINIMAX_H3_TURBO_DIT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_TURBO_DIT must point at a MiniMax-H3 `transformer/` directory"
    );
    let dit_dir = PathBuf::from(raw);

    let mut dit = MiniMaxH3Dit::load_dir(&dit_dir, Dtype::Bfloat16)
        .unwrap_or_else(|e| panic!("load {}: {e}", dit_dir.display()));
    println!(
        "[turbo e2e] loaded {} — {} blocks, hidden {}",
        dit_dir.display(),
        dit.num_layers(),
        dit.config().hidden_size
    );

    // A probe activation through one real block projection, before and after.
    let hidden = dit.config().hidden_size;
    let x = tensor(&[2, hidden], 0.3).as_dtype(Dtype::Bfloat16).unwrap();
    let segs: Vec<&str> = PROBE.split('.').collect();
    let y0 = dit
        .adaptable_mut(&segs)
        .expect("probe module")
        .forward(&x)
        .expect("base forward");
    mlx_rs::transforms::eval([&y0]).expect("force the base forward");

    let report = apply_minimax_h3_adapters(&mut dit, &[spec(lora.clone(), 1.0)])
        .unwrap_or_else(|e| panic!("install {}: {e}", lora.display()));
    println!(
        "[turbo e2e] applied {} module(s), {} unmatched",
        report.applied,
        report.unmatched_paths.len()
    );
    assert_eq!(report.applied, 312, "every published module must fold");
    assert!(report.unmatched_paths.is_empty());

    let probe = dit.adaptable_mut(&segs).expect("probe module");
    println!(
        "[turbo e2e] probe base: shape {:?}, packed={}, {} adapter(s) stacked",
        probe.base_shape(),
        probe.is_quantized(),
        probe.adapters().len()
    );
    assert_eq!(probe.adapters().len(), 1, "exactly one residual per module");
    let y1 = probe.forward(&x).expect("adapted forward");
    let delta = max_abs(&subtract(&y1, &y0).unwrap());
    let base = max_abs(&y0);
    println!("[turbo e2e] probe |Δy| = {delta:.4e} against |y| = {base:.4e}");
    assert!(
        delta > 0.0,
        "the fold must change the forward — a 0 delta means it landed on nothing"
    );

    // The refiner arm, run for real: `embed_context` is `context_embedder` then the whole 2-block
    // token refiner, so this exercises the 24 published tensors that land there — the ones a stubbed
    // refiner would have dropped. `context_embedder` itself is NOT adapted, so any difference here
    // comes from the refiner blocks.
    let context = tensor(&[1, 4, dit.config().text_dim], 1.7)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let refined = dit.embed_context(&context).expect("embed + refine");
    mlx_rs::transforms::eval([&refined]).expect("force the refiner chain");
    let clean = MiniMaxH3Dit::load_dir(&dit_dir, Dtype::Bfloat16).expect("reload");
    let refined_base = clean.embed_context(&context).expect("embed + refine");
    let refiner_delta = max_abs(&subtract(&refined, &refined_base).unwrap());
    println!(
        "[turbo e2e] refiner chain |Δ| = {refiner_delta:.4e} against |base| = {:.4e}",
        max_abs(&refined_base)
    );
    assert!(
        refiner_delta > 0.0,
        "the refiner's 24 tensors must fold — a 0 delta means token_refiner took nothing"
    );
    // Keep `clean` alive to the end so the comparison arm cannot be optimized away.
    assert_eq!(clean.num_layers(), dit.num_layers());

    // The 8-step file declares alpha 8 at rank 128, so it must fold at 0.0625 here too.
    let w = Weights::from_file(&lora).expect("read the LoRA");
    assert_eq!(resolve_alpha(&w).unwrap(), 8.0);
    assert_eq!(alpha_rank_fold(8.0, 128.0), 0.0625);
    println!("[turbo e2e] EXAMINED the real DiT at fold 0.0625");
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED turbo_lora::the_real_turbo_lora_folds_onto_the_real_dit"
    );
}

/// **The sc-18729 render.** One clip, fully measured, written to disk.
///
/// Every knob is an env var so the same binary produces each arm of the comparison in its **own
/// process** — which is not a convenience: MLX's peak counter is process-wide, and two renders in
/// one process would report the larger one's high-water for both.
///
/// ```sh
/// MINIMAX_H3_SNAPSHOT=<upstream root> \
/// MINIMAX_H3_DIT=<tier>/transformer \
/// MINIMAX_H3_TURBO_LORA=<lightx2v/Minimax-h3-Turbo dir> \
/// MINIMAX_H3_RENDER_LORA=minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16 \
/// MINIMAX_H3_RENDER_STEPS=4 MINIMAX_H3_RENDER_SHIFT=6 \
/// MINIMAX_H3_RENDER_WIDTH=1344 MINIMAX_H3_RENDER_HEIGHT=768 \
/// MINIMAX_H3_RENDER_OUT=<dir> MINIMAX_H3_RENDER_LABEL=turbo4-768p \
///   cargo test -p mlx-gen-minimax-h3 --test integration turbo_lora:: -- --ignored --nocapture \
///   --test-threads=1 turbo_render_records_a_measured_clip
/// ```
///
/// `MINIMAX_H3_RENDER_LORA=none` renders the base arm at the same geometry, which is what makes the
/// speedup a measurement rather than arithmetic.
///
/// The artifacts are raw (`frames.rgb`, `audio.f32le`) rather than an encoded container: this crate
/// has no muxer and must not grow one for a validation harness. `receipt.json` carries everything
/// needed to encode them and everything measured.
#[test]
#[ignore = "sc-18729: needs MINIMAX_H3_SNAPSHOT + the turbo LoRA + Metal; the 1344x768 4-step arm is ~15 min, the 50-step base at that canvas ~2 h"]
fn turbo_render_records_a_measured_clip() {
    let root = PathBuf::from(
        render_env("MINIMAX_H3_SNAPSHOT").expect("MINIMAX_H3_SNAPSHOT=<upstream snapshot root>"),
    );
    let dit = render_env("MINIMAX_H3_DIT").map(PathBuf::from);
    let lora_stem = render_env("MINIMAX_H3_RENDER_LORA").expect(
        "MINIMAX_H3_RENDER_LORA=<file stem>, or `none` for the base arm — an unset value would \
         silently render the base and be recorded as turbo",
    );
    let lora = if lora_stem == "none" {
        None
    } else {
        let p = turbo_dir().join(format!("{lora_stem}.safetensors"));
        assert!(p.is_file(), "missing {}", p.display());
        Some(p)
    };
    let steps = render_env_u32("MINIMAX_H3_RENDER_STEPS").unwrap_or(4);
    let shift = render_env_f32("MINIMAX_H3_RENDER_SHIFT");
    let width = render_env_u32("MINIMAX_H3_RENDER_WIDTH").unwrap_or(1344);
    let height = render_env_u32("MINIMAX_H3_RENDER_HEIGHT").unwrap_or(768);
    let frames_req =
        render_env_u32("MINIMAX_H3_RENDER_FRAMES").unwrap_or(SMALLEST_LEGAL_FRAMES as u32);
    let seed = render_env_u32("MINIMAX_H3_RENDER_SEED").unwrap_or(18_729) as u64;
    let label = render_env("MINIMAX_H3_RENDER_LABEL").unwrap_or_else(|| "render".into());
    let out_dir = PathBuf::from(
        render_env("MINIMAX_H3_RENDER_OUT").expect("MINIMAX_H3_RENDER_OUT=<output dir>"),
    );
    std::fs::create_dir_all(&out_dir).expect("create the output dir");

    // A prompt with picture AND sound in it — the model is a joint AV generator and an A/B that
    // only looks at pixels cannot speak to half of what distillation might have cost.
    let prompt = render_env("MINIMAX_H3_RENDER_PROMPT").unwrap_or_else(|| {
        "a lighthouse on a rocky coast at dusk, waves breaking against the rocks, seagulls calling"
            .into()
    });

    println!(
        "\n[render] {label}: {width}x{height} / {frames_req} frames / {steps} steps / video shift \
         {} / seed {seed}\n[render]   root {}\n[render]   dit  {}\n[render]   lora {}",
        shift.map_or("12 (default)".to_string(), |s| s.to_string()),
        root.display(),
        dit.as_ref().map_or("<flat root/transformer>".into(), |d| d
            .display()
            .to_string()),
        lora.as_ref()
            .map_or("<none — base arm>".into(), |l| l.display().to_string()),
    );

    let r = render_once(&RenderRecipe {
        root: &root,
        dit: dit.as_deref(),
        lora: lora.as_deref(),
        steps,
        shift,
        width,
        height,
        frames: frames_req,
        seed,
        prompt: &prompt,
    });

    // --- evidence the render happened ---------------------------------------------------------
    assert_eq!(r.frames.len(), frames_req as usize, "decoded frame count");
    assert_eq!(r.steps_seen, steps, "one progress tick per evaluation");
    assert_eq!(r.fps, 24);
    for (i, f) in r.frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (width, height), "frame {i} size");
        assert_eq!(f.pixels.len(), (width * height * 3) as usize, "frame {i}");
    }
    assert!(
        r.peak_bytes > 1_000_000_000,
        "MLX peak {} B is too small for a real 33 B forward — nothing was materialized",
        r.peak_bytes
    );
    let (mean_sd, motion) = clip_stats(&r.frames);
    let rms = {
        let s: f64 = r
            .audio
            .samples
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        (s / r.audio.samples.len() as f64).sqrt()
    };
    assert!(
        r.audio.samples.iter().all(|s| s.is_finite()),
        "the soundtrack carries NaN/Inf"
    );

    // --- the artifacts ------------------------------------------------------------------------
    let rgb_path = out_dir.join(format!("{label}.frames.rgb"));
    let pcm_path = out_dir.join(format!("{label}.audio.f32le"));
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(&rgb_path).expect("create the frame file"),
        );
        for frame in &r.frames {
            f.write_all(&frame.pixels).expect("write frames");
        }
        f.flush().expect("flush frames");
        let mut a =
            std::io::BufWriter::new(std::fs::File::create(&pcm_path).expect("create the pcm file"));
        for s in &r.audio.samples {
            a.write_all(&s.to_le_bytes()).expect("write pcm");
        }
        a.flush().expect("flush pcm");
    }

    let per_channel = r.audio.samples.len() / usize::from(r.audio.channels);
    let audio_seconds = per_channel as f64 / f64::from(r.audio.sample_rate);
    let video_seconds = f64::from(frames_req) / f64::from(r.fps);
    let intervals = r
        .step_intervals_s
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    let s_per_step = r.s_per_step.map_or("null".into(), |v| format!("{v:.4}"));
    let receipt = format!(
        "{{\n  \"label\": \"{label}\",\n  \"width\": {width},\n  \"height\": {height},\n  \
         \"frames\": {frames_req},\n  \"fps\": {},\n  \"steps\": {steps},\n  \
         \"video_shift\": {},\n  \"audio_shift\": 3.0,\n  \"seed\": {seed},\n  \
         \"lora\": \"{lora_stem}\",\n  \"tier_dir\": \"{}\",\n  \
         \"wall_s\": {:.3},\n  \"setup_and_first_eval_s\": {:.3},\n  \
         \"denoise_tail_s\": {:.3},\n  \"denoise_tail_evals\": {},\n  \
         \"decode_s\": {:.3},\n  \"s_per_step\": {s_per_step},\n  \
         \"step_intervals_s\": [{intervals}],\n  \"peak_bytes\": {},\n  \
         \"peak_gb\": {:.3},\n  \"mean_frame_stddev\": {mean_sd:.4},\n  \
         \"inter_frame_motion\": {motion:.4},\n  \"audio_rms\": {rms:.6},\n  \
         \"audio_seconds\": {audio_seconds:.5},\n  \"video_seconds\": {video_seconds:.5},\n  \
         \"sample_rate\": {},\n  \"channels\": {},\n  \"prompt\": \"{}\"\n}}\n",
        r.fps,
        shift.unwrap_or(12.0),
        dit.as_ref()
            .map_or(String::new(), |d| d.display().to_string()),
        r.wall_s,
        r.setup_and_first_eval_s,
        r.denoise_tail_s,
        r.step_intervals_s.len(),
        r.decode_s,
        r.peak_bytes,
        r.peak_bytes as f64 / 1e9,
        r.audio.sample_rate,
        r.audio.channels,
        prompt.replace('"', "'"),
    );
    let receipt_path = out_dir.join(format!("{label}.receipt.json"));
    std::fs::write(&receipt_path, &receipt).expect("write the receipt");

    println!(
        "\n[render] {label} DONE\n  wall {:.1} s = (setup + eval 1) {:.1} + {} further eval(s) \
         {:.1} + decode {:.1}\n  s/step {} over {} interval(s): [{intervals}]\n  MLX peak {:.2} \
         GB\n  frame stddev {mean_sd:.2}, inter-frame motion {motion:.3}, audio rms {rms:.4} \
         ({audio_seconds:.4} s vs {video_seconds:.4} s of picture)\n  {}\n  {}\n  {}",
        r.wall_s,
        r.setup_and_first_eval_s,
        r.step_intervals_s.len(),
        r.denoise_tail_s,
        r.decode_s,
        s_per_step,
        r.step_intervals_s.len(),
        r.peak_bytes as f64 / 1e9,
        rgb_path.display(),
        pcm_path.display(),
        receipt_path.display(),
    );

    // Structure gates last, so the artifacts and the receipt survive a failing one and can be
    // inspected. A turbo arm that renders mush is a *result*, not a lost run.
    assert!(
        mean_sd > 8.0,
        "{label}: frames are nearly flat (stddev {mean_sd:.2}) — the render collapsed"
    );
    assert!(
        motion > 0.05,
        "{label}: frozen clip (inter-frame motion {motion:.4})"
    );
    assert!(
        motion < 60.0,
        "{label}: per-frame noise, not video (inter-frame motion {motion:.2})"
    );
    assert!(
        rms > 1e-4,
        "{label}: the soundtrack is silent (rms {rms:.3e})"
    );
}

/// Header and conversion receipt for one exact `networks.lora_minimax_h3` artifact.
///
/// Run this only with the supplied proprietary file:
///
/// ```text
/// MINIMAX_H3_TRAINER_LORA=/absolute/path/adapter.safetensors \
///   cargo test -p mlx-gen-minimax-h3 --test integration \
///   turbo_lora::exact_h3_trainer_file_receipt -- --ignored --nocapture
/// ```
///
/// This is deliberately **not** a substitute for the full-model MLX/Metal receipt: that separate
/// run must install this same digest into a real 50-block transformer and exercise every target.
#[test]
#[ignore = "needs MINIMAX_H3_TRAINER_LORA=<exact trainer safetensors>; run with --ignored"]
fn exact_h3_trainer_file_receipt() {
    let path = trainer_lora_path();
    let bytes = std::fs::metadata(&path)
        .unwrap_or_else(|error| panic!("stat {}: {error}", path.display()))
        .len();
    let sha256 = sha256_of(&path);
    assert_optional_trainer_artifact_identity(&path, bytes, &sha256);

    let adapter = Weights::from_file(&path)
        .unwrap_or_else(|error| panic!("read exact trainer file {}: {error}", path.display()));
    let (converted, layout, mut alphas) = convert_minimax_h3_trainer_key_space(&adapter)
        .unwrap_or_else(|error| panic!("validate exact trainer file {}: {error}", path.display()));
    assert_eq!(layout.source_targets, 200, "50 blocks × four source leaves");
    assert_eq!(layout.ranks, vec![16], "unique source factor ranks");
    assert!(layout.trunk_only, "the exact trainer export is trunk-only");
    alphas.sort_by(f32::total_cmp);
    alphas.dedup_by(|left, right| *left == *right);
    assert_eq!(alphas, vec![16.0], "unique per-target alphas");

    let runtime_targets = adapter_target_paths(&MiniMaxH3DitConfig::default())
        .into_iter()
        .filter(|target| target.starts_with("transformer_blocks."))
        .collect::<BTreeSet<_>>();
    assert_eq!(runtime_targets.len(), 300, "50 blocks × six runtime leaves");
    let runtime_modules = converted
        .keys()
        .map(|key| {
            converted_lora_module(key)
                .unwrap_or_else(|| panic!("trainer conversion emitted unexpected key {key}"))
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(converted.len(), 900, "300 runtime leaves × down/up/alpha");
    assert_eq!(
        runtime_modules.len(),
        300,
        "fused QKV expands 200 source targets to 300 leaves"
    );
    let unmatched = runtime_modules
        .difference(&runtime_targets)
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        unmatched.is_empty(),
        "unmatched runtime targets: {unmatched:?}"
    );

    println!(
        "SC21028_TRAINER_RECEIPT backend=mlx file={} bytes={bytes} sha256={sha256} \
         source_targets={} runtime_leaves={} unmatched={} unique_ranks={:?} unique_alphas={:?} \
         parser_layout_only=true",
        path.display(),
        layout.source_targets,
        runtime_modules.len(),
        unmatched.len(),
        layout.ranks,
        alphas,
    );
    println!(
        "SC21028_TRAINER_RUNTIME_RECEIPT_REQUIRED backend=mlx full_model=MINIMAX_H3 transformer \
         blocks=50 accelerator=Metal status=not_run"
    );
}

/// Manual MLX/Metal receipt for the exact SC-21028 trainer artifact against a real 50-block DiT.
///
/// This intentionally probes one real projection rather than rendering a clip: installation walks
/// and audits all 300 trunk projections, while the projection forward proves the installed factors
/// execute numerically on Metal. The change gate is relative max-abs-diff, never shape or cosine.
///
/// ```text
/// MINIMAX_H3_TRAINER_LORA=/absolute/path/deepthroat_v02.safetensors \
/// MINIMAX_H3_DIT=/absolute/path/to/a/real/tier/transformer \
/// cargo test --release --locked -p mlx-gen-minimax-h3 --test integration \
///   turbo_lora::manual_metal_exact_h3_trainer_runtime_receipt -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore = "manual MLX/Metal real-weight receipt; needs exact trainer LoRA and a 50-block DiT"]
fn manual_metal_exact_h3_trainer_runtime_receipt() {
    let (runtime_device, runtime_device_index) = require_default_mlx_metal_device();
    let lora = trainer_lora_path();
    let bytes = std::fs::metadata(&lora)
        .unwrap_or_else(|error| panic!("stat {}: {error}", lora.display()))
        .len();
    let sha256 = sha256_of(&lora);
    assert_sc21028_trainer_artifact_identity(&lora, bytes, &sha256);

    let dit_dir = trainer_dit_path();
    let config_path = dit_dir.join("config.json");
    let config_text = mlx_gen::gen_core::__model_input_reader!(
std::fs::read_to_string(&config_path),
"crates/media/mlx-gen/mlx-gen-minimax-h3/tests/turbo_lora.rs::manual_metal_exact_h3_trainer_runtime_receipt::std::fs::read_to_string@111922",
Configuration,
&config_path,
|__model_input_path| std::fs::read_to_string(__model_input_path)
,
origin = mlx_gen::gen_core::model_input_proof::SourceOrigin::ExternalSelected
)
        .unwrap_or_else(|error| panic!("read {}: {error}", config_path.display()));
    let config_sha256 = sha256_of(&config_path);
    let cfg = MiniMaxH3DitConfig::from_diffusers_json(&config_text)
        .expect("parse the real MiniMax-H3 transformer config");
    assert_eq!(
        cfg,
        MiniMaxH3DitConfig::default(),
        "the runtime receipt requires the published 50-block MiniMax-H3 geometry"
    );

    let tier = match mlx_gen::quant::packed_quant_bits_at(&dit_dir)
        .unwrap_or_else(|error| panic!("read staged DiT tier {}: {error}", dit_dir.display()))
    {
        Some(bits) => format!("q{bits}"),
        None => "bf16".to_owned(),
    };
    let mut dit = MiniMaxH3Dit::load_dir(&dit_dir, Dtype::Bfloat16)
        .unwrap_or_else(|error| panic!("load real DiT {}: {error}", dit_dir.display()));
    assert_eq!(
        dit.num_layers(),
        50,
        "the real DiT must contain all 50 blocks"
    );

    let probe = "transformer_blocks.0.attn.to_q";
    let probe_segments = probe.split('.').collect::<Vec<_>>();
    let x = tensor(&[1, 3, cfg.hidden_size], 0.21028)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let (probe_packed, base_native) = {
        let projection = dit
            .adaptable_mut(&probe_segments)
            .expect("resolve real probe projection before install");
        (
            projection.is_quantized(),
            projection
                .forward(&x)
                .expect("real base projection forward"),
        )
    };
    let base = base_native.as_dtype(Dtype::Float32).unwrap();
    let base_peak = max_abs(&base);
    assert!(
        base_peak.is_finite() && base_peak > 1e-6,
        "real base projection is non-finite or vacuous: max|y|={base_peak:.3e}"
    );

    let report = apply_minimax_h3_adapters(
        &mut dit,
        &[spec(lora.clone(), SC21028_RUNTIME_ADAPTER_SCALE)],
    )
    .expect("install the exact trainer LoRA into the real 50-block DiT");
    assert_eq!(report.applied, 300, "50 blocks × six runtime leaves");
    assert!(report.unmatched_paths.is_empty(), "zero unmatched targets");
    assert_eq!(report.converted_from_trainer, 1, "exact trainer namespace");
    assert_eq!(
        report.trainer_source_targets_applied, 200,
        "50 × four source leaves"
    );
    assert_eq!(report.trainer_ranks, vec![16], "exact source rank");
    assert_eq!(report.trainer_alphas, vec![16.0], "exact per-target alpha");

    let mut adapted_paths = Vec::new();
    for path in adapter_target_paths(&cfg) {
        let segments = path.split('.').collect::<Vec<_>>();
        let linear = dit
            .adaptable_mut(&segments)
            .unwrap_or_else(|| panic!("declared adapter target {path} does not resolve"));
        if !linear.adapters().is_empty() {
            adapted_paths.push(path);
        }
    }
    assert_eq!(
        adapted_paths.len(),
        300,
        "every trunk runtime leaf is adapted"
    );
    assert!(
        adapted_paths
            .iter()
            .all(|path| path.starts_with("transformer_blocks.")),
        "the exact trunk-only artifact must not fabricate token-refiner targets"
    );

    let adapted_native = dit
        .adaptable_mut(&probe_segments)
        .expect("resolve real probe projection after install")
        .forward(&x)
        .expect("real adapted projection forward");
    let observed_delta = subtract(&adapted_native, &base_native)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let independent_residual = exact_trainer_q_probe_residual(&lora, &x, &cfg);
    // The production linear narrows the residual to the base output dtype before adding it. Build
    // the oracle's expected *observed* delta through the same public dtype boundary so BF16 add
    // rounding does not masquerade as a conversion error. QKV selection/orientation and the
    // non-unit runtime scale above remain independent; non-identity alpha/rank coverage belongs to
    // `trainer_namespace_converts_to_independent_expected_factors_at_relative_max_abs` because this
    // exact artifact's alpha/rank is 16/16 = 1.
    let expected_adapted = base_native
        .add(independent_residual.as_dtype(base_native.dtype()).unwrap())
        .unwrap();
    let expected_delta = subtract(&expected_adapted, &base_native)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let expected_peak = max_abs(&expected_delta);
    let relative_max_abs_diff = expected_peak / base_peak;
    assert!(
        relative_max_abs_diff.is_finite() && relative_max_abs_diff > 1e-6,
        "the exact trainer LoRA did not measurably change the real {probe} forward: \
         relative-max-abs-diff={relative_max_abs_diff:.3e}"
    );
    let residual_correctness_rel_max =
        max_abs(&subtract(&observed_delta, &expected_delta).unwrap()) / expected_peak;
    assert!(
        residual_correctness_rel_max.is_finite()
            && residual_correctness_rel_max <= SC21028_RUNTIME_RESIDUAL_REL_MAX,
        "the installed {probe} residual disagrees with independent raw-factor math by \
         relative-max-abs={residual_correctness_rel_max:.3e} (limit \
         {SC21028_RUNTIME_RESIDUAL_REL_MAX:.3e}); wrong QKV slice/orientation or runtime scale \
         (non-identity alpha/rank is covered by the independent trainer mutation fixture)"
    );

    println!(
         "SC21028_TRAINER_RUNTIME_RECEIPT backend=mlx accelerator=Metal file={} bytes={bytes} \
         sha256={sha256} model_id={} dit={} config_sha256={} tier={} probe_packed={} \
         runtime_device={} runtime_device_index={} adapter_scale={} blocks={} source_targets={} applied={} adapted_modules={} \
         unmatched={} unique_ranks={:?} unique_alphas={:?} probe={} relative_max_abs_diff={:.6e}",
        lora.display(),
        mlx_gen_minimax_h3::MODEL_ID,
        dit_dir.display(),
        config_sha256,
        tier,
        probe_packed,
        runtime_device,
        runtime_device_index,
        SC21028_RUNTIME_ADAPTER_SCALE,
        dit.num_layers(),
        report.trainer_source_targets_applied,
        report.applied,
        adapted_paths.len(),
        report.unmatched_paths.len(),
        report.trainer_ranks,
        report.trainer_alphas,
        probe,
        relative_max_abs_diff,
    );
    println!(
        "SC21028_TRAINER_RUNTIME_CORRECTNESS backend=mlx probe={} \
         observed_vs_independent_expected_relative_max_abs={:.6e} limit={:.6e}",
        probe, residual_correctness_rel_max, SC21028_RUNTIME_RESIDUAL_REL_MAX
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED turbo_lora::manual_metal_exact_h3_trainer_runtime_receipt"
    );
}
