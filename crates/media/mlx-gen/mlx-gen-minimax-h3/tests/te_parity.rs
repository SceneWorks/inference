//! sc-17143 — committed-fixture parity for the MiniMax-H3 Qwen3-VL-32B context extraction against
//! the **transformers** `Qwen3VLTextModel` forward (an independent graph), at tiny dims.
//!
//! Exercises bias-less GQA, per-head q/k RMSNorm, HF half-split RoPE, the causal mask and the
//! select-layer capture — the `context` the H3 DiT consumes. The fixture is produced by
//! `tools/dump_minimax_h3_te.py` and committed, so this runs by default.
//!
//! # sc-18741 — there is no template-prefix slice
//!
//! This file used to assert that the port dropped 3 leading rows. The official conditioner
//! (`diffusers.modular_pipelines.minimax_h3.encoders.MiniMaxH3TextEncoderStep`) applies **no chat
//! template and no special tokens**, so the context keeps one row per presentation token. The
//! `presentation_*` tests below pin the *absence* of the template — including against the exact ids
//! the old render produced — rather than merely pinning a token count, which a reintroduced
//! template with a compensating slice could still satisfy.
//!
//! The suite is built around the fact that **an off-by-one in the layer selection still produces a
//! plausible tensor**. Equality against `hidden_states[50]` alone cannot catch it, so the fixture
//! also carries both neighbours and this file asserts *inequality* against them.

mod common;

use common::{assert_parity, rel, std_dev, te_fixture_config, TE_FIXTURE};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{
    MiniMaxH3TeConfig, MiniMaxH3TextEncoder, APPLIES_CHAT_TEMPLATE, MINIMAX_ADDED_SPECIALS,
    NUM_HIDDEN_LAYERS, SELECT_HIDDEN,
};
use mlx_rs::Array;

/// Metal fp32 matmul tolerance, matching the sibling parity harnesses.
const TOL: f32 = 1e-2;

fn load() -> Weights {
    Weights::from_file(TE_FIXTURE)
        .unwrap_or_else(|e| panic!("load te fixture (run tools/dump_minimax_h3_te.py): {e}"))
}

fn encoder(w: &Weights, cfg: &MiniMaxH3TeConfig) -> MiniMaxH3TextEncoder {
    MiniMaxH3TextEncoder::from_weights(w, "language_model", cfg).expect("build encoder")
}

fn run(w: &Weights, cfg: &MiniMaxH3TeConfig) -> Array {
    encoder(w, cfg)
        .forward(
            w.require("in.input_ids").unwrap(),
            w.require("in.attention_mask").unwrap(),
        )
        .expect("forward")
}

/// The headline parity gate: the port reproduces the reference's select-layer context.
#[test]
fn context_matches_the_reference() {
    let w = load();
    let cfg = te_fixture_config();
    let got = run(&w, &cfg);
    let want = w.require("out.context").unwrap();

    assert_eq!(got.shape(), want.shape(), "context shape");
    let (peak, mean) = rel(&got, want);
    println!("H3 TE context parity: peak-rel={peak:.3e} mean-rel={mean:.3e}");
    assert_parity(&got, want, TOL, "h3 te context");
}

/// **The single highest-risk detail in this story.** HF `hidden_states[k]` is the state after `k`
/// layers, so the card's "50th layer" is index 50 = the output of 0-indexed layer 49.
///
/// A fixture that only carries the right answer cannot catch a shift, because a shifted
/// implementation still returns a well-formed tensor of the right shape. This asserts the port is
/// **unequal** to both neighbours by a wide margin, so moving the tap one layer in either direction
/// fails here.
#[test]
fn layer_selection_is_off_by_one_safe_in_both_directions() {
    let w = load();
    let cfg = te_fixture_config();
    let got = run(&w, &cfg);

    // Correct tap.
    assert_parity(&got, w.require("out.context").unwrap(), TOL, "selected");

    // One layer early / one layer late must both be clearly different.
    for (key, direction) in [
        ("out.context_at_3", "one layer EARLY"),
        ("out.context_at_5", "one layer LATE"),
    ] {
        let neighbour = w.require(key).unwrap();
        assert_eq!(got.shape(), neighbour.shape(), "{key} shape");
        let (peak, _) = rel(&got, neighbour);
        println!("  vs {direction} ({key}): peak-rel={peak:.3e}");
        assert!(
            peak > TOL * 10.0,
            "{direction} is only {peak:.3e} away — the fixture cannot gate an off-by-one, so this \
             test would pass against a shifted selection"
        );
    }
}

/// Shifting `select_hidden` by one must actually change the output — the converse of the test
/// above, driven through the config rather than the fixture. This proves the knob is wired, not
/// merely that three reference tensors differ from each other.
#[test]
fn shifting_select_hidden_changes_the_context() {
    let w = load();
    let base = te_fixture_config();

    let got = run(&w, &base);
    for (delta, key) in [(-1i64, "out.context_at_3"), (1, "out.context_at_5")] {
        let mut shifted = base.clone();
        shifted.select_hidden = (base.select_hidden as i64 + delta) as usize;
        let out = run(&w, &shifted);

        // The shifted port must match the corresponding reference neighbour...
        assert_parity(&out, w.require(key).unwrap(), TOL, key);
        // ...and must NOT match the correct answer.
        let (peak, _) = rel(&out, &got);
        assert!(
            peak > TOL * 10.0,
            "select_hidden {delta:+} produced the same context as the correct tap (peak {peak:.3e})"
        );
    }
}

/// Only `select_hidden` layers are ever loaded — the evidence sc-17139's hosting decision rests on.
/// Layers `select_hidden..num_layers`, the final `norm` and `lm_head` are never touched, so they
/// can be trimmed from an uploaded artifact without changing a single output value.
#[test]
fn only_the_selected_layers_are_loaded() {
    let w = load();
    let cfg = te_fixture_config();
    let te = encoder(&w, &cfg);

    assert_eq!(te.num_loaded_layers(), cfg.select_hidden);
    assert_eq!(te.num_loaded_layers(), 4);
    assert!(
        cfg.select_hidden < cfg.num_layers as usize,
        "trim must be real"
    );

    // Scaled to the shipped model: 50 of 64 run, 14 unused.
    let shipped = MiniMaxH3TeConfig::qwen3_vl_32b();
    assert_eq!(shipped.select_hidden, SELECT_HIDDEN);
    assert_eq!(shipped.layers_to_run(), 50);
    assert_eq!(shipped.num_layers, NUM_HIDDEN_LAYERS);
    assert_eq!(shipped.num_layers as usize - shipped.layers_to_run(), 14);

    // The encoder must load without the trimmable tensors present at all: removing layers 4-5, the
    // final norm and lm_head from the fixture must still build and produce identical output.
    let mut trimmed = w.clone();
    for i in cfg.select_hidden..cfg.num_layers as usize {
        trimmed.remove_prefix(&format!("language_model.layers.{i}."));
    }
    trimmed.remove("language_model.norm.weight");
    trimmed.remove("lm_head.weight");
    let out = run(&trimmed, &cfg);
    assert_parity(
        &out,
        w.require("out.context").unwrap(),
        TOL,
        "context from a trimmed checkpoint",
    );
}

/// **sc-18741.** The context keeps ONE ROW PER PRESENTATION TOKEN — nothing is sliced off the
/// front. The reference conditioner returns `outputs.hidden_states[layer]` whole.
///
/// Asserted against the reference tensor's own length rather than a constant, so a slice
/// reintroduced anywhere in the forward shortens the output and fails here.
#[test]
fn context_keeps_every_presentation_row() {
    let w = load();
    let cfg = te_fixture_config();
    let got = run(&w, &cfg);

    let seq = w.require("in.input_ids").unwrap().shape()[1];
    assert_eq!(
        got.shape()[1],
        seq,
        "the context must have one row per input token; a shorter context means a prefix slice \
         came back (sc-18741)"
    );
    assert_eq!(w.require("out.context").unwrap().shape()[1], seq);
}

/// A short prompt must work. Under sc-17143's prefix slice, any prompt of 3 tokens or fewer was a
/// hard error ("must exceed the 3 dropped template-prefix tokens") — a real capability loss that
/// fell out of a slice the reference never performs. `MiniMax-H3` prompts like `"东京的夜景"`
/// tokenize to 4 ids, and `"\ttab start"` to 2.
#[test]
fn short_prompts_are_accepted() {
    let w = load();
    let cfg = te_fixture_config();
    let te = encoder(&w, &cfg);

    for s in [1i32, 2, 3] {
        let ids = Array::from_slice(&vec![0i32; s as usize], &[1, s]);
        let mask = Array::from_slice(&vec![1i32; s as usize], &[1, s]);
        let out = te
            .forward(&ids, &mask)
            .unwrap_or_else(|e| panic!("{s}-token prompt must encode, got: {e}"));
        assert_eq!(out.shape(), &[1, s, cfg.hidden_size], "{s}-token context");
    }
}

/// **Mutation check.** Perturbing any weight the forward depends on must move the output; a parity
/// test that passes against a mutated checkpoint is not testing the math.
#[test]
fn mutating_a_weight_changes_the_context() {
    let w = load();
    let cfg = te_fixture_config();
    let baseline = run(&w, &cfg);

    // One key from each distinct kind of tensor the forward touches.
    for key in [
        "language_model.embed_tokens.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.0.self_attn.q_norm.weight",
        "language_model.layers.0.mlp.gate_proj.weight",
        "language_model.layers.0.input_layernorm.weight",
        // The LAST layer that is actually run — proves the tap is at the end of the stack.
        "language_model.layers.3.mlp.down_proj.weight",
    ] {
        let mut mutated = w.clone();
        let orig = w.require(key).unwrap();
        let bumped = mlx_rs::ops::add(orig, Array::from_f32(0.05)).unwrap();
        mutated.insert(key, bumped);

        let out = run(&mutated, &cfg);
        let (peak, _) = rel(&out, &baseline);
        assert!(
            peak > TOL,
            "mutating {key} moved the context by only {peak:.3e} — it is not load-bearing"
        );
    }
}

/// Mutating a tensor the encoder must NOT read (a layer past the tap, the final norm, `lm_head`)
/// must leave the output bit-identical. This is the other half of the trim evidence: it proves the
/// unused tail is genuinely unused rather than merely unlisted.
#[test]
fn mutating_an_unused_tensor_changes_nothing() {
    let w = load();
    let cfg = te_fixture_config();
    let baseline = run(&w, &cfg);

    for key in [
        "language_model.layers.4.mlp.down_proj.weight",
        "language_model.layers.5.self_attn.q_proj.weight",
        "language_model.norm.weight",
    ] {
        let mut mutated = w.clone();
        let orig = w.require(key).unwrap();
        let bumped = mlx_rs::ops::add(orig, Array::from_f32(10.0)).unwrap();
        mutated.insert(key, bumped);

        let out = run(&mutated, &cfg);
        let (peak, _) = rel(&out, &baseline);
        assert!(
            peak < 1e-6,
            "{key} is past the layer-{SELECT_HIDDEN} tap but moved the context by {peak:.3e}"
        );
    }
}

/// **Non-constant-field check.** A golden whose expected output is ~constant would pass against an
/// implementation that returns a constant.
#[test]
fn reference_context_is_not_constant() {
    let w = load();
    let want = w.require("out.context").unwrap();
    let sd = std_dev(want);
    println!("reference context std-dev = {sd:.5}");
    assert!(sd > 1e-3, "reference context is ~constant (std {sd:.3e})");

    // And every neighbour differs from the selected one, so the off-by-one gate has real signal.
    for key in ["out.context_at_3", "out.context_at_5"] {
        let (peak, _) = rel(w.require(key).unwrap(), want);
        assert!(
            peak > TOL * 10.0,
            "{key} is too close to the selected layer"
        );
    }
}

// ── The template / special-token half, verified against the SHIPPED files ────────────────────────
//
// The fixture's safetensors metadata carries values derived by rendering the real
// `chat_template.json` and reading the real `tokenizer_config.json` — not transcribed by hand into
// this crate. These tests assert the constants in `text_encoder::tokenizer` agree with them.

fn meta(w: &Weights, key: &str) -> String {
    w.metadata(key)
        .unwrap_or_else(|| panic!("fixture metadata is missing `{key}`; re-run the dump script"))
        .to_string()
}

/// **The sc-18741 gate: pin the ABSENCE of template application, not just a token count.**
///
/// The fixture metadata carries three id vectors for one probe prompt, all produced against the
/// real shipped tokenizer:
///
/// | key | what it is |
/// |---|---|
/// | `presentation_ids` | the official conditioner's `tokenizer(prompt, add_special_tokens=False)` |
/// | `templated_ids` | sc-17143's `chat_template.json` render of the same prompt |
/// | `sc17143_ids` | that render with its 3-token prefix dropped — what the port actually fed the DiT |
///
/// This asserts the presentation is the first and is neither of the others. A test that only
/// checked "no rows are dropped" would still pass if a template came back together with a
/// compensating slice; this cannot.
#[test]
fn presentation_applies_no_chat_template() {
    let w = load();
    // A compile-time assertion: reintroducing the chat template must not even build.
    const { assert!(!APPLIES_CHAT_TEMPLATE, "the crate's presentation contract") };
    assert_eq!(meta(&w, "applies_chat_template"), "false");
    assert_eq!(meta(&w, "add_special_tokens"), "false");

    let ids =
        |k: &str| -> Vec<i32> { meta(&w, k).split(',').map(|s| s.parse().unwrap()).collect() };
    let reference = ids("presentation_ids");
    let templated = ids("templated_ids");
    let sc17143 = ids("sc17143_ids");

    // The chat-template render opens with `<|im_start|>user\n` = [151644, 872, 198] and closes with
    // the 5-token generation cue [151645, 198, 151644, 77091, 198]. The reference presentation has
    // neither.
    assert_eq!(
        &templated[..3],
        &[151644, 872, 198],
        "the template's prefix"
    );
    assert_eq!(
        &templated[templated.len() - 5..],
        &[151645, 198, 151644, 77091, 198],
        "the template's generation cue"
    );
    assert_ne!(
        reference, templated,
        "the presentation is not the template render"
    );
    assert_ne!(
        reference, sc17143,
        "the presentation is not the template render with 3 ids sliced off — that was the shipped \
         behaviour and it is what this test exists to keep out (sc-18741)"
    );

    // What the port actually shipped: the prompt PLUS the 5-token generation cue. Slicing 3 removed
    // the prefix but nothing ever removed the suffix.
    assert_eq!(
        sc17143.len(),
        reference.len() + 5,
        "sc-17143 conditioned on 5 extra generation-cue rows"
    );
    assert_eq!(
        &sc17143[sc17143.len() - 5..],
        &[151645, 198, 151644, 77091, 198],
        "and those 5 rows are chat-turn control tokens, not prompt"
    );
    // No special token may appear in the reference presentation at all.
    for id in &reference {
        assert!(
            *id < 151643,
            "reference presentation contains special token {id}; it is plain text only"
        );
    }
    println!(
        "presentation {:?}: reference {} ids, sc-17143 emitted {} ids",
        meta(&w, "probe_prompt"),
        reference.len(),
        sc17143.len()
    );
}

/// The fixture must record which reference produced which half. A regeneration that reverts to
/// deriving the presentation from `chat_template.json` writes different provenance and fails here.
#[test]
fn fixture_provenance_records_the_official_conditioner() {
    let w = load();
    assert_eq!(meta(&w, "provenance"), "official-conditioner");
    assert_eq!(
        meta(&w, "tensor_reference"),
        "transformers.Qwen3VLTextModel"
    );
    assert_eq!(
        meta(&w, "presentation_reference"),
        "diffusers.MiniMaxH3TextEncoderStep"
    );
    println!(
        "te fixture provenance: tensors from {}, presentation from {} ({})",
        meta(&w, "tensor_reference"),
        meta(&w, "presentation_reference"),
        meta(&w, "reference_version"),
    );
}

/// The `<d>` token and its six siblings: declared only in `tokenizer_config.json`, assigned ids
/// positionally at load time. The fixture records what `transformers` actually resolved them to.
#[test]
fn minimax_special_tokens_match_the_shipped_tokenizer_config() {
    let w = load();
    let declared: Vec<String> = meta(&w, "additional_special_tokens")
        .split(',')
        .map(str::to_owned)
        .collect();
    let ids: serde_json::Value = serde_json::from_str(&meta(&w, "special_ids")).unwrap();

    // The shipped list is upstream Qwen's 13 followed by MiniMax's 7, in that order.
    assert_eq!(
        declared.len(),
        20,
        "shipped additional_special_tokens count"
    );
    assert_eq!(
        &declared[13..],
        &MINIMAX_ADDED_SPECIALS[..],
        "MiniMax's additions, in declaration order"
    );

    // `<d>` = 151669, contiguous through `<|caption_end|>` = 151675.
    for (i, tok) in MINIMAX_ADDED_SPECIALS.iter().enumerate() {
        let want = 151669 + i as i64;
        assert_eq!(
            ids.get(*tok).and_then(serde_json::Value::as_i64),
            Some(want),
            "{tok} must resolve to {want}"
        );
    }
    assert_eq!(
        ids.get("<d>").and_then(serde_json::Value::as_i64),
        Some(151669)
    );

    // All seven land inside the embedding table (vocab_size 151936), and the tokenizer's reported
    // length is exactly one past the last of them.
    let len_tok: i64 = meta(&w, "len_tokenizer").parse().unwrap();
    assert_eq!(len_tok, 151676);
    assert!(len_tok < MiniMaxH3TeConfig::qwen3_vl_32b().vocab_size as i64);
}
