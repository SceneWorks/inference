//! sc-17143 — committed-fixture parity for the MiniMax-H3 Qwen3-VL-32B context extraction against
//! the **transformers** `Qwen3VLTextModel` forward (an independent graph), at tiny dims.
//!
//! Exercises bias-less GQA, per-head q/k RMSNorm, HF half-split RoPE, the causal mask, the
//! select-layer capture and the template-prefix slice — the `context` the H3 DiT consumes. The
//! fixture is produced by `tools/dump_minimax_h3_te.py` and committed, so this runs by default.
//!
//! The suite is built around the fact that **an off-by-one in the layer selection still produces a
//! plausible tensor**. Equality against `hidden_states[50]` alone cannot catch it, so the fixture
//! also carries both neighbours and this file asserts *inequality* against them.

mod common;

use common::{assert_parity, rel, std_dev, te_fixture_config, TE_FIXTURE};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{
    MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MINIMAX_ADDED_SPECIALS, NUM_HIDDEN_LAYERS,
    SELECT_HIDDEN,
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

/// The headline parity gate: the port reproduces the reference's select-layer, prefix-trimmed
/// context.
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

/// The prefix slice drops exactly `prefix_tokens` leading positions off the selected state — the
/// trim is checked against the reference's UNtrimmed tensor, so the drop count is pinned
/// independently of the layer selection.
#[test]
fn prefix_trim_drops_exactly_the_template_tokens() {
    let w = load();
    let cfg = te_fixture_config();
    let got = run(&w, &cfg);

    let untrimmed = w.require("out.hidden_untrimmed").unwrap();
    let seq = untrimmed.shape()[1];
    assert_eq!(
        got.shape()[1],
        seq - cfg.prefix_tokens as i32,
        "trimmed length must be seq - prefix_tokens"
    );

    let tail = untrimmed
        .split_axis(&[cfg.prefix_tokens as i32], 1)
        .unwrap()
        .swap_remove(1);
    assert_parity(&got, &tail, TOL, "prefix-trimmed tail");
}

/// Dropping the prefix needs strictly more tokens than `prefix_tokens`; a shorter prompt must be a
/// typed error, not an opaque `split_axis` panic.
#[test]
fn rejects_prompt_shorter_than_the_prefix() {
    let w = load();
    let cfg = te_fixture_config(); // prefix_tokens = 3
    let te = encoder(&w, &cfg);

    for s in [1i32, 3] {
        let ids = Array::from_slice(&vec![0i32; s as usize], &[1, s]);
        let mask = Array::from_slice(&vec![1i32; s as usize], &[1, s]);
        let e = te.forward(&ids, &mask).unwrap_err().to_string();
        assert!(
            e.contains("template-prefix"),
            "expected the prefix-drop guard for s={s}, got: {e}"
        );
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

/// The chat-template prefix constant must equal what the SHIPPED `chat_template.json` actually
/// renders, and must tokenize to exactly `PREFIX_TOKENS` ids.
#[test]
fn template_prefix_matches_the_shipped_chat_template() {
    let w = load();
    assert_eq!(
        meta(&w, "prefix_str"),
        mlx_gen_minimax_h3::text_encoder::PREFIX,
        "the rendered chat-template prefix disagrees with the constant in this crate"
    );
    assert_eq!(
        meta(&w, "suffix_str"),
        mlx_gen_minimax_h3::text_encoder::SUFFIX,
        "the rendered generation cue disagrees with the constant in this crate"
    );

    let ids: Vec<i32> = meta(&w, "prefix_ids")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(ids, vec![151644, 872, 198], "<|im_start|>user\\n");
    assert_eq!(ids.len(), mlx_gen_minimax_h3::PREFIX_TOKENS);
    assert_eq!(
        ids.len(),
        meta(&w, "prefix_tokens").parse::<usize>().unwrap()
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
