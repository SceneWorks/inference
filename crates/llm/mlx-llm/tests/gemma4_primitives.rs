//! Gemma 4 shared-primitive goldens (sc-18769).
//!
//! Each primitive Gemma 4 adds to the shared LLM crates is pinned here against
//! `crates/llm/testdata/gemma4/gemma4_goldens.json` — a fixture transcribed from the public
//! reference (`huggingface/transformers` `gemma4_unified` + `_compute_proportional_rope_parameters`)
//! and **shared with `candle-llm`**, which asserts the identical numbers.
//!
//! Every golden is paired with a *mutation* assertion: the plausible wrong implementation (a
//! leading-slice partial RoPE instead of proportional; a plain causal mask instead of a windowed
//! one; a separate value projection instead of a shared one; the raw last hidden state instead of
//! the final-normed one) is shown to produce different numbers. A golden that a no-op or a
//! near-miss also passes is not asking its question.
//!
//! Assertions are on **absolute** error, not cosine similarity — a rope schedule with the right
//! shape and the wrong base is cosine-close and numerically wrong.

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};

use mlx_llm::config::{LayerAttentionType, ModelConfig, RopeType};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::nn::{linear, rms_norm, rms_norm_unscaled};
use mlx_llm::primitives::projection::{KvProjection, Projection};
use mlx_llm::primitives::rope::{apply_rope, Rope};
use mlx_llm::primitives::sampler::{SplitMix64, TokenRng};
use mlx_llm::primitives::{
    input_ids, input_ids_batch, sdpa, sliding_causal_mask, AttnMask, KvCache, Weights,
};

const GOLDENS: &str = include_str!("../../testdata/gemma4/gemma4_goldens.json");
const GEMMA4_CONFIG: &str = include_str!("../../testdata/gemma4/gemma4_unified_config.json");
/// The **real** LTX-2.5 packed text encoder's safetensors header (sc-18756 evidence capture), whose
/// `metadata.gemma_config` is the shipped `Gemma4UnifiedConfig` verbatim. Weightless — 6 KB of
/// header JSON, no gated download — so the config layer can be asserted against the actual
/// checkpoint rather than only against a hand-shaped fixture.
const REAL_TE_HEADER: &str = include_str!(
    "../../../../docs/reference/sc-18756-headers/text_encoders/\
     gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors.json"
);

/// The per-layer-type RoPE widths the table must resolve to (`Rope::dim()` is `i32` on
/// this backend).
const SLIDING_ROPE_DIM: i32 = 256;
const FULL_ROPE_DIM: i32 = 512;

fn goldens() -> Value {
    serde_json::from_str(GOLDENS).expect("parse gemma4 goldens")
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("float array")
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Assert on **absolute** error, with the budget scaled by the golden's own magnitude.
///
/// The goldens are computed in f64 and these primitives run in f32, and anything that passes
/// through a matmul runs on Metal (whose GEMM is a few units in the last place of f32 looser than
/// an f64 host reduction). `rel` is that budget as a fraction of `max|want|`, floored at 1.0 so a
/// near-zero golden still gets an honest absolute floor rather than a vanishing one.
fn assert_abs_close(got: &[f32], want: &[f32], rel: f32, what: &str) {
    let scale = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1.0);
    let err = max_abs_err(got, want);
    assert!(
        err <= rel * scale,
        "{what}: max|delta| = {err} exceeds {} (rel {rel} of magnitude {scale})",
        rel * scale
    );
}

fn host(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

// ---------------------------------------------------------------------------------------------
// layer_types resolution
// ---------------------------------------------------------------------------------------------

/// The 48-entry `layer_types` table resolves to 40 `sliding_attention` (head_dim 256, theta 10 000,
/// window 1024, 8 KV heads) and 8 `full_attention` (head_dim 512, theta 1 000 000, partial 0.25,
/// no window, 1 KV head, shared K/V) — every 6th layer, the last one included.
///
/// Getting this table wrong is invisible in a smoke render: the model still runs, every layer still
/// attends something, and the output is merely worse. It is only ever caught by asserting the table.
#[test]
fn gemma4_layer_types_resolve_to_forty_sliding_and_eight_full() {
    let g = goldens();
    let expect = &g["layer_types"];
    let cfg: Value = serde_json::from_str(GEMMA4_CONFIG).expect("parse gemma4 config fixture");
    let cfg = ModelConfig::from_json(&cfg).expect("gemma4 config parses");

    assert!(cfg.is_gemma4());
    assert_eq!(cfg.architecture.family(), "gemma4_unified");
    assert_eq!(cfg.num_layers, 48);
    let table = cfg.gemma4.as_ref().expect("gemma 4 carries a layer table");
    assert_eq!(table.layer_types.len(), 48);

    let want_types: Vec<&str> = expect["types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    let got_types: Vec<&str> = table.layer_types.iter().map(|t| t.as_str()).collect();
    assert_eq!(got_types, want_types, "the 48-entry layer_types pattern");

    let full: Vec<usize> = (0..48)
        .filter(|&i| cfg.layer_type(i) == LayerAttentionType::Full)
        .collect();
    let want_full: Vec<usize> = expect["full_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(full, want_full, "full-attention layer indices");
    assert_eq!(full.len(), expect["num_full"].as_u64().unwrap() as usize);
    assert_eq!(
        48 - full.len(),
        expect["num_sliding"].as_u64().unwrap() as usize
    );
    assert_eq!(
        *full.last().unwrap(),
        47,
        "the last layer is full attention"
    );

    // Every layer's resolved descriptor, not just the type tag.
    for i in 0..48 {
        let la = cfg.layer_attention(i);
        if cfg.layer_type(i) == LayerAttentionType::Sliding {
            assert_eq!(la.head_dim, 256, "layer {i} sliding head_dim");
            assert_eq!(la.num_kv_heads, 8, "layer {i} sliding kv heads");
            assert_eq!(la.rope_type, RopeType::Default, "layer {i}");
            assert_eq!(la.rope_theta, 10_000.0, "layer {i} sliding theta");
            assert_eq!(la.partial_rotary_factor, 1.0, "layer {i}");
            assert_eq!(la.sliding_window, Some(1024), "layer {i} window");
            assert!(!la.k_eq_v, "sliding layers never share k/v");
            assert_eq!(cfg.layer_groups(i), 2, "layer {i} groups (16 heads / 8 kv)");
            // The convenience accessors the decoder stories will actually call, not just the
            // descriptor behind them — a dropped index inside one of these is invisible otherwise.
            assert_eq!(
                cfg.layer_head_dim(i),
                256,
                "layer {i} head dim via accessor"
            );
            assert_eq!(
                cfg.layer_sliding_window(i),
                Some(1024),
                "layer {i} window via accessor"
            );
            assert_eq!(cfg.layer_rope(i).dim(), SLIDING_ROPE_DIM, "layer {i} rope");
        } else {
            assert_eq!(la.head_dim, 512, "layer {i} full head_dim");
            assert_eq!(la.num_kv_heads, 1, "layer {i} full kv heads");
            assert_eq!(la.rope_type, RopeType::Proportional, "layer {i}");
            assert_eq!(la.rope_theta, 1_000_000.0, "layer {i} full theta");
            assert_eq!(la.partial_rotary_factor, 0.25, "layer {i} partial");
            assert_eq!(la.sliding_window, None, "layer {i} must be un-windowed");
            assert!(la.k_eq_v, "full layers share one k/v projection");
            assert_eq!(
                cfg.layer_groups(i),
                16,
                "layer {i} groups (16 heads / 1 kv)"
            );
            assert_eq!(
                cfg.layer_head_dim(i),
                512,
                "layer {i} head dim via accessor"
            );
            assert_eq!(
                cfg.layer_sliding_window(i),
                None,
                "layer {i} window via accessor"
            );
            assert_eq!(cfg.layer_rope(i).dim(), FULL_ROPE_DIM, "layer {i} rope");
        }
    }

    // Gemma 4's attention scale is 1.0, not head_dim^-0.5 — the q/k norms take its place.
    assert_eq!(cfg.attn_scale(), 1.0);
    assert_eq!(cfg.final_logit_softcap, Some(30.0));
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.hidden_size, 3840);
    assert_eq!(cfg.intermediate_size, 15360);
    assert_eq!(cfg.vocab_size, 262144);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
}

/// The shipped `google/gemma-4-12B-it` text config, read straight out of the LTX-2.5 packed text
/// encoder's safetensors metadata, must produce the same table the synthetic fixture does.
///
/// This is what turns the fixture from "shaped after the epic's measurements" into "agrees with the
/// checkpoint": every number below is the checkpoint's own, and the 48-entry `layer_types` array it
/// ships is compared against the schedule this crate derives when the key is absent.
#[test]
fn gemma4_real_packed_te_config_parses_to_the_same_table() {
    let header: Value = serde_json::from_str(REAL_TE_HEADER).expect("parse TE header dump");
    let real = &header["metadata"]["gemma_config"];
    assert_eq!(real["model_type"], "gemma4_unified");
    assert_eq!(real["gemma_version"], "gemma4-12b-ltx-v1");

    let cfg = ModelConfig::from_json(real).expect("the shipped Gemma 4 config parses");
    assert_eq!(cfg.architecture.family(), "gemma4_unified");
    assert_eq!(cfg.num_layers, 48);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.hidden_size, 3840);
    assert_eq!(cfg.intermediate_size, 15360);
    assert_eq!(cfg.vocab_size, 262144);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.final_logit_softcap, Some(30.0));
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.attn_scale(), 1.0);
    assert!(!cfg.is_moe(), "`num_experts: null` is not a MoE model");
    assert!(!cfg.is_mla());

    // The checkpoint ships `layer_types` explicitly; it must equal the derived 5:1 schedule (which
    // is what makes the derive path safe for any Gemma 4 config that omits the key).
    let shipped: Vec<&str> = real["text_config"]["layer_types"]
        .as_array()
        .expect("the shipped config carries layer_types")
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    assert_eq!(shipped.len(), 48);
    let parsed: Vec<&str> = cfg
        .gemma4
        .as_ref()
        .unwrap()
        .layer_types
        .iter()
        .map(|t| t.as_str())
        .collect();
    assert_eq!(parsed, shipped, "parsed table vs the shipped layer_types");

    // And it must agree with the synthetic fixture, which omits `layer_types` entirely.
    let synthetic =
        ModelConfig::from_json(&serde_json::from_str::<Value>(GEMMA4_CONFIG).unwrap()).unwrap();
    assert_eq!(
        cfg.gemma4.as_ref().unwrap().layer_types,
        synthetic.gemma4.as_ref().unwrap().layer_types,
        "the derived schedule must match the shipped one"
    );
    assert_eq!(
        cfg.gemma4.as_ref().unwrap().sliding,
        synthetic.gemma4.as_ref().unwrap().sliding
    );
    assert_eq!(
        cfg.gemma4.as_ref().unwrap().full,
        synthetic.gemma4.as_ref().unwrap().full
    );
}

/// Attempt a `CausalLm` build against an empty weight map — enough to reach (and assert) the
/// architecture guard, which runs before any tensor is looked up.
fn build_empty_causal_lm(cfg: ModelConfig) -> Result<CausalLm, mlx_llm::Error> {
    CausalLm::from_weights(&Weights::from_map(HashMap::new()), "", cfg)
}

/// sc-18769's blanket "Gemma 4 is unsupported" refusal is **gone** (sc-18760 built the decoder), so
/// a Gemma 4 build must now fail the ordinary way — on the first tensor it cannot find — rather
/// than on an architecture check.
///
/// The empty weight map makes that the whole assertion: reaching a missing-tensor error proves the
/// loader got past the architecture guard and into real weight lookups.
#[test]
fn generic_decoder_no_longer_refuses_gemma4_by_architecture() {
    let cfg: Value = serde_json::from_str(GEMMA4_CONFIG).unwrap();
    let cfg = ModelConfig::from_json(&cfg).expect("the config layer must still parse it");
    assert!(cfg.is_gemma4());
    match build_empty_causal_lm(cfg) {
        Err(e) => {
            let m = e.to_string();
            assert!(
                !m.contains("sc-18760") && !m.contains("sc-18761"),
                "the placeholder refusal must be gone now that the decoder is built: {m}"
            );
            assert!(
                m.contains("embed_tokens") || m.contains("missing") || m.contains("Missing"),
                "an empty weight map must fail on a weight lookup, not an architecture check: {m}"
            );
        }
        Ok(_) => panic!("an empty weight map cannot produce a model"),
    }
}

/// `use_bidirectional_attention: "all"` is refused at load: it makes every token attend both ways,
/// which needs a *symmetric* window this decoder does not build. Refusing beats silently running
/// the model causally — the numbers would look plausible and be wrong everywhere.
#[test]
fn gemma4_bidirectional_all_is_refused_by_the_decoder() {
    let mut cfg: Value = serde_json::from_str(GEMMA4_CONFIG).unwrap();
    cfg["text_config"]["use_bidirectional_attention"] = Value::from("all");
    let cfg = ModelConfig::from_json(&cfg).expect("the config layer parses it");
    let err = build_empty_causal_lm(cfg).expect_err("must refuse");
    let m = err.to_string();
    assert!(
        m.contains("bidirectional") && m.contains("symmetric"),
        "the refusal must name what is unsupported and why: {m}"
    );
}

/// The schedule is derived when `layer_types` is absent and read verbatim when present — and the
/// two agree on the shipped config, which is the only reason omitting it is safe.
#[test]
fn gemma4_layer_types_derived_schedule_matches_an_explicit_one() {
    let g = goldens();
    let types: Vec<Value> = g["layer_types"]["types"].as_array().unwrap().clone();
    let base: Value = serde_json::from_str(GEMMA4_CONFIG).unwrap();

    let derived = ModelConfig::from_json(&base).unwrap();
    let mut explicit_json = base.clone();
    explicit_json["text_config"]["layer_types"] = Value::Array(types);
    let explicit = ModelConfig::from_json(&explicit_json).unwrap();
    assert_eq!(
        derived.gemma4.as_ref().unwrap().layer_types,
        explicit.gemma4.as_ref().unwrap().layer_types
    );

    // Mutation: an explicit table that disagrees must be honoured, not silently re-derived.
    let mut odd = base.clone();
    odd["text_config"]["layer_types"] =
        Value::Array((0..48).map(|_| json!("full_attention")).collect());
    let odd = ModelConfig::from_json(&odd).unwrap();
    assert!(
        (0..48).all(|i| odd.layer_type(i) == LayerAttentionType::Full),
        "an explicit layer_types array must override the derived schedule"
    );
}

/// `use_bidirectional_attention: "all"` halves the stored window (`w/2 + 1`), as upstream's
/// `__post_init__` does; the default `"vision"` leaves it alone.
#[test]
fn gemma4_bidirectional_all_halves_the_sliding_window() {
    let base: Value = serde_json::from_str(GEMMA4_CONFIG).unwrap();
    assert_eq!(
        ModelConfig::from_json(&base)
            .unwrap()
            .layer_sliding_window(0),
        Some(1024)
    );
    let mut all = base.clone();
    all["text_config"]["use_bidirectional_attention"] = json!("all");
    assert_eq!(
        ModelConfig::from_json(&all)
            .unwrap()
            .layer_sliding_window(0),
        Some(513),
        "1024 / 2 + 1"
    );
}

// ---------------------------------------------------------------------------------------------
// proportional RoPE
// ---------------------------------------------------------------------------------------------

/// Both layer types' real inverse-frequency tables, against the reference oracle.
#[test]
fn gemma4_real_rope_inv_freq_matches_golden() {
    let g = goldens();
    let rope = &g["rope"];

    let sliding = Rope::standard(256, 10_000.0);
    assert_abs_close(
        sliding.inv_freq(),
        &floats(&rope["sliding_real_inv_freq"]),
        1e-6,
        "sliding (default, theta 1e4) inv_freq",
    );

    let full = Rope::proportional(512, 1_000_000.0, 0.25, 1.0);
    let want = floats(&rope["full_real_inv_freq"]);
    assert_eq!(full.dim(), 512, "proportional keeps the full head width");
    assert_eq!(full.inv_freq().len(), 256);
    assert_abs_close(
        full.inv_freq(),
        &want,
        1e-6,
        "full (proportional, theta 1e6) inv_freq",
    );

    // The tail is exactly zero, not merely small: those channels are an identity rotation.
    let angles = rope["full_real_rope_angles"].as_u64().unwrap() as usize;
    assert_eq!(angles, 64, "0.25 * 512 / 2");
    assert!(full.inv_freq()[..angles].iter().all(|&f| f > 0.0));
    assert!(
        full.inv_freq()[angles..].iter().all(|&f| f == 0.0),
        "the un-rotated channels must be exactly 0.0"
    );
}

/// The rotation itself, on small synthetic configs for **both** layer types: cos/sin tables and the
/// rotated tensor, against the reference `apply_rotary_pos_emb`.
#[test]
fn gemma4_rope_rotation_matches_golden_for_both_layer_types() {
    let g = goldens();
    for kind in ["sliding", "full"] {
        let c = &g["rope"]["small"][kind];
        let head_dim = c["head_dim"].as_i64().unwrap() as i32;
        let theta = c["theta"].as_f64().unwrap() as f32;
        let rope = if kind == "sliding" {
            Rope::standard(head_dim, theta)
        } else {
            Rope::proportional(
                head_dim,
                theta,
                c["partial_rotary_factor"].as_f64().unwrap() as f32,
                1.0,
            )
        };
        assert_abs_close(
            rope.inv_freq(),
            &floats(&c["inv_freq"]),
            1e-6,
            &format!("{kind}: inv_freq"),
        );

        let positions: Vec<i32> = c["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        let (cos, sin) = rope.cos_sin_at(&positions, Dtype::Float32).unwrap();
        assert!(
            max_abs_err(&host(&cos), &floats(&c["cos"])) < 1e-6,
            "{kind}: cos table"
        );
        assert!(
            max_abs_err(&host(&sin), &floats(&c["sin"])) < 1e-6,
            "{kind}: sin table"
        );

        let shape: Vec<i32> = c["x_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        let x = Array::from_slice(&floats(&c["x"]), &shape);
        let rotated = apply_rope(&x, &cos, &sin, false).unwrap();
        let err = max_abs_err(&host(&rotated), &floats(&c["rotated"]));
        assert!(err < 1e-5, "{kind}: rotated tensor, max|delta| = {err}");
    }
}

/// **Mutation guard.** Proportional RoPE is not a leading-slice partial RoPE with the same rotated
/// width, and it is not plain standard RoPE. Both are the near-misses an implementer reaches for,
/// and both are numerically wrong; if either matched, the golden above would be proving nothing.
#[test]
fn gemma4_proportional_rope_differs_from_partial_and_standard() {
    let head_dim = 16;
    let theta = 1_000_000.0;
    let partial_factor = 0.25;
    let proportional = Rope::proportional(head_dim, theta, partial_factor, 1.0);

    // A leading-slice partial RoPE over the same 4 rotated dims re-bases the exponent on the slice,
    // so its frequencies differ where they overlap...
    let rotary_dim = (head_dim as f32 * partial_factor) as i32; // 4
    let partial = Rope::partial(rotary_dim, theta, false);
    assert_ne!(
        partial.dim(),
        proportional.dim(),
        "partial narrows the table; proportional keeps the full head"
    );
    assert!(
        (partial.inv_freq()[1] - proportional.inv_freq()[1]).abs() > 1e-3,
        "partial re-bases the exponent on the rotated span; proportional does not"
    );

    // ...and the rotation itself differs, because the NeoX pair partner is a different channel.
    let positions = [3i32];
    let x = Array::from_slice(
        &(0..head_dim)
            .map(|i| (i as f32) * 0.1 - 0.7)
            .collect::<Vec<_>>(),
        &[1, 1, 1, head_dim],
    );
    let (pc, ps) = proportional.cos_sin_at(&positions, Dtype::Float32).unwrap();
    let (qc, qs) = partial.cos_sin_at(&positions, Dtype::Float32).unwrap();
    let a = host(&apply_rope(&x, &pc, &ps, false).unwrap());
    let b = host(&apply_rope(&x, &qc, &qs, false).unwrap());
    assert!(
        max_abs_err(&a, &b) > 1e-3,
        "proportional and partial RoPE must not agree"
    );

    // Standard RoPE over the same head rotates every channel; proportional leaves most of it alone.
    let standard = Rope::standard(head_dim, theta);
    let (sc, ss) = standard.cos_sin_at(&positions, Dtype::Float32).unwrap();
    let c = host(&apply_rope(&x, &sc, &ss, false).unwrap());
    assert!(
        max_abs_err(&a, &c) > 1e-3,
        "proportional must not collapse to standard RoPE at factor 0.25"
    );

    // ...but it does at factor 1.0, which is the documented degenerate case.
    let full = Rope::proportional(head_dim, theta, 1.0, 1.0);
    assert_eq!(
        full.inv_freq(),
        standard.inv_freq(),
        "factor 1.0 is exactly standard RoPE"
    );
}

/// **`rope_parameters.factor`.** The reference divides the proportional schedule's inverse
/// frequencies by `factor` as its last step. Every shipped Gemma 4 config carries `1.0` (a no-op),
/// but a config that carried anything else used to be parsed with the key silently dropped — a
/// wrong RoPE base that still renders.
///
/// A non-1.0 factor must scale the rotated channels and leave the zero tail zero; and because the
/// reference defines `factor` only for the *proportional* schedule, a `default` layer type carrying
/// one is refused outright rather than ignored.
#[test]
fn gemma4_rope_factor_scales_the_proportional_schedule_and_is_refused_elsewhere() {
    let plain = Rope::proportional(16, 1_000_000.0, 0.25, 1.0);
    let scaled = Rope::proportional(16, 1_000_000.0, 0.25, 4.0);
    assert_eq!(plain.inv_freq().len(), scaled.inv_freq().len());
    for (i, (p, s)) in plain.inv_freq().iter().zip(scaled.inv_freq()).enumerate() {
        if *p == 0.0 {
            assert_eq!(
                *s, 0.0,
                "channel {i}: the un-rotated tail stays exactly zero"
            );
        } else {
            assert!(
                (s - p / 4.0).abs() < 1e-9,
                "channel {i}: factor must divide the frequency ({s} vs {})",
                p / 4.0
            );
        }
    }
    assert!(
        plain.inv_freq() != scaled.inv_freq(),
        "a non-unit factor must change the schedule"
    );

    // Parsed off a real config, not just constructed.
    let mut cfg: Value = serde_json::from_str(GEMMA4_CONFIG).unwrap();
    cfg["text_config"]["rope_parameters"]["full_attention"]["factor"] = json!(4.0);
    let parsed = ModelConfig::from_json(&cfg).expect("a proportional factor is supported");
    let full = parsed.gemma4.as_ref().unwrap().full;
    assert_eq!(full.rope_factor, 4.0);
    let rope = full.build_rope();
    let unscaled = Rope::proportional(
        full.head_dim,
        full.rope_theta,
        full.partial_rotary_factor,
        1.0,
    );
    assert!(
        (rope.inv_freq()[0] - unscaled.inv_freq()[0] / 4.0).abs() < 1e-9,
        "the parsed factor must reach the built RoPE"
    );

    // The shipped config's implicit factor is 1.0 on both layer types.
    let shipped =
        ModelConfig::from_json(&serde_json::from_str::<Value>(GEMMA4_CONFIG).unwrap()).unwrap();
    let g = shipped.gemma4.as_ref().unwrap();
    assert_eq!(g.full.rope_factor, 1.0);
    assert_eq!(g.sliding.rope_factor, 1.0);

    // A `default` layer type with a factor is refused by name, not silently dropped.
    let mut bad: Value = serde_json::from_str(GEMMA4_CONFIG).unwrap();
    bad["text_config"]["rope_parameters"]["sliding_attention"]["factor"] = json!(2.0);
    let err = ModelConfig::from_json(&bad).expect_err("a default-schedule factor must be refused");
    let m = err.to_string();
    assert!(m.contains("sliding_attention"), "{m}");
    assert!(m.contains("factor"), "{m}");
    assert!(m.contains("proportional"), "{m}");
}

// ---------------------------------------------------------------------------------------------
// sliding-window mask
// ---------------------------------------------------------------------------------------------

/// The additive sliding-window mask's visibility pattern, against the reference semantics
/// (`0 <= q - j < window`, bottom-right aligned over the cached keys).
#[test]
fn gemma4_sliding_causal_mask_matches_golden_visibility() {
    let g = goldens();
    for case in g["sliding_masks"].as_array().unwrap() {
        let (q_len, k_len, window) = (
            case["q_len"].as_i64().unwrap() as i32,
            case["k_len"].as_i64().unwrap() as i32,
            case["window"].as_i64().unwrap() as i32,
        );
        let mask = sliding_causal_mask(q_len, k_len, window).unwrap();
        assert_eq!(mask.shape(), &[1, 1, q_len, k_len]);
        let flat = host(&mask);
        let allowed = case["allowed"].as_array().unwrap();
        for r in 0..q_len as usize {
            let row = allowed[r].as_array().unwrap();
            for j in 0..k_len as usize {
                let want_open = row[j].as_bool().unwrap();
                let value = flat[r * k_len as usize + j];
                if want_open {
                    assert_eq!(
                        value, 0.0,
                        "q{q_len}/k{k_len}/w{window} [{r},{j}] must be open"
                    );
                } else {
                    assert!(
                        value < -1e20,
                        "q{q_len}/k{k_len}/w{window} [{r},{j}] must be blocked, got {value}"
                    );
                }
            }
        }
    }

    // A window wide enough to cover the cache degenerates to a plain causal mask (case q3/k7/w32).
    let wide = host(&sliding_causal_mask(3, 7, 32).unwrap());
    for r in 0..3usize {
        for j in 0..7usize {
            let causal_open = j <= (7 - 3) + r;
            assert_eq!(wide[r * 7 + j] == 0.0, causal_open, "[{r},{j}]");
        }
    }
    // A non-positive window is an error, not an all-blocked row.
    assert!(sliding_causal_mask(2, 2, 0).is_err());
}

/// **Behavioural mutation guard.** Attention under a sliding mask must be *independent* of keys
/// outside the window and *dependent* on keys inside it. A mask that silently degraded to plain
/// causal (or to no mask at all) fails the first half; a mask that blocked everything fails the
/// second.
#[test]
fn gemma4_sliding_window_attention_ignores_keys_outside_the_window() {
    let (heads, k_len, head_dim, window) = (2i32, 12i32, 8i32, 3i32);
    let mut rng = SplitMix64::new(0x18769);
    let randn = |n: usize, rng: &mut SplitMix64| -> Vec<f32> {
        (0..n).map(|_| (rng.next_f32() - 0.5) * 2.0).collect()
    };
    let q = Array::from_slice(
        &randn((heads * head_dim) as usize, &mut rng),
        &[1, heads, 1, head_dim],
    );
    let k_data = randn((heads * k_len * head_dim) as usize, &mut rng);
    let v_data = randn((heads * k_len * head_dim) as usize, &mut rng);
    let shape = [1, heads, k_len, head_dim];
    let k = Array::from_slice(&k_data, &shape);
    let v = Array::from_slice(&v_data, &shape);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let base = host(&sdpa(&q, &k, &v, scale, AttnMask::SlidingCausal { window }).unwrap());

    // The single query sits at position k_len - 1 = 11 and may see keys 9, 10, 11 only.
    let perturb = |data: &[f32], pos: i32| -> Array {
        let mut d = data.to_vec();
        for h in 0..heads {
            for c in 0..head_dim {
                let idx = ((h * k_len + pos) * head_dim + c) as usize;
                d[idx] += 5.0;
            }
        }
        Array::from_slice(&d, &shape)
    };

    for outside in [0, 4, 8] {
        let k2 = perturb(&k_data, outside);
        let v2 = perturb(&v_data, outside);
        let out = host(&sdpa(&q, &k2, &v2, scale, AttnMask::SlidingCausal { window }).unwrap());
        let err = max_abs_err(&out, &base);
        assert!(
            err < 1e-5,
            "key/value {outside} is outside the {window}-wide window but changed the output by {err}"
        );
    }
    for inside in [9, 10, 11] {
        let v2 = perturb(&v_data, inside);
        let out = host(&sdpa(&q, &k, &v2, scale, AttnMask::SlidingCausal { window }).unwrap());
        assert!(
            max_abs_err(&out, &base) > 1e-3,
            "value {inside} is inside the window and must change the output"
        );
    }

    // And the whole point: a plain causal mask over the same tensors is a *different* answer.
    let causal = host(&sdpa(&q, &k, &v, scale, AttnMask::Causal).unwrap());
    assert!(
        max_abs_err(&base, &causal) > 1e-3,
        "a sliding window must not reduce to plain causal attention at window {window} < {k_len}"
    );
}

/// The sliding mask survives the prefill path too (`q_len > 8` chunks through MLX's
/// broken-kernel mitigation), where the mask is sliced per chunk rather than applied whole.
#[test]
fn gemma4_sliding_window_prefill_matches_the_eager_reference() {
    // `head_dim = 64` and `q_len = 16` deliberately land on MLX's broken-fused-kernel envelope
    // (`q_len > 8`, multi-head, power-of-two head dim), so `sdpa` routes this through the chunked
    // prefill — the path that slices the materialized window mask's query axis per chunk. At a
    // smaller head dim the chunker never runs and this test would prove nothing about it.
    let (heads, seq, head_dim, window) = (2i32, 16i32, 64i32, 5i32);
    let mut rng = SplitMix64::new(0xB0B);
    let randn = |n: usize, rng: &mut SplitMix64| -> Vec<f32> {
        (0..n).map(|_| (rng.next_f32() - 0.5) * 2.0).collect()
    };
    let shape = [1, heads, seq, head_dim];
    let n = (heads * seq * head_dim) as usize;
    let q = Array::from_slice(&randn(n, &mut rng), &shape);
    let k = Array::from_slice(&randn(n, &mut rng), &shape);
    let v = Array::from_slice(&randn(n, &mut rng), &shape);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let got = host(&sdpa(&q, &k, &v, scale, AttnMask::SlidingCausal { window }).unwrap());
    // The same mask handed over explicitly, which takes the ordinary additive path.
    let m = sliding_causal_mask(seq, seq, window).unwrap();
    let want = host(&sdpa(&q, &k, &v, scale, AttnMask::Additive(&m)).unwrap());
    let err = max_abs_err(&got, &want);
    assert!(
        err < 2e-3,
        "sliding prefill vs explicit additive mask: {err}"
    );

    // Each query row must see at most `window` keys — check via the mask the path builds.
    let flat = host(&m);
    for r in 0..seq as usize {
        let open = (0..seq as usize)
            .filter(|&j| flat[r * seq as usize + j] == 0.0)
            .count();
        assert!(open <= window as usize, "row {r} sees {open} keys");
        assert!(open >= 1, "row {r} must see at least itself");
    }
}

// ---------------------------------------------------------------------------------------------
// attention_k_eq_v projection
// ---------------------------------------------------------------------------------------------

/// `attention_k_eq_v`: one projection feeds both key and value heads. K takes the scaled `k_norm`,
/// V takes the **scale-free** `v_norm` of the *raw* projection output — so the two are different
/// tensors built from one matmul, which is exactly what the goldens pin.
#[test]
fn gemma4_k_eq_v_projection_matches_golden() {
    let g = goldens();
    let c = &g["k_eq_v"];
    let hidden = c["hidden_size"].as_i64().unwrap() as i32;
    let kv_heads = c["num_kv_heads"].as_i64().unwrap() as i32;
    let head_dim = c["head_dim"].as_i64().unwrap() as i32;
    let seq = c["seq"].as_i64().unwrap() as i32;
    let eps = c["rms_norm_eps"].as_f64().unwrap() as f32;

    let x = Array::from_slice(&floats(&c["x"]), &[1, seq, hidden]);
    let w_k = Array::from_slice(&floats(&c["w_k"]), &[kv_heads * head_dim, hidden]);
    let w_v = Array::from_slice(&floats(&c["w_v"]), &[kv_heads * head_dim, hidden]);
    let k_norm_w = Array::from_slice(&floats(&c["k_norm_weight"]), &[head_dim]);
    let heads_shape = [1, seq, kv_heads, head_dim];

    // --- shared (attention_k_eq_v: true) ---
    let shared = KvProjection::shared(Projection::load(w_k.clone(), None).unwrap());
    assert!(shared.k_eq_v());
    assert!(shared.value().is_none(), "no v_proj weight exists at all");
    let (raw_k, raw_v) = shared.forward(&x).unwrap();
    assert_abs_close(
        &host(&raw_k),
        &floats(&c["raw_k"]),
        2e-3,
        "raw key projection",
    );
    assert_eq!(
        host(&raw_k),
        host(&raw_v),
        "a shared projection must hand both paths the same raw tensor"
    );

    let k = rms_norm(&raw_k.reshape(&heads_shape).unwrap(), &k_norm_w, eps).unwrap();
    let v = rms_norm_unscaled(&raw_v.reshape(&heads_shape).unwrap(), eps).unwrap();
    assert_abs_close(
        &host(&k),
        &floats(&c["k_shared"]),
        2e-3,
        "shared k after k_norm",
    );
    assert_abs_close(
        &host(&v),
        &floats(&c["v_shared"]),
        2e-3,
        "shared v after the scale-free v_norm",
    );

    // **Mutation guard.** Sharing the projection must not make K and V the same tensor: the two
    // norms differ (one scaled, one not). An implementation that reused `k` for `v` would pass a
    // shape check and fail here.
    assert!(
        max_abs_err(&host(&k), &host(&v)) > 1e-3,
        "k_norm and the scale-free v_norm must not agree"
    );
    // ...nor may the value path silently take the scaled norm.
    let wrong_v = rms_norm(&raw_v.reshape(&heads_shape).unwrap(), &k_norm_w, eps).unwrap();
    assert!(
        max_abs_err(&host(&wrong_v), &floats(&c["v_shared"])) > 1e-3,
        "the value norm is scale-free; applying k_norm's weight must fail the golden"
    );

    // --- separate (attention_k_eq_v: false) ---
    let separate = KvProjection::separate(
        Projection::load(w_k, None).unwrap(),
        Projection::load(w_v, None).unwrap(),
    );
    assert!(!separate.k_eq_v());
    assert!(separate.value().is_some());
    let (sep_k, sep_v) = separate.forward(&x).unwrap();
    assert_abs_close(
        &host(&sep_k),
        &floats(&c["raw_k"]),
        2e-3,
        "separate: key path is unchanged",
    );
    assert_abs_close(
        &host(&sep_v),
        &floats(&c["raw_v_separate"]),
        2e-3,
        "separate: value comes from its own weight",
    );
    let sep_v = rms_norm_unscaled(&sep_v.reshape(&heads_shape).unwrap(), eps).unwrap();
    assert_abs_close(
        &host(&sep_v),
        &floats(&c["v_separate"]),
        2e-3,
        "separate: normed value",
    );
    // The two modes must not coincide — otherwise the flag would be untestable.
    assert!(
        max_abs_err(&host(&sep_v), &floats(&c["v_shared"])) > 1e-3,
        "a separate v_proj must not reproduce the shared one"
    );
}

// ---------------------------------------------------------------------------------------------
// hidden-state-stack forward
// ---------------------------------------------------------------------------------------------

/// Build a tiny deterministic Llama and drive the hidden-state-stack forward.
///
/// The stack is `num_layers + 1` entries in Hugging Face's `output_hidden_states` layout — the
/// input embeddings, each layer's output, and a **final-normed** last entry. The check that the
/// last entry is normed is the load-bearing one: it is invisible to any logits assertion (the
/// logits path applies the same norm either way) and silently shifts every feature an encoder
/// consumer builds. Here it is pinned by pushing the stack's last entry through the (tied) LM head
/// and requiring it to reproduce the model's own all-position logits.
#[test]
fn hidden_state_stack_matches_the_logits_forward() {
    const HIDDEN: i32 = 32;
    const VOCAB: i32 = 48;
    const HEAD_DIM: i32 = 8;
    const LAYERS: usize = 3;
    let (heads, kv, inter) = (4i32, 2i32, 64i32);

    let cfg_json = json!({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": HIDDEN, "intermediate_size": inter, "num_hidden_layers": LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": HEAD_DIM,
        "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "tie_word_embeddings": true
    });
    let cfg = ModelConfig::from_json(&cfg_json).unwrap();

    let mut rng = SplitMix64::new(0x5C18769);
    let mut randn = |shape: &[i32]| -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Array::from_slice(&data, shape)
    };
    let mut m: HashMap<String, Array> = HashMap::new();
    let embed = randn(&[VOCAB, HIDDEN]);
    m.insert("model.embed_tokens.weight".into(), embed.clone());
    m.insert(
        "model.norm.weight".into(),
        Array::ones::<f32>(&[HIDDEN]).unwrap(),
    );
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(
            p("self_attn.q_proj.weight"),
            randn(&[heads * HEAD_DIM, HIDDEN]),
        );
        m.insert(
            p("self_attn.k_proj.weight"),
            randn(&[kv * HEAD_DIM, HIDDEN]),
        );
        m.insert(
            p("self_attn.v_proj.weight"),
            randn(&[kv * HEAD_DIM, HIDDEN]),
        );
        m.insert(
            p("self_attn.o_proj.weight"),
            randn(&[HIDDEN, heads * HEAD_DIM]),
        );
        m.insert(p("mlp.gate_proj.weight"), randn(&[inter, HIDDEN]));
        m.insert(p("mlp.up_proj.weight"), randn(&[inter, HIDDEN]));
        m.insert(p("mlp.down_proj.weight"), randn(&[HIDDEN, inter]));
        m.insert(
            p("input_layernorm.weight"),
            Array::ones::<f32>(&[HIDDEN]).unwrap(),
        );
        m.insert(
            p("post_attention_layernorm.weight"),
            Array::ones::<f32>(&[HIDDEN]).unwrap(),
        );
    }
    let model = CausalLm::from_weights(&Weights::from_map(m), "", cfg).unwrap();

    let prompt = [3i32, 9, 14, 2, 7];
    let ids = input_ids(&prompt);
    let seq = prompt.len() as i32;

    let mut cache = model.new_cache();
    let stack = model.hidden_states(&ids, &mut cache, 0).unwrap();
    assert_eq!(
        stack.len(),
        LAYERS + 1,
        "num_layers + 1 entries (embeddings, then each layer)"
    );
    for (i, h) in stack.iter().enumerate() {
        assert_eq!(h.shape(), &[1, seq, HIDDEN], "entry {i} shape");
        assert!(host(h).iter().all(|x| x.is_finite()), "entry {i} finite");
    }

    // [0] is the input embeddings, verbatim.
    let embeds = model.embed(&ids).unwrap();
    assert!(
        max_abs_err(&host(&stack[0]), &host(&embeds)) < 1e-6,
        "entry 0 must be the input embeddings"
    );
    // Every entry must actually differ from its predecessor — otherwise a stack that pushed the
    // same handle `L + 1` times would pass every shape check above.
    for i in 1..stack.len() {
        assert!(
            max_abs_err(&host(&stack[i]), &host(&stack[i - 1])) > 1e-4,
            "entry {i} must differ from entry {}",
            i - 1
        );
    }

    // The last entry is final-normed: pushing it through the tied LM head reproduces the model's
    // own all-position logits. A raw (un-normed) last entry fails this by a wide margin.
    let mut cache2 = model.new_cache();
    let logits = model.decode_logits_all(&ids, &mut cache2, 0).unwrap();
    let from_stack = linear(
        &stack[LAYERS],
        &embed.as_dtype(stack[LAYERS].dtype()).unwrap(),
        None,
    )
    .unwrap();
    let err = max_abs_err(&host(&from_stack), &host(&logits));
    assert!(
        err < 5e-2,
        "normed last entry -> logits: max|delta| = {err}"
    );

    // Mutation: the *un-normed* last layer output would not reproduce the logits.
    let unnormed = model
        .hidden_states(&ids, &mut model.new_cache(), 0)
        .unwrap();
    let raw_last = unnormed[LAYERS - 1].clone(); // the previous entry stands in for "forgot the norm"
    let wrong = linear(&raw_last, &embed.as_dtype(raw_last.dtype()).unwrap(), None).unwrap();
    assert!(
        max_abs_err(&host(&wrong), &host(&logits)) > 1e-1,
        "an un-normed final hidden state must not reproduce the logits"
    );

    // The stack shares the decoder's cache semantics: one position per prompt token.
    assert_eq!(cache.offset(), seq);

    // Decode continues correctly from the same cache (the stack is a real forward, not a probe).
    let step = model
        .hidden_states(&input_ids(&[5]), &mut cache, seq)
        .unwrap();
    assert_eq!(step.len(), LAYERS + 1);
    assert_eq!(step[0].shape(), &[1, 1, HIDDEN]);
    assert_eq!(cache.offset(), seq + 1);

    // Batched: the encoder path prefills many prompts at once, so the batch axis must survive every
    // entry — and row 0 of a batch-2 stack must equal the batch-1 stack of the same prompt.
    let other = [11i32, 1, 4, 6, 0];
    let batched_ids = input_ids_batch(&[&prompt, &other]).unwrap();
    let mut batched_cache = model.new_cache();
    let batched = model
        .hidden_states(&batched_ids, &mut batched_cache, 0)
        .unwrap();
    assert_eq!(batched.len(), LAYERS + 1);
    for (i, h) in batched.iter().enumerate() {
        assert_eq!(h.shape(), &[2, seq, HIDDEN], "batched entry {i} shape");
    }
    let row0: Vec<f32> = host(&batched[LAYERS])[..(seq * HIDDEN) as usize].to_vec();
    let err = max_abs_err(&row0, &host(&stack[LAYERS]));
    assert!(
        err < 1e-3,
        "batch row 0 must match the batch-1 stack: {err}"
    );
}

/// A stray guard: the mask enum's new variant must not have quietly changed how the existing masks
/// behave (the enum is `Copy` and matched in several places).
#[test]
fn existing_attention_masks_are_unchanged() {
    let (heads, seq, head_dim) = (2i32, 6i32, 8i32);
    let mut rng = SplitMix64::new(7);
    let n = (heads * seq * head_dim) as usize;
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 2.0).collect();
    let t = Array::from_slice(&data, &[1, heads, seq, head_dim]);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let causal = host(&sdpa(&t, &t, &t, scale, AttnMask::Causal).unwrap());
    let none = host(&sdpa(&t, &t, &t, scale, AttnMask::None).unwrap());
    assert!(causal.iter().all(|x| x.is_finite()));
    assert!(none.iter().all(|x| x.is_finite()));
    assert!(
        max_abs_err(&causal, &none) > 1e-3,
        "causal and unmasked attention still differ"
    );
    // A single query attending a single key is a softmax over one logit, so the output is that key's
    // value row verbatim — the sanity check that the causal mask still means "see yourself".
    // The slice is cut on the **host** rather than with a lazy view: `as_slice` reads the physical
    // buffer and ignores strides, so reading a view back would not be the values the kernel saw.
    let first_rows: Vec<f32> = (0..heads as usize)
        .flat_map(|h| {
            let base = h * (seq * head_dim) as usize;
            data[base..base + head_dim as usize].to_vec()
        })
        .collect();
    let first = Array::from_slice(&first_rows, &[1, heads, 1, head_dim]);
    let first_causal = host(&sdpa(&first, &first, &first, scale, AttnMask::Causal).unwrap());
    assert!(
        max_abs_err(&first_causal, &first_rows) < 1e-4,
        "a lone causal query must return its own value row"
    );
}
