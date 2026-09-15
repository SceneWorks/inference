//! Regression fixture: every already-shipped `Architecture` still parses and derives identically
//! after sc-18769 extended the shared `ModelConfig` with Gemma 4's per-layer-type attention table.
//!
//! `ModelConfig` is the struct **every** LLM architecture in this crate shares, so a change to it
//! has the widest blast radius in the crate — and a regression here is silent: a slightly wrong RoPE
//! base or attention scale still produces finite logits and still decodes text, just worse. The
//! smoke suites (`tests/breadth.rs`) would stay green through it.
//!
//! The fixture (`crates/llm/testdata/architectures/architecture_regression.json`) is **shared with
//! `candle-llm`**, which asserts the identical numbers, so the two backends cannot drift apart
//! either. Its expectations are an independent oracle computed from the documented formulas by
//! `generate_regression.py`, not read back out of this crate.

use std::collections::BTreeSet;

use serde_json::Value;

use mlx_llm::config::{Architecture, LayerAttentionType, ModelConfig, RopeType};

const FIXTURE: &str = include_str!("../../testdata/architectures/architecture_regression.json");

fn f32_at(v: &Value, key: &str) -> f32 {
    v.get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("fixture missing f64 `{key}`")) as f32
}

fn i32_at(v: &Value, key: &str) -> i32 {
    v.get(key)
        .and_then(Value::as_i64)
        .unwrap_or_else(|| panic!("fixture missing int `{key}`")) as i32
}

fn bool_at(v: &Value, key: &str) -> bool {
    v.get(key)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("fixture missing bool `{key}`"))
}

fn floats(v: &Value, key: &str) -> Vec<f32> {
    v.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("fixture missing array `{key}`"))
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// Relative tolerance for the derived floats. The oracle computes in f64 and this crate in f32, so
/// the schedules (llama3 / YaRN) can differ in the last f32 ulp; anything larger is a real change.
fn assert_close(got: f32, want: f32, what: &str) {
    let tol = 1e-6 * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (|delta| = {})",
        (got - want).abs()
    );
}

/// The fixture must cover **exactly** the architectures the shared [`ModelConfig`] serves — no more,
/// no fewer.
///
/// A `>= N` floor lets a newly-added architecture land uncovered, which is the regression this whole
/// file exists to prevent: the guard has to fail when the enum grows, not merely when it shrinks.
/// `Qwen35` is excluded because `ModelConfig::from_json` declines it (it has its own config and
/// model); the Gemma 4 variants are excluded because the generic decoder refuses them until
/// sc-18760 / sc-18761. Any other new variant must appear here **and** in the fixture.
fn assert_family_coverage(cases: &[Value]) {
    let covered: BTreeSet<&str> = cases
        .iter()
        .map(|c| c["expect"]["family"].as_str().expect("case family"))
        .collect();
    let expected: BTreeSet<&str> = [
        Architecture::Llama,
        Architecture::Qwen3,
        Architecture::Phi3,
        Architecture::Qwen2Moe,
        Architecture::Gemma2,
        Architecture::Glm4,
        Architecture::DeepseekV2,
        Architecture::Qwen3Vl,
    ]
    .into_iter()
    .map(Architecture::family)
    .collect();
    assert_eq!(
        covered, expected,
        "the regression fixture must cover exactly the generic-decoder architectures"
    );
    // And the excluded variants must still be excluded for the stated reason.
    assert!(
        !covered.contains(Architecture::Qwen35.family()),
        "Qwen3.6 is served by Qwen35Config, not ModelConfig"
    );
    for gemma4 in [Architecture::Gemma4Unified, Architecture::Gemma4] {
        assert!(
            gemma4.is_gemma4() && !covered.contains(gemma4.family()),
            "{gemma4:?} has no generic decoder yet (sc-18760 / sc-18761)"
        );
    }
}

#[test]
fn every_shipped_architecture_parses_and_derives_unchanged() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("parse regression fixture");
    let cases = fixture["cases"].as_array().expect("cases array");
    assert_family_coverage(cases);

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let expect = &case["expect"];
        let cfg = ModelConfig::from_json(&case["config"])
            .unwrap_or_else(|e| panic!("{name}: config must still parse: {e}"));

        assert_eq!(
            cfg.architecture.family(),
            expect["family"].as_str().unwrap(),
            "{name}: family"
        );
        assert_eq!(cfg.head_dim, i32_at(expect, "head_dim"), "{name}: head_dim");
        assert_eq!(
            cfg.num_heads,
            i32_at(expect, "num_heads"),
            "{name}: num_heads"
        );
        assert_eq!(
            cfg.num_kv_heads,
            i32_at(expect, "num_kv_heads"),
            "{name}: num_kv_heads"
        );
        assert_eq!(cfg.groups(), i32_at(expect, "groups"), "{name}: groups");
        assert_close(
            cfg.partial_rotary_factor,
            f32_at(expect, "partial_rotary_factor"),
            &format!("{name}: partial_rotary_factor"),
        );
        assert_eq!(
            cfg.rotary_dim(),
            i32_at(expect, "rotary_dim"),
            "{name}: rotary_dim"
        );
        assert_close(
            cfg.attn_scale(),
            f32_at(expect, "attn_scale"),
            &format!("{name}: attn_scale"),
        );
        assert_eq!(cfg.is_moe(), bool_at(expect, "is_moe"), "{name}: is_moe");
        assert_eq!(cfg.is_mla(), bool_at(expect, "is_mla"), "{name}: is_mla");
        assert_eq!(
            cfg.has_qk_norm(),
            bool_at(expect, "has_qk_norm"),
            "{name}: has_qk_norm"
        );
        assert_eq!(
            cfg.architecture.is_sandwich(),
            bool_at(expect, "is_sandwich"),
            "{name}: is_sandwich"
        );
        assert_eq!(
            cfg.tie_word_embeddings,
            bool_at(expect, "tie_word_embeddings"),
            "{name}: tie_word_embeddings"
        );
        // No pre-Gemma-4 architecture may acquire the new table.
        assert_eq!(
            cfg.is_gemma4(),
            bool_at(expect, "is_gemma4"),
            "{name}: is_gemma4"
        );
        assert!(cfg.gemma4.is_none(), "{name}: must carry no Gemma 4 table");

        // The RoPE the decoder actually builds — dim, pairing, and both ends of the schedule.
        let rope = cfg.build_rope();
        assert_eq!(rope.dim(), i32_at(expect, "rope_dim"), "{name}: rope dim");
        assert_eq!(
            rope.interleaved(),
            bool_at(expect, "rope_interleaved"),
            "{name}: rope interleaved"
        );
        let inv = rope.inv_freq();
        assert_eq!(
            inv.len(),
            (i32_at(expect, "rope_dim") / 2) as usize,
            "{name}: inv_freq length"
        );
        for (i, want) in floats(expect, "rope_inv_freq_head").iter().enumerate() {
            assert_close(inv[i], *want, &format!("{name}: inv_freq[{i}]"));
        }
        let tail = floats(expect, "rope_inv_freq_tail");
        for (k, want) in tail.iter().enumerate() {
            let i = inv.len() - tail.len() + k;
            assert_close(inv[i], *want, &format!("{name}: inv_freq[{i}] (tail)"));
        }

        // Optional per-family blocks.
        if let Some(cap) = expect.get("attn_logit_softcap").and_then(Value::as_f64) {
            assert_eq!(
                cfg.attn_logit_softcap,
                Some(cap as f32),
                "{name}: attn_logit_softcap"
            );
        }
        if let Some(cap) = expect.get("final_logit_softcap").and_then(Value::as_f64) {
            assert_eq!(
                cfg.final_logit_softcap,
                Some(cap as f32),
                "{name}: final_logit_softcap"
            );
        }
        if let Some(sec) = expect.get("mrope_section").and_then(Value::as_array) {
            let want: Vec<i32> = sec.iter().map(|x| x.as_i64().unwrap() as i32).collect();
            assert_eq!(
                cfg.mrope_section,
                Some([want[0], want[1], want[2]]),
                "{name}: mrope_section"
            );
        }
        if let Some(moe) = cfg.moe {
            assert_eq!(
                moe.num_experts,
                i32_at(expect, "moe_num_experts") as usize,
                "{name}: moe experts"
            );
            assert_eq!(
                moe.num_experts_per_tok,
                i32_at(expect, "moe_experts_per_tok") as usize,
                "{name}: moe experts_per_tok"
            );
            assert_eq!(
                moe.moe_intermediate_size,
                i32_at(expect, "moe_intermediate_size"),
                "{name}: moe intermediate"
            );
            assert_eq!(
                moe.shared_expert_intermediate_size,
                i32_at(expect, "moe_shared_expert_intermediate_size"),
                "{name}: moe shared intermediate"
            );
            assert_eq!(
                moe.first_k_dense_replace,
                i32_at(expect, "moe_first_k_dense_replace") as usize,
                "{name}: moe first_k_dense_replace"
            );
        }
        if let Some(mla) = cfg.mla {
            assert_eq!(
                mla.qk_nope_head_dim,
                i32_at(expect, "mla_qk_nope_head_dim"),
                "{name}: mla nope"
            );
            assert_eq!(
                mla.qk_rope_head_dim,
                i32_at(expect, "mla_qk_rope_head_dim"),
                "{name}: mla rope"
            );
            assert_eq!(
                mla.v_head_dim,
                i32_at(expect, "mla_v_head_dim"),
                "{name}: mla v"
            );
            assert_eq!(
                mla.kv_lora_rank,
                i32_at(expect, "mla_kv_lora_rank"),
                "{name}: mla kv_lora_rank"
            );
        }
    }
}

/// The per-layer accessors sc-18769 introduces must be *pure additions* for a uniform model: every
/// layer of every pre-Gemma-4 architecture resolves to exactly the scalars the decoder read before.
///
/// This is the assertion that would fail if the new table were ever wired in as the primary source
/// and the scalar fallback drifted — the failure mode a fixture of parsed values alone cannot see,
/// because the parse would still be right while what the decoder *reads* would not.
#[test]
fn uniform_architectures_resolve_every_layer_to_the_scalar_shape() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("parse regression fixture");
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cfg = ModelConfig::from_json(&case["config"]).unwrap();
        let whole_model_rope = cfg.build_rope();
        for layer in 0..cfg.num_layers {
            let la = cfg.layer_attention(layer);
            assert_eq!(cfg.layer_type(layer), LayerAttentionType::Full, "{name}");
            assert_eq!(la.head_dim, cfg.head_dim, "{name}: layer {layer} head_dim");
            assert_eq!(
                la.num_kv_heads, cfg.num_kv_heads,
                "{name}: layer {layer} kv heads"
            );
            assert_eq!(la.rope_type, RopeType::Default, "{name}: layer {layer}");
            assert_eq!(
                la.rope_theta, cfg.rope_theta,
                "{name}: layer {layer} rope_theta"
            );
            assert_eq!(
                la.partial_rotary_factor, cfg.partial_rotary_factor,
                "{name}: layer {layer} partial"
            );
            assert!(
                la.sliding_window.is_none(),
                "{name}: layer {layer} must not be windowed"
            );
            assert!(!la.k_eq_v, "{name}: layer {layer} must not share k/v");
            assert_eq!(
                cfg.layer_groups(layer),
                cfg.groups(),
                "{name}: layer {layer} groups"
            );
            assert_eq!(
                cfg.layer_head_dim(layer),
                cfg.head_dim,
                "{name}: layer {layer} head dim"
            );
            assert_eq!(
                cfg.layer_sliding_window(layer),
                None,
                "{name}: layer {layer} window"
            );
            // The per-layer RoPE must be the whole-model RoPE, verbatim — including the llama3 /
            // YaRN / interleaved schedules the per-layer-type table cannot express.
            let lr = cfg.layer_rope(layer);
            assert_eq!(lr.dim(), whole_model_rope.dim(), "{name}: layer {layer}");
            assert_eq!(
                lr.interleaved(),
                whole_model_rope.interleaved(),
                "{name}: layer {layer}"
            );
            assert_eq!(
                lr.inv_freq(),
                whole_model_rope.inv_freq(),
                "{name}: layer {layer} inv_freq"
            );
        }
    }
}

/// The Gemma-2 → Gemma 4 predicate split. `is_gemma2` (which drives the `(1 + weight)` norm fold)
/// must still mean *Gemma-2 only*; the new `is_gemma` must not leak the fold to Gemma 4.
#[test]
fn gemma_predicates_did_not_widen_the_norm_fold() {
    for arch in [
        Architecture::Llama,
        Architecture::Qwen3,
        Architecture::Phi3,
        Architecture::Qwen2Moe,
        Architecture::Glm4,
        Architecture::DeepseekV2,
        Architecture::Qwen3Vl,
        Architecture::Gemma4Unified,
        Architecture::Gemma4,
    ] {
        assert!(!arch.norm_unit_offset(), "{arch:?} must not fold +1");
        assert!(!arch.is_gemma2(), "{arch:?} is not Gemma-2");
    }
    assert!(Architecture::Gemma2.norm_unit_offset());
    assert!(Architecture::Gemma2.is_gemma2());
    assert!(Architecture::Gemma2.is_gemma());
    assert!(!Architecture::Gemma2.is_gemma4());
    for g4 in [Architecture::Gemma4Unified, Architecture::Gemma4] {
        assert!(g4.is_gemma4());
        assert!(g4.is_gemma(), "Gemma 4 shares the embed scale + GeGLU");
        assert!(g4.has_v_norm(), "Gemma 4 alone has a scale-free value norm");
        assert!(g4.nests_text_config());
    }
    assert!(!Architecture::Gemma2.has_v_norm());
}
