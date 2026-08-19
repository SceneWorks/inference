//! Gemma 4 unified **decoder** goldens (sc-18761).
//!
//! `tests/gemma4_primitives.rs` pins each primitive Gemma 4 adds. This pins the *assembled block*:
//! a whole tiny `Gemma4Unified` model — four layers alternating `sliding_attention` and
//! `full_attention`, two head dims (8 / 16), two GQA group counts, `attention_k_eq_v` on the full
//! layers, a per-layer `layer_scalar`, and a sliding window (3) shorter than the prompt (7).
//!
//! The fixture `crates/llm/testdata/gemma4/gemma4_decoder_goldens.json` is **owned by sc-18760**
//! (the MLX twin) and shared verbatim: both backends assert the identical numbers, so neither is
//! the other's oracle and cross-backend drift is caught by construction rather than by comparing
//! two ports to each other. Its generator is committed beside it, is numpy-only and deterministic,
//! and is transcribed from the pinned reference (`huggingface/transformers` 5.14.1
//! `gemma4_unified`, the revision `Lightricks/LTX-2` pins) with a `--verify-reference` cross-check
//! against the real module.
//!
//! Both entry points are covered: the causal-LM forward (`decode_logits_all` / `decode_logits`) and
//! the all-hidden-states forward (`hidden_states`).
//!
//! Assertions are on **absolute** error, not cosine similarity — a RoPE schedule with the right
//! shape and the wrong base is cosine-close and numerically wrong. The fixture also ships four
//! **mutation** oracles (the same forward with one plausible mistake); each is asserted to differ
//! from the truth, so a golden that a near-miss would also satisfy cannot stand.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use serde_json::Value;

use candle_llm::config::{LayerAttentionType, ModelConfig};
use candle_llm::models::CausalLm;
use candle_llm::primitives::{input_ids, KvCache, Weights};

const DECODER_GOLDENS: &str = include_str!("../../testdata/gemma4/gemma4_decoder_goldens.json");

fn goldens() -> Value {
    serde_json::from_str(DECODER_GOLDENS).expect("parse gemma4 decoder goldens")
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("float array")
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("shape array")
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn ids_of(v: &Value) -> Vec<i32> {
    v.as_array()
        .expect("id array")
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect()
}

fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Assert on **absolute** error, with the budget scaled by the golden's own magnitude (the fixture
/// is computed in f64 and the decoder runs in f32 on CPU).
fn assert_abs_close(got: &[f32], want: &[f32], rel: f32, what: &str) {
    let scale = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1.0);
    let err = max_abs_err(got, want);
    assert!(
        err <= rel * scale,
        "{what}: max|delta| = {err} exceeds {} (rel {rel} of magnitude {scale})",
        rel * scale
    );
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The **one** documented difference between this backend's numbers and the shared fixture's.
///
/// `Gemma4UnifiedTextScaledWordEmbedding` computes `sqrt(hidden_size)` and casts it to the running
/// dtype *before* the multiply, so the scale a run actually applies depends on that dtype. The
/// fixture carries the **bf16** rounding (5.65625 for hidden 32), which is what both backends apply
/// on a GPU — candle's `compute_dtype` is bf16 on CUDA and Metal, and `affine` rounds the scalar to
/// the tensor dtype. On **CPU** candle computes in f32 and therefore applies f32 `sqrt(32)`
/// (5.6568542), which is 1.07e-4 relatively *closer* to the f64 oracle, not further.
///
/// So the CPU stack's entry 0 is the golden times this ratio, exactly. Asserting that — rather than
/// widening the budget until the gap fits — keeps the claim precise: the only divergence is the
/// scale's dtype, and any *other* drift in the embedding path still fails.
fn embed_scale_ratio(g: &Value) -> f32 {
    let hidden = g["config"]["text_config"]["hidden_size"].as_f64().unwrap();
    let fixture = g["embed_scale"].as_f64().unwrap();
    ((hidden.sqrt() as f32) as f64 / fixture) as f32
}

/// Run the decoder stack from the fixture's **own** entry-0 embeddings and compare every entry.
///
/// Feeding the oracle's input rather than re-deriving it takes [`embed_scale_ratio`] out of the
/// comparison entirely: the two runs then differ only by f32-vs-f64 arithmetic, so the whole stack
/// can be held to a tight budget instead of one widened to absorb a scale that propagates through
/// the residual stream. The embedding path itself is pinned separately, as an equality, by
/// [`gemma4_embedding_scale_is_the_dtype_rounded_sqrt_hidden`].
fn assert_stack_from_golden_embeds(m: &CausalLm, want: &[Value], shape: &[usize], label: &str) {
    let embeds = Tensor::from_vec(floats(&want[0]), shape, &Device::Cpu).unwrap();
    let mut cache = m.new_cache();
    let stack = m.hidden_states_from_embeds(&embeds, &mut cache, 0).unwrap();
    assert_eq!(stack.len(), want.len(), "{label}: entry count");
    for (i, (got, expect)) in stack.iter().zip(want).enumerate() {
        assert_eq!(got.dims(), shape, "{label} hidden[{i}] shape");
        assert_abs_close(
            &host(got),
            &floats(expect),
            1e-4,
            &format!("{label} hidden[{i}]"),
        );
    }
}

/// Lift a committed `{shape, data}` weight map onto the CPU.
fn weight_map(weights: &Value) -> HashMap<String, Tensor> {
    let mut m = HashMap::new();
    for (key, spec) in weights.as_object().expect("weights object") {
        let shape = usizes(&spec["shape"]);
        let data = floats(&spec["data"]);
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "{key}: data does not fill {shape:?}"
        );
        m.insert(
            key.clone(),
            Tensor::from_vec(data, shape, &Device::Cpu).unwrap(),
        );
    }
    m
}

/// Build a decoder from a fixture's own `config` + `weights`, with `patch` applied to the
/// `text_config` first.
fn build(cfg_json: &Value, weights: &Value, patch: &[(&str, Value)]) -> Result<CausalLm, String> {
    let mut cfg_json = cfg_json.clone();
    for (key, value) in patch {
        cfg_json["text_config"][*key] = value.clone();
    }
    let cfg = ModelConfig::from_json(&cfg_json).expect("gemma4 decoder fixture config parses");
    CausalLm::from_weights(
        &Weights::from_map(weight_map(weights), Device::Cpu),
        "",
        cfg,
    )
    .map_err(|e| e.to_string())
}

fn model(g: &Value) -> CausalLm {
    match build(&g["config"], &g["weights"], &[]) {
        Ok(m) => m,
        Err(e) => panic!("the decoder must build a Gemma 4 model: {e}"),
    }
}

/// The all-position logits of a fresh forward over the fixture's prompt.
fn prefill_logits(m: &CausalLm, prompt: &[i32]) -> Vec<f32> {
    let ids = input_ids(prompt, &Device::Cpu).unwrap();
    let mut cache = m.new_cache();
    host(&m.decode_logits_all(&ids, &mut cache, 0).unwrap())
}

// ---------------------------------------------------------------------------------------------
// the two entry points
// ---------------------------------------------------------------------------------------------

/// The **all-hidden-states** forward — the entry point a text encoder consumes (LTX-2.5 stacks
/// every layer's output into its feature extractor).
///
/// `num_layers + 1` entries in Hugging Face's `output_hidden_states` layout: the scaled input
/// embeddings, each layer's output, and a **final-normed** last entry. The fixture computes the same
/// stack from the reference, so a wrong sandwich order, a wrong norm fold, a missed `layer_scalar`,
/// or a missing final norm all fail here — none of which a logits-only assertion can see.
#[test]
fn gemma4_hidden_state_stack_matches_the_shared_goldens() {
    let g = goldens();
    let m = model(&g);
    let prompt = ids_of(&g["prompt"]);
    let want_shape = usizes(&g["hidden_states"]["shape"]);

    let ids = input_ids(&prompt, &Device::Cpu).unwrap();
    let mut cache = m.new_cache();
    let stack = m.hidden_states(&ids, &mut cache, 0).unwrap();

    let want = g["hidden_states"]["layers"].as_array().unwrap();
    assert_eq!(
        stack.len(),
        g["hidden_states"]["count"].as_u64().unwrap() as usize,
        "num_layers + 1 hidden-state entries"
    );
    for h in &stack {
        assert_eq!(h.dims(), want_shape.as_slice(), "entry shape");
    }
    // A real forward, not a probe: the cache advanced one position per prompt token.
    assert_eq!(cache.offset(), prompt.len() as i32);
    // The numbers, driven from the oracle's own embeddings.
    assert_stack_from_golden_embeds(&m, want, &want_shape, "prefill");

    // Every entry must differ from its predecessor — otherwise a stack that pushed one handle
    // `L + 1` times would satisfy every shape check above.
    for i in 1..stack.len() {
        assert!(
            max_abs_err(&host(&stack[i]), &host(&stack[i - 1])) > 1e-4,
            "entry {i} must differ from entry {}",
            i - 1
        );
    }
}

/// The embedding scale is `sqrt(hidden_size)` **rounded to the running dtype** before the multiply,
/// which is the single documented reason this backend's entry 0 is not the fixture's verbatim.
///
/// Pinned as an equality rather than absorbed into a tolerance: on CPU the scale is f32
/// `sqrt(hidden)`, the fixture's is its bf16 rounding, and the ratio between them is
/// [`embed_scale_ratio`] — so `embed()` must equal the golden times exactly that, to f32 precision.
/// Any *other* change in the embedding path (a missing scale, a `1 + weight` fold, the wrong gather)
/// moves this far outside the budget.
#[test]
fn gemma4_embedding_scale_is_the_dtype_rounded_sqrt_hidden() {
    let g = goldens();
    let m = model(&g);
    let prompt = ids_of(&g["prompt"]);
    let ids = input_ids(&prompt, &Device::Cpu).unwrap();

    let hidden = g["config"]["text_config"]["hidden_size"].as_f64().unwrap();
    let fixture_scale = g["embed_scale"].as_f64().unwrap();
    // The fixture carries the bf16 rounding, which is what a GPU run of this decoder applies.
    assert!(
        (fixture_scale - hidden.sqrt()).abs() / hidden.sqrt() < 5e-3,
        "the fixture's scale must be sqrt(hidden) to within bf16 rounding"
    );
    let ratio = embed_scale_ratio(&g);
    assert!(
        (ratio - 1.0).abs() < 5e-3,
        "the two roundings differ only at bf16 precision, got ratio {ratio}"
    );

    let mut want = floats(&g["hidden_states"]["layers"][0]);
    for v in &mut want {
        *v *= ratio;
    }
    let got = host(&m.embed(&ids).unwrap());
    assert_abs_close(&got, &want, 1e-6, "scaled input embeddings");
}

/// The **causal-LM** forward: every position's logits, soft-capped at 30.0 through the tied head.
#[test]
fn gemma4_all_position_logits_match_the_shared_goldens() {
    let g = goldens();
    let m = model(&g);
    let prompt = ids_of(&g["prompt"]);
    let ids = input_ids(&prompt, &Device::Cpu).unwrap();

    let mut cache = m.new_cache();
    let logits = m.decode_logits_all(&ids, &mut cache, 0).unwrap();
    assert_eq!(logits.dims(), usizes(&g["logits"]["shape"]).as_slice());
    assert_abs_close(
        &host(&logits),
        &floats(&g["logits"]["data"]),
        1e-4,
        "prefill logits",
    );

    // The last-position forward must agree with the all-position one's last row.
    let vocab = *usizes(&g["logits"]["shape"]).last().unwrap();
    let want_last = floats(&g["logits"]["data"])[(prompt.len() - 1) * vocab..].to_vec();
    let mut cache2 = m.new_cache();
    let last = m.decode_logits(&ids, &mut cache2, 0).unwrap();
    assert_abs_close(&host(&last), &want_last, 1e-4, "prefill last logits");
}

/// A **cached decode step** continues from the prefill's cache: the sliding window now spans the
/// cache rather than the prompt, which a prefill-only golden cannot exercise. The fixture pins no
/// decode step, so this asserts the invariant instead — a one-shot forward over `prompt + [t]` and
/// a cached step after `prompt` must agree on that token's logits.
#[test]
fn gemma4_cached_decode_step_agrees_with_a_one_shot_forward() {
    let g = goldens();
    let m = model(&g);
    let prompt = ids_of(&g["prompt"]);
    let next = 6i32;

    let mut cache = m.new_cache();
    m.decode_logits(&input_ids(&prompt, &Device::Cpu).unwrap(), &mut cache, 0)
        .unwrap();
    let stepped = host(
        &m.decode_logits(
            &input_ids(&[next], &Device::Cpu).unwrap(),
            &mut cache,
            prompt.len() as i32,
        )
        .unwrap(),
    );
    assert_eq!(cache.offset(), prompt.len() as i32 + 1);

    let mut whole: Vec<i32> = prompt.clone();
    whole.push(next);
    let mut cold = m.new_cache();
    let one_shot = host(
        &m.decode_logits(&input_ids(&whole, &Device::Cpu).unwrap(), &mut cold, 0)
            .unwrap(),
    );
    assert_abs_close(&stepped, &one_shot, 1e-5, "cached step vs one-shot");
}

// ---------------------------------------------------------------------------------------------
// mutations: the fixture's own oracles for the plausible wrong decoders
// ---------------------------------------------------------------------------------------------

/// Each committed mutation is a full forward with one plausible mistake. This decoder must match
/// the **truth** and differ from every one of them — that is what proves the goldens discriminate
/// rather than merely matching something.
///
/// The four (see the fixture's `mutations[*].why`): a `layer_scalar` left at its 1.0 initializer;
/// `attention_k_eq_v` reading V off the normed+rotated key instead of the raw projection; a
/// `sliding_attention` layer running a plain causal mask; and the `full_attention` layers using a
/// leading-slice partial RoPE instead of the proportional schedule.
#[test]
fn gemma4_forward_differs_from_every_committed_mutation() {
    let g = goldens();
    let m = model(&g);
    let prompt = ids_of(&g["prompt"]);
    let got = prefill_logits(&m, &prompt);
    let truth = floats(&g["logits"]["data"]);
    assert_abs_close(&got, &truth, 1e-4, "truth");

    let mutations = g["mutations"].as_object().expect("mutation oracles");
    assert!(!mutations.is_empty(), "the fixture must ship mutations");
    for (name, m) in mutations {
        let wrong = floats(&m["logits"]);
        let delta = max_abs_err(&got, &wrong);
        assert!(
            delta > 1e-3,
            "mutation `{name}` ({}) produces the same logits as the correct forward \
             (max|delta| = {delta}); the golden does not discriminate it",
            m["why"].as_str().unwrap_or("")
        );
    }
}

/// `layer_scalar` gets its own named check because it is the one term whose *initializer* is 1.0 —
/// a port that never reads the buffer passes any fixture that left it there, and the real
/// checkpoint's values are not 1.0. The fixture's scalars are 0.9 / 1.1 / 0.95 / 1.05.
#[test]
fn gemma4_layer_scalar_is_read_and_is_not_the_initializer() {
    let g = goldens();
    let scalars = floats(&g["layer_scalars"]);
    assert!(
        scalars.iter().any(|s| (s - 1.0).abs() > 1e-3),
        "the fixture must not leave layer_scalar at its 1.0 initializer"
    );
    for (i, layer) in g["config"]["text_config"]["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let key = format!("model.layers.{i}.layer_scalar");
        assert!(
            g["weights"].get(&key).is_some(),
            "{key} must be a checkpoint weight ({layer})"
        );
    }

    let m = model(&g);
    let got = prefill_logits(&m, &ids_of(&g["prompt"]));
    let skipped = floats(&g["mutations"]["no_layer_scalar"]["logits"]);
    assert!(
        max_abs_err(&got, &skipped) > 1e-3,
        "a decoder that ignored layer_scalar would match the `no_layer_scalar` oracle"
    );
}

// ---------------------------------------------------------------------------------------------
// num_kv_shared_layers + use_double_wide_mlp
// ---------------------------------------------------------------------------------------------

/// The fixture's second model sets `num_kv_shared_layers: 2` and `use_double_wide_mlp: true`: its
/// trailing layers project **no** K/V of their own (no `k_proj`, no `v_proj`, no `k_norm` in the
/// checkpoint) and attend the keys/values of the last earlier layer of their **own type**, with a
/// double-width MLP. Layer 2 (sliding) reuses layer 0's, layer 3 (full) reuses layer 1's.
///
/// The shipped LTX-2.5 encoder leaves both off, so this is the only place the capability is
/// exercised — and a decoder that ignored the two keys would run a different model in silence on
/// any checkpoint that set them.
#[test]
fn gemma4_kv_sharing_tail_matches_the_shared_goldens() {
    let g = goldens();
    let kv = &g["kv_shared"];
    assert_eq!(kv["num_kv_shared_layers"].as_u64(), Some(2));
    assert_eq!(kv["use_double_wide_mlp"].as_bool(), Some(true));

    // The tail layers really are absent from the checkpoint, not merely unused.
    for i in 2..4 {
        for absent in ["k_proj.weight", "v_proj.weight", "k_norm.weight"] {
            let key = format!("model.layers.{i}.self_attn.{absent}");
            assert!(
                kv["weights"].get(&key).is_none(),
                "{key} must not exist on a KV-sharing layer"
            );
        }
    }

    let m = match build(&kv["config"], &kv["weights"], &[]) {
        Ok(m) => m,
        Err(e) => panic!("the decoder must build a KV-sharing Gemma 4 model: {e}"),
    };
    let prompt = ids_of(&g["prompt"]);
    let ids = input_ids(&prompt, &Device::Cpu).unwrap();

    let want = kv["hidden_states"].as_array().unwrap();
    let shape = usizes(&g["hidden_states"]["shape"]);
    assert_stack_from_golden_embeds(&m, want, &shape, "kv_shared");

    let mut cache2 = m.new_cache();
    let logits = m.decode_logits_all(&ids, &mut cache2, 0).unwrap();
    assert_abs_close(
        &host(&logits),
        &floats(&kv["logits"]),
        1e-4,
        "kv_shared logits",
    );
}

/// The donor is resolved **per layer type**: a sliding tail layer reuses the last pre-tail sliding
/// layer, never the full one (their head dims do not even match, so getting it wrong is not a
/// subtle numeric error — it is a different tensor shape).
#[test]
fn gemma4_kv_donor_is_the_last_pre_tail_layer_of_the_same_type() {
    let g = goldens();
    let cfg = ModelConfig::from_json(&g["kv_shared"]["config"]).unwrap();
    let table = cfg.gemma4.as_ref().expect("gemma 4 table");

    assert_eq!(table.first_kv_shared_layer(), Some(2));
    assert_eq!(
        (0..4).map(|i| table.is_kv_shared(i)).collect::<Vec<_>>(),
        vec![false, false, true, true]
    );
    // Layers 0 (sliding) and 1 (full) are each the last of their type before the tail.
    assert_eq!(
        (0..4)
            .map(|i| table.stores_shared_kv(i))
            .collect::<Vec<_>>(),
        vec![true, true, false, false]
    );
    assert_eq!(table.kv_donor(2), Some(0), "sliding tail -> sliding donor");
    assert_eq!(table.kv_donor(3), Some(1), "full tail -> full donor");
    assert_eq!(table.layer_type(2), LayerAttentionType::Sliding);
    assert_eq!(table.layer_type(3), LayerAttentionType::Full);
    assert_eq!(table.kv_donor(0), None);

    // `use_double_wide_mlp` applies to exactly the sharing tail.
    assert_eq!(
        (0..4)
            .map(|i| table.layer_intermediate_size(i, 48))
            .collect::<Vec<_>>(),
        vec![48, 48, 96, 96]
    );

    // A model without a tail must be entirely unaffected — `num_kv_shared_layers: 0` does not make
    // every layer "shared" even though `len - 0 == len`.
    let plain = ModelConfig::from_json(&g["config"]).unwrap();
    let plain = plain.gemma4.as_ref().unwrap();
    assert_eq!(plain.first_kv_shared_layer(), None);
    assert!((0..4).all(|i| !plain.is_kv_shared(i) && !plain.stores_shared_kv(i)));
    assert!((0..4).all(|i| plain.layer_intermediate_size(i, 48) == 48));

    // The fixture has one pre-tail layer of each type, so *first* and *last* coincide there and it
    // cannot tell the two rules apart. This schedule can: the pre-tail run is
    // [sliding, sliding, full, sliding], so the sliding donor is layer 3 under "last" and layer 0
    // under "first".
    let mut wide = g["kv_shared"]["config"].clone();
    wide["text_config"]["num_hidden_layers"] = Value::from(6);
    wide["text_config"]["layer_types"] = serde_json::json!([
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]);
    let wide = ModelConfig::from_json(&wide).unwrap();
    let wide = wide.gemma4.as_ref().unwrap();
    assert_eq!(wide.first_kv_shared_layer(), Some(4));
    assert_eq!(
        wide.kv_donor(4),
        Some(3),
        "the sliding tail takes the LAST pre-tail sliding layer, not the first"
    );
    assert_eq!(wide.kv_donor(5), Some(2), "the full tail takes layer 2");
    assert_eq!(
        (0..6).map(|i| wide.stores_shared_kv(i)).collect::<Vec<_>>(),
        vec![false, false, true, true, false, false],
        "only the last pre-tail layer of each type publishes"
    );
}

/// The sharing tail's keys/values are published **per forward**, not carried in the cache, so a
/// continued (cached) generation is refused rather than silently attending only the current step's
/// keys. A single forward from an empty cache — what a text encoder runs — is exact.
#[test]
fn gemma4_kv_sharing_refuses_a_continued_generation() {
    let g = goldens();
    let kv = &g["kv_shared"];
    let m = build(&kv["config"], &kv["weights"], &[]).expect("build");
    let prompt = ids_of(&g["prompt"]);

    let mut cache = m.new_cache();
    m.decode_logits(&input_ids(&prompt, &Device::Cpu).unwrap(), &mut cache, 0)
        .expect("the first forward from an empty cache is exact");
    let err = match m.decode_logits(
        &input_ids(&[6], &Device::Cpu).unwrap(),
        &mut cache,
        prompt.len() as i32,
    ) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a cached continuation cannot serve a KV-sharing tail"),
    };
    assert!(err.contains("num_kv_shared_layers"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// weight-key contract
// ---------------------------------------------------------------------------------------------

/// **`attention_k_eq_v` must drive the weight requirement, not just the arithmetic.** The fixture's
/// checkpoint has no `v_proj` on its `full_attention` layers at all; turning the flag off must make
/// the load *fail* for want of one. A decoder that "supported" sharing by running the key weight
/// through two projections would load either way, at twice the matmul and twice the quantized
/// footprint — which is the whole point of the flag.
#[test]
fn gemma4_k_eq_v_layers_have_no_value_projection_in_the_checkpoint() {
    let g = goldens();
    for i in [1usize, 3] {
        assert!(
            g["weights"]
                .get(format!("model.layers.{i}.self_attn.v_proj.weight"))
                .is_none(),
            "layer {i} is a full_attention layer and must ship no v_proj"
        );
    }
    let err = match build(
        &g["config"],
        &g["weights"],
        &[("attention_k_eq_v", Value::from(false))],
    ) {
        Err(e) => e,
        Ok(_) => panic!("without k_eq_v the full layers need a v_proj the checkpoint lacks"),
    };
    assert!(err.contains("v_proj"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// paths that cannot serve a per-layer-type stack
// ---------------------------------------------------------------------------------------------

/// A **paged** cache is sized once from layer 0's head shape, so an ordinary
/// `decode_logits`-on-a-paged-cache is unserviceable for Gemma 4 — and it is reachable through
/// `new_paged_cache` + `generate_with_cache`, not just the Throughput path. It must say what is
/// wrong at the first disagreeing layer rather than surfacing a raw tensor-shape mismatch.
#[test]
fn gemma4_paged_cache_names_the_per_layer_head_shape_it_cannot_hold() {
    let g = goldens();
    let m = model(&g);
    let ids = input_ids(&ids_of(&g["prompt"]), &Device::Cpu).unwrap();
    let mut cache = m.new_paged_cache(8);
    let err = match m.decode_logits(&ids, &mut cache, 0) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("one pooled head shape cannot serve two layer types"),
    };
    assert!(err.contains("head shape"), "{err}");
    assert!(err.contains("Gemma 4"), "{err}");
    // The contiguous cache, which holds each layer's own shape, does serve it.
    let mut contiguous = m.new_cache();
    m.decode_logits(&ids, &mut contiguous, 0)
        .expect("the contiguous cache holds per-layer shapes");
}

/// The continuous-batching `Throughput` path sizes one shared block pool from a single
/// `(n_kv_heads, head_dim)`; Gemma 4's layer types disagree on both, so it is refused by name (as
/// MLA is) rather than quietly attending the wrong shape. The bit-exact `Exact` mode serves it.
#[test]
fn gemma4_refuses_the_paged_throughput_path_by_name() {
    let g = goldens();
    let m = model(&g);
    let ids = input_ids(&[3], &Device::Cpu).unwrap();
    let mut cache = m.new_paged_cache(8);
    let mut caches = vec![&mut cache];
    let err = match m.decode_logits_per_seq(&ids, &mut caches, &[0]) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the single-shape paged pool cannot hold a Gemma 4 stack"),
    };
    assert!(err.contains("Gemma 4"), "{err}");
    assert!(err.contains("Exact"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// real weights
// ---------------------------------------------------------------------------------------------

/// The **real-weight** encoder forward, on the CUDA lane (`#[ignore]`; point
/// `CANDLE_LLM_GEMMA4_MODEL` at a Gemma 4 unified snapshot and run with `--features cuda`).
///
/// The deterministic CPU goldens above carry the numeric pinning — this asserts the thing only real
/// weights can: that a 48-layer stack whose per-layer table resolves to 40 sliding + 8 full actually
/// *builds* from a shipped checkpoint (the full layers' missing `v_proj`, the two `o_proj` widths,
/// the 256/512 q/k norms, the per-layer `layer_scalar`) and produces a finite, soft-capped,
/// correctly-shaped hidden-state stack and logit row.
#[test]
#[ignore = "needs a Gemma 4 unified snapshot via CANDLE_LLM_GEMMA4_MODEL"]
fn gemma4_real_weight_forward_produces_a_finite_hidden_state_stack() {
    let Some(dir) = std::env::var("CANDLE_LLM_GEMMA4_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
    else {
        eprintln!("skip: set CANDLE_LLM_GEMMA4_MODEL");
        return;
    };
    let device = candle_llm::device::select_device().expect("select the compile-time device");
    let cfg = ModelConfig::from_dir(&dir).expect("parse the snapshot's config.json");
    assert!(cfg.is_gemma4(), "not a Gemma 4 snapshot");
    let table = cfg.gemma4.as_ref().expect("gemma 4 layer table");
    let sliding = table
        .layer_types
        .iter()
        .filter(|t| **t == LayerAttentionType::Sliding)
        .count();
    assert_eq!(
        (sliding, table.layer_types.len() - sliding),
        (40, 8),
        "the shipped 48-layer schedule is 40 sliding / 8 full"
    );

    let w = Weights::from_dir(&dir, &device).expect("load weights");
    let m = CausalLm::from_weights(&w, "", cfg.clone()).expect("build the Gemma 4 decoder");

    let prompt: Vec<i32> = vec![2, 1596, 3384, 611, 1002, 3055];
    let ids = input_ids(&prompt, m.device()).unwrap();
    let mut cache = m.new_cache();
    let stack = m.hidden_states(&ids, &mut cache, 0).expect("hidden states");
    assert_eq!(stack.len(), cfg.num_layers + 1);
    for (i, h) in stack.iter().enumerate() {
        assert_eq!(
            h.dims(),
            &[1, prompt.len(), cfg.hidden_size as usize],
            "entry {i} shape"
        );
        assert!(host(h).iter().all(|x| x.is_finite()), "entry {i} finite");
    }

    let mut cache2 = m.new_cache();
    let logits = m.decode_logits(&ids, &mut cache2, 0).expect("logits");
    assert_eq!(logits.dims(), &[1, cfg.vocab_size as usize]);
    let row = host(&logits);
    assert!(row.iter().all(|x| x.is_finite()), "logits must be finite");
    // `final_logit_softcapping: 30.0` bounds every logit; an uncapped head blows past it.
    let cap = cfg
        .final_logit_softcap
        .expect("gemma 4 caps its final logits");
    assert!(
        row.iter().all(|x| x.abs() <= cap * 1.001),
        "soft-capped logits must stay within +/-{cap}"
    );
}
