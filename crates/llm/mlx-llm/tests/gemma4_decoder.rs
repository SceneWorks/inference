//! Gemma 4 unified **decoder** goldens (sc-18760).
//!
//! `gemma4_primitives.rs` pins the pieces Gemma 4 adds; this pins the assembled block. A complete
//! small Gemma-4-unified model — weights, prompt, every layer's hidden states, and the final
//! logits — is loaded from `crates/llm/testdata/gemma4/gemma4_decoder_goldens.json` and driven
//! through both entry points the story requires: the all-hidden-states forward LTX-2.5 consumes
//! and the ordinary causal-LM logits forward. The fixture is **shared with `candle-llm`**, which
//! asserts the identical numbers.
//!
//! The oracle is a numpy transcription of `huggingface/transformers` 5.14.1 `gemma4_unified` — the
//! revision `Lightricks/LTX-2` @ `d151147` pins — and was checked against the real
//! `Gemma4UnifiedTextModel` (`generate_decoder_goldens.py --verify-reference`), which it reproduces
//! to 2.0e-7 relative: float32 ULP, because the reference runs its RMSNorm and RoPE tables in
//! float32 whatever the model dtype.
//!
//! **Both layer types, and it matters.** The fixture's `layer_types` is
//! `[sliding, full, sliding, full]`. Sliding and full differ in head dim (8 vs 16), KV-head count
//! (2 vs 1), RoPE schedule (default vs proportional), mask (windowed vs plain causal), *and*
//! whether K and V share a projection — five independent things, so a sliding-only fixture would
//! be a false green on all of the second column.
//!
//! **Absolute error, never cosine.** Cosine similarity is scale-invariant, so it passes a
//! uniformly mis-scaled port — which is exactly the failure `layer_scalar` invites. Every
//! assertion here is on `max|delta|`, budgeted against the golden's own magnitude.
//!
//! **Every golden is paired with a mutation.** The fixture carries the reference output of four
//! plausible wrong implementations, and each is asserted to be *far* outside the tolerance the
//! real output sits inside. A golden that the near-miss also passes is not asking its question.

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};
use serde_json::Value;

use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::{input_ids, Weights};

const GOLDENS: &str = include_str!("../../testdata/gemma4/gemma4_decoder_goldens.json");

/// Absolute-error budget as a fraction of each tensor's own magnitude.
///
/// The decoders compute in bf16 (`COMPUTE_DTYPE`), whose 8-bit mantissa carries ~4e-3 of relative
/// error per operation. The fixture's weights are emitted *already rounded to bf16*, so this budget
/// covers arithmetic rounding only — the model this crate builds is bit-identical to the one the
/// oracle ran, and none of the gap is a different set of weights.
///
/// **Measured**, not guessed: the worst relative deviation across all five hidden states is
/// **5.8e-3** (logits 2.9e-3, stepped-decode-vs-prefill 3.9e-4) on an idle machine. The budget is
/// ~3.4x that, which is the headroom `tests/architecture_forward.rs` documents MLX's Metal kernels
/// needing under runner load (drift of ~1.6e-3 absolute observed there on values near 1).
///
/// Loose in absolute terms and still decisive: the four mutations land at 1.1e-1 to 5.8e-1
/// relative — 6x to 30x this budget. [`mutations_are_all_outside_the_tolerance`] pins that gap
/// every run rather than trusting this comment.
const ABS_TOL: f32 = 2.0e-2;

/// How far outside [`ABS_TOL`] a mutation must land for the fixture to count as discriminating.
///
/// At `ABS_TOL` 2e-2 this demands 8e-2, and the *narrowest* mutation (`no_layer_scalar`) measures
/// 1.1e-1 — so the fixture clears the bar with room, and a future edit that quietly widened the
/// budget until a real bug fit inside it would fail here first.
const MUTATION_MARGIN: f32 = 4.0;

fn goldens() -> Value {
    serde_json::from_str(GOLDENS).expect("parse gemma4 decoder goldens")
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

/// The golden's own magnitude, floored at 1.0 so a near-zero tensor still gets an honest absolute
/// budget rather than a vanishing one.
fn scale_of(want: &[f32]) -> f32 {
    want.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1.0)
}

fn assert_abs_close(got: &[f32], want: &[f32], what: &str) {
    let scale = scale_of(want);
    let err = max_abs_err(got, want);
    assert!(
        err <= ABS_TOL * scale,
        "{what}: max|delta| = {err} exceeds {} (rel {ABS_TOL} of magnitude {scale})",
        ABS_TOL * scale
    );
}

fn host(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// Build a fixture's model: its `config.json` through the ordinary parse, its weights through the
/// ordinary loader. Nothing about this path is test-only — it is exactly what a real snapshot takes.
///
/// `section` is the fixture root (the top level, or the `kv_shared` variant), both of which carry a
/// `config` and a `weights` map.
fn model_from(section: &Value) -> CausalLm {
    let cfg = ModelConfig::from_json(&section["config"]).expect("fixture config parses");
    assert!(
        cfg.is_gemma4(),
        "the fixture must exercise the Gemma 4 path"
    );
    let mut map: HashMap<String, Array> = HashMap::new();
    for (key, entry) in section["weights"].as_object().expect("weights object") {
        let shape: Vec<i32> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        map.insert(
            key.clone(),
            Array::from_slice(&floats(&entry["data"]), &shape),
        );
    }
    CausalLm::from_weights(&Weights::from_map(map), "", cfg).expect("fixture model loads")
}

fn fixture_model(g: &Value) -> CausalLm {
    model_from(g)
}

fn prompt_ids(g: &Value) -> Vec<i32> {
    g["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The two entry points the story requires
// ---------------------------------------------------------------------------------------------

/// `hidden_states` — the `output_hidden_states=True` stack LTX-2.5's feature extractor consumes —
/// matches the reference at **every** layer, both types.
///
/// Asserting only the last entry would hide a layer whose error cancels downstream; asserting each
/// one localizes a break to the layer that caused it, and because the schedule alternates, a
/// sliding-only or full-only mistake shows up as every other entry failing.
#[test]
fn gemma4_hidden_states_match_the_reference_at_every_layer() {
    let g = goldens();
    let model = fixture_model(&g);
    let ids = prompt_ids(&g);
    let want = g["hidden_states"]["layers"].as_array().unwrap();

    let mut cache = model.new_cache();
    let states = model
        .hidden_states(&input_ids(&ids), &mut cache, 0)
        .expect("hidden states forward");

    assert_eq!(
        states.len(),
        want.len(),
        "the stack must return num_layers + 1 entries (input embeddings, every layer, \
         final-normed last)"
    );
    let types = g["layer_types"].as_array().unwrap();
    for (i, (got, expected)) in states.iter().zip(want).enumerate() {
        let label = match i {
            0 => "input embeddings".to_string(),
            _ => format!("layer {} ({})", i - 1, types[i - 1].as_str().unwrap_or("?")),
        };
        assert_abs_close(
            &host(got),
            &floats(expected),
            &format!("hidden_states[{i}] — {label}"),
        );
    }
}

/// The standard causal-LM logits forward — the prompt-enhancer path — matches the reference at
/// every position, through the tied `lm_head` and the `final_logit_softcapping: 30.0` tanh.
#[test]
fn gemma4_logits_match_the_reference_at_every_position() {
    let g = goldens();
    let model = fixture_model(&g);
    let ids = prompt_ids(&g);

    let mut cache = model.new_cache();
    let logits = model
        .decode_logits_all(&input_ids(&ids), &mut cache, 0)
        .expect("all-position logits forward");
    assert_abs_close(&host(&logits), &floats(&g["logits"]["data"]), "logits");
}

/// The two forwards agree where they overlap: the last position's logits from the single-position
/// entry point equal the last row of the all-position one.
///
/// This is the seam that lets the hidden-state stack and the logits stack share one decoder loop.
/// If they ever diverge, one of them is running a different model.
#[test]
fn gemma4_last_position_logits_agree_between_entry_points() {
    let g = goldens();
    let model = fixture_model(&g);
    let ids = prompt_ids(&g);
    let vocab = model.config().vocab_size as usize;

    let mut cache_all = model.new_cache();
    let all = host(
        &model
            .decode_logits_all(&input_ids(&ids), &mut cache_all, 0)
            .expect("all-position forward"),
    );
    let mut cache_last = model.new_cache();
    let last = host(
        &model
            .decode_logits(&input_ids(&ids), &mut cache_last, 0)
            .expect("last-position forward"),
    );

    let tail = &all[all.len() - vocab..];
    assert_eq!(last.len(), vocab);
    assert_abs_close(&last, tail, "last-position logits vs the all-position tail");
}

// ---------------------------------------------------------------------------------------------
// The batched seam: one caller-built RoPE pair cannot serve two head dims
// ---------------------------------------------------------------------------------------------

/// `decode_logits_masked` must **refuse** a Gemma 4 model.
///
/// It takes one caller-built `(cos, sin)` pair. That is every uniform architecture's entire RoPE,
/// but Gemma 4's `full_attention` layers rotate a 512-wide head against the sliding layers' 256 —
/// no single pair can carry both. Accepting it would silently rotate eight of the forty-eight
/// layers with the wrong table and the wrong frequencies, and the output would still be finite,
/// still be plausibly-scaled, and still be wrong. The refusal is the contract; this pins it.
#[test]
fn gemma4_is_refused_by_the_single_rope_table_entry_point() {
    let g = goldens();
    let model = fixture_model(&g);
    let ids = prompt_ids(&g);
    let s = ids.len() as i32;

    // The primary (sliding) tables, exactly what a uniform caller would have built.
    let (cos, sin) = model
        .rope_tables(&(0..s).collect::<Vec<_>>(), 1, s)
        .expect("primary rope tables");
    let mask = Array::zeros::<f32>(&[1, 1, s, s])
        .unwrap()
        .as_dtype(model.compute_dtype())
        .unwrap();

    let err = model
        .decode_logits_masked(&input_ids(&ids), &mut model.new_cache(), &cos, &sin, &mask)
        .expect_err("a Gemma 4 model must not accept a single RoPE pair");
    let m = err.to_string();
    assert!(
        m.contains("decode_logits_masked_at"),
        "the refusal must name the entry point that does work: {m}"
    );
    assert!(
        m.contains("Gemma 4"),
        "the refusal must say what it is refusing: {m}"
    );
}

/// A left-padded batched prefill through `decode_logits_masked_at` reproduces each row's
/// single-sequence logits.
///
/// This is the only test that reaches the `AttnMask::Additive` arm of the per-layer window logic:
/// a sliding layer must intersect its window with the caller's padding mask (they are both
/// "0 keeps, a large negative blocks", so the intersection is their sum) and hand the fused kernel
/// a mask in the query dtype — an f32 mask does not promote to bf16 and the kernel rejects it
/// outright. Nothing else in the suite passes an explicit mask to a Gemma 4 model.
///
/// It also pins the per-row RoPE: the two rows sit at different pad offsets, so every layer type's
/// table has to be built from the row's own positions rather than a shared arange.
///
/// Rows are compared against `decode_logits_all` on the *unpadded* prompt, so padding that leaked
/// into attention, a window applied at the wrong offset, or a dropped second RoPE table all fail.
#[test]
fn gemma4_left_padded_batch_matches_per_row_single_sequence_logits() {
    let g = goldens();
    let model = fixture_model(&g);
    let full = prompt_ids(&g);
    let vocab = model.config().vocab_size as usize;

    // Row 0 is the whole prompt; row 1 is a shorter one, left-padded to the same width. The shorter
    // row is what makes the padding mask load-bearing.
    let rows: [Vec<i32>; 2] = [full.clone(), full[..4].to_vec()];
    let width = full.len() as i32;
    assert!(
        rows[1].len() < full.len(),
        "the second row must be shorter or nothing is padded"
    );

    // The left-padded batch, its per-row absolute positions, and its additive mask — the same
    // construction `decode::batch`'s prefill uses.
    let mut ids: Vec<i32> = Vec::new();
    let mut positions: Vec<i32> = Vec::new();
    let mut mask: Vec<f32> = Vec::new();
    for row in &rows {
        let pad = width - row.len() as i32;
        for c in 0..width {
            if c < pad {
                ids.push(0); // pad id
                positions.push(0);
            } else {
                ids.push(row[(c - pad) as usize]);
                positions.push(c - pad);
            }
        }
        for i in 0..width {
            for j in 0..width {
                // Causal, and a key is attendable only if it is a real token. The diagonal is
                // always kept so a pure-padding query row is never fully masked (whose softmax
                // would be NaN); those rows are discarded anyway.
                let ok = j <= i && (j >= pad || j == i);
                mask.push(if ok { 0.0 } else { f32::NEG_INFINITY });
            }
        }
    }
    let batch = rows.len() as i32;
    let ids = Array::from_slice(&ids, &[batch, width]);
    let mask = Array::from_slice(&mask, &[batch, 1, width, width])
        .as_dtype(model.compute_dtype())
        .unwrap();

    let mut cache = model.new_cache();
    let batched = host(
        &model
            .decode_logits_masked_at(&ids, &mut cache, &positions, &mask)
            .expect("batched left-padded prefill"),
    );
    assert_eq!(batched.len(), rows.len() * vocab);

    for (r, row) in rows.iter().enumerate() {
        let mut solo_cache = model.new_cache();
        let solo = host(
            &model
                .decode_logits_all(&input_ids(row), &mut solo_cache, 0)
                .expect("single-sequence forward"),
        );
        let want = &solo[solo.len() - vocab..];
        let got = &batched[r * vocab..(r + 1) * vocab];
        assert_abs_close(
            got,
            want,
            &format!(
                "batch row {r} (pad {}) vs its single-sequence logits",
                width - row.len() as i32
            ),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Cached decode: the per-layer window and the two RoPE schedules must survive an offset
// ---------------------------------------------------------------------------------------------

/// Prefilling the whole prompt and prefilling a prefix then decoding the rest one token at a time
/// must reach the same final hidden state.
///
/// This is where a per-layer mask or RoPE schedule that is only correct at `offset == 0` breaks:
/// a sliding layer's window is bottom-right aligned over the *cached* keys, so a decode step at
/// offset `n` must still see exactly its last `sliding_window` positions, and each layer type must
/// keep reading its own RoPE table as the offset advances.
#[test]
fn gemma4_cached_decode_matches_a_single_prefill() {
    let g = goldens();
    let model = fixture_model(&g);
    let ids = prompt_ids(&g);
    let split = 4;
    assert!(
        split < ids.len(),
        "the fixture prompt must be long enough to split"
    );

    let mut whole = model.new_cache();
    let want = host(
        model
            .hidden_states(&input_ids(&ids), &mut whole, 0)
            .expect("single prefill")
            .last()
            .expect("final state"),
    );
    let hidden = model.config().hidden_size as usize;
    let want_last = &want[want.len() - hidden..];

    let mut stepped = model.new_cache();
    model
        .hidden_states(&input_ids(&ids[..split]), &mut stepped, 0)
        .expect("prefix prefill");
    let mut got_last = Vec::new();
    for (i, tok) in ids[split..].iter().enumerate() {
        let states = model
            .hidden_states(&input_ids(&[*tok]), &mut stepped, (split + i) as i32)
            .expect("decode step");
        got_last = host(states.last().expect("final state"));
    }

    assert_abs_close(
        &got_last,
        want_last,
        "final hidden state: stepped decode vs one prefill",
    );
}

// ---------------------------------------------------------------------------------------------
// Mutation coverage — proving the goldens discriminate
// ---------------------------------------------------------------------------------------------

/// Every mutation the fixture carries must land **far** outside [`ABS_TOL`], while the real output
/// sits inside it.
///
/// Without this the tolerance is unjustified: a budget wide enough to absorb bf16 noise might also
/// absorb a real bug, and nothing in the passing test would say. The four mutations are the four
/// most plausible ways to port this block wrongly:
///
/// * `no_layer_scalar` — never read the `layer_scalar` buffer, leaving it at its `ones`
///   initializer. This is the one cosine similarity cannot see at all.
/// * `v_from_normed_k` — under `attention_k_eq_v`, take V from the normed *and rotated* key
///   instead of the raw projection output.
/// * `no_window` — run `sliding_attention` layers on a plain causal mask.
/// * `full_rope_as_partial` — read `partial_rotary_factor` as a leading-slice partial RoPE rather
///   than the proportional schedule.
#[test]
fn mutations_are_all_outside_the_tolerance() {
    let g = goldens();
    let model = fixture_model(&g);
    let ids = prompt_ids(&g);
    let want = g["hidden_states"]["layers"].as_array().unwrap();

    let mut cache = model.new_cache();
    let states = model
        .hidden_states(&input_ids(&ids), &mut cache, 0)
        .expect("hidden states forward");

    /// The worst per-entry error, as a fraction of that entry's own magnitude.
    ///
    /// Judged across **every** hidden state rather than only the last, because the last entry is
    /// the final-normed one and RMSNorm is scale-invariant: a uniform mis-scaling — precisely what
    /// dropping `layer_scalar` is — is largely normalized away there while being glaring one layer
    /// earlier. Scoring only the last entry would understate exactly the mutation this fixture
    /// most needs to catch.
    fn worst_relative(got: &[Vec<f32>], want: &[Value]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| {
                let w = floats(w);
                max_abs_err(g, &w) / scale_of(&w)
            })
            .fold(0.0f32, f32::max)
    }

    let got: Vec<Vec<f32>> = states.iter().map(host).collect();
    let real = worst_relative(&got, want);
    assert!(
        real <= ABS_TOL,
        "the real forward is {real} (rel) from the reference, outside the {ABS_TOL} budget"
    );

    let mutations = g["mutations"].as_object().expect("mutations object");
    assert_eq!(
        mutations.len(),
        4,
        "the fixture must carry all four mutations"
    );
    for (name, entry) in mutations {
        let mutated: Vec<Vec<f32>> = entry["hidden_states"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        let err = worst_relative(&mutated, want);
        assert!(
            err > ABS_TOL * MUTATION_MARGIN,
            "mutation {name:?} ({}) deviates from the golden by only {err} (rel), within \
             {MUTATION_MARGIN}x the {ABS_TOL} budget the real forward sits inside at {real} — the \
             fixture does not discriminate it, so passing proves nothing",
            entry["why"].as_str().unwrap_or("")
        );
    }
}

/// The fixture's schedule really is interleaved, and the decoder really did resolve two different
/// shapes from it.
///
/// A fixture whose layers all came out the same shape would pass every numeric assertion above
/// while testing half of what the story is about, and nothing else here would notice.
#[test]
fn the_fixture_exercises_two_genuinely_different_layer_shapes() {
    let g = goldens();
    let cfg = ModelConfig::from_json(&g["config"]).expect("fixture config parses");
    let table = cfg.gemma4.as_ref().expect("a Gemma 4 table");

    let types: Vec<&str> = g["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    assert!(
        types.contains(&"sliding_attention") && types.contains(&"full_attention"),
        "a single-type fixture cannot pin per-layer-type attention: {types:?}"
    );

    // The five things that differ between the types — every one of them is a way to be wrong.
    assert_ne!(
        table.sliding.head_dim, table.full.head_dim,
        "head dim must differ"
    );
    assert_ne!(
        table.sliding.num_kv_heads, table.full.num_kv_heads,
        "KV-head count must differ"
    );
    assert_ne!(
        table.sliding.rope_type, table.full.rope_type,
        "RoPE schedule must differ"
    );
    assert!(
        table.sliding.sliding_window.is_some() && table.full.sliding_window.is_none(),
        "only the sliding layers may carry a window"
    );
    assert!(
        !table.sliding.k_eq_v && table.full.k_eq_v,
        "`attention_k_eq_v` gates the full layers only"
    );

    // And the window must be narrow enough to actually clip the fixture's prompt, or the
    // `no_window` mutation would be a no-op.
    let window = table.sliding.sliding_window.unwrap();
    assert!(
        (window as usize) < g["prompt"].as_array().unwrap().len(),
        "window {window} does not bite on this prompt — the sliding mask would be untested"
    );
}

/// `layer_scalar` is loaded per layer, and none of the fixture's values is the `1.0` initializer.
///
/// A fixture whose scalars were all 1.0 would pass identically whether the decoder read the buffer
/// or ignored it entirely — which is the single most likely way to port this block wrongly, since
/// it is a *buffer* rather than a parameter and does not appear in the config at all.
#[test]
fn layer_scalars_are_all_non_identity_in_the_fixture() {
    let g = goldens();
    for (i, s) in g["layer_scalars"].as_array().unwrap().iter().enumerate() {
        let v = s.as_f64().unwrap();
        assert!(
            (v - 1.0).abs() > 0.01,
            "layer {i}'s scalar is {v}, indistinguishable from the initializer"
        );
    }
    // And the weight map really carries them, under the key the loader reads.
    for i in 0..g["layer_types"].as_array().unwrap().len() {
        let key = format!("model.layers.{i}.layer_scalar");
        assert!(
            g["weights"].get(&key).is_some(),
            "the fixture must ship {key} or the decoder has nothing to load"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// `num_kv_shared_layers` + `use_double_wide_mlp`
// ---------------------------------------------------------------------------------------------

/// The KV-sharing tail matches the reference, with a double-width MLP.
///
/// The shipped LTX-2.5 encoder sets `num_kv_shared_layers: 0` and `use_double_wide_mlp: false`, so
/// nothing in the real-weights path exercises either. The **config layer parses both**, which means
/// a decoder that ignored them would not refuse such a checkpoint — it would quietly run a
/// different model. This fixture is the only thing standing between that and a silent wrong answer.
///
/// With 4 layers and 2 shared, the tail spans both types: layer 2 (`sliding`) must reuse layer 0's
/// K/V and layer 3 (`full`) must reuse layer 1's. Reusing the immediately-preceding layer instead —
/// the obvious wrong reading — crosses the types and changes the numbers.
#[test]
fn gemma4_kv_shared_tail_matches_the_reference() {
    let g = goldens();
    let section = &g["kv_shared"];
    let model = model_from(section);
    let ids = prompt_ids(&g);
    let want = section["hidden_states"].as_array().unwrap();

    let mut cache = model.new_cache();
    let states = model
        .hidden_states(&input_ids(&ids), &mut cache, 0)
        .expect("kv-shared hidden states forward");
    assert_eq!(states.len(), want.len());
    for (i, (got, expected)) in states.iter().zip(want).enumerate() {
        assert_abs_close(
            &host(got),
            &floats(expected),
            &format!("kv_shared hidden_states[{i}]"),
        );
    }

    let mut cache = model.new_cache();
    let logits = model
        .decode_logits_all(&input_ids(&ids), &mut cache, 0)
        .expect("kv-shared logits forward");
    assert_abs_close(
        &host(&logits),
        &floats(&section["logits"]),
        "kv_shared logits",
    );
}

/// The KV-shared fixture really omits the tail layers' key/value weights, and really widens their
/// MLP — otherwise it would be the base fixture under a different name.
#[test]
fn the_kv_shared_fixture_omits_tail_kv_weights_and_widens_its_mlp() {
    let g = goldens();
    let section = &g["kv_shared"];
    let shared = section["num_kv_shared_layers"].as_u64().unwrap() as usize;
    let layers = g["layer_types"].as_array().unwrap().len();
    let first_shared = layers - shared;
    assert!(
        shared > 0 && first_shared > 0,
        "the tail must be a real suffix"
    );

    let w = &section["weights"];
    let base_inter = g["config"]["text_config"]["intermediate_size"]
        .as_i64()
        .unwrap();
    for i in 0..layers {
        let tail = i >= first_shared;
        for suffix in ["k_proj.weight", "k_norm.weight"] {
            let key = format!("model.layers.{i}.self_attn.{suffix}");
            assert_eq!(
                w.get(&key).is_none(),
                tail,
                "layer {i}: {key} presence must be the inverse of its KV-sharing"
            );
        }
        // `use_double_wide_mlp` doubles `intermediate_size` for exactly the sharing tail.
        let gate = &w[format!("model.layers.{i}.mlp.gate_proj.weight")]["shape"];
        let inter = gate.as_array().unwrap()[0].as_i64().unwrap();
        let want = if tail { base_inter * 2 } else { base_inter };
        assert_eq!(inter, want, "layer {i}: MLP width");
    }

    // Both layer types must appear in the tail, or only one sharing path is covered.
    let types = g["layer_types"].as_array().unwrap();
    let tail_types: Vec<&str> = types[first_shared..]
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    assert!(
        tail_types.contains(&"sliding_attention") && tail_types.contains(&"full_attention"),
        "the sharing tail must span both layer types, got {tail_types:?}"
    );
}

/// The `full_attention` layers ship **no** `v_proj` — `attention_k_eq_v` means the weight does not
/// exist — while the `sliding_attention` layers do.
///
/// This is a fixture-shape assertion rather than a numeric one: a decoder that "supported"
/// `k_eq_v` by loading the key weight twice would produce identical numbers and quietly double the
/// KV-projection footprint, so no output comparison can catch it. The absent weight can.
#[test]
fn only_the_sliding_layers_ship_a_value_projection() {
    let g = goldens();
    let types = g["layer_types"].as_array().unwrap();
    for (i, kind) in types.iter().enumerate() {
        let key = format!("model.layers.{i}.self_attn.v_proj.weight");
        let present = g["weights"].get(&key).is_some();
        match kind.as_str().unwrap() {
            "sliding_attention" => assert!(present, "sliding layer {i} must have its own {key}"),
            "full_attention" => assert!(
                !present,
                "full layer {i} shares K/V — {key} must not exist in the checkpoint"
            ),
            other => panic!("unexpected layer type {other:?}"),
        }
    }
}
