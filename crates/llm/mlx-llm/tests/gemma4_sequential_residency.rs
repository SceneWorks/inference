//! Gemma 4 decoder stack under [`EncoderResidency::Sequential`] (sc-18798).
//!
//! LTX-2.5's text phase binds its peak — a 26.3 GB Gemma 4 encoder against a ~10.6 GiB q4 DiT — so
//! the lever that moves LTX's memory is streaming the encoder's layer stack, not bounding the DiT
//! harder. `mlx_llm::residency` implements that; this file is its regression.
//!
//! The model under test is the real `gemma4_decoder_goldens.json` fixture `gemma4_decoder.rs`
//! pins: a 4-layer `gemma4_unified` whose `layer_types` is `[sliding, full, sliding, full]`, so
//! both attention shapes, both RoPE schedules and the shared-K/V projection all run. It is written
//! to a temp `.safetensors` file because `from_file_sequential` reopens its source on every
//! forward — that reopen *is* the mechanism, so a test that handed it an in-memory map would not be
//! testing it.
//!
//! # The three questions, and why each needs its own shape of assertion
//!
//! 1. **Did the streamed loader run, and is it absent under `Resident`?** Not answerable from
//!    output. A correct stream is numerically identical to a resident pass by construction — that
//!    is [`streamed_stack_reproduces_the_resident_one`] — so output comparison cannot distinguish
//!    "streamed" from "resident", nor either from a stream that silently fell back. The claim has
//!    to be observed on the loader: [`loader_identity_streamed_ran_and_resident_did_not`].
//!
//! 2. **Did the loaded layers actually participate?** This is the one that matters. A streamable
//!    encoder that materializes nothing still returns `num_layers + 1` correctly shaped, finite,
//!    non-zero, prompt-varying hidden states — they are just the bare token embeddings repeated.
//!    Downstream that is "plausible images that ignore the prompt", and **no memory assertion sees
//!    it**: the memory numbers get *better*. [`prompt_context_reaches_the_shared_suffix`] is the
//!    guard, and its discriminator is built so an empty stack cannot pass it — see that test.
//!
//! 3. **Does a streamed model refuse the paths it is not wired for?** An empty resident `Vec` would
//!    have let every decode path run a zero-layer stack silently.
//!    [`decode_paths_refuse_a_sequential_model`] pins the typed refusal.

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};
use serde_json::Value;

use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::{input_ids, AttnMask, Weights};

use crate::common::Fixture;

const GOLDENS: &str = include_str!("../../testdata/gemma4/gemma4_decoder_goldens.json");

/// Additive-mask fill for a blocked `(query, key)` pair — the same large finite negative the LTX
/// adapter uses, deliberately rather than bf16-min: a Gemma 4 sliding layer *sums* this with its
/// window band, and bf16-min plus a band is one rounding step from `inf`.
const MASK_NEG: f32 = -1e30;

/// How far above the shared-suffix hidden states' own magnitude the two prompts' conditioning must
/// diverge for the layers to count as having participated.
///
/// Scale-relative, so nothing machine-dependent is committed. It is not a guess about how big the
/// effect "should" be: [`prompt_context_reaches_the_shared_suffix`] proves in the same run that the
/// *inputs* to those positions are bit-identical, so the entire measured difference is the layer
/// stack's doing and the only question is whether it is above noise. bf16 arithmetic noise on this
/// fixture is ~4e-3 relative; 5e-2 is an order of magnitude clear of it, and the empty-stack
/// mutation drives the measured value to exactly 0.0.
const CONTEXT_MIN_REL: f32 = 5.0e-2;

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

fn host(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors must have equal length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn magnitude(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1.0)
}

/// The fixture's config, and its weights written to a real `.safetensors` file.
///
/// The file is what makes this test meaningful: `from_file_sequential` re-opens its source on every
/// forward, so the source has to be a file. The returned [`Fixture`] owns the directory and must
/// outlive every model built from it — MLX's `load_safetensors` is lazy, so the tensors stay bound
/// to the path.
fn fixture_on_disk() -> (Fixture, std::path::PathBuf, ModelConfig) {
    let g = goldens();
    let cfg = ModelConfig::from_json(&g["config"]).expect("fixture config parses");
    assert!(
        cfg.is_gemma4(),
        "the fixture must exercise the Gemma 4 path"
    );
    assert_eq!(
        cfg.num_layers, 4,
        "the fixture is the 4-layer sliding/full alternation; a different depth would change what \
         the stream is proven over"
    );

    let mut map: HashMap<String, Array> = HashMap::new();
    for (key, entry) in g["weights"].as_object().expect("weights object") {
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

    let dir = Fixture::new("mlx-llm-sc18798-", None);
    let path = dir.root().join("model.safetensors");
    let refs: Vec<(&str, &Array)> = map.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &path).expect("write fixture safetensors");
    (dir, path, cfg)
}

fn resident(path: &std::path::Path, cfg: &ModelConfig) -> CausalLm {
    let w = Weights::from_file(path).expect("open fixture weights");
    CausalLm::from_weights(&w, "", cfg.clone()).expect("resident model loads")
}

fn streamed(path: &std::path::Path, cfg: &ModelConfig) -> CausalLm {
    CausalLm::from_file_sequential(path, "", cfg.clone(), None).expect("streamed model loads")
}

/// The additive causal + left-padding mask `[1, 1, L, L]`: `valid(i, j) = j <= i && keep[j]`.
///
/// The encoder path this stream serves runs a **left-padded** sequence, and `AttnMask::Causal`
/// alone does not mask padding — every valid token would attend the pad run and every hidden state
/// would be wrong while staying finite, non-zero and correctly shaped. Threading the real mask here
/// also means the streamed arm is proven to carry the mask *per layer*: a stream that dropped it
/// after the first layer would diverge from the resident pass in
/// [`streamed_stack_reproduces_the_resident_one`].
fn causal_padding_mask(keep: &[bool], dtype: Dtype) -> Array {
    let l = keep.len();
    let mut data = vec![0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            data[i * l + j] = if j <= i && keep[j] { 0.0 } else { MASK_NEG };
        }
    }
    Array::from_slice(&data, &[1, 1, l as i32, l as i32])
        .as_dtype(dtype)
        .expect("mask casts to the compute dtype")
}

fn hidden_stack(model: &CausalLm, ids: &[i32], keep: &[bool]) -> Vec<Array> {
    let mask = causal_padding_mask(keep, model.compute_dtype());
    let mut cache = model.new_cache();
    model
        .hidden_states_with_mask(&input_ids(ids), &mut cache, 0, AttnMask::Additive(&mask))
        .expect("hidden-state forward")
}

// ---------------------------------------------------------------------------------------------
// 1. Loader identity — observed on the loader, because output cannot see it.
// ---------------------------------------------------------------------------------------------

/// AC1: the streamed loader **ran** under `Sequential`, and is **not engaged** under `Resident`.
///
/// The resident side asserts `None`, not "a counter that reads zero". Those are different claims:
/// `None` says the streaming loader was never constructed. A zero counter would also be produced by
/// a stream that was constructed and silently did nothing — which is exactly the failure mode the
/// next test guards, and it must not be able to masquerade as "resident" here.
///
/// MUTATION: skip the layer in the `Stack::Sequential` arm of `run_decoder_stack_collecting`
/// (keeping the `sink.push`) and `layers_materialized` goes to 0 while the output stays correctly
/// shaped — RED here. Delete `view.remove_accessed()` and `view_drains` goes to 0 — RED here.
#[test]
fn loader_identity_streamed_ran_and_resident_did_not() {
    let (_dir, path, cfg) = fixture_on_disk();
    let n = cfg.num_layers;

    let resident = resident(&path, &cfg);
    let streamed = streamed(&path, &cfg);

    assert!(
        resident.stream_observation().is_none(),
        "a resident model must not carry a streaming-loader record at all — `None` is the claim \
         that the streamed loader was never constructed"
    );
    let obs = streamed
        .stream_observation()
        .expect("a sequential model must expose its streaming-loader record");
    assert_eq!(
        obs.passes(),
        0,
        "no forward has run yet, so the stream cannot have completed a pass"
    );

    let ids = [3, 1, 4, 1, 5, 9, 2];
    let keep = [true; 7];
    let states = hidden_stack(&streamed, &ids, &keep);

    assert_eq!(
        states.len(),
        n + 1,
        "the hidden-state stack is `num_layers + 1` (input embeds, then each layer's output)"
    );
    assert_eq!(
        obs.layers_materialized(),
        n,
        "every layer must have been built out of a reopened view — this is the claim that the \
         streamed loader RAN"
    );
    assert_eq!(
        obs.view_drains(),
        n,
        "every materialized layer must be followed by a view drain; fewer drains than \
         materializations means the view still holds the layers' tensors and the stream bounds \
         nothing"
    );
    assert_eq!(obs.passes(), 1, "exactly one stack pass ran");
    assert_eq!(
        obs.layers_in_last_pass(),
        n,
        "the completed pass must have covered the whole stack, not an empty range"
    );

    // A second forward accumulates rather than resetting — the record is cumulative, which is what
    // lets a caller snapshot and diff around a single pass.
    let _ = hidden_stack(&streamed, &ids, &keep);
    assert_eq!(obs.layers_materialized(), 2 * n);
    assert_eq!(obs.passes(), 2);
}

// ---------------------------------------------------------------------------------------------
// 2. Parity — streaming changes residency and nothing else.
// ---------------------------------------------------------------------------------------------

/// Streaming must be numerically invisible: the streamed stack reproduces the resident one.
///
/// This is what makes the loader-identity test necessary rather than redundant — the two passes are
/// *supposed* to be indistinguishable in their output, so output is not evidence about which ran.
///
/// It is also the assertion that catches a stream that mis-threads per-layer state: the fixture
/// alternates sliding and full layers, so a stream that reused one layer's RoPE table, dropped the
/// additive mask after the first layer, or mis-ordered the shared-K/V donor would diverge here.
///
/// MUTATION: off-by-one the layer index in the streamed arm (`i + 1`), or drop the `quant` replay
/// in `SequentialStack::run_layer` — RED.
#[test]
fn streamed_stack_reproduces_the_resident_one() {
    let (_dir, path, cfg) = fixture_on_disk();
    let resident = resident(&path, &cfg);
    let streamed = streamed(&path, &cfg);

    // Left-padded, like the encoder path this serves.
    let ids = [0, 0, 3, 1, 4, 1, 5];
    let keep = [false, false, true, true, true, true, true];

    let want = hidden_stack(&resident, &ids, &keep);
    let got = hidden_stack(&streamed, &ids, &keep);
    assert_eq!(
        want.len(),
        got.len(),
        "both stacks are `num_layers + 1` deep"
    );

    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert_eq!(w.shape(), g.shape(), "hidden state {i} shape");
        let (w, g) = (host(w), host(g));
        let err = max_abs_diff(&w, &g);
        assert_eq!(
            err, 0.0,
            "hidden state {i}: a streamed layer is built from the same bytes with the same quant \
             replay and run by the same forward, so it is bit-identical to its resident twin — \
             max|delta| was {err}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// 3. AC2 — the empty-stack `forward` trap.
// ---------------------------------------------------------------------------------------------

/// AC2: prove the materialized layers **participated**, by a discriminator an empty stack fails.
///
/// # Why the obvious test does not work
///
/// "Two different prompts produce different conditioning" is NOT a guard here. With an empty stack
/// each position's hidden state is a pure per-token embedding lookup, so two prompts with different
/// tokens still produce different outputs — the assertion passes while the encoder is returning
/// bare embeddings, which is the exact fault it was supposed to catch.
///
/// # The discriminator
///
/// Two sequences that share an identical **suffix** and differ only in their **prefix**, compared
/// *only over the shared-suffix positions*:
///
/// * with the layers running, causal attention carries the differing prefix forward into those
///   positions, so their hidden states must differ;
/// * with an empty stack there is no mixing at all — each suffix position is the lookup of the same
///   token in both sequences — so they are **bit-identical** and the difference vanishes.
///
/// The test proves that second clause rather than asserting it: it checks that the stack's entry
/// `[0]`, the input embeddings, is already exactly equal across the two prompts on the suffix
/// positions. That is the empty-stack output, computed in the same run. So every bit of the
/// divergence measured at the top of the stack is the layer stack's doing, and there is no
/// threshold-tuning question left — only "is it above bf16 noise", which [`CONTEXT_MIN_REL`] is an
/// order of magnitude clear of.
///
/// MUTATION: in the `Stack::Sequential` arm, keep `sink.push(h.clone())` but skip the layer, so the
/// stack is still `num_layers + 1` deep and every entry is the embeddings. Measured divergence goes
/// to exactly 0.0 — RED here, while the shape and length assertions above stay green. That is the
/// whole point: no memory assertion and no shape assertion sees this failure.
#[test]
fn prompt_context_reaches_the_shared_suffix() {
    let (_dir, path, cfg) = fixture_on_disk();
    let model = streamed(&path, &cfg);

    // Position:      0  1  2  |  3  4  5  6
    //                <- differing ->|<- shared suffix ->
    let a = [7, 11, 23, 5, 9, 2, 6];
    let b = [31, 19, 3, 5, 9, 2, 6];
    const SHARED_FROM: usize = 3;
    let keep = [true; 7];
    assert_eq!(
        a[SHARED_FROM..],
        b[SHARED_FROM..],
        "the two prompts must share their suffix exactly"
    );
    assert!(
        a[..SHARED_FROM]
            .iter()
            .zip(&b[..SHARED_FROM])
            .all(|(x, y)| x != y),
        "and differ at every prefix position"
    );

    let sa = hidden_stack(&model, &a, &keep);
    let sb = hidden_stack(&model, &b, &keep);
    assert_eq!(sa.len(), cfg.num_layers + 1);
    assert_eq!(sb.len(), cfg.num_layers + 1);

    let hidden = cfg.hidden_size as usize;
    let suffix = |t: &Array| -> Vec<f32> { host(t)[SHARED_FROM * hidden..].to_vec() };

    // The empty-stack output, measured rather than assumed: entry [0] is the input embeddings, and
    // over the shared suffix it is identical between the two prompts. Any divergence downstream is
    // therefore attributable to the layers alone.
    let (e_a, e_b) = (suffix(&sa[0]), suffix(&sb[0]));
    let embed_gap = max_abs_diff(&e_a, &e_b);
    assert_eq!(
        embed_gap, 0.0,
        "the shared-suffix INPUT EMBEDDINGS must be bit-identical across the two prompts — this is \
         what a zero-layer stack would return, and it is why a difference at the top of the stack \
         can only have come from the layers. Got max|delta| {embed_gap}"
    );

    // With the layers running, the same positions must diverge materially.
    let (t_a, t_b) = (suffix(sa.last().unwrap()), suffix(sb.last().unwrap()));
    let gap = max_abs_diff(&t_a, &t_b);
    let scale = magnitude(&t_a);
    println!(
        "shared-suffix divergence: max|delta| = {gap} against magnitude {scale} \
         (relative {:.4}); embedding-level gap = {embed_gap}",
        gap / scale
    );
    assert!(
        gap >= CONTEXT_MIN_REL * scale,
        "the loaded layers did not reach the shared suffix: max|delta| {gap} over positions \
         {SHARED_FROM}.. is below {} ({CONTEXT_MIN_REL} of magnitude {scale}). The prompts differ \
         only BEFORE these positions, so a stack that mixed context would separate them. A stack \
         that materialized no layer returns the bare token embeddings — identical here, gap 0.0 — \
         which is 'plausible output that ignores the prompt' and which no memory or shape \
         assertion detects.",
        CONTEXT_MIN_REL * scale
    );

    // Every intermediate layer output must also be prompt-separated on the suffix — a stream that
    // materialized only the first layer would pass a last-entry-only check.
    for (i, (x, y)) in sa.iter().zip(&sb).enumerate().skip(1) {
        let (x, y) = (suffix(x), suffix(y));
        assert!(
            max_abs_diff(&x, &y) > 0.0,
            "layer {} output is identical across the two prompts on the shared suffix — that \
             layer did not mix context, i.e. it did not run",
            i - 1
        );
    }
}

// ---------------------------------------------------------------------------------------------
// 4. A streamed model refuses the paths it is not wired for.
// ---------------------------------------------------------------------------------------------

/// A sequential model has no resident layer stack, and every path that needs one says so.
///
/// Represented as an `Option`/empty `Vec` instead, a zero-layer decode would have run silently and
/// returned logits off the un-decoded embeddings — the same class of fault as AC2, one layer up.
///
/// MUTATION: make `resident_layers` return `&[]` for the sequential arm instead of erroring — RED.
#[test]
fn decode_paths_refuse_a_sequential_model() {
    let (_dir, path, cfg) = fixture_on_disk();
    let model = streamed(&path, &cfg);

    let ids = input_ids(&[3, 1, 4]);
    let mut cache = model.new_paged_cache(8);
    let mut refs = [&mut cache];
    let err = model
        .decode_logits_per_seq(&ids, &mut refs, &[0, 1, 2])
        .expect_err("a sequential model must refuse the per-seq decode path");
    assert!(
        matches!(err, mlx_llm::Error::Unsupported(_)),
        "the refusal must be the typed Unsupported variant, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("sequential residency"),
        "the refusal must name why, got: {msg}"
    );
}
