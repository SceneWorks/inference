//! sc-6266 — activation-chunking equivalence gate for the FLUX.2 MMDiT.
//!
//! Proves the [`MemoryConfig`] memory levers change only the activation *schedule*, not the result —
//! so the correctness gate (`transformer_parity.rs`, which runs with the default
//! [`MemoryConfig::OFF`]) keeps covering the math while the gated long-sequence multi-reference edit
//! path runs with the levers on (`model.rs`). Two equivalence classes, asserted separately on the
//! committed tiny fixture the parity test already carries (`tests/fixtures/transformer_golden.safetensors`):
//!   * **the graph-evaluation cadence is exactly bit-identical** (max|Δ| == 0) — it only forces
//!     materialization of the same graph, so the multi-reference edit's pixels are unchanged. This is
//!     the dominant memory lever and the production default ([`MemoryConfig::LONG_SEQ`]), so the win
//!     is bit-exact. sc-18317 generalized the retired `eval_per_block: bool` into a cadence, so the
//!     assertion now runs across cadences rather than at one boolean point.
//!   * **FFN sequence-chunking is numerically equivalent** (cosine ≥ 0.9999999) — the FFN is
//!     per-token so the math is identical, but MLX's Metal GEMM is tile-specialized by the row (M)
//!     dimension, so a `[chunk, k]` matmul can round slightly differently from the full `[L, k]` one
//!     (the same class as the model's own torch parity). It is off by default; on as env-tunable
//!     headroom for extreme configs.
//!
//! Self-consistent: it compares `forward(OFF)` against `forward(levered)` on the **same** model +
//! inputs, so it needs no torch reference. The deliberately tiny chunk size (down to 1 token) forces
//! the multi-chunk + ragged-remainder paths on the fixture's 4-token image sequence.
//!
//! sc-18317 adds the **request seam**: the same equivalence, reached the way epic 18304's planner
//! reaches it — through `GenerationMemory`'s typed domains and `MemoryConfig::with_request` — so the
//! proof covers the production wiring and not only a hand-built config.

use mlx_gen::gen_core::{FfnChunk, GenerationMemory, GraphEvalCadence};
use mlx_gen::weights::Weights;
use mlx_gen_flux2::{Flux2Config, Flux2ForwardInputs, Flux2Transformer, MemoryConfig};
use mlx_rs::{Array, Dtype};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/transformer_golden.safetensors"
);

/// The tiny config the dump script used (inner = 2·8 = 16) — identical to `transformer_parity.rs`.
fn tiny_config() -> Flux2Config {
    Flux2Config {
        num_double_layers: 1,
        num_single_layers: 1,
        num_heads: 2,
        head_dim: 8,
        in_channels: 4,
        out_channels: 4,
        joint_attention_dim: 12,
        mlp_ratio: 3.0,
        timestep_channels: 16,
        axes_dim: [2, 2, 2, 2],
        rope_theta: 2000.0,
        te_hidden_size: 4,
        te_intermediate_size: 12,
        te_out_layers: [0, 1, 2],
        max_sequence_length: 512,
        num_latent_channels: 1,
        vae_scale_factor: 8,
    }
}

fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// (cosine similarity, max abs diff) between two same-shape tensors.
fn compare(a: &Array, b: &Array) -> (f32, f32) {
    let (va, vb) = (flat(a), flat(b));
    assert_eq!(va.len(), vb.len(), "shape mismatch");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut max_abs = 0f32;
    for (x, y) in va.iter().zip(&vb) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        max_abs = max_abs.max((x - y).abs());
    }
    let cos = (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32;
    (cos, max_abs)
}

fn forward_mem(t: &Flux2Transformer, w: &Weights, mem: &MemoryConfig) -> Array {
    t.forward_with_mem(
        &Flux2ForwardInputs {
            hidden_states: w.require("hidden").unwrap(),
            encoder_hidden_states: w.require("encoder").unwrap(),
            img_ids: w.require("img_ids").unwrap(),
            txt_ids: w.require("txt_ids").unwrap(),
            timestep: 500.0,
            guidance: None,
        },
        None,
        mem,
        mlx_gen::attention::AttentionPlan::UNBOUNDED,
        None,
    )
    .unwrap()
}

#[test]
fn every_graph_eval_cadence_is_bit_identical() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();
    let base = forward_mem(&t, &w, &MemoryConfig::OFF);
    // LONG_SEQ = per-block evaluation only (the production long-sequence default), then the wider
    // cadences sc-18317 made reachable, including one past the fixture's stack depth (which degrades
    // to a single evaluation at the stack boundary).
    let mut configs = vec![MemoryConfig::LONG_SEQ];
    for blocks in [2u32, 3, 64] {
        configs.push(MemoryConfig {
            eval_cadence: Some(GraphEvalCadence::new(blocks).unwrap()),
            ..MemoryConfig::OFF
        });
    }
    for mem in configs {
        let levered = forward_mem(&t, &w, &mem);
        assert_eq!(base.shape(), levered.shape(), "{mem:?} out shape");
        let (cos, max_abs) = compare(&base, &levered);
        assert_eq!(
            max_abs, 0.0,
            "{mem:?} must be bit-identical (max|Δ| {max_abs}, cos {cos})"
        );
    }
}

/// **sc-18317 default preservation, at the request seam.** A request that selects no execution domain
/// must leave whichever base config the route chose completely alone — the property that makes this
/// story inert for every existing render. Asserted on the tensor, not only on the config, so a future
/// overlay bug that silently enables a lever is caught here and not only in `chunk.rs`.
#[test]
fn an_unset_request_selection_is_bit_identical_to_the_route_default() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();
    let unset = GenerationMemory::default();
    for base_config in [MemoryConfig::OFF, MemoryConfig::LONG_SEQ] {
        let base = forward_mem(&t, &w, &base_config);
        let overlaid = MemoryConfig::with_request(base_config, Some(&unset));
        assert_eq!(overlaid, base_config, "the overlay must be a no-op");
        let (_, max_abs) = compare(&base, &forward_mem(&t, &w, &overlaid));
        assert_eq!(
            max_abs, 0.0,
            "{base_config:?} perturbed by an unset request"
        );
    }
}

/// **sc-18317 reach.** A selection made the way the planner makes it — typed fields on
/// `GenerationMemory` — must arrive at the forward as the corresponding `MemoryConfig`, and the
/// forward must still be equivalent. This is the link between the request and the consumer that
/// `chunk.rs`'s unit tests cannot cover on their own (they stop at the config).
#[test]
fn a_request_selection_reaches_the_forward_and_stays_equivalent() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();
    let base = forward_mem(&t, &w, &MemoryConfig::OFF);

    let selected = GenerationMemory {
        graph_eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
        ffn_chunk: Some(FfnChunk::new(2).unwrap()),
        ..Default::default()
    };
    let mem = MemoryConfig::with_request(MemoryConfig::OFF, Some(&selected));
    // The request's values, not the route default, are what the forward will read.
    assert_eq!(mem.ffn_chunk_rows(), Some(2));
    assert!(mem.evaluates_after_block(0));
    let (cos, max_abs) = compare(&base, &forward_mem(&t, &w, &mem));
    assert!(
        cos >= 0.999_999_9,
        "a request-selected schedule diverged (cos {cos}, max|Δ| {max_abs})"
    );
}

#[test]
fn ffn_seq_chunk_is_numerically_equivalent() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = Flux2Transformer::from_weights(&w, &tiny_config()).unwrap();
    let base = forward_mem(&t, &w, &MemoryConfig::OFF);

    // chunk 1/2/3 over the 4-token image FFN exercise the multi-chunk + ragged-remainder paths.
    for chunk in [1u32, 2, 3] {
        let mem = MemoryConfig {
            ffn_seq_chunk: Some(FfnChunk::new(chunk).unwrap()),
            eval_cadence: None,
        };
        let chunked = forward_mem(&t, &w, &mem);
        assert_eq!(base.shape(), chunked.shape(), "chunk {chunk} out shape");
        let (cos, max_abs) = compare(&base, &chunked);
        assert!(
            cos >= 0.999_999_9,
            "ffn chunk {chunk} diverged (cos {cos}, max|Δ| {max_abs})"
        );
    }

    // Production-style combination (eval-to-free + FFN chunk) is still equivalent.
    let combined = forward_mem(
        &t,
        &w,
        &MemoryConfig {
            ffn_seq_chunk: Some(FfnChunk::new(2).unwrap()),
            eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
        },
    );
    let (cos, max_abs) = compare(&base, &combined);
    assert!(
        cos >= 0.999_999_9,
        "eval + ffn chunk diverged (cos {cos}, max|Δ| {max_abs})"
    );
}
