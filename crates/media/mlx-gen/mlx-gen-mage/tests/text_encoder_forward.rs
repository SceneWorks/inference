//! sc-14038 — **weights-free** structural tests for the Qwen3-VL text-encoder forward.
//!
//! These run a real [`Qwen3VlTextEncoder`] over a synthetic, deterministically-generated
//! checkpoint at toy dimensions, so every invariant below is exercised by the production code path
//! without the 8.875 GB snapshot. They pin the two things a parity golden alone could not:
//!
//! 1. **the conditioning is the final layer's output AFTER the final RMSNorm** — asserted against
//!    a hand-assembled reference built from the same public pieces, *and* shown to differ from
//!    both wrong candidates (penultimate layer, final-but-un-normed);
//! 2. **packed segments are isolated** — `forward_packed` over a two-prompt pack reproduces each
//!    prompt's independent encode exactly. That is true by construction today (the port evaluates
//!    per segment, mirroring the SDPA varlen backend the goldens were dumped with), and this test
//!    is what keeps it true if the implementation is ever swapped for a block-diagonal mask.
//!
//! Real-weight parity against the torch golden lives in `te_parity_real_weights.rs`.

use std::collections::HashMap;

use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_mage::config::QwenVlTextConfig;
use mlx_gen_mage::text_encoder::{MRopePositions, Qwen3VlTextEncoder};

const PREFIX: &str = "lm";
const EPS: f32 = 1e-6;
const THETA: f64 = 5_000_000.0;

/// A toy Qwen3-VL: the real topology (GQA, decoupled `head_dim`, interleaved M-RoPE over a
/// `[1,1,1]` section) at dimensions small enough to build in memory.
fn tiny() -> QwenVlTextConfig {
    QwenVlTextConfig {
        hidden_size: 8,
        num_layers: 3,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        // Decoupled, exactly as production is: 4 × 6 = 24 ≠ 8.
        head_dim: 6,
        intermediate_size: 16,
        vocab_size: 20,
        mrope_section: [1, 1, 1],
        attention_bias: false,
        tie_word_embeddings: true,
    }
}

/// Deterministic pseudo-random weights — a plain LCG, so the fixture is reproducible without a
/// dependency and without committing a binary.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // [-0.5, 0.5) — small enough that three layers stay numerically tame.
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    fn array(&mut self, shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.next_f32()).collect();
        Array::from_slice(&data, shape)
    }

    /// A norm scale: centred on 1.0 so RMSNorm behaves like the real thing.
    fn norm(&mut self, dim: i32) -> Array {
        let data: Vec<f32> = (0..dim).map(|_| 1.0 + 0.1 * self.next_f32()).collect();
        Array::from_slice(&data, &[dim])
    }
}

fn synthetic_weights(cfg: &QwenVlTextConfig) -> Weights {
    let mut rng = Lcg(0x5EED_1403_8000_0001);
    let mut map: HashMap<String, Array> = HashMap::new();
    let (h, hd) = (cfg.hidden_size, cfg.head_dim);
    let (q_out, kv_out) = (cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd);

    map.insert(
        format!("{PREFIX}.embed_tokens.weight"),
        rng.array(&[cfg.vocab_size, h]),
    );
    for i in 0..cfg.num_layers {
        let p = format!("{PREFIX}.layers.{i}");
        map.insert(format!("{p}.input_layernorm.weight"), rng.norm(h));
        map.insert(format!("{p}.post_attention_layernorm.weight"), rng.norm(h));
        map.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rng.array(&[q_out, h]),
        );
        map.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rng.array(&[kv_out, h]),
        );
        map.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rng.array(&[kv_out, h]),
        );
        map.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rng.array(&[h, q_out]),
        );
        map.insert(format!("{p}.self_attn.q_norm.weight"), rng.norm(hd));
        map.insert(format!("{p}.self_attn.k_norm.weight"), rng.norm(hd));
        map.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rng.array(&[cfg.intermediate_size, h]),
        );
        map.insert(
            format!("{p}.mlp.up_proj.weight"),
            rng.array(&[cfg.intermediate_size, h]),
        );
        map.insert(
            format!("{p}.mlp.down_proj.weight"),
            rng.array(&[h, cfg.intermediate_size]),
        );
    }
    map.insert(format!("{PREFIX}.norm.weight"), rng.norm(h));
    Weights::from_map(map)
}

fn encoder(cfg: &QwenVlTextConfig) -> Qwen3VlTextEncoder {
    Qwen3VlTextEncoder::from_weights(&synthetic_weights(cfg), PREFIX, cfg, EPS, THETA).unwrap()
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// The forward produces the documented shape, and every layer plus the final norm participates.
#[test]
fn forward_segment_returns_the_post_final_norm_state() {
    let cfg = tiny();
    let te = encoder(&cfg);
    let ids: Vec<i32> = vec![3, 7, 1, 12, 5, 0, 19];

    let got = te.forward_segment(&ids).unwrap();
    assert_eq!(got.shape(), [ids.len() as i32, cfg.hidden_size]);

    // Hand-assemble the same computation from the public seam pieces, stopping one step short.
    let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);
    let hidden = te.embed(&ids_arr).unwrap();
    let pos = MRopePositions::text(ids.len());
    let (cos, sin) = mlx_gen_mage::text_encoder::mrope_cos_sin(
        &pos,
        cfg.head_dim,
        THETA,
        cfg.mrope_section,
        hidden.dtype(),
    )
    .unwrap();
    let ones = vec![1i32; ids.len()];
    let mask = mlx_gen::nn::build_mask(
        &Array::from_slice(&ones, &[1, ids.len() as i32]),
        1,
        ids.len() as i32,
    )
    .unwrap();

    let mut h = hidden;
    let mut penultimate = h.clone();
    for layer in te.layers() {
        let next = layer.forward(&h, &cos, &sin, &mask).unwrap();
        penultimate = std::mem::replace(&mut h, next);
    }
    let final_prenorm = h.clone();
    let final_postnorm = te.final_norm(&final_prenorm).unwrap();

    let flat = |x: &Array| {
        x.reshape(&[ids.len() as i32, cfg.hidden_size])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
    };

    // (a) The port returns the post-norm final state, exactly.
    assert_eq!(
        max_abs(&got, &flat(&final_postnorm)),
        0.0,
        "forward_segment is not the post-final-norm final layer"
    );

    // (b) …and the two wrong candidates are visibly different. This is the weights-free half of
    // the GAP-1 discrimination the real-weight test repeats against the torch golden.
    let vs_prenorm = max_abs(&got, &flat(&final_prenorm));
    let vs_penultimate = max_abs(&got, &flat(&penultimate));
    println!("final-pre-norm differs by {vs_prenorm:e}; penultimate differs by {vs_penultimate:e}");
    assert!(
        vs_prenorm > 1e-3,
        "the final RMSNorm is a no-op on this fixture ({vs_prenorm:e}) — the check cannot discriminate"
    );
    assert!(
        vs_penultimate > 1e-3,
        "the last decoder layer is a no-op on this fixture ({vs_penultimate:e})"
    );
}

/// A packed two-prompt forward reproduces each prompt's independent encode **bit for bit**: the
/// segments cannot see each other, and position ids restart at 0 per segment.
#[test]
fn packed_segments_are_isolated_from_each_other() {
    let te = encoder(&tiny());
    let a: Vec<i32> = vec![3, 7, 1, 12, 5];
    let b: Vec<i32> = vec![9, 2, 14];

    let mut packed_ids = a.clone();
    packed_ids.extend_from_slice(&b);
    let cu = mlx_gen_mage::text_encoder::cu_seqlens_from_lens(&[a.len(), b.len()]);
    let packed = te.forward_packed(&packed_ids, &cu).unwrap();
    assert_eq!(packed.shape(), [(a.len() + b.len()) as i32, 8]);

    let want_a = te.forward_segment(&a).unwrap();
    let want_b = te.forward_segment(&b).unwrap();
    let mut parts = packed.split_axis(&[a.len() as i32], 0).unwrap().into_iter();
    let got_a = parts.next().unwrap();
    let got_b = parts.next().unwrap();

    assert_eq!(max_abs(&got_a, &want_a), 0.0, "segment 0 leaked");
    assert_eq!(max_abs(&got_b, &want_b), 0.0, "segment 1 leaked");

    // The isolation claim is only meaningful if the two prompts would otherwise interact — i.e.
    // if B's encode genuinely depends on its own content and position, not on being second.
    let b_first = te
        .forward_packed(
            &{
                let mut v = b.clone();
                v.extend_from_slice(&a);
                v
            },
            &mlx_gen_mage::text_encoder::cu_seqlens_from_lens(&[b.len(), a.len()]),
        )
        .unwrap();
    let b_first_seg = b_first.split_axis(&[b.len() as i32], 0).unwrap().remove(0);
    assert_eq!(
        max_abs(&b_first_seg, &want_b),
        0.0,
        "segment order changed a segment's own encode"
    );
}

/// Attention is causal: appending tokens must not change the states already computed for the
/// prefix. A bidirectional mask would fail this, and no shape check would notice.
#[test]
fn attention_is_causal_within_a_segment() {
    let te = encoder(&tiny());
    let short: Vec<i32> = vec![3, 7, 1];
    let long: Vec<i32> = vec![3, 7, 1, 12, 5];

    let a = te.forward_segment(&short).unwrap();
    let b = te.forward_segment(&long).unwrap();
    let b_prefix = b.split_axis(&[short.len() as i32], 0).unwrap().remove(0);
    assert_eq!(
        max_abs(&a, &b_prefix),
        0.0,
        "extending the sequence changed the prefix — attention is not causal"
    );
}

/// The M-RoPE positions really reach the forward: running the **same embeddings** under two
/// different position layouts must give different states. A port that built `cos`/`sin` and then
/// dropped them would pass the causality and isolation checks unchanged.
///
/// Comparing two position layouts is the only formulation that actually tests this. The obvious
/// alternative — encode one repeated token and compare row 0 with row 1 — is a **false green**:
/// RoPE rotates q and k but never `v`, so with identical tokens every position's attention output
/// is `Σᵢ wᵢ·v` = `v` regardless of the weights, and the rows are mathematically identical. Such a
/// test measures float rounding, which is exactly what it did — it passed on one machine and was
/// bit-identical (and so failed) on another.
#[test]
fn position_affects_the_encoded_state() {
    let cfg = tiny();
    let te = encoder(&cfg);
    let ids: Vec<i32> = vec![3, 7, 1, 12];
    let n = ids.len() as i32;
    let hidden = te.embed(&Array::from_slice(&ids, &[1, n])).unwrap();

    let sequential = MRopePositions::text(ids.len());
    // Same order, different spacing — every pairwise relative distance changes.
    let stretched = MRopePositions {
        t: vec![0, 2, 4, 6],
        h: vec![0, 2, 4, 6],
        w: vec![0, 2, 4, 6],
    };

    let a = te.forward_embeds(&hidden, &sequential).unwrap();
    let b = te.forward_embeds(&hidden, &stretched).unwrap();
    let moved = max_abs(&a, &b);
    println!("stretching the M-RoPE positions moved the state by {moved:e}");
    assert!(
        moved > 1e-3,
        "changing the M-RoPE positions did not change the encoded state ({moved:e}) — the \
         rotary embedding is not reaching attention"
    );

    // …and the positions are what changed it: the same layout twice is bit-identical.
    assert_eq!(
        max_abs(&a, &te.forward_embeds(&hidden, &sequential).unwrap()),
        0.0,
        "the forward is not deterministic, so the comparison above proves nothing"
    );
}

/// A malformed or empty pack is an error, never a silently shorter conditioning.
#[test]
fn degenerate_inputs_are_rejected() {
    let te = encoder(&tiny());
    assert!(te.forward_segment(&[]).is_err());
    assert!(te.forward_packed(&[1, 2, 3], &[0, 4]).is_err());
    assert!(te.forward_packed(&[1, 2, 3], &[1, 3]).is_err());
}

/// A checkpoint missing the final `norm.weight` must fail loudly. That tensor is exactly what the
/// `mlx-gen-z-image` sibling deliberately does *not* load, so a copy-paste of that loader would
/// otherwise produce a working-looking encoder returning the wrong conditioning.
#[test]
fn a_checkpoint_without_the_final_norm_is_rejected() {
    let cfg = tiny();
    let mut w = synthetic_weights(&cfg);
    w.remove(&format!("{PREFIX}.norm.weight"));
    let err = match Qwen3VlTextEncoder::from_weights(&w, PREFIX, &cfg, EPS, THETA) {
        Err(e) => e,
        Ok(_) => panic!("the final RMSNorm must be required"),
    };
    assert!(
        format!("{err}").contains("norm.weight"),
        "unhelpful error: {err}"
    );
}

/// Every one of the configured layers is loaded and run — there is no "later layers cannot matter"
/// shortcut here, because the conditioning is the last layer's output.
#[test]
fn all_configured_layers_are_loaded() {
    let cfg = tiny();
    assert_eq!(encoder(&cfg).layers().len(), cfg.num_layers);

    let production = QwenVlTextConfig::mage_flow();
    assert_eq!(production.num_layers, 36);
}
