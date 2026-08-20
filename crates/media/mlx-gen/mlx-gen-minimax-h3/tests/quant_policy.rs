//! The converter's pack predicates held against the loaders that consume them — weights-free.
//!
//! Both halves of MiniMax-H3's tiering have the same failure mode and it is silent in both: a
//! tensor packed by [`crate::convert`](mlx_gen_minimax_h3::convert) that some loader reads with a
//! raw `Weights::require` loads u32 codes where a float is expected and reports **no error at all**
//! (sc-14980's Mage `pos_embed`). Nothing downstream can see it — the shapes are plausible, the
//! render completes, the output is wrong.
//!
//! The only defence is that the converter's predicate and the loaders' split are the *same* set,
//! so this file asserts that directly instead of trusting two comment blocks to agree.
//!
//! # The text-encoder half asserts equality, not agreement
//!
//! [`MiniMaxH3TextEncoder::quantize`] packs the token table and every loaded projection in place.
//! It is not a render path (see its own docs), it is the modules' own statement of what is
//! packable — so `converter_pack_matches_the_encoders_own_quantize` builds one encoder each way,
//! from the same dense weights, and requires the two forwards to be **bit-identical**. If the
//! converter's suffix list ever drifts from what the modules actually pack, one side quantizes a
//! tensor the other leaves dense and the difference is immediate.
//!
//! `mlx_gen::quant::quantize_map` and `AdaptableLinear::quantize` both cast to bf16 before packing
//! (the sc-2609 fork-parity downcast), so bit-identity is the honest bar here rather than a
//! tolerance.

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;

use mlx_gen_minimax_h3::convert::{is_te_pack_target, GROUP_SIZE};
use mlx_gen_minimax_h3::text_encoder::{MiniMaxH3TeConfig, MiniMaxH3TextEncoder};

/// The tiny geometry the synthetic encoder is built at.
///
/// Every packable input width is a multiple of [`GROUP_SIZE`] (128, 256), because a width that is
/// not would be left dense by `quantize_map`'s shape guard and the comparison would pass for the
/// wrong reason. `head_dim` is deliberately not `hidden / heads`, and `select_hidden < num_layers`,
/// so the real model's two structural quirks are exercised.
fn tiny_config() -> MiniMaxH3TeConfig {
    MiniMaxH3TeConfig {
        hidden_size: 128,
        num_layers: 3,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        intermediate_size: 256,
        select_hidden: 2,
        vocab_size: 128,
        ..MiniMaxH3TeConfig::qwen3_vl_32b()
    }
}

const PREFIX: &str = "model.language_model";

/// A deterministic, non-constant dense weight. Non-constant matters: a table of equal values
/// quantizes exactly, and every packing bug would then be invisible.
fn weight(seed: u64, shape: &[i32]) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let vals: Vec<f32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // A wide, sign-mixed spread so group scales and biases both do real work.
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect();
    Array::from_slice(&vals, shape)
        .as_dtype(Dtype::Bfloat16)
        .expect("bf16 cast")
}

/// The full dense key→tensor map of a `tiny_config()` text encoder, in the published key layout.
fn dense_map(cfg: &MiniMaxH3TeConfig) -> HashMap<String, Array> {
    let mut m = HashMap::new();
    let mut seed = 1u64;
    let put = |m: &mut HashMap<String, Array>, key: String, shape: &[i32], seed: &mut u64| {
        *seed += 1;
        m.insert(key, weight(*seed, shape));
    };
    let h = cfg.hidden_size;
    let q_out = cfg.num_heads * cfg.head_dim;
    let kv_out = cfg.num_kv_heads * cfg.head_dim;

    put(
        &mut m,
        format!("{PREFIX}.embed_tokens.weight"),
        &[cfg.vocab_size, h],
        &mut seed,
    );
    for i in 0..cfg.num_layers {
        let l = format!("{PREFIX}.layers.{i}");
        for (leaf, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q_out, h]),
            ("self_attn.k_proj.weight", vec![kv_out, h]),
            ("self_attn.v_proj.weight", vec![kv_out, h]),
            ("self_attn.o_proj.weight", vec![h, q_out]),
            ("self_attn.q_norm.weight", vec![cfg.head_dim]),
            ("self_attn.k_norm.weight", vec![cfg.head_dim]),
            ("mlp.gate_proj.weight", vec![cfg.intermediate_size, h]),
            ("mlp.up_proj.weight", vec![cfg.intermediate_size, h]),
            ("mlp.down_proj.weight", vec![h, cfg.intermediate_size]),
        ] {
            put(&mut m, format!("{l}.{leaf}"), &shape, &mut seed);
        }
    }
    m
}

fn weights_from(map: &HashMap<String, Array>) -> Weights {
    let mut w = Weights::empty();
    for (k, v) in map {
        w.insert(k.clone(), v.clone());
    }
    w
}

/// `max|a−b|` over two same-shaped arrays. Never a norm and never a cosine: a cosine is
/// scale-invariant and a norm averages a local catastrophe away over the channel axis.
fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a: Vec<f32> = a
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let b: Vec<f32> = b
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn probe_ids(cfg: &MiniMaxH3TeConfig) -> (Array, Array) {
    let ids: Vec<i32> = (0..6).map(|i| (i * 7) % cfg.vocab_size).collect();
    (
        Array::from_slice(&ids, &[1, 6]),
        Array::from_slice(&[1i32; 6], &[1, 6]),
    )
}

/// **The drift guard.** The converter's [`is_te_pack_target`] and
/// [`MiniMaxH3TextEncoder::quantize`] must pack exactly the same set at exactly the same widths.
///
/// Built two ways from one dense map:
/// * load dense, then `quantize(bits)` in place — the modules' own definition;
/// * run the source map through `quantize_map(.., is_te_pack_target)` as the converter does, then
///   load the packed map through the auto-detecting loader.
///
/// Bit-identical forwards is the bar. A predicate that missed `.embed_tokens`, or picked up
/// `.q_norm`, or packed at a different width, moves the output.
#[test]
fn converter_pack_matches_the_encoders_own_quantize() {
    let cfg = tiny_config();
    let source = dense_map(&cfg);
    let (ids, mask) = probe_ids(&cfg);

    for bits in [4, 8] {
        // (a) The modules' own definition.
        let mut in_place =
            MiniMaxH3TextEncoder::from_weights(&weights_from(&source), PREFIX, &cfg).unwrap();
        in_place.quantize(bits).unwrap();
        let a = in_place.forward(&ids, &mask).unwrap();

        // (b) The converter's predicate, through the auto-detecting loader.
        let packed =
            mlx_gen::quant::quantize_map(source.clone(), bits, GROUP_SIZE, is_te_pack_target)
                .unwrap();
        let from_tier =
            MiniMaxH3TextEncoder::from_weights(&weights_from(&packed), PREFIX, &cfg).unwrap();
        let b = from_tier.forward(&ids, &mask).unwrap();

        assert_eq!(
            max_abs_diff(&a, &b),
            0.0,
            "q{bits}: the converter's pack set differs from the encoder's own `quantize`"
        );
        // Both sides must actually be packed at the requested width — an equality that held
        // because neither packed anything would be the emptiest possible green.
        assert_eq!(in_place.packed_bits().unwrap(), Some(bits));
        assert_eq!(from_tier.packed_bits().unwrap(), Some(bits));
        assert!(in_place.token_table_is_quantized());
        assert!(
            from_tier.token_table_is_quantized(),
            "q{bits}: the converter left the token table dense"
        );
    }
}

/// Quantization must actually change the conditioning, or the equality above proves nothing.
///
/// Also pins the direction of the tier ladder: q8 is closer to dense than q4 is. Relative
/// max-abs-diff against the dense context, which is the same gate `quant_tiers_real.rs` uses.
#[test]
fn packing_moves_the_context_and_q8_moves_it_less_than_q4() {
    let cfg = tiny_config();
    let source = dense_map(&cfg);
    let (ids, mask) = probe_ids(&cfg);

    let dense = MiniMaxH3TextEncoder::from_weights(&weights_from(&source), PREFIX, &cfg).unwrap();
    let reference = dense.forward(&ids, &mask).unwrap();
    let scale = reference
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "a zero reference context cannot gate anything");

    let relative = |bits: i32| {
        let packed =
            mlx_gen::quant::quantize_map(source.clone(), bits, GROUP_SIZE, is_te_pack_target)
                .unwrap();
        let te = MiniMaxH3TextEncoder::from_weights(&weights_from(&packed), PREFIX, &cfg).unwrap();
        max_abs_diff(&reference, &te.forward(&ids, &mask).unwrap()) / scale
    };
    let (q4, q8) = (relative(4), relative(8));
    assert!(q4 > 0.0, "q4 left the context untouched — nothing packed");
    assert!(q8 > 0.0, "q8 left the context untouched — nothing packed");
    assert!(
        q8 < q4,
        "q8 relative max-abs-diff {q8} must be under q4's {q4}"
    );
}

/// The trim survives packing: a packed tier still loads only `select_hidden` layers, and the
/// unloaded tail's presence or absence in the map cannot change the context.
#[test]
fn a_packed_tier_still_loads_only_the_selected_layers() {
    let cfg = tiny_config();
    let source = dense_map(&cfg);
    let packed =
        mlx_gen::quant::quantize_map(source.clone(), 4, GROUP_SIZE, is_te_pack_target).unwrap();
    let full = MiniMaxH3TextEncoder::from_weights(&weights_from(&packed), PREFIX, &cfg).unwrap();
    assert_eq!(full.num_loaded_layers(), cfg.select_hidden);

    // Physically remove the never-run tail; the context must be unchanged, not merely close.
    let trimmed: HashMap<String, Array> = packed
        .iter()
        .filter(|(k, _)| !k.starts_with(&format!("{PREFIX}.layers.{}.", cfg.num_layers - 1)))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    assert!(trimmed.len() < packed.len(), "nothing was removed");
    let te = MiniMaxH3TextEncoder::from_weights(&weights_from(&trimmed), PREFIX, &cfg).unwrap();

    let (ids, mask) = probe_ids(&cfg);
    assert_eq!(
        max_abs_diff(
            &full.forward(&ids, &mask).unwrap(),
            &te.forward(&ids, &mask).unwrap()
        ),
        0.0
    );
}

/// A tier assembled at two widths is a mis-built artifact, and reporting the first layer's width
/// would hide it. [`MiniMaxH3TextEncoder::packed_bits`] names it instead.
#[test]
fn a_mixed_width_tier_is_a_named_error_not_a_plausible_width() {
    let cfg = tiny_config();
    let source = dense_map(&cfg);
    // Pack layer 0 at q8 and everything else at q4 — the shape a resumed/re-run conversion makes.
    let q4 = mlx_gen::quant::quantize_map(source.clone(), 4, GROUP_SIZE, |b: &str| {
        is_te_pack_target(b) && !b.starts_with(&format!("{PREFIX}.layers.0."))
    })
    .unwrap();
    let mixed = mlx_gen::quant::quantize_map(q4, 8, GROUP_SIZE, |b: &str| {
        is_te_pack_target(b) && b.starts_with(&format!("{PREFIX}.layers.0."))
    })
    .unwrap();

    let te = MiniMaxH3TextEncoder::from_weights(&weights_from(&mixed), PREFIX, &cfg).unwrap();
    let err = te.packed_bits().unwrap_err().to_string();
    assert!(
        err.contains("mixes quantization widths"),
        "unexpected error: {err}"
    );
}
