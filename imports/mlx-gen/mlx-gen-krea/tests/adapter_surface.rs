//! sc-11720 (mlx mirror of candle-gen sc-11720) — default-run lock on the Krea 2 DiT's **wide** adapter
//! surface: user LoRA/LoKr must reach the front-end / final / shared-modulation projections and the
//! per-block `to_gate` + SwiGLU FFN, **not just attention**, and each install must ride the shared
//! additive [`mlx_gen::adapters::AdaptableLinear`] residual over an **untouched** base (mlx-gen never
//! folds — CLAUDE.md "the base is never fused/mutated").
//!
//! candle-gen retired an eager `merge_into_weights` fold to reach this state; mlx-gen was additive by
//! construction, so the mirror is the *coverage* candle-gen added: `install_additive_drives_lora_and_
//! adapt_leaves_wide_surface`. Block-level attention aliases are already locked in `transformer::block`
//! unit tests; this locks the TOP-LEVEL [`Krea2Transformer`] global surface (`img_in` / `txt_in` /
//! `time_embed` / `time_mod_proj` / `final_layer`) + a block's `to_gate`/FFN, which had no default-run
//! coverage. Uses the committed tiny `dit_golden` fixture (runs on a fresh clone, no weights/Metal beyond
//! the default suite).

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen_krea::{Krea2Config, Krea2Transformer};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

/// The tiny config the `dit_golden` fixture was dumped against (mirrors `dit_parity::tiny_dit_config`).
fn tiny_dit_config() -> Krea2Config {
    Krea2Config {
        in_channels: 16,
        patch_size: 2,
        hidden_size: 128,
        num_attention_heads: 4,
        num_kv_heads: 2,
        attention_head_dim: 32,
        num_layers: 2,
        intermediate_size: 384,
        norm_eps: 1e-5,
        axes_dims_rope: [8, 12, 12],
        rope_theta: 1000.0,
        timestep_embed_dim: 64,
        num_text_layers: 3,
        num_layerwise_text_blocks: 2,
        num_refiner_text_blocks: 2,
        text_hidden_dim: 64,
        text_intermediate_size: 256,
        text_num_attention_heads: 2,
        text_num_kv_heads: 2,
    }
}

fn load() -> Weights {
    Weights::from_file(format!("{FIX}dit_golden.safetensors")).unwrap_or_else(|e| {
        panic!("load dit_golden fixture (run tools/dump_krea_dit_golden.py): {e}")
    })
}

fn dit(w: &Weights) -> Krea2Transformer {
    let cfg = tiny_dit_config();
    cfg.validate().unwrap();
    Krea2Transformer::from_weights(w, &cfg).unwrap()
}

/// A zero LoRA sized to `leaf` (logical `[out, in]`): installs cleanly and increments the adapter stack
/// without perturbing the forward (the "is this path on the surface" probe).
fn zero_lora(out: i32, in_: i32) -> Adapter {
    let rank = 2;
    Adapter::Lora {
        a: Array::zeros::<f32>(&[in_, rank]).unwrap(),
        b: Array::zeros::<f32>(&[rank, out]).unwrap(),
        scale: 1.0,
    }
}

/// Every path the widened surface must reach (sc-11720): the global front-end/final/shared-modulation
/// projections AND a representative block's `to_gate` + SwiGLU FFN — i.e. beyond attention. Each must
/// resolve through [`AdaptableHost::adaptable_mut`] and accept an additive adapter.
#[test]
fn wide_surface_routes_beyond_attention() {
    let w = load();
    let mut dit = dit(&w);

    // Global leaves (transformer/mod.rs) + a block's non-attention/attention-gate leaves (block.rs).
    let wide: &[&[&str]] = &[
        &["img_in"],
        &["txt_in", "linear_1"],
        &["txt_in", "linear_2"],
        &["time_embed", "linear_1"],
        &["time_embed", "linear_2"],
        &["time_mod_proj"],
        &["final_layer", "linear"],
        // Beyond attention, inside a single-stream block: the attention gate + the whole SwiGLU FFN.
        &["transformer_blocks", "0", "attn", "to_gate"],
        &["transformer_blocks", "0", "ff", "gate"],
        &["transformer_blocks", "0", "ff", "up"],
        &["transformer_blocks", "0", "ff", "down"],
        // The text-fusion aggregator's collapse projection.
        &["text_fusion", "projector"],
    ];

    for path in wide {
        let leaf = dit.adaptable_mut(path).unwrap_or_else(|| {
            panic!("wide-surface path {path:?} did not route to an AdaptableLinear")
        });
        let shape = leaf.base_shape();
        assert_eq!(shape.len(), 2, "path {path:?} base_shape {shape:?}");
        let before = leaf.adapters().len();
        leaf.push(zero_lora(shape[0], shape[1]));
        assert_eq!(
            dit.adaptable_mut(path).unwrap().adapters().len(),
            before + 1,
            "adapter did not install on wide-surface path {path:?}"
        );
    }

    // Norms / non-projection leaves are NOT adapter targets (the surface widened to the projections, not
    // to everything): a norm path must route to None.
    assert!(
        dit.adaptable_mut(&["final_layer", "norm"]).is_none(),
        "final_layer.norm must not be an adapter target"
    );
    assert!(
        dit.adaptable_mut(&["transformer_blocks", "0", "norm1"])
            .is_none(),
        "block norm must not be an adapter target"
    );
}

/// The install is a live **additive** residual over an untouched base: a nonzero LoRA on the front-end
/// `img_in` (which feeds the whole net) shifts the DiT velocity. Proves the widened leaves actually drive
/// the forward — not merely accept an adapter record (candle-gen `install_additive_drives_lora_*`).
#[test]
fn additive_lora_on_frontend_drives_forward() {
    let w = load();
    let latent = w.require("in.latent").unwrap();
    let timestep = w.require("in.timestep").unwrap();
    let context = w.require("in.context").unwrap();

    let base = dit(&w).forward(latent, timestep, context, None).unwrap();

    // A nonzero LoRA on `img_in` (base_shape [hidden, c·p·p]) → nonzero residual → shifted velocity.
    let mut adapted = dit(&w);
    let shape = adapted.adaptable_mut(&["img_in"]).unwrap().base_shape();
    let (out, in_) = (shape[0], shape[1]);
    let rank = 4;
    let a = Array::from_slice(
        &(0..in_ * rank)
            .map(|i| 0.02 * (i as f32).sin())
            .collect::<Vec<_>>(),
        &[in_, rank],
    );
    let b = Array::from_slice(
        &(0..rank * out)
            .map(|i| 0.02 * (i as f32).cos())
            .collect::<Vec<_>>(),
        &[rank, out],
    );
    adapted
        .adaptable_mut(&["img_in"])
        .unwrap()
        .push(Adapter::Lora { a, b, scale: 1.0 });
    let with_lora = adapted.forward(latent, timestep, context, None).unwrap();

    assert_eq!(with_lora.shape(), base.shape(), "velocity shape unchanged");
    let diff = subtract(
        with_lora.as_dtype(Dtype::Float32).unwrap(),
        base.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap();
    let delta = max(abs(diff).unwrap(), false).unwrap().item::<f32>();
    assert!(
        delta > 1e-3,
        "front-end LoRA did not move the velocity (max_abs_diff {delta:e}) — additive residual is inert"
    );
}
