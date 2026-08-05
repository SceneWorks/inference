//! sc-17616 — the CFG-off conditioning gate is **output-equivalent** to the per-lane
//! `split(.., 2, 0)[0]` narrow it replaces.
//!
//! Gating at the encoder changes *what the denoise lane receives* when CFG is off: previously the
//! lane got the full `[pos; neg]` stack and sliced row 0 itself; now the encoder returns the cond-only
//! tensor directly. The two agree trivially for an empty negative (`s_neg == s_pos`, so the joint
//! stack's row 0 is the un-padded positive). They can only diverge when the **negative is longer than
//! the positive** at guidance exactly `1.0`: the old row 0 carried a zero-padded tail out to
//! `S_txt = max(s_pos, s_neg)` with those positions marked invalid in the mask, while the gated encode
//! returns the positive at its own length. This test drives both shapes through the real DiT and
//! asserts the conditional output is unchanged — i.e. the padded tail was inert, as the key-side
//! additive `-1e9` joint mask intends.
//!
//! Runs on a **tiny synthetic DiT** (dimension-parametric modules, seeded random weights), so it is a
//! default-suite test: no snapshot, no golden. The real-weights render parity for this story is the
//! shared `check_cfg_off_render` conformance arm (sc-17418) on the weekly real-weights lane.

use std::collections::HashMap;

use mlx_rs::ops::{abs, concatenate_axis, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_lens::dit::{LensDitConfig, LensTransformer};

const DTYPE: Dtype = Dtype::Float32;

/// A tiny Lens DiT: same architecture, toy dimensions. `axes_dims_rope` must sum to `head_dim`.
fn tiny_config() -> LensDitConfig {
    LensDitConfig {
        patch_size: 2,
        in_channels: 8,
        out_channels: 2, // proj_out width = patch²·out_channels = 8
        num_layers: 2,
        num_heads: 2,
        head_dim: 8,
        inner_dim: 16, // num_heads · head_dim
        enc_hidden_dim: 6,
        axes_dims_rope: [2, 2, 4], // Σ = head_dim
        num_text_layers: 2,
    }
}

/// Deterministic `[shape]` normal draw — a fixed key per tensor keeps the whole model reproducible.
fn draw(shape: &[i32], key: u64) -> Array {
    mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(key).unwrap()))
        .unwrap()
        .as_dtype(DTYPE)
        .unwrap()
}

/// Every tensor `LensTransformer::from_weights` requires, at the tiny config's shapes.
fn tiny_weights(cfg: &LensDitConfig) -> Weights {
    let dim = cfg.inner_dim;
    let mut t: HashMap<String, Array> = HashMap::new();
    let mut k = 0u64;
    let put = |t: &mut HashMap<String, Array>, name: &str, shape: &[i32], k: &mut u64| {
        *k += 1;
        t.insert(name.to_string(), draw(shape, *k));
    };

    put(&mut t, "img_in.weight", &[dim, cfg.in_channels], &mut k);
    put(&mut t, "img_in.bias", &[dim], &mut k);
    for i in 0..cfg.num_text_layers {
        put(
            &mut t,
            &format!("txt_norm.{i}.weight"),
            &[cfg.enc_hidden_dim],
            &mut k,
        );
    }
    let txt_in_dim = cfg.enc_hidden_dim * cfg.num_text_layers as i32;
    put(&mut t, "txt_in.weight", &[dim, txt_in_dim], &mut k);
    put(&mut t, "txt_in.bias", &[dim], &mut k);
    put(
        &mut t,
        "time_text_embed.timestep_embedder.linear_1.weight",
        &[dim, 256],
        &mut k,
    );
    put(
        &mut t,
        "time_text_embed.timestep_embedder.linear_1.bias",
        &[dim],
        &mut k,
    );
    put(
        &mut t,
        "time_text_embed.timestep_embedder.linear_2.weight",
        &[dim, dim],
        &mut k,
    );
    put(
        &mut t,
        "time_text_embed.timestep_embedder.linear_2.bias",
        &[dim],
        &mut k,
    );
    put(&mut t, "norm_out.linear.weight", &[2 * dim, dim], &mut k);
    put(&mut t, "norm_out.linear.bias", &[2 * dim], &mut k);
    let out_w = cfg.patch_size * cfg.patch_size * cfg.out_channels;
    put(&mut t, "proj_out.weight", &[out_w, dim], &mut k);
    put(&mut t, "proj_out.bias", &[out_w], &mut k);

    let hidden = 32; // SwiGLU hidden width — free here, the loader takes it from the weight shape
    for b in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{b}");
        // AdaLN modulations: 2 × 3·dim (each split into shift/scale/gate).
        put(
            &mut t,
            &format!("{p}.img_mod.1.weight"),
            &[6 * dim, dim],
            &mut k,
        );
        put(&mut t, &format!("{p}.img_mod.1.bias"), &[6 * dim], &mut k);
        put(
            &mut t,
            &format!("{p}.txt_mod.1.weight"),
            &[6 * dim, dim],
            &mut k,
        );
        put(&mut t, &format!("{p}.txt_mod.1.bias"), &[6 * dim], &mut k);
        for n in ["img_norm1", "img_norm2", "txt_norm1", "txt_norm2"] {
            put(&mut t, &format!("{p}.{n}.weight"), &[dim], &mut k);
        }
        for n in ["img_qkv", "txt_qkv"] {
            put(
                &mut t,
                &format!("{p}.attn.{n}.weight"),
                &[3 * dim, dim],
                &mut k,
            );
            put(&mut t, &format!("{p}.attn.{n}.bias"), &[3 * dim], &mut k);
        }
        for n in ["to_out.0", "to_add_out"] {
            put(&mut t, &format!("{p}.attn.{n}.weight"), &[dim, dim], &mut k);
            put(&mut t, &format!("{p}.attn.{n}.bias"), &[dim], &mut k);
        }
        for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
            put(
                &mut t,
                &format!("{p}.attn.{n}.weight"),
                &[cfg.head_dim],
                &mut k,
            );
        }
        for m in ["img_mlp", "txt_mlp"] {
            put(
                &mut t,
                &format!("{p}.{m}.w1.weight"),
                &[hidden, dim],
                &mut k,
            );
            put(
                &mut t,
                &format!("{p}.{m}.w3.weight"),
                &[hidden, dim],
                &mut k,
            );
            put(
                &mut t,
                &format!("{p}.{m}.w2.weight"),
                &[dim, hidden],
                &mut k,
            );
        }
    }
    Weights::from_map(t)
}

/// `num_text_layers` feature layers `[1, s, enc_hidden_dim]`, deterministic in `key`.
fn text_layers(cfg: &LensDitConfig, s: i32, key: u64) -> Vec<Array> {
    (0..cfg.num_text_layers)
        .map(|i| draw(&[1, s, cfg.enc_hidden_dim], key + i as u64))
        .collect()
}

/// Zero-pad each `[1, s, C]` layer to `target` along the sequence axis (what the joint path's
/// `pad_features` does to the positive when the negative is longer).
fn pad_layers(layers: &[Array], target: i32) -> Vec<Array> {
    layers
        .iter()
        .map(|f| {
            let pad = target - f.shape()[1];
            let z = mlx_rs::ops::zeros::<f32>(&[1, pad, f.shape()[2]])
                .unwrap()
                .as_dtype(DTYPE)
                .unwrap();
            concatenate_axis(&[f, &z], 1).unwrap()
        })
        .collect()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    max(abs(subtract(a, b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

/// The divergent case: at guidance `1.0` with a negative LONGER than the positive, the pre-sc-17616
/// lane narrowed row 0 out of a stack padded to `S_txt = max(s_pos, s_neg)` — a zero-padded positive
/// with a zero-padded mask. The gated encode returns the positive un-padded. Both must produce the
/// same conditional velocity, or the refactor would have changed CFG-off renders.
#[test]
fn padded_and_unpadded_cond_streams_agree() {
    let cfg = tiny_config();
    let dit = LensTransformer::from_weights(&tiny_weights(&cfg), &cfg, DTYPE).expect("build dit");

    let (frame, h, w) = (1usize, 2usize, 2usize);
    let latent = draw(&[1, (frame * h * w) as i32, cfg.in_channels], 9_001);
    let timestep = Array::from_slice(&[0.7f32], &[1]).as_dtype(DTYPE).unwrap();

    let (s_pos, s_neg) = (3i32, 5i32); // a longer negative ⇒ the old row 0 was padded to 5
    let pos = text_layers(&cfg, s_pos, 4_100);
    let pos_mask = mlx_rs::ops::ones::<f32>(&[1, s_pos]).unwrap();

    // New (gated): the positive at its own length.
    let gated = dit
        .forward(&latent, &pos, Some(&pos_mask), &timestep, frame, h, w)
        .expect("gated forward");

    // Old (narrowed): the positive zero-padded to S_txt, its pad tail marked invalid in the mask.
    let padded = pad_layers(&pos, s_neg);
    let padded_mask = concatenate_axis(
        &[
            &pos_mask,
            &mlx_rs::ops::zeros::<f32>(&[1, s_neg - s_pos]).unwrap(),
        ],
        1,
    )
    .unwrap();
    let narrowed = dit
        .forward(&latent, &padded, Some(&padded_mask), &timestep, frame, h, w)
        .expect("narrowed forward");

    assert_eq!(gated.shape(), narrowed.shape());
    let diff = max_abs_diff(&gated, &narrowed);
    assert!(
        diff < 1e-5,
        "the masked pad tail must be inert: cond-only vs narrowed-from-padded differ by {diff}"
    );
}

/// The empty-negative case (the `lens_turbo` default at guidance 1.0): the joint stack's row 0 IS the
/// un-padded positive, so the gated encode and the old narrow feed byte-identical tensors — the
/// forward must be bit-for-bit equal, not merely close.
#[test]
fn empty_negative_cond_only_is_bit_identical_to_the_narrowed_row() {
    let cfg = tiny_config();
    let dit = LensTransformer::from_weights(&tiny_weights(&cfg), &cfg, DTYPE).expect("build dit");

    let (frame, h, w) = (1usize, 2usize, 2usize);
    let latent = draw(&[1, (frame * h * w) as i32, cfg.in_channels], 9_002);
    let timestep = Array::from_slice(&[0.3f32], &[1]).as_dtype(DTYPE).unwrap();

    let s = 4i32;
    let pos = text_layers(&cfg, s, 5_200);
    let pos_mask = mlx_rs::ops::ones::<f32>(&[1, s]).unwrap();

    let gated = dit
        .forward(&latent, &pos, Some(&pos_mask), &timestep, frame, h, w)
        .expect("gated forward");

    // Reproduce the retired lane: stack `[pos; zeros]`, then slice row 0 back out.
    let stacked: Vec<Array> = pos
        .iter()
        .map(|f| concatenate_axis(&[f, &mlx_rs::ops::zeros_like(f).unwrap()], 0).unwrap())
        .collect();
    let stacked_mask = concatenate_axis(
        &[&pos_mask, &mlx_rs::ops::zeros_like(&pos_mask).unwrap()],
        0,
    )
    .unwrap();
    let narrowed_feats: Vec<Array> = stacked
        .iter()
        .map(|f| mlx_rs::ops::split(f, 2, 0).unwrap().swap_remove(0))
        .collect();
    let narrowed_mask = mlx_rs::ops::split(&stacked_mask, 2, 0)
        .unwrap()
        .swap_remove(0);
    let narrowed = dit
        .forward(
            &latent,
            &narrowed_feats,
            Some(&narrowed_mask),
            &timestep,
            frame,
            h,
            w,
        )
        .expect("narrowed forward");

    assert_eq!(
        max_abs_diff(&gated, &narrowed),
        0.0,
        "an empty negative makes the gate and the narrow byte-identical"
    );
}
