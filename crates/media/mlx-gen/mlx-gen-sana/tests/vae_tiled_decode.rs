//! sc-19753 — the DC-AE bounded decode keeps the `EfficientVit` attention whole.
//!
//! SANA's decoder ends in deep `EfficientVit` stages whose ReLU-linear attention contracts over
//! **every** `H·W` token — `matmul(v, kᵀ)` and `sum_axes(&k, &[3])` in `LinearAttn::forward`. The MLX
//! lane used to tile the *entire* decoder, so nine of those blocks ran per tile and each aggregated
//! over its own crop rather than the image. That is the same defect class as a per-tile GroupNorm,
//! and no overlap width can fix it. `decode_tiled` now runs [`DcAeDecoder::decode_head`] once and
//! tiles only the attention-free [`DcAeDecoder::decode_tail`], matching the candle sibling.
//!
//! The existing weights-free fixture in `pipeline_contract.rs` is all-`Res`, so it cannot construct
//! a single attention block and could never have observed this. This file builds a decoder that
//! genuinely carries `EfficientVit` stages.
//!
//! Weights-free and unit-scale: deterministic synthetic tensors, a narrow channel width, and a
//! latent a few cells across.

use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_sana::config::{BlockType, DcAeConfig};
use mlx_gen_sana::dc_ae::DcAeDecoder;
use mlx_rs::random::{key, normal};
use mlx_rs::Array;

const HEAD_DIM: i32 = 4;

fn rand(shape: &[i32], seed: u64) -> Array {
    normal::<f32>(shape, None, None, Some(&key(seed).unwrap())).unwrap()
}

/// `[out, in_per_group, k, k]` — the PyTorch layout `Conv::load` transposes.
fn conv(
    w: &mut Weights,
    prefix: &str,
    o: i32,
    i_per_group: i32,
    k: i32,
    bias: bool,
    seed: &mut u64,
) {
    w.insert(
        format!("{prefix}.weight"),
        rand(&[o, i_per_group, k, k], *seed),
    );
    *seed += 1;
    if bias {
        w.insert(format!("{prefix}.bias"), rand(&[o], *seed));
        *seed += 1;
    }
}

fn affine(w: &mut Weights, prefix: &str, ch: i32, seed: &mut u64) {
    w.insert(format!("{prefix}.weight"), rand(&[ch], *seed));
    *seed += 1;
    w.insert(format!("{prefix}.bias"), rand(&[ch], *seed));
    *seed += 1;
}

fn res_block(w: &mut Weights, prefix: &str, ch: i32, seed: &mut u64) {
    conv(w, &format!("{prefix}.conv1"), ch, ch, 3, true, seed);
    conv(w, &format!("{prefix}.conv2"), ch, ch, 3, false, seed);
    affine(w, &format!("{prefix}.norm"), ch, seed);
}

/// One `EfficientVit` block: the multiscale ReLU-linear attention plus its `GluMbConv` epilogue.
fn evit_block(w: &mut Weights, prefix: &str, ch: i32, cfg: &DcAeConfig, seed: &mut u64) {
    let num_heads = ch / HEAD_DIM;
    let attn = format!("{prefix}.attn");
    // q/k/v are `[inner, in]` (loaded transposed); `inner == ch` at mult 1.0.
    for name in ["to_q", "to_k", "to_v"] {
        w.insert(format!("{attn}.{name}.weight"), rand(&[ch, ch], *seed));
        *seed += 1;
    }
    // to_out consumes the concatenated raw + multiscale heads: `inner · (1 + scales)`.
    let fan_in = ch * (1 + cfg.qkv_multiscales.len() as i32);
    w.insert(format!("{attn}.to_out.weight"), rand(&[ch, fan_in], *seed));
    *seed += 1;
    for (i, k) in cfg.qkv_multiscales.iter().enumerate() {
        let proj = format!("{attn}.to_qkv_multiscale.{i}");
        // proj_in is depthwise over the concatenated [q,k,v] = 3·ch channels.
        conv(w, &format!("{proj}.proj_in"), 3 * ch, 1, *k, false, seed);
        // proj_out is grouped by 3·num_heads.
        conv(
            w,
            &format!("{proj}.proj_out"),
            3 * ch,
            3 * ch / (3 * num_heads),
            1,
            false,
            seed,
        );
    }
    affine(w, &format!("{attn}.norm_out"), ch, seed);

    let glu = format!("{prefix}.conv_out");
    let hidden = ch * 4;
    conv(
        w,
        &format!("{glu}.conv_inverted"),
        2 * hidden,
        ch,
        1,
        true,
        seed,
    );
    conv(
        w,
        &format!("{glu}.conv_depth"),
        2 * hidden,
        1,
        3,
        true,
        seed,
    );
    conv(w, &format!("{glu}.conv_point"), ch, hidden, 1, false, seed);
    affine(w, &format!("{glu}.norm"), ch, seed);
}

/// A decoder that really carries attention: two shallow `Res` stages (the tileable tail) under two
/// deep `EfficientVit` stages (the dense head) — the shipped `[R,R,R,E,E,E]` shape, narrowed.
fn config() -> DcAeConfig {
    DcAeConfig {
        in_channels: 3,
        latent_channels: 4,
        attention_head_dim: HEAD_DIM,
        block_out_channels: vec![8, 8, 8, 8],
        layers_per_block: vec![1, 1, 1, 1],
        encoder_layers_per_block: vec![1, 1, 1, 1],
        block_types: vec![
            BlockType::Res,
            BlockType::Res,
            BlockType::EfficientVit,
            BlockType::EfficientVit,
        ],
        qkv_multiscales: vec![5],
        upsample_interpolate: true,
        norm_eps: 1e-5,
        attn_eps: 1e-15,
        scaling_factor: 0.41407,
    }
}

fn weights(cfg: &DcAeConfig) -> Weights {
    let mut w = Weights::empty();
    let mut s = 90_000_u64;
    let n = cfg.num_stages();
    let deepest = cfg.block_out_channels[n - 1];
    conv(
        &mut w,
        "decoder.conv_in",
        deepest,
        cfg.latent_channels,
        3,
        true,
        &mut s,
    );

    for i in 0..n {
        let ch = cfg.block_out_channels[i];
        let has_up = i + 1 < n;
        let mut slot = 0;
        if has_up {
            conv(
                &mut w,
                &format!("decoder.up_blocks.{i}.0.conv"),
                ch,
                cfg.block_out_channels[i + 1],
                3,
                true,
                &mut s,
            );
            slot = 1;
        }
        for j in 0..cfg.layers_per_block[i] {
            let prefix = format!("decoder.up_blocks.{i}.{}", j + slot);
            match cfg.block_types[i] {
                BlockType::Res => res_block(&mut w, &prefix, ch, &mut s),
                BlockType::EfficientVit => evit_block(&mut w, &prefix, ch, cfg, &mut s),
            }
        }
    }

    let shallow = cfg.block_out_channels[0];
    affine(&mut w, "decoder.norm_out", shallow, &mut s);
    conv(
        &mut w,
        "decoder.conv_out",
        cfg.in_channels,
        shallow,
        3,
        true,
        &mut s,
    );
    w
}

/// Position-dependent latents, so a per-crop attention aggregate differs observably from the dense
/// one and the comparisons below discriminate.
fn latent(h: i32, w_: i32) -> Array {
    let c = config().latent_channels;
    let data = (0..(c * h * w_))
        .map(|i| {
            let y = (i / w_ % h) as f32;
            let x = (i % w_) as f32;
            (i as f32 * 0.037).sin() + y * 0.11 - x * 0.07
        })
        .collect::<Vec<_>>();
    Array::from_slice(&data, &[1, c, h, w_])
}

fn max_abs(a: &Array, b: &Array) -> f32 {
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Mean absolute error relative to the reference's own scale — the seam-tolerant measure the Wan
/// tiling gate uses, since a max-abs on random weights is dominated by a handful of edge pixels.
fn mean_rel(got: &Array, want: &Array) -> f64 {
    let (g, w) = (got.as_slice::<f32>(), want.as_slice::<f32>());
    let num: f64 = g.iter().zip(w).map(|(a, b)| (a - b).abs() as f64).sum();
    let den: f64 = w.iter().map(|b| b.abs() as f64).sum();
    num / den.max(1e-9)
}

fn decoder() -> DcAeDecoder {
    let cfg = config();
    DcAeDecoder::from_weights(&weights(&cfg), cfg).expect("build synthetic DC-AE decoder")
}

/// The head/tail split must be an exact decomposition of the dense decode — otherwise the bounded
/// route would be tracking the wrong reference.
#[test]
fn head_then_tail_is_exactly_the_dense_decode() {
    let dec = decoder();
    let z = latent(8, 9);
    let cancel = CancelFlag::new();
    let dense = dec.decode(&z, &cancel).unwrap();
    let composed = dec.decode_tail(&dec.decode_head(&z).unwrap()).unwrap();
    dense.eval().unwrap();
    composed.eval().unwrap();
    assert_eq!(composed.shape(), dense.shape());
    assert_eq!(max_abs(&dense, &composed), 0.0);
}

/// The split lands where the attention does: two shallow `Res` stages are tileable, and the tail's
/// upsample factor is ×4 — **not** the whole decoder's ×8. A tile plan keyed on the wrong scale
/// would silently partition the head feature map at the wrong granularity.
#[test]
fn the_tail_is_the_attention_free_prefix_and_carries_its_own_scale() {
    let dec = decoder();
    assert_eq!(dec.num_tail_stages(), 2);
    assert_eq!(dec.tail_scale(), 4);

    // Mutation control: an all-`Res` config has no head at all, and an attention-first config has no
    // tileable tail — both are handled, and neither is what the shipped geometry does.
    let mut all_res = config();
    all_res.block_types = vec![BlockType::Res; 4];
    let dec_all_res = DcAeDecoder::from_weights(&weights(&all_res), all_res.clone()).unwrap();
    assert_eq!(dec_all_res.num_tail_stages(), 4);
    assert_eq!(dec_all_res.tail_scale(), 8);

    let mut attn_first = config();
    attn_first.block_types = vec![BlockType::EfficientVit; 4];
    let dec_attn = DcAeDecoder::from_weights(&weights(&attn_first), attn_first).unwrap();
    assert_eq!(
        dec_attn.num_tail_stages(),
        0,
        "no attention-free prefix ⇒ no tileable tail ⇒ the pipeline must decode single-pass"
    );
}

/// The **executed** defect control: running the whole decoder per crop — the retired route — is
/// observably different from the dense decode, because each crop's `LinearAttn` sums over its own
/// token set. This is what makes the bounded-vs-dense comparison a real guard.
#[test]
fn per_crop_whole_decoder_attention_is_observably_wrong() {
    let dec = decoder();
    let z = latent(8, 9);
    let cancel = CancelFlag::new();
    let dense = dec.decode(&z, &cancel).unwrap();

    // Two independent latent crops, decoded whole and stitched.
    let take = |start: i32, len: i32| {
        let idx = (start..start + len).collect::<Vec<i32>>();
        z.take_axis(Array::from_slice(&idx, &[len]), 3).unwrap()
    };
    let left = dec.decode(&take(0, 4), &cancel).unwrap();
    let right = dec.decode(&take(4, 5), &cancel).unwrap();
    let per_crop = mlx_rs::ops::concatenate_axis(&[&left, &right], 2).unwrap();
    dense.eval().unwrap();
    per_crop.eval().unwrap();
    assert_eq!(per_crop.shape(), dense.shape());
    let delta = max_abs(&dense, &per_crop);
    assert!(
        delta > 1e-2,
        "per-crop whole-decoder attention must be observably different from the dense decode: \
         max|delta|={delta:.3e}"
    );
}

/// The bounded tail must track the dense decode **materially better** than the retired per-crop
/// whole-decoder route, driven through the real shared tiling driver on the tail's own geometry.
///
/// A relative claim, not a tolerance. On a tiny random-weight fixture there is no learned
/// smoothness, so each tile's convolutions zero-padding at their crop boundary dominates the
/// absolute error for *any* tiling policy — the same reason the Wan z16 gate
/// (`mlx-gen-wan/tests/tiling_parity.rs`) is comparative. What this story changed is the attention,
/// and the comparison isolates exactly that: both sides carry the same conv seam, only the retired
/// side also aggregates attention per crop.
#[test]
fn bounded_tail_tracks_dense_better_than_per_crop_decoding() {
    let dec = decoder();
    let z = latent(8, 9);
    let cancel = CancelFlag::new();
    let dense = dec.decode(&z, &cancel).unwrap();

    let head = dec.decode_head(&z).unwrap();
    let hs = head.shape();
    let vae = VaeTiling {
        spatial_scale: dec.tail_scale(),
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: 3,
    };
    // Tile edge in OUTPUT pixels; the head is already ×2 the latent here.
    let cfg = TilingConfig::spatial_only(4 * dec.tail_scale(), dec.tail_scale());
    assert!(
        cfg.needs_tiling(vae, 1, hs[1], hs[2]),
        "the bounded request must actually split the {}x{} head",
        hs[1],
        hs[2]
    );
    let plan = cfg.plan(vae, 1, hs[1], hs[2]);
    assert!(plan.h.len() > 1 && plan.w.len() > 1, "both axes must split");

    let lifted = head.reshape(&[1, 1, hs[1], hs[2], hs[3]]).unwrap();
    let bounded = mlx_gen::vae_tiling::tiled_decode(&lifted, &plan, [1, 2, 3], None, |tile| {
        let ts = tile.shape();
        let out = dec.decode_tail(&tile.reshape(&[1, ts[2], ts[3], ts[4]])?)?;
        let os = out.shape();
        Ok(out.reshape(&[1, 1, os[1], os[2], os[3]])?)
    })
    .unwrap();
    let bs = bounded.shape();
    let bounded = bounded.reshape(&[1, bs[2], bs[3], bs[4]]).unwrap();

    // The retired route: tile the whole decoder, on the whole decoder's ×8 geometry.
    let whole = VaeTiling {
        spatial_scale: 8,
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: 3,
    };
    let zs = z.shape();
    let legacy_plan = TilingConfig::spatial_only(4 * 8, 8).plan(whole, 1, zs[2], zs[3]);
    let nhwc = z.transpose_axes(&[0, 2, 3, 1]).unwrap();
    let legacy_in = nhwc.reshape(&[1, 1, zs[2], zs[3], zs[1]]).unwrap();
    let legacy =
        mlx_gen::vae_tiling::tiled_decode(&legacy_in, &legacy_plan, [1, 2, 3], None, |tile| {
            let ts = tile.shape();
            let t = tile
                .reshape(&[1, ts[2], ts[3], ts[4]])?
                .transpose_axes(&[0, 3, 1, 2])?;
            let out = dec.decode(&t, &CancelFlag::new())?;
            let os = out.shape();
            Ok(out.reshape(&[1, 1, os[1], os[2], os[3]])?)
        })
        .unwrap();
    let ls = legacy.shape();
    let legacy = legacy.reshape(&[1, ls[2], ls[3], ls[4]]).unwrap();

    dense.eval().unwrap();
    bounded.eval().unwrap();
    legacy.eval().unwrap();
    assert_eq!(bounded.shape(), dense.shape());
    assert_eq!(legacy.shape(), dense.shape());

    let bounded_rel = mean_rel(&bounded, &dense);
    let legacy_rel = mean_rel(&legacy, &dense);
    println!("[vs dense] legacy={legacy_rel:.3e} bounded={bounded_rel:.3e}");
    assert!(
        bounded_rel < legacy_rel * 0.75,
        "keeping the EfficientVit attention whole must materially reduce the divergence from a \
         dense decode: bounded={bounded_rel:.3e} vs legacy={legacy_rel:.3e}"
    );
}
