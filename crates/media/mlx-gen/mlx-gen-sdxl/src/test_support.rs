//! A tiny random-init SDXL-family U-Net for weights-free trainer tests (epic 2123 sc-24830) — this
//! crate's and `mlx-gen-kolors`'s (the Kolors U-Net IS this U-Net). Not a stable API.

use mlx_gen::weights::Weights;
use mlx_rs::ops::multiply;
use mlx_rs::{random, Array};

use crate::config::UNetConfig;
use crate::unet::UNet2DConditionModel;
use mlx_gen::Result;

/// Cross-attention context width of [`tiny_unet_config`].
pub const TINY_CONTEXT_DIM: i32 = 16;
/// Pooled text-embedding width of [`tiny_unet_config`].
pub const TINY_POOLED_DIM: i32 = 16;

/// Two blocks (`[32, 64]` channels — the U-Net's GroupNorm is a fixed 32 groups), one resnet per
/// block, a cross-attention transformer on the inner block + the mid block, 4 latent channels,
/// SDXL's `text_time` added conditioning (6 time ids × 8 + the 16-wide pooled text).
pub fn tiny_unet_config() -> UNetConfig {
    UNetConfig {
        in_channels: 4,
        out_channels: 4,
        conv_in_kernel: 3,
        conv_out_kernel: 3,
        block_out_channels: vec![32, 64],
        layers_per_block: vec![1, 1],
        transformer_layers_per_block: vec![1, 1],
        num_attention_heads: vec![1, 2],
        cross_attention_dim: vec![TINY_CONTEXT_DIM, TINY_CONTEXT_DIM],
        norm_num_groups: 32,
        down_block_types: vec!["DownBlock2D".into(), "CrossAttnDownBlock2D".into()],
        // Stored already-reversed, like `UNetConfig::sdxl_base`.
        up_block_types: vec!["UpBlock2D".into(), "CrossAttnUpBlock2D".into()],
        addition_embed_type: Some("text_time".into()),
        addition_time_embed_dim: Some(8),
        projection_class_embeddings_input_dim: Some(TINY_POOLED_DIM + 6 * 8),
    }
}

struct Gen {
    w: Weights,
    seed: u64,
    n: u64,
}

impl Gen {
    fn rnd(&mut self, shape: &[i32], scale: f32) -> Result<Array> {
        self.n += 1;
        let key = random::key(self.seed.wrapping_mul(1_000_003).wrapping_add(self.n))?;
        Ok(multiply(
            &random::normal::<f32>(shape, None, None, Some(&key))?,
            Array::from_f32(scale),
        )?)
    }
    fn put(&mut self, key: String, shape: &[i32], scale: f32) -> Result<()> {
        let t = self.rnd(shape, scale)?;
        self.w.insert(key, t);
        Ok(())
    }
    /// A dense layer `[out, in]` (+ bias), fan-in scaled.
    fn linear(&mut self, p: &str, out: i32, inp: i32, bias: bool) -> Result<()> {
        self.put(
            format!("{p}.weight"),
            &[out, inp],
            (1.0 / inp as f32).sqrt(),
        )?;
        if bias {
            self.put(format!("{p}.bias"), &[out], 0.02)?;
        }
        Ok(())
    }
    /// A torch OIHW conv `[out, in, k, k]` + bias.
    fn conv(&mut self, p: &str, out: i32, inp: i32, k: i32) -> Result<()> {
        self.put(
            format!("{p}.weight"),
            &[out, inp, k, k],
            (1.0 / (inp * k * k) as f32).sqrt(),
        )?;
        self.put(format!("{p}.bias"), &[out], 0.02)
    }
    /// A norm's affine pair (weight ≈ 1).
    fn norm(&mut self, p: &str, c: i32) -> Result<()> {
        let ones = Array::ones::<f32>(&[c])?;
        let jitter = self.rnd(&[c], 0.02)?;
        self.w
            .insert(format!("{p}.weight"), mlx_rs::ops::add(&ones, &jitter)?);
        self.put(format!("{p}.bias"), &[c], 0.02)
    }
    fn resnet(&mut self, p: &str, inp: i32, out: i32, temb: i32) -> Result<()> {
        self.norm(&format!("{p}.norm1"), inp)?;
        self.conv(&format!("{p}.conv1"), out, inp, 3)?;
        self.linear(&format!("{p}.time_emb_proj"), out, temb, true)?;
        self.norm(&format!("{p}.norm2"), out)?;
        self.conv(&format!("{p}.conv2"), out, out, 3)?;
        if inp != out {
            self.conv(&format!("{p}.conv_shortcut"), out, inp, 1)?;
        }
        Ok(())
    }
    fn transformer(&mut self, p: &str, c: i32, ctx: i32, layers: i32) -> Result<()> {
        self.norm(&format!("{p}.norm"), c)?;
        self.linear(&format!("{p}.proj_in"), c, c, true)?;
        self.linear(&format!("{p}.proj_out"), c, c, true)?;
        for i in 0..layers {
            let b = format!("{p}.transformer_blocks.{i}");
            for n in ["norm1", "norm2", "norm3"] {
                self.norm(&format!("{b}.{n}"), c)?;
            }
            for (attn, kv) in [("attn1", c), ("attn2", ctx)] {
                self.linear(&format!("{b}.{attn}.to_q"), c, c, false)?;
                self.linear(&format!("{b}.{attn}.to_k"), c, kv, false)?;
                self.linear(&format!("{b}.{attn}.to_v"), c, kv, false)?;
                self.linear(&format!("{b}.{attn}.to_out.0"), c, c, true)?;
            }
            self.linear(&format!("{b}.ff.net.0.proj"), 8 * c, c, true)?;
            self.linear(&format!("{b}.ff.net.2"), c, 4 * c, true)?;
        }
        Ok(())
    }
}

/// Random weights for every key [`UNet2DConditionModel::from_weights`] reads under
/// [`tiny_unet_config`] (diffusers key layout, torch OIHW convs). Deterministic in `seed`.
pub fn tiny_unet_weights(seed: u64) -> Result<Weights> {
    tiny_unet_weights_for(&tiny_unet_config(), seed)
}

/// Three blocks (`[32, 32, 64]`), two resnets per block and cross-attention on the inner two — the
/// smallest SDXL-shaped U-Net whose down path yields the nine skip residuals a ControlNet's nine
/// zero-convs pair with (sc-26270).
pub fn tiny_control_unet_config() -> UNetConfig {
    UNetConfig {
        block_out_channels: vec![32, 32, 64],
        layers_per_block: vec![2, 2, 2],
        transformer_layers_per_block: vec![1, 1, 1],
        num_attention_heads: vec![1, 1, 2],
        cross_attention_dim: vec![TINY_CONTEXT_DIM; 3],
        down_block_types: vec![
            "DownBlock2D".into(),
            "CrossAttnDownBlock2D".into(),
            "CrossAttnDownBlock2D".into(),
        ],
        up_block_types: vec![
            "UpBlock2D".into(),
            "CrossAttnUpBlock2D".into(),
            "CrossAttnUpBlock2D".into(),
        ],
        ..tiny_unet_config()
    }
}

/// Random weights for a diffusers SDXL `ControlNetModel` over `cfg` (which must yield nine down
/// residuals, e.g. [`tiny_control_unet_config`]): the U-Net encoder copy, a small conditioning
/// embedding (`3 → 8 → … → block_out_channels[0]`, three stride-2 convs) and the nine + one 1x1
/// zero-convs. Deterministic in `seed`.
pub fn tiny_controlnet_weights(cfg: &UNetConfig, seed: u64) -> Result<Weights> {
    let unet = tiny_unet_weights_for(cfg, seed)?;
    let mut g = Gen {
        w: Weights::empty(),
        seed: seed.wrapping_add(17),
        n: 0,
    };
    for key in unet.keys().map(str::to_owned).collect::<Vec<_>>() {
        if !(key.starts_with("up_blocks.")
            || key.starts_with("conv_norm_out")
            || key.starts_with("conv_out"))
        {
            let value = unet.get(&key).expect("listed key").clone();
            g.w.insert(key, value);
        }
    }
    let boc = &cfg.block_out_channels;
    g.conv("controlnet_cond_embedding.conv_in", 8, 3, 3)?;
    for i in 0..6 {
        g.conv(&format!("controlnet_cond_embedding.blocks.{i}"), 8, 8, 3)?;
    }
    g.conv("controlnet_cond_embedding.conv_out", boc[0], 8, 3)?;
    // The skip-residual channel sequence: stem, then each down block's resnets and downsample.
    let mut channels = vec![boc[0]];
    for (i, c) in boc.iter().enumerate() {
        for _ in 0..cfg.layers_per_block[i] {
            channels.push(*c);
        }
        if i + 1 < boc.len() {
            channels.push(*c);
        }
    }
    for (i, c) in channels.iter().enumerate() {
        g.conv(&format!("controlnet_down_blocks.{i}"), *c, *c, 1)?;
    }
    let mid = *boc.last().expect("non-empty config");
    g.conv("controlnet_mid_block", mid, mid, 1)?;
    Ok(g.w)
}

/// [`tiny_unet_weights`] for an arbitrary SDXL-shaped `cfg`.
pub fn tiny_unet_weights_for(cfg: &UNetConfig, seed: u64) -> Result<Weights> {
    let cfg = cfg.clone();
    let boc = cfg.block_out_channels.clone();
    let n = boc.len();
    let temb = cfg.time_embed_dim();
    let ctx = cfg.cross_attention_dim[0];
    let mut g = Gen {
        w: Weights::empty(),
        seed,
        n: 0,
    };
    g.conv("conv_in", boc[0], cfg.in_channels, 3)?;
    g.linear("time_embedding.linear_1", temb, boc[0], true)?;
    g.linear("time_embedding.linear_2", temb, temb, true)?;
    g.linear(
        "add_embedding.linear_1",
        temb,
        cfg.projection_class_embeddings_input_dim.unwrap(),
        true,
    )?;
    g.linear("add_embedding.linear_2", temb, temb, true)?;
    for i in 0..n {
        let p = format!("down_blocks.{i}");
        let inp = if i == 0 { boc[0] } else { boc[i - 1] };
        for j in 0..cfg.layers_per_block[i] {
            let rin = if j == 0 { inp } else { boc[i] };
            g.resnet(&format!("{p}.resnets.{j}"), rin, boc[i], temb)?;
            if cfg.down_block_types[i].contains("CrossAttn") {
                g.transformer(
                    &format!("{p}.attentions.{j}"),
                    boc[i],
                    ctx,
                    cfg.transformer_layers_per_block[i],
                )?;
            }
        }
        if i < n - 1 {
            g.conv(&format!("{p}.downsamplers.0.conv"), boc[i], boc[i], 3)?;
        }
    }
    let c = boc[n - 1];
    g.resnet("mid_block.resnets.0", c, c, temb)?;
    g.transformer(
        "mid_block.attentions.0",
        c,
        ctx,
        cfg.transformer_layers_per_block[n - 1],
    )?;
    g.resnet("mid_block.resnets.1", c, c, temb)?;
    for k in 0..n {
        let ci = n - 1 - k;
        let p = format!("up_blocks.{k}");
        let out = boc[ci];
        let prev = if k == 0 { boc[n - 1] } else { boc[ci + 1] };
        let input = boc[ci.saturating_sub(1)];
        let layers = cfg.layers_per_block[ci] + 1;
        for j in 0..layers {
            let skip = if j < layers - 1 { out } else { input };
            let rin = if j == 0 { prev } else { out };
            g.resnet(&format!("{p}.resnets.{j}"), rin + skip, out, temb)?;
            if cfg.up_block_types[ci].contains("CrossAttn") {
                g.transformer(
                    &format!("{p}.attentions.{j}"),
                    out,
                    ctx,
                    cfg.transformer_layers_per_block[ci],
                )?;
            }
        }
        if ci > 0 {
            g.conv(&format!("{p}.upsamplers.0.conv"), out, out, 3)?;
        }
    }
    g.norm("conv_norm_out", boc[0])?;
    g.conv("conv_out", cfg.out_channels, boc[0], 3)?;
    Ok(g.w)
}

/// The tiny U-Net built from [`tiny_unet_weights`]; every generated key is consumed.
pub fn tiny_unet(seed: u64) -> Result<UNet2DConditionModel> {
    let w = tiny_unet_weights(seed)?;
    let unet = UNet2DConditionModel::from_weights(&w, &tiny_unet_config())?;
    let unused = w.unused_keys();
    if !unused.is_empty() {
        return Err(mlx_gen::Error::Msg(format!(
            "tiny U-Net fixture: unconsumed keys {unused:?}"
        )));
    }
    Ok(unet)
}
