// Tiny CPU fixture shared VERBATIM by the base-rev golden generator and the in-crate golden tests
// (`pipeline::quantum_tests::default_paths_match_the_base_rev_goldens`, ...). It is `include!`d into
// a module that brings `Array`, `HashMap`, `Image`, `BlockType`, `DcAeConfig`,
// `SanaTransformerConfig` and `SanaConditioning` into scope, so the same items compile against
// `91ffb4a81`'s public API (an integration test there) and against this crate (`crate::` paths).
//
// Every input is fixed: seeded MLX normal draws (the same pinned MLX at both revs), fixed
// position-dependent image bytes, a fixed request seed, steps and strength.

/// Request geometry and knobs of the pipeline goldens.
pub const WIDTH: u32 = 128;
pub const HEIGHT: u32 = 96;
pub const SEED: u64 = 5;
pub const STRENGTH: f32 = 0.6;
pub const GUIDANCE: f32 = 4.5;
pub const SPRINT_GUIDANCE_EMBEDS_SCALE: f32 = 0.1;

/// The img2img-encode golden's image: large enough that a 192-pixel encode tile splits both axes.
pub const ENCODE_WIDTH: u32 = 448;
pub const ENCODE_HEIGHT: u32 = 384;

pub fn rand(shape: &[i32], seed: u64) -> Array {
    mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
        .unwrap()
}

/// One-block trunk at dim 64 (4 heads x 16), Mix-FFN ratio 2.5, a 96-wide caption; Sprint adds the
/// guidance embedder and the qk RMSNorms.
pub fn trunk_config(sprint: bool) -> SanaTransformerConfig {
    SanaTransformerConfig {
        num_attention_heads: 4,
        attention_head_dim: 16,
        num_cross_attention_heads: 4,
        cross_attention_head_dim: 16,
        num_layers: 1,
        mlp_ratio: 2.5,
        caption_channels: 96,
        guidance_embeds: sprint,
        qk_norm: sprint,
        ..SanaTransformerConfig::sana_1600m()
    }
}

/// F16 trunk weights (the shipped storage dtype), scaled N(0, 0.05²).
pub fn trunk_weights(cfg: &SanaTransformerConfig) -> HashMap<String, Array> {
    let mut map = HashMap::new();
    let mut seed = 7000u64;
    let mut put = |key: String, shape: &[i32]| {
        seed += 1;
        let a = rand(shape, seed)
            .multiply(Array::from_slice(&[0.05f32], &[1]))
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float16)
            .unwrap();
        a.eval().unwrap();
        map.insert(key, a);
    };
    let d = cfg.num_attention_heads * cfg.attention_head_dim;
    let p = cfg.patch_size;
    put(
        "patch_embed.proj.weight".into(),
        &[d, cfg.in_channels, p, p],
    );
    put("patch_embed.proj.bias".into(), &[d]);
    let ts = if cfg.guidance_embeds {
        put(
            "time_embed.guidance_embedder.linear_1.weight".into(),
            &[d, 256],
        );
        put("time_embed.guidance_embedder.linear_1.bias".into(), &[d]);
        put(
            "time_embed.guidance_embedder.linear_2.weight".into(),
            &[d, d],
        );
        put("time_embed.guidance_embedder.linear_2.bias".into(), &[d]);
        "time_embed.timestep_embedder"
    } else {
        "time_embed.emb.timestep_embedder"
    };
    put(format!("{ts}.linear_1.weight"), &[d, 256]);
    put(format!("{ts}.linear_1.bias"), &[d]);
    put(format!("{ts}.linear_2.weight"), &[d, d]);
    put(format!("{ts}.linear_2.bias"), &[d]);
    put("time_embed.linear.weight".into(), &[6 * d, d]);
    put("time_embed.linear.bias".into(), &[6 * d]);
    put(
        "caption_projection.linear_1.weight".into(),
        &[d, cfg.caption_channels],
    );
    put("caption_projection.linear_1.bias".into(), &[d]);
    put("caption_projection.linear_2.weight".into(), &[d, d]);
    put("caption_projection.linear_2.bias".into(), &[d]);
    put("caption_norm.weight".into(), &[d]);
    put("scale_shift_table".into(), &[2, d]);
    put("proj_out.weight".into(), &[p * p * cfg.out_channels, d]);
    put("proj_out.bias".into(), &[p * p * cfg.out_channels]);
    let hidden = (cfg.mlp_ratio * d as f32) as i32;
    for i in 0..cfg.num_layers {
        let b = format!("transformer_blocks.{i}");
        put(format!("{b}.scale_shift_table"), &[6, d]);
        for leaf in ["to_q", "to_k", "to_v"] {
            put(format!("{b}.attn1.{leaf}.weight"), &[d, d]);
            put(format!("{b}.attn2.{leaf}.weight"), &[d, d]);
            put(format!("{b}.attn2.{leaf}.bias"), &[d]);
        }
        for a in ["attn1", "attn2"] {
            put(format!("{b}.{a}.to_out.0.weight"), &[d, d]);
            put(format!("{b}.{a}.to_out.0.bias"), &[d]);
            if cfg.qk_norm {
                put(format!("{b}.{a}.norm_q.weight"), &[d]);
                put(format!("{b}.{a}.norm_k.weight"), &[d]);
            }
        }
        put(
            format!("{b}.ff.conv_inverted.weight"),
            &[2 * hidden, d, 1, 1],
        );
        put(format!("{b}.ff.conv_inverted.bias"), &[2 * hidden]);
        put(format!("{b}.ff.conv_depth.weight"), &[2 * hidden, 1, 3, 3]);
        put(format!("{b}.ff.conv_depth.bias"), &[2 * hidden]);
        put(format!("{b}.ff.conv_point.weight"), &[d, hidden, 1, 1]);
    }
    map
}

/// A six-stage (x32) DC-AE with SANA's block layout (three ResBlock stages, three EfficientViT).
pub fn dcae_config() -> DcAeConfig {
    use BlockType::{EfficientVit as E, Res as R};
    DcAeConfig {
        in_channels: 3,
        latent_channels: 32,
        attention_head_dim: 8,
        block_out_channels: vec![8, 8, 8, 16, 32, 32],
        layers_per_block: vec![1; 6],
        encoder_layers_per_block: vec![1; 6],
        block_types: vec![R, R, R, E, E, E],
        qkv_multiscales: vec![5],
        upsample_interpolate: true,
        norm_eps: 1e-5,
        attn_eps: 1e-15,
        scaling_factor: 0.41407,
    }
}

/// F32 DC-AE encoder + decoder weights, scaled N(0, 0.1²).
pub fn dcae_weights(cfg: &DcAeConfig) -> HashMap<String, Array> {
    let mut map = HashMap::new();
    let mut seed = 4000u64;
    let mut put = |key: String, shape: &[i32]| {
        seed += 1;
        let a = rand(shape, seed)
            .multiply(Array::from_slice(&[0.1f32], &[1]))
            .unwrap();
        a.eval().unwrap();
        map.insert(key, a);
    };
    type Put<'a> = &'a mut dyn FnMut(String, &[i32]);
    fn conv(put: Put<'_>, p: &str, o: i32, i: i32, k: i32, bias: bool) {
        put(format!("{p}.weight"), &[o, i, k, k]);
        if bias {
            put(format!("{p}.bias"), &[o]);
        }
    }
    fn affine(put: Put<'_>, p: &str, c: i32) {
        put(format!("{p}.weight"), &[c]);
        put(format!("{p}.bias"), &[c]);
    }
    fn block(put: Put<'_>, cfg: &DcAeConfig, p: &str, c: i32, ty: BlockType) {
        match ty {
            BlockType::Res => {
                conv(put, &format!("{p}.conv1"), c, c, 3, true);
                conv(put, &format!("{p}.conv2"), c, c, 3, false);
                affine(put, &format!("{p}.norm"), c);
            }
            BlockType::EfficientVit => {
                let heads = c / cfg.attention_head_dim;
                for n in ["to_q", "to_k", "to_v"] {
                    put(format!("{p}.attn.{n}.weight"), &[c, c]);
                }
                let scales = cfg.qkv_multiscales.len() as i32;
                put(format!("{p}.attn.to_out.weight"), &[c, c * (1 + scales)]);
                for (i, k) in cfg.qkv_multiscales.iter().enumerate() {
                    let m = format!("{p}.attn.to_qkv_multiscale.{i}");
                    conv(put, &format!("{m}.proj_in"), 3 * c, 1, *k, false);
                    conv(
                        put,
                        &format!("{m}.proj_out"),
                        3 * c,
                        3 * c / (3 * heads),
                        1,
                        false,
                    );
                }
                affine(put, &format!("{p}.attn.norm_out"), c);
                let hidden = 4 * c;
                conv(
                    put,
                    &format!("{p}.conv_out.conv_inverted"),
                    2 * hidden,
                    c,
                    1,
                    true,
                );
                conv(
                    put,
                    &format!("{p}.conv_out.conv_depth"),
                    2 * hidden,
                    1,
                    3,
                    true,
                );
                conv(
                    put,
                    &format!("{p}.conv_out.conv_point"),
                    c,
                    hidden,
                    1,
                    false,
                );
                affine(put, &format!("{p}.conv_out.norm"), c);
            }
        }
    }
    let n = cfg.block_out_channels.len();
    let ch = cfg.block_out_channels.clone();
    conv(
        &mut put,
        "decoder.conv_in",
        ch[n - 1],
        cfg.latent_channels,
        3,
        true,
    );
    conv(&mut put, "encoder.conv_in", ch[0], cfg.in_channels, 3, true);
    for i in 0..n {
        let has_up = i + 1 < n;
        if has_up {
            conv(
                &mut put,
                &format!("decoder.up_blocks.{i}.0.conv"),
                ch[i],
                ch[i + 1],
                3,
                true,
            );
            let layers = cfg.encoder_layers_per_block[i];
            conv(
                &mut put,
                &format!("encoder.down_blocks.{i}.{layers}.conv"),
                ch[i + 1],
                ch[i],
                3,
                true,
            );
        }
        for j in 0..cfg.layers_per_block[i] {
            let p = format!("decoder.up_blocks.{i}.{}", j + i32::from(has_up));
            block(&mut put, cfg, &p, ch[i], cfg.block_types[i]);
        }
        for j in 0..cfg.encoder_layers_per_block[i] {
            block(
                &mut put,
                cfg,
                &format!("encoder.down_blocks.{i}.{j}"),
                ch[i],
                cfg.block_types[i],
            );
        }
    }
    affine(&mut put, "decoder.norm_out", ch[0]);
    conv(
        &mut put,
        "decoder.conv_out",
        cfg.in_channels,
        ch[0],
        3,
        true,
    );
    conv(
        &mut put,
        "encoder.conv_out",
        cfg.latent_channels,
        ch[n - 1],
        3,
        true,
    );
    map
}

/// Position-dependent RGB bytes (a seam or a resize change shows).
pub fn image(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: (0..width * height * 3)
            .map(|i| ((i * 53 + (i / (3 * width)) * 7) % 241) as u8)
            .collect(),
    }
}

/// Fixed caption conditioning; base SANA carries the CFG uncond branch, Sprint is CFG-free.
pub fn conditioning(cfg: &SanaTransformerConfig, sprint: bool) -> SanaConditioning {
    let cond = SanaConditioning {
        cond: rand(&[1, 300, cfg.caption_channels], 11),
        cond_mask: Array::ones::<f32>(&[1, 300]).unwrap(),
        uncond: (!sprint).then(|| {
            (
                rand(&[1, 300, cfg.caption_channels], 12),
                Array::ones::<f32>(&[1, 300]).unwrap(),
            )
        }),
    };
    cond.cond.eval().unwrap();
    if let Some((u, _)) = &cond.uncond {
        u.eval().unwrap();
    }
    cond
}

/// An image's bytes as a u8 `[H, W, 3]` array (the golden's storage form).
pub fn pixels_array(image: &Image) -> Array {
    Array::from_slice(&image.pixels, &[image.height as i32, image.width as i32, 3])
}
