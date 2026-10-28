//! sc-26270 (E2): the Mac default SDXL txt2img / img2img path, pinned against a golden produced by
//! the PRE-epic code (inference rev `91ffb4a81`) on a tiny deterministic CPU fixture.
//!
//! The fixture is built entirely from host `sin` arithmetic (no RNG), so the weights, conditioning
//! and init image are bit-identical on every machine; the only RNG is MLX's seeded threefry stream
//! (the prior and the ancestral noise), exactly as the production generate draws it. This file's
//! fixture and [`run_default_paths`] were copied verbatim into a scratch worktree at `91ffb4a81` and
//! run there with `SDXL_GOLDEN_WRITE` set to produce
//! `tests/fixtures/sdxl_default_path_golden.safetensors` (the two tests below that name post-base
//! APIs were left out of that copy).
//!
//! The chain is the default Mac generate's — the ancestral Euler schedule, CFG 7 over a cond/uncond
//! pair, the U-Net and the f32 VAE; img2img encodes the init image with the whole-image
//! [`crate::pipeline::encode_init_latents`] (the Mac default — no tiling), noises it to
//! `max_time · 0.8` (`DEFAULT_STRENGTH`) and runs `int(steps · 0.8)` steps — computed at
//! [`CHAIN_DTYPE`] = **f32**, not the production fp16 (`model::DTYPE`, pinned exactly by
//! [`the_default_compute_dtype_is_fp16`]).
//!
//! Why f32: the golden has to hold on another Apple CPU, whose BLAS GEMM and vectorized
//! transcendentals round differently in the last f32 place. In an fp16 chain those differences flip
//! fp16 roundings, and CFG 7 over this random fixture amplifies them: one fp16 ulp on 1% of the
//! prior's elements moves the fp16 chain's latents by 1.7e-2 of peak even at ONE step — a
//! cross-machine golden on it cannot separate machine noise from a behaviour change. In f32 a 1e-6
//! relative perturbation of the prior moves the latents by 6e-5 of peak (measured), so latents are
//! held to `1e-3` of peak and pixels to `±2` of 255 — still far below a behaviour change (a
//! different noise draw, schedule, CFG or graph moves latents by O(1) of peak).
//!
//! Item 3 (the opt-in bounded-decode rung's tiled img2img encode,
//! [`crate::pipeline::encode_init_latents_tiled`]) is pinned against the SAME golden's whole-image
//! encode from `91ffb4a81` — the old code's own output rather than today's untiled path, so a change
//! to the untiled encode could not silently move the reference — within the tiled-decode parity
//! standard for an f32 VAE (`1e-4` of peak; `quantum_tests::tiled_encode_meets_the_tiled_parity_
//! standard`).

use mlx_rs::{Array, Dtype};

use crate::config::{DiffusionConfig, UNetConfig, VaeConfig};
use mlx_gen::weights::Weights;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sdxl_default_path_golden.safetensors"
);
const CTX: i32 = 16;
const POOLED: i32 = 16;
const WIDTH: u32 = 32;
const HEIGHT: u32 = 24;
const STEPS: usize = 3;
/// The golden chain's compute dtype (see the module doc).
const CHAIN_DTYPE: Dtype = Dtype::Float32;
const CFG: f32 = 7.0;
const STRENGTH: f32 = 0.8;
const SEED: u64 = 4242;

/// Deterministic host values: `sin((i + phase) · 0.071) · scale`.
fn values(shape: &[i32], phase: f32, scale: f32) -> Array {
    let count = shape.iter().product::<i32>();
    let v: Vec<f32> = (0..count)
        .map(|i| ((i as f32 + phase) * 0.071).sin() * scale)
        .collect();
    Array::from_slice(&v, shape)
}

struct Gen {
    w: Weights,
    phase: f32,
}

impl Gen {
    fn put(&mut self, key: String, shape: &[i32], scale: f32) {
        self.phase += 7.3;
        self.w.insert(key, values(shape, self.phase, scale));
    }
    fn linear(&mut self, p: &str, out: i32, inp: i32, bias: bool) {
        self.put(
            format!("{p}.weight"),
            &[out, inp],
            1.5 / (inp as f32).sqrt(),
        );
        if bias {
            self.put(format!("{p}.bias"), &[out], 0.02);
        }
    }
    fn conv(&mut self, p: &str, out: i32, inp: i32, k: i32) {
        self.put(
            format!("{p}.weight"),
            &[out, inp, k, k],
            1.5 / ((inp * k * k) as f32).sqrt(),
        );
        self.put(format!("{p}.bias"), &[out], 0.02);
    }
    fn norm(&mut self, p: &str, c: i32) {
        self.phase += 3.1;
        let w = mlx_rs::ops::add(values(&[c], self.phase, 0.05), Array::from_f32(1.0)).unwrap();
        self.w.insert(format!("{p}.weight"), w);
        self.put(format!("{p}.bias"), &[c], 0.02);
    }
    fn resnet(&mut self, p: &str, inp: i32, out: i32, temb: Option<i32>) {
        self.norm(&format!("{p}.norm1"), inp);
        self.conv(&format!("{p}.conv1"), out, inp, 3);
        if let Some(t) = temb {
            self.linear(&format!("{p}.time_emb_proj"), out, t, true);
        }
        self.norm(&format!("{p}.norm2"), out);
        self.conv(&format!("{p}.conv2"), out, out, 3);
        if inp != out {
            self.conv(&format!("{p}.conv_shortcut"), out, inp, 1);
        }
    }
    fn transformer(&mut self, p: &str, c: i32) {
        self.norm(&format!("{p}.norm"), c);
        self.linear(&format!("{p}.proj_in"), c, c, true);
        self.linear(&format!("{p}.proj_out"), c, c, true);
        let b = format!("{p}.transformer_blocks.0");
        for n in ["norm1", "norm2", "norm3"] {
            self.norm(&format!("{b}.{n}"), c);
        }
        for (attn, kv) in [("attn1", c), ("attn2", CTX)] {
            self.linear(&format!("{b}.{attn}.to_q"), c, c, false);
            self.linear(&format!("{b}.{attn}.to_k"), c, kv, false);
            self.linear(&format!("{b}.{attn}.to_v"), c, kv, false);
            self.linear(&format!("{b}.{attn}.to_out.0"), c, c, true);
        }
        self.linear(&format!("{b}.ff.net.0.proj"), 8 * c, c, true);
        self.linear(&format!("{b}.ff.net.2"), c, 4 * c, true);
    }
}

fn unet_config() -> UNetConfig {
    UNetConfig {
        in_channels: 4,
        out_channels: 4,
        conv_in_kernel: 3,
        conv_out_kernel: 3,
        block_out_channels: vec![32, 64],
        layers_per_block: vec![1, 1],
        transformer_layers_per_block: vec![1, 1],
        num_attention_heads: vec![1, 2],
        cross_attention_dim: vec![CTX, CTX],
        norm_num_groups: 32,
        down_block_types: vec!["DownBlock2D".into(), "CrossAttnDownBlock2D".into()],
        up_block_types: vec!["UpBlock2D".into(), "CrossAttnUpBlock2D".into()],
        addition_embed_type: Some("text_time".into()),
        addition_time_embed_dim: Some(8),
        projection_class_embeddings_input_dim: Some(POOLED + 6 * 8),
    }
}

/// The tiny U-Net (two blocks, cross-attention on the inner block and the mid block) at the
/// production fp16 compute dtype.
fn unet() -> crate::unet::UNet2DConditionModel {
    let mut g = Gen {
        w: Weights::empty(),
        phase: 0.0,
    };
    let temb = 4 * 32;
    g.linear("time_embedding.linear_1", temb, 32, true);
    g.linear("time_embedding.linear_2", temb, temb, true);
    g.linear("add_embedding.linear_1", temb, POOLED + 6 * 8, true);
    g.linear("add_embedding.linear_2", temb, temb, true);
    g.conv("conv_in", 32, 4, 3);
    g.resnet("down_blocks.0.resnets.0", 32, 32, Some(temb));
    g.conv("down_blocks.0.downsamplers.0.conv", 32, 32, 3);
    g.resnet("down_blocks.1.resnets.0", 32, 64, Some(temb));
    g.transformer("down_blocks.1.attentions.0", 64);
    g.resnet("mid_block.resnets.0", 64, 64, Some(temb));
    g.transformer("mid_block.attentions.0", 64);
    g.resnet("mid_block.resnets.1", 64, 64, Some(temb));
    // Up block 0 (64 channels, cross-attention): skips 64 (down 1) and 32 (the downsample).
    g.resnet("up_blocks.0.resnets.0", 128, 64, Some(temb));
    g.transformer("up_blocks.0.attentions.0", 64);
    g.resnet("up_blocks.0.resnets.1", 96, 64, Some(temb));
    g.transformer("up_blocks.0.attentions.1", 64);
    g.conv("up_blocks.0.upsamplers.0.conv", 64, 64, 3);
    // Up block 1 (32 channels): skips 32 (down 0) and 32 (conv_in).
    g.resnet("up_blocks.1.resnets.0", 96, 32, Some(temb));
    g.resnet("up_blocks.1.resnets.1", 64, 32, Some(temb));
    g.norm("conv_norm_out", 32);
    g.conv("conv_out", 4, 32, 3);
    let mut w = g.w;
    w.cast_all(CHAIN_DTYPE).unwrap();
    crate::unet::UNet2DConditionModel::from_weights(&w, &unet_config()).unwrap()
}

/// The tiny f32 VAE (32 channels, one 2x down/upsample).
fn vae() -> crate::vae::Autoencoder {
    const C: i32 = 32;
    let mut g = Gen {
        w: Weights::empty(),
        phase: 1000.0,
    };
    let mid = |g: &mut Gen, p: &str| {
        g.resnet(&format!("{p}.resnets.0"), C, C, None);
        g.norm(&format!("{p}.attentions.0.group_norm"), C);
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            g.linear(&format!("{p}.attentions.0.{n}"), C, C, true);
        }
        g.resnet(&format!("{p}.resnets.1"), C, C, None);
    };
    g.conv("decoder.conv_in", C, 4, 3);
    mid(&mut g, "decoder.mid_block");
    for block in 0..2 {
        for resnet in 0..2 {
            g.resnet(
                &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
                C,
                C,
                None,
            );
        }
    }
    g.conv("decoder.up_blocks.0.upsamplers.0.conv", C, C, 3);
    g.norm("decoder.conv_norm_out", C);
    g.conv("decoder.conv_out", 3, C, 3);
    g.conv("encoder.conv_in", C, 3, 3);
    for block in 0..2 {
        g.resnet(
            &format!("encoder.down_blocks.{block}.resnets.0"),
            C,
            C,
            None,
        );
    }
    g.conv("encoder.down_blocks.0.downsamplers.0.conv", C, C, 3);
    mid(&mut g, "encoder.mid_block");
    g.norm("encoder.conv_norm_out", C);
    g.conv("encoder.conv_out", 8, C, 3);
    g.conv("quant_conv", 8, 8, 1);
    g.conv("post_quant_conv", 4, 4, 1);
    let cfg = VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels_out: 8,
        latent_channels_in: 4,
        block_out_channels: vec![C, C],
        layers_per_block: 1,
        norm_num_groups: 32,
        scaling_factor: 0.13025,
    };
    crate::vae::Autoencoder::from_weights(&g.w, &cfg).unwrap()
}

/// `(conditioning, pooled, time ids)` for the CFG pair (cond row first), fp16 like the encoders'.
fn conditioning() -> (Array, Array, Array) {
    let half = |a: Array| a.as_dtype(CHAIN_DTYPE).unwrap();
    (
        half(values(&[2, 5, CTX], 3.0, 1.0)),
        half(values(&[2, POOLED], 5.0, 1.0)),
        crate::pipeline::text_time_ids(2),
    )
}

fn init_image() -> mlx_gen::Image {
    let n = (WIDTH * HEIGHT * 3) as usize;
    mlx_gen::Image {
        width: WIDTH,
        height: HEIGHT,
        pixels: (0..n).map(|i| (i * 37 % 251) as u8).collect(),
    }
}

fn pixels(image: &mlx_gen::Image) -> Array {
    let v: Vec<f32> = image.pixels.iter().map(|&p| p as f32).collect();
    Array::from_slice(&v, &[v.len() as i32])
}

/// Run the Mac default txt2img and img2img chains; returns the named outputs.
fn run_default_paths() -> Vec<(&'static str, Array)> {
    use crate::pipeline::{decode_image, denoise, encode_init_latents, Denoiser};
    use crate::sampler::{AncestralEuler, EulerSampler};
    let unet = unet();
    let vae = vae();
    // The production sampler at the production compute dtype (`model::DTYPE`, fp16).
    let euler =
        EulerSampler::new_with_dtype(&DiffusionConfig::sdxl_base(), true, CHAIN_DTYPE).unwrap();
    let (cond, pooled, ids) = conditioning();
    let cancel = mlx_gen::CancelFlag::default();
    let max_time = euler.max_time();

    // txt2img: seeded prior, the full ancestral schedule.
    mlx_rs::random::seed(SEED).unwrap();
    let shape = [1, (HEIGHT / 2) as i32, (WIDTH / 2) as i32, 4];
    let prior = euler.sample_prior(&shape).unwrap();
    let sampler = AncestralEuler::new(&euler, STEPS, max_time).unwrap();
    let t2i = (
        crate::test_device::record_model_inference_device(),
        denoise(
            &Denoiser::new(&unet, &sampler),
            prior,
            &cond,
            &pooled,
            &ids,
            CFG,
            &cancel,
            &mut |_| {},
        ),
    )
        .1
        .unwrap();
    let t2i_pixels = pixels(
        &(
            crate::test_device::record_model_inference_device(),
            decode_image(&vae, &t2i, None),
        )
            .1
            .unwrap(),
    );

    // img2img: the whole-image init encode, noised to `max_time · strength`.
    let init = (
        crate::test_device::record_model_inference_device(),
        encode_init_latents(&vae, &init_image(), WIDTH, HEIGHT),
    )
        .1
        .unwrap();
    mlx_rs::random::seed(SEED + 1).unwrap();
    let eff = (STEPS as f32 * STRENGTH) as usize;
    let start = max_time * STRENGTH;
    let x_t = euler.add_noise(&init, start).unwrap();
    let sampler = AncestralEuler::new(&euler, eff, start).unwrap();
    let i2i = (
        crate::test_device::record_model_inference_device(),
        denoise(
            &Denoiser::new(&unet, &sampler),
            x_t,
            &cond,
            &pooled,
            &ids,
            CFG,
            &cancel,
            &mut |_| {},
        ),
    )
        .1
        .unwrap();
    let i2i_pixels = pixels(
        &(
            crate::test_device::record_model_inference_device(),
            decode_image(&vae, &i2i, None),
        )
            .1
            .unwrap(),
    );
    // Row-major f32 copies: an output may be a strided view (the encode's channel gather), whose
    // raw buffer is not in logical order.
    let f32 = |a: Array| Array::from_slice(&host(&a), a.shape());
    vec![
        ("txt2img_latents", f32(t2i)),
        ("txt2img_pixels", t2i_pixels),
        ("img2img_init_latents", f32(init)),
        ("img2img_latents", f32(i2i)),
        ("img2img_pixels", i2i_pixels),
    ]
}

fn cpu() {
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(crate::test_metal_guard::forbid_metal());
}

/// `a`'s values as f32 in logical row-major order (a flattening reshape copies a strided view).
fn host(a: &Array) -> Vec<f32> {
    let flat = a.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
    flat.eval().unwrap();
    flat.as_slice::<f32>().to_vec()
}

fn relative_max_delta(reference: &Array, candidate: &Array) -> f32 {
    let (r, c) = (host(reference), host(candidate));
    assert_eq!(r.len(), c.len());
    let peak = r.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    r.iter().zip(c).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / peak
}

fn max_abs_delta(reference: &Array, candidate: &Array) -> f32 {
    let (r, c) = (host(reference), host(candidate));
    assert_eq!(r.len(), c.len());
    r.iter()
        .zip(&c)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
}

/// Re-run test `name` in a child pinned to the CPU device with Metal made fatal.
fn cpu_child(name: &str) -> bool {
    const CHILD: &str = "SDXL_DEFAULT_PATH_GOLDEN_CHILD";
    if std::env::var_os(CHILD).is_some() {
        cpu();
        return true;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("default_path_golden::{name}"),
            "--nocapture",
            "--ignored",
        ])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success(), "CPU child for {name} failed");
    false
}

fn golden() -> Weights {
    Weights::from_file(std::path::Path::new(GOLDEN)).expect("the committed golden")
}

/// The Mac default txt2img / img2img outputs match the pre-epic (`91ffb4a81`) golden. With
/// `SDXL_GOLDEN_WRITE=<path>` set it writes that golden instead (how it was produced, at the base).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn default_txt2img_and_img2img_match_the_base_rev_golden() {
    if !cpu_child("default_txt2img_and_img2img_match_the_base_rev_golden") {
        return;
    }
    let outputs = (
        crate::test_device::record_model_inference_device(),
        run_default_paths(),
    )
        .1;
    if let Some(path) = std::env::var_os("SDXL_GOLDEN_WRITE") {
        let named: Vec<(&str, &Array)> = outputs.iter().map(|(k, a)| (*k, a)).collect();
        Array::save_safetensors(named, None, std::path::Path::new(&path)).unwrap();
        eprintln!("wrote the default-path golden to {path:?}");
        return;
    }
    let golden = golden();
    for (name, out) in &outputs {
        let reference = golden.require(name).unwrap();
        assert_eq!(reference.shape(), out.shape(), "{name} shape");
        if name.ends_with("pixels") {
            let delta = max_abs_delta(reference, out);
            eprintln!("GOLDEN {name}: max |Δ| {delta} of 255");
            assert!(
                delta <= 2.0,
                "{name} moved {delta} levels from the base rev"
            );
        } else {
            let delta = relative_max_delta(reference, out);
            eprintln!("GOLDEN {name}: relative max Δ {delta:.3e}");
            assert!(
                delta <= 1e-3,
                "{name} moved {delta:.3e} of peak from the base rev"
            );
        }
    }

    println!("\nMODEL_INFERENCE_BODY_COMPLETED default_path_golden::default_txt2img_and_img2img_match_the_base_rev_golden");
}

/// The production compute dtype the f32 golden chain stands in for (see the module doc): the
/// default Mac generate's U-Net, text encoders and sampler run fp16.
#[test]
fn the_default_compute_dtype_is_fp16() {
    assert_eq!(crate::model::DTYPE, Dtype::Float16);
}

/// Item 3: the bounded-decode rung's tiled img2img encode tracks the base rev's whole-image encode
/// of the same init image within the f32 tiled parity standard, at an even and a ragged tile edge.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn tiled_init_encode_tracks_the_base_rev_whole_image_encode() {
    if !cpu_child("tiled_init_encode_tracks_the_base_rev_whole_image_encode") {
        return;
    }
    let reference = golden().require("img2img_init_latents").unwrap().clone();
    let vae = vae();
    for edge in [8, 12, 16] {
        let tiling = mlx_gen::tiling::TilingConfig::spatial_only(edge, 0);
        let tiled = (
            crate::test_device::record_model_inference_device(),
            crate::pipeline::encode_init_latents_tiled(
                &vae,
                &init_image(),
                WIDTH,
                HEIGHT,
                Some(&tiling),
                None,
            ),
        )
            .1
            .unwrap();
        let delta = relative_max_delta(&reference, &tiled);
        eprintln!("TILED ENCODE edge {edge}: relative max Δ {delta:.3e} vs 91ffb4a81");
        assert!(
            delta <= 1e-4,
            "tiled encode at edge {edge} drifted {delta:.3e}"
        );
    }
    // The untiled route (no rung) is still the whole-image encode.
    let untiled = (
        crate::test_device::record_model_inference_device(),
        crate::pipeline::encode_init_latents_tiled(&vae, &init_image(), WIDTH, HEIGHT, None, None),
    )
        .1
        .unwrap();
    let delta = relative_max_delta(&reference, &untiled);
    eprintln!("UNTILED ENCODE: relative max Δ {delta:.3e} vs 91ffb4a81");
    assert!(delta <= 1e-4);

    println!("\nMODEL_INFERENCE_BODY_COMPLETED default_path_golden::tiled_init_encode_tracks_the_base_rev_whole_image_encode");
}
