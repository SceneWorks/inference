//! The registered generator type's `Generator` surface on a tiny random-weight pipeline (sc-24217),
//! on the MLX **CPU** device so every lane that builds this crate runs it:
//!
//! * `generate_with_report` reports `prompt_truncated` (with the token counts) exactly when the
//!   prompt overflows the context, and `generate` returns the same render without the report;
//! * `Generator::generate` honours a pre-tripped cancel flag and one tripped mid-denoise.
//!
//! The parts are [`Supra2Image::from_parts`]'s: a word-level tokenizer with Flan-T5's special
//! tokens and `</s>` post-processor, the tiny T5 parity fixture (2 blocks × 4 heads, width 32), a
//! deterministic tiny DiT and a two-stage AutoencoderKL.

use std::collections::HashMap;
use std::path::Path;

use gen_core_checkpoint::supra2::{
    Supra2Config, Supra2Tokenizer, PROMPT_TRUNCATED_WARNING, VAE_SCALE,
};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, GenerationOutput, GenerationRequest, Generator, Progress};
use mlx_gen_flux::text_encoder::{T5Geometry, T5TextEncoder};
use mlx_gen_sdxl::config::VaeConfig;
use mlx_gen_sdxl::vae::Autoencoder;
use mlx_gen_supra2::{Supra2Image, SupraDit};
use mlx_rs::Array;

/// Selects the MLX CPU device for one test and restores the GPU default afterwards. Run this
/// binary with `--test-threads 1` when no GPU may be touched: the default device is process-wide.
struct CpuDevice;
impl CpuDevice {
    fn select() -> Self {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        CpuDevice
    }
}
impl Drop for CpuDevice {
    fn drop(&mut self) {
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    }
}

/// An 8-token context, so the long prompt below (10 words + `</s>`) overflows it.
fn tiny() -> Supra2Config {
    Supra2Config {
        latent_channels: 4,
        latent_size: 16,
        patch: 2,
        d_model: 32,
        depth: 2,
        num_heads: 2,
        mlp_hidden: 128,
        ctx_dim: 32,
        ctx_len: 8,
        freq_dim: 256,
    }
}

const SHORT: &str = "a b c";
const LONG: &str = "a b c d e f g h i j";

/// Flan-T5's special tokens and `</s>` template over a whitespace word-level vocabulary.
const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "WhitespaceSplit"},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
    "special_tokens": {"</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}}
  },
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"<pad>": 0, "</s>": 1, "<unk>": 2, "a": 3, "b": 4, "c": 5, "d": 6, "e": 7, "f": 8, "g": 9, "h": 10, "i": 11, "j": 12},
    "unk_token": "<unk>"
  }
}"#;

fn tokenizer() -> Supra2Tokenizer {
    let dir = tempfile::Builder::new()
        .prefix(&format!("supra2-tiny-tokenizer-{}-", std::process::id()))
        .tempdir()
        .unwrap();
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, TOKENIZER_JSON).unwrap();
    Supra2Tokenizer::from_file(&path).unwrap()
}

fn values(shape: &[i32], phase: f32, scale: f32) -> Array {
    let count = shape.iter().product::<i32>();
    let v = (0..count)
        .map(|i| ((i as f32 + phase) * 0.071).sin() * scale)
        .collect::<Vec<_>>();
    Array::from_slice(&v, shape)
}

fn text_encoder() -> T5TextEncoder {
    let all = Weights::from_file(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_t5_parity.safetensors"),
    )
    .unwrap();
    let w: HashMap<String, Array> = all
        .keys()
        .filter_map(|k| {
            k.strip_prefix("w.")
                .map(|name| (name.to_string(), all.require(k).unwrap().clone()))
        })
        .collect();
    T5TextEncoder::from_weights_with_geometry(
        &Weights::from_map(w),
        "",
        32,
        T5Geometry {
            blocks: 2,
            heads: 4,
        },
    )
    .unwrap()
}

fn denoiser(cfg: &Supra2Config) -> SupraDit {
    let w: HashMap<String, Array> = cfg
        .expected_tensors()
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            (name, values(&shape, i as f32 * 3.0, 0.05))
        })
        .collect();
    SupraDit::from_weights(&Weights::from_map(w), cfg).unwrap()
}

const CHANNELS: i32 = 32;

fn insert_conv(t: &mut HashMap<String, Array>, prefix: &str, input: i32, output: i32, k: i32) {
    let phase = t.len() as f32;
    t.insert(
        format!("{prefix}.weight"),
        values(&[output, input, k, k], phase, 0.025),
    );
    t.insert(
        format!("{prefix}.bias"),
        values(&[output], phase + 3.0, 0.01),
    );
}

fn insert_norm(t: &mut HashMap<String, Array>, prefix: &str) {
    let phase = t.len() as f32;
    t.insert(
        format!("{prefix}.weight"),
        mlx_rs::ops::add(values(&[CHANNELS], phase, 0.15), Array::from_f32(1.0)).unwrap(),
    );
    t.insert(
        format!("{prefix}.bias"),
        values(&[CHANNELS], phase + 2.0, 0.05),
    );
}

fn insert_resnet(t: &mut HashMap<String, Array>, prefix: &str) {
    insert_norm(t, &format!("{prefix}.norm1"));
    insert_norm(t, &format!("{prefix}.norm2"));
    insert_conv(t, &format!("{prefix}.conv1"), CHANNELS, CHANNELS, 3);
    insert_conv(t, &format!("{prefix}.conv2"), CHANNELS, CHANNELS, 3);
}

fn insert_mid(t: &mut HashMap<String, Array>, prefix: &str) {
    insert_resnet(t, &format!("{prefix}.resnets.0"));
    insert_norm(t, &format!("{prefix}.attentions.0.group_norm"));
    for name in ["to_q", "to_k", "to_v", "to_out.0"] {
        let phase = t.len() as f32;
        t.insert(
            format!("{prefix}.attentions.0.{name}.weight"),
            values(&[CHANNELS, CHANNELS], phase, 0.02),
        );
        t.insert(
            format!("{prefix}.attentions.0.{name}.bias"),
            values(&[CHANNELS], phase + 5.0, 0.005),
        );
    }
    insert_resnet(t, &format!("{prefix}.resnets.1"));
}

/// A two-stage AutoencoderKL (one 2× upsampler), the SDXL key layout the provider's loader reads.
fn vae() -> Autoencoder {
    let cfg = VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels_out: 8,
        latent_channels_in: 4,
        block_out_channels: vec![CHANNELS, CHANNELS],
        layers_per_block: 1,
        norm_num_groups: 32,
        scaling_factor: VAE_SCALE,
    };
    let mut t = HashMap::new();
    insert_conv(&mut t, "decoder.conv_in", 4, CHANNELS, 3);
    insert_mid(&mut t, "decoder.mid_block");
    for block in 0..2 {
        for resnet in 0..2 {
            insert_resnet(
                &mut t,
                &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
            );
        }
    }
    insert_conv(
        &mut t,
        "decoder.up_blocks.0.upsamplers.0.conv",
        CHANNELS,
        CHANNELS,
        3,
    );
    insert_norm(&mut t, "decoder.conv_norm_out");
    insert_conv(&mut t, "decoder.conv_out", CHANNELS, 3, 3);
    insert_conv(&mut t, "encoder.conv_in", 3, CHANNELS, 3);
    for block in 0..2 {
        insert_resnet(&mut t, &format!("encoder.down_blocks.{block}.resnets.0"));
    }
    insert_conv(
        &mut t,
        "encoder.down_blocks.0.downsamplers.0.conv",
        CHANNELS,
        CHANNELS,
        3,
    );
    insert_mid(&mut t, "encoder.mid_block");
    insert_norm(&mut t, "encoder.conv_norm_out");
    insert_conv(&mut t, "encoder.conv_out", CHANNELS, 8, 3);
    insert_conv(&mut t, "quant_conv", 8, 8, 1);
    insert_conv(&mut t, "post_quant_conv", 4, 4, 1);
    Autoencoder::from_weights(&Weights::from_map(t), &cfg).unwrap()
}

/// The provider behind the `Generator` trait object the registry hands out. No stored
/// unconditional embedding, so CFG exercises the empty-string fallback encode too.
fn generator() -> Box<dyn Generator> {
    let cfg = tiny();
    Box::new(Supra2Image::from_parts(
        cfg,
        tokenizer(),
        text_encoder(),
        denoiser(&cfg),
        vae(),
        None,
    ))
}

fn request(prompt: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        width: 256,
        height: 256,
        count: 1,
        steps: Some(2),
        guidance: Some(3.0),
        seed: Some(7),
        ..Default::default()
    }
}

fn images(output: Option<GenerationOutput>) -> Vec<mlx_gen::Image> {
    match output {
        Some(GenerationOutput::Images(images)) => images,
        _ => panic!("expected images"),
    }
}

#[test]
fn report_warns_exactly_when_the_prompt_is_truncated() {
    let _cpu = CpuDevice::select();
    let g = generator();

    let short = g
        .generate_with_report(&request(SHORT), &mut |_| {})
        .unwrap();
    assert_eq!(short.warnings, []);
    assert_eq!(images(short.output).len(), 1);

    let long = g.generate_with_report(&request(LONG), &mut |_| {}).unwrap();
    assert_eq!(long.warnings.len(), 1, "{:?}", long.warnings);
    assert_eq!(long.warnings[0].code, PROMPT_TRUNCATED_WARNING);
    assert!(
        long.warnings[0]
            .message
            .contains("prompt_tokens=11 ctx_len=8"),
        "{}",
        long.warnings[0].message
    );
    // A truncated prompt still renders, and `generate` is the same render minus the report.
    let reported = images(long.output);
    let plain = images(Some(g.generate(&request(LONG), &mut |_| {}).unwrap()));
    assert_eq!(reported.len(), 1);
    assert_eq!(plain[0].pixels, reported[0].pixels);
}

#[test]
fn generate_honours_cancellation_before_and_during_the_denoise() {
    let _cpu = CpuDevice::select();
    let g = generator();

    let pre = request(SHORT);
    pre.cancel.cancel();
    assert!(matches!(
        g.generate(&pre, &mut |_| {}),
        Err(mlx_gen::gen_core::Error::Canceled)
    ));

    // Tripped from the first step's progress callback: the loop stops before the second step.
    let mid = GenerationRequest {
        cancel: CancelFlag::new(),
        ..request(SHORT)
    };
    let flag = mid.cancel.clone();
    let mut steps = 0;
    let result = g.generate(&mid, &mut |p| {
        if matches!(p, Progress::Step { .. }) {
            steps += 1;
            flag.cancel();
        }
    });
    assert!(
        matches!(result, Err(mlx_gen::gen_core::Error::Canceled)),
        "{:?}",
        result.err()
    );
    assert_eq!(steps, 1);
}
