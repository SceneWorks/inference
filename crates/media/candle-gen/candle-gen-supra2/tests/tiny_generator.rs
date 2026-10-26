//! The registered generator type's report surface on a tiny random-weight pipeline (sc-24217), on
//! the CPU: `generate_with_report` reports `prompt_truncated` (with the token counts) exactly when
//! the prompt overflows the context — the MLX twin's code and fields — and `generate` returns the
//! same render without the report.
//!
//! The parts are [`Supra2Image::from_parts`]'s: a word-level tokenizer with Flan-T5's special tokens
//! and `</s>` post-processor, the MLX twin's tiny T5 parity fixture (2 blocks × 4 heads, width 32),
//! a deterministic tiny DiT and a two-stage SD AutoencoderKL decoder.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{VarBuilder, VarMap};
use candle_gen::gen_core::supra2::{Supra2Config, Supra2Tokenizer, PROMPT_TRUNCATED_WARNING};
use candle_gen::gen_core::{GenerationOutput, GenerationRequest, Generator, Image};
use candle_gen_sdxl::SdxlVaeDecoder;
use candle_gen_supra2::{Supra2Image, SupraDit};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};

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

/// The MLX twin's tiny T5 fixture (its `T5Config`, as `tiny_t5_parity.rs` reads it).
fn text_encoder(device: &Device) -> T5EncoderModel {
    let cfg: T5Config = serde_json::from_value(serde_json::json!({
        "vocab_size": 64, "d_model": 32, "d_kv": 8, "d_ff": 48, "num_layers": 2,
        "num_heads": 4, "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128, "dropout_rate": 0.0,
        "layer_norm_epsilon": 1e-6, "initializer_factor": 1.0,
        "feed_forward_proj": "gated-gelu", "is_encoder_decoder": true,
        "pad_token_id": 0, "eos_token_id": 1
    }))
    .unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/mlx-gen-supra2/tests/fixtures/tiny_t5_parity.safetensors");
    let all = candle_gen::candle_core::safetensors::load(path, device).unwrap();
    let weights: HashMap<String, Tensor> = all
        .iter()
        .filter_map(|(k, v)| k.strip_prefix("w.").map(|n| (n.to_string(), v.clone())))
        .collect();
    T5EncoderModel::load(VarBuilder::from_tensors(weights, DType::F32, device), &cfg).unwrap()
}

fn denoiser(cfg: &Supra2Config, device: &Device) -> SupraDit {
    let w: HashMap<String, Tensor> = cfg
        .expected_tensors()
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let count: usize = shape.iter().product();
            let v: Vec<f32> = (0..count)
                .map(|j| ((j as f32 + i as f32 * 3.0) * 0.071).sin() * 0.05)
                .collect();
            (name, Tensor::from_vec(v, shape, device).unwrap())
        })
        .collect();
    SupraDit::from_tensors(&w, cfg).unwrap()
}

/// A two-stage decoder (one 2× upsampler) on the keys and shapes upstream's constructor declares.
fn vae(device: &Device) -> SdxlVaeDecoder {
    let vars = VarMap::new();
    SdxlVaeDecoder::new(
        VarBuilder::from_varmap(&vars, DType::F32, device),
        3,
        &AutoEncoderKLConfig {
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 4,
            use_quant_conv: true,
            use_post_quant_conv: true,
        },
    )
    .unwrap()
}

/// The provider behind the `Generator` trait object the registry hands out. No stored
/// unconditional embedding, so CFG exercises the empty-string fallback encode too.
fn generator() -> Box<dyn Generator> {
    let cfg = tiny();
    let device = Device::Cpu;
    Box::new(Supra2Image::from_parts(
        cfg,
        device.clone(),
        tokenizer(),
        text_encoder(&device),
        denoiser(&cfg, &device),
        vae(&device),
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

fn images(output: Option<GenerationOutput>) -> Vec<Image> {
    match output {
        Some(GenerationOutput::Images(images)) => images,
        _ => panic!("expected images"),
    }
}

#[test]
fn report_warns_exactly_when_the_prompt_is_truncated() {
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
