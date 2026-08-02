//! sc-8777 (Group-B): maintainer's on-device proof that a **pre-quantized packed** Chroma tier built
//! by [`mlx_gen_chroma::convert::prequantize_turnkey`] loads directly via the packed-detect loader
//! ([`mlx_gen_chroma::quant`] wired into [`mlx_gen_chroma::transformer::Lin::load`]) and renders a
//! coherent T2I image — no dense transient, no in-app `.quantize` (epic 8506). This render is the
//! completeness gate for the loader packed-detect refactor: a missed quantized site loads u32 codes
//! as dense floats → a degenerate (flat) render, which the pixel-std assertion catches.
//!
//! Chroma is a FLUX.1-schnell-derived DiT with a shared T5-XXL text encoder and FLUX.1 VAE. The
//! converter packs the **DiT `transformer/` block Linears**, every group-quantizable T5-XXL 2-D
//! weight, and the FLUX.1 VAE mid-block attention. Shipping q4 preserves the existing q4 transformer
//! and uses q8 T5 primaries plus q4-packed T5 residuals because hosted image calibration rejects
//! single-term q4 and q8 quality; both routes load without a full dense auxiliary transient. A
//! packed tier is loaded with `Quant::None` (the
//! loader packed-detects via `{base}.scales`, so no in-app re-quantize is needed). The `bf16` (dense)
//! tier is the mirrored source, loaded directly.
//!
//! `#[ignore]`d — needs a real ~18GB Chroma diffusers snapshot. Run per tier:
//!   SC8777_SRC=<snap> SC8777_BITS=4 SC8777_MODEL=chroma1_base \
//!     cargo test -p mlx-gen-chroma --release --test prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs: SC8777_SRC (source snapshot dir; default the cached Chroma1-Base snapshot),
//! SC8777_OUT (tier output dir), SC8777_BITS (4 default / 8 / 0 = dense bf16 mirror), SC8777_MODEL
//! (registry id: `chroma1_base` default / `chroma1_hd` / `chroma1_flash`), SC8777_KEEP (retain the
//! tier), SC16462_AUX_BITS (optional T5/VAE bit width for mixed-precision shipping evidence), and
//! SC16462_T5_GROUP_SIZE (optional T5 affine group size; 64 by default).

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_flux::T5Sublayer;
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};
use std::io::Read;
use std::path::PathBuf;

/// Resolve the cached HF snapshot dir for a `lodestones/<repo>` model, or `None` if absent.
fn cached_snapshot(repo: &str) -> Option<PathBuf> {
    let cache = std::path::PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)"));
    let snaps = cache.join(format!("models--lodestones--{repo}/snapshots"));
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
}

fn model_id() -> String {
    std::env::var("SC8777_MODEL").unwrap_or_else(|_| "chroma1_base".into())
}

/// The source HF repo for a registry id (the three variants ship distinct checkpoints).
fn repo_for(id: &str) -> &'static str {
    match id {
        "chroma1_hd" => "Chroma1-HD",
        "chroma1_flash" => "Chroma1-Flash",
        _ => "Chroma1-Base",
    }
}

/// Resolve the source snapshot: `SC8777_SRC`, else the cached snapshot for the selected model.
fn chroma_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC8777_SRC") {
        return Some(PathBuf::from(p));
    }
    cached_snapshot(repo_for(&model_id()))
}

fn bits_env() -> i32 {
    std::env::var("SC8777_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

fn auxiliary_bits_env(route_bits: i32) -> i32 {
    std::env::var("SC16462_AUX_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(route_bits)
}

fn t5_group_size_env() -> i32 {
    std::env::var("SC16462_T5_GROUP_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(mlx_gen::quant::DEFAULT_GROUP_SIZE)
}

fn packed_tier(src: &std::path::Path, out: &std::path::Path, bits: i32) {
    let complete = out
        .join("transformer/diffusion_pytorch_model.safetensors")
        .is_file()
        && out.join("text_encoder/model.safetensors").is_file()
        && out.join("vae/model.safetensors").is_file();
    if !complete {
        assert!(
            !out.exists(),
            "refusing to reuse incomplete packed destination {}",
            out.display()
        );
        mlx_gen_chroma::convert::prequantize_turnkey_with_t5_group_size(
            src,
            out,
            bits,
            t5_group_size_env(),
        )
        .expect("prequantize_turnkey succeeds");
    }
    for component in ["transformer", "text_encoder", "vae"] {
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(out.join(component).join("config.json"))
                .unwrap_or_else(|error| panic!("{component} config: {error}")),
        )
        .unwrap_or_else(|error| panic!("parse {component} config: {error}"));
        let expected_bits = if component == "transformer" {
            bits
        } else {
            auxiliary_bits_env(bits)
        };
        assert_eq!(
            config["quantization"]["bits"], expected_bits,
            "{component} packed bit-width provenance"
        );
        let expected_group_size = if component == "text_encoder" {
            t5_group_size_env()
        } else {
            mlx_gen::quant::DEFAULT_GROUP_SIZE
        };
        assert_eq!(
            config["quantization"]["group_size"], expected_group_size,
            "{component} packed group-size provenance"
        );
        if component == "text_encoder" {
            assert_eq!(
                config["quantization"]["residual_bits"],
                mlx_gen_chroma::convert::T5_RESIDUAL_BITS,
                "T5 progressive residual bit-width provenance"
            );
        }
        let safetensors = std::fs::read_dir(out.join(component))
            .expect("packed component dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension().and_then(|ext| ext.to_str()) == Some("safetensors")
            })
            .count();
        assert_eq!(
            safetensors, 1,
            "{component} must contain exactly one packed safetensors file"
        );
    }
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / 2_f64.powi(30)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "component output length mismatch");
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    dot / (aa.sqrt() * bb.sqrt()).max(f64::EPSILON)
}

fn exact_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

#[derive(Clone, Copy)]
enum T5ProbePolicy {
    Dense,
    Q8Q4Progressive,
    Q8Linears,
    Q8Except { block: usize, sublayer: T5Sublayer },
}

type T5ProbeOutputs = Vec<(Vec<f32>, Vec<f32>)>;

fn t5_probe_outputs(
    root: &std::path::Path,
    max_length: usize,
    prompts: &[&str],
    policy: T5ProbePolicy,
) -> (T5ProbeOutputs, usize) {
    clear_cache();
    reset_peak_memory();
    let tokenizer =
        mlx_gen_chroma::loader::load_tokenizer_with_max_len(max_length).expect("tokenizer");
    let mut t5 = mlx_gen_chroma::loader::load_t5_encoder(root).expect("T5 weights");
    match policy {
        T5ProbePolicy::Dense => {}
        T5ProbePolicy::Q8Q4Progressive => t5
            .quantize_progressive(
                8,
                mlx_gen_chroma::convert::T5_RESIDUAL_BITS,
                t5_group_size_env(),
            )
            .expect("load-time progressive T5 quantization"),
        T5ProbePolicy::Q8Linears => t5
            .quantize_linears(8)
            .expect("load-time T5 Linear quantization"),
        T5ProbePolicy::Q8Except { block, sublayer } => t5
            .quantize_linears_except(8, block, sublayer)
            .expect("load-time T5 sensitivity quantization"),
    }

    let mut outputs = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let (output, text_mask) =
            mlx_gen_chroma::text::encode_prompt(&tokenizer, &t5, prompt).expect("T5 prompt encode");
        let output = output.as_dtype(Dtype::Float32).expect("T5 output f32");
        let text_mask = text_mask
            .as_dtype(Dtype::Float32)
            .expect("T5 text mask f32");
        eval([&output, &text_mask]).expect("materialize T5 output and mask");

        let all_values = output.as_slice::<f32>().to_vec();
        let token_width = output.shape()[2] as usize;
        let mask = text_mask.as_slice::<f32>();
        let mut active_values = Vec::with_capacity(mask.len() * token_width);
        for (token, &active) in mask.iter().enumerate() {
            if active > 0.5 {
                let start = token * token_width;
                active_values.extend_from_slice(&all_values[start..start + token_width]);
            }
        }
        assert!(
            !active_values.is_empty(),
            "T5 prompt mask must retain tokens"
        );
        outputs.push((all_values, active_values));
    }

    let peak = get_peak_memory();
    drop(t5);
    clear_cache();
    (outputs, peak)
}

fn t5_sublayer_dense_bytes(root: &std::path::Path, block: usize, sublayer: T5Sublayer) -> usize {
    let weights = Weights::from_dir(root.join("text_encoder")).expect("T5 weights for byte count");
    let (prefix, names): (String, &[&str]) = match sublayer {
        T5Sublayer::Attention => (
            format!("encoder.block.{block}.layer.0.SelfAttention"),
            &["q", "k", "v", "o"],
        ),
        T5Sublayer::FeedForward => (
            format!("encoder.block.{block}.layer.1.DenseReluDense"),
            &["wi_0", "wi_1", "wo"],
        ),
    };
    names
        .iter()
        .map(|name| {
            weights
                .require(&format!("{prefix}.{name}.weight"))
                .expect("T5 sensitivity tensor")
                .nbytes()
        })
        .sum()
}

fn t5_output(root: &std::path::Path, quantize_at_load: Option<i32>) -> (Vec<f32>, Vec<f32>, usize) {
    clear_cache();
    reset_peak_memory();
    let tokenizer = mlx_gen_chroma::loader::load_tokenizer_with_max_len(64).expect("tokenizer");
    let mut t5 = mlx_gen_chroma::loader::load_t5_encoder(root).expect("T5 weights");
    if let Some(bits) = quantize_at_load {
        mlx_gen_chroma::loader::quantize_t5_for_dense_source(&mut t5, bits, t5_group_size_env())
            .expect("load-time complete progressive T5 quantization");
    }
    let (output, text_mask) = mlx_gen_chroma::text::encode_prompt(
        &tokenizer,
        &t5,
        "a red fox trotting across a snowy meadow at sunrise, cinematic",
    )
    .expect("T5 prompt encode");
    let output = output.as_dtype(Dtype::Float32).expect("T5 output f32");
    let text_mask = text_mask
        .as_dtype(Dtype::Float32)
        .expect("T5 text mask f32");
    eval([&output, &text_mask]).expect("materialize T5 output and mask");

    let all_values = output.as_slice::<f32>().to_vec();

    // Keep this as a semantic-span diagnostic only. Chroma's reference-compatible 0/1 additive
    // attention mask does not prove the other positions are inert; the full-output exact check and
    // full-pipeline image gate below remain authoritative for runtime parity and quality.
    let token_width = output.shape()[2] as usize;
    let mask = text_mask.as_slice::<f32>();
    let mut active_values = Vec::with_capacity(mask.len() * token_width);
    for (token, &active) in mask.iter().enumerate() {
        if active > 0.5 {
            let start = token * token_width;
            active_values.extend_from_slice(&all_values[start..start + token_width]);
        }
    }
    assert!(
        !active_values.is_empty(),
        "T5 prompt mask must retain tokens"
    );
    let peak = get_peak_memory();
    drop(text_mask);
    drop(output);
    drop(t5);
    clear_cache();
    (all_values, active_values, peak)
}

fn vae_output(
    root: &std::path::Path,
    quantize_at_load: Option<i32>,
) -> (Vec<f32>, Vec<f32>, usize) {
    clear_cache();
    reset_peak_memory();
    let mut vae = mlx_gen_chroma::loader::load_vae(root).expect("VAE weights");
    if let Some(bits) = quantize_at_load {
        vae.quantize(bits).expect("load-time VAE quantization");
    }
    let latent_values = (0..16 * 8 * 8)
        .map(|i| ((i as f32 * 0.013).sin() * 0.5).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    let latents = Array::from_slice(&latent_values, &[1, 16, 1, 8, 8]);
    let decoded = vae.decode(&latents).expect("VAE decode");
    let decoded = decoded.as_dtype(Dtype::Float32).expect("VAE decode f32");
    let image_values = (0..3 * 64 * 64)
        .map(|i| ((i as f32 * 0.007).cos() * 0.5).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    let image = Array::from_slice(&image_values, &[1, 3, 1, 64, 64]);
    let encoded = vae.encode(&image).expect("VAE encode");
    let encoded = encoded.as_dtype(Dtype::Float32).expect("VAE encode f32");
    eval([&decoded, &encoded]).expect("materialize VAE outputs");
    let decoded_values = decoded.as_slice::<f32>().to_vec();
    let encoded_values = encoded.as_slice::<f32>().to_vec();
    let peak = get_peak_memory();
    drop(decoded);
    drop(encoded);
    drop(vae);
    clear_cache();
    (decoded_values, encoded_values, peak)
}

fn render_samples(root: PathBuf, id: &str) -> (Vec<Image>, usize) {
    clear_cache();
    reset_peak_memory();
    let generator = mlx_gen_chroma::provider_registry()
        .unwrap()
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .expect("Chroma tier loads");
    let prompts = [
        "a studio photograph of a red fox on fresh snow, detailed fur",
        "a watercolor lighthouse above a stormy ocean at sunset",
    ];
    let images = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| {
            let request = GenerationRequest {
                prompt: (*prompt).into(),
                width: 256,
                height: 256,
                count: 1,
                seed: Some(16462 + index as u64),
                steps: Some(8),
                ..Default::default()
            };
            match generator
                .generate(&request, &mut |_| {})
                .expect("Chroma quality render")
            {
                GenerationOutput::Images(mut images) => {
                    assert_eq!(images.len(), 1);
                    images.pop().unwrap()
                }
                other => panic!("expected Images, got {other:?}"),
            }
        })
        .collect::<Vec<_>>();
    let peak = get_peak_memory();
    drop(generator);
    clear_cache();
    (images, peak)
}

fn image_metrics(reference: &Image, candidate: &Image) -> (f64, f64) {
    assert_eq!(reference.pixels.len(), candidate.pixels.len());
    let reference_f32 = reference
        .pixels
        .iter()
        .map(|&pixel| pixel as f32)
        .collect::<Vec<_>>();
    let candidate_f32 = candidate
        .pixels
        .iter()
        .map(|&pixel| pixel as f32)
        .collect::<Vec<_>>();
    let mean_absolute_error = reference
        .pixels
        .iter()
        .zip(&candidate.pixels)
        .map(|(&left, &right)| (left as f64 - right as f64).abs())
        .sum::<f64>()
        / reference.pixels.len() as f64;
    (cosine(&reference_f32, &candidate_f32), mean_absolute_error)
}

fn files_are_identical(left: &std::path::Path, right: &std::path::Path) -> std::io::Result<bool> {
    if std::fs::metadata(left)?.len() != std::fs::metadata(right)?.len() {
        return Ok(false);
    }
    let mut left = std::io::BufReader::with_capacity(8 * 1024 * 1024, std::fs::File::open(left)?);
    let mut right = std::io::BufReader::with_capacity(8 * 1024 * 1024, std::fs::File::open(right)?);
    let mut left_chunk = vec![0u8; 8 * 1024 * 1024];
    let mut right_chunk = vec![0u8; 8 * 1024 * 1024];
    loop {
        let left_len = left.read(&mut left_chunk)?;
        let right_len = right.read(&mut right_chunk)?;
        if left_len != right_len || left_chunk[..left_len] != right_chunk[..right_len] {
            return Ok(false);
        }
        if left_len == 0 {
            return Ok(true);
        }
    }
}

/// Build-only harness for producing **hostable** tiers (epic 8506 rollout): pack a tier from a Chroma
/// snapshot into `SC8777_OUT` and keep it — no load/generate. `SC8777_BITS=0` (dense bf16) is a
/// verbatim mirror of the source; copy the snapshot dir directly rather than running the packer. Run:
///   SC8777_SRC=<snap> SC8777_OUT=<staging/q4> SC8777_BITS=4 \
///     cargo test -p mlx-gen-chroma --release --test prequantize_real_weights -- --ignored build_tier_only --nocapture
#[test]
#[ignore = "build-only tier producer for hosting; set SC8777_SRC/OUT/BITS"]
fn build_tier_only() {
    let src =
        PathBuf::from(std::env::var("SC8777_SRC").expect("SC8777_SRC (source snapshot) required"));
    let out =
        PathBuf::from(std::env::var("SC8777_OUT").expect("SC8777_OUT (tier output dir) required"));
    let bits = bits_env();
    if bits == 0 {
        panic!(
            "SC8777_BITS=0 (dense bf16) is a verbatim mirror of the source — copy the snapshot dir \
             directly (deref symlinks) rather than running the packer"
        );
    }
    println!(
        "building Q{bits} tier: {} -> {}",
        src.display(),
        out.display()
    );
    packed_tier(&src, &out, bits);
    let f = out
        .join("transformer")
        .join("diffusion_pytorch_model.safetensors");
    let sz = std::fs::metadata(&f)
        .expect("missing packed transformer safetensors")
        .len();
    println!(
        "  transformer/diffusion_pytorch_model.safetensors = {:.3} GB",
        sz as f64 / 1e9
    );
    for asset in ["model_index.json", "transformer/config.json"] {
        assert!(out.join(asset).is_file(), "missing {asset} in turnkey");
    }
    assert!(
        out.join("vae/model.safetensors").is_file(),
        "missing packed vae/model.safetensors"
    );
    assert!(
        out.join("text_encoder/model.safetensors").is_file(),
        "missing packed text_encoder/model.safetensors"
    );
    println!("✓ built {}", out.display());
}

/// Rank the smallest credible T5 source-precision carve-outs before spending another full render
/// matrix. The probes use Chroma's production sequence length, both strict-gate positive prompts,
/// and the empty negative prompt exercised by Base/HD true CFG.
#[test]
#[ignore = "needs a real Chroma T5-XXL snapshot and Apple Silicon MLX"]
fn t5_precision_sensitivity_sweep() {
    let src = chroma_snapshot().expect("SC8777_SRC or cached Chroma snapshot required");
    let prompts = [
        "a studio photograph of a red fox on fresh snow, detailed fur",
        "a watercolor lighthouse above a stormy ocean at sunset",
        "",
    ];
    let (dense, dense_peak) = t5_probe_outputs(
        &src,
        mlx_gen_chroma::MAX_SEQUENCE_LENGTH,
        &prompts,
        T5ProbePolicy::Dense,
    );
    let candidates = [
        (
            "q8-plus-q4-packed-residual",
            T5ProbePolicy::Q8Q4Progressive,
            None,
        ),
        ("q8-linears", T5ProbePolicy::Q8Linears, None),
        (
            "q8-except-block0-attention",
            T5ProbePolicy::Q8Except {
                block: 0,
                sublayer: T5Sublayer::Attention,
            },
            Some((0usize, T5Sublayer::Attention)),
        ),
        (
            "q8-except-block0-ffn",
            T5ProbePolicy::Q8Except {
                block: 0,
                sublayer: T5Sublayer::FeedForward,
            },
            Some((0usize, T5Sublayer::FeedForward)),
        ),
        (
            "q8-except-block23-attention",
            T5ProbePolicy::Q8Except {
                block: 23,
                sublayer: T5Sublayer::Attention,
            },
            Some((23usize, T5Sublayer::Attention)),
        ),
        (
            "q8-except-block23-ffn",
            T5ProbePolicy::Q8Except {
                block: 23,
                sublayer: T5Sublayer::FeedForward,
            },
            Some((23usize, T5Sublayer::FeedForward)),
        ),
    ];

    for (policy_name, policy, carveout) in candidates {
        let (candidate, peak) =
            t5_probe_outputs(&src, mlx_gen_chroma::MAX_SEQUENCE_LENGTH, &prompts, policy);
        let prompt_metrics = dense
            .iter()
            .zip(&candidate)
            .enumerate()
            .map(
                |(index, ((dense_all, dense_active), (candidate_all, candidate_active)))| {
                    serde_json::json!({
                        "promptIndex": index,
                        "allPositionsCosine": cosine(dense_all, candidate_all),
                        "activeSpanCosine": cosine(dense_active, candidate_active),
                    })
                },
            )
            .collect::<Vec<_>>();
        let worst_all = prompt_metrics
            .iter()
            .filter_map(|row| row["allPositionsCosine"].as_f64())
            .fold(f64::INFINITY, f64::min);
        let worst_active = prompt_metrics
            .iter()
            .filter_map(|row| row["activeSpanCosine"].as_f64())
            .fold(f64::INFINITY, f64::min);
        let dense_sublayer_bytes = carveout
            .map(|(block, sublayer)| t5_sublayer_dense_bytes(&src, block, sublayer))
            .unwrap_or(0);
        let dense_block = carveout.map(|(block, _)| block);
        let dense_sublayer = carveout.map(|(_, sublayer)| match sublayer {
            T5Sublayer::Attention => "attention",
            T5Sublayer::FeedForward => "ffn",
        });
        println!(
            "SC16462_T5_SENSITIVITY {}",
            serde_json::json!({
                "model": model_id(),
                "policy": policy_name,
                "sequenceLength": mlx_gen_chroma::MAX_SEQUENCE_LENGTH,
                "denseBlock": dense_block,
                "denseSublayer": dense_sublayer,
                "denseSublayerBytes": dense_sublayer_bytes,
                "worstAllPositionsCosine": worst_all,
                "worstActiveSpanCosine": worst_active,
                "peakBytes": peak,
                "densePeakBytes": dense_peak,
                "prompts": prompt_metrics,
            })
        );
    }
}

/// SC-16462: real T5-XXL and FLUX.1 VAE weights must take the packed loader path, produce exactly
/// the same tensors as the established dense-load-then-quantize seam, and stay inside the measured
/// direct semantic-span diagnostic versus bf16. Because Chroma uses the reference's literal 0/1
/// additive attention mask, the separate full-pipeline image gate is authoritative for runtime
/// quality. T5 is also the residency discriminator: a packed load must avoid the dense-plus-packed
/// high-water mark that load-time quantization necessarily incurs.
#[test]
#[ignore = "needs real Chroma weights and Apple Silicon MLX"]
fn packed_auxiliaries_match_load_time_quantization() {
    let Some(src) = chroma_snapshot() else {
        panic!("no Chroma snapshot (set SC8777_SRC or populate the explicit MLX_GEN_MODELS_ROOT)");
    };
    let bits = bits_env();
    assert!(matches!(bits, 4 | 8), "SC8777_BITS must be 4 or 8");
    let auxiliary_bits = auxiliary_bits_env(bits);
    assert!(
        matches!(auxiliary_bits, 4 | 8),
        "SC16462_AUX_BITS must be 4 or 8"
    );
    let out = std::env::var("SC8777_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("chroma-tier-q{bits}")));
    packed_tier(&src, &out, bits);

    let packed_weights =
        mlx_gen::weights::Weights::from_dir(out.join("text_encoder")).expect("packed T5 weights");
    assert_eq!(
        packed_weights.require("shared.weight").unwrap().dtype(),
        Dtype::Uint32,
        "T5 token embedding must be stored as packed codes"
    );
    assert!(packed_weights.get("shared.scales").is_some());
    let relative_bias = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias";
    assert!(
        packed_weights
            .get(&format!("{relative_bias}.scales"))
            .is_some(),
        "T5 relative-position bias must be stored as packed codes"
    );
    let t5_probe = "encoder.block.0.layer.0.SelfAttention.q";
    assert_eq!(
        packed_weights
            .require(&format!("{t5_probe}.weight"))
            .unwrap()
            .dtype(),
        Dtype::Uint32,
        "T5 attention/FFN surface must be stored as packed codes"
    );
    assert!(packed_weights.get(&format!("{t5_probe}.scales")).is_some());
    assert!(
        packed_weights
            .get(&format!("{t5_probe}.residual.scales"))
            .is_some(),
        "T5 progressive correction must also be stored as packed codes"
    );
    let packed_weights =
        mlx_gen::weights::Weights::from_dir(out.join("vae")).expect("packed VAE weights");
    let vae_probe = "decoder.mid_block.attentions.0.to_q";
    assert_eq!(
        packed_weights
            .require(&format!("{vae_probe}.weight"))
            .unwrap()
            .dtype(),
        Dtype::Uint32,
        "VAE attention must be stored as packed codes"
    );
    assert!(packed_weights.get(&format!("{vae_probe}.scales")).is_some());

    let (dense_t5, dense_t5_active, dense_t5_peak) = t5_output(&src, None);
    let (load_time_t5, load_time_t5_active, load_time_t5_peak) =
        t5_output(&src, Some(auxiliary_bits));
    let (packed_t5, packed_t5_active, packed_t5_peak) = t5_output(&out, None);
    assert!(
        exact_f32(&packed_t5, &load_time_t5),
        "packed T5 output differs from load-time Q{auxiliary_bits} output"
    );
    assert!(
        exact_f32(&packed_t5_active, &load_time_t5_active),
        "packed T5 active-span output differs from load-time Q{auxiliary_bits} output"
    );
    let t5_all_positions_cosine = cosine(&dense_t5, &packed_t5);
    let t5_active_span_cosine = cosine(&dense_t5_active, &packed_t5_active);
    assert!(
        t5_all_positions_cosine.is_finite() && t5_active_span_cosine.is_finite(),
        "Q{auxiliary_bits} T5 diagnostic cosines must be finite"
    );
    assert!(
        packed_t5_peak < load_time_t5_peak,
        "packed T5 peak {:.2} GiB must stay below load-time quantization peak {:.2} GiB",
        gib(packed_t5_peak),
        gib(load_time_t5_peak)
    );

    let (dense_vae_decode, dense_vae_encode, dense_vae_peak) = vae_output(&src, None);
    let (load_time_vae_decode, load_time_vae_encode, load_time_vae_peak) =
        vae_output(&src, Some(auxiliary_bits));
    let (packed_vae_decode, packed_vae_encode, packed_vae_peak) = vae_output(&out, None);
    assert!(
        exact_f32(&packed_vae_decode, &load_time_vae_decode),
        "packed VAE decode differs from load-time Q{auxiliary_bits} decode"
    );
    assert!(
        exact_f32(&packed_vae_encode, &load_time_vae_encode),
        "packed VAE encode differs from load-time Q{auxiliary_bits} encode"
    );
    let vae_decode_cosine = cosine(&dense_vae_decode, &packed_vae_decode);
    let vae_encode_cosine = cosine(&dense_vae_encode, &packed_vae_encode);
    // q4 group quantization has a wider direct component envelope; the full rendered-image gate
    // below remains the stricter user-visible no-regression contract for both bit widths.
    let vae_floor = if auxiliary_bits == 8 { 0.99999 } else { 0.9995 };
    assert!(
        vae_decode_cosine >= vae_floor && vae_encode_cosine >= vae_floor,
        "Q{auxiliary_bits} VAE cosine fell below {vae_floor:.5} (decode={vae_decode_cosine:.7}, encode={vae_encode_cosine:.7})"
    );

    println!(
        "SC16462_COMPONENT {{\"model\":\"{}\",\"tier\":\"q{}\",\"auxiliaryBits\":{},\"t5Policy\":\"q{}-plus-q{}-residual-complete-group{}\",\"t5AllPositionsCosine\":{:.8},\"t5ActiveSpanDiagnosticCosine\":{:.8},\"vaeDecodeCosine\":{:.8},\"vaeEncodeCosine\":{:.8},\"t5PeakBytes\":{{\"dense\":{},\"loadTimeQuantized\":{},\"packed\":{}}},\"vaePeakBytes\":{{\"dense\":{},\"loadTimeQuantized\":{},\"packed\":{}}}}}",
        model_id(),
        bits,
        auxiliary_bits,
        auxiliary_bits,
        mlx_gen_chroma::convert::T5_RESIDUAL_BITS,
        t5_group_size_env(),
        t5_all_positions_cosine,
        t5_active_span_cosine,
        vae_decode_cosine,
        vae_encode_cosine,
        dense_t5_peak,
        load_time_t5_peak,
        packed_t5_peak,
        dense_vae_peak,
        load_time_vae_peak,
        packed_vae_peak,
    );
}

/// Compare the newly packed-auxiliary tier against the exact currently shipped tier, whose
/// transformer is already packed at the same bit width but whose T5/VAE are dense. This isolates
/// auxiliary quality at full-pipeline level and emits the q4/q8 calibration row consumed by the
/// hosted lane. The tight pixel envelope is the story's measurable no-regression contract.
#[test]
#[ignore = "needs the shipped and candidate Chroma tiers plus Apple Silicon MLX"]
fn packed_auxiliaries_preserve_full_pipeline_quality_and_reduce_peak() {
    let baseline = PathBuf::from(
        std::env::var("SC16462_BASELINE")
            .expect("SC16462_BASELINE must point to the exact shipped q4/q8 tier"),
    );
    assert!(
        baseline.join("transformer").is_dir()
            && baseline.join("text_encoder").is_dir()
            && baseline.join("vae").is_dir(),
        "baseline tier is incomplete: {}",
        baseline.display()
    );
    let candidate = PathBuf::from(
        std::env::var("SC8777_OUT").expect("SC8777_OUT must point to the candidate packed tier"),
    );
    let bits = bits_env();
    let auxiliary_bits = auxiliary_bits_env(bits);
    let id = model_id();
    let transformer = "transformer/diffusion_pytorch_model.safetensors";
    assert!(
        files_are_identical(&baseline.join(transformer), &candidate.join(transformer))
            .expect("compare shipped and candidate transformers"),
        "Q{bits} candidate transformer differs from the shipped transformer; auxiliary quality isolation is invalid"
    );
    let (baseline_images, baseline_peak) = render_samples(baseline, &id);
    let (candidate_images, candidate_peak) = render_samples(candidate, &id);
    let metrics = baseline_images
        .iter()
        .zip(&candidate_images)
        .map(|(reference, candidate)| image_metrics(reference, candidate))
        .collect::<Vec<_>>();
    let minimum_cosine = metrics
        .iter()
        .map(|(cosine, _)| *cosine)
        .fold(f64::INFINITY, f64::min);
    let maximum_mae = metrics.iter().map(|(_, mae)| *mae).fold(0.0f64, f64::max);
    assert!(
        minimum_cosine >= 0.9999,
        "Q{bits} auxiliary packing changed a render: minimum cosine {minimum_cosine:.6} < 0.9999"
    );
    assert!(
        maximum_mae <= 1.0,
        "Q{bits} auxiliary packing changed a render: maximum mean absolute pixel error {maximum_mae:.4} > 1.0"
    );
    assert!(
        candidate_peak < baseline_peak,
        "Q{bits} packed tier peak {:.2} GiB must stay below shipped dense-auxiliary peak {:.2} GiB",
        gib(candidate_peak),
        gib(baseline_peak)
    );
    println!(
        "SC16462_CALIBRATION {{\"model\":\"{}\",\"tier\":\"q{}\",\"auxiliaryBits\":{},\"t5Policy\":\"q{}-plus-q{}-residual-complete-group{}\",\"minimumImageCosine\":{:.8},\"maximumMeanAbsolutePixelError\":{:.8},\"peakBytes\":{{\"shippedDenseAuxiliary\":{},\"packedAuxiliary\":{}}},\"prompts\":{}}}",
        id,
        bits,
        auxiliary_bits,
        auxiliary_bits,
        mlx_gen_chroma::convert::T5_RESIDUAL_BITS,
        t5_group_size_env(),
        minimum_cosine,
        maximum_mae,
        baseline_peak,
        candidate_peak,
        metrics.len(),
    );
}

#[test]
#[ignore = "needs a ~18GB Chroma snapshot; builds a packed tier + renders (set SC8777_SRC/BITS/MODEL)"]
fn prequantize_turnkey_loads_packed_and_renders() {
    let Some(src) = chroma_snapshot() else {
        eprintln!("skip: no Chroma snapshot (set SC8777_SRC or populate the HF cache)");
        return;
    };
    let bits = bits_env();
    let out = std::env::var("SC8777_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("chroma-tier-q{bits}")));
    let id = model_id();

    // Build the packed tier (Q4/Q8). For the dense bf16 tier the source snapshot IS the tier, so we
    // load `src` directly.
    let load_root: PathBuf = if bits == 0 {
        println!(
            "dense (bf16) tier: loading source snapshot directly {}",
            src.display()
        );
        src.clone()
    } else {
        println!(
            "building Q{bits} turnkey: {} -> {}",
            src.display(),
            out.display()
        );
        packed_tier(&src, &out, bits);
        assert!(
            out.join("transformer")
                .join("diffusion_pytorch_model.safetensors")
                .is_file(),
            "missing packed transformer safetensors"
        );
        assert!(
            out.join("text_encoder/model.safetensors").is_file(),
            "missing packed T5 safetensors"
        );
        assert!(
            out.join("vae/model.safetensors").is_file(),
            "missing packed VAE safetensors"
        );
        out.clone()
    };

    // Load DIRECTLY from the tier dir. A packed tier packed-detects via `{base}.scales` (no dense
    // transient, no in-app re-quantize), so we load with `Quant::None`; the dense bf16 tier loads
    // dense the same way.
    let spec = LoadSpec::new(WeightsSource::Dir(load_root));
    let generator = mlx_gen_chroma::provider_registry()
        .unwrap()
        .load(&id, &spec)
        .expect("packed chroma loads");

    // 256² / few-step — packed load-path proof, not a quality bench.
    let req = GenerationRequest {
        prompt: "a photograph of an astronaut riding a horse".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        ..Default::default()
    };
    let img = match generator
        .generate(&req, &mut |_| {})
        .expect("packed generate succeeds")
    {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1);
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (256, 256), "image size");

    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as u64).sum::<u64>() as f64 / img.pixels.len() as f64;
    let var = img
        .pixels
        .iter()
        .map(|&p| {
            let d = p as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / img.pixels.len() as f64;
    let std = var.sqrt();
    let tier = if bits == 0 {
        "bf16".into()
    } else {
        format!("Q{bits}")
    };
    println!("✓ packed {tier} {id}: 256x256; px min={min} max={max} mean={mean:.1} std={std:.1}");
    assert!(
        std > 20.0,
        "degenerate render: pixel std {std:.1} too flat (a missed packed site loads codes as dense \
         floats)"
    );
    assert!(
        max as i32 - min as i32 > 32,
        "degenerate render: pixel range {min}..={max} too flat"
    );

    if bits != 0 && std::env::var("SC8777_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC8777_KEEP to retain)", out.display());
    }
}
