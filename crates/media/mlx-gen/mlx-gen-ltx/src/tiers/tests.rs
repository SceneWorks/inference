//! Tier-converter unit tests over a **miniature but structurally faithful** LTX-2.5 bundle.
//!
//! The real bundle is 73 GB; these run in a temp dir in under a second. What they keep from the
//! real thing is every structure the converter can get wrong: the `model.diffusion_model.` prefix
//! and both embeddings connectors riding the transformer file, the Gemma 4 checkpoint's per-layer
//! `layer_scalar` buffers and its `full_attention` layers with no `v_proj`, the `U8` packed HF
//! assets, the video VAE's two halves sharing one file, and the per-component `__metadata__` the
//! loader reads its config from.
//!
//! Everything measured here is measured **by re-reading the produced files' safetensors headers** —
//! never from the converter's own report alone, and never by hashing (`save_file` orders
//! `__metadata__` nondeterministically, so a hash-based check would be flaky *and* wrong).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::ltx_checkpoint::{LtxBundleBuilder, LtxComponent};

use super::*;

// =================================================================================================
// Fixture: a miniature LTX-2.5 split bundle
// =================================================================================================

/// Transformer blocks in the fixture (the real checkpoint has 48).
const BLOCKS: usize = 2;
/// Connector blocks per tower (the real checkpoint has 8).
const CONNECTOR_BLOCKS: usize = 2;
/// Gemma decoder layers in the fixture (the real checkpoint has 48).
const GEMMA_LAYERS: usize = 4;
/// Which fixture Gemma layers are `full_attention` — those ship **no** `v_proj`, exactly as the
/// real checkpoint's every-6th-layer schedule does.
const GEMMA_FULL_LAYERS: [usize; 1] = [3];
/// A width that divides the group size, so every fixture Linear is quantizable.
const DIM: i32 = 128;

fn ones(shape: &[i32]) -> Array {
    Array::ones::<f32>(shape)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// A `U8` payload tensor, the shape the packed HF assets take.
fn bytes_tensor(payload: &[u8]) -> Array {
    Array::from_slice(payload, &[payload.len() as i32])
}

fn write_file(path: &Path, tensors: Vec<(String, Array)>, metadata: &[(&str, &str)]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let meta: HashMap<String, String> = metadata
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    eval(tensors.iter().map(|(_, v)| v)).unwrap();
    Array::save_safetensors(
        tensors.iter().map(|(k, v)| (k.as_str(), v)),
        Some(&meta),
        path,
    )
    .unwrap();
}

/// The transformer config block, carrying the dims the connector and the caption path read.
fn transformer_config() -> String {
    serde_json::json!({
        "_class_name": "AVTransformer3DModel",
        "num_layers": BLOCKS,
        "num_attention_heads": 2,
        "attention_head_dim": DIM / 2,
        "cross_attention_dim": DIM,
        "in_channels": 128,
        "out_channels": 128,
        "norm_eps": 1e-6,
        "ff_bias": false,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": DIM / 2,
        "audio_cross_attention_dim": DIM,
        "use_embeddings_connector": true,
        "connector_num_layers": CONNECTOR_BLOCKS,
        "connector_num_attention_heads": 2,
        "connector_attention_head_dim": DIM / 2,
        "connector_positional_embedding_max_pos": [4096],
        "connector_num_learnable_registers": 4,
        "caption_projection_first_linear": false,
        "caption_projection_second_linear": false,
        "caption_proj_input_norm": false,
        "caption_proj_before_connector": true,
        "text_encoder_norm_type": "PER_TOKEN_RMS",
        "rope_type": "split",
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
    })
    .to_string()
}

fn write_transformer(root: &Path) -> PathBuf {
    write_transformer_with(root, &[])
}

/// [`write_transformer`] plus `extra` tensors, for the tests that need a transformer carrying a
/// weight the converter's quant list and expected-dense allowlist have never heard of. Keys are
/// written under the checkpoint's `model.diffusion_model.` prefix, exactly as the real file's are.
fn write_transformer_with(root: &Path, extra: &[(&str, Array)]) -> PathBuf {
    let mut t: Vec<(String, Array)> = Vec::new();
    let p = "model.diffusion_model.";
    for (key, value) in extra {
        t.push((format!("{p}{key}"), value.clone()));
    }
    for b in 0..BLOCKS {
        for attn in [
            "attn1",
            "attn2",
            "audio_attn1",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
        ] {
            for proj in ["to_q", "to_k", "to_v"] {
                t.push((
                    format!("{p}transformer_blocks.{b}.{attn}.{proj}.weight"),
                    ones(&[DIM, DIM]),
                ));
                t.push((
                    format!("{p}transformer_blocks.{b}.{attn}.{proj}.bias"),
                    ones(&[DIM]),
                ));
            }
            t.push((
                format!("{p}transformer_blocks.{b}.{attn}.to_out.0.weight"),
                ones(&[DIM, DIM]),
            ));
            t.push((
                format!("{p}transformer_blocks.{b}.{attn}.to_out.0.bias"),
                ones(&[DIM]),
            ));
            // Dense by the reference predicate: norms and the sigmoid gate projection.
            t.push((
                format!("{p}transformer_blocks.{b}.{attn}.q_norm.weight"),
                ones(&[DIM / 2]),
            ));
            t.push((
                format!("{p}transformer_blocks.{b}.{attn}.to_gate_logits.weight"),
                ones(&[2, DIM]),
            ));
        }
        // The 2.5 video FFN carries no bias (`ff_bias: false`); the audio FFN does.
        t.push((
            format!("{p}transformer_blocks.{b}.ff.net.0.proj.weight"),
            ones(&[DIM, DIM]),
        ));
        t.push((
            format!("{p}transformer_blocks.{b}.ff.net.2.weight"),
            ones(&[DIM, DIM]),
        ));
        t.push((
            format!("{p}transformer_blocks.{b}.audio_ff.net.0.proj.weight"),
            ones(&[DIM, DIM]),
        ));
        t.push((
            format!("{p}transformer_blocks.{b}.audio_ff.net.0.proj.bias"),
            ones(&[DIM]),
        ));
        t.push((
            format!("{p}transformer_blocks.{b}.audio_ff.net.2.weight"),
            ones(&[DIM, DIM]),
        ));
        t.push((
            format!("{p}transformer_blocks.{b}.audio_ff.net.2.bias"),
            ones(&[DIM]),
        ));
        // f32 in the real checkpoint — the bf16 cast is load-bearing here.
        t.push((
            format!("{p}transformer_blocks.{b}.scale_shift_table"),
            Array::ones::<f32>(&[9, DIM]).unwrap(),
        ));
    }
    t.push((
        format!("{p}adaln_single.emb.timestep_embedder.linear_1.weight"),
        ones(&[DIM, DIM]),
    ));
    t.push((
        format!("{p}adaln_single.emb.timestep_embedder.linear_1.bias"),
        ones(&[DIM]),
    ));
    t.push((format!("{p}patchify_proj.weight"), ones(&[DIM, DIM])));
    t.push((format!("{p}patchify_proj.bias"), ones(&[DIM])));
    t.push((format!("{p}proj_out.weight"), ones(&[DIM, DIM])));
    t.push((format!("{p}proj_out.bias"), ones(&[DIM])));
    t.push((format!("{p}keyframes_abs_pos_embedding"), ones(&[1, DIM])));

    for tower in ["video_embeddings_connector", "audio_embeddings_connector"] {
        for b in 0..CONNECTOR_BLOCKS {
            let base = format!("{p}{tower}.transformer_1d_blocks.{b}");
            for proj in ["to_q", "to_k", "to_v"] {
                t.push((format!("{base}.attn1.{proj}.weight"), ones(&[DIM, DIM])));
                t.push((format!("{base}.attn1.{proj}.bias"), ones(&[DIM])));
            }
            t.push((format!("{base}.attn1.to_out.0.weight"), ones(&[DIM, DIM])));
            t.push((format!("{base}.attn1.to_out.0.bias"), ones(&[DIM])));
            t.push((format!("{base}.attn1.q_norm.weight"), ones(&[DIM / 2])));
            t.push((format!("{base}.attn1.k_norm.weight"), ones(&[DIM / 2])));
            t.push((
                format!("{base}.attn1.to_gate_logits.weight"),
                ones(&[2, DIM]),
            ));
            t.push((format!("{base}.attn1.to_gate_logits.bias"), ones(&[2])));
            t.push((format!("{base}.ff.net.0.proj.weight"), ones(&[DIM, DIM])));
            t.push((format!("{base}.ff.net.2.weight"), ones(&[DIM, DIM])));
        }
        t.push((
            format!("{p}{tower}.{tower}.learnable_registers"),
            ones(&[4, DIM]),
        ));
    }

    let path = root.join("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors");
    write_file(
        &path,
        t,
        &[
            ("model_version", LTX_2_5_MODEL_VERSION),
            (
                "gemma_source_checkpoint",
                r#"{"ltx_version":"2.5.0","gemma_version":"gemma4-12b-ltx-v1"}"#,
            ),
            (
                "config",
                &serde_json::json!({
                    "transformer": serde_json::from_str::<serde_json::Value>(&transformer_config()).unwrap(),
                    "scheduler": {"_class_name": "RectifiedFlowScheduler", "sampler": "LinearQuadratic"},
                    "vae": null,
                    "audio_vae": null,
                    "vocoder": null,
                })
                .to_string(),
            ),
            ("license", "LTX-2.x Community License Agreement (fixture)"),
        ],
    );
    path
}

fn write_text_encoder(root: &Path) -> PathBuf {
    let mut t: Vec<(String, Array)> = Vec::new();
    t.push(("model.embed_tokens.weight".into(), ones(&[256, DIM])));
    t.push(("model.norm.weight".into(), ones(&[DIM])));
    for l in 0..GEMMA_LAYERS {
        for proj in ["q_proj", "k_proj", "o_proj"] {
            t.push((
                format!("model.layers.{l}.self_attn.{proj}.weight"),
                ones(&[DIM, DIM]),
            ));
        }
        // `attention_k_eq_v` on the full-attention layers: no `v_proj` exists at all.
        if !GEMMA_FULL_LAYERS.contains(&l) {
            t.push((
                format!("model.layers.{l}.self_attn.v_proj.weight"),
                ones(&[DIM, DIM]),
            ));
        }
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            t.push((
                format!("model.layers.{l}.mlp.{proj}.weight"),
                ones(&[DIM, DIM]),
            ));
        }
        t.push((
            format!("model.layers.{l}.input_layernorm.weight"),
            ones(&[DIM]),
        ));
        // The Gemma 4 per-layer trained scalar — a `[1]` buffer that must survive verbatim.
        t.push((format!("model.layers.{l}.layer_scalar"), ones(&[1])));
    }
    // The LTX feature-extractor heads, which live in the 2.5 text-encoder file.
    for head in ["video_aggregate_embed", "audio_aggregate_embed"] {
        t.push((
            format!("text_embedding_projection.{head}.weight"),
            ones(&[DIM, DIM]),
        ));
        t.push((
            format!("text_embedding_projection.{head}.bias"),
            ones(&[DIM]),
        ));
    }
    // Towers LTX never runs, carried so the pack stays self-contained.
    t.push(("vision_model.patch_dense.weight".into(), ones(&[DIM, DIM])));
    t.push((
        "multi_modal_projector.embedding_projection.weight".into(),
        ones(&[DIM, DIM]),
    ));
    // The packed HF assets — `U8` payloads that must pass through byte-identical.
    t.push((
        "tokenizer_json".into(),
        bytes_tensor(br#"{"version":"1.0","model":{"type":"WordLevel","vocab":{}}}"#),
    ));
    t.push((
        "hf_asset__tokenizer_config.json".into(),
        bytes_tensor(br#"{"bos_token":"<bos>","pad_token":"<pad>"}"#),
    ));
    t.push((
        "hf_asset__processor_config.json".into(),
        bytes_tensor(b"{}"),
    ));

    let path = root.join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors");
    write_file(
        &path,
        t,
        &[
            ("format", "pt"),
            (
                "gemma_config",
                &serde_json::json!({
                    "model_type": "gemma4_unified",
                    "gemma_version": "gemma4-12b-ltx-v1",
                    "text_config": {"hidden_size": DIM, "num_hidden_layers": GEMMA_LAYERS},
                })
                .to_string(),
            ),
        ],
    );
    path
}

fn write_conv_vae(root: &Path) -> PathBuf {
    let mut t: Vec<(String, Array)> = Vec::new();
    for half in ["encoder", "decoder"] {
        t.push((
            format!("{half}.conv_in.conv.weight"),
            ones(&[8, 4, 3, 3, 3]),
        ));
        t.push((format!("{half}.conv_in.conv.bias"), ones(&[8])));
        t.push((format!("{half}.norm_out.weight"), ones(&[8])));
    }
    t.push(("per_channel_statistics.mean-of-means".into(), ones(&[4])));
    t.push(("per_channel_statistics.std-of-means".into(), ones(&[4])));
    let path = root.join("vae/ltx-2.5-video-vae-conv-bf16.safetensors");
    write_file(
        &path,
        t,
        &[
            ("model_version", LTX_2_5_MODEL_VERSION),
            (
                "config",
                &serde_json::json!({
                    "vae": {
                        "_class_name": "CausalVideoAutoencoder",
                        "dims": 3, "in_channels": 3, "out_channels": 3,
                        "latent_channels": 128, "patch_size": 4,
                        "latent_log_var": "uniform",
                    }
                })
                .to_string(),
            ),
        ],
    );
    path
}

fn write_audio_vae(root: &Path) -> PathBuf {
    let t: Vec<(String, Array)> = vec![
        (
            "audio_vae.decoder.conv_in.conv.weight".into(),
            ones(&[8, 4, 3, 3]),
        ),
        ("audio_vae.decoder.conv_in.conv.bias".into(), ones(&[8])),
        (
            "audio_vae.per_channel_statistics.mean-of-means".into(),
            ones(&[4]),
        ),
        (
            "audio_vae.per_channel_statistics.std-of-means".into(),
            ones(&[4]),
        ),
        ("vocoder.vocoder.conv_pre.weight".into(), ones(&[8, 4, 3])),
        ("vocoder.vocoder.conv_pre.bias".into(), ones(&[8])),
        (
            "vocoder.bwe_generator.ups.0.weight".into(),
            ones(&[4, 8, 3]),
        ),
    ];
    let path = root.join("vae/ltx-2.5-audio-vae-bf16.safetensors");
    write_file(
        &path,
        t,
        &[
            ("model_version", LTX_2_5_MODEL_VERSION),
            (
                "config",
                &serde_json::json!({
                    "audio_vae": {"model": {"params": {"ddconfig": {"ch": 128, "z_channels": 8}}}},
                    "vocoder": {"vocoder": {"resblock": "AMP1"}},
                })
                .to_string(),
            ),
        ],
    );
    path
}

fn write_upsampler(root: &Path, temporal: bool) -> PathBuf {
    let t: Vec<(String, Array)> = vec![
        ("initial_conv.weight".into(), ones(&[8, 4, 3, 3, 3])),
        ("initial_conv.bias".into(), ones(&[8])),
        ("initial_norm.weight".into(), ones(&[8])),
    ];
    let name = if temporal { "temporal" } else { "spatial" };
    let path = root.join(format!(
        "latent_upscale_models/ltx-2.5-latent-{name}-upscaler-x2-bf16-1.0.safetensors"
    ));
    write_file(
        &path,
        t,
        &[(
            "config",
            &serde_json::json!({
                "_class_name": "LatentUpsampler",
                "in_channels": 128, "mid_channels": 1024,
                "spatial_upsample": !temporal, "temporal_upsample": temporal,
            })
            .to_string(),
        )],
    );
    path
}

/// The DiffVAE: a conv encoder half plus the `NADiffusionDecoder` half whose 99.96 %-of-parameters
/// Linear population is what makes it the one non-conv VAE component a tier quantizes (sc-18775,
/// on sc-18766's port).
///
/// The shapes are the real component's in miniature, including the one that matters most:
/// `conv_in_x_t`'s input axis is 48, which the group width 64 does not divide, so it is the Linear
/// the quantized set must leave out.
fn write_diff_vae(root: &Path) -> PathBuf {
    write_diff_vae_with(root, true)
}

/// [`write_diff_vae`], optionally **without** the `NADiffusionDecoder` half — a file that still
/// declares `CausalDiffusionVAE` and still yields a non-empty "decoder" map (the two
/// `per_channel_statistics` tensors survive `sanitize_vae_decoder_component`), which is exactly the
/// shape that used to be silently skipped.
fn write_diff_vae_with(root: &Path, with_decoder: bool) -> PathBuf {
    /// The DiffVAE decoder's latent width — `conv_out`'s output and `conv_in_x_t`'s input, and
    /// deliberately **not** a multiple of [`DEFAULT_GROUP_SIZE`], exactly as upstream's 48 is not.
    const LATENT: i32 = 48;
    let hidden = DIM * 2;
    let mut t: Vec<(String, Array)> = Vec::new();

    // The encoder half — conv kernels only, which is why `diffusion_vae_encoder` keeps its
    // `no-linear-weights` exemption while the decoder loses its `no-mlx-port` one.
    t.push(("encoder.conv_in.conv.weight".into(), ones(&[8, 4, 3, 3, 3])));
    t.push(("encoder.conv_in.conv.bias".into(), ones(&[8])));
    t.push(("encoder.norm_out.weight".into(), ones(&[8])));

    fn linear(t: &mut Vec<(String, Array)>, name: &str, out: i32, inp: i32) {
        t.push((format!("decoder.{name}.weight"), ones(&[out, inp])));
        t.push((format!("decoder.{name}.bias"), ones(&[out])));
    }
    if !with_decoder {
        t.push((
            "per_channel_statistics.mean-of-means".into(),
            ones(&[LATENT]),
        ));
        t.push((
            "per_channel_statistics.std-of-means".into(),
            ones(&[LATENT]),
        ));
        let path = root.join("vae/ltx-2.5-video-vae-diffusion-bf16.safetensors");
        let config = diff_vae_config(LATENT);
        write_file(
            &path,
            t,
            &[
                ("model_version", LTX_2_5_MODEL_VERSION),
                ("config", &config),
            ],
        );
        return path;
    }
    linear(&mut t, "conv_in", DIM, DIM);
    linear(&mut t, "conv_in_x_t", DIM, LATENT);
    linear(&mut t, "conv_out", LATENT, DIM);
    linear(&mut t, "shared_adaln.proj", 7 * DIM, DIM);
    linear(&mut t, "t_embedder.mlp.0", DIM, DIM);
    linear(&mut t, "t_embedder.mlp.2", DIM, DIM);
    linear(&mut t, "upsamples.0.proj", DIM * 8, DIM);
    for (stage, diffusion) in [("det_stages.0.0", false), ("diff_blocks.0", true)] {
        linear(&mut t, &format!("{stage}.attn.qkv"), 3 * DIM, DIM);
        linear(&mut t, &format!("{stage}.attn.proj"), DIM, DIM);
        if diffusion {
            linear(&mut t, &format!("{stage}.context_proj"), DIM, DIM);
            t.push((
                format!("decoder.{stage}.scale_shift_table"),
                ones(&[7, DIM]),
            ));
        }
        for (proj, out, inp) in [
            ("w_gate", hidden, DIM),
            ("w_up", hidden, DIM),
            ("w_down", DIM, hidden),
        ] {
            // The SwiGLU projections are biasless upstream, and the tier must keep them that way.
            t.push((
                format!("decoder.{stage}.mlp.{proj}.weight"),
                ones(&[out, inp]),
            ));
        }
        for norm in ["norm1", "norm2", "attn.q_norm", "attn.k_norm"] {
            t.push((format!("decoder.{stage}.{norm}.weight"), ones(&[DIM])));
        }
    }
    t.push(("decoder.norm_out.weight".into(), ones(&[DIM])));
    t.push((
        "per_channel_statistics.mean-of-means".into(),
        ones(&[LATENT]),
    ));
    t.push((
        "per_channel_statistics.std-of-means".into(),
        ones(&[LATENT]),
    ));

    let path = root.join("vae/ltx-2.5-video-vae-diffusion-bf16.safetensors");
    let config = diff_vae_config(LATENT);
    write_file(
        &path,
        t,
        &[
            ("model_version", LTX_2_5_MODEL_VERSION),
            ("config", &config),
        ],
    );
    path
}

/// The `CausalDiffusionVAE` config slice both DiffVAE fixtures declare — the same block either way,
/// so "the file says it is a diffusion VAE" is held fixed while the decoder tensors vary.
fn diff_vae_config(latent: i32) -> String {
    serde_json::json!({
        "vae": {
            "_class_name": "CausalDiffusionVAE",
            "dims": 3, "in_channels": 3, "out_channels": 3,
            "latent_channels": latent, "patch_size": 4,
            "latent_log_var": "constant",
        }
    })
    .to_string()
}

fn write_duration_head(root: &Path) -> PathBuf {
    let t: Vec<(String, Array)> = vec![
        (
            "duration_head.video_input_proj.weight".into(),
            ones(&[DIM, DIM]),
        ),
        ("duration_head.video_input_proj.bias".into(), ones(&[DIM])),
        ("duration_head.mlp_out.weight".into(), ones(&[1, DIM])),
        ("duration_head.video_modality_emb".into(), ones(&[DIM])),
    ];
    let path = root.join("model_patches/ltx-2.5-duration-head-bf16.safetensors");
    write_file(
        &path,
        t,
        &[
            ("model_version", LTX_2_5_MODEL_VERSION),
            (
                "config",
                &serde_json::json!({
                    "transformer": {"cross_attention_dim": DIM, "audio_cross_attention_dim": DIM},
                    "duration_head": {},
                })
                .to_string(),
            ),
        ],
    );
    path
}

/// Assemble the fixture bundle through the **real** split resolver, so a fixture that would not
/// resolve is a test failure rather than a private shortcut.
fn fixture_bundle(root: &Path) -> mlx_gen::gen_core::ltx_checkpoint::LtxBundle {
    LtxBundleBuilder::new()
        .with_component(LtxComponent::Transformer, write_transformer(root))
        .with_component(LtxComponent::TextEncoder, write_text_encoder(root))
        .with_component(LtxComponent::ConvVideoVae, write_conv_vae(root))
        .with_component(LtxComponent::DiffusionVideoVae, write_diff_vae(root))
        .with_component(LtxComponent::AudioVae, write_audio_vae(root))
        .with_component(LtxComponent::SpatialUpsampler, write_upsampler(root, false))
        .with_component(LtxComponent::TemporalUpsampler, write_upsampler(root, true))
        .with_component(LtxComponent::DurationHead, write_duration_head(root))
        .build()
        .expect("the fixture must resolve through the real split resolver")
}

// =================================================================================================
// Measuring a produced file (header only — never a hash)
// =================================================================================================

/// One tensor's declared dtype and shape, read from a produced file's safetensors header.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TensorHeader {
    dtype: String,
    shape: Vec<i64>,
}

/// A produced component file's header: `__metadata__` plus every tensor's dtype/shape.
///
/// Deliberately parses the header rather than loading through [`Weights`]: the point of these
/// assertions is what is **on disk**, and a loader that normalizes dtypes would launder exactly the
/// mistake being looked for.
struct FileHeader {
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, TensorHeader>,
}

impl FileHeader {
    fn read(path: &Path) -> FileHeader {
        let mut f =
            std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let mut len = [0u8; 8];
        f.read_exact(&mut len).unwrap();
        let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut buf).unwrap();
        let json: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&buf).unwrap();
        let mut metadata = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for (key, value) in json {
            if key == "__metadata__" {
                for (k, v) in value.as_object().unwrap() {
                    metadata.insert(k.clone(), v.as_str().unwrap().to_string());
                }
                continue;
            }
            tensors.insert(
                key,
                TensorHeader {
                    dtype: value["dtype"].as_str().unwrap().to_string(),
                    shape: value["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_i64().unwrap())
                        .collect(),
                },
            );
        }
        FileHeader { metadata, tensors }
    }

    fn dtype(&self, key: &str) -> &str {
        &self
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("tensor {key} missing"))
            .dtype
    }

    fn keys_ending(&self, suffix: &str) -> Vec<&str> {
        let mut keys: Vec<&str> = self
            .tensors
            .keys()
            .map(String::as_str)
            .filter(|k| k.ends_with(suffix))
            .collect();
        keys.sort_unstable();
        keys
    }
}

fn tier_header(dir: &Path, tier: LtxTier, component: &str) -> FileHeader {
    FileHeader::read(&dir.join(tier.id()).join(format!("{component}.safetensors")))
}

// =================================================================================================
// Tests
// =================================================================================================

/// All three tiers build, and **every** quantizable segment of a quantized tier is packed at that
/// tier's bit-width — measured from the produced files.
///
/// This is the whole-pipeline-contract assertion. It fails if the transformer is quantized but the
/// connector or the text encoder quietly stayed bf16, which is the specific way a component-split
/// converter goes wrong.
#[test]
fn every_quantizable_segment_lands_at_the_tier_precision() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let out = tmp.path().join("tiers");
    let reports =
        convert_2_5_tiers(&bundle, &out, LtxTier::ALL, DEFAULT_GROUP_SIZE).expect("build tiers");
    assert_eq!(reports.len(), 3);

    // Counts derived from the fixture's declared geometry, not pinned literals: 6 attentions x 4
    // projections + 4 FFN Linears per DiT block; 6 Linears per connector block across two towers;
    // Gemma's 3-or-4 attention projections + 3 MLP projections per layer, plus the two LTX
    // aggregate embeds.
    let dit = BLOCKS * (6 * 4 + 4);
    // Both connector towers, plus the two `text_embedding_projection` aggregate embeds the tier
    // moves here out of the text-encoder file.
    let connector = CONNECTOR_BLOCKS * 6 * 2 + 2;
    let gemma = GEMMA_LAYERS * 6 + (GEMMA_LAYERS - GEMMA_FULL_LAYERS.len());

    for report in &reports {
        let quantized = report.bits.is_some();
        for (component, expected) in [
            ("transformer", dit),
            ("connector", connector),
            ("text_encoder", gemma),
        ] {
            // The text encoder is the one measured exception: `q4` ships it dense
            // (`TEXT_ENCODER_Q4_QUALITY`). Asking the policy rather than hardcoding "q4 too" keeps
            // this test and the converter unable to disagree about which tiers pack it.
            let packs = quantized
                && !(component == "text_encoder"
                    && text_encoder_dense_reason(report.tier).is_some());
            let entry = report.component(component).expect(component);
            assert_eq!(
                entry.quantized_linears,
                if packs { expected } else { 0 },
                "{} / {component}: quantized-Linear count",
                report.tier
            );
            assert_eq!(
                entry.dense_reason.is_some(),
                !packs,
                "{} / {component}: a quantized component must carry no dense reason",
                report.tier
            );
        }

        // ...and the same claim re-derived from the files themselves.
        for (component, expected) in [
            ("transformer", dit),
            ("connector", connector),
            ("text_encoder", gemma),
        ] {
            let packs = quantized
                && !(component == "text_encoder"
                    && text_encoder_dense_reason(report.tier).is_some());
            let header = tier_header(&out, report.tier, component);
            let scales = header.keys_ending(".scales");
            let biases = header.keys_ending(".biases");
            assert_eq!(
                scales.len(),
                if packs { expected } else { 0 },
                "{} / {component}: `.scales` tensors on disk",
                report.tier
            );
            assert_eq!(scales.len(), biases.len());
            for key in &scales {
                let base = key.strip_suffix(".scales").unwrap();
                assert_eq!(
                    header.dtype(&format!("{base}.weight")),
                    "U32",
                    "{} / {component}: {base}.weight must be packed",
                    report.tier
                );
                assert_eq!(header.dtype(key), "BF16");
            }
            // Nothing that should be packed is still a float weight.
            for key in header.keys_ending(".weight") {
                let base = key.strip_suffix(".weight").unwrap();
                let packed = header.tensors.contains_key(&format!("{base}.scales"));
                if packed {
                    assert_eq!(header.dtype(key), "U32");
                } else {
                    assert!(
                        header.dtype(key) == "BF16" || header.dtype(key) == "U8",
                        "{} / {component}: {key} is {} — every unpacked float must be bf16",
                        report.tier,
                        header.dtype(key)
                    );
                }
            }
        }
    }
}

/// A quantized tier is strictly smaller than a less-quantized one, component by component, for the
/// components that quantize — and the exempt components are byte-identical in size across tiers.
#[test]
fn tier_sizes_are_ordered_and_the_exempt_components_do_not_move() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let out = tmp.path().join("tiers");
    let reports =
        convert_2_5_tiers(&bundle, &out, LtxTier::ALL, DEFAULT_GROUP_SIZE).expect("build tiers");
    let by_tier: BTreeMap<LtxTier, &LtxTierReport> = reports.iter().map(|r| (r.tier, r)).collect();

    let q4 = by_tier[&LtxTier::Q4];
    let q8 = by_tier[&LtxTier::Q8];
    let bf16 = by_tier[&LtxTier::Bf16];
    assert!(
        q4.bytes < q8.bytes && q8.bytes < bf16.bytes,
        "tier sizes must be strictly ordered: q4 {} < q8 {} < bf16 {}",
        q4.bytes,
        q8.bytes,
        bf16.bytes
    );
    for component in ["transformer", "connector"] {
        let a = q4.component(component).unwrap().bytes;
        let b = q8.component(component).unwrap().bytes;
        let c = bf16.component(component).unwrap().bytes;
        assert!(a < b && b < c, "{component}: {a} < {b} < {c}");
    }
    // The text encoder is the measured exception and breaks the ladder on purpose:
    // `q4` ships it dense (`TEXT_ENCODER_Q4_QUALITY`), so it is the **q8** file that is smallest,
    // and `q4`'s matches the dense tier's. Asserting the plain ordering here would either fail or,
    // worse, have to be relaxed into `<=` — which would stop noticing a q8 encoder that silently
    // went dense too.
    {
        let a = q4.component("text_encoder").unwrap().bytes;
        let b = q8.component("text_encoder").unwrap().bytes;
        let c = bf16.component("text_encoder").unwrap().bytes;
        assert!(b < a, "q8 must pack the encoder: q8 {b} < q4 {a}");
        assert!(
            a.abs_diff(c) < 64,
            "q4 ships the dense encoder, so it must match bf16's size: {a} vs {c}"
        );
    }
    // An exempt component carries the *same tensors* in every tier. Its file size is not literally
    // equal — the `sceneworks_tier` metadata value is 2 characters in `q4` and 4 in `bf16`, and the
    // safetensors header pads to an 8-byte boundary — so the claim is made on content, which is
    // also the only thing a downstream consumer may compare (see the metadata-ordering trap).
    for component in ["vae_decoder", "vae_encoder", "audio_vae", "vocoder"] {
        let a = tier_header(&out, LtxTier::Q4, component);
        let b = tier_header(&out, LtxTier::Bf16, component);
        assert_eq!(
            a.tensors, b.tensors,
            "{component} quantizes nothing, so every tier must carry identical tensors"
        );
        let (sa, sb) = (
            q4.component(component).unwrap().bytes,
            bf16.component(component).unwrap().bytes,
        );
        assert!(
            sa.abs_diff(sb) < 64,
            "{component}: {sa} vs {sb} differ by more than the tier stamp's header cost"
        );
    }
}

/// **The DiffVAE decoder is packed like any other Linear-bearing component**, and the one Linear the
/// affine grid cannot describe is the only one left dense.
///
/// Until sc-18766 landed the MLX `NADiffusionDecoder` this component was exempt under
/// [`DenseReason::NoMlxPort`] — an exemption that was correct when written and false the moment the
/// port merged, with nothing in the tree to notice. This test is what notices: it reads the produced
/// files and asserts on the exact tensors, so a decoder that silently reverts to dense inside a q4
/// tier fails here rather than shipping as a 4x-oversized "q4".
///
/// `conv_in_x_t` is checked by name and by dtype, not by "some tensor stayed dense": its input axis
/// is 48 and the group width is 64, so it is the single Linear with no grid — and pinning it by name
/// is what stops the exclusion list from quietly growing.
#[test]
fn the_diff_vae_decoder_is_packed_except_the_linear_no_affine_group_divides() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let out = tmp.path().join("tiers");
    convert_2_5_tiers(&bundle, &out, LtxTier::ALL, DEFAULT_GROUP_SIZE).unwrap();

    let component = crate::diff_vae::DIFFUSION_DECODER_COMPONENT;
    let dense = tier_header(&out, LtxTier::Bf16, component);
    for tier in [LtxTier::Q4, LtxTier::Q8] {
        let packed = tier_header(&out, tier, component);
        // Every Linear the loader binds through `.scales` must have become a triple.
        let mut triples = 0;
        for (key, header) in &dense.tensors {
            let Some(base) = key.strip_suffix(".weight") else {
                continue;
            };
            if !is_diff_vae_decoder_quantizable(key) {
                continue;
            }
            triples += 1;
            assert_eq!(
                packed.tensors[key].dtype, "U32",
                "{tier}: {key} must be packed"
            );
            for part in ["scales", "biases"] {
                assert!(
                    packed.tensors.contains_key(&format!("{base}.{part}")),
                    "{tier}: {key} is packed but carries no `{part}` — `quantized_matmul` cannot \
                     decode it"
                );
            }
            assert_eq!(
                header.dtype, "BF16",
                "the dense tier must keep {key} a float weight"
            );
        }
        assert!(
            triples >= 10,
            "{tier}: the fixture must exercise more than a token Linear or two, got {triples}"
        );
        // …and the one that cannot be, is not.
        assert_eq!(
            packed.tensors["conv_in_x_t.weight"].dtype, "BF16",
            "{tier}: conv_in_x_t's input axis is 48, which the group width \
             {DEFAULT_GROUP_SIZE} does not divide — it has no affine grid and must stay dense"
        );
        assert!(
            !packed.tensors.contains_key("conv_in_x_t.scales"),
            "{tier}: a dense weight must not carry a scales grid"
        );
        // The SwiGLU projections are biasless upstream; packing must not invent an output bias.
        assert!(
            !packed.tensors.contains_key("diff_blocks.0.mlp.w_gate.bias"),
            "{tier}: the SwiGLU projections carry no output bias in any tier"
        );
    }
}

/// **The measured TE exception, shipped**: `q8` packs the Gemma 4 encoder, `q4` does not.
///
/// `q4`'s encoder failed the quality bar in `mlx-llm`'s `ltx_2_5_te_tier_quality` measurement — worst
/// cos 0.889414 / rel L2 0.53488 over the 49 hidden states the LTX-2.5 feature extractor
/// concatenates — so the `q4` tier ships it dense. Three things have to agree for that to be a
/// decision rather than a bug, and this pins all three: no packed tensors, a declared
/// `below-quality-bar` reason carrying the evidence, and **no `quantization` block** (which would
/// send `mlx_llm` looking for `.scales` that are not there).
#[test]
fn q4_ships_the_text_encoder_dense_on_measured_evidence_and_q8_packs_it() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let out = tmp.path().join("tiers");
    let reports = convert_2_5_tiers(&bundle, &out, LtxTier::ALL, DEFAULT_GROUP_SIZE).unwrap();

    for report in &reports {
        let te = report.component("text_encoder").expect("a text encoder");
        let header = tier_header(&out, report.tier, "text_encoder");
        let packed = header
            .tensors
            .keys()
            .filter(|k| k.ends_with(".scales"))
            .count();
        match report.tier {
            LtxTier::Q8 => {
                assert!(
                    packed > 0 && te.dense_reason.is_none(),
                    "q8 packs the encoder: {packed} scales, reason {:?}",
                    te.dense_reason
                );
            }
            LtxTier::Q4 => {
                assert_eq!(packed, 0, "q4 must ship the encoder dense");
                let reason = te
                    .dense_reason
                    .expect("q4's dense encoder must declare why");
                assert_eq!(reason.id(), "below-quality-bar");
                assert!(
                    reason.describe().contains("0.889414"),
                    "the exemption must carry the measurement that drove it, got: {}",
                    reason.describe()
                );
            }
            LtxTier::Bf16 => assert_eq!(packed, 0),
        }
    }

    // q4's encoder is byte-for-byte the dense tier's, modulo the tier stamp — the only honest
    // meaning of "q4 ships the bf16 encoder".
    let q4 = tier_header(&out, LtxTier::Q4, "text_encoder");
    let dense = tier_header(&out, LtxTier::Bf16, "text_encoder");
    assert_eq!(
        q4.tensors, dense.tensors,
        "q4's encoder must carry exactly the dense tier's tensors"
    );
}

/// Every dense component of a quantized tier declares **why**, and the structural reason is checked
/// against the weights rather than trusted.
#[test]
fn dense_components_declare_a_reason_and_the_structural_one_is_verified() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let out = tmp.path().join("q4");
    let report = convert_2_5_tier(&bundle, &out, LtxTier::Q4, DEFAULT_GROUP_SIZE).unwrap();

    for entry in &report.components {
        if entry.quantized_linears > 0 {
            assert!(
                entry.dense_reason.is_none(),
                "{}: a component that quantized {} Linears must not declare a dense reason",
                entry.name,
                entry.quantized_linears
            );
            continue;
        }
        let reason = entry.dense_reason.unwrap_or_else(|| {
            panic!("{}: dense in a q4 tier with no declared reason", entry.name)
        });
        assert_ne!(
            reason,
            DenseReason::DenseTier,
            "{}: `dense-tier` is only valid for the bf16 tier",
            entry.name
        );
    }

    // The conv/audio components claim `no-linear-weights`; the emitter verifies that claim against
    // the weights, so a component that gained a Linear would fail the conversion rather than ship.
    for name in ["vae_decoder", "vae_encoder", "audio_vae", "vocoder"] {
        assert_eq!(
            report.component(name).unwrap().dense_reason,
            Some(DenseReason::NoLinearWeights),
            "{name}"
        );
    }
    assert_eq!(
        report.component("duration_head").unwrap().dense_reason,
        Some(DenseReason::NoMlxPort("sc-18777")),
    );

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join(TIER_MANIFEST_FILE)).unwrap())
            .unwrap();
    let detail = manifest["component_detail"].as_array().unwrap();
    for entry in detail {
        if entry["quantized_linears"].as_u64().unwrap() == 0 {
            assert!(
                entry["dense_reason"].is_string() && entry["dense_reason_detail"].is_string(),
                "the manifest must carry the reason and its justification: {entry}"
            );
        }
    }
}

/// A component whose declared exemption is wrong is a conversion **error**, not a silent pass.
///
/// The executed control on [`DenseReason::NoLinearWeights`]: without it the exemption would be a
/// comment, and a VAE that grew a Linear would ship dense inside a q4 tier with a reason that no
/// longer applies.
#[test]
fn a_no_linear_weights_exemption_is_refused_when_the_component_has_linears() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("src");
    let vae = write_conv_vae(&root);
    // Rewrite the conv VAE with a rank-2 Linear in the decoder.
    let mut t: Vec<(String, Array)> = vec![
        ("decoder.conv_in.conv.weight".into(), ones(&[8, 4, 3, 3, 3])),
        ("decoder.proj.weight".into(), ones(&[DIM, DIM])),
        ("encoder.conv_in.conv.weight".into(), ones(&[8, 4, 3, 3, 3])),
        ("per_channel_statistics.mean-of-means".into(), ones(&[4])),
        ("per_channel_statistics.std-of-means".into(), ones(&[4])),
    ];
    t.sort_by(|a, b| a.0.cmp(&b.0));
    write_file(
        &vae,
        t,
        &[
            ("model_version", LTX_2_5_MODEL_VERSION),
            (
                "config",
                &serde_json::json!({"vae": {"_class_name": "CausalVideoAutoencoder",
                    "latent_channels": 128, "patch_size": 4, "latent_log_var": "uniform"}})
                .to_string(),
            ),
        ],
    );
    let bundle = LtxBundleBuilder::new()
        .with_component(LtxComponent::Transformer, write_transformer(&root))
        .with_component(LtxComponent::TextEncoder, write_text_encoder(&root))
        .with_component(LtxComponent::ConvVideoVae, vae)
        .build()
        .unwrap();
    let err = convert_2_5_tier(
        &bundle,
        tmp.path().join("q4"),
        LtxTier::Q4,
        DEFAULT_GROUP_SIZE,
    )
    .expect_err("a VAE with a Linear must not pass the `no-linear-weights` exemption");
    let text = err.to_string();
    assert!(text.contains("no-linear-weights"), "{text}");
    assert!(text.contains("rank-2 float weight"), "{text}");
}

/// Per-component `__metadata__` survives into the packed output, and the text encoder's
/// `gemma_config` gains the `quantization` block that binds its packed projections — in the
/// quantized tiers only.
#[test]
fn component_metadata_travels_and_the_gemma_quantization_block_is_tier_scoped() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let out = tmp.path().join("tiers");
    convert_2_5_tiers(&bundle, &out, LtxTier::ALL, DEFAULT_GROUP_SIZE).unwrap();

    for tier in LtxTier::ALL {
        let transformer = tier_header(&out, *tier, "transformer");
        assert_eq!(
            transformer
                .metadata
                .get("model_version")
                .map(String::as_str),
            Some(LTX_2_5_MODEL_VERSION),
            "{tier}: the transformer must keep its model_version — the split layout is keyed on it"
        );
        assert!(
            transformer.metadata.contains_key("gemma_source_checkpoint"),
            "{tier}: the Gemma version assertion reads this off the transformer"
        );
        assert!(
            transformer.metadata.contains_key("license"),
            "{tier}: the embedded licence travels with the weights it licenses"
        );
        let config: serde_json::Value =
            serde_json::from_str(&transformer.metadata["config"]).unwrap();
        assert_eq!(config["transformer"]["num_layers"], BLOCKS);
        assert!(
            config["vae"].is_null(),
            "{tier}: 2.5 nulls the sibling sections"
        );
        assert_eq!(transformer.metadata[TIER_METADATA_KEY], tier.id());

        // The connector is split out of the transformer file and must NOT re-declare
        // `config.transformer` — two files claiming the transformer component makes the directory
        // scan ambiguous.
        let connector = tier_header(&out, *tier, "connector");
        assert!(
            !connector.metadata.contains_key("config"),
            "{tier}: the connector must not carry a config section of its own"
        );
        assert_eq!(connector.metadata["sceneworks_derived_from"], "transformer");

        let te = tier_header(&out, *tier, "text_encoder");
        let gemma: serde_json::Value = serde_json::from_str(&te.metadata["gemma_config"]).unwrap();
        assert_eq!(gemma["gemma_version"], "gemma4-12b-ltx-v1");
        // Read the block from the exact object `mlx_llm::config::ModelConfig::from_json` reads it
        // from. Gemma 4 `nests_text_config()`, so `from_json` rebinds to `text_config` *before* it
        // looks for `quantization`; a block at the wrapper's top level is invisible to it and the
        // tier's packed encoder then fails to load at all. Asserting on the top level is the
        // convergence-point mistake that let that ship — this reads `text_config` and pins the top
        // level empty so the block has exactly one home.
        //
        // The block is present iff *this tier packs the encoder*, which is not the same as "this
        // tier packs" — `q4` ships the encoder dense on measured evidence (`TEXT_ENCODER_Q4_QUALITY`)
        // and must therefore carry no block, or `mlx_llm` would look for `.scales` that are not
        // there.
        //
        // The expectation is **hardcoded per tier**, deliberately: keying it off
        // `text_encoder_dense_reason` — the decision function under test — would make this pass no
        // matter which way that decision flipped, which is the one thing it exists to catch.
        let nested = &gemma["text_config"];
        assert!(
            gemma.get("quantization").is_none(),
            "{tier}: the `quantization` block belongs in `text_config`, not the wrapper's top level"
        );
        match tier {
            LtxTier::Q8 => {
                assert_eq!(nested["quantization"]["bits"], 8);
                assert_eq!(nested["quantization"]["group_size"], DEFAULT_GROUP_SIZE);
                assert_eq!(nested["quantization"]["mode"], "affine");
            }
            LtxTier::Q4 | LtxTier::Bf16 => assert!(
                nested.get("quantization").is_none(),
                "{tier}: a tier that ships the encoder dense must not claim a quantization block"
            ),
        }
    }
}

/// The Gemma 4 shapes the loader depends on survive the repack: the 48 (here 4) per-layer
/// `layer_scalar` buffers, the missing `v_proj` on the `full_attention` layers, the embedding table
/// left dense, and the `U8` packed HF assets byte-identical.
#[test]
fn the_gemma_checkpoints_load_bearing_shapes_survive_the_repack() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let bundle = fixture_bundle(&src);
    let out = tmp.path().join("tiers");
    convert_2_5_tiers(&bundle, &out, LtxTier::ALL, DEFAULT_GROUP_SIZE).unwrap();

    let source =
        FileHeader::read(&src.join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"));
    for tier in LtxTier::ALL {
        let te = tier_header(&out, *tier, "text_encoder");
        for l in 0..GEMMA_LAYERS {
            let scalar = format!("model.layers.{l}.layer_scalar");
            assert_eq!(
                te.tensors.get(&scalar),
                source.tensors.get(&scalar),
                "{tier}: layer {l}'s trained `layer_scalar` must survive verbatim"
            );
            let v = format!("model.layers.{l}.self_attn.v_proj.weight");
            assert_eq!(
                te.tensors.contains_key(&v),
                !GEMMA_FULL_LAYERS.contains(&l),
                "{tier}: layer {l}'s v_proj presence must match the source"
            );
        }
        // The embedding table is an exempt lookup, never packed.
        assert_eq!(te.dtype("model.embed_tokens.weight"), "BF16", "{tier}");
        assert!(
            !te.tensors.contains_key("model.embed_tokens.scales"),
            "{tier}: the embedding table has no quantized read path and must stay dense"
        );
        // Packed assets pass through untouched — a cast or a quantize here corrupts the tokenizer.
        for asset in [
            "tokenizer_json",
            "hf_asset__tokenizer_config.json",
            "hf_asset__processor_config.json",
        ] {
            assert_eq!(
                te.tensors.get(asset),
                source.tensors.get(asset),
                "{tier}: {asset}"
            );
        }
    }

    // The asset payloads themselves, byte-for-byte, through the real unpacker.
    for tier in LtxTier::ALL {
        let packed = out.join(tier.id()).join("text_encoder.safetensors");
        let assets = mlx_gen::gen_core::gemma_assets::GemmaAssets::from_single_file(&packed)
            .unwrap_or_else(|e| panic!("{tier}: unpack the tier's Gemma assets: {e}"));
        let original = mlx_gen::gen_core::gemma_assets::GemmaAssets::from_single_file(
            src.join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        )
        .unwrap();
        assert_eq!(
            assets.tokenizer_json(),
            original.tokenizer_json(),
            "{tier}: the packed tokenizer must be byte-identical"
        );
        assert_eq!(
            assets.sidecar("tokenizer_config.json").unwrap(),
            original.sidecar("tokenizer_config.json").unwrap()
        );
    }
}

/// A tier directory resolves as a 2.5 split bundle and its sidecars parse through the shipped
/// config readers — the "loads end-to-end" half that does not need a GPU.
#[test]
fn a_tier_directory_resolves_and_its_sidecars_parse() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let out = tmp.path().join("q8");
    convert_2_5_tier(&bundle, &out, LtxTier::Q8, DEFAULT_GROUP_SIZE).unwrap();

    // The manifest declares 2.5, so the tree keeps the split layout.
    assert_eq!(
        crate::bundle::declared_model_version(&out)
            .unwrap()
            .as_deref(),
        Some(LTX_2_5_MODEL_VERSION)
    );
    assert_eq!(
        crate::bundle::declared_layout(&out).unwrap(),
        mlx_gen::gen_core::ltx_checkpoint::LtxCheckpointLayout::Split
    );

    // The quant geometry the DiT loader reads.
    let split = crate::config::SplitModel::from_model_dir(&out).unwrap();
    assert!(split.quantized);
    assert_eq!(split.bits, 8);
    assert_eq!(split.group, DEFAULT_GROUP_SIZE);

    // The merged config sidecar the shipped ports read.
    let cfg = crate::config::LtxConfig::from_model_dir(&out).unwrap();
    assert_eq!(cfg.num_layers, BLOCKS as i32);
    assert_eq!(cfg.connector_num_layers, CONNECTOR_BLOCKS as i32);
    let vae = crate::config::LtxVaeConfig::from_model_dir(&out).unwrap();
    assert_eq!(vae.latent_channels, 128);
    assert_eq!(vae.patch_size, 4);

    // **Every component slot the source bundle carried must resolve from a scan of the tier tree**,
    // and no slot may be ambiguous. This is the property that makes a tier a drop-in for
    // `resolve_split_bundle`, and it is exactly what breaks if a split half keeps a config section
    // it should have dropped (two files claiming one slot) or drops one it should have kept (no
    // file claiming it).
    let scanned = mlx_gen::gen_core::ltx_checkpoint::discover_split_bundle(&out)
        .expect("a tier tree must resolve without an ambiguous component");
    for component in [
        LtxComponent::Transformer,
        LtxComponent::TextEncoder,
        LtxComponent::ConvVideoVae,
        LtxComponent::AudioVae,
        LtxComponent::SpatialUpsampler,
        LtxComponent::TemporalUpsampler,
        LtxComponent::DurationHead,
    ] {
        let resolved = scanned
            .require(component)
            .unwrap_or_else(|e| panic!("{} must resolve from the tier tree: {e}", component.id()));
        // ...and it must resolve to the file that actually holds that component's weights.
        let expected = match component {
            LtxComponent::ConvVideoVae => "vae_decoder.safetensors",
            LtxComponent::TextEncoder => "text_encoder.safetensors",
            LtxComponent::Transformer => "transformer.safetensors",
            LtxComponent::AudioVae => "audio_vae.safetensors",
            LtxComponent::SpatialUpsampler => "spatial_upsampler.safetensors",
            LtxComponent::TemporalUpsampler => "temporal_upsampler.safetensors",
            LtxComponent::DurationHead => "duration_head.safetensors",
            other => panic!("unexpected component {}", other.id()),
        };
        assert!(
            resolved.path().ends_with(expected),
            "{} resolved to {}, expected {expected}",
            component.id(),
            resolved.path().display()
        );
    }
    // The secondary halves declare nothing, so they are searched-but-unclaimed rather than rival
    // candidates for a slot.
    for name in ["connector", "vae_encoder", "vocoder"] {
        let path = out.join(format!("{name}.safetensors"));
        let meta = mlx_gen::gen_core::ltx_checkpoint::LtxCheckpointMetadata::from_file(&path)
            .unwrap_or_else(|e| panic!("read {name} metadata: {e}"));
        assert_eq!(
            meta.classify(),
            None,
            "{name} must not claim a component slot"
        );
    }
}

/// Two conversions of the same input agree on **content** even though their bytes may differ.
///
/// This is the safetensors metadata-ordering trap made executable: the tier verification downstream
/// (sc-18780's rehost, sc-18781's manifest footprints) must compare key sets, dtypes and shapes —
/// never a file hash — because `save_file` writes `__metadata__` in an unstable order.
#[test]
fn two_conversions_agree_on_content_not_necessarily_on_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = fixture_bundle(&tmp.path().join("src"));
    let a = tmp.path().join("a");
    let b = tmp.path().join("b");
    let ra = convert_2_5_tier(&bundle, &a, LtxTier::Q4, DEFAULT_GROUP_SIZE).unwrap();
    let rb = convert_2_5_tier(&bundle, &b, LtxTier::Q4, DEFAULT_GROUP_SIZE).unwrap();
    assert_eq!(ra.quantized_linears(), rb.quantized_linears());
    for entry in &ra.components {
        let ha = FileHeader::read(&entry.file);
        let hb = FileHeader::read(&b.join(format!("{}.safetensors", entry.name)));
        assert_eq!(
            ha.tensors, hb.tensors,
            "{}: two conversions must agree on every tensor's dtype and shape",
            entry.name
        );
        assert_eq!(ha.metadata, hb.metadata, "{}: metadata content", entry.name);
    }
}

/// The tier ids round-trip and cover the shipped subdirectory names.
#[test]
fn tier_ids_round_trip() {
    for tier in LtxTier::ALL {
        assert_eq!(LtxTier::from_id(tier.id()), Some(*tier));
    }
    assert_eq!(LtxTier::from_id("fp8"), None);
    assert_eq!(LtxTier::Q4.bits(), Some(4));
    assert_eq!(LtxTier::Q8.bits(), Some(8));
    assert_eq!(LtxTier::Bf16.bits(), None);
}

/// The connector predicate selects the connector's attention/FFN Linears under their **raw**
/// naming and leaves the gate, the norms and the registers dense.
#[test]
fn the_connector_predicate_matches_the_raw_checkpoint_naming() {
    let q = |k: &str| matches_quant_suffix(k, CONNECTOR_QUANT_SUFFIXES);
    for key in [
        "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight",
        "video_embeddings_connector.transformer_1d_blocks.7.attn1.to_out.0.weight",
        "audio_embeddings_connector.transformer_1d_blocks.3.ff.net.0.proj.weight",
        "audio_embeddings_connector.transformer_1d_blocks.3.ff.net.2.weight",
    ] {
        assert!(q(key), "should quantize: {key}");
    }
    for key in [
        "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.bias",
        "video_embeddings_connector.transformer_1d_blocks.0.attn1.q_norm.weight",
        "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight",
        "video_embeddings_connector.video_embeddings_connector.learnable_registers",
    ] {
        assert!(!q(key), "should stay dense: {key}");
    }
}

/// The text-encoder predicate covers the Gemma projections plus the LTX aggregate embeds, and never
/// the packed assets or the embedding table.
#[test]
fn the_text_encoder_predicate_covers_the_projections_and_nothing_else() {
    for key in [
        "model.layers.11.self_attn.q_proj.weight",
        "model.layers.11.self_attn.o_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    ] {
        assert!(is_text_encoder_quantizable(key), "should quantize: {key}");
    }
    // The aggregate embeds ARE quantized — but under the connector's predicate, because the tier
    // moves them into that component. Asserting both halves here keeps the split honest: a future
    // change that stopped moving them would leave them matched by neither list and silently dense.
    for key in [
        "text_embedding_projection.video_aggregate_embed.weight",
        "text_embedding_projection.audio_aggregate_embed.weight",
    ] {
        assert!(
            !is_text_encoder_quantizable(key),
            "the text encoder no longer owns: {key}"
        );
        assert!(
            matches_quant_suffix(key, CONNECTOR_QUANT_SUFFIXES),
            "the connector must own: {key}"
        );
    }
    for key in [
        "model.embed_tokens.weight",
        "model.layers.0.layer_scalar",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
        "text_embedding_projection.video_aggregate_embed.bias",
        "tokenizer_json",
        "hf_asset__tokenizer_config.json",
        "vision_model.patch_dense.weight",
        "multi_modal_projector.embedding_projection.weight",
    ] {
        assert!(
            !is_text_encoder_quantizable(key),
            "should stay dense: {key}"
        );
    }
}

/// **A Linear nobody listed is a conversion error, not a bf16 tensor inside a q4 tier.**
///
/// [`quantize_selected`] is an allowlist over *names*, so before this check a renamed `conv_in_x_t`,
/// a new upstream Linear, or a drifted suffix sailed through dense with no [`DenseReason`] anywhere:
/// the component still quantized hundreds of other Linears, so the "quantized nothing" guard stayed
/// quiet and the tier shipped a weight the loader would bind dense where the manifest promised
/// packed. The doctored transformer here carries exactly one such weight.
#[test]
fn an_unlisted_rank2_linear_is_refused_rather_than_shipped_dense() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("src");
    let transformer = write_transformer_with(&root, &[("mystery_proj.weight", ones(&[DIM, DIM]))]);
    let bundle = LtxBundleBuilder::new()
        .with_component(LtxComponent::Transformer, transformer)
        .with_component(LtxComponent::TextEncoder, write_text_encoder(&root))
        .with_component(LtxComponent::ConvVideoVae, write_conv_vae(&root))
        .build()
        .unwrap();
    let err = convert_2_5_tier(
        &bundle,
        tmp.path().join("q4"),
        LtxTier::Q4,
        DEFAULT_GROUP_SIZE,
    )
    .expect_err("an unlisted rank-2 Linear must fail the conversion");
    let text = err.to_string();
    assert!(text.contains("mystery_proj.weight"), "{text}");
    assert!(text.contains("expected-dense"), "{text}");

    // …and the same bundle without that one weight converts, so the refusal is about the weight
    // rather than about anything else the doctored fixture did.
    let root = tmp.path().join("clean");
    let bundle = LtxBundleBuilder::new()
        .with_component(LtxComponent::Transformer, write_transformer(&root))
        .with_component(LtxComponent::TextEncoder, write_text_encoder(&root))
        .with_component(LtxComponent::ConvVideoVae, write_conv_vae(&root))
        .build()
        .unwrap();
    convert_2_5_tier(
        &bundle,
        tmp.path().join("q4-clean"),
        LtxTier::Q4,
        DEFAULT_GROUP_SIZE,
    )
    .expect("the same bundle minus the unlisted Linear must convert");
}

/// The expected-dense allowlist names **whole keys** for index-free weights and only pattern-matches
/// where an index forces it, so an entry cannot quietly cover a per-block weight that happens to end
/// the same way.
///
/// `proj_out` is the live collision: the DiT's top-level unpatchify projection is expected-dense,
/// while every block's `ff.proj_out` is a **quantized** Linear. An `ends_with` allowlist would cover
/// both and re-open the hole the check above closes.
#[test]
fn the_expected_dense_allowlist_does_not_swallow_a_quantized_namesake() {
    assert!(is_expected_dense("proj_out", TRANSFORMER_EXPECTED_DENSE));
    assert!(is_expected_dense(
        "audio_proj_out",
        TRANSFORMER_EXPECTED_DENSE
    ));
    assert!(
        !is_expected_dense(
            "transformer_blocks.7.ff.proj_out",
            TRANSFORMER_EXPECTED_DENSE
        ),
        "a block's FFN output projection is quantized, and the allowlist must not claim it"
    );
    // The indexed sites still match through their `*` patterns.
    assert!(is_expected_dense(
        "transformer_blocks.7.attn1.to_gate_logits",
        TRANSFORMER_EXPECTED_DENSE
    ));
    assert!(is_expected_dense(
        "audio_adaln_single.emb.timestep_embedder.linear1",
        TRANSFORMER_EXPECTED_DENSE
    ));
    assert!(!is_expected_dense(
        "transformer_blocks.7.attn1.to_q",
        TRANSFORMER_EXPECTED_DENSE
    ));
}

/// **A DiffVAE source with no diffusion decoder fails the conversion** rather than producing a tier
/// that declares one.
///
/// The tier's `embedded_config.json` gains its `diffusion_vae` section from the same source file, so
/// skipping the decoder emission — which is what the old `if has_stages && has_blocks` did — shipped
/// a bundle whose declared decoder had no file behind it, with nothing said at conversion time.
#[test]
fn a_diffusion_vae_with_no_decoder_is_an_error_not_a_silent_skip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("src");
    let bundle = LtxBundleBuilder::new()
        .with_component(LtxComponent::Transformer, write_transformer(&root))
        .with_component(LtxComponent::TextEncoder, write_text_encoder(&root))
        .with_component(LtxComponent::ConvVideoVae, write_conv_vae(&root))
        .with_component(
            LtxComponent::DiffusionVideoVae,
            write_diff_vae_with(&root, false),
        )
        .build()
        .unwrap();
    let err = convert_2_5_tier(
        &bundle,
        tmp.path().join("q4"),
        LtxTier::Q4,
        DEFAULT_GROUP_SIZE,
    )
    .expect_err("a diffusion VAE with no decoder tensors must fail the conversion");
    let text = err.to_string();
    assert!(
        text.contains(crate::diff_vae::DIFFUSION_DECODER_COMPONENT),
        "{text}"
    );
    assert!(text.contains("det_stages"), "{text}");
}

/// A selected weight whose input axis does not divide the group size is a hard error, never a
/// silent skip — a skip is exactly how a component quietly stays bf16 inside a q4 tier.
#[test]
fn a_misaligned_quantizable_weight_is_an_error_not_a_skip() {
    let mut map = HashMap::new();
    map.insert("blk.attn1.to_q.weight".to_string(), ones(&[8, 100]));
    let err = quantize_selected(map, 4, 64, "connector", |k| {
        matches_quant_suffix(k, CONNECTOR_QUANT_SUFFIXES)
    })
    .expect_err("100 is not a multiple of 64");
    let text = err.to_string();
    assert!(text.contains("group size 64"), "{text}");
    assert!(text.contains("to_q.weight"), "{text}");
}
