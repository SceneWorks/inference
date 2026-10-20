//! Real-checkpoint loading for Z-Image-turbo: assemble the tokenizer, Qwen text encoder, DiT
//! transformer, and VAE decoder from a `Tongyi-MAI/Z-Image-Turbo` snapshot directory, applying
//! the diffusers-checkpoint → internal-name remaps (the fork's `z_image_weight_mapping`).
//!
//! The snapshot layout is the standard diffusers multi-component tree:
//! ```text
//!   <root>/tokenizer/tokenizer.json
//!   <root>/text_encoder/*.safetensors
//!   <root>/transformer/*.safetensors
//!   <root>/vae/*.safetensors
//! ```
//! These loaders were validated stage-by-stage against the Python fork on real bf16 weights
//! (sc-2352); they are the production path behind [`crate::model::load`].

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Result, WeightsSource};

use crate::control_transformer::ZImageControlTransformer;
use crate::text_encoder::{TextEncoder, ZTextEncoderConfig};
use crate::transformer::{ZImageTransformer, ZImageTransformerConfig};
use crate::vae::{Vae, VaeDecoderConfig, VaeEncoderConfig};

/// Qwen3 pad token id (`<|endoftext|>`).
const PAD_TOKEN_ID: i32 = 151643;
/// Prompts pad to this length (the fork's `padding="max_length"`).
pub(crate) const MAX_LENGTH: usize = 512;

/// The tensors a streamable text-encoder load reads up front: everything but the layer stack, whose
/// layers are read inside their own priced quanta (an eager load reads everything).
pub(crate) fn te_load_keeps(streamable: bool, name: &str) -> bool {
    !(streamable && name.starts_with("model.layers."))
}

/// The tensors a DiT load reads up front: a streamed stack's block bodies are read inside each
/// block's own priced quantum instead.
pub(crate) fn dit_load_keeps(streamed: bool, name: &str) -> bool {
    !(streamed && name.starts_with("layers."))
}

/// The tensors a ControlNet load reads up front: a streamed control stack's block bodies are read
/// inside each block's own priced quantum instead.
pub(crate) fn control_load_keeps(streamed: bool, name: &str) -> bool {
    !(streamed && name.starts_with("control_layers."))
}

/// Load the Qwen tokenizer with the Z-Image chat-template + padding policy.
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let source = crate::ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    load_validated_tokenizer(&source)
}

pub(crate) fn load_validated_tokenizer(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<TextTokenizer> {
    let tokenizer = source.tokenizer_source().ok_or_else(|| {
        mlx_gen::Error::Unsupported(
            "Z-Image tokenizer parsing requires a retained encoder-source receipt".into(),
        )
    })?;
    load_tokenizer_from_receipt(tokenizer)
}

pub(crate) fn load_tokenizer_from_receipt(
    source: &mlx_gen::gen_core::ValidatedTokenizerSource,
) -> Result<TextTokenizer> {
    source.read_unchanged(|path| {
        TextTokenizer::from_file(
            path,
            TokenizerConfig {
                max_length: MAX_LENGTH,
                pad_token_id: PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstruct,
                pad_to_max_length: true,
            },
        )
        .map_err(Into::into)
    })
}

/// Load the Qwen3-style text encoder (prompt → `cap_feats`). The checkpoint keys are prefixed
/// `model.` (the encoder's own modules); no other remap is needed.
pub fn load_text_encoder(root: &Path) -> Result<TextEncoder> {
    let selected = crate::ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    load_validated_text_encoder(&selected)
}

pub(crate) fn load_text_encoder_from_source(source: &WeightsSource) -> Result<TextEncoder> {
    load_text_encoder_from_source_with(source, &ZTextEncoderConfig::z_image())
}

fn load_text_encoder_from_source_with(
    source: &WeightsSource,
    cfg: &ZTextEncoderConfig,
) -> Result<TextEncoder> {
    // iOS: the header-only pre-load floor, checked before any tensor is read.
    crate::quantum::before_load(crate::quantum::TE_LOAD, source, |name| {
        te_load_keeps(false, name)
    })?;
    let w = weights_from_source(source)?;
    let encoder = TextEncoder::from_weights(&w, "model", cfg)?;
    w.materialize_accessed()?;
    Ok(encoder)
}

pub(crate) fn load_validated_text_encoder(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<TextEncoder> {
    source.read_unchanged(load_text_encoder_from_source)
}

/// The same validated eager loader at an explicit encoder geometry, for weights-free fixtures.
#[cfg(test)]
pub(crate) fn load_validated_text_encoder_with(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
    cfg: &ZTextEncoderConfig,
) -> Result<TextEncoder> {
    source.read_unchanged(|weights| load_text_encoder_from_source_with(weights, cfg))
}

/// The **streamable** encoder for rung 4's text-encoder scope (SC-15794): it records the re-openable
/// directory so the layer stack can be materialized a window at a time.
///
/// Selected by [`mlx_gen::LoadShape::DeferredMaterialization`], independently from phase-level
/// residency. A Resident+Deferred generator intentionally re-materializes the 36-layer window per
/// encode while retaining its tokenizer and non-windowed generator state. SC-15998 measured that
/// shape with no phase reload, so the repeated materialization is an explicit load-shape cost rather
/// than an accidental rung-4 fallback.
pub fn load_text_encoder_streamable(root: &Path) -> Result<TextEncoder> {
    let selected = crate::ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    load_validated_text_encoder_streamable(&selected)
}

pub(crate) fn load_validated_text_encoder_streamable(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<TextEncoder> {
    load_validated_text_encoder_streamable_with(source, &ZTextEncoderConfig::z_image())
}

/// [`load_validated_text_encoder_streamable`] at an explicit encoder geometry (the production one,
/// or a tiny fixture's).
pub(crate) fn load_validated_text_encoder_streamable_with(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
    cfg: &ZTextEncoderConfig,
) -> Result<TextEncoder> {
    source.read_unchanged(|weights| {
        // iOS: only the token embedding is read here; every layer is read inside its own priced
        // quantum when its window materializes it.
        crate::quantum::before_load(crate::quantum::TE_LOAD, weights, |name| {
            te_load_keeps(true, name)
        })?;
        let w = weights_from_source(weights)?;
        let encoder =
            TextEncoder::from_validated_streamable_source(&w, source.clone(), "model", cfg)?;
        w.materialize_accessed()?;
        Ok(encoder)
    })
}

fn weights_from_source(source: &WeightsSource) -> Result<Weights> {
    match source {
        WeightsSource::Dir(path) => Weights::from_dir(path),
        WeightsSource::File(path) => Weights::from_file(path),
    }
}

/// The in-memory / ComfyUI-normalized path: no re-openable source, so no stream. The contract declares
/// the text-encoder component unavailable for such a load, exactly as it does for the DiT stream.
pub(crate) fn load_text_encoder_from_weights(w: Weights) -> Result<TextEncoder> {
    TextEncoder::from_weights(&w, "model", &ZTextEncoderConfig::z_image())
}

/// Load the DiT transformer, applying the timestep-embedder + final-layer key remaps.
///
/// KEEP-F32 (sc-2609): Z-Image-Turbo ships f32 weights on disk and we deliberately keep them f32
/// (the fork downcasts to bf16). The dense render is then *sharper / more contrast* than the fork's
/// bf16 — Michael's preferred output — and it costs nothing today (activations already run f32, so
/// the matmul is f32 either way). Do NOT downcast these to bf16 for parity. The only place we cast
/// to bf16 is the quantize path (`AdaptableLinear::quantize`, tagged PARITY-BF16) to byte-match the
/// fork's Q8/Q4 golden; that too is a flip-to-f32 candidate once parity stops being the goal.
pub fn load_transformer(root: &Path) -> Result<ZImageTransformer> {
    // Armed, but its callers (training, fixtures) run the resident stack: materialize every block.
    load_transformer_with_stream(root, true, false)
}

/// Load the transformer in the component form selected for this request.
///
/// `streamed` (the request's deferred-materialization shape, which always runs a block window)
/// leaves the `layers.*` bodies lazy: the stream reads each block itself, and reading them here
/// would defeat bounded residency. Everything else is materialized at load (sc-24245; see
/// `mlx_gen_qwen_image::loader::load_transformer_with`).
pub(crate) fn load_transformer_with_stream(
    root: &Path,
    streamable: bool,
    streamed: bool,
) -> Result<ZImageTransformer> {
    load_transformer_with_stream_cfg(root, ZImageTransformerConfig::turbo(), streamable, streamed)
}

/// [`load_transformer_with_stream`] at an explicit DiT geometry (the production one, or a tiny
/// fixture's).
pub(crate) fn load_transformer_with_stream_cfg(
    root: &Path,
    cfg: ZImageTransformerConfig,
    streamable: bool,
    streamed: bool,
) -> Result<ZImageTransformer> {
    let dir = root.join("transformer");
    // iOS: the header-only floor of what this load reads (a streamed stack's block bodies are read
    // inside each block's own priced quantum instead).
    crate::quantum::before_load(
        crate::quantum::DIT_LOAD,
        &WeightsSource::Dir(dir.clone()),
        |name| dit_load_keeps(streamed, name),
    )?;
    let mut w = Weights::from_dir(&dir)?;
    remap_transformer_keys(&mut w);
    let transformer = ZImageTransformer::from_weights(&w, "", cfg)?;
    materialize_built(&w, streamed.then_some("layers."))?;
    // SC-15754: a snapshot directory is re-openable, so rung 4 (bounded transformer residency) can
    // rebuild `layers` per window from exactly these files. `remap_transformer_keys` only aliases the
    // timestep-embedder and final-layer keys — never a `layers.*` key — so a streamed block reads the
    // same names off disk that the resident load did, with no remap to replay.
    Ok(if streamable {
        transformer.with_block_stream(WeightsSource::Dir(dir), "")
    } else {
        transformer
    })
}

pub(crate) fn load_transformer_from_weights(mut w: Weights) -> Result<ZImageTransformer> {
    build_transformer(&mut w)
}

fn build_transformer(w: &mut Weights) -> Result<ZImageTransformer> {
    remap_transformer_keys(w);
    ZImageTransformer::from_weights(w, "", ZImageTransformerConfig::turbo())
}

/// Materialize the tensors a build read, except a streamed block stack's bodies (`streamed_blocks`
/// key prefix), which its block stream reads per window (sc-24245). Left lazy, the first forward's
/// Metal command buffers wait on the safetensors reads — past the GPU watchdog on a cold page cache.
fn materialize_built(w: &Weights, streamed_blocks: Option<&str>) -> Result<()> {
    let Some(prefix) = streamed_blocks else {
        return w.materialize_accessed();
    };
    let resident: Vec<(String, mlx_rs::Array)> = w
        .accessed_entries()
        .into_iter()
        .filter(|(key, _)| !key.starts_with(prefix))
        .collect();
    let mut named: Vec<(&str, &mlx_rs::Array)> =
        resident.iter().map(|(k, a)| (k.as_str(), a)).collect();
    Weights::materialize_named(&mut named)
}

/// Load the ControlNet transformer (sc-2349): the base DiT from the snapshot `root`, overlaid with
/// the Fun-Controlnet-Union control branch from `control` (a single `.safetensors` `File`, or a
/// `Dir` of them — the HF cache stores the checkpoint as one file inside a snapshot dir). The
/// control keys map 1:1 onto the control branch's param tree, so no remap is needed.
pub fn load_control_transformer(
    root: &Path,
    control: &WeightsSource,
) -> Result<ZImageControlTransformer> {
    // Armed, but run resident by its callers: materialize every block (see `load_transformer`).
    load_control_transformer_with_stream(root, control, true, false)
}

/// `streamed` as for [`load_transformer_with_stream`], for both the base and control stacks.
pub(crate) fn load_control_transformer_with_stream(
    root: &Path,
    control: &WeightsSource,
    streamable: bool,
    streamed: bool,
) -> Result<ZImageControlTransformer> {
    load_control_transformer_with_stream_cfg(
        root,
        control,
        ZImageTransformerConfig::turbo(),
        streamable,
        streamed,
    )
}

/// [`load_control_transformer_with_stream`] at an explicit DiT geometry.
pub(crate) fn load_control_transformer_with_stream_cfg(
    root: &Path,
    control: &WeightsSource,
    cfg: ZImageTransformerConfig,
    streamable: bool,
    streamed: bool,
) -> Result<ZImageControlTransformer> {
    let base = load_transformer_with_stream_cfg(root, cfg, streamable, streamed)?;
    before_control_load(control, streamed)?;
    let control_weights = match control {
        WeightsSource::File(p) => Weights::from_file(p)?,
        WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    // SC-15754: the ControlNet checkpoint is re-openable too, so the 15-block main control stack gets
    // its own rung-4 stream alongside the base DiT's. The control keys map 1:1 onto the tree (no
    // remap), so a streamed control block reads the same names the resident load did.
    let transformer = ZImageControlTransformer::from_weights(base, &control_weights, "")?;
    materialize_built(&control_weights, streamed.then_some("control_layers."))?;
    let transformer = if streamed {
        transformer.with_deferred_main_blocks()
    } else {
        transformer
    };
    Ok(if streamable {
        transformer.with_control_block_stream(control.clone(), "")
    } else {
        transformer
    })
}

/// iOS: the ControlNet checkpoint's header-only floor of what its load reads (a streamed control
/// stack's block bodies are read inside each block's own priced quantum instead).
pub(crate) fn before_control_load(control: &WeightsSource, streamed: bool) -> Result<()> {
    crate::quantum::before_load(crate::quantum::CONTROL_LOAD, control, |name| {
        control_load_keeps(streamed, name)
    })
}

/// Load the full VAE (decoder + encoder), remapping both diffusers trees to the internal naming
/// and transposing conv weights to NHWC. The encoder powers img2img (`Conditioning::Reference`).
pub fn load_vae(root: &Path) -> Result<Vae> {
    crate::quantum::before_load(
        crate::quantum::VAE_LOAD,
        &WeightsSource::Dir(root.join("vae")),
        |_| true,
    )?;
    let mut w = Weights::from_dir(root.join("vae"))?;
    let vae = build_vae(&mut w)?;
    // Materialize at load (sc-24245; see `materialize_built`).
    w.materialize_accessed()?;
    Ok(vae)
}

/// The ComfyUI path: its source arrays are materialized by the caller (`model::ComfyUiSource`).
pub(crate) fn load_vae_from_weights(mut w: Weights) -> Result<Vae> {
    build_vae(&mut w)
}

pub(crate) fn build_vae(w: &mut Weights) -> Result<Vae> {
    remap_vae_decoder(w)?;
    remap_vae_encoder(w)?;
    build_remapped_vae(w)
}

/// The SAME constructor factory, after remapping. Keeping this boundary explicit lets metadata
/// inventory distinguish remapper reads from module fields without changing native load ordering.
pub(crate) fn build_remapped_vae(w: &Weights) -> Result<Vae> {
    Vae::from_weights(w, "", &VaeDecoderConfig::default_z_image())?.with_encoder(
        w,
        "encoder",
        &VaeEncoderConfig::default_z_image(),
    )
}

pub(crate) const TRANSFORMER_KEY_ALIASES: [(&str, &str); 3] = [
    ("t_embedder.mlp.0", "t_embedder.linear1"),
    ("t_embedder.mlp.2", "t_embedder.linear2"),
    (
        "all_final_layer.2-1.adaLN_modulation.1",
        "all_final_layer.2-1.adaLN_modulation.0",
    ),
];

/// Source-key identity of each header after the executable loader's ordered, per-suffix aliases.
/// A source weight does NOT imply a source bias/scales sibling: a missing source suffix leaves the
/// authored target suffix intact. Both raw spellings remain dictionary entries, but an overwritten
/// target names the source's same Array handle in the module view. This is logical correspondence,
/// not proof that distinct source keys have disjoint native backing or that discarded roots release.
pub(crate) fn transformer_header_origins(
    headers: &std::collections::BTreeMap<
        String,
        mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader,
    >,
) -> std::collections::BTreeMap<String, String> {
    let mut origins = headers.keys().map(|key| (key.clone(), key.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    for (from, to) in TRANSFORMER_KEY_ALIASES {
        for suffix in [".weight", ".bias", ".scales", ".biases"] {
            if let Some(source) = origins.get(&format!("{from}{suffix}")).cloned() {
                origins.insert(format!("{to}{suffix}"), source);
            }
        }
    }
    origins
}

/// The same aliases as the executable loader, applied only to metadata. Aliases are views of one
/// source tensor, not new source payloads; callers must never sum both names as separate owners.
pub(crate) fn remap_transformer_header_keys(
    headers: &mut std::collections::BTreeMap<
        String,
        mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader,
    >,
) {
    for (from, to) in TRANSFORMER_KEY_ALIASES {
        for suffix in [".weight", ".bias", ".scales", ".biases"] {
            if let Some(source) = headers.get(&format!("{from}{suffix}")).cloned() {
                let mut alias = source;
                alias.name = format!("{to}{suffix}");
                headers.insert(alias.name.clone(), alias);
            }
        }
    }
}

/// diffusers DiT checkpoint → internal names: the timestep embedder is `t_embedder.mlp.{0,2}` on
/// disk but `linear{1,2}` internally, and the final layer's adaLN is `Sequential(SiLU, Linear)`
/// (Linear at index 1 on disk, index 0 internally). Everything else matches directly.
pub fn remap_transformer_keys(w: &mut Weights) {
    // Base (suffix-less) renames. Each is aliased across the four tensor suffixes a Linear can carry
    // so a **pre-quantized** snapshot (sc-8670) remaps intact: `.weight`+`.bias` for the dense case,
    // plus the packed `.scales`+`.biases` for the quantized case. `alias` is a no-op when the source
    // key is absent, so a dense snapshot (no `.scales`/`.biases`) and a packed one are both handled
    // by the same list.
    for (from, to) in TRANSFORMER_KEY_ALIASES {
        for suffix in [".weight", ".bias", ".scales", ".biases"] {
            w.alias(&format!("{from}{suffix}"), &format!("{to}{suffix}"));
        }
    }
}

/// Map one diffusers VAE key *suffix* (the part after the `decoder.`/`encoder.` prefix) to its
/// internal target suffix and whether the tensor is an NCHW→NHWC conv weight needing transpose.
/// Pure (no `Weights`), so the remap's regression-prone naming/transpose logic is unit-testable
/// without a checkpoint fixture. `sampler_substr` is the resample-conv path that also carries a
/// conv weight: `.upsamplers.0.conv.` for the decoder up path, `.downsamplers.0.conv.` for the
/// encoder down path.
fn vae_key_target(rest: &str, sampler_substr: &str) -> (String, bool) {
    match rest {
        "conv_in.weight" => ("conv_in.conv.weight".into(), true),
        "conv_in.bias" => ("conv_in.conv.bias".into(), false),
        "conv_out.weight" => ("conv_out.conv.weight".into(), true),
        "conv_out.bias" => ("conv_out.conv.bias".into(), false),
        "conv_norm_out.weight" => ("conv_norm_out.norm.weight".into(), false),
        "conv_norm_out.bias" => ("conv_norm_out.norm.bias".into(), false),
        _ => {
            let is_conv_w = rest.ends_with(".weight")
                && (rest.contains(".conv1.")
                    || rest.contains(".conv2.")
                    || rest.contains(".conv_shortcut.")
                    || rest.contains(sampler_substr));
            (rest.to_string(), is_conv_w)
        }
    }
}

/// Remap a diffusers VAE sub-tree (`{prefix}*`, flat conv names, NCHW conv weights) → the crate's
/// internal `conv.`/`norm.` wrapper naming, conv weights transposed to NHWC `[out,kH,kW,in]`.
/// Inserts the remapped keys alongside the originals. `keep_prefix` controls whether `{prefix}` is
/// retained on the target key — the encoder keeps it, the decoder drops it, and both must coexist in
/// one `Weights`. Shared by [`remap_vae_decoder`]/[`remap_vae_encoder`] (F-037).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VaeRemapStep {
    pub(crate) source: String,
    pub(crate) target: String,
    pub(crate) transpose: bool,
}

/// Metadata and native execution share the exact one-key selector. The phase snapshot is
/// deliberately supplied by the caller: encoder keys do not exist until decoder inserts finish.
pub(crate) fn vae_remap_step(source: &str, encoder: bool) -> Option<VaeRemapStep> {
    let (prefix, sampler) = if encoder {
        ("encoder.", ".downsamplers.0.conv.")
    } else {
        ("decoder.", ".upsamplers.0.conv.")
    };
    let rest = source.strip_prefix(prefix)?;
    let (suffix, transpose) = vae_key_target(rest, sampler);
    Some(VaeRemapStep {
        source: source.to_owned(),
        target: if encoder { format!("{prefix}{suffix}") } else { suffix },
        transpose,
    })
}

fn remap_vae_tree(
    w: &mut Weights,
    prefix: &str,
    keep_prefix: bool,
    sampler_substr: &str,
    trace: Option<&mut Vec<VaeRemapStep>>,
) -> Result<()> {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with(prefix))
        .map(String::from)
        .collect();
    remap_vae_keys(w, prefix, keep_prefix, sampler_substr, keys, trace)
}

/// `keys` is the native HashMap iteration snapshot, NOT sorted. The private seam also allows a
/// fixture to exercise both real overwrite orders without changing the public loader's behavior.
fn remap_vae_keys(
    w: &mut Weights,
    prefix: &str,
    keep_prefix: bool,
    sampler_substr: &str,
    keys: Vec<String>,
    mut trace: Option<&mut Vec<VaeRemapStep>>,
) -> Result<()> {
    for k in keys {
        // The wrappers still supply the original unsorted snapshot and native phase order.
        debug_assert!(k.starts_with(prefix));
        debug_assert_eq!(sampler_substr, if keep_prefix {
            ".downsamplers.0.conv."
        } else {
            ".upsamplers.0.conv."
        });
        let step = vae_remap_step(&k, keep_prefix).expect("phase snapshot key");
        let transpose = step.transpose;
        let target = step.target;
        let t = w.require(&k)?.clone();
        let t = if transpose {
            t.transpose_axes(&[0, 2, 3, 1])? // NCHW -> NHWC conv weight
        } else {
            t
        };
        if let Some(trace) = trace.as_mut() {
            trace.push(VaeRemapStep {
                source: k,
                target: target.clone(),
                transpose,
            });
        }
        w.insert(target, t);
    }
    Ok(())
}

/// diffusers VAE checkpoint (`decoder.*`) → internal decoder naming, conv weights transposed to
/// NHWC. Drops the `decoder.` prefix (the encoder remap keeps its prefix; both coexist in one
/// `Weights`). Up path → `upsamplers`.
pub fn remap_vae_decoder(w: &mut Weights) -> Result<()> {
    remap_vae_tree(w, "decoder.", false, ".upsamplers.0.conv.", None)
}

/// Test-only order seam; the public remapper always snapshots native unsorted `keys()`.
#[cfg(test)]
pub(crate) fn remap_vae_decoder_ordered(w: &mut Weights, keys: Vec<String>) -> Result<()> {
    remap_vae_keys(w, "decoder.", false, ".upsamplers.0.conv.", keys, None)
}

/// diffusers VAE checkpoint (`encoder.*`) → internal encoder naming, conv weights transposed to
/// NHWC. Keeps the `encoder.` prefix. Down path → `downsamplers`.
pub fn remap_vae_encoder(w: &mut Weights) -> Result<()> {
    remap_vae_tree(w, "encoder.", true, ".downsamplers.0.conv.", None)
}

#[cfg(test)]
mod vae_remap_tests {
    use super::vae_key_target;

    const UP: &str = ".upsamplers.0.conv.";
    const DOWN: &str = ".downsamplers.0.conv.";

    #[test]
    fn special_conv_and_norm_keys_map_and_transpose() {
        assert_eq!(
            vae_key_target("conv_in.weight", UP),
            ("conv_in.conv.weight".to_string(), true)
        );
        assert_eq!(
            vae_key_target("conv_in.bias", UP),
            ("conv_in.conv.bias".to_string(), false)
        );
        assert_eq!(
            vae_key_target("conv_out.weight", UP),
            ("conv_out.conv.weight".to_string(), true)
        );
        // conv_norm_out is a norm, never transposed.
        assert_eq!(
            vae_key_target("conv_norm_out.weight", UP),
            ("conv_norm_out.norm.weight".to_string(), false)
        );
    }

    #[test]
    fn resnet_conv_weights_transpose_but_keep_name() {
        let (t, tr) = vae_key_target("mid_block.resnets.1.conv2.weight", UP);
        assert_eq!(t, "mid_block.resnets.1.conv2.weight");
        assert!(tr, "resnet conv2 weight transposes");
        // The matching bias is not a conv weight.
        assert_eq!(
            vae_key_target("mid_block.resnets.1.conv2.bias", UP),
            ("mid_block.resnets.1.conv2.bias".to_string(), false)
        );
        // A norm weight inside a resnet is not a conv → no transpose.
        assert!(!vae_key_target("up_blocks.0.resnets.0.norm1.weight", UP).1);
    }

    #[test]
    fn sampler_substr_is_path_specific() {
        // The up path's upsampler conv counts as a conv weight only under the decoder substr...
        assert!(vae_key_target("up_blocks.0.upsamplers.0.conv.weight", UP).1);
        assert!(!vae_key_target("up_blocks.0.upsamplers.0.conv.weight", DOWN).1);
        // ...and symmetrically for the down path's downsampler conv under the encoder substr.
        assert!(vae_key_target("down_blocks.2.downsamplers.0.conv.weight", DOWN).1);
        assert!(!vae_key_target("down_blocks.2.downsamplers.0.conv.weight", UP).1);
    }
}

/// Test-only fixture shared by the tier-guard tests here and in `model`/`model_control` (F-009,
/// sc-12461): a fresh temp snapshot root whose `transformer/config.json` carries the Group-B packed
/// quantization marker at `bits` — the turnkey shape `needs_load_time_quant` reads — and nothing
/// else (weight-free). `tag` disambiguates the per-test temp dirs. Callers remove the dir.
#[cfg(test)]
pub(crate) fn packed_snapshot_fixture(
    tmp: &tempfile::TempDir,
    tag: &str,
    bits: i32,
) -> std::path::PathBuf {
    let root = tmp.path().join(format!(
        "zimage-packed-{tag}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
    ));
    let td = root.join("transformer");
    std::fs::create_dir_all(&td).unwrap();
    std::fs::write(
        td.join("config.json"),
        format!(r#"{{"quantization": {{"bits": {bits}, "group_size": 64}}}}"#),
    )
    .unwrap();
    root
}

/// F-181: the `Sequential` re-quant warn (and the `needs_load_time_quant` tier guard) must fire only
/// when a load-time (re)quant over a **dense** snapshot will actually happen — an already-packed
/// turnkey loads packed and must NOT warn. Weight-free: writes only a `transformer/config.json`.
#[cfg(test)]
mod quant_tier_tests {
    use mlx_gen::quant::needs_load_time_quant;

    /// Make a fresh temp snapshot root with `transformer/config.json` = `body` (skip the file when
    /// `body` is `None` — a dense snapshot with no quantization marker).
    fn snapshot(tmp: &tempfile::TempDir, body: Option<&str>) -> std::path::PathBuf {
        let root = tmp.path().join(format!(
            "zimage-tier-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let td = root.join("transformer");
        std::fs::create_dir_all(&td).unwrap();
        if let Some(b) = body {
            std::fs::write(td.join("config.json"), b).unwrap();
        }
        root
    }

    #[test]
    fn dense_snapshot_needs_quant_and_warns() {
        let tmp = tempfile::tempdir().unwrap();
        // No config.json at all, and a config with no `quantization` marker, both read as dense.
        for body in [None, Some("{}"), Some(r#"{"in_channels": 16}"#)] {
            let root = snapshot(&tmp, body);
            assert!(
                needs_load_time_quant(&root, "transformer", 4, "z-image").unwrap(),
                "dense snapshot must report a load-time quant (→ warn)"
            );
            std::fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn already_packed_at_requested_bits_does_not_warn() {
        let tmp = tempfile::tempdir().unwrap();
        let root = snapshot(
            &tmp,
            Some(r#"{"quantization": {"bits": 4, "group_size": 64}}"#),
        );
        assert!(
            !needs_load_time_quant(&root, "transformer", 4, "z-image").unwrap(),
            "an already-packed Q4 turnkey must NOT report a load-time quant (no warn)"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn tier_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let root = snapshot(
            &tmp,
            Some(r#"{"quantization": {"bits": 8, "group_size": 64}}"#),
        );
        let err = needs_load_time_quant(&root, "transformer", 4, "z-image").unwrap_err();
        assert!(
            format!("{err}").contains("pre-quantized Q8"),
            "requesting Q4 over a packed Q8 turnkey must error, got: {err}"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}

/// SC-15754 guard: the rung-4 block stream re-reads `layers.*` straight off the snapshot and does
/// **not** replay [`remap_transformer_keys`], because that remap touches only the timestep embedder
/// and the final layer. That is true today and load-bearing — a streamed block built from unremapped
/// keys would silently differ from its resident twin, and the failure would look like a numerical
/// drift rather than a missing remap.
///
/// So this pins the coupling rather than leaving it to the comment: if a future checkpoint needs a
/// `layers.*` alias, this test fails and whoever adds it has to teach the stream about it.
#[cfg(test)]
mod block_stream_remap_guard {
    use super::*;
    use mlx_gen::weights::Weights;
    use mlx_rs::Array;

    #[test]
    fn the_transformer_remap_never_touches_a_layers_key() {
        let mut w = Weights::empty();
        // A representative slice of the real key set: both remap sources, and a trunk block.
        for key in [
            "t_embedder.mlp.0.weight",
            "t_embedder.mlp.2.weight",
            "all_final_layer.2-1.adaLN_modulation.1.weight",
            "layers.0.attention.to_q.weight",
            "layers.29.feed_forward.w1.weight",
            "layers.7.adaLN_modulation.0.weight",
        ] {
            w.insert(key, Array::from_f32(1.0));
        }
        let before: Vec<String> = w
            .keys()
            .filter(|k| k.starts_with("layers."))
            .map(String::from)
            .collect();

        remap_transformer_keys(&mut w);

        let after: Vec<String> = w
            .keys()
            .filter(|k| k.starts_with("layers."))
            .map(String::from)
            .collect();
        let mut before_sorted = before.clone();
        let mut after_sorted = after.clone();
        before_sorted.sort();
        after_sorted.sort();
        assert_eq!(
            before_sorted, after_sorted,
            "remap_transformer_keys added or renamed a `layers.*` key. The rung-4 block stream reads \
             those keys directly off the snapshot and does not replay this remap, so a streamed block \
             would no longer match its resident twin. Teach `ZImageBlockStream` about the new alias."
        );
        // ...and it did do its actual job, so this is not vacuously green on a no-op remap.
        assert!(w.get("t_embedder.linear1.weight").is_some());
        assert!(w
            .get("all_final_layer.2-1.adaLN_modulation.0.weight")
            .is_some());
    }
}
