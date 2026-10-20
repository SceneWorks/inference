//! Real-checkpoint loading for SDXL: assemble the components from a
//! `stabilityai/stable-diffusion-xl-base-1.0` snapshot directory (the diffusers multi-component
//! tree). Grows component-by-component as the slices land (tokenizers → text encoders → U-Net →
//! VAE).
//!
//! Snapshot layout:
//! ```text
//!   <root>/tokenizer/{vocab.json,merges.txt}      (+ tokenizer_2/ — byte-identical)
//!   <root>/text_encoder/model.safetensors          CLIP-L (f32)
//!   <root>/text_encoder_2/model.safetensors        OpenCLIP-bigG (f32)
//!   <root>/unet/diffusion_pytorch_model.safetensors
//!   <root>/vae/diffusion_pytorch_model.safetensors
//! ```

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

use crate::config::{ClipTextConfig, UNetConfig, VaeConfig};
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// Load the SDXL CLIP-BPE tokenizer (one instance serves both encoders — `tokenizer/` and
/// `tokenizer_2/` ship byte-identical vocab+merges).
pub fn load_tokenizer(root: &Path) -> Result<ClipBpeTokenizer> {
    ClipBpeTokenizer::from_dir(root.join("tokenizer"))
}

/// `true` when a loaded component is a **pre-quantized** (packed Q4/Q8) snapshot — detected by any
/// `{base}.scales` key (sc-8746). A packed component must NOT be `cast_all`-ed: its `.weight` are u32
/// codes and its `.scales`/`.biases` carry the quantization at a fixed dtype, so a blanket
/// `astype(f16)` would corrupt the codes/scales. The `crate::quant::lin` packed-detect then builds the
/// quantized module directly (no post-load `.quantize`, which no-ops on an already-quantized base).
fn is_packed(w: &Weights) -> bool {
    w.keys().any(|k| k.ends_with(".scales"))
}

/// The load-time downcast for one component at `dtype`, for both snapshot shapes.
///
/// A dense snapshot casts every tensor (the vendored reference's `v.astype(dtype)`). A **packed**
/// snapshot must leave its affine packs alone (a blanket cast corrupts the u32 codes) — but its
/// *dense* leaves (convs, norms, embeddings, the packed linears' own biases) still follow the
/// compute dtype, exactly as they would on a dense load of the same model. Before sc-26268 a packed
/// snapshot skipped the cast entirely, so a producer that wrote its dense leaves at f32 shipped them
/// resident at f32: the published `illustrious-xl-v{1,2}` q4 U-Net carries 1.33 GB of f32 dense
/// leaves (`sdxl` q4: 0.67 GB at f16), which the memory contract already prices at 16-bit — the load
/// now matches the price.
///
/// `pub(crate)` for ladder rung 4: [`crate::block_stream::SdxlBlockStream::open`] applies this same
/// rule to every re-read window, so a streamed block's dense leaves carry the dtype its resident twin
/// was cast to rather than the file's stored width.
///
/// **The packed-leaf cast is iOS-only** (sc-26270, epic requirement E2): on the host a packed
/// snapshot keeps every stored dtype exactly as every release before sc-26268 did, so macOS output
/// bits do not move; on iOS (and the host CPU fixture that exercises the iOS path) the dense leaves
/// follow the compute dtype — the resident bytes the iOS memory contract prices at 16-bit.
pub(crate) fn cast_for_load(w: &mut Weights, dtype: Dtype) -> Result<()> {
    if is_packed(w) {
        if casts_packed_dense_leaves() {
            w.cast_unpacked_floats(dtype)
        } else {
            Ok(())
        }
    } else {
        w.cast_all(dtype)
    }
}

/// Whether [`cast_for_load`] casts a packed snapshot's dense leaves: on iOS (device or simulator),
/// or under the host CPU fixture that runs the iOS quantum path. Never on a host render.
pub(crate) fn casts_packed_dense_leaves() -> bool {
    cfg!(target_os = "ios") || mlx_gen::ios_quantum::enabled()
}

/// On the iOS path ([`casts_packed_dense_leaves`]), the handles of a load's weights as read, taken
/// BEFORE its load-time cast and dropped only once the load has materialized (epic 16774): while a
/// source is still referenced its cast cannot be donated the source's buffer, so every cast copy
/// ends resident at exactly its own (narrower) width — not at a wider source buffer it inherited —
/// and each materialize batch's peak is exactly its read sources plus their copies, the floor the
/// guard prices (`quantum::cast_load_parts`). Measured: an f32 VAE cast to f16 otherwise stays
/// resident 44% above its f16 bytes. A handle copy, never a tensor copy; `None` off the iOS path,
/// where every load is byte-for-byte what it was.
fn hold_cast_sources(w: &Weights) -> Option<Weights> {
    casts_packed_dense_leaves().then(|| w.clone())
}

/// iOS pre-load floor (sc-26270) for a component `files` that [`cast_for_load`] casts to `dtype`
/// after reading: every tensor's resident bytes after the cast (its `dtype` copy exactly where
/// `cast_for_load` casts it — all differing tensors of a dense snapshot; on a packed one the dense
/// float leaves, never the codes or their `.scales`/`.biases`) plus the largest tensor's transient
/// (source and copy, or one read). Header-only; a no-op off device.
fn before_cast_for_load(label: &str, files: &[PathBuf], dtype: Dtype) -> Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    mlx_gen::ios_quantum::before(
        label,
        cast_load_floor_bytes(files, dtype, casts_packed_dense_leaves(), |_| true)?,
    )
}

/// The header-only floor [`before_cast_for_load_where`] checks: the kept tensors' resident bytes
/// after [`cast_for_load`]'s cast to `dtype` plus the largest kept tensor's transient.
/// `packed_leaves` is [`casts_packed_dense_leaves`] (always `true` on the device, which header-only
/// pricing passes from any host).
pub(crate) fn cast_load_floor_bytes(
    files: &[PathBuf],
    dtype: Dtype,
    packed_leaves: bool,
    keep: impl Fn(&str) -> bool,
) -> Result<u64> {
    Ok(cast_load_parts(files, dtype, packed_leaves, keep)?.1)
}

/// `(resident bytes after the load, the load's floor)` ([`crate::quantum::cast_load_parts`]) under
/// [`cast_for_load`]'s rule at `dtype`.
pub(crate) fn cast_load_parts(
    files: &[PathBuf],
    dtype: Dtype,
    packed_leaves: bool,
    keep: impl Fn(&str) -> bool,
) -> Result<(u64, u64)> {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let target = match dtype {
        Dtype::Float16 => Stored::F16,
        Dtype::Bfloat16 => Stored::BF16,
        Dtype::Float32 => Stored::F32,
        other => {
            return Err(Error::Msg(format!(
                "sdxl load: unsupported compute dtype {other:?}"
            )))
        }
    };
    let width = match target {
        Stored::F32 => 4,
        _ => 2,
    };
    let mut packed = false;
    for file in files {
        packed |= mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
            .map_err(|e| Error::Msg(e.to_string()))?
            .iter()
            .any(|h| h.name.ends_with(".scales"));
    }
    crate::quantum::cast_load_parts(
        files,
        width,
        |name, stored| {
            if stored == target {
                return false;
            }
            if !packed {
                return true;
            }
            let floating = matches!(
                stored,
                Stored::F16 | Stored::BF16 | Stored::F32 | Stored::F64
            );
            let companion = name.ends_with(".scales") || name.ends_with(".biases");
            packed_leaves && floating && !companion
        },
        keep,
    )
}

/// The host index arrays `quant::lin_geglu_half` builds for every `TransformerBlock` whose
/// `ff.net.0.proj` `keep(name)` selects (two `[hidden]` i32 row selections per block, one
/// allocation each). A constructed-but-unread (rung-4 streamed) block holds them for as long as it
/// lives; a materialized block releases them once its halves are evaluated. Header-only.
pub(crate) fn geglu_index_bytes(files: &[PathBuf], keep: impl Fn(&str) -> bool) -> Result<u64> {
    let mut bytes = 0u64;
    for file in files {
        for header in mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
            .map_err(|e| Error::Msg(e.to_string()))?
        {
            let proj = header.name.ends_with(".ff.net.0.proj.scales")
                || (header.name.ends_with(".ff.net.0.proj.weight")
                    && header.dtype != mlx_gen::gen_core::weightsmeta::Dtype::U32);
            if !proj || !keep(&header.name) {
                continue;
            }
            let half = header.shape.first().copied().unwrap_or(0) as u64 / 2;
            bytes = bytes
                .saturating_add(2 * mlx_gen::ios_quantum::allocated_bytes(half.saturating_mul(4)));
        }
    }
    Ok(bytes)
}

/// The U-Net load's `(resident bytes, floor)` at `dtype`, header-only: [`cast_load_parts`] over
/// the tensors the load materializes (all, or — `streamed` — the resident remainder; the streamed
/// block bodies are read by their own quanta), plus the GEGLU index arrays every constructed block
/// builds before the materialization (which the constructed-but-unread streamed blocks keep).
pub(crate) fn unet_load_parts(
    files: &[PathBuf],
    dtype: Dtype,
    packed_leaves: bool,
    streamed: bool,
    cfg: &UNetConfig,
) -> Result<(u64, u64)> {
    let (resident, floor) = cast_load_parts(files, dtype, packed_leaves, |name| {
        !(streamed && is_streamed_block_tensor(name))
    })?;
    let index = geglu_index_bytes(files, |_| true)?;
    // The timestep and `text_time` sinusoid tables the constructor builds on the host.
    use crate::unet::SinusoidalPositionalEncoding as Sinusoid;
    let tables = Sinusoid::timestep_table_bytes(cfg.block_out_channels[0])
        + Sinusoid::timestep_table_bytes(cfg.addition_time_embed_dim.unwrap_or(256));
    Ok((
        resident
            .saturating_add(if streamed { index } else { 0 })
            .saturating_add(tables),
        floor.saturating_add(index).saturating_add(tables),
    ))
}

/// Whether a U-Net tensor is a rung-4 streamed `TransformerBlock` body: a streamed load leaves it
/// lazy (the block stream reads it per window), so its load floor and its resident bytes omit it.
pub(crate) fn is_streamed_block_tensor(name: &str) -> bool {
    name.contains(".transformer_blocks.")
}

/// Refuse a trainer load whose `components` include a **pre-quantized** (packed Q4/Q8) tier.
///
/// Training reads the dense base — the contract the `mage_flow_base` trainer already enforces, the
/// engine-side twin of the app's `TrainingTierMissing` pre-flight. An SDXL-family packed tier would
/// not fail on its own: the gradient path runs over the packed codes, and the load-time cast upcasts
/// the snapshot's f16 dense leaves to the trainer's f32, so it would train against the q4/q8
/// approximation of the base rather than the base. `trainer` names the caller in the refusal.
pub fn require_dense_training_base(root: &Path, trainer: &str, components: &[&str]) -> Result<()> {
    for component in components {
        if let Some(bits) = mlx_gen::quant::packed_quant_bits(root, component)? {
            return Err(Error::Msg(format!(
                "{trainer} trainer requires the dense base, but {component} at {} is a \
                 pre-quantized Q{bits} artifact; install the dense tier to train",
                root.join(component).display()
            )));
        }
    }
    Ok(())
}

/// Resolve a component's weight file inside `subdir`, picking the variant that best matches `dtype`.
///
/// `pub(crate)` for ladder rung 4 (SC-15525): `model::load_heavy` records the U-Net file this
/// resolves to as the block stream's re-openable source, so a streamed block reads the **same file**
/// the resident stack was built from rather than re-deriving the fp16/f32 variant rule.
/// diffusers snapshots ship the f32 master (`<stem>.safetensors`) and/or an fp16 variant
/// (`<stem>.fp16.safetensors`); the fp16 file is exactly `astype(f16)` of the f32 master, so for an
/// f16 load the two are equivalent. We prefer the variant matching `dtype` (fp16 file for f16, the
/// f32 file otherwise) and fall back to the other when only one is cached — the caller casts to
/// `dtype` regardless, so the result is identical when both exist.
pub(crate) fn resolve_weight_file(
    root: &Path,
    subdir: &str,
    stem: &str,
    dtype: Dtype,
) -> Result<PathBuf> {
    let plain = root.join(subdir).join(format!("{stem}.safetensors"));
    let fp16 = root.join(subdir).join(format!("{stem}.fp16.safetensors"));
    let (first, second) = if dtype == Dtype::Float16 {
        (&fp16, &plain)
    } else {
        (&plain, &fp16)
    };
    if first.exists() {
        Ok(first.clone())
    } else if second.exists() {
        Ok(second.clone())
    } else {
        Err(Error::Msg(format!(
            "sdxl: missing {subdir}/{stem}.safetensors (and no .fp16 variant)"
        )))
    }
}

/// Load one CLIP text encoder from a component subdir (`text_encoder` or `text_encoder_2`) at a
/// given compute dtype. Reads the best-matching `model{,.fp16}.safetensors` and casts every tensor to
/// `dtype` — the vendored reference loads the f32 master and applies `v.astype(dtype)`, so f16 here
/// byte-matches the production `StableDiffusionXL(float16=True)` text encoder.
pub(crate) fn load_clip_dtype(
    root: &Path,
    subdir: &str,
    cfg: &ClipTextConfig,
    dtype: Dtype,
) -> Result<ClipTextEncoder> {
    let file = resolve_weight_file(root, subdir, "model", dtype)?;
    // sc-26270: iOS header-only pre-load floor before this encoder is constructed, including the
    // load-time cast's copies.
    let label = match subdir {
        "text_encoder" => crate::quantum::TEXT_ENCODER_1_LOAD,
        "text_encoder_2" => crate::quantum::TEXT_ENCODER_2_LOAD,
        other => {
            return Err(Error::Msg(format!(
                "sdxl load: unknown text encoder component {other}"
            )))
        }
    };
    before_cast_for_load(label, std::slice::from_ref(&file), dtype)?;
    let mut w = Weights::from_file(&file)?;
    let sources = hold_cast_sources(&w);
    // Packs keep their on-disk dtypes; every dense leaf follows `dtype` (sc-26268).
    cast_for_load(&mut w, dtype)?;
    let encoder = ClipTextEncoder::from_weights(&w, "text_model", cfg)?;
    // Materialize at load (after the cast, before any `quantize`): left lazy, the first forward's
    // command buffers wait on the safetensors reads — past the GPU watchdog on a cold page cache
    // (sc-24245; see `mlx_gen_qwen_image::loader::load_transformer_with`).
    w.materialize_accessed()?;
    drop(sources);
    Ok(encoder)
}

pub(crate) fn load_clip_from_weights(
    mut weights: Weights,
    cfg: &ClipTextConfig,
    dtype: Dtype,
) -> Result<ClipTextEncoder> {
    cast_for_load(&mut weights, dtype)?;
    ClipTextEncoder::from_weights(&weights, "text_model", cfg)
}

pub(crate) fn load_text_encoder_1_from_weights(
    weights: Weights,
    dtype: Dtype,
) -> Result<ClipTextEncoder> {
    load_clip_from_weights(weights, &ClipTextConfig::sdxl_te1(), dtype)
}

pub(crate) fn load_text_encoder_2_from_weights(
    weights: Weights,
    dtype: Dtype,
) -> Result<ClipTextEncoder> {
    load_clip_from_weights(weights, &ClipTextConfig::sdxl_te2(), dtype)
}

pub(crate) fn load_unet_from_weights(
    mut weights: Weights,
    dtype: Dtype,
) -> Result<UNet2DConditionModel> {
    cast_for_load(&mut weights, dtype)?;
    UNet2DConditionModel::from_weights(&weights, &UNetConfig::sdxl_base())
}

pub(crate) fn load_vae_from_weights(mut weights: Weights) -> Result<Autoencoder> {
    weights.cast_all(Dtype::Float32)?;
    Autoencoder::from_weights(&weights, &VaeConfig::sdxl_base())
}

/// Load CLIP-L (`text_encoder`) — the 768-wide encoder, no projection — at `dtype`.
pub fn load_text_encoder_1_dtype(root: &Path, dtype: Dtype) -> Result<ClipTextEncoder> {
    load_clip_dtype(root, "text_encoder", &ClipTextConfig::sdxl_te1(), dtype)
}

/// Load OpenCLIP-bigG (`text_encoder_2`) — the 1280-wide encoder with the pooled projection — at
/// `dtype`.
pub fn load_text_encoder_2_dtype(root: &Path, dtype: Dtype) -> Result<ClipTextEncoder> {
    load_clip_dtype(root, "text_encoder_2", &ClipTextConfig::sdxl_te2(), dtype)
}

/// f32 CLIP-L — the tight-stage-gate path (validated against the `float16=False` golden).
pub fn load_text_encoder_1(root: &Path) -> Result<ClipTextEncoder> {
    load_text_encoder_1_dtype(root, Dtype::Float32)
}

/// f32 OpenCLIP-bigG — the tight-stage-gate path.
pub fn load_text_encoder_2(root: &Path) -> Result<ClipTextEncoder> {
    load_text_encoder_2_dtype(root, Dtype::Float32)
}

/// Load the SDXL U-Net at `dtype` from `unet/diffusion_pytorch_model{,.fp16}.safetensors`. The chosen
/// file is cast to `dtype` (f16 byte-matches the production `float16=True` U-Net).
pub fn load_unet_dtype(root: &Path, dtype: Dtype) -> Result<UNet2DConditionModel> {
    load_unet_with_config(root, dtype, &UNetConfig::sdxl_base())
}

/// Load the U-Net at `dtype` with an explicit [`UNetConfig`] — the shared body of
/// [`load_unet_dtype`] (SDXL) and the Kolors loader. The `encoder_hid_proj` (Kolors) is auto-detected
/// from the weights, so the same file-resolution + cast path serves both.
pub fn load_unet_with_config(
    root: &Path,
    dtype: Dtype,
    cfg: &UNetConfig,
) -> Result<UNet2DConditionModel> {
    load_unet_with_config_streamed(root, dtype, cfg, false)
}

/// [`load_unet_with_config`] for a load that will arm ladder rung 4
/// ([`UNet2DConditionModel::arm_block_streams`]) when `streamed` is set.
///
/// Materializes at load (sc-24245): left lazy, the first denoise step's command buffers wait on the
/// safetensors reads — past the GPU watchdog on a cold page cache (see
/// `mlx_gen_qwen_image::loader::load_transformer_with`). `streamed` leaves every
/// `*.transformer_blocks.*` body lazy: the block stream re-opens and materializes each block itself,
/// so reading them here would defeat bounded residency.
pub fn load_unet_with_config_streamed(
    root: &Path,
    dtype: Dtype,
    cfg: &UNetConfig,
    streamed: bool,
) -> Result<UNet2DConditionModel> {
    let file = resolve_weight_file(root, "unet", "diffusion_pytorch_model", dtype)?;
    // sc-26270: iOS header-only pre-load floor before the U-Net is constructed.
    // The load-time cast's copies are priced with the stored tensors they replace.
    // A streamed load materializes only the resident remainder: the transformer-block bodies are
    // read per window by the block stream's own priced quanta.
    if mlx_gen::ios_quantum::enabled() {
        let (_, floor) = unet_load_parts(
            std::slice::from_ref(&file),
            dtype,
            casts_packed_dense_leaves(),
            streamed,
            cfg,
        )?;
        mlx_gen::ios_quantum::before(crate::quantum::UNET_LOAD, floor)?;
    }
    let mut w = Weights::from_file(&file)?;
    let sources = hold_cast_sources(&w);
    // Packs keep their on-disk dtypes; every dense leaf follows `dtype` (sc-26268).
    cast_for_load(&mut w, dtype)?;
    let unet = UNet2DConditionModel::from_weights(&w, cfg)?;
    if streamed {
        let resident: Vec<(String, mlx_rs::Array)> = w
            .accessed_entries()
            .into_iter()
            .filter(|(key, _)| !is_streamed_block_tensor(key))
            .collect();
        let mut named: Vec<(&str, &mlx_rs::Array)> =
            resident.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Weights::materialize_named(&mut named)?;
    } else {
        w.materialize_accessed()?;
    }
    drop(sources);
    Ok(unet)
}

/// f32 U-Net — the tight-stage-gate path (validated against the `float16=False` golden).
pub fn load_unet(root: &Path) -> Result<UNet2DConditionModel> {
    load_unet_dtype(root, Dtype::Float32)
}

/// Load the **Kolors** U-Net (epic 3090) at `dtype` — [`UNetConfig::kolors`] + the auto-detected
/// `encoder_hid_proj`. `root` is the `Kwai-Kolors/Kolors-diffusers` snapshot.
pub fn load_unet_kolors_dtype(root: &Path, dtype: Dtype) -> Result<UNet2DConditionModel> {
    load_unet_with_config(root, dtype, &UNetConfig::kolors())
}

/// The **exact** U-Net weight file [`load_unet_with_config`] would read out of `root` at `dtype` —
/// the re-openable source ladder rung 4 records when it arms a block stream.
///
/// `pub` for `mlx-gen-kolors` (SC-15521): Kolors re-exports this crate's [`UNet2DConditionModel`]
/// verbatim but registers its own provider and arms its own streams, so it needs the same
/// fp16/f32-variant resolution the resident load performed rather than a second derivation of the
/// rule. Getting a *different* file here would silently stream blocks from a different snapshot
/// variant than the resident stack was built from.
pub fn resolve_unet_weight_file(root: &Path, dtype: Dtype) -> Result<PathBuf> {
    resolve_weight_file(root, "unet", "diffusion_pytorch_model", dtype)
}

/// Load an SDXL **ControlNet** branch (sc-3058) from a diffusers `ControlNetModel` checkpoint — a
/// single `.safetensors` file or a directory containing `diffusion_pytorch_model.safetensors`. Cast
/// to `dtype` (fp16 in production, matching the U-Net it injects into).
pub fn load_controlnet(
    src: &mlx_gen::WeightsSource,
    dtype: Dtype,
) -> Result<crate::unet::ControlNet> {
    load_controlnet_with(src, dtype, &UNetConfig::sdxl_base())
}

/// [`load_controlnet`] over an explicit encoder-copy config (a tiny proof fixture's).
pub(crate) fn load_controlnet_with(
    src: &mlx_gen::WeightsSource,
    dtype: Dtype,
    cfg: &UNetConfig,
) -> Result<crate::unet::ControlNet> {
    // sc-26270: iOS header-only pre-load floor before the branch is constructed (a directory
    // source counts every safetensors file it holds, as `Weights::from_dir` reads them all).
    let source_path = match src {
        mlx_gen::WeightsSource::File(p) | mlx_gen::WeightsSource::Dir(p) => p.clone(),
    };
    before_cast_for_load(crate::quantum::CONTROLNET_LOAD, &[source_path], dtype)?;
    let mut w = match src {
        mlx_gen::WeightsSource::File(p) => Weights::from_file(p)?,
        mlx_gen::WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    let sources = hold_cast_sources(&w);
    // F-082 / sc-26268: packs keep their stored dtypes; every dense leaf follows `dtype`.
    cast_for_load(&mut w, dtype)?;
    let control = crate::unet::ControlNet::from_weights(&w, cfg)?;
    // Materialize at load, before any `quantize` (sc-24245; see [`load_unet_with_config_streamed`]).
    w.materialize_accessed()?;
    drop(sources);
    Ok(control)
}

/// The ViT-H image encoder inside an `h94/IP-Adapter`-layout snapshot, relative to its root.
pub(crate) const IP_ADAPTER_IMAGE_ENCODER_FILE: &str = "models/image_encoder/model.safetensors";

/// The IP weights [`load_ip_adapter`] opens, in preference order: plus-face first, plus as the
/// fallback (they share the Resampler architecture). Shared with the memory contract (SC-22667)
/// so the overlay it prices is the file this loader materializes and never a third candidate.
pub(crate) const IP_ADAPTER_WEIGHT_FILES: [&str; 2] = [
    "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
    "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
];

/// The actual configured IP source selector, shared with header-only semantic normalization.
/// Returning paths is local loader plumbing; the exported semantic facts never serialize them.
pub(crate) fn resolve_ip_adapter_files(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let weights = IP_ADAPTER_WEIGHT_FILES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists())
        .ok_or_else(|| {
            Error::Msg(format!(
                "ip-adapter: no plus/plus-face sdxl_vit-h weights under {}/sdxl_models",
                dir.display(),
            ))
        })?;
    Ok((dir.join(IP_ADAPTER_IMAGE_ENCODER_FILE), weights))
}

/// Load the **IP-Adapter** (sc-3059) from an `h94/IP-Adapter`-layout snapshot directory: the ViT-H
/// image encoder at `models/image_encoder/model.safetensors` and the IP weights (Resampler +
/// decoupled-attn K/V pairs) at `sdxl_models/ip-adapter-plus[-face]_sdxl_vit-h.safetensors`
/// (plus-face preferred, plus as fallback — they share the Resampler architecture). Returns the
/// image-token encoder + the per-cross-attn K/V pairs to install into the U-Net. Cast to `dtype`.
pub fn load_ip_adapter(
    dir: &Path,
    dtype: Dtype,
) -> Result<(
    crate::ip_adapter::IpImageEncoder,
    Vec<(mlx_rs::Array, mlx_rs::Array)>,
)> {
    load_ip_adapter_with(
        dir,
        dtype,
        &crate::vision_encoder::VisionConfig::vit_h_14(),
        &crate::ip_adapter::ResamplerConfig::plus_sdxl_vit_h(),
    )
}

/// [`load_ip_adapter`] over explicit tower / Resampler configs (a tiny proof fixture's).
pub(crate) fn load_ip_adapter_with(
    dir: &Path,
    dtype: Dtype,
    vision: &crate::vision_encoder::VisionConfig,
    resampler_cfg: &crate::ip_adapter::ResamplerConfig,
) -> Result<(
    crate::ip_adapter::IpImageEncoder,
    Vec<(mlx_rs::Array, mlx_rs::Array)>,
)> {
    use crate::ip_adapter::{load_ip_kv_pairs, IpImageEncoder, Resampler};
    use crate::vision_encoder::ClipVisionEncoder;

    // sc-26270: iOS header-only pre-load floors before each IP component is constructed.
    before_cast_for_load(
        crate::quantum::IP_IMAGE_ENCODER_LOAD,
        &[dir.join(IP_ADAPTER_IMAGE_ENCODER_FILE)],
        dtype,
    )?;
    let mut enc_w = Weights::from_file(dir.join(IP_ADAPTER_IMAGE_ENCODER_FILE))?;
    let enc_sources = hold_cast_sources(&enc_w);
    // F-082 / sc-26268: packs keep their stored dtypes; every dense leaf follows `dtype`.
    cast_for_load(&mut enc_w, dtype)?;
    let encoder = ClipVisionEncoder::from_weights(&enc_w, vision)?;
    // Materialize at load (sc-24245; see [`load_unet_with_config_streamed`]).
    enc_w.materialize_accessed()?;
    drop(enc_sources);

    let (_, ip_file) = resolve_ip_adapter_files(dir)?;
    before_cast_for_load(
        crate::quantum::IP_ADAPTER_LOAD,
        std::slice::from_ref(&ip_file),
        dtype,
    )?;
    let mut ip_w = Weights::from_file(&ip_file)?;
    let ip_sources = hold_cast_sources(&ip_w);
    // F-082 / sc-26268: packs keep their stored dtypes; every dense leaf follows `dtype`.
    cast_for_load(&mut ip_w, dtype)?;
    let resampler = Resampler::from_weights(&ip_w, "image_proj", resampler_cfg)?;
    let pairs = load_ip_kv_pairs(&ip_w)?;
    // Materialize at load, before the pairs are installed and quantized with the U-Net (sc-24245).
    ip_w.materialize_accessed()?;
    drop(ip_sources);

    Ok((
        IpImageEncoder::with_image_size(encoder, resampler, vision.image_size as usize),
        pairs,
    ))
}

/// Load the SDXL VAE (encoder + decoder). The VAE always runs **f32**, even when the U-Net/TEs are
/// fp16 — the vendored `StableDiffusion.__init__` loads `load_autoencoder(model, float16=False)`
/// unconditionally (the SDXL VAE is fp16-unstable). Prefers the f32 master; if only the fp16 variant
/// is cached it is upcast to f32 (fp16-precision weights — note: not bit-identical to the true f32
/// VAE; fetch `vae/diffusion_pytorch_model.safetensors` for an exact decode).
pub fn load_vae(root: &Path) -> Result<Autoencoder> {
    let file = resolve_vae_weight_file(root)?;
    // sc-26270: iOS header-only pre-load floor before the VAE is constructed, with the f32 upcast's
    // copies of every non-f32 tensor (a no-op cast of the f32 master costs nothing).
    before_cast_for_load(
        crate::quantum::VAE_LOAD,
        std::slice::from_ref(&file),
        Dtype::Float32,
    )?;
    let mut w = Weights::from_file(&file)?;
    let sources = hold_cast_sources(&w);
    w.cast_all(Dtype::Float32)?;
    let vae = Autoencoder::from_weights(&w, &VaeConfig::sdxl_base())?;
    // Materialize at load (sc-24245; see [`load_unet_with_config_streamed`]).
    w.materialize_accessed()?;
    drop(sources);
    Ok(vae)
}

/// The **exact** VAE weight file [`load_vae`] would read out of `root`.
///
/// `pub` for the same reason [`resolve_unet_weight_file`] is: a memory contract that prices the
/// decoder must size the file the resident load actually opens, not a directory sum. It matters
/// more here than for the U-Net, because [`load_vae`] `cast_all`s to **f32 unconditionally** while
/// every SceneWorks SDXL-family tier ships only `diffusion_pytorch_model.fp16.safetensors` — so a
/// decoder footprint taken from stored bytes is underpriced by exactly 2x, at every tier
/// (sc-15839). Pair this with `mlx_gen::asset_facts::ResidentProjection::Float32`.
pub fn resolve_vae_weight_file(root: &Path) -> Result<PathBuf> {
    resolve_weight_file(root, "vae", "diffusion_pytorch_model", Dtype::Float32)
}

/// The caller-staged component id of the **fp16-fix VAE** (`madebyollin/sdxl-vae-fp16-fix`) — the
/// same id the Candle SDXL engine requires and SceneWorks already provisions for it
/// (`LoadSpec::components` registry, epic 13657).
///
/// On MLX the component is **optional** (sc-26268). Absent, the VAE is the in-snapshot one at f32,
/// byte-identical to every earlier release. Present, the VAE is that checkpoint at **f16** — the
/// fp16-fix finetune keeps the decoder's activations inside the f16 range, which is the whole reason
/// it exists (the stock SDXL VAE overflows at f16, which is why the in-snapshot path runs f32). It
/// halves the VAE's resident weights and, more importantly, the decode's full-resolution activation
/// working set — the low-memory path an 8 GB device needs.
pub const VAE_FP16_FIX_COMPONENT: &str = "vae_fp16_fix";

/// The f16 compute dtype the fp16-fix VAE runs at.
pub const VAE_FP16_FIX_DTYPE: Dtype = Dtype::Float16;

/// The exact checkpoint file a staged fp16-fix VAE component resolves to: a `File` is used as-is; a
/// `Dir` (the `madebyollin/sdxl-vae-fp16-fix` snapshot) resolves `diffusion_pytorch_model.safetensors`,
/// then its `.fp16` variant — the same rule the Candle loader applies to the same component.
pub fn resolve_vae_fp16_fix_file(source: &mlx_gen::WeightsSource) -> Result<PathBuf> {
    match source {
        mlx_gen::WeightsSource::File(file) => Ok(file.clone()),
        mlx_gen::WeightsSource::Dir(dir) => [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.fp16.safetensors",
        ]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| {
            Error::Msg(format!(
                "sdxl: staged `{VAE_FP16_FIX_COMPONENT}` component has no \
                         diffusion_pytorch_model(.fp16).safetensors in {}",
                dir.display()
            ))
        }),
    }
}

/// Load the staged fp16-fix VAE at [`VAE_FP16_FIX_DTYPE`].
pub fn load_vae_fp16_fix(source: &mlx_gen::WeightsSource) -> Result<Autoencoder> {
    load_vae_fp16_fix_with(source, &VaeConfig::sdxl_base())
}

/// [`load_vae_fp16_fix`] over an explicit config (a tiny proof fixture's).
pub(crate) fn load_vae_fp16_fix_with(
    source: &mlx_gen::WeightsSource,
    cfg: &VaeConfig,
) -> Result<Autoencoder> {
    let file = resolve_vae_fp16_fix_file(source)?;
    // sc-26270: iOS header-only pre-load floor before the fp16-fix VAE is constructed, with the
    // cast copies of every tensor not already at VAE_FP16_FIX_DTYPE.
    before_cast_for_load(
        crate::quantum::VAE_FP16_FIX_LOAD,
        std::slice::from_ref(&file),
        VAE_FP16_FIX_DTYPE,
    )?;
    let mut w = Weights::from_file(file)?;
    let sources = hold_cast_sources(&w);
    w.cast_all(VAE_FP16_FIX_DTYPE)?;
    let vae = Autoencoder::from_weights(&w, cfg)?;
    // Materialize at load (sc-24245; see [`load_unet_with_config_streamed`]).
    w.materialize_accessed()?;
    drop(sources);
    Ok(vae)
}

/// The VAE a load runs: the staged fp16-fix component at f16 when the spec carries one, else the
/// in-snapshot VAE at f32 (the historical path, byte-identical).
pub(crate) fn load_vae_for_spec(spec: &mlx_gen::LoadSpec, root: &Path) -> Result<Autoencoder> {
    match spec.components.get(VAE_FP16_FIX_COMPONENT) {
        Some(source) => load_vae_fp16_fix(source),
        None => load_vae(root),
    }
}

/// The weights of `files` as the iOS load casts them, LAZILY: the same `Weights::from_file`
/// handles and the device's [`cast_for_load`] rule (a packed snapshot's dense float leaves follow
/// `dtype` on iOS, whatever host prices them; `cast_all` when `all`), with nothing read or
/// evaluated. Header-only pricing (`crate::ios_pricing`) builds its components from these.
pub(crate) fn lazy_weights(files: &[PathBuf], dtype: Dtype, all: bool) -> Result<Weights> {
    let mut w = Weights::empty();
    for file in files {
        let mut part = if file.is_dir() {
            Weights::from_dir(file)?
        } else {
            Weights::from_file(file)?
        };
        if all || !is_packed(&part) {
            part.cast_all(dtype)?;
        } else {
            part.cast_unpacked_floats(dtype)?;
        }
        for key in part.keys().map(str::to_owned).collect::<Vec<_>>() {
            if let Some(array) = part.get(&key) {
                w.insert(key, array.clone());
            }
        }
    }
    Ok(w)
}

/// sc-26268: the staged fp16-fix VAE component resolves exactly like the Candle loader resolves
/// the same component — a file as-is, a snapshot directory's master then its `.fp16` variant — and
/// a directory holding neither is a load error naming the component.
#[cfg(test)]
mod vae_fp16_fix_tests {
    use super::*;
    use mlx_gen::WeightsSource;

    #[test]
    fn the_component_resolves_like_the_candle_loader() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("fix");
        std::fs::create_dir_all(&dir).unwrap();
        let missing = resolve_vae_fp16_fix_file(&WeightsSource::Dir(dir.clone())).unwrap_err();
        assert!(
            missing.to_string().contains(VAE_FP16_FIX_COMPONENT),
            "{missing}"
        );

        let fp16 = dir.join("diffusion_pytorch_model.fp16.safetensors");
        std::fs::write(&fp16, b"").unwrap();
        assert_eq!(
            resolve_vae_fp16_fix_file(&WeightsSource::Dir(dir.clone())).unwrap(),
            fp16
        );
        let master = dir.join("diffusion_pytorch_model.safetensors");
        std::fs::write(&master, b"").unwrap();
        assert_eq!(
            resolve_vae_fp16_fix_file(&WeightsSource::Dir(dir.clone())).unwrap(),
            master,
            "the master wins when both are staged"
        );
        assert_eq!(
            resolve_vae_fp16_fix_file(&WeightsSource::File(fp16.clone())).unwrap(),
            fp16
        );
    }
}

/// F-181: the `Sequential` re-quant warn (and the `needs_load_time_quant` tier guard) must fire only
/// when a load-time (re)quant over a **dense** snapshot will actually happen — an already-packed
/// turnkey loads packed and must NOT warn. Weight-free: writes only a `unet/config.json`.
#[cfg(test)]
mod quant_tier_tests {
    use mlx_gen::quant::needs_load_time_quant;

    /// Make a fresh temp snapshot root with `unet/config.json` = `body` (skip the file when `body` is
    /// `None` — a dense snapshot with no quantization marker).
    fn snapshot(tmp: &tempfile::TempDir, body: Option<&str>) -> std::path::PathBuf {
        let root = tmp.path().join(format!(
            "sdxl-tier-{:?}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let ud = root.join("unet");
        std::fs::create_dir_all(&ud).unwrap();
        if let Some(b) = body {
            std::fs::write(ud.join("config.json"), b).unwrap();
        }
        root
    }

    #[test]
    fn dense_snapshot_needs_quant_and_warns() {
        let tmp = tempfile::tempdir().unwrap();
        // No config.json at all, and a config with no `quantization` marker, both read as dense.
        for body in [None, Some("{}"), Some(r#"{"in_channels": 4}"#)] {
            let root = snapshot(&tmp, body);
            assert!(
                needs_load_time_quant(&root, "unet", 4, "sdxl").unwrap(),
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
            Some(r#"{"quantization": {"bits": 8, "group_size": 64}}"#),
        );
        assert!(
            !needs_load_time_quant(&root, "unet", 8, "sdxl").unwrap(),
            "an already-packed Q8 turnkey must NOT report a load-time quant (no warn)"
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
        let err = needs_load_time_quant(&root, "unet", 4, "sdxl").unwrap_err();
        assert!(
            format!("{err}").contains("pre-quantized Q8"),
            "requesting Q4 over a packed Q8 turnkey must error, got: {err}"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
