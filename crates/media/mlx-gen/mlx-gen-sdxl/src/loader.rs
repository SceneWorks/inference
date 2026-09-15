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
fn load_clip_dtype(
    root: &Path,
    subdir: &str,
    cfg: &ClipTextConfig,
    dtype: Dtype,
) -> Result<ClipTextEncoder> {
    let file = resolve_weight_file(root, subdir, "model", dtype)?;
    let mut w = Weights::from_file(&file)?;
    // A packed (pre-quantized) snapshot keeps its on-disk dtypes; only a dense snapshot downcasts.
    if !is_packed(&w) {
        w.cast_all(dtype)?;
    }
    ClipTextEncoder::from_weights(&w, "text_model", cfg)
}

pub(crate) fn load_clip_from_weights(
    mut weights: Weights,
    cfg: &ClipTextConfig,
    dtype: Dtype,
) -> Result<ClipTextEncoder> {
    weights.cast_all(dtype)?;
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
    weights.cast_all(dtype)?;
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
    let file = resolve_weight_file(root, "unet", "diffusion_pytorch_model", dtype)?;
    let mut w = Weights::from_file(&file)?;
    // A packed (pre-quantized) snapshot keeps its on-disk dtypes; only a dense snapshot downcasts.
    if !is_packed(&w) {
        w.cast_all(dtype)?;
    }
    UNet2DConditionModel::from_weights(&w, cfg)
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
    let mut w = match src {
        mlx_gen::WeightsSource::File(p) => Weights::from_file(p)?,
        mlx_gen::WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    // F-082: same packed guard as the U-Net load — casting a pre-quantized checkpoint's packed u32
    // payloads to a float dtype would corrupt them; only a dense checkpoint downcasts.
    if !is_packed(&w) {
        w.cast_all(dtype)?;
    }
    crate::unet::ControlNet::from_weights(&w, &UNetConfig::sdxl_base())
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
    use crate::ip_adapter::{load_ip_kv_pairs, IpImageEncoder, Resampler, ResamplerConfig};
    use crate::vision_encoder::{ClipVisionEncoder, VisionConfig};

    let mut enc_w = Weights::from_file(dir.join(IP_ADAPTER_IMAGE_ENCODER_FILE))?;
    // F-082: packed guard, as in `load_unet_with_config` — never cast pre-quantized payloads.
    if !is_packed(&enc_w) {
        enc_w.cast_all(dtype)?;
    }
    let encoder = ClipVisionEncoder::from_weights(&enc_w, &VisionConfig::vit_h_14())?;

    let ip_file = IP_ADAPTER_WEIGHT_FILES
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| {
            Error::Msg(format!(
                "ip-adapter: no plus/plus-face sdxl_vit-h weights under {}/sdxl_models",
                dir.display()
            ))
        })?;
    let mut ip_w = Weights::from_file(&ip_file)?;
    // F-082: packed guard, as in `load_unet_with_config` — never cast pre-quantized payloads.
    if !is_packed(&ip_w) {
        ip_w.cast_all(dtype)?;
    }
    let resampler =
        Resampler::from_weights(&ip_w, "image_proj", &ResamplerConfig::plus_sdxl_vit_h())?;
    let pairs = load_ip_kv_pairs(&ip_w)?;

    Ok((IpImageEncoder::new(encoder, resampler), pairs))
}

/// Load the SDXL VAE (encoder + decoder). The VAE always runs **f32**, even when the U-Net/TEs are
/// fp16 — the vendored `StableDiffusion.__init__` loads `load_autoencoder(model, float16=False)`
/// unconditionally (the SDXL VAE is fp16-unstable). Prefers the f32 master; if only the fp16 variant
/// is cached it is upcast to f32 (fp16-precision weights — note: not bit-identical to the true f32
/// VAE; fetch `vae/diffusion_pytorch_model.safetensors` for an exact decode).
pub fn load_vae(root: &Path) -> Result<Autoencoder> {
    let file = resolve_vae_weight_file(root)?;
    let mut w = Weights::from_file(&file)?;
    w.cast_all(Dtype::Float32)?;
    Autoencoder::from_weights(&w, &VaeConfig::sdxl_base())
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
