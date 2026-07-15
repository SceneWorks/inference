//! SDXL component loaders for the InstantID provider (sc-5491, epic 5480) — the candle twins of
//! `mlx-gen-sdxl`'s `load_unet_dtype` / `load_vae` / `load_controlnet`. The txt2img `crate::pipeline`
//! loads the **stock** candle-transformers UNet internally; InstantID needs the **vendored** UNet (the
//! one carrying the `add_embedding` micro-conditioning + the decoupled IP-Adapter cross-attention from
//! phase 2c), so these build that stack from an SDXL snapshot + a diffusers ControlNet checkpoint.
//!
//! The IP-Adapter K/V install is NOT done here — the caller (the `candle-gen-instantid` glue) loads the
//! converted `ip-adapter.safetensors` (`image_proj.*` Resampler + `ip_adapter.*` pairs) and calls
//! [`UNet2DConditionModel::install_ip_adapter`] on the returned UNet, mirroring `InstantId::load`.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;

use candle_gen::gen_core::{AdapterSpec, WeightsSource};
use candle_gen::{CandleError, Result};

use crate::pipeline::{hf_get, snapshot_file, VAE_FIX_FILE, VAE_FIX_REPO, VAE_SCALE};
use crate::unet::{
    sdxl_unet_config, ControlNet, ControlNetConfig, UNet2DConditionModel, VaeMomentsEncoder,
};

/// SDXL `add_embedding` dims (diffusers `unet/config.json`): `addition_time_embed_dim = 256`,
/// `projection_class_embeddings_input_dim = 2816` (pooled 1280 + 6·256). The InstantID UNet needs the
/// `add_embedding` head the plain `forward` omits.
const ADDITION_TIME_EMBED_DIM: usize = 256;
const PROJECTION_INPUT_DIM: usize = 2816;

/// Load the **vendored** SDXL UNet with the `add_embedding` head loaded (so
/// [`UNet2DConditionModel::forward_instantid`] runs). Math attention (`use_flash_attn = false`) — the
/// vendored flash path is a stub; perf tuning is later. The caller installs the IP-Adapter K/V pairs.
///
/// sc-10813: packed-detect the tier the SAME way the base txt2img load does. When `root` is a packed
/// MLX q4/q8 tier (`crate::pipeline::detect_packed_unet` — a `quantization` block in
/// `unet/config.json` at group 64), feed the packed `unet/diffusion_pytorch_model.safetensors`; the
/// vendored UNet body + `add_embedding` head packed-detect per-Linear off the `.scales` siblings (their
/// `linear_detect_gs` seams take the packed path automatically), so the edit / inpaint / IP-Adapter
/// lanes fit a low-VRAM budget instead of forcing the dense bf16 tier. A dense diffusers snapshot has no
/// such block, so it loads the `.fp16` weights through the identical dense path (byte-unchanged).
pub fn load_instantid_unet(
    root: &Path,
    device: &Device,
    dtype: DType,
) -> Result<UNet2DConditionModel> {
    let unet_file = instantid_unet_file(root)?;
    // One mmap'd VarBuilder feeds both the UNet body and the `add_embedding` head (both live in the
    // same `unet/` checkpoint). `VarBuilder` is Arc-backed, so the clone is cheap.
    let vs = candle_gen::mmap_var_builder(&[unet_file], dtype, device)?;
    let unet = UNet2DConditionModel::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
        .with_add_embedding(vs, ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
    Ok(unet)
}

/// The `unet/` weight file the vendored InstantID/edit/IP-Adapter UNet loads from (sc-10813): the packed
/// `diffusion_pytorch_model.safetensors` on a packed MLX q4/q8 tier (via
/// [`crate::pipeline::detect_packed_unet`], group 64 validated there), else the dense `.fp16` file. Split
/// out so the packed-vs-dense fork is unit-testable without weights or a GPU.
fn instantid_unet_file(root: &Path) -> Result<PathBuf> {
    Ok(match crate::pipeline::detect_packed_unet(root)? {
        Some((packed_file, _group_size)) => packed_file,
        None => snapshot_file(root, "unet/diffusion_pytorch_model.fp16.safetensors")?,
    })
}

/// As [`load_instantid_unet`], but apply user LoRA/LoKr `adapters` to the UNet at load (sc-6038).
/// InstantID runs on a stock SDXL (RealVisXL) UNet, so SDXL-family LoRAs apply on top of the IdentityNet
/// and face IP-Adapter. Mirrors the SDXL generator's adapter path (`crate::pipeline`'s
/// `load_dense_vendored_unet_with_adapters` / `load_packed_unet_with_adapters`).
///
/// sc-11176 (F-084): fork on the packed tier the SAME way the non-adapter [`load_instantid_unet`]
/// (sc-10813) and the txt2img adapter lane do — the pre-fix loader hard-coded the dense `.fp16` file, so
/// an InstantID/edit/IP-Adapter LoRA job against a packed MLX q4/q8 tier hard-failed with a misleading
/// "snapshot is missing …fp16.safetensors" even though the packed tier was present.
///
/// - **Packed tier** (`crate::pipeline::detect_packed_unet` ⇒ `Some`, sc-11103): the distill LoRA rides
///   the packed Linears **additively** (`y = base(x) + Σ scale·((x·A)·B)`,
///   `crate::adapters::install_additive`) — the u32 codes are never dequantized, so the q4/q8 footprint
///   survives — and any conv LoRA **folds** into the dense convs
///   (`crate::adapters::fold_conv_adapters`) before the UNet body + `add_embedding` head are built. The
///   additive residual equals the dense fold to f32 tolerance.
/// - **Dense snapshot** (`None`, sc-11682): keep the `.fp16` base a **pristine mmap** (evictable —
///   epic 10765) and apply the adapter **additively** on both the Linear
///   (`crate::adapters::install_additive`) and conv (`crate::adapters::install_additive_conv`)
///   surfaces, instead of folding into a host `from_tensors` map. Additive equals the old fold to f32
///   tolerance (~1-ULP golden shift).
///
/// An empty `adapters` slice applies nothing; a non-empty slice that matches no target errors (it never
/// renders an unadapted image silently). The caller installs the IP-Adapter K/V pairs on the returned
/// UNet. `VarBuilder::from_tensors` is Arc-backed, so the body/head clone is cheap.
pub fn load_instantid_unet_with_adapters(
    root: &Path,
    device: &Device,
    dtype: DType,
    adapters: &[AdapterSpec],
) -> Result<UNet2DConditionModel> {
    match crate::pipeline::detect_packed_unet(root)? {
        Some((packed_file, group_size)) => {
            // The vendored UNet threads only the default MLX group 64 through its blocks; a non-64 tier
            // would pack/read at mismatched grids. Refuse it loudly rather than mis-apply.
            crate::adapters::assert_group_size_supported(group_size)?;
            let mut raw = candle_core::safetensors::load(&packed_file, &Device::Cpu)?;
            let table = crate::adapters::build_sdxl_kohya_table(&raw);
            // Conv LoRA folds into the dense convs; the packed Linears stay packed for the additive push.
            let conv = crate::adapters::fold_conv_adapters(&mut raw, adapters, &table)?;
            let vs = VarBuilder::from_tensors(raw, dtype, device);
            let mut unet = UNet2DConditionModel::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
                .with_add_embedding(vs, ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
            let add = crate::adapters::install_additive(&mut unet, adapters, &table, device)?;
            crate::adapters::guard_additive_matched(adapters.len(), conv.merged + add.applied)?;
            Ok(unet)
        }
        None => {
            // sc-11682: keep the bf16 base a pristine mmap (evictable — epic 10765) and apply the
            // adapter additively (Linear + conv residuals) instead of folding into a host `from_tensors`
            // map. The `add_embedding` head shares the same mmap VarBuilder.
            let unet_file = snapshot_file(root, "unet/diffusion_pytorch_model.fp16.safetensors")?;
            let table = crate::adapters::build_sdxl_kohya_table_from_file(&unet_file)?;
            let vs = candle_gen::mmap_var_builder(&[unet_file], dtype, device)?;
            let mut unet = UNet2DConditionModel::new(vs.clone(), 4, 4, false, sdxl_unet_config())?
                .with_add_embedding(vs, ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
            let lin = crate::adapters::install_additive(&mut unet, adapters, &table, device)?;
            let conv = crate::adapters::install_additive_conv(&mut unet, adapters, &table, device)?;
            crate::adapters::guard_additive_matched(adapters.len(), lin.applied + conv.applied)?;
            Ok(unet)
        }
    }
}

/// Load the f16-stable SDXL VAE (`madebyollin/sdxl-vae-fp16-fix`, resolved via `hf-hub` exactly as the
/// txt2img path) at `dtype`. Resolution-agnostic — `build_vae` reads only the autoencoder sub-config.
pub fn load_sdxl_vae(device: &Device, dtype: DType) -> Result<AutoEncoderKL> {
    let config = StableDiffusionConfig::sdxl(None, None, None);
    Ok(config.build_vae(hf_get(VAE_FIX_REPO, VAE_FIX_FILE)?, device, dtype)?)
}

/// Load the **deterministic VAE moments-encoder** for the SDXL edit path (sc-6037) — the encode
/// counterpart of [`load_sdxl_vae`], built from the SAME f16-stable VAE checkpoint
/// (`madebyollin/sdxl-vae-fp16-fix`). candle's stock `AutoEncoderKL` exposes only `decode` plus a
/// device-RNG `sample` (non-portable; the very thing sc-3673 banned), so `VaeMomentsEncoder`
/// (vendored for the trainer, sc-5165) is reused to take the clean latent **mean** × `VAE_SCALE`
/// (0.13025) — the launch-portable img2img/inpaint init latent (no sampling, no device RNG).
pub fn load_sdxl_vae_encoder(device: &Device, dtype: DType) -> Result<VaeMomentsEncoder> {
    let vae_file = hf_get(VAE_FIX_REPO, VAE_FIX_FILE)?;
    let vs = candle_gen::mmap_var_builder(&[vae_file], dtype, device)?;
    Ok(VaeMomentsEncoder::new(vs, VAE_SCALE)?)
}

/// Load a stock diffusers SDXL `ControlNetModel` (the InstantID IdentityNet, or the OpenPose CN for
/// pose mode) from a `WeightsSource`. A `Dir` resolves `diffusion_pytorch_model.safetensors` (then the
/// `.fp16` variant); a `File` is used directly. No conversion — the diffusers key layout is what
/// [`ControlNet::new`] reads.
pub fn load_sdxl_controlnet(
    source: &WeightsSource,
    device: &Device,
    dtype: DType,
) -> Result<ControlNet> {
    let file = match source {
        WeightsSource::File(f) => f.clone(),
        WeightsSource::Dir(d) => {
            let primary = d.join("diffusion_pytorch_model.safetensors");
            if primary.is_file() {
                primary
            } else {
                let fp16 = d.join("diffusion_pytorch_model.fp16.safetensors");
                if fp16.is_file() {
                    fp16
                } else {
                    return Err(CandleError::Msg(format!(
                        "sdxl controlnet: no diffusion_pytorch_model(.fp16).safetensors in {}",
                        d.display()
                    )));
                }
            }
        }
    };
    let vs = candle_gen::mmap_var_builder(&[file], dtype, device)?;
    ControlNet::new(vs, &ControlNetConfig::sdxl())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-10813: the InstantID/edit/IP-Adapter UNet loader picks the packed
    /// `diffusion_pytorch_model.safetensors` on a packed MLX q4/q8 tier (a `quantization` block in
    /// `unet/config.json`) and the dense `.fp16` file on a plain diffusers snapshot — the same fork the
    /// base txt2img load takes, so the edit / inpaint / IP-Adapter lanes serve a low-VRAM tier instead of
    /// forcing dense bf16. GPU-free: asserts file selection only (no mmap / weights).
    #[test]
    fn instantid_unet_file_forks_packed_vs_dense() {
        let tmp =
            std::env::temp_dir().join(format!("sc10813_instantid_unet_{}", std::process::id()));
        let unet_dir = tmp.join("unet");
        std::fs::create_dir_all(&unet_dir).unwrap();

        // Packed tier: a group-64 `quantization` block + the packed weight file at the non-`.fp16` name.
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 4, "group_size": 64}, "cross_attention_dim": 2048}"#,
        )
        .unwrap();
        std::fs::write(
            unet_dir.join("diffusion_pytorch_model.safetensors"),
            b"stub",
        )
        .unwrap();
        assert_eq!(
            instantid_unet_file(&tmp).unwrap(),
            unet_dir.join("diffusion_pytorch_model.safetensors"),
            "a packed tier ⇒ the packed weight file (not the dense .fp16)"
        );

        // Dense snapshot: no quantization block ⇒ the `.fp16` file.
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"cross_attention_dim": 2048, "sample_size": 128}"#,
        )
        .unwrap();
        std::fs::write(
            unet_dir.join("diffusion_pytorch_model.fp16.safetensors"),
            b"stub",
        )
        .unwrap();
        assert_eq!(
            instantid_unet_file(&tmp).unwrap(),
            unet_dir.join("diffusion_pytorch_model.fp16.safetensors"),
            "a dense snapshot ⇒ the .fp16 weight file (unchanged pre-sc-10813 behavior)"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// sc-11176 (F-084): the InstantID/edit/IP-Adapter **adapter** loader forks on the packed tier the
    /// same way the non-adapter load does — a packed q4/q8 tier routes through the packed weight file
    /// (loaded on CPU for the sc-11103 conv-fold + additive install), NOT the dense `.fp16` snapshot.
    /// Pre-fix it hard-coded `.fp16`, so a packed-tier LoRA job hard-failed with a misleading
    /// "snapshot is missing …fp16.safetensors" even though the packed tier was present. GPU-free: drives
    /// the branch on stub weights and asserts the packed arm never surfaces the dense `.fp16` diagnosis
    /// (and the dense arm still does), proving the fork is taken.
    #[test]
    fn instantid_adapter_load_forks_packed_vs_dense() {
        let tmp =
            std::env::temp_dir().join(format!("sc11176_instantid_adapter_{}", std::process::id()));
        let unet_dir = tmp.join("unet");
        std::fs::create_dir_all(&unet_dir).unwrap();
        let dev = Device::Cpu;

        // Packed tier: a group-64 `quantization` block + the packed weight file at the non-`.fp16`
        // name. The adapter load must take the packed fork — it fails parsing the stub safetensors, NOT
        // with a missing-`.fp16` diagnosis (the pre-fix bug). Empty adapters is fine: the fork is chosen
        // before any delta math, and the packed arm loads the packed file (which the stub bytes fail).
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"quantization": {"bits": 4, "group_size": 64}, "cross_attention_dim": 2048}"#,
        )
        .unwrap();
        std::fs::write(
            unet_dir.join("diffusion_pytorch_model.safetensors"),
            b"not-a-real-safetensor",
        )
        .unwrap();
        let err = load_instantid_unet_with_adapters(&tmp, &dev, DType::F32, &[])
            .unwrap_err()
            .to_string();
        assert!(
            !err.contains("fp16"),
            "a packed tier must take the packed fork, not resolve the dense .fp16 file (got: {err})"
        );

        // Dense snapshot: no `quantization` block AND no weight file present ⇒ the dense fork resolves
        // the `.fp16` name and surfaces the missing-snapshot diagnosis naming it — proving the other arm.
        std::fs::write(
            unet_dir.join("config.json"),
            br#"{"cross_attention_dim": 2048, "sample_size": 128}"#,
        )
        .unwrap();
        std::fs::remove_file(unet_dir.join("diffusion_pytorch_model.safetensors")).ok();
        let err = load_instantid_unet_with_adapters(&tmp, &dev, DType::F32, &[])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("fp16"),
            "a dense snapshot with no weights ⇒ the missing-.fp16 diagnosis (got: {err})"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }
}
