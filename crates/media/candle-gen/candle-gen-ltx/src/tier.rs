//! LTX-2.3 **MLX packed-tier ingestion** (sc-9545, sc-9089 umbrella) — the split-file / subfolder
//! resolver + key-remap layer that lets the merged sc-9417 packed-detect seam ([`crate::quant`]) fire on
//! the REAL `SceneWorks/ltx-2.3-mlx` q4/q8 tier weights, with **no dense staging**.
//!
//! ## The gap sc-9417 left (why the wiring alone could not render)
//!
//! sc-9417 wired [`crate::quant::qlinear`] / [`crate::quant::qembedding`] to packed-DETECT on a
//! `{key}.scales` sibling, validated with SYNTHETIC fixtures on the crate's own key layout. But the
//! crate's dense loader (`crate::Pipeline::load_components`) consumes a **single bundled** Lightricks
//! checkpoint (`ltx-2.3-22b-distilled.safetensors`, keys under `model.diffusion_model.*` with the dense
//! Lightricks spelling), whereas the hosted MLX tier ships **split per-component** safetensors in a
//! `q4/` (or `q8/`) subfolder whose `transformer.safetensors` uses **remapped** keys. So the packed
//! `.scales` siblings live under different names than the crate asks for, and the DiT body, connectors,
//! VAE, and gemma each live in a different file. This module bridges that: it resolves the tier's files
//! and rewrites the crate's key requests to the tier's names via candle's [`Rename`] backend, so the
//! *existing* loaders ([`crate::transformer::AvDiT::new`], [`crate::connector`], [`crate::vae`],
//! [`crate::gemma`]) load straight from the packed/dense tier parts unchanged.
//!
//! ## The real q4 tier layout (hf-header audit, sc-9545)
//!
//! ```text
//! <snapshot>/q4/               (or q8/)
//!   quantize_config.json       { "quantization": { "bits": 4, "group_size": 64 } }
//!   config.json embedded_config.json split_model.json
//!   transformer.safetensors    PACKED — DiT body, 1344 `.scales`; keys REMAPPED (see below)
//!   connector.safetensors      DENSE  — `*_embeddings_connector.*` + `text_embedding_projection.*`,
//!                                       crate-native spelling (`to_out.0`, `ff.net.0.proj`, `ff.net.2`)
//!   vae_decoder.safetensors    DENSE  — `up_blocks.*` / `conv_in` / `conv_out` / `per_channel_statistics`,
//!                                       conv weights CHANNELS-LAST `[O,kt,kh,kw,I]`, stats `mean`/`std`
//!   vae_encoder  DENSE  — loaded for conditioning
//!   upsampler    DENSE  — learned stage-one→stage-two latent refinement
//!   audio_vae vocoder DENSE — outside the packed tier's final audio decode
//! <snapshot>/gemma/            DENSE  — standard `language_model.model.*` 5-shard set + tokenizer.json
//! ```
//!
//! ## The transformer key remap (crate spelling → tier spelling), unambiguous per the audit
//!
//! | crate loader asks for | tier `transformer.safetensors` |
//! |---|---|
//! | `…attn*.to_out.0.{weight,scales,biases,bias}` | `…attn*.to_out.…` |
//! | `…ff.net.0.proj.…` | `…ff.proj_in.…` |
//! | `…ff.net.2.…` | `…ff.proj_out.…` |
//! | `…emb.timestep_embedder.linear_1.…` | `…linear1.…` |
//! | `…emb.timestep_embedder.linear_2.…` | `…linear2.…` |
//! | `model.diffusion_model.<X>` | `<X>` (the DiT sits at the file root) |
//!
//! The tier transformer carries **zero** keys in the crate spelling (audited: 0 `net.0.proj`, 0
//! `to_out.0`, 0 `linear_1`), so the rewrite is total and never collides. The connector file *does* use
//! the crate spelling natively, so [`remap_transformer_key`] is applied only to the DiT builder, never
//! the connector one.
//!
//! ## group_size from config (sc-9545 AC)
//!
//! [`TierPaths::packed_config`] reads `quantize_config.json`'s `quantization.group_size` via the shared
//! [`candle_gen::quant::PackedConfig`]; [`TierPaths::validate_group_size`] asserts it equals the
//! [`crate::quant::GROUP_SIZE`] the loaders repack at (64), failing loudly rather than silently
//! mis-repacking if a future tier ever ships a different group.
//!
//! # LTX-**2.5** packed tiers (sc-18776) — a different file set, a different manifest
//!
//! Everything above describes the LTX-**2.3** tier and is unchanged. The LTX-2.5 tier
//! ([`Ltx25Tier`]) mirrors the same idea — one directory per precision, split per-component
//! safetensors, q4-to-all-OSes — but it is **not** the same layout, and the two are told apart by
//! declaration rather than by sniffing:
//!
//! | | LTX-2.3 tier ([`TierPaths`]) | LTX-2.5 tier ([`Ltx25Tier`]) |
//! |---|---|---|
//! | marker | `quantize_config.json` | `split_model.json` declaring a 2.5+ `model_version` |
//! | quant geometry | `quantization.{bits,group_size}` | `quantization_bits` / `quantization_group_size` |
//! | components | 7 files + a sibling `gemma/` snapshot dir | **12 files**, text encoder packed in-bundle |
//! | per-component config | one `embedded_config.json` | each file's own `__metadata__` |
//! | packed beyond the DiT | no | **yes** — connector, DiffVAE decoder, and (q8) text encoder |
//!
//! [`TierPaths::detect`] requires `quantize_config.json`, which no 2.5 tier ships, so a 2.5
//! directory can never take the 2.3 path; [`Ltx25Tier::detect`] gates on the manifest's declared
//! `model_version` (via the shared [`candle_gen::gen_core::ltx_checkpoint::layout_for_declared_version`]), so a
//! SceneWorks-converted **2.3** tree — which also ships a `split_model.json` — can never take the
//! 2.5 path either.
//!
//! ## Why the manifest is the authority, and what is checked against it
//!
//! A packed affine weight is stored as `[out, in·bits/32]` `U32` beside an `[out, in/group]` scales
//! grid. That is two equations in three unknowns (`in`, `bits`, `group`), so the geometry **cannot**
//! be recovered from the tensors alone — the same reasoning `mlx_gen_ltx::diff_vae::DiffVaeQuant`
//! records. The manifest declares `bits` and `group`; [`Ltx25Tier::validate`] then asserts the
//! declaration against every packed triple's own shapes and refuses the bundle when they disagree,
//! rather than picking a geometry and decoding the weights into noise.
//!
//! The same manifest carries the whole-pipeline tier contract: each component reports how many of
//! its Linears were packed, and a component that is **dense inside a quantized tier** must declare a
//! `dense_reason` (plus its detail). That is the mechanism that stops a "q4 transformer, bf16
//! quietly everything else" bundle from passing as q4 — the shipped q4 tier's dense Gemma 4 text
//! encoder is admitted only because it declares `below-quality-bar` with the measurement behind it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::var_builder::Rename;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::ltx_checkpoint::{
    layout_for_declared_version, LtxCheckpointLayout, LtxCheckpointMetadata,
};
use candle_gen::gen_core::weightsmeta::Dtype;
use candle_gen::gen_core::{safetensors_path_tensor_headers, SafetensorsTensorHeader};
use candle_gen::{CandleError, Result as CResult};

/// The resolved file set of a packed LTX tier subfolder (`.../q4` or `.../q8`) + its sibling `gemma/`.
pub struct TierPaths {
    /// The `q4/` (or `q8/`) subfolder holding the split per-component safetensors.
    pub tier_dir: PathBuf,
    /// The Gemma-3-12B encoder snapshot dir (the tier's sibling `gemma/`, or an override).
    pub gemma_dir: PathBuf,
}

impl TierPaths {
    /// Detect a packed tier at `dir`: a directory that directly holds `transformer.safetensors` **and**
    /// `quantize_config.json` (the MLX split-tier marker). Returns `None` for the dense single-bundle
    /// layout so `crate::Pipeline` keeps the legacy path unchanged.
    ///
    /// `gemma_override` (from `LoadSpec::text_encoder`) wins for the Gemma dir; else the tier's sibling
    /// `gemma/` (one level up from the `q4/` subdir) is used. Both are passed-in paths — sc-13749 deleted
    /// the dense path's environment side-channel, which the tier path never consulted anyway.
    pub fn detect(dir: &Path, gemma_override: Option<&Path>) -> Option<Self> {
        let marker = dir.join("transformer.safetensors");
        let cfg = dir.join("quantize_config.json");
        if !(marker.is_file() && cfg.is_file()) {
            return None;
        }
        let gemma_dir = gemma_override
            .map(Path::to_path_buf)
            .or_else(|| {
                // The tier nests `<snapshot>/{q4,q8,gemma}`; from `<snapshot>/q4` the gemma sibling is
                // `../gemma`.
                dir.parent().map(|p| p.join("gemma")).filter(|g| g.is_dir())
            })
            .unwrap_or_else(|| dir.join("gemma"));
        Some(Self {
            tier_dir: dir.to_path_buf(),
            gemma_dir,
        })
    }

    fn file(&self, name: &str) -> CResult<PathBuf> {
        let p = self.tier_dir.join(name);
        if !p.is_file() {
            return Err(CandleError::Msg(format!(
                "ltx tier: missing `{name}` in {} (expected a split MLX tier: transformer / connector \
                 / vae_decoder / vae_encoder / upsampler / audio_vae / vocoder)",
                self.tier_dir.display()
            )));
        }
        Ok(p)
    }

    /// Parse the tier's `quantize_config.json` → the shared [`candle_gen::quant::PackedConfig`]
    /// (`quantization.{bits, group_size}`). Errors if the file is absent/unparseable or carries no
    /// `quantization` block (a packed tier always has one).
    pub fn packed_config(&self) -> CResult<candle_gen::quant::PackedConfig> {
        let p = self.tier_dir.join("quantize_config.json");
        let text = std::fs::read_to_string(&p)
            .map_err(|e| CandleError::Msg(format!("ltx tier: read {}: {e}", p.display())))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| CandleError::Msg(format!("ltx tier: parse {}: {e}", p.display())))?;
        candle_gen::quant::PackedConfig::from_config(&json).ok_or_else(|| {
            CandleError::Msg(format!(
                "ltx tier: {} has no `quantization.bits` — not a packed tier config",
                p.display()
            ))
        })
    }

    /// Read + **validate** the tier's `group_size` against the [`crate::quant::GROUP_SIZE`] the packed
    /// loaders repack at (sc-9545 AC). The MLX-packed→GGML repack ([`candle_gen::quant`]) is done at a
    /// fixed group; if the tier ever ships a different group the repack would mis-align, so fail loudly
    /// rather than render garbage. Returns the validated group size.
    pub fn validate_group_size(&self) -> CResult<usize> {
        let cfg = self.packed_config()?;
        let g = cfg.group_size as usize;
        if g != crate::quant::GROUP_SIZE {
            return Err(CandleError::Msg(format!(
                "ltx tier: quantize_config.json group_size {g} != the loader's repack group {} — the \
                 MLX→GGML repack would mis-align. A tier at a new group needs the group threaded into \
                 the packed loaders (candle_gen::quant::*_gs already accepts it).",
                crate::quant::GROUP_SIZE
            )));
        }
        Ok(g)
    }

    /// The DiT VarBuilder over the tier's **packed** `transformer.safetensors`, with the crate→tier key
    /// remap applied so [`crate::transformer::AvDiT::new`] (which asks for `model.diffusion_model.<X>`
    /// with the dense Lightricks spelling) resolves the tier's rootless remapped keys — firing the
    /// packed-detect seam on the real `.scales` siblings. Loaded at `dtype` (bf16).
    pub fn dit_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner =
            candle_gen::mmap_var_builder(&[self.file("transformer.safetensors")?], dtype, device)?;
        Ok(rename_vb(inner, dtype, device, remap_transformer_key))
    }

    /// The connector + text-projection VarBuilder over the tier's **dense** `connector.safetensors`.
    /// Its keys already use the crate spelling (`to_out.0`, `ff.net.0.proj`, `ff.net.2`), so only the
    /// `model.diffusion_model.` prefix (which the crate prepends for the connectors) is stripped — the
    /// text projection is at the file root, the connectors too.
    pub fn connector_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner =
            candle_gen::mmap_var_builder(&[self.file("connector.safetensors")?], dtype, device)?;
        Ok(rename_vb(inner, dtype, device, strip_diffusion_prefix))
    }

    /// The video-VAE decoder VarBuilder over the tier's **dense** `vae_decoder.safetensors`, remapped so
    /// [`crate::vae::LtxVideoVae::new`] (which asks for `vae.decoder.<X>` and `vae.per_channel_statistics.
    /// {mean-of-means,std-of-means}`) resolves the tier's rootless `<X>` / `per_channel_statistics.{mean,
    /// std}`. Conv weights are additionally **permuted** from the tier's channels-last `[O,kt,kh,kw,I]` to
    /// the crate's PyTorch `[O,I,kt,kh,kw]` on load (see `VaeRemapBackend`).
    pub fn vae_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner =
            candle_gen::mmap_var_builder(&[self.file("vae_decoder.safetensors")?], dtype, device)?;
        Ok(VarBuilder::from_backend(
            Box::new(VaeRemapBackend { inner }),
            dtype,
            device.clone(),
        ))
    }

    /// The video-VAE encoder builder over `vae_encoder.safetensors`. It presents the same synthetic
    /// `vae.encoder.*` / `vae.per_channel_statistics.{mean-of-means,std-of-means}` namespace as the
    /// unified checkpoint and transposes channels-last conv weights on demand.
    pub fn vae_encoder_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner =
            candle_gen::mmap_var_builder(&[self.file("vae_encoder.safetensors")?], dtype, device)?;
        Ok(VarBuilder::from_backend(
            Box::new(VaeEncoderRemapBackend { inner }),
            dtype,
            device.clone(),
        ))
    }

    /// Learned spatial refinement weights co-located with the packed tier,
    /// accepting either upstream staged filename without silently choosing an
    /// ambiguous directory.
    ///
    /// Returns the **path**, not a `VarBuilder`: `LatentUpsampler::from_checkpoint` must read the
    /// file's `__metadata__` as well as its tensors, and a builder has already discarded it.
    pub fn upsampler_file(&self) -> CResult<PathBuf> {
        crate::canonical_upsampler_file(&self.tier_dir)
    }

    /// The Gemma-3-12B encoder VarBuilder rooted at `language_model.model.` over the tier's sibling
    /// `gemma/` shards. The tier ships Gemma **dense** with the standard `language_model.model.*` keys
    /// (matches the crate exactly), so no remap — just the sorted-shard resolve.
    pub fn gemma_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let files = candle_gen::sorted_safetensors(&self.gemma_dir, "ltx tier gemma")?;
        Ok(candle_gen::mmap_var_builder(&files, dtype, device)?.pp("language_model.model"))
    }

    /// The Gemma tokenizer path (`gemma/tokenizer.json`).
    pub fn tokenizer_path(&self) -> PathBuf {
        self.gemma_dir.join("tokenizer.json")
    }
}

/// Wrap `inner` in candle's [`Rename`] backend applying `f` to every requested key. Boxed as a
/// `Renamer` fn so the DiT / connector remaps share one path.
fn rename_vb(
    inner: VarBuilder<'static>,
    dtype: DType,
    device: &Device,
    f: fn(&str) -> String,
) -> VarBuilder<'static> {
    let renamer: Box<dyn Fn(&str) -> String + Send + Sync> = Box::new(f);
    VarBuilder::from_backend(Box::new(Rename::new(inner, renamer)), dtype, device.clone())
}

/// Rewrite a crate DiT key (what [`crate::transformer`] asks for) to the tier's `transformer.safetensors`
/// spelling. Strips the `model.diffusion_model.` prefix (the DiT is at the file root) and applies the
/// four projection renames. Order matters: strip the prefix first, then the sub-key renames.
pub fn remap_transformer_key(key: &str) -> String {
    let k = key
        .strip_prefix("model.diffusion_model.")
        .unwrap_or(key)
        .to_string();
    // `attn*.to_out.0.<suffix>` → `attn*.to_out.<suffix>` (candle `Linear` under a `.0` ModuleList slot).
    let k = k.replace(".to_out.0.", ".to_out.");
    // FeedForward: `ff.net.0.proj.<suffix>` → `ff.proj_in.<suffix>`, `ff.net.2.<suffix>` → `ff.proj_out`.
    let k = k
        .replace(".net.0.proj.", ".proj_in.")
        .replace(".net.2.", ".proj_out.");
    // AdaLayerNormSingle timestep MLP: `linear_1`/`linear_2` → `linear1`/`linear2`.
    k.replace(".linear_1.", ".linear1.")
        .replace(".linear_2.", ".linear2.")
}

/// Strip the crate's `model.diffusion_model.` connector prefix — the tier's `connector.safetensors`
/// roots the connectors + text projection at the file top, in the crate's own key spelling.
fn strip_diffusion_prefix(key: &str) -> String {
    key.strip_prefix("model.diffusion_model.")
        .unwrap_or(key)
        .to_string()
}

/// A `SimpleBackend` for the tier's `vae_decoder.safetensors` that (1) strips the crate's `vae.` /
/// `vae.decoder.` prefix, (2) renames the `mean-of-means`/`std-of-means` stats to the tier's
/// `mean`/`std`, and (3) **permutes** every conv weight from the tier's channels-last `[O,kt,kh,kw,I]`
/// to the crate's `[O,I,kt,kh,kw]`. Kept a bespoke backend (not [`Rename`]) because the conv permute is a
/// tensor transform, not just a key rewrite.
struct VaeRemapBackend {
    inner: VarBuilder<'static>,
}

impl VaeRemapBackend {
    /// crate VAE key → tier `vae_decoder.safetensors` key. Returns `(tier_key, is_conv_weight)`.
    fn remap(key: &str) -> (String, bool) {
        // The crate roots the VAE at `vae.` then descends into `decoder.` for the block body and
        // `per_channel_statistics.` for the stats — the tier drops the `vae.`/`vae.decoder.` wrappers.
        let k = key
            .strip_prefix("vae.decoder.")
            .or_else(|| key.strip_prefix("vae."))
            .unwrap_or(key);
        // Stats: crate `per_channel_statistics.mean-of-means` → tier `.mean`; `std-of-means` → `.std`.
        let k = k
            .replace(
                "per_channel_statistics.mean-of-means",
                "per_channel_statistics.mean",
            )
            .replace(
                "per_channel_statistics.std-of-means",
                "per_channel_statistics.std",
            );
        // A conv weight (`….conv.weight`) is channels-last in the tier and must be permuted on load.
        let is_conv_weight = k.ends_with(".conv.weight");
        (k, is_conv_weight)
    }

    /// Permute a tier channels-last conv weight back to Candle's channels-first layout.
    fn permute_conv(w: Tensor) -> candle_gen::candle_core::Result<Tensor> {
        match w.rank() {
            // Conv3d `[O,kt,kh,kw,I]` → `[O,I,kt,kh,kw]`.
            5 => w.permute((0, 4, 1, 2, 3))?.contiguous(),
            // Conv2d `[O,kh,kw,I]` → `[O,I,kh,kw]`.
            4 => w.permute((0, 3, 1, 2))?.contiguous(),
            _ => Ok(w),
        }
    }
}

impl candle_gen::candle_nn::var_builder::SimpleBackend for VaeRemapBackend {
    fn get(
        &self,
        s: candle_gen::candle_core::Shape,
        name: &str,
        _h: candle_gen::candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        // Shape-checked reads are only used for non-conv leaves (the crate reads convs via
        // `get_unchecked`, then infers dims from the permuted shape), so no permute is needed here.
        let (k, _) = Self::remap(name);
        self.inner
            .get_with_hints_dtype(s, &k, Default::default(), dtype)?
            .to_device(dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let (k, is_conv) = Self::remap(name);
        let t = self.inner.get_unchecked_dtype(&k, dtype)?.to_device(dev)?;
        if is_conv {
            Self::permute_conv(t)
        } else {
            Ok(t)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let (k, _) = Self::remap(name);
        self.inner.contains_tensor(&k)
    }
}

struct VaeEncoderRemapBackend {
    inner: VarBuilder<'static>,
}

impl VaeEncoderRemapBackend {
    fn remap(key: &str) -> (String, bool) {
        let k = key
            .strip_prefix("vae.encoder.")
            .or_else(|| key.strip_prefix("vae."))
            .unwrap_or(key)
            .replace(
                "per_channel_statistics.mean-of-means",
                "per_channel_statistics._mean_of_means",
            )
            .replace(
                "per_channel_statistics.std-of-means",
                "per_channel_statistics._std_of_means",
            );
        let is_conv_weight = k.ends_with(".conv.weight");
        (k, is_conv_weight)
    }
}

impl candle_gen::candle_nn::var_builder::SimpleBackend for VaeEncoderRemapBackend {
    fn get(
        &self,
        s: candle_gen::candle_core::Shape,
        name: &str,
        _h: candle_gen::candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let (k, _) = Self::remap(name);
        self.inner
            .get_with_hints_dtype(s, &k, Default::default(), dtype)?
            .to_device(dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let (k, is_conv) = Self::remap(name);
        let t = self.inner.get_unchecked_dtype(&k, dtype)?.to_device(dev)?;
        if is_conv {
            VaeRemapBackend::permute_conv(t)
        } else {
            Ok(t)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let (k, _) = Self::remap(name);
        self.inner.contains_tensor(&k)
    }
}

/// Inverse of `mlx-gen-ltx::convert::sanitize_audio_vae`: the converted component drops the
/// `audio_vae.decoder.` wrapper, renames the two statistics with a leading underscore, and stores
/// Conv2d kernels channels-last. The Candle decoder deliberately keeps the released-checkpoint
/// namespace, so the bridge belongs at the component boundary rather than inside the decoder.
struct AudioVaeRemapBackend {
    inner: VarBuilder<'static>,
}

impl AudioVaeRemapBackend {
    fn remap(key: &str) -> (String, bool) {
        let k = key
            .strip_prefix("audio_vae.decoder.")
            .or_else(|| key.strip_prefix("audio_vae."))
            .unwrap_or(key)
            .replace(
                "per_channel_statistics.mean-of-means",
                "per_channel_statistics._mean_of_means",
            )
            .replace(
                "per_channel_statistics.std-of-means",
                "per_channel_statistics._std_of_means",
            );
        let is_conv = k.ends_with(".conv.weight");
        (k, is_conv)
    }

    fn load(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let (key, is_conv) = Self::remap(name);
        let tensor = self
            .inner
            .get_unchecked_dtype(&key, dtype)?
            .to_device(dev)?;
        if is_conv {
            VaeRemapBackend::permute_conv(tensor)
        } else {
            Ok(tensor)
        }
    }
}

impl candle_gen::candle_nn::var_builder::SimpleBackend for AudioVaeRemapBackend {
    fn get(
        &self,
        shape: candle_gen::candle_core::Shape,
        name: &str,
        _hints: candle_gen::candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let tensor = self.load(name, dtype, dev)?;
        if tensor.shape() != &shape {
            candle_gen::candle_core::bail!(
                "shape mismatch for {name}: expected {:?}, got {:?}",
                shape.dims(),
                tensor.dims()
            )
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        self.load(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let (key, _) = Self::remap(name);
        self.inner.contains_tensor(&key)
    }
}

/// Inverse of `mlx-gen-ltx::convert::sanitize_vocoder`: every `vocoder.` segment was removed and
/// every rank-3 convolution was stored channels-last. Transposed-convolution `ups.*.weight` uses a
/// different inverse permutation from ordinary Conv1d weights.
struct VocoderRemapBackend {
    inner: VarBuilder<'static>,
}

impl VocoderRemapBackend {
    fn remap(key: &str) -> (String, bool, bool) {
        let converted = key.replace("vocoder.", "");
        let rank3_weight = converted.contains("weight");
        let transposed = converted.contains("ups");
        (converted, rank3_weight, transposed)
    }

    fn permute_weight(tensor: Tensor, transposed: bool) -> candle_gen::candle_core::Result<Tensor> {
        if transposed {
            // Converted ConvTranspose1d `[O,K,I]` → released/Candle `[I,O,K]`.
            tensor.permute((2, 0, 1))?.contiguous()
        } else {
            // Converted Conv1d `[O,K,I]` → released/Candle `[O,I,K]`.
            tensor.permute((0, 2, 1))?.contiguous()
        }
    }

    fn load(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let (key, rank3_weight, transposed) = Self::remap(name);
        let tensor = self
            .inner
            .get_unchecked_dtype(&key, dtype)?
            .to_device(dev)?;
        if !rank3_weight || tensor.rank() != 3 {
            return Ok(tensor);
        }
        Self::permute_weight(tensor, transposed)
    }
}

impl candle_gen::candle_nn::var_builder::SimpleBackend for VocoderRemapBackend {
    fn get(
        &self,
        shape: candle_gen::candle_core::Shape,
        name: &str,
        _hints: candle_gen::candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        let tensor = self.load(name, dtype, dev)?;
        if tensor.shape() != &shape {
            candle_gen::candle_core::bail!(
                "shape mismatch for {name}: expected {:?}, got {:?}",
                shape.dims(),
                tensor.dims()
            )
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_gen::candle_core::Result<Tensor> {
        self.load(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let (key, _, _) = Self::remap(name);
        self.inner.contains_tensor(&key)
    }
}

/// Builders for SceneWorks-converted LTX-2.5 components. They are path-based so the resolved
/// `LoadSpec` remains authoritative; the tier manifest validates the directory, while explicit
/// component selection still chooses the exact file that is mapped.
pub(crate) fn ltx25_transformer_vb(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let inner = candle_gen::mmap_var_builder(&[path.to_path_buf()], dtype, device)?;
    Ok(rename_vb(inner, dtype, device, remap_transformer_key))
}

pub(crate) fn ltx25_connector_vb(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let inner = candle_gen::mmap_var_builder(&[path.to_path_buf()], dtype, device)?;
    Ok(rename_vb(inner, dtype, device, strip_diffusion_prefix))
}

pub(crate) fn ltx25_vae_decoder_vb(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let inner = candle_gen::mmap_var_builder(&[path.to_path_buf()], dtype, device)?;
    Ok(VarBuilder::from_backend(
        Box::new(VaeRemapBackend { inner }),
        dtype,
        device.clone(),
    ))
}

pub(crate) fn ltx25_vae_encoder_vb(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let inner = candle_gen::mmap_var_builder(&[path.to_path_buf()], dtype, device)?;
    Ok(VarBuilder::from_backend(
        Box::new(VaeEncoderRemapBackend { inner }),
        dtype,
        device.clone(),
    ))
}

pub(crate) fn ltx25_diff_vae_vb(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> CResult<(VarBuilder<'static>, VarBuilder<'static>)> {
    let body = candle_gen::mmap_var_builder(&[path.to_path_buf()], dtype, device)?;
    let stats = rename_vb(body.clone(), dtype, device, remap_diff_vae_stat_key);
    Ok((body, stats))
}

pub(crate) fn ltx25_audio_vae_vb(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let inner = candle_gen::mmap_var_builder(&[path.to_path_buf()], dtype, device)?;
    Ok(VarBuilder::from_backend(
        Box::new(AudioVaeRemapBackend { inner }),
        dtype,
        device.clone(),
    ))
}

pub(crate) fn ltx25_vocoder_vb(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let inner = candle_gen::mmap_var_builder(&[path.to_path_buf()], dtype, device)?;
    Ok(VarBuilder::from_backend(
        Box::new(VocoderRemapBackend { inner }),
        dtype,
        device.clone(),
    ))
}

// =================================================================================================
// LTX-2.5 packed tiers (sc-18776)
// =================================================================================================

/// The tier manifest an LTX-2.5 bundle ships beside its per-component files — the same file
/// `mlx_gen_ltx::tiers::TIER_MANIFEST_FILE` writes.
pub const TIER_MANIFEST_FILE: &str = "split_model.json";

/// The `__metadata__` key the tier converter stamps every component with, so a file moved out of
/// its directory still says which tier it came from
/// (`mlx_gen_ltx::tiers::TIER_METADATA_KEY`).
pub const TIER_METADATA_KEY: &str = "sceneworks_tier";

/// The merged per-component config sidecar the tier ships beside the manifest.
pub const EMBEDDED_CONFIG_FILE: &str = "embedded_config.json";

/// One component of an LTX-2.5 tier, by the id the tier manifest uses.
///
/// This is deliberately the **tier's** vocabulary, not
/// [`candle_gen::gen_core::ltx_checkpoint::LtxComponent`]'s: the tier splits what that enum calls one
/// `conv_video_vae` slot into a separate `vae_decoder` / `vae_encoder` pair (and likewise for the
/// diffusion VAE), and it carries a `connector` file that the upstream 2.5 release keeps inside the
/// transformer. Mapping one onto the other would have to invent or drop a file, so the reader speaks
/// the manifest's own names and the two vocabularies stay honest about being different.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ltx25Component {
    /// The `AVTransformer3DModel` denoiser (packed at q4/q8).
    Transformer,
    /// The packed Gemma 4 text encoder — dense at q4 (`below-quality-bar`), packed at q8.
    TextEncoder,
    /// The video/audio embedding connectors + the text-embedding projection (packed at q4/q8).
    Connector,
    /// The convolutional video VAE decoder (`CausalVideoAutoencoder`), always dense.
    VaeDecoder,
    /// The convolutional video VAE encoder, always dense.
    VaeEncoder,
    /// The diffusion video VAE's encoder, always dense.
    DiffusionVaeEncoder,
    /// The diffusion video VAE decoder (`CausalDiffusionVAE`) — packed at q4/q8.
    VaeDiffusionDecoder,
    /// The audio VAE, always dense.
    AudioVae,
    /// The vocoder, always dense.
    Vocoder,
    /// The spatial `LatentUpsampler`, always dense.
    SpatialUpsampler,
    /// The temporal `LatentUpsampler`, always dense.
    TemporalUpsampler,
    /// The duration head, always dense.
    DurationHead,
}

/// Every component a complete LTX-2.5 tier must carry.
///
/// Private, and reached through [`Ltx25Component::all`], for a reason that lives outside this
/// module: the cross-backend geometry gate in `scripts/check-workspace.py` diffs every **public**
/// `const` the two backends of a model family declare, keyed on the bare constant name.
/// `mlx-gen-ltx` already declares an `LtxTier::ALL` — its three *precision tiers* — so a public
/// `Ltx25Component::ALL` here would be diffed against it and red as a backend divergence between
/// two unrelated lists that happen to share a name. The gate is right to compare same-named
/// constants; this list simply is not one it means.
const ALL_COMPONENTS: &[Ltx25Component] = &[
    Ltx25Component::Transformer,
    Ltx25Component::TextEncoder,
    Ltx25Component::Connector,
    Ltx25Component::VaeDecoder,
    Ltx25Component::VaeEncoder,
    Ltx25Component::DiffusionVaeEncoder,
    Ltx25Component::VaeDiffusionDecoder,
    Ltx25Component::AudioVae,
    Ltx25Component::Vocoder,
    Ltx25Component::SpatialUpsampler,
    Ltx25Component::TemporalUpsampler,
    Ltx25Component::DurationHead,
];

impl Ltx25Component {
    /// Every component a complete LTX-2.5 tier must carry, in manifest order.
    pub fn all() -> &'static [Ltx25Component] {
        ALL_COMPONENTS
    }

    /// The manifest id for this component.
    pub fn id(self) -> &'static str {
        match self {
            Ltx25Component::Transformer => "transformer",
            Ltx25Component::TextEncoder => "text_encoder",
            Ltx25Component::Connector => "connector",
            Ltx25Component::VaeDecoder => "vae_decoder",
            Ltx25Component::VaeEncoder => "vae_encoder",
            Ltx25Component::DiffusionVaeEncoder => "diffusion_vae_encoder",
            Ltx25Component::VaeDiffusionDecoder => "vae_diffusion_decoder",
            Ltx25Component::AudioVae => "audio_vae",
            Ltx25Component::Vocoder => "vocoder",
            Ltx25Component::SpatialUpsampler => "spatial_upsampler",
            Ltx25Component::TemporalUpsampler => "temporal_upsampler",
            Ltx25Component::DurationHead => "duration_head",
        }
    }

    /// Parse a manifest id (the inverse of [`id`](Self::id)).
    pub fn from_id(id: &str) -> Option<Ltx25Component> {
        Ltx25Component::all().iter().copied().find(|c| c.id() == id)
    }
}

/// The affine-quant geometry a packed 2.5 tier was written at — the candle twin of
/// `mlx_gen_ltx::diff_vae::DiffVaeQuant`, read from the tier manifest's `quantization_bits` /
/// `quantization_group_size`.
///
/// It is a **declaration**, never an inference: `[out, in·bits/32]` codes plus an `[out, in/group]`
/// scales grid leave `in`, `bits` and `group` under-determined, so a loader that guessed would decode
/// a q8 file as q4 (or a group-32 file as group-64) and render noise. Every consumer is handed this
/// value and refuses a `.scales` sibling it was not told to expect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ltx25Quant {
    /// Bits per packed weight (4 or 8).
    pub bits: usize,
    /// The affine group width along the input axis.
    pub group: usize,
}

/// One `component_detail` row of the tier manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx25ManifestComponent {
    /// The manifest id, verbatim — kept as a string so a component this reader does not name is
    /// still carried and still validated, instead of being silently dropped.
    pub name: String,
    /// The component's file name, relative to the tier directory. Read from the manifest rather
    /// than guessed, so a rehost that renames a file resolves from its own declaration.
    pub file: String,
    /// How many tensors the file is declared to hold.
    pub tensors: usize,
    /// How many of its Linears were packed at this tier's width.
    pub quantized_linears: usize,
    /// Why this component is dense, when it is. Required for a dense component inside a quantized
    /// tier — that requirement is the whole-pipeline tier contract.
    pub dense_reason: Option<String>,
    /// The human-readable justification behind [`dense_reason`](Self::dense_reason).
    pub dense_reason_detail: Option<String>,
}

impl Ltx25ManifestComponent {
    /// The typed component this row names, or `None` for a row this reader does not name.
    pub fn component(&self) -> Option<Ltx25Component> {
        Ltx25Component::from_id(&self.name)
    }
}

/// A parsed LTX-2.5 tier manifest (`split_model.json`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx25TierManifest {
    /// The tier id (`q4` / `q8` / `bf16`), verbatim.
    pub tier: String,
    /// The `model_version` every component of this tier declares.
    pub model_version: String,
    /// Whether this tier packs anything at all.
    pub quantized: bool,
    /// The declared affine geometry. Present even on a dense tier (the converter writes its
    /// defaults), so it is only *meaningful* when [`quantized`](Self::quantized).
    pub quant: Ltx25Quant,
    /// One row per component, in manifest order.
    pub components: Vec<Ltx25ManifestComponent>,
}

impl Ltx25TierManifest {
    /// The row for `name`, if the manifest carries one.
    pub fn component(&self, name: &str) -> Option<&Ltx25ManifestComponent> {
        self.components.iter().find(|c| c.name == name)
    }

    /// The affine geometry a packed loader must be handed, or `None` for a dense tier.
    pub fn quant(&self) -> Option<Ltx25Quant> {
        self.quantized.then_some(self.quant)
    }

    /// Parse `<dir>/split_model.json` as a **2.5** manifest.
    ///
    /// Every key of the 2.5 schema is required. Callers that do not already know the manifest is a
    /// 2.5 one must go through [`Ltx25Tier::detect`], which gates on the declared `model_version`
    /// first — an LTX-**2.3** converted tree ships a `split_model.json` too, and it carries none of
    /// the keys below.
    pub fn from_dir(dir: &Path) -> CResult<Self> {
        let path = dir.join(TIER_MANIFEST_FILE);
        let json = read_manifest_json(&path)?;
        Self::from_value(&path, &json)
    }

    /// Parse an already-loaded manifest value. `source` names the file in every error.
    pub fn from_value(source: &Path, json: &serde_json::Value) -> CResult<Self> {
        let rows = json
            .get("component_detail")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "ltx 2.5 tier: {} has no `component_detail` array — not a tier manifest",
                    source.display()
                ))
            })?;
        let mut components = Vec::with_capacity(rows.len());
        for row in rows {
            components.push(Ltx25ManifestComponent {
                name: json_string(source, row, "name")?,
                file: json_string(source, row, "file")?,
                tensors: json_usize(source, row, "tensors")?,
                quantized_linears: json_usize(source, row, "quantized_linears")?,
                dense_reason: optional_json_string(source, row, "dense_reason")?,
                dense_reason_detail: optional_json_string(source, row, "dense_reason_detail")?,
            });
        }
        Ok(Ltx25TierManifest {
            tier: json_string(source, json, "tier")?,
            model_version: json_string(source, json, "model_version")?,
            quantized: json_bool(source, json, "quantized")?,
            quant: Ltx25Quant {
                bits: json_usize(source, json, "quantization_bits")?,
                group: json_usize(source, json, "quantization_group_size")?,
            },
            components,
        })
    }
}

/// A resolved LTX-2.5 packed tier directory: its manifest, and the per-component files it names.
#[derive(Debug)]
pub struct Ltx25Tier {
    dir: PathBuf,
    manifest: Ltx25TierManifest,
}

impl Ltx25Tier {
    /// Detect an LTX-2.5 tier at `dir`.
    ///
    /// * `Ok(None)` — `dir` ships no [`TIER_MANIFEST_FILE`], or its manifest does not declare a
    ///   `model_version` that ships split.
    /// * `Err` — the manifest declares a split version and is then unreadable or malformed. A
    ///   broken 2.5 manifest reported as `Ok(None)` would fall through to a loader that picks files
    ///   by name.
    ///
    /// # The order of these two steps is the contract
    ///
    /// The `model_version` gate runs **alone, and first** — before a single key of the 2.5 schema
    /// is required. A SceneWorks-converted LTX-**2.3** tree ships a `split_model.json` too, and it
    /// carries `format` / `model_version` / `components` / `quantized` / `quantization_*` but
    /// **no** `component_detail` and **no** `tier` (see
    /// `mlx-gen-ltx/tests/fixtures/ltx_2_3_split_model.json`, the committed real one). Parsing the
    /// 2.5 schema before gating would turn every 2.3 tier into a hard error here instead of a clean
    /// "not mine" — and the caller that error reaches is a dispatcher whose whole job is to fall
    /// through to [`TierPaths`].
    ///
    /// A manifest with **no** `model_version` key at all is likewise `Ok(None)`, not an error:
    /// [`layout_for_declared_version`] reads an undeclared version as the oldest layout, and
    /// pre-`model_version` trees exist.
    pub fn detect(dir: &Path) -> CResult<Option<Self>> {
        let path = dir.join(TIER_MANIFEST_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let json = read_manifest_json(&path)?;
        // Step 1: the version gate, on `model_version` and nothing else.
        let declared = json
            .get(candle_gen::gen_core::ltx_checkpoint::MODEL_VERSION_METADATA_KEY)
            .and_then(serde_json::Value::as_str);
        if layout_for_declared_version(declared) != LtxCheckpointLayout::Split {
            return Ok(None);
        }
        // Step 2: this manifest says it is a split-layout release, so the 2.5 schema is now
        // required and anything missing from it is a real fault.
        let manifest = Ltx25TierManifest::from_value(&path, &json)?;
        Ok(Some(Ltx25Tier {
            dir: dir.to_path_buf(),
            manifest,
        }))
    }

    /// The tier directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The parsed manifest.
    pub fn manifest(&self) -> &Ltx25TierManifest {
        &self.manifest
    }

    /// The affine geometry to hand a packed loader, or `None` for the dense tier.
    pub fn quant(&self) -> Option<Ltx25Quant> {
        self.manifest.quant()
    }

    /// The manifest row for `component`, or a typed error naming the tier when it has none.
    pub fn row(
        &self,
        component: Ltx25Component,
    ) -> Result<&Ltx25ManifestComponent, Ltx25TierError> {
        self.manifest.component(component.id()).ok_or_else(|| {
            Ltx25TierError::MissingComponentEntry {
                component: component.id().to_string(),
                tier: self.manifest.tier.clone(),
                manifest: self.dir.join(TIER_MANIFEST_FILE),
            }
        })
    }

    /// The on-disk path of `component`, from the manifest's own `file` declaration.
    ///
    /// Errors when the manifest does not list the component, or lists it and the file is absent —
    /// two different faults with two different messages, because they need two different fixes.
    pub fn file(&self, component: Ltx25Component) -> Result<PathBuf, Ltx25TierError> {
        let row = self.row(component)?;
        let path = self.dir.join(&row.file);
        if !path.is_file() {
            return Err(Ltx25TierError::MissingComponentFile {
                component: component.id().to_string(),
                path,
            });
        }
        Ok(path)
    }

    /// `component`'s own `__metadata__` — the per-component config an LTX-2.5 bundle carries
    /// instead of one shared sidecar. Header-only, so it is safe on the 22 GB text encoder.
    pub fn component_metadata(&self, component: Ltx25Component) -> CResult<LtxCheckpointMetadata> {
        let path = self.file(component)?;
        LtxCheckpointMetadata::from_file(&path)
            .map_err(|e| CandleError::Msg(format!("ltx 2.5 tier: {}: {e}", path.display())))
    }

    /// A plain mmap [`VarBuilder`] over `component`'s file, rooted at the file root.
    ///
    /// No key remap: unlike the 2.3 tier, the 2.5 converter emits each component under the key
    /// spelling this crate's loaders already ask for.
    pub fn component_vb(
        &self,
        component: Ltx25Component,
        dtype: DType,
        device: &Device,
    ) -> CResult<VarBuilder<'static>> {
        let path = self.file(component)?;
        candle_gen::mmap_var_builder(&[path], dtype, device)
    }

    /// The connector + text-projection builder over the tier's `connector.safetensors`.
    ///
    /// The tier roots the connectors and the text projection at the **file root**, already in this
    /// crate's own key spelling (`to_out.0`, `ff.net.0.proj`, `ff.net.2`). The crate's loaders ask
    /// for them under the DiT's `model.diffusion_model.` prefix, so — exactly as the 2.3 tier's
    /// [`TierPaths::connector_vb`] does — only that prefix is stripped. The transformer projection
    /// renames must **not** be applied here; the connector file does not use the DiT's spelling.
    pub fn connector_vb(&self, dtype: DType, device: &Device) -> CResult<VarBuilder<'static>> {
        let inner = self.component_vb(Ltx25Component::Connector, dtype, device)?;
        Ok(rename_vb(inner, dtype, device, strip_diffusion_prefix))
    }

    /// The two builders [`crate::diff_vae::NaDiffusionDecoder::load_quantized`] takes, over the
    /// tier's `vae_diffusion_decoder.safetensors`: `(body, stats)`.
    ///
    /// The tier's layout differs from the released checkpoint's in two ways, and both are handled
    /// here rather than by the decoder:
    ///
    /// * the released file nests the decoder under `decoder.`; the tier hoists it to the **file
    ///   root**, so the body builder is the file itself with no `pp`;
    /// * the released file names the latent statistics `per_channel_statistics.{mean,std}-of-means`;
    ///   the tier writes `per_channel_statistics.{mean,std}`, so the stats builder renames them.
    ///
    /// Only the two statistic keys are rewritten. Renaming by suffix across the whole namespace
    /// would also rewrite any future key that happened to end in `-of-means`, which is how a remap
    /// silently binds the wrong tensor.
    pub fn diff_vae_vb(
        &self,
        dtype: DType,
        device: &Device,
    ) -> CResult<(VarBuilder<'static>, VarBuilder<'static>)> {
        let body = self.component_vb(Ltx25Component::VaeDiffusionDecoder, dtype, device)?;
        let stats = rename_vb(body.clone(), dtype, device, remap_diff_vae_stat_key);
        Ok((body, stats))
    }

    /// **Validate the whole bundle against its own manifest, before any weights are loaded.**
    ///
    /// Every check reads safetensors *headers* only, so validating all three shipped tiers costs a
    /// few seconds and no GPU. The faults it refuses, each at the exact value that carries it:
    ///
    /// 1. a component the manifest does not list, or lists and does not ship;
    /// 2. a file whose tensor count disagrees with the manifest (a truncated or substituted file);
    /// 3. a file whose packed-Linear count disagrees with the manifest (a bf16 file dropped into a
    ///    quantized tier, or the reverse);
    /// 4. a packed triple whose shapes disagree with the declared `bits`/`group` — the check that
    ///    catches a **q8 file inside a q4 tier**, which every count above passes unchanged. It pins
    ///    the *product* `bits·group` rather than each independently; check 9 pinning `group` is
    ///    what makes it decisive for the shipped tiers (see the note at the check itself);
    /// 5. an incomplete packed triple, or packed codes stored at a float dtype;
    /// 6. a component stamped with a different `sceneworks_tier` or `model_version`;
    /// 7. a component that is dense inside a quantized tier without declaring why;
    /// 8. any packed tensor inside the dense tier;
    /// 9. a manifest group width the packed loaders cannot repack at.
    ///
    /// Returns a [`Ltx25TierReport`] naming what was checked and what was legitimately skipped, so a
    /// green run can be read rather than trusted.
    pub fn validate(&self) -> Result<Ltx25TierReport, Ltx25TierError> {
        let manifest = &self.manifest;
        // (9) The packed loaders repack at one fixed group ([`crate::quant::GROUP_SIZE`]). A tier at
        // a different group would repack mis-aligned, so refuse it here rather than downstream.
        if manifest.quantized && manifest.quant.group != crate::quant::GROUP_SIZE {
            return Err(Ltx25TierError::UnsupportedGroupSize {
                tier: manifest.tier.clone(),
                declared: manifest.quant.group,
                supported: crate::quant::GROUP_SIZE,
            });
        }
        // (1a) Every component this engine needs must be listed. Extra rows are fine — a future
        // converter may split a component further — but they are still validated below.
        for component in Ltx25Component::all() {
            self.row(*component)?;
        }

        let mut report = Ltx25TierReport {
            tier: manifest.tier.clone(),
            checked: Vec::new(),
            skipped: Vec::new(),
        };
        for row in &manifest.components {
            // (1b) Listed and shipped. Resolved through the manifest's own `file`, never a guess.
            let path = self.dir.join(&row.file);
            if !path.is_file() {
                return Err(Ltx25TierError::MissingComponentFile {
                    component: row.name.clone(),
                    path,
                });
            }
            let headers = safetensors_path_tensor_headers(&path).map_err(|error| {
                Ltx25TierError::UnreadableComponent {
                    component: row.name.clone(),
                    path: path.clone(),
                    detail: error.to_string(),
                }
            })?;
            // (2) Tensor count. A file substituted from another tier usually differs here; when it
            // does not (q4 and q8 transformers hold the same 6779), check (4) still catches it.
            if headers.len() != row.tensors {
                return Err(Ltx25TierError::TensorCountMismatch {
                    component: row.name.clone(),
                    path,
                    declared: row.tensors,
                    actual: headers.len(),
                });
            }
            let packed = self.check_packed(row, &path, &headers)?;
            // (3) Packed-Linear count.
            if packed != row.quantized_linears {
                return Err(Ltx25TierError::PackedCountMismatch {
                    component: row.name.clone(),
                    path,
                    declared: row.quantized_linears,
                    actual: packed,
                });
            }
            // (7) The whole-pipeline tier contract: dense inside a quantized tier needs a reason.
            if manifest.quantized && packed == 0 {
                let declared = row
                    .dense_reason
                    .as_deref()
                    .zip(row.dense_reason_detail.as_deref());
                if declared.is_none() {
                    return Err(Ltx25TierError::UndeclaredDenseComponent {
                        component: row.name.clone(),
                        tier: manifest.tier.clone(),
                    });
                }
            }
            self.check_stamps(row, &path, &mut report)?;
            report.checked.push(Ltx25CheckedComponent {
                name: row.name.clone(),
                tensors: headers.len(),
                packed,
                dense_reason: row.dense_reason.clone(),
                known: row.component().is_some(),
            });
        }
        Ok(report)
    }

    /// Walk one component's packed triples, returning how many it holds.
    ///
    /// Checks (4), (5) and (8) live here because they are per-tensor facts: the count they feed is
    /// only meaningful once every triple has been proved complete and correctly shaped.
    fn check_packed(
        &self,
        row: &Ltx25ManifestComponent,
        path: &Path,
        headers: &[SafetensorsTensorHeader],
    ) -> Result<usize, Ltx25TierError> {
        let by_name: BTreeMap<&str, &SafetensorsTensorHeader> =
            headers.iter().map(|h| (h.name.as_str(), h)).collect();
        let mut packed = 0_usize;
        for header in headers {
            let Some(base) = header.name.strip_suffix(".scales") else {
                continue;
            };
            // (8) The dense tier packs nothing, anywhere.
            let Some(quant) = self.manifest.quant() else {
                return Err(Ltx25TierError::PackedTensorInDenseTier {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    tensor: header.name.clone(),
                });
            };
            packed += 1;
            // (5) A triple is `{base}.weight` + `{base}.scales` + `{base}.biases`. A missing leg
            // would otherwise surface as a `get_unchecked` failure mid-load, after the caller has
            // already committed to building a pipeline.
            let weight = by_name.get(format!("{base}.weight").as_str()).copied();
            let biases = by_name.get(format!("{base}.biases").as_str()).copied();
            let (Some(weight), Some(biases)) = (weight, biases) else {
                return Err(Ltx25TierError::IncompletePackedTriple {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    base: base.to_string(),
                    has_weight: weight.is_some(),
                    has_biases: biases.is_some(),
                });
            };
            // (5) The codes are bit-packed, not numeric. Stored at a float dtype they would be
            // *readable* — and every value would be wrong.
            if weight.dtype != Dtype::U32 {
                return Err(Ltx25TierError::PackedWeightDtype {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    tensor: weight.name.clone(),
                    dtype: weight.dtype.as_str(),
                });
            }
            if biases.shape != header.shape {
                return Err(Ltx25TierError::PackedTripleShape {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    base: base.to_string(),
                    shapes: PackedShapes::boxed(&header.shape, &biases.shape),
                });
            }
            // (4) The declared geometry, asserted against this triple's own shapes.
            //
            // `weight` is `[out, in·bits/32]` and `scales` is `[out, in/group]`, so
            // `32·weight_cols == bits·group·scales_cols` for every packed weight in the tier — an
            // exact integer identity, no tolerance and no inference.
            //
            // What it pins, precisely: the **product** `bits·group`, not `bits` and `group`
            // independently. A q8/group-32 triple satisfies a q4/group-64 declaration exactly
            // (8·32 == 4·64), and this check alone would pass it. What makes it decisive for every
            // tier this loader will actually see is check (9), which already ran and pinned
            // `group` to `crate::quant::GROUP_SIZE`: with `group` fixed, the product determines
            // `bits`. A future tier at a different group has to widen (9) first, and the pairing
            // must be re-argued there rather than inherited from here.
            //
            // Within that pinned group, a q8 file inside a q4 tier doubles the left side alone and
            // is refused here, having passed every count above.
            let (Some(&out_w), Some(&cols_w)) = (weight.shape.first(), weight.shape.get(1)) else {
                return Err(Ltx25TierError::PackedTripleShape {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    base: base.to_string(),
                    shapes: PackedShapes::boxed(&header.shape, &weight.shape),
                });
            };
            let (Some(&out_s), Some(&cols_s)) = (header.shape.first(), header.shape.get(1)) else {
                return Err(Ltx25TierError::PackedTripleShape {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    base: base.to_string(),
                    shapes: PackedShapes::boxed(&header.shape, &biases.shape),
                });
            };
            if out_w != out_s || 32 * cols_w != quant.bits * quant.group * cols_s {
                return Err(Ltx25TierError::PackedGeometryMismatch {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    base: base.to_string(),
                    declared: quant,
                    shapes: PackedShapes::boxed(&header.shape, &weight.shape),
                });
            }
        }
        Ok(packed)
    }

    /// Check (6): the tier stamp and the model version this component's own `__metadata__` carries.
    fn check_stamps(
        &self,
        row: &Ltx25ManifestComponent,
        path: &Path,
        report: &mut Ltx25TierReport,
    ) -> Result<(), Ltx25TierError> {
        let metadata = LtxCheckpointMetadata::from_file(path).map_err(|error| {
            Ltx25TierError::UnreadableComponent {
                component: row.name.clone(),
                path: path.to_path_buf(),
                detail: error.to_string(),
            }
        })?;
        match metadata.raw().get(TIER_METADATA_KEY) {
            None => {
                return Err(Ltx25TierError::TierStampMissing {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    expected: self.manifest.tier.clone(),
                })
            }
            Some(stamped) if *stamped != self.manifest.tier => {
                return Err(Ltx25TierError::TierStampMismatch {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    expected: self.manifest.tier.clone(),
                    stamped: stamped.clone(),
                })
            }
            Some(_) => {}
        }
        match metadata.model_version() {
            Some(version) if version != self.manifest.model_version => {
                return Err(Ltx25TierError::ModelVersionMismatch {
                    component: row.name.clone(),
                    path: path.to_path_buf(),
                    expected: self.manifest.model_version.clone(),
                    declared: version.to_string(),
                })
            }
            Some(_) => {}
            // The packed text encoder is upstream's own file re-stamped, and upstream stamps it
            // `format`/`gemma_config` only. Recorded rather than waved through, so a green run says
            // out loud which components did not answer this question.
            None => report.skipped.push(format!(
                "{}: no `model_version` in __metadata__ — version agreement not checked for this \
                 component (its `{TIER_METADATA_KEY}` stamp was)",
                row.name
            )),
        }
        Ok(())
    }
}

/// One component as [`Ltx25Tier::validate`] found it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx25CheckedComponent {
    /// The manifest id.
    pub name: String,
    /// Tensors in the file (equal to the manifest's declaration, or validation would have failed).
    pub tensors: usize,
    /// Packed triples in the file.
    pub packed: usize,
    /// The declared reason this component is dense, when it is.
    pub dense_reason: Option<String>,
    /// Whether [`Ltx25Component`] names this component. A `false` here is not a fault — the row was
    /// fully validated — but it is worth printing, because this engine will not load it.
    pub known: bool,
}

/// What [`Ltx25Tier::validate`] checked, and what it could not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx25TierReport {
    /// The tier id the manifest declares.
    pub tier: String,
    /// One row per validated component.
    pub checked: Vec<Ltx25CheckedComponent>,
    /// Checks that were legitimately skipped, each saying which component and why.
    pub skipped: Vec<String>,
}

impl Ltx25TierReport {
    /// Total packed triples across the bundle.
    pub fn packed_total(&self) -> usize {
        self.checked.iter().map(|c| c.packed).sum()
    }

    /// The components that are dense, paired with the reason each declares (`None` on a dense
    /// tier, where nothing has to justify itself).
    pub fn dense_components(&self) -> Vec<(&str, Option<&str>)> {
        self.checked
            .iter()
            .filter(|c| c.packed == 0)
            .map(|c| (c.name.as_str(), c.dense_reason.as_deref()))
            .collect()
    }
}

/// The two shapes a packed-triple refusal compares.
///
/// Boxed inside [`Ltx25TierError`] so one oversized variant does not set the size of every
/// `Result` in this module — the shapes are only read when a refusal is being explained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedShapes {
    /// The `.scales` grid's shape.
    pub scales: Vec<usize>,
    /// The shape compared against it — the packed weight, or the `.biases` leg.
    pub compared: Vec<usize>,
}

impl PackedShapes {
    fn boxed(scales: &[usize], compared: &[usize]) -> Box<Self> {
        Box::new(PackedShapes {
            scales: scales.to_vec(),
            compared: compared.to_vec(),
        })
    }
}

/// A refusal from [`Ltx25Tier::validate`] (or from resolving one of its components).
///
/// Typed rather than a formatted string so a caller — and the validation tests — can assert on the
/// fault that occurred instead of on the wording that describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ltx25TierError {
    /// The manifest lists no row for a component this engine needs.
    MissingComponentEntry {
        /// The component id.
        component: String,
        /// The tier that is missing it.
        tier: String,
        /// The manifest that should have listed it.
        manifest: PathBuf,
    },
    /// The manifest lists the component, and the file it names is not there.
    MissingComponentFile {
        /// The component id.
        component: String,
        /// The path the manifest named.
        path: PathBuf,
    },
    /// A component file exists and its safetensors header could not be read.
    UnreadableComponent {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The underlying reader's message.
        detail: String,
    },
    /// The file holds a different number of tensors than the manifest declares.
    TensorCountMismatch {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// What the manifest declared.
        declared: usize,
        /// What the file holds.
        actual: usize,
    },
    /// The file holds a different number of packed Linears than the manifest declares.
    PackedCountMismatch {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// What the manifest declared.
        declared: usize,
        /// What the file holds.
        actual: usize,
    },
    /// A packed triple is missing its weight or its biases.
    IncompletePackedTriple {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The triple's base key.
        base: String,
        /// Whether `{base}.weight` was present.
        has_weight: bool,
        /// Whether `{base}.biases` was present.
        has_biases: bool,
    },
    /// A packed weight is stored at a dtype other than `U32`.
    PackedWeightDtype {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The tensor.
        tensor: String,
        /// The dtype it was stored at.
        dtype: &'static str,
    },
    /// A packed triple's legs disagree on shape, or are not rank 2.
    PackedTripleShape {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The triple's base key.
        base: String,
        /// The `.scales` shape and the leg that disagreed with it.
        shapes: Box<PackedShapes>,
    },
    /// A packed triple's shapes do not match the tier's declared bits/group.
    PackedGeometryMismatch {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The triple's base key.
        base: String,
        /// The geometry the manifest declared.
        declared: Ltx25Quant,
        /// The `.scales` grid's shape and the packed weight's.
        shapes: Box<PackedShapes>,
    },
    /// A packed tensor was found inside a tier the manifest declares dense.
    PackedTensorInDenseTier {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The packed tensor.
        tensor: String,
    },
    /// A component carries no `sceneworks_tier` stamp.
    TierStampMissing {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The tier it should have been stamped with.
        expected: String,
    },
    /// A component is stamped with a different tier than the one it sits in.
    TierStampMismatch {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The tier the manifest declares.
        expected: String,
        /// The tier the file is stamped with.
        stamped: String,
    },
    /// A component declares a different `model_version` than the rest of the bundle.
    ModelVersionMismatch {
        /// The component id.
        component: String,
        /// The file.
        path: PathBuf,
        /// The manifest's version.
        expected: String,
        /// The file's version.
        declared: String,
    },
    /// A component is dense inside a quantized tier and does not say why.
    UndeclaredDenseComponent {
        /// The component id.
        component: String,
        /// The tier it is dense in.
        tier: String,
    },
    /// The manifest's affine group width is not the one the packed loaders repack at.
    UnsupportedGroupSize {
        /// The tier id.
        tier: String,
        /// The manifest's group width.
        declared: usize,
        /// The width the loaders repack at.
        supported: usize,
    },
}

impl std::fmt::Display for Ltx25TierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ltx25TierError::MissingComponentEntry {
                component,
                tier,
                manifest,
            } => write!(
                f,
                "ltx 2.5 tier: the `{tier}` manifest {} lists no `{component}` component — this \
                 engine needs all {} components of a 2.5 bundle",
                manifest.display(),
                Ltx25Component::all().len(),
            ),
            Ltx25TierError::MissingComponentFile { component, path } => write!(
                f,
                "ltx 2.5 tier: the manifest names `{component}` as {} and that file is not there",
                path.display()
            ),
            Ltx25TierError::UnreadableComponent {
                component,
                path,
                detail,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` at {} could not be read: {detail}",
                path.display()
            ),
            Ltx25TierError::TensorCountMismatch {
                component,
                path,
                declared,
                actual,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) holds {actual} tensors, but the manifest declares \
                 {declared} — this is not the file the tier was built from",
                path.display()
            ),
            Ltx25TierError::PackedCountMismatch {
                component,
                path,
                declared,
                actual,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) holds {actual} packed Linear(s), but the manifest \
                 declares {declared}. A dense file in a quantized tier (or the reverse) loads and \
                 runs at the WRONG precision for the whole pipeline",
                path.display()
            ),
            Ltx25TierError::IncompletePackedTriple {
                component,
                path,
                base,
                has_weight,
                has_biases,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) has `{base}.scales` but \
                 weight={has_weight} biases={has_biases} — an affine triple needs all three legs",
                path.display()
            ),
            Ltx25TierError::PackedWeightDtype {
                component,
                path,
                tensor,
                dtype,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) stores the packed weight `{tensor}` as {dtype}, \
                 not U32 — bit-packed codes read as floats decode to noise",
                path.display()
            ),
            Ltx25TierError::PackedTripleShape {
                component,
                path,
                base,
                shapes,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) packed triple `{base}` is mis-shaped: scales \
                 {:?} against {:?}",
                path.display(),
                shapes.scales,
                shapes.compared,
            ),
            Ltx25TierError::PackedGeometryMismatch {
                component,
                path,
                base,
                declared,
                shapes,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) packed triple `{base}` is weight {:?} / scales \
                 {:?}, which is not {} bits at group {} (the manifest's declaration). Repacking it \
                 under the declared geometry would decode the weights into noise",
                path.display(),
                shapes.compared,
                shapes.scales,
                declared.bits,
                declared.group,
            ),
            Ltx25TierError::PackedTensorInDenseTier {
                component,
                path,
                tensor,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) carries the packed tensor `{tensor}`, but the \
                 manifest declares this tier dense",
                path.display()
            ),
            Ltx25TierError::TierStampMissing {
                component,
                path,
                expected,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) carries no `{TIER_METADATA_KEY}` stamp; every \
                 component of the `{expected}` tier is stamped with the tier it was built for",
                path.display()
            ),
            Ltx25TierError::TierStampMismatch {
                component,
                path,
                expected,
                stamped,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) is stamped `{stamped}` but sits in the \
                 `{expected}` tier — a component from another tier",
                path.display()
            ),
            Ltx25TierError::ModelVersionMismatch {
                component,
                path,
                expected,
                declared,
            } => write!(
                f,
                "ltx 2.5 tier: `{component}` ({}) declares model_version {declared:?} but the \
                 bundle is {expected:?}; every component must come from the same release",
                path.display()
            ),
            Ltx25TierError::UndeclaredDenseComponent { component, tier } => write!(
                f,
                "ltx 2.5 tier: `{component}` is dense inside the quantized `{tier}` tier and \
                 declares no `dense_reason`. A tier is a whole-pipeline contract: a component that \
                 stays dense must say why (no-linear-weights / no-mlx-port / below-quality-bar), or \
                 the bundle is only nominally `{tier}`"
            ),
            Ltx25TierError::UnsupportedGroupSize {
                tier,
                declared,
                supported,
            } => write!(
                f,
                "ltx 2.5 tier: the `{tier}` manifest declares group_size {declared}, but the packed \
                 loaders repack at {supported}; the MLX→GGML repack would mis-align"
            ),
        }
    }
}

impl std::error::Error for Ltx25TierError {}

impl From<Ltx25TierError> for CandleError {
    fn from(error: Ltx25TierError) -> CandleError {
        CandleError::Msg(error.to_string())
    }
}

/// Rewrite the two DiffVAE latent-statistic keys from the released spelling to the tier's.
///
/// Exact whole-key matches, not a suffix rewrite: the decoder asks for exactly these two keys, and
/// matching on `-of-means` would rewrite any future key ending the same way.
fn remap_diff_vae_stat_key(key: &str) -> String {
    match key {
        crate::diff_vae::STAT_MEAN_KEY => "per_channel_statistics.mean".to_string(),
        crate::diff_vae::STAT_STD_KEY => "per_channel_statistics.std".to_string(),
        other => other.to_string(),
    }
}

/// Read and JSON-parse a tier manifest. Errors only on "cannot read" / "not JSON" — every schema
/// question is the caller's, so [`Ltx25Tier::detect`] can ask the version question first.
fn read_manifest_json(path: &Path) -> CResult<serde_json::Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| CandleError::Msg(format!("ltx 2.5 tier: read {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| CandleError::Msg(format!("ltx 2.5 tier: parse {}: {e}", path.display())))
}

/// A required string field. Present-but-not-a-string is an error, never a default.
fn json_string(source: &Path, value: &serde_json::Value, key: &str) -> CResult<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            CandleError::Msg(format!(
                "ltx 2.5 tier: {} has no string `{key}`",
                source.display()
            ))
        })
}

/// An optional string field. Absent and JSON `null` both read as absent (the null-is-absent
/// convention the rest of the LTX readers apply); any other non-string is an error.
fn optional_json_string(
    source: &Path,
    value: &serde_json::Value,
    key: &str,
) -> CResult<Option<String>> {
    match value.get(key) {
        None => Ok(None),
        Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text.clone())),
        Some(other) => Err(CandleError::Msg(format!(
            "ltx 2.5 tier: {} has `{key}` = {other}, which is not a string",
            source.display()
        ))),
    }
}

/// A required unsigned integer, accepted as a JSON number **or** as a decimal string.
///
/// The string form is not hypothetical: safetensors types every `__metadata__` value as a string,
/// and converters that round-trip a manifest through that representation (the lightx2v LoRA packs
/// are the case on record) emit `"4"` where the schema says `4`. Rejecting it would refuse a
/// bundle that is entirely well-formed; silently defaulting would pick a geometry.
fn json_usize(source: &Path, value: &serde_json::Value, key: &str) -> CResult<usize> {
    let field = value.get(key).ok_or_else(|| {
        CandleError::Msg(format!("ltx 2.5 tier: {} has no `{key}`", source.display()))
    })?;
    let parsed = match field {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    };
    parsed.and_then(|n| usize::try_from(n).ok()).ok_or_else(|| {
        CandleError::Msg(format!(
            "ltx 2.5 tier: {} has `{key}` = {field}, which is not a non-negative integer",
            source.display()
        ))
    })
}

/// A required boolean, accepted as a JSON bool **or** as the string `"true"` / `"false"` — same
/// round-tripping reason as [`json_usize`].
fn json_bool(source: &Path, value: &serde_json::Value, key: &str) -> CResult<bool> {
    let field = value.get(key).ok_or_else(|| {
        CandleError::Msg(format!("ltx 2.5 tier: {} has no `{key}`", source.display()))
    })?;
    match field {
        serde_json::Value::Bool(b) => Ok(*b),
        serde_json::Value::String(text) => match text.trim() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(CandleError::Msg(format!(
                "ltx 2.5 tier: {} has `{key}` = {field}, which is not a boolean",
                source.display()
            ))),
        },
        _ => Err(CandleError::Msg(format!(
            "ltx 2.5 tier: {} has `{key}` = {field}, which is not a boolean",
            source.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_f32(path: &Path, key: &str, shape: Vec<usize>) {
        let data = vec![0_u8; shape.iter().product::<usize>() * std::mem::size_of::<f32>()];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, &data)
            .expect("valid fixture view");
        safetensors::serialize_to_file(vec![(key, view)], None, path)
            .expect("write fixture safetensors");
    }

    /// The transformer remap turns every crate DiT key into the audited tier spelling (sc-9545) — the
    /// exact `to_out.0`/`ff.net.*`/`linear_*` → `to_out`/`proj_in`/`proj_out`/`linear1/2` rewrites the
    /// hf-header audit of `SceneWorks/ltx-2.3-mlx` q4 found, plus the `model.diffusion_model.` strip.
    #[test]
    fn transformer_remap_matches_real_tier_layout() {
        // attn `to_out.0` (packed triple + dense bias) → `to_out`.
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.scales"
            ),
            "transformer_blocks.0.attn1.to_out.scales"
        );
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight"
            ),
            "transformer_blocks.0.attn1.to_out.weight"
        );
        // FeedForward `net.0.proj` / `net.2` → `proj_in` / `proj_out`.
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.5.ff.net.0.proj.scales"
            ),
            "transformer_blocks.5.ff.proj_in.scales"
        );
        assert_eq!(
            remap_transformer_key("model.diffusion_model.transformer_blocks.5.ff.net.2.weight"),
            "transformer_blocks.5.ff.proj_out.weight"
        );
        // audio_ff too (the `.net.*` rewrite is prefix-agnostic).
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.5.audio_ff.net.0.proj.bias"
            ),
            "transformer_blocks.5.audio_ff.proj_in.bias"
        );
        // AdaLayerNormSingle timestep MLP linear_1/2 → linear1/2.
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"
            ),
            "adaln_single.emb.timestep_embedder.linear1.weight"
        );
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.bias"
            ),
            "adaln_single.emb.timestep_embedder.linear2.bias"
        );
        // Un-nested keys (patchify_proj, to_q, scale_shift_table, gate) only lose the prefix.
        assert_eq!(
            remap_transformer_key("model.diffusion_model.patchify_proj.weight"),
            "patchify_proj.weight"
        );
        assert_eq!(
            remap_transformer_key("model.diffusion_model.keyframes_abs_pos_embedding"),
            "keyframes_abs_pos_embedding"
        );
        assert_eq!(
            remap_transformer_key("model.diffusion_model.transformer_blocks.0.attn1.to_q.scales"),
            "transformer_blocks.0.attn1.to_q.scales"
        );
        assert_eq!(
            remap_transformer_key(
                "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight"
            ),
            "transformer_blocks.0.attn1.to_gate_logits.weight"
        );
    }

    /// The connector prefix strip leaves the crate-native connector spelling intact (the tier connector
    /// file uses `to_out.0` / `ff.net.*` natively, so ONLY the `model.diffusion_model.` prefix is removed
    /// — the transformer projection renames must NOT be applied here).
    #[test]
    fn connector_strip_keeps_native_spelling() {
        assert_eq!(
            strip_diffusion_prefix(
                "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight"
            ),
            "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight"
        );
        assert_eq!(
            strip_diffusion_prefix(
                "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight"
            ),
            "audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight"
        );
        // The text projection is already at the file root (no prefix to strip).
        assert_eq!(
            strip_diffusion_prefix("text_embedding_projection.video_aggregate_embed.weight"),
            "text_embedding_projection.video_aggregate_embed.weight"
        );
    }

    /// The VAE remap drops the `vae.`/`vae.decoder.` wrapper, renames the stats, and flags conv weights
    /// for the channels-last→torch permute — matching the audited tier `vae_decoder.safetensors` layout.
    #[test]
    fn vae_remap_matches_real_tier_layout() {
        let (k, conv) =
            VaeRemapBackend::remap("vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight");
        assert_eq!(k, "up_blocks.0.res_blocks.0.conv1.conv.weight");
        assert!(conv, "a `.conv.weight` must be flagged for the permute");
        let (k, conv) = VaeRemapBackend::remap("vae.decoder.conv_in.conv.bias");
        assert_eq!(k, "conv_in.conv.bias");
        assert!(!conv, "a bias is not permuted");
        let (k, _) = VaeRemapBackend::remap("vae.per_channel_statistics.mean-of-means");
        assert_eq!(k, "per_channel_statistics.mean");
        let (k, _) = VaeRemapBackend::remap("vae.per_channel_statistics.std-of-means");
        assert_eq!(k, "per_channel_statistics.std");
    }

    #[test]
    fn vae_encoder_remap_matches_real_tier_layout() {
        let (k, conv) = VaeEncoderRemapBackend::remap("vae.encoder.down_blocks.1.conv.conv.weight");
        assert_eq!(k, "down_blocks.1.conv.conv.weight");
        assert!(conv);
        let (k, _) = VaeEncoderRemapBackend::remap("vae.per_channel_statistics.mean-of-means");
        assert_eq!(k, "per_channel_statistics._mean_of_means");
        let (k, _) = VaeEncoderRemapBackend::remap("vae.per_channel_statistics.std-of-means");
        assert_eq!(k, "per_channel_statistics._std_of_means");
    }

    #[test]
    fn audio_vae_remap_matches_converted_component_layout() {
        let (key, conv) =
            AudioVaeRemapBackend::remap("audio_vae.decoder.up.2.block.0.conv1.conv.weight");
        assert_eq!(key, "up.2.block.0.conv1.conv.weight");
        assert!(conv);
        let (key, conv) = AudioVaeRemapBackend::remap("audio_vae.decoder.conv_out.conv.bias");
        assert_eq!(key, "conv_out.conv.bias");
        assert!(!conv);
        let (key, _) =
            AudioVaeRemapBackend::remap("audio_vae.per_channel_statistics.mean-of-means");
        assert_eq!(key, "per_channel_statistics._mean_of_means");
        let (key, _) = AudioVaeRemapBackend::remap("audio_vae.per_channel_statistics.std-of-means");
        assert_eq!(key, "per_channel_statistics._std_of_means");
    }

    #[test]
    fn vocoder_remap_matches_converted_component_layout() {
        let (key, weight, transposed) =
            VocoderRemapBackend::remap("vocoder.vocoder.conv_pre.weight");
        assert_eq!(key, "conv_pre.weight");
        assert!(weight);
        assert!(!transposed);
        let (key, weight, transposed) =
            VocoderRemapBackend::remap("vocoder.bwe_generator.ups.0.weight");
        assert_eq!(key, "bwe_generator.ups.0.weight");
        assert!(weight);
        assert!(transposed);
        let (key, weight, _) = VocoderRemapBackend::remap("vocoder.mel_stft.mel_basis");
        assert_eq!(key, "mel_stft.mel_basis");
        assert!(!weight);
    }

    /// The permute turns a tier channels-last conv `[O,kt,kh,kw,I]` into the crate `[O,I,kt,kh,kw]`.
    #[test]
    fn conv_permute_channels_last_to_torch() -> candle_gen::candle_core::Result<()> {
        use candle_gen::candle_core::Device;
        // [O=2, kt=3, kh=3, kw=3, I=4] → [2,4,3,3,3].
        let w = Tensor::arange(0f32, (2 * 3 * 3 * 3 * 4) as f32, &Device::Cpu)?
            .reshape((2, 3, 3, 3, 4))?;
        let p = VaeRemapBackend::permute_conv(w.clone())?;
        assert_eq!(p.dims(), &[2, 4, 3, 3, 3]);
        // Spot-check via flat buffers: p[o,i,t,h,ww] == w[o,t,h,ww,i].
        let wv = w.flatten_all()?.to_vec1::<f32>()?; // strides [O]:108 [kt]:36 [kh]:12 [kw]:4 [I]:1
        let pv = p.flatten_all()?.to_vec1::<f32>()?; // strides [O]:108 [I]:27 [kt]:9 [kh]:3 [kw]:1
        let w_idx = |o: usize, t: usize, h: usize, ww: usize, i: usize| {
            o * 108 + t * 36 + h * 12 + ww * 4 + i
        };
        let p_idx = |o: usize, i: usize, t: usize, h: usize, ww: usize| {
            o * 108 + i * 27 + t * 9 + h * 3 + ww
        };
        for o in 0..2 {
            for i in 0..4 {
                for t in 0..3 {
                    for h in 0..3 {
                        for ww in 0..3 {
                            assert_eq!(pv[p_idx(o, i, t, h, ww)], wv[w_idx(o, t, h, ww, i)]);
                        }
                    }
                }
            }
        }
        // The same converter also stores Conv2d `[O,kh,kw,I]`; audio/video bridges share this
        // inverse and must recover `[O,I,kh,kw]`.
        let w2 = Tensor::zeros((2, 3, 5, 4), DType::F32, &Device::Cpu)?;
        let p2 = VaeRemapBackend::permute_conv(w2)?;
        assert_eq!(p2.dims(), &[2, 4, 3, 5]);
        Ok(())
    }

    #[test]
    fn vocoder_permute_restores_conv_and_transposed_conv_layouts(
    ) -> candle_gen::candle_core::Result<()> {
        use candle_gen::candle_core::Device;

        let conv = Tensor::zeros((8, 3, 4), DType::F32, &Device::Cpu)?;
        assert_eq!(
            VocoderRemapBackend::permute_weight(conv, false)?.dims(),
            &[8, 4, 3]
        );

        let transposed = Tensor::zeros((6, 5, 7), DType::F32, &Device::Cpu)?;
        assert_eq!(
            VocoderRemapBackend::permute_weight(transposed, true)?.dims(),
            &[7, 6, 5]
        );
        Ok(())
    }

    /// Regression for the terminal Blackwell failure: production builders must apply the inverse
    /// converter namespace/layout, not ask a rootless component for released prefixed keys.
    #[test]
    fn converted_component_builders_reach_rootless_tensors() -> CResult<()> {
        let dir = tempfile::tempdir().expect("tempdir");
        let device = Device::Cpu;

        let transformer = dir.path().join("transformer.safetensors");
        write_f32(&transformer, "keyframes_abs_pos_embedding", vec![1, 4]);
        let dit =
            ltx25_transformer_vb(&transformer, DType::F32, &device)?.pp("model.diffusion_model");
        assert_eq!(
            dit.get_unchecked("keyframes_abs_pos_embedding")?.dims(),
            &[1, 4]
        );

        let audio = dir.path().join("audio_vae.safetensors");
        write_f32(&audio, "conv_in.conv.weight", vec![2, 3, 5, 4]);
        let audio = ltx25_audio_vae_vb(&audio, DType::F32, &device)?.pp("audio_vae");
        assert_eq!(
            audio.get_unchecked("decoder.conv_in.conv.weight")?.dims(),
            &[2, 4, 3, 5]
        );

        let vocoder = dir.path().join("vocoder.safetensors");
        write_f32(&vocoder, "conv_pre.weight", vec![8, 3, 4]);
        let vocoder = ltx25_vocoder_vb(&vocoder, DType::F32, &device)?;
        assert_eq!(
            vocoder
                .get_unchecked("vocoder.vocoder.conv_pre.weight")?
                .dims(),
            &[8, 4, 3]
        );
        Ok(())
    }
}
