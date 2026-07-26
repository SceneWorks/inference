//! Explicit Stable Audio 3 snapshot layout and safetensors namespace routing.
//!
//! There is no repository-id resolver, network client, or cache-path derivation here. A caller
//! supplies a [`crate::gen_core::WeightsSource::Dir`] containing one immutable snapshot.
//!
//! The upstream files use two key layouts:
//!
//! | Snapshot | File prefix | Candle component prefix |
//! |---|---|---|
//! | full SA3 | `pretransform.model.encoder.` | `pretransform.model.encoder` |
//! | full SA3 | `pretransform.model.decoder.` | `pretransform.model.decoder` |
//! | full SA3 | `pretransform.model.bottleneck.` | `pretransform.model.bottleneck` |
//! | full SA3 | `model.` | `model` (DiT) |
//! | full SA3 | `conditioner.` | `conditioner` |
//! | standalone SAME | `encoder.` / `decoder.` / `bottleneck.` | matching root |
//!
//! `svd_bases.pt` is intentionally absent from the required-file list: some base repositories
//! carry that training artifact, but no inference component consumes it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_audio::candle_core::safetensors::MmapedSafetensors;
use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core::{LoadSpec, WeightsSource};
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;

use crate::config::{ModelConfig, StableAudioConfig};

pub const CONFIG_FILE: &str = "model_config.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const TEXT_ENCODER_DIR: &str = "t5gemma-b-b-ul2";
pub const TEXT_CONFIG_FILE: &str = "config.json";
pub const TEXT_WEIGHTS_FILE: &str = "model.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const OPTIONAL_TOKENIZER_MODEL: &str = "tokenizer.model";
const METAL_WEIGHT_PACK_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotKind {
    Full,
    StandaloneAutoencoder,
}

/// Fully recognized snapshot. Every path is explicit and rooted at the caller's directory.
#[derive(Debug, Clone)]
pub struct SnapshotLayout {
    pub root: PathBuf,
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
    pub text_config_path: Option<PathBuf>,
    pub text_weights_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub tokenizer_model_path: Option<PathBuf>,
    pub config: StableAudioConfig,
    pub kind: SnapshotKind,
    pub keys: KeyMapSummary,
    pub text_keys: Option<TextWeightSummary>,
}

impl SnapshotLayout {
    pub fn from_load_spec(spec: &LoadSpec) -> Result<Self> {
        Self::from_weights(&spec.weights)
    }

    pub fn from_weights(source: &WeightsSource) -> Result<Self> {
        let root = match source {
            WeightsSource::Dir(root) => root,
            WeightsSource::File(path) => {
                return Err(AudioError::Msg(format!(
                    "Stable Audio 3 requires WeightsSource::Dir, got file {}",
                    path.display()
                )));
            }
        };
        Self::from_dir(root)
    }

    pub fn from_dir(root: &Path) -> Result<Self> {
        if !root.is_dir() {
            return Err(AudioError::Msg(format!(
                "Stable Audio 3 snapshot directory does not exist: {}",
                root.display()
            )));
        }
        let config_path = require_file(root, CONFIG_FILE)?;
        let weights_path = require_file(root, WEIGHTS_FILE)?;
        let config = StableAudioConfig::from_path(&config_path)?;
        let kind = match &config.model {
            ModelConfig::Diffusion(_) => SnapshotKind::Full,
            ModelConfig::Autoencoder(_) => SnapshotKind::StandaloneAutoencoder,
        };

        let (text_config_path, text_weights_path, tokenizer_path, tokenizer_model_path) =
            if kind == SnapshotKind::Full {
                let text_root = root.join(TEXT_ENCODER_DIR);
                let config = require_file(&text_root, TEXT_CONFIG_FILE)?;
                let weights = require_file(&text_root, TEXT_WEIGHTS_FILE)?;
                let tokenizer = require_file(&text_root, TOKENIZER_FILE)?;
                let optional = text_root.join(OPTIONAL_TOKENIZER_MODEL);
                (
                    Some(config),
                    Some(weights),
                    Some(tokenizer),
                    optional.is_file().then_some(optional),
                )
            } else {
                (None, None, None, None)
            };

        let keys = KeyMapSummary::inspect(&weights_path, kind)?;
        let text_keys = text_weights_path
            .as_deref()
            .map(TextWeightSummary::inspect)
            .transpose()?;
        Ok(Self {
            root: root.to_path_buf(),
            config_path,
            weights_path,
            text_config_path,
            text_weights_path,
            tokenizer_path,
            tokenizer_model_path,
            config,
            kind,
            keys,
            text_keys,
        })
    }

    /// Mmap the root checkpoint and return component-scoped Candle builders.
    ///
    /// Like every Candle mmap loader, the snapshot must remain immutable while these builders are
    /// alive. The explicit snapshot contract makes that invariant the caller's responsibility.
    pub fn mmap_builders(
        &self,
        root_dtype: DType,
        device: &Device,
    ) -> Result<StableAudioVarBuilders<'static>> {
        self.mmap_builders_with_text_dtype(root_dtype, DType::BF16, device)
    }

    /// Mmap the F32 SA3 root and bundled text weights with independent compute dtypes.
    ///
    /// Shipped checkpoints use F32 root tensors and BF16 T5Gemma tensors. The ordinary
    /// [`Self::mmap_builders`] path therefore fixes the text side to BF16; this explicit form exists
    /// for tests and callers that need to state both sides.
    pub fn mmap_builders_with_text_dtype(
        &self,
        root_dtype: DType,
        text_dtype: DType,
        device: &Device,
    ) -> Result<StableAudioVarBuilders<'static>> {
        let root = mmap_builder(
            &self.weights_path,
            root_dtype,
            device,
            packs_root_on_metal(self.kind),
        )?;
        let text_encoder = match &self.text_weights_path {
            Some(path) => Some(mmap_builder(path, text_dtype, device, false)?),
            None => None,
        };

        Ok(match self.kind {
            SnapshotKind::Full => StableAudioVarBuilders {
                encoder: root.pp("pretransform.model.encoder"),
                decoder: root.pp("pretransform.model.decoder"),
                bottleneck: root.pp("pretransform.model.bottleneck"),
                dit: Some(root.pp("model")),
                conditioner: Some(root.pp("conditioner")),
                text_encoder,
            },
            SnapshotKind::StandaloneAutoencoder => StableAudioVarBuilders {
                encoder: root.pp("encoder"),
                decoder: root.pp("decoder"),
                bottleneck: root.pp("bottleneck"),
                dit: None,
                conditioner: None,
                text_encoder: None,
            },
        })
    }

    /// Build the complete full-model graph through one shared weight backing.
    ///
    /// The ordinary component loaders stay lazy because standalone callers may request only one
    /// component. A registered full provider consumes every root tensor, so on Metal this path
    /// coalesces the 685 root tensors and the used T5 encoder tensors into bounded shared buffers.
    /// This avoids Metal's persistent-resource ceiling while preserving independently shaped
    /// views. CPU and CUDA retain the zero-copy mmap backend.
    pub fn full_pipeline_builders(
        &self,
        root_dtype: DType,
        text_dtype: DType,
        device: &Device,
    ) -> Result<StableAudioVarBuilders<'static>> {
        if self.kind != SnapshotKind::Full {
            return Err(AudioError::Msg(
                "full pipeline builders require a full Stable Audio 3 snapshot".into(),
            ));
        }
        let metal = matches!(device, Device::Metal(_));
        let root = if metal {
            packed_metal_builder(&self.weights_path, root_dtype, device, None)?
        } else {
            mmap_builder(&self.weights_path, root_dtype, device, false)?
        };
        let text_path = self
            .text_weights_path
            .as_deref()
            .ok_or_else(|| AudioError::Msg("full snapshot has no text weights".into()))?;
        let text_encoder = if metal {
            packed_metal_builder(text_path, text_dtype, device, Some("model.encoder."))?
        } else {
            mmap_builder(text_path, text_dtype, device, false)?
        };
        Ok(StableAudioVarBuilders {
            encoder: root.pp("pretransform.model.encoder"),
            decoder: root.pp("pretransform.model.decoder"),
            bottleneck: root.pp("pretransform.model.bottleneck"),
            dit: Some(root.pp("model")),
            conditioner: Some(root.pp("conditioner")),
            text_encoder: Some(text_encoder),
        })
    }
}

fn packs_root_on_metal(kind: SnapshotKind) -> bool {
    kind == SnapshotKind::StandaloneAutoencoder
}

fn mmap_builder(
    path: &Path,
    dtype: DType,
    device: &Device,
    pack_on_metal: bool,
) -> Result<VarBuilder<'static>> {
    if pack_on_metal && matches!(device, Device::Metal(_)) {
        packed_metal_builder(path, dtype, device, None)
    } else {
        // Safety: callers of SnapshotLayout require an immutable snapshot. The returned backend
        // owns its mmap for the lifetime of the 'static VarBuilder.
        Ok(unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), dtype, device)?
        })
    }
}

/// Coalesce persisted weights into bounded shared-storage packs on Metal.
///
/// Candle's ordinary mmap backend creates one MTLBuffer per requested tensor. Standalone SAME-L
/// has 472 tensors, which exceeds the empirical Metal resource-allocation ceiling before the final
/// tensors load. Packing preserves independently shaped tensor views while reducing persistent
/// Metal resources to a few dozen bounded buffers. Full checkpoints stay on the lazy mmap path so
/// component-only loaders do not eagerly materialize unrelated autoencoder, DiT, or text weights.
fn packed_metal_builder(
    path: &Path,
    dtype: DType,
    device: &Device,
    prefix: Option<&str>,
) -> Result<VarBuilder<'static>> {
    // Safety: SnapshotLayout requires this file to remain immutable for the duration of loading.
    // All values are copied into owned Candle storage before this function returns.
    let safetensors = unsafe { MmapedSafetensors::new(path)? };
    let mut names = safetensors
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| prefix.is_none_or(|prefix| name.starts_with(prefix)))
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err(AudioError::Msg(format!(
            "{} has no tensors matching packed prefix {prefix:?}",
            path.display()
        )));
    }
    names.sort_unstable();

    let max_elements = METAL_WEIGHT_PACK_BYTES / dtype.size_in_bytes();
    let mut tensors = HashMap::with_capacity(names.len());
    let mut pending = Vec::<(String, Tensor)>::new();
    let mut pending_elements = 0usize;

    for name in names {
        let tensor = safetensors.load(&name, &Device::Cpu)?.to_dtype(dtype)?;
        let elements = tensor.elem_count();
        if !pending.is_empty() && pending_elements.saturating_add(elements) > max_elements {
            flush_weight_pack(&mut pending, &mut tensors, device)?;
            pending_elements = 0;
        }
        pending_elements = pending_elements.checked_add(elements).ok_or_else(|| {
            AudioError::Msg(format!("weight pack element count overflows: {name}"))
        })?;
        pending.push((name, tensor));
    }
    flush_weight_pack(&mut pending, &mut tensors, device)?;

    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

fn flush_weight_pack(
    pending: &mut Vec<(String, Tensor)>,
    output: &mut HashMap<String, Tensor>,
    device: &Device,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let flattened = pending
        .iter()
        .map(|(_, tensor)| tensor.flatten_all())
        .collect::<candle_audio::candle_core::Result<Vec<_>>>()?;
    let refs = flattened.iter().collect::<Vec<_>>();
    let packed = Tensor::cat(&refs, 0)?.to_device(device)?;
    let mut offset = 0usize;
    for (name, tensor) in pending.drain(..) {
        let elements = tensor.elem_count();
        let view = packed
            .narrow(0, offset, elements)?
            .reshape(tensor.shape())?;
        output.insert(name, view);
        offset += elements;
    }
    Ok(())
}

/// Exact bundled T5Gemma safetensors inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextWeightSummary {
    pub total: usize,
    pub encoder: usize,
    pub decoder: usize,
    pub encoder_params: usize,
    pub decoder_params: usize,
}

impl TextWeightSummary {
    pub const TOTAL: usize = 340;
    pub const ENCODER: usize = 134;
    pub const DECODER: usize = 206;
    pub const ENCODER_PARAMS: usize = 281_580_288;
    pub const DECODER_PARAMS: usize = 309_910_272;

    pub fn inspect(path: &Path) -> Result<Self> {
        let entries = safetensors_header(path)?;
        let expected: std::collections::BTreeSet<_> =
            crate::t5gemma::encoder_weight_keys().into_iter().collect();
        let mut actual_encoder = std::collections::BTreeSet::new();
        let mut summary = Self {
            total: 0,
            encoder: 0,
            decoder: 0,
            encoder_params: 0,
            decoder_params: 0,
        };
        for (key, value) in entries {
            if key == "__metadata__" {
                continue;
            }
            summary.total += 1;
            let dtype = value
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if dtype != "BF16" {
                return Err(AudioError::Msg(format!(
                    "{} tensor {key} has dtype {dtype}, expected BF16",
                    path.display()
                )));
            }
            let shape = value
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    AudioError::Msg(format!("{} tensor {key} has no shape", path.display()))
                })?;
            let params = shape.iter().try_fold(1usize, |acc, dim| {
                let dim = dim.as_u64().ok_or_else(|| {
                    AudioError::Msg(format!("{} tensor {key} has invalid shape", path.display()))
                })?;
                let dim: usize = dim.try_into().map_err(|_| {
                    AudioError::Msg(format!("{} tensor {key} shape overflows", path.display()))
                })?;
                acc.checked_mul(dim).ok_or_else(|| {
                    AudioError::Msg(format!(
                        "{} tensor {key} parameter count overflows",
                        path.display()
                    ))
                })
            })?;
            if key.starts_with("model.encoder.") {
                summary.encoder += 1;
                summary.encoder_params += params;
                actual_encoder.insert(key);
            } else if key.starts_with("model.decoder.") {
                summary.decoder += 1;
                summary.decoder_params += params;
            } else {
                return Err(AudioError::Msg(format!(
                    "{} has unexpected T5Gemma tensor {key}",
                    path.display()
                )));
            }
        }
        if actual_encoder != expected
            || summary.total != Self::TOTAL
            || summary.encoder != Self::ENCODER
            || summary.decoder != Self::DECODER
            || summary.encoder_params != Self::ENCODER_PARAMS
            || summary.decoder_params != Self::DECODER_PARAMS
        {
            return Err(AudioError::Msg(format!(
                "{} T5Gemma inventory mismatch: {summary:?}",
                path.display()
            )));
        }
        Ok(summary)
    }
}

fn require_file(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = root.join(relative);
    if !path.is_file() {
        return Err(AudioError::Msg(format!(
            "Stable Audio 3 snapshot {} is missing required file {relative}",
            root.display()
        )));
    }
    Ok(path)
}

#[derive(Clone)]
pub struct StableAudioVarBuilders<'a> {
    pub encoder: VarBuilder<'a>,
    pub decoder: VarBuilder<'a>,
    pub bottleneck: VarBuilder<'a>,
    pub dit: Option<VarBuilder<'a>>,
    pub conditioner: Option<VarBuilder<'a>>,
    pub text_encoder: Option<VarBuilder<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSection {
    Encoder,
    Decoder,
    Bottleneck,
    Dit,
    Conditioner,
}

/// One checkpoint key after mapping into a component builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedWeightKey<'a> {
    pub section: WeightSection,
    pub local_key: &'a str,
}

pub fn map_weight_key(kind: SnapshotKind, key: &str) -> Option<MappedWeightKey<'_>> {
    let prefixes: &[(WeightSection, &str)] = match kind {
        SnapshotKind::Full => &[
            (WeightSection::Encoder, "pretransform.model.encoder."),
            (WeightSection::Decoder, "pretransform.model.decoder."),
            (WeightSection::Bottleneck, "pretransform.model.bottleneck."),
            (WeightSection::Dit, "model."),
            (WeightSection::Conditioner, "conditioner."),
        ],
        SnapshotKind::StandaloneAutoencoder => &[
            (WeightSection::Encoder, "encoder."),
            (WeightSection::Decoder, "decoder."),
            (WeightSection::Bottleneck, "bottleneck."),
        ],
    };
    prefixes.iter().find_map(|(section, prefix)| {
        key.strip_prefix(prefix).map(|local_key| MappedWeightKey {
            section: *section,
            local_key,
        })
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyMapSummary {
    pub total: usize,
    pub encoder: usize,
    pub decoder: usize,
    pub bottleneck: usize,
    pub dit: usize,
    pub conditioner: usize,
}

impl KeyMapSummary {
    pub fn inspect(path: &Path, kind: SnapshotKind) -> Result<Self> {
        let keys = safetensors_keys(path)?;
        let mut summary = Self {
            total: keys.len(),
            encoder: 0,
            decoder: 0,
            bottleneck: 0,
            dit: 0,
            conditioner: 0,
        };
        let mut unmapped = Vec::new();
        for key in &keys {
            match map_weight_key(kind, key).map(|mapped| mapped.section) {
                Some(WeightSection::Encoder) => summary.encoder += 1,
                Some(WeightSection::Decoder) => summary.decoder += 1,
                Some(WeightSection::Bottleneck) => summary.bottleneck += 1,
                Some(WeightSection::Dit) => summary.dit += 1,
                Some(WeightSection::Conditioner) => summary.conditioner += 1,
                None => unmapped.push(key.as_str()),
            }
        }
        if !unmapped.is_empty() {
            return Err(AudioError::Msg(format!(
                "{} has {} unmapped Stable Audio 3 keys; first: {:?}",
                path.display(),
                unmapped.len(),
                &unmapped[..unmapped.len().min(5)]
            )));
        }
        let required_nonempty = summary.encoder > 0
            && summary.decoder > 0
            && summary.bottleneck > 0
            && (kind == SnapshotKind::StandaloneAutoencoder
                || (summary.dit > 0 && summary.conditioner > 0));
        if !required_nonempty {
            return Err(AudioError::Msg(format!(
                "{} is missing a required Stable Audio 3 key namespace: {summary:?}",
                path.display()
            )));
        }
        Ok(summary)
    }
}

/// Read only the safetensors header and return its tensor keys.
pub fn safetensors_keys(path: &Path) -> Result<Vec<String>> {
    let object = safetensors_header(path)?;
    let mut keys: Vec<String> = object
        .into_iter()
        .map(|(key, _)| key)
        .filter(|key| key != "__metadata__")
        .collect();
    keys.sort();
    Ok(keys)
}

fn safetensors_header(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut file =
        File::open(path).map_err(|e| AudioError::Msg(format!("open {}: {e}", path.display())))?;
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| AudioError::Msg(format!("read {} safetensors length: {e}", path.display())))?;
    let header_len_u64 = u64::from_le_bytes(len_bytes);
    let header_len: usize = header_len_u64.try_into().map_err(|_| {
        AudioError::Msg(format!(
            "{} safetensors header length does not fit this platform",
            path.display()
        ))
    })?;
    let file_len = file
        .seek(SeekFrom::End(0))
        .map_err(|e| AudioError::Msg(format!("seek {}: {e}", path.display())))?;
    if header_len_u64 > file_len.saturating_sub(8) {
        return Err(AudioError::Msg(format!(
            "{} has a truncated safetensors header",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(8))
        .map_err(|e| AudioError::Msg(format!("seek {} header: {e}", path.display())))?;
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|e| AudioError::Msg(format!("read {} header: {e}", path.display())))?;
    serde_json::from_slice(&header)
        .map_err(|e| AudioError::Msg(format!("parse {} header: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_documented_namespaces_without_overlap() {
        let mapped = map_weight_key(
            SnapshotKind::Full,
            "pretransform.model.encoder.layers.0.mapping.weight",
        )
        .unwrap();
        assert_eq!(mapped.section, WeightSection::Encoder);
        assert_eq!(mapped.local_key, "layers.0.mapping.weight");
        let mapped =
            map_weight_key(SnapshotKind::Full, "model.model.transformer.layers.0.x").unwrap();
        assert_eq!(mapped.section, WeightSection::Dit);
        assert_eq!(mapped.local_key, "model.transformer.layers.0.x");
        assert!(map_weight_key(SnapshotKind::Full, "svd_bases.pt").is_none());

        let mapped = map_weight_key(
            SnapshotKind::StandaloneAutoencoder,
            "decoder.layers.1.weight",
        )
        .unwrap();
        assert_eq!(mapped.section, WeightSection::Decoder);
        assert_eq!(mapped.local_key, "layers.1.weight");
    }

    #[test]
    fn packed_weight_views_preserve_names_shapes_and_values() {
        let device = Device::Cpu;
        let mut pending = vec![
            (
                "encoder.weight".to_owned(),
                Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &device).unwrap(),
            ),
            (
                "bottleneck.scalar".to_owned(),
                Tensor::from_vec(vec![5f32], (), &device).unwrap(),
            ),
            (
                "decoder.weight".to_owned(),
                Tensor::from_vec(vec![6f32, 7., 8.], 3, &device).unwrap(),
            ),
        ];
        let mut packed = HashMap::new();
        flush_weight_pack(&mut pending, &mut packed, &device).unwrap();

        assert!(pending.is_empty());
        assert_eq!(packed.len(), 3);
        assert_eq!(packed["encoder.weight"].dims(), &[2, 2]);
        assert_eq!(
            packed["encoder.weight"]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1., 2., 3., 4.]
        );
        assert_eq!(packed["bottleneck.scalar"].to_scalar::<f32>().unwrap(), 5.);
        assert_eq!(
            packed["decoder.weight"].to_vec1::<f32>().unwrap(),
            vec![6., 7., 8.]
        );
    }

    #[test]
    fn metal_packing_is_scoped_to_standalone_autoencoder_roots() {
        assert!(packs_root_on_metal(SnapshotKind::StandaloneAutoencoder));
        assert!(!packs_root_on_metal(SnapshotKind::Full));
    }
}
