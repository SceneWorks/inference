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

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device};
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
        // Safety: the API contract above requires a caller-provisioned immutable snapshot. The
        // returned backend owns its mmap for the lifetime of the 'static VarBuilder.
        let root = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.weights_path),
                root_dtype,
                device,
            )?
        };
        let text_encoder = match &self.text_weights_path {
            Some(path) => {
                // Safety: same immutable-snapshot invariant as the root checkpoint.
                Some(unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        std::slice::from_ref(path),
                        text_dtype,
                        device,
                    )?
                })
            }
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
}
