//! Exact FLUX3 publisher/native storage identities and physical DiT inventories.
//!
//! Candle GGML Q4_1/Q8_0 (Q8_1 activation compute) are NOT MLX affine W4/8A16
//! artifacts. A shared element width never permits cross-format reinterpretation.
use std::collections::BTreeMap;

use super::quantization::{self, DitFormat, TensorSpec};
use super::DitConfig;
use crate::weightsmeta::{Dtype, SafetensorsTensorHeader};
use crate::{Error, Precision, Quant, Result};

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 artifact: {detail}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactBackend {
    Mlx,
    CandleCpu,
    CandleMetal,
    CandleCuda { major: u32, minor: u32 },
}

/// Storage of ONLY the DiT. The bundled encoder, F32 AE and optional separate
/// PUP retain their individually declared representations and identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeDitStorage {
    Publisher(DitFormat),
    MlxAffine {
        bits: u32,
        group_size: u32,
    },
    /// U8 tensor rows contain little-endian GGML blocks: F16 d,m + 16 code bytes.
    GgmlQ4_1,
    /// U8 tensor rows contain little-endian GGML blocks: F16 d + 32 signed codes.
    GgmlQ8_0,
}

impl NativeDitStorage {
    fn validate_parameters(self) -> Result<()> {
        if let Self::MlxAffine { bits, group_size } = self {
            if !matches!(bits, 4 | 8) || !matches!(group_size, 32 | 64 | 128) {
                return Err(invalid("unsupported MLX affine bits/group parameters"));
            }
        }
        Ok(())
    }

    /// Source null/string markers remain unchanged. A native object MUST name
    /// its exact format, bits, group and version; no scale-dtype/filename guesses.
    pub fn from_marker(marker: &serde_json::Value) -> Result<Self> {
        let Some(object) = marker.as_object() else {
            return DitFormat::from_marker(marker).map(Self::Publisher);
        };
        if object.keys().any(|key| {
            !matches!(
                key.as_str(),
                "format" | "bits" | "group_size" | "algorithm_version"
            )
        }) {
            return Err(invalid("unknown native quantization parameter"));
        }
        let bits = marker["bits"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| invalid("missing native bits"))?;
        let group_size = marker["group_size"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| invalid("missing native group_size"))?;
        if marker["algorithm_version"].as_u64() != Some(1) {
            return Err(invalid("unsupported/missing native algorithm_version"));
        }
        let storage =
            match marker["format"].as_str() {
                Some("mlx-affine") => Self::MlxAffine { bits, group_size },
                Some("ggml-q4_1") if bits == 4 && group_size == 32 => Self::GgmlQ4_1,
                Some("ggml-q8_0") if bits == 8 && group_size == 32 => Self::GgmlQ8_0,
                _ => return Err(invalid(
                    "declare the exact supported native format and group, not just its bit width",
                )),
            };
        storage.validate_parameters()?;
        Ok(storage)
    }

    pub fn marker(self) -> Result<serde_json::Value> {
        self.validate_parameters()?;
        Ok(match self {
            Self::Publisher(DitFormat::Dense) => serde_json::Value::Null,
            Self::Publisher(DitFormat::Fp8Rowwise) => serde_json::json!("fp8r"),
            Self::Publisher(DitFormat::Nvfp4) => serde_json::json!("nvfp4"),
            Self::MlxAffine { bits, group_size } => {
                serde_json::json!({"format":"mlx-affine","bits":bits,"group_size":group_size,"algorithm_version":1})
            }
            Self::GgmlQ4_1 => {
                serde_json::json!({"format":"ggml-q4_1","bits":4,"group_size":32,"algorithm_version":1})
            }
            Self::GgmlQ8_0 => {
                serde_json::json!({"format":"ggml-q8_0","bits":8,"group_size":32,"algorithm_version":1})
            }
        })
    }

    /// A distinct artifact/cache key, never a universal cross-backend q4/q8 label.
    pub fn key(self) -> Result<String> {
        self.validate_parameters()?;
        Ok(match self {
            Self::Publisher(DitFormat::Dense) => "dense".into(),
            Self::Publisher(DitFormat::Fp8Rowwise) => "fp8r".into(),
            Self::Publisher(DitFormat::Nvfp4) => "nvfp4".into(),
            Self::MlxAffine { bits, group_size } => format!("mlx-affine-q{bits}-g{group_size}"),
            Self::GgmlQ4_1 => "ggml-q4_1".into(),
            Self::GgmlQ8_0 => "ggml-q8_0".into(),
        })
    }

    /// Compact identity retained by the historical parsed config. Native group/
    /// bit parameters are validated by this parser and read from the raw marker.
    pub fn config_marker_name(self) -> Option<&'static str> {
        match self {
            Self::Publisher(DitFormat::Dense) => None,
            Self::Publisher(DitFormat::Fp8Rowwise) => Some("fp8r"),
            Self::Publisher(DitFormat::Nvfp4) => Some("nvfp4"),
            Self::MlxAffine { .. } => Some("mlx-affine"),
            Self::GgmlQ4_1 => Some("ggml-q4_1"),
            Self::GgmlQ8_0 => Some("ggml-q8_0"),
        }
    }

    pub fn quant(self) -> Result<Option<Quant>> {
        self.validate_parameters()?;
        Ok(match self {
            Self::Publisher(DitFormat::Dense) => None,
            Self::Publisher(DitFormat::Fp8Rowwise) => Some(Quant::Fp8Rowwise),
            Self::Publisher(DitFormat::Nvfp4) => Some(Quant::Nvfp4),
            Self::MlxAffine { bits: 4, .. } | Self::GgmlQ4_1 => Some(Quant::Q4),
            Self::MlxAffine { bits: 8, .. } | Self::GgmlQ8_0 => Some(Quant::Q8),
            Self::MlxAffine { .. } => return Err(invalid("unsupported affine quantization width")),
        })
    }

    pub fn validate_selection(self, requested: Option<Quant>, precision: Precision) -> Result<()> {
        self.validate_parameters()?;
        if self.quant()? != requested {
            return Err(invalid(format!("selected {requested:?}, but artifact is {}; install/convert the exact requested native tier (no model/tier fallback)",self.key()?)));
        }
        if matches!(
            self,
            Self::Publisher(DitFormat::Fp8Rowwise | DitFormat::Nvfp4)
        ) && precision != Precision::Bf16
        {
            return Err(invalid(
                "publisher FP8R/NVFP4 kernels return BF16; FP32 is not silently substituted",
            ));
        }
        Ok(())
    }

    pub fn check_backend(self, backend: ArtifactBackend) -> Result<()> {
        self.validate_parameters()?;
        if matches!(self, Self::GgmlQ4_1 | Self::GgmlQ8_0) && !cfg!(target_endian = "little") {
            return Err(invalid(
                "native GGML block bytes require a little-endian execution backend",
            ));
        }
        let supported = match self {
            Self::Publisher(DitFormat::Dense) => true,
            Self::Publisher(format) => {
                return format.check_device(match backend {
                    ArtifactBackend::Mlx => quantization::QuantDevice::Mlx,
                    ArtifactBackend::CandleCuda { major, minor } => {
                        quantization::QuantDevice::Cuda { major, minor }
                    }
                    _ => quantization::QuantDevice::Cpu,
                })
            }
            Self::MlxAffine { .. } => backend == ArtifactBackend::Mlx,
            Self::GgmlQ4_1 | Self::GgmlQ8_0 => matches!(
                backend,
                ArtifactBackend::CandleCpu | ArtifactBackend::CandleCuda { .. }
            ),
        };
        if supported {
            Ok(())
        } else {
            Err(invalid(format!("{} is not a supported artifact on {backend:?}; convert the ORIGINAL dense checkpoint with this backend's native converter instead",self.key()?)))
        }
    }

    /// Native conversions use the source FP8 target/exclusion partition, NOT a
    /// FLUX2 alias or an all-parameters * bits estimate. GGML scales live INSIDE
    /// each byte block; MLX affine has explicit BF16 scale/bias companion tensors.
    pub fn expected_tensors(self, cfg: &DitConfig) -> Result<BTreeMap<String, TensorSpec>> {
        self.validate_parameters()?;
        if let Self::Publisher(format) = self {
            return quantization::expected_tensors(cfg, format);
        }
        let mut tensors = quantization::expected_tensors(cfg, DitFormat::Dense)?;
        let targets = quantization::expected_tensors(cfg, DitFormat::Fp8Rowwise)?;
        for (name, target) in targets {
            if target.dtype != Dtype::F8_E4M3 {
                continue;
            }
            let [n, k] = target.shape.as_slice() else {
                return Err(invalid("quantization target is not a matrix"));
            };
            let (n, k) = (*n, *k);
            match self {
                Self::MlxAffine { bits, group_size } => {
                    let group = group_size as usize;
                    if k < group || !k.is_multiple_of(group) {
                        return Err(invalid(format!("{name}: width {k} does not support requested group {group}; no target is silently kept dense")));
                    }
                    let stem = name
                        .strip_suffix(".weight")
                        .ok_or_else(|| invalid("bad target stem"))?;
                    tensors.insert(
                        name.clone(),
                        TensorSpec {
                            dtype: Dtype::U32,
                            shape: vec![n, k / (32 / bits as usize)],
                        },
                    );
                    for suffix in ["scales", "biases"] {
                        tensors.insert(
                            format!("{stem}.{suffix}"),
                            TensorSpec {
                                dtype: Dtype::BF16,
                                shape: vec![n, k / group],
                            },
                        );
                    }
                }
                Self::GgmlQ4_1 | Self::GgmlQ8_0 => {
                    if !k.is_multiple_of(32) {
                        return Err(invalid(format!("{name}: GGML contraction width must divide into complete 32-element blocks")));
                    }
                    let block = if self == Self::GgmlQ4_1 { 20 } else { 34 };
                    let bytes = (k / 32)
                        .checked_mul(block)
                        .ok_or_else(|| invalid("GGML row byte size overflow"))?;
                    tensors.insert(
                        name,
                        TensorSpec {
                            dtype: Dtype::U8,
                            shape: vec![n, bytes],
                        },
                    );
                }
                Self::Publisher(_) => unreachable!("publisher returned above"),
            }
        }
        Ok(tensors)
    }

    pub fn validate_headers(
        self,
        cfg: &DitConfig,
        headers: &[SafetensorsTensorHeader],
    ) -> Result<()> {
        let mut expected = self.expected_tensors(cfg)?;
        if self == Self::Publisher(DitFormat::Dense) {
            // The existing native dense path accepts F32 exports/miniature
            // fixtures and casts to the explicitly requested compute precision.
            // Do not regress it by pretending every dense source is pinned BF16.
            for header in headers {
                if header.dtype == Dtype::F32 {
                    if let Some(tensor) = expected.get_mut(&header.name) {
                        tensor.dtype = Dtype::F32;
                    }
                }
            }
        }
        quantization::validate_tensor_inventory(&expected, headers)
    }
}

#[cfg(test)]
#[path = "artifacts_tests.rs"]
mod tests;
