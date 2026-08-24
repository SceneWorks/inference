//! Native **GGUF k-quant** loader for the Wan DiT (sc-12735, Pillar 2 of epic 12732 — the 24 GB lever).
//!
//! Where the MLX-packed path ([`crate::quant`], sc-10025) reads a `SceneWorks/wan2.2-*-mlx` tier's affine
//! `.scales`/`.biases` triple, and the ComfyUI scaled-fp8 seam ([`crate::comfyui`]) *pre-dequantizes* to a
//! dense bf16 map, this module holds the DiT as **resident Q4_K_M [`QTensor`]s and dequantizes per-block
//! per-matmul** — the ComfyUI-GGUF posture. A `QuantStack/Wan2.2-TI2V-5B-GGUF` `.gguf` is opened with
//! candle's native reader ([`candle_gen::candle_core::quantized::gguf_file`], k-quant CUDA-supported in the
//! SceneWorks candle pin), and **every k-quant weight stays quantized-resident**: the dequant happens on
//! the matmul ([`candle_gen::quant::QLinear::from_qtensor_dequant`], the sc-7702-safe
//! [`candle_gen::quant::MatmulStrategy::DequantDense`] forward), **never at load**. That is the whole win —
//! copying the ComfyUI seam's `from_tensors(dense bf16)` naively would erase it.
//!
//! ## Scope (sub-story 1)
//!
//! The loader mechanism, proven on the **5B** (single dense DiT — the simplest vehicle). The manifest /
//! catalog / tier routing (sub-story 2), the A14B dual-expert GGUF (sub-story 3), and a GGUF text encoder
//! (sub-story 4) are **separate** stories. The 5B GGUF path is selected here by the
//! [`env_gguf_path`] test seam (an env var pointing at a downloaded `.gguf`), NOT by the manifest.
//!
//! ## The two transforms (shared with the ComfyUI seam)
//!
//! 1. **Native-Wan → diffusers key remap** — QuantStack GGUF ships the **native-Wan** tensor names
//!    (`blocks.N.self_attn.q`, `cross_attn`, `ffn.0/2`, `modulation`, `norm3`, `head.head`,
//!    `text_embedding.0/2`, `time_projection.1`); the loader reuses the ONE
//!    [`crate::comfyui::remap_wan_key`] rename to the diffusers schema [`crate::transformer::WanTransformer`]
//!    reads.
//! 2. **Resident k-quant vs dense sidecar split** — the attention/FFN/embedder/`proj_out` Linears are
//!    k-quant `QTensor`s held resident; the dense sidecars (norms, biases, `modulation`, `patch_embedding`,
//!    `scale_shift_table`) are the GGUF's F16/F32 blocks, dequantized on read to the DiT compute dtype.
//!
//! The build routes through the SAME [`WeightSrc`] the dense path uses (so `WanTransformer::new` and
//! `WanTransformer::from_gguf` share every shape rule), and the resulting DiT reports `is_packed()` — it
//! drops into the sc-12757 sequential residency as the (now quantized-resident) denoise component,
//! unchanged staging.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_gen::candle_core::{Device, Error as CError, Result as CResult, Shape, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    CheckpointCodecRegistry, CodecResidencyReport, LogicalKeyMapping, LogicalReadMaterialization,
    LogicalTensorPlan, LogicalWeightPlan, LogicalWeightReceipt, PlannedResidency, ResidencyMode,
    TensorCodecSpec, WeightEncoding, GGUF_CONTAINER_CODEC,
};

use crate::comfyui::remap_wan_key;
use crate::config::TransformerConfig;
use crate::quant::QLinear;
use crate::transformer::WanTransformer;

/// The env var the sub-story-1 test seam reads: an absolute path to a `QuantStack/Wan2.2-TI2V-5B-GGUF`
/// `.gguf` file. When set (and non-empty), [`crate::Pipeline::build_dit`] builds the 5B DiT natively from
/// its k-quant `QTensor`s (this module) instead of the snapshot's `transformer/`. This is the deliberate
/// minimal seam so the loader can be GPU-validated **without** the manifest/catalog wiring (sub-story 2).
pub(crate) const GGUF_ENV: &str = "CANDLE_GEN_WAN_GGUF";

/// The 5B-GGUF path selected by the [`GGUF_ENV`] test seam (`None` ⇒ the normal snapshot path). A
/// present-but-empty value is treated as unset. Clearly marked as the sub-story-1 seam — sub-story 2
/// replaces this env probe with manifest/catalog tier routing.
pub(crate) fn env_gguf_path() -> Option<PathBuf> {
    match std::env::var(GGUF_ENV) {
        Ok(v) if !v.trim().is_empty() => Some(PathBuf::from(v)),
        _ => None,
    }
}

/// Open a Wan DiT `.gguf` and build the [`WanTransformer`] with its k-quant weights held
/// **quantized-resident** (sc-12735). The entry point [`crate::Pipeline::build_dit`] calls on the GGUF seam.
pub(crate) fn load_wan_dit_gguf(
    path: &Path,
    cfg: &TransformerConfig,
    device: &Device,
    dtype: candle_gen::candle_core::DType,
) -> CResult<WanTransformer> {
    load_wan_dit_gguf_with_receipt(path, cfg, device, dtype).map(|(dit, _)| dit)
}

/// `load_wan_dit_gguf` plus the read's [`LogicalWeightReceipt`] — the registered-codec route's
/// truthful accounting of what the container left resident (measured from the materialized
/// `QTensor`s, not predicted from the header).
pub fn load_wan_dit_gguf_with_receipt(
    path: &Path,
    cfg: &TransformerConfig,
    device: &Device,
    dtype: candle_gen::candle_core::DType,
) -> CResult<(WanTransformer, LogicalWeightReceipt)> {
    let dit = GgufDit::open(path, device, dtype)?;
    let receipt = dit.receipt().clone();
    let transformer = WanTransformer::from_gguf(cfg, &dit)?;
    Ok((transformer, receipt))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// The registered checkpoint-codec route (epic 20398, sc-20649/sc-20651)
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The registered codec id this module implements — the GGUF half of the checkpoint-codec registry.
/// `candle-gen-catalog` registers [`GGUF_CONTAINER_CODEC`] and asserts this id against it, so the
/// row can never be declared without an implementation (or implemented without being declared).
pub const GGUF_CODEC_IMPLEMENTATION_ID: &str = GGUF_CONTAINER_CODEC.codec_id;

/// The codec registry the GGUF route plans against: the engine's safetensors table
/// ([`candle_gen::logical_weights::baseline_codec_registry`]) plus the
/// [`GGUF_CONTAINER_CODEC`] row this crate implements.
///
/// The row lives here rather than in the engine table because the implementation does: decoding a
/// GGUF container needs candle's `gguf_file` reader and the ggml block constants, which the engine
/// crate does not carry. `candle-gen-catalog` registers the identical row into the provider
/// registry and its conformance test asserts the two sets agree, so this is one declaration read
/// from two places, never a second, drifting table.
pub fn gguf_codec_registry() -> &'static CheckpointCodecRegistry {
    static REGISTRY: std::sync::OnceLock<CheckpointCodecRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        CheckpointCodecRegistry::new(
            candle_gen::logical_weights::BASELINE_CODECS
                .iter()
                .copied()
                .chain(std::iter::once(GGUF_CONTAINER_CODEC)),
        )
        .expect("the engine codec table plus the GGUF container row is a valid registry")
    })
}

/// The **refusing** implementation of the Wan adapter's registered
/// `wan-comfyui-to-diffusers-v1` canonical mapping.
///
/// `remap_wan_key` is a total function: a key it does not recognise falls through **unchanged**,
/// which is right for the ComfyUI dense seam (the DiT then asks for the keys it wants and ignores
/// the rest) but wrong for a plan-driven route, whose whole contract is that nothing is skipped
/// silently. So this mapping carries its own explicit recognizer over the native-Wan DiT key
/// surface and answers `None` — a typed `UnmappedKey` naming the tensor — for everything else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WanNativeToDiffusersMapping;

impl WanNativeToDiffusersMapping {
    pub const MAPPING_ID: &'static str = "wan-comfyui-to-diffusers-v1";
}

impl LogicalKeyMapping for WanNativeToDiffusersMapping {
    fn mapping_id(&self) -> &'static str {
        Self::MAPPING_ID
    }

    fn logical_key(&self, physical_key: &str) -> Option<String> {
        is_native_wan_dit_key(physical_key).then(|| remap_wan_key(physical_key))
    }
}

/// Whether `key` is a tensor of the native-Wan DiT surface this route reads. The enumeration
/// mirrors [`remap_wan_key`]'s own rename table plus the two keys that pass through unchanged
/// (`patch_embedding.{weight,bias}`), so the recognizer and the renamer cannot drift apart without
/// the coverage test below failing.
fn is_native_wan_dit_key(key: &str) -> bool {
    // Top-level: the head, the two condition embedders, the time projection, the patch embedder.
    const TOP_LEVEL: &[&str] = &[
        "head.modulation",
        "patch_embedding.weight",
        "patch_embedding.bias",
    ];
    if TOP_LEVEL.contains(&key) {
        return true;
    }
    const TOP_LEVEL_LINEARS: &[&str] = &[
        "head.head",
        "text_embedding.0",
        "text_embedding.2",
        "time_embedding.0",
        "time_embedding.2",
        "time_projection.1",
    ];
    if let Some((base, leaf)) = key.rsplit_once('.') {
        if TOP_LEVEL_LINEARS.contains(&base) && matches!(leaf, "weight" | "bias") {
            return true;
        }
    }

    // Per-block: `blocks.<N>.<leaf>`.
    let Some(rest) = key.strip_prefix("blocks.") else {
        return false;
    };
    let Some((index, leaf)) = rest.split_once('.') else {
        return false;
    };
    if index.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if leaf == "modulation" {
        return true;
    }
    let Some((base, tail)) = leaf.rsplit_once('.') else {
        return false;
    };
    if !matches!(tail, "weight" | "bias") {
        return false;
    }
    const BLOCK_BASES: &[&str] = &[
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "self_attn.norm_q",
        "self_attn.norm_k",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "cross_attn.norm_q",
        "cross_attn.norm_k",
        "ffn.0",
        "ffn.2",
        "norm3",
    ];
    BLOCK_BASES.contains(&base)
}

/// Why a `.gguf` container could not compile to a [`LogicalWeightPlan`]. Every variant names the
/// exact tensor (or file) the fault appears on — nothing is swallowed into a default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GgufPlanError {
    /// The file could not be opened or its GGUF header could not be parsed. An unknown ggml type
    /// *code* is refused here, by candle's own reader, before any tensor is planned.
    Container { path: String, defect: String },
    /// The container declares no DiT tensors at all (only the dropped `freqs` buffer, or nothing).
    EmptyContainer { path: String },
    /// The mapping recognises no logical key for this container tensor.
    UnmappedKey { physical_key: String },
    /// Two container tensors map onto one logical key.
    KeyCollision {
        logical_key: String,
        first_physical_key: String,
        second_physical_key: String,
    },
    /// A ggml quantization type this route does not read. `Q8_1` and `Q8_K` are ggml *activation*
    /// intermediates — they are not a stored weight type in any GGUF export — so they are refused
    /// rather than decoded through an unvalidated path.
    UnsupportedQuantType {
        physical_key: String,
        ggml_type: &'static str,
    },
    /// The tensor's element count is not a whole number of ggml blocks for its type, so its
    /// container size is not computable and the file is malformed.
    BlockGeometry {
        physical_key: String,
        ggml_type: &'static str,
        elements: usize,
        block_size: usize,
    },
    /// No GGUF codec is registered on this backend's codec registry.
    UnregisteredCodec { codec_id: &'static str },
}

impl fmt::Display for GgufPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Container { path, defect } => {
                write!(f, "wan gguf: {path}: {defect}")
            }
            Self::EmptyContainer { path } => write!(
                f,
                "wan gguf: {path} declares no DiT tensors; an empty container cannot be a model component"
            ),
            Self::UnmappedKey { physical_key } => write!(
                f,
                "wan gguf: container tensor {physical_key:?} is not a native-Wan DiT tensor this route reads (wrong family, a VACE/adapter export, or a key outside the DiT namespace)"
            ),
            Self::KeyCollision {
                logical_key,
                first_physical_key,
                second_physical_key,
            } => write!(
                f,
                "wan gguf: container tensors {first_physical_key:?} and {second_physical_key:?} both map onto logical key {logical_key:?}"
            ),
            Self::UnsupportedQuantType {
                physical_key,
                ggml_type,
            } => write!(
                f,
                "wan gguf: container tensor {physical_key:?} is stored as ggml {ggml_type}, which the `{}` codec does not read",
                GGUF_CONTAINER_CODEC.codec_id
            ),
            Self::BlockGeometry {
                physical_key,
                ggml_type,
                elements,
                block_size,
            } => write!(
                f,
                "wan gguf: container tensor {physical_key:?} holds {elements} elements, not a whole number of ggml {ggml_type} blocks of {block_size}"
            ),
            Self::UnregisteredCodec { codec_id } => write!(
                f,
                "wan gguf: no checkpoint codec '{codec_id}' is registered on this backend"
            ),
        }
    }
}

impl std::error::Error for GgufPlanError {}

impl From<GgufPlanError> for CError {
    fn from(error: GgufPlanError) -> Self {
        CError::msg(error.to_string())
    }
}

/// A ggml block type's stable label, for refusal messages and the per-tensor quant table.
fn ggml_label(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::F32 => "F32",
        GgmlDType::F16 => "F16",
        GgmlDType::BF16 => "BF16",
        GgmlDType::Q4_0 => "Q4_0",
        GgmlDType::Q4_1 => "Q4_1",
        GgmlDType::Q5_0 => "Q5_0",
        GgmlDType::Q5_1 => "Q5_1",
        GgmlDType::Q8_0 => "Q8_0",
        GgmlDType::Q8_1 => "Q8_1",
        GgmlDType::Q2K => "Q2_K",
        GgmlDType::Q3K => "Q3_K",
        GgmlDType::Q4K => "Q4_K",
        GgmlDType::Q5K => "Q5_K",
        GgmlDType::Q6K => "Q6_K",
        GgmlDType::Q8K => "Q8_K",
    }
}

/// The ggml block types the `gguf-container-v1` codec reads: the k-quants and legacy quants a
/// real export stores weights in, plus the dense F32/F16/BF16 blocks its sidecars use. `Q8_1` and
/// `Q8_K` are deliberately absent — they are activation-side intermediates, never a stored weight
/// type, so this route refuses them by name rather than decoding an unvalidated layout.
fn quant_type_is_readable(dtype: GgmlDType) -> bool {
    match dtype {
        GgmlDType::F32
        | GgmlDType::F16
        | GgmlDType::BF16
        | GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q2K
        | GgmlDType::Q3K
        | GgmlDType::Q4K
        | GgmlDType::Q5K
        | GgmlDType::Q6K => true,
        GgmlDType::Q8_1 | GgmlDType::Q8K => false,
    }
}

/// A compiled GGUF read plan: the portable [`LogicalWeightPlan`] plus the ggml block type each
/// logical tensor is stored in.
///
/// The plan is the *same* type the safetensors route compiles, so admission and pricing consume one
/// shape. The side table exists because [`TensorCodecSpec`] is a safetensors-shaped vocabulary with
/// no ggml arm; rather than misreport a Q4_K payload as one of its variants, every GGUF entry is
/// [`TensorCodecSpec::Dense`] — a byte-preserving pass-through into a resident `QTensor` — and the
/// block type it actually holds is recorded here, typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufDitPlan {
    pub plan: LogicalWeightPlan,
    /// logical key → the ggml block type its container payload is stored in.
    pub quant_types: BTreeMap<String, GgmlDType>,
}

/// Compile a parsed `.gguf` header into a [`GgufDitPlan`] against the registered GGUF codec.
///
/// # Residency, and why every entry is `Packed`
///
/// `GgufDit` holds **every** container tensor as a resident [`QTensor`] — that is the sc-12735
/// posture and the whole reason a GGUF tier fits where a bf16 one does not. So the plan prices each
/// tensor at its *container* size (`elements / block_size × type_size`, from ggml's own constants),
/// never at `logical shape × dense element width`; the codec row's
/// `resident_encoding` is the bf16 a dense fallback **would** leave, and is unused on this path.
/// `GgufDit::open` then measures the same quantity from the materialized `QTensor`s
/// ([`QTensor::storage_size_in_bytes`]) — plan and receipt are independent computations of the same
/// number, so a drift in either shows up rather than being asserted away.
pub fn compile_gguf_dit_plan(
    path: &Path,
    content: &gguf_file::Content,
    mapping: &dyn LogicalKeyMapping,
    codecs: &CheckpointCodecRegistry,
) -> Result<GgufDitPlan, GgufPlanError> {
    let codec = codecs.for_encoding(WeightEncoding::GgufContainer).ok_or(
        GgufPlanError::UnregisteredCodec {
            codec_id: GGUF_CONTAINER_CODEC.codec_id,
        },
    )?;

    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort();
    let mut tensors: Vec<LogicalTensorPlan> = Vec::with_capacity(names.len());
    let mut quant_types: BTreeMap<String, GgmlDType> = BTreeMap::new();
    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    let mut source_bytes: u64 = 0;

    for name in names {
        // The DiT derives RoPE from theta; a precomputed `freqs` buffer (present on some exports)
        // is dropped rather than mapped, mirroring the ComfyUI seam. It is the one documented
        // exclusion, taken before the mapping is consulted so it can never look like a refusal.
        if name == "freqs" || name.ends_with(".freqs") {
            continue;
        }
        let info = &content.tensor_infos[name];
        let logical_key = mapping
            .logical_key(name)
            .ok_or_else(|| GgufPlanError::UnmappedKey {
                physical_key: name.clone(),
            })?;
        if let Some(first) = owners.insert(logical_key.clone(), name.clone()) {
            return Err(GgufPlanError::KeyCollision {
                logical_key,
                first_physical_key: first,
                second_physical_key: name.clone(),
            });
        }
        let ggml_type = info.ggml_dtype;
        if !quant_type_is_readable(ggml_type) {
            return Err(GgufPlanError::UnsupportedQuantType {
                physical_key: name.clone(),
                ggml_type: ggml_label(ggml_type),
            });
        }
        let elements = info.shape.elem_count();
        let block_size = ggml_type.block_size();
        // `is_multiple_of(0)` is `elements == 0`, so a zero block size refuses on its own.
        if block_size == 0 || !elements.is_multiple_of(block_size) {
            return Err(GgufPlanError::BlockGeometry {
                physical_key: name.clone(),
                ggml_type: ggml_label(ggml_type),
                elements,
                block_size,
            });
        }
        let container_bytes = (elements / block_size * ggml_type.type_size()) as u64;
        source_bytes = source_bytes.saturating_add(container_bytes);
        quant_types.insert(logical_key.clone(), ggml_type);
        tensors.push(LogicalTensorPlan {
            logical_key,
            physical_key: name.clone(),
            encoding: WeightEncoding::GgufContainer,
            shape: info.shape.dims().to_vec(),
            source_bytes: container_bytes,
            codec_id: codec.codec_id,
            resident_encoding: codec.resident_encoding,
            codec: TensorCodecSpec::Dense,
            residency: PlannedResidency {
                mode: ResidencyMode::Packed,
                resident_bytes: container_bytes,
            },
        });
    }

    if tensors.is_empty() {
        return Err(GgufPlanError::EmptyContainer {
            path: path.display().to_string(),
        });
    }
    tensors.sort_by(|a, b| a.logical_key.cmp(&b.logical_key));
    Ok(GgufDitPlan {
        plan: LogicalWeightPlan {
            mapping_id: mapping.mapping_id(),
            tensors,
            // A GGUF container has no `.comfy_quant` descriptors or scale companions: the block
            // scales live inside each quantized block, not as sibling tensors.
            companions: Vec::new(),
            source_bytes,
        },
        quant_types,
    })
}

/// A Wan DiT `.gguf` parsed into **resident** [`QTensor`]s, diffusers-keyed (post native→diffusers remap).
/// k-quant Linear weights (`Q4_K` etc.) are held quantized — NEVER dequantized to a dense `[out,in]` weight
/// at load. Dense sidecars (norms, biases, `modulation`, `patch_embedding`, `scale_shift_table`) are the
/// GGUF's F16/F32 blocks, dequantized on read.
pub(crate) struct GgufDit {
    /// diffusers-key → resident GGUF tensor (k-quant Linear weight, or an F16/F32 dense sidecar block).
    tensors: HashMap<String, Arc<QTensor>>,
    device: Device,
    /// The DiT compute dtype (bf16) dense sidecars are cast to on read — matching the dense path's
    /// `VarBuilder::get` cast, so the GGUF and snapshot builds agree tensor-for-tensor on the sidecars.
    dtype: candle_gen::candle_core::DType,
    /// The receipt of the read that produced this DiT: mapping id, tensor count, and the
    /// **measured** container residency (see [`GgufDit::receipt`]).
    receipt: LogicalWeightReceipt,
}

impl GgufDit {
    /// Open `path` through the **registered** `gguf-container-v1` codec route (sc-20649): parse the
    /// header, compile a [`LogicalWeightPlan`] against the backend's codec registry and the
    /// refusing [`WanNativeToDiffusersMapping`], then materialize exactly the planned tensors and
    /// measure what they left resident.
    ///
    /// Every refusal is typed and names its tensor: an unrecognised key, two keys colliding on one
    /// logical key, a ggml quant type this codec does not read. Tensors are read in logical-key
    /// order so any error message is stable.
    pub(crate) fn open(
        path: &Path,
        device: &Device,
        dtype: candle_gen::candle_core::DType,
    ) -> CResult<Self> {
        Self::open_with(
            path,
            device,
            dtype,
            &WanNativeToDiffusersMapping,
            gguf_codec_registry(),
        )
    }

    /// [`GgufDit::open`] with the mapping and codec registry injected — the drivable form the tests
    /// use to exercise a foreign key or an unregistered codec against a real container.
    pub(crate) fn open_with(
        path: &Path,
        device: &Device,
        dtype: candle_gen::candle_core::DType,
        mapping: &dyn LogicalKeyMapping,
        codecs: &CheckpointCodecRegistry,
    ) -> CResult<Self> {
        let mut file = std::fs::File::open(path).map_err(|e| GgufPlanError::Container {
            path: path.display().to_string(),
            defect: format!("open: {e}"),
        })?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| GgufPlanError::Container {
                path: path.display().to_string(),
                defect: format!("parse: {e}"),
            })?;
        let GgufDitPlan { plan, .. } = compile_gguf_dit_plan(path, &content, mapping, codecs)?;

        let mut tensors: HashMap<String, Arc<QTensor>> = HashMap::with_capacity(plan.tensors.len());
        // Measured, not predicted: the receipt's resident bytes come from the materialized
        // QTensors' own storage, so a plan that mis-priced the container cannot hide behind it.
        let mut resident_bytes: u64 = 0;
        for tensor in &plan.tensors {
            let qt = content
                .tensor(&mut file, &tensor.physical_key, device)
                .map_err(|e| {
                    CError::msg(format!(
                        "wan gguf: read tensor {:?}: {e}",
                        tensor.physical_key
                    ))
                })?;
            resident_bytes = resident_bytes.saturating_add(qt.storage_size_in_bytes() as u64);
            tensors.insert(tensor.logical_key.clone(), Arc::new(qt));
        }
        let receipt = LogicalWeightReceipt {
            mapping_id: plan.mapping_id,
            tensor_count: plan.tensor_count(),
            source_bytes: plan.source_bytes,
            materialization: LogicalReadMaterialization::Materialized,
            residency: vec![CodecResidencyReport {
                codec_id: GGUF_CONTAINER_CODEC.codec_id,
                tensor_count: plan.tensor_count(),
                source_bytes: plan.source_bytes,
                resident_bytes,
            }],
        };
        Ok(Self {
            tensors,
            device: device.clone(),
            dtype,
            receipt,
        })
    }

    /// The receipt of the read that produced this DiT. `resident_bytes` is measured from the
    /// materialized `QTensor`s, never copied from the plan.
    pub(crate) fn receipt(&self) -> &LogicalWeightReceipt {
        &self.receipt
    }

    /// The resident tensor at diffusers `key`, or a **loud** error naming the missing key (a renamed /
    /// absent tensor must fail the load, not silently degrade — the sc-12735 "fail loudly" contract).
    fn require(&self, key: &str) -> CResult<&Arc<QTensor>> {
        self.tensors.get(key).ok_or_else(|| {
            CError::msg(format!(
                "wan gguf: missing tensor {key:?} — the 5B DiT expects it (renamed/absent GGUF key, or \
                 an unmapped native-Wan name); a native→diffusers remap gap fails the load here"
            ))
        })
    }

    /// Build a **resident-QTensor** [`QLinear`] for `{base}` from the k-quant `{base}.weight` (held
    /// quantized) plus the optional dense `{base}.bias`. The QTensor is shared by `Arc` (no copy); the
    /// forward dequantizes it per-matmul (the sc-7702-safe [`candle_gen::quant::MatmulStrategy::DequantDense`]
    /// path). A `[out, in]` shape mismatch is a loud error (a wrong-dim GGUF, never silent garbage).
    fn qlinear(&self, base: &str, in_dim: usize, out_dim: usize, bias: bool) -> CResult<QLinear> {
        let wkey = format!("{base}.weight");
        let qt = self.require(&wkey)?.clone();
        let dims = qt.shape().dims();
        if dims != [out_dim, in_dim] {
            return Err(CError::msg(format!(
                "wan gguf: {wkey:?} shape {dims:?} != expected [{out_dim}, {in_dim}]"
            )));
        }
        let bias = if bias {
            Some(self.dense(&format!("{base}.bias"), Shape::from((out_dim,)))?)
        } else {
            None
        };
        // Ingest the resident k-quant QTensor WITHOUT dequantizing to dense (the whole point), then wrap
        // it as the packed base of an `AdaptLinear` so `transformer.rs` keeps calling `QLinear` unchanged.
        let base = candle_gen::quant::QLinear::from_qtensor_dequant(qt, bias);
        Ok(QLinear::from_packed(base, in_dim, out_dim))
    }

    /// Dequantize a dense sidecar `{key}` (an F16/F32 GGUF block) to the DiT compute dtype, verifying its
    /// element count matches `shape` (reshaping when the block's logical shape differs but the count
    /// agrees — e.g. a flattened bias). Mirrors the dense path's `VarBuilder::get(shape, key)` (which also
    /// shape-checks + casts to the builder dtype), so the two builds agree on every sidecar.
    fn dense(&self, key: &str, shape: Shape) -> CResult<Tensor> {
        let qt = self.require(key)?;
        let t = qt.dequantize(&self.device)?.to_dtype(self.dtype)?;
        if t.dims() == shape.dims() {
            Ok(t)
        } else if t.elem_count() == shape.elem_count() {
            t.reshape(shape)
        } else {
            Err(CError::msg(format!(
                "wan gguf: {key:?} element count {} != expected {} (shape {:?} vs {:?})",
                t.elem_count(),
                shape.elem_count(),
                t.dims(),
                shape.dims()
            )))
        }
    }
}

/// The weight source a [`WanTransformer`] build reads from: a **dense** [`VarBuilder`] (a snapshot or an
/// MLX-packed tier, the unchanged path) or a **native-GGUF** [`GgufDit`] (resident k-quant, sc-12735). It
/// unifies the two so `WanTransformer::new` and `WanTransformer::from_gguf` share every shape rule — the
/// dense arm forwards to the exact `VarBuilder`/`QLinear::linear_detect` calls as before (byte-identical),
/// the GGUF arm routes each Linear through [`GgufDit::qlinear`] (resident QTensor) and each sidecar through
/// [`GgufDit::dense`].
pub(crate) enum WeightSrc<'a> {
    /// The dense / MLX-packed path (unchanged) — packed-detects `.scales` per [`QLinear::linear_detect`].
    Dense(VarBuilder<'a>),
    /// The native-GGUF k-quant path (sc-12735) — a resident [`GgufDit`] plus the current dotted key prefix.
    Gguf { dit: &'a GgufDit, prefix: String },
}

impl<'a> WeightSrc<'a> {
    /// The dense / MLX-packed source over `vb` (the unchanged path).
    pub(crate) fn dense(vb: VarBuilder<'a>) -> Self {
        Self::Dense(vb)
    }

    /// The native-GGUF source at the root prefix (sc-12735).
    pub(crate) fn gguf(dit: &'a GgufDit) -> Self {
        Self::Gguf {
            dit,
            prefix: String::new(),
        }
    }

    /// A sub-scope under `seg` (the [`VarBuilder::pp`] analogue) — appends `seg.` to the key prefix so a
    /// `to_out.0`-style nesting survives on both arms.
    pub(crate) fn pp(&self, seg: impl std::fmt::Display) -> WeightSrc<'a> {
        match self {
            Self::Dense(vb) => Self::Dense(vb.pp(seg.to_string())),
            Self::Gguf { dit, prefix } => Self::Gguf {
                dit,
                prefix: format!("{prefix}{seg}."),
            },
        }
    }

    /// The device this source builds on.
    pub(crate) fn device(&self) -> Device {
        match self {
            Self::Dense(vb) => vb.device().clone(),
            Self::Gguf { dit, .. } => dit.device.clone(),
        }
    }

    /// The compute dtype (bf16 for the DiT).
    pub(crate) fn dtype(&self) -> candle_gen::candle_core::DType {
        match self {
            Self::Dense(vb) => vb.dtype(),
            Self::Gguf { dit, .. } => dit.dtype,
        }
    }

    /// A dense tensor `{key}` (relative to this scope) at the source dtype — a norm / bias / modulation /
    /// `patch_embedding` / `scale_shift_table` sidecar. Dense arm: `vb.get`; GGUF arm: [`GgufDit::dense`].
    pub(crate) fn get(&self, shape: impl Into<Shape>, key: &str) -> CResult<Tensor> {
        match self {
            Self::Dense(vb) => vb.get(shape, key),
            Self::Gguf { dit, prefix } => dit.dense(&format!("{prefix}{key}"), shape.into()),
        }
    }

    /// A [`QLinear`] for `{base}` (relative to this scope). Dense arm: [`QLinear::linear_detect`]
    /// (packed-detecting, unchanged); GGUF arm: [`GgufDit::qlinear`] (resident k-quant QTensor).
    pub(crate) fn qlinear(
        &self,
        in_dim: usize,
        out_dim: usize,
        base: &str,
        bias: bool,
    ) -> CResult<QLinear> {
        match self {
            Self::Dense(vb) => QLinear::linear_detect(in_dim, out_dim, vb, base, bias),
            Self::Gguf { dit, prefix } => {
                dit.qlinear(&format!("{prefix}{base}"), in_dim, out_dim, bias)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::WanRope;
    use candle_gen::candle_core::quantized::{gguf_file, GgmlDType};
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::quant::MatmulStrategy;

    /// A small config whose every Linear contraction (`in`) is a multiple of the Q4_K block (256), so a
    /// real k-quant fixture can be written: dim 256 (= 2 heads × 128), ffn 512, freq/text 256. 2 blocks.
    fn gguf_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 4,
            out_channels: 4,
            num_layers: 2,
            num_heads: 2,
            head_dim: 128,
            dim: 256,
            ffn_dim: 512,
            freq_dim: 256,
            text_dim: 256,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 64,
        }
    }

    /// A k-quant `QTensor` of shape `[out, in]` (`in` must be a multiple of 256) from a deterministic
    /// small-magnitude grid — the resident weight the loader must keep quantized.
    fn q4k(out: usize, inn: usize) -> QTensor {
        let data: Vec<f32> = (0..out * inn)
            .map(|i| ((i % 17) as f32 / 17.0 - 0.5) * 0.1)
            .collect();
        let w = Tensor::from_vec(data, (out, inn), &Device::Cpu).unwrap();
        QTensor::quantize(&w, GgmlDType::Q4K).unwrap()
    }

    /// A dense (F32-block) `QTensor` of `shape` — the sidecar form (norms / biases / modulation /
    /// `patch_embedding` / `scale_shift_table`), stored uncompressed so it dequantizes back exactly.
    fn dense_qt(shape: &[usize]) -> QTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.2).collect();
        let t = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::F32).unwrap()
    }

    /// Emit the FULL **native-Wan** keyed tensor set a `WanTransformer` of `cfg` needs — Q4_K Linear
    /// weights + F32 dense sidecars/biases — so the fixture exercises [`remap_wan_key`] end-to-end (the
    /// same key layout a real `QuantStack/Wan2.2-*-GGUF` ships). Returned as an owned map so the caller
    /// can borrow `(&str, &QTensor)` pairs for [`gguf_file::write`].
    fn native_wan_tensors(cfg: &TransformerConfig) -> HashMap<String, QTensor> {
        let d = cfg.dim;
        let (pt, ph, pw) = cfg.patch;
        let mut m: HashMap<String, QTensor> = HashMap::new();
        // Linear weight (Q4_K) + dense bias, under a native base key.
        let lin = |m: &mut HashMap<String, QTensor>, base: &str, out: usize, inn: usize| {
            m.insert(format!("{base}.weight"), q4k(out, inn));
            m.insert(format!("{base}.bias"), dense_qt(&[out]));
        };
        // Top-level embedders + head (native names).
        lin(&mut m, "text_embedding.0", d, cfg.text_dim);
        lin(&mut m, "text_embedding.2", d, d);
        lin(&mut m, "time_embedding.0", d, cfg.freq_dim);
        lin(&mut m, "time_embedding.2", d, d);
        lin(&mut m, "time_projection.1", 6 * d, d);
        lin(&mut m, "head.head", cfg.out_channels * pt * ph * pw, d);
        m.insert("head.modulation".into(), dense_qt(&[1, 2, d]));
        // patch_embedding (native == diffusers key), a 5-D conv weight + bias, dense.
        m.insert(
            "patch_embedding.weight".into(),
            dense_qt(&[d, cfg.in_channels, pt, ph, pw]),
        );
        m.insert("patch_embedding.bias".into(), dense_qt(&[d]));
        // Per-block (native names): self/cross attn q/k/v/o (+ norm_q/k), ffn.0/2, norm3, modulation.
        for i in 0..cfg.num_layers {
            let b = format!("blocks.{i}");
            for attn in ["self_attn", "cross_attn"] {
                for leaf in ["q", "k", "v", "o"] {
                    lin(&mut m, &format!("{b}.{attn}.{leaf}"), d, d);
                }
                m.insert(format!("{b}.{attn}.norm_q.weight"), dense_qt(&[d]));
                m.insert(format!("{b}.{attn}.norm_k.weight"), dense_qt(&[d]));
            }
            lin(&mut m, &format!("{b}.ffn.0"), cfg.ffn_dim, d);
            lin(&mut m, &format!("{b}.ffn.2"), d, cfg.ffn_dim);
            m.insert(format!("{b}.norm3.weight"), dense_qt(&[d]));
            m.insert(format!("{b}.norm3.bias"), dense_qt(&[d]));
            m.insert(format!("{b}.modulation"), dense_qt(&[1, 6, d]));
        }
        m
    }

    /// Write `tensors` to a fresh `.gguf` at a unique temp path and return it.
    fn write_gguf(
        tmp: &tempfile::TempDir,
        tensors: &HashMap<String, QTensor>,
        tag: &str,
    ) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let uniq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = tmp.path().join(format!("sc12735_{tag}_{}.gguf", uniq));
        let refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut f = std::fs::File::create(&path).unwrap();
        // A minimal metadata table (the DiT config comes from `TransformerConfig`, not the GGUF here).
        let arch = gguf_file::Value::String("wan".to_string());
        gguf_file::write(&mut f, &[("general.architecture", &arch)], &refs).unwrap();
        path
    }

    // ── the registered codec route: plan, refusals, measured receipt (sc-20649/sc-20651) ─────────────

    /// Read a container's header back the way the plan compiler does.
    fn read_content(path: &Path) -> gguf_file::Content {
        let mut f = std::fs::File::open(path).unwrap();
        gguf_file::Content::read(&mut f).unwrap()
    }

    /// The recognizer accepts **exactly** the native-Wan DiT surface a real container ships and
    /// nothing else. The accept corpus is the full fixture (the same key layout a
    /// `QuantStack/Wan2.2-*-GGUF` writes); the refuse corpus is the tensors a *neighbouring* Wan
    /// export really carries — VACE hint blocks, the I2V CLIP image embedder, an adapter's LoRA
    /// leaf — each of which `remap_wan_key` would otherwise pass through unchanged.
    #[test]
    fn native_key_recognizer_matches_the_real_dit_surface() {
        let cfg = gguf_cfg();
        for key in native_wan_tensors(&cfg).keys() {
            assert!(
                is_native_wan_dit_key(key),
                "{key:?} is a real native-Wan DiT tensor and must be recognized"
            );
        }
        for key in [
            "vace_blocks.0.before_proj.weight",
            "vace_blocks.0.after_proj.bias",
            "img_emb.proj.1.weight",
            "blocks.0.self_attn.q.lora_A.weight",
            "blocks.0.self_attn.wq.weight",
            "blocks.x.self_attn.q.weight",
            "blocks.0.ffn.1.weight",
            "text_embedding.1.weight",
            "head.head.scale_weight",
            "freqs",
        ] {
            assert!(
                !is_native_wan_dit_key(key),
                "{key:?} is outside the DiT surface this route reads and must refuse"
            );
        }
    }

    /// A native-keyed container compiles to a [`LogicalWeightPlan`] against the **registered**
    /// `gguf-container-v1` row: every tensor is diffusers-keyed, `Packed`, and priced at its ggml
    /// container size (`elements / block_size × type_size`) — never at `logical shape × dense
    /// element width`, which for a Q4_K weight would overstate residency by ~3.5×.
    #[test]
    fn plan_prices_every_tensor_at_its_ggml_container_size() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let tensors = native_wan_tensors(&cfg);
        let path = write_gguf(&tmp, &tensors, "plan");
        let content = read_content(&path);
        let GgufDitPlan { plan, quant_types } = compile_gguf_dit_plan(
            &path,
            &content,
            &WanNativeToDiffusersMapping,
            gguf_codec_registry(),
        )
        .expect("the native fixture compiles");

        assert_eq!(plan.mapping_id, WanNativeToDiffusersMapping::MAPPING_ID);
        assert_eq!(plan.tensor_count(), tensors.len());
        assert!(plan.companions.is_empty(), "a GGUF has no scale companions");
        assert_eq!(plan.codec_ids(), vec![GGUF_CONTAINER_CODEC.codec_id]);

        // The remap fired: the native `blocks.0.self_attn.q.weight` is planned at its diffusers key.
        let q = plan
            .tensors
            .iter()
            .find(|t| t.logical_key == "blocks.0.attn1.to_q.weight")
            .expect("the remapped q projection is planned");
        assert_eq!(q.physical_key, "blocks.0.self_attn.q.weight");
        assert_eq!(q.encoding, WeightEncoding::GgufContainer);
        assert_eq!(q.residency.mode, ResidencyMode::Packed);
        assert_eq!(quant_types["blocks.0.attn1.to_q.weight"], GgmlDType::Q4K);
        // Q4_K: 144 bytes per 256-element block, so a [256, 256] weight is 256 blocks × 144.
        assert_eq!(q.shape, vec![cfg.dim, cfg.dim]);
        let blocks = (cfg.dim * cfg.dim) / GgmlDType::Q4K.block_size();
        assert_eq!(
            q.residency.resident_bytes,
            (blocks * GgmlDType::Q4K.type_size()) as u64
        );
        assert!(
            q.residency.resident_bytes < (cfg.dim * cfg.dim * 2) as u64,
            "a Q4_K weight priced at or above bf16 means the container size was not used"
        );

        // Every planned tensor prices at its own container size, and the plan's total is their sum.
        let mut total = 0u64;
        for tensor in &plan.tensors {
            let ggml = quant_types[&tensor.logical_key];
            let elements: usize = tensor.shape.iter().product();
            let expected = (elements / ggml.block_size() * ggml.type_size()) as u64;
            assert_eq!(
                tensor.residency.resident_bytes,
                expected,
                "{} priced wrong for ggml {}",
                tensor.logical_key,
                ggml_label(ggml)
            );
            assert_eq!(tensor.source_bytes, expected);
            assert_eq!(tensor.residency.mode, ResidencyMode::Packed);
            total += expected;
        }
        assert_eq!(plan.resident_bytes(), total);
        assert_eq!(plan.source_bytes, total);
        std::fs::remove_file(&path).ok();
    }

    /// The receipt's resident bytes are **measured** from the materialized `QTensor`s
    /// (`storage_size_in_bytes`), computed independently of the plan's ggml arithmetic — and the two
    /// agree. A reader that invented its receipt from the plan could not fail this; a plan that
    /// mis-sized the container could not hide behind it.
    #[test]
    fn receipt_measures_the_same_residency_the_plan_predicted() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let tensors = native_wan_tensors(&cfg);
        let path = write_gguf(&tmp, &tensors, "receipt");
        let content = read_content(&path);
        let GgufDitPlan { plan, .. } = compile_gguf_dit_plan(
            &path,
            &content,
            &WanNativeToDiffusersMapping,
            gguf_codec_registry(),
        )
        .unwrap();

        let dit = GgufDit::open(&path, &Device::Cpu, DType::F32).expect("the container opens");
        let receipt = dit.receipt();
        assert_eq!(receipt.mapping_id, WanNativeToDiffusersMapping::MAPPING_ID);
        assert_eq!(receipt.tensor_count, plan.tensor_count());
        assert_eq!(
            receipt.materialization,
            LogicalReadMaterialization::Materialized
        );
        assert_eq!(receipt.residency.len(), 1);
        assert_eq!(receipt.residency[0].codec_id, GGUF_CONTAINER_CODEC.codec_id);
        assert_eq!(
            receipt.resident_bytes(),
            plan.resident_bytes(),
            "the measured container residency must equal what the plan priced"
        );
        // And it really is measured: the same number falls out of the resident QTensors directly.
        let measured: u64 = dit
            .tensors
            .values()
            .map(|qt| qt.storage_size_in_bytes() as u64)
            .sum();
        assert_eq!(receipt.resident_bytes(), measured);
        std::fs::remove_file(&path).ok();
    }

    /// A container tensor outside the Wan DiT surface REFUSES the whole plan, by name — it is not
    /// passed through unchanged (which `remap_wan_key` alone would do) and not skipped.
    #[test]
    fn a_foreign_container_tensor_refuses_the_plan_by_name() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let mut tensors = native_wan_tensors(&cfg);
        // A real neighbour: the VACE hint stack a `Wan2.1-VACE-*-GGUF` ships.
        tensors.insert(
            "vace_blocks.0.before_proj.weight".into(),
            q4k(cfg.dim, cfg.dim),
        );
        let path = write_gguf(&tmp, &tensors, "foreign");
        let content = read_content(&path);
        let error = compile_gguf_dit_plan(
            &path,
            &content,
            &WanNativeToDiffusersMapping,
            gguf_codec_registry(),
        )
        .expect_err("a foreign tensor must refuse");
        assert_eq!(
            error,
            GgufPlanError::UnmappedKey {
                physical_key: "vace_blocks.0.before_proj.weight".into(),
            }
        );
        assert!(
            error
                .to_string()
                .contains("vace_blocks.0.before_proj.weight"),
            "{error}"
        );
        // And the loader surfaces it rather than degrading.
        let opened = GgufDit::open(&path, &Device::Cpu, DType::F32);
        let message = match opened {
            Ok(_) => panic!("a foreign tensor must fail the load"),
            Err(e) => e.to_string(),
        };
        assert!(
            message.contains("vace_blocks.0.before_proj.weight"),
            "{message}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// A ggml quant type this codec does not read refuses **by type name**, at plan time, before a
    /// single byte is decoded.
    #[test]
    fn an_unreadable_ggml_quant_type_refuses_by_name() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let mut tensors = native_wan_tensors(&cfg);
        // Q8_1 is a ggml activation intermediate, never a stored weight type.
        let data: Vec<f32> = (0..cfg.dim * cfg.dim)
            .map(|i| (i % 7) as f32 * 0.01)
            .collect();
        let t = Tensor::from_vec(data, (cfg.dim, cfg.dim), &Device::Cpu).unwrap();
        tensors.insert(
            "blocks.0.self_attn.q.weight".into(),
            QTensor::quantize(&t, GgmlDType::Q8_1).unwrap(),
        );
        let path = write_gguf(&tmp, &tensors, "q8_1");
        let content = read_content(&path);
        let error = compile_gguf_dit_plan(
            &path,
            &content,
            &WanNativeToDiffusersMapping,
            gguf_codec_registry(),
        )
        .expect_err("an unreadable quant type must refuse");
        assert_eq!(
            error,
            GgufPlanError::UnsupportedQuantType {
                physical_key: "blocks.0.self_attn.q.weight".into(),
                ggml_type: "Q8_1",
            }
        );
        let message = error.to_string();
        assert!(
            message.contains("blocks.0.self_attn.q.weight")
                && message.contains("Q8_1")
                && message.contains(GGUF_CONTAINER_CODEC.codec_id),
            "{message}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Planning against a registry that carries no GGUF row refuses by codec id, rather than
    /// inventing a codec or falling back to a safetensors row. This is what makes the codec
    /// *registration* load-bearing instead of decorative.
    #[test]
    fn an_unregistered_gguf_codec_refuses_the_plan() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let tensors = native_wan_tensors(&cfg);
        let path = write_gguf(&tmp, &tensors, "unregistered");
        let content = read_content(&path);
        // The engine's safetensors-only table: dense/fp8/mxfp8/int8/nvfp4, no GGUF container row.
        let error = compile_gguf_dit_plan(
            &path,
            &content,
            &WanNativeToDiffusersMapping,
            candle_gen::logical_weights::baseline_codec_registry(),
        )
        .expect_err("no GGUF codec means no plan");
        assert_eq!(
            error,
            GgufPlanError::UnregisteredCodec {
                codec_id: GGUF_CONTAINER_CODEC.codec_id,
            }
        );
        std::fs::remove_file(&path).ok();
    }

    // ── remap coverage: every Linear the 5B DiT expects (native → diffusers) ──────────────────────────

    /// [`remap_wan_key`] covers **every** Linear (and its bias) the 5B DiT reads — attention q/k/v/out,
    /// FFN, the condition embedders, `time_proj`, `proj_out`, `scale_shift_table` — for a native-Wan GGUF.
    /// A gap here would surface as a missing-tensor load error; pinning it as a pure-function test makes a
    /// remap regression fail loudly and locally.
    #[test]
    fn remap_covers_every_5b_linear() {
        // attention (self → attn1, cross → attn2) weights + biases + qk-norms
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.q.weight"),
            "blocks.7.attn1.to_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.k.bias"),
            "blocks.7.attn1.to_k.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.v.weight"),
            "blocks.7.attn1.to_v.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.o.weight"),
            "blocks.7.attn1.to_out.0.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.self_attn.norm_q.weight"),
            "blocks.7.attn1.norm_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.cross_attn.q.weight"),
            "blocks.7.attn2.to_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.cross_attn.o.bias"),
            "blocks.7.attn2.to_out.0.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.7.cross_attn.norm_k.weight"),
            "blocks.7.attn2.norm_k.weight"
        );
        // ffn + norm3 + block modulation
        assert_eq!(
            remap_wan_key("blocks.7.ffn.0.weight"),
            "blocks.7.ffn.net.0.proj.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.ffn.2.bias"),
            "blocks.7.ffn.net.2.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.7.norm3.weight"),
            "blocks.7.norm2.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.7.modulation"),
            "blocks.7.scale_shift_table"
        );
        // top-level embedders / head
        assert_eq!(
            remap_wan_key("text_embedding.0.weight"),
            "condition_embedder.text_embedder.linear_1.weight"
        );
        assert_eq!(
            remap_wan_key("text_embedding.2.bias"),
            "condition_embedder.text_embedder.linear_2.bias"
        );
        assert_eq!(
            remap_wan_key("time_embedding.0.weight"),
            "condition_embedder.time_embedder.linear_1.weight"
        );
        assert_eq!(
            remap_wan_key("time_embedding.2.weight"),
            "condition_embedder.time_embedder.linear_2.weight"
        );
        assert_eq!(
            remap_wan_key("time_projection.1.weight"),
            "condition_embedder.time_proj.weight"
        );
        assert_eq!(remap_wan_key("head.head.weight"), "proj_out.weight");
        assert_eq!(remap_wan_key("head.modulation"), "scale_shift_table");
        // dense sidecar that must pass through unchanged
        assert_eq!(
            remap_wan_key("patch_embedding.weight"),
            "patch_embedding.weight"
        );
    }

    // ── loader: resident (NOT dense), remap applied, missing key fails loud ───────────────────────────

    /// A native-keyed `.gguf` opens with each Linear held as a **resident k-quant `QTensor`** (Q4_K, not a
    /// dense `[out,in]` weight), reachable at its **diffusers** key (the remap fired), and the bias is a
    /// dense companion. The resident-not-dense guarantee — a naive `dequantize`-at-load would fail this.
    #[test]
    fn loader_keeps_kquant_resident_and_remaps() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let tensors = native_wan_tensors(&cfg);
        let path = write_gguf(&tmp, &tensors, "resident");
        let dit = GgufDit::open(&path, &Device::Cpu, DType::BF16).unwrap();

        // The native `blocks.0.self_attn.q` is reachable at the diffusers `blocks.0.attn1.to_q`.
        let q = dit
            .qlinear("blocks.0.attn1.to_q", cfg.dim, cfg.dim, true)
            .expect("remapped diffusers key resolves");
        assert!(
            q.is_packed(),
            "the k-quant weight must load quantized-resident, not dense"
        );
        let inner = q.base_qlinear().expect("packed base exposes the QLinear");
        assert_eq!(
            inner.quant_dtype(),
            Some(GgmlDType::Q4K),
            "the resident weight must stay Q4_K (NOT dequantized to a dense [out,in] at load)"
        );
        assert_eq!(
            inner.matmul_strategy(),
            Some(MatmulStrategy::DequantDense),
            "the forward must dequant-on-matmul (sc-7702-safe), not the int8-fast path"
        );
        std::fs::remove_file(&path).ok();
    }

    /// A missing / renamed key fails the load **loudly** (naming the key), never a silent dense fallback —
    /// the sc-12735 fail-loud contract. Drop `blocks.0.self_attn.q.weight` and the diffusers lookup errors.
    #[test]
    fn loader_missing_key_fails_loud() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let mut tensors = native_wan_tensors(&cfg);
        tensors.remove("blocks.0.self_attn.q.weight");
        let path = write_gguf(&tmp, &tensors, "missing");
        let dit = GgufDit::open(&path, &Device::Cpu, DType::BF16).unwrap();
        // `AdaptLinear` isn't `Debug`, so match rather than `expect_err`.
        let err = match dit.qlinear("blocks.0.attn1.to_q", cfg.dim, cfg.dim, true) {
            Ok(_) => panic!("a missing weight must error, not silently degrade"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("blocks.0.attn1.to_q.weight") && msg.contains("missing"),
            "the error must name the missing diffusers key: {msg}"
        );
        std::fs::remove_file(&path).ok();
    }

    // ── integration: the full WanTransformer::from_gguf build is entirely resident + runs a forward ───

    /// `WanTransformer::from_gguf` builds the whole 5B-shaped DiT with **every** adaptable projection
    /// packed-resident at `Q4_K` (walked via `visit_adaptable_mut`) — no projection accidentally
    /// dequantized to dense at load — reports `is_packed()`, and produces a finite velocity on a CPU
    /// forward (the dequant-on-matmul path executes end-to-end).
    #[test]
    fn from_gguf_builds_fully_resident_dit_and_forwards() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = gguf_cfg();
        let tensors = native_wan_tensors(&cfg);
        let path = write_gguf(&tmp, &tensors, "full");
        // F32 compute dtype so the CPU forward runs (CPU has no bf16 matmul); the k-quant weights stay
        // Q4_K resident regardless of the compute dtype — only the activation/sidecar dtype changes. On
        // CUDA the production path uses bf16 (DIT_DTYPE); this asserts the resident-QTensor mechanism.
        let dit = GgufDit::open(&path, &Device::Cpu, DType::F32).unwrap();
        let mut model = WanTransformer::from_gguf(&cfg, &dit).expect("from_gguf builds");
        assert!(
            model.is_packed(),
            "a GGUF DiT must report packed (is_packed)"
        );

        // Every adaptable projection is a resident Q4_K base — none dequantized to dense at load.
        let mut count = 0usize;
        model
            .visit_adaptable_mut(&mut |path, ql| {
                assert!(ql.is_packed(), "{path} must be packed-resident");
                let inner = ql.base_qlinear().expect("packed base");
                assert_eq!(
                    inner.quant_dtype(),
                    Some(GgmlDType::Q4K),
                    "{path} must stay Q4_K resident (not dense) after load"
                );
                count += 1;
                Ok(())
            })
            .unwrap();
        // 5 condition-embedder + num_layers×(4+4+2) block projections + proj_out.
        assert_eq!(
            count,
            5 + cfg.num_layers * 10 + 1,
            "every DiT Linear walked"
        );

        // A CPU forward exercises the dequant-on-matmul path end-to-end and stays finite.
        let latents =
            Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 2, 4, 4), &Device::Cpu).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &Device::Cpu).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(2, 2, 2, &Device::Cpu).unwrap();
        let vel = model
            .forward(&latents, &context, 700.0, &cos, &sin)
            .unwrap();
        assert_eq!(vel.dims(), &[1, cfg.out_channels, 2, 4, 4]);
        let max = vel
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max.is_finite(),
            "GGUF-resident DiT forward must be finite (got {max})"
        );
        std::fs::remove_file(&path).ok();
    }
}
