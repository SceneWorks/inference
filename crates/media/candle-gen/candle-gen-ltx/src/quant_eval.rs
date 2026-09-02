//! LTX-2.5 quant selection and terminal measurement evidence (sc-18777).
//!
//! Production and measurement are intentionally separate surfaces. Production `Quant::Q8` names the
//! hosted packed-q8 tier (`<snapshot>/q8`) — the released MLX-affine bundle that also packs the
//! Gemma 4 text encoder — promoted to a first-class Candle route in sc-18791 once the packed-q8
//! projection loader landed. It shares the ordinary packed loader with q4 and is admitted
//! unconditionally, exactly like bf16 and q4.
//!
//! The materially different *advanced* operators are unchanged by that promotion. INT8-ConvRot has
//! no [`LoadSpec`] selector at all (it is reached only by the terminal controller, which names its
//! mode explicitly), and NVFP4 stays fail-closed until a same-run, identity-bound receipt is
//! deliberately copied into [`ACCEPTED_MEASUREMENT_RECEIPTS`].

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{self, ltx_checkpoint::LtxBundle, LoadSpec, LtxComponent, Quant};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{dev_sampler::TransformerVariant, MODEL_25_ID};

pub const RUNTIME_BINDING_FILE: &str = "ltx25-quant-runtime-binding.json";
pub const RUNTIME_BINDING_SCHEMA: &str = "sceneworks-ltx25-quant-runtime-bindings-v1";
pub const LTX25_PUBLIC_REPOSITORY: &str = "SceneWorks/ltx-2.5-mlx";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ltx25QuantRuntimeBindings {
    pub schema_version: String,
    pub bindings: Vec<Ltx25QuantRuntimeIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ltx25QuantRuntimeIdentity {
    pub mode: Ltx25QuantMode,
    pub transformer_variant: TransformerVariant,
    pub inference_revision: String,
    pub executable_contract_sha256: String,
    pub executable_sha256: String,
    pub source_model_revision: String,
    pub source_model_inventory_sha256: String,
    pub source_bundle_subdir: String,
    pub source_bf16_text_encoder_subpath: String,
    pub source_runtime_bundle_sha256: String,
    pub source_selected_bundle_sha256: String,
    pub model_revision: String,
    pub model_inventory_sha256: String,
    pub bundle_subdir: String,
    pub bf16_text_encoder_subpath: String,
    pub runtime_bundle_sha256: String,
    pub selected_bundle_sha256: String,
    pub public_repository: String,
    pub public_readback_sha256: String,
    pub public_replay_receipt_sha256: String,
    pub public_replay_output_sha256: String,
    pub promotion_copy_sha256: String,
    pub reference_model_revision: String,
    pub reference_model_inventory_sha256: String,
    pub reference_bundle_subdir: String,
    pub reference_runtime_bundle_sha256: String,
    pub receipt_sha256: String,
    pub transcript_sha256: String,
    pub evidence_manifest_sha256: String,
    pub output_sha256: String,
    pub reference_output_sha256: String,
    pub reference_receipt_sha256: String,
    pub operator_kind: String,
    pub operator_contract_sha256: String,
    pub operator_weight_inventory_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ltx25QuantAcceptedMeasurement {
    pub receipt: Ltx25QuantMeasurementReceipt,
    pub runtime: Ltx25QuantRuntimeIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SnapshotInventoryEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub symlink_target: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SnapshotInventory {
    pub schema_version: &'static str,
    pub entries: Vec<SnapshotInventoryEntry>,
}

/// Every numeric source compared by the terminal controller.
///
/// `PackedQ8` is the production meaning of `Quant::Q8` (sc-18791): the hosted `q8/` bundle, loaded
/// by the same MLX-affine packed path as `Q4`. `Int8ConvRot` is the deliberately separate
/// terminal-only variant — keeping it unconstructible from [`LoadSpec`] prevents a packed-q8
/// observation from being reported under the ConvRot label, and vice versa.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Ltx25QuantMode {
    Bf16,
    Q4,
    PackedQ8,
    Int8ConvRot,
    Nvfp4,
}

impl Ltx25QuantMode {
    pub fn from_load_spec(spec: &LoadSpec) -> gen_core::Result<Self> {
        match spec.quantize {
            None => Ok(Self::Bf16),
            Some(Quant::Q4) => Ok(Self::Q4),
            Some(Quant::Q8) => Ok(Self::PackedQ8),
            Some(Quant::Nvfp4) => Ok(Self::Nvfp4),
        }
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q4 => "packed-q4",
            Self::PackedQ8 => "packed-q8",
            Self::Int8ConvRot => "int8-convrot",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// Bind the declared mode to the transformer's actual tensor encoding and quant descriptors.
    /// File names are not evidence and are deliberately ignored.
    pub fn validate_bundle_source(self, bundle: &LtxBundle) -> gen_core::Result<()> {
        let transformer = bundle.require(LtxComponent::Transformer)?.path();
        crate::advanced_quant::inspect_transformer_source(transformer, self)
            .map(|_| ())
            .map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "{MODEL_25_ID}: transformer does not satisfy the semantic {} source contract: {error}",
                    self.id()
                ))
            })
    }
}

/// CUDA identities represented by the current terminal matrix.
///
/// Datacenter Blackwell (`sm_100`/`sm_103`) deliberately falls into `OtherCuda`: it must never be
/// reported as the consumer `sm_120` NVFP4 lane merely because both products are called Blackwell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Ltx25GpuGeneration {
    NotCuda,
    AdaSm89,
    ConsumerBlackwellSm120,
    OtherCuda,
}

impl Ltx25GpuGeneration {
    pub const fn id(self) -> &'static str {
        match self {
            Self::NotCuda => "not-cuda",
            Self::AdaSm89 => "ada-sm89",
            Self::ConsumerBlackwellSm120 => "consumer-blackwell-sm120",
            Self::OtherCuda => "other-cuda",
        }
    }

    pub const fn from_compute_cap(compute_cap: Option<(i32, i32)>) -> Self {
        match compute_cap {
            None => Self::NotCuda,
            Some((8, 9)) => Self::AdaSm89,
            Some((12, 0)) => Self::ConsumerBlackwellSm120,
            Some(_) => Self::OtherCuda,
        }
    }

    /// Read the bound CUDA device's exact compute capability. No name-based generation guess is
    /// permitted, and a driver query failure is an error rather than an Ada fallback.
    pub fn from_device(device: &Device) -> gen_core::Result<Self> {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(Self::NotCuda);
        }
        #[cfg(feature = "cuda")]
        {
            use candle_gen::candle_core::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
            let Device::Cuda(cuda) = device else {
                unreachable!("CUDA variant checked above")
            };
            let stream = cuda.cuda_stream();
            let context = stream.context();
            let major = context
                .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .map_err(|error| {
                    gen_core::Error::Msg(format!(
                        "{MODEL_25_ID}: CUDA driver did not report compute-capability major: {error}"
                    ))
                })?;
            let minor = context
                .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(|error| {
                    gen_core::Error::Msg(format!(
                        "{MODEL_25_ID}: CUDA driver did not report compute-capability minor: {error}"
                    ))
                })?;
            Ok(Self::from_compute_cap(Some((major, minor))))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: a CUDA device was bound to a build without the cuda feature"
            )))
        }
    }
}

/// Immutable terminal input. Quality is comparable only within this exact fixture and geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx25QuantMeasurementCase {
    pub id: &'static str,
    pub mode: Ltx25QuantMode,
    pub gpu: Ltx25GpuGeneration,
    pub transformer_variant: TransformerVariant,
    pub fixture: &'static str,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub seed: u64,
}

const fn terminal_case(
    id: &'static str,
    mode: Ltx25QuantMode,
    gpu: Ltx25GpuGeneration,
) -> Ltx25QuantMeasurementCase {
    Ltx25QuantMeasurementCase {
        id,
        mode,
        gpu,
        transformer_variant: TransformerVariant::Distilled,
        fixture: "ltx25-production-latent-v1",
        width: 512,
        height: 512,
        frames: 17,
        fps: 24,
        seed: 18777,
    }
}

const fn terminal_dev_case(
    id: &'static str,
    mode: Ltx25QuantMode,
    gpu: Ltx25GpuGeneration,
) -> Ltx25QuantMeasurementCase {
    let mut case = terminal_case(id, mode, gpu);
    case.transformer_variant = TransformerVariant::Dev;
    case
}

/// The active physical campaign pool is consumer Blackwell `sm_120`. Unsupported generations stay
/// classified above so selecting them still fails closed, but they are not measurement rows.
///
/// sc-18791: every row must name a bundle the public `SceneWorks/ltx-2.5-mlx` release actually
/// ships. That release publishes only `distilled/{bf16,q4,q8}` and `dev/{bf16,q4,q8}` — it has no
/// `bundles/` tree at all — so the INT8-ConvRot and NVFP4 candidates have no measurable case and
/// are not rows here. `Ltx25QuantMode::Int8ConvRot` and `::Nvfp4` remain classified selectors:
/// `admit` refuses them because no terminal case covers them, which is the same fail-closed
/// outcome the empty receipt allowlist already produced. Publishing those bundles is what makes
/// the rows representable again.
pub const TERMINAL_MEASUREMENT_CASES: &[Ltx25QuantMeasurementCase] = &[
    terminal_case(
        "ltx25-bf16-blackwell-v1",
        Ltx25QuantMode::Bf16,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_case(
        "ltx25-packed-q4-blackwell-v1",
        Ltx25QuantMode::Q4,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_case(
        "ltx25-packed-q8-blackwell-v1",
        Ltx25QuantMode::PackedQ8,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_dev_case(
        "ltx25-bf16-blackwell-dev-v1",
        Ltx25QuantMode::Bf16,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_dev_case(
        "ltx25-packed-q4-blackwell-dev-v1",
        Ltx25QuantMode::Q4,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_dev_case(
        "ltx25-packed-q8-blackwell-dev-v1",
        Ltx25QuantMode::PackedQ8,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
];

pub fn measurement_case(id: &str) -> Option<&'static Ltx25QuantMeasurementCase> {
    TERMINAL_MEASUREMENT_CASES.iter().find(|case| case.id == id)
}

/// Quality values emitted beside the output hash.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ltx25QuantQuality {
    pub reference_psnr: f64,
    pub reference_ssim: f64,
    pub temporal_boundary_drift: f64,
    pub silent_zero_video_passed: bool,
    pub silent_zero_audio_passed: bool,
}

/// One real-weight observation, bound to exact case, model/code inventory, hardware, transcript,
/// evidence manifest, and generated artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ltx25QuantMeasurementReceipt {
    pub schema_version: String,
    pub case_id: String,
    pub mode: Ltx25QuantMode,
    pub gpu_generation: Ltx25GpuGeneration,
    pub transformer_variant: TransformerVariant,
    pub fixture: String,
    pub observed_width: u32,
    pub observed_height: u32,
    pub observed_frames: u32,
    pub observed_fps: u32,
    pub seed: u64,
    pub inference_revision: String,
    pub executable_contract_sha256: String,
    pub executable_sha256: String,
    pub model_revision: String,
    pub model_inventory_sha256: String,
    pub bundle_subdir: String,
    pub bf16_text_encoder_subpath: String,
    pub runtime_bundle_sha256: String,
    pub selected_bundle_sha256: String,
    pub reference_model_revision: String,
    pub reference_model_inventory_sha256: String,
    pub reference_bundle_subdir: String,
    pub reference_runtime_bundle_sha256: String,
    pub reference_selected_bundle_sha256: String,
    pub gpu_name: String,
    pub compute_capability: String,
    pub driver_version: String,
    pub harness_version: String,
    pub run_nonce_sha256: String,
    pub transcript_sha256: String,
    pub evidence_manifest_sha256: String,
    pub output_sha256: String,
    pub reference_output_sha256: String,
    pub reference_receipt_sha256: String,
    pub operator_kind: String,
    pub operator_contract_sha256: String,
    pub operator_weight_inventory_sha256: String,
    pub executed_projection_count: u32,
    pub declared_projection_count: u32,
    pub baseline_vram_bytes: u64,
    pub peak_vram_bytes: u64,
    pub wall_clock_ms: u64,
    pub quality: Ltx25QuantQuality,
    pub receipt_sha256: String,
}

/// Internal pre-seal shape. Only the terminal controller can turn an observation into a receipt;
/// production code only validates an already-reviewed entry.
#[cfg(any(test, feature = "terminal-quant-measurement"))]
pub(crate) struct Ltx25QuantMeasurementDraft {
    pub case_id: String,
    pub mode: Ltx25QuantMode,
    pub gpu_generation: Ltx25GpuGeneration,
    pub transformer_variant: TransformerVariant,
    pub fixture: String,
    pub observed_width: u32,
    pub observed_height: u32,
    pub observed_frames: u32,
    pub observed_fps: u32,
    pub seed: u64,
    pub inference_revision: String,
    pub executable_contract_sha256: String,
    pub executable_sha256: String,
    pub model_revision: String,
    pub model_inventory_sha256: String,
    pub bundle_subdir: String,
    pub bf16_text_encoder_subpath: String,
    pub runtime_bundle_sha256: String,
    pub selected_bundle_sha256: String,
    pub reference_model_revision: String,
    pub reference_model_inventory_sha256: String,
    pub reference_bundle_subdir: String,
    pub reference_runtime_bundle_sha256: String,
    pub reference_selected_bundle_sha256: String,
    pub gpu_name: String,
    pub compute_capability: String,
    pub driver_version: String,
    pub harness_version: String,
    pub run_nonce_sha256: String,
    pub transcript_sha256: String,
    pub evidence_manifest_sha256: String,
    pub output_sha256: String,
    pub reference_output_sha256: String,
    pub reference_receipt_sha256: String,
    pub operator_kind: String,
    pub operator_contract_sha256: String,
    pub operator_weight_inventory_sha256: String,
    pub executed_projection_count: u32,
    pub declared_projection_count: u32,
    pub baseline_vram_bytes: u64,
    pub peak_vram_bytes: u64,
    pub wall_clock_ms: u64,
    pub quality: Ltx25QuantQuality,
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_normalized_bundle_subdir(value: &str) -> bool {
    value == "."
        || (!value.is_empty()
            && !value.starts_with('/')
            && !value.ends_with('/')
            && !value.contains(['\\', ':'])
            && value
                .split('/')
                .all(|component| !component.is_empty() && component != "." && component != ".."))
}

impl Ltx25QuantMeasurementReceipt {
    fn canonical_payload(&self) -> String {
        fn push_string(fields: &mut Vec<String>, label: &str, value: &str) {
            fields.push(format!("{label}:{}:{value}", value.len()));
        }
        let mut fields = Vec::new();
        push_string(&mut fields, "schema", &self.schema_version);
        push_string(&mut fields, "case", &self.case_id);
        push_string(&mut fields, "mode", self.mode.id());
        push_string(&mut fields, "gpu_generation", self.gpu_generation.id());
        push_string(
            &mut fields,
            "transformer_variant",
            self.transformer_variant.id(),
        );
        push_string(&mut fields, "fixture", &self.fixture);
        fields.push(format!(
            "geometry:{}x{}x{}@{}",
            self.observed_width, self.observed_height, self.observed_frames, self.observed_fps
        ));
        fields.push(format!("seed:{}", self.seed));
        push_string(&mut fields, "inference_revision", &self.inference_revision);
        push_string(
            &mut fields,
            "executable_contract_sha256",
            &self.executable_contract_sha256,
        );
        push_string(&mut fields, "executable_sha256", &self.executable_sha256);
        push_string(&mut fields, "model_revision", &self.model_revision);
        push_string(
            &mut fields,
            "model_inventory_sha256",
            &self.model_inventory_sha256,
        );
        push_string(&mut fields, "bundle_subdir", &self.bundle_subdir);
        push_string(
            &mut fields,
            "bf16_text_encoder_subpath",
            &self.bf16_text_encoder_subpath,
        );
        push_string(
            &mut fields,
            "runtime_bundle_sha256",
            &self.runtime_bundle_sha256,
        );
        push_string(
            &mut fields,
            "selected_bundle_sha256",
            &self.selected_bundle_sha256,
        );
        push_string(
            &mut fields,
            "reference_model_revision",
            &self.reference_model_revision,
        );
        push_string(
            &mut fields,
            "reference_model_inventory_sha256",
            &self.reference_model_inventory_sha256,
        );
        push_string(
            &mut fields,
            "reference_bundle_subdir",
            &self.reference_bundle_subdir,
        );
        push_string(
            &mut fields,
            "reference_runtime_bundle_sha256",
            &self.reference_runtime_bundle_sha256,
        );
        push_string(
            &mut fields,
            "reference_selected_bundle_sha256",
            &self.reference_selected_bundle_sha256,
        );
        push_string(&mut fields, "gpu_name", &self.gpu_name);
        push_string(&mut fields, "compute_capability", &self.compute_capability);
        push_string(&mut fields, "driver_version", &self.driver_version);
        push_string(&mut fields, "harness_version", &self.harness_version);
        push_string(&mut fields, "run_nonce_sha256", &self.run_nonce_sha256);
        push_string(&mut fields, "transcript_sha256", &self.transcript_sha256);
        push_string(
            &mut fields,
            "evidence_manifest_sha256",
            &self.evidence_manifest_sha256,
        );
        push_string(
            &mut fields,
            "reference_receipt_sha256",
            &self.reference_receipt_sha256,
        );
        push_string(&mut fields, "operator_kind", &self.operator_kind);
        push_string(
            &mut fields,
            "operator_contract_sha256",
            &self.operator_contract_sha256,
        );
        push_string(
            &mut fields,
            "operator_weight_inventory_sha256",
            &self.operator_weight_inventory_sha256,
        );
        fields.push(format!(
            "operator_counts:{}:{}",
            self.executed_projection_count, self.declared_projection_count
        ));
        push_string(&mut fields, "output_sha256", &self.output_sha256);
        push_string(
            &mut fields,
            "reference_output_sha256",
            &self.reference_output_sha256,
        );
        fields.push(format!("baseline_vram_bytes:{}", self.baseline_vram_bytes));
        fields.push(format!("peak_vram_bytes:{}", self.peak_vram_bytes));
        fields.push(format!("wall_clock_ms:{}", self.wall_clock_ms));
        fields.push(format!(
            "quality:{:016x}:{:016x}:{:016x}:{}:{}",
            self.quality.reference_psnr.to_bits(),
            self.quality.reference_ssim.to_bits(),
            self.quality.temporal_boundary_drift.to_bits(),
            self.quality.silent_zero_video_passed,
            self.quality.silent_zero_audio_passed,
        ));
        fields.join("\n")
    }

    #[cfg(any(test, feature = "terminal-quant-measurement"))]
    pub(crate) fn seal(draft: Ltx25QuantMeasurementDraft) -> Self {
        let mut receipt = Self {
            schema_version: "sceneworks-ltx25-quant-receipt-v6".to_owned(),
            case_id: draft.case_id,
            mode: draft.mode,
            gpu_generation: draft.gpu_generation,
            transformer_variant: draft.transformer_variant,
            fixture: draft.fixture,
            observed_width: draft.observed_width,
            observed_height: draft.observed_height,
            observed_frames: draft.observed_frames,
            observed_fps: draft.observed_fps,
            seed: draft.seed,
            inference_revision: draft.inference_revision,
            executable_contract_sha256: draft.executable_contract_sha256,
            executable_sha256: draft.executable_sha256,
            model_revision: draft.model_revision,
            model_inventory_sha256: draft.model_inventory_sha256,
            bundle_subdir: draft.bundle_subdir,
            bf16_text_encoder_subpath: draft.bf16_text_encoder_subpath,
            runtime_bundle_sha256: draft.runtime_bundle_sha256,
            selected_bundle_sha256: draft.selected_bundle_sha256,
            reference_model_revision: draft.reference_model_revision,
            reference_model_inventory_sha256: draft.reference_model_inventory_sha256,
            reference_bundle_subdir: draft.reference_bundle_subdir,
            reference_runtime_bundle_sha256: draft.reference_runtime_bundle_sha256,
            reference_selected_bundle_sha256: draft.reference_selected_bundle_sha256,
            gpu_name: draft.gpu_name,
            compute_capability: draft.compute_capability,
            driver_version: draft.driver_version,
            harness_version: draft.harness_version,
            run_nonce_sha256: draft.run_nonce_sha256,
            transcript_sha256: draft.transcript_sha256,
            evidence_manifest_sha256: draft.evidence_manifest_sha256,
            output_sha256: draft.output_sha256,
            reference_output_sha256: draft.reference_output_sha256,
            reference_receipt_sha256: draft.reference_receipt_sha256,
            operator_kind: draft.operator_kind,
            operator_contract_sha256: draft.operator_contract_sha256,
            operator_weight_inventory_sha256: draft.operator_weight_inventory_sha256,
            executed_projection_count: draft.executed_projection_count,
            declared_projection_count: draft.declared_projection_count,
            baseline_vram_bytes: draft.baseline_vram_bytes,
            peak_vram_bytes: draft.peak_vram_bytes,
            wall_clock_ms: draft.wall_clock_ms,
            quality: draft.quality,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = sha256_hex(receipt.canonical_payload().as_bytes());
        receipt
    }

    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let case = measurement_case(&self.case_id);
        if case.is_none() {
            errors.push(format!(
                "unknown LTX-2.5 quant measurement case {:?}",
                self.case_id
            ));
        }
        if let Some(case) = case {
            if self.mode != case.mode {
                errors.push(format!(
                    "case {} requires mode {}, got {}",
                    case.id,
                    case.mode.id(),
                    self.mode.id()
                ));
            }
            if self.gpu_generation != case.gpu {
                errors.push(format!(
                    "case {} requires GPU generation {}, got {}",
                    case.id,
                    case.gpu.id(),
                    self.gpu_generation.id()
                ));
            }
            if self.transformer_variant != case.transformer_variant {
                errors.push(format!(
                    "case {} requires transformer variant {}, got {}",
                    case.id,
                    case.transformer_variant.id(),
                    self.transformer_variant.id()
                ));
            }
            if self.fixture != case.fixture
                || self.observed_width != case.width
                || self.observed_height != case.height
                || self.observed_frames != case.frames
                || self.observed_fps != case.fps
                || self.seed != case.seed
            {
                errors.push(format!(
                    "case {} fixture/geometry/seed identity changed",
                    case.id
                ));
            }
        }
        if self.schema_version != "sceneworks-ltx25-quant-receipt-v6" {
            errors.push("unknown receipt schema version".to_owned());
        }
        for (label, subdir) in [
            ("bundle subdir", self.bundle_subdir.as_str()),
            (
                "reference bundle subdir",
                self.reference_bundle_subdir.as_str(),
            ),
        ] {
            if !is_normalized_bundle_subdir(subdir) {
                errors.push(format!(
                    "{label} must be a normalized relative path using forward slashes"
                ));
            }
        }
        let advanced = matches!(
            self.mode,
            Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4
        );
        if advanced {
            if !is_normalized_bundle_subdir(&self.bf16_text_encoder_subpath)
                || self.bf16_text_encoder_subpath == "."
            {
                errors.push(
                    "advanced receipt requires a normalized explicit bf16 text encoder subpath"
                        .to_owned(),
                );
            }
        } else if !self.bf16_text_encoder_subpath.is_empty() {
            errors.push(
                "non-advanced receipt must not bind an external bf16 text encoder".to_owned(),
            );
        }
        for (label, value, expected) in [
            ("inference revision", self.inference_revision.as_str(), 40),
            (
                "executable contract SHA-256",
                self.executable_contract_sha256.as_str(),
                64,
            ),
            ("executable SHA-256", self.executable_sha256.as_str(), 64),
            ("model revision", self.model_revision.as_str(), 40),
            (
                "model inventory SHA-256",
                self.model_inventory_sha256.as_str(),
                64,
            ),
            (
                "runtime bundle SHA-256",
                self.runtime_bundle_sha256.as_str(),
                64,
            ),
            (
                "selected bundle SHA-256",
                self.selected_bundle_sha256.as_str(),
                64,
            ),
            (
                "reference model revision",
                self.reference_model_revision.as_str(),
                40,
            ),
            (
                "reference model inventory SHA-256",
                self.reference_model_inventory_sha256.as_str(),
                64,
            ),
            (
                "reference runtime bundle SHA-256",
                self.reference_runtime_bundle_sha256.as_str(),
                64,
            ),
            (
                "reference selected bundle SHA-256",
                self.reference_selected_bundle_sha256.as_str(),
                64,
            ),
            ("run nonce SHA-256", self.run_nonce_sha256.as_str(), 64),
            ("transcript SHA-256", self.transcript_sha256.as_str(), 64),
            (
                "evidence manifest SHA-256",
                self.evidence_manifest_sha256.as_str(),
                64,
            ),
            ("output SHA-256", self.output_sha256.as_str(), 64),
            (
                "reference output SHA-256",
                self.reference_output_sha256.as_str(),
                64,
            ),
            (
                "reference receipt SHA-256",
                self.reference_receipt_sha256.as_str(),
                64,
            ),
            (
                "operator contract SHA-256",
                self.operator_contract_sha256.as_str(),
                64,
            ),
            (
                "operator weight inventory SHA-256",
                self.operator_weight_inventory_sha256.as_str(),
                64,
            ),
            ("receipt SHA-256", self.receipt_sha256.as_str(), 64),
        ] {
            if !is_lower_hex(value, expected) {
                errors.push(format!(
                    "{label} must be {expected} lowercase hexadecimal characters"
                ));
            }
        }
        let expected_operator = match self.mode {
            Ltx25QuantMode::Bf16 => "dense-linear",
            Ltx25QuantMode::Q4 | Ltx25QuantMode::PackedQ8 => "mlx-affine-dequant",
            Ltx25QuantMode::Int8ConvRot => "int8-convrot-rht-cublaslt-igemm",
            Ltx25QuantMode::Nvfp4 => "native-nvfp4-cublaslt-w4a4",
        };
        if self.operator_kind != expected_operator {
            errors.push(format!(
                "mode {} requires operator {expected_operator}, got {}",
                self.mode.id(),
                self.operator_kind
            ));
        }
        if self.executed_projection_count == 0 {
            errors.push("executed projection count must be positive".to_owned());
        }
        if matches!(
            self.mode,
            Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4
        ) && (self.declared_projection_count == 0
            || self.declared_projection_count > self.executed_projection_count)
        {
            errors.push("advanced receipt did not execute every declared projection".to_owned());
        }
        for (label, value) in [
            ("GPU name", self.gpu_name.as_str()),
            ("compute capability", self.compute_capability.as_str()),
            ("driver version", self.driver_version.as_str()),
            ("harness version", self.harness_version.as_str()),
        ] {
            if value.trim().is_empty() {
                errors.push(format!("{label} must be non-empty"));
            }
        }
        let expected_cap = match self.gpu_generation {
            Ltx25GpuGeneration::AdaSm89 => Some("sm_89"),
            Ltx25GpuGeneration::ConsumerBlackwellSm120 => Some("sm_120"),
            Ltx25GpuGeneration::NotCuda | Ltx25GpuGeneration::OtherCuda => None,
        };
        if let Some(expected_cap) = expected_cap {
            if self.compute_capability != expected_cap {
                errors.push(format!(
                    "GPU generation {} requires compute capability {expected_cap}, got {}",
                    self.gpu_generation.id(),
                    self.compute_capability
                ));
            }
        }
        if self.mode == Ltx25QuantMode::Nvfp4
            && (self.gpu_generation != Ltx25GpuGeneration::ConsumerBlackwellSm120
                || self.compute_capability != "sm_120")
        {
            errors.push("nvfp4 evidence requires exact consumer Blackwell sm_120; datacenter Blackwell is not interchangeable".to_owned());
        }
        if self.mode == Ltx25QuantMode::Bf16
            && (self.reference_model_revision != self.model_revision
                || self.reference_model_inventory_sha256 != self.model_inventory_sha256
                || self.reference_bundle_subdir != self.bundle_subdir
                || self.reference_runtime_bundle_sha256 != self.runtime_bundle_sha256
                || self.reference_selected_bundle_sha256 != self.selected_bundle_sha256
                || self.reference_output_sha256 != self.output_sha256)
        {
            errors.push("bf16 receipt must be self-bound as its own reference identity".to_owned());
        }
        if self.peak_vram_bytes == 0 {
            errors.push("peak VRAM bytes must be positive".to_owned());
        }
        if self.peak_vram_bytes < self.baseline_vram_bytes {
            errors.push("peak VRAM bytes cannot be below the sampled baseline".to_owned());
        }
        if self.wall_clock_ms == 0 {
            errors.push("wall-clock milliseconds must be positive".to_owned());
        }
        for (label, value) in [
            ("reference PSNR", self.quality.reference_psnr),
            ("reference SSIM", self.quality.reference_ssim),
            (
                "temporal-boundary drift",
                self.quality.temporal_boundary_drift,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                errors.push(format!("{label} must be finite and non-negative"));
            }
        }
        if self.quality.reference_ssim > 1.0 {
            errors.push("reference SSIM must be at most 1.0".to_owned());
        }
        if !self.quality.silent_zero_video_passed {
            errors.push("silent/zero video check did not pass".to_owned());
        }
        if !self.quality.silent_zero_audio_passed {
            errors.push("silent/zero audio check did not pass".to_owned());
        }
        if is_lower_hex(&self.receipt_sha256, 64) {
            let expected = sha256_hex(self.canonical_payload().as_bytes());
            if self.receipt_sha256 != expected {
                errors.push(
                    "receipt seal does not match its canonical identity/evidence payload"
                        .to_owned(),
                );
            }
        }
        errors
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ltx25QuantAdmission {
    Admitted,
    Refused { reason: String },
}

/// Generated promotion data is deliberately outside the executable source-contract digest.
/// Adding a reviewed receipt must not change the contract that the already-sealed campaign names.
/// The Rust admission implementation and receipt schema remain inside that digest; only this inert
/// allowlist payload is excluded, avoiding a self-referential promotion cycle.
pub static ACCEPTED_MEASUREMENT_RECEIPTS: LazyLock<Vec<Ltx25QuantAcceptedMeasurement>> =
    LazyLock::new(|| {
        serde_json::from_str(include_str!("accepted_quant_receipts.allowlist"))
            .expect("accepted LTX-2.5 quant allowlist must be valid JSON")
    });

pub fn admit(
    mode: Ltx25QuantMode,
    gpu: Ltx25GpuGeneration,
    variant: TransformerVariant,
    runtime: Option<&Ltx25QuantRuntimeIdentity>,
    accepted: &[Ltx25QuantAcceptedMeasurement],
) -> Ltx25QuantAdmission {
    // sc-18791: packed-q8 is an ordinary released tier alongside bf16 and packed-q4 — it falls
    // through to the unconditional arm below. Only the advanced operators are receipt-gated.
    if mode == Ltx25QuantMode::Nvfp4 && gpu != Ltx25GpuGeneration::ConsumerBlackwellSm120 {
        return Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: nvfp4 requires exact consumer Blackwell sm_120; detected {}. Datacenter Blackwell is not this lane, and fallback to bf16/q4 is forbidden", gpu.id()) };
    }
    if !matches!(mode, Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4) {
        return Ltx25QuantAdmission::Admitted;
    }
    let Some(case) = TERMINAL_MEASUREMENT_CASES
        .iter()
        .find(|case| case.mode == mode && case.gpu == gpu && case.transformer_variant == variant)
    else {
        return Ltx25QuantAdmission::Refused {
            reason: format!(
                "{MODEL_25_ID}: {} has no supported terminal measurement case for {}",
                mode.id(),
                gpu.id()
            ),
        };
    };
    match accepted
        .iter()
        .find(|accepted| accepted.receipt.case_id == case.id)
    {
        Some(accepted)
            if accepted.receipt.validation_errors().is_empty()
                && runtime.is_some_and(|runtime| {
                    runtime == &accepted.runtime
                        && receipt_matches_runtime(&accepted.receipt, runtime)
                }) =>
        {
            Ltx25QuantAdmission::Admitted
        }
        Some(accepted) if !accepted.receipt.validation_errors().is_empty() => Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: {} measurement receipt is invalid: {}", case.id, accepted.receipt.validation_errors().join("; ")) },
        Some(_) => Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: {} receipt does not match the active code/model/bundle/evidence runtime identity; replay is refused", case.id) },
        None => Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: {} is selectable but not catalog-adopted until the terminal campaign records the {} receipt (exact code/model/GPU, VRAM, wall-clock, output, transcript, and quality)", mode.id(), case.id) },
    }
}

fn receipt_matches_runtime(
    receipt: &Ltx25QuantMeasurementReceipt,
    runtime: &Ltx25QuantRuntimeIdentity,
) -> bool {
    receipt.mode == runtime.mode
        && receipt.transformer_variant == runtime.transformer_variant
        && receipt.inference_revision == runtime.inference_revision
        && receipt.executable_contract_sha256 == runtime.executable_contract_sha256
        && receipt.executable_sha256 == runtime.executable_sha256
        && receipt.model_revision == runtime.source_model_revision
        && receipt.model_inventory_sha256 == runtime.source_model_inventory_sha256
        && receipt.bundle_subdir == runtime.source_bundle_subdir
        && receipt.bf16_text_encoder_subpath == runtime.source_bf16_text_encoder_subpath
        && receipt.runtime_bundle_sha256 == runtime.source_runtime_bundle_sha256
        && receipt.selected_bundle_sha256 == runtime.source_selected_bundle_sha256
        && receipt.reference_model_revision == runtime.reference_model_revision
        && receipt.reference_model_inventory_sha256 == runtime.reference_model_inventory_sha256
        && receipt.reference_bundle_subdir == runtime.reference_bundle_subdir
        && receipt.reference_runtime_bundle_sha256 == runtime.reference_runtime_bundle_sha256
        && receipt.receipt_sha256 == runtime.receipt_sha256
        && receipt.transcript_sha256 == runtime.transcript_sha256
        && receipt.evidence_manifest_sha256 == runtime.evidence_manifest_sha256
        && receipt.output_sha256 == runtime.output_sha256
        && receipt.reference_output_sha256 == runtime.reference_output_sha256
        && receipt.reference_receipt_sha256 == runtime.reference_receipt_sha256
        && receipt.operator_kind == runtime.operator_kind
        && receipt.operator_contract_sha256 == runtime.operator_contract_sha256
        && receipt.operator_weight_inventory_sha256 == runtime.operator_weight_inventory_sha256
        && runtime.public_repository == LTX25_PUBLIC_REPOSITORY
        && is_lower_hex(&runtime.public_readback_sha256, 64)
        && is_lower_hex(&runtime.public_replay_receipt_sha256, 64)
        && runtime.public_replay_output_sha256 == receipt.output_sha256
        && runtime.source_selected_bundle_sha256 == runtime.selected_bundle_sha256
        && runtime.promotion_copy_sha256 == promotion_copy_sha256(receipt, runtime)
}

pub(crate) fn promotion_copy_sha256(
    receipt: &Ltx25QuantMeasurementReceipt,
    runtime: &Ltx25QuantRuntimeIdentity,
) -> String {
    let rows = [
        format!("receipt:{}", receipt.receipt_sha256),
        format!("source-revision:{}", runtime.source_model_revision),
        format!("source-inventory:{}", runtime.source_model_inventory_sha256),
        format!("source-subdir:{}", runtime.source_bundle_subdir),
        format!(
            "source-bf16-text-encoder:{}",
            runtime.source_bf16_text_encoder_subpath
        ),
        format!("source-bundle:{}", runtime.source_runtime_bundle_sha256),
        format!("source-selected:{}", runtime.source_selected_bundle_sha256),
        format!("public-revision:{}", runtime.model_revision),
        format!("public-inventory:{}", runtime.model_inventory_sha256),
        format!("public-subdir:{}", runtime.bundle_subdir),
        format!(
            "public-bf16-text-encoder:{}",
            runtime.bf16_text_encoder_subpath
        ),
        format!("public-bundle:{}", runtime.runtime_bundle_sha256),
        format!("public-selected:{}", runtime.selected_bundle_sha256),
        format!("public-repository:{}", runtime.public_repository),
        format!("public-readback:{}", runtime.public_readback_sha256),
        format!(
            "public-replay-receipt:{}",
            runtime.public_replay_receipt_sha256
        ),
        format!(
            "public-replay-output:{}",
            runtime.public_replay_output_sha256
        ),
    ];
    sha256_hex(rows.join("\n").as_bytes())
}

pub const fn catalog_advertised(mode: Ltx25QuantMode) -> bool {
    matches!(mode, Ltx25QuantMode::Q4)
}

fn file_sha256(path: &Path) -> gen_core::Result<String> {
    let bytes = fs::read(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "read runtime identity file {}: {error}",
            path.display()
        ))
    })?;
    Ok(sha256_hex(&bytes))
}

fn snapshot_root(spec: &LoadSpec) -> gen_core::Result<PathBuf> {
    let root = match &spec.weights {
        gen_core::WeightsSource::Dir(path) => path.clone(),
        gen_core::WeightsSource::File(path) => path
            .parent()
            .ok_or_else(|| gen_core::Error::Msg("weights file has no parent".into()))?
            .to_path_buf(),
    };
    fs::canonicalize(&root).map_err(|error| {
        gen_core::Error::Msg(format!(
            "canonicalize model snapshot {}: {error}",
            root.display()
        ))
    })
}

fn snapshot_revision(root: &Path) -> gen_core::Result<String> {
    let revision = root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|revision| is_lower_hex(revision, 40))
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: model root {} is not an immutable 40-hex Hugging Face snapshot revision",
                root.display()
            ))
        })?;
    if root.parent().and_then(Path::file_name) != Some(std::ffi::OsStr::new("snapshots")) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: model root {} is not under a Hugging Face snapshots directory",
            root.display()
        )));
    }
    Ok(revision.to_owned())
}

fn canonical_public_snapshot_for_selected(
    selected_root: &Path,
    bundle_subdir: &str,
) -> gen_core::Result<PathBuf> {
    if !is_normalized_bundle_subdir(bundle_subdir) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: public bundle subdir {bundle_subdir:?} is not normalized"
        )));
    }
    let selected_root = fs::canonicalize(selected_root).map_err(|error| {
        gen_core::Error::Msg(format!(
            "canonicalize selected public bundle {}: {error}",
            selected_root.display()
        ))
    })?;
    let snapshot = if snapshot_revision(&selected_root).is_ok() {
        selected_root.clone()
    } else {
        if bundle_subdir == "." {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: nested public weights require a non-root reviewed bundle subdir"
            )));
        }
        let mut candidate = selected_root.clone();
        for _ in Path::new(bundle_subdir).components() {
            if !candidate.pop() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{MODEL_25_ID}: selected public bundle cannot be mapped to its reviewed snapshot root"
                )));
            }
        }
        let expected_selected =
            fs::canonicalize(candidate.join(bundle_subdir)).map_err(|error| {
                gen_core::Error::Msg(format!(
                    "canonicalize reviewed public bundle {}: {error}",
                    candidate.join(bundle_subdir).display()
                ))
            })?;
        if expected_selected != selected_root {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: active weights {} do not equal reviewed public bundle subdir {bundle_subdir:?}",
                selected_root.display()
            )));
        }
        snapshot_revision(&candidate)?;
        candidate
    };
    let expected_repo_dir = format!("models--{}", LTX25_PUBLIC_REPOSITORY.replace('/', "--"));
    let actual_repo_dir = snapshot
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .and_then(|name| name.to_str());
    if actual_repo_dir != Some(expected_repo_dir.as_str()) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: public snapshot must be under canonical Hugging Face repository cache {expected_repo_dir:?}"
        )));
    }
    Ok(snapshot)
}

/// Select one logical file inside an immutable snapshot without resolving its final HF-cache
/// symlink out to `blobs/`. Parent-directory symlinks are forbidden. A final symlink is accepted
/// only when its target is one file in the same canonical repository blob store.
pub(crate) fn snapshot_bound_file(
    snapshot: &Path,
    relative: &Path,
    label: &str,
) -> gen_core::Result<PathBuf> {
    let components = relative.components().collect::<Vec<_>>();
    if components.is_empty()
        || components
            .iter()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: {label} must be a non-empty traversal-free relative path"
        )));
    }
    let mut logical = snapshot.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        logical.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&logical).map_err(|error| {
            gen_core::Error::Msg(format!(
                "inspect {label} {} inside immutable snapshot: {error}",
                logical.display()
            ))
        })?;
        let final_component = index + 1 == components.len();
        if !final_component && metadata.file_type().is_symlink() {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: {label} parent {} may not be a symlink",
                logical.display()
            )));
        }
        if final_component {
            if metadata.file_type().is_symlink() {
                let target = fs::canonicalize(&logical)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
                let repo_root = snapshot
                    .parent()
                    .filter(|path| path.file_name().is_some_and(|name| name == "snapshots"))
                    .and_then(Path::parent)
                    .ok_or_else(|| {
                        gen_core::Error::Unsupported(format!(
                            "{MODEL_25_ID}: symlinked {label} requires canonical <repo>/snapshots/<revision> layout"
                        ))
                    })?;
                let blobs = fs::canonicalize(repo_root.join("blobs"))
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
                if !target.is_file() || !target.starts_with(&blobs) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{MODEL_25_ID}: symlinked {label} {} resolves outside canonical blob store {}",
                        logical.display(),
                        blobs.display()
                    )));
                }
            } else if !metadata.is_file() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{MODEL_25_ID}: {label} {} must be one file",
                    logical.display()
                )));
            }
        }
    }
    Ok(logical)
}

/// Convert the ordinary nested SceneWorks tier selection into the exact full-snapshot-plus-explicit-
/// components shape used by terminal measurement. The reviewed bundle subdir must map the active
/// weights to one canonical public HF snapshot; sibling variants can never be discovered implicitly.
pub fn stage_public_runtime_spec(
    spec: &LoadSpec,
    promotion: &Ltx25QuantRuntimeIdentity,
) -> gen_core::Result<LoadSpec> {
    if promotion.public_repository != LTX25_PUBLIC_REPOSITORY
        || !is_normalized_bundle_subdir(&promotion.bf16_text_encoder_subpath)
        || promotion.bf16_text_encoder_subpath == "."
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: promoted runtime lacks an exact public BF16 text-encoder binding"
        )));
    }
    let selected_root = snapshot_root(spec)?;
    let snapshot =
        canonical_public_snapshot_for_selected(&selected_root, &promotion.bundle_subdir)?;
    let encoder = snapshot_bound_file(
        &snapshot,
        Path::new(&promotion.bf16_text_encoder_subpath),
        "promoted BF16 text encoder",
    )?;
    let mut staged = spec.clone();
    staged.components.insert(
        LtxComponent::TextEncoder.id().to_owned(),
        gen_core::WeightsSource::File(encoder),
    );
    Ok(staged)
}

pub fn bind_public_runtime_spec(
    spec: &LoadSpec,
    bundle: &LtxBundle,
    promotion: &Ltx25QuantRuntimeIdentity,
) -> gen_core::Result<LoadSpec> {
    if promotion.public_repository != LTX25_PUBLIC_REPOSITORY {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: promotion repository is not the canonical public repository"
        )));
    }
    let selected_root = snapshot_root(spec)?;
    let snapshot =
        canonical_public_snapshot_for_selected(&selected_root, &promotion.bundle_subdir)?;
    let bundle_root = snapshot.join(&promotion.bundle_subdir);
    let expected_text_encoder = snapshot_bound_file(
        &snapshot,
        Path::new(&promotion.bf16_text_encoder_subpath),
        "promoted BF16 text encoder",
    )?;
    let mut bound = spec.clone();
    bound.weights = gen_core::WeightsSource::Dir(snapshot);
    bound.components.clear();
    for component in bundle.components() {
        let path = component.path().to_path_buf();
        if !(path.starts_with(&bundle_root)
            || component.component() == LtxComponent::TextEncoder && path == expected_text_encoder)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: selected component {} escapes reviewed public bundle {}",
                path.display(),
                bundle_root.display()
            )));
        }
        let source = if path.is_dir() {
            gen_core::WeightsSource::Dir(path)
        } else {
            gen_core::WeightsSource::File(path)
        };
        bound
            .components
            .insert(component.component().id().to_owned(), source);
    }
    Ok(bound)
}

pub(crate) fn inventory_for_snapshot(root: &Path) -> gen_core::Result<SnapshotInventory> {
    fn visit(dir: &Path, files: &mut Vec<PathBuf>) -> gen_core::Result<()> {
        for entry in fs::read_dir(dir).map_err(|error| gen_core::Error::Msg(error.to_string()))? {
            let entry = entry.map_err(|error| gen_core::Error::Msg(error.to_string()))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
            if metadata.file_type().is_symlink() {
                if fs::metadata(&path)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .is_file()
                {
                    files.push(path);
                } else {
                    return Err(gen_core::Error::Msg(format!(
                        "runtime inventory refuses non-file symlink {}",
                        path.display()
                    )));
                }
            } else if metadata.is_dir() {
                visit(&path, files)?;
            } else if metadata.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(gen_core::Error::Msg(
            "model snapshot inventory is empty".to_owned(),
        ));
    }
    let snapshots_dir = root
        .parent()
        .filter(|path| path.file_name().is_some_and(|name| name == "snapshots"));
    let repo_root = snapshots_dir.and_then(Path::parent);
    let contains_symlink = files.iter().any(|path| {
        fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink())
    });
    let canonical_blobs = if contains_symlink {
        repo_root
            .map(|repo| fs::canonicalize(repo.join("blobs")))
            .transpose()
            .map_err(|error| gen_core::Error::Msg(error.to_string()))?
    } else {
        None
    };
    let entries = files
        .into_iter()
        .map(|path| {
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
            let symlink_target = if metadata.file_type().is_symlink() {
                let blobs = canonical_blobs.as_ref().ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "{MODEL_25_ID}: symlinked snapshot file {} is not under a Hugging Face <repo>/snapshots/<revision> root",
                        path.display()
                    ))
                })?;
                let physical = fs::canonicalize(&path)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
                if !physical.starts_with(blobs) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{MODEL_25_ID}: snapshot symlink {} resolves outside the repository blob cache {}",
                        path.display(),
                        blobs.display()
                    )));
                }
                let blob_relative = physical
                    .strip_prefix(blobs)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .to_string_lossy()
                    .replace('\\', "/");
                Some(format!("blobs/{blob_relative}"))
            } else {
                None
            };
            Ok(SnapshotInventoryEntry {
                path: path
                    .strip_prefix(root)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .to_string_lossy()
                    .replace('\\', "/"),
                bytes: fs::metadata(&path)
                    .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                    .len(),
                sha256: file_sha256(&path)?,
                symlink_target,
            })
        })
        .collect::<gen_core::Result<Vec<_>>>()?;
    Ok(SnapshotInventory {
        schema_version: "sceneworks-model-inventory-v1",
        entries,
    })
}

pub(crate) fn snapshot_inventory_sha256(inventory: &SnapshotInventory) -> gen_core::Result<String> {
    serde_json::to_vec_pretty(inventory)
        .map(|mut bytes| {
            bytes.push(b'\n');
            sha256_hex(&bytes)
        })
        .map_err(|error| gen_core::Error::Msg(error.to_string()))
}

pub(crate) fn bundle_identity_sha256(
    bundle: &LtxBundle,
    root: &Path,
    inventory: &SnapshotInventory,
    inventory_sha256: &str,
    variant: TransformerVariant,
    mode: Ltx25QuantMode,
) -> gen_core::Result<String> {
    let mut rows = Vec::new();
    for component in bundle.components() {
        let path = component.path();
        let logical = path.strip_prefix(root).map_err(|_| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: resolved component {} escapes the identity-bound snapshot {}",
                path.display(),
                root.display()
            ))
        })?;
        let logical = logical.to_string_lossy().replace('\\', "/");
        let entry = inventory
            .entries
            .iter()
            .find(|entry| entry.path == logical)
            .ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "{MODEL_25_ID}: resolved component {} is absent from the identity-bound snapshot inventory",
                    component.path().display()
                ))
            })?;
        rows.push(format!(
            "{}:{logical}:{}",
            component.component().id(),
            entry.sha256
        ));
    }
    rows.sort();
    rows.insert(0, format!("mode:{}", mode.id()));
    rows.insert(0, format!("variant:{}", variant.id()));
    rows.insert(0, format!("inventory:{inventory_sha256}"));
    Ok(sha256_hex(rows.join("\n").as_bytes()))
}

/// Content identity shared by a measured source bundle and its final public copy. Unlike
/// [`bundle_identity_sha256`], this deliberately excludes the containing snapshot inventory and
/// revision, but includes every selected component id, bundle-relative path, and exact file hash.
/// Promotion requires this digest to be identical across the two independently inventoried roots.
pub(crate) fn selected_bundle_identity_sha256(
    bundle: &LtxBundle,
    snapshot_root: &Path,
    bundle_subdir: &str,
    inventory: &SnapshotInventory,
    variant: TransformerVariant,
    mode: Ltx25QuantMode,
) -> gen_core::Result<String> {
    if !is_normalized_bundle_subdir(bundle_subdir) {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: selected bundle subdir {bundle_subdir:?} is not normalized"
        )));
    }
    let bundle_root = if bundle_subdir == "." {
        snapshot_root.to_path_buf()
    } else {
        snapshot_root.join(bundle_subdir)
    };
    let mut rows = Vec::new();
    for component in bundle.components() {
        let snapshot_relative = component.path().strip_prefix(snapshot_root).map_err(|_| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: selected component {} escapes snapshot {}",
                component.path().display(),
                snapshot_root.display()
            ))
        })?;
        let snapshot_relative = snapshot_relative.to_string_lossy().replace('\\', "/");
        let bundle_relative = match component.path().strip_prefix(&bundle_root) {
            Ok(path) => path.to_string_lossy().replace('\\', "/"),
            Err(_) if component.component() == LtxComponent::TextEncoder => {
                format!("@snapshot/{snapshot_relative}")
            }
            Err(_) => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{MODEL_25_ID}: selected component {} escapes bundle subdir {bundle_subdir:?}",
                    component.path().display()
                )))
            }
        };
        let entry = inventory
            .entries
            .iter()
            .find(|entry| entry.path == snapshot_relative)
            .ok_or_else(|| {
                gen_core::Error::Unsupported(format!(
                    "{MODEL_25_ID}: selected component {} is absent from the full snapshot inventory",
                    component.path().display()
                ))
            })?;
        rows.push(format!(
            "{}:{bundle_relative}:{}:{}",
            component.component().id(),
            entry.bytes,
            entry.sha256
        ));
    }
    rows.sort();
    rows.insert(0, format!("mode:{}", mode.id()));
    rows.insert(0, format!("variant:{}", variant.id()));
    Ok(sha256_hex(rows.join("\n").as_bytes()))
}

/// Reconstruct the active production identity from the exact public bundle on disk and a reviewed
/// source-to-public promotion binding. The final public revision, full inventory, selected bundle,
/// and transformer operator contract are re-derived live; no model-provided sidecar can authorize
/// itself.
pub fn runtime_identity_from_bundle(
    spec: &LoadSpec,
    bundle: &LtxBundle,
    mode: Ltx25QuantMode,
    variant: TransformerVariant,
    promotion: &Ltx25QuantRuntimeIdentity,
) -> gen_core::Result<Ltx25QuantRuntimeIdentity> {
    let root =
        canonical_public_snapshot_for_selected(&snapshot_root(spec)?, &promotion.bundle_subdir)?;
    let inventory = inventory_for_snapshot(&root)?;
    let model_revision = snapshot_revision(&root)?;
    let inventory_sha256 = snapshot_inventory_sha256(&inventory)?;
    let bundle_hash =
        bundle_identity_sha256(bundle, &root, &inventory, &inventory_sha256, variant, mode)?;
    let selected_hash = selected_bundle_identity_sha256(
        bundle,
        &root,
        &promotion.bundle_subdir,
        &inventory,
        variant,
        mode,
    )?;
    let transformer = bundle.require(LtxComponent::Transformer)?.path();
    let inspection = crate::advanced_quant::inspect_transformer_source(transformer, mode)
        .map_err(|error| gen_core::Error::Unsupported(error.to_string()))?;
    if promotion.mode != mode
        || promotion.transformer_variant != variant
        || promotion.public_repository != LTX25_PUBLIC_REPOSITORY
        || !is_normalized_bundle_subdir(&promotion.bf16_text_encoder_subpath)
        || promotion.bf16_text_encoder_subpath == "."
        || !is_lower_hex(&promotion.public_readback_sha256, 64)
        || !is_lower_hex(&promotion.public_replay_receipt_sha256, 64)
        || !is_lower_hex(&promotion.public_replay_output_sha256, 64)
        || promotion.executable_contract_sha256 != env!("LTX25_EXECUTABLE_CONTRACT_SHA256")
        || promotion.model_revision != model_revision
        || promotion.model_inventory_sha256 != inventory_sha256
        || promotion.runtime_bundle_sha256 != bundle_hash
        || promotion.selected_bundle_sha256 != selected_hash
        || promotion.source_selected_bundle_sha256 != selected_hash
        || promotion.operator_contract_sha256 != inspection.operator_contract_sha256
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: promotion binding disagrees with active code/public-model/bundle/operator identity; receipt replay is refused"
        )));
    }
    let mut identity = promotion.clone();
    identity.model_inventory_sha256 = inventory_sha256;
    identity.runtime_bundle_sha256 = bundle_hash;
    identity.selected_bundle_sha256 = selected_hash;
    Ok(identity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(case_id: &str) -> Ltx25QuantMeasurementReceipt {
        let case = measurement_case(case_id).expect("known case");
        let cap = match case.gpu {
            Ltx25GpuGeneration::AdaSm89 => "sm_89",
            Ltx25GpuGeneration::ConsumerBlackwellSm120 => "sm_120",
            _ => unreachable!(),
        };
        Ltx25QuantMeasurementReceipt::seal(Ltx25QuantMeasurementDraft {
            case_id: case.id.to_owned(),
            mode: case.mode,
            gpu_generation: case.gpu,
            transformer_variant: case.transformer_variant,
            fixture: case.fixture.to_owned(),
            observed_width: case.width,
            observed_height: case.height,
            observed_frames: case.frames,
            observed_fps: case.fps,
            seed: case.seed,
            inference_revision: "a".repeat(40),
            executable_contract_sha256: "3".repeat(64),
            executable_sha256: "4".repeat(64),
            model_revision: "b".repeat(40),
            model_inventory_sha256: "c".repeat(64),
            bundle_subdir: "bundles/distilled/int8".to_owned(),
            bf16_text_encoder_subpath: if matches!(
                case.mode,
                Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4
            ) {
                "shared/gemma4-bf16.safetensors".to_owned()
            } else {
                String::new()
            },
            runtime_bundle_sha256: "5".repeat(64),
            selected_bundle_sha256: "0".repeat(64),
            reference_model_revision: "9".repeat(40),
            reference_model_inventory_sha256: "a".repeat(64),
            reference_bundle_subdir: "bundles/distilled/bf16".to_owned(),
            reference_runtime_bundle_sha256: "b".repeat(64),
            reference_selected_bundle_sha256: "2".repeat(64),
            gpu_name: if case.gpu == Ltx25GpuGeneration::AdaSm89 {
                "NVIDIA GeForce RTX 4090".to_owned()
            } else {
                "NVIDIA RTX PRO 6000 Blackwell".to_owned()
            },
            compute_capability: cap.to_owned(),
            driver_version: "580.12".to_owned(),
            harness_version: "sc-18777-terminal-v6".to_owned(),
            run_nonce_sha256: "d".repeat(64),
            transcript_sha256: "e".repeat(64),
            evidence_manifest_sha256: "f".repeat(64),
            output_sha256: "1".repeat(64),
            reference_output_sha256: "2".repeat(64),
            reference_receipt_sha256: "6".repeat(64),
            operator_kind: match case.mode {
                Ltx25QuantMode::Bf16 => "dense-linear",
                Ltx25QuantMode::Q4 | Ltx25QuantMode::PackedQ8 => "mlx-affine-dequant",
                Ltx25QuantMode::Int8ConvRot => "int8-convrot-rht-cublaslt-igemm",
                Ltx25QuantMode::Nvfp4 => "native-nvfp4-cublaslt-w4a4",
            }
            .to_owned(),
            operator_contract_sha256: "7".repeat(64),
            operator_weight_inventory_sha256: "8".repeat(64),
            executed_projection_count: 2,
            declared_projection_count: if matches!(
                case.mode,
                Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4
            ) {
                1
            } else {
                0
            },
            baseline_vram_bytes: 1,
            peak_vram_bytes: 2,
            wall_clock_ms: 1,
            quality: Ltx25QuantQuality {
                reference_psnr: 40.0,
                reference_ssim: 0.99,
                temporal_boundary_drift: 0.01,
                silent_zero_video_passed: true,
                silent_zero_audio_passed: true,
            },
        })
    }

    fn runtime(receipt: &Ltx25QuantMeasurementReceipt) -> Ltx25QuantRuntimeIdentity {
        let mut runtime = Ltx25QuantRuntimeIdentity {
            mode: receipt.mode,
            transformer_variant: receipt.transformer_variant,
            inference_revision: receipt.inference_revision.clone(),
            executable_contract_sha256: receipt.executable_contract_sha256.clone(),
            executable_sha256: receipt.executable_sha256.clone(),
            source_model_revision: receipt.model_revision.clone(),
            source_model_inventory_sha256: receipt.model_inventory_sha256.clone(),
            source_bundle_subdir: receipt.bundle_subdir.clone(),
            source_bf16_text_encoder_subpath: receipt.bf16_text_encoder_subpath.clone(),
            source_runtime_bundle_sha256: receipt.runtime_bundle_sha256.clone(),
            source_selected_bundle_sha256: receipt.selected_bundle_sha256.clone(),
            model_revision: receipt.model_revision.clone(),
            model_inventory_sha256: receipt.model_inventory_sha256.clone(),
            bundle_subdir: receipt.bundle_subdir.clone(),
            bf16_text_encoder_subpath: receipt.bf16_text_encoder_subpath.clone(),
            runtime_bundle_sha256: receipt.runtime_bundle_sha256.clone(),
            selected_bundle_sha256: receipt.selected_bundle_sha256.clone(),
            public_repository: LTX25_PUBLIC_REPOSITORY.to_owned(),
            public_readback_sha256: "8".repeat(64),
            public_replay_receipt_sha256: "9".repeat(64),
            public_replay_output_sha256: receipt.output_sha256.clone(),
            promotion_copy_sha256: String::new(),
            reference_model_revision: receipt.reference_model_revision.clone(),
            reference_model_inventory_sha256: receipt.reference_model_inventory_sha256.clone(),
            reference_bundle_subdir: receipt.reference_bundle_subdir.clone(),
            reference_runtime_bundle_sha256: receipt.reference_runtime_bundle_sha256.clone(),
            receipt_sha256: receipt.receipt_sha256.clone(),
            transcript_sha256: receipt.transcript_sha256.clone(),
            evidence_manifest_sha256: receipt.evidence_manifest_sha256.clone(),
            output_sha256: receipt.output_sha256.clone(),
            reference_output_sha256: receipt.reference_output_sha256.clone(),
            reference_receipt_sha256: receipt.reference_receipt_sha256.clone(),
            operator_kind: receipt.operator_kind.clone(),
            operator_contract_sha256: receipt.operator_contract_sha256.clone(),
            operator_weight_inventory_sha256: receipt.operator_weight_inventory_sha256.clone(),
        };
        runtime.promotion_copy_sha256 = promotion_copy_sha256(receipt, &runtime);
        runtime
    }

    fn write_minimal_safetensors(path: &Path) {
        let header = r#"{"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0, 0]);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn production_selectors_name_packed_tiers_and_never_convrot_evidence() {
        let base = LoadSpec::new(gen_core::WeightsSource::Dir("/weights".into()));
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base).unwrap(),
            Ltx25QuantMode::Bf16
        );
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base.clone().with_quant(Quant::Q4)).unwrap(),
            Ltx25QuantMode::Q4
        );
        // sc-18791: `Quant::Q8` selects the hosted packed q8 bundle, never the ConvRot operator.
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base.clone().with_quant(Quant::Q8)).unwrap(),
            Ltx25QuantMode::PackedQ8
        );
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base.with_quant(Quant::Nvfp4)).unwrap(),
            Ltx25QuantMode::Nvfp4
        );
        assert_ne!(Ltx25QuantMode::PackedQ8, Ltx25QuantMode::Int8ConvRot);
        // The released packed tiers are admitted unconditionally, on every classified generation:
        // no receipt, no runtime identity, no GPU allowlist.
        for gpu in [
            Ltx25GpuGeneration::ConsumerBlackwellSm120,
            Ltx25GpuGeneration::AdaSm89,
        ] {
            for variant in [TransformerVariant::Distilled, TransformerVariant::Dev] {
                for mode in [
                    Ltx25QuantMode::Bf16,
                    Ltx25QuantMode::Q4,
                    Ltx25QuantMode::PackedQ8,
                ] {
                    assert_eq!(
                        admit(mode, gpu, variant, None, &[]),
                        Ltx25QuantAdmission::Admitted,
                        "{} must be a first-class production tier on {}",
                        mode.id(),
                        gpu.id()
                    );
                }
            }
        }
        // The advanced operators keep their unchanged receipt gate.
        for mode in [Ltx25QuantMode::Int8ConvRot, Ltx25QuantMode::Nvfp4] {
            assert!(matches!(
                admit(
                    mode,
                    Ltx25GpuGeneration::ConsumerBlackwellSm120,
                    TransformerVariant::Distilled,
                    None,
                    &[]
                ),
                Ltx25QuantAdmission::Refused { .. }
            ));
        }
    }

    #[test]
    fn promoted_nested_selection_binds_full_public_snapshot_and_explicit_components() {
        let dir = tempfile::tempdir().unwrap();
        let revision = "b".repeat(40);
        let snapshot = dir
            .path()
            .join("models--SceneWorks--ltx-2.5-mlx")
            .join("snapshots")
            .join(&revision);
        let bundle_root = snapshot.join("bundles/distilled/int8");
        let transformer = bundle_root.join("transformer.safetensors");
        let encoder = snapshot.join("shared/gemma4-bf16.safetensors");
        write_minimal_safetensors(&transformer);
        write_minimal_safetensors(&encoder);

        // sc-18791: public-snapshot staging/binding reads only the runtime identity, never the
        // case table, so this exercises it through a published case.
        let measured = receipt("ltx25-packed-q8-blackwell-v1");
        let mut promotion = runtime(&measured);
        promotion.model_revision = revision;
        promotion.bundle_subdir = "bundles/distilled/int8".to_owned();
        promotion.bf16_text_encoder_subpath = "shared/gemma4-bf16.safetensors".to_owned();
        let selected = LoadSpec::new(gen_core::WeightsSource::Dir(bundle_root.clone()));
        let staged = stage_public_runtime_spec(&selected, &promotion).unwrap();
        assert_eq!(staged.weights, selected.weights);
        let canonical_encoder = fs::canonicalize(&encoder).unwrap();
        assert_eq!(
            staged.components.get(LtxComponent::TextEncoder.id()),
            Some(&gen_core::WeightsSource::File(canonical_encoder.clone()))
        );

        let bundle = candle_gen::gen_core::ltx_checkpoint::LtxBundleBuilder::new()
            .with_component(
                LtxComponent::Transformer,
                fs::canonicalize(&transformer).unwrap(),
            )
            .with_component(LtxComponent::TextEncoder, canonical_encoder.clone())
            .build()
            .unwrap();
        let bound = bind_public_runtime_spec(&staged, &bundle, &promotion).unwrap();
        assert_eq!(
            bound.weights,
            gen_core::WeightsSource::Dir(fs::canonicalize(snapshot).unwrap())
        );
        assert_eq!(
            bound.components.get(LtxComponent::Transformer.id()),
            Some(&gen_core::WeightsSource::File(
                fs::canonicalize(transformer).unwrap()
            ))
        );
        assert_eq!(
            bound.components.get(LtxComponent::TextEncoder.id()),
            Some(&gen_core::WeightsSource::File(canonical_encoder))
        );

        let private_root = dir
            .path()
            .join("models--Private--ltx")
            .join("snapshots")
            .join("c".repeat(40))
            .join("bundles/distilled/int8");
        fs::create_dir_all(&private_root).unwrap();
        let private = LoadSpec::new(gen_core::WeightsSource::Dir(private_root));
        assert!(stage_public_runtime_spec(&private, &promotion).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_bound_file_preserves_hf_logical_path_and_rejects_escape() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--SceneWorks--ltx-2.5-mlx");
        let snapshot = repo.join("snapshots").join("b".repeat(40));
        let blobs = repo.join("blobs");
        fs::create_dir_all(snapshot.join("shared")).unwrap();
        fs::create_dir_all(&blobs).unwrap();
        let blob = blobs.join("object");
        write_minimal_safetensors(&blob);
        let logical = snapshot.join("shared/gemma.safetensors");
        symlink(&blob, &logical).unwrap();

        assert_eq!(
            snapshot_bound_file(
                &fs::canonicalize(&snapshot).unwrap(),
                Path::new("shared/gemma.safetensors"),
                "test encoder"
            )
            .unwrap(),
            fs::canonicalize(&snapshot)
                .unwrap()
                .join("shared/gemma.safetensors")
        );

        let outside = dir.path().join("outside.safetensors");
        write_minimal_safetensors(&outside);
        let escaped = snapshot.join("shared/escaped.safetensors");
        symlink(outside, &escaped).unwrap();
        assert!(snapshot_bound_file(
            &fs::canonicalize(snapshot).unwrap(),
            Path::new("shared/escaped.safetensors"),
            "test encoder"
        )
        .is_err());
    }

    #[test]
    fn compute_capability_never_labels_datacenter_blackwell_as_consumer_blackwell() {
        assert_eq!(
            Ltx25GpuGeneration::from_compute_cap(Some((12, 0))),
            Ltx25GpuGeneration::ConsumerBlackwellSm120
        );
        for cap in [(10, 0), (10, 3), (9, 0), (8, 0)] {
            assert_eq!(
                Ltx25GpuGeneration::from_compute_cap(Some(cap)),
                Ltx25GpuGeneration::OtherCuda,
                "{cap:?} must not be mislabeled consumer Blackwell"
            );
        }
    }

    #[test]
    fn nvfp4_refuses_everything_except_exact_consumer_sm120() {
        for gpu in [
            Ltx25GpuGeneration::NotCuda,
            Ltx25GpuGeneration::AdaSm89,
            Ltx25GpuGeneration::OtherCuda,
        ] {
            let result = admit(
                Ltx25QuantMode::Nvfp4,
                gpu,
                TransformerVariant::Distilled,
                None,
                &[],
            );
            assert!(
                matches!(result, Ltx25QuantAdmission::Refused { ref reason } if reason.contains("exact consumer Blackwell sm_120")),
                "{gpu:?}: {result:?}"
            );
        }
    }

    #[test]
    fn advanced_production_modes_remain_fail_closed_without_accepted_receipts() {
        assert!(ACCEPTED_MEASUREMENT_RECEIPTS.is_empty());
        assert!(!catalog_advertised(Ltx25QuantMode::Int8ConvRot));
        assert!(!catalog_advertised(Ltx25QuantMode::Nvfp4));
        // sc-18791: the public release ships no advanced bundle, so no terminal case covers these
        // modes on any generation or variant. The refusal now names the missing case instead of
        // the empty allowlist; the outcome — never admitted — is unchanged.
        for mode in [Ltx25QuantMode::Int8ConvRot, Ltx25QuantMode::Nvfp4] {
            for variant in [TransformerVariant::Distilled, TransformerVariant::Dev] {
                let result = admit(
                    mode,
                    Ltx25GpuGeneration::ConsumerBlackwellSm120,
                    variant,
                    None,
                    ACCEPTED_MEASUREMENT_RECEIPTS.as_slice(),
                );
                assert!(
                    matches!(result, Ltx25QuantAdmission::Refused { ref reason } if reason.contains("no supported terminal measurement case")),
                    "{mode:?}/{variant:?}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn promotion_allowlist_is_external_to_the_stable_code_contract() {
        assert_eq!(
            include_str!("accepted_quant_receipts.allowlist").trim(),
            "[]"
        );
        let source = include_str!("quant_eval.rs");
        assert!(source.contains("include_str!(\"accepted_quant_receipts.allowlist\")"));
        assert!(source.contains("LazyLock<Vec<Ltx25QuantAcceptedMeasurement>>"));
        let build = include_str!("../build.rs");
        assert!(build.contains("extension == \"rs\""));
        assert!(!build.contains("accepted_quant_receipts.allowlist"));
        assert!(build.contains("strip_prefix(\"ref: \")"));
        assert!(build.contains("cargo:rerun-if-changed={}\", branch_ref.display()"));

        let runtime = source
            .split("pub fn runtime_identity_from_bundle(")
            .nth(1)
            .unwrap();
        assert!(runtime.contains("promotion.executable_contract_sha256"));
        assert!(runtime.contains("promotion.source_selected_bundle_sha256 != selected_hash"));
        assert!(!runtime.contains("fs::read(&binding_path)"));
    }

    #[test]
    fn receipt_cannot_omit_peak_wall_quality_or_identity() {
        let mut row = receipt("ltx25-packed-q8-blackwell-v1");
        row.peak_vram_bytes = 0;
        row.wall_clock_ms = 0;
        row.quality.temporal_boundary_drift = f64::NAN;
        row.driver_version.clear();
        row.model_inventory_sha256 = "wrong".to_owned();
        row.quality.silent_zero_audio_passed = false;
        let errors = row.validation_errors().join("; ");
        for expected in [
            "peak VRAM",
            "wall-clock",
            "temporal-boundary",
            "driver",
            "model inventory",
            "silent/zero audio",
            "receipt seal",
        ] {
            assert!(errors.contains(expected), "{errors}");
        }
    }

    #[test]
    fn sealed_receipt_cannot_be_replayed_across_code_model_gpu_or_case_identity() {
        let original = receipt("ltx25-packed-q8-blackwell-v1");
        assert!(original.validation_errors().is_empty());
        let mut mutations = Vec::new();
        let mut code = original.clone();
        code.inference_revision = "9".repeat(40);
        mutations.push(code);
        let mut executable = original.clone();
        executable.executable_contract_sha256 = "0".repeat(64);
        mutations.push(executable);
        let mut model = original.clone();
        model.model_inventory_sha256 = "8".repeat(64);
        mutations.push(model);
        let mut gpu = original.clone();
        gpu.gpu_name = "different GPU".to_owned();
        mutations.push(gpu);
        let mut case = original.clone();
        case.case_id = "ltx25-packed-q8-blackwell-dev-v1".to_owned();
        mutations.push(case);
        let mut variant = original.clone();
        variant.transformer_variant = TransformerVariant::Dev;
        mutations.push(variant);
        for replay in mutations {
            let errors = replay.validation_errors().join("; ");
            assert!(errors.contains("receipt seal"), "{errors}");
        }
    }

    #[test]
    fn production_admission_compares_every_replay_sensitive_runtime_field() {
        // sc-18791: `admit`'s receipt arm is only reachable for a mode that owns a terminal case,
        // and the public release ships no advanced bundle, so the gate is exercised through the
        // predicate `admit` delegates to. Every field below is still proven replay-sensitive.
        let accepted = receipt("ltx25-packed-q8-blackwell-v1");
        let identity = runtime(&accepted);
        assert!(receipt_matches_runtime(&accepted, &identity));
        let mutations: Vec<fn(&mut Ltx25QuantRuntimeIdentity)> = vec![
            |value| value.inference_revision = "0".repeat(40),
            |value| value.executable_contract_sha256 = "0".repeat(64),
            |value| value.executable_sha256 = "0".repeat(64),
            |value| value.source_model_revision = "0".repeat(40),
            |value| value.source_model_inventory_sha256 = "0".repeat(64),
            |value| value.source_bundle_subdir = "bundles/other".to_owned(),
            |value| value.source_runtime_bundle_sha256 = "0".repeat(64),
            |value| value.source_selected_bundle_sha256 = "f".repeat(64),
            |value| value.model_revision = "0".repeat(40),
            |value| value.model_inventory_sha256 = "0".repeat(64),
            |value| value.bundle_subdir = "bundles/other".to_owned(),
            |value| value.runtime_bundle_sha256 = "0".repeat(64),
            |value| value.selected_bundle_sha256 = "f".repeat(64),
            |value| value.public_repository = "private/repository".to_owned(),
            |value| value.public_readback_sha256 = "0".repeat(64),
            |value| value.public_replay_receipt_sha256 = "0".repeat(64),
            |value| value.public_replay_output_sha256 = "0".repeat(64),
            |value| value.promotion_copy_sha256 = "0".repeat(64),
            |value| value.reference_model_revision = "0".repeat(40),
            |value| value.reference_model_inventory_sha256 = "0".repeat(64),
            |value| value.reference_bundle_subdir = "bundles/other".to_owned(),
            |value| value.reference_runtime_bundle_sha256 = "0".repeat(64),
            |value| value.receipt_sha256 = "0".repeat(64),
            |value| value.transcript_sha256 = "0".repeat(64),
            |value| value.evidence_manifest_sha256 = "0".repeat(64),
            |value| value.output_sha256 = "0".repeat(64),
            |value| value.reference_output_sha256 = "0".repeat(64),
            |value| value.reference_receipt_sha256 = "0".repeat(64),
            |value| value.operator_kind = "dense-linear".to_owned(),
            |value| value.operator_contract_sha256 = "0".repeat(64),
            |value| value.operator_weight_inventory_sha256 = "0".repeat(64),
        ];
        for (index, mutate) in mutations.into_iter().enumerate() {
            let mut replay = identity.clone();
            mutate(&mut replay);
            assert_ne!(replay, identity, "mutation {index} changed nothing");
            assert!(
                !receipt_matches_runtime(&accepted, &replay),
                "mutation {index} was not replay-sensitive"
            );
        }
    }

    #[test]
    fn promotion_copy_proof_cannot_be_replayed_across_public_identity_or_selected_bytes() {
        let measured = receipt("ltx25-packed-q8-blackwell-v1");
        let original = runtime(&measured);
        assert!(receipt_matches_runtime(&measured, &original));

        let mut public_revision = original.clone();
        public_revision.model_revision = "f".repeat(40);
        assert!(!receipt_matches_runtime(&measured, &public_revision));

        let mut different_selected_bytes = original;
        different_selected_bytes.selected_bundle_sha256 = "f".repeat(64);
        different_selected_bytes.promotion_copy_sha256 =
            promotion_copy_sha256(&measured, &different_selected_bytes);
        assert!(!receipt_matches_runtime(
            &measured,
            &different_selected_bytes
        ));
    }

    #[test]
    fn current_pool_matrix_is_sm120_only_and_covers_exactly_the_published_bundles() {
        // sc-18791: the public SceneWorks/ltx-2.5-mlx release ships `distilled/{bf16,q4,q8}` and
        // `dev/{bf16,q4,q8}` and nothing else, so the matrix is exactly those six rows.
        assert_eq!(TERMINAL_MEASUREMENT_CASES.len(), 6);
        assert!(TERMINAL_MEASUREMENT_CASES
            .iter()
            .all(|case| case.gpu == Ltx25GpuGeneration::ConsumerBlackwellSm120));
        for variant in [TransformerVariant::Distilled, TransformerVariant::Dev] {
            for mode in [
                Ltx25QuantMode::Bf16,
                Ltx25QuantMode::Q4,
                Ltx25QuantMode::PackedQ8,
            ] {
                assert!(TERMINAL_MEASUREMENT_CASES
                    .iter()
                    .any(|case| case.mode == mode && case.transformer_variant == variant));
            }
        }
        // No row may name an unpublished advanced bundle on any variant or generation.
        assert!(!TERMINAL_MEASUREMENT_CASES.iter().any(|case| matches!(
            case.mode,
            Ltx25QuantMode::Int8ConvRot | Ltx25QuantMode::Nvfp4
        )));
        assert!(matches!(
            admit(
                Ltx25QuantMode::Int8ConvRot,
                Ltx25GpuGeneration::AdaSm89,
                TransformerVariant::Distilled,
                None,
                &[],
            ),
            Ltx25QuantAdmission::Refused { reason } if reason.contains("no supported terminal measurement case")
        ));
    }
}
