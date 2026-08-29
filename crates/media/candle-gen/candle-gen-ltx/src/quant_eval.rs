//! LTX-2.5 quant selection and terminal measurement evidence (sc-18777).
//!
//! Production and measurement are intentionally separate surfaces. Production `Quant::Q8` means
//! the released INT8-ConvRot transformer/text-encoder pair. The terminal controller can additionally
//! measure the hosted packed-q8 tier, but that mode has no production selector and cannot enter the
//! ordinary catalog. Advanced production modes remain fail-closed until a same-run, identity-bound
//! receipt is deliberately copied into [`ACCEPTED_MEASUREMENT_RECEIPTS`].

use std::fs;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{self, ltx_checkpoint::LtxBundle, LoadSpec, LtxComponent, Quant};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{dev_sampler::TransformerVariant, MODEL_25_ID};

pub const RUNTIME_BINDING_FILE: &str = "ltx25-quant-runtime-binding.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ltx25QuantRuntimeIdentity {
    pub mode: Ltx25QuantMode,
    pub transformer_variant: TransformerVariant,
    pub inference_revision: String,
    pub executable_contract_sha256: String,
    pub executable_sha256: String,
    pub model_revision: String,
    pub model_inventory_sha256: String,
    pub runtime_bundle_sha256: String,
    pub reference_model_revision: String,
    pub reference_model_inventory_sha256: String,
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
/// `PackedQ8` is intentionally not constructible from [`LoadSpec`]: the production `Quant::Q8`
/// contract already names INT8-ConvRot. Keeping a separate terminal-only variant prevents a q8
/// packed observation from being promoted under the ConvRot label.
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
            Some(Quant::Q8) => Ok(Self::Int8ConvRot),
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
            let context = cuda.cuda_stream().context();
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

/// Both current physical generations compare bf16, packed q4/q8, and ConvRot. Native NVFP4 exists
/// only for the exact consumer `sm_120` row.
pub const TERMINAL_MEASUREMENT_CASES: &[Ltx25QuantMeasurementCase] = &[
    terminal_case(
        "ltx25-bf16-ada-v1",
        Ltx25QuantMode::Bf16,
        Ltx25GpuGeneration::AdaSm89,
    ),
    terminal_case(
        "ltx25-packed-q4-ada-v1",
        Ltx25QuantMode::Q4,
        Ltx25GpuGeneration::AdaSm89,
    ),
    terminal_case(
        "ltx25-packed-q8-ada-v1",
        Ltx25QuantMode::PackedQ8,
        Ltx25GpuGeneration::AdaSm89,
    ),
    terminal_case(
        "ltx25-int8-convrot-ada-v1",
        Ltx25QuantMode::Int8ConvRot,
        Ltx25GpuGeneration::AdaSm89,
    ),
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
    terminal_case(
        "ltx25-int8-convrot-blackwell-v1",
        Ltx25QuantMode::Int8ConvRot,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_case(
        "ltx25-nvfp4-blackwell-v1",
        Ltx25QuantMode::Nvfp4,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_dev_case(
        "ltx25-bf16-ada-dev-v1",
        Ltx25QuantMode::Bf16,
        Ltx25GpuGeneration::AdaSm89,
    ),
    terminal_dev_case(
        "ltx25-packed-q4-ada-dev-v1",
        Ltx25QuantMode::Q4,
        Ltx25GpuGeneration::AdaSm89,
    ),
    terminal_dev_case(
        "ltx25-packed-q8-ada-dev-v1",
        Ltx25QuantMode::PackedQ8,
        Ltx25GpuGeneration::AdaSm89,
    ),
    terminal_dev_case(
        "ltx25-int8-convrot-ada-dev-v1",
        Ltx25QuantMode::Int8ConvRot,
        Ltx25GpuGeneration::AdaSm89,
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
    terminal_dev_case(
        "ltx25-int8-convrot-blackwell-dev-v1",
        Ltx25QuantMode::Int8ConvRot,
        Ltx25GpuGeneration::ConsumerBlackwellSm120,
    ),
    terminal_dev_case(
        "ltx25-nvfp4-blackwell-dev-v1",
        Ltx25QuantMode::Nvfp4,
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
    pub runtime_bundle_sha256: String,
    pub reference_model_revision: String,
    pub reference_model_inventory_sha256: String,
    pub reference_runtime_bundle_sha256: String,
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
    pub runtime_bundle_sha256: String,
    pub reference_model_revision: String,
    pub reference_model_inventory_sha256: String,
    pub reference_runtime_bundle_sha256: String,
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
        push_string(
            &mut fields,
            "runtime_bundle_sha256",
            &self.runtime_bundle_sha256,
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
            "reference_runtime_bundle_sha256",
            &self.reference_runtime_bundle_sha256,
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

    pub(crate) fn seal(draft: Ltx25QuantMeasurementDraft) -> Self {
        let mut receipt = Self {
            schema_version: "sceneworks-ltx25-quant-receipt-v4".to_owned(),
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
            runtime_bundle_sha256: draft.runtime_bundle_sha256,
            reference_model_revision: draft.reference_model_revision,
            reference_model_inventory_sha256: draft.reference_model_inventory_sha256,
            reference_runtime_bundle_sha256: draft.reference_runtime_bundle_sha256,
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
        if self.schema_version != "sceneworks-ltx25-quant-receipt-v4" {
            errors.push("unknown receipt schema version".to_owned());
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
                || self.reference_runtime_bundle_sha256 != self.runtime_bundle_sha256
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
pub const ACCEPTED_MEASUREMENT_RECEIPTS: &[Ltx25QuantMeasurementReceipt] =
    include!("accepted_quant_receipts.allowlist");

pub fn admit(
    mode: Ltx25QuantMode,
    gpu: Ltx25GpuGeneration,
    variant: TransformerVariant,
    runtime: Option<&Ltx25QuantRuntimeIdentity>,
    receipts: &[Ltx25QuantMeasurementReceipt],
) -> Ltx25QuantAdmission {
    if mode == Ltx25QuantMode::PackedQ8 {
        return Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: packed-q8 is a terminal comparison source, not a production selector; production Quant::Q8 means int8-convrot") };
    }
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
    match receipts.iter().find(|receipt| receipt.case_id == case.id) {
        Some(receipt)
            if receipt.validation_errors().is_empty()
                && runtime.is_some_and(|runtime| receipt_matches_runtime(receipt, runtime)) =>
        {
            Ltx25QuantAdmission::Admitted
        }
        Some(receipt) if !receipt.validation_errors().is_empty() => Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: {} measurement receipt is invalid: {}", case.id, receipt.validation_errors().join("; ")) },
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
        && receipt.model_revision == runtime.model_revision
        && receipt.model_inventory_sha256 == runtime.model_inventory_sha256
        && receipt.runtime_bundle_sha256 == runtime.runtime_bundle_sha256
        && receipt.reference_model_revision == runtime.reference_model_revision
        && receipt.reference_model_inventory_sha256 == runtime.reference_model_inventory_sha256
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

pub(crate) fn inventory_for_snapshot(root: &Path) -> gen_core::Result<SnapshotInventory> {
    fn visit(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) -> gen_core::Result<()> {
        for entry in fs::read_dir(dir).map_err(|error| gen_core::Error::Msg(error.to_string()))? {
            let entry = entry.map_err(|error| gen_core::Error::Msg(error.to_string()))?;
            let path = entry.path();
            if path == root.join(RUNTIME_BINDING_FILE) {
                continue;
            }
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
                visit(root, &path, files)?;
            } else if metadata.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, root, &mut files)?;
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

/// Reconstruct the active production identity from the exact bundle on disk. The binding sidecar is
/// only a carrier for the evidence hashes/model revision; live code, inventory, bundle resolution,
/// transformer descriptor contract, and sidecar agreement are all re-verified here.
pub fn runtime_identity_from_bundle(
    spec: &LoadSpec,
    bundle: &LtxBundle,
    mode: Ltx25QuantMode,
    variant: TransformerVariant,
) -> gen_core::Result<Ltx25QuantRuntimeIdentity> {
    let root = snapshot_root(spec)?;
    let binding_path = root.join(RUNTIME_BINDING_FILE);
    let mut identity: Ltx25QuantRuntimeIdentity =
        serde_json::from_slice(&fs::read(&binding_path).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: advanced quant bundle lacks readable {}: {error}",
                binding_path.display()
            ))
        })?)
        .map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: parse runtime binding {}: {error}",
                binding_path.display()
            ))
        })?;
    let inventory = inventory_for_snapshot(&root)?;
    let model_revision = snapshot_revision(&root)?;
    let inventory_sha256 = snapshot_inventory_sha256(&inventory)?;
    let bundle_hash =
        bundle_identity_sha256(bundle, &root, &inventory, &inventory_sha256, variant, mode)?;
    let transformer = bundle.require(LtxComponent::Transformer)?.path();
    let inspection = crate::advanced_quant::inspect_transformer_source(transformer, mode)
        .map_err(|error| gen_core::Error::Unsupported(error.to_string()))?;
    if identity.mode != mode
        || identity.transformer_variant != variant
        || identity.executable_contract_sha256 != env!("LTX25_EXECUTABLE_CONTRACT_SHA256")
        || identity.model_revision != model_revision
        || identity.model_inventory_sha256 != inventory_sha256
        || identity.runtime_bundle_sha256 != bundle_hash
        || identity.operator_contract_sha256 != inspection.operator_contract_sha256
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_25_ID}: runtime binding disagrees with active code/model/bundle/operator identity; receipt replay is refused"
        )));
    }
    identity.model_inventory_sha256 = inventory_sha256;
    identity.runtime_bundle_sha256 = bundle_hash;
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
            runtime_bundle_sha256: "5".repeat(64),
            reference_model_revision: "9".repeat(40),
            reference_model_inventory_sha256: "a".repeat(64),
            reference_runtime_bundle_sha256: "b".repeat(64),
            gpu_name: if case.gpu == Ltx25GpuGeneration::AdaSm89 {
                "NVIDIA GeForce RTX 4090".to_owned()
            } else {
                "NVIDIA RTX PRO 6000 Blackwell".to_owned()
            },
            compute_capability: cap.to_owned(),
            driver_version: "580.12".to_owned(),
            harness_version: "sc-18777-terminal-v4".to_owned(),
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
        Ltx25QuantRuntimeIdentity {
            mode: receipt.mode,
            transformer_variant: receipt.transformer_variant,
            inference_revision: receipt.inference_revision.clone(),
            executable_contract_sha256: receipt.executable_contract_sha256.clone(),
            executable_sha256: receipt.executable_sha256.clone(),
            model_revision: receipt.model_revision.clone(),
            model_inventory_sha256: receipt.model_inventory_sha256.clone(),
            runtime_bundle_sha256: receipt.runtime_bundle_sha256.clone(),
            reference_model_revision: receipt.reference_model_revision.clone(),
            reference_model_inventory_sha256: receipt.reference_model_inventory_sha256.clone(),
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
        }
    }

    #[test]
    fn production_selectors_do_not_alias_packed_q8_to_convrot_evidence() {
        let base = LoadSpec::new(gen_core::WeightsSource::Dir("/weights".into()));
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base).unwrap(),
            Ltx25QuantMode::Bf16
        );
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base.clone().with_quant(Quant::Q4)).unwrap(),
            Ltx25QuantMode::Q4
        );
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base.clone().with_quant(Quant::Q8)).unwrap(),
            Ltx25QuantMode::Int8ConvRot
        );
        assert_eq!(
            Ltx25QuantMode::from_load_spec(&base.with_quant(Quant::Nvfp4)).unwrap(),
            Ltx25QuantMode::Nvfp4
        );
        assert_ne!(Ltx25QuantMode::PackedQ8, Ltx25QuantMode::Int8ConvRot);
        assert!(
            matches!(admit(Ltx25QuantMode::PackedQ8, Ltx25GpuGeneration::AdaSm89, TransformerVariant::Distilled, None, &[]), Ltx25QuantAdmission::Refused { ref reason } if reason.contains("terminal comparison source"))
        );
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
        assert!(
            matches!(admit(Ltx25QuantMode::Int8ConvRot, Ltx25GpuGeneration::AdaSm89, TransformerVariant::Distilled, None, ACCEPTED_MEASUREMENT_RECEIPTS), Ltx25QuantAdmission::Refused { ref reason } if reason.contains("not catalog-adopted"))
        );
    }

    #[test]
    fn promotion_allowlist_is_external_to_the_stable_code_contract() {
        assert_eq!(
            include_str!("accepted_quant_receipts.allowlist").trim(),
            "&[]"
        );
        let source = include_str!("quant_eval.rs");
        assert!(source.contains("include!(\"accepted_quant_receipts.allowlist\")"));
        let build = include_str!("../build.rs");
        assert!(build.contains("extension == \"rs\""));
        assert!(!build.contains("accepted_quant_receipts.allowlist"));
        assert!(build.contains("strip_prefix(\"ref: \")"));
        assert!(build.contains("cargo:rerun-if-changed={}\", branch_ref.display()"));

        let runtime = source
            .split("pub fn runtime_identity_from_bundle(")
            .nth(1)
            .unwrap();
        assert!(runtime.contains("identity.executable_contract_sha256"));
        assert!(!runtime
            .contains("identity.inference_revision != env!(\"LTX25_BUILD_INFERENCE_REVISION\")"));
    }

    #[test]
    fn receipt_cannot_omit_peak_wall_quality_or_identity() {
        let mut row = receipt("ltx25-int8-convrot-ada-v1");
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
        let original = receipt("ltx25-int8-convrot-ada-v1");
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
        case.case_id = "ltx25-int8-convrot-blackwell-v1".to_owned();
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
        let accepted = receipt("ltx25-int8-convrot-ada-v1");
        let identity = runtime(&accepted);
        assert_eq!(
            admit(
                accepted.mode,
                accepted.gpu_generation,
                accepted.transformer_variant,
                Some(&identity),
                std::slice::from_ref(&accepted),
            ),
            Ltx25QuantAdmission::Admitted
        );
        let mutations: Vec<fn(&mut Ltx25QuantRuntimeIdentity)> = vec![
            |value| value.inference_revision = "0".repeat(40),
            |value| value.executable_contract_sha256 = "0".repeat(64),
            |value| value.executable_sha256 = "0".repeat(64),
            |value| value.model_revision = "0".repeat(40),
            |value| value.model_inventory_sha256 = "0".repeat(64),
            |value| value.runtime_bundle_sha256 = "0".repeat(64),
            |value| value.reference_model_revision = "0".repeat(40),
            |value| value.reference_model_inventory_sha256 = "0".repeat(64),
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
        for mutate in mutations {
            let mut replay = identity.clone();
            mutate(&mut replay);
            let result = admit(
                accepted.mode,
                accepted.gpu_generation,
                accepted.transformer_variant,
                Some(&replay),
                std::slice::from_ref(&accepted),
            );
            assert!(
                matches!(result, Ltx25QuantAdmission::Refused { ref reason } if reason.contains("replay")),
                "{result:?}"
            );
        }
    }

    #[test]
    fn full_matrix_distinguishes_all_five_modes_and_nvfp4_is_sm120_only() {
        for gpu in [
            Ltx25GpuGeneration::AdaSm89,
            Ltx25GpuGeneration::ConsumerBlackwellSm120,
        ] {
            for mode in [
                Ltx25QuantMode::Bf16,
                Ltx25QuantMode::Q4,
                Ltx25QuantMode::PackedQ8,
                Ltx25QuantMode::Int8ConvRot,
            ] {
                assert!(TERMINAL_MEASUREMENT_CASES
                    .iter()
                    .any(|case| case.mode == mode && case.gpu == gpu));
            }
        }
        assert!(TERMINAL_MEASUREMENT_CASES
            .iter()
            .any(|case| case.mode == Ltx25QuantMode::Nvfp4
                && case.gpu == Ltx25GpuGeneration::ConsumerBlackwellSm120));
        assert!(!TERMINAL_MEASUREMENT_CASES
            .iter()
            .any(|case| case.mode == Ltx25QuantMode::Nvfp4
                && case.gpu != Ltx25GpuGeneration::ConsumerBlackwellSm120));
    }
}
