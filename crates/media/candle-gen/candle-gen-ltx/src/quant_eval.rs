//! LTX-2.5 quant selection and terminal measurement evidence (sc-18777).
//!
//! Production and measurement are intentionally separate surfaces. Production `Quant::Q8` means
//! the released INT8-ConvRot transformer/text-encoder pair. The terminal controller can additionally
//! measure the hosted packed-q8 tier, but that mode has no production selector and cannot enter the
//! ordinary catalog. Advanced production modes remain fail-closed until a same-run, identity-bound
//! receipt is deliberately copied into [`ACCEPTED_MEASUREMENT_RECEIPTS`].

use std::path::Path;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{self, ltx_checkpoint::LtxBundle, LoadSpec, LtxComponent, Quant};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::MODEL_25_ID;

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

    /// Bind the declared mode to the physical transformer/text-encoder artifact names.
    pub fn validate_bundle_source(self, bundle: &LtxBundle) -> gen_core::Result<()> {
        let transformer = bundle.require(LtxComponent::Transformer)?.path();
        let text_encoder = bundle.require(LtxComponent::TextEncoder)?.path();
        let has = |path: &Path, marker: &str| {
            path.to_string_lossy().to_ascii_lowercase().contains(marker)
        };
        match self {
            Self::Bf16 | Self::Q4 => Ok(()),
            Self::PackedQ8 if has(transformer, "q8") && has(text_encoder, "q8") => Ok(()),
            Self::PackedQ8 => Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: terminal packed-q8 measurement requires q8 transformer and text \
                 encoder artifacts; got '{}' / '{}'",
                transformer.display(),
                text_encoder.display(),
            ))),
            Self::Int8ConvRot
                if has(transformer, "int8-convrot") && has(text_encoder, "int8-convrot") =>
            {
                Ok(())
            }
            Self::Int8ConvRot => Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: requested int8-convrot but transformer '{}' and text encoder '{}' \
                 are not the released matching ConvRot artifacts; refusing a bf16/q4/q8 fallback",
                transformer.display(),
                text_encoder.display(),
            ))),
            Self::Nvfp4 if has(transformer, "nvfp4") && has(text_encoder, "bf16") => Ok(()),
            Self::Nvfp4 => Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: requested nvfp4 but transformer '{}' / text encoder '{}' do not \
                 match the released nvfp4-transformer + bf16-Gemma pairing; refusing a fallback",
                transformer.display(),
                text_encoder.display(),
            ))),
        }
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
        fixture: "ltx25-production-latent-v1",
        width: 512,
        height: 512,
        frames: 17,
        fps: 24,
        seed: 18777,
    }
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
    pub fixture: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub seed: u64,
    pub inference_revision: String,
    pub model_revision: String,
    pub model_inventory_sha256: String,
    pub gpu_name: String,
    pub compute_capability: String,
    pub driver_version: String,
    pub harness_version: String,
    pub run_nonce_sha256: String,
    pub transcript_sha256: String,
    pub evidence_manifest_sha256: String,
    pub output_sha256: String,
    pub reference_output_sha256: String,
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
    pub fixture: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub seed: u64,
    pub inference_revision: String,
    pub model_revision: String,
    pub model_inventory_sha256: String,
    pub gpu_name: String,
    pub compute_capability: String,
    pub driver_version: String,
    pub harness_version: String,
    pub run_nonce_sha256: String,
    pub transcript_sha256: String,
    pub evidence_manifest_sha256: String,
    pub output_sha256: String,
    pub reference_output_sha256: String,
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
        push_string(&mut fields, "fixture", &self.fixture);
        fields.push(format!(
            "geometry:{}x{}x{}@{}",
            self.width, self.height, self.frames, self.fps
        ));
        fields.push(format!("seed:{}", self.seed));
        push_string(&mut fields, "inference_revision", &self.inference_revision);
        push_string(&mut fields, "model_revision", &self.model_revision);
        push_string(
            &mut fields,
            "model_inventory_sha256",
            &self.model_inventory_sha256,
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
            schema_version: "sceneworks-ltx25-quant-receipt-v2".to_owned(),
            case_id: draft.case_id,
            mode: draft.mode,
            gpu_generation: draft.gpu_generation,
            fixture: draft.fixture,
            width: draft.width,
            height: draft.height,
            frames: draft.frames,
            fps: draft.fps,
            seed: draft.seed,
            inference_revision: draft.inference_revision,
            model_revision: draft.model_revision,
            model_inventory_sha256: draft.model_inventory_sha256,
            gpu_name: draft.gpu_name,
            compute_capability: draft.compute_capability,
            driver_version: draft.driver_version,
            harness_version: draft.harness_version,
            run_nonce_sha256: draft.run_nonce_sha256,
            transcript_sha256: draft.transcript_sha256,
            evidence_manifest_sha256: draft.evidence_manifest_sha256,
            output_sha256: draft.output_sha256,
            reference_output_sha256: draft.reference_output_sha256,
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
            if self.fixture != case.fixture
                || self.width != case.width
                || self.height != case.height
                || self.frames != case.frames
                || self.fps != case.fps
                || self.seed != case.seed
            {
                errors.push(format!(
                    "case {} fixture/geometry/seed identity changed",
                    case.id
                ));
            }
        }
        if self.schema_version != "sceneworks-ltx25-quant-receipt-v2" {
            errors.push("unknown receipt schema version".to_owned());
        }
        for (label, value, expected) in [
            ("inference revision", self.inference_revision.as_str(), 40),
            ("model revision", self.model_revision.as_str(), 40),
            (
                "model inventory SHA-256",
                self.model_inventory_sha256.as_str(),
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
            ("receipt SHA-256", self.receipt_sha256.as_str(), 64),
        ] {
            if !is_lower_hex(value, expected) {
                errors.push(format!(
                    "{label} must be {expected} lowercase hexadecimal characters"
                ));
            }
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

/// Intentionally empty: this story adds the producer, not an unmeasured acceptance decision.
pub const ACCEPTED_MEASUREMENT_RECEIPTS: &[Ltx25QuantMeasurementReceipt] = &[];

pub fn admit(
    mode: Ltx25QuantMode,
    gpu: Ltx25GpuGeneration,
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
        .find(|case| case.mode == mode && case.gpu == gpu)
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
        Some(receipt) if receipt.validation_errors().is_empty() => Ltx25QuantAdmission::Admitted,
        Some(receipt) => Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: {} measurement receipt is invalid: {}", case.id, receipt.validation_errors().join("; ")) },
        None => Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: {} is selectable but not catalog-adopted until the terminal campaign records the {} receipt (exact code/model/GPU, VRAM, wall-clock, output, transcript, and quality)", mode.id(), case.id) },
    }
}

pub const fn catalog_advertised(mode: Ltx25QuantMode) -> bool {
    matches!(mode, Ltx25QuantMode::Q4)
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
            fixture: case.fixture.to_owned(),
            width: case.width,
            height: case.height,
            frames: case.frames,
            fps: case.fps,
            seed: case.seed,
            inference_revision: "a".repeat(40),
            model_revision: "b".repeat(40),
            model_inventory_sha256: "c".repeat(64),
            gpu_name: if case.gpu == Ltx25GpuGeneration::AdaSm89 {
                "NVIDIA GeForce RTX 4090".to_owned()
            } else {
                "NVIDIA RTX PRO 6000 Blackwell".to_owned()
            },
            compute_capability: cap.to_owned(),
            driver_version: "580.12".to_owned(),
            harness_version: "sc-18777-terminal-v2".to_owned(),
            run_nonce_sha256: "d".repeat(64),
            transcript_sha256: "e".repeat(64),
            evidence_manifest_sha256: "f".repeat(64),
            output_sha256: "1".repeat(64),
            reference_output_sha256: "2".repeat(64),
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
            matches!(admit(Ltx25QuantMode::PackedQ8, Ltx25GpuGeneration::AdaSm89, &[]), Ltx25QuantAdmission::Refused { ref reason } if reason.contains("terminal comparison source"))
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
            let result = admit(Ltx25QuantMode::Nvfp4, gpu, &[]);
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
            matches!(admit(Ltx25QuantMode::Int8ConvRot, Ltx25GpuGeneration::AdaSm89, ACCEPTED_MEASUREMENT_RECEIPTS), Ltx25QuantAdmission::Refused { ref reason } if reason.contains("not catalog-adopted"))
        );
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
        let mut model = original.clone();
        model.model_inventory_sha256 = "8".repeat(64);
        mutations.push(model);
        let mut gpu = original.clone();
        gpu.gpu_name = "different GPU".to_owned();
        mutations.push(gpu);
        let mut case = original.clone();
        case.case_id = "ltx25-int8-convrot-blackwell-v1".to_owned();
        mutations.push(case);
        for replay in mutations {
            let errors = replay.validation_errors().join("; ");
            assert!(errors.contains("receipt seal"), "{errors}");
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
