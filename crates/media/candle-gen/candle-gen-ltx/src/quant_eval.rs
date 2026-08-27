//! LTX-2.5 pre-packed quant selection and terminal measurement contract (sc-18777).
//!
//! `Quant::Q8` names LTX's released pre-packed INT8-ConvRot pair; it is not an instruction to
//! quantize a bf16 bundle. `Quant::Nvfp4` likewise names the released NVFP4 transformer. Neither
//! advanced mode is catalog-advertised until sc-18783 records a receipt for exact code, model,
//! driver, VRAM, wall-clock, and quality evidence.

use std::path::Path;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{self, ltx_checkpoint::LtxBundle, LoadSpec, LtxComponent, Quant};

use crate::MODEL_25_ID;

/// All LTX-2.5 numeric source modes. Q8 and NVFP4 are never collapsed into Q4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltx25QuantMode {
    Bf16,
    Q4,
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
            Self::Q4 => "q4",
            Self::Int8ConvRot => "int8-convrot",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// Advanced selectors must bind to their own upstream artifacts; accepting a bf16 component
    /// here would silently misreport the active numeric mode.
    pub fn validate_bundle_source(self, bundle: &LtxBundle) -> gen_core::Result<()> {
        let transformer = bundle.require(LtxComponent::Transformer)?.path();
        let text_encoder = bundle.require(LtxComponent::TextEncoder)?.path();
        let has = |path: &Path, marker: &str| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.to_ascii_lowercase().contains(marker))
        };
        match self {
            Self::Bf16 | Self::Q4 => Ok(()),
            Self::Int8ConvRot
                if has(transformer, "int8-convrot") && has(text_encoder, "int8-convrot") =>
            {
                Ok(())
            }
            Self::Int8ConvRot => Err(gen_core::Error::Unsupported(format!(
                "{MODEL_25_ID}: requested int8-convrot but transformer '{}' and text encoder '{}' \
                 are not the released matching ConvRot artifacts; refusing a bf16/q4 fallback",
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

/// CUDA generations with defined LTX terminal-harness cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ltx25GpuGeneration {
    NotCuda,
    Ada,
    Blackwell,
}

impl Ltx25GpuGeneration {
    pub const fn id(self) -> &'static str {
        match self {
            Self::NotCuda => "not-cuda",
            Self::Ada => "ada",
            Self::Blackwell => "blackwell",
        }
    }

    /// The shared NVFP4 context is the runtime capability gate: it identifies Blackwell only when
    /// the bound device can create the consumer-sm_120 FP4 context.
    pub fn from_device(device: &Device) -> gen_core::Result<Self> {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(Self::NotCuda);
        }
        let context =
            candle_gen::quant::Nvfp4Context::new(device).map_err(candle_gen::CandleError::from)?;
        Ok(if context.is_fp4() {
            Self::Blackwell
        } else {
            Self::Ada
        })
    }
}

/// Immutable terminal-harness input. Quality is comparable only within this exact fixture/shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ltx25QuantMeasurementCase {
    pub id: &'static str,
    pub mode: Ltx25QuantMode,
    pub gpu: Ltx25GpuGeneration,
    pub fixture: &'static str,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub seed: u64,
}

/// sc-18783 executes these cases once. NVFP4 has no Ada row because a dequant fallback must never
/// be measured or reported as native NVFP4.
pub const TERMINAL_MEASUREMENT_CASES: &[Ltx25QuantMeasurementCase] = &[
    Ltx25QuantMeasurementCase {
        id: "ltx25-int8-convrot-ada-v1",
        mode: Ltx25QuantMode::Int8ConvRot,
        gpu: Ltx25GpuGeneration::Ada,
        fixture: "ltx25-production-latent-v1",
        width: 512,
        height: 512,
        frames: 17,
        seed: 18777,
    },
    Ltx25QuantMeasurementCase {
        id: "ltx25-int8-convrot-blackwell-v1",
        mode: Ltx25QuantMode::Int8ConvRot,
        gpu: Ltx25GpuGeneration::Blackwell,
        fixture: "ltx25-production-latent-v1",
        width: 512,
        height: 512,
        frames: 17,
        seed: 18777,
    },
    Ltx25QuantMeasurementCase {
        id: "ltx25-nvfp4-blackwell-v1",
        mode: Ltx25QuantMode::Nvfp4,
        gpu: Ltx25GpuGeneration::Blackwell,
        fixture: "ltx25-production-latent-v1",
        width: 512,
        height: 512,
        frames: 17,
        seed: 18777,
    },
];

/// Quality values emitted beside the output hash.
#[derive(Clone, Debug, PartialEq)]
pub struct Ltx25QuantQuality {
    pub reference_psnr: f64,
    pub reference_ssim: f64,
    pub temporal_boundary_drift: f64,
}

/// One real-weight observation, bound to exact model/code/driver and output identity.
#[derive(Clone, Debug, PartialEq)]
pub struct Ltx25QuantMeasurementReceipt {
    pub case_id: String,
    pub inference_revision: String,
    pub model_revision: String,
    pub model_inventory_sha256: String,
    pub driver_version: String,
    pub harness_version: String,
    pub output_sha256: String,
    pub peak_vram_bytes: u64,
    pub wall_clock_ms: u64,
    pub quality: Ltx25QuantQuality,
}

impl Ltx25QuantMeasurementReceipt {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if TERMINAL_MEASUREMENT_CASES
            .iter()
            .all(|case| case.id != self.case_id)
        {
            errors.push(format!(
                "unknown LTX-2.5 quant measurement case {:?}",
                self.case_id
            ));
        }
        for (label, value, expected) in [
            ("inference revision", self.inference_revision.as_str(), 40),
            ("model revision", self.model_revision.as_str(), 40),
            (
                "model inventory SHA-256",
                self.model_inventory_sha256.as_str(),
                64,
            ),
            ("output SHA-256", self.output_sha256.as_str(), 64),
        ] {
            if value.len() != expected
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                errors.push(format!(
                    "{label} must be {expected} lowercase hexadecimal characters"
                ));
            }
        }
        if self.driver_version.trim().is_empty() {
            errors.push("driver version must be non-empty".to_owned());
        }
        if self.harness_version.trim().is_empty() {
            errors.push("harness version must be non-empty".to_owned());
        }
        if self.peak_vram_bytes == 0 {
            errors.push("peak VRAM bytes must be positive".to_owned());
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
        errors
    }
}

/// An explicit admission result, never an implicit fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ltx25QuantAdmission {
    Admitted,
    Refused { reason: String },
}

/// Intentionally empty until sc-18783 promotes terminal real-weight measurements.
pub const ACCEPTED_MEASUREMENT_RECEIPTS: &[Ltx25QuantMeasurementReceipt] = &[];

pub fn admit(
    mode: Ltx25QuantMode,
    gpu: Ltx25GpuGeneration,
    receipts: &[Ltx25QuantMeasurementReceipt],
) -> Ltx25QuantAdmission {
    if mode == Ltx25QuantMode::Nvfp4 && gpu != Ltx25GpuGeneration::Blackwell {
        return Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: nvfp4 requires consumer Blackwell (sm_120); detected {}. Refusing rather than falling back to bf16/q4", gpu.id()) };
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
        None => Ltx25QuantAdmission::Refused { reason: format!("{MODEL_25_ID}: {} is selectable but not catalog-adopted until sc-18783 records the {} receipt (VRAM, wall-clock, and quality)", mode.id(), case.id) },
    }
}

/// Catalog truth for this commit; new modes need accepted receipts before joining this surface.
pub const fn catalog_advertised(mode: Ltx25QuantMode) -> bool {
    matches!(mode, Ltx25QuantMode::Q4)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn receipt(case_id: &str) -> Ltx25QuantMeasurementReceipt {
        Ltx25QuantMeasurementReceipt {
            case_id: case_id.to_owned(),
            inference_revision: "a".repeat(40),
            model_revision: "b".repeat(40),
            model_inventory_sha256: "c".repeat(64),
            driver_version: "580.12".to_owned(),
            harness_version: "sc-18783-v1".to_owned(),
            output_sha256: "d".repeat(64),
            peak_vram_bytes: 1,
            wall_clock_ms: 1,
            quality: Ltx25QuantQuality {
                reference_psnr: 1.0,
                reference_ssim: 1.0,
                temporal_boundary_drift: 0.0,
            },
        }
    }
    #[test]
    fn selectors_preserve_all_four_modes_without_aliasing() {
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
    }
    #[test]
    fn nvfp4_refuses_non_blackwell_even_with_a_valid_blackwell_receipt() {
        let result = admit(
            Ltx25QuantMode::Nvfp4,
            Ltx25GpuGeneration::Ada,
            &[receipt("ltx25-nvfp4-blackwell-v1")],
        );
        assert!(
            matches!(result, Ltx25QuantAdmission::Refused { ref reason } if reason.contains("requires consumer Blackwell"))
        );
    }
    #[test]
    fn advanced_mode_is_not_advertised_or_admitted_without_terminal_receipt() {
        assert!(!catalog_advertised(Ltx25QuantMode::Int8ConvRot));
        assert!(!catalog_advertised(Ltx25QuantMode::Nvfp4));
        assert!(
            matches!(admit(Ltx25QuantMode::Int8ConvRot, Ltx25GpuGeneration::Ada, &[]), Ltx25QuantAdmission::Refused { ref reason } if reason.contains("not catalog-adopted"))
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
        let errors = row.validation_errors().join("; ");
        for expected in [
            "peak VRAM",
            "wall-clock",
            "temporal-boundary",
            "driver",
            "model inventory",
        ] {
            assert!(errors.contains(expected), "{errors}");
        }
    }
    #[test]
    fn terminal_matrix_covers_int8_on_each_supported_generation_and_nvfp4_only_blackwell() {
        assert!(TERMINAL_MEASUREMENT_CASES
            .iter()
            .any(|case| case.mode == Ltx25QuantMode::Int8ConvRot
                && case.gpu == Ltx25GpuGeneration::Ada));
        assert!(TERMINAL_MEASUREMENT_CASES
            .iter()
            .any(|case| case.mode == Ltx25QuantMode::Int8ConvRot
                && case.gpu == Ltx25GpuGeneration::Blackwell));
        assert!(TERMINAL_MEASUREMENT_CASES
            .iter()
            .any(|case| case.mode == Ltx25QuantMode::Nvfp4
                && case.gpu == Ltx25GpuGeneration::Blackwell));
        assert!(!TERMINAL_MEASUREMENT_CASES
            .iter()
            .any(|case| case.mode == Ltx25QuantMode::Nvfp4 && case.gpu == Ltx25GpuGeneration::Ada));
    }
}
