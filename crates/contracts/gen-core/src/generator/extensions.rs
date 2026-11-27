//! Typed feature-parameter map ([`FeatureParams`]) and typed family extension blocks
//! ([`FamilyExtensions`], [`FamilyExtension`], [`FamilyExtensionKind`]) carried on
//! [`GenerationRequest`](super::GenerationRequest) (`REG-4`, `E3`).
//!
//! Moving family-specific knobs into typed extension blocks and feature parameters into
//! [`FeatureParams`] guarantees that the core request has zero family-specific top-level fields
//! while preserving exhaustive float-finiteness checking across every extension and feature block.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::GenerationPhase;

/// Optional cross-family PiD super-resolving decoder parameters carried in [`FeatureParams::pid`].
///
/// Presence (`Some(...)`) opts into PiD decode (`use_pid = true`); `None` is the native VAE path.
/// This makes the invalid combination (`use_pid = false` with `capture_sigma = Some(...)`)
/// unrepresentable by construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PidParams {
    /// PiD `from_ldm` early-stop capture σ (flow-matching noise ceiling). `None` / `<= 0` runs the
    /// full denoise before PiD decode.
    pub capture_sigma: Option<f32>,
}

impl PidParams {
    /// Default PiD decode with clean σ=0 full denoise.
    pub const fn enabled() -> Self {
        Self {
            capture_sigma: None,
        }
    }

    /// PiD decode with early-stop capture σ.
    pub const fn with_capture_sigma(sigma: f32) -> Self {
        Self {
            capture_sigma: Some(sigma),
        }
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self { capture_sigma } = self;
        if let Some(sigma) = *capture_sigma {
            if !sigma.is_finite() {
                return Some(("pid_capture_sigma", sigma));
            }
        }
        None
    }
}

/// Typed cross-family feature parameters carried on [`GenerationRequest::features`](super::GenerationRequest::features)
/// (`REG-4`).
///
/// Covers shared, capability-gated features (`pag`, `phases`, `low_res_start`, `pid`) plus an
/// extensible parameter map (`extra`) keyed by registered `FeatureSpec` id (e.g. `hires_fix`,
/// `inert_phase_split`, future features).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureParams {
    /// Perturbed-Attention Guidance (PAG) controls (`Capabilities::pag`).
    pub pag: Option<crate::PagConfig>,
    /// Multi-phase denoise schedule (`Capabilities::supports_phases`).
    pub phases: Option<Vec<GenerationPhase>>,
    /// Two-phase low-resolution start recipe (`Capabilities::supports_low_res_start`).
    pub low_res_start: Option<crate::LowResStartSpec>,
    /// Cross-family PiD super-resolving decoder opt-in (`use_pid` + optional `capture_sigma`).
    pub pid: Option<PidParams>,
    /// Extensible feature parameter map keyed by feature id for generic `FeatureSpec` invocations.
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl FeatureParams {
    pub fn is_empty(&self) -> bool {
        let Self {
            pag,
            phases,
            low_res_start,
            pid,
            extra,
        } = self;
        pag.is_none()
            && phases.is_none()
            && low_res_start.is_none()
            && pid.is_none()
            && extra.is_empty()
    }

    /// Exhaustive non-finite float check across every feature parameter block.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            pag,
            phases,
            low_res_start,
            pid,
            extra: _,
        } = self;
        if let Some(nonfinite) = pag.as_ref().and_then(|v| v.first_nonfinite_float()) {
            return Some(nonfinite);
        }
        if let Some(crate::LowResStartSpec {
            version: _,
            k: _,
            scale: Some(scale),
            estimate_source: _,
        }) = low_res_start
        {
            if !scale.is_finite() {
                return Some(("low_res_start.scale", *scale));
            }
        }
        if let Some(nonfinite) = pid.as_ref().and_then(PidParams::first_nonfinite_float) {
            return Some(nonfinite);
        }
        if let Some(phases) = phases {
            for ph in phases {
                if let Some(g) = ph.guidance {
                    if !g.is_finite() {
                        return Some(("phases.guidance", g));
                    }
                }
                for pa in &ph.adapters {
                    if let Some(w) = pa.weight {
                        if !w.is_finite() {
                            return Some(("phases.adapter.weight", w));
                        }
                    }
                }
            }
        }
        None
    }
}

/// Machine-readable family-extension kind declared on [`crate::RouteStructure`] and
/// [`crate::Capabilities::supported_family_extensions`] so shared validation checks primary route
/// declarations instead of comparing model ID strings (`REG-4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FamilyExtensionKind {
    /// MOSS-Image options and native edit inputs (`MossImageExtension`).
    MossImage,
    /// FLUX 3 reference-output sizing and caption-only spatial layout controls (`Flux3Extension`).
    Flux3,
    /// MiniMax-H3-Image explicit sampler parameters (`H3Extension::image_sampler`).
    H3ImageSampler,
    /// MiniMax-H3 learned sparse attention (`H3Extension::veda`).
    H3Veda,
    /// MiniMax-H3 reference sizing / terminal telemetry controls (`H3Extension`).
    H3Reference,
    /// LTX-2.3 / LTX-2.5 DFR / duration-head / uncensored-enhancer controls (`LtxExtension`).
    Ltx,
    /// SVD-XT micro-conditioning controls (`SvdExtension`).
    Svd,
    /// PuLID-FLUX CFG start-step scheduling (`PulidExtension`).
    Pulid,
    /// SeedVR2 pre-blur softness (`SeedVr2Extension`).
    SeedVr2,
    /// Wan leading temporal chunk trimming (`WanExtension`).
    Wan,
    /// Krea / Qwen-Image multi-tap text style gain (`KreaExtension`).
    Krea,
}

/// MOSS-Image family extension (`REG-4`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MossImageExtension {
    /// Frozen MOSS controls (DMD acceleration, SLG, skip layers, training shift).
    pub options: Option<crate::moss_image::MossImageOptions>,
    /// Native MOSS-Image-Edit inputs executed through the unified [`crate::Generator`] contract.
    pub edit: Option<crate::moss_image_edit::MossImageEditRequest>,
}

impl MossImageExtension {
    pub fn from_options(options: crate::moss_image::MossImageOptions) -> Self {
        Self {
            options: Some(options),
            edit: None,
        }
    }

    pub fn from_edit(edit: crate::moss_image_edit::MossImageEditRequest) -> Self {
        Self {
            options: None,
            edit: Some(edit),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.options.is_none() && self.edit.is_none()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self { options, edit } = self;
        if let Some(v) = options
            .as_ref()
            .and_then(|opts| opts.first_nonfinite_float())
        {
            return Some(v);
        }
        edit.as_ref()
            .and_then(crate::moss_image_edit::MossImageEditRequest::first_nonfinite_float)
    }
}

/// FLUX 3 family extension (`REG-4`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Flux3Extension {
    /// FLUX 3 reference-output intent before API defaults are inserted.
    pub reference_output: Option<crate::flux3::references::ReferenceOutputSize>,
    /// Lossless FLUX 3 placement/edit layout document.
    pub layout: Option<crate::flux3::layouts::Layout>,
    /// Caller-resolved attachment identities and decoded geometry in conditioning order.
    pub layout_bindings: Vec<crate::flux3::layouts::ReferenceBinding>,
}

impl Flux3Extension {
    pub fn is_empty(&self) -> bool {
        self.reference_output.is_none()
            && self.layout.is_none()
            && self.layout_bindings.is_empty()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            reference_output: _,
            layout: _,
            layout_bindings: _,
        } = self;
        None
    }
}

/// MiniMax-H3 / MiniMax-H3-Image family extension (`REG-4`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct H3Extension {
    /// MiniMax-H3-Image explicit sampler parameters (SW-7106).
    pub image_sampler: Option<crate::h3_image::H3ImageSampler>,
    /// MiniMax-H3 learned sparse attention request.
    pub veda: Option<crate::VedaRequest>,
    /// Operator-owned terminal H3 measurement toggle.
    pub terminal_telemetry: bool,
    /// Short edge an image reference is encoded at in pixels (ref2va).
    pub reference_image_short_edge: Option<u32>,
    /// Reference-image resize policy before encoding (ref2va).
    pub reference_image_resize_mode: Option<crate::ReferenceImageResizeMode>,
}

impl H3Extension {
    pub fn is_empty(&self) -> bool {
        self.image_sampler.is_none()
            && self.veda.is_none()
            && !self.terminal_telemetry
            && self.reference_image_short_edge.is_none()
            && self.reference_image_resize_mode.is_none()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            image_sampler,
            veda: _,
            terminal_telemetry: _,
            reference_image_short_edge: _,
            reference_image_resize_mode: _,
        } = self;
        image_sampler.and_then(|v| v.first_nonfinite_float())
    }
}

/// LTX-2.3 / LTX-2.5 family extension (`REG-4`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LtxExtension {
    /// Number of extra generated keyframe slots placed at evenly spaced interior positions.
    pub num_generated_keyframes: Option<u32>,
    /// Number of DFR temporal ×2 refine rounds (`0..=2`).
    pub temporal_upsample_rounds: Option<u32>,
    /// Auto-duration opt-in range clamped by the checkpoint's duration head.
    pub auto_duration: Option<crate::duration_head::AutoDurationRange>,
    /// Use the separate uncensored 4-bit Gemma enhancer instead of the loaded text encoder.
    pub use_uncensored_enhancer: bool,
}

impl LtxExtension {
    pub fn is_empty(&self) -> bool {
        self.num_generated_keyframes.is_none()
            && self.temporal_upsample_rounds.is_none()
            && self.auto_duration.is_none()
            && !self.use_uncensored_enhancer
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            num_generated_keyframes: _,
            temporal_upsample_rounds: _,
            auto_duration: _,
            use_uncensored_enhancer: _,
        } = self;
        None
    }
}

/// SVD-XT micro-conditioning family extension (`REG-4`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SvdExtension {
    /// SVD `motion_bucket_id` micro-conditioning.
    pub motion_bucket_id: Option<f32>,
    /// SVD `noise_aug_strength` Gaussian noise added to conditioning image.
    pub noise_aug_strength: Option<f32>,
    /// Frames decoded per temporal-VAE pass (`decode_chunk_size`).
    pub decode_chunk_size: Option<u32>,
    /// SVD motion conditioning fps (`fps - 1` in `added_time_ids`), distinct from output `fps`.
    pub conditioning_fps: Option<u32>,
}

impl SvdExtension {
    pub fn is_empty(&self) -> bool {
        self.motion_bucket_id.is_none()
            && self.noise_aug_strength.is_none()
            && self.decode_chunk_size.is_none()
            && self.conditioning_fps.is_none()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            motion_bucket_id,
            noise_aug_strength,
            decode_chunk_size: _,
            conditioning_fps: _,
        } = self;
        if let Some(v) = *motion_bucket_id {
            if !v.is_finite() {
                return Some(("motion_bucket_id", v));
            }
        }
        if let Some(v) = *noise_aug_strength {
            if !v.is_finite() {
                return Some(("noise_aug_strength", v));
            }
        }
        None
    }
}

/// PuLID-FLUX family extension (`REG-4`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PulidExtension {
    /// CFG-scheduling start step (`timestep_to_start_cfg`).
    pub timestep_to_start_cfg: Option<u32>,
}

impl PulidExtension {
    pub fn is_empty(&self) -> bool {
        self.timestep_to_start_cfg.is_none()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            timestep_to_start_cfg: _,
        } = self;
        None
    }
}

/// SeedVR2 super-resolution family extension (`REG-4`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SeedVr2Extension {
    /// Pre-blur softness applied to bicubic-upscaled input before VAE encode.
    pub softness: Option<f32>,
}

impl SeedVr2Extension {
    pub fn is_empty(&self) -> bool {
        self.softness.is_none()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self { softness } = self;
        if let Some(v) = *softness {
            if !v.is_finite() {
                return Some(("softness", v));
            }
        }
        None
    }
}

/// Wan video family extension (`REG-4`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WanExtension {
    /// Extra leading temporal chunks generated and discarded after decode (`trim_first_frames`).
    pub trim_first_frames: Option<u32>,
}

impl WanExtension {
    pub fn is_empty(&self) -> bool {
        self.trim_first_frames.is_none()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            trim_first_frames: _,
        } = self;
        None
    }
}

/// Krea / Qwen-Image multi-tap text encoder family extension (`REG-4`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KreaExtension {
    /// Krea "text style" gain reweighting the 12 stacked Qwen3-VL select-layer taps.
    pub text_style_gain: Option<f32>,
}

impl KreaExtension {
    pub fn is_empty(&self) -> bool {
        self.text_style_gain.is_none()
    }

    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self { text_style_gain } = self;
        if let Some(v) = *text_style_gain {
            if !v.is_finite() {
                return Some(("text_style_gain", v));
            }
        }
        None
    }
}

/// Single typed family extension block (`REG-4`).
#[derive(Clone, Debug, PartialEq)]
pub enum FamilyExtension {
    MossImage(Box<MossImageExtension>),
    Flux3(Flux3Extension),
    H3(H3Extension),
    Ltx(LtxExtension),
    Svd(SvdExtension),
    Pulid(PulidExtension),
    SeedVr2(SeedVr2Extension),
    Wan(WanExtension),
    Krea(KreaExtension),
}

/// Composable container of typed family extension blocks carried on
/// [`GenerationRequest::extensions`](super::GenerationRequest::extensions) (`REG-4`).
///
/// Using typed optional slots inside one container allows a request to carry composed extensions
/// when needed while guaranteeing zero family-specific top-level fields on `GenerationRequest`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FamilyExtensions {
    pub moss_image: Option<MossImageExtension>,
    pub flux3: Option<Flux3Extension>,
    pub h3: Option<H3Extension>,
    pub ltx: Option<LtxExtension>,
    pub svd: Option<SvdExtension>,
    pub pulid: Option<PulidExtension>,
    pub seedvr2: Option<SeedVr2Extension>,
    pub wan: Option<WanExtension>,
    pub krea: Option<KreaExtension>,
}

impl FamilyExtensions {
    pub fn is_empty(&self) -> bool {
        let Self {
            moss_image,
            flux3,
            h3,
            ltx,
            svd,
            pulid,
            seedvr2,
            wan,
            krea,
        } = self;
        moss_image.as_ref().is_none_or(MossImageExtension::is_empty)
            && flux3.as_ref().is_none_or(Flux3Extension::is_empty)
            && h3.as_ref().is_none_or(H3Extension::is_empty)
            && ltx.as_ref().is_none_or(LtxExtension::is_empty)
            && svd.as_ref().is_none_or(SvdExtension::is_empty)
            && pulid.as_ref().is_none_or(PulidExtension::is_empty)
            && seedvr2.as_ref().is_none_or(SeedVr2Extension::is_empty)
            && wan.as_ref().is_none_or(WanExtension::is_empty)
            && krea.as_ref().is_none_or(KreaExtension::is_empty)
    }

    /// Insert or replace a single typed [`FamilyExtension`] block.
    pub fn set(&mut self, extension: FamilyExtension) {
        match extension {
            FamilyExtension::MossImage(v) => self.moss_image = Some(*v),
            FamilyExtension::Flux3(v) => self.flux3 = Some(v),
            FamilyExtension::H3(v) => self.h3 = Some(v),
            FamilyExtension::Ltx(v) => self.ltx = Some(v),
            FamilyExtension::Svd(v) => self.svd = Some(v),
            FamilyExtension::Pulid(v) => self.pulid = Some(v),
            FamilyExtension::SeedVr2(v) => self.seedvr2 = Some(v),
            FamilyExtension::Wan(v) => self.wan = Some(v),
            FamilyExtension::Krea(v) => self.krea = Some(v),
        }
    }

    /// Mutable access to the MOSS-Image extension block, initializing with defaults if absent.
    pub fn moss_image_mut(&mut self) -> &mut MossImageExtension {
        self.moss_image.get_or_insert_with(Default::default)
    }

    /// Mutable access to the FLUX.3 extension block, initializing with defaults if absent.
    pub fn flux3_mut(&mut self) -> &mut Flux3Extension {
        self.flux3.get_or_insert_with(Default::default)
    }

    /// Mutable access to the MiniMax-H3 extension block, initializing with defaults if absent.
    pub fn h3_mut(&mut self) -> &mut H3Extension {
        self.h3.get_or_insert_with(Default::default)
    }

    /// Mutable access to the LTX extension block, initializing with defaults if absent.
    pub fn ltx_mut(&mut self) -> &mut LtxExtension {
        self.ltx.get_or_insert_with(Default::default)
    }

    /// Mutable access to the SVD extension block, initializing with defaults if absent.
    pub fn svd_mut(&mut self) -> &mut SvdExtension {
        self.svd.get_or_insert_with(Default::default)
    }

    /// Mutable access to the PuLID extension block, initializing with defaults if absent.
    pub fn pulid_mut(&mut self) -> &mut PulidExtension {
        self.pulid.get_or_insert_with(Default::default)
    }

    /// Mutable access to the SeedVR2 extension block, initializing with defaults if absent.
    pub fn seedvr2_mut(&mut self) -> &mut SeedVr2Extension {
        self.seedvr2.get_or_insert_with(Default::default)
    }

    /// Mutable access to the Wan extension block, initializing with defaults if absent.
    pub fn wan_mut(&mut self) -> &mut WanExtension {
        self.wan.get_or_insert_with(Default::default)
    }

    /// Mutable access to the Krea extension block, initializing with defaults if absent.
    pub fn krea_mut(&mut self) -> &mut KreaExtension {
        self.krea.get_or_insert_with(Default::default)
    }

    /// Exhaustive non-finite float check across every family extension slot.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            moss_image,
            flux3,
            h3,
            ltx,
            svd,
            pulid,
            seedvr2,
            wan,
            krea,
        } = self;
        if let Some(v) = moss_image
            .as_ref()
            .and_then(MossImageExtension::first_nonfinite_float)
        {
            return Some(v);
        }
        if let Some(v) = flux3
            .as_ref()
            .and_then(Flux3Extension::first_nonfinite_float)
        {
            return Some(v);
        }
        if let Some(v) = h3.as_ref().and_then(H3Extension::first_nonfinite_float) {
            return Some(v);
        }
        if let Some(v) = ltx.as_ref().and_then(LtxExtension::first_nonfinite_float) {
            return Some(v);
        }
        if let Some(v) = svd.as_ref().and_then(SvdExtension::first_nonfinite_float) {
            return Some(v);
        }
        if let Some(v) = pulid
            .as_ref()
            .and_then(PulidExtension::first_nonfinite_float)
        {
            return Some(v);
        }
        if let Some(v) = seedvr2
            .as_ref()
            .and_then(SeedVr2Extension::first_nonfinite_float)
        {
            return Some(v);
        }
        if let Some(v) = wan.as_ref().and_then(WanExtension::first_nonfinite_float) {
            return Some(v);
        }
        krea.as_ref()
            .and_then(KreaExtension::first_nonfinite_float)
    }
}
