//! Shared ENG-6 multi-stream scheduling, typed stream-transition, and `MEM-3` live-set
//! builders (`PIPE-4`, `PIPE-6`, `DEC-3`, `MEM-3`).
//!
//! Zero tensor dependencies: every type here operates over descriptor metadata, immutable `f32`
//! sigma tables, typed boundary states, and integer byte budgets so `SW-6972..SW-6977` can share
//! one uniform contract without merge contention on monolithic files.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::driver::DriverStepInfo;
use crate::{
    BoundaryState, Error, LatentNormalization, LatentPatchLayout, LatentSpace, LatentTemporalLaw,
    MemoryPhase, PhaseOperationLiveSet, PricedAllocation, Result, StreamPhaseGeometry,
    TransitionMechanismKind, TransitionOperationLiveSet,
};

/// Canonical identifier for an independently shaped/timed latent stream (`PIPE-4`, `PIPE-6`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamId {
    Video,
    Audio,
    Joint,
    Named(String),
}

impl StreamId {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Joint => "joint",
            Self::Named(name) => name.as_str(),
        }
    }
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How timestep/sigma values are assigned across tokens of a stream (`PIPE-6`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerTokenTimePolicy {
    /// Every token in the stream shares one scalar σ per step (e.g., LTX audio or pure T2V).
    Uniform,
    /// Per-token σ derived from `σ * denoise_mask` plus clean-latent pinning (e.g., LTX
    /// I2V / keyframes / IC-LoRA clips / DFR anchors & slots).
    PerTokenMask,
}

/// Serializable descriptor capturing a stream's [`LatentSpace`] identity and normalization law.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamLatentDescriptor {
    pub channels: u16,
    pub spatial_compression: (u16, u16),
    pub temporal_scale: u16,
    pub patch_layout: String,
    pub normalization: String,
}

impl StreamLatentDescriptor {
    /// Describe an unpacked audio mel-band latent stream (such as LTX `(B, 8, T, 16)`).
    pub fn audio(channels: u16, mel_bins: u16, normalization: impl Into<String>) -> Self {
        Self {
            channels,
            spatial_compression: (1, mel_bins),
            temporal_scale: 1,
            patch_layout: "unpacked".to_owned(),
            normalization: normalization.into(),
        }
    }

    pub fn validate(&self, stream: &StreamId) -> Result<()> {
        if self.channels == 0 {
            return Err(Error::Unsupported(format!(
                "stream '{stream}': latent channel count must be non-zero"
            )));
        }
        if self.spatial_compression.0 == 0 || self.spatial_compression.1 == 0 {
            return Err(Error::Unsupported(format!(
                "stream '{stream}': spatial compression {:?} must be non-zero",
                self.spatial_compression
            )));
        }
        if self.temporal_scale == 0 {
            return Err(Error::Unsupported(format!(
                "stream '{stream}': temporal_scale must be non-zero"
            )));
        }
        if self.normalization.trim().is_empty() {
            return Err(Error::Unsupported(format!(
                "stream '{stream}': latent normalization description must not be empty"
            )));
        }
        Ok(())
    }
}

impl From<&LatentSpace> for StreamLatentDescriptor {
    fn from(space: &LatentSpace) -> Self {
        let temporal_scale = match space.temporal_law {
            LatentTemporalLaw::None => 1,
            LatentTemporalLaw::Causal4x => 4,
            LatentTemporalLaw::Causal6x => 6,
            LatentTemporalLaw::Causal8x => 8,
        };
        let patch_layout = match space.patch_layout {
            LatentPatchLayout::Unpacked => "unpacked".to_owned(),
            LatentPatchLayout::Packed {
                patch_height,
                patch_width,
            } => format!("packed({patch_height}x{patch_width})"),
        };
        let normalization = match space.normalization {
            LatentNormalization::Identity => "identity".to_owned(),
            LatentNormalization::Affine {
                scale_bits,
                shift_bits,
            } => format!(
                "affine(scale={},shift={})",
                f32::from_bits(scale_bits),
                f32::from_bits(shift_bits)
            ),
            LatentNormalization::PerChannel(stats) => {
                format!(
                    "per_channel({},ch={},hash={:#x})",
                    stats.identity, stats.channels, stats.content_hash
                )
            }
            LatentNormalization::LearnedPerChannel { identity } => {
                format!("learned_per_channel({identity})")
            }
        };
        Self {
            channels: space.channels,
            spatial_compression: (
                space.spatial_compression.height,
                space.spatial_compression.width,
            ),
            temporal_scale,
            patch_layout,
            normalization,
        }
    }
}

/// Physical latent geometry carried by a stream within one pass or stage (`MEM-3`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamGeometry {
    VideoGrid {
        width: u32,
        height: u32,
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        channels: u16,
    },
    VideoTokens {
        width: u32,
        height: u32,
        latent_frames: usize,
        target_tokens: usize,
        conditioning_tokens: usize,
        channels: u16,
    },
    AudioGrid {
        audio_frames: usize,
        mel_bins: u16,
        channels: u16,
        duration_millis: u64,
    },
    Custom {
        shape: Vec<usize>,
    },
}

impl StreamGeometry {
    pub fn validate(&self, stream: &StreamId) -> Result<()> {
        match *self {
            Self::VideoGrid {
                width,
                height,
                latent_frames,
                latent_height,
                latent_width,
                channels,
            } => {
                if width == 0
                    || height == 0
                    || latent_frames == 0
                    || latent_height == 0
                    || latent_width == 0
                    || channels == 0
                {
                    return Err(Error::Unsupported(format!(
                        "stream '{stream}': video grid geometry must have non-zero dimensions"
                    )));
                }
            }
            Self::VideoTokens {
                width,
                height,
                latent_frames,
                target_tokens,
                channels,
                ..
            } => {
                if width == 0
                    || height == 0
                    || latent_frames == 0
                    || target_tokens == 0
                    || channels == 0
                {
                    return Err(Error::Unsupported(format!(
                        "stream '{stream}': video token geometry must have non-zero target dimensions"
                    )));
                }
            }
            Self::AudioGrid {
                audio_frames,
                mel_bins,
                channels,
                ..
            } => {
                if audio_frames == 0 || mel_bins == 0 || channels == 0 {
                    return Err(Error::Unsupported(format!(
                        "stream '{stream}': audio grid geometry must have non-zero dimensions"
                    )));
                }
            }
            Self::Custom { ref shape } => {
                if shape.is_empty() || shape.contains(&0) {
                    return Err(Error::Unsupported(format!(
                        "stream '{stream}': custom geometry must have non-zero dimensions"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn spatial_dimensions(&self) -> Option<(u32, u32)> {
        match *self {
            Self::VideoGrid { width, height, .. } | Self::VideoTokens { width, height, .. } => {
                Some((width, height))
            }
            Self::AudioGrid { .. } | Self::Custom { .. } => None,
        }
    }

    /// Convert into [`StreamPhaseGeometry`] for [`crate::memory_strategy::pricing`].
    pub fn to_phase_geometry(&self, stream: &StreamId) -> StreamPhaseGeometry {
        let mut geom = StreamPhaseGeometry::image(stream.as_str(), 0, 0, 1);
        match *self {
            Self::VideoGrid {
                width,
                height,
                latent_frames,
                ..
            }
            | Self::VideoTokens {
                width,
                height,
                latent_frames,
                ..
            } => {
                geom.width = width;
                geom.height = height;
                geom.latent_frames = latent_frames.min(u32::MAX as usize) as u32;
            }
            Self::AudioGrid {
                audio_frames,
                duration_millis,
                ..
            } => {
                geom.latent_frames = 0;
                geom.audio_tokens = audio_frames as u64;
                geom.audio_duration_millis = duration_millis;
            }
            Self::Custom { .. } => {}
        }
        geom
    }
}

/// Complete typed specification of one active latent stream (`PIPE-4`, `PIPE-6`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSpec {
    pub id: StreamId,
    pub latent_space: StreamLatentDescriptor,
    pub geometry: StreamGeometry,
    pub token_time_policy: PerTokenTimePolicy,
}

impl StreamSpec {
    pub fn from_latent_space(
        id: StreamId,
        space: &LatentSpace,
        geometry: StreamGeometry,
        token_time_policy: PerTokenTimePolicy,
    ) -> Result<Self> {
        let latent_space = StreamLatentDescriptor::from(space);
        let spec = Self {
            id,
            latent_space,
            geometry,
            token_time_policy,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        self.latent_space.validate(&self.id)?;
        self.geometry.validate(&self.id)?;
        Ok(())
    }
}

/// Ordered bundle of concurrent streams operating within one phase or stage (`PIPE-4`, `MEM-3`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamBundleSpec {
    pub bundle_id: String,
    pub streams: Vec<StreamSpec>,
}

impl StreamBundleSpec {
    pub fn new(bundle_id: impl Into<String>, streams: Vec<StreamSpec>) -> Result<Self> {
        let bundle = Self {
            bundle_id: bundle_id.into(),
            streams,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<()> {
        if self.bundle_id.trim().is_empty() {
            return Err(Error::Unsupported(
                "StreamBundleSpec: bundle_id must not be empty".into(),
            ));
        }
        if self.streams.is_empty() {
            return Err(Error::Unsupported(format!(
                "stream bundle '{}': bundle must declare at least one stream",
                self.bundle_id
            )));
        }
        let mut seen = BTreeSet::new();
        for stream in &self.streams {
            stream.validate()?;
            if !seen.insert(stream.id.clone()) {
                return Err(Error::Unsupported(format!(
                    "stream bundle '{}': duplicate stream id '{}'",
                    self.bundle_id, stream.id
                )));
            }
        }
        Ok(())
    }

    pub fn stream(&self, id: &StreamId) -> Option<&StreamSpec> {
        self.streams.iter().find(|s| &s.id == id)
    }

    pub fn contains(&self, id: &StreamId) -> bool {
        self.stream(id).is_some()
    }

    pub fn stream_ids(&self) -> Vec<StreamId> {
        self.streams.iter().map(|s| s.id.clone()).collect()
    }
}

/// Classification tag preserved in [`ResolvedSchedule`] after resolution (`DEC-3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleSourceKind {
    GlobalSliced,
    FreshPerPass,
    ExplicitTable,
}

/// Declarative provenance of a stage/pass sigma schedule before resolution to one immutable
/// vector (`DEC-3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleSource {
    /// Contiguous `[start_step..=end_step]` slice of a pass-wide schedule table.
    GlobalSliced {
        table_id: String,
        full_schedule: Vec<f32>,
        start_step: usize,
        end_step: usize,
    },
    /// Pass-generated discrete schedule (such as Dev linear `N=30` steps from `1.0 -> 0.0`).
    FreshPerPass { table_id: String, sigmas: Vec<f32> },
    /// Explicit baked waypoint table (such as `STAGE1_SIGMAS`, `STAGE2_SIGMAS`, `TEMPORAL_SIGMAS`).
    ExplicitTable { table_id: String, sigmas: Vec<f32> },
}

impl ScheduleSource {
    /// Resolve any [`ScheduleSource`] variant into one immutable, validated [`ResolvedSchedule`]
    /// (`DEC-3`).
    pub fn resolve(&self) -> Result<ResolvedSchedule> {
        let (table_id, source_kind, sigmas) = match self {
            Self::GlobalSliced {
                table_id,
                full_schedule,
                start_step,
                end_step,
            } => {
                if *start_step >= *end_step || *end_step >= full_schedule.len() {
                    return Err(Error::Unsupported(format!(
                        "schedule '{table_id}': invalid GlobalSliced span {start_step}..={end_step} for table length {}",
                        full_schedule.len()
                    )));
                }
                (
                    table_id.clone(),
                    ScheduleSourceKind::GlobalSliced,
                    full_schedule[*start_step..=*end_step].to_vec(),
                )
            }
            Self::FreshPerPass { table_id, sigmas } => (
                table_id.clone(),
                ScheduleSourceKind::FreshPerPass,
                sigmas.clone(),
            ),
            Self::ExplicitTable { table_id, sigmas } => (
                table_id.clone(),
                ScheduleSourceKind::ExplicitTable,
                sigmas.clone(),
            ),
        };
        ResolvedSchedule::new(table_id, source_kind, sigmas)
    }
}

/// Single immutable sigma schedule produced from [`ScheduleSource::resolve`] (`DEC-3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedSchedule {
    pub table_id: String,
    pub source_kind: ScheduleSourceKind,
    sigmas: Vec<f32>,
}

impl ResolvedSchedule {
    pub fn new(
        table_id: impl Into<String>,
        source_kind: ScheduleSourceKind,
        sigmas: Vec<f32>,
    ) -> Result<Self> {
        let table_id = table_id.into();
        if table_id.trim().is_empty() {
            return Err(Error::Unsupported(
                "ResolvedSchedule: table_id must not be empty".into(),
            ));
        }
        if sigmas.len() < 2 {
            return Err(Error::Unsupported(format!(
                "schedule '{table_id}': expected at least 2 sigma entries (>= 1 step), got {}",
                sigmas.len()
            )));
        }
        for (idx, &sigma) in sigmas.iter().enumerate() {
            if !sigma.is_finite() || sigma < 0.0 {
                return Err(Error::Unsupported(format!(
                    "schedule '{table_id}': sigma[{idx}]={sigma} must be finite and >= 0.0"
                )));
            }
            if idx + 1 < sigmas.len() {
                if sigma <= 0.0 {
                    return Err(Error::Unsupported(format!(
                        "schedule '{table_id}': active transition sigma[{idx}]={sigma} must be > 0.0"
                    )));
                }
                if sigmas[idx + 1] > sigma {
                    return Err(Error::Unsupported(format!(
                        "schedule '{table_id}': non-monotone transition sigma[{idx}]={sigma} -> sigma[{}]={}",
                        idx + 1,
                        sigmas[idx + 1]
                    )));
                }
            }
        }
        Ok(Self {
            table_id,
            source_kind,
            sigmas,
        })
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    pub fn entry_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    pub fn terminal_sigma(&self) -> f32 {
        self.sigmas[self.sigmas.len() - 1]
    }

    pub fn step_window(&self, step: usize) -> Option<(f32, f32)> {
        if step + 1 < self.sigmas.len() {
            Some((self.sigmas[step], self.sigmas[step + 1]))
        } else {
            None
        }
    }

    pub fn stream_step_sigma(&self, stream: StreamId, step: usize) -> Option<StreamStepSigma> {
        self.step_window(step)
            .map(|(sigma, next_sigma)| StreamStepSigma {
                stream,
                sigma,
                next_sigma,
            })
    }
}

/// Per-stream σ transition coordinates at one evaluation/step (`PIPE-6`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamStepSigma {
    pub stream: StreamId,
    pub sigma: f32,
    pub next_sigma: f32,
}

/// Step metadata pairing shared scalar [`DriverStepInfo`] with explicit per-stream sigmas
/// (`PIPE-6`, `DEC-19`).
#[derive(Clone, Debug, PartialEq)]
pub struct MultiStreamStepInfo {
    pub step: DriverStepInfo,
    pub stream_sigmas: Vec<StreamStepSigma>,
}

impl MultiStreamStepInfo {
    pub fn sigma_for(&self, stream: &StreamId) -> Option<(f32, f32)> {
        self.stream_sigmas
            .iter()
            .find(|entry| &entry.stream == stream)
            .map(|entry| (entry.sigma, entry.next_sigma))
    }
}

/// Explicit learned component metadata carried by a [`StreamTransitionKind`] (`MEM-3`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearnedUpsamplerComponent {
    pub name: String,
    pub normalization: String,
    pub workspace_bytes: u64,
}

/// Typed state semantics across transition chaining (`PIPE-4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamBoundarySemantics {
    NoisyXSigma,
    CleanEstimate,
    RouteSpecific,
}

/// Typed boundary transition operator between passes or stages (`PIPE-4`, `MEM-3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamTransitionKind {
    /// Contiguous continuation at the current noisy state `BoundaryState::XSigma`.
    Continue,
    /// Fresh stage or round re-noising from [`BoundaryState::CleanEstimate`] ONLY (`PIPE-4`).
    Renoise { sigma: f32 },
    /// Latent-space spatial or temporal resample (optionally via a [`LearnedUpsamplerComponent`])
    /// operating ONLY on [`BoundaryState::CleanEstimate`].
    LatentResample {
        upsampler: Option<LearnedUpsamplerComponent>,
        spatial_factor: (u16, u16),
        temporal_factor: u16,
    },
    /// Pixel-space decode -> edit/upsample -> encode round trip from [`BoundaryState::CleanEstimate`].
    PixelRoundTrip {
        decoder_name: String,
        encoder_name: String,
        workspace_bytes: u64,
    },
    /// Explicit route-owned boundary state (`BoundaryState::RouteSpecific`).
    RouteSupplied { route_key: String },
    /// Ordered sequence of boundary operators evaluated left-to-right (for example,
    /// `[LatentResample { .. }, Renoise { sigma }]`).
    Composed(Vec<StreamTransitionKind>),
}

impl StreamTransitionKind {
    /// Validate internal parameters AND compose boundary semantics (`CleanEstimate` vs `XSigma`),
    /// returning `(required_entry, produced_exit)`.
    pub fn boundary_semantics(
        &self,
        transition_id: &str,
    ) -> Result<(StreamBoundarySemantics, StreamBoundarySemantics)> {
        match self {
            Self::Continue => Ok((
                StreamBoundarySemantics::NoisyXSigma,
                StreamBoundarySemantics::NoisyXSigma,
            )),
            Self::Renoise { sigma } => {
                if !sigma.is_finite() || !(0.0..=1.0).contains(sigma) || *sigma == 0.0 {
                    return Err(Error::Unsupported(format!(
                        "transition '{transition_id}': Renoise sigma={sigma} must be finite and in (0.0, 1.0]"
                    )));
                }
                Ok((
                    StreamBoundarySemantics::CleanEstimate,
                    StreamBoundarySemantics::NoisyXSigma,
                ))
            }
            Self::LatentResample {
                upsampler,
                spatial_factor,
                temporal_factor,
            } => {
                if spatial_factor.0 == 0 || spatial_factor.1 == 0 || *temporal_factor == 0 {
                    return Err(Error::Unsupported(format!(
                        "transition '{transition_id}': resample factors spatial={spatial_factor:?} temporal={temporal_factor} must be non-zero"
                    )));
                }
                if let Some(comp) = upsampler {
                    if comp.name.trim().is_empty() || comp.normalization.trim().is_empty() {
                        return Err(Error::Unsupported(format!(
                            "transition '{transition_id}': learned upsampler name and normalization must not be empty"
                        )));
                    }
                    if comp.workspace_bytes == 0 {
                        return Err(Error::Unsupported(format!(
                            "transition '{transition_id}': learned upsampler '{}' workspace_bytes must be > 0 (MEM-3)",
                            comp.name
                        )));
                    }
                }
                Ok((
                    StreamBoundarySemantics::CleanEstimate,
                    StreamBoundarySemantics::CleanEstimate,
                ))
            }
            Self::PixelRoundTrip {
                decoder_name,
                encoder_name,
                workspace_bytes,
            } => {
                if decoder_name.trim().is_empty() || encoder_name.trim().is_empty() {
                    return Err(Error::Unsupported(format!(
                        "transition '{transition_id}': PixelRoundTrip decoder and encoder names must not be empty"
                    )));
                }
                if *workspace_bytes == 0 {
                    return Err(Error::Unsupported(format!(
                        "transition '{transition_id}': PixelRoundTrip workspace_bytes must be > 0 (MEM-3)"
                    )));
                }
                Ok((
                    StreamBoundarySemantics::CleanEstimate,
                    StreamBoundarySemantics::CleanEstimate,
                ))
            }
            Self::RouteSupplied { route_key } => {
                if route_key.trim().is_empty() {
                    return Err(Error::Unsupported(format!(
                        "transition '{transition_id}': RouteSupplied route_key must not be empty"
                    )));
                }
                Ok((
                    StreamBoundarySemantics::RouteSpecific,
                    StreamBoundarySemantics::RouteSpecific,
                ))
            }
            Self::Composed(steps) => {
                if steps.is_empty() {
                    return Err(Error::Unsupported(format!(
                        "transition '{transition_id}': Composed transition must contain at least one step"
                    )));
                }
                let mut iter = steps.iter();
                let first = iter.next().expect("non-empty checked above");
                let (entry, mut current) = first.boundary_semantics(transition_id)?;
                for (idx, step) in iter.enumerate() {
                    let (req, next) = step.boundary_semantics(transition_id)?;
                    if req != current {
                        return Err(Error::Unsupported(format!(
                            "transition '{transition_id}': Composed step {} expects {:?} boundary state, but prior step emitted {:?} (PIPE-4 double-noise / incompatible state guard)",
                            idx + 1,
                            req,
                            current
                        )));
                    }
                    current = next;
                }
                Ok((entry, current))
            }
        }
    }

    /// Learned transition workspace bytes (`MEM-3`), or `None` when purely latent spatial.
    pub fn learned_workspace_bytes(&self) -> Option<u64> {
        match self {
            Self::Continue | Self::Renoise { .. } | Self::RouteSupplied { .. } => None,
            Self::LatentResample { upsampler, .. } => upsampler.as_ref().map(|u| u.workspace_bytes),
            Self::PixelRoundTrip {
                workspace_bytes, ..
            } => Some(*workspace_bytes),
            Self::Composed(steps) => {
                let max_learned = steps.iter().filter_map(Self::learned_workspace_bytes).max();
                max_learned
            }
        }
    }

    /// Map this transition into `MEM-3`'s [`TransitionMechanismKind`].
    pub fn to_transition_mechanism(&self) -> TransitionMechanismKind {
        match self.learned_workspace_bytes() {
            Some(bytes) => TransitionMechanismKind::LearnedComponentTransition {
                learned_component_workspace_bytes: bytes,
            },
            None => TransitionMechanismKind::LatentSpatialTransition,
        }
    }
}

/// Stream-targeted transition specification (`PIPE-4`).
///
/// Only the named `targets` undergo `kind`; streams omitted from `targets` are explicitly
/// untouched at this boundary (for example, LTX audio skips spatial/temporal upsampling).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamTransition {
    pub transition_id: String,
    pub targets: Vec<StreamId>,
    pub kind: StreamTransitionKind,
}

impl StreamTransition {
    pub fn new(
        transition_id: impl Into<String>,
        targets: Vec<StreamId>,
        kind: StreamTransitionKind,
    ) -> Result<Self> {
        let transition = Self {
            transition_id: transition_id.into(),
            targets,
            kind,
        };
        transition.validate_structure()?;
        Ok(transition)
    }

    pub fn validate_structure(&self) -> Result<()> {
        if self.transition_id.trim().is_empty() {
            return Err(Error::Unsupported(
                "StreamTransition: transition_id must not be empty".into(),
            ));
        }
        if self.targets.is_empty() {
            return Err(Error::Unsupported(format!(
                "transition '{}': must declare at least one target stream",
                self.transition_id
            )));
        }
        let mut seen = BTreeSet::new();
        for target in &self.targets {
            if !seen.insert(target.clone()) {
                return Err(Error::Unsupported(format!(
                    "transition '{}': duplicate target stream '{target}'",
                    self.transition_id
                )));
            }
        }
        let _ = self.kind.boundary_semantics(&self.transition_id)?;
        Ok(())
    }

    /// Validate that every target stream exists in `bundle` (`PIPE-4`).
    pub fn validate_against_bundle(&self, bundle: &StreamBundleSpec) -> Result<()> {
        self.validate_structure()?;
        for target in &self.targets {
            if !bundle.contains(target) {
                return Err(Error::Unsupported(format!(
                    "transition '{}': targets stream '{target}' which is absent from bundle '{}'",
                    self.transition_id, bundle.bundle_id
                )));
            }
        }
        Ok(())
    }

    /// Enforce [`BoundaryState`] guard at runtime across all target streams (`PIPE-4`).
    ///
    /// Rejects `BoundaryState::XSigma` entering any `Renoise`, `LatentResample`, or
    /// `PixelRoundTrip` transition (prevents double-noising and noisy upsampling).
    pub fn validate_boundary_states<L>(
        &self,
        stream_states: &[(StreamId, &BoundaryState<L>)],
    ) -> Result<()> {
        self.validate_structure()?;
        let (required_entry, _) = self.kind.boundary_semantics(&self.transition_id)?;
        for target in &self.targets {
            let Some((_, state)) = stream_states.iter().find(|(id, _)| id == target) else {
                return Err(Error::Unsupported(format!(
                    "transition '{}': missing boundary state for target stream '{target}'",
                    self.transition_id
                )));
            };
            let actual = match state {
                BoundaryState::XSigma(_) => StreamBoundarySemantics::NoisyXSigma,
                BoundaryState::CleanEstimate { .. } => StreamBoundarySemantics::CleanEstimate,
                BoundaryState::RouteSpecific { .. } => StreamBoundarySemantics::RouteSpecific,
            };
            if actual != required_entry {
                return Err(Error::Unsupported(format!(
                    "transition '{}': target stream '{target}' supplied {:?}, but {:?} requires {:?} (PIPE-4 double-noising / boundary guard)",
                    self.transition_id, actual, self.kind, required_entry
                )));
            }
        }
        Ok(())
    }
}

/// Enforces `BoundaryState::CleanEstimate` -> transition -> `BoundaryState::XSigma` sequencing for
/// one stream boundary (`PIPE-4`).
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionBoundaryGuard<L> {
    pub stream: StreamId,
    pub state: BoundaryState<L>,
}

impl<L> TransitionBoundaryGuard<L> {
    pub fn new(stream: StreamId, state: BoundaryState<L>) -> Self {
        Self { stream, state }
    }

    /// Apply `transition` to `self`, refusing double-noise or uncleaned `XSigma` entry (`PIPE-4`).
    pub fn apply<Out, F>(self, transition: &StreamTransition, op: F) -> Result<BoundaryState<Out>>
    where
        F: FnOnce(BoundaryState<L>) -> Result<BoundaryState<Out>>,
    {
        if !transition.targets.contains(&self.stream) {
            return Err(Error::Unsupported(format!(
                "transition '{}': does not target stream '{}'",
                transition.transition_id, self.stream
            )));
        }
        let check = [(self.stream.clone(), &self.state)];
        transition.validate_boundary_states(&check)?;
        let (_, expected_exit) = transition
            .kind
            .boundary_semantics(&transition.transition_id)?;
        let out = op(self.state)?;
        let actual_exit = match &out {
            BoundaryState::XSigma(_) => StreamBoundarySemantics::NoisyXSigma,
            BoundaryState::CleanEstimate { .. } => StreamBoundarySemantics::CleanEstimate,
            BoundaryState::RouteSpecific { .. } => StreamBoundarySemantics::RouteSpecific,
        };
        if actual_exit != expected_exit {
            return Err(Error::Unsupported(format!(
                "transition '{}': expected output {:?} for stream '{}', produced {:?}",
                transition.transition_id, expected_exit, self.stream, actual_exit
            )));
        }
        Ok(out)
    }
}

/// Per-stream `MEM-3` contribution fed into [`build_multistream_phase_live_set`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamMemorySpec {
    pub spec: StreamSpec,
    /// Concurrent activations owned by this stream (always SUMMED across active streams, never maxed).
    pub activations: Vec<PricedAllocation>,
    /// Retained boundary/history buffers owned by this stream during the phase.
    pub retained: Vec<PricedAllocation>,
}

/// Build a [`PhaseOperationLiveSet`] for a concurrent multi-stream phase (`MEM-3`).
///
/// - Concurrent stream activations are strictly SUMMED alongside `joint_workspaces` (never maxed).
/// - `resident_weights`, `resident_overlays`, and aliased allocations carry deduplication keys
///   via [`PricedAllocation::alias_group`].
/// - `DeviceVram`, `HostRam`, and `UnifiedMemory` allocations remain strictly isolated per domain.
#[allow(clippy::too_many_arguments)]
pub fn build_multistream_phase_live_set(
    operation_id: impl Into<String>,
    phase: MemoryPhase,
    pass_index: usize,
    phase_index: usize,
    streams: &[StreamMemorySpec],
    resident_weights: Vec<PricedAllocation>,
    resident_overlays: Vec<PricedAllocation>,
    joint_workspaces: Vec<PricedAllocation>,
    extra_retained_state: Vec<PricedAllocation>,
) -> Result<PhaseOperationLiveSet> {
    if streams.is_empty() {
        return Err(Error::Unsupported(
            "build_multistream_phase_live_set: at least one active stream is required".into(),
        ));
    }
    let mut stream_geometries = Vec::with_capacity(streams.len());
    let mut concurrent_stream_activations = Vec::new();
    let mut retained_state = extra_retained_state;

    for stream in streams {
        stream.spec.validate()?;
        stream_geometries.push(stream.spec.geometry.to_phase_geometry(&stream.spec.id));
        concurrent_stream_activations.extend(stream.activations.iter().cloned());
        retained_state.extend(stream.retained.iter().cloned());
    }

    Ok(PhaseOperationLiveSet {
        operation_id: operation_id.into(),
        phase,
        pass_index,
        phase_index,
        stream_geometries,
        resident_weights,
        resident_overlays,
        concurrent_stream_activations,
        joint_workspaces,
        retained_state,
        evidence_peak: None,
    })
}

/// Build a [`TransitionOperationLiveSet`] for a [`StreamTransition`] (`MEM-3`).
///
/// Prices `resident_weights` + `resident_overlays` (deduplicated across aliases) + executing
/// learned/spatial component workspace + live boundary handoff buffers across strict budget
/// domains.
#[allow(clippy::too_many_arguments)]
pub fn build_stream_transition_live_set(
    transition: &StreamTransition,
    from_operation_id: impl Into<String>,
    to_operation_id: impl Into<String>,
    resident_weights: Vec<PricedAllocation>,
    resident_overlays: Vec<PricedAllocation>,
    transition_workspaces: Vec<PricedAllocation>,
    live_handoff_buffers: Vec<PricedAllocation>,
) -> Result<TransitionOperationLiveSet> {
    transition.validate_structure()?;
    Ok(TransitionOperationLiveSet {
        transition_id: transition.transition_id.clone(),
        from_operation_id: from_operation_id.into(),
        to_operation_id: to_operation_id.into(),
        mechanism: transition.kind.to_transition_mechanism(),
        resident_weights,
        resident_overlays,
        transition_workspaces,
        live_handoff_buffers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EstimateSource, MemoryBudgetDomain, RequestLiveSetPlan, LTX_VIDEO_LATENT_SPACE};

    #[test]
    fn explicit_schedule_sources_resolve_to_single_immutable_representation() {
        let global = ScheduleSource::GlobalSliced {
            table_id: "global_flow".into(),
            full_schedule: vec![1.0, 0.8, 0.6, 0.4, 0.0],
            start_step: 1,
            end_step: 4,
        }
        .resolve()
        .unwrap();
        assert_eq!(global.source_kind, ScheduleSourceKind::GlobalSliced);
        assert_eq!(global.sigmas(), &[0.8, 0.6, 0.4, 0.0]);
        assert_eq!(global.num_steps(), 3);

        let fresh = ScheduleSource::FreshPerPass {
            table_id: "dev_linear_30".into(),
            sigmas: vec![1.0, 0.5, 0.0],
        }
        .resolve()
        .unwrap();
        assert_eq!(fresh.source_kind, ScheduleSourceKind::FreshPerPass);
        assert_eq!(fresh.step_window(0), Some((1.0, 0.5)));

        let explicit = ScheduleSource::ExplicitTable {
            table_id: "ltx_stage2".into(),
            sigmas: vec![0.909_375, 0.725, 0.421_875, 0.0],
        }
        .resolve()
        .unwrap();
        assert_eq!(explicit.source_kind, ScheduleSourceKind::ExplicitTable);
        assert_eq!(explicit.entry_sigma(), 0.909_375);

        // Non-monotone, non-finite, and zero active transition sigmas are refused.
        assert!(ScheduleSource::ExplicitTable {
            table_id: "bad_up".into(),
            sigmas: vec![0.5, 0.9, 0.0],
        }
        .resolve()
        .is_err());
        assert!(ScheduleSource::ExplicitTable {
            table_id: "bad_nan".into(),
            sigmas: vec![1.0, f32::NAN, 0.0],
        }
        .resolve()
        .is_err());
        assert!(ScheduleSource::ExplicitTable {
            table_id: "bad_zero_active".into(),
            sigmas: vec![1.0, 0.0, 0.0],
        }
        .resolve()
        .is_err());
    }

    #[test]
    fn targeted_transitions_and_boundary_guards_refuse_double_noise_and_missing_targets() {
        let video = StreamSpec::from_latent_space(
            StreamId::Video,
            &LTX_VIDEO_LATENT_SPACE,
            StreamGeometry::VideoGrid {
                width: 512,
                height: 512,
                latent_frames: 7,
                latent_height: 8,
                latent_width: 8,
                channels: 128,
            },
            PerTokenTimePolicy::PerTokenMask,
        )
        .unwrap();
        let audio = StreamSpec {
            id: StreamId::Audio,
            latent_space: StreamLatentDescriptor::audio(8, 16, "identity"),
            geometry: StreamGeometry::AudioGrid {
                audio_frames: 25,
                mel_bins: 16,
                channels: 8,
                duration_millis: 1000,
            },
            token_time_policy: PerTokenTimePolicy::Uniform,
        };
        let bundle = StreamBundleSpec::new("ltx_av", vec![video, audio]).unwrap();

        let upsample_then_renoise = StreamTransition::new(
            "stage1_to_stage2_video",
            vec![StreamId::Video],
            StreamTransitionKind::Composed(vec![
                StreamTransitionKind::LatentResample {
                    upsampler: Some(LearnedUpsamplerComponent {
                        name: "ltx_spatial_x2".into(),
                        normalization: "vae_per_channel_denorm_renorm".into(),
                        workspace_bytes: 256 * 1024 * 1024,
                    }),
                    spatial_factor: (2, 2),
                    temporal_factor: 1,
                },
                StreamTransitionKind::Renoise { sigma: 0.909_375 },
            ]),
        )
        .unwrap();
        upsample_then_renoise
            .validate_against_bundle(&bundle)
            .unwrap();

        // Audio-only renoise (no upsampling).
        let audio_renoise = StreamTransition::new(
            "stage1_to_stage2_audio",
            vec![StreamId::Audio],
            StreamTransitionKind::Renoise { sigma: 0.909_375 },
        )
        .unwrap();
        audio_renoise.validate_against_bundle(&bundle).unwrap();

        // Entering Renoise/LatentResample with XSigma must be refused (double-noising guard, PIPE-4).
        let noisy_state = BoundaryState::XSigma(vec![1.0_f32]);
        let clean_state = BoundaryState::CleanEstimate {
            latent: vec![0.5_f32],
            source: EstimateSource::EndpointEvaluation,
        };
        let err = upsample_then_renoise
            .validate_boundary_states(&[(StreamId::Video, &noisy_state)])
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        upsample_then_renoise
            .validate_boundary_states(&[(StreamId::Video, &clean_state)])
            .unwrap();

        let guard = TransitionBoundaryGuard::new(StreamId::Video, clean_state);
        let out = guard
            .apply(&upsample_then_renoise, |state| match state {
                BoundaryState::CleanEstimate { latent, .. } => Ok(BoundaryState::XSigma(latent)),
                _ => unreachable!(),
            })
            .unwrap();
        assert!(matches!(out, BoundaryState::XSigma(_)));

        // Illegal composed order `[Renoise, LatentResample]` is rejected statically.
        assert!(StreamTransition::new(
            "illegal_order",
            vec![StreamId::Video],
            StreamTransitionKind::Composed(vec![
                StreamTransitionKind::Renoise { sigma: 0.9 },
                StreamTransitionKind::LatentResample {
                    upsampler: None,
                    spatial_factor: (2, 2),
                    temporal_factor: 1,
                },
            ]),
        )
        .is_err());

        // Learned upsampler with 0 workspace bytes violates MEM-3 and is refused.
        assert!(StreamTransition::new(
            "unpriced_upsampler",
            vec![StreamId::Video],
            StreamTransitionKind::LatentResample {
                upsampler: Some(LearnedUpsamplerComponent {
                    name: "ltx_spatial_x2".into(),
                    normalization: "vae".into(),
                    workspace_bytes: 0,
                }),
                spatial_factor: (2, 2),
                temporal_factor: 1,
            },
        )
        .is_err());
    }

    #[test]
    fn multistream_mem3_sums_concurrent_streams_deduplicates_aliases_and_keeps_domains_separate() {
        let video_spec = StreamSpec::from_latent_space(
            StreamId::Video,
            &LTX_VIDEO_LATENT_SPACE,
            StreamGeometry::VideoGrid {
                width: 704,
                height: 512,
                latent_frames: 7,
                latent_height: 8,
                latent_width: 11,
                channels: 128,
            },
            PerTokenTimePolicy::Uniform,
        )
        .unwrap();
        let audio_spec = StreamSpec {
            id: StreamId::Audio,
            latent_space: StreamLatentDescriptor::audio(8, 16, "identity"),
            geometry: StreamGeometry::AudioGrid {
                audio_frames: 51,
                mel_bins: 16,
                channels: 8,
                duration_millis: 2040,
            },
            token_time_policy: PerTokenTimePolicy::Uniform,
        };

        // Both streams alias the shared 10 GiB AvDiT trunk on DeviceVram; HostRam carries pinned staging.
        let op1 = build_multistream_phase_live_set(
            "stage1_denoise",
            MemoryPhase::Denoise,
            0,
            0,
            &[
                StreamMemorySpec {
                    spec: video_spec.clone(),
                    activations: vec![PricedAllocation::new(
                        "video_act",
                        MemoryBudgetDomain::DeviceVram,
                        400,
                    )],
                    retained: vec![PricedAllocation::new(
                        "host_video_pin",
                        MemoryBudgetDomain::HostRam,
                        50,
                    )],
                },
                StreamMemorySpec {
                    spec: audio_spec.clone(),
                    activations: vec![PricedAllocation::new(
                        "audio_act",
                        MemoryBudgetDomain::DeviceVram,
                        100,
                    )],
                    retained: vec![],
                },
            ],
            vec![
                PricedAllocation::new("av_dit_for_video", MemoryBudgetDomain::DeviceVram, 10_000)
                    .with_alias_group("av_dit_trunk"),
                PricedAllocation::new("av_dit_for_audio", MemoryBudgetDomain::DeviceVram, 10_000)
                    .with_alias_group("av_dit_trunk"),
            ],
            vec![PricedAllocation::new(
                "av_cross_attn_ws",
                MemoryBudgetDomain::DeviceVram,
                250,
            )],
            vec![PricedAllocation::new(
                "unified_diag",
                MemoryBudgetDomain::UnifiedMemory,
                75,
            )],
            vec![],
        )
        .unwrap();

        let upsample = StreamTransition::new(
            "spatial_upsample",
            vec![StreamId::Video],
            StreamTransitionKind::LatentResample {
                upsampler: Some(LearnedUpsamplerComponent {
                    name: "spatial_upsampler".into(),
                    normalization: "vae_stats".into(),
                    workspace_bytes: 1_500,
                }),
                spatial_factor: (2, 2),
                temporal_factor: 1,
            },
        )
        .unwrap();
        let tr = build_stream_transition_live_set(
            &upsample,
            "stage1_denoise",
            "stage2_denoise",
            vec![
                PricedAllocation::new("av_dit_resident", MemoryBudgetDomain::DeviceVram, 10_000)
                    .with_alias_group("av_dit_trunk"),
                PricedAllocation::new("spatial_weights", MemoryBudgetDomain::DeviceVram, 900),
            ],
            vec![],
            vec![PricedAllocation::new(
                "spatial_workspace",
                MemoryBudgetDomain::DeviceVram,
                1_500,
            )],
            vec![PricedAllocation::new(
                "handoff_bufs",
                MemoryBudgetDomain::DeviceVram,
                200,
            )],
        )
        .unwrap();

        let op2 = build_multistream_phase_live_set(
            "stage2_denoise",
            MemoryPhase::Denoise,
            1,
            1,
            &[
                StreamMemorySpec {
                    spec: video_spec,
                    activations: vec![PricedAllocation::new(
                        "video_act_s2",
                        MemoryBudgetDomain::DeviceVram,
                        800,
                    )],
                    retained: vec![],
                },
                StreamMemorySpec {
                    spec: audio_spec,
                    activations: vec![PricedAllocation::new(
                        "audio_act_s2",
                        MemoryBudgetDomain::DeviceVram,
                        100,
                    )],
                    retained: vec![],
                },
            ],
            vec![
                PricedAllocation::new("av_dit_s2", MemoryBudgetDomain::DeviceVram, 10_000)
                    .with_alias_group("av_dit_trunk"),
            ],
            vec![],
            vec![],
            vec![],
        )
        .unwrap();

        let plan = RequestLiveSetPlan {
            operations: vec![op1, op2],
            transitions: vec![tr],
        };
        let all = plan.price_all_domains().unwrap();
        // Stage 1: 10000 (deduped) + 400 + 100 + 250 = 10750
        // Stage 2: 10000 + 800 + 100 = 10900
        // Transition: 10000 + 900 + 1500 + 200 = 12600 (dominates!)
        assert_eq!(
            all[&MemoryBudgetDomain::DeviceVram].max_operation_peak_bytes,
            10_900
        );
        assert_eq!(
            all[&MemoryBudgetDomain::DeviceVram].max_transition_peak_bytes,
            12_600
        );
        assert_eq!(
            all[&MemoryBudgetDomain::DeviceVram].request_peak_bytes,
            12_600
        );
        assert_eq!(all[&MemoryBudgetDomain::HostRam].request_peak_bytes, 50);
        assert_eq!(
            all[&MemoryBudgetDomain::UnifiedMemory].request_peak_bytes,
            75
        );
    }
}
