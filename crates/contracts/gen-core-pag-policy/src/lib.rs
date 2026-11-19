//! Portable Perturbed-Attention Guidance (PAG) request, capability, resolution, and execution
//! report contracts (SW-7173).
//!
//! This crate owns the pure host and browser (`wasm32-unknown-unknown`) PAG validation and
//! step-window math; it has zero tensor, RNG, regex, filesystem, or OS dependencies.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Default upper bound for [`PagConfig::scale`] across PAG-capable descriptors.
pub const DEFAULT_PAG_MAX_SCALE: f32 = 15.0;

/// Classification of a [`PagPolicyError`] so native callers can map capability gaps to
/// `Error::Unsupported` and malformed parameters to `Error::Msg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PagPolicyErrorKind {
    /// The target model/backend does not advertise PAG support (`Capabilities::pag` is `None`).
    Unsupported,
    /// A PAG control or capability definition is non-finite, out of range, or malformed.
    Invalid,
}

/// Portable error returned by PAG validation and resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagPolicyError {
    pub kind: PagPolicyErrorKind,
    pub message: String,
}

impl PagPolicyError {
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: PagPolicyErrorKind::Unsupported,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: PagPolicyErrorKind::Invalid,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> PagPolicyErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for PagPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PagPolicyError {}

/// Unified request-level Perturbed-Attention Guidance (PAG) configuration.
///
/// Setting `scale == 0.0` (or omitting `PagConfig` altogether) resolves to a strict zero-pass
/// no-op while still validating any supplied floats, window bounds, and block indices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagConfig {
    /// Guidance scale (`>= 0.0`, bounded by [`PagCapability::max_scale`]). `0.0` disables PAG.
    pub scale: f32,
    /// Optional zero-based block indices to perturb. `None` selects
    /// [`PagCapability::default_layers`]. Explicit empty `Some(vec![])` is rejected when
    /// `scale > 0.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<u32>>,
    /// Normalized start of the active step window in `[0.0, 1.0]`. `None` uses
    /// [`PagCapability::default_start`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<f32>,
    /// Normalized exclusive stop of the active step window in `[0.0, 1.0]`. `None` uses
    /// [`PagCapability::default_stop`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<f32>,
    /// Optional linear decay slope (`>= 0.0`, default `0.0`) applied across normalized step
    /// progress: `effective_scale = (scale - adaptive_scale * progress).max(0.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_scale: Option<f32>,
}

impl Default for PagConfig {
    fn default() -> Self {
        Self {
            scale: 0.0,
            layers: None,
            start: None,
            stop: None,
            adaptive_scale: None,
        }
    }
}

impl PagConfig {
    /// Return the first non-finite float field in canonical order, using exhaustive destructuring
    /// so any newly added float field must be classified at compile time.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            scale,
            layers: _,
            start,
            stop,
            adaptive_scale,
        } = self;
        if !scale.is_finite() {
            return Some(("pag.scale", *scale));
        }
        if let Some(x) = start.filter(|x| !x.is_finite()) {
            return Some(("pag.start", x));
        }
        if let Some(x) = stop.filter(|x| !x.is_finite()) {
            return Some(("pag.stop", x));
        }
        if let Some(x) = adaptive_scale.filter(|x| !x.is_finite()) {
            return Some(("pag.adaptive_scale", x));
        }
        None
    }

    /// Capability-independent structural validation of `PagConfig` (finiteness, non-negative
    /// scales, `[0, 1]` window bounds, and non-empty explicit `layers` when `scale > 0`).
    ///
    /// Full validation against a model's `max_scale`, default window, and `0..num_blocks-1` block
    /// count runs through [`PagCapability::resolve`].
    pub fn validate(&self) -> Result<(), PagPolicyError> {
        if let Some((field, value)) = self.first_nonfinite_float() {
            return Err(PagPolicyError::invalid(format!(
                "{field} must be finite (got {value})"
            )));
        }
        if self.scale < 0.0 {
            return Err(PagPolicyError::invalid(format!(
                "pag.scale must be >= 0 (got {})",
                self.scale
            )));
        }
        if let Some(adaptive) = self.adaptive_scale {
            if adaptive < 0.0 {
                return Err(PagPolicyError::invalid(format!(
                    "pag.adaptive_scale must be >= 0 (got {adaptive})"
                )));
            }
        }
        if let Some(start) = self.start {
            if !(0.0..=1.0).contains(&start) {
                return Err(PagPolicyError::invalid(format!(
                    "pag.start {start} must be in 0..=1"
                )));
            }
        }
        if let Some(stop) = self.stop {
            if !(0.0..=1.0).contains(&stop) {
                return Err(PagPolicyError::invalid(format!(
                    "pag.stop {stop} must be in 0..=1"
                )));
            }
        }
        if let (Some(start), Some(stop)) = (self.start, self.stop) {
            if start > stop {
                return Err(PagPolicyError::invalid(format!(
                    "pag step window [{start}, {stop}] must satisfy 0 <= start <= stop <= 1"
                )));
            }
        }
        if self.scale > 0.0 && self.layers.as_ref().is_some_and(Vec::is_empty) {
            return Err(PagPolicyError::invalid(
                "pag.layers must not be empty when pag.scale > 0; omit `layers` to use model defaults or set `scale` to 0",
            ));
        }
        Ok(())
    }
}

/// Per-model Perturbed-Attention Guidance capability advertisement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagCapability {
    /// Total addressable self-attention blocks (`0..num_blocks`, i.e. valid indices
    /// `0..=num_blocks - 1`).
    pub num_blocks: u32,
    /// Human/UI-facing name of the block index space (e.g. `"layers"`, `"joint_blocks"`,
    /// `"single_blocks"`).
    pub block_domain: Cow<'static, str>,
    /// Curated default zero-based block indices used when [`PagConfig::layers`] is omitted.
    pub default_layers: Vec<u32>,
    /// Default normalized window start in `[0.0, 1.0]`.
    pub default_start: f32,
    /// Default normalized window stop in `[0.0, 1.0]`.
    pub default_stop: f32,
    /// Recommended non-zero scale for UI presets.
    pub recommended_scale: f32,
    /// Maximum supported [`PagConfig::scale`] (typically [`DEFAULT_PAG_MAX_SCALE`]).
    pub max_scale: f32,
}

impl PagCapability {
    /// Validate that the descriptor's own PAG capability metadata is self-consistent.
    pub fn validate_definition(&self) -> Result<(), PagPolicyError> {
        if self.num_blocks == 0 {
            return Err(PagPolicyError::invalid(
                "PagCapability.num_blocks must be >= 1",
            ));
        }
        if self.block_domain.trim().is_empty() {
            return Err(PagPolicyError::invalid(
                "PagCapability.block_domain must be non-empty",
            ));
        }
        if self.default_layers.is_empty() {
            return Err(PagPolicyError::invalid(
                "PagCapability.default_layers must be non-empty",
            ));
        }
        for (i, &layer) in self.default_layers.iter().enumerate() {
            if layer >= self.num_blocks {
                return Err(PagPolicyError::invalid(format!(
                    "PagCapability.default_layers index {layer} out of range 0..={} ({} has {} blocks)",
                    self.num_blocks - 1,
                    self.block_domain,
                    self.num_blocks
                )));
            }
            if self.default_layers[..i].contains(&layer) {
                return Err(PagPolicyError::invalid(format!(
                    "PagCapability.default_layers contains duplicate block index {layer}"
                )));
            }
        }
        if !self.default_start.is_finite()
            || !self.default_stop.is_finite()
            || self.default_start < 0.0
            || self.default_start >= self.default_stop
            || self.default_stop > 1.0
        {
            return Err(PagPolicyError::invalid(format!(
                "PagCapability default window [{}, {}] must be finite with 0 <= default_start < default_stop <= 1",
                self.default_start, self.default_stop
            )));
        }
        if !self.max_scale.is_finite() || self.max_scale <= 0.0 {
            return Err(PagPolicyError::invalid(format!(
                "PagCapability.max_scale must be finite and > 0 (got {})",
                self.max_scale
            )));
        }
        if !self.recommended_scale.is_finite()
            || self.recommended_scale <= 0.0
            || self.recommended_scale > self.max_scale
        {
            return Err(PagPolicyError::invalid(format!(
                "PagCapability.recommended_scale {} must be finite and in (0, {}]",
                self.recommended_scale, self.max_scale
            )));
        }
        Ok(())
    }

    /// Validate an optional [`PagConfig`] against this capability and resolve defaults, sorting
    /// and deduplicating block indices.
    ///
    /// When `config` is `None` or `scale == 0.0`, the returned [`PagResolved`] is a strict
    /// zero-pass no-op (`is_active() == false`, `step_scale(..) == 0.0`, `active_steps(..) == 0`).
    /// Explicit block indices `>= num_blocks` are always rejected, even when `scale == 0.0`.
    pub fn resolve(&self, config: Option<&PagConfig>) -> Result<PagResolved, PagPolicyError> {
        self.validate_definition()?;

        let Some(cfg) = config else {
            let mut layers = self.default_layers.clone();
            layers.sort_unstable();
            layers.dedup();
            return Ok(PagResolved {
                scale: 0.0,
                adaptive_scale: 0.0,
                layers,
                start: self.default_start,
                stop: self.default_stop,
            });
        };

        cfg.validate()?;

        if cfg.scale > self.max_scale {
            return Err(PagPolicyError::invalid(format!(
                "pag.scale {} exceeds maximum supported scale {}",
                cfg.scale, self.max_scale
            )));
        }

        let adaptive_scale = cfg.adaptive_scale.unwrap_or(0.0);
        let start = cfg.start.unwrap_or(self.default_start);
        let stop = cfg.stop.unwrap_or(self.default_stop);
        if start < 0.0 || stop > 1.0 || start > stop {
            return Err(PagPolicyError::invalid(format!(
                "pag step window [{start}, {stop}] must satisfy 0 <= start <= stop <= 1"
            )));
        }

        let mut layers = match &cfg.layers {
            Some(explicit) => {
                if let Some(&bad) = explicit.iter().find(|&&idx| idx >= self.num_blocks) {
                    return Err(PagPolicyError::invalid(format!(
                        "pag.layers block index {bad} out of range 0..={} ({} has {} blocks)",
                        self.num_blocks - 1,
                        self.block_domain,
                        self.num_blocks
                    )));
                }
                if explicit.is_empty() {
                    // `cfg.validate()` already refused empty `layers` when `cfg.scale > 0.0`;
                    // for `scale == 0.0` fall back to `default_layers`.
                    self.default_layers.clone()
                } else {
                    explicit.clone()
                }
            }
            None => self.default_layers.clone(),
        };
        layers.sort_unstable();
        layers.dedup();

        Ok(PagResolved {
            scale: cfg.scale,
            adaptive_scale,
            layers,
            start,
            stop,
        })
    }
}

/// Resolve an optional [`PagConfig`] against an optional model [`PagCapability`].
///
/// - `config == None` always succeeds as `Ok(None)` (zero-pass no-op), even on models without PAG.
/// - `config == Some(_)` when `capability == None` fails with [`PagPolicyErrorKind::Unsupported`].
/// - `config == Some(cfg)` with `cfg.scale == 0.0` validates all bounds/layers and resolves to
///   `Ok(None)` (strict zero-pass no-op).
/// - `config == Some(cfg)` with `cfg.scale > 0.0` validates and returns `Ok(Some(resolved))`.
pub fn resolve_pag(
    capability: Option<&PagCapability>,
    model_id: &str,
    config: Option<&PagConfig>,
) -> Result<Option<PagResolved>, PagPolicyError> {
    let Some(cfg) = config else {
        return Ok(None);
    };
    let Some(cap) = capability else {
        return Err(PagPolicyError::unsupported(format!(
            "{model_id}: Perturbed-Attention Guidance (PAG) is not supported by this model"
        )));
    };
    let resolved = cap.resolve(Some(cfg))?;
    if resolved.scale == 0.0 {
        Ok(None)
    } else {
        Ok(Some(resolved))
    }
}

/// Validated, defaulted Perturbed-Attention Guidance parameters ready for step-loop execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagResolved {
    pub scale: f32,
    pub adaptive_scale: f32,
    /// Sorted, deduplicated zero-based block indices in `0..num_blocks`.
    pub layers: Vec<u32>,
    pub start: f32,
    pub stop: f32,
}

impl PagResolved {
    /// Whether PAG can be active on at least one step (`scale > 0`, non-empty window, non-empty
    /// layers). When `false`, zero perturbed forward passes may execute.
    pub fn is_active(&self) -> bool {
        self.scale > 0.0 && self.start < self.stop && !self.layers.is_empty()
    }

    /// Effective PAG scale at 0-based `step_index` out of `total_steps`, aligned with MOSS SLG's
    /// normalized `[start, stop)` step-fraction convention.
    ///
    /// Returns `0.0` whenever PAG is inactive, `total_steps == 0`, `step_index >= total_steps`,
    /// `progress < start`, or `progress >= stop`. Otherwise returns
    /// `(scale - adaptive_scale * progress).max(0.0)` with
    /// `progress = step_index as f32 / total_steps.max(1) as f32`.
    pub fn step_scale(&self, step_index: usize, total_steps: usize) -> f32 {
        if !self.is_active() || total_steps == 0 || step_index >= total_steps {
            return 0.0;
        }
        let progress = step_index as f32 / total_steps.max(1) as f32;
        if !(self.start <= progress && progress < self.stop) {
            return 0.0;
        }
        (self.scale - self.adaptive_scale * progress).max(0.0)
    }

    /// Block indices perturbed at `(step_index, total_steps)`; empty slice when `step_scale == 0`.
    pub fn step_layers(&self, step_index: usize, total_steps: usize) -> &[u32] {
        if self.step_scale(step_index, total_steps) > 0.0 {
            &self.layers
        } else {
            &[]
        }
    }

    /// Whether `block_index` has identity self-attention perturbation at `(step_index, total_steps)`.
    pub fn perturbs_block(&self, block_index: u32, step_index: usize, total_steps: usize) -> bool {
        self.step_scale(step_index, total_steps) > 0.0
            && self.layers.binary_search(&block_index).is_ok()
    }

    /// Number of denoise steps in `0..total_steps` where `step_scale > 0.0` (i.e. where a
    /// perturbed forward pass actually executes).
    pub fn active_steps(&self, total_steps: u32) -> u32 {
        let total = total_steps as usize;
        (0..total)
            .filter(|&step| self.step_scale(step, total) > 0.0)
            .count() as u32
    }

    /// Build the execution report for a completed trajectory of `total_steps` denoise steps.
    pub fn execution_report(&self, total_steps: u32) -> PagExecutionReport {
        PagExecutionReport {
            layers: self.layers.clone(),
            active_steps: self.active_steps(total_steps),
            total_steps,
            scale: self.scale,
            adaptive_scale: self.adaptive_scale,
            start: self.start,
            stop: self.stop,
        }
    }

    /// Convert back into an explicit [`PagConfig`] carrying all resolved values.
    pub fn to_config(&self) -> PagConfig {
        PagConfig {
            scale: self.scale,
            layers: Some(self.layers.clone()),
            start: Some(self.start),
            stop: Some(self.stop),
            adaptive_scale: Some(self.adaptive_scale),
        }
    }
}

/// Completed Perturbed-Attention Guidance execution parameters recorded on generation reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagExecutionReport {
    /// Sorted, deduplicated zero-based block indices configured for perturbation.
    pub layers: Vec<u32>,
    /// Number of denoise steps where a perturbed pass actually executed (`step_scale > 0`).
    pub active_steps: u32,
    /// Total denoise steps in the executed trajectory.
    pub total_steps: u32,
    /// Resolved base PAG scale.
    pub scale: f32,
    /// Resolved linear adaptive decay slope (`0.0` when disabled).
    pub adaptive_scale: f32,
    /// Resolved normalized window start in `[0.0, 1.0]`.
    pub start: f32,
    /// Resolved normalized window stop in `[0.0, 1.0]`.
    pub stop: f32,
}

impl PagExecutionReport {
    /// Whether at least one perturbed forward pass ran during the trajectory.
    pub fn is_active(&self) -> bool {
        self.active_steps > 0
    }

    /// Validate that this execution report represents a canonical [`PagResolved`] outcome for
    /// `capability` and `self.total_steps`.
    ///
    /// Zero-scale PAG no-ops are canonically represented only as `None` on outer execution
    /// reports, so `self.scale` must be strictly positive.
    pub fn validate_with_capability(
        &self,
        capability: &PagCapability,
    ) -> Result<(), PagPolicyError> {
        let cfg = PagConfig {
            scale: self.scale,
            layers: Some(self.layers.clone()),
            start: Some(self.start),
            stop: Some(self.stop),
            adaptive_scale: Some(self.adaptive_scale),
        };
        let resolved = capability.resolve(Some(&cfg))?;
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(PagPolicyError::invalid(format!(
                "PagExecutionReport.scale must be > 0 (got {}); represent zero-scale PAG no-ops as None",
                self.scale
            )));
        }
        if resolved.layers != self.layers {
            return Err(PagPolicyError::invalid(
                "PagExecutionReport.layers must be sorted, deduplicated, and non-empty",
            ));
        }
        let expected_active = resolved.active_steps(self.total_steps);
        if self.active_steps != expected_active {
            return Err(PagPolicyError::invalid(format!(
                "PagExecutionReport.active_steps {} disagrees with resolved window ({expected_active} of {} steps)",
                self.active_steps, self.total_steps
            )));
        }
        Ok(())
    }
}

/// Pure FP32 slice reference for Perturbed-Attention Guidance combination:
/// `eps_base + pag_scale * (eps_cond - eps_perturbed)`.
///
/// When `pag_scale == 0.0`, returns `eps_base.to_vec()` after validating slice lengths and
/// finiteness.
pub fn pag_apply_slice(
    eps_base: &[f32],
    eps_cond: &[f32],
    eps_perturbed: &[f32],
    pag_scale: f32,
) -> Result<Vec<f32>, PagPolicyError> {
    if !pag_scale.is_finite() || pag_scale < 0.0 {
        return Err(PagPolicyError::invalid(format!(
            "pag_scale must be finite and >= 0 (got {pag_scale})"
        )));
    }
    if eps_base.len() != eps_cond.len() || eps_cond.len() != eps_perturbed.len() {
        return Err(PagPolicyError::invalid(format!(
            "PAG branch length mismatch: base={} cond={} perturbed={}",
            eps_base.len(),
            eps_cond.len(),
            eps_perturbed.len()
        )));
    }
    if !eps_base
        .iter()
        .chain(eps_cond)
        .chain(eps_perturbed)
        .all(|v| v.is_finite())
    {
        return Err(PagPolicyError::invalid(
            "PAG prediction branches must contain finite FP32 values",
        ));
    }
    if pag_scale == 0.0 {
        return Ok(eps_base.to_vec());
    }
    let out: Vec<f32> = eps_base
        .iter()
        .zip(eps_cond)
        .zip(eps_perturbed)
        .map(|((&b, &c), &p)| b + pag_scale * (c - p))
        .collect();
    if !out.iter().all(|v| v.is_finite()) {
        return Err(PagPolicyError::invalid(
            "PAG guided prediction overflowed to non-finite FP32",
        ));
    }
    Ok(out)
}
