//! Tensor-free arguments shared by denoise phase preparation and forward evaluation.

use std::collections::{hash_map::Entry, HashMap};

use crate::MemoryGeometry;

/// Overrides for one projection's installed adapter order and one denoise pass.
///
/// These indices are local to the projection. For a request-wide phase selection, bind each
/// installed residual to the original request slot using [`PhaseAdapterScale`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScaleContext<'a> {
    /// Optional per-adapter scale overrides indexed by adapter position.
    pub adapter_scales: Option<&'a [f32]>,
    /// Optional uniform scale multiplier applied across all adapters (defaults to `1.0` when `None`).
    pub uniform_scale: Option<f32>,
    /// Denoise pass index, such as the base or distilled refinement pass.
    pub pass: usize,
}

impl<'a> ScaleContext<'a> {
    /// Default pass-0 context with no scale overrides.
    pub const fn new() -> Self {
        Self {
            adapter_scales: None,
            uniform_scale: None,
            pass: 0,
        }
    }

    /// Override per-adapter scales by position.
    pub const fn with_adapter_scales(adapter_scales: &'a [f32]) -> Self {
        Self {
            adapter_scales: Some(adapter_scales),
            uniform_scale: None,
            pass: 0,
        }
    }

    /// Apply a uniform scale multiplier across all installed adapters for this request/pass.
    pub const fn with_uniform_scale(scale: f32) -> Self {
        Self {
            adapter_scales: None,
            uniform_scale: Some(scale),
            pass: 0,
        }
    }

    /// Attach a pass index to this scale context.
    pub const fn with_pass(mut self, pass: usize) -> Self {
        self.pass = pass;
        self
    }

    /// Resolve the effective `f32` scale for adapter at `index` with installed `default_scale`.
    #[inline]
    pub fn scale_for(&self, index: usize, default_scale: f32) -> f32 {
        let base = self
            .adapter_scales
            .and_then(|s| s.get(index).copied())
            .unwrap_or(default_scale);
        match self.uniform_scale {
            Some(mult) => base * mult,
            None => base,
        }
    }

    /// Resolve the effective `f64` scale for adapter at `index` with installed `default_scale`.
    #[inline]
    pub fn scale_for_f64(&self, index: usize, default_scale: f64) -> f64 {
        let base = self
            .adapter_scales
            .and_then(|s| s.get(index).copied().map(f64::from))
            .unwrap_or(default_scale);
        match self.uniform_scale {
            Some(mult) => base * f64::from(mult),
            None => base,
        }
    }
}

/// Effective strength for one slot in the request's installed stack. A phase's list is ordered
/// and can reference the same slot more than once. Individual projections can have sparse
/// installations, so the slot is not a projection-local residual index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhaseAdapterScale {
    slot: usize,
    scale_bits: u32,
}

impl PhaseAdapterScale {
    pub const fn new(slot: usize, scale: f32) -> Self {
        Self {
            slot,
            scale_bits: scale.to_bits(),
        }
    }

    pub const fn slot(self) -> usize {
        self.slot
    }

    pub const fn scale(self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
}

/// The text conditioning branches prepared for this phase. Routes declare their native layout
/// independently of the guidance number (for example, an embedded guidance input needs no
/// unconditional preparation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConditioningBranches {
    Positive,
    PositiveAndNegative,
}

/// Immutable arguments of one phase. Preparation and forward read the same ordered, effective
/// residual stack and branch selection. An empty stack means the phase uses the bare base.
#[derive(Clone, Debug)]
pub struct PhaseArguments {
    adapters: Vec<PhaseAdapterScale>,
    guidance: f32,
    branches: ConditioningBranches,
}

impl PhaseArguments {
    pub fn new(
        adapters: Vec<PhaseAdapterScale>,
        guidance: f32,
        branches: ConditioningBranches,
    ) -> Self {
        Self {
            adapters,
            guidance,
            branches,
        }
    }

    pub fn adapters(&self) -> &[PhaseAdapterScale] {
        &self.adapters
    }

    pub fn guidance(&self) -> f32 {
        self.guidance
    }

    pub fn uses_classifier_free_guidance(&self) -> bool {
        self.branches == ConditioningBranches::PositiveAndNegative
    }
}

/// Exactly the phase dependencies available to preparation. Guidance magnitude and any
/// per-evaluation perturbation are forward-only inputs.
pub struct PrepareArguments<'a> {
    adapters: &'a [PhaseAdapterScale],
    branches: ConditioningBranches,
}

impl<'a> PrepareArguments<'a> {
    pub fn adapters(&self) -> &'a [PhaseAdapterScale] {
        self.adapters
    }

    pub fn uses_classifier_free_guidance(&self) -> bool {
        self.branches == ConditioningBranches::PositiveAndNegative
    }
}

/// One phase's forward arguments and the conditioning prepared with them.
pub struct PreparedPhase<T> {
    arguments: PhaseArguments,
    prepared: T,
}

impl<T> PreparedPhase<T> {
    pub fn parts(&self) -> (&PhaseArguments, &T) {
        (&self.arguments, &self.prepared)
    }
}

#[derive(PartialEq, Eq, Hash)]
struct PreparationKey {
    adapters: Vec<PhaseAdapterScale>,
    geometry: MemoryGeometry,
    branches: ConditioningBranches,
}

/// Prepare a request's phases with one fixed effective model, installed adapter stack and set of
/// encoded conditioning inputs, captured by `prepare`. Use a separate call when any of those
/// dependencies changes. The per-call cache keys also include the phase's geometry, ordered
/// effective scales and prepared branches. Guidance magnitude changes how the same branches are
/// combined, so only the branch layout affects preparation. A perturbed-attention evaluation can
/// use the positive preparation if its route leaves conditioning unchanged.
///
/// This is for drivers that already reserve and retain preparation across their phase plans.
/// Prepared values must clone by sharing immutable allocations. Equal dependencies clone from
/// the first phase's preparation, and the lookup only stores indices into the returned plans.
pub fn prepare_phases<T: Clone, E>(
    phases: impl IntoIterator<Item = (PhaseArguments, MemoryGeometry)>,
    mut prepare: impl FnMut(PrepareArguments<'_>, MemoryGeometry) -> Result<T, E>,
) -> Result<Vec<PreparedPhase<T>>, E> {
    let mut cache: HashMap<_, usize> = HashMap::new();
    let mut plans: Vec<PreparedPhase<T>> = Vec::new();
    for (arguments, geometry) in phases {
        let key = PreparationKey {
            adapters: arguments.adapters.clone(),
            geometry,
            branches: arguments.branches,
        };
        let prepared = match cache.entry(key) {
            Entry::Occupied(entry) => plans[*entry.get()].prepared.clone(),
            Entry::Vacant(entry) => {
                let value = prepare(
                    PrepareArguments {
                        adapters: arguments.adapters(),
                        branches: arguments.branches,
                    },
                    geometry,
                )?;
                entry.insert(plans.len());
                value
            }
        };
        plans.push(PreparedPhase {
            arguments,
            prepared,
        });
    }
    Ok(plans)
}

#[cfg(test)]
mod tests;
