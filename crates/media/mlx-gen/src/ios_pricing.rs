//! Header-only, per-phase COUNTED pricing of one iOS request: the architecture facts the iOS
//! anchor law (epic 16774, `mlx-gen-ios-catalog::anchor`) scales a measured device anchor by.
//!
//! Every iOS provider crate exposes `ios_pricing(&IosPricingRequest) -> Result<IosPricing>` built
//! from the SAME `ios_quantum` formulas its per-quantum guards price (component load floors,
//! resident component bytes under the request's regime, and each phase's largest counted quantum
//! scratch), and `ios_quantum_phase(label) -> Option<IosPhase>`, which attributes each of its
//! quantum labels to the phase it runs in (a device capture splits its kernel lifetime peak by
//! phase through those labels). Pricing reads safetensors headers and small configs only: it never
//! builds a graph or reads a tensor payload, so it runs on device at admission time.
//!
//! Prices are NOT admission. Admission is a measured device anchor plus the anchor law; a price
//! only says how a measured residue scales between the anchor's regime/geometry and a request's.

use crate::gen_core::{GenerationMemory, LoadSpec, MemoryGeometry, MemoryMode, MemorySelection};

/// The execution phases a request's memory peak is attributed to. Load floors are their own phase
/// (a load's peak is the component's stored bytes plus its read scratch, not an activation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IosPhase {
    /// A component load floor (`ios_quantum::before_load*`), whichever component it is.
    Load,
    /// Text / image-prompt conditioning (text encoders, CLIP vision, conditioners).
    Conditioning,
    /// An image ENCODE: an img2img / inpaint init latent, a control hint, a reference image.
    Encode,
    /// The denoiser (DiT / U-Net / ControlNet blocks, sampler steps).
    Denoise,
    /// The decode to pixels (VAE, tile assembly, RGB conversion), and post-processing tiles.
    Decode,
}

impl IosPhase {
    pub const ALL: [IosPhase; 5] = [
        IosPhase::Load,
        IosPhase::Conditioning,
        IosPhase::Encode,
        IosPhase::Denoise,
        IosPhase::Decode,
    ];

    /// Stable key used in evidence rows.
    pub fn as_key(self) -> &'static str {
        match self {
            IosPhase::Load => "load",
            IosPhase::Conditioning => "conditioning",
            IosPhase::Encode => "encode",
            IosPhase::Denoise => "denoise",
            IosPhase::Decode => "decode",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.as_key() == key)
    }
}

/// The counted price of ONE phase: the component bytes resident in it under the request's regime,
/// plus its LARGEST counted quantum (live input + scratch, the same bytes its guard checks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IosPhasePrice {
    pub phase: IosPhase,
    /// Component (weight) bytes resident throughout this phase under the request's regime.
    pub resident_bytes: u64,
    /// The largest counted quantum of the phase (its guard's priced scratch), bytes.
    pub scratch_bytes: u64,
    /// The quantum label that binds `scratch_bytes` (the anchor law calibrates a measured residue
    /// against a counted formula only where the request's binding quantum is the anchor's).
    pub binding: String,
}

impl IosPhasePrice {
    pub fn peak_bytes(&self) -> u64 {
        self.resident_bytes.saturating_add(self.scratch_bytes)
    }
}

/// One request's counted price, one entry per phase the request runs (a phase it does not run is
/// absent — e.g. no `Encode` for text-to-image).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IosPricing {
    pub phases: Vec<IosPhasePrice>,
}

impl IosPricing {
    /// The phase's price, if the request runs it.
    pub fn phase(&self, phase: IosPhase) -> Option<&IosPhasePrice> {
        self.phases.iter().find(|p| p.phase == phase)
    }

    /// The whole-run counted peak: phases run in sequence, so the worst phase.
    pub fn peak_bytes(&self) -> u64 {
        self.phases
            .iter()
            .map(IosPhasePrice::peak_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Fold one quantum into its phase, keeping the phase's worst (largest resident + scratch)
    /// quantum. Providers build their pricing with this so a phase is always its worst quantum.
    pub fn take(&mut self, phase: IosPhase, resident_bytes: u64, scratch_bytes: u64, label: &str) {
        let candidate = IosPhasePrice {
            phase,
            resident_bytes,
            scratch_bytes,
            binding: label.to_string(),
        };
        match self.phases.iter_mut().find(|p| p.phase == phase) {
            Some(existing) if existing.peak_bytes() >= candidate.peak_bytes() => {}
            Some(existing) => *existing = candidate,
            None => {
                self.phases.push(candidate);
                self.phases.sort_by_key(|p| p.phase);
            }
        }
    }
}

/// What a provider prices: the request's load (tier, components), route, geometry, the selected
/// ladder parameters and the execution levers actually written onto the request, and the encoded
/// text window. Exactly the memory-relevant facts of the iOS evidence key.
#[derive(Clone, Copy, Debug)]
pub struct IosPricingRequest<'a> {
    pub spec: &'a LoadSpec,
    pub mode: &'a MemoryMode,
    pub geometry: MemoryGeometry,
    pub selection: &'a MemorySelection,
    pub execution: &'a GenerationMemory,
    /// Encoded text window in tokens where it is request-dependent (SDXL CLIP windows x 77); 0
    /// where the provider pads to a fixed, artifact-determined window.
    pub token_window: u32,
}

/// A provider's pricing entry point.
pub type IosPricer = fn(&IosPricingRequest<'_>) -> crate::Result<IosPricing>;
/// A provider's quantum-label to phase attribution.
pub type IosQuantumPhase = fn(&str) -> Option<IosPhase>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phase_keeps_its_worst_quantum_and_the_run_its_worst_phase() {
        let mut p = IosPricing::default();
        p.take(IosPhase::Decode, 100, 10, "a");
        p.take(IosPhase::Decode, 100, 50, "b");
        p.take(IosPhase::Decode, 100, 20, "c");
        p.take(IosPhase::Denoise, 300, 1, "d");
        assert_eq!(p.phase(IosPhase::Decode).unwrap().binding, "b");
        assert_eq!(p.peak_bytes(), 301);
        assert_eq!(
            p.phases.iter().map(|p| p.phase).collect::<Vec<_>>(),
            [IosPhase::Denoise, IosPhase::Decode]
        );
        assert!(p.phase(IosPhase::Encode).is_none());
        for phase in IosPhase::ALL {
            assert_eq!(IosPhase::from_key(phase.as_key()), Some(phase));
        }
    }
}
