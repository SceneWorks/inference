//! Adapter specs, apply reports, and adapter kinds.

use super::*;

/// A single adapter to stack at load time. Multiples + mixed LoRA/LoKr are supported by
/// construction — see the provider `adapters` modules. Carried by [`LoadSpec::adapters`].
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct AdapterSpec {
    pub path: PathBuf,
    pub scale: f32,
    pub kind: AdapterKind,
    /// Per-denoise-pass strength override (LTX-2.3 only). When `Some`, the slice gives this
    /// adapter's strength for each distilled stage (LTX runs a 2-stage denoise, so a length-2
    /// `[stage1, stage2]`); when `None`, [`scale`](Self::scale) is applied uniformly to every pass.
    /// This is the LTX "per-pass strength" feature (sc-2687) — the reference has no per-stage
    /// schedule, so it is net-new. Like [`LoadSpec::control`], it is a model-specific knob on the
    /// shared spec: **only LTX reads it**; every other model ignores it (its denoise is single-pass).
    pub pass_scales: Option<Vec<f32>>,
    /// Which expert of a dual-expert MoE model (Wan2.2 A14B) this adapter targets (sc-2683).
    /// `None` = shared: merged onto **both** the high- and low-noise experts (the reference
    /// `--lora` file → `(loras)+(loras_high/low)`); `Some(High)`/`Some(Low)` = one expert only
    /// (`--lora-high` / `--lora-low`). Like [`pass_scales`](Self::pass_scales), this is a
    /// model-specific knob on the shared spec: **only the Wan MoE models read it**; every
    /// single-stream model ignores it (a `Some(_)` there is surfaced, not silently honored).
    pub moe_expert: Option<MoeExpert>,
}

/// One adapter file's actual provider-side install outcome.
///
/// Providers that can partially accept an adapter publish these reports inside
/// [`GenerationReport::adapter_apply_reports`](crate::GenerationReport::adapter_apply_reports)
/// returned by [`Generator::generate_with_report`](crate::Generator::generate_with_report) and
/// [`Generator::generate_with_execution_report`](crate::Generator::generate_with_execution_report).
/// The report is deliberately tensor-free and additive to the existing generator contract: providers
/// that do not opt in return an empty list, preserving their current behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterApplyReport {
    /// The adapter file the report describes. This is the resolved local path from
    /// [`AdapterSpec::path`], allowing a consumer to correlate reports with load-order specs without
    /// guessing from file headers.
    pub adapter_path: PathBuf,
    /// Target deltas/factors the provider accepted for this adapter.
    pub applied: usize,
    /// Target stems the provider did not apply. Empty means the provider accepted the whole adapter
    /// surface it recognized.
    pub skipped: Vec<String>,
}

impl AdapterSpec {
    /// A uniform-strength adapter (the common case): [`scale`](Self::scale) on every denoise pass,
    /// no per-pass override, shared across both MoE experts. Equivalent to a literal with
    /// `pass_scales: None, moe_expert: None`.
    pub fn new(path: PathBuf, scale: f32, kind: AdapterKind) -> Self {
        Self {
            path,
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }
    }

    /// Builder-style per-pass strength override (LTX only — see [`pass_scales`](Self::pass_scales)).
    pub fn with_pass_scales(mut self, pass_scales: Vec<f32>) -> Self {
        self.pass_scales = Some(pass_scales);
        self
    }

    /// Builder-style MoE expert target (Wan2.2 A14B only — see [`moe_expert`](Self::moe_expert)).
    pub fn with_moe_expert(mut self, expert: MoeExpert) -> Self {
        self.moe_expert = Some(expert);
        self
    }

    /// Shared adapters apply to either expert; tagged adapters apply only to their target.
    pub fn applies_to_expert(&self, expert: MoeExpert) -> bool {
        self.moe_expert.is_none() || self.moe_expert == Some(expert)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum AdapterKind {
    Lora,
    Lokr,
}

/// One expert of a dual-expert MoE denoiser (Wan2.2 A14B), naming which checkpoint an adapter
/// merges onto. The A14B splits denoising at a noise `boundary` between a **high**-noise expert
/// (early, noisy steps) and a **low**-noise expert (late steps); a trained Wan MoE LoRA ships as a
/// high/low pair (e.g. `*_wan22_high` + `*_wan22_low`). See [`AdapterSpec::moe_expert`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum MoeExpert {
    High,
    Low,
}

/// The adapter application order declared by an expert route. Keep this independent of the
/// adapter format: reordering a fold can change its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertAdapterOrder {
    /// Keep the original request order, with shared and expert-specific adapters interleaved.
    Request,
    /// Apply the shared stack first, then this expert's stack, keeping order within each group.
    SharedFirst,
}

/// Select an expert's adapters without changing strengths or replacing their request-slot
/// identities. The original slots let install reports correlate with the request even when two
/// entries name the same file or a route changes their application order.
pub fn select_expert_adapters(
    adapters: &[AdapterSpec],
    expert: MoeExpert,
    order: ExpertAdapterOrder,
) -> impl Iterator<Item = (usize, &AdapterSpec)> {
    let indexed = adapters.iter().enumerate();
    let request_order = order == ExpertAdapterOrder::Request;
    indexed
        .clone()
        .filter(move |(_, spec)| {
            if request_order {
                spec.applies_to_expert(expert)
            } else {
                spec.moe_expert.is_none()
            }
        })
        .chain(indexed.filter(move |(_, spec)| !request_order && spec.moe_expert == Some(expert)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_adapter_selection_preserves_slots_strengths_and_route_order() {
        let specs = [
            AdapterSpec::new("high.safetensors".into(), 0.25, AdapterKind::Lokr)
                .with_moe_expert(MoeExpert::High),
            AdapterSpec::new("shared.safetensors".into(), 0.5, AdapterKind::Lora),
            AdapterSpec::new("low.safetensors".into(), 0.75, AdapterKind::Lora)
                .with_moe_expert(MoeExpert::Low),
            AdapterSpec::new("shared.safetensors".into(), -0.0, AdapterKind::Lora),
        ];

        for (expert, order, expected) in [
            (MoeExpert::High, ExpertAdapterOrder::Request, [0, 1, 3]),
            (MoeExpert::Low, ExpertAdapterOrder::Request, [1, 2, 3]),
            (MoeExpert::High, ExpertAdapterOrder::SharedFirst, [1, 3, 0]),
            (MoeExpert::Low, ExpertAdapterOrder::SharedFirst, [1, 3, 2]),
        ] {
            let selected: Vec<_> = select_expert_adapters(&specs, expert, order).collect();
            assert_eq!(
                selected.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(),
                expected
            );
            for (slot, spec) in selected {
                assert!(std::ptr::eq(spec, &specs[slot]));
            }
        }
    }
}
