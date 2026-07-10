//! Anima LoRA/LoKr adapter consumption (sc-10521). The model-specific piece is the key→module map:
//! the trained files carry BOTH the Cosmos DiT (`diffusion_model.blocks.*` + globals) AND the bundled
//! `AnimaTextConditioner` (`diffusion_model.llm_adapter.blocks.*`) under ComfyUI's `diffusion_model.`
//! prefix. Everything else — per-file LoKr/LoRA dispatch, `diffusion_model.`/`transformer.` prefix
//! detection, stacking + mixed LoRA/LoKr, and the strict no-silent-drop policy — is the shared core
//! seam ([`apply_adapters_strict`], sc-2534).
//!
//! **The verified trap (sc-10274 class):** the `anima-turbo-lora-v0.2` file is 508 target pairs =
//! 448 DiT (`blocks.*`) + **60 `llm_adapter.*`**, while `anima-greg-rutkowski-style` is 448 DiT-only,
//! zero adapter. The conditioner is therefore a first-class injectable target: this host strips the
//! leading `llm_adapter.` segment and routes into the [`AnimaTextConditioner`] host; everything else
//! routes into the [`CosmosDiT`] host. Because the install is `apply_adapters_strict`, an unrouted
//! `llm_adapter.*` target is a hard error, not a silent partial — the count is proven, not assumed.
//!
//! One nuance the count hides: for `anima-turbo-lora-v0.2` specifically, all 60 conditioner `lora_B`
//! are **zero-initialized** (untrained), so `B·A ≡ 0` and dropping them would be numerically inert for
//! *this* file. The guard is about the MECHANISM, not this one file's magnitudes: a future non-zero
//! conditioner LoRA — and the already-shipped `anima-rl-v0.1`, which also carries 60 `llm_adapter.*`
//! targets — must not silently load at partial strength. Enforcing routing by count, independent of the
//! trained delta, is what keeps the sc-10274 "loads partial, looks fine" class un-repeatable here.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

use crate::conditioner::AnimaTextConditioner;
use crate::transformer::CosmosDiT;

/// A mutable borrow of the two injectable Anima sub-models, presented to the core adapter loader as a
/// single [`AdaptableHost`]. `llm_adapter.*` keys route into the conditioner; all others into the DiT.
pub struct AnimaAdapterHost<'a> {
    pub dit: &'a mut CosmosDiT,
    pub conditioner: &'a mut AnimaTextConditioner,
}

impl AdaptableHost for AnimaAdapterHost<'_> {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            // The bundled conditioner is addressed under `llm_adapter.` (the Cosmos checkpoint's own
            // sub-module name) — strip it and route into the conditioner host.
            ["llm_adapter", rest @ ..] => self.conditioner.adaptable_mut(rest),
            // Everything else is a Cosmos DiT target (`blocks.*` + globals).
            _ => self.dit.adaptable_mut(path),
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = self.dit.adaptable_paths();
        out.extend(prefixed_paths("llm_adapter", self.conditioner));
        out
    }
}

/// Apply every adapter in `specs` onto an Anima model (DiT + conditioner), stacked and mixed
/// LoRA/LoKr, via the core [`apply_adapters_strict`]. Errors — never silently drops — on an unmatched
/// target, so a DiT-only regression that skips the 60 `llm_adapter.*` targets surfaces as a hard
/// failure rather than a partially-loaded distillation (sc-10274). Returns the [`ApplyReport`] so a
/// caller can assert the injected-target count (508 for the turbo LoRA, 448 for the DiT-only style
/// LoRA).
pub fn apply_anima_adapters(
    dit: &mut CosmosDiT,
    conditioner: &mut AnimaTextConditioner,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    let mut host = AnimaAdapterHost { dit, conditioner };
    apply_adapters_strict(&mut host, specs, "anima")
}
