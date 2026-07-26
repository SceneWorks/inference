//! Mage-Flow adapter consumption (sc-14055): install a trained (or imported) LoRA/LoKr adapter onto
//! the NR-MMDiT for inference.
//!
//! The model-specific piece is the key→module map — the top-level [`AdaptableHost`] for
//! [`MageTransformer`](crate::transformer::MageTransformer) plus the per-module hosts on the joint
//! attention, FFN, and block, all in the DiT source. Everything else — per-file LoKr/LoRA dispatch,
//! LoRA-prefix detection, stacking + mixed, and the strict no-silent-drop policy — is the shared
//! core seam ([`mlx_gen::adapters::loader`]).
//!
//! A trainer writes `{path}.lora_A.weight` / `.lora_B.weight` / `.alpha` (or the LoKr
//! `{path}.lokr_w1` + factors) at exactly the [`AdaptableHost::adaptable_paths`] the reload
//! resolves against, so a Mage adapter trained here reloads bit-for-bit.
//!
//! **Community adapters (sc-14057).** The host exposes **every** DiT `Linear`, including the
//! globals our own default target set never touches
//! (`time_text_embed.timestep_embedder.linear_{1,2}`, `norm_out.linear`, `proj_out`) — see the
//! [`AdaptableHost`] impl on [`MageTransformer`](crate::transformer::MageTransformer). A PEFT
//! `target_modules="all-linear"` export names all of them; before sc-14057 they resolved to
//! nothing and [`apply_adapters_strict`] failed the whole file (loud, never silent — but the
//! adapter was unusable). Enumerating them also makes them kohya-reachable, since the shared
//! `flattened → dotted` table is built from that list. The loader supplies the spelling coverage
//! on top: PEFT/diffusers dotted, kohya `lora_unet_`, LoKr, and ComfyUI `.diff`/`.diff_b`
//! diff-patches. Mage has no fused BFL surface (its q/k/v projections are already split), so
//! `bfl_targets` stays empty and a BFL-fused file surfaces its keys as unmatched rather than
//! being silently dropped. Covered by `tests/adapter_routing.rs`.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a Mage-Flow NR-MMDiT `host` (stacked, mixed LoRA/LoKr), via
/// the shared strict installer — it errors, never silently drops, on an unmatched target.
pub fn apply_mage_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "mage_flow")
}
