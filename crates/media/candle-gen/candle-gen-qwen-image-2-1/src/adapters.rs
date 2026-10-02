//! LoRA / LoKr / LoHa on the Qwen-Image 2.1 DiT (sc-24157) — the candle twin of the MLX host's
//! adapter consumption.
//!
//! Two routes, chosen per adapter file from the file's **own** keys and metadata:
//!
//! * **LoRA and PEFT-stamped LoKr** ride as forward-time additive residuals through the shared
//!   [`candle_gen::quant::install_dotted_adapters`] over
//!   [`QwenImage21Transformer::visit_adaptable_mut`] — the FLUX.2 pattern. The base weight is never
//!   touched, so the same install serves the dense bf16 tier and the packed q8/q4 tiers at the base's
//!   footprint. Adapters stack in request order, each at its own `AdapterSpec::scale`.
//! * **LyCORIS LoHa** (`hada_w1_a/b`, `hada_w2_a/b`) has no deferred additive form — the Hadamard
//!   product of two low-rank pairs is not a low-rank matrix — so it is **folded into the dense
//!   weight** (`W ← W + δ`), the `candle-gen-qwen-image` precedent. That needs a dense weight, so it
//!   is served on the **bf16 tier only**: on a packed q8/q4 tier LoHa is refused with a typed
//!   `Unsupported` naming LoHa and the tier ([`loha_on_packed_tier_refusal`]) — never silently
//!   skipped and never served by dequantizing the base.
//!
//! **Strict.** Every target in every selected file must reach a DiT projection: an adapter key that
//! resolves to no projection (another model's module tree, a text-encoder LoRA, a mis-shaped factor)
//! is a load error naming the file and the key, not a silently un-adapted render.

use candle_core::Device;
use candle_gen::gen_core::weightsmeta as wmeta;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::merge::{parse_loha_thirdparty, read_adapter, AdapterFile};
use candle_gen::{CandleError as Error, Result};

use crate::quant::Tier;
use crate::transformer::QwenImage21Transformer;
use crate::MODEL_ID;

/// What one install applied.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterInstallReport {
    /// LoRA / LoKr residuals attached (one per `(projection, file)` hit; stacked files count once
    /// each).
    pub residuals: usize,
    /// LoHa deltas folded into dense projection weights.
    pub loha_folds: usize,
}

/// The typed refusal for a LoHa adapter on a packed tier. Names LoHa, the tier and the way out.
pub fn loha_on_packed_tier_refusal(tier: Tier, adapter: &std::path::Path) -> String {
    format!(
        "{MODEL_ID}: LoHa adapter {} cannot apply on the packed {} tier — LoHa has no additive \
         form over quantized weights and is merged into dense weights only; load the bf16 tier to \
         use it",
        adapter.display(),
        tier.dir_name()
    )
}

/// Whether `file` is a LyCORIS LoHa (keyed by its own tensors, never by the declared kind).
fn is_loha(file: &AdapterFile) -> bool {
    wmeta::keys_contain_loha(file.tensors.keys().map(String::as_str))
}

fn check_spec(spec: &AdapterSpec) -> Result<()> {
    if !spec.scale.is_finite() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: adapter {} scale must be finite, got {}",
            spec.path.display(),
            spec.scale
        )));
    }
    if let Some(expert) = spec.moe_expert {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: adapter {} targets the {expert:?} MoE expert, but this model has a single \
             denoiser",
            spec.path.display()
        )));
    }
    Ok(())
}

/// Weight-free admission: reads each selected adapter file and refuses — before any weight loads —
/// a LoHa on a packed tier and a LoHa declared as LoKr. Everything else is decided at install.
pub fn preflight(specs: &[AdapterSpec], tier: Tier) -> Result<()> {
    for spec in specs {
        check_spec(spec)?;
        let file = read_adapter(&spec.path)?;
        if is_loha(&file) {
            admit_loha(spec, tier)?;
        }
    }
    Ok(())
}

fn admit_loha(spec: &AdapterSpec, tier: Tier) -> Result<()> {
    if spec.kind == AdapterKind::Lokr {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: adapter {} was declared LoKr but its keys are LyCORIS LoHa (hada_*)",
            spec.path.display()
        )));
    }
    if tier != Tier::Bf16 {
        return Err(Error::Unsupported(loha_on_packed_tier_refusal(
            tier, &spec.path,
        )));
    }
    Ok(())
}

/// The dotted / kohya-flattened spellings an adapter key may use for the projection at `path`.
fn key_candidates(path: &str) -> [String; 2] {
    [
        path.to_string(),
        format!("lora_unet_{}", path.replace('.', "_")),
    ]
}

/// Strip a leading `transformer.` / `diffusion_model.` namespace.
fn strip_namespace(key: &str) -> &str {
    for prefix in wmeta::COMMON_LORA_PREFIXES {
        if let Some(rest) = key.strip_prefix(prefix) {
            return rest;
        }
    }
    key
}

/// Fold one LoHa file into the dense DiT at `spec.scale`, strictly: every module group must reach a
/// projection, and every key must be a LoHa factor or a per-module `.alpha`.
fn fold_loha(
    transformer: &mut QwenImage21Transformer,
    spec: &AdapterSpec,
    file: &AdapterFile,
) -> Result<usize> {
    let stray: Vec<&str> = file
        .tensors
        .keys()
        .map(String::as_str)
        .filter(|key| {
            !key.ends_with(".alpha")
                && wmeta::split_factor_key(key, &wmeta::LOHA_TP_SUFFIXES).is_none()
        })
        .collect();
    if let Some(key) = stray.first() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoHa adapter {} carries {} key(s) that are not LoHa factors (first: `{key}`); \
             refusing a partial apply",
            spec.path.display(),
            stray.len()
        )));
    }
    let groups: std::collections::BTreeMap<String, _> = parse_loha_thirdparty(file)?
        .into_iter()
        .map(|(raw, group)| (strip_namespace(&raw).to_string(), group))
        .collect();
    let mut matched = std::collections::HashSet::new();
    let mut folded = 0usize;
    transformer.visit_adaptable_mut(&mut |path, linear| {
        for key in key_candidates(path) {
            let Some(group) = groups.get(&key) else {
                continue;
            };
            matched.insert(key.clone());
            let delta = group
                .delta(linear.base_shape(), spec.scale)
                .map_err(|e| candle_core::Error::Msg(format!("LoHa `{key}`: {e}")))?;
            linear.fold_dense_delta(&delta).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "{MODEL_ID}: LoHa `{key}` cannot fold into `{path}`: {e}"
                ))
            })?;
            folded += 1;
        }
        Ok(())
    })?;
    let unmatched: Vec<&String> = groups.keys().filter(|k| !matched.contains(*k)).collect();
    if let Some(first) = unmatched.first() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoHa adapter {} has {} module(s) that match no DiT projection (first: \
             `{first}`); every target must apply",
            spec.path.display(),
            unmatched.len()
        )));
    }
    if folded == 0 {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: LoHa adapter {} matched no DiT projection",
            spec.path.display()
        )));
    }
    Ok(folded)
}

/// Install `specs` on a loaded DiT: LoHa files fold into the dense weights (bf16 tier only), then
/// every LoRA / LoKr file attaches as a stacked additive residual. `tier` is the tier the DiT was
/// loaded from; `device` the DiT's device. A no-op for an empty stack.
pub fn install(
    transformer: &mut QwenImage21Transformer,
    specs: &[AdapterSpec],
    tier: Tier,
    device: &Device,
) -> Result<AdapterInstallReport> {
    let mut report = AdapterInstallReport::default();
    if specs.is_empty() {
        return Ok(report);
    }
    let mut additive = Vec::new();
    for spec in specs {
        check_spec(spec)?;
        let file = read_adapter(&spec.path)?;
        if is_loha(&file) {
            admit_loha(spec, tier)?;
            report.loha_folds += fold_loha(transformer, spec, &file)?;
        } else {
            additive.push(spec.clone());
        }
    }
    if additive.is_empty() {
        return Ok(report);
    }
    let applied =
        candle_gen::quant::install_dotted_adapters(MODEL_ID, &additive, device, |visitor| {
            transformer.visit_adaptable_mut(visitor)
        })?;
    if let Some(first) = applied.skipped_targets.first() {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: {} adapter target(s) match no DiT projection (first: `{first}`); every \
             target must apply — expected diffusers/PEFT keys over `transformer_blocks.{{i}}.\
             {{attn.to_q|to_k|to_v|to_out.0, img_mlp.gate_layer|proj|out}}` and the embedder / \
             modulation / output projections",
            applied.skipped_targets.len()
        )));
    }
    if applied.skipped_keys > 0 {
        return Err(Error::Msg(format!(
            "{MODEL_ID}: {} adapter key(s) were not applicable (non-factor keys, half pairs or \
             factors whose shape does not match the projection); refusing a partial apply",
            applied.skipped_keys
        )));
    }
    report.residuals = applied.applied;
    Ok(report)
}
