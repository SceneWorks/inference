//! LoRA / LoKr adapters on the Iris backbone (sc-25681), the MLX family convention: every adapter
//! file is installed through the shared strict loader
//! ([`mlx_gen::adapters::loader::apply_adapters_strict`]) as a **forward-time residual** on the
//! projection it targets — `y = base(x) + Σ adapter(x)` — so the base weight is never mutated. A
//! LoKr applies through the structured Kronecker product (`Y = w1 · X · w2ᵀ`, never a materialized
//! `[out, in]` delta); a LoRA as `scale · (x·Aᵀ)·Bᵀ`.
//!
//! Upstream Iris-3B ships no adapter code, so the surface is the repo's: PEFT/diffusers LoRA
//! (`transformer.` / `diffusion_model.` prefixes or bare), kohya flattened LoRA, PEFT-stamped LoKr
//! (`networkType=lokr`), third-party LyCORIS LoKr/LoHa — keyed by the upstream `IrisDiT` module
//! path (`blocks.3.attn_proj`, `y_embedder.refiner.proj`, …), which is also the checkpoint key stem.
//!
//! **Identity first.** Before a tensor is read, the file's `__metadata__` must name the Iris family
//! and this route's task ([`check_adapter_identity`]): the three Iris task backbones share one
//! architecture, so a depth or restoration adapter would otherwise resolve onto the generation
//! backbone without complaint. **Strict install:** an adapter target that resolves to no
//! projection, a file that lands nothing, and a ComfyUI diff-patch file (`.diff` / `.diff_b`, which
//! this residual path does not fold) are typed errors — never a partially adapted render. Each file's
//! outcome is returned as an [`AdapterApplyReport`] (the provider's provenance surface).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use mlx_gen::adapters::loader::{apply_adapters_strict, has_diff_patch_keys};
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::gen_core::iris::{
    check_adapter_identity, IrisTask, ADAPTER_BASE_MODEL_KEY, ADAPTER_FAMILY_KEY, ADAPTER_TASK_KEY,
};
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterApplyReport, AdapterSpec, Error, Result};
use mlx_rs::Dtype;

use crate::nn::AdaptedLinears;

/// Every 2-D `‹path›.weight` of the checkpoint as an [`AdaptableLinear`] (with its `‹path›.bias`),
/// addressable by dotted module path — the install's host.
struct ProjectionHost {
    linears: BTreeMap<String, AdaptableLinear>,
}

impl ProjectionHost {
    /// Weights and biases in `compute`, exactly as [`crate::nn::Loader::linear`] builds a bare
    /// projection, so an adapted projection differs from the bare one only by its adapters.
    fn from_weights(weights: &Weights, compute: Dtype) -> Result<Self> {
        let mut linears = BTreeMap::new();
        for key in weights.keys() {
            let Some(path) = key.strip_suffix(".weight") else {
                continue;
            };
            let weight = weights.require(key)?;
            if weight.ndim() != 2 {
                continue;
            }
            let bias = weights
                .get(&format!("{path}.bias"))
                .map(|b| b.as_dtype(compute))
                .transpose()?;
            linears.insert(
                path.to_owned(),
                AdaptableLinear::dense(weight.as_dtype(compute)?, bias),
            );
        }
        Ok(Self { linears })
    }
}

impl AdaptableHost for ProjectionHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        self.linears.get_mut(&path.join("."))
    }

    fn adaptable_paths(&self) -> Vec<String> {
        self.linears.keys().cloned().collect()
    }
}

/// The identity stamps of an adapter file's `__metadata__`.
pub fn adapter_identity(path: &Path) -> Result<HashMap<String, String>> {
    let file = Weights::from_file(path)
        .map_err(|e| Error::Msg(format!("iris: reading adapter {}: {e}", path.display())))?;
    Ok(
        [ADAPTER_FAMILY_KEY, ADAPTER_BASE_MODEL_KEY, ADAPTER_TASK_KEY]
            .into_iter()
            .filter_map(|key| {
                file.metadata(key)
                    .map(|value| (key.to_owned(), value.to_owned()))
            })
            .collect(),
    )
}

/// Install `specs` (in order, stacking) onto the projections of `weights` at the `compute` dtype for
/// `task` on `base_model`. Returns the adapted projections for the model build and one report per
/// file. An empty `specs` is `Ok(None)` (the bare build).
pub fn install(
    weights: &Weights,
    compute: Dtype,
    specs: &[AdapterSpec],
    task: IrisTask,
    base_model: &str,
) -> Result<Option<(AdaptedLinears, Vec<AdapterApplyReport>)>> {
    if specs.is_empty() {
        return Ok(None);
    }
    for spec in specs {
        check_adapter_identity(&adapter_identity(&spec.path)?, task, base_model, &spec.path)?;
        if spec.pass_scales.is_some() || spec.moe_expert.is_some() {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {} sets per-pass scales or an MoE expert; the Iris backbone \
                 is one single-pass denoiser, so only `scale` applies",
                spec.path.display()
            )));
        }
        let file = Weights::from_file(&spec.path)?;
        if has_diff_patch_keys(&file) {
            return Err(Error::Unsupported(format!(
                "{base_model}: adapter {} is a ComfyUI diff-patch (`.diff`/`.diff_b`) file; the Iris \
                 adapter surface is LoRA / LoKr / LoHa low-rank factors",
                spec.path.display()
            )));
        }
    }
    let mut host = ProjectionHost::from_weights(weights, compute)?;
    let mut reports = Vec::with_capacity(specs.len());
    for spec in specs {
        // One file at a time: the strict install errors on a file that matched nothing or any
        // target that resolved to no projection, and yields this file's own report.
        let report = apply_adapters_strict(&mut host, std::slice::from_ref(spec), base_model)?;
        reports.push(AdapterApplyReport {
            adapter_path: spec.path.clone(),
            applied: report.applied,
            skipped: report.unmatched_paths,
        });
    }
    let adapted: BTreeSet<String> = host
        .linears
        .iter()
        .filter(|(_, linear)| !linear.adapters().is_empty())
        .map(|(path, _)| path.clone())
        .collect();
    Ok(Some((AdaptedLinears::new(host.linears, adapted), reports)))
}
