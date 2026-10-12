//! The **training adapter** (sc-25213) — ai-toolkit's `assistant_lora_path` on the MLX Z-Image
//! trainer: a frozen LoRA (ostris' `zimage_turbo_training_adapter` de-distill adapter) applied to the
//! step-distilled Z-Image-Turbo base for the training forward/backward only.
//!
//! The semantics follow ai-toolkit's `ZImageModel.load_training_adapter` exactly: the adapter is
//! **merged into the frozen base** at strength 1.0 (`W += B·A·alpha/rank`, alpha defaulting to the
//! rank) before training, so every training forward and backward runs on base + adapter while the
//! user's trainable factors stay separate; at preview time the adapter is applied again at **-1.0**
//! as a forward-time residual on top of the merged base, cancelling it so the in-training samples
//! render as the bare distilled base + the user's LoRA — what the user will get at generation time
//! (ai-toolkit's `invert_assistant_lora`). The saved adapter is written from the trainable factors
//! alone, so it never carries a de-distill key.
//!
//! Key format: the ai-toolkit files key their factors `diffusion_model.<module>.lora_{A,B}.weight`
//! over the main `layers.N` stack (`attention.to_{q,k,v}`, `attention.to_out.0`,
//! `feed_forward.w{1,2,3}`, `adaLN_modulation.0`), no `.alpha` (ai-toolkit builds the network with
//! `alpha = rank`, i.e. scale 1). They are read by the same strict loader inference uses
//! ([`crate::adapters::apply_z_image_adapters`]), which strips the `diffusion_model.` prefix and
//! routes each dotted path through the DiT's `AdaptableHost` map — so every key format inference
//! accepts is accepted here, and a target the DiT does not have is an error, never a silent drop.

use std::path::{Path, PathBuf};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, Adapter};
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::Result;
use mlx_rs::ops::{matmul, multiply};
use mlx_rs::{Array, Dtype};

use crate::adapters::apply_z_image_adapters;

fn spec(path: &Path, scale: f32) -> AdapterSpec {
    AdapterSpec {
        path: path.to_path_buf(),
        scale,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }
}

fn linear<'a, H: AdaptableHost>(host: &'a mut H, path: &str) -> Result<&'a mut AdaptableLinear> {
    let parts: Vec<&str> = path.split('.').collect();
    host.adaptable_mut(&parts).ok_or_else(|| {
        mlx_gen::Error::Msg(format!(
            "training adapter: no adaptable linear at `{path}` on the Z-Image DiT"
        ))
    })
}

/// The dense `[out, in]` delta one installed residual contributes, in f32 — the merge counterpart of
/// [`Adapter::residual`] (`scale · x·a·b` for LoRA, `scale · x·Δᵀ` for LoKr).
fn residual_delta(adapter: &Adapter) -> Result<Array> {
    match adapter {
        Adapter::Lora { a, b, scale } => {
            let ab = matmul(&a.as_dtype(Dtype::Float32)?, &b.as_dtype(Dtype::Float32)?)?;
            Ok(multiply(ab.t(), Array::from_slice(&[*scale], &[1]))?)
        }
        Adapter::Lokr { delta, scale } => Ok(multiply(
            &delta.as_dtype(Dtype::Float32)?,
            Array::from_slice(&[*scale], &[1]),
        )?),
        Adapter::LokrStructured { .. } => Err(mlx_gen::Error::Msg(
            "training adapter: a structured (deferred-Kronecker) LoKr cannot be merged into the \
             training base"
                .into(),
        )),
    }
}

/// The training adapter merged into a Z-Image DiT's frozen base: the file it came from, the
/// dotted module paths it changed (the ones the preview cancel pushes onto and pops off), and the
/// cancelling (-1.0) residual per path, built once on the first preview and reused after.
#[derive(Clone)]
pub(crate) struct MergedTrainingAdapter {
    pub(crate) file: PathBuf,
    pub(crate) paths: Vec<String>,
    cancel: Option<Vec<Adapter>>,
}

/// Merge the training adapter at `file` into `host`'s dense base weights (`W += Δ`, strength 1.0).
///
/// The sum is formed in f32 and rounded ONCE back to the weight's loaded precision: each linear is
/// widened to f32, merged, and narrowed back (a bf16-loaded base therefore rounds `W + Δ` once,
/// instead of rounding `Δ` to bf16 and then the bf16 sum). The trainer merges before its compute
/// cast ([`crate::training`]'s `prepare_training_base`), so an f32-loaded base also rounds once.
///
/// Every block-indexed adaptable path's adapter stack is cleared first (a reused trainer instance
/// may still hold a previous run's installed factors, which must not be folded in); the fresh run
/// installs its own trainable factors afterwards. The adapter is then installed by the strict
/// inference loader and each installed residual is folded into its linear's base and removed, so
/// the result is a plain base carrying the adapter's delta. A file whose targets are not all folded
/// (e.g. a global, non-block module) is refused rather than half-applied.
pub(crate) fn merge_training_adapter<H: AdaptableHost>(
    host: &mut H,
    file: &Path,
) -> Result<MergedTrainingAdapter> {
    let candidates = host.adaptable_paths();
    for path in &candidates {
        linear(host, path)?.set_adapters(Vec::new());
    }
    let report = apply_z_image_adapters(host, &[spec(file, 1.0)]).map_err(|e| {
        mlx_gen::Error::Msg(format!(
            "training adapter {} could not be applied: {e}",
            file.display()
        ))
    })?;
    let mut paths = Vec::new();
    let mut folded = 0usize;
    for path in candidates {
        let lin = linear(host, &path)?;
        if lin.adapters().is_empty() {
            continue;
        }
        let deltas = lin
            .adapters()
            .iter()
            .map(residual_delta)
            .collect::<Result<Vec<_>>>()?;
        let loaded = lin.weight_dtype().ok_or_else(|| {
            mlx_gen::Error::Msg(format!(
                "training adapter: `{path}` has a quantized base; the adapter merges into a dense base"
            ))
        })?;
        lin.cast_weights(Dtype::Float32)?;
        for delta in &deltas {
            lin.merge_dense_delta(delta)?;
            folded += 1;
        }
        lin.cast_weights(loaded)?;
        lin.set_adapters(Vec::new());
        // Materialize this linear's merged weight now, so the f32 delta is freed per layer instead of
        // the whole DiT's deltas staying live in one lazy graph.
        lin.materialize_weights()?;
        paths.push(path);
    }
    if folded != report.applied {
        return Err(mlx_gen::Error::Msg(format!(
            "training adapter {}: {} of its {} targets are outside the DiT's block layers and \
             cannot be merged into the training base",
            file.display(),
            report.applied - folded.min(report.applied),
            report.applied
        )));
    }
    Ok(MergedTrainingAdapter {
        file: file.to_path_buf(),
        paths,
        cancel: None,
    })
}

impl MergedTrainingAdapter {
    /// Push the adapter at strength **-1.0** on top of every merged linear's stack, cancelling the
    /// merge for an inference-only forward (the preview render) — ai-toolkit's inverted assistant
    /// LoRA. Pair with [`pop_cancel`](Self::pop_cancel) before the next training step.
    ///
    /// The file is read and parsed once — on the first preview, through the strict loader — and the
    /// resulting low-rank residuals (the adapter's own factors, not a copy of the base) are
    /// materialized and kept; every later preview pushes those same residuals.
    pub(crate) fn push_cancel<H: AdaptableHost>(&mut self, host: &mut H) -> Result<()> {
        if let Some(cancel) = &self.cancel {
            for (path, adapter) in self.paths.iter().zip(cancel) {
                linear(host, path)?.push(adapter.clone());
            }
            return Ok(());
        }
        let mut before = Vec::with_capacity(self.paths.len());
        for path in &self.paths {
            before.push(linear(host, path)?.adapters().len());
        }
        apply_z_image_adapters(host, &[spec(&self.file, -1.0)])?;
        let mut cancel = Vec::with_capacity(self.paths.len());
        for (path, len) in self.paths.iter().zip(before) {
            let lin = linear(host, path)?;
            if lin.adapters().len() != len + 1 {
                return Err(mlx_gen::Error::Msg(format!(
                    "training adapter {}: the preview cancel did not land exactly once on `{path}`",
                    self.file.display()
                )));
            }
            let adapter = lin.adapters()[len].clone();
            adapter.materialize()?;
            cancel.push(adapter);
        }
        self.cancel = Some(cancel);
        Ok(())
    }

    /// Whether the preview-cancel residuals have been built (after the first preview). Test seam for
    /// the build-once cache.
    #[cfg(test)]
    pub(crate) fn cancel_built(&self) -> bool {
        self.cancel.is_some()
    }

    /// Remove the cancelling residual [`push_cancel`](Self::push_cancel) pushed (the top of each
    /// merged linear's stack), restoring the merged base for training.
    pub(crate) fn pop_cancel<H: AdaptableHost>(&self, host: &mut H) -> Result<()> {
        for path in &self.paths {
            let lin = linear(host, path)?;
            let mut stack = lin.adapters().to_vec();
            stack.pop();
            lin.set_adapters(stack);
        }
        Ok(())
    }
}
