//! The **training adapter** (sc-25213) — ai-toolkit's `assistant_lora_path` on the candle Z-Image
//! trainer: a frozen LoRA (ostris' `zimage_turbo_training_adapter` de-distill adapter) applied to the
//! step-distilled Z-Image-Turbo base for the training forward/backward only.
//!
//! ai-toolkit merges the adapter into the base at strength 1.0 for training and re-applies it at
//! -1.0 for sampling. The candle DiT holds its base weights as immutable mmap-backed tensors, so here
//! the adapter rides as a **frozen forward-time residual** on every projection it targets instead —
//! the same function `(W + B·A)·x` for every training forward and backward, with an *exact* off
//! switch for previews: the residual is installed with two per-pass strengths, `1.0` on
//! [`TRAIN_PASS`] and `0.0` on [`PREVIEW_PASS`] (a zero strength skips the residual entirely, so a
//! preview forward is bit-identical to the bare base + the user's LoRA). Its factors are plain
//! tensors, never `Var`s, so no gradient or optimizer state reaches them and the saved adapter —
//! written from the trainable `Var`s alone — never carries a de-distill key.
//!
//! Key format: the ai-toolkit files key their factors `diffusion_model.<module>.lora_{A,B}.weight`
//! over the main `layers.N` stack (`attention.to_{q,k,v}`, `attention.to_out.0`,
//! `feed_forward.w{1,2,3}`, `adaLN_modulation.0`), no `.alpha` (scale 1). They resolve through the
//! inference key classifier ([`crate::adapters::resolve_training_adapter`]); every resolved module
//! must exist on the DiT with matching shapes, or the run is refused.

use std::collections::BTreeMap;
use std::path::Path;

use candle_core::{DType, Device};
use candle_gen::{CandleError, Result};

use crate::dit::ZImageTransformer2DModel;

/// The additive pass a training forward runs on — the training adapter at strength 1.0.
pub(crate) const TRAIN_PASS: usize = 0;
/// The additive pass a preview render runs on — the training adapter at strength 0.0 (skipped).
pub(crate) const PREVIEW_PASS: usize = 1;

/// Install the training adapter at `file` on `dit` as a frozen residual (strength 1.0 for training,
/// 0.0 for previews) in the DiT's compute `dtype` on its `device`. Returns the number of projections
/// it adapts. Every module the file names must exist on the DiT with the factor shapes it expects.
pub(crate) fn install_training_adapter(
    dit: &mut ZImageTransformer2DModel,
    file: &Path,
    dtype: DType,
    device: &Device,
) -> Result<usize> {
    let mut paths = Vec::new();
    dit.visit_block_linears(&mut |lin| paths.push(lin.path().to_string()));
    let table: BTreeMap<String, String> = paths
        .iter()
        .map(|p| (p.replace('.', "_"), p.clone()))
        .collect();
    let mut factors = crate::adapters::resolve_training_adapter(file, &table)?;
    let mut applied = 0usize;
    dit.visit_block_linears_mut(&mut |lin| {
        let Some((a, b)) = factors.remove(lin.path()) else {
            return Ok(());
        };
        if a.dims() != [lin.in_features(), a.dims()[1]]
            || b.dims() != [a.dims()[1], lin.out_features()]
        {
            return Err(CandleError::Msg(format!(
                "z_image: training adapter {}: factors for `{}` are {:?}·{:?}, the projection is \
                 [{}, {}]",
                file.display(),
                lin.path(),
                a.dims(),
                b.dims(),
                lin.in_features(),
                lin.out_features()
            )));
        }
        let mut scales = vec![0.0; PREVIEW_PASS + 1];
        scales[TRAIN_PASS] = 1.0;
        lin.push_additive_lora_per_pass(
            a.to_device(device)?.to_dtype(dtype)?,
            b.to_device(device)?.to_dtype(dtype)?,
            scales,
        )?;
        lin.set_additive_pass(TRAIN_PASS);
        applied += 1;
        Ok(())
    })?;
    if !factors.is_empty() {
        return Err(CandleError::Msg(format!(
            "z_image: training adapter {}: {} target(s) match no Z-Image DiT projection: {:?}",
            file.display(),
            factors.len(),
            factors.keys().collect::<Vec<_>>()
        )));
    }
    Ok(applied)
}

/// Select the additive pass on every block projection: [`TRAIN_PASS`] (training adapter on) or
/// [`PREVIEW_PASS`] (off). A projection without the residual ignores it.
pub(crate) fn set_training_adapter_pass(dit: &ZImageTransformer2DModel, pass: usize) {
    dit.visit_block_linears(&mut |lin| lin.set_additive_pass(pass));
}

/// The bytes the installed training adapter keeps resident on the device (its factors in the
/// compute dtype): the file's tensor bytes rescaled from the file dtype to `dtype` — counted by the
/// trainer's memory preflight (epic 2123 E7). `0` when no adapter is configured.
pub(crate) fn training_adapter_bytes(file: Option<&Path>, dtype: DType) -> Result<u64> {
    let Some(file) = file else {
        return Ok(0);
    };
    // Only the header is needed (shapes), so map rather than buffer the file.
    // SAFETY: read-only mapping of a file the job owns for the run; the same pattern as
    // `candle_gen::weights`.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(file) }
        .map_err(|e| CandleError::Msg(format!("read training adapter {}: {e}", file.display())))?;
    Ok(st
        .tensors()
        .iter()
        .map(|(_, view)| {
            view.shape().iter().product::<usize>() as u64 * dtype.size_in_bytes() as u64
        })
        .sum())
}
