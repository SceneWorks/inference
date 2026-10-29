//! Test-only seams: views of the 2.3 transformer (the unit tests drive `LtxDiT` methods through
//! `TrainingTransformer` directly) and the loss-only wrappers over both step-loss functions.

#![cfg(test)]

use super::*;

impl std::ops::Deref for TrainingTransformer {
    type Target = LtxDiT;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ltx23(dit) => dit,
            Self::Ltx25(_) => panic!("2.3-only test helper used with LTX-2.5"),
        }
    }
}

impl std::ops::DerefMut for TrainingTransformer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ltx23(dit) => dit,
            Self::Ltx25(_) => panic!("2.3-only test helper used with LTX-2.5"),
        }
    }
}

/// One forward+backward over the trainable factors: build the rectified-flow input `x_t`, inject the
/// factors, run the video DiT, regress the raw velocity toward `noise - clean`, return `(loss, grads)`.
///
/// `checkpoint_block`, when `Some`, lists each block's trainable targets and switches the forward to
/// the gradient-checkpointed path (sc-4942) — each block recomputes its activations in the backward
/// instead of retaining them. `None` runs the dense (attention-segment-checkpointed) forward.
/// `dtype` is the training compute dtype (sc-4942): for bf16 the DiT weights were cast once in
/// `train_impl` and `preprocess` casts the activation stream, so the LoRA factors are cast at install
/// (here and inside the checkpoint segment) to keep the whole graph bf16; the noising / loss / grads
/// stay f32.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(
        dit,
        params,
        targets,
        alpha,
        rank,
        clean,
        context,
        positions,
        sigma,
        noise,
        mae,
        mask_weight,
        checkpoint_block,
        dtype,
        None,
    )?;
    Ok((losses.total, grads))
}

/// Full flexible-strategy AV loss: each present modality has its own target and loss mask; frozen
/// modalities participate in cross-modal attention but contribute zero loss. Audio-only and
/// video-only workflows execute only their stream, matching upstream's optional modalities.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_ltx25_loss_grads(
    dit: &mut AvDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    batch: &Ltx25PreparedBatch,
    mae: bool,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) =
        compute_ltx25_step_loss_grads(dit, params, targets, alpha, rank, batch, mae, dtype, None)?;
    Ok((losses.total, grads))
}

impl TrainingTransformer {
    pub(super) fn set_sdpa_checkpoint(&mut self, on: bool) {
        match self {
            Self::Ltx23(dit) => dit.set_sdpa_checkpoint(on),
            Self::Ltx25(dit) => dit.set_sdpa_checkpoint(on),
        }
    }

    pub(super) fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        match self {
            Self::Ltx23(dit) => dit.cast_weights(dtype),
            Self::Ltx25(_) => Err("ltx_2_5 test helper cannot cast the full AV base".into()),
        }
    }
}
