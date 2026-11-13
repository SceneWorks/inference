//! One H3-Image training micro-step on candle (SW-7098): the render's own forward at one distilled
//! step, regressed onto the flow-matching target, differentiated with respect to the adapter
//! factors only.
//!
//! The forward is [`crate::dit::ImageDit`]'s, piece for piece: [`MiniMaxH3Dit::image_stack_input`]
//! (`proj_in` over the packed image rows, scattered with the refined caption rows), the 50 blocks
//! gathering their modulation from the run's precomputed tables at the step's AdaLN indices,
//! `norm_out` and the step's own head. Only the generated rows enter the loss. The attention runs
//! inside [`crate::nn::with_differentiable_attention`]: candle's shared training SDPA, which holds
//! one float32 score tile at a time and recomputes it in the backward (the fused kernels have no
//! backward).
//!
//! # Streamed, checkpointed backward (the default)
//!
//! The frozen blocks never sit on the device together. Each block is one segment of the shared
//! segmented VJP ([`checkpointed_backward`]): the forward copies block `l` from the host into the
//! device window ([`MiniMaxH3Dit::window_block`]), runs it, drops it and keeps only its output —
//! the next block's input. The backward walks the blocks in reverse, copying block `l` in again and
//! back-propagating the cotangent from block `l + 1` through one recomputed block, which yields
//! block `l`'s factor gradients and the cotangent for block `l − 1`. Checkpointing is therefore
//! structural: one block's weights and one block's backward are live, whatever the depth.
//!
//! # Resident backward (`gradient_checkpointing = false`)
//!
//! The whole DiT is on the device and one `loss.backward()` walks the full forward, retaining every
//! block's activations — the configuration the memory estimate refuses unless the device holds it.

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Tensor, Var};
use candle_gen::train::flow_match::velocity_loss;
use candle_gen::train::gradient_checkpoint::{checkpointed_backward, Segment};
use candle_gen::{CandleError, Result};

use gen_core_train::h3_image::{training_mix, TrainingStep};

use crate::denoise::packing::gather_rows;
use crate::dit::heads::AdaLayerNormOut;
use crate::dit::model::ImageModulation;
use crate::dit::rope::MmRopeTables;
use crate::dit::MiniMaxH3Dit;
use crate::image_pipeline::ImageLayout;

/// One micro-step's inputs.
pub(crate) struct StepInputs<'a> {
    /// The caption's refined text rows `[1, T, hidden]` (`context_embedder` + `token_refiner`).
    pub(crate) text_rows: &'a Tensor,
    pub(crate) layout: &'a ImageLayout,
    /// The packed image stream `[1, num_packed_image_rows, 96]`, float32: an edit's noised
    /// reference rows, then the noised target.
    pub(crate) packed: &'a Tensor,
    /// The regression target over the generated rows, `x0 − ε`, float32.
    pub(crate) target: &'a Tensor,
    pub(crate) step: TrainingStep,
}

/// The noised target rows and the regression target at `step`: `x_σ = (1 − σ)·x0 + σ·ε`, target
/// `x0 − ε` ([`gen_core_train::h3_image`]'s contract), in float32.
pub(crate) fn noised_rows(
    x0: &Tensor,
    noise: &Tensor,
    step: TrainingStep,
) -> Result<(Tensor, Tensor)> {
    let (data, noise_coef) = training_mix(step.sigma);
    let x0 = x0.to_dtype(DType::F32)?;
    let noise = noise.to_dtype(DType::F32)?;
    let x_t = ((&x0 * f64::from(data))? + (&noise * f64::from(noise_coef))?)?;
    Ok((x_t, (x0 - noise)?))
}

fn core(e: CandleError) -> candle_gen::candle_core::Error {
    candle_gen::candle_core::Error::Msg(e.to_string())
}

/// The per-step constants of the forward: the rotary tables, and the AdaLN and bare-timestep
/// gather indices at the step's head.
struct StepConstants {
    tables: MmRopeTables,
    adaln: Tensor,
    timesteps: Tensor,
}

impl StepConstants {
    fn new(
        dit: &MiniMaxH3Dit,
        modulation: &ImageModulation,
        inputs: &StepInputs<'_>,
    ) -> Result<Self> {
        let device = dit.device();
        let layout = inputs.layout;
        let tables = dit.rope().tables_from_rows(layout.position_ids(), device)?;
        let adaln = modulation.cache.schedule().adaln_indices(
            inputs.step.index,
            &layout.row_classes(),
            layout.token_tags(),
            device,
        )?;
        crate::dit::block::check_adaln_indices(
            &adaln,
            modulation.cache.modulation(0)?.scale_msa.dims()[0],
        )?;
        let timesteps = AdaLayerNormOut::timestep_indices_from_adaln(&adaln)?;
        Ok(Self {
            tables,
            adaln,
            timesteps,
        })
    }
}

/// Block `layer` at the step's indices, through the device window when the stack is streamed.
fn run_block(
    dit: &MiniMaxH3Dit,
    modulation: &ImageModulation,
    constants: &StepConstants,
    layer: usize,
    x: &Tensor,
) -> Result<Tensor> {
    let block = dit.window_block(layer)?;
    block.forward(
        x,
        modulation.cache.modulation(layer)?,
        &constants.adaln,
        dit.rope(),
        &constants.tables,
    )
}

/// `norm_out`, the step's head over the image rows, and the loss over the generated rows.
fn tail_loss(
    dit: &MiniMaxH3Dit,
    modulation: &ImageModulation,
    constants: &StepConstants,
    inputs: &StepInputs<'_>,
    x: &Tensor,
    mae: bool,
) -> Result<Tensor> {
    let layout = inputs.layout;
    let normed = dit
        .projections()
        .norm_out
        .apply(x, &modulation.norm_out, &constants.timesteps)?;
    let image = gather_rows(&normed, layout.image_indices(), normed.dims()[1])?;
    let v = dit
        .projections()
        .output_head(Some(inputs.step.index))?
        .forward(&image)?;
    let generated = v
        .narrow(1, layout.num_reference_rows(), layout.num_image_rows())?
        .to_dtype(DType::F32)?;
    Ok(velocity_loss(&generated, inputs.target, mae)?)
}

/// One micro-step's `(loss, gradients)` for the adapter `vars` installed on `dit`.
pub(crate) fn loss_and_grads(
    dit: &MiniMaxH3Dit,
    modulation: &ImageModulation,
    vars: &[Var],
    inputs: &StepInputs<'_>,
    mae: bool,
) -> Result<(f32, GradStore)> {
    crate::nn::with_differentiable_attention(|| {
        let constants = StepConstants::new(dit, modulation, inputs)?;
        let h0 = dit.image_stack_input(
            inputs.packed,
            inputs.text_rows,
            inputs.layout.text_indices(),
            inputs.layout.image_indices(),
        )?;
        if dit.is_streamed() {
            let constants = &constants;
            let mut segments: Vec<Segment<'_>> = (0..dit.num_layers())
                .map(|layer| -> Segment<'_> {
                    Box::new(move |state: &[Tensor]| {
                        Ok(vec![run_block(
                            dit, modulation, constants, layer, &state[0],
                        )
                        .map_err(core)?])
                    })
                })
                .collect();
            segments.push(Box::new(move |state: &[Tensor]| {
                Ok(vec![tail_loss(
                    dit, modulation, constants, inputs, &state[0], mae,
                )
                .map_err(core)?])
            }));
            checkpointed_backward(&segments, &[h0], vars)
        } else {
            let mut x = h0;
            for layer in 0..dit.num_layers() {
                x = run_block(dit, modulation, &constants, layer, &x)?;
            }
            let loss = tail_loss(dit, modulation, &constants, inputs, &x, mae)?;
            drop(x);
            let value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            Ok((value, loss.backward()?))
        }
    })
}

/// Diagnostics for the real-weights gradient gate (SW-7098): the cotangent arriving at block
/// `from`'s input — the stack restarted at `from` from its forward input `h_from`, then the
/// streamed path's segmented VJP or, resident, one `backward()` through a leaf input — and, for
/// one block, the vector-Jacobian product against a given output cotangent under either attention.
#[cfg(test)]
pub(crate) mod diagnostics {
    use super::*;

    /// `h_from`: the stack's input to block `from`, no graph kept.
    pub(crate) fn block_input(
        dit: &MiniMaxH3Dit,
        modulation: &ImageModulation,
        inputs: &StepInputs<'_>,
        from: usize,
    ) -> Result<Tensor> {
        crate::nn::with_differentiable_attention(|| {
            let constants = StepConstants::new(dit, modulation, inputs)?;
            let mut x = dit
                .image_stack_input(
                    inputs.packed,
                    inputs.text_rows,
                    inputs.layout.text_indices(),
                    inputs.layout.image_indices(),
                )?
                .detach();
            for layer in 0..from {
                x = run_block(dit, modulation, &constants, layer, &x)?.detach();
            }
            Ok(x)
        })
    }

    /// `dL/d h_from` through blocks `from..` and the tail.
    pub(crate) fn boundary_cotangent(
        dit: &MiniMaxH3Dit,
        modulation: &ImageModulation,
        vars: &[Var],
        inputs: &StepInputs<'_>,
        h_from: &Tensor,
        from: usize,
        mae: bool,
    ) -> Result<Tensor> {
        crate::nn::with_differentiable_attention(|| {
            let constants = StepConstants::new(dit, modulation, inputs)?;
            if dit.is_streamed() {
                let constants = &constants;
                let mut segments: Vec<Segment<'_>> = (from..dit.num_layers())
                    .map(|layer| -> Segment<'_> {
                        Box::new(move |state: &[Tensor]| {
                            Ok(vec![run_block(
                                dit, modulation, constants, layer, &state[0],
                            )
                            .map_err(core)?])
                        })
                    })
                    .collect();
                segments.push(Box::new(move |state: &[Tensor]| {
                    Ok(vec![tail_loss(
                        dit, modulation, constants, inputs, &state[0], mae,
                    )
                    .map_err(core)?])
                }));
                let (_, _, mut cot) =
                    candle_gen::train::gradient_checkpoint::checkpointed_backward_with_input_grad(
                        &segments,
                        std::slice::from_ref(h_from),
                        vars,
                    )?;
                Ok(cot.remove(0))
            } else {
                let leaf = Var::from_tensor(h_from)?;
                let mut x = leaf.as_tensor().clone();
                for layer in from..dit.num_layers() {
                    x = run_block(dit, modulation, &constants, layer, &x)?;
                }
                let grads = tail_loss(dit, modulation, &constants, inputs, &x, mae)?.backward()?;
                grads.get(leaf.as_tensor()).cloned().ok_or_else(|| {
                    CandleError::Msg(format!("no cotangent reached block {from}'s input"))
                })
            }
        })
    }

    /// Block `layer`'s VJP against `cot` at input `x`: the gradients of `⟨block(x), cot⟩` for `x`
    /// and the adapter `vars`, under the bounded training attention (`composable == false`) or the
    /// unbounded composable one.
    pub(crate) fn block_vjp(
        dit: &MiniMaxH3Dit,
        modulation: &ImageModulation,
        inputs: &StepInputs<'_>,
        layer: usize,
        x: &Tensor,
        cot: &Tensor,
        composable: bool,
    ) -> Result<(Tensor, GradStore)> {
        let run = || -> Result<(Tensor, GradStore)> {
            let constants = StepConstants::new(dit, modulation, inputs)?;
            let leaf = Var::from_tensor(x)?;
            let y = run_block(dit, modulation, &constants, layer, leaf.as_tensor())?;
            let grads = (y.to_dtype(DType::F32)? * cot.to_dtype(DType::F32)?)?
                .sum_all()?
                .backward()?;
            let gx = grads
                .get(leaf.as_tensor())
                .cloned()
                .ok_or_else(|| CandleError::Msg("no input cotangent".into()))?;
            Ok((gx, grads))
        };
        crate::nn::with_differentiable_attention(|| {
            if composable {
                crate::nn::with_composable_training_attention(run)
            } else {
                run()
            }
        })
    }
}
