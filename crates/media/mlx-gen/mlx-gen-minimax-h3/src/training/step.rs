//! One H3-Image training micro-step on MLX (SW-7097): the render's own forward at one distilled
//! step, regressed onto the flow-matching target, differentiated with respect to the adapter
//! factors only.
//!
//! The forward is [`crate::h3_image::ImageDenoiser::velocity`]'s, piece for piece: `proj_in` over
//! the packed image rows, the refined caption rows in front ([`denoiser_parts`]), the 50
//! [`DitBlock`]s gathering their modulation from the run's precomputed tables at the step's AdaLN
//! indices, `norm_out` and the step's own head. Only the generated rows enter the loss.
//!
//! # Streamed, checkpointed backward (the default)
//!
//! The frozen blocks never sit in memory together. The forward walks them one at a time —
//! rebuilt from the staged tier by H3's [`DitBlockStream`], the current factors installed, run,
//! forced, dropped — keeping only each block's **input** (51 residual-width rows). The backward
//! then walks them in reverse: block `l` is rebuilt again, and one `vjp` of `block_l(h_l, θ_l)`
//! against the cotangent arriving from block `l + 1` yields the cotangent for block `l − 1` and
//! block `l`'s factor gradients. So the working set is one block's weights and one block's
//! backward, whatever the depth; the price is a second read of each block and its recompute.
//!
//! # Resident backward (`gradient_checkpointing = false`)
//!
//! The whole DiT is resident and one `keyed_value_and_grad` traces the full forward, retaining
//! every block's activations — the configuration the memory estimate refuses unless the device
//! holds all of it.

use std::rc::Rc;

use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::transforms::{eval, fallible_vjp, keyed_value_and_grad};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{AttentionChunkAxis, AttentionPlan, BoundedAttention};
use mlx_gen::train::lora::{LoraParams, TrainAdapter};
use mlx_gen::train::loss::reduce_loss;
use mlx_gen::{Error, Result};

use gen_core_train::h3_image::{training_mix, TrainingStep};

use crate::denoise::schedule::SigmaSchedule;
use crate::dit::block::DitBlock;
use crate::h3_image::dit::{denoiser_parts, DenoiserParts, MiniMaxH3ImageDit, Modulation};
use crate::h3_image::layout::ImageLayout;
use crate::tensor::slice_axis;

/// The LoKr reconstruct dtype the H3-Image adapter install uses (the host's LyCORIS dtype, bf16),
/// so a trained LoKr round-trips through it.
pub(crate) const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// One block's trainable factors: an adapter restricted to that block's targets (built against
/// the whole DiT, so its keys are the run's `{path}.<factor>` keys) and those keys, sorted.
pub(crate) struct BlockTrainables {
    pub(crate) adapter: Option<TrainAdapter>,
    pub(crate) keys: Vec<Rc<str>>,
}

/// The fixed install / loss settings of a run.
pub(crate) struct LossSpec<'a> {
    /// Every target (the resident path installs them all at once).
    pub(crate) adapter: &'a TrainAdapter,
    /// Per block (the streamed path installs one block's at a time).
    pub(crate) blocks: &'a [BlockTrainables],
    pub(crate) alpha: f32,
    pub(crate) rank: f32,
    pub(crate) mae: bool,
    /// The folded LoRA residual's dtype (the activation stream's under bf16 training).
    pub(crate) lora_dtype: Option<Dtype>,
    /// Streamed + checkpointed (`true`) or resident (`false`).
    pub(crate) streamed: bool,
    /// Attention chunked on whole heads, each chunk recomputed in the backward, at inference rung
    /// 3's plan (E20): no attention call ever holds the whole `[56, S, S]` score domain.
    pub(crate) chunked_attention: bool,
}

/// One micro-step's inputs.
pub(crate) struct StepInputs<'a> {
    /// The caption's layer-50 context `[1, T, 5120]`.
    pub(crate) context: &'a Array,
    pub(crate) layout: &'a ImageLayout,
    /// The packed image stream `[1, num_packed_image_rows, 96]`, float32: an edit's noised
    /// reference rows, then the noised target.
    pub(crate) packed: &'a Array,
    /// The regression target over the generated rows, `x0 − ε`, float32.
    pub(crate) target: &'a Array,
    pub(crate) step: TrainingStep,
}

/// The noised target rows and the regression target at `step`: `x_σ = (1 − σ)·x0 + σ·ε`, target
/// `x0 − ε` ([`gen_core_train::h3_image`]'s contract), in float32.
pub(crate) fn noised_rows(x0: &Array, noise: &Array, step: TrainingStep) -> Result<(Array, Array)> {
    let (data, noise_coef) = training_mix(step.sigma);
    let x0 = x0.as_dtype(Dtype::Float32)?;
    let noise = noise.as_dtype(Dtype::Float32)?;
    let x_t = add(
        &multiply(&x0, Array::from_f32(data))?,
        &multiply(&noise, Array::from_f32(noise_coef))?,
    )?;
    Ok((x_t, subtract(&x0, &noise)?))
}

/// A one-block adapter host: `transformer_blocks.{index}.<leaf>` resolves on `block`, so a block's
/// [`TrainAdapter`] installs on a block rebuilt outside the DiT.
pub(crate) struct BlockHost<'a> {
    pub(crate) index: usize,
    pub(crate) block: &'a mut DitBlock,
}

impl AdaptableHost for BlockHost<'_> {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", i, rest @ ..] if i.parse::<usize>().ok()? == self.index => {
                self.block.adaptable_mut(rest)
            }
            _ => None,
        }
    }
}

fn mlx_err(e: impl std::fmt::Display) -> Exception {
    Exception::custom(e.to_string())
}

/// The per-step constants of the forward: the refined caption rows, rotary tables and gather
/// indices for `inputs.layout`.
pub(crate) fn parts(dit: &mut MiniMaxH3ImageDit, inputs: &StepInputs<'_>) -> Result<DenoiserParts> {
    denoiser_parts(
        dit,
        inputs.layout,
        inputs.context,
        &SigmaSchedule::h3_image_distilled()?,
    )
}

/// `[text | proj_in(image)]` — the stack's input.
pub(crate) fn stack_input(
    dit: &MiniMaxH3ImageDit,
    parts: &DenoiserParts,
    packed: &Array,
) -> Result<Array> {
    let stream = parts.text_rows.dtype();
    let image = dit
        .projections()
        .proj_in
        .forward(packed)?
        .as_dtype(stream)?;
    Ok(concatenate_axis(&[parts.text_rows.clone(), image], 1)?)
}

/// The attention plan every training block runs under: [`AttentionChunkAxis::HeadsRecomputed`] at
/// the training budget ([`super::memory::training_attention_budget`]) when chunked.
pub(crate) fn attention(spec: &LossSpec<'_>, seq_len: u64) -> BoundedAttention<'static> {
    if spec.chunked_attention {
        BoundedAttention::new(
            AttentionPlan::budgeted(super::memory::training_attention_budget(seq_len)),
            AttentionChunkAxis::HeadsRecomputed,
        )
    } else {
        BoundedAttention::UNBOUNDED
    }
}

/// Block `layer` at the step's indices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_block(
    dit: &MiniMaxH3ImageDit,
    block: &DitBlock,
    layer: usize,
    x: &Array,
    modulation: &Modulation,
    parts: &DenoiserParts,
    step: usize,
    bounded: BoundedAttention<'_>,
) -> Result<Array> {
    block.forward_bounded(
        x,
        modulation.cache.modulation(layer)?,
        &parts.adaln_indices[step],
        dit.rope(),
        &parts.tables,
        bounded,
    )
}

/// `norm_out`, the step's head over the image rows, and the loss over the generated rows.
pub(crate) fn tail_loss(
    dit: &MiniMaxH3ImageDit,
    x: &Array,
    modulation: &Modulation,
    parts: &DenoiserParts,
    inputs: &StepInputs<'_>,
    mae: bool,
) -> Result<Array> {
    let step = inputs.step.index;
    let x =
        dit.projections()
            .norm_out
            .apply(x, &modulation.norm_out, &parts.timestep_indices[step])?;
    let layout = inputs.layout;
    let text = layout.num_text_tokens();
    let rows = layout.num_packed_image_rows();
    let image = slice_axis(&x, 1, text, text + rows)?;
    let v = dit.projections().pdd_heads.select(step)?.forward(&image)?;
    let generated =
        slice_axis(&v, 1, layout.num_reference_rows(), rows)?.as_dtype(Dtype::Float32)?;
    Ok(reduce_loss(
        &subtract(&generated, inputs.target)?,
        None,
        mae,
    )?)
}

/// One micro-step's `(loss, gradients)` at `params`.
pub(crate) fn loss_and_grads(
    dit: &mut MiniMaxH3ImageDit,
    modulation: &Modulation,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    inputs: &StepInputs<'_>,
) -> Result<(f32, LoraParams)> {
    let parts = parts(dit, inputs)?;
    // Every array the step's graph shares across blocks is forced before the graph is built, so
    // no block re-derives a lazy shared input inside it.
    eval([
        &parts.text_rows,
        &parts.tables.cos,
        &parts.tables.sin,
        &parts.adaln_indices[inputs.step.index],
        &parts.timestep_indices[inputs.step.index],
    ])?;
    if spec.streamed {
        streamed_loss_and_grads(dit, modulation, &parts, params, spec, inputs, None)
    } else {
        resident_loss_and_grads(dit, modulation, &parts, params, spec, inputs)
    }
}

fn resident_loss_and_grads(
    dit: &mut MiniMaxH3ImageDit,
    modulation: &Modulation,
    parts: &DenoiserParts,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    inputs: &StepInputs<'_>,
) -> Result<(f32, LoraParams)> {
    if dit.is_deferred() {
        return Err(Error::Msg(
            "minimax_h3_image trainer: the resident backward needs a resident DiT".into(),
        ));
    }
    let bounded = attention(spec, inputs.layout.seq_len() as u64);
    let loss_fn = |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        spec.adapter.install_as(
            &mut *dit,
            &p,
            spec.alpha,
            spec.rank,
            spec.lora_dtype,
            LOKR_DTYPE,
        )?;
        let dit: &MiniMaxH3ImageDit = dit;
        let mut x = stack_input(dit, parts, inputs.packed).map_err(mlx_err)?;
        for (layer, block) in dit.blocks().iter().enumerate() {
            x = run_block(
                dit,
                block,
                layer,
                &x,
                modulation,
                parts,
                inputs.step.index,
                bounded,
            )
            .map_err(mlx_err)?;
        }
        Ok(vec![tail_loss(
            dit, &x, modulation, parts, inputs, spec.mae,
        )
        .map_err(mlx_err)?])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (value, grads) = vg(params.clone(), 0)?;
    Ok((value[0].item::<f32>(), grads))
}

/// Block `layer` rebuilt from the staged tier, by itself.
pub(crate) fn materialize(dit: &MiniMaxH3ImageDit, layer: usize) -> Result<DitBlock> {
    let stream = dit.stream().ok_or_else(|| {
        Error::Msg("minimax_h3_image trainer: the streamed backward needs a deferred DiT".into())
    })?;
    // One fresh view per block: a retained view would keep every block it ever read alive.
    let mut view = stream.open()?;
    stream.materialize(&mut view, layer)
}

/// The concrete (untraced) factors of block `layer`, installed for a forward that is not
/// differentiated.
pub(crate) fn install_concrete(
    block: &mut DitBlock,
    layer: usize,
    trainables: &BlockTrainables,
    params: &LoraParams,
    spec: &LossSpec<'_>,
) -> Result<()> {
    if let Some(adapter) = &trainables.adapter {
        adapter.install_as(
            &mut BlockHost {
                index: layer,
                block,
            },
            params,
            spec.alpha,
            spec.rank,
            spec.lora_dtype,
            LOKR_DTYPE,
        )?;
    }
    Ok(())
}

/// A block's factor gradients, in its trainables' key order.
pub(crate) type FactorGrads = Vec<(Rc<str>, Array)>;

/// One block's backward: `vjp` of `block(input, θ_block)` against the cotangent of its output.
/// Returns the input's cotangent and the block's factor gradients, in its trainables' key order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_vjp(
    dit: &MiniMaxH3ImageDit,
    block: &mut DitBlock,
    layer: usize,
    trainables: &BlockTrainables,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    modulation: &Modulation,
    parts: &DenoiserParts,
    step: usize,
    bounded: BoundedAttention<'_>,
    input: Array,
    cotangent: &Array,
) -> Result<(Array, FactorGrads)> {
    let mut primals = Vec::with_capacity(1 + trainables.keys.len());
    primals.push(input);
    for key in &trainables.keys {
        primals.push(
            params
                .get(key)
                .ok_or_else(|| Error::Msg(format!("LoRA param missing: {key}")))?
                .clone(),
        );
    }
    let (_, grads) = fallible_vjp(
        |xs: &[Array]| -> MlxResult<Vec<Array>> {
            if let Some(adapter) = &trainables.adapter {
                let traced: LoraParams = trainables
                    .keys
                    .iter()
                    .cloned()
                    .zip(xs[1..].iter().cloned())
                    .collect();
                adapter.install_as(
                    &mut BlockHost {
                        index: layer,
                        block: &mut *block,
                    },
                    &traced,
                    spec.alpha,
                    spec.rank,
                    spec.lora_dtype,
                    LOKR_DTYPE,
                )?;
            }
            Ok(vec![run_block(
                dit, block, layer, &xs[0], modulation, parts, step, bounded,
            )
            .map_err(mlx_err)?])
        },
        &primals,
        std::slice::from_ref(cotangent),
    )?;
    let mut grads = grads.into_iter();
    let input_cotangent = grads.next().ok_or_else(|| {
        Error::Msg(format!(
            "minimax_h3_image trainer: block {layer} returned no input cotangent"
        ))
    })?;
    Ok((
        input_cotangent,
        trainables.keys.iter().cloned().zip(grads).collect(),
    ))
}

/// The streamed backward. `trace`, when given, receives the cotangent of every block's output
/// `(l + 1, dL/dh_{l+1})` from the stack output down, then block 0's input `(0, dL/dh_0)` — the
/// gradient gate's per-boundary diagnostic.
pub(crate) fn streamed_loss_and_grads(
    dit: &mut MiniMaxH3ImageDit,
    modulation: &Modulation,
    parts: &DenoiserParts,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    inputs: &StepInputs<'_>,
    mut trace: Option<&mut Vec<(usize, Array)>>,
) -> Result<(f32, LoraParams)> {
    let dit: &MiniMaxH3ImageDit = dit;
    let layers = dit.num_layers();
    if spec.blocks.len() != layers {
        return Err(Error::Msg(format!(
            "minimax_h3_image trainer: {} block trainables for a {layers}-block stack",
            spec.blocks.len()
        )));
    }
    let step = inputs.step.index;
    let bounded = attention(spec, inputs.layout.seq_len() as u64);

    // Forward: keep every block's input, forced, and nothing else.
    let mut saved = Vec::with_capacity(layers + 1);
    let h0 = stack_input(dit, parts, inputs.packed)?;
    eval([&h0])?;
    saved.push(h0);
    for layer in 0..layers {
        let mut block = materialize(dit, layer)?;
        install_concrete(&mut block, layer, &spec.blocks[layer], params, spec)?;
        let h = run_block(
            dit,
            &block,
            layer,
            &saved[layer],
            modulation,
            parts,
            step,
            bounded,
        )?;
        eval([&h])?;
        drop(block);
        saved.push(h);
    }

    // The tail's loss and its cotangent into the last block's output.
    let last = saved.pop().expect("the stack output was saved");
    let (value, grads) = fallible_vjp(
        |xs: &[Array]| -> MlxResult<Vec<Array>> {
            Ok(vec![tail_loss(
                dit, &xs[0], modulation, parts, inputs, spec.mae,
            )
            .map_err(mlx_err)?])
        },
        &[last],
        &[Array::from_f32(1.0)],
    )?;
    let loss = value[0].item::<f32>();
    let mut cotangent = grads.into_iter().next().ok_or_else(|| {
        Error::Msg("minimax_h3_image trainer: the tail returned no cotangent".into())
    })?;
    eval([&cotangent])?;
    if let Some(trace) = trace.as_deref_mut() {
        trace.push((layers, cotangent.clone()));
    }

    // Backward: block `l` again, one vjp against the cotangent from block `l + 1`.
    let mut out = LoraParams::new();
    for layer in (0..layers).rev() {
        let input = saved.pop().expect("one saved input per block");
        let trainables = &spec.blocks[layer];
        let mut block = materialize(dit, layer)?;
        let (next, block_grads) = block_vjp(
            dit, &mut block, layer, trainables, params, spec, modulation, parts, step, bounded,
            input, &cotangent,
        )?;
        cotangent = next;
        let mut forced: Vec<&Array> = block_grads.iter().map(|(_, g)| g).collect();
        if layer > 0 {
            forced.push(&cotangent);
        }
        eval(forced)?;
        drop(block);
        out.extend(block_grads);
        if let Some(trace) = trace.as_deref_mut() {
            trace.push((layer, cotangent.clone()));
        }
    }
    Ok((loss, out))
}
