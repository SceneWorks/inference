//! `UNetBlock2D` — the down/up macro-block: a run of `ResnetBlock2D`s, optional cross-attention
//! `Transformer2D`s interleaved, and an optional downsample (stride-2 conv) or upsample
//! (nearest-2× + conv). Port of the vendored `unet.UNetBlock2D`. On the up path each resnet is fed
//! `concat(x, residual.pop())` (the U-Net skip connections); the resnets' loaded conv weights
//! already carry the post-concat channel counts, so no channel math is needed here.

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableConv2d, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{conv2d, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::nchw_to_nhwc;
use super::resnet::ResnetBlock2D;
use super::transformer::Transformer2D;

#[derive(Clone)]
pub struct UNetBlock2D {
    resnets: Vec<ResnetBlock2D>,
    attentions: Option<Vec<Transformer2D>>,
    /// Downsample conv (stride 2, pad 1) on NHWC — a conv-layer LoRA target (sc-2919).
    downsample: Option<AdaptableConv2d>,
    /// Upsample conv (stride 1, pad 1) applied after a nearest-2× resize — a conv-layer LoRA target.
    upsample: Option<AdaptableConv2d>,
}

/// Per-block construction parameters resolved from [`crate::config::UNetConfig`].
pub struct BlockSpec<'a> {
    /// Checkpoint module prefix, e.g. `down_blocks.0` or `up_blocks.2`.
    pub prefix: &'a str,
    pub num_resnets: i32,
    pub out_channels: i32,
    pub num_heads: i32,
    pub transformer_layers: i32,
    pub add_cross_attention: bool,
    pub add_downsample: bool,
    pub add_upsample: bool,
}

impl UNetBlock2D {
    pub fn from_weights(w: &Weights, spec: &BlockSpec) -> Result<Self> {
        let resnets = (0..spec.num_resnets)
            .map(|j| ResnetBlock2D::from_weights(w, &format!("{}.resnets.{j}", spec.prefix)))
            .collect::<Result<Vec<_>>>()?;
        let attentions = if spec.add_cross_attention {
            Some(
                (0..spec.num_resnets)
                    .map(|j| {
                        Transformer2D::from_weights(
                            w,
                            &format!("{}.attentions.{j}", spec.prefix),
                            spec.out_channels,
                            spec.num_heads,
                            spec.transformer_layers,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let downsample = if spec.add_downsample {
            Some(AdaptableConv2d::new(
                nchw_to_nhwc(w.require(&format!("{}.downsamplers.0.conv.weight", spec.prefix))?)?,
                Some(
                    w.require(&format!("{}.downsamplers.0.conv.bias", spec.prefix))?
                        .clone(),
                ),
            ))
        } else {
            None
        };
        let upsample = if spec.add_upsample {
            Some(AdaptableConv2d::new(
                nchw_to_nhwc(w.require(&format!("{}.upsamplers.0.conv.weight", spec.prefix))?)?,
                Some(
                    w.require(&format!("{}.upsamplers.0.conv.bias", spec.prefix))?
                        .clone(),
                ),
            ))
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            downsample,
            upsample,
        })
    }

    pub(crate) fn quantize_using(
        &mut self,
        prefix: &str,
        bits: i32,
        visit: &mut crate::quant::QuantizeLinear<'_>,
    ) -> Result<()> {
        for (index, r) in self.resnets.iter_mut().enumerate() {
            r.quantize_using(
                &crate::quant::child_path(prefix, &format!("resnets.{index}")),
                bits,
                visit,
            )?;
        }
        if let Some(attns) = &mut self.attentions {
            for (index, a) in attns.iter_mut().enumerate() {
                a.quantize_using(
                    &crate::quant::child_path(prefix, &format!("attentions.{index}")),
                    bits,
                    visit,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn materialize_weights(&self) -> Result<()> {
        for resnet in &self.resnets {
            resnet.materialize_weights()?;
        }
        if let Some(attentions) = &self.attentions {
            for attention in attentions {
                attention.materialize_weights()?;
            }
        }
        if let Some(downsample) = &self.downsample {
            downsample.materialize_weights()?;
        }
        if let Some(upsample) = &self.upsample {
            upsample.materialize_weights()?;
        }
        Ok(())
    }

    /// Number of skip tensors this block pops on the up path (one per resnet) — the count
    /// `forward_block_checkpointed` (sc-4941) peels off the residual stack to thread into the block's
    /// checkpoint segment as explicit inputs.
    pub fn num_skip_inputs(&self) -> usize {
        self.resnets.len()
    }

    /// Toggle SDPA-segment checkpointing on every cross-attention transformer in this block (sc-4941).
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        if let Some(attns) = &mut self.attentions {
            for a in attns {
                a.set_sdpa_checkpoint(on);
            }
        }
    }

    /// Cast every dtype-bearing leaf (resnets, attention transformers, samplers) to `dtype` (sc-4941).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for r in &mut self.resnets {
            r.cast_weights(dtype)?;
        }
        if let Some(attns) = &mut self.attentions {
            for a in attns {
                a.cast_weights(dtype)?;
            }
        }
        if let Some(c) = &mut self.downsample {
            c.cast_weights(dtype)?;
        }
        if let Some(c) = &mut self.upsample {
            c.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// Run the block. `residuals` is `Some` on the up path (each resnet pops one skip tensor and
    /// concatenates it onto `x` first). Returns the block output plus the per-step output states
    /// (the down path pushes these onto the residual stack). Threads the IP-Adapter tokens + scale
    /// into each cross-attention transformer (sc-3059), and the ladder's rung-3/rung-4 `plan`
    /// (SC-15525) — whose `cfg_row` also reaches the resnets (sc-26268).
    ///
    /// The resnets, samplers and the skip stack this block builds are the **resident remainder**:
    /// `plan` reaches only the `attentions`, because only a `Transformer2D` holds a uniform block
    /// stack a window can bound and an attention a chunk can bound.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_ip(
        &self,
        x: &Array,
        encoder_x: &Array,
        temb: &Array,
        mut residuals: Option<&mut Vec<Array>>,
        ip: Option<(&Array, f32)>,
        plan: crate::plan::SdxlForwardPlan<'_>,
    ) -> Result<(Array, Vec<Array>)> {
        let mut x = x.clone();
        let mut output_states = Vec::with_capacity(self.resnets.len() + 1);
        // sc-26270: each resnet (with its skip concat), each attention block and each sampler is one
        // guarded iOS quantum; every guard is a pass-through off device.
        for i in 0..self.resnets.len() {
            let skip = match residuals.as_deref_mut() {
                Some(res) => Some(res.pop().ok_or_else(|| {
                    mlx_gen::Error::Msg("sdxl unet: residual stack underflow".into())
                })?),
                None => None,
            };
            crate::quantum::diagnostic_owned_tensors(
                "block-local-before-resnet",
                i as u64,
                output_states.iter(),
            );
            x = self.resnets[i].forward_quantum(&x, skip, Some(temb), plan)?;
            if let Some(attns) = &self.attentions {
                crate::quantum::diagnostic_owned_tensors(
                    "block-local-before-attention",
                    i as u64,
                    output_states.iter().chain(std::iter::once(&x)),
                );
                x = attns[i].forward_ip(&x, encoder_x, ip, plan)?;
            }
            output_states.push(x.clone());
        }
        if let Some(c) = &self.downsample {
            crate::quantum::diagnostic_owned_tensors(
                "block-local-before-downsample",
                0,
                output_states.iter(),
            );
            let scratch = downsample_scratch(&x, c);
            x = mlx_gen::ios_quantum::run(crate::quantum::UNET_DOWNSAMPLE, scratch, || {
                conv2d(&x, c.weight(), c.bias(), 2, 1)
            })?;
            output_states.push(x.clone());
        }
        if let Some(c) = &self.upsample {
            crate::quantum::diagnostic_owned_tensors(
                "block-local-before-upsample",
                0,
                output_states.iter(),
            );
            let scratch = upsample_scratch(&x, c);
            x = mlx_gen::ios_quantum::run(crate::quantum::UNET_UPSAMPLE, scratch, || {
                conv2d(&upsample_nearest(&x, 2)?, c.weight(), c.bias(), 1, 1)
            })?;
            output_states.push(x.clone());
        }
        Ok((x, output_states))
    }

    #[cfg(test)]
    pub(crate) fn price_forward_with_effects(
        &self,
        x: &Array,
        inputs: super::pricing::BlockForwardInputs<'_, Array>,
        mut residuals: Option<&mut Vec<Array>>,
        out: &mut crate::quantum::Quanta,
        ownership: super::pricing::BlockOwners<'_>,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<(Array, Vec<Array>)> {
        let super::pricing::BlockForwardInputs {
            encoder_x,
            temb,
            ip,
            plan,
        } = inputs;
        let super::pricing::BlockOwners {
            outer_held,
            mut owners,
        } = ownership;
        let mut x = x.clone();
        let mut output_states = Vec::with_capacity(self.resnets.len() + 1);
        for i in 0..self.resnets.len() {
            let skip = match residuals.as_deref_mut() {
                Some(res) => Some(res.pop().ok_or_else(|| {
                    mlx_gen::Error::Msg("sdxl unet pricing: residual stack underflow".into())
                })?),
                None => None,
            };
            let resnet = &self.resnets[i];
            if let Some(owners) = owners.as_deref_mut() {
                owners.push(
                    outer_held + output_states.iter().map(crate::quantum::bytes).sum::<u64>(),
                );
            }
            out.push((
                crate::quantum::UNET_RESNET,
                resnet.quantum_scratch_with_effects(
                    &x,
                    Some(temb),
                    &effects.child(&format!("resnets.{i}")),
                )?,
            ));
            let input = skip.as_ref().map_or(x.dtype(), |s| {
                Dtype::from_promoting_types(x.dtype(), s.dtype())
            });
            let shape = resnet.output_shape_with_effects(
                x.shape(),
                skip.as_ref().map(Array::shape),
                &effects.child(&format!("resnets.{i}")),
            )?;
            x = crate::quantum::shaped(
                &shape,
                resnet.output_dtype_with_effects(
                    input,
                    Some(temb.dtype()),
                    &effects.child(&format!("resnets.{i}")),
                )?,
            )?;
            if let Some(attns) = &self.attentions {
                let first = out.len();
                // The caller's resnet output remains borrowed for the whole attention call;
                // only PREVIOUS output_states are already in the retained stack at this point.
                let held = outer_held
                    + output_states.iter().map(crate::quantum::bytes).sum::<u64>()
                    + crate::quantum::bytes(&x);
                x = attns[i].price_forward_with_effects(
                    &x,
                    encoder_x,
                    ip,
                    plan,
                    out,
                    &effects.child(&format!("attentions.{i}")),
                )?;
                if let Some(owners) = owners.as_deref_mut() {
                    owners.extend(std::iter::repeat_n(held, out.len() - first));
                }
            }
            output_states.push(x.clone());
        }
        if let Some(c) = &self.downsample {
            if let Some(owners) = owners.as_deref_mut() {
                owners.push(
                    outer_held + output_states.iter().map(crate::quantum::bytes).sum::<u64>(),
                );
            }
            let conv = effects.conv("downsamplers.0.conv", c);
            out.push((
                crate::quantum::UNET_DOWNSAMPLE,
                sampler_scratch(&x, &conv, false),
            ));
            x = crate::quantum::shaped(
                &conv.output_shape(x.shape(), 2, 1)?,
                conv.output_dtype(x.dtype()),
            )?;
            output_states.push(x.clone());
        }
        if let Some(c) = &self.upsample {
            if let Some(owners) = owners {
                owners.push(
                    outer_held + output_states.iter().map(crate::quantum::bytes).sum::<u64>(),
                );
            }
            let conv = effects.conv("upsamplers.0.conv", c);
            out.push((
                crate::quantum::UNET_UPSAMPLE,
                sampler_scratch(&x, &conv, true),
            ));
            let s = x.shape();
            let input = [s[0], 2 * s[1], 2 * s[2], s[3]];
            x = crate::quantum::shaped(
                &conv.output_shape(&input, 1, 1)?,
                conv.output_dtype(x.dtype()),
            )?;
            output_states.push(x.clone());
        }
        Ok((x, output_states))
    }

    /// Arm ladder rung 4 on every `Transformer2D` this block owns (SC-16355). `prefix` is the
    /// block's own checkpoint prefix (`down_blocks.2`, `up_blocks.0`, …); each sub-stack records
    /// `{prefix}.attentions.{j}`, which is byte-identical to what `from_weights` read. A
    /// no-attention block arms nothing, which is correct — `down_blocks.0` / `up_blocks.2` have no
    /// windowable stack.
    pub(crate) fn arm_block_streams(
        &mut self,
        source: &mlx_gen::WeightsSource,
        prefix: &str,
        dtype: mlx_rs::Dtype,
        quant_bits: Option<i32>,
    ) {
        if let Some(attns) = &mut self.attentions {
            for (j, attn) in attns.iter_mut().enumerate() {
                attn.arm_block_stream(
                    source.clone(),
                    &format!("{prefix}.attentions.{j}"),
                    dtype,
                    quant_bits,
                );
            }
        }
    }

    /// Disarm rung 4 on every sub-stack this block owns.
    pub(crate) fn disarm_block_streams(&mut self) {
        if let Some(attns) = &mut self.attentions {
            for attn in attns {
                attn.disarm_block_stream();
            }
        }
    }

    pub(crate) fn bind_source_inventory(&mut self, guard: &mlx_gen::gen_core::SourceInventory) {
        if let Some(attentions) = &mut self.attentions {
            for attention in attentions {
                attention.bind_source_inventory(guard);
            }
        }
    }

    /// This block's `Transformer2D` depths, in order — the per-sub-stack window domains.
    pub(crate) fn attention_depths(&self) -> Vec<usize> {
        self.attentions
            .as_ref()
            .map(|attns| {
                attns
                    .iter()
                    .map(super::transformer::Transformer2D::depth)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether every `Transformer2D` this block owns can stream. Vacuously true for a block with no
    /// attention.
    pub(crate) fn block_streams_armed(&self) -> bool {
        self.attentions
            .as_ref()
            .map(|attns| {
                attns
                    .iter()
                    .all(super::transformer::Transformer2D::can_stream_blocks)
            })
            .unwrap_or(true)
    }

    /// Install IP-Adapter K/V projections into this block's cross-attention transformers (sc-3059),
    /// consuming pairs in transformer/block order. A no-attention block consumes nothing.
    pub fn install_ip(&mut self, pairs: &mut impl Iterator<Item = (Array, Array)>) -> Result<()> {
        if let Some(attns) = &mut self.attentions {
            for a in attns {
                a.install_ip(pairs)?;
            }
        }
        Ok(())
    }

    /// Emit the diffusers paths of this block's LoRA-targetable Linears (resnet `time_emb_proj`s +
    /// each `attentions.{j}` transformer's projections), prefixed by `prefix` (e.g. `down_blocks.1`).
    pub fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        for (j, r) in self.resnets.iter().enumerate() {
            r.lora_target_paths(&format!("{prefix}.resnets.{j}"), out);
        }
        if let Some(attns) = &self.attentions {
            for (j, a) in attns.iter().enumerate() {
                a.lora_target_paths(&format!("{prefix}.attentions.{j}"), out);
            }
        }
    }

    /// Emit this block's GEGLU feed-forward LoRA targets (sc-2671 complete coverage) — the FF of
    /// each cross-attention transformer, beyond the vendored-reachable surface.
    pub fn lora_target_paths_ff(&self, prefix: &str, out: &mut Vec<String>) {
        if let Some(attns) = &self.attentions {
            for (j, a) in attns.iter().enumerate() {
                a.lora_target_paths_ff(&format!("{prefix}.attentions.{j}"), out);
            }
        }
    }

    /// Emit this block's conv-layer LoRA targets (sc-2919): each resnet's convs plus the
    /// down/up-sampler's `conv` (`downsamplers.0.conv` / `upsamplers.0.conv`).
    pub fn conv_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        for (j, r) in self.resnets.iter().enumerate() {
            r.conv_target_paths(&format!("{prefix}.resnets.{j}"), out);
        }
        if self.downsample.is_some() {
            out.push(format!("{prefix}.downsamplers.0.conv"));
        }
        if self.upsample.is_some() {
            out.push(format!("{prefix}.upsamplers.0.conv"));
        }
    }
}

impl AdaptableHost for UNetBlock2D {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["resnets", j, rest @ ..] => self
                .resnets
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["attentions", j, rest @ ..] => self
                .attentions
                .as_mut()?
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_conv_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableConv2d> {
        match path {
            ["resnets", j, rest @ ..] => self
                .resnets
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_conv_mut(rest),
            ["downsamplers", "0", "conv"] => self.downsample.as_mut(),
            ["upsamplers", "0", "conv"] => self.upsample.as_mut(),
            _ => None,
        }
    }
}

/// Shared sampler price from actual OHWI metadata. Native wrappers keep their original trace
/// operands; selected-world pricing substitutes only the installed weight facts.
fn sampler_scratch(x: &Array, work: &crate::quantum::ConvWork, up: bool) -> u64 {
    super::pricing::sampler_scratch(&crate::quantum::activation_facts(x), work, up)
}

fn downsample_scratch(x: &Array, c: &AdaptableConv2d) -> u64 {
    #[cfg(test)]
    {
        crate::quantum::trace_layout(crate::quantum::UNET_DOWNSAMPLE, x);
        crate::quantum::trace_layout("downsample weight", c.weight());
    }
    sampler_scratch(x, &crate::quantum::EffectScope::native().conv("", c), false)
}

fn upsample_scratch(x: &Array, c: &AdaptableConv2d) -> u64 {
    #[cfg(test)]
    {
        crate::quantum::trace_layout(crate::quantum::UNET_UPSAMPLE, x);
        crate::quantum::trace_layout("upsample weight", c.weight());
    }
    sampler_scratch(x, &crate::quantum::EffectScope::native().conv("", c), true)
}
