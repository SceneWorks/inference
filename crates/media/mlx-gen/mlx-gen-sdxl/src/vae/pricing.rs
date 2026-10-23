//! Header-only VAE hierarchy, bound to the exact prospective Root metadata APIs in proposal625b.
//! No native weights/model/activation construction; logical operands are not backing identities.
use super::{max_conv_tile, VAE_TILING};
use crate::config::VaeConfig;
use crate::quantum as q;
use crate::unet::pricing::{
    element_bytes, nchw_to_nhwc, promote, tensor, tensor_bytes, FactView, HeaderWeights,
    ResnetFacts,
};
use mlx_gen::adapters::facts::TensorFacts as Tensor;
use mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader;
use mlx_gen::Result;
use mlx_rs::Dtype;

pub(crate) struct AttentionFacts {
    pub(crate) _gn_w: Tensor,
    pub(crate) _gn_b: Tensor,
    pub(crate) _q: q::LinearWork,
    pub(crate) _k: q::LinearWork,
    pub(crate) _v: q::LinearWork,
    pub(crate) out: q::LinearWork,
}
pub(crate) struct VaeBlockFacts {
    pub(crate) resnets: Vec<ResnetFacts>,
    pub(crate) downsample: Option<(Tensor, Tensor)>,
    pub(crate) upsample: Option<(Tensor, Tensor)>,
}
pub(crate) struct DecoderFacts {
    pub(crate) conv_in_w: Tensor,
    pub(crate) conv_in_b: Tensor,
    pub(crate) mid_resnet0: ResnetFacts,
    pub(crate) mid_attn: AttentionFacts,
    pub(crate) mid_resnet1: ResnetFacts,
    pub(crate) up_blocks: Vec<VaeBlockFacts>,
    pub(crate) _norm_out_w: Tensor,
    pub(crate) _norm_out_b: Tensor,
    pub(crate) conv_out_w: Tensor,
    pub(crate) conv_out_b: Tensor,
}
pub(crate) struct EncoderFacts {
    pub(crate) conv_in_w: Tensor,
    pub(crate) conv_in_b: Tensor,
    pub(crate) down_blocks: Vec<VaeBlockFacts>,
    pub(crate) mid_resnet0: ResnetFacts,
    pub(crate) mid_attn: AttentionFacts,
    pub(crate) mid_resnet1: ResnetFacts,
    pub(crate) _norm_out_w: Tensor,
    pub(crate) _norm_out_b: Tensor,
    pub(crate) conv_out_w: Tensor,
    pub(crate) _conv_out_b: Tensor,
}
pub(crate) struct AutoencoderFacts {
    pub(crate) encoder: EncoderFacts,
    pub(crate) decoder: DecoderFacts,
    pub(crate) _quant_proj: q::LinearWork,
    pub(crate) post_quant_proj: q::LinearWork,
    _scaling_factor: f32,
    pub(crate) dtype: Dtype,
}
fn linear_output_dtype(linear: &q::LinearWork, input: Dtype) -> Dtype {
    linear.output_dtype(input)
}
fn conv_channels(weight: &Tensor) -> (i32, i32) {
    (weight.shape[weight.shape.len() - 1], weight.shape[0])
}
fn conv_output(
    input: &[i32],
    weight: &Tensor,
    bias: &Tensor,
    stride: i32,
    pad: i32,
) -> Result<Vec<i32>> {
    q::ConvWork {
        weight: weight.clone(),
        bias: Some(bias.clone()),
    }
    .output_shape(input, stride, pad)
}

pub(crate) fn validate_tiled_weight(weight: &Tensor) -> Result<()> {
    if weight.shape.len() != 4 || weight.shape[1] != 3 || weight.shape[2] != 3 {
        return Err(q::native_shape_failure(
            "tiled conv2d requires [out,3,3,in] weights",
        ));
    }
    Ok(())
}

fn conv_call(input: &[i32], act: u64, weight: &Tensor, stride: i32, pad: i32) -> q::ConvCall {
    q::ConvCall::from_metadata(
        input,
        act,
        &weight.shape,
        element_bytes(weight),
        stride,
        pad,
    )
}
fn tile_bytes(
    tile: &[i32],
    source: q::TileSource,
    act: u64,
    weight: &Tensor,
    f32_accum: bool,
    backend: q::Backend,
) -> u64 {
    q::tile_bytes_from_metadata(
        tile,
        source,
        act,
        &weight.shape,
        element_bytes(weight),
        f32_accum,
        backend,
    )
}
fn downsample_tile_bytes(crop: &[i32], act: u64, weight: &Tensor, backend: q::Backend) -> u64 {
    q::downsample_tile_bytes_from_metadata(crop, act, &weight.shape, element_bytes(weight), backend)
}
fn tiled_stage_f32_accum(out: &[i32], weight: &Tensor, bias: &Tensor, act: u64) -> u64 {
    q::tiled_stage_f32_accum_from_metadata(
        out,
        &weight.shape,
        weight.elements(),
        bias.elements(),
        act,
    )
}

impl AttentionFacts {
    pub(crate) fn from_weights(w: &HeaderWeights, prefix: &str) -> Result<Self> {
        let lin = |n: &str| -> Result<q::LinearWork> {
            Ok(q::LinearWork::dense(
                w.require(&format!("{prefix}.{n}.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.{n}.bias"))?.clone()),
            ))
        };
        Ok(Self {
            _gn_w: w.require(&format!("{prefix}.group_norm.weight"))?.clone(),
            _gn_b: w.require(&format!("{prefix}.group_norm.bias"))?.clone(),
            _q: lin("to_q")?,
            _k: lin("to_k")?,
            _v: lin("to_v")?,
            out: lin("to_out.0")?,
        })
    }

    pub(crate) fn bounded_plan() -> mlx_gen::attention::AttentionPlan<'static> {
        mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::from_score_elements(
                crate::quantum::VAE_ATTENTION_SCORE_ELEMENTS,
                true,
            ),
        )
    }

    fn scratch(x: &Tensor, plan: mlx_gen::attention::AttentionPlan<'_>) -> u64 {
        attention_scratch(x.shape(), x.dtype(), plan)
    }

    fn price(
        &self,
        x: &Tensor,
        label: &'static str,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        out.push((label, Self::scratch(x, Self::bounded_plan())));
        q::validate_group_norm(x.shape(), &self._gn_w, &self._gn_b, false)?;
        let width = x.shape()[3];
        for projection in [&self._q, &self._k, &self._v] {
            if crate::text_encoder::linear_output(projection, width)? != width {
                return Err(q::native_shape_failure(
                    "VAE attention projection reshape differs",
                ));
            }
        }
        let branch = crate::text_encoder::linear_output(&self.out, width)?;
        if branch != width && branch != 1 {
            return Err(q::native_shape_failure(
                "VAE attention residual does not broadcast",
            ));
        }
        tensor(
            x.shape(),
            Dtype::from_promoting_types(x.dtype(), linear_output_dtype(&self.out, x.dtype())),
        )
    }
}

impl VaeBlockFacts {
    fn from_weights(
        w: &HeaderWeights,
        prefix: &str,
        num_resnets: i32,
        add_downsample: bool,
        add_upsample: bool,
    ) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| ResnetFacts::from_weights(w, &format!("{prefix}.resnets.{j}")))
            .collect::<Result<Vec<_>>>()?;
        let conv = |which: &str| -> Result<(Tensor, Tensor)> {
            Ok((
                nchw_to_nhwc(w.require(&format!("{prefix}.{which}.0.conv.weight"))?)?,
                w.require(&format!("{prefix}.{which}.0.conv.bias"))?.clone(),
            ))
        };
        Ok(Self {
            resnets,
            downsample: if add_downsample {
                Some(conv("downsamplers")?)
            } else {
                None
            },
            upsample: if add_upsample {
                Some(conv("upsamplers")?)
            } else {
                None
            },
        })
    }

    fn upsamples(&self) -> bool {
        self.upsample.is_some()
    }

    fn price_forward(
        &self,
        prefix: &str,
        x: &Tensor,
        labels: crate::quantum::VaeLabels,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for (j, r) in self.resnets.iter().enumerate() {
            x = price_resnet(labels.resnet, &format!("{prefix}.resnets.{j}"), r, &x, out)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            let s = x.shape();
            let padded = [s[0], s[1] + 1, s[2] + 1, s[3]];
            let call = conv_call(&padded, element_bytes(&x), cw, 2, 0);
            out.push((
                crate::quantum::VAE_DOWNSAMPLE,
                crate::quantum::single_conv_bytes(&call, 1, 0, crate::quantum::backend()),
            ));
            let shape = conv_output(&padded, cw, cb, 2, 0)?;
            x = tensor(&shape, promote(x.dtype(), &[cw, cb]))?;
            q::trace_module(&format!("{prefix}.downsamplers"), &x);
        }
        if let Some((cw, cb)) = &self.upsample {
            out.push((crate::quantum::VAE_UPSAMPLE, upsample_scratch(&x, cw)));
            let s = x.shape();
            let shape = conv_output(&[s[0], 2 * s[1], 2 * s[2], s[3]], cw, cb, 1, 1)?;
            x = tensor(&shape, promote(x.dtype(), &[cw, cb]))?;
            q::trace_module(&format!("{prefix}.upsamplers"), &x);
        }
        Ok(x)
    }

    fn price_tiled_decode(
        &self,
        prefix: &str,
        x: &Tensor,
        tile_edge: i32,
        upsampled_tile_edge: i32,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for (j, resnet) in self.resnets.iter().enumerate() {
            x = resnet.price_tiled_vae(&x, tile_edge, crate::quantum::VAE_DECODE_LABELS, out)?;
            q::trace_module(&format!("{prefix}.resnets.{j}"), &x);
        }
        if let Some((cw, cb)) = &self.upsample {
            x = price_upsample_tiled(&x, cw, cb, upsampled_tile_edge, out)?;
            q::trace_module(&format!("{prefix}.upsamplers"), &x);
        }
        Ok(x)
    }
}

impl DecoderFacts {
    fn from_weights(w: &HeaderWeights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        // decoder layers_per_block = config.layers_per_block + 1.
        let num_resnets = cfg.layers_per_block + 1;
        let up_blocks = (0..n)
            .map(|i| {
                VaeBlockFacts::from_weights(
                    w,
                    &format!("decoder.up_blocks.{i}"),
                    num_resnets,
                    false,
                    i < n - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: nchw_to_nhwc(w.require("decoder.conv_in.weight")?)?,
            conv_in_b: w.require("decoder.conv_in.bias")?.clone(),
            mid_resnet0: ResnetFacts::from_weights(w, "decoder.mid_block.resnets.0")?,
            mid_attn: AttentionFacts::from_weights(w, "decoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetFacts::from_weights(w, "decoder.mid_block.resnets.1")?,
            up_blocks,
            _norm_out_w: w.require("decoder.conv_norm_out.weight")?.clone(),
            _norm_out_b: w.require("decoder.conv_norm_out.bias")?.clone(),
            conv_out_w: nchw_to_nhwc(w.require("decoder.conv_out.weight")?)?,
            conv_out_b: w.require("decoder.conv_out.bias")?.clone(),
        })
    }

    fn conv_out_scratch(&self, x: &Tensor) -> u64 {
        let call = conv_call(x.shape(), element_bytes(x), &self.conv_out_w, 1, 1);
        crate::quantum::norm_conv_bytes(&call, crate::quantum::backend())
    }

    fn price_head(
        &self,
        z: &Tensor,
        pending_input: u64,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let call = conv_call(z.shape(), element_bytes(z), &self.conv_in_w, 1, 1);
        out.push((
            crate::quantum::VAE_DECODE_IN,
            crate::quantum::single_conv_bytes(&call, 2, 0, crate::quantum::backend())
                .saturating_add(pending_input),
        ));
        let shape = conv_output(z.shape(), &self.conv_in_w, &self.conv_in_b, 1, 1)?;
        let x = tensor(
            &shape,
            promote(z.dtype(), &[&self.conv_in_w, &self.conv_in_b]),
        )?;
        q::trace_module("decoder.in", &x);
        let x = price_resnet(
            crate::quantum::VAE_RESNET,
            "decoder.mid_block.resnets.0",
            &self.mid_resnet0,
            &x,
            out,
        )?;
        let x = self
            .mid_attn
            .price(&x, crate::quantum::VAE_ATTENTION, out)?;
        q::trace_module("decoder.mid_block.attentions.0", &x);
        price_resnet(
            crate::quantum::VAE_RESNET,
            "decoder.mid_block.resnets.1",
            &self.mid_resnet1,
            &x,
            out,
        )
    }

    fn price_conv_out_output(&self, x: &Tensor) -> Result<Tensor> {
        let shape = conv_output(x.shape(), &self.conv_out_w, &self.conv_out_b, 1, 1)?;
        let output = tensor(
            &shape,
            promote(x.dtype(), &[&self.conv_out_w, &self.conv_out_b]),
        )?;
        q::trace_module("decoder.out", &output);
        Ok(output)
    }

    fn price_tail(&self, x: &Tensor, out: &mut crate::quantum::Quanta) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, ub) in self.up_blocks.iter().enumerate() {
            x = ub.price_forward(
                &format!("decoder.up_blocks.{i}"),
                &x,
                crate::quantum::VAE_DECODE_LABELS,
                out,
            )?;
        }
        out.push((crate::quantum::VAE_DECODE_OUT, self.conv_out_scratch(&x)));
        q::validate_group_norm(x.shape(), &self._norm_out_w, &self._norm_out_b, false)?;
        self.price_conv_out_output(&x)
    }

    fn price_tail_tiled(
        &self,
        x: &Tensor,
        output_tile_edge: i32,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let tile_edge_at = |remaining_scale: i32| {
            ((output_tile_edge + remaining_scale - 1) / remaining_scale).max(3)
        };
        let mut remaining_scale = self
            .up_blocks
            .iter()
            .filter(|block| block.upsamples())
            .fold(1_i32, |scale, _| scale.saturating_mul(2));
        let mut x = x.clone();
        for (i, up) in self.up_blocks.iter().enumerate() {
            let current_edge = tile_edge_at(remaining_scale);
            let next_scale = if up.upsamples() {
                (remaining_scale / 2).max(1)
            } else {
                remaining_scale
            };
            x = up.price_tiled_decode(
                &format!("decoder.up_blocks.{i}"),
                &x,
                current_edge,
                tile_edge_at(next_scale),
                out,
            )?;
            remaining_scale = next_scale;
        }
        let (cin, cout) = conv_channels(&self.conv_out_w);
        out.push((
            crate::quantum::VAE_TILED_STAGE,
            crate::quantum::tiled_stage_bytes(x.shape(), cin, cout, element_bytes(&x), false),
        ));
        q::validate_group_norm(x.shape(), &self._norm_out_w, &self._norm_out_b, true)?;
        validate_tiled_weight(&self.conv_out_w)?;
        price_tiles(
            crate::quantum::VAE_TILE,
            &x,
            cin,
            &self.conv_out_w,
            output_tile_edge.max(3),
            TileMode {
                f32_accum: false,
                normalized: true,
            },
            out,
        );
        self.price_conv_out_output(&x)
    }
}

impl EncoderFacts {
    fn from_weights(w: &HeaderWeights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        let num_resnets = cfg.layers_per_block;
        let down_blocks = (0..n)
            .map(|i| {
                VaeBlockFacts::from_weights(
                    w,
                    &format!("encoder.down_blocks.{i}"),
                    num_resnets,
                    i < n - 1,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: nchw_to_nhwc(w.require("encoder.conv_in.weight")?)?,
            conv_in_b: w.require("encoder.conv_in.bias")?.clone(),
            down_blocks,
            mid_resnet0: ResnetFacts::from_weights(w, "encoder.mid_block.resnets.0")?,
            mid_attn: AttentionFacts::from_weights(w, "encoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetFacts::from_weights(w, "encoder.mid_block.resnets.1")?,
            _norm_out_w: w.require("encoder.conv_norm_out.weight")?.clone(),
            _norm_out_b: w.require("encoder.conv_norm_out.bias")?.clone(),
            conv_out_w: nchw_to_nhwc(w.require("encoder.conv_out.weight")?)?,
            _conv_out_b: w.require("encoder.conv_out.bias")?.clone(),
        })
    }

    fn price_body(&self, x: &Tensor, out: &mut crate::quantum::Quanta) -> Result<Tensor> {
        let call = conv_call(x.shape(), element_bytes(x), &self.conv_in_w, 1, 1);
        out.push((
            crate::quantum::VAE_ENCODE_IN,
            crate::quantum::single_conv_bytes(&call, 1, 1, crate::quantum::backend()),
        ));
        let shape = conv_output(x.shape(), &self.conv_in_w, &self.conv_in_b, 1, 1)?;
        let mut x = tensor(
            &shape,
            promote(x.dtype(), &[&self.conv_in_w, &self.conv_in_b]),
        )?;
        q::trace_module("encoder.in", &x);
        for (i, db) in self.down_blocks.iter().enumerate() {
            x = db.price_forward(
                &format!("encoder.down_blocks.{i}"),
                &x,
                crate::quantum::VAE_ENCODE_LABELS,
                out,
            )?;
        }
        self.price_mid(&x, out)
    }

    fn price_body_tiled(
        &self,
        x: &Tensor,
        tile: i32,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let s = x.shape();
        let (cin, cout) = conv_channels(&self.conv_in_w);
        out.push((
            crate::quantum::VAE_ENCODE_TILED_STAGE,
            crate::quantum::tiled_stage_bytes(
                &[s[0], s[1], s[2], cout],
                cin,
                cout,
                element_bytes(x),
                false,
            ),
        ));
        validate_tiled_weight(&self.conv_in_w)?;
        price_tiles(
            crate::quantum::VAE_ENCODE_TILE,
            x,
            cin,
            &self.conv_in_w,
            tile.max(3),
            TileMode {
                f32_accum: false,
                normalized: false,
            },
            out,
        );
        let shape = conv_output(x.shape(), &self.conv_in_w, &self.conv_in_b, 1, 1)?;
        let mut x = tensor(
            &shape,
            promote(x.dtype(), &[&self.conv_in_w, &self.conv_in_b]),
        )?;
        q::trace_module("encoder.in", &x);
        let mut scale = 1;
        for (i, db) in self.down_blocks.iter().enumerate() {
            let edge = ((tile + scale - 1) / scale).max(3);
            for (j, r) in db.resnets.iter().enumerate() {
                x = r.price_tiled_vae(&x, edge, crate::quantum::VAE_ENCODE_LABELS, out)?;
                q::trace_module(&format!("encoder.down_blocks.{i}.resnets.{j}"), &x);
            }
            if let Some((cw, cb)) = &db.downsample {
                scale *= 2;
                x = price_downsample_tiled(&x, cw, cb, ((tile + scale - 1) / scale).max(1), out)?;
                q::trace_module(&format!("encoder.down_blocks.{i}.downsamplers"), &x);
            }
        }
        self.price_mid(&x, out)
    }

    fn price_mid(&self, x: &Tensor, out: &mut crate::quantum::Quanta) -> Result<Tensor> {
        let x = price_resnet(
            crate::quantum::VAE_ENCODE_RESNET,
            "encoder.mid_block.resnets.0",
            &self.mid_resnet0,
            x,
            out,
        )?;
        let x = self
            .mid_attn
            .price(&x, crate::quantum::VAE_ENCODE_ATTENTION, out)?;
        q::trace_module("encoder.mid_block.attentions.0", &x);
        price_resnet(
            crate::quantum::VAE_ENCODE_RESNET,
            "encoder.mid_block.resnets.1",
            &self.mid_resnet1,
            &x,
            out,
        )
    }

    fn out_scratch(&self, x: &Tensor) -> u64 {
        let call = conv_call(x.shape(), element_bytes(x), &self.conv_out_w, 1, 1);
        crate::quantum::norm_conv_bytes(&call, crate::quantum::backend())
            .saturating_add(call.out_bytes() / call_act(x) * 4 * 4)
    }
}

impl AutoencoderFacts {
    pub fn from_weights(w: &HeaderWeights, cfg: &VaeConfig) -> Result<Self> {
        let squeeze_lin = |name: &str| -> Result<q::LinearWork> {
            let cw = w.require(&format!("{name}.weight"))?;
            let sh = cw.shape();
            Ok(q::LinearWork::dense(
                cw.reshape(&[sh[0], sh[1]])?,
                Some(w.require(&format!("{name}.bias"))?.clone()),
            ))
        };
        Ok(Self {
            encoder: EncoderFacts::from_weights(w, cfg)?,
            decoder: DecoderFacts::from_weights(w, cfg)?,
            _quant_proj: squeeze_lin("quant_conv")?,
            post_quant_proj: squeeze_lin("post_quant_conv")?,
            _scaling_factor: cfg.scaling_factor,
            dtype: w.require("decoder.conv_in.weight")?.dtype(),
        })
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn compute_dtype(&self, x: Dtype) -> Dtype {
        compute_dtype(self.dtype, x)
    }

    pub(crate) fn decode_input_retained_bytes(&self, latents: &Tensor) -> u64 {
        input_retained_bytes(self.dtype, latents)
    }

    pub(crate) fn price_decode(
        &self,
        latents: &Tensor,
        tiling: Option<&mlx_gen::tiling::TilingConfig>,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let sh = latents.shape();
        let scale = if self.dtype == Dtype::Float32 {
            Dtype::Float32
        } else {
            self.dtype
        };
        let z = Dtype::from_promoting_types(self.compute_dtype(latents.dtype()), scale);
        let width = crate::text_encoder::linear_output(&self.post_quant_proj, sh[3])?;
        let z = tensor(
            &[sh[0], sh[1], sh[2], width],
            linear_output_dtype(&self.post_quant_proj, z),
        )?;
        let pending_input = self.decode_input_retained_bytes(latents) - tensor_bytes(latents);
        let head = self.decoder.price_head(&z, pending_input, out)?;
        let tile = tiling
            .filter(|cfg| cfg.needs_tiling(VAE_TILING, 1, sh[1], sh[2]))
            .and_then(|cfg| cfg.spatial.as_ref().map(|s| s.tile_px));
        let decoded = match tile {
            Some(edge) => self.decoder.price_tail_tiled(&head, edge, out)?,
            None => self.decoder.price_tail(&head, out)?,
        };
        tensor(decoded.shape(), Dtype::Float32)
    }

    pub(crate) fn price_encode(
        &self,
        image: &Tensor,
        tile_px: i32,
        out: &mut crate::quantum::Quanta,
    ) -> Result<()> {
        let s = image.shape();
        let x = tensor(s, self.compute_dtype(image.dtype()))?;
        let body = if s[1] <= tile_px && s[2] <= tile_px {
            self.encoder.price_body(&x, out)?
        } else {
            self.encoder.price_body_tiled(&x, tile_px, out)?
        };
        out.push((
            crate::quantum::VAE_ENCODE_OUT,
            self.encoder.out_scratch(&body),
        ));
        q::validate_group_norm(
            body.shape(),
            &self.encoder._norm_out_w,
            &self.encoder._norm_out_b,
            false,
        )?;
        let moments = conv_output(
            body.shape(),
            &self.encoder.conv_out_w,
            &self.encoder._conv_out_b,
            1,
            1,
        )?;
        let width = crate::text_encoder::linear_output(&self._quant_proj, moments[3])?;
        q::trace_module(
            "encoder.out",
            &tensor(&[moments[0], moments[1], moments[2], width], body.dtype())?,
        );
        Ok(())
    }
}

/// How one tiled convolution accumulates and whether its input tile is a normalized view.
#[derive(Clone, Copy)]
struct TileMode {
    f32_accum: bool,
    normalized: bool,
}

fn price_tiles(
    label: &'static str,
    x: &Tensor,
    channels: i32,
    weight: &Tensor,
    edge: i32,
    mode: TileMode,
    out: &mut crate::quantum::Quanta,
) {
    let TileMode {
        f32_accum,
        normalized,
    } = mode;
    let s = x.shape();
    let (th, tw) = max_conv_tile(s[1], s[2], edge);
    out.push((
        label,
        tile_bytes(
            &[s[0], th, tw, channels],
            q::TileSource {
                full_width: s[2],
                normalized,
            },
            element_bytes(x),
            weight,
            f32_accum,
            crate::quantum::backend(),
        ),
    ));
}

fn price_upsample_tiled(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    tile: i32,
    out: &mut crate::quantum::Quanta,
) -> Result<Tensor> {
    let s = x.shape();
    let (cin, cout) = conv_channels(weight);
    let act = element_bytes(x);
    let up = [s[0], 2 * s[1], 2 * s[2], s[3]];
    let f32_accum = x.dtype() == Dtype::Float16;
    out.push((
        crate::quantum::VAE_TILED_STAGE,
        if f32_accum {
            tiled_stage_f32_accum(&up, weight, bias, act)
        } else {
            crate::quantum::tiled_stage_bytes(&up, cin, cout, act, false)
        },
    ));
    let upsampled = tensor(&up, x.dtype())?;
    validate_tiled_weight(weight)?;
    price_tiles(
        crate::quantum::VAE_TILE,
        &upsampled,
        s[3],
        weight,
        tile,
        TileMode {
            f32_accum,
            normalized: false,
        },
        out,
    );
    let dtype = if f32_accum {
        x.dtype()
    } else {
        promote(x.dtype(), &[weight, bias])
    };
    let shape = conv_output(&up, weight, bias, 1, 1)?;
    tensor(&shape, dtype)
}

fn price_downsample_tiled(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    tile: i32,
    out: &mut crate::quantum::Quanta,
) -> Result<Tensor> {
    let s = x.shape();
    let (oh, ow) = ((s[1] - 2) / 2 + 1, (s[2] - 2) / 2 + 1);
    let core = tile.max(1);
    let act = element_bytes(x);
    out.push((
        crate::quantum::VAE_ENCODE_TILED_STAGE,
        crate::quantum::downsample_stage_bytes(&[s[0], oh, ow, weight.shape()[0]], act),
    ));
    out.push((
        crate::quantum::VAE_ENCODE_TILE,
        downsample_tile_bytes(
            &[s[0], 2 * core.min(oh) + 1, 2 * core.min(ow) + 1, s[3]],
            act,
            weight,
            crate::quantum::backend(),
        ),
    ));
    let shape = conv_output(&[s[0], s[1] + 1, s[2] + 1, s[3]], weight, bias, 2, 0)?;
    tensor(&shape, promote(x.dtype(), &[weight, bias]))
}

fn price_resnet(
    label: &'static str,
    path: &str,
    resnet: &ResnetFacts,
    x: &Tensor,
    out: &mut crate::quantum::Quanta,
) -> Result<Tensor> {
    out.push((label, resnet.quantum_scratch(x, None)));
    let shape = resnet.validate_vae(x.shape())?;
    let output = tensor(&shape, resnet.output_dtype(x.dtype(), None))?;
    q::trace_module(path, &output);
    Ok(output)
}

fn upsample_scratch(x: &Tensor, weight: &Tensor) -> u64 {
    let s = x.shape();
    let call = conv_call(
        &[s[0], 2 * s[1], 2 * s[2], s[3]],
        element_bytes(x),
        weight,
        1,
        1,
    );
    crate::quantum::single_conv_bytes(&call, 1, 0, crate::quantum::backend())
}

fn call_act(x: &Tensor) -> u64 {
    element_bytes(x).max(1)
}

pub(super) fn attention_scratch(
    shape: &[i32],
    dtype: Dtype,
    plan: mlx_gen::attention::AttentionPlan<'_>,
) -> u64 {
    let tokens = shape[1] * shape[2];
    let rows = plan.budget.query_block(shape[0], 1, tokens, tokens);
    q::vae_attention_bytes(
        shape[0],
        tokens,
        shape[3],
        rows,
        q::dtype_bytes(dtype),
        q::backend(),
    )
}
pub(super) fn compute_dtype(loaded: Dtype, input: Dtype) -> Dtype {
    if loaded == Dtype::Float32 {
        input
    } else {
        loaded
    }
}
pub(super) fn input_retained_bytes(loaded: Dtype, latents: &Tensor) -> u64 {
    let caller = latents.bytes();
    let compute = compute_dtype(loaded, latents.dtype);
    caller
        + if compute == latents.dtype {
            0
        } else {
            latents.elements() * q::dtype_bytes(compute)
        }
}
impl AutoencoderFacts {
    pub(crate) fn from_headers(
        headers: &[SafetensorsTensorHeader],
        cfg: &VaeConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        Self::from_weights(&HeaderWeights::new_all_cast(headers, dtype)?, cfg)
    }
}
