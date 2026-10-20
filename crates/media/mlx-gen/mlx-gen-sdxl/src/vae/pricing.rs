//! Header-only VAE hierarchy, bound to the exact prospective Root metadata APIs in proposal625b.
//! No native weights/model/activation construction; logical operands are not backing identities.
use mlx_gen::Result;
use mlx_rs::Dtype;
use mlx_gen::adapters::facts::TensorFacts as Tensor;
use mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader;
use crate::config::VaeConfig;
use crate::quantum as q;
use crate::unet::pricing::{HeaderWeights, ResnetFacts, FactView, tensor, tensor_bytes,
    element_bytes, promote, nchw_to_nhwc};
use super::{VAE_TILING, max_conv_tile};

struct AttentionFacts {
    _gn_w: Tensor, _gn_b: Tensor, _q: q::LinearWork, _k: q::LinearWork, _v: q::LinearWork,
    out: q::LinearWork,
}
struct VaeBlockFacts { resnets: Vec<ResnetFacts>, downsample: Option<(Tensor, Tensor)>, upsample: Option<(Tensor, Tensor)> }
struct DecoderFacts {
    conv_in_w: Tensor, conv_in_b: Tensor, mid_resnet0: ResnetFacts, mid_attn: AttentionFacts,
    mid_resnet1: ResnetFacts, up_blocks: Vec<VaeBlockFacts>, _norm_out_w: Tensor, _norm_out_b: Tensor,
    conv_out_w: Tensor, conv_out_b: Tensor,
}
struct EncoderFacts {
    conv_in_w: Tensor, conv_in_b: Tensor, down_blocks: Vec<VaeBlockFacts>, mid_resnet0: ResnetFacts,
    mid_attn: AttentionFacts, mid_resnet1: ResnetFacts, _norm_out_w: Tensor, _norm_out_b: Tensor,
    conv_out_w: Tensor, _conv_out_b: Tensor,
}
pub(crate) struct AutoencoderFacts {
    encoder: EncoderFacts, decoder: DecoderFacts, _quant_proj: q::LinearWork,
    post_quant_proj: q::LinearWork, _scaling_factor: f32, dtype: Dtype,
}
fn linear_output_dtype(linear: &q::LinearWork, input: Dtype) -> Dtype { linear.output_dtype(input) }
fn conv_channels(weight: &Tensor) -> (i32, i32) { (weight.shape[weight.shape.len() - 1], weight.shape[0]) }
fn conv_call(input: &[i32], act: u64, weight: &Tensor, stride: i32, pad: i32) -> q::ConvCall {
    q::ConvCall::from_metadata(input, act, &weight.shape, element_bytes(weight), stride, pad)
}
fn tile_bytes(tile: &[i32], source: q::TileSource, act: u64, weight: &Tensor,
    f32_accum: bool, backend: q::Backend) -> u64 {
    q::tile_bytes_from_metadata(tile, source, act, &weight.shape, element_bytes(weight), f32_accum, backend)
}
fn downsample_tile_bytes(crop: &[i32], act: u64, weight: &Tensor, backend: q::Backend) -> u64 {
    q::downsample_tile_bytes_from_metadata(crop, act, &weight.shape, element_bytes(weight), backend)
}
fn tiled_stage_f32_accum(out: &[i32], weight: &Tensor, bias: &Tensor, act: u64) -> u64 {
    q::tiled_stage_f32_accum_from_metadata(out, &weight.shape, weight.elements(), bias.elements(), act)
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
        tensor(
            x.shape(),
            Dtype::from_promoting_types(
                x.dtype(),
                linear_output_dtype(&self.out, x.dtype()),
            ),
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
        x: &Tensor,
        labels: crate::quantum::VaeLabels,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = price_resnet(labels.resnet, r, &x, out)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            let s = x.shape();
            let padded = [s[0], s[1] + 1, s[2] + 1, s[3]];
            let call =
                conv_call(&padded, element_bytes(&x), cw, 2, 0);
            out.push((
                crate::quantum::VAE_DOWNSAMPLE,
                crate::quantum::single_conv_bytes(&call, 1, 0, crate::quantum::backend()),
            ));
            let half = |n: i32| (n + 1 - 3) / 2 + 1;
            x = tensor(
                &[s[0], half(s[1]), half(s[2]), cw.shape()[0]],
                promote(x.dtype(), &[cw, cb]),
            )?;
        }
        if let Some((cw, cb)) = &self.upsample {
            out.push((crate::quantum::VAE_UPSAMPLE, upsample_scratch(&x, cw)));
            let s = x.shape();
            x = tensor(
                &[s[0], 2 * s[1], 2 * s[2], cw.shape()[0]],
                promote(x.dtype(), &[cw, cb]),
            )?;
        }
        Ok(x)
    }

fn price_tiled_decode(
        &self,
        x: &Tensor,
        tile_edge: i32,
        upsampled_tile_edge: i32,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.price_tiled_vae(&x, tile_edge, crate::quantum::VAE_DECODE_LABELS, out)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = price_upsample_tiled(&x, cw, cb, upsampled_tile_edge, out)?;
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
        let call = conv_call(
            x.shape(),
            element_bytes(x),
            &self.conv_out_w,
            1,
            1,
        );
        crate::quantum::norm_conv_bytes(&call, crate::quantum::backend())
    }

fn price_head(
        &self,
        z: &Tensor,
        pending_input: u64,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let call = conv_call(
            z.shape(),
            element_bytes(z),
            &self.conv_in_w,
            1,
            1,
        );
        out.push((
            crate::quantum::VAE_DECODE_IN,
            crate::quantum::single_conv_bytes(&call, 2, 0, crate::quantum::backend())
                .saturating_add(pending_input),
        ));
        let s = z.shape();
        let x = tensor(
            &[s[0], s[1], s[2], self.conv_in_w.shape()[0]],
            promote(z.dtype(), &[&self.conv_in_w, &self.conv_in_b]),
        )?;
        let x = price_resnet(crate::quantum::VAE_RESNET, &self.mid_resnet0, &x, out)?;
        let x = self
            .mid_attn
            .price(&x, crate::quantum::VAE_ATTENTION, out)?;
        price_resnet(crate::quantum::VAE_RESNET, &self.mid_resnet1, &x, out)
    }

fn price_conv_out_output(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.shape();
        tensor(
            &[s[0], s[1], s[2], self.conv_out_w.shape()[0]],
            promote(x.dtype(), &[&self.conv_out_w, &self.conv_out_b]),
        )
    }

fn price_tail(&self, x: &Tensor, out: &mut crate::quantum::Quanta) -> Result<Tensor> {
        let mut x = x.clone();
        for ub in &self.up_blocks {
            x = ub.price_forward(&x, crate::quantum::VAE_DECODE_LABELS, out)?;
        }
        out.push((crate::quantum::VAE_DECODE_OUT, self.conv_out_scratch(&x)));
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
        for up in &self.up_blocks {
            let current_edge = tile_edge_at(remaining_scale);
            let next_scale = if up.upsamples() {
                (remaining_scale / 2).max(1)
            } else {
                remaining_scale
            };
            x = up.price_tiled_decode(&x, current_edge, tile_edge_at(next_scale), out)?;
            remaining_scale = next_scale;
        }
        let (cin, cout) = conv_channels(&self.conv_out_w);
        out.push((
            crate::quantum::VAE_TILED_STAGE,
            crate::quantum::tiled_stage_bytes(
                x.shape(),
                cin,
                cout,
                element_bytes(&x),
                false,
            ),
        ));
        price_tiles(
            crate::quantum::VAE_TILE,
            &x,
            cin,
            &self.conv_out_w,
            output_tile_edge.max(3),
            false,
            true,
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
        let call = conv_call(
            x.shape(),
            element_bytes(x),
            &self.conv_in_w,
            1,
            1,
        );
        out.push((
            crate::quantum::VAE_ENCODE_IN,
            crate::quantum::single_conv_bytes(&call, 1, 1, crate::quantum::backend()),
        ));
        let s = x.shape();
        let mut x = tensor(
            &[s[0], s[1], s[2], self.conv_in_w.shape()[0]],
            promote(x.dtype(), &[&self.conv_in_w, &self.conv_in_b]),
        )?;
        for db in &self.down_blocks {
            x = db.price_forward(&x, crate::quantum::VAE_ENCODE_LABELS, out)?;
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
        price_tiles(
            crate::quantum::VAE_ENCODE_TILE,
            x,
            cin,
            &self.conv_in_w,
            tile.max(3),
            false,
            false,
            out,
        );
        let mut x = tensor(
            &[s[0], s[1], s[2], cout],
            promote(x.dtype(), &[&self.conv_in_w, &self.conv_in_b]),
        )?;
        let mut scale = 1;
        for db in &self.down_blocks {
            let edge = ((tile + scale - 1) / scale).max(3);
            for r in &db.resnets {
                x = r.price_tiled_vae(&x, edge, crate::quantum::VAE_ENCODE_LABELS, out)?;
            }
            if let Some((cw, cb)) = &db.downsample {
                scale *= 2;
                x = price_downsample_tiled(&x, cw, cb, ((tile + scale - 1) / scale).max(1), out)?;
            }
        }
        self.price_mid(&x, out)
    }

fn price_mid(&self, x: &Tensor, out: &mut crate::quantum::Quanta) -> Result<Tensor> {
        let x = price_resnet(crate::quantum::VAE_ENCODE_RESNET, &self.mid_resnet0, x, out)?;
        let x = self
            .mid_attn
            .price(&x, crate::quantum::VAE_ENCODE_ATTENTION, out)?;
        price_resnet(
            crate::quantum::VAE_ENCODE_RESNET,
            &self.mid_resnet1,
            &x,
            out,
        )
    }

fn out_scratch(&self, x: &Tensor) -> u64 {
        let call = conv_call(
            x.shape(),
            element_bytes(x),
            &self.conv_out_w,
            1,
            1,
        );
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
        let z = tensor(
            sh,
            linear_output_dtype(&self.post_quant_proj, z),
        )?;
        let pending_input =
            self.decode_input_retained_bytes(latents) - tensor_bytes(latents);
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
        Ok(())
    }

}

fn price_tiles(
    label: &'static str,
    x: &Tensor,
    channels: i32,
    weight: &Tensor,
    edge: i32,
    f32_accum: bool,
    normalized: bool,
    out: &mut crate::quantum::Quanta,
) {
    let s = x.shape();
    let (th, tw) = max_conv_tile(s[1], s[2], edge);
    out.push((
        label,
        tile_bytes(
            &[s[0], th, tw, channels],
            q::TileSource { full_width: s[2], normalized },
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
    price_tiles(
        crate::quantum::VAE_TILE,
        &upsampled,
        s[3],
        weight,
        tile,
        f32_accum,
        false,
        out,
    );
    let dtype = if f32_accum {
        x.dtype()
    } else {
        promote(x.dtype(), &[weight, bias])
    };
    tensor(&[up[0], up[1], up[2], cout], dtype)
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
    tensor(
        &[s[0], oh, ow, weight.shape()[0]],
        promote(x.dtype(), &[weight, bias]),
    )
}

fn price_resnet(
    label: &'static str,
    resnet: &ResnetFacts,
    x: &Tensor,
    out: &mut crate::quantum::Quanta,
) -> Result<Tensor> {
    out.push((label, resnet.quantum_scratch(x, None)));
    let s = x.shape();
    tensor(
        &[s[0], s[1], s[2], resnet.conv_weights().1.shape()[0]],
        resnet.output_dtype(x.dtype(), None),
    )
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

pub(super) fn attention_scratch(shape: &[i32], dtype: Dtype, plan: mlx_gen::attention::AttentionPlan<'_>) -> u64 {
    let tokens = shape[1] * shape[2];
    let rows = plan.budget.query_block(shape[0], 1, tokens, tokens);
    q::vae_attention_bytes(shape[0], tokens, shape[3], rows, q::dtype_bytes(dtype), q::backend())
}
pub(super) fn compute_dtype(loaded: Dtype, input: Dtype) -> Dtype {
    if loaded == Dtype::Float32 { input } else { loaded }
}
pub(super) fn input_retained_bytes(loaded: Dtype, latents: &Tensor) -> u64 {
    let caller = latents.bytes();
    let compute = compute_dtype(loaded, latents.dtype);
    caller + if compute == latents.dtype { 0 } else { latents.elements() * q::dtype_bytes(compute) }
}
impl AutoencoderFacts {
    pub(crate) fn from_headers(headers: &[SafetensorsTensorHeader], cfg: &VaeConfig, dtype: Dtype) -> Result<Self> {
        Self::from_weights(&HeaderWeights::new_all_cast(headers, dtype)?, cfg)
    }
}
