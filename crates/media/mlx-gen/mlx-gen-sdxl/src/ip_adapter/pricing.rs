//! Complete IP conditioning price hierarchy: pure loaded headers/activation facts, no native graph.
use super::ResamplerConfig;
use crate::quantum as q;
use crate::unet::pricing::{element_bytes, nchw_to_nhwc, promote, tensor, FactView, HeaderWeights};
use crate::vision_encoder::VisionConfig;
use mlx_gen::adapters::facts::TensorFacts as Tensor;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

pub(crate) struct VisionLayerFacts {
    pub(super) ln1_w: Tensor,
    pub(super) _ln1_b: Tensor,
    pub(super) _ln2_w: Tensor,
    pub(super) _ln2_b: Tensor,
    pub(super) q: q::LinearWork,
    pub(super) _k: q::LinearWork,
    pub(super) _v: q::LinearWork,
    pub(super) out: q::LinearWork,
    pub(super) fc1: q::LinearWork,
    pub(super) fc2: q::LinearWork,
    pub(super) num_heads: i32,
    pub(super) head_dim: i32,
}
pub(crate) struct VisionFacts {
    pub(super) patch_embedding: Tensor,
    pub(super) class_embedding: Tensor,
    pub(super) position_embedding: Tensor,
    pub(super) pre_ln_w: Tensor,
    pub(super) pre_ln_b: Tensor,
    pub(super) layers: Vec<VisionLayerFacts>,
    pub(super) patch: i32,
    pub(super) hidden: i32,
}
pub(crate) struct PerceiverFacts {
    pub(super) _norm1_w: Tensor,
    pub(super) _norm1_b: Tensor,
    pub(super) _norm2_w: Tensor,
    pub(super) _norm2_b: Tensor,
    pub(super) to_q: q::LinearWork,
    pub(super) _to_kv: q::LinearWork,
    pub(super) _to_out: q::LinearWork,
    pub(super) heads: i32,
    pub(super) dim_head: i32,
}
pub(crate) struct FeedForwardFacts {
    pub(super) _ln_w: Tensor,
    pub(super) _ln_b: Tensor,
    pub(super) fc1: q::LinearWork,
    pub(super) _fc2: q::LinearWork,
}
pub(crate) struct ResamplerFacts {
    pub(super) latents: Tensor,
    pub(super) proj_in: q::LinearWork,
    pub(super) proj_out: q::LinearWork,
    pub(super) norm_out_w: Tensor,
    pub(super) norm_out_b: Tensor,
    pub(super) layers: Vec<(PerceiverFacts, FeedForwardFacts)>,
    pub(super) dim: i32,
    pub(super) num_queries: i32,
}
pub(crate) struct IpImageEncoderFacts {
    encoder: VisionFacts,
    resampler: ResamplerFacts,
    image_size: usize,
}

fn linear_output_dtype(linear: &q::LinearWork, input: Dtype) -> Dtype {
    linear.output_dtype(input)
}

fn conv_call(shape: &[i32], act: u64, weight: &Tensor, stride: i32, pad: i32) -> q::ConvCall {
    q::ConvCall::from_metadata(
        shape,
        act,
        &weight.shape,
        element_bytes(weight),
        stride,
        pad,
    )
}

impl VisionLayerFacts {
    fn from_weights(w: &HeaderWeights, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let g = |name: &str| w.require(&format!("{prefix}.{name}")).cloned();
        let dense = |wn: &str, bn: &str| -> Result<q::LinearWork> {
            Ok(q::LinearWork::dense(
                w.require(&format!("{prefix}.{wn}"))?.clone(),
                Some(w.require(&format!("{prefix}.{bn}"))?.clone()),
            ))
        };
        let head_dim = cfg.hidden / cfg.num_heads;
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            _ln1_b: g("layer_norm1.bias")?,
            _ln2_w: g("layer_norm2.weight")?,
            _ln2_b: g("layer_norm2.bias")?,
            q: dense("self_attn.q_proj.weight", "self_attn.q_proj.bias")?,
            _k: dense("self_attn.k_proj.weight", "self_attn.k_proj.bias")?,
            _v: dense("self_attn.v_proj.weight", "self_attn.v_proj.bias")?,
            out: dense("self_attn.out_proj.weight", "self_attn.out_proj.bias")?,
            fc1: dense("mlp.fc1.weight", "mlp.fc1.bias")?,
            fc2: dense("mlp.fc2.weight", "mlp.fc2.bias")?,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }

    pub(crate) fn quantum_scratch(&self, x: &Tensor) -> u64 {
        vision_layer_scratch(
            x,
            self.ln1_w.shape[0] as u64,
            self.fc1.output_width(),
            self.num_heads,
            self.head_dim,
        )
    }

    pub(crate) fn output_dtype(&self, x: Dtype) -> Dtype {
        [&self.q, &self.out, &self.fc1, &self.fc2]
            .iter()
            .fold(x, |d, linear| {
                Dtype::from_promoting_types(d, linear_output_dtype(linear, d))
            })
    }
}

impl VisionFacts {
    pub fn from_weights(w: &HeaderWeights, cfg: &VisionConfig) -> Result<Self> {
        let p = "vision_model";
        let patch_nchw = w.require(&format!("{p}.embeddings.patch_embedding.weight"))?;
        let patch_embedding = nchw_to_nhwc(patch_nchw)?; // [O, kH, kW, in]
        let class_embedding = w
            .require(&format!("{p}.embeddings.class_embedding"))?
            .clone();
        let position_embedding = w
            .require(&format!("{p}.embeddings.position_embedding.weight"))?
            .clone();
        let pre_ln_w = w.require(&format!("{p}.pre_layrnorm.weight"))?.clone();
        let pre_ln_b = w.require(&format!("{p}.pre_layrnorm.bias"))?.clone();
        let layers = (0..cfg.num_layers)
            .map(|i| VisionLayerFacts::from_weights(w, &format!("{p}.encoder.layers.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            pre_ln_w,
            pre_ln_b,
            layers,
            patch: cfg.patch,
            hidden: cfg.hidden,
        })
    }

    pub(crate) fn embed_scratch(&self, pixel_values: &Tensor) -> u64 {
        vision_embedding_scratch(pixel_values, &self.patch_embedding, self.patch, self.hidden)
    }

    pub(crate) fn price_penultimate(
        &self,
        pixel_values: &Tensor,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        out.push((
            crate::quantum::VISION_EMBEDDINGS,
            self.embed_scratch(pixel_values),
        ));
        let s = pixel_values.shape();
        let tokens = (s[1] / self.patch) * (s[2] / self.patch) + 1;
        let embedded = promote(
            pixel_values.dtype(),
            &[
                &self.patch_embedding,
                &self.class_embedding,
                &self.position_embedding,
                &self.pre_ln_w,
                &self.pre_ln_b,
            ],
        );
        let mut x = tensor(&[s[0], tokens, self.hidden], embedded)?;
        let keep = self.layers.len().saturating_sub(1);
        for layer in &self.layers[..keep] {
            out.push((crate::quantum::VISION_LAYER, layer.quantum_scratch(&x)));
            x = tensor(x.shape(), layer.output_dtype(x.dtype()))?;
        }
        Ok(x)
    }
}

impl PerceiverFacts {
    fn from_weights(w: &HeaderWeights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        let g = |name: &str| w.require(&format!("{prefix}.{name}")).cloned();
        let lin = |name: &str| -> Result<q::LinearWork> {
            Ok(q::LinearWork::dense(
                w.require(&format!("{prefix}.{name}"))?.clone(),
                None,
            ))
        };
        Ok(Self {
            _norm1_w: g("norm1.weight")?,
            _norm1_b: g("norm1.bias")?,
            _norm2_w: g("norm2.weight")?,
            _norm2_b: g("norm2.bias")?,
            to_q: lin("to_q.weight")?,
            _to_kv: lin("to_kv.weight")?,
            _to_out: lin("to_out.weight")?,
            heads: cfg.heads,
            dim_head: cfg.dim_head,
        })
    }
}

impl FeedForwardFacts {
    fn from_weights(w: &HeaderWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            _ln_w: w.require(&format!("{prefix}.0.weight"))?.clone(),
            _ln_b: w.require(&format!("{prefix}.0.bias"))?.clone(),
            fc1: q::LinearWork::dense(w.require(&format!("{prefix}.1.weight"))?.clone(), None),
            _fc2: q::LinearWork::dense(w.require(&format!("{prefix}.3.weight"))?.clone(), None),
        })
    }
}

impl ResamplerFacts {
    pub fn dtype(&self) -> Dtype {
        self.latents.dtype()
    }

    pub fn output_dim(&self) -> i32 {
        self.norm_out_w.shape()[0]
    }

    pub fn from_weights(w: &HeaderWeights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        let latents = w.require(&format!("{prefix}.latents"))?.clone();
        let proj_in = q::LinearWork::dense(
            w.require(&format!("{prefix}.proj_in.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.proj_in.bias"))?.clone()),
        );
        let proj_out = q::LinearWork::dense(
            w.require(&format!("{prefix}.proj_out.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.proj_out.bias"))?.clone()),
        );
        let layers = (0..cfg.depth)
            .map(|i| -> Result<_> {
                let attn = PerceiverFacts::from_weights(w, &format!("{prefix}.layers.{i}.0"), cfg)?;
                let ff = FeedForwardFacts::from_weights(w, &format!("{prefix}.layers.{i}.1"))?;
                Ok((attn, ff))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            latents,
            proj_in,
            proj_out,
            norm_out_w: w.require(&format!("{prefix}.norm_out.weight"))?.clone(),
            norm_out_b: w.require(&format!("{prefix}.norm_out.bias"))?.clone(),
            layers,
            dim: cfg.dim,
            num_queries: cfg.num_queries,
        })
    }

    pub(crate) fn price_forward(
        &self,
        image_features: &Tensor,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let b = image_features.shape()[0];
        validate_latents(self.latents.shape(), self.num_queries, self.dim)?;
        let nx = image_features.shape()[1];
        out.push((
            crate::quantum::RESAMPLER_PROJ_IN,
            resampler_in_scratch(
                b,
                nx,
                element_bytes(&self.latents),
                self.proj_in.output_width(),
            ),
        ));
        for (attn, ff) in &self.layers {
            out.push((
                crate::quantum::RESAMPLER_LAYER,
                self.layer_scratch(attn, ff, b, nx),
            ));
        }
        out.push((
            crate::quantum::RESAMPLER_PROJ_OUT,
            self.proj_out_scratch(b, self.latents.dtype()),
        ));
        tensor(&[b, self.num_queries, self.output_dim()], self.dtype())
    }

    pub(crate) fn proj_out_scratch(&self, b: i32, input: Dtype) -> u64 {
        resampler_out_scratch(
            b,
            self.num_queries,
            self.dim,
            input,
            &self.proj_out,
            &self.norm_out_w,
            &self.norm_out_b,
        )
    }

    pub(crate) fn layer_scratch(
        &self,
        attn: &PerceiverFacts,
        ff: &FeedForwardFacts,
        b: i32,
        nx: i32,
    ) -> u64 {
        resampler_layer_scratch(
            element_bytes(&self.latents),
            self.num_queries,
            self.dim,
            attn.to_q.output_width(),
            ff.fc1.output_width(),
            attn.heads,
            attn.dim_head,
            b,
            nx,
        )
    }
}

impl IpImageEncoderFacts {
    pub fn with_image_size(
        encoder: VisionFacts,
        resampler: ResamplerFacts,
        image_size: usize,
    ) -> Self {
        Self {
            encoder,
            resampler,
            image_size,
        }
    }

    pub(crate) fn price_tokens(&self, out: &mut crate::quantum::Quanta) -> Result<Tensor> {
        let size = self.image_size as i32;
        let pixels = tensor(&[1, size, size, 3], self.resampler.dtype())?;
        let penultimate = self.encoder.price_penultimate(&pixels, out)?;
        self.resampler.price_forward(&penultimate, out)
    }
}

pub(crate) fn cfg_pair(tokens: &Tensor, enabled: bool) -> Result<Tensor> {
    if !enabled {
        return Ok(tokens.clone());
    }
    let mut shape = tokens.shape.clone();
    shape[0] = shape[0]
        .checked_mul(2)
        .ok_or_else(|| Error::Msg("sdxl IP CFG token row overflow".into()))?;
    tensor(&shape, tokens.dtype)
}

pub(crate) fn vision_layer_scratch(
    x: &Tensor,
    hidden: u64,
    ff: u64,
    heads: i32,
    head_dim: i32,
) -> u64 {
    q::clip_layer_bytes(
        x.shape[0],
        x.shape[1],
        hidden,
        ff,
        heads,
        head_dim,
        false,
        element_bytes(x),
        q::backend(),
    )
}
pub(crate) fn vision_embedding_scratch(
    pixels: &Tensor,
    weight: &Tensor,
    patch: i32,
    hidden: i32,
) -> u64 {
    let s = pixels.shape();
    let tokens = (s[1] / patch) * (s[2] / patch) + 1;
    let act = element_bytes(pixels);
    let call = conv_call(s, act, weight, patch, 0);
    let backend = q::backend();
    let units = 1 + q::layer_norm_buffers(backend);
    q::single_conv_bytes(&call, 0, 0, backend)
        .saturating_add(mlx_gen::ios_quantum::tensor_bytes(
            act,
            &[units, s[0] as u64, tokens as u64, hidden as u64],
        ))
        .saturating_add(q::page_slack(units))
}
pub(crate) fn validate_latents(shape: &[i32], queries: i32, dim: i32) -> Result<()> {
    if shape != [1, queries, dim] {
        return Err(Error::Msg(format!(
            "resampler latents shape {:?} != [1, {}, {}]",
            shape, queries, dim
        )));
    }
    Ok(())
}
pub(crate) fn resampler_in_scratch(b: i32, nx: i32, act: u64, width: u64) -> u64 {
    2 * mlx_gen::ios_quantum::tensor_bytes(act, &[b as u64, nx as u64, width])
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn resampler_layer_scratch(
    act: u64,
    queries: i32,
    dim: i32,
    projected_width: u64,
    ff_width: u64,
    heads: i32,
    head_dim: i32,
    batch: i32,
    image_tokens: i32,
) -> u64 {
    let (b, nq) = (batch as u64, queries as u64);
    let s = image_tokens as u64 + nq;
    let width = (dim as u64).max(projected_width);
    let scores = q::scores(q::backend(), head_dim, queries, act);
    let fallback = u64::from(matches!(
        scores,
        mlx_gen::ios_quantum::Scores::Materialized { .. }
    ));
    let ln = q::layer_norm_buffers(q::backend());
    let key_units = 6 + 2 * fallback;
    let query_units = 5 + 3 * ln + fallback;
    mlx_gen::ios_quantum::tensor_bytes(act, &[key_units, b, s, width])
        .saturating_add(mlx_gen::ios_quantum::tensor_bytes(
            act,
            &[b, nq, query_units * width + 2 * ff_width],
        ))
        .saturating_add(q::page_slack(key_units + query_units + 4))
        .saturating_add(mlx_gen::ios_quantum::scores_bytes(
            scores,
            b,
            heads as u64,
            nq,
            s,
        ))
}
pub(crate) fn resampler_out_scratch(
    b: i32,
    queries: i32,
    dim: i32,
    input: Dtype,
    projection: &q::LinearWork,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
) -> u64 {
    let width = norm_weight.shape[0];
    let projected = projection.output_dtype(input);
    let shape = [b, queries, width];
    let output = mlx_gen::ios_quantum::tensor_bytes(
        q::dtype_bytes(projected),
        &[b as u64, queries as u64, width as u64],
    );
    let scratch = projection.operand_bytes(&[b, queries, dim], input) + output;
    scratch
        .max(output + q::final_norm_bytes_from_metadata(&shape, projected, norm_weight, norm_bias))
}

impl IpImageEncoderFacts {
    pub(crate) fn preprocess_geometry(
        &self,
        source: crate::sampling::ReferenceImageFacts,
    ) -> Result<super::ClipPreprocessGeometry> {
        super::preprocess_geometry(
            source.width,
            source.height,
            source.buffer_bytes,
            self.image_size,
        )
    }
}
