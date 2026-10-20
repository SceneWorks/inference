//! CLIP text encoders — a Rust port of the vendored `_vendor/mlx_sd/clip.py` (`CLIPTextModel` +
//! `CLIPEncoderLayer`). SDXL conditions on **two** of these: CLIP-L (`text_encoder`, 768-wide, no
//! projection) and OpenCLIP-bigG (`text_encoder_2`, 1280-wide, with a final projection feeding the
//! pooled micro-conditioning). Both run **fp16**, matching the production reference
//! (`StableDiffusionXL(float16=True)`); `load_text_encoder_*_dtype` keeps an f32 path for stage gates.
//!
//! The SDXL conditioning is `concat(te1.hidden_states[-2], te2.hidden_states[-2])` (the
//! penultimate-layer hidden states, BEFORE the final layer-norm) over the full padded sequence, plus
//! `te2.pooled` (the projected EOS token of the final-layer-norm'd output). The encoder applies a
//! causal mask only — there is no padding mask, matching the reference (padding positions flow into
//! the UNet cross-attention).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::add;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::host_i32;
use mlx_gen::nn::{gelu_exact, gelu_quick};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{ClipActivation, ClipTextConfig};

/// CLIP's LayerNorm epsilon (HF `layer_norm_eps`, mlx `nn.LayerNorm` default).
const LN_EPS: f32 = 1e-5;

/// The six actual per-layer Linear names. Constructor routing and header pack projection share
/// this table; token/position embeddings, norms and ordinary biases are not packing targets.
const LAYER_LINEAR_NAMES: [&str; 6] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
];
const TEXT_PROJECTION: &str = "text_projection";

/// `(path, has_dense_bias)` of exactly the actual CLIP quantization walk.
pub(crate) fn quantizable_text_paths(cfg: &ClipTextConfig) -> Vec<(String, bool)> {
    let mut paths = Vec::new();
    for layer in 0..cfg.num_layers {
        for name in LAYER_LINEAR_NAMES {
            paths.push((format!("text_model.encoder.layers.{layer}.{name}"), true));
        }
    }
    if cfg.projection_dim.is_some() {
        paths.push((TEXT_PROJECTION.to_owned(), false));
    }
    paths
}

/// One actual projection-pack point. These declared logical owners can overlap native storage;
/// they are neither an allocation identity nor a subtractable physical lower bound.
pub(crate) struct TextPackPoint {
    pub before_resident: u64,
    pub scratch: u64,
}

pub(crate) struct TextPackFacts {
    pub retained: u64,
    pub points: Vec<TextPackPoint>,
}

/// Header-only post-pack declared owners and EACH coincident raw pack budget. Never combine an
/// earlier owner set with a future final maximum: page-rounded tiny packs can initially grow.
/// No native payload/readiness/allocator query or physical lower/freshness proof. The pre-pack
/// read/cast footprint stays separate at the caller; pending raw/cast parents must not be canceled
/// merely because declared packed outputs are smaller.
pub(crate) fn text_pack_facts(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    cfg: &ClipTextConfig,
    bits: Option<i32>,
) -> Result<TextPackFacts> {
    use mlx_gen::ios_quantum::allocated_bytes;
    let overflow = || Error::Msg("sdxl text pack header byte extent overflow".into());
    let resident = |h: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader| -> Result<u64> {
        let source = crate::quant::header_dtype(h.dtype)?;
        let loaded = crate::quant::loaded_header_dtype(headers, h)?;
        let bytes = if source == loaded {
            h.data_bytes
        } else {
            h.materialized_bytes(crate::quantum::dtype_bytes(loaded))
                .map_err(|e| Error::Msg(e.to_string()))?
        };
        Ok(allocated_bytes(bytes))
    };
    let mut retained = 0u64;
    for header in headers {
        retained = retained
            .checked_add(resident(header)?)
            .ok_or_else(overflow)?;
    }
    let Some(bits) = bits else {
        return Ok(TextPackFacts {
            retained,
            points: Vec::new(),
        });
    };
    let bit_width = u64::try_from(bits)
        .ok()
        .filter(|b| *b != 0)
        .ok_or_else(|| Error::Msg("sdxl text pack requires positive bits".into()))?;
    let mut points = Vec::new();
    for (path, has_bias) in quantizable_text_paths(cfg) {
        let linear = crate::quant::linear_from_headers(headers, &path)?;
        if linear.source_packed_bits.is_some() {
            continue;
        } // Exact existing quantize() no-op.
        let [output, input] = linear.shape.as_slice() else {
            return Err(Error::Msg("sdxl text pack requires a matrix".into()));
        };
        let (output, input) = (
            u64::try_from(*output).map_err(|_| overflow())?,
            u64::try_from(*input).map_err(|_| overflow())?,
        );
        let group = crate::quant::GROUP_SIZE as u64;
        if output == 0 || input == 0 || input % group != 0 {
            return Err(Error::Msg(
                "sdxl text pack input is not group-aligned".into(),
            ));
        }
        let weight_name = format!("{path}.weight");
        let weight = headers
            .iter()
            .find(|h| h.name == weight_name)
            .ok_or_else(|| Error::MissingTensor(weight_name))?;
        let bias_bytes = if has_bias {
            let name = format!("{path}.bias");
            headers
                .iter()
                .find(|h| h.name == name)
                .ok_or_else(|| Error::MissingTensor(name))?
                .materialized_bytes(2)
                .map_err(|e| Error::Msg(e.to_string()))?
        } else {
            0
        };
        let elements = output.checked_mul(input).ok_or_else(overflow)?;
        points.push(TextPackPoint {
            before_resident: retained,
            scratch: crate::quant::quantize_bytes_from_metadata(elements, bias_bytes, bits),
        });
        // QuantizedLinear's codes plus separate BF16 scales/affine biases, each page-rounded.
        let code_row = input
            .checked_mul(bit_width)
            .ok_or_else(overflow)?
            .div_ceil(32)
            .checked_mul(4)
            .ok_or_else(overflow)?;
        let codes = output.checked_mul(code_row).ok_or_else(overflow)?;
        let companion = output
            .checked_mul(input / group)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        retained = retained
            .checked_sub(resident(weight)?)
            .ok_or_else(overflow)?
            .checked_add(allocated_bytes(codes))
            .and_then(|n| n.checked_add(allocated_bytes(companion)))
            .and_then(|n| n.checked_add(allocated_bytes(companion)))
            .ok_or_else(overflow)?;
    }
    Ok(TextPackFacts { retained, points })
}

/// Complete CLIP pricing component, built from file headers without a native weight/model graph.
/// Facts describe logical operands only: no layout normalization here grants backing/release credit.
pub(crate) struct ClipPrice {
    token_embedding: mlx_gen::adapters::facts::TensorFacts,
    position_embedding: mlx_gen::adapters::facts::TensorFacts,
    layers: Vec<ClipLayerPrice>,
    final_ln_w: mlx_gen::adapters::facts::TensorFacts,
    final_ln_b: mlx_gen::adapters::facts::TensorFacts,
    text_projection: Option<crate::quantum::LinearWork>,
}

fn clip_embeddings_scratch(b: i32, n: i32, dim: i32, dtype: Dtype) -> u64 {
    3 * mlx_gen::ios_quantum::tensor_bytes(crate::quantum::dtype_bytes(dtype),
        &[b as u64, n as u64, dim as u64])
}

struct ClipLayerPrice {
    ln1_w: mlx_gen::adapters::facts::TensorFacts,
    // Actual constructor/pack order: q, k, v, out, fc1, fc2.
    projections: [crate::quantum::LinearWork; 6],
    num_heads: i32,
    head_dim: i32,
}

pub(crate) fn clip_header_tensor(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader], name: &str,
) -> Result<mlx_gen::adapters::facts::TensorFacts> {
    let header = headers.iter().find(|h| h.name == name)
        .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
    let shape = header.shape.iter().map(|&d| i32::try_from(d).map_err(|_| Error::Msg(
        "sdxl CLIP header dimension exceeds native i32".into()))).collect::<Result<Vec<_>>>()?;
    Ok(mlx_gen::adapters::facts::TensorFacts {
        shape, dtype: crate::quant::loaded_header_dtype(headers, header)?, strides: Vec::new(),
    })
}

/// Exact shared quant::lin lookup order: ordinary bias first, scales selector, weight + packed
/// validator, then affine biases. Dense weights gain no new exact-rank/shape/dtype restriction.
pub(crate) fn clip_header_linear(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader], base: &str, biased: bool,
) -> Result<crate::quantum::LinearWork> {
    use mlx_gen::adapters::LinearFacts;
    let bias = if biased { Some(clip_header_tensor(headers, &format!("{base}.bias"))?) } else { None };
    let packed = headers.iter().any(|h| h.name == format!("{base}.scales"));
    let weight = clip_header_tensor(headers, &format!("{base}.weight"))?;
    let (weights, shape, bits) = if packed {
        let scales = clip_header_tensor(headers, &format!("{base}.scales"))?;
        let bits = mlx_gen::quant::packed_bits_from_metadata(
            &weight.shape, weight.dtype, &scales.shape, scales.dtype, crate::quant::GROUP_SIZE)?;
        let offsets = clip_header_tensor(headers, &format!("{base}.biases"))?;
        // The shared validator already checked the exact 2-D native packed geometry and overflow.
        let shape = vec![scales.shape[0], scales.shape[1] * crate::quant::GROUP_SIZE];
        (vec![weight, scales, offsets], shape, Some(bits))
    } else {
        let shape = weight.shape.clone();
        (vec![weight], shape, None)
    };
    Ok(crate::quantum::LinearWork {
        base: LinearFacts { base_shape: shape, is_quantized: bits.is_some(),
            has_live_adapters: false, adapter_count: 0,
            bias_shape: bias.as_ref().map(|b| b.shape.clone()) },
        weights, bias, residuals: Vec::new(), group_size: crate::quant::GROUP_SIZE, bits,
    })
}

impl ClipLayerPrice {
    fn of(layer: &ClipEncoderLayer) -> Self {
        Self {
            ln1_w: mlx_gen::adapters::facts::TensorFacts::of(&layer.ln1_w),
            projections: [&layer.q, &layer.k, &layer.v, &layer.out, &layer.linear1, &layer.linear2]
                .map(crate::quantum::LinearWork::of),
            num_heads: layer.num_heads, head_dim: layer.head_dim,
        }
    }

    fn output_dtype(&self, x: Dtype) -> Dtype {
        [0, 3, 4, 5].iter().fold(x, |d, &index| {
            Dtype::from_promoting_types(d, self.projections[index].output_dtype(d))
        })
    }

    fn quantum_scratch(&self, b: i32, n: i32, x: Dtype) -> u64 {
        let act = crate::quantum::dtype_bytes(self.output_dtype(x));
        let casts = if act > crate::quantum::dtype_bytes(x) {
            mlx_gen::ios_quantum::tensor_bytes(
                act, &[3, b as u64, n as u64, self.ln1_w.shape[0] as u64],
            ).saturating_add(crate::quantum::page_slack(3))
        } else { 0 };
        casts.saturating_add(crate::quantum::clip_layer_bytes(
            b, n, self.ln1_w.shape[0] as u64, self.projections[4].output_width(),
            self.num_heads, self.head_dim, true, act, crate::quantum::backend(),
        ))
    }
}

fn clip_final_scratch(
    b: i32, n: i32, dim: i32, input: Dtype,
    weight: &mlx_gen::adapters::facts::TensorFacts,
    bias: &mlx_gen::adapters::facts::TensorFacts,
    projection: Option<&crate::quantum::LinearWork>,
) -> u64 {
    use crate::quantum as q;
    let normed = Dtype::from_promoting_types(
        Dtype::from_promoting_types(input, weight.dtype), bias.dtype);
    let row = mlx_gen::ios_quantum::tensor_bytes(q::dtype_bytes(normed), &[b as u64, dim as u64]);
    let projection = projection.map_or(0, |p| {
        p.operand_bytes(&[b, dim], normed) + mlx_gen::ios_quantum::tensor_bytes(
            q::dtype_bytes(p.output_dtype(normed)), &[b as u64, p.output_width()],
        )
    });
    let pool = row * n as u64 + row + b as u64 * 4 + projection;
    q::final_norm_bytes_from_metadata(&[b, n, dim], input, weight, bias).max(pool)
}

impl ClipPrice {
    pub(crate) fn from_headers(
        headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
        prefix: &str, cfg: &ClipTextConfig,
    ) -> Result<Self> {
        // The native safetensors decoder visits all source dtypes before constructor lookups,
        // even unused keys. This is the same decoder acceptance, not an extra precision policy.
        for header in headers { crate::quant::header_dtype(header.dtype)?; }
        let tensor = |suffix: &str| clip_header_tensor(headers, &format!("{prefix}.{suffix}"));
        let token_embedding = tensor("embeddings.token_embedding.weight")?;
        let position_embedding = tensor("embeddings.position_embedding.weight")?;
        let mut layers = Vec::new();
        for index in 0..cfg.num_layers {
            let layer = format!("{prefix}.encoder.layers.{index}");
            let head_dim = cfg.model_dims / cfg.num_heads;
            let ln1_w = clip_header_tensor(headers, &format!("{layer}.layer_norm1.weight"))?;
            // These native constructor requirements remain authoritative even though their shapes
            // are not consumed by the existing layer budget. Never price a missing norm silently.
            clip_header_tensor(headers, &format!("{layer}.layer_norm1.bias"))?;
            clip_header_tensor(headers, &format!("{layer}.layer_norm2.weight"))?;
            clip_header_tensor(headers, &format!("{layer}.layer_norm2.bias"))?;
            let linear = |index| clip_header_linear(headers,
                &format!("{layer}.{}", LAYER_LINEAR_NAMES[index]), true);
            layers.push(ClipLayerPrice { ln1_w,
                projections: [linear(0)?, linear(1)?, linear(2)?, linear(3)?, linear(4)?, linear(5)?],
                num_heads: cfg.num_heads, head_dim });
        }
        let final_ln_w = tensor("final_layer_norm.weight")?;
        let final_ln_b = tensor("final_layer_norm.bias")?;
        let text_projection = if cfg.projection_dim.is_some() {
            Some(clip_header_linear(headers, TEXT_PROJECTION, false)?)
        } else { None };
        Ok(Self { token_embedding, position_embedding, layers, final_ln_w, final_ln_b, text_projection })
    }

    /// Both encoder constructors precede this pair's ordered pack, exactly as the actual loader.
    pub(crate) fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            for projection in &mut layer.projections { projection.pack(bits)?; }
        }
        if let Some(projection) = &mut self.text_projection { projection.pack(bits)?; }
        Ok(())
    }

    pub(crate) fn pooled_width(&self) -> i32 {
        match &self.text_projection {
            Some(projection) => projection.output_width() as i32,
            None => self.token_embedding.shape[1],
        }
    }

    pub(crate) fn price_forward(&self, b: i32, n: i32, out: &mut crate::quantum::Quanta) -> (Dtype, Dtype) {
        let dim = self.token_embedding.shape[1];
        out.push((crate::quantum::CLIP_EMBEDDINGS,
            clip_embeddings_scratch(b, n, dim, self.token_embedding.dtype)));
        let mut x = Dtype::from_promoting_types(self.token_embedding.dtype, self.position_embedding.dtype);
        let mut hidden_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            out.push((crate::quantum::CLIP_LAYER, layer.quantum_scratch(b, n, x)));
            x = layer.output_dtype(x);
            hidden_states.push(x);
        }
        out.push((crate::quantum::CLIP_POOLED, clip_final_scratch(
            b, n, dim, x, &self.final_ln_w, &self.final_ln_b, self.text_projection.as_ref())));
        let normed = Dtype::from_promoting_types(
            Dtype::from_promoting_types(x, self.final_ln_w.dtype), self.final_ln_b.dtype);
        let pooled = self.text_projection.as_ref().map_or(normed, |p| p.output_dtype(normed));
        let penultimate = hidden_states.len().checked_sub(2).map_or(x, |i| hidden_states[i]);
        (penultimate, pooled)
    }

    #[cfg(test)]
    pub(crate) fn projection_facts(&self) -> Vec<(String, crate::quantum::LinearWork)> {
        let mut result = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, facts) in LAYER_LINEAR_NAMES.iter().zip(&layer.projections) {
                result.push((format!("text_model.encoder.layers.{index}.{name}"), facts.clone()));
            }
        }
        if let Some(projection) = &self.text_projection {
            result.push((TEXT_PROJECTION.to_owned(), projection.clone()));
        }
        result
    }
}

/// The outputs the SDXL pipeline reads off a CLIP encoder.
pub struct ClipOutput {
    /// Final-layer-norm'd output `[B, N, D]`.
    pub last_hidden_state: Array,
    /// Per-encoder-layer outputs (before the final layer-norm); `hidden_states[-2]` feeds the
    /// SDXL conditioning.
    pub hidden_states: Vec<Array>,
    /// The (optionally projected) EOS-token pooled output `[B, proj_or_D]`.
    pub pooled: Array,
}

/// One CLIP transformer encoder layer (pre-norm self-attention + pre-norm MLP, both residual).
struct ClipEncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    ln2_w: Array,
    ln2_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
    linear1: AdaptableLinear,
    linear2: AdaptableLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    act: ClipActivation,
}

impl ClipEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ClipTextConfig) -> Result<Self> {
        let g = |name: &str| w.require(&format!("{prefix}.{name}")).cloned();
        // Packed-detect (sc-8746): every per-layer Linear (q/k/v/out + mlp fc1/fc2) is quantized in
        // [`Self::quantize`], all biased — `crate::quant::lin` loads packed-or-dense. The token /
        // position embeddings (below, on `ClipTextEncoder`) stay dense (gather lookups).
        let lin = |base: &str| crate::quant::lin(w, &format!("{prefix}.{base}"), true);
        let head_dim = cfg.model_dims / cfg.num_heads;
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            ln1_b: g("layer_norm1.bias")?,
            ln2_w: g("layer_norm2.weight")?,
            ln2_b: g("layer_norm2.bias")?,
            q: lin(LAYER_LINEAR_NAMES[0])?,
            k: lin(LAYER_LINEAR_NAMES[1])?,
            v: lin(LAYER_LINEAR_NAMES[2])?,
            out: lin(LAYER_LINEAR_NAMES[3])?,
            linear1: lin(LAYER_LINEAR_NAMES[4])?,
            linear2: lin(LAYER_LINEAR_NAMES[5])?,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            act: cfg.hidden_act,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for (projection, lin) in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.out,
            &mut self.linear1,
            &mut self.linear2,
        ]
        .into_iter()
        .enumerate()
        {
            mlx_gen::ios_quantum::diagnostic_metadata(
                "sdxl",
                "clip-pack-projection",
                &[
                    ("schema", 1),
                    ("projection", projection as u64),
                    ("pooled_projection", 0),
                ],
            );
            crate::quant::quantize_linear(lin, bits)?;
        }
        Ok(())
    }

    fn materialize_weights(&self) -> Result<()> {
        mlx_rs::transforms::eval([&self.ln1_w, &self.ln1_b, &self.ln2_w, &self.ln2_b])?;
        for linear in [
            &self.q,
            &self.k,
            &self.v,
            &self.out,
            &self.linear1,
            &self.linear2,
        ] {
            linear.materialize_weights()?;
        }
        Ok(())
    }

    fn activation(&self, x: &Array) -> Result<Array> {
        match self.act {
            // CLIP-L "quick_gelu" = `x · sigmoid(1.702·x)` — the vendored `mlx.nn.gelu_fast_approx`
            // (NOT mlx-rs `gelu_fast_approximate`, which uses 1.773). The core `gelu_quick` matches it
            // byte-for-byte: dtype-weak `1.702` (an f32 scalar promotes f16→f32) + `mx.compile` (the
            // fused fp16 rounding — see `gelu_exact`/sc-2721). CLIP-bigG uses exact `gelu`.
            ClipActivation::QuickGelu => gelu_quick(x),
            ClipActivation::Gelu => gelu_exact(x),
        }
    }

    /// iOS quantum scratch (sc-26270) of this layer over `b` rows of `n` tokens of `x`-typed
    /// input, from the loaded widths: hidden from `layer_norm1`, MLP width from `fc1`,
    /// heads/head_dim from the config ([`crate::quantum::clip_layer_bytes`]), at the activation
    /// width the layer's projections promote `x` to (a packed snapshot's BF16 scales make an f16
    /// layer's every projection output and its residual f32 —
    /// [`crate::quantum::linear_output_dtype`]). The causal mask is an additive array mask, which
    /// MLX's fused Metal SDPA accepts at head dim 64.
    fn quantum_scratch(&self, b: i32, n: i32, x: Dtype) -> u64 {
        ClipLayerPrice::of(self).quantum_scratch(b, n, x)
    }

    /// `x`: `[B, N, D]`; `mask`: additive causal `[1, 1, N, N]`.
    fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
        // Self-attention (pre-norm, residual).
        let y = layer_norm(x, Some(&self.ln1_w), Some(&self.ln1_b), LN_EPS)?;
        let y = self.attention(&y, mask)?;
        let x = add(x, &y)?;
        // sc-26270: on iOS the attention half is materialized before the MLP is built, bounding the
        // layer quantum to its wider half (`quantum::clip_layer_bytes`); values are unchanged and
        // off device the graph is the historical one.
        if mlx_gen::ios_quantum::enabled() {
            x.eval()?;
        }

        // MLP (pre-norm, residual).
        let y = layer_norm(&x, Some(&self.ln2_w), Some(&self.ln2_b), LN_EPS)?;
        let y = self.linear1.forward(&y)?;
        let y = self.activation(&y)?;
        let y = self.linear2.forward(&y)?;
        Ok(add(&x, &y)?)
    }

    fn attention(&self, x: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, n, self.num_heads * self.head_dim])?;
        self.out.forward(&o)
    }
}

/// A loaded CLIP text encoder.
pub struct ClipTextEncoder {
    token_embedding: Array,
    position_embedding: Array,
    layers: Vec<ClipEncoderLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
    /// TE2 only (`CLIPTextModelWithProjection`): projects the pooled EOS token. Bias-free.
    text_projection: Option<AdaptableLinear>,
}

/// The actual pair's post-load packing policy, shared by snapshot/LDM loads and raw pricing.
/// Only the existing `quantize` walk selects weights: embeddings/norms/bias exceptions are not
/// replaced by a header-name approximation. Header pricing mirrors this ordered base-only policy
/// without invoking a native quantizer. Runtime keeps its existing per-projection materialization
/// and load pack before-checks, not a new evaluation or precision rule.
pub(crate) fn quantize_pair(
    te1: &mut ClipTextEncoder,
    te2: &mut ClipTextEncoder,
    bits: i32,
) -> Result<()> {
    let result = crate::quant::guarded_load_quantize(|| {
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "clip-pack-encoder",
            &[
                ("schema", 1),
                ("encoder", 0),
                ("bits", bits.max(0) as u64),
                ("active", 1),
            ],
        );
        te1.quantize(bits)?;
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "clip-pack-encoder",
            &[
                ("schema", 1),
                ("encoder", 1),
                ("bits", bits.max(0) as u64),
                ("active", 1),
            ],
        );
        te2.quantize(bits)
    });
    mlx_gen::ios_quantum::diagnostic_metadata(
        "sdxl",
        "clip-pack-encoder",
        &[
            ("schema", 1),
            ("encoder", 1),
            ("bits", bits.max(0) as u64),
            ("active", 0),
            ("completed", u64::from(result.is_ok())),
        ],
    );
    result
}

impl ClipTextEncoder {
    /// Load from a CLIP-text checkpoint. `prefix` is the encoder namespace (`text_model`); the
    /// projection (TE2) is read from the bare top-level `text_projection.weight`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &ClipTextConfig) -> Result<Self> {
        let token_embedding = w
            .require(&format!("{prefix}.embeddings.token_embedding.weight"))?
            .clone();
        let position_embedding = w
            .require(&format!("{prefix}.embeddings.position_embedding.weight"))?
            .clone();
        let layers = (0..cfg.num_layers)
            .map(|i| {
                ClipEncoderLayer::from_weights(w, &format!("{prefix}.encoder.layers.{i}"), cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        let final_ln_w = w
            .require(&format!("{prefix}.final_layer_norm.weight"))?
            .clone();
        let final_ln_b = w
            .require(&format!("{prefix}.final_layer_norm.bias"))?
            .clone();
        // TE2's `text_projection` (bias-free) is quantized — packed-detect (sc-8746). The bare
        // top-level key (no `text_model.` prefix) matches the on-disk layout the converter packs.
        let text_projection = match cfg.projection_dim {
            Some(_) => Some(crate::quant::lin(w, TEXT_PROJECTION, false)?),
            None => None,
        };
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_ln_w,
            final_ln_b,
            text_projection,
        })
    }

    /// The pooled output's width: the projection's rows (TE2), else the hidden width.
    #[cfg(test)]
    pub(crate) fn pooled_width(&self) -> i32 {
        match &self.text_projection {
            Some(p) => crate::quantum::linear_out(p) as i32,
            None => self.token_embedding.shape()[1],
        }
    }

    /// Quantize every Linear (the per-layer projections + the text projection) to Q4/Q8. The token /
    /// position embeddings stay dense (gather lookups, not matmuls).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for (layer, l) in self.layers.iter_mut().enumerate() {
            mlx_gen::ios_quantum::diagnostic_metadata(
                "sdxl",
                "clip-pack-layer",
                &[("schema", 1), ("layer", layer as u64)],
            );
            l.quantize(bits)?;
        }
        if let Some(p) = &mut self.text_projection {
            mlx_gen::ios_quantum::diagnostic_metadata(
                "sdxl",
                "clip-pack-projection",
                &[("schema", 1), ("projection", 0), ("pooled_projection", 1)],
            );
            crate::quant::quantize_linear(p, bits)?;
        }
        Ok(())
    }

    pub(crate) fn materialize_weights(&self) -> Result<()> {
        mlx_rs::transforms::eval([
            &self.token_embedding,
            &self.position_embedding,
            &self.final_ln_w,
            &self.final_ln_b,
        ])?;
        for layer in &self.layers {
            layer.materialize_weights()?;
        }
        if let Some(projection) = &self.text_projection {
            projection.materialize_weights()?;
        }
        Ok(())
    }

    /// Borrowed native shape/dtype observations for fixture proofs, never a unique-backing or
    /// physical lower certificate. No Array clone, value read or evaluation is performed.
    #[cfg(test)]
    pub(crate) fn payload_layouts_for_test(
        &self,
    ) -> (std::collections::BTreeMap<String, (Vec<i32>, Dtype)>, u64) {
        use std::collections::BTreeMap;
        type Layouts = BTreeMap<String, (Vec<i32>, Dtype)>;
        fn record(out: &mut Layouts, bytes: &mut u64, key: String, tensor: &Array) {
            *bytes += mlx_gen::ios_quantum::allocated_bytes(crate::quantum::bytes(tensor));
            out.insert(key, (tensor.shape().to_vec(), tensor.dtype()));
        }
        fn linear(out: &mut Layouts, bytes: &mut u64, key: &str, value: &AdaptableLinear) {
            if let Some((weight, bias)) = value.dense_weight() {
                record(out, bytes, format!("{key}.weight"), weight);
                if let Some(bias) = bias {
                    record(out, bytes, format!("{key}.bias"), bias);
                }
            } else if let Some((codes, scales, offsets, bias, _, _)) = value.quantized_params() {
                for (suffix, tensor) in [("weight", codes), ("scales", scales), ("biases", offsets)]
                {
                    record(out, bytes, format!("{key}.{suffix}"), tensor);
                }
                if let Some(bias) = bias {
                    record(out, bytes, format!("{key}.bias"), bias);
                }
            }
        }
        let (mut out, mut bytes) = (BTreeMap::new(), 0);
        for (key, tensor) in [
            (
                "text_model.embeddings.token_embedding.weight",
                &self.token_embedding,
            ),
            (
                "text_model.embeddings.position_embedding.weight",
                &self.position_embedding,
            ),
            ("text_model.final_layer_norm.weight", &self.final_ln_w),
            ("text_model.final_layer_norm.bias", &self.final_ln_b),
        ] {
            record(&mut out, &mut bytes, key.into(), tensor);
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("text_model.encoder.layers.{index}");
            for (key, tensor) in [
                ("layer_norm1.weight", &layer.ln1_w),
                ("layer_norm1.bias", &layer.ln1_b),
                ("layer_norm2.weight", &layer.ln2_w),
                ("layer_norm2.bias", &layer.ln2_b),
            ] {
                record(&mut out, &mut bytes, format!("{prefix}.{key}"), tensor);
            }
            for (name, value) in LAYER_LINEAR_NAMES.iter().zip([
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.out,
                &layer.linear1,
                &layer.linear2,
            ]) {
                linear(&mut out, &mut bytes, &format!("{prefix}.{name}"), value);
            }
        }
        if let Some(projection) = &self.text_projection {
            linear(&mut out, &mut bytes, TEXT_PROJECTION, projection);
        }
        (out, bytes)
    }

    /// One layer's guarded quantum body and price (sc-26270 proof tests): `layer.forward(x, mask)`
    /// and its counted scratch over `x`.
    #[cfg(test)]
    pub(crate) fn layer_quantum_for_test(
        &self,
        index: usize,
        x: &Array,
    ) -> (u64, impl Fn() -> Result<Array> + '_) {
        let s = x.shape();
        let layer = &self.layers[index];
        let scratch = layer.quantum_scratch(s[0], s[1], x.dtype());
        let (n, dtype) = (s[1], x.dtype());
        let x = x.clone();
        (scratch, move || layer.forward(&x, &causal_mask(n, dtype)?))
    }

    /// Native test witness, projected from the actual constructed/packed model. Production guards
    /// use the same metadata formulas directly; header construction never calls this method.
    #[cfg(test)]
    pub(crate) fn price_facts(&self) -> ClipPrice {
        use mlx_gen::adapters::facts::TensorFacts;
        ClipPrice {
            token_embedding: TensorFacts::of(&self.token_embedding),
            position_embedding: TensorFacts::of(&self.position_embedding),
            layers: self.layers.iter().map(ClipLayerPrice::of).collect(),
            final_ln_w: TensorFacts::of(&self.final_ln_w),
            final_ln_b: TensorFacts::of(&self.final_ln_b),
            text_projection: self.text_projection.as_ref().map(crate::quantum::LinearWork::of),
        }
    }

    /// Final norm and EOS pool/projection keep their original competing live sets. Only operand
    /// metadata crosses into the common formula; actual normalization/projection graphs are intact.
    fn final_scratch(&self, b: i32, n: i32, input: Dtype) -> u64 {
        use mlx_gen::adapters::facts::TensorFacts;
        clip_final_scratch(b, n, self.token_embedding.shape()[1], input,
            &TensorFacts::of(&self.final_ln_w), &TensorFacts::of(&self.final_ln_b),
            self.text_projection.as_ref().map(crate::quantum::LinearWork::of).as_ref())
    }

    /// Run the encoder over `input_ids` `[B, N]` (int32). Returns the last hidden state, the
    /// per-layer hidden states, and the pooled EOS token (projected for TE2).
    pub fn forward(&self, input_ids: &Array) -> Result<ClipOutput> {
        let sh = input_ids.shape();
        let (b, n) = (sh[0], sh[1]);
        let dim = self.token_embedding.shape()[1];

        // Hard bound: the position-embedding gather below indexes `0..n` into the `[max_len, D]`
        // table. MLX gathers are not bounds-checked, so `n` past the table would silently produce
        // garbage embeddings (F-062). Since sc-20528 this is the ONLY thing enforcing the context
        // window — the tokenizer no longer truncates; it splits an over-long prompt into windows
        // (`ClipBpeTokenizer::tokenize_windows`) and each window is forwarded here on its own, so a
        // long prompt reaches this guard as `n == max_len`, never past it. A caller that hands over
        // an unwindowed batch gets a typed error rather than silent garbage.
        let max_pos = self.position_embedding.shape()[0];
        if n > max_pos {
            return Err(Error::Msg(format!(
                "sdxl clip: sequence length {n} exceeds position-embedding capacity {max_pos}"
            )));
        }
        // sc-26270: embeddings, each layer, and the final norm + pooled token are guarded iOS
        // quanta (pass-throughs off device), each counted from its own ops.
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let embeddings_scratch = clip_embeddings_scratch(b, n, dim, self.token_embedding.dtype());

        // Token + position embeddings. `position_embedding.weight[:N]` broadcast over the batch.
        // Counted: the gathered tokens, the gathered positions and the sum.
        let mut x = mlx_gen::ios_quantum::run(crate::quantum::CLIP_EMBEDDINGS, embeddings_scratch, || {
            let ids_flat = input_ids.reshape(&[b * n])?;
            let tok = self
                .token_embedding
                .take_axis(&ids_flat, 0)?
                .reshape(&[b, n, dim])?;
            let pos_idx = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n]);
            let pos = self.position_embedding.take_axis(&pos_idx, 0)?; // [N, D]
            Ok(add(&tok, &pos.reshape(&[1, n, dim])?)?)
        })?;

        let mask = causal_mask(n, x.dtype())?;
        let mut hidden_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            x = mlx_gen::ios_quantum::run(
                crate::quantum::CLIP_LAYER,
                layer.quantum_scratch(b, n, x.dtype()),
                || layer.forward(&x, &mask),
            )?;
            hidden_states.push(x.clone());
        }
        mlx_gen::ios_quantum::before(
            crate::quantum::CLIP_POOLED,
            self.final_scratch(b, n, x.dtype()),
        )?;
        let last_hidden_state =
            layer_norm(&x, Some(&self.final_ln_w), Some(&self.final_ln_b), LN_EPS)?;

        // Pooled: the hidden state at each row's EOS token (argmax of the ids — EOS is the max id),
        // optionally projected. Compute EOS indices on the host (ids are small).
        let ids_host = host_i32(input_ids)?;
        let mut gather = Vec::with_capacity(b as usize);
        for row in 0..b as usize {
            let r = &ids_host[row * n as usize..(row + 1) * n as usize];
            gather.push(row as i32 * n + first_argmax(r));
        }
        let gather = Array::from_slice(&gather, &[b]);
        let pooled = last_hidden_state
            .reshape(&[b * n, dim])?
            .take_axis(&gather, 0)?;
        let pooled = match &self.text_projection {
            Some(p) => p.forward(&pooled)?,
            None => pooled,
        };
        mlx_gen::ios_quantum::finish(crate::quantum::CLIP_POOLED, &[&last_hidden_state, &pooled])?;

        Ok(ClipOutput {
            last_hidden_state,
            hidden_states,
            pooled,
        })
    }
}

/// Build the additive causal attention mask `[1, 1, N, N]` in the model `dtype`: `0` where a query
/// may attend to a key (`j <= i`), a large negative above the diagonal. Mirrors the vendored
/// `CLIPTextModel._get_mask(N, dtype)` exactly, including its dtype-specific masked value —
/// **`-6e4` for fp16** (≈ f16's finite floor; `-1e9` would overflow to `-inf`) and `-1e9` for f32.
/// The mask must share the q/k/v dtype or `scaled_dot_product_attention` rejects it.
fn causal_mask(n: i32, dtype: Dtype) -> Result<Array> {
    let nu = n as usize;
    let masked = if dtype == Dtype::Float16 {
        -6.0e4
    } else {
        -1.0e9
    };
    let mut m = vec![0f32; nu * nu];
    for i in 0..nu {
        for (j, slot) in m[i * nu..(i + 1) * nu].iter_mut().enumerate() {
            if j > i {
                *slot = masked;
            }
        }
    }
    Ok(Array::from_slice(&m, &[1, 1, n, n]).as_dtype(dtype)?)
}

/// Index of the **first** maximal element — `np.argmax` / `mx.argmax` semantics, which the reference
/// uses to locate the EOS token (the max token id) for the pooled output. Rust's `max_by_key` returns
/// the *last* maximal element on ties, so it diverged for prompts whose text contains a literal
/// `<|endoftext|>` (the CLIP regex matches it → multiple EOS ids) — a silent parity break (F-063).
/// Empty input → `0`.
fn first_argmax(ids: &[i32]) -> i32 {
    ids.iter()
        .enumerate()
        .fold((0i32, i32::MIN), |(best_i, best_v), (i, &v)| {
            if v > best_v {
                (i as i32, v)
            } else {
                (best_i, best_v)
            }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_argmax_picks_first_max_on_ties() {
        // F-063: EOS id appears twice; the *first* index wins (np.argmax), not max_by_key's last.
        assert_eq!(first_argmax(&[1, 49407, 5, 49407]), 1);
        assert_eq!(first_argmax(&[3, 3, 3]), 0); // all-equal → first
        assert_eq!(first_argmax(&[1, 2, 3]), 2); // strictly increasing → last is the max
        assert_eq!(first_argmax(&[]), 0); // empty → 0
    }

    #[test]
    fn causal_mask_is_lower_triangular() {
        let m = causal_mask(3, Dtype::Float32).unwrap();
        let v = m.reshape(&[9]).unwrap();
        let s = v.as_slice::<f32>();
        // row 0 attends only to 0; row 1 to {0,1}; row 2 to all.
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], -1e9);
        assert_eq!(s[2], -1e9);
        assert_eq!(s[3], 0.0);
        assert_eq!(s[4], 0.0);
        assert_eq!(s[5], -1e9);
        assert_eq!(s[6], 0.0);
        assert_eq!(s[8], 0.0);
    }
}
