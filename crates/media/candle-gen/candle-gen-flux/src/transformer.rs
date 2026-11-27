//! Candle FLUX.1 DiT for native BFL and diffusers checkpoints.
//!
//! The model uses the same double/single block topology, with checkpoint-specific projection
//! layouts and numerical conventions. The BFL weight names and fused projection shapes follow
//! candle-transformers' FLUX model. The diffusers loaders preserve split and packed projections
//! and can materialize bounded windows from device-format sidecars.

use std::ops::Range;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use candle_gen::gen_core::CancelFlag;
use candle_gen::quant::PackedWeightSidecars;
use candle_gen::rope::{FreqsCisLayout, RopeLayout, RopeMath, RopeSpec};
use candle_nn::{LayerNorm, RmsNorm, VarBuilder};

pub use candle_transformers::models::flux::model::Config;

use crate::ip_adapter::FluxIpInjector;
use crate::quant::QLinear;

/// A per-block **additive residual** injector for the FLUX DiT image stream — the generic *post-block*
/// seam the PuLID-FLUX id cross-attn (sc-5492, `candle-gen-pulid`) plugs into, the candle twin of
/// `mlx-gen-flux`'s `transformer::DitImageInjector`. Kept model-agnostic so this crate carries no
/// PuLID-specific code: the DiT just asks "is there a residual to add to the image tokens after block
/// N?" and the injector (which owns the id_embedding + cross-attn modules) answers.
///
/// This is **distinct** from the XLabs IP-Adapter seam ([`FluxIpInjector`]), which injects *mid*-block
/// on the image attention output using the captured pre-RoPE image query. This trait is consulted
/// *after* a block completes, on the image hidden stream: `after_double` sees the image hidden directly;
/// `after_single` sees the image-token tail of the joint `cat(txt, img)` stream (the DiT slices it out
/// and writes the residual back). Returning `None` (a non-injection block index, or a 0 weight) leaves
/// the stream untouched. The two seams are orthogonal — [`FluxDit::forward`] drives the IP one,
/// [`FluxDit::forward_injected`] drives this one.
pub trait DitImageInjector {
    /// Resolved injector settings. Model/adapter bytes are bound separately by the application.
    /// Unknown implementations cannot claim a complete execution identity.
    fn execution_policy(&self) -> Option<serde_json::Value> {
        None
    }

    /// Residual to add to the image stream after double block `block_idx`, or `None`.
    fn after_double(&self, block_idx: usize, img_hidden: &Tensor) -> Result<Option<Tensor>>;
    /// Cheap gate so the DiT skips slicing the image-token tail on single blocks with no injection.
    fn injects_after_single(&self, block_idx: usize) -> bool;
    /// Residual to add to the image-token tail after single block `block_idx`, or `None`.
    fn after_single(&self, block_idx: usize, img_tokens: &Tensor) -> Result<Option<Tensor>>;
}

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

/// i32-overflow-safe SDPA over an N-D `q`/`k`/`v` (leading dims folded to a single batch). Flattens the
/// leading dims to `[N, Sq, D]`, delegates the budgeted query-row chunking to the shared
/// [`candle_gen::sdpa_budgeted_flat`] (sc-9570) — which chunks once the `[N,Sq,Sk]` scores tensor would
/// exceed [`candle_gen::ATTN_SCORES_BUDGET`] (the candle CUDA i32-index limit) — then reshapes back.
/// scale = `1/sqrt(head_dim)`, `softmax_last_dim` closure keeps the exact fused softmax; each query row's
/// softmax is independent, so the chunked result is *mathematically* equal to the single pass — not
/// bitwise equal, since narrowing the query axis changes the GEMM `M` and so may change the f32
/// accumulation order (SC-15943). The FLUX.1 joint
/// `[txt, img]` sequence at the largest advertised sizes trips the guard; the common sizes stay a single
/// un-chunked pass. The vendored upstream SDPA is otherwise unchanged.
pub(crate) fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    scaled_dot_product_attention_planned(
        q,
        k,
        v,
        AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            candle_gen::ATTN_SCORES_BUDGET as u64,
            false,
        )),
    )
    .map_err(|error| match error {
        candle_gen::CandleError::Candle(error) => error,
        other => candle_core::Error::Msg(other.to_string()),
    })
}

pub(crate) fn scaled_dot_product_attention_planned(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    plan: AttentionPlan<'_>,
) -> candle_gen::Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;

    let attn = candle_gen::sdpa_planned_flat(
        &q,
        &k,
        &v,
        scale_factor,
        candle_nn::ops::softmax_last_dim,
        plan,
    )?; // [N, Sq, dim_v]
    batch_dims.push(attn.dim(D::Minus2)?);
    batch_dims.push(attn.dim(D::Minus1)?);
    Ok(attn.reshape(batch_dims)?)
}

fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle_core::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?;
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

/// FLUX.1's RoPE convention: interleaved pairs on `[B, H, S, D]` q/k, the products and sums in the
/// activation dtype (the reference `fr0·x0 + fr1·x1` over the `freqs_cis` rotation matrix).
pub(crate) const FLUX_ROPE: RopeSpec =
    RopeSpec::interleaved(RopeLayout::Bhsd).with_math(RopeMath::Native);

/// Rotate `x` `[B, H, S, D]` by FLUX's `freq_cis` `[B, 1, S, D/2, 2, 2]` (the `[[cos, −sin], [sin,
/// cos]]` matrix [`rope`] builds): the shared [`candle_gen::rope::apply_rope_freqs_cis`] (SW-6129) —
/// the fused strided kernel reading the matrix's cos / sin as views on CUDA, and everywhere else the
/// shared reference chain, bit-identical to the `fr0·x0 + fr1·x1` chain this used to carry
/// (`fr0 = (cos, sin)`, `fr1 = (−sin, cos)`: `x0·cos − x1·sin`, `x0·sin + x1·cos`, the negation exact).
pub(crate) fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    candle_gen::rope::apply_rope_freqs_cis(x, freq_cis, FreqsCisLayout::Matrix, &FLUX_ROPE)
}

/// Perturbed-attention guidance over FLUX.1 `[txt, img]` joint post-RoPE BHSD tensors:
/// `Q_txt` (`0..txt_seq`) runs normal SDPA over full `[K_txt, K_img]` / `[V_txt, V_img]`, while
/// `Q_img` (`txt_seq..`) receives `V_img` directly.
pub(crate) fn pag_joint_attention_post_rope(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    txt_seq: usize,
    plan: AttentionPlan<'_>,
) -> candle_gen::Result<(Tensor, Tensor)> {
    let total_seq = v.dim(2)?;
    let img_seq = total_seq - txt_seq;
    let q_txt = q.narrow(2, 0, txt_seq)?.contiguous()?;
    let out_txt = scaled_dot_product_attention_planned(&q_txt, k, v, plan)?
        .transpose(1, 2)?
        .flatten_from(2)?;
    let out_img = v
        .narrow(2, txt_seq, img_seq)?
        .transpose(1, 2)?
        .flatten_from(2)?
        .to_dtype(out_txt.dtype())?;
    Ok((out_txt, out_img))
}

pub(crate) fn attention_planned_pag(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pe: &Tensor,
    plan: AttentionPlan<'_>,
    pag_txt_seq: Option<usize>,
) -> candle_gen::Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    match pag_txt_seq {
        Some(txt_seq) => {
            let (out_txt, out_img) = pag_joint_attention_post_rope(&q, &k, v, txt_seq, plan)?;
            Ok(Tensor::cat(&[&out_txt, &out_img], 1)?)
        }
        None => {
            let x = scaled_dot_product_attention_planned(&q, &k, v, plan)?;
            Ok(x.transpose(1, 2)?.flatten_from(2)?)
        }
    }
}

pub(crate) fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        candle_core::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(candle_core::DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(candle_core::DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

#[derive(Debug, Clone)]
pub(crate) struct EmbedNd {
    theta: usize,
    axes_dim: Vec<usize>,
    #[cfg(test)]
    forward_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl EmbedNd {
    pub(crate) fn new(theta: usize, axes_dim: Vec<usize>) -> Self {
        Self {
            theta,
            axes_dim,
            #[cfg(test)]
            forward_calls: Default::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn forward_calls(&self) -> usize {
        self.forward_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl candle_core::Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        #[cfg(test)]
        self.forward_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            let r = rope(
                &ids.get_on_dim(D::Minus1, idx)?,
                self.axes_dim[idx],
                self.theta,
            )?;
            emb.push(r)
        }
        let emb = Tensor::cat(&emb, 2)?;
        emb.unsqueeze(1)
    }
}

/// Request-owned FLUX positional conditioning. The denoise latent changes every evaluation, but its
/// token geometry and the text/image position ids do not.
pub(crate) struct PreparedConditioning {
    layout: Option<Layout>,
    pub(crate) img_shape: Vec<usize>,
    pub(crate) img_ids_shape: Vec<usize>,
    pub(crate) txt_shape: Vec<usize>,
    pub(crate) txt_ids_shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) img_ids_dtype: DType,
    pub(crate) txt_dtype: DType,
    pub(crate) txt_ids_dtype: DType,
    pub(crate) device: candle_core::DeviceLocation,
    pub(crate) pe: Tensor,
}

impl PreparedConditioning {
    pub(crate) fn from_positions(
        image: &Tensor,
        image_ids: &Tensor,
        text: &Tensor,
        text_ids: &Tensor,
        pe: Tensor,
    ) -> Self {
        Self {
            layout: None,
            img_shape: image.dims().to_vec(),
            img_ids_shape: image_ids.dims().to_vec(),
            txt_shape: text.dims().to_vec(),
            txt_ids_shape: text_ids.dims().to_vec(),
            dtype: image.dtype(),
            img_ids_dtype: image_ids.dtype(),
            txt_dtype: text.dtype(),
            txt_ids_dtype: text_ids.dtype(),
            device: image.device().location(),
            pe,
        }
    }

    pub(crate) fn validate(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
    ) -> candle_gen::Result<()> {
        let matches = img.dims() == self.img_shape
            && img_ids.dims() == self.img_ids_shape
            && txt.dims() == self.txt_shape
            && txt_ids.dims() == self.txt_ids_shape
            && img.dtype() == self.dtype
            && img_ids.dtype() == self.img_ids_dtype
            && txt.dtype() == self.txt_dtype
            && txt_ids.dtype() == self.txt_ids_dtype
            && img.device().location() == self.device
            && img_ids.device().location() == self.device
            && txt.device().location() == self.device
            && txt_ids.device().location() == self.device;
        if !matches {
            return Err(candle_gen::CandleError::Msg(
                "flux: prepared conditioning geometry, dtype, or device does not match inputs"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Bfl,
    Diffusers,
}

impl Layout {
    fn double_prefix(self) -> &'static str {
        match self {
            Self::Bfl => "double_blocks",
            Self::Diffusers => "transformer_blocks",
        }
    }

    fn single_prefix(self) -> &'static str {
        match self {
            Self::Bfl => "single_blocks",
            Self::Diffusers => "single_transformer_blocks",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Dims {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    mlp: usize,
    qkv_bias: bool,
}

impl Dims {
    fn from_config(cfg: &Config) -> Self {
        Self {
            hidden: cfg.hidden_size,
            heads: cfg.num_heads,
            head_dim: cfg.hidden_size / cfg.num_heads,
            mlp: (cfg.hidden_size as f64 * cfg.mlp_ratio) as usize,
            qkv_bias: cfg.qkv_bias,
        }
    }
}

/// Retain the checkpoint's relative parameter path for adapter routing and sidecar lookups.
struct ProjectionLoader<'weights, 'sidecars> {
    vb: VarBuilder<'weights>,
    prefix: Option<String>,
    sidecars: Option<&'sidecars PackedWeightSidecars>,
}

impl<'weights, 'sidecars> ProjectionLoader<'weights, 'sidecars> {
    fn root(vb: VarBuilder<'weights>) -> Self {
        Self {
            vb,
            prefix: Some(String::new()),
            sidecars: None,
        }
    }

    fn parameter_path(&self, name: &str) -> Option<String> {
        self.prefix.as_ref().map(|prefix| {
            if prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{prefix}.{name}")
            }
        })
    }

    fn at(&self, name: &str) -> Self {
        Self {
            vb: self.vb.pp(name),
            prefix: self.parameter_path(name),
            sidecars: self.sidecars,
        }
    }

    fn dense(&self, input: usize, output: usize, key: &str) -> Result<Projection> {
        Ok(Projection {
            key: self.parameter_path(key),
            linear: QLinear::linear(input, output, self.vb.pp(key))?,
        })
    }

    fn detect(&self, input: usize, output: usize, key: &str, bias: bool) -> Result<Projection> {
        let path = self.parameter_path(key);
        let linear = match (self.sidecars, path.as_deref()) {
            (Some(sidecars), Some(path)) if self.vb.contains_tensor(&format!("{key}.scales")) => {
                if !sidecars.contains(path) {
                    candle_core::bail!(
                        "FLUX streamed packed projection `{path}` has no prepared device-format sidecar"
                    );
                }
                let dense_bias = bias
                    .then(|| self.vb.get(output, &format!("{key}.bias")))
                    .transpose()?;
                let qtensor = sidecars.load(path, self.vb.device())?;
                QLinear::from_packed(
                    candle_gen::quant::QLinear::from_qtensor_dequant(Arc::new(qtensor), dense_bias),
                    input,
                    output,
                )
            }
            _ => QLinear::linear_detect(input, output, &self.vb, key, bias)?,
        };
        Ok(Projection { key: path, linear })
    }
}

type AdapterVisitor<'a> = dyn FnMut(&str, &mut QLinear) -> Result<()> + 'a;

#[derive(Clone)]
struct Projection {
    key: Option<String>,
    linear: QLinear,
}

impl Projection {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.linear.forward(input)
    }

    fn visit(&mut self, visitor: &mut AdapterVisitor<'_>) -> Result<()> {
        let key = self.key.as_deref().ok_or_else(|| {
            candle_core::Error::Msg("FLUX streamed projections cannot be adapted".into())
        })?;
        visitor(key, &mut self.linear)
    }
}

#[derive(Clone)]
struct MlpEmbedder {
    first: Projection,
    second: Projection,
}

impl MlpEmbedder {
    fn new(
        layout: Layout,
        input: usize,
        hidden: usize,
        loader: &ProjectionLoader<'_, '_>,
    ) -> Result<Self> {
        let (first, second) = match layout {
            Layout::Bfl => (
                loader.dense(input, hidden, "in_layer")?,
                loader.dense(hidden, hidden, "out_layer")?,
            ),
            Layout::Diffusers => (
                loader.detect(input, hidden, "linear_1", true)?,
                loader.detect(hidden, hidden, "linear_2", true)?,
            ),
        };
        Ok(Self { first, second })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.second.forward(&self.first.forward(input)?.silu()?)
    }

    fn visit(&mut self, visitor: &mut AdapterVisitor<'_>) -> Result<()> {
        self.first.visit(visitor)?;
        self.second.visit(visitor)
    }
}

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, input: &Tensor) -> Result<Tensor> {
        candle_gen::modulate::modulate(input, &self.scale, &self.shift)
    }

    fn gate(&self, input: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(input)
    }
}

fn modulation<const N: usize>(
    linear: &Projection,
    embedding: &Tensor,
) -> Result<[ModulationOut; N]> {
    let params = linear
        .forward(&embedding.silu()?)?
        .unsqueeze(1)?
        .chunk(3 * N, D::Minus1)?;
    if params.len() != 3 * N {
        candle_core::bail!("unexpected FLUX modulation count: {}", params.len());
    }
    Ok(std::array::from_fn(|i| ModulationOut {
        shift: params[3 * i].clone(),
        scale: params[3 * i + 1].clone(),
        gate: params[3 * i + 2].clone(),
    }))
}

type QueryKeyValue = (Tensor, Tensor, Tensor);

fn fused_qkv(projected: Tensor, heads: usize) -> Result<QueryKeyValue> {
    let (batch, seq, _) = projected.dims3()?;
    let projected = projected.reshape((batch, seq, 3, heads, ()))?;
    Ok((
        projected.i((.., .., 0))?.transpose(1, 2)?,
        projected.i((.., .., 1))?.transpose(1, 2)?,
        projected.i((.., .., 2))?.transpose(1, 2)?,
    ))
}

#[derive(Clone)]
struct SplitQkv {
    q: Projection,
    k: Projection,
    v: Projection,
}

impl SplitQkv {
    fn new(dims: Dims, loader: &ProjectionLoader<'_, '_>, [q, k, v]: [&str; 3]) -> Result<Self> {
        let p = |name| loader.detect(dims.hidden, dims.hidden, name, true);
        Ok(Self {
            q: p(q)?,
            k: p(k)?,
            v: p(v)?,
        })
    }

    fn forward(&self, input: &Tensor, norm: &QkNorm, dims: Dims) -> Result<QueryKeyValue> {
        let (batch, seq, _) = input.dims3()?;
        let to_heads = |projection: &Projection| {
            projection
                .forward(input)?
                .reshape((batch, seq, dims.heads, dims.head_dim))?
                .transpose(1, 2)
        };
        Ok((
            __model_proof_output!(to_heads(&self.q)?.apply(&norm.query))?,
            __model_proof_output!(to_heads(&self.k)?.apply(&norm.key))?,
            to_heads(&self.v)?,
        ))
    }

    fn visit(&mut self, visitor: &mut AdapterVisitor<'_>) -> Result<()> {
        self.q.visit(visitor)?;
        self.k.visit(visitor)?;
        self.v.visit(visitor)
    }
}

#[derive(Clone)]
struct QkNorm {
    query: RmsNorm,
    key: RmsNorm,
}

impl QkNorm {
    fn new(size: usize, vb: &VarBuilder, query: &str, key: &str) -> Result<Self> {
        Ok(Self {
            query: RmsNorm::new(vb.get(size, query)?, 1e-6),
            key: RmsNorm::new(vb.get(size, key)?, 1e-6),
        })
    }

    fn fused(&self, qkv: Tensor, heads: usize) -> Result<QueryKeyValue> {
        let (q, k, v) = fused_qkv(qkv, heads)?;
        Ok((
            __model_proof_output!(q.apply(&self.query))?,
            __model_proof_output!(k.apply(&self.key))?,
            v,
        ))
    }
}

#[derive(Clone)]
enum AttentionInput {
    Fused(Projection),
    Split(Box<SplitQkv>),
}

#[derive(Clone)]
struct StreamAttention {
    input: AttentionInput,
    norm: QkNorm,
    output: Projection,
}

impl StreamAttention {
    fn bfl(dims: Dims, loader: &ProjectionLoader<'_, '_>) -> Result<Self> {
        Ok(Self {
            input: AttentionInput::Fused(loader.detect(
                dims.hidden,
                dims.hidden * 3,
                "qkv",
                dims.qkv_bias,
            )?),
            norm: QkNorm::new(
                dims.head_dim,
                &loader.vb,
                "norm.query_norm.scale",
                "norm.key_norm.scale",
            )?,
            output: loader.detect(dims.hidden, dims.hidden, "proj", true)?,
        })
    }

    fn diffusers(dims: Dims, loader: &ProjectionLoader<'_, '_>, text: bool) -> Result<Self> {
        let (qkv, norms, out) = if text {
            (
                ["add_q_proj", "add_k_proj", "add_v_proj"],
                ["norm_added_q.weight", "norm_added_k.weight"],
                "to_add_out",
            )
        } else {
            (
                ["to_q", "to_k", "to_v"],
                ["norm_q.weight", "norm_k.weight"],
                "to_out.0",
            )
        };
        Ok(Self {
            input: AttentionInput::Split(Box::new(SplitQkv::new(dims, loader, qkv)?)),
            norm: QkNorm::new(dims.head_dim, &loader.vb, norms[0], norms[1])?,
            output: loader.detect(dims.hidden, dims.hidden, out, true)?,
        })
    }

    fn qkv(&self, input: &Tensor, dims: Dims) -> Result<QueryKeyValue> {
        match &self.input {
            AttentionInput::Fused(qkv) => self.norm.fused(qkv.forward(input)?, dims.heads),
            AttentionInput::Split(qkv) => qkv.forward(input, &self.norm, dims),
        }
    }

    fn visit(&mut self, visitor: &mut AdapterVisitor<'_>) -> Result<()> {
        match &mut self.input {
            AttentionInput::Fused(qkv) => qkv.visit(visitor)?,
            AttentionInput::Split(qkv) => qkv.visit(visitor)?,
        }
        self.output.visit(visitor)
    }
}

/// Modulation, attention projections and feed-forward for one stream of the double block.
#[derive(Clone)]
struct DoubleStream {
    modulation: Projection,
    before_attention: LayerNorm,
    attention: StreamAttention,
    before_mlp: LayerNorm,
    first: Projection,
    second: Projection,
    erf_gelu: bool,
}

impl DoubleStream {
    fn new(
        layout: Layout,
        dims: Dims,
        loader: &ProjectionLoader<'_, '_>,
        text: bool,
        before_mlp: LayerNorm,
    ) -> Result<Self> {
        let (modulation, attention, first, second, erf_gelu) = match layout {
            Layout::Bfl => {
                let (mod_key, attention_key, mlp_key) = if text {
                    ("txt_mod.lin", "txt_attn", "txt_mlp")
                } else {
                    ("img_mod.lin", "img_attn", "img_mlp")
                };
                let mlp = loader.at(mlp_key);
                (
                    loader.dense(dims.hidden, 6 * dims.hidden, mod_key)?,
                    StreamAttention::bfl(dims, &loader.at(attention_key))?,
                    mlp.detect(dims.hidden, dims.mlp, "0", true)?,
                    mlp.detect(dims.mlp, dims.hidden, "2", true)?,
                    false,
                )
            }
            Layout::Diffusers => {
                let (mod_key, mlp_key) = if text {
                    ("norm1_context.linear", "ff_context")
                } else {
                    ("norm1.linear", "ff")
                };
                let mlp = loader.at(mlp_key);
                (
                    loader.detect(dims.hidden, 6 * dims.hidden, mod_key, true)?,
                    StreamAttention::diffusers(dims, &loader.at("attn"), text)?,
                    mlp.detect(dims.hidden, dims.mlp, "net.0.proj", true)?,
                    mlp.detect(dims.mlp, dims.hidden, "net.2", true)?,
                    !text,
                )
            }
        };
        Ok(Self {
            modulation,
            before_attention: layer_norm(dims.hidden, loader.vb.clone())?,
            attention,
            before_mlp,
            first,
            second,
            erf_gelu,
        })
    }

    fn qkv(&self, input: &Tensor, modulation: &ModulationOut, dims: Dims) -> Result<QueryKeyValue> {
        let normed = __model_proof_output!(input.apply(&self.before_attention))?;
        self.attention.qkv(&modulation.scale_shift(&normed)?, dims)
    }

    fn finish(
        &self,
        input: &Tensor,
        attention: &Tensor,
        attention_mod: &ModulationOut,
        mlp_mod: &ModulationOut,
    ) -> Result<Tensor> {
        let attention = self.attention.output.forward(attention)?;
        let hidden = (input + attention_mod.gate(&attention)?)?;
        let normed = __model_proof_output!(hidden.apply(&self.before_mlp))?;
        let hidden_mlp = self.first.forward(&mlp_mod.scale_shift(&normed)?)?;
        let activated = if self.erf_gelu {
            hidden_mlp.gelu_erf()?
        } else {
            hidden_mlp.gelu()?
        };
        let mlp_out = self.second.forward(&activated)?;
        &hidden + mlp_mod.gate(&mlp_out)?
    }
}

#[derive(Clone)]
struct DoubleStreamBlock {
    image: DoubleStream,
    text: DoubleStream,
    layout: Layout,
    dims: Dims,
}

impl DoubleStreamBlock {
    fn new(layout: Layout, dims: Dims, loader: &ProjectionLoader<'_, '_>) -> Result<Self> {
        let image_norm = layer_norm(dims.hidden, loader.vb.clone())?;
        let text_norm = match layout {
            Layout::Bfl => layer_norm(dims.hidden, loader.vb.clone())?,
            Layout::Diffusers => image_norm.clone(),
        };
        Ok(Self {
            image: DoubleStream::new(layout, dims, loader, false, image_norm)?,
            text: DoubleStream::new(layout, dims, loader, true, text_norm)?,
            layout,
            dims,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        image: &Tensor,
        text: &Tensor,
        embedding: &Tensor,
        pe: &Tensor,
        ip: Option<(&FluxIpInjector<'_>, usize)>,
        plan: AttentionPlan<'_>,
        perturb_image: bool,
    ) -> candle_gen::Result<(Tensor, Tensor)> {
        let [image_attn, image_mlp] = modulation(&self.image.modulation, embedding)?;
        let [text_attn, text_mlp] = modulation(&self.text.modulation, embedding)?;
        let (image_q, image_k, image_v) = self.image.qkv(image, &image_attn, self.dims)?;
        // Both layouts use the pre-RoPE normalized image query. Diffusers computes the IP
        // attention early; BFL keeps the query view and computes it after the feed-forward.
        let (deferred_ip, ip_residual) = match ip {
            Some((injector, index)) if self.layout == Layout::Diffusers => {
                (None, injector.double_block_residual(index, &image_q)?)
            }
            Some((injector, index)) => (Some((injector, index, image_q.clone())), None),
            None => (None, None),
        };
        let (text_q, text_k, text_v) = self.text.qkv(text, &text_attn, self.dims)?;
        let q = Tensor::cat(&[text_q, image_q], 2)?;
        let k = Tensor::cat(&[text_k, image_k], 2)?;
        let v = Tensor::cat(&[text_v, image_v], 2)?;
        let text_len = text.dim(1)?;
        let (text_attention, image_attention) = if perturb_image && self.layout == Layout::Diffusers
        {
            let q = apply_rope(&q, pe)?.contiguous()?;
            let k = apply_rope(&k, pe)?.contiguous()?;
            pag_joint_attention_post_rope(&q, &k, &v, text_len, plan)?
        } else {
            let out =
                attention_planned_pag(&q, &k, &v, pe, plan, perturb_image.then_some(text_len))?;
            (
                out.narrow(1, 0, text_len)?,
                out.narrow(1, text_len, image.dim(1)?)?,
            )
        };

        let image = self
            .image
            .finish(image, &image_attention, &image_attn, &image_mlp)?;
        let ip_residual = match deferred_ip {
            Some((injector, index, q)) => injector.double_block_residual(index, &q)?,
            None => ip_residual,
        };
        let image = match ip_residual {
            Some(residual) => (&image + residual.to_dtype(image.dtype())?)?,
            None => image,
        };
        let text = self
            .text
            .finish(text, &text_attention, &text_attn, &text_mlp)?;
        Ok((image, text))
    }

    fn visit(&mut self, visitor: &mut AdapterVisitor<'_>) -> Result<()> {
        match self.layout {
            Layout::Bfl => {
                for stream in [&mut self.image, &mut self.text] {
                    stream.modulation.visit(visitor)?;
                    stream.attention.visit(visitor)?;
                    stream.first.visit(visitor)?;
                    stream.second.visit(visitor)?;
                }
            }
            Layout::Diffusers => {
                // Retain the diffusers adapter visitor and receipt ordering.
                self.image.modulation.visit(visitor)?;
                self.text.modulation.visit(visitor)?;
                self.image.attention.visit(visitor)?;
                self.text.attention.visit(visitor)?;
                for stream in [&mut self.image, &mut self.text] {
                    stream.first.visit(visitor)?;
                    stream.second.visit(visitor)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct SplitSingleInput {
    qkv: SplitQkv,
    mlp: Projection,
}

#[derive(Clone)]
enum SingleInput {
    Fused(Projection),
    Split(Box<SplitSingleInput>),
}

/// BFL keeps the fused projection's MLP view live; diffusers projects the MLP after attention.
enum SingleMlp<'a> {
    Fused(Tensor),
    Split(&'a Projection, &'a Tensor),
}

impl SingleMlp<'_> {
    fn activate(self) -> Result<Tensor> {
        match self {
            Self::Fused(view) => view.gelu(),
            Self::Split(projection, input) => projection.forward(input)?.gelu(),
        }
    }
}

#[derive(Clone)]
struct SingleStreamBlock {
    input: SingleInput,
    output: Projection,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Projection,
    dims: Dims,
}

impl SingleStreamBlock {
    fn new(layout: Layout, dims: Dims, loader: &ProjectionLoader<'_, '_>) -> Result<Self> {
        let (input, output, norm, modulation) = match layout {
            Layout::Bfl => (
                SingleInput::Fused(loader.detect(
                    dims.hidden,
                    3 * dims.hidden + dims.mlp,
                    "linear1",
                    true,
                )?),
                loader.detect(dims.hidden + dims.mlp, dims.hidden, "linear2", true)?,
                QkNorm::new(
                    dims.head_dim,
                    &loader.vb,
                    "norm.query_norm.scale",
                    "norm.key_norm.scale",
                )?,
                loader.dense(dims.hidden, 3 * dims.hidden, "modulation.lin")?,
            ),
            Layout::Diffusers => (
                SingleInput::Split(Box::new(SplitSingleInput {
                    qkv: SplitQkv::new(dims, &loader.at("attn"), ["to_q", "to_k", "to_v"])?,
                    mlp: loader.detect(dims.hidden, dims.mlp, "proj_mlp", true)?,
                })),
                loader.detect(dims.hidden + dims.mlp, dims.hidden, "proj_out", true)?,
                QkNorm::new(
                    dims.head_dim,
                    &loader.vb,
                    "attn.norm_q.weight",
                    "attn.norm_k.weight",
                )?,
                loader.detect(dims.hidden, 3 * dims.hidden, "norm.linear", true)?,
            ),
        };
        Ok(Self {
            input,
            output,
            norm,
            pre_norm: layer_norm(dims.hidden, loader.vb.clone())?,
            modulation,
            dims,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        embedding: &Tensor,
        pe: &Tensor,
        plan: AttentionPlan<'_>,
        pag_text_len: Option<usize>,
    ) -> candle_gen::Result<Tensor> {
        let [modulation] = modulation(&self.modulation, embedding)?;
        let normed =
            modulation.scale_shift(&__model_proof_output!(hidden.apply(&self.pre_norm))?)?;
        let ((q, k, v), mlp) = match &self.input {
            SingleInput::Fused(projection) => {
                let projected = projection.forward(&normed)?;
                let qkv_end = 3 * self.dims.hidden;
                let qkv = self
                    .norm
                    .fused(projected.narrow(D::Minus1, 0, qkv_end)?, self.dims.heads)?;
                (
                    qkv,
                    SingleMlp::Fused(projected.narrow(D::Minus1, qkv_end, self.dims.mlp)?),
                )
            }
            SingleInput::Split(projections) => (
                projections.qkv.forward(&normed, &self.norm, self.dims)?,
                SingleMlp::Split(&projections.mlp, &normed),
            ),
        };
        let attention = attention_planned_pag(&q, &k, &v, pe, plan, pag_text_len)?;
        let out = self
            .output
            .forward(&Tensor::cat(&[attention, mlp.activate()?], 2)?)?;
        Ok((hidden + modulation.gate(&out)?)?)
    }

    fn visit(&mut self, visitor: &mut AdapterVisitor<'_>) -> Result<()> {
        match &mut self.input {
            SingleInput::Fused(input) => {
                input.visit(visitor)?;
                self.output.visit(visitor)?;
                self.modulation.visit(visitor)
            }
            SingleInput::Split(input) => {
                self.modulation.visit(visitor)?;
                input.qkv.visit(visitor)?;
                input.mlp.visit(visitor)?;
                self.output.visit(visitor)
            }
        }
    }
}

#[derive(Clone)]
struct OutputHead {
    modulation: Projection,
    output: Projection,
    norm: LayerNorm,
    layout: Layout,
}

impl OutputHead {
    fn new(
        layout: Layout,
        hidden: usize,
        channels: usize,
        loader: &ProjectionLoader<'_, '_>,
    ) -> Result<Self> {
        let (modulation, output) = match layout {
            Layout::Bfl => (
                loader.dense(hidden, 2 * hidden, "final_layer.adaLN_modulation.1")?,
                loader.detect(hidden, channels, "final_layer.linear", true)?,
            ),
            Layout::Diffusers => (
                loader.detect(hidden, 2 * hidden, "norm_out.linear", true)?,
                loader.detect(hidden, channels, "proj_out", true)?,
            ),
        };
        Ok(Self {
            modulation,
            output,
            norm: layer_norm(hidden, loader.vb.clone())?,
            layout,
        })
    }

    fn forward(&self, hidden: &Tensor, embedding: &Tensor) -> Result<Tensor> {
        let params = self
            .modulation
            .forward(&embedding.silu()?)?
            .chunk(2, D::Minus1)?;
        // Diffusers AdaLN-Continuous stores scale first; BFL stores shift first.
        let (shift, scale) = match self.layout {
            Layout::Bfl => (&params[0], &params[1]),
            Layout::Diffusers => (&params[1], &params[0]),
        };
        let normalized = candle_gen::modulate::modulate(
            &__model_proof_output!(hidden.apply(&self.norm))?,
            &scale.unsqueeze(1)?,
            &shift.unsqueeze(1)?,
        )?;
        self.output.forward(&normalized)
    }
}

#[derive(Clone)]
struct BlockSource {
    weights: VarBuilder<'static>,
    prefix: &'static str,
    count: usize,
    sidecars: Option<Arc<PackedWeightSidecars>>,
}

#[derive(Clone)]
enum Blocks<Block> {
    Resident(Vec<Block>),
    Streamed(BlockSource),
}

impl<Block> Blocks<Block> {
    fn len(&self) -> usize {
        match self {
            Self::Resident(blocks) => blocks.len(),
            Self::Streamed(source) => source.count,
        }
    }

    fn load_resident(
        root: &ProjectionLoader<'_, '_>,
        prefix: &str,
        count: usize,
        load: impl Fn(&ProjectionLoader<'_, '_>) -> Result<Block>,
    ) -> Result<Self> {
        let parent = root.at(prefix);
        (0..count)
            .map(|index| load(&parent.at(&index.to_string())))
            .collect::<Result<_>>()
            .map(Self::Resident)
    }

    #[allow(clippy::too_many_arguments)]
    fn run<State>(
        &self,
        device: &Device,
        window: usize,
        cancel: &CancelFlag,
        state: State,
        load: impl Fn(&ProjectionLoader<'_, '_>) -> Result<Block>,
        step: impl Fn(State, usize, &Block) -> candle_gen::Result<State>,
    ) -> candle_gen::Result<State> {
        let run = |mut state: State, first: usize, blocks: &[Block]| -> candle_gen::Result<State> {
            for (offset, block) in blocks.iter().enumerate() {
                candle_gen::check_cancel(cancel)?;
                state = step(state, first + offset, block)?;
            }
            Ok(state)
        };
        match self {
            Self::Resident(blocks) => run(state, 0, blocks),
            Self::Streamed(source) => {
                let plan = candle_gen::block_window::BlockPlan::new(source.count, window.max(1))?;
                candle_gen::block_window::run_windowed(
                    device,
                    &plan,
                    cancel,
                    state,
                    || Ok(source.weights.clone()),
                    |state, view, range: Range<usize>| {
                        let first = range.start;
                        let loader = ProjectionLoader {
                            vb: view.pp(source.prefix),
                            prefix: source.sidecars.as_ref().map(|_| source.prefix.to_owned()),
                            sidecars: source.sidecars.as_deref(),
                        };
                        let blocks = range
                            .map(|index| load(&loader.at(&index.to_string())))
                            .collect::<Result<Vec<_>>>()?;
                        run(state, first, &blocks)
                    },
                )
            }
        }
    }

    fn resident_mut(&mut self) -> Result<&mut [Block]> {
        match self {
            Self::Resident(blocks) => Ok(blocks),
            Self::Streamed(_) => {
                candle_core::bail!(
                    "FLUX adapters require resident transformer blocks; disable block streaming"
                )
            }
        }
    }
}

/// The FLUX.1 DiT, with one forward graph across checkpoint formats and residency policies.
#[derive(Clone)]
pub struct FluxDit {
    image: Projection,
    text: Projection,
    time: MlpEmbedder,
    pooled: MlpEmbedder,
    guidance: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    double: Blocks<DoubleStreamBlock>,
    single: Blocks<SingleStreamBlock>,
    output: OutputHead,
    layout: Layout,
    dims: Dims,
    device: Device,
}

// FLUX.1's diffusers component uses fixed pooled CLIP and T5 widths.
const POOLED_DIM: usize = 768;
const CONTEXT_DIM: usize = 4096;

impl FluxDit {
    /// Load the native BFL layout, with fused QKV weights and depth from the variant preset.
    pub fn new(cfg: &Config, vb: VarBuilder<'static>) -> Result<Self> {
        Self::resident(Layout::Bfl, cfg, cfg.depth, cfg.depth_single_blocks, vb)
    }

    /// Load the diffusers layout, with split projections and depths from the component config.
    pub(crate) fn new_diffusers(
        cfg: &Config,
        num_double: usize,
        num_single: usize,
        vb: VarBuilder<'static>,
    ) -> Result<Self> {
        Self::resident(Layout::Diffusers, cfg, num_double, num_single, vb)
    }

    fn resident(
        layout: Layout,
        cfg: &Config,
        num_double: usize,
        num_single: usize,
        vb: VarBuilder<'static>,
    ) -> Result<Self> {
        let dims = Dims::from_config(cfg);
        let loader = ProjectionLoader::root(vb);
        Self::load(layout, cfg, &loader, || {
            let double =
                Blocks::load_resident(&loader, layout.double_prefix(), num_double, |layer| {
                    DoubleStreamBlock::new(layout, dims, layer)
                })?;
            let single =
                Blocks::load_resident(&loader, layout.single_prefix(), num_single, |layer| {
                    SingleStreamBlock::new(layout, dims, layer)
                })?;
            Ok((double, single))
        })
    }

    pub(crate) fn new_block_streamed(cfg: &Config, vb: VarBuilder<'static>) -> Result<Self> {
        Self::streamed(
            Layout::Bfl,
            cfg,
            cfg.depth,
            cfg.depth_single_blocks,
            vb,
            None,
        )
    }

    pub(crate) fn new_diffusers_block_streamed(
        cfg: &Config,
        num_double: usize,
        num_single: usize,
        vb: VarBuilder<'static>,
        sidecars: Option<(Arc<PackedWeightSidecars>, Arc<PackedWeightSidecars>)>,
    ) -> Result<Self> {
        Self::streamed(Layout::Diffusers, cfg, num_double, num_single, vb, sidecars)
    }

    fn streamed(
        layout: Layout,
        cfg: &Config,
        num_double: usize,
        num_single: usize,
        vb: VarBuilder<'static>,
        sidecars: Option<(Arc<PackedWeightSidecars>, Arc<PackedWeightSidecars>)>,
    ) -> Result<Self> {
        let (double_sidecars, single_sidecars) =
            sidecars.map_or((None, None), |(d, s)| (Some(d), Some(s)));
        let double = Blocks::Streamed(BlockSource {
            weights: vb.clone(),
            prefix: layout.double_prefix(),
            count: num_double,
            sidecars: double_sidecars,
        });
        let single = Blocks::Streamed(BlockSource {
            weights: vb.clone(),
            prefix: layout.single_prefix(),
            count: num_single,
            sidecars: single_sidecars,
        });
        Self::load(layout, cfg, &ProjectionLoader::root(vb), || {
            Ok((double, single))
        })
    }

    fn load(
        layout: Layout,
        cfg: &Config,
        loader: &ProjectionLoader<'_, '_>,
        make_blocks: impl FnOnce() -> Result<(Blocks<DoubleStreamBlock>, Blocks<SingleStreamBlock>)>,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let (image_key, text_key, time_key, pooled_key, guidance_key, text_width, pooled_width) =
            match layout {
                Layout::Bfl => (
                    "img_in",
                    "txt_in",
                    "time_in",
                    "vector_in",
                    "guidance_in",
                    cfg.context_in_dim,
                    cfg.vec_in_dim,
                ),
                Layout::Diffusers => (
                    "x_embedder",
                    "context_embedder",
                    "time_text_embed.timestep_embedder",
                    "time_text_embed.text_embedder",
                    "time_text_embed.guidance_embedder",
                    CONTEXT_DIM,
                    POOLED_DIM,
                ),
            };
        let image = loader.detect(cfg.in_channels, hidden, image_key, true)?;
        let text = loader.detect(text_width, hidden, text_key, true)?;
        let load_embedders = || -> Result<_> {
            let time = MlpEmbedder::new(layout, 256, hidden, &loader.at(time_key))?;
            let pooled = MlpEmbedder::new(layout, pooled_width, hidden, &loader.at(pooled_key))?;
            let guidance = cfg
                .guidance_embed
                .then(|| MlpEmbedder::new(layout, 256, hidden, &loader.at(guidance_key)))
                .transpose()?;
            Ok((time, pooled, guidance))
        };
        let ((double, single), (time, pooled, guidance)) = match layout {
            Layout::Bfl => (make_blocks()?, load_embedders()?),
            Layout::Diffusers => {
                let embedders = load_embedders()?;
                (make_blocks()?, embedders)
            }
        };
        Ok(Self {
            image,
            text,
            time,
            pooled,
            guidance,
            pe_embedder: EmbedNd::new(cfg.theta, cfg.axes_dim.clone()),
            double,
            single,
            output: OutputHead::new(layout, hidden, cfg.in_channels, loader)?,
            layout,
            dims: Dims::from_config(cfg),
            device: loader.vb.device().clone(),
        })
    }

    pub(crate) fn prepare_conditioning(
        &self,
        image: &Tensor,
        image_ids: &Tensor,
        text: &Tensor,
        text_ids: &Tensor,
    ) -> candle_gen::Result<PreparedConditioning> {
        let ids = Tensor::cat(&[text_ids, image_ids], 1)?;
        let pe = ids.apply(&self.pe_embedder)?;
        let mut prepared =
            PreparedConditioning::from_positions(image, image_ids, text, text_ids, pe);
        prepared.layout = Some(self.layout);
        Ok(prepared)
    }

    pub fn num_double_blocks(&self) -> usize {
        self.double.len()
    }

    /// Walk the projection names of the actual checkpoint, including fused native targets.
    pub fn visit_adaptable_mut(&mut self, visitor: &mut AdapterVisitor<'_>) -> Result<()> {
        self.image.visit(visitor)?;
        self.text.visit(visitor)?;
        self.time.visit(visitor)?;
        self.pooled.visit(visitor)?;
        if let Some(guidance) = &mut self.guidance {
            guidance.visit(visitor)?;
        }
        for block in self.double.resident_mut()? {
            block.visit(visitor)?;
        }
        for block in self.single.resident_mut()? {
            block.visit(visitor)?;
        }
        match self.layout {
            Layout::Bfl => {
                self.output.output.visit(visitor)?;
                self.output.modulation.visit(visitor)
            }
            Layout::Diffusers => {
                self.output.modulation.visit(visitor)?;
                self.output.output.visit(visitor)
            }
        }
    }

    fn prepare_pe(
        &self,
        image: &Tensor,
        image_ids: &Tensor,
        text: &Tensor,
        text_ids: &Tensor,
        prepared: Option<&PreparedConditioning>,
    ) -> candle_gen::Result<Tensor> {
        match prepared {
            Some(prepared) => {
                prepared.validate(image, image_ids, text, text_ids)?;
                Ok(prepared.pe.clone())
            }
            None => Ok(Tensor::cat(&[text_ids, image_ids], 1)?.apply(&self.pe_embedder)?),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        ip: Option<&FluxIpInjector>,
    ) -> Result<Tensor> {
        self.forward_with_memory(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            ip,
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(
                candle_gen::ATTN_SCORES_BUDGET as u64,
                false,
            )),
            self.num_double_blocks() + self.single.len(),
            &CancelFlag::default(),
        )
        .map_err(|error| match error {
            candle_gen::CandleError::Candle(error) => error,
            other => candle_core::Error::Msg(other.to_string()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_memory(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        ip: Option<&FluxIpInjector>,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
    ) -> candle_gen::Result<Tensor> {
        self.forward_core(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            ip,
            None,
            None,
            attention_plan,
            transformer_window,
            cancel,
            None,
            None,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_prepared_with_memory(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        ip: Option<&FluxIpInjector>,
        injector: Option<&dyn DitImageInjector>,
        control: Option<(&[Tensor], f64)>,
        prepared: &PreparedConditioning,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
    ) -> candle_gen::Result<Tensor> {
        self.forward_prepared_with_memory_pag(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            ip,
            injector,
            control,
            prepared,
            attention_plan,
            transformer_window,
            cancel,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_prepared_with_memory_pag(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        ip: Option<&FluxIpInjector>,
        injector: Option<&dyn DitImageInjector>,
        control: Option<(&[Tensor], f64)>,
        prepared: &PreparedConditioning,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
        pag_layers: Option<&[u32]>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_core(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            ip,
            injector,
            control,
            attention_plan,
            transformer_window,
            cancel,
            Some(prepared),
            pag_layers,
        )
    }

    /// The upstream FLUX `forward`, plus an optional **post-block** image-stream residual injector —
    /// the seam the PuLID-FLUX id cross-attn (sc-5492) hooks into. `injector = None` is byte-identical
    /// to [`forward`](Self::forward) with `ip = None` (the plain FLUX path), so the no-id ablation
    /// (`id_weight = 0` ⇒ every residual `None`) carries zero overhead.
    ///
    /// The injector is consulted after every double block (the image stream `img`) and after the single
    /// blocks it opts into (the image-token tail of `joint = cat(txt, img)`), matching the reference
    /// `flux/model.py` PuLID injection points (every 2nd double, every 4th single). The XLabs IP seam is
    /// NOT engaged here (`ip = None` into the blocks) — PuLID uses only the post-block residuals.
    ///
    /// A thin wrapper over `forward_core` engaging only the post-block `injector`
    /// (`ip = None`, `control = None`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_injected(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
    ) -> Result<Tensor> {
        self.forward_core(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            None,
            injector,
            None,
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(
                candle_gen::ATTN_SCORES_BUDGET as u64,
                false,
            )),
            usize::MAX,
            &CancelFlag::default(),
            None,
            None,
        )
        .map_err(|error| match error {
            candle_gen::CandleError::Candle(error) => error,
            other => candle_core::Error::Msg(other.to_string()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_injected_with_memory_pag(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
        pag_layers: Option<&[u32]>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_core(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            None,
            injector,
            None,
            attention_plan,
            transformer_window,
            cancel,
            None,
            pag_layers,
        )
    }

    /// As `forward_injected`, but ALSO threading the Fun-Controlnet-Union per-double-block residuals
    /// (sc-8412) — the Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0` path. `control = (residuals,
    /// scale)` is the (already-computed, pre-injection) control-branch output: one residual per control
    /// double block, added to the base **image** stream after base double block `i` at the diffusers
    /// interval `ceil(num_double_blocks / num_residuals)`, scaled by `scale`. With `control = None` this
    /// is byte-identical to `forward_injected` (and so, with `injector = None` too, to the plain
    /// [`forward`](Self::forward)).
    ///
    /// **Compose-ready** (the candle twin of the mlx `forward_control` seam): the per-block `injector`
    /// (PuLID / XLabs IP-Adapter) is consulted at the SAME points as in `forward_injected`, so a
    /// future epic can stack identity + control in one denoise (`injector = Some(..)` AND
    /// `control = Some(..)`). The control residual is added AFTER the injector's `after_double` residual,
    /// matching diffusers' FLUX ControlNet order (the controlnet sample is added to the post-block
    /// hidden state).
    ///
    /// A thin wrapper over `forward_core` engaging the post-block `injector` and
    /// `control` (the XLabs mid-block `ip` seam is `None` here).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_control(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
        control: Option<(&[Tensor], f64)>,
    ) -> Result<Tensor> {
        self.forward_core(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            None,
            injector,
            control,
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(
                candle_gen::ATTN_SCORES_BUDGET as u64,
                false,
            )),
            usize::MAX,
            &CancelFlag::default(),
            None,
            None,
        )
        .map_err(|error| match error {
            candle_gen::CandleError::Candle(error) => error,
            other => candle_core::Error::Msg(other.to_string()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_control_with_memory(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
        control: Option<(&[Tensor], f64)>,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
    ) -> candle_gen::Result<Tensor> {
        self.forward_control_with_memory_pag(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            injector,
            control,
            attention_plan,
            transformer_window,
            cancel,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_control_with_memory_pag(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
        control: Option<(&[Tensor], f64)>,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
        pag_layers: Option<&[u32]>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_core(
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            None,
            injector,
            control,
            attention_plan,
            transformer_window,
            cancel,
            None,
            pag_layers,
        )
    }

    /// One block loop for both checkpoint layouts. The resident and windowed storage paths
    /// differ only in when they load blocks. IP, post-block identity, and control compose here.
    #[allow(clippy::too_many_arguments)]
    fn forward_core(
        &self,
        image: &Tensor,
        image_ids: &Tensor,
        text: &Tensor,
        text_ids: &Tensor,
        timesteps: &Tensor,
        pooled: &Tensor,
        guidance: Option<&Tensor>,
        ip: Option<&FluxIpInjector<'_>>,
        injector: Option<&dyn DitImageInjector>,
        control: Option<(&[Tensor], f64)>,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
        prepared: Option<&PreparedConditioning>,
        pag_layers: Option<&[u32]>,
    ) -> candle_gen::Result<Tensor> {
        if prepared.is_some_and(|prepared| prepared.layout != Some(self.layout)) {
            return Err(candle_gen::CandleError::Msg(
                "flux: prepared conditioning belongs to another DiT variant".into(),
            ));
        }
        if self.layout == Layout::Bfl {
            for (name, tensor) in [("txt", text), ("img", image)] {
                if tensor.rank() != 3 {
                    return Err(candle_gen::CandleError::Msg(format!(
                        "unexpected shape for {name} {:?}",
                        tensor.shape()
                    )));
                }
            }
        }
        let compute_pe = || self.prepare_pe(image, image_ids, text, text_ids, prepared);
        let (pe, image_hidden, text_hidden, timestep_dtype, control_dtype) = match self.layout {
            Layout::Bfl => {
                let pe = compute_pe()?;
                let text_hidden = self.text.forward(text)?;
                let image_hidden = self.image.forward(image)?;
                (
                    Some(pe),
                    image_hidden,
                    text_hidden,
                    image.dtype(),
                    image.dtype(),
                )
            }
            Layout::Diffusers => {
                let image_hidden = self.image.forward(image)?;
                let text_hidden = self.text.forward(text)?;
                let control_dtype = image_hidden.dtype();
                (
                    None,
                    image_hidden,
                    text_hidden,
                    pooled.dtype(),
                    control_dtype,
                )
            }
        };

        let mut embedding =
            self.time
                .forward(&timestep_embedding(timesteps, 256, timestep_dtype)?)?;
        if let (Some(embedder), Some(guidance)) = (&self.guidance, guidance) {
            embedding = (embedding
                + embedder.forward(&timestep_embedding(guidance, 256, timestep_dtype)?)?)?;
        }
        let embedding = (embedding + self.pooled.forward(pooled)?)?;
        let pe = match pe {
            Some(pe) => pe,
            None => compute_pe()?,
        };

        let scaled_control = control
            .filter(|(residuals, _)| !residuals.is_empty())
            .map(|(residuals, scale)| -> Result<_> {
                let interval = control_residual_interval(self.num_double_blocks(), residuals.len());
                let scaled = residuals
                    .iter()
                    .map(|residual| residual.to_dtype(control_dtype)? * scale)
                    .collect::<Result<Vec<_>>>()?;
                Ok((scaled, interval))
            })
            .transpose()?;

        let perturbed =
            |index| pag_layers.is_some_and(|layers| layers.binary_search(&(index as u32)).is_ok());
        let (image, text) = self.double.run(
            &self.device,
            transformer_window,
            cancel,
            (image_hidden, text_hidden),
            |loader| DoubleStreamBlock::new(self.layout, self.dims, loader),
            |(image, text), index, block| {
                let (mut image, text) = block.forward(
                    &image,
                    &text,
                    &embedding,
                    &pe,
                    ip.map(|injector| (injector, index)),
                    attention_plan,
                    perturbed(index),
                )?;
                // Identity is added after the block (and IP), then the scaled control sample.
                if let Some(injector) = injector {
                    if let Some(residual) = injector.after_double(index, &image)? {
                        image = (&image + residual.to_dtype(image.dtype())?)?;
                    }
                }
                if let Some((residuals, interval)) = &scaled_control {
                    let control_index = (index / interval).min(residuals.len() - 1);
                    image = (&image + &residuals[control_index])?;
                }
                Ok((image, text))
            },
        )?;

        let text_len = text.dim(1)?;
        let image_len = image.dim(1)?;
        let num_double = self.num_double_blocks();
        let joint = Tensor::cat(&[&text, &image], 1)?;
        let joint = self.single.run(
            &self.device,
            transformer_window,
            cancel,
            joint,
            |loader| SingleStreamBlock::new(self.layout, self.dims, loader),
            |joint, index, block| {
                let mut joint = block.forward(
                    &joint,
                    &embedding,
                    &pe,
                    attention_plan,
                    perturbed(num_double + index).then_some(text_len),
                )?;
                if let Some(injector) =
                    injector.filter(|injector| injector.injects_after_single(index))
                {
                    let image = joint.narrow(1, text_len, image_len)?;
                    if let Some(residual) = injector.after_single(index, &image)? {
                        let image = (image + residual.to_dtype(joint.dtype())?)?;
                        let text = joint.narrow(1, 0, text_len)?;
                        joint = Tensor::cat(&[&text, &image], 1)?;
                    }
                }
                Ok(joint)
            },
        )?;
        let image = match self.layout {
            Layout::Bfl => joint.i((.., text_len..))?,
            Layout::Diffusers => joint.narrow(1, text_len, image_len)?,
        };
        Ok(self.output.forward(&image, &embedding)?)
    }
}

/// ControlNet residual injection interval (diffusers `FluxTransformer2DModel`):
/// `interval = ceil(num_double_blocks / num_residuals)`. After base double block `i` the controlnet
/// residual `controlnet_block_samples[i / interval]` is added. For FLUX.1 (19 double blocks) and the
/// Shakker Union-Pro-2.0 (6 control residuals) this is `ceil(19/6) = 4` (sc-8412). `num_residuals = 0`
/// would never reach here (the caller filters an empty residual slice), so it is clamped to 1.
pub(crate) fn control_residual_interval(num_double_blocks: usize, num_residuals: usize) -> usize {
    num_double_blocks.div_ceil(num_residuals.max(1))
}

#[cfg(test)]
mod bfl_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    use std::cell::RefCell;

    fn assert_close(a: &Tensor, b: &Tensor) {
        assert_eq!(a.dims(), b.dims());
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x - y).abs() < 1e-6,
                "chunked attention diverged: {x} vs {y}"
            );
        }
    }

    #[test]
    fn chunked_sdpa_matches_single_pass() {
        // Per-query-row softmax is independent, so chunking over query rows (forced via a tiny budget)
        // must match the single pass bit-for-bit — the guard for the i32-overflow fix (sc-8983,
        // ported from FLUX.2's sc-5487). Retargeted onto the shared `candle_gen::sdpa_budgeted_flat`
        // (sc-9570) — the crate's `scaled_dot_product_attention` folds `[B,H,S,D]` → `[B·H, S, D]` and
        // delegates to it, with scale `1/sqrt(d)` and the fused `softmax_last_dim`. Covers the self-attn
        // shape and a cross-shape `Sq != Sk` (the flat helper is generic over the key length).
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let n = b * h;
        let scale = 1.0 / (d as f64).sqrt();
        let sm = candle_nn::ops::softmax_last_dim;
        let q = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (n, s, d), &dev).unwrap();
        // Huge budget → single pass; tiny budget (1) → single-row chunks; a MID-SIZE budget forces
        // multi-row chunks + a remainder (block=3 over s=7 → 3,3,1) — the sc-9116 hardening ask.
        let single =
            candle_gen::sdpa_budgeted_flat_materialized(&q, &k, &v, scale, sm, usize::MAX).unwrap();
        // budget = n·s·block = 2·7·3 = 42 → block = 3.
        for budget in [1usize, 42] {
            assert_close(
                &single,
                &candle_gen::sdpa_budgeted_flat_materialized(&q, &k, &v, scale, sm, budget)
                    .unwrap(),
            );
        }

        let sk = 5usize;
        let kx = Tensor::randn(0f32, 1f32, (n, sk, d), &dev).unwrap();
        let vx = Tensor::randn(0f32, 1f32, (n, sk, d), &dev).unwrap();
        let single =
            candle_gen::sdpa_budgeted_flat_materialized(&q, &kx, &vx, scale, sm, usize::MAX)
                .unwrap();
        // budget = n·sk·block = 2·5·3 = 30 → block = 3 (chunks 3, 3, 1 over the 7 query rows).
        for budget in [1usize, 30] {
            assert_close(
                &single,
                &candle_gen::sdpa_budgeted_flat_materialized(&q, &kx, &vx, scale, sm, budget)
                    .unwrap(),
            );
        }
    }

    /// A tiny FLUX DiT config (real hyperparameter *shape*, minimal depth) for a CPU-only forward parity
    /// test — small enough to run every gate, large enough to exercise the double + single block loops
    /// and the final layer.
    fn tiny_cfg() -> Config {
        Config {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 32,
            mlp_ratio: 2.0,
            num_heads: 2,
            depth: 3,
            depth_single_blocks: 4,
            axes_dim: vec![4, 6, 6],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
        }
    }

    /// Build a tiny random-weight [`FluxDit`] on CPU. VarMap zero-init is randomized so the forward is a
    /// non-degenerate function (a zero DiT would make the wrapper-vs-wrapper parity vacuous).
    fn tiny_ipflux(dev: &Device) -> FluxDit {
        tiny_ipflux_pair(dev).0
    }

    fn tiny_ipflux_pair(dev: &Device) -> (FluxDit, FluxDit) {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let resident = FluxDit::new(&cfg, vb.clone()).expect("tiny resident FluxDit");
        let streamed = FluxDit::new_block_streamed(&cfg, vb).expect("tiny block-streamed FluxDit");
        // Randomize every parameter (deterministic seed) — otherwise every weight is 0 and the forward
        // collapses to a constant, hiding any drift between the wrappers.
        let mut seed = 9003u64;
        for var in vm.data().lock().unwrap().values() {
            let n = var.shape().elem_count();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    // xorshift64* — a self-contained deterministic PRNG, no rand dep needed here.
                    seed ^= seed >> 12;
                    seed ^= seed << 25;
                    seed ^= seed >> 27;
                    let u =
                        (seed.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64;
                    ((u as f32) - 0.5) * 0.1
                })
                .collect();
            let t = Tensor::from_vec(data, var.shape(), dev).expect("randomize");
            var.set(&t).expect("set var");
        }
        (resident, streamed)
    }

    struct RecordingInjector {
        double: RefCell<Vec<usize>>,
        single: RefCell<Vec<usize>>,
    }

    impl RecordingInjector {
        fn new() -> Self {
            Self {
                double: RefCell::new(Vec::new()),
                single: RefCell::new(Vec::new()),
            }
        }

        fn residual(like: &Tensor, scale: f64) -> Result<Tensor> {
            let (b, s, c) = like.dims3()?;
            Tensor::arange(0u32, c as u32, like.device())?
                .to_dtype(like.dtype())?
                .affine(scale / c as f64, scale)?
                .reshape((1, 1, c))?
                .broadcast_as((b, s, c))?
                .contiguous()
        }
    }

    impl DitImageInjector for RecordingInjector {
        fn after_double(&self, block_idx: usize, img_hidden: &Tensor) -> Result<Option<Tensor>> {
            self.double.borrow_mut().push(block_idx);
            Ok(Some(Self::residual(img_hidden, 0.03)?))
        }

        fn injects_after_single(&self, _block_idx: usize) -> bool {
            true
        }

        fn after_single(&self, block_idx: usize, img_tokens: &Tensor) -> Result<Option<Tensor>> {
            self.single.borrow_mut().push(block_idx);
            Ok(Some(Self::residual(img_tokens, 0.02)?))
        }
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_window_one_preserves_injector_indices_and_control_order() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let (resident, streamed) = tiny_ipflux_pair(&dev);
        let (img, img_ids, txt, txt_ids, timesteps, y) = tiny_inputs(&dev, &cfg);
        let guidance = Tensor::from_vec(vec![3.5f32], (1,), &dev).unwrap();
        let control = [
            Tensor::full(0.01f32, (1, 4, cfg.hidden_size), &dev).unwrap(),
            Tensor::full(0.02f32, (1, 4, cfg.hidden_size), &dev).unwrap(),
        ];
        let expected_injector = RecordingInjector::new();
        let actual_injector = RecordingInjector::new();
        let plan =
            || AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false));
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let expected = resident
            .forward_control_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&guidance),
                Some(&expected_injector),
                Some((&control, 0.7)),
                plan(),
                1,
                &cancel,
            )
            .unwrap();
        let actual = streamed
            .forward_control_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&guidance),
                Some(&actual_injector),
                Some((&control, 0.7)),
                plan(),
                1,
                &cancel,
            )
            .unwrap();

        assert_close(&expected, &actual);
        assert_eq!(*actual_injector.double.borrow(), vec![0, 1, 2]);
        assert_eq!(*actual_injector.single.borrow(), vec![0, 1, 2, 3]);
        assert_eq!(
            *actual_injector.double.borrow(),
            *expected_injector.double.borrow()
        );
        assert_eq!(
            *actual_injector.single.borrow(),
            *expected_injector.single.borrow()
        );
        // Two residuals over three double blocks retain diffusers' idx=min(i/ceil(3/2), 1):
        // blocks 0/1 use residual 0 and block 2 uses residual 1. Exact resident parity above is the
        // mutation guard for applying control before the injector or resetting indices per window.
        assert_eq!(control_residual_interval(3, 2), 2);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::bfl_tests::streamed_window_one_preserves_injector_indices_and_control_order");
    }

    /// The FLUX DiT `img`/`img_ids`/`txt`/`txt_ids`/`timesteps`/`y` inputs for a tiny 2×2-token image and
    /// a 3-token text sequence — the geometry `State::new` produces, at the tiny config's channel counts.
    fn tiny_inputs(dev: &Device, cfg: &Config) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (img_seq, txt_seq) = (4usize, 3usize);
        let fill = |shape: (usize, usize, usize), scale: f32| {
            let n = shape.0 * shape.1 * shape.2;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.31).sin()) * scale).collect();
            Tensor::from_vec(data, shape, dev).unwrap()
        };
        let img = fill((1, img_seq, cfg.in_channels), 1.0);
        let txt = fill((1, txt_seq, cfg.context_in_dim), 1.0);
        // Position ids are 3-axis (axes_dim.len() == 3), integer-valued floats.
        let ids = |seq: usize| {
            let data: Vec<f32> = (0..seq * 3).map(|i| (i % 5) as f32).collect();
            Tensor::from_vec(data, (1, seq, 3), dev).unwrap()
        };
        let img_ids = ids(img_seq);
        let txt_ids = ids(txt_seq);
        let timesteps = Tensor::from_vec(vec![0.7f32], (1,), dev).unwrap();
        let y = fill((1, 1, cfg.vec_in_dim), 1.0).squeeze(1).unwrap(); // (1, vec_in_dim)
        (img, img_ids, txt, txt_ids, timesteps, y)
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn prepared_ipflux_conditioning_is_reused_by_base_and_control_forwards() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let model = tiny_ipflux(&dev);
        let (img, img_ids, txt, txt_ids, timesteps, y) = tiny_inputs(&dev, &cfg);
        let guidance = Tensor::from_vec(vec![3.5f32], (1,), &dev).unwrap();
        let mut prepared = model
            .prepare_conditioning(&img, &img_ids, &txt, &txt_ids)
            .unwrap();
        assert_eq!(model.pe_embedder.forward_calls(), 1);
        let plan =
            || AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false));
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let first = model
            .forward_prepared_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&guidance),
                None,
                None,
                None,
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .unwrap();
        let residual = Tensor::zeros((1, 4, cfg.hidden_size), DType::F32, &dev).unwrap();
        let control = model
            .forward_prepared_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&guidance),
                None,
                None,
                Some((std::slice::from_ref(&residual), 0.7)),
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .unwrap();
        assert_close(&first, &control);
        assert_eq!(
            model.pe_embedder.forward_calls(),
            1,
            "base/control forwards must not reconstruct PE"
        );
        let stale = Tensor::zeros((1, 5, cfg.in_channels), DType::F32, &dev).unwrap();
        assert!(model
            .forward_prepared_with_memory(
                &stale,
                &Tensor::zeros((1, 5, 3), DType::F32, &dev).unwrap(),
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&guidance),
                None,
                None,
                None,
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .is_err());
        let wrong_dtype = img.to_dtype(DType::F64).unwrap();
        assert!(model
            .forward_prepared_with_memory(
                &wrong_dtype,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&guidance),
                None,
                None,
                None,
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .is_err());
        prepared.device = candle_core::DeviceLocation::Cuda { gpu_id: 7 };
        assert!(prepared.validate(&img, &img_ids, &txt, &txt_ids).is_err());

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::bfl_tests::prepared_ipflux_conditioning_is_reused_by_base_and_control_forwards");
    }

    /// sc-9003 / F-023: the three public forwards were consolidated onto one `forward_core`. Each is a
    /// thin wrapper engaging only its own seam, so with every optional injector `None` all three must be
    /// byte-identical — the documented invariant (`forward_control(.., None) ≡ forward_injected(.., None)
    /// ≡ forward(.., None)`). This locks that the consolidation didn't perturb the shared body.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn wrappers_agree_when_all_injectors_none() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let model = tiny_ipflux(&dev);
        let (img, img_ids, txt, txt_ids, timesteps, y) = tiny_inputs(&dev, &cfg);
        let g = Tensor::from_vec(vec![3.5f32], (1,), &dev).unwrap();

        let base = model
            .forward(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&g),
                None,
            )
            .expect("forward");
        let injected = model
            .forward_injected(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&g),
                None,
            )
            .expect("forward_injected");
        let control = model
            .forward_control(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&g),
                None,
                None,
            )
            .expect("forward_control");

        // All three engage no seam ⇒ identical to the plain FLUX path, and to each other.
        assert_close(&base, &injected);
        assert_close(&base, &control);

        // An empty control residual slice is also "no control" — must match the None arm.
        let empty: Vec<Tensor> = Vec::new();
        let control_empty = model
            .forward_control(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &timesteps,
                &y,
                Some(&g),
                None,
                Some((&empty, 0.7)),
            )
            .expect("forward_control empty residuals");
        assert_close(&base, &control_empty);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::bfl_tests::wrappers_agree_when_all_injectors_none");
    }
}

#[cfg(test)]
mod diffusers_tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::candle_nn::VarMap;
    use std::cell::RefCell;

    /// A tiny FLUX [`Config`] for the GPU-free DiT smoke test: inner 16 (heads 2 · head_dim 8),
    /// `axes_dim` summing to 8, real input widths (in 64 / context 4096 / pooled 768).
    fn tiny_cfg(guidance: bool) -> Config {
        Config {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 16,
            mlp_ratio: 4.0,
            num_heads: 2,
            depth: 0,
            depth_single_blocks: 0,
            axes_dim: vec![2, 2, 4],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: guidance,
        }
    }

    #[test]
    fn checkpoint_layouts_keep_their_adapter_projection_names_and_shapes() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg(false);
        cfg.depth = 1;
        cfg.depth_single_blocks = 1;
        let native_weights = VarMap::new();
        let diffusers_weights = VarMap::new();
        let mut native = FluxDit::new(
            &cfg,
            VarBuilder::from_varmap(&native_weights, DType::F32, &device),
        )?;
        let mut diffusers = FluxDit::new_diffusers(
            &cfg,
            1,
            1,
            VarBuilder::from_varmap(&diffusers_weights, DType::F32, &device),
        )?;

        let collect = |model: &mut FluxDit| -> Result<std::collections::BTreeMap<_, _>> {
            let mut routes = std::collections::BTreeMap::new();
            model.visit_adaptable_mut(&mut |name, projection| {
                assert!(routes
                    .insert(
                        name.to_owned(),
                        (projection.in_features, projection.out_features)
                    )
                    .is_none());
                Ok(())
            })?;
            Ok(routes)
        };

        let native = collect(&mut native)?;
        let diffusers = collect(&mut diffusers)?;
        assert_eq!(native["double_blocks.0.img_attn.qkv"], (16, 48));
        assert_eq!(native["single_blocks.0.linear1"], (16, 112));
        assert_eq!(native["final_layer.adaLN_modulation.1"], (16, 32));
        assert_eq!(diffusers["transformer_blocks.0.attn.to_q"], (16, 16));
        assert_eq!(diffusers["transformer_blocks.0.attn.to_out.0"], (16, 16));
        assert_eq!(diffusers["single_transformer_blocks.0.proj_mlp"], (16, 64));
        assert_eq!(diffusers["norm_out.linear"], (16, 32));
        assert!(!native.keys().any(|key| key.contains("to_q")));
        assert!(!diffusers.keys().any(|key| key.contains("qkv")));
        Ok(())
    }

    /// The vendored diffusers DiT loads (dense, no `.scales`) and forwards to the FLUX velocity shape
    /// `[B, img_seq, 64]` with finite values — exercising the double/single/output-head wiring and the
    /// `cat(txt,img)` single-block plumbing end-to-end. `guidance_embed = true` also exercises the dev
    /// guidance-embedder path. A 2-double / 2-single tiny config keeps it GPU-free.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_dit_dense_forward_shape_and_finite() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg(true);
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let dit = FluxDit::new_diffusers(&cfg, 2, 2, vb)?;

        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let g = Tensor::full(3.5f32, b, &dev)?;

        let out = dit.forward(&img, &img_ids, &txt, &txt_ids, &ts, &pooled, Some(&g), None)?;
        assert_eq!(out.dims(), &[b, img_seq, 64]);
        let max = out.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max.is_finite(), "DiT output must be finite, got max {max}");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::packed_dit_dense_forward_shape_and_finite");
        Ok(())
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn prepared_packed_conditioning_is_reused_by_base_and_control_forwards() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg(true);
        let vm = VarMap::new();
        let dit =
            FluxDit::new_diffusers(&cfg, 2, 2, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let guidance = Tensor::full(3.5f32, b, &dev)?;
        let mut prepared = dit
            .prepare_conditioning(&img, &img_ids, &txt, &txt_ids)
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        assert_eq!(dit.pe_embedder.forward_calls(), 1);
        let plan =
            || AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false));
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let base = dit
            .forward_prepared_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                None,
                None,
                None,
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        let residual = Tensor::zeros((b, img_seq, cfg.hidden_size), DType::F32, &dev)?;
        let control = dit
            .forward_prepared_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                None,
                None,
                Some((std::slice::from_ref(&residual), 0.7)),
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        assert_eq!(base.to_vec3::<f32>()?, control.to_vec3::<f32>()?);
        assert_eq!(
            dit.pe_embedder.forward_calls(),
            1,
            "packed base/control forwards must not reconstruct PE"
        );
        let stale = Tensor::zeros((b, img_seq + 1, 64), DType::F32, &dev)?;
        let stale_ids = Tensor::zeros((b, img_seq + 1, 3), DType::F32, &dev)?;
        assert!(dit
            .forward_prepared_with_memory(
                &stale,
                &stale_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                None,
                None,
                None,
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .is_err());
        let wrong_dtype = img.to_dtype(DType::F64)?;
        assert!(dit
            .forward_prepared_with_memory(
                &wrong_dtype,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                None,
                None,
                None,
                &prepared,
                plan(),
                usize::MAX,
                &cancel,
            )
            .is_err());
        prepared.device = candle_gen::candle_core::DeviceLocation::Metal { gpu_id: 3 };
        assert!(prepared.validate(&img, &img_ids, &txt, &txt_ids).is_err());
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::prepared_packed_conditioning_is_reused_by_base_and_control_forwards");
        Ok(())
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn dense_streamed_blocks_match_resident_window_one() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = tiny_cfg(true);
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let resident = FluxDit::new_diffusers(&cfg, 2, 2, vb.clone())?;
        let streamed = FluxDit::new_diffusers_block_streamed(&cfg, 2, 2, vb, None)?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let guidance = Tensor::full(3.5f32, b, &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let expected = resident.forward(
            &img,
            &img_ids,
            &txt,
            &txt_ids,
            &ts,
            &pooled,
            Some(&guidance),
            None,
        )?;
        let actual = streamed
            .forward_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                None,
                AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false)),
                1,
                &candle_gen::gen_core::CancelFlag::default(),
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        assert_eq!(expected.to_vec3::<f32>()?, actual.to_vec3::<f32>()?);
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::dense_streamed_blocks_match_resident_window_one");
        Ok(())
    }

    /// A test [`DitImageInjector`] that adds a constant residual to the image stream after one double
    /// block and one single block, so the effect of the seam is observable (the candle twin of the
    /// PuLID CA modules, without the identity math).
    struct StubInjector {
        after_double_idx: usize,
        after_single_idx: usize,
        value: f64,
    }

    impl StubInjector {
        /// A **channel-varying** residual (a per-channel ramp, broadcast over batch/tokens). A constant
        /// or pure-scalar-multiple residual would be normalized away by the output-head LayerNorm
        /// (shift/scale-invariant per token) after passing near-unchanged through the residual single
        /// blocks — so it could not be observed. The real PuLID CA residual likewise varies per channel.
        fn residual(&self, like: &Tensor) -> Result<Tensor> {
            let (b, s, c) = like.dims3()?;
            let ramp = Tensor::arange(0u32, c as u32, like.device())?
                .to_dtype(like.dtype())?
                .affine(self.value / c as f64, self.value)?;
            ramp.reshape((1, 1, c))?
                .broadcast_as((b, s, c))?
                .contiguous()
        }
    }

    impl DitImageInjector for StubInjector {
        fn after_double(&self, block_idx: usize, img_hidden: &Tensor) -> Result<Option<Tensor>> {
            if block_idx == self.after_double_idx {
                Ok(Some(self.residual(img_hidden)?))
            } else {
                Ok(None)
            }
        }
        fn injects_after_single(&self, block_idx: usize) -> bool {
            block_idx == self.after_single_idx
        }
        fn after_single(&self, block_idx: usize, img_tokens: &Tensor) -> Result<Option<Tensor>> {
            if block_idx == self.after_single_idx {
                Ok(Some(self.residual(img_tokens)?))
            } else {
                Ok(None)
            }
        }
    }

    /// Build a tiny dense diffusers DiT with deterministically-randomized weights. A zero-init VarMap
    /// (as in the finite-shape smoke above) makes the forward degenerate — the output-head LayerNorm
    /// normalizes away a constant residual and the zero `proj_out` maps everything to 0 — so an injected
    /// residual could not be observed. Randomizing every parameter (mirrors `bfl_tests::tiny_ipflux`)
    /// makes the forward a non-degenerate function of the injector.
    fn randomized_tiny_dit(dev: &Device) -> Result<FluxDit> {
        Ok(randomized_tiny_dit_pair(dev)?.0)
    }

    fn randomized_tiny_dit_pair(dev: &Device) -> Result<(FluxDit, FluxDit)> {
        let cfg = tiny_cfg(true);
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let resident = FluxDit::new_diffusers(&cfg, 2, 2, vb.clone())?;
        let streamed = FluxDit::new_diffusers_block_streamed(&cfg, 2, 2, vb, None)?;
        let mut seed = 0x9E3779B97F4A7C15u64;
        for var in vm.data().lock().unwrap().values() {
            let n = var.shape().elem_count();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    // xorshift64* — deterministic, self-contained (no rand dep in this crate's tests).
                    seed ^= seed >> 12;
                    seed ^= seed << 25;
                    seed ^= seed >> 27;
                    let u =
                        (seed.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64;
                    ((u as f32) - 0.5) * 0.1
                })
                .collect();
            var.set(&Tensor::from_vec(data, var.shape(), dev)?)?;
        }
        Ok((resident, streamed))
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_injected_window_one_matches_resident_across_both_namespaces() -> Result<()> {
        let dev = Device::Cpu;
        let (resident, streamed) = randomized_tiny_dit_pair(&dev)?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let guidance = Tensor::full(3.5f32, b, &dev)?;
        let injector = StubInjector {
            after_double_idx: 1,
            after_single_idx: 0,
            value: 0.5,
        };
        let plan =
            || AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false));
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let expected = resident
            .forward_injected_with_memory_pag(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                Some(&injector),
                plan(),
                1,
                &cancel,
                None,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        let actual = streamed
            .forward_injected_with_memory_pag(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                Some(&injector),
                plan(),
                1,
                &cancel,
                None,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        assert_eq!(expected.to_vec3::<f32>()?, actual.to_vec3::<f32>()?);
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::streamed_injected_window_one_matches_resident_across_both_namespaces");
        Ok(())
    }

    /// The packed XLabs seam uses the double block's post-QKNorm, pre-RoPE image query (not the
    /// generic post-block injector). Window-one streaming must preserve its absolute block index and
    /// populate exactly one cached image-token K/V pair per adapted block.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_xlabs_window_one_matches_resident_and_populates_each_block_cache() -> Result<()> {
        let dev = Device::Cpu;
        let (resident, streamed) = randomized_tiny_dit_pair(&dev)?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let guidance = Tensor::full(3.5f32, b, &dev)?;
        let adapter = crate::ip_adapter::FluxIpAdapter::synthetic_for_hidden(2, 4096, 16, &dev)?;
        let token_data = (0..4 * 4096)
            .map(|index| (index as f32 * 0.019).sin() * 0.1)
            .collect::<Vec<_>>();
        let tokens = Tensor::from_vec(token_data, (1, 4, 4096), &dev)?;
        let resident_injector = FluxIpInjector::new(&adapter, tokens.clone(), 0.7);
        let streamed_injector = FluxIpInjector::new(&adapter, tokens, 0.7);
        let plan =
            || AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false));
        let cancel = candle_gen::gen_core::CancelFlag::default();

        let expected = resident
            .forward_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                Some(&resident_injector),
                plan(),
                1,
                &cancel,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        let actual = streamed
            .forward_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                Some(&streamed_injector),
                plan(),
                1,
                &cancel,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        assert_eq!(expected.to_vec3::<f32>()?, actual.to_vec3::<f32>()?);
        assert_eq!(resident_injector.cached_block_count(), 2);
        assert_eq!(streamed_injector.cached_block_count(), 2);

        let baseline = resident.forward(
            &img,
            &img_ids,
            &txt,
            &txt_ids,
            &ts,
            &pooled,
            Some(&guidance),
            None,
        )?;
        assert!(
            tensor_max_abs_diff_f32(&baseline, &expected) > 1e-6,
            "nonzero XLabs residual must affect the packed DiT output"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::streamed_xlabs_window_one_matches_resident_and_populates_each_block_cache");
        Ok(())
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_control_window_one_matches_resident_with_absolute_residual_indices() -> Result<()> {
        let dev = Device::Cpu;
        let (resident, streamed) = randomized_tiny_dit_pair(&dev)?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let guidance = Tensor::full(3.5f32, b, &dev)?;
        let ramp = Tensor::arange(0f32, 16f32, &dev)?
            .reshape((1, 1, 16))?
            .broadcast_as((b, img_seq, 16))?
            .contiguous()?;
        let residuals = vec![ramp.affine(0.025, 0.1)?, ramp.affine(-0.05, -0.2)?];
        let plan =
            || AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false));
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let expected = resident
            .forward_control_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                None,
                Some((&residuals, 0.7)),
                plan(),
                1,
                &cancel,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        let actual = streamed
            .forward_control_with_memory(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                None,
                Some((&residuals, 0.7)),
                plan(),
                1,
                &cancel,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        assert_eq!(expected.to_vec3::<f32>()?, actual.to_vec3::<f32>()?);
        let baseline = resident.forward(
            &img,
            &img_ids,
            &txt,
            &txt_ids,
            &ts,
            &pooled,
            Some(&guidance),
            None,
        )?;
        assert!(tensor_max_abs_diff_f32(&baseline, &expected) > 1e-6);
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::streamed_control_window_one_matches_resident_with_absolute_residual_indices");
        Ok(())
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn streamed_window_one_observes_cancel_before_next_block_materialization() -> Result<()> {
        struct CancelAfterFirst<'a> {
            cancel: &'a candle_gen::gen_core::CancelFlag,
            visited: RefCell<Vec<usize>>,
        }
        impl DitImageInjector for CancelAfterFirst<'_> {
            fn after_double(&self, block_idx: usize, _hidden: &Tensor) -> Result<Option<Tensor>> {
                self.visited.borrow_mut().push(block_idx);
                if block_idx == 0 {
                    self.cancel.cancel();
                }
                Ok(None)
            }

            fn injects_after_single(&self, _block_idx: usize) -> bool {
                false
            }

            fn after_single(
                &self,
                _block_idx: usize,
                _img_tokens: &Tensor,
            ) -> Result<Option<Tensor>> {
                Ok(None)
            }
        }

        let dev = Device::Cpu;
        let (_, streamed) = randomized_tiny_dit_pair(&dev)?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::zeros((b, img_seq, 64), DType::F32, &dev)?;
        let txt = Tensor::zeros((b, txt_seq, 4096), DType::F32, &dev)?;
        let pooled = Tensor::zeros((b, 768), DType::F32, &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let guidance = Tensor::full(3.5f32, b, &dev)?;
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let injector = CancelAfterFirst {
            cancel: &cancel,
            visited: RefCell::new(Vec::new()),
        };
        let error = streamed
            .forward_injected_with_memory_pag(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &ts,
                &pooled,
                Some(&guidance),
                Some(&injector),
                AttentionPlan::budgeted(AttentionBudget::from_score_elements(u64::MAX, false)),
                1,
                &cancel,
                None,
            )
            .expect_err("cancel after block zero must stop the next window");
        assert!(matches!(error, candle_gen::CandleError::Canceled));
        assert_eq!(&*injector.visited.borrow(), &[0]);
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::streamed_window_one_observes_cancel_before_next_block_materialization");
        Ok(())
    }

    /// `forward_injected(.., None)` is byte-identical to `forward(..)` — the seam is inert with no
    /// injector, so the txt2img packed path is unperturbed (mirrors the BFL invariant).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn forward_injected_none_matches_forward() -> Result<()> {
        let dev = Device::Cpu;
        let dit = randomized_tiny_dit(&dev)?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let g = Tensor::full(3.5f32, b, &dev)?;

        let base = dit.forward(&img, &img_ids, &txt, &txt_ids, &ts, &pooled, Some(&g), None)?;
        let injected =
            dit.forward_injected(&img, &img_ids, &txt, &txt_ids, &ts, &pooled, Some(&g), None)?;
        assert_eq!(
            tensor_max_abs_diff_f32(&base, &injected),
            0.0,
            "None injector must be a no-op"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::forward_injected_none_matches_forward");
        Ok(())
    }

    /// A nonzero injected residual (after a double block AND a single block) changes the output — the
    /// seam is actually wired into both block loops, not silently dropped.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn forward_injected_residual_changes_output() -> Result<()> {
        let dev = Device::Cpu;
        let dit = randomized_tiny_dit(&dev)?;
        let (b, img_seq, txt_seq) = (1usize, 4usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, 64), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, 4096), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, 768), &dev)?;
        let img_ids = Tensor::zeros((b, img_seq, 3), DType::F32, &dev)?;
        let txt_ids = Tensor::zeros((b, txt_seq, 3), DType::F32, &dev)?;
        let ts = Tensor::full(0.5f32, b, &dev)?;
        let g = Tensor::full(3.5f32, b, &dev)?;

        let base = dit.forward(&img, &img_ids, &txt, &txt_ids, &ts, &pooled, Some(&g), None)?;
        // Inject after double block 1 and single block 0 (both must be in-range for the tiny 2/2 DiT).
        let inj = StubInjector {
            after_double_idx: 1,
            after_single_idx: 0,
            value: 0.5,
        };
        let out = dit.forward_injected(
            &img,
            &img_ids,
            &txt,
            &txt_ids,
            &ts,
            &pooled,
            Some(&g),
            Some(&inj),
        )?;
        let d = tensor_max_abs_diff_f32(&base, &out);
        assert!(
            d > 1e-4,
            "a nonzero injected residual must change the output: max|Δ| = {d}"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::forward_injected_residual_changes_output");
        Ok(())
    }

    // ========================================================================================
    // Stock-vs-vendored DiT numeric parity (sc-9443).
    //
    // The vendored DiT above is the **diffusers**-layout FLUX.1 MMDiT (split `attn.to_q`/`to_k`/`to_v`,
    // `transformer_blocks`/`single_transformer_blocks`, `norm1.linear` AdaLN-Zero). The stock
    // `candle_transformers::models::flux::model::Flux` is the **BFL** layout (fused `img_attn.qkv`,
    // `double_blocks`/`single_blocks`, `img_mod.lin` modulation). They compute the same FLUX velocity —
    // so a dense build of the vendored DiT, its diffusers weights remapped into the BFL key layout, must
    // match the stock `Flux` forward at ~1e-4 on shared weights. This anchors the vendored DiT's RoPE /
    // QK-norm / QKV-split / modulation-chunk ordering against the reference (mirrors the CLIP/T5 encoder
    // parity tests in [`crate::packed_te`]). The remap (below) is the load-time transform the packed tier
    // implies: diffusers split-QKV → BFL fused-QKV, `to_out.0` → `proj`, per-embedder rename.
    // ========================================================================================

    use candle_gen::candle_core::Var;
    use candle_gen::testkit::tensor_max_abs_diff_f32;
    use std::collections::HashMap;

    /// A tiny parity [`Config`]: heads 2 · head_dim 8 = hidden 16, `axes_dim` summing to 8.
    /// `context_in_dim`/`vec_in_dim` MUST be the real FLUX widths (4096 T5 / 768 pooled) because the
    /// vendored `context_embedder`/`text_embedder` are built at the module constants [`CONTEXT_DIM`] /
    /// [`POOLED_DIM`], not from the cfg — the stock `Flux` reads them from cfg, so pinning cfg to those
    /// constants keeps the two input projections shape-matched. Only the tiny seq lengths / hidden width
    /// keep it GPU-free (the two input `Linear`s are the cost, but a 3-token / 5-token forward is cheap).
    /// `guidance_embed` toggles the dev guidance-embedder path.
    fn parity_cfg(guidance: bool) -> Config {
        Config {
            in_channels: 32,
            vec_in_dim: POOLED_DIM,
            context_in_dim: CONTEXT_DIM,
            hidden_size: 16,
            mlp_ratio: 4.0,
            num_heads: 2,
            depth: 2,
            depth_single_blocks: 2,
            axes_dim: vec![2, 2, 4],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: guidance,
        }
    }

    /// Make the QK-RMSNorm scales **Q/K-asymmetric** so the parity anchors actually discriminate a
    /// Q↔K transposition in the QKV split (sc-9443, review). QK-RMSNorm + a tiny head_dim (8) +
    /// statistically-symmetric random-init `to_q`/`to_k` weights make Q and K interchangeable at this
    /// config's scale: swapping which projection feeds Q vs K leaves `softmax(QKᵀ)` unchanged to well
    /// inside 1e-4, so a wrong-QKV-split port would escape CI. The learned RMSNorm scale vectors are the
    /// one per-channel signal that survives the (scale-invariant) RMS normalization, so we overwrite them
    /// in the shared `VarMap` — *before* both the vendored and stock models read them — with distinct,
    /// deterministic per-channel patterns for the query vs key norms. Correct remap still matches exactly
    /// (both sides read the same asymmetric scales); a Q↔K swap now mismatches the channel weighting of
    /// the attention score and diverges by O(1), tripping the tight 1e-4 anchor.
    ///
    /// Query norm gets an ascending ramp `1.0 + 0.5·c/(D-1)` (∈[1.0, 1.5]); key norm gets a *descending*,
    /// sign-alternating pattern `-(1.5 - 0.5·c/(D-1))` — different magnitude curve AND different sign per
    /// channel, so the two are not related by any permutation/scale and the swap cannot cancel out.
    fn make_qk_asymmetric(
        vm: &mut VarMap,
        num_double: usize,
        num_single: usize,
        head_dim: usize,
        dev: &Device,
    ) -> Result<()> {
        let d = head_dim;
        let q_scale: Vec<f32> = (0..d)
            .map(|c| 1.0 + 0.5 * (c as f32) / ((d - 1).max(1) as f32))
            .collect();
        let k_scale: Vec<f32> = (0..d)
            .map(|c| {
                let m = 1.5 - 0.5 * (c as f32) / ((d - 1).max(1) as f32);
                if c % 2 == 0 {
                    -m
                } else {
                    m
                }
            })
            .collect();
        let q = Tensor::from_vec(q_scale, d, dev)?;
        let k = Tensor::from_vec(k_scale, d, dev)?;
        let mut set = |name: String, v: &Tensor| vm.set_one(name, v);
        for i in 0..num_double {
            let s = format!("transformer_blocks.{i}.attn");
            set(format!("{s}.norm_q.weight"), &q)?;
            set(format!("{s}.norm_k.weight"), &k)?;
            // The text stream carries its own QK-norm; make it asymmetric too so a Q↔K swap in the
            // `add_q_proj`/`add_k_proj` concat is likewise caught.
            set(format!("{s}.norm_added_q.weight"), &q)?;
            set(format!("{s}.norm_added_k.weight"), &k)?;
        }
        for i in 0..num_single {
            let s = format!("single_transformer_blocks.{i}.attn");
            set(format!("{s}.norm_q.weight"), &q)?;
            set(format!("{s}.norm_k.weight"), &k)?;
        }
        Ok(())
    }

    /// Swap the two halves of a tensor along dim 0 — the (scale, shift) → (shift, scale) re-ordering the
    /// diffusers→BFL output-head conversion needs (`norm_out.linear` is scale-first `AdaLayerNormContinuous`,
    /// BFL `adaLN_modulation.1` is shift-first). Applies to the `[2·hidden, hidden]` weight and `[2·hidden]`
    /// bias alike (sc-10914).
    fn swap_halves(x: &Tensor) -> Tensor {
        let h = x.dim(0).unwrap() / 2;
        let lo = x.narrow(0, 0, h).unwrap();
        let hi = x.narrow(0, h, h).unwrap();
        Tensor::cat(&[&hi, &lo], 0).unwrap()
    }

    /// Read a vendored tensor out of the populated `VarMap` by its diffusers key.
    fn t(map: &HashMap<String, Var>, key: &str) -> Tensor {
        map.get(key)
            .unwrap_or_else(|| panic!("missing vendored key {key}"))
            .as_tensor()
            .clone()
    }

    /// Remap the vendored **diffusers** DiT weights (already materialized in `vm`) into the **BFL** key
    /// layout the stock `candle_transformers` `Flux` reads, performing the split-QKV → fused-QKV
    /// concatenation the packed tier's layout difference requires. Returns a `HashMap` ready for
    /// `VarBuilder::from_tensors`.
    fn remap_to_bfl(
        vm: &VarMap,
        num_double: usize,
        num_single: usize,
        guidance: bool,
    ) -> HashMap<String, Tensor> {
        let src = vm.data().lock().unwrap();
        let src: HashMap<String, Var> = src.clone();
        let mut out: HashMap<String, Tensor> = HashMap::new();
        let mut put = |k: String, v: Tensor| {
            out.insert(k, v);
        };
        let g = |k: &str| t(&src, k);

        // Input projections + per-embedder rename.
        for wb in ["weight", "bias"] {
            put(format!("img_in.{wb}"), g(&format!("x_embedder.{wb}")));
            put(format!("txt_in.{wb}"), g(&format!("context_embedder.{wb}")));
            put(
                format!("time_in.in_layer.{wb}"),
                g(&format!("time_text_embed.timestep_embedder.linear_1.{wb}")),
            );
            put(
                format!("time_in.out_layer.{wb}"),
                g(&format!("time_text_embed.timestep_embedder.linear_2.{wb}")),
            );
            put(
                format!("vector_in.in_layer.{wb}"),
                g(&format!("time_text_embed.text_embedder.linear_1.{wb}")),
            );
            put(
                format!("vector_in.out_layer.{wb}"),
                g(&format!("time_text_embed.text_embedder.linear_2.{wb}")),
            );
            if guidance {
                put(
                    format!("guidance_in.in_layer.{wb}"),
                    g(&format!("time_text_embed.guidance_embedder.linear_1.{wb}")),
                );
                put(
                    format!("guidance_in.out_layer.{wb}"),
                    g(&format!("time_text_embed.guidance_embedder.linear_2.{wb}")),
                );
            }
            // Output head: diffusers `norm_out.linear` is (scale, shift) — **scale-first**
            // (`AdaLayerNormContinuous`) — whereas BFL `final_layer.adaLN_modulation.1` is (shift, scale)
            // shift-first. So the conversion **swaps the two halves** (verified against the real checkpoint:
            // the diffusers `norm_out.linear` halves are the byte-swapped BFL halves). `proj_out` ≡ BFL
            // `final_layer.linear`. (sc-10914 — swapping here is what anchors the corrected scale-first
            // `OutputHead` chunk order against stock; a straight copy would re-hide the original bug.)
            put(
                format!("final_layer.adaLN_modulation.1.{wb}"),
                swap_halves(&g(&format!("norm_out.linear.{wb}"))),
            );
            put(
                format!("final_layer.linear.{wb}"),
                g(&format!("proj_out.{wb}")),
            );
        }

        // Double blocks: `transformer_blocks.{i}` → `double_blocks.{i}`.
        for i in 0..num_double {
            let s = format!("transformer_blocks.{i}");
            let d = format!("double_blocks.{i}");
            // Modulation: diffusers chunk order [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
            // gate_mlp] matches BFL Modulation2's (mod1=shift,scale,gate ; mod2=shift,scale,gate) — a
            // straight rename of `norm1.linear` → `img_mod.lin`, `norm1_context.linear` → `txt_mod.lin`.
            for wb in ["weight", "bias"] {
                put(
                    format!("{d}.img_mod.lin.{wb}"),
                    g(&format!("{s}.norm1.linear.{wb}")),
                );
                put(
                    format!("{d}.txt_mod.lin.{wb}"),
                    g(&format!("{s}.norm1_context.linear.{wb}")),
                );
            }
            // Fused QKV (image stream): concat split `to_q`/`to_k`/`to_v` on the output rows (dim 0).
            for wb in ["weight", "bias"] {
                let qkv = Tensor::cat(
                    &[
                        &g(&format!("{s}.attn.to_q.{wb}")),
                        &g(&format!("{s}.attn.to_k.{wb}")),
                        &g(&format!("{s}.attn.to_v.{wb}")),
                    ],
                    0,
                )
                .unwrap();
                put(format!("{d}.img_attn.qkv.{wb}"), qkv);
                let add_qkv = Tensor::cat(
                    &[
                        &g(&format!("{s}.attn.add_q_proj.{wb}")),
                        &g(&format!("{s}.attn.add_k_proj.{wb}")),
                        &g(&format!("{s}.attn.add_v_proj.{wb}")),
                    ],
                    0,
                )
                .unwrap();
                put(format!("{d}.txt_attn.qkv.{wb}"), add_qkv);
                // Attn output proj: diffusers `to_out.0` → BFL `img_attn.proj`; `to_add_out` → `txt_attn.proj`.
                put(
                    format!("{d}.img_attn.proj.{wb}"),
                    g(&format!("{s}.attn.to_out.0.{wb}")),
                );
                put(
                    format!("{d}.txt_attn.proj.{wb}"),
                    g(&format!("{s}.attn.to_add_out.{wb}")),
                );
                // FF: diffusers `net.0.proj`/`net.2` → BFL Mlp `0`/`2` (image = img_mlp, context = txt_mlp).
                put(
                    format!("{d}.img_mlp.0.{wb}"),
                    g(&format!("{s}.ff.net.0.proj.{wb}")),
                );
                put(
                    format!("{d}.img_mlp.2.{wb}"),
                    g(&format!("{s}.ff.net.2.{wb}")),
                );
                put(
                    format!("{d}.txt_mlp.0.{wb}"),
                    g(&format!("{s}.ff_context.net.0.proj.{wb}")),
                );
                put(
                    format!("{d}.txt_mlp.2.{wb}"),
                    g(&format!("{s}.ff_context.net.2.{wb}")),
                );
            }
            // QK RMSNorm scales: diffusers `.weight` → BFL `.scale`.
            put(
                format!("{d}.img_attn.norm.query_norm.scale"),
                g(&format!("{s}.attn.norm_q.weight")),
            );
            put(
                format!("{d}.img_attn.norm.key_norm.scale"),
                g(&format!("{s}.attn.norm_k.weight")),
            );
            put(
                format!("{d}.txt_attn.norm.query_norm.scale"),
                g(&format!("{s}.attn.norm_added_q.weight")),
            );
            put(
                format!("{d}.txt_attn.norm.key_norm.scale"),
                g(&format!("{s}.attn.norm_added_k.weight")),
            );
        }

        // Single blocks: `single_transformer_blocks.{i}` → `single_blocks.{i}`.
        for i in 0..num_single {
            let s = format!("single_transformer_blocks.{i}");
            let d = format!("single_blocks.{i}");
            for wb in ["weight", "bias"] {
                // Modulation (3-chunk shift,scale,gate) → BFL Modulation1.
                put(
                    format!("{d}.modulation.lin.{wb}"),
                    g(&format!("{s}.norm.linear.{wb}")),
                );
                // linear1 = concat[to_q; to_k; to_v; proj_mlp] on output rows (BFL packs qkv then mlp).
                let l1 = Tensor::cat(
                    &[
                        &g(&format!("{s}.attn.to_q.{wb}")),
                        &g(&format!("{s}.attn.to_k.{wb}")),
                        &g(&format!("{s}.attn.to_v.{wb}")),
                        &g(&format!("{s}.proj_mlp.{wb}")),
                    ],
                    0,
                )
                .unwrap();
                put(format!("{d}.linear1.{wb}"), l1);
                // linear2 = diffusers `proj_out`.
                put(
                    format!("{d}.linear2.{wb}"),
                    g(&format!("{s}.proj_out.{wb}")),
                );
            }
            put(
                format!("{d}.norm.query_norm.scale"),
                g(&format!("{s}.attn.norm_q.weight")),
            );
            put(
                format!("{d}.norm.key_norm.scale"),
                g(&format!("{s}.attn.norm_k.weight")),
            );
        }

        out
    }

    /// **Vendored diffusers FLUX DiT ≡ stock candle-transformers BFL `Flux` (dense, 1e-4).** The
    /// vendored DiT is built dense into a `VarMap`; its diffusers weights are then remapped into the BFL
    /// key layout (split-QKV → fused-QKV concat, `to_out.0` → `proj`, per-embedder rename) and loaded
    /// into the stock `Flux`. Feeding both the same FLUX inputs, the velocity outputs must agree at
    /// against the reference model (sc-9443). A tiny config keeps it GPU-free.
    ///
    /// **One documented reference divergence.** The diffusers image-stream FF activation is *exact*
    /// GELU (`gelu_erf`) — the vendored DiT reproduces that ([`FeedForward::forward`], `approx=false`
    /// for `ff`) — whereas the stock BFL `Mlp` uses the *tanh-approx* GELU (`candle .gelu()`) on **both**
    /// streams. So a full double-block network differs by the compounded `gelu_erf` vs `gelu_tanh` gap
    /// (~5e-4 per element; ~1e-3 at the DiT output) — a genuine formulation difference between the two
    /// reference impls, not a port bug. Everything else (input/output projections, RoPE, QK-norm, joint
    /// attention, single blocks whose MLP uses the same tanh-GELU on both sides, the AdaLN output head)
    /// is bit-exact. So the test pins **two** anchors on the SAME shared weights:
    ///  - a **tight 1e-4** anchor with `depth = 0` (no double blocks) — the single-block + IO + output-
    ///    head path, which uses the same tanh-GELU on both sides, must match exactly (this is where the
    ///    single-block QKV split→fuse remap, the joint RoPE, and the AdaLN head are pinned);
    ///  - a **full-network** anchor (2 double + 2 single) bounded at 5e-3 — loose enough to absorb the
    ///    documented image-FF GELU gap yet far tighter than any structural bug (a wrong RoPE / QKV split
    ///    / modulation-chunk order diverges by O(1)), so it still guards the double-block wiring.
    ///
    /// **Discriminating the QKV split (sc-9443, review).** QK-RMSNorm + a tiny head_dim (8) + symmetric
    /// random-init `to_q`/`to_k` weights would make Q and K interchangeable at this scale — a Q↔K swap in
    /// the split→fuse concat leaves the output unchanged to inside 1e-4, so the anchor would *not* catch a
    /// transposed QKV split. To close that hole, [`make_qk_asymmetric`] overwrites the shared QK-norm
    /// scales with distinct per-channel patterns for the query vs key norms before either model reads them
    /// (correct remap still matches exactly; a Q↔K swap now diverges by O(1) — verified ~2.9 on the tight
    /// anchor). See that fn for the mechanism.
    fn run_parity(num_double: usize, num_single: usize, guidance: bool, tol: f32) -> Result<f32> {
        run_parity_cfg(parity_cfg(guidance), num_double, num_single, guidance, tol)
    }

    /// As [`run_parity`] but with a caller-supplied base [`Config`] — so a test can pin the vendored DiT
    /// against stock at the **real** FLUX.1 hyperparameters (head_dim 128, `axes_dim` [16,56,56], packed
    /// `in_channels` 64) rather than only the tiny parity config. The tiny config's `axes_dim` [2,2,4]
    /// and head_dim 8 cannot exercise a RoPE / attention-reshape bug that only manifests at the real
    /// per-axis rope widths.
    fn run_parity_cfg(
        base: Config,
        num_double: usize,
        num_single: usize,
        guidance: bool,
        tol: f32,
    ) -> Result<f32> {
        use candle_transformers::models::flux::model::{Config as StockConfig, Flux};
        use candle_transformers::models::flux::WithForward;

        let dev = Device::Cpu;
        let mut cfg = base;
        cfg.guidance_embed = guidance;
        cfg.depth = num_double;
        cfg.depth_single_blocks = num_single;

        // Build the vendored diffusers DiT — this populates the VarMap with diffusers-keyed weights.
        let mut vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let vendored = FluxDit::new_diffusers(&cfg, num_double, num_single, vb)?;

        // Make the shared QK-norm scales Q/K-asymmetric so the parity anchors actually catch a Q↔K
        // transposition in the QKV split (sc-9443); both models read these mutated scales, so a correct
        // remap still matches exactly. Must run before `remap_to_bfl` snapshots the VarMap.
        let head_dim = Dims::from_config(&cfg).head_dim;
        make_qk_asymmetric(&mut vm, num_double, num_single, head_dim, &dev)?;

        // Remap into the BFL layout and build the stock model from those exact weights.
        let bfl = remap_to_bfl(&vm, num_double, num_single, guidance);
        let stock_cfg = StockConfig {
            in_channels: cfg.in_channels,
            vec_in_dim: cfg.vec_in_dim,
            context_in_dim: cfg.context_in_dim,
            hidden_size: cfg.hidden_size,
            mlp_ratio: cfg.mlp_ratio,
            num_heads: cfg.num_heads,
            depth: cfg.depth,
            depth_single_blocks: cfg.depth_single_blocks,
            axes_dim: cfg.axes_dim.clone(),
            theta: cfg.theta,
            qkv_bias: cfg.qkv_bias,
            guidance_embed: cfg.guidance_embed,
        };
        let vb_stock = VarBuilder::from_tensors(bfl, DType::F32, &dev);
        let stock = Flux::new(&stock_cfg, vb_stock)?;

        // Shared FLUX inputs.
        let (b, img_seq, txt_seq) = (1usize, 5usize, 3usize);
        let img = Tensor::randn(0f32, 1f32, (b, img_seq, cfg.in_channels), &dev)?;
        let txt = Tensor::randn(0f32, 1f32, (b, txt_seq, cfg.context_in_dim), &dev)?;
        let pooled = Tensor::randn(0f32, 1f32, (b, cfg.vec_in_dim), &dev)?;
        // Non-trivial position ids (rows differ) so RoPE is actually exercised.
        let mk_ids = |n: usize| -> Result<Tensor> {
            let mut v = Vec::with_capacity(n * 3);
            for r in 0..n {
                v.push(r as f32);
                v.push((r % 2) as f32);
                v.push((r % 3) as f32);
            }
            Tensor::from_vec(v, (b, n, 3), &dev)
        };
        let img_ids = mk_ids(img_seq)?;
        let txt_ids = mk_ids(txt_seq)?;
        let ts = Tensor::full(0.37f32, b, &dev)?;
        let guid = if guidance {
            Some(Tensor::full(3.5f32, b, &dev)?)
        } else {
            None
        };

        let v = vendored.forward(
            &img,
            &img_ids,
            &txt,
            &txt_ids,
            &ts,
            &pooled,
            guid.as_ref(),
            None,
        )?;
        let s = __model_proof_output!(stock.forward(
            &img,
            &img_ids,
            &txt,
            &txt_ids,
            &ts,
            &pooled,
            guid.as_ref()
        ))?;
        assert_eq!(v.dims(), s.dims());
        let d = tensor_max_abs_diff_f32(&v, &s);
        assert!(
            d < tol,
            "vendored diffusers DiT vs stock BFL Flux (double={num_double} single={num_single} \
             guidance={guidance}) max|Δ| = {d} (tol {tol})"
        );
        Ok(d)
    }

    /// Tight anchor: single-block + IO + output-head path is bit-exact against stock (no image-FF GELU
    /// divergence, since `depth = 0`). Runs both the schnell (no guidance) and dev (guidance) heads.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn vendored_dit_single_and_io_matches_stock_bfl_dense() -> Result<()> {
        for guidance in [false, true] {
            let d = run_parity(0, 2, guidance, 1e-4)?;
            assert!(d < 1e-4, "single-block/IO parity max|Δ| = {d}");
        }
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::vendored_dit_single_and_io_matches_stock_bfl_dense");
        Ok(())
    }

    /// Full-network anchor: 2 double + 2 single blocks, bounded at 5e-3 (absorbs only the documented
    /// image-FF `gelu_erf` vs stock `gelu_tanh` gap; a structural bug would diverge by O(1)).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn vendored_dit_full_network_close_to_stock_bfl_dense() -> Result<()> {
        for guidance in [false, true] {
            let d = run_parity(2, 2, guidance, 5e-3)?;
            // Sanity: the divergence must be well within the documented GELU-variant band, not O(1).
            assert!(d < 5e-3, "full-network parity max|Δ| = {d}");
        }
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::vendored_dit_full_network_close_to_stock_bfl_dense");
        Ok(())
    }

    /// The **real** FLUX.1 hyperparameters (sc-10914 investigation): head_dim 128, `axes_dim` [16,56,56],
    /// packed `in_channels` 64, hidden 3072/24 heads. Only the tiny seq lengths keep it GPU-free. This
    /// exercises the RoPE per-axis rope widths + the attention reshape at the widths the shipping tiers
    /// actually use — which the tiny [`parity_cfg`] ([2,2,4], head_dim 8) cannot.
    fn real_cfg() -> Config {
        Config {
            in_channels: 64,
            vec_in_dim: POOLED_DIM,
            context_in_dim: CONTEXT_DIM,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 0,
            depth_single_blocks: 0,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
        }
    }

    /// Real-dims single-block + IO + **output-head** path must be bit-exact against stock. This is the
    /// anchor for the scale-first `OutputHead` chunk order at the real `axes_dim`/head_dim/64-channel
    /// widths the shipping tiers use — paired with the half-swap in [`remap_to_bfl`], reverting the
    /// output head to shift-first breaks it (sc-10914). One double block is already covered by the tiny
    /// [`vendored_dit_full_network_close_to_stock_bfl_dense`]; kept to depth=0 here to bound CI cost.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn vendored_dit_real_config_matches_stock_bfl_dense() -> Result<()> {
        let d = run_parity_cfg(real_cfg(), 0, 1, true, 1e-4)?;
        assert!(d < 1e-4, "real-config single/IO parity max|Δ| = {d}");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::diffusers_tests::vendored_dit_real_config_matches_stock_bfl_dense");
        Ok(())
    }
}

/// SW-6129: FLUX.1's RoPE is the shared `candle_gen::rope::apply_rope_freqs_cis` on the
/// `freqs_cis` rotation matrix.
#[cfg(test)]
mod sw6129_rope_route_tests {
    use super::*;

    /// Deterministic xorshift values in [-2, 2) (no device RNG).
    fn values(dims: &[usize], seed: u64, device: &Device) -> Tensor {
        let n: usize = dims.iter().product();
        let mut s = seed | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0) as f32
            })
            .collect();
        Tensor::from_vec(v, dims, device).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// The former private apply, verbatim.
    fn former_apply(x: &Tensor, freq_cis: &Tensor) -> Tensor {
        let dims = x.dims();
        let (b_sz, n_head, seq_len, n_embd) = x.dims4().unwrap();
        let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2)).unwrap();
        let x0 = x.narrow(D::Minus1, 0, 1).unwrap();
        let x1 = x.narrow(D::Minus1, 1, 1).unwrap();
        let fr0 = freq_cis.get_on_dim(D::Minus1, 0).unwrap();
        let fr1 = freq_cis.get_on_dim(D::Minus1, 1).unwrap();
        (fr0.broadcast_mul(&x0).unwrap() + fr1.broadcast_mul(&x1).unwrap())
            .unwrap()
            .reshape(dims.to_vec())
            .unwrap()
    }

    /// FLUX.1's `EmbedNd` table (`theta 10000`, axes `[16, 56, 56]`) built straight from the `rope`
    /// table builder `EmbedNd` runs per axis (no module call).
    fn pe_table(ids: &Tensor) -> Tensor {
        let axes = [16usize, 56, 56];
        let parts: Vec<Tensor> = (0..3)
            .map(|i| rope(&ids.get_on_dim(D::Minus1, i).unwrap(), axes[i], 10_000).unwrap())
            .collect();
        Tensor::cat(&parts, 2).unwrap().unsqueeze(1).unwrap()
    }

    /// FLUX.1 `[t, h, w]` ids for `txt` text tokens then an `h2 x w2` image grid, `[1, S, 3]`.
    fn ids(txt: usize, h2: usize, w2: usize, dtype: DType, device: &Device) -> Tensor {
        let mut v = vec![0f32; txt * 3];
        for i in 0..h2 {
            for j in 0..w2 {
                v.extend([0.0, i as f32, j as f32]);
            }
        }
        Tensor::from_vec(v, (1, txt + h2 * w2, 3), device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    /// The shared route equals the former apply bit for bit on CPU in f32 / bf16 / f16, with the
    /// real `EmbedNd` (`theta 10000`, axes `[16, 56, 56]`) table in the activation dtype, for a
    /// shared table and a 2-batch q.
    #[test]
    fn shared_rope_route_matches_the_former_private_chain() {
        let x32 = values(&[2, 3, 4 + 12, 128], 7, &Device::Cpu);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let pe = pe_table(&ids(4, 3, 4, dtype, &Device::Cpu));
            let x = x32.to_dtype(dtype).unwrap();
            let got = apply_rope(&x, &pe).unwrap();
            assert_eq!(got.dtype(), dtype);
            assert_eq!(bits(&got), bits(&former_apply(&x, &pe)), "{dtype:?}");
        }
    }

    /// SW-6129 CUDA smoke: flux apply_rope (bf16, freqs_cis matrix, [1,24,4608,128]) on CUDA device 0 takes the fused strided RoPE (every call counted
    /// `fused`) and is bit-identical to the same call with the kernel switch off (the shared
    /// reference chain) — the bf16 / production-dtype run the CPU goldens cannot see.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_bf16_rope_route_is_fused_and_bit_identical() {
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::rope::ROPE_PATH;
        let dev = Device::new_cuda(0).unwrap();
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ROPE_PATH.guard(Some(true));
        // FLUX.1-dev at 1024^2: 512 text + 64x64 image tokens, 24 heads x 128, bf16.
        let pe = pe_table(&ids(512, 64, 64, DType::BF16, &dev));
        let q = values(&[1, 24, 512 + 4096, 128], 3, &dev)
            .to_dtype(DType::BF16)
            .unwrap();
        let k = values(&[1, 24, 512 + 4096, 128], 4, &dev)
            .to_dtype(DType::BF16)
            .unwrap();
        let run = || {
            Tensor::cat(
                &[apply_rope(&q, &pe).unwrap(), apply_rope(&k, &pe).unwrap()],
                0,
            )
            .unwrap()
        };
        ROPE_PATH.set_enabled(Some(true));
        let start = ROPE_PATH.thread_counts();
        let fused = run();
        let on = ROPE_PATH.thread_counts().since(&start);
        assert_eq!((on.fused, on.reference_total()), (2, 0), "{on}");
        ROPE_PATH.set_enabled(Some(false));
        let start = ROPE_PATH.thread_counts();
        let reference = run();
        let off = ROPE_PATH.thread_counts().since(&start);
        assert_eq!(off.reference_for("disabled"), 2, "{off}");
        ROPE_PATH.set_enabled(Some(true));
        let (f, r) = (bits(&fused), bits(&reference));
        let mismatches = f.iter().zip(&r).filter(|(a, b)| a != b).count();
        println!("flux apply_rope (bf16, freqs_cis matrix, [1,24,4608,128]): {mismatches} of {} elements differ; on {on}, off {off}", f.len());
        assert_eq!(mismatches, 0);
    }
}
