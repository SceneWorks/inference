//! Vendored FLUX DiT with the XLabs **IP-Adapter** decoupled-cross-attention seam (sc-5872, epic 5480).
//!
//! A faithful copy of `candle-transformers::models::flux::model` at the workspace candle pin
//! (`65ecb58`), vendored because the stock [`flux::model::Flux`] exposes **no** per-double-block
//! injection point (all blocks/fields are private) — the same "vendor the model to get a seam" move
//! `candle-gen-sdxl` made for the UNet. The txt2img pipeline keeps using the **stock** candle-transformers
//! `Flux` untouched ([`crate::pipeline`]); only this reference/IP path uses the fork.
//!
//! The one structural change vs upstream: [`DoubleStreamBlock::forward`] takes an optional
//! [`FluxIpInjector`] + the block index, and — after the standard joint-attention + gated FF — adds the
//! XLabs IP residual **raw (ungated)** to the image stream's block output (diffusers
//! `hidden_states = hidden_states + ip_attn_output`). The IP query is the image stream's **post-QkNorm,
//! pre-RoPE** query (candle's `img_attn.qkv` already produces exactly that), matching the MLX
//! `mlx-gen-flux` port and diffusers' `FluxIPAdapterAttnProcessor`. Single-stream blocks are not
//! injected (XLabs only adapts the 19 double blocks). With `ip = None` this is byte-identical to upstream.
//!
//! Config is **reused** from candle-transformers (`Config::dev()` / `Config::schnell()`) so the fork
//! cannot drift on the FLUX hyperparameters; the BFL checkpoint layout is identical, so `IpFlux::new`
//! loads the same `flux1-{dev,schnell}.safetensors` the stock model does.

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
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
/// the stream untouched. The two seams are orthogonal — [`IpFlux::forward`] drives the IP one,
/// [`IpFlux::forward_injected`] drives this one.
pub trait DitImageInjector {
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

pub(crate) fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())
}

pub(crate) fn attention_planned(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pe: &Tensor,
    plan: AttentionPlan<'_>,
) -> candle_gen::Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention_planned(&q, &k, v, plan)?;
    Ok(x.transpose(1, 2)?.flatten_from(2)?)
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
    #[allow(unused)]
    dim: usize,
    theta: usize,
    axes_dim: Vec<usize>,
    #[cfg(test)]
    forward_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl EmbedNd {
    pub(crate) fn new(dim: usize, theta: usize, axes_dim: Vec<usize>) -> Self {
        Self {
            dim,
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

#[derive(Debug, Clone)]
struct MlpEmbedder {
    in_layer: QLinear,
    out_layer: QLinear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: VarBuilder) -> Result<Self> {
        let in_layer = QLinear::linear(in_sz, h_sz, vb.pp("in_layer"))?;
        let out_layer = QLinear::linear(h_sz, h_sz, vb.pp("out_layer"))?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }
}

impl candle_core::Module for MlpEmbedder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.out_layer.forward(&self.in_layer.forward(xs)?.silu()?)
    }
}

#[derive(Debug, Clone)]
struct QkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl QkNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let query_norm = vb.get(dim, "query_norm.scale")?;
        let query_norm = RmsNorm::new(query_norm, 1e-6);
        let key_norm = vb.get(dim, "key_norm.scale")?;
        let key_norm = RmsNorm::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
}

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1 {
    lin: QLinear,
}

impl Modulation1 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = QLinear::linear(dim, 3 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Modulation2 {
    lin: QLinear,
}

impl Modulation2 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = QLinear::linear(dim, 6 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = vec_
            .silu()?
            .apply(&self.lin)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    qkv: QLinear,
    norm: QkNorm,
    proj: QLinear,
    num_heads: usize,
}

impl SelfAttention {
    fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = QLinear::linear_detect(dim, dim * 3, &vb, "qkv", qkv_bias)?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = QLinear::linear_detect(dim, dim, &vb, "proj", true)?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.qkv.forward(xs)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    lin1: QLinear,
    lin2: QLinear,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = QLinear::linear_detect(in_sz, mlp_sz, &vb, "0", true)?;
        let lin2 = QLinear::linear_detect(mlp_sz, in_sz, &vb, "2", true)?;
        Ok(Self { lin1, lin2 })
    }
}

impl candle_core::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.lin2.forward(&self.lin1.forward(xs)?.gelu()?)
    }
}

#[derive(Debug, Clone)]
struct DoubleStreamBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let img_mod = Modulation2::new(h_sz, vb.pp("img_mod"))?;
        let img_norm1 = layer_norm(h_sz, vb.pp("img_norm1"))?;
        let img_attn = SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"))?;
        let img_norm2 = layer_norm(h_sz, vb.pp("img_norm2"))?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("img_mlp"))?;
        let txt_mod = Modulation2::new(h_sz, vb.pp("txt_mod"))?;
        let txt_norm1 = layer_norm(h_sz, vb.pp("txt_norm1"))?;
        let txt_attn = SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"))?;
        let txt_norm2 = layer_norm(h_sz, vb.pp("txt_norm2"))?;
        let txt_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("txt_mlp"))?;
        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
        })
    }

    /// As upstream `DoubleStreamBlock::forward`, but consulting the XLabs IP-Adapter seam when
    /// `ip = Some((injector, block_idx))`. The IP query is the image stream's **post-QkNorm, pre-RoPE**
    /// query (`img_attn.qkv`'s `img_q`, captured before the text concat + RoPE); the injector computes
    /// the decoupled-cross-attention residual and the block adds it **raw (ungated)** to the image
    /// stream's block output (after the gated FF) — diffusers `hidden_states = hidden_states +
    /// ip_attn_output`. `ip = None` is byte-identical to upstream.
    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
        ip: Option<(&FluxIpInjector, usize)>,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?; // shift, scale, gate
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?; // shift, scale, gate
        let img_modulated = img.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_modulated)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;
        // The IP query is captured here — post-QkNorm, pre-RoPE — before `img_q` is consumed by the
        // text concat below. Cloned only when an injector is present (otherwise zero overhead).
        let ip_img_q = ip.map(|_| img_q.clone());

        let txt_modulated = txt.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_modulated)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention_planned(&q, &k, &v, pe, attention_plan)?;
        let txt_attn = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&self.img_attn.proj.forward(&img_attn)?))?;
        let img = (&img
            + img_mod2.gate(
                &img_mod2
                    .scale_shift(&img.apply(&self.img_norm2)?)?
                    .apply(&self.img_mlp)?,
            )?)?;

        // XLabs IP-Adapter: add the decoupled-cross-attention residual RAW (ungated) to the image
        // stream's block output — after the gated FF, bypassing `gate_msa` and the FF input entirely
        // (diffusers `transformer_flux.py`). `ip = None` / a 0-scale injector leaves `img` untouched.
        let img = match (ip, ip_img_q) {
            (Some((inj, block_idx)), Some(q)) => match inj.double_block_residual(block_idx, &q)? {
                Some(r) => (&img + r.to_dtype(img.dtype())?)?,
                None => img,
            },
            _ => img,
        };

        let txt = (txt + txt_mod1.gate(&self.txt_attn.proj.forward(&txt_attn)?))?;
        let txt = (&txt
            + txt_mod2.gate(
                &txt_mod2
                    .scale_shift(&txt.apply(&self.txt_norm2)?)?
                    .apply(&self.txt_mlp)?,
            )?)?;

        Ok((img, txt))
    }
}

#[derive(Debug, Clone)]
struct SingleStreamBlock {
    linear1: QLinear,
    linear2: QLinear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_heads;
        let linear1 = QLinear::linear_detect(h_sz, h_sz * 3 + mlp_sz, &vb, "linear1", true)?;
        let linear2 = QLinear::linear_detect(h_sz + mlp_sz, h_sz, &vb, "linear2", true)?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let pre_norm = layer_norm(h_sz, vb.pp("pre_norm"))?;
        let modulation = Modulation1::new(h_sz, vb.pp("modulation"))?;
        Ok(Self {
            linear1,
            linear2,
            norm,
            pre_norm,
            modulation,
            h_sz,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        let mod_ = self.modulation.forward(vec_)?;
        let x_mod = mod_.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = self.linear1.forward(&x_mod)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        let attn = attention_planned(&q, &k, &v, pe, attention_plan)?;
        let output = self
            .linear2
            .forward(&Tensor::cat(&[attn, mlp.gelu()?], 2)?)?;
        Ok((xs + mod_.gate(&output)?)?)
    }
}

#[derive(Debug, Clone)]
struct LastLayer {
    norm_final: LayerNorm,
    linear: QLinear,
    ada_ln_modulation: QLinear,
}

impl LastLayer {
    fn new(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.pp("norm_final"))?;
        let linear = QLinear::linear_detect(h_sz, p_sz * p_sz * out_c, &vb, "linear", true)?;
        let ada_ln_modulation = QLinear::linear(h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        self.linear.forward(&xs)
    }
}

/// The vendored FLUX DiT with the XLabs IP-Adapter seam — a fork of candle-transformers'
/// `flux::model::Flux`. [`forward`](Self::forward) is the upstream `WithForward::forward` plus an
/// optional [`FluxIpInjector`] threaded into the 19 double blocks.
#[derive(Clone)]
pub struct IpFlux {
    img_in: QLinear,
    txt_in: QLinear,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    blocks: FluxBlocks,
    final_layer: LastLayer,
    device: candle_core::Device,
}

#[derive(Clone)]
enum FluxBlocks {
    Resident {
        double: Vec<DoubleStreamBlock>,
        single: Vec<SingleStreamBlock>,
    },
    Streamed {
        weights: VarBuilder<'static>,
        config: Config,
    },
}

impl IpFlux {
    pub(crate) fn prepare_conditioning(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
    ) -> candle_gen::Result<PreparedConditioning> {
        let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
        let pe = ids.apply(&self.pe_embedder)?;
        Ok(PreparedConditioning {
            img_shape: img.dims().to_vec(),
            img_ids_shape: img_ids.dims().to_vec(),
            txt_shape: txt.dims().to_vec(),
            txt_ids_shape: txt_ids.dims().to_vec(),
            dtype: img.dtype(),
            img_ids_dtype: img_ids.dtype(),
            txt_dtype: txt.dtype(),
            txt_ids_dtype: txt_ids.dtype(),
            device: img.device().location(),
            pe,
        })
    }

    /// The number of XLabs-adapted double blocks (the IP adapter carries exactly this many K/V pairs).
    pub fn num_double_blocks(&self) -> usize {
        match &self.blocks {
            FluxBlocks::Resident { double, .. } => double.len(),
            FluxBlocks::Streamed { config, .. } => config.depth,
        }
    }

    /// Walk the native BFL-layout FLUX.1 adapter surface. This includes the fused qkv/linear1
    /// projections used by native BFL/ComfyUI LoRAs and all global embedders/modulation projections.
    pub fn visit_adaptable_mut(
        &mut self,
        visitor: &mut dyn FnMut(&str, &mut candle_gen::quant::AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        visitor("img_in", &mut self.img_in)?;
        visitor("txt_in", &mut self.txt_in)?;
        for (prefix, embedder) in [
            ("time_in", &mut self.time_in),
            ("vector_in", &mut self.vector_in),
        ] {
            visitor(&format!("{prefix}.in_layer"), &mut embedder.in_layer)?;
            visitor(&format!("{prefix}.out_layer"), &mut embedder.out_layer)?;
        }
        if let Some(embedder) = &mut self.guidance_in {
            visitor("guidance_in.in_layer", &mut embedder.in_layer)?;
            visitor("guidance_in.out_layer", &mut embedder.out_layer)?;
        }
        match &mut self.blocks {
            FluxBlocks::Resident { double, single } => {
                for (index, block) in double.iter_mut().enumerate() {
                    let prefix = format!("double_blocks.{index}");
                    visitor(&format!("{prefix}.img_mod.lin"), &mut block.img_mod.lin)?;
                    visitor(&format!("{prefix}.img_attn.qkv"), &mut block.img_attn.qkv)?;
                    visitor(&format!("{prefix}.img_attn.proj"), &mut block.img_attn.proj)?;
                    visitor(&format!("{prefix}.img_mlp.0"), &mut block.img_mlp.lin1)?;
                    visitor(&format!("{prefix}.img_mlp.2"), &mut block.img_mlp.lin2)?;
                    visitor(&format!("{prefix}.txt_mod.lin"), &mut block.txt_mod.lin)?;
                    visitor(&format!("{prefix}.txt_attn.qkv"), &mut block.txt_attn.qkv)?;
                    visitor(&format!("{prefix}.txt_attn.proj"), &mut block.txt_attn.proj)?;
                    visitor(&format!("{prefix}.txt_mlp.0"), &mut block.txt_mlp.lin1)?;
                    visitor(&format!("{prefix}.txt_mlp.2"), &mut block.txt_mlp.lin2)?;
                }
                for (index, block) in single.iter_mut().enumerate() {
                    let prefix = format!("single_blocks.{index}");
                    visitor(&format!("{prefix}.linear1"), &mut block.linear1)?;
                    visitor(&format!("{prefix}.linear2"), &mut block.linear2)?;
                    visitor(
                        &format!("{prefix}.modulation.lin"),
                        &mut block.modulation.lin,
                    )?;
                }
            }
            FluxBlocks::Streamed { .. } => {
                candle_core::bail!(
                    "FLUX adapters require resident transformer blocks; disable block streaming"
                )
            }
        }
        visitor("final_layer.linear", &mut self.final_layer.linear)?;
        visitor(
            "final_layer.adaLN_modulation.1",
            &mut self.final_layer.ada_ln_modulation,
        )
    }

    pub fn new(cfg: &Config, vb: VarBuilder<'static>) -> Result<Self> {
        Self::new_with_blocks(cfg, vb, false)
    }

    pub(crate) fn new_block_streamed(cfg: &Config, vb: VarBuilder<'static>) -> Result<Self> {
        Self::new_with_blocks(cfg, vb, true)
    }

    fn new_with_blocks(cfg: &Config, vb: VarBuilder<'static>, stream_blocks: bool) -> Result<Self> {
        let img_in = QLinear::linear_detect(cfg.in_channels, cfg.hidden_size, &vb, "img_in", true)?;
        let txt_in =
            QLinear::linear_detect(cfg.context_in_dim, cfg.hidden_size, &vb, "txt_in", true)?;
        let blocks = if stream_blocks {
            FluxBlocks::Streamed {
                weights: vb.clone(),
                config: cfg.clone(),
            }
        } else {
            let mut double = Vec::with_capacity(cfg.depth);
            let vb_d = vb.pp("double_blocks");
            for idx in 0..cfg.depth {
                double.push(DoubleStreamBlock::new(cfg, vb_d.pp(idx))?);
            }
            let mut single = Vec::with_capacity(cfg.depth_single_blocks);
            let vb_s = vb.pp("single_blocks");
            for idx in 0..cfg.depth_single_blocks {
                single.push(SingleStreamBlock::new(cfg, vb_s.pp(idx))?);
            }
            FluxBlocks::Resident { double, single }
        };
        let time_in = MlpEmbedder::new(256, cfg.hidden_size, vb.pp("time_in"))?;
        let vector_in = MlpEmbedder::new(cfg.vec_in_dim, cfg.hidden_size, vb.pp("vector_in"))?;
        let guidance_in = if cfg.guidance_embed {
            let mlp = MlpEmbedder::new(256, cfg.hidden_size, vb.pp("guidance_in"))?;
            Some(mlp)
        } else {
            None
        };
        let final_layer =
            LastLayer::new(cfg.hidden_size, 1, cfg.in_channels, vb.pp("final_layer"))?;
        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, cfg.theta, cfg.axes_dim.to_vec());
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            blocks,
            final_layer,
            device: vb.device().clone(),
        })
    }

    /// The upstream FLUX `forward`, plus the optional XLabs IP injector threaded into every double
    /// block. `ip = None` is byte-identical to candle-transformers `Flux::forward`.
    ///
    /// A thin wrapper over the shared `forward_core`: the XLabs IP seam is the
    /// mid-block one (threaded into `block.forward`), so this engages ONLY `ip` — no post-block
    /// [`DitImageInjector`] and no control residuals. Byte-identical to the pre-consolidation body: with
    /// `injector = None` / `control = None` every post-block `if let Some(..)` is skipped and the loop
    /// reduces to the exact upstream sequence.
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
            self.num_double_blocks()
                + match &self.blocks {
                    FluxBlocks::Resident { single, .. } => single.len(),
                    FluxBlocks::Streamed { config, .. } => config.depth_single_blocks,
                },
            &candle_gen::gen_core::CancelFlag::default(),
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
        cancel: &candle_gen::gen_core::CancelFlag,
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
        )
    }

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
        cancel: &candle_gen::gen_core::CancelFlag,
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
            &candle_gen::gen_core::CancelFlag::default(),
            None,
        )
        .map_err(|error| match error {
            candle_gen::CandleError::Candle(error) => error,
            other => candle_core::Error::Msg(other.to_string()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_injected_with_memory(
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
        cancel: &candle_gen::gen_core::CancelFlag,
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
            &candle_gen::gen_core::CancelFlag::default(),
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
        cancel: &candle_gen::gen_core::CancelFlag,
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
        )
    }

    /// The single shared FLUX DiT forward body backing [`forward`], [`forward_injected`], and
    /// [`forward_control`] (sc-9003 / F-023 — the three used to triplicate this ~60-line body, letting
    /// the parity-critical preamble and block loop drift independently). It threads all three orthogonal
    /// seams as explicit parameters; each public entry point engages exactly the seam(s) it owns and
    /// leaves the rest `None`, so every arm stays byte-identical to its pre-consolidation body:
    ///
    /// - `ip` — the XLabs IP-Adapter **mid-block** seam (the decoupled-cross-attn residual computed from
    ///   the captured pre-RoPE image query INSIDE each double block). Only [`forward`] engages it.
    /// - `injector` — the generic **post-block** [`DitImageInjector`] seam (PuLID id cross-attn):
    ///   `after_double` on the image stream and `after_single` on the image-token tail of the joint
    ///   stream. [`forward_injected`] and [`forward_control`] engage it.
    /// - `control` — the Fun-Controlnet-Union per-double-block residuals `(residuals, scale)`, pre-scaled
    ///   once and added after the injector residual at the diffusers interval. Only [`forward_control`]
    ///   engages it.
    ///
    /// With `ip = None`, `injector = None`, `control = None` this is the verbatim upstream
    /// candle-transformers `Flux::forward`.
    #[allow(clippy::too_many_arguments)]
    fn forward_core(
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
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &candle_gen::gen_core::CancelFlag,
        prepared: Option<&PreparedConditioning>,
    ) -> candle_gen::Result<Tensor> {
        if txt.rank() != 3 {
            return Err(candle_gen::CandleError::Msg(format!(
                "unexpected shape for txt {:?}",
                txt.shape()
            )));
        }
        if img.rank() != 3 {
            return Err(candle_gen::CandleError::Msg(format!(
                "unexpected shape for img {:?}",
                img.shape()
            )));
        }
        let dtype = img.dtype();
        let pe = match prepared {
            Some(prepared) => {
                prepared.validate(img, img_ids, txt, txt_ids)?;
                prepared.pe.clone()
            }
            None => {
                let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
                ids.apply(&self.pe_embedder)?
            }
        };
        let mut txt = self.txt_in.forward(txt)?;
        let mut img = self.img_in.forward(img)?;
        let vec_ = timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        // ControlNet residual injection interval (diffusers `FluxTransformer2DModel`):
        // `interval = ceil(num_double_blocks / num_control_residuals)`, and after base double block `i`
        // we add `controlnet_block_samples[i / interval]` (scaled). The Shakker Union-Pro-2.0 ships 6
        // control double blocks → interval `ceil(19/6) = 4`. An empty residual slice is treated as "no
        // control". The residuals are pre-scaled once before the 19-block loop rather than per block.
        let control = control.filter(|(res, _)| !res.is_empty());
        let scaled_control = match control {
            Some((res, scale)) => {
                let interval = control_residual_interval(self.num_double_blocks(), res.len());
                let scaled = res
                    .iter()
                    .map(|r| r.to_dtype(dtype)? * scale)
                    .collect::<Result<Vec<_>>>()?;
                Some((scaled, interval))
            }
            None => None,
        };

        if let FluxBlocks::Streamed { weights, config } = &self.blocks {
            let window = transformer_window.max(1);
            let double_plan = candle_gen::block_window::BlockPlan::new(config.depth, window)?;
            let (img, txt) = candle_gen::block_window::run_windowed(
                &self.device,
                &double_plan,
                cancel,
                (img, txt),
                || Ok(weights.clone()),
                |(mut img, mut txt), view, range| {
                    let blocks = range
                        .map(|index| {
                            DoubleStreamBlock::new(config, view.pp("double_blocks").pp(index))
                                .map(|block| (index, block))
                                .map_err(candle_gen::CandleError::from)
                        })
                        .collect::<candle_gen::Result<Vec<_>>>()?;
                    for (index, block) in &blocks {
                        candle_gen::check_cancel(cancel)?;
                        (img, txt) = block.forward(
                            &img,
                            &txt,
                            &vec_,
                            &pe,
                            ip.map(|inj| (inj, *index)),
                            attention_plan,
                        )?;
                        if let Some(inj) = injector {
                            if let Some(r) = inj.after_double(*index, &img)? {
                                img = (&img + r.to_dtype(img.dtype())?)?;
                            }
                        }
                        if let Some((res, interval)) = &scaled_control {
                            let idx = (*index / interval).min(res.len() - 1);
                            img = (&img + &res[idx])?;
                        }
                    }
                    Ok((img, txt))
                },
            )?;
            let txt_len = txt.dim(1)?;
            let joint = Tensor::cat(&[&txt, &img], 1)?;
            let single_plan =
                candle_gen::block_window::BlockPlan::new(config.depth_single_blocks, window)?;
            let joint = candle_gen::block_window::run_windowed(
                &self.device,
                &single_plan,
                cancel,
                joint,
                || Ok(weights.clone()),
                |mut joint, view, range| {
                    let blocks = range
                        .map(|index| {
                            SingleStreamBlock::new(config, view.pp("single_blocks").pp(index))
                                .map(|block| (index, block))
                                .map_err(candle_gen::CandleError::from)
                        })
                        .collect::<candle_gen::Result<Vec<_>>>()?;
                    for (index, block) in &blocks {
                        candle_gen::check_cancel(cancel)?;
                        joint = block.forward(&joint, &vec_, &pe, attention_plan)?;
                        if let Some(inj) = injector {
                            if inj.injects_after_single(*index) {
                                let seq = joint.dim(1)?;
                                let img_part = joint.narrow(1, txt_len, seq - txt_len)?;
                                if let Some(r) = inj.after_single(*index, &img_part)? {
                                    let added = (img_part + r.to_dtype(joint.dtype())?)?;
                                    let txt_part = joint.narrow(1, 0, txt_len)?;
                                    joint = Tensor::cat(&[&txt_part, &added], 1)?;
                                }
                            }
                        }
                    }
                    Ok(joint)
                },
            )?;
            let img = joint.i((.., txt_len..))?;
            return Ok(self.final_layer.forward(&img, &vec_)?);
        }
        let FluxBlocks::Resident {
            double: double_blocks,
            single: single_blocks,
        } = &self.blocks
        else {
            unreachable!("streamed FLUX blocks returned above")
        };

        // Double blocks: the XLabs IP seam is consulted INSIDE each block (`ip`, the image-query seam);
        // the post-block `injector` residual and then the control residual are added to the image stream
        // afterwards. Each seam is inert when its argument is `None`, so a single-seam caller reduces to
        // exactly its former body.
        for (i, block) in double_blocks.iter().enumerate() {
            candle_gen::check_cancel(cancel)?;
            (img, txt) = block.forward(
                &img,
                &txt,
                &vec_,
                &pe,
                ip.map(|inj| (inj, i)),
                attention_plan,
            )?;
            // Identity injector (PuLID / IP-Adapter) first — composes with control below.
            if let Some(inj) = injector {
                if let Some(r) = inj.after_double(i, &img)? {
                    img = (&img + r.to_dtype(img.dtype())?)?;
                }
            }
            // Fun-Controlnet-Union residual, added AFTER the identity injector so the two compose:
            // `img = img + controlnet_block_samples[i / interval]·scale`.
            if let Some((res, interval)) = &scaled_control {
                let idx = (i / interval).min(res.len() - 1);
                img = (&img + &res[idx])?;
            }
        }

        // Single blocks operate on the joint `cat(txt, img)`; XLabs adapts only the double stream, so the
        // IP seam is never engaged here. The post-block `injector` residual is added to the image-token
        // tail (and written back) after the blocks it opts into. The Shakker Union-Pro-2.0 checkpoint has
        // 0 control SINGLE blocks (diffusers `controlnet_single_block_samples = None`), so control is not
        // consulted here.
        let txt_len = txt.dim(1)?;
        let mut joint = Tensor::cat(&[&txt, &img], 1)?;
        for (i, block) in single_blocks.iter().enumerate() {
            candle_gen::check_cancel(cancel)?;
            joint = block.forward(&joint, &vec_, &pe, attention_plan)?;
            if let Some(inj) = injector {
                if inj.injects_after_single(i) {
                    let seq = joint.dim(1)?;
                    let img_part = joint.narrow(1, txt_len, seq - txt_len)?;
                    if let Some(r) = inj.after_single(i, &img_part)? {
                        let added = (img_part + r.to_dtype(joint.dtype())?)?;
                        let txt_part = joint.narrow(1, 0, txt_len)?;
                        joint = Tensor::cat(&[&txt_part, &added], 1)?;
                    }
                }
            }
        }
        let img = joint.i((.., txt_len..))?;
        Ok(self.final_layer.forward(&img, &vec_)?)
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
mod tests {
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
        let single = candle_gen::sdpa_budgeted_flat(&q, &k, &v, scale, sm, usize::MAX).unwrap();
        // budget = n·s·block = 2·7·3 = 42 → block = 3.
        for budget in [1usize, 42] {
            assert_close(
                &single,
                &candle_gen::sdpa_budgeted_flat(&q, &k, &v, scale, sm, budget).unwrap(),
            );
        }

        let sk = 5usize;
        let kx = Tensor::randn(0f32, 1f32, (n, sk, d), &dev).unwrap();
        let vx = Tensor::randn(0f32, 1f32, (n, sk, d), &dev).unwrap();
        let single = candle_gen::sdpa_budgeted_flat(&q, &kx, &vx, scale, sm, usize::MAX).unwrap();
        // budget = n·sk·block = 2·5·3 = 30 → block = 3 (chunks 3, 3, 1 over the 7 query rows).
        for budget in [1usize, 30] {
            assert_close(
                &single,
                &candle_gen::sdpa_budgeted_flat(&q, &kx, &vx, scale, sm, budget).unwrap(),
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

    /// Build a tiny random-weight [`IpFlux`] on CPU. VarMap zero-init is randomized so the forward is a
    /// non-degenerate function (a zero DiT would make the wrapper-vs-wrapper parity vacuous).
    fn tiny_ipflux(dev: &Device) -> IpFlux {
        tiny_ipflux_pair(dev).0
    }

    fn tiny_ipflux_pair(dev: &Device) -> (IpFlux, IpFlux) {
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
        let resident = IpFlux::new(&cfg, vb.clone()).expect("tiny resident IpFlux");
        let streamed = IpFlux::new_block_streamed(&cfg, vb).expect("tiny block-streamed IpFlux");
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
    }

    /// sc-9003 / F-023: the three public forwards were consolidated onto one `forward_core`. Each is a
    /// thin wrapper engaging only its own seam, so with every optional injector `None` all three must be
    /// byte-identical — the documented invariant (`forward_control(.., None) ≡ forward_injected(.., None)
    /// ≡ forward(.., None)`). This locks that the consolidation didn't perturb the shared body.
    #[test]
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
    }
}
