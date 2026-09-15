//! The **`WanTransformer3DModel`** DiT (TI2V-5B, dense) — a port of diffusers `transformer_wan.py`.
//! 30 blocks, each: AdaLN-modulated self-attention (3-axis interleaved RoPE, full-dim qk-RMSNorm) →
//! ungated cross-attention to the UMT5 context → AdaLN-modulated gated GELU FFN. The per-block
//! 6-vector modulation is `scale_shift_table + time_proj`; the head uses a separate 2-vector.
//!
//! Runs in **bf16** (the 5B checkpoint's native dtype) with norms / modulation / RoPE upcast to f32,
//! mirroring diffusers' `FP32LayerNorm` + `.float()` modulation.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Linear, VarBuilder};

use crate::config::TransformerConfig;
use crate::gguf::{GgufDit, WeightSrc};
use crate::quant::QLinear;
use crate::rope::apply_rope;

/// Dense Linear loader — retained for the VACE model (`vace.rs`) and the training DiT (`dit_train.rs`),
/// whose tiers are not packed. The inference DiT Linears route through [`qlinear`] (packed-detect).
pub(crate) fn linear(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((out_c, in_c), "weight")?,
        Some(vb.get(out_c, "bias")?),
    ))
}

/// LayerNorm over the last dim with no learnable affine, in f32.
pub(crate) fn ln_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// RMSNorm over the last dim (qk-norm "across heads") with affine weight, in f32.
pub(crate) fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(dt)
}

/// sc-12894 — Wan's **denoise self-attention** scores-chunk budget, deliberately smaller than the
/// shared [`candle_gen::ATTN_SCORES_BUDGET`] (1e9). sc-12434 chunks the query rows so the full
/// `[B,H,S,S]` matrix is never materialized, but at the A14B 720p ceiling (S ≈ 75.6k tokens, 40 heads)
/// the *per-chunk* transient — the block's `[B,H,block,Sk]` scores + softmax probs, upcast to f32 for
/// the softmax — is the DENOISE peak's dominant term (~8 GiB at the 1e9 budget, GPU-proven via the #95
/// `USED_MEM_HIGH` probe: the A14B q4 true peak was 30.11 GiB, denoise-owned). The transient scales
/// linearly with the budget (fewer score elements per chunk, proportionally more chunks), so an 8×
/// smaller budget shrinks it to ~1 GiB and drops the A14B true peak under the 24 GiB card. Still far
/// below `i32::MAX` (2.147e9) — sc-12434's CUDA-i32 overflow safety is preserved (strictly tightened),
/// and cross-attention (S_kv = text tokens, ≪ budget) stays a single un-chunked pass, untouched.
const WAN_SELF_ATTN_SCORES_BUDGET: usize = 125_000_000;

// sc-12894 compile-time invariants: the Wan budget must not exceed the shared i32-safe ceiling, and
// must itself stay under candle's CUDA i32 element limit — so the sc-12434 overflow safety is preserved
// (strictly tightened) no matter how this const is edited.
const _: () = assert!(WAN_SELF_ATTN_SCORES_BUDGET <= candle_gen::ATTN_SCORES_BUDGET);
const _: () = assert!(WAN_SELF_ATTN_SCORES_BUDGET < i32::MAX as usize);

/// sc-12894 measurement knob (mirrors the `WAN_VRAM_*` probe envs): override
/// [`WAN_SELF_ATTN_SCORES_BUDGET`] for a GPU A/B budget sweep without a rebuild. Unset / non-positive /
/// unparseable ⇒ the shipped default. Read once (the sweep sets it before the first render).
fn wan_self_attn_budget() -> usize {
    use std::sync::OnceLock;
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("WAN_ATTN_SCORES_BUDGET")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&b| b > 0)
            .unwrap_or(WAN_SELF_ATTN_SCORES_BUDGET)
    })
}

/// sc-12894 measurement knob: when set (`WAN_ATTN_SOFTMAX_BF16=1`), run the scores softmax in the
/// scores' native **bf16** instead of the f32 upcast — the candle CUDA softmax kernel max-stabilizes
/// and accumulates the sum in f32 regardless (`SOFTMAX_OP(__nv_bfloat16, float, …)`), so only the
/// `exp`/probs carry bf16 rounding, which halves the per-chunk transient. Off by default (the f32
/// upcast is numerically exact, and the chunk lever alone perturbs only the f32 rounding order — close
/// to, but not bit-identical to, the un-chunked pass; see SC-15943 and [`sdpa_budgeted`]); gated on only
/// after a parity A/B confirms the bf16 path stays within tolerance. Read once.
fn wan_self_attn_bf16_softmax() -> bool {
    use std::sync::OnceLock;
    static BF16: OnceLock<bool> = OnceLock::new();
    *BF16.get_or_init(|| {
        std::env::var("WAN_ATTN_SOFTMAX_BF16")
            .ok()
            .map(|v| {
                let v = v.trim();
                v == "1" || v.eq_ignore_ascii_case("true")
            })
            .unwrap_or(false)
    })
}

/// Scaled-dot-product attention. `q,k,v`: `[B, H, S*, d]`; softmax upcast to f32 (bf16 under the
/// sc-12894 knob).
///
/// Delegates to the shared i32-overflow-safe [`candle_gen::sdpa_budgeted_bhsd`] — the sc-6217 /
/// sc-9116 query-row chunking Qwen and Krea already carry, ported to Wan here (sc-12434). It chunks
/// over the query rows once the `[B, H, Sq, Sk]` score block would exceed [`wan_self_attn_budget`], so
/// the full `[B, H, S, S]` matrix is never materialized. Both A14B experts and the 5B share this one
/// attention; at every advertised A14B geometry the un-chunked score block (S ≈ 33k tokens at 480p, up
/// to ≈ 76k at the 720p `MAX_AREA_14B` ceiling; 40 heads) is hundreds of GiB and OOMs a 96 GB card
/// before the first denoise step — chunking caps each block's transient near the budget instead. The
/// budget is Wan's own reduced [`WAN_SELF_ATTN_SCORES_BUDGET`] (sc-12894): small enough that the
/// per-chunk f32 scores + probs transient fits the denoise peak under 24 GiB, still ≪ `i32::MAX`. Each
/// query row's softmax is over all keys and independent of the others, so the chunked result is
/// *mathematically* equal to the single pass — but **not bitwise** equal to it: narrowing the query axis
/// changes the GEMM `M`, so the f32 accumulation order may change (SC-15943). The
/// chunking engages only on the over-budget denoise self-attention and stays a no-op single pass — that
/// one byte-identical by construction, being literally the same call — for the small cross-attention
/// (S_kv = text tokens) and every in-budget size.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let dtype = q.dtype();
    let bf16 = wan_self_attn_bf16_softmax();
    sdpa_budgeted(
        q,
        k,
        v,
        scale,
        wan_self_attn_budget(),
        move |scores: &Tensor| {
            if bf16 {
                // sc-12894: softmax in the scores' native bf16 (the CUDA kernel still max-stabilizes
                // and sums in f32) — no f32 upcast/downcast pair, so the per-chunk transient halves.
                softmax_last_dim(scores)
            } else {
                softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(dtype)
            }
        },
    )
}

/// [`sdpa`] with an explicit scores-element `budget` (the query-row chunk threshold) and `softmax`
/// closure — the shared budgeted attention both production and the tests route through, so the test's
/// call-counting proof exercises the same chunking the render uses. Production fixes the budget at
/// [`candle_gen::ATTN_SCORES_BUDGET`] and passes the f32-upcast softmax; the test drives a tiny budget
/// and a counting wrapper of that same softmax. Wan self- and cross-attention carry no mask, so `mask`
/// is always `None`.
fn sdpa_budgeted(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    budget: usize,
    softmax: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    candle_gen::sdpa_budgeted_bhsd(q, k, v, scale, None, softmax, budget)
}

struct Attention {
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
}

/// Request-scoped K/V heads for one block's cross-attention.  These depend only on the projected
/// text context, never on the noisy latent or timestep, so the denoise loop must reuse them.
pub(crate) struct PreparedBlockCrossKv {
    key: Tensor,
    value: Tensor,
}

/// Request-scoped cross-attention K/V heads for every base Wan block.
pub(crate) struct PreparedWanCrossKv {
    blocks: Vec<PreparedBlockCrossKv>,
}

#[cfg(test)]
static CROSS_KV_PREPARATION_PAIRS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
static CROSS_KV_PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn lock_cross_kv_probe() -> std::sync::MutexGuard<'static, ()> {
    candle_gen::lock_recover(&CROSS_KV_PROBE_LOCK)
}

#[cfg(test)]
pub(crate) fn reset_cross_kv_preparation_pairs() {
    CROSS_KV_PREPARATION_PAIRS.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn cross_kv_preparation_pairs() -> usize {
    CROSS_KV_PREPARATION_PAIRS.load(std::sync::atomic::Ordering::Relaxed)
}

impl Attention {
    /// Build this attention's projections + qk-norms from `src` — the dense (`WeightSrc::Dense`) and
    /// native-GGUF k-quant (`WeightSrc::Gguf`, sc-12735) paths share this ONE builder, so the resident-
    /// QTensor path reads the identical projection set the dense/packed path does. On the dense arm each
    /// `qlinear` is `QLinear::linear_detect` (packed-detecting, unchanged); on the GGUF arm it is a
    /// resident k-quant QTensor. The qk-norms are dense sidecars either way.
    fn build(cfg: &TransformerConfig, src: WeightSrc) -> Result<Self> {
        let inner = cfg.dim;
        Ok(Self {
            to_q: src.qlinear(cfg.dim, inner, "to_q", true)?,
            to_k: src.qlinear(cfg.dim, inner, "to_k", true)?,
            to_v: src.qlinear(cfg.dim, inner, "to_v", true)?,
            to_out: src.qlinear(inner, cfg.dim, "to_out.0", true)?,
            norm_q: src.get(inner, "norm_q.weight")?,
            norm_k: src.get(inner, "norm_k.weight")?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            eps: cfg.eps,
        })
    }

    #[cfg(test)]
    fn all_projections_packed(&self) -> bool {
        self.to_q.is_packed()
            && self.to_k.is_packed()
            && self.to_v.is_packed()
            && self.to_out.is_packed()
    }

    /// Visit this attention's four adaptable projections (`{prefix}.{to_q,to_k,to_v,to_out.0}`) for the
    /// additive-adapter walk (sc-10094).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.to_q"), &mut self.to_q)?;
        f(&format!("{prefix}.to_k"), &mut self.to_k)?;
        f(&format!("{prefix}.to_v"), &mut self.to_v)?;
        f(&format!("{prefix}.to_out.0"), &mut self.to_out)?;
        Ok(())
    }

    /// Project a step-invariant cross-attention source into K/V heads once per request conditioning
    /// payload.  The resulting tensors remain owned by the caller's request scope.
    fn prepare_kv(&self, context: &Tensor) -> Result<PreparedBlockCrossKv> {
        let (b, s_kv, _) = context.dims3()?;
        let k = rms(&self.to_k.forward(context)?, &self.norm_k, self.eps)?;
        let v = self.to_v.forward(context)?;
        let to_heads = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, s_kv, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        Ok(PreparedBlockCrossKv {
            key: to_heads(&k)?,
            value: to_heads(&v)?,
        })
    }

    /// `hidden`: `[B, S, dim]`; `kv`: preprojected K/V heads. RoPE is applied only when
    /// `cos`/`sin` are given (self-attention).
    fn forward_prepared(
        &self,
        hidden: &Tensor,
        kv: &PreparedBlockCrossKv,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, _) = hidden.dims3()?;
        let q = rms(&self.to_q.forward(hidden)?, &self.norm_q, self.eps)?;
        let to_heads = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, s, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let mut q = to_heads(&q)?; // [B,H,S,d]
        let mut k = kv.key.clone();
        if let Some((cos, sin)) = rope {
            q = apply_rope(&q, cos, sin)?;
            k = apply_rope(&k, cos, sin)?;
        }
        let scale = (self.head_dim as f64).powf(-0.5);
        let out = sdpa(&q, &k, &kv.value, scale)?; // [B,H,S,d]
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        self.to_out.forward(&out)
    }

    /// Compatibility path for self-attention and one-off callers. Request render paths use
    /// [`Self::prepare_kv`] outside the step loop instead.
    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        self.forward_prepared(hidden, &self.prepare_kv(context)?, rope)
    }
}

struct Ffn {
    proj: QLinear, // net.0.proj
    out: QLinear,  // net.2
}

impl Ffn {
    /// Build the FFN's two projections from `src` (dense or native-GGUF k-quant, sc-12735) — one builder
    /// for both paths.
    fn build(cfg: &TransformerConfig, src: WeightSrc) -> Result<Self> {
        Ok(Self {
            proj: src.qlinear(cfg.dim, cfg.ffn_dim, "net.0.proj", true)?,
            out: src.qlinear(cfg.ffn_dim, cfg.dim, "net.2", true)?,
        })
    }

    #[cfg(test)]
    fn all_projections_packed(&self) -> bool {
        self.proj.is_packed() && self.out.is_packed()
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.proj.forward(x)?.gelu()?)
    }

    /// Visit the FFN's two adaptable projections (`{prefix}.net.0.proj`, `{prefix}.net.2`) for the
    /// additive-adapter walk (sc-10094).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(&format!("{prefix}.net.0.proj"), &mut self.proj)?;
        f(&format!("{prefix}.net.2"), &mut self.out)?;
        Ok(())
    }
}

pub(crate) struct Block {
    scale_shift_table: Tensor, // [1,6,dim] f32
    attn1: Attention,
    norm2_w: Tensor, // affine cross-attn norm
    norm2_b: Tensor,
    attn2: Attention,
    ffn: Ffn,
    eps: f64,
}

impl Block {
    /// Dense (or MLX-packed) block loader — the unchanged path (used by the VACE model, `vace.rs`, and the
    /// inference DiT's snapshot/packed tiers). Delegates to [`Self::build`] with a dense [`WeightSrc`], so
    /// its behavior is byte-identical to before.
    pub(crate) fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, WeightSrc::dense(vb))
    }

    #[cfg(test)]
    pub(crate) fn all_projections_packed(&self) -> bool {
        self.attn1.all_projections_packed()
            && self.attn2.all_projections_packed()
            && self.ffn.all_projections_packed()
    }

    /// Build a block from `src` — the ONE builder the dense/MLX-packed path (`WeightSrc::Dense`) and the
    /// native-GGUF k-quant path (`WeightSrc::Gguf`, sc-12735) share, so the resident-QTensor DiT reads the
    /// identical block structure. The `scale_shift_table` / `norm2` / qk-norms are dense sidecars either
    /// way; the attention/FFN projections route through [`WeightSrc::qlinear`].
    fn build(cfg: &TransformerConfig, src: WeightSrc) -> Result<Self> {
        Ok(Self {
            scale_shift_table: src
                .get((1, 6, cfg.dim), "scale_shift_table")?
                .to_dtype(DType::F32)?,
            attn1: Attention::build(cfg, src.pp("attn1"))?,
            norm2_w: src.get(cfg.dim, "norm2.weight")?.to_dtype(DType::F32)?,
            norm2_b: src.get(cfg.dim, "norm2.bias")?.to_dtype(DType::F32)?,
            attn2: Attention::build(cfg, src.pp("attn2"))?,
            ffn: Ffn::build(cfg, src.pp("ffn"))?,
            eps: cfg.eps,
        })
    }

    pub(crate) fn prepare_cross_kv(&self, context: &Tensor) -> Result<PreparedBlockCrossKv> {
        #[cfg(test)]
        CROSS_KV_PREPARATION_PAIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.attn2.prepare_kv(context)
    }

    /// `hidden`: `[B,S,dim]` (bf16); `temb6`: `[B,6,dim]` (f32); `cross_kv`: request-scoped
    /// prepared text K/V for this block.
    pub(crate) fn forward_prepared(
        &self,
        hidden: &Tensor,
        temb6: &Tensor,
        cross_kv: &PreparedBlockCrossKv,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let dt = hidden.dtype();
        // mods: scale_shift_table[1,6,dim] + temb6[B,6,dim] → 6 × [B,1,dim] (f32).
        let table = if temb6.rank() == 4 {
            self.scale_shift_table.unsqueeze(1)?
        } else {
            self.scale_shift_table.clone()
        };
        let mods = table.broadcast_add(temb6)?;
        let modulation_axis = if mods.rank() == 4 { 2 } else { 1 };
        let m = |i: usize| -> Result<Tensor> {
            let value = mods.narrow(modulation_axis, i, 1)?;
            if modulation_axis == 2 {
                value.squeeze(2)
            } else {
                Ok(value)
            }
        };
        let (shift_msa, scale_msa, gate_msa) = (m(0)?, m(1)?, m(2)?);
        let (c_shift, c_scale, c_gate) = (m(3)?, m(4)?, m(5)?);

        let hf = hidden.to_dtype(DType::F32)?;
        // 1. self-attention
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(scale_msa + 1.0)?)?
            .broadcast_add(&shift_msa)?
            .to_dtype(dt)?;
        let a = self.attn1.forward(&n, &n, Some((cos, sin)))?;
        let hf = (hf + a.to_dtype(DType::F32)?.broadcast_mul(&gate_msa)?)?;

        // 2. cross-attention (affine norm2, ungated)
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&self.norm2_w)?
            .broadcast_add(&self.norm2_b)?
            .to_dtype(dt)?;
        let a = self.attn2.forward_prepared(&n, cross_kv, None)?;
        let hf = (hf + a.to_dtype(DType::F32)?)?;

        // 3. feed-forward
        let n = ln_no_affine(&hf, self.eps)?
            .broadcast_mul(&(c_scale + 1.0)?)?
            .broadcast_add(&c_shift)?
            .to_dtype(dt)?;
        let f = self.ffn.forward(&n)?;
        let hf = (hf + f.to_dtype(DType::F32)?.broadcast_mul(&c_gate)?)?;
        hf.to_dtype(dt)
    }

    /// Visit this block's adaptable projections (`{prefix}.attn1/attn2.*`, `{prefix}.ffn.*`) for the
    /// additive-adapter walk (sc-10094).
    fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        self.attn1
            .visit_adaptable_mut(&format!("{prefix}.attn1"), f)?;
        self.attn2
            .visit_adaptable_mut(&format!("{prefix}.attn2"), f)?;
        self.ffn.visit_adaptable_mut(&format!("{prefix}.ffn"), f)?;
        Ok(())
    }
}

/// Build the `[B, freq_dim]` sinusoidal timestep embedding (diffusers `Timesteps`,
/// `flip_sin_to_cos=True`, `downscale_freq_shift=0`): `[cos(t·ω) | sin(t·ω)]`.
pub(crate) fn timestep_sinusoid(t: f64, freq_dim: usize, b: usize, dev: &Device) -> Result<Tensor> {
    let half = freq_dim / 2;
    let mut row = vec![0f32; freq_dim];
    for i in 0..half {
        let freq = (-(10000f64.ln()) * i as f64 / half as f64).exp();
        let ang = t * freq;
        row[i] = ang.cos() as f32;
        row[half + i] = ang.sin() as f32;
    }
    let one = Tensor::from_vec(row, (1, freq_dim), dev)?;
    if b == 1 {
        Ok(one)
    } else {
        Ok(one.broadcast_as((b, freq_dim))?.contiguous()?)
    }
}

/// Vectorized timestep embedding for TI2V mask blending. `timesteps` is `[B,L]`; the result is
/// `[B,L,freq_dim]`, so pinned tokens can carry timestep zero independently of generated tokens.
fn timestep_sinusoid_tokens(timesteps: &Tensor, freq_dim: usize, dev: &Device) -> Result<Tensor> {
    let half = freq_dim / 2;
    let freqs = (0..half)
        .map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32)
        .collect::<Vec<_>>();
    let freqs = Tensor::from_vec(freqs, (1, 1, half), dev)?;
    let angles = timesteps
        .to_dtype(DType::F32)?
        .unsqueeze(2)?
        .broadcast_mul(&freqs)?;
    Tensor::cat(&[&angles.cos()?, &angles.sin()?], 2)
}

pub struct WanTransformer {
    patch_w: Tensor, // [dim,48,p_h,p_w]
    patch_b: Tensor, // [1,dim,1,1]
    text_l1: QLinear,
    text_l2: QLinear,
    time_l1: QLinear,
    time_l2: QLinear,
    time_proj: QLinear,
    blocks: Vec<Block>,
    norm_out_eps: f64,
    proj_out: QLinear,
    scale_shift_table: Tensor, // [1,2,dim] f32
    cfg: TransformerConfig,
    device: Device,
    dtype: DType,
    /// When set, the block stack drains the CUDA stream (`device.synchronize()`) after **each** DiT
    /// block during the denoise forward. Off (fully async) by default — the resident render never sets
    /// it. The **sequential-offload** render (sc-12733) sets it on each staged expert: that path frees
    /// the ~21 GB UMT5 encoder and the inactive expert into candle's in-process cudarc caching pool
    /// (which never returns pages to the driver) and then reuses that churned pool for the next expert's
    /// weights + the full-res denoise activations. Left fully async, the deep unsynced kernel pipeline
    /// racing against that pool's free→realloc **deterministically faults with a CUDA illegal-memory
    /// access at the A14B 720p geometry** (S ≈ 75 600 tokens) — a hard exit, no Rust panic — right at the
    /// high→low expert-swap boundary; the resident co-resident pool (nothing freed mid-render) never
    /// trips it, and small geometries stay under the collision size. Draining per block bounds the
    /// in-flight set so the reuse is ordered. Numerically inert: `synchronize()` changes only ordering,
    /// never a value (sc-12768).
    bounded_offload: bool,
}

impl WanTransformer {
    /// Build the DiT from a dense (or MLX-packed) [`VarBuilder`] — the unchanged path (the
    /// `Wan-AI/*-Diffusers` snapshot and the `SceneWorks/*-mlx` packed tiers). Delegates to the shared
    /// `build` with a dense `WeightSrc`, so its behavior is byte-identical to before.
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Self::build(cfg, WeightSrc::dense(vb))
    }

    /// Build the DiT from a **native-GGUF k-quant** source ([`GgufDit`], sc-12735) — the Linears are held
    /// as resident `Q4_K` [`QTensor`]s that dequantize per-matmul (ComfyUI-GGUF parity), the dense sidecars
    /// are the GGUF's F16/F32 blocks. Shares [`Self::build`] with [`Self::new`], so the resident-QTensor DiT
    /// reads the identical structure the dense/packed path does; the result reports [`Self::is_packed`].
    pub(crate) fn from_gguf(cfg: &TransformerConfig, dit: &GgufDit) -> Result<Self> {
        Self::build(cfg, WeightSrc::gguf(dit))
    }

    /// The ONE DiT builder shared by [`Self::new`] (dense / MLX-packed) and [`Self::from_gguf`] (native
    /// GGUF k-quant, sc-12735). Every projection routes through [`WeightSrc::qlinear`] and every dense
    /// sidecar through [`WeightSrc::get`], so the two load paths can never drift in shape/key handling —
    /// the dense arm forwards to the exact `VarBuilder`/`QLinear::linear_detect` calls as before.
    fn build(cfg: &TransformerConfig, src: WeightSrc) -> Result<Self> {
        let (pt, ph, pw) = cfg.patch;
        // patch_embedding is a Conv3d (1,2,2); temporal kernel 1 → squeeze to a per-frame conv2d.
        let pw_full = src.get(
            (cfg.dim, cfg.in_channels, pt, ph, pw),
            "patch_embedding.weight",
        )?;
        let patch_w = pw_full.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [dim,48,ph,pw]
        let patch_b = src
            .get(cfg.dim, "patch_embedding.bias")?
            .reshape((1, cfg.dim, 1, 1))?;

        let ce = src.pp("condition_embedder");
        let text_l1 = ce.qlinear(cfg.text_dim, cfg.dim, "text_embedder.linear_1", true)?;
        let text_l2 = ce.qlinear(cfg.dim, cfg.dim, "text_embedder.linear_2", true)?;
        let time_l1 = ce.qlinear(cfg.freq_dim, cfg.dim, "time_embedder.linear_1", true)?;
        let time_l2 = ce.qlinear(cfg.dim, cfg.dim, "time_embedder.linear_2", true)?;
        let time_proj = ce.qlinear(cfg.dim, 6 * cfg.dim, "time_proj", true)?;

        let blocks_src = src.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::build(cfg, blocks_src.pp(i))?);
        }

        let proj_out = src.qlinear(cfg.dim, cfg.out_channels * pt * ph * pw, "proj_out", true)?;
        let scale_shift_table = src
            .get((1, 2, cfg.dim), "scale_shift_table")?
            .to_dtype(DType::F32)?;

        Ok(Self {
            patch_w,
            patch_b,
            text_l1,
            text_l2,
            time_l1,
            time_l2,
            time_proj,
            blocks,
            norm_out_eps: cfg.eps,
            proj_out,
            scale_shift_table,
            cfg: *cfg,
            device: src.device(),
            dtype: src.dtype(),
            bounded_offload: false,
        })
    }

    /// Enable per-block CUDA-stream draining for the **sequential-offload** denoise (sc-12768). The
    /// resident render leaves this off (fully async); the staged A14B render sets it on each expert
    /// right after loading it, before the denoise forward — draining the stream after each DiT block so
    /// the deep async pipeline cannot race candle's churned cudarc caching pool (the TE + inactive-expert
    /// frees that path reuses), which otherwise faults with a CUDA illegal-memory access at the 720p
    /// A14B geometry. Ordering-only, so it never changes a denoised value.
    pub fn set_bounded_offload(&mut self, on: bool) {
        self.bounded_offload = on;
    }

    /// Project UMT5 prompt embeds `[B,S,4096]` → cross-attn context `[B,S,dim]` (constant across the
    /// denoise loop). `gelu_tanh` between the two linears (PixArtAlphaTextProjection).
    pub fn embed_text(&self, prompt_embeds: &Tensor) -> Result<Tensor> {
        let x = prompt_embeds.to_dtype(self.dtype)?;
        self.text_l2.forward(&self.text_l1.forward(&x)?.gelu()?)
    }

    /// Prepare all step-invariant text cross-attention K/V heads for one projected conditioning
    /// payload. The returned cache is intentionally request-scoped: callers create it after the DiT
    /// loads and retain it only for the matching denoise branch.
    pub(crate) fn prepare_cross_kv(&self, context: &Tensor) -> Result<PreparedWanCrossKv> {
        Ok(PreparedWanCrossKv {
            blocks: self
                .blocks
                .iter()
                .map(|block| block.prepare_cross_kv(context))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    /// One DiT forward: `latents [B,in_c,F,Hl,Wl]`, projected `context [B,S,dim]`, scalar `t`,
    /// RoPE `cos`/`sin [L,64]` → predicted velocity `[B,out_c,F,Hl,Wl]`.
    ///
    /// Composed from the three seams below (patch-embed → block-stack/head → unpatchify), byte-identical
    /// to the previous monolithic body. The seams are exposed additively for the Bernini renderer's
    /// token-axis packed conditioning (sc-11004), which patch-embeds the target + each source separately
    /// and runs one packed [`forward_packed`](Self::forward_packed) over the concatenated token axis.
    pub fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        t: f64,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (tokens, grid) = self.patch_embed_tokens(latents)?;
        let cross_kv = self.prepare_cross_kv(context)?;
        let out = self.forward_packed_prepared(&tokens, t, &cross_kv, cos, sin)?;
        self.unpatchify_tokens(&out, grid)
    }

    /// Denoise one scalar-timestep latent against request-scoped prepared text K/V.
    pub(crate) fn forward_prepared(
        &self,
        latents: &Tensor,
        t: f64,
        cross_kv: &PreparedWanCrossKv,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (tokens, grid) = self.patch_embed_tokens(latents)?;
        let out = self.forward_packed_prepared(&tokens, t, cross_kv, cos, sin)?;
        self.unpatchify_tokens(&out, grid)
    }

    /// Patch-embed `latents [B, in_channels, F, Hl, Wl]` into the DiT token stream `[B, L, dim]` (bf16)
    /// plus the patch grid `(ppf, pph, ppw)` — the embedding half of [`forward`](Self::forward), exposed
    /// as a seam for the Bernini renderer (sc-11004), which patch-embeds the noisy target **and** each
    /// conditioning source separately (each with its own source-id RoPE) and concatenates them on the
    /// token axis before a single packed forward. `L = ppf·pph·ppw`.
    pub fn patch_embed_tokens(&self, latents: &Tensor) -> Result<(Tensor, (usize, usize, usize))> {
        let (b, _c, f, hl, wl) = latents.dims5()?;
        let (pt, ph, pw) = self.cfg.patch;
        let (ppf, pph, ppw) = (f / pt, hl / ph, wl / pw);

        // Patch embed: per-frame strided conv2d, then flatten to tokens (f outer, then h, w).
        let merged = latents
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * f, self.cfg.in_channels, hl, wl))?
            .contiguous()?
            .to_dtype(self.dtype)?;
        let y = merged.conv2d(&self.patch_w, 0, ph, 1, 1)?; // [B*F,dim,pph,ppw]
        let y = y.broadcast_add(&self.patch_b)?;
        let hidden = y
            .reshape((b, f, self.cfg.dim, pph, ppw))?
            .permute((0, 1, 3, 4, 2))? // [B,F,pph,ppw,dim]
            .reshape((b, ppf * pph * ppw, self.cfg.dim))?
            .contiguous()?;
        Ok((hidden, (ppf, pph, ppw)))
    }

    /// Run the block stack + output head over a **pre-embedded, pre-packed** token sequence
    /// `tokens [B, L, dim]` (bf16) with caller-supplied RoPE `cos`/`sin [L, head_dim/2]` and the
    /// projected cross-attention `context [B, S, dim]` — returning the per-token velocity
    /// `[B, L, out_channels·∏patch]` (this DiT's dtype) **without** unpatchifying. This is
    /// [`forward`](Self::forward)'s body minus the patch-embed in / unpatchify out, the seam the Bernini
    /// renderer (sc-11004) uses: at batch 1 the packed `[sources…, target]` sequence is plain full
    /// self-attention (the reference's varlen attention with a single `cu_seqlens` segment), so the
    /// caller assembles the token + RoPE concat, calls this once, then slices the target tokens and
    /// [`unpatchify_tokens`](Self::unpatchify_tokens) them.
    pub fn forward_packed(
        &self,
        tokens: &Tensor,
        t: f64,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let cross_kv = self.prepare_cross_kv(context)?;
        self.forward_packed_prepared(tokens, t, &cross_kv, cos, sin)
    }

    /// Run the block stack + head with preprojected request-scoped text K/V.
    pub(crate) fn forward_packed_prepared(
        &self,
        tokens: &Tensor,
        t: f64,
        cross_kv: &PreparedWanCrossKv,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, _l, _dim) = tokens.dims3()?;
        if cross_kv.blocks.len() != self.blocks.len() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "wan prepared cross K/V has {} blocks; transformer has {}",
                cross_kv.blocks.len(),
                self.blocks.len()
            )));
        }
        // Time embedding → temb [B,dim], and the per-block 6-vector temb6 [B,6,dim] (f32).
        let sinus =
            timestep_sinusoid(t, self.cfg.freq_dim, b, &self.device)?.to_dtype(self.dtype)?;
        let temb = self
            .time_l2
            .forward(&self.time_l1.forward(&sinus)?.silu()?)?; // [B,dim]
        let temb6 = self
            .time_proj
            .forward(&temb.silu()?)?
            .reshape((b, 6, self.cfg.dim))?
            .to_dtype(DType::F32)?;

        let mut hidden = tokens.clone();
        for (blk, kv) in self.blocks.iter().zip(&cross_kv.blocks) {
            hidden = blk.forward_prepared(&hidden, &temb6, kv, cos, sin)?;
            // sc-12768: on the sequential-offload path, drain the stream after each block so the deep
            // async denoise pipeline cannot race candle's churned cudarc caching pool (the freed TE /
            // inactive-expert pages the next expert's weights + full-res activations reuse) — the
            // illegal-memory access at the A14B 720p geometry. No-op (untouched, fully async) on the
            // resident path. Ordering-only; the denoised value is unchanged.
            if self.bounded_offload {
                self.device.synchronize()?;
            }
        }

        // Head: norm_out (non-affine) modulated by scale_shift_table + temb.
        let head_mod = self
            .scale_shift_table
            .broadcast_add(&temb.unsqueeze(1)?.to_dtype(DType::F32)?)?;
        let shift = head_mod.narrow(1, 0, 1)?;
        let scale = head_mod.narrow(1, 1, 1)?;
        let hf = hidden.to_dtype(DType::F32)?;
        let normed = ln_no_affine(&hf, self.norm_out_eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?
            .to_dtype(self.dtype)?;
        self.proj_out.forward(&normed) // [B,L,out_c*patch]
    }

    /// TI2V mask-blend forward with per-token timesteps `[B,L]`. The scalar T2V entry point above is
    /// intentionally unchanged; this only generalizes AdaLN/head modulation to the token axis.
    pub fn forward_tokens(
        &self,
        latents: &Tensor,
        timestep_tokens: &Tensor,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let cross_kv = self.prepare_cross_kv(context)?;
        self.forward_tokens_prepared(latents, timestep_tokens, &cross_kv, cos, sin)
    }

    /// TI2V mask-blend forward with request-scoped prepared text K/V.
    pub(crate) fn forward_tokens_prepared(
        &self,
        latents: &Tensor,
        timestep_tokens: &Tensor,
        cross_kv: &PreparedWanCrossKv,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (tokens, grid) = self.patch_embed_tokens(latents)?;
        let (b, l, _dim) = tokens.dims3()?;
        if cross_kv.blocks.len() != self.blocks.len() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "wan prepared cross K/V has {} blocks; transformer has {}",
                cross_kv.blocks.len(),
                self.blocks.len()
            )));
        }
        let (tb, tl) = timestep_tokens.dims2()?;
        if (tb, tl) != (b, l) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "wan TI2V timestep tokens must be [{b},{l}] (got [{tb},{tl}])"
            )));
        }
        let sinus = timestep_sinusoid_tokens(timestep_tokens, self.cfg.freq_dim, &self.device)?
            .to_dtype(self.dtype)?;
        let temb = self
            .time_l2
            .forward(&self.time_l1.forward(&sinus)?.silu()?)?; // [B,L,dim]
        let temb6 = self
            .time_proj
            .forward(&temb.silu()?)?
            .reshape((b, l, 6, self.cfg.dim))?
            .to_dtype(DType::F32)?;

        let mut hidden = tokens;
        for (block, kv) in self.blocks.iter().zip(&cross_kv.blocks) {
            hidden = block.forward_prepared(&hidden, &temb6, kv, cos, sin)?;
            if self.bounded_offload {
                self.device.synchronize()?;
            }
        }

        let head_mod = self
            .scale_shift_table
            .unsqueeze(1)?
            .broadcast_add(&temb.unsqueeze(2)?.to_dtype(DType::F32)?)?; // [B,L,2,dim]
        let shift = head_mod.narrow(2, 0, 1)?.squeeze(2)?;
        let scale = head_mod.narrow(2, 1, 1)?.squeeze(2)?;
        let hidden = hidden.to_dtype(DType::F32)?;
        let normed = ln_no_affine(&hidden, self.norm_out_eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?
            .to_dtype(self.dtype)?;
        let out = self.proj_out.forward(&normed)?;
        self.unpatchify_tokens(&out, grid)
    }

    /// Unpatchify a per-token velocity `[B, L, out_channels·∏patch]` (with `L = ppf·pph·ppw`) back to a
    /// spatial latent `[B, out_channels, F, Hl, Wl]` (f32) — the tail of [`forward`](Self::forward),
    /// exposed so the Bernini renderer can unpatchify the **target-sliced** packed output (sc-11004).
    pub fn unpatchify_tokens(&self, out: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (ppf, pph, ppw) = grid;
        let (b, _l, _op) = out.dims3()?;
        let (pt, ph, pw) = self.cfg.patch;
        let oc = self.cfg.out_channels;
        out.reshape(&[b, ppf, pph, ppw, pt, ph, pw, oc][..])?
            .permute(&[0usize, 7, 1, 4, 2, 5, 3, 6][..])?
            .reshape((b, oc, ppf * pt, pph * ph, ppw * pw))?
            .to_dtype(DType::F32)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether this DiT loaded from a **packed** MLX tier (its projections are quantized) — the additive
    /// router uses this to select packed-safe additive residuals (sc-10094). Probed on `proj_out` (every
    /// projection in a tier packs together; a dense checkpoint packs none).
    pub fn is_packed(&self) -> bool {
        self.proj_out.is_packed()
    }

    /// The canonical dotted paths of every adaptable projection (attention q/k/v/out, FFN, the
    /// condition-embedder projections, `time_proj`, `proj_out`) — the LoRA merge surface, in the diffusers
    /// key namespace. Drives the additive-adapter kohya `flat→dotted` table (sc-10094).
    pub fn adaptable_paths(&self) -> Vec<String> {
        let mut paths = vec![
            "condition_embedder.text_embedder.linear_1".to_string(),
            "condition_embedder.text_embedder.linear_2".to_string(),
            "condition_embedder.time_embedder.linear_1".to_string(),
            "condition_embedder.time_embedder.linear_2".to_string(),
            "condition_embedder.time_proj".to_string(),
        ];
        for i in 0..self.blocks.len() {
            for attn in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                    paths.push(format!("blocks.{i}.{attn}.{leaf}"));
                }
            }
            paths.push(format!("blocks.{i}.ffn.net.0.proj"));
            paths.push(format!("blocks.{i}.ffn.net.2"));
        }
        paths.push("proj_out".to_string());
        paths
    }

    /// Walk every adaptable projection, invoking `f(path, &mut QLinear)` once each with the projection's
    /// canonical dotted path — the host visitor the additive-adapter installer routes residuals through
    /// (sc-10094; the candle analog of mlx-gen's `AdaptableHost`). The order matches
    /// [`Self::adaptable_paths`].
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> Result<()>,
    ) -> Result<()> {
        f(
            "condition_embedder.text_embedder.linear_1",
            &mut self.text_l1,
        )?;
        f(
            "condition_embedder.text_embedder.linear_2",
            &mut self.text_l2,
        )?;
        f(
            "condition_embedder.time_embedder.linear_1",
            &mut self.time_l1,
        )?;
        f(
            "condition_embedder.time_embedder.linear_2",
            &mut self.time_l2,
        )?;
        f("condition_embedder.time_proj", &mut self.time_proj)?;
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("blocks.{i}"), f)?;
        }
        f("proj_out", &mut self.proj_out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::{apply_source_id, WanRope};
    use std::collections::HashMap;

    /// A tiny dense config the CPU synthetic weights below fill (dim 16 = 2 heads × head_dim 8, z16
    /// in/out, patch (1,2,2)). Keeps the packed-forward geometry (`ppf·pph·ppw` tokens, 3-axis RoPE) but
    /// small enough to run on CPU without weights.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            dim: 16,
            ffn_dim: 32,
            freq_dim: 16,
            text_dim: 16,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 64,
        }
    }

    /// Build a synthetic `WanTransformer` (all dense) from randn weights — every tensor key
    /// [`WanTransformer::new`] reads, at [`DType::F32`] so the whole forward runs on CPU.
    fn tiny_dit(cfg: &TransformerConfig, dev: &Device) -> WanTransformer {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut put = |k: &str, shape: &[usize]| {
            m.insert(
                k.to_string(),
                Tensor::randn(0f32, 0.2f32, shape, dev).unwrap(),
            );
        };
        let (pt, ph, pw) = cfg.patch;
        let d = cfg.dim;
        put("patch_embedding.weight", &[d, cfg.in_channels, pt, ph, pw]);
        put("patch_embedding.bias", &[d]);
        put(
            "condition_embedder.text_embedder.linear_1.weight",
            &[d, cfg.text_dim],
        );
        put("condition_embedder.text_embedder.linear_1.bias", &[d]);
        put("condition_embedder.text_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.text_embedder.linear_2.bias", &[d]);
        put(
            "condition_embedder.time_embedder.linear_1.weight",
            &[d, cfg.freq_dim],
        );
        put("condition_embedder.time_embedder.linear_1.bias", &[d]);
        put("condition_embedder.time_embedder.linear_2.weight", &[d, d]);
        put("condition_embedder.time_embedder.linear_2.bias", &[d]);
        put("condition_embedder.time_proj.weight", &[6 * d, d]);
        put("condition_embedder.time_proj.bias", &[6 * d]);
        for i in 0..cfg.num_layers {
            let b = format!("blocks.{i}");
            put(&format!("{b}.scale_shift_table"), &[1, 6, d]);
            for attn in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                    put(&format!("{b}.{attn}.{leaf}.weight"), &[d, d]);
                    put(&format!("{b}.{attn}.{leaf}.bias"), &[d]);
                }
                put(&format!("{b}.{attn}.norm_q.weight"), &[d]);
                put(&format!("{b}.{attn}.norm_k.weight"), &[d]);
            }
            put(&format!("{b}.norm2.weight"), &[d]);
            put(&format!("{b}.norm2.bias"), &[d]);
            put(&format!("{b}.ffn.net.0.proj.weight"), &[cfg.ffn_dim, d]);
            put(&format!("{b}.ffn.net.0.proj.bias"), &[cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.weight"), &[d, cfg.ffn_dim]);
            put(&format!("{b}.ffn.net.2.bias"), &[d]);
        }
        put("proj_out.weight", &[cfg.out_channels * pt * ph * pw, d]);
        put("proj_out.bias", &[cfg.out_channels * pt * ph * pw]);
        put("scale_shift_table", &[1, 2, d]);
        let vb = VarBuilder::from_tensors(m, DType::F32, dev);
        WanTransformer::new(cfg, vb).unwrap()
    }

    /// Max-abs bound for a **chunked-vs-un-chunked SDPA** comparison (SC-15943). Query-row chunking is
    /// mathematically equivalent, never bitwise equal: it changes the GEMM `M` dimension, and candle's
    /// CPU `gemm` and cuBLAS may pick a different tiling and accumulation order per `M`, so the f32
    /// rounding differs. This is the metric + limit the parity-evidence rule asks for, not a "looks
    /// similar" fudge.
    ///
    /// **Measured, on the host that exposed the defect** (macOS 26.x / Darwin 25.5.0, Apple silicon,
    /// toolchain 1.96.0) — 1,000,000 random draws at this test's `[1,2,7,4]` shape, both budgets, i.e.
    /// 2e6 tensor comparisons / ~3.8e7 differing elements:
    ///
    /// | quantity                                    | measured                  |
    /// |---------------------------------------------|---------------------------|
    /// | draws where `budget 42` differs from single  | 99.7 % (never 0.0)        |
    /// | draws where `budget 1` differs from single   | 100 %                     |
    /// | worst max\|Δ\| over 1e6 draws                | **1.31e-6**               |
    /// | differing elements above `1e-6`             | 1 of 37,978,005           |
    /// | differing elements above `2e-6`             | 0                         |
    /// | `probs·v` alone, bit-identical inputs        | diverges (up to 2.4e-7)   |
    ///
    /// The last row is the mechanism isolated: feeding *bit-identical* `probs` and `v` and varying only
    /// `M` already diverges, so this is GEMM shape — not the softmax and not the f32 upcast.
    ///
    /// Deliberately stated as an **absolute** bound rather than in ULP. The deltas land on
    /// near-cancelling output elements, so a per-element ULP ratio is unbounded (a measured
    /// 1.4e7 "ULP" at an element near zero) and describes nothing useful; against the tensor's own
    /// scale (|out| ≲ 4.5 for unit-normal inputs) the same worst case is only ~3 ULP. Two references,
    /// six orders apart, which is exactly why the assertion tests the absolute delta.
    ///
    /// `1e-5` sits ~8× above the measured worst case, and ~5 orders below what any real chunking
    /// regression produces — a mis-narrowed offset, a mis-ordered `cat`, or a dropped softmax all move
    /// whole rows, i.e. O(1); the two mutations run for SC-15943 gave 1.6e0 and 9.6e-1. It is also the
    /// bound the shared kernel's own equivalence helper uses for this comparison
    /// (`candle_gen::attention`'s `approx_eq`), so the two crates agree.
    ///
    /// Do **not** tighten this. `1e-6` — the bound SC-15943's own analysis first proposed, from a
    /// 20k-draw sample whose worst case was 7.2e-7 — is *below* the measured 1e6-draw tail of 1.31e-6:
    /// it does not merely flake, it fails. Do **not** "fix" this by seeding the RNG either; that hides
    /// the tail behind one lucky draw and leaves a false invariant standing for the next `M`, host, or
    /// BLAS version.
    ///
    /// **This host is the outlier, and no CI lane checks it.** On x86-64 Linux the same comparison is
    /// exactly `0.0` — `Candle CPU packages (Linux)` is green on main while main still asserts
    /// `== 0.0`. That lane (`ubuntu-latest`) is the only one that runs these lib tests; the macOS lane
    /// reaches `candle-gen*` through Clippy alone, so on arm64 this test is compiled and never
    /// executed. The invariant was never evaluated on the architecture that breaks it.
    const CHUNK_PARITY_MAX_ABS: f32 = 1e-5;

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// The refactored [`WanTransformer::forward`] is byte-identical to the explicit
    /// `patch_embed_tokens → forward_packed → unpatchify_tokens` composition — pins the additive seams
    /// to the validated monolithic forward (the many-crates-depend-on-it invariant).
    #[test]
    fn forward_equals_seam_composition() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let latents = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &dev).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(2, 2, 2, &dev).unwrap(); // L = 8
        let t = 833.0;
        let want = dit.forward(&latents, &context, t, &cos, &sin).unwrap();

        let (tokens, grid) = dit.patch_embed_tokens(&latents).unwrap();
        assert_eq!(grid, (2, 2, 2));
        assert_eq!(tokens.dims(), &[1, 8, cfg.dim]);
        let out = dit
            .forward_packed(&tokens, t, &context, &cos, &sin)
            .unwrap();
        assert_eq!(out.dims(), &[1, 8, cfg.out_channels * 4]);
        let got = dit.unpatchify_tokens(&out, grid).unwrap();
        assert_eq!(
            max_abs(&got, &want),
            0.0,
            "seam composition must equal forward"
        );
    }

    /// The Candle work-count probe for SC-21692. Each projected text payload gets one K/V pair per
    /// block; scalar and TI2V-token denoise forwards reuse that request-scoped cache without another
    /// text K/V projection. The prepared result remains exactly pinned to the prior small CPU fixture.
    #[test]
    fn prepared_cross_kv_runs_once_per_payload_and_preserves_small_fixture_output() {
        let _probe_lock = lock_cross_kv_probe();
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let latents = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let pos = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &dev).unwrap();
        let neg = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &dev).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(2, 2, 2, &dev).unwrap();
        let timestep = 833.0;

        let prior = dit.forward(&latents, &pos, timestep, &cos, &sin).unwrap();
        reset_cross_kv_preparation_pairs();
        let pos_kv = dit.prepare_cross_kv(&pos).unwrap();
        let neg_kv = dit.prepare_cross_kv(&neg).unwrap();
        assert_eq!(
            cross_kv_preparation_pairs(),
            cfg.num_layers * 2,
            "one K/V pair per block and conditioning payload"
        );

        let prepared = dit
            .forward_prepared(&latents, timestep, &pos_kv, &cos, &sin)
            .unwrap();
        assert_eq!(
            max_abs(&prior, &prepared),
            0.0,
            "prepared scalar forward must preserve the prior fixture"
        );
        let tokens = Tensor::full(timestep as f32, (1, 8), &dev).unwrap();
        let tokenized = dit
            .forward_tokens_prepared(&latents, &tokens, &pos_kv, &cos, &sin)
            .unwrap();
        assert!(
            max_abs(&prepared, &tokenized) < 1e-5,
            "prepared TI2V-token forward must retain scalar parity"
        );
        // Repeat both CFG branches across multiple denoise steps. Only Q changes; text K/V is untouched.
        for step in [700.0, 500.0, 250.0] {
            dit.forward_prepared(&latents, step, &pos_kv, &cos, &sin)
                .unwrap();
            dit.forward_prepared(&latents, step, &neg_kv, &cos, &sin)
                .unwrap();
        }
        assert_eq!(
            cross_kv_preparation_pairs(),
            cfg.num_layers * 2,
            "denoise repetition must not reproject text K/V"
        );
    }

    #[test]
    fn per_token_timestep_forward_reduces_to_scalar_and_honors_pinned_tokens() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let latents = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &dev).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(2, 2, 2, &dev).unwrap();
        let timestep = 833.0;
        let scalar = dit
            .forward(&latents, &context, timestep, &cos, &sin)
            .unwrap();
        let all_equal = Tensor::full(timestep as f32, (1, 8), &dev).unwrap();
        let tokenized = dit
            .forward_tokens(&latents, &all_equal, &context, &cos, &sin)
            .unwrap();
        assert!(
            max_abs(&scalar, &tokenized) < 1e-5,
            "all-equal token timesteps must reduce to the scalar forward"
        );

        let mixed = Tensor::from_vec(
            vec![0f32, 0.0, 0.0, 0.0, 833.0, 833.0, 833.0, 833.0],
            (1, 8),
            &dev,
        )
        .unwrap();
        let pinned = dit
            .forward_tokens(&latents, &mixed, &context, &cos, &sin)
            .unwrap();
        assert!(
            max_abs(&tokenized, &pinned) > 1e-5,
            "zero-timestep pinned tokens must change the modulation path"
        );
    }

    /// A conditioning source concatenated on the token axis extends the packed sequence, but the sliced
    /// target velocity keeps the target's shape — and the source actually couples into the target through
    /// self-attention (the packed target-slice differs from the target-only forward), with the source-id
    /// RoPE shifting the result. Mirrors the mlx `conditioning_source_preserves_target_shape` intent.
    #[test]
    fn packed_source_preserves_target_shape_and_couples() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let hd = cfg.head_dim;
        let rope = WanRope::new(&cfg);
        let t = 700.0;

        let target = Tensor::randn(0f32, 1f32, (1, 16, 2, 4, 4), &dev).unwrap();
        let source = Tensor::randn(0f32, 1f32, (1, 16, 1, 4, 4), &dev).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 3, cfg.dim), &dev).unwrap();

        let (tok_t, grid_t) = dit.patch_embed_tokens(&target).unwrap();
        let (cos_t, sin_t) = rope.cos_sin(grid_t.0, grid_t.1, grid_t.2, &dev).unwrap();
        let (tok_s, grid_s) = dit.patch_embed_tokens(&source).unwrap();
        let (cos_s0, sin_s0) = rope.cos_sin(grid_s.0, grid_s.1, grid_s.2, &dev).unwrap();
        // source id 1 shifts the source segment's RoPE; the target stays id 0.
        let (cos_s, sin_s) = apply_source_id(&cos_s0, &sin_s0, 1.0, hd).unwrap();

        let l_t = grid_t.0 * grid_t.1 * grid_t.2;
        let tokens = Tensor::cat(&[&tok_s, &tok_t], 1).unwrap();
        let cos = Tensor::cat(&[&cos_s, &cos_t], 0).unwrap();
        let sin = Tensor::cat(&[&sin_s, &sin_t], 0).unwrap();
        let out = dit
            .forward_packed(&tokens, t, &context, &cos, &sin)
            .unwrap();
        let total = out.dim(1).unwrap();
        let target_tokens = out.narrow(1, total - l_t, l_t).unwrap();
        let vel = dit.unpatchify_tokens(&target_tokens, grid_t).unwrap();
        assert_eq!(
            vel.dims(),
            target.dims(),
            "target velocity keeps target shape"
        );

        // Coupling: the packed target-slice differs from the target-only forward (the source tokens
        // entered the target through self-attention).
        let solo = dit.forward(&target, &context, t, &cos_t, &sin_t).unwrap();
        assert!(
            max_abs(&vel, &solo) > 1e-5,
            "a conditioning source must couple into the target velocity"
        );

        // The source-id RoPE matters: id 0 on the source segment yields a different target velocity.
        let cos0 = Tensor::cat(&[&cos_s0, &cos_t], 0).unwrap();
        let sin0 = Tensor::cat(&[&sin_s0, &sin_t], 0).unwrap();
        let out0 = dit
            .forward_packed(&tokens, t, &context, &cos0, &sin0)
            .unwrap();
        let vel0 = dit
            .unpatchify_tokens(&out0.narrow(1, total - l_t, l_t).unwrap(), grid_t)
            .unwrap();
        assert!(
            max_abs(&vel, &vel0) > 1e-6,
            "source-id RoPE (id 1 vs id 0) must change the coupled velocity"
        );
    }

    /// The ported sc-6217 query-row chunking (sc-12434): forcing a tiny scores budget must split the
    /// query rows yet reproduce the single un-chunked pass to within [`CHUNK_PARITY_MAX_ABS`], since
    /// each query row's softmax is over all keys and independent of the other rows. This is the
    /// guarantee that stops the A14B self-attention from materializing the whole `[B,H,S,S]` block.
    /// Counting softmax invocations **through the production `sdpa_budgeted`** proves the render's own
    /// path chunks (one call per query block), so a regression back to a single materialized pass fails
    /// here, not just a silently-slower one.
    ///
    /// **Per-row independence is a statement about the math, not about the bits** (SC-15943). This
    /// asserted exact `0.0` until SC-15943: narrowing the query axis changes the GEMM `M` dimension
    /// (7 → 3/3/1 here), and candle's CPU `gemm` and cuBLAS are both free to select a different tiling
    /// and accumulation order at a different `M`. A different summation order over f32 perturbs the low
    /// bits, so bit-identity is not available on this path and never was — see [`CHUNK_PARITY_MAX_ABS`]
    /// for the measured distribution and bound, and `candle_gen::sdpa_budgeted_bhsd`'s own contract,
    /// which says the same thing.
    #[test]
    fn sdpa_chunks_query_rows_and_matches_single_pass() {
        use std::cell::Cell;
        let dev = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, h, s, d), &dev).unwrap();
        let scale = (d as f64).powf(-0.5);
        let dtype = q.dtype();

        // Production default budget (ATTN_SCORES_BUDGET ≫ this size) is a single un-chunked pass. Pin
        // that rather than assume it: `single` is the reference both tolerance checks below compare
        // against, and the counting closure is wired only into the `sdpa_budgeted` calls, so nothing
        // else would notice if `single` itself started chunking. `sdpa_bhsd_impl` takes the un-chunked
        // branch iff `budget / (b·h·sk) >= sq`, i.e. `budget >= b·h·sk·sq` — so a `WAN_ATTN_SCORES_BUDGET`
        // override small enough to chunk the reference fails here loudly instead of being absorbed by
        // the tolerance (SC-15943).
        assert!(
            wan_self_attn_budget() >= b * h * s * s,
            "the reference pass must be un-chunked: budget {} < b·h·sk·sq {}",
            wan_self_attn_budget(),
            b * h * s * s
        );
        let single = sdpa(&q, &k, &v, scale).unwrap();

        // Drive the PRODUCTION `sdpa_budgeted` with a call-counting wrapper of the exact f32-upcast
        // softmax and tiny budgets. budget 42 → block = 42/(b·h·sk) = 42/14 = 3 → blocks 3,3,1 over
        // S=7 (3 calls); budget 1 → 7 single-row blocks (7 more calls). A regression that stopped
        // chunking would report 1 and fail. Each chunked result matches the single pass to
        // `CHUNK_PARITY_MAX_ABS` — a rounding-order difference, not a bit-identity (SC-15943).
        let calls = Cell::new(0usize);
        let counting = |scores: &Tensor| {
            calls.set(calls.get() + 1);
            softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(dtype)
        };
        let chunked = sdpa_budgeted(&q, &k, &v, scale, 42, counting).unwrap();
        assert_eq!(
            calls.get(),
            3,
            "budget 42 must split S=7 into 3 query-row blocks (3,3,1)"
        );
        let d_chunked = max_abs(&single, &chunked);
        assert!(
            d_chunked < CHUNK_PARITY_MAX_ABS,
            "chunked attention diverged from the single pass: max|Δ| {d_chunked:e} ≥ {CHUNK_PARITY_MAX_ABS:e}"
        );
        let block1 = sdpa_budgeted(&q, &k, &v, scale, 1, counting).unwrap();
        assert_eq!(calls.get(), 10, "budget 1 adds 7 single-row blocks (3 + 7)");
        let d_block1 = max_abs(&single, &block1);
        assert!(
            d_block1 < CHUNK_PARITY_MAX_ABS,
            "single-row chunks diverged from the single pass: max|Δ| {d_block1:e} ≥ {CHUNK_PARITY_MAX_ABS:e}"
        );

        // The production budget genuinely engages at the story's 832x480 A14B proof geometry
        // (h = 40, S ≈ 32,760, b = 1 under Lightning CFG-off): one query row's score contribution
        // already exceeds the budget, so `sdpa`'s Wan budget forces chunking there, and each resulting
        // block stays under candle's CUDA i32 element ceiling. sc-12894 tightened the budget below the
        // shared 1e9 ceiling to cap the denoise transient — so chunking engages harder, never less (the
        // budget-vs-ceiling invariants are the compile-time `const _` assertions by the const itself).
        let rows_per_query = 40 * 32_760usize;
        assert!(
            rows_per_query * 32_760 > WAN_SELF_ATTN_SCORES_BUDGET,
            "A14B 480p self-attention must be over budget (chunking engages)"
        );
        let block = WAN_SELF_ATTN_SCORES_BUDGET / rows_per_query;
        assert!(
            rows_per_query * block <= i32::MAX as usize,
            "each chunk's score block must stay under the CUDA i32 element ceiling"
        );
    }

    /// sc-12894 parity gate for the **bf16-softmax** lever (the `WAN_ATTN_SOFTMAX_BF16` knob). The
    /// chunk lever is a rounding-order effect bounded by [`CHUNK_PARITY_MAX_ABS`] (`1e-5`); the bf16
    /// softmax is a far coarser trade — it gives up the f32 upcast for half the per-chunk transient, so
    /// its bound below is `0.05`, nearly four orders wider. This bounds that trade against the actual
    /// CUDA kernels (bf16 matmul and softmax are CUDA-only — CPU has no bf16 matmul), so a regression
    /// that widened the gap (say a bf16 sum accumulator) is caught here and the delta the GPU render's
    /// PSNR check corroborates is quantified. The two paths must stay tightly correlated: the CUDA bf16 softmax still max-stabilizes
    /// and sums in f32, so only the `exp`/probs carry ~2^-8 bf16 rounding and the attention output — a
    /// convex combination of unit-scale values — tracks the f32 path to well within a distilled sampler's
    /// tolerance.
    #[cfg(feature = "cuda")]
    #[test]
    fn bf16_softmax_attention_matches_f32_within_tolerance() {
        let dev = Device::new_cuda(0).expect("cuda:0");
        let (b, h, s, d) = (1usize, 4usize, 96usize, 64usize);
        let mk = || {
            Tensor::randn(0f32, 1f32, (b, h, s, d), &dev)
                .and_then(|t| t.to_dtype(DType::BF16))
                .unwrap()
        };
        let (q, k, v) = (mk(), mk(), mk());
        let scale = (d as f64).powf(-0.5);

        let f32_path = sdpa_budgeted(&q, &k, &v, scale, usize::MAX, |scores: &Tensor| {
            softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)
        })
        .unwrap();
        let bf16_path = sdpa_budgeted(&q, &k, &v, scale, usize::MAX, |scores: &Tensor| {
            softmax_last_dim(scores)
        })
        .unwrap();

        // Compare in f32. bf16 has ~7-bit mantissa (step ~1/128 near 1.0); the attention output is a
        // convex combination of the (unit-scale) values, so a max-abs a few bf16 ULPs wide is expected
        // and acceptable — an order of magnitude tighter would demand the f32 upcast we are dropping.
        let a = f32_path.to_dtype(DType::F32).unwrap();
        let bpath = bf16_path.to_dtype(DType::F32).unwrap();
        let max_abs = max_abs(&a, &bpath);
        eprintln!("[sc-12894] bf16-vs-f32 softmax attention max_abs = {max_abs:.5}");
        assert!(
            max_abs < 0.05,
            "bf16-softmax attention drifted {max_abs} from the f32 path (> 0.05 bf16 ULPs)"
        );
    }
}
