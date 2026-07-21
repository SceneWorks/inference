//! The **trainable** Krea 2 single-stream DiT (sc-7577) — the candle twin of `mlx-gen-krea`'s
//! training DiT, vendored alongside the inference [`crate::transformer::Krea2Transformer`] for the same
//! reason the Z-Image trainer vendors its DiT: candle's fused `softmax_last_dim` / `ops::rms_norm`
//! kernels are `CustomOp`s with **no backward** (they silently yield `None` grads), so any module the
//! gradient must flow *through* to reach a LoRA factor has to use composable ops instead.
//!
//! ## What is — and is NOT — re-implemented
//!
//! In the backward, the chain runs `loss → final_layer → block_{N-1} → … → block_0`. Every module on
//! that path must be differentiable, so the **single-stream blocks** ([`TrainBlock`]) and the
//! **`final_layer`** are re-implemented here with composable softmax ([`candle_nn::ops::softmax`]) and a
//! composable `+1` RMSNorm ([`rms_scale_diff`]). The **trainable** seam is each block's attention
//! `to_q/to_k/to_v/to_out.0` projections ([`KREA_ATTN_TARGETS`], what [`LoraHost::visit_lora_mut`]
//! exposes); the modulation table stays frozen.
//!
//! Since sc-11720, every Linear leaf on the frozen base — the attention `to_gate`, the SwiGLU FFN, and
//! the pre-main front-end (`img_in`, the timestep MLP, `txt_in.linear_1/2`, `final_layer.linear`) — is
//! also wrapped so it can host a **forward-time additive USER LoRA** at control-lane inference (the
//! [`crate::adapters::AdditiveDit`] surface, disjoint from the trainer surface above). These wrappers
//! carry no trainable factor, so training and `control_scale=0` are byte-identical to a plain `Linear`:
//! candle's `sorted_nodes` prunes any backward branch that reaches no `Var`, so the front-end is still
//! never differentiated and train/infer conditioning parity holds at zero cost. `time_mod_proj` stays a
//! plain `Linear` (out of the adapter surface, matching the txt2img front-end set).
//!
//! ## Velocity sign
//!
//! Unlike the Z-Image trainer (which negates the DiT output to match its inference pipeline's
//! `noise_pred.neg()`), Krea's inference pipeline consumes the **raw** velocity directly
//! (`x + v·Δσ`, [`crate::pipeline`]). So [`KreaTrainDit::forward`] returns the raw velocity and the
//! trainer regresses it toward `noise − x0` with no negation — the Lens convention.
//!
//! ## Gradient checkpointing
//!
//! Because the default training surface is the 28 single-stream blocks' attention, **all** adapters
//! live in the checkpointed main stack — there is no retained-pre-main adapter to stitch back (the
//! Z-Image complication). So the checkpointed path is the plain
//! [`checkpointed_backward`](candle_gen::train::gradient_checkpoint::checkpointed_backward): run
//! [`forward_pre_main`](KreaTrainDit::forward_pre_main) once (frozen, detached at the boundary),
//! checkpoint the [`main_layer_segments`](KreaTrainDit::main_layer_segments), and recompute the loss in
//! the final segment via [`velocity_out`](KreaTrainDit::velocity_out).

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::ops::{sigmoid, softmax};
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::train::gradient_checkpoint::Segment;
use candle_gen::train::lora::{LoraHost, LoraLinear};

use crate::config::Krea2Config;
use candle_gen::quant::QLinear as SharedQLinear;

use crate::loader::{linear, rms_scale_weight, Weights};
use crate::transformer::block::{RmsScale, SwiGlu, TextFusionTransformer};
use crate::transformer::rope::{apply_interleaved_rope, RopeTables};
use crate::transformer::{patchify, temb, unpatchify, RopeCache, ROPE_CACHE_CAP};

/// Default LoRA target suffixes — the single-stream blocks' attention projections (`to_out.0` is the
/// first element of diffusers' `to_out` `ModuleList`, so the suffix literally carries the `.0`). With
/// 28 blocks this is the **112-target** default surface the MLX trainer uses (sc-7577); `to_gate` and
/// the SwiGLU FFN are intentionally not in the default set.
pub const KREA_ATTN_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// Composable `+1` RMSNorm — the differentiable twin of [`crate::loader::rms_scale`] (which calls the
/// no-backward fused `ops::rms_norm`). `weight` is the pre-folded `scale + 1` f32 tensor; the reduction
/// runs in f32 (the reference upcasts) and the result is cast back to `x`'s dtype, so it is numerically
/// the same op the inference path applies — just one the autograd can traverse.
pub(crate) fn rms_scale_diff(x: &Tensor, weight_f32: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let hidden = xf.dim(D::Minus1)? as f64;
    let norm = (xf.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
    let y = xf.broadcast_div(&(norm + eps)?.sqrt()?)?;
    y.broadcast_mul(weight_f32)?.to_dtype(dt)
}

/// Repeat each kv head `groups` times consecutively (`[b,s,hkv,hd] → [b,s,hkv·groups,hd]`) — the
/// composable `repeat_interleave` matching the inference block's `repeat_kv` (reference `enable_gqa`).
pub(crate) fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, s, hkv, hd) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b, s, hkv, groups, hd))?
        .contiguous()?
        .reshape((b, s, hkv * groups, hd))
}

/// The attention-scores element budget the Krea control **activation-chunking rung** (sc-11745) uses
/// once the fit-gate engages it ([`Krea2ControlPaths::chunk_attention`](crate::control_provider)). At
/// `128 Mi` elements each per-block joint `[ctx; img]` scores (and probs) block is ≤ ~256 MiB in bf16 —
/// ~8× under the i32 guard ([`candle_gen::ATTN_SCORES_BUDGET`], `1e9`) — so a 1024² render (per-block
/// scores ~8.9e8 elems: 48 heads × ~4.3k joint tokens²) splits into ~7 query-row chunks, trading a
/// small speed cost for the bounded activation peak. **Only** applied when the fit-gate flips the knob;
/// on a card with headroom the forward runs unchunked at the i32-guard budget (full speed). Set via
/// [`KreaTrainDit::set_attention_budget`] / [`crate::control::ControlBranch::set_attention_budget`] and
/// consumed by `sdpa_diff_budgeted`.
pub const KREA_ATTN_CHUNK_BUDGET: usize = 128 * 1024 * 1024;

/// Bidirectional, unmasked scaled-dot-product attention with a **composable** softmax (the inference
/// `sdpa` uses the no-backward fused `softmax_last_dim`), with **query-row chunking bounded by
/// `budget`** — the Krea control activation-peak lever (sc-11745). Delegates to the shared
/// i32-overflow-safe [`candle_gen::sdpa_budgeted_bhsd`] with the grad-carrying composable
/// `softmax(_, D::Minus1)`, so:
/// - below `budget` it is a **single un-chunked pass** — the plain `q·kᵀ·scale → softmax → ·v` (the
///   ≤1024² default, [`candle_gen::ATTN_SCORES_BUDGET`] — that guard only ever engages past the i32
///   element limit);
/// - above `budget` it chunks over query rows (each row's softmax is over all keys and independent of
///   the others, so the result is numerically identical up to the associativity-free `cat`).
///
/// The Krea control fit-ladder (sc-11754) lowers `budget` from the i32 guard to
/// [`KREA_ATTN_CHUNK_BUDGET`] to bound the per-block joint-attention scratch on a constrained card.
/// `q`/`k`/`v`: `[b, h, s, hd]`.
pub(crate) fn sdpa_diff_budgeted(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    budget: usize,
) -> Result<Tensor> {
    candle_gen::sdpa_budgeted_bhsd(q, k, v, scale, None, |s| softmax(s, D::Minus1), budget)
}

/// Build a frozen base `Linear` (no bias) from the mmap'd `Weights` and wrap it as a trainable
/// [`LoraLinear`], reading `in`/`out` from the on-disk shape (`[out, in]`) and recording `path` as the
/// PEFT module path the harness matches against.
fn lora_proj(w: &Weights, path: &str, bias: bool) -> Result<LoraLinear> {
    let base = linear(w, path, bias)?;
    let (out_f, in_f) = base.weight().dims2()?;
    Ok(LoraLinear::from_linear(base, in_f, out_f, path.to_string()))
}

/// Packed-inference twin of [`lora_proj`] (sc-11727). On a packed q4/q8 tier, build the projection's
/// frozen base as a **packed** [`QLinear`] (`linear_detect`) — the codes STAY packed in VRAM and
/// dequantize on-forward (sc-7702), which is the whole point for a small-card user; a dense-materialized
/// bf16 weight would defeat the tier they installed. Wrap it as an inference-only
/// [`LoraLinear::from_qlinear`] so a USER LoRA (sc-11720/sc-11721) still rides additively over the packed
/// base. The logical `[out, in]` dims come from the `.scales` shape (`[out, in/group]`), so no dense
/// weight is materialized just to read them. On a dense (bf16) tier there is no packed base to keep, so
/// defer to [`lora_proj`] (identical). INFERENCE-ONLY: a packed base cannot host a trainable `Var`
/// residual, so the trainer keeps [`lora_proj`].
fn lora_proj_packed(w: &Weights, path: &str, bias: bool) -> Result<LoraLinear> {
    let scales_key = format!("{path}.scales");
    match w.packed() {
        Some(cfg) if w.contains(&scales_key) => {
            // Build the SHARED `candle_gen::quant::QLinear` straight from the MLX packed triple (the same
            // source `crate::loader::linear_detect` reads, but that returns the krea-local `QLinear`
            // wrapper). `from_packed_gs` repacks into a GGUF `QTensor` that stays quantized in VRAM and
            // dequantizes on-forward (sc-7702) — the codes are NOT materialized to a dense weight.
            let group = cfg.group_size as usize;
            let wq = w.get_native(&format!("{path}.weight"))?;
            let scales = w.get_f32(&scales_key)?;
            let biases = w.get_f32(&format!("{path}.biases"))?;
            let dense_bias = if bias {
                Some(w.get(&format!("{path}.bias"))?)
            } else {
                None
            };
            let sdims = scales.dims();
            let (out_f, in_f) = (sdims[0], sdims[1] * group);
            let ql = SharedQLinear::from_packed_gs(
                &wq,
                &scales,
                &biases,
                dense_bias,
                group,
                w.device(),
            )?;
            Ok(LoraLinear::from_qlinear(ql, in_f, out_f, path.to_string()))
        }
        _ => lora_proj(w, path, bias),
    }
}

/// The projection loader a [`KreaTrainDit`] build uses: the packed-detecting [`lora_proj_packed`] on the
/// control-INFERENCE path (`packed = true`, codes stay in VRAM), else the dense trainable [`lora_proj`].
type ProjLoader = fn(&Weights, &str, bool) -> Result<LoraLinear>;

fn proj_loader(packed: bool) -> ProjLoader {
    if packed {
        lora_proj_packed
    } else {
        lora_proj
    }
}

/// Sigmoid-gated GQA attention with the four attention projections as adaptable [`LoraLinear`]s and a
/// frozen `to_gate` / per-head `+1` RMSNorm — the trainable twin of [`crate::transformer::block`]'s
/// `GatedAttention`.
struct TrainAttention {
    q: LoraLinear,
    k: LoraLinear,
    v: LoraLinear,
    gate: LoraLinear,
    o: LoraLinear,
    norm_q: Tensor, // f32, scale + 1
    norm_k: Tensor, // f32, scale + 1
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
    scale: f64,
    /// Query-row chunking budget for the joint `[ctx; img]` attention (sc-11745). Defaults to the i32
    /// guard ([`candle_gen::ATTN_SCORES_BUDGET`]) — unchunked at ≤1024²; the Krea control fit-gate
    /// lowers it to [`KREA_ATTN_CHUNK_BUDGET`] via [`KreaTrainDit::set_attention_budget`].
    attn_budget: usize,
}

impl TrainAttention {
    fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
        packed: bool,
    ) -> Result<Self> {
        let proj = proj_loader(packed);
        Ok(Self {
            q: proj(w, &format!("{prefix}.to_q"), false)?,
            k: proj(w, &format!("{prefix}.to_k"), false)?,
            v: proj(w, &format!("{prefix}.to_v"), false)?,
            gate: proj(w, &format!("{prefix}.to_gate"), false)?,
            o: proj(w, &format!("{prefix}.to_out.0"), false)?,
            norm_q: rms_scale_weight(w, &format!("{prefix}.norm_q.weight"))?,
            norm_k: rms_scale_weight(w, &format!("{prefix}.norm_k.weight"))?,
            heads,
            kv_heads,
            head_dim,
            eps,
            scale: (head_dim as f64).powf(-0.5),
            attn_budget: candle_gen::ATTN_SCORES_BUDGET,
        })
    }

    /// Visit the four adaptable projections in install order.
    fn visit(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.q)?;
        f(&mut self.k)?;
        f(&mut self.v)?;
        f(&mut self.o)?;
        Ok(())
    }

    fn forward(&self, x: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;
        let gate = self.gate.forward(x)?;

        let q = rms_scale_diff(&q, &self.norm_q, self.eps)?;
        let k = rms_scale_diff(&k, &self.norm_k, self.eps)?;
        let (q, k) = match rope {
            Some((cos, sin)) => (
                apply_interleaved_rope(&q, cos, sin)?,
                apply_interleaved_rope(&k, cos, sin)?,
            ),
            None => (q, k),
        };

        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa_diff_budgeted(&q, &k, &v, self.scale, self.attn_budget)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;

        let gated = (o * sigmoid(&gate)?)?;
        self.o.forward(&gated)
    }
}

/// One trainable single-stream block (`DoubleSharedModulation`) — the differentiable twin of
/// [`crate::transformer::block`]'s `SingleStreamBlock`. The SwiGLU FFN is the inference crate's
/// (composable, frozen) [`SwiGlu`]; only the norms are swapped for the composable [`rms_scale_diff`].
pub(crate) struct TrainBlock {
    scale_shift_table: Tensor, // [1, 1, 6·hidden]
    prenorm: Tensor,           // f32, scale + 1
    postnorm: Tensor,          // f32, scale + 1
    attn: TrainAttention,
    mlp: SwiGlu,
    eps: f64,
}

impl TrainBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        eps: f64,
        packed: bool,
    ) -> Result<Self> {
        let sst = w
            .get(&format!("{prefix}.scale_shift_table"))?
            .reshape((1, 1, 6 * hidden))?;
        Ok(Self {
            scale_shift_table: sst,
            prenorm: rms_scale_weight(w, &format!("{prefix}.norm1.weight"))?,
            postnorm: rms_scale_weight(w, &format!("{prefix}.norm2.weight"))?,
            attn: TrainAttention::load(
                w,
                &format!("{prefix}.attn"),
                heads,
                kv_heads,
                head_dim,
                eps,
                packed,
            )?,
            mlp: SwiGlu::load(w, &format!("{prefix}.ff"))?,
            eps,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        tvec: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let m = tvec.broadcast_add(&self.scale_shift_table)?; // [b, 1, 6·hidden]
        let chunks = m.chunk(6, D::Minus1)?;
        let (prescale, preshift, pregate) = (&chunks[0], &chunks[1], &chunks[2]);
        let (postscale, postshift, postgate) = (&chunks[3], &chunks[4], &chunks[5]);

        let pre = rms_scale_diff(x, &self.prenorm, self.eps)?
            .broadcast_mul(&(prescale + 1.0)?)?
            .broadcast_add(preshift)?;
        let attn = self.attn.forward(&pre, Some((cos, sin)))?;
        let x = (x + attn.broadcast_mul(pregate)?)?;

        let post = rms_scale_diff(&x, &self.postnorm, self.eps)?
            .broadcast_mul(&(postscale + 1.0)?)?
            .broadcast_add(postshift)?;
        let mlp = self.mlp.forward(&post)?;
        &x + mlp.broadcast_mul(postgate)?
    }
}

/// The constants the single-stream stack + final layer need, computed once in
/// [`KreaTrainDit::forward_pre_main`] and threaded (cloned) into the per-block segments / the loss
/// segment. None of these carry a trainable factor, so they are the detached boundary of the
/// checkpointed backward.
pub struct MainCtx {
    pub(crate) tvec: Tensor, // [b, 1, 6·hidden] shared modulation
    pub(crate) rcos: Tensor, // joint RoPE cos table
    pub(crate) rsin: Tensor, // joint RoPE sin table
    t: Tensor,               // [b, 1, hidden] for the final SimpleModulation
    pub(crate) cap_len: usize,
    pub(crate) img_len: usize,
    ht: usize,
    wt: usize,
    latent_ch: usize,
    patch: usize,
}

/// The trainable Krea 2 single-stream DiT. Built from the same mmap'd `transformer/` `Weights` the
/// inference path loads — the frozen base is shared; only the attention projections grow a `Var`-backed
/// LoRA residual (installed by [`build_lora_targets`](candle_gen::train::lora::build_lora_targets)).
pub struct KreaTrainDit {
    cfg: Krea2Config,
    device: Device,
    dtype: DType,
    // --- pre-main front-end. Upstream of the control branch (never trained); the Linear leaves are
    //     `LoraLinear` so a USER LoRA (control-lane inference, sc-11720) can ride additively — identity
    //     to the plain Linear when unadapted, so training / control_scale=0 stay byte-exact. `time_mod_
    //     proj` stays plain Linear (out of the adapter surface, matching the txt2img front-end set). ---
    img_in: LoraLinear,
    time_embed_l1: LoraLinear,
    time_embed_l2: LoraLinear,
    time_mod_proj: Linear,
    txt_in_norm: RmsScale,
    txt_in_l1: LoraLinear,
    txt_in_l2: LoraLinear,
    text_fusion: TextFusionTransformer,
    // --- trainable single-stream stack ---
    blocks: Vec<TrainBlock>,
    // --- final layer (composable; on the backward path to every adapter) ---
    final_norm: Tensor, // f32, scale + 1
    final_linear: LoraLinear,
    final_sstable: Tensor, // [1, 2, hidden]
    /// The control/training front-end sees fixed `(caption, height, width)` geometry throughout a
    /// denoise loop, so share the same bounded RoPE cache used by the inference DiT.
    rope_cache: RopeCache<(usize, usize, usize), (Tensor, Tensor)>,
}

impl KreaTrainDit {
    /// Build the composable DiT for **training** (the control-branch trainer / LoRA trainer): every
    /// adaptable projection is a dense `Var`-trainable [`LoraLinear`], so a packed tier is dequantized to
    /// dense on load. This is the historical `load`; the control **inference** provider should use
    /// [`load_inference`](Self::load_inference) to keep a packed q4/q8 base packed in VRAM (sc-11727).
    pub fn load(w: &Weights, cfg: &Krea2Config) -> Result<Self> {
        Self::load_impl(w, cfg, false)
    }

    /// Build the composable DiT for the **control-inference** lane: on a packed q4/q8 tier the attention
    /// and front-end projections load as **packed** [`candle_gen::quant::QLinear`] bases (dequant-on-forward, codes stay in
    /// VRAM — a q4 DiT keeps its ~¼ footprint instead of ballooning to dense bf16), the FFN / text-fusion
    /// leaves already packed-detect, and a USER LoRA still rides additively. The frozen base is never
    /// trained, so nothing needs `Var` master weights. On a dense (bf16) tier this is identical to
    /// [`load`](Self::load). sc-11727 packed pose-control forward.
    pub fn load_inference(w: &Weights, cfg: &Krea2Config) -> Result<Self> {
        Self::load_impl(w, cfg, true)
    }

    /// Shared build. `packed = true` routes the adaptable projections through the packed-keeping
    /// [`lora_proj_packed`] (control inference); `false` keeps the dense trainable [`lora_proj`].
    fn load_impl(w: &Weights, cfg: &Krea2Config, packed: bool) -> Result<Self> {
        let (heads, kv, hd, eps) = (
            cfg.num_attention_heads,
            cfg.num_kv_heads,
            cfg.attention_head_dim,
            cfg.norm_eps,
        );
        let (theads, tkv) = (cfg.text_num_attention_heads, cfg.text_num_kv_heads);
        let hidden = cfg.hidden_size;
        let proj = proj_loader(packed);

        let final_sstable = w
            .get("final_layer.scale_shift_table")?
            .reshape((1, 2, hidden))?;

        Ok(Self {
            cfg: cfg.clone(),
            device: w.device().clone(),
            dtype: w.dtype(),
            img_in: proj(w, "img_in", true)?,
            time_embed_l1: proj(w, "time_embed.linear_1", true)?,
            time_embed_l2: proj(w, "time_embed.linear_2", true)?,
            time_mod_proj: linear(w, "time_mod_proj", true)?,
            txt_in_norm: RmsScale::load(w, "txt_in.norm.weight", eps)?,
            txt_in_l1: proj(w, "txt_in.linear_1", true)?,
            txt_in_l2: proj(w, "txt_in.linear_2", true)?,
            text_fusion: TextFusionTransformer::load(
                w,
                cfg.num_layerwise_text_blocks,
                cfg.num_refiner_text_blocks,
                theads,
                tkv,
                hd,
                eps,
            )?,
            blocks: (0..cfg.num_layers)
                .map(|i| {
                    TrainBlock::load(
                        w,
                        &format!("transformer_blocks.{i}"),
                        heads,
                        kv,
                        hd,
                        hidden,
                        eps,
                        packed,
                    )
                })
                .collect::<Result<_>>()?,
            final_norm: rms_scale_weight(w, "final_layer.norm.weight")?,
            final_linear: proj(w, "final_layer.linear", true)?,
            final_sstable,
            rope_cache: RopeCache::new(ROPE_CACHE_CAP),
        })
    }

    fn rope_tables(&self, cap_len: usize, ht: usize, wt: usize) -> Result<(Tensor, Tensor)> {
        self.rope_cache.get_or_build((cap_len, ht, wt), || {
            let rope = RopeTables::build_t2i(
                cap_len,
                ht,
                wt,
                self.cfg.axes_dims_rope,
                self.cfg.rope_theta as f64,
                &self.device,
            )?;
            Ok(rope.joint())
        })
    }

    /// Run the frozen front-end: patch-embed the latent, build the shared modulation, aggregate +
    /// project the text conditioning, and fuse to the joint `[ctx; img]` sequence. Returns that joint
    /// sequence (the differentiable boundary entering block 0) plus the `MainCtx` the stack/final
    /// need. `latent`: `[b, 16, H, W]`; `timestep`: `[b]` (the raw flow σ); `context`:
    /// `[b, n_tok, num_text_layers, text_hidden]` (the stacked Qwen3-VL select layers).
    pub fn forward_pre_main(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, MainCtx)> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        let dt = self.dtype;
        let (_, _, h, w) = latent.dims4()?;
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;
        let latent_ch = cfg.in_channels / (p * p);
        let cap_len = context.dim(1)?;
        let context = context.to_dtype(dt)?;

        let img = self.img_in.forward(&patchify(&latent.to_dtype(dt)?, p)?)?;

        let t_sin = temb(timestep, cfg.timestep_embed_dim, &self.device)?.to_dtype(dt)?;
        let t = self
            .time_embed_l2
            .forward(&self.time_embed_l1.forward(&t_sin)?.gelu()?)?;
        let tvec = self.time_mod_proj.forward(&t.gelu()?)?;

        let ctx = self.text_fusion.forward(&context)?;
        let ctx = self.txt_in_norm.forward(&ctx)?;
        let ctx = self
            .txt_in_l2
            .forward(&self.txt_in_l1.forward(&ctx)?.gelu()?)?;

        let combined = Tensor::cat(&[&ctx, &img], 1)?;
        let (rcos, rsin) = self.rope_tables(cap_len, ht, wt)?;
        Ok((
            combined,
            MainCtx {
                tvec,
                rcos,
                rsin,
                t,
                cap_len,
                img_len,
                ht,
                wt,
                latent_ch,
                patch: p,
            },
        ))
    }

    /// One [`Segment`] per single-stream block (for the checkpointed backward): each recomputes its
    /// block forward over the incoming joint sequence, threading the (constant) shared modulation +
    /// RoPE tables borrowed from `ctx`. The `Segment` lifetime ties to both `self` (the block refs) and
    /// `ctx`, so the trainer can push a `ctx`-borrowing loss segment after these.
    pub fn main_layer_segments<'a>(&'a self, ctx: &'a MainCtx) -> Vec<Segment<'a>> {
        self.blocks
            .iter()
            .map(|blk| -> Segment<'a> {
                Box::new(move |st: &[Tensor]| {
                    Ok(vec![blk.forward(&st[0], &ctx.tvec, &ctx.rcos, &ctx.rsin)?])
                })
            })
            .collect()
    }

    /// The continuous-AdaLN output head: `LastLayer` (SimpleModulation on `t`) over the joint sequence,
    /// then slice the image tokens and unpatchify back to a velocity `[b, 16, H, W]`. Composable (it is
    /// on the backward path to every block adapter).
    pub fn velocity_out(&self, combined: &Tensor, ctx: &MainCtx) -> Result<Tensor> {
        let m = ctx.t.broadcast_add(&self.final_sstable)?; // [b, 2, hidden]
        let scale = m.narrow(1, 0, 1)?;
        let shift = m.narrow(1, 1, 1)?;
        let normed = rms_scale_diff(combined, &self.final_norm, self.cfg.norm_eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        let out = self.final_linear.forward(&normed)?; // [b, cap+img_len, in_channels]
        let img_out = out.narrow(1, ctx.cap_len, ctx.img_len)?;
        unpatchify(&img_out, ctx.ht, ctx.wt, ctx.patch, ctx.latent_ch)
    }

    /// Dense (retained) velocity prediction — the same surface as the inference
    /// [`Krea2Transformer::forward`](crate::transformer::Krea2Transformer::forward), built from the
    /// composable trainable blocks. Returns the **raw** velocity `[b, 16, H, W]` (no negation).
    pub fn forward(&self, latent: &Tensor, timestep: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (mut combined, ctx) = self.forward_pre_main(latent, timestep, context)?;
        for blk in &self.blocks {
            combined = blk.forward(&combined, &ctx.tvec, &ctx.rcos, &ctx.rsin)?;
        }
        self.velocity_out(&combined, &ctx)
    }

    /// The single-stream stack, exposed for the pose-ControlNet spike (sc-8460): the control branch
    /// injects per-block residuals into this stack from the outside
    /// ([`crate::control::forward_with_control`]).
    pub(crate) fn blocks(&self) -> &[TrainBlock] {
        &self.blocks
    }

    /// Patch-embed an extra (control) latent through the **frozen** base `img_in` — the control
    /// branch's conditioning embedder (sc-8460). `latent`: `[b, 16, H, W]` → `[b, img_len, hidden]`,
    /// exactly the embedding [`forward_pre_main`](Self::forward_pre_main) gives the noisy latent.
    pub(crate) fn embed_latent(&self, latent: &Tensor) -> Result<Tensor> {
        let p = self.cfg.patch_size;
        self.img_in
            .forward(&patchify(&latent.to_dtype(self.dtype)?, p)?)
    }

    /// Set the per-block joint-attention scores budget on the single-stream stack — the Krea control
    /// **activation-chunking rung** (sc-11745). The load default (`ATTN_SCORES_BUDGET`) leaves the
    /// forward unchunked at ≤1024² (full speed); the fit-gate lowers it to [`KREA_ATTN_CHUNK_BUDGET`]
    /// to bound the activation peak on a constrained card, forcing sc-6217-style query-row chunking
    /// (numerically identical). Only the main single-stream blocks are affected — the tiny text-fusion
    /// attention (≤ caption tokens) is never a peak, so it is left at its default. Idempotent; call
    /// before the sampler loop.
    pub fn set_attention_budget(&mut self, budget: usize) {
        for blk in &mut self.blocks {
            blk.attn.attn_budget = budget;
        }
    }
}

impl LoraHost for KreaTrainDit {
    fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for blk in &mut self.blocks {
            blk.attn.visit(f)?;
        }
        Ok(())
    }
}

impl crate::adapters::AdditiveDit for KreaTrainDit {
    /// The control-lane adapter surface (sc-11720): per-block attention (`to_q|to_k|to_v|to_gate|
    /// to_out.0`, all [`LoraLinear`]) + SwiGLU FFN (`ff.gate|ff.up|ff.down`, the shared inference
    /// [`SwiGlu`]'s `AdaptLinear` leaves) + the text-fusion blocks + the front-end / final leaves. This is
    /// the INFERENCE user-LoRA surface only — disjoint from [`LoraHost::visit_lora_mut`], which the
    /// control-branch trainer walks and which stays attention-only, so training is unaffected. USER
    /// residuals fold onto this frozen base DiT; the control branch is never adapted.
    fn visit_additive(
        &mut self,
        f: &mut dyn FnMut(&str, &mut dyn crate::adapters::AdditiveProj) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            let p = format!("transformer_blocks.{i}");
            f(&format!("{p}.attn.to_q"), &mut blk.attn.q)?;
            f(&format!("{p}.attn.to_k"), &mut blk.attn.k)?;
            f(&format!("{p}.attn.to_v"), &mut blk.attn.v)?;
            f(&format!("{p}.attn.to_gate"), &mut blk.attn.gate)?;
            f(&format!("{p}.attn.to_out.0"), &mut blk.attn.o)?;
            blk.mlp
                .visit_adaptable_mut(&format!("{p}.ff"), &mut |path, a| f(path, a))?;
        }
        self.text_fusion
            .visit_adaptable_mut(&mut |path, a| f(path, a))?;
        for (path, proj) in [
            ("img_in", &mut self.img_in),
            ("time_embed.linear_1", &mut self.time_embed_l1),
            ("time_embed.linear_2", &mut self.time_embed_l2),
            ("txt_in.linear_1", &mut self.txt_in_l1),
            ("txt_in.linear_2", &mut self.txt_in_l2),
            ("final_layer.linear", &mut self.final_linear),
        ] {
            f(path, proj)?;
        }
        Ok(())
    }

    fn adapter_device(&self) -> Device {
        self.device.clone()
    }

    fn adapter_surface_hint(&self) -> &'static str {
        "expected bare/PEFT `<path>.lora_A/B.weight` (LoRA) or `<module>.lokr_w1/w2` (LoKr) over the \
         control-base DiT attention (to_q|to_k|to_v|to_gate|to_out.0) + SwiGLU FFN (ff.gate|ff.up|ff.\
         down) across the single-stream transformer_blocks and text_fusion blocks, plus the front-end \
         (img_in|time_embed.linear_1/2|txt_in.linear_1/2|final_layer.linear) projections; or a ComfyUI/\
         lightx2v `<module>.diff`/`.diff_b` diff-patch (full-weight/bias delta, incl. the \
         text_fusion.projector 12→1 collapse). The pose control branch is never adapted; conv-layer / \
         text-encoder adapters are out of surface"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn control_rope_cache_builds_once_and_matches_fresh_tables() -> Result<()> {
        let dev = Device::Cpu;
        let key = (3usize, 2usize, 2usize);
        let axes = [4usize, 6usize, 6usize];
        let cache: RopeCache<(usize, usize, usize), (Tensor, Tensor)> =
            RopeCache::new(ROPE_CACHE_CAP);
        let builds = Cell::new(0usize);
        let build = || {
            builds.set(builds.get() + 1);
            Ok(RopeTables::build_t2i(key.0, key.1, key.2, axes, 1000.0, &dev)?.joint())
        };

        let first = cache.get_or_build(key, build)?;
        let second = cache.get_or_build(key, build)?;
        let fresh = RopeTables::build_t2i(key.0, key.1, key.2, axes, 1000.0, &dev)?.joint();

        assert_eq!(builds.get(), 1, "fixed denoise geometry must build once");
        assert_eq!(
            first.0.flatten_all()?.to_vec1::<f32>()?,
            fresh.0.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            first.1.flatten_all()?.to_vec1::<f32>()?,
            fresh.1.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            second.0.flatten_all()?.to_vec1::<f32>()?,
            first.0.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
