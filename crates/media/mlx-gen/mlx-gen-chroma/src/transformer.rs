//! The Chroma DiT (`ChromaTransformer2DModel`).
//!
//! The FLUX MMDiT skeleton (19 dual + 38 single blocks, FluxPosEmbed RoPE, gelu-tanh FFN) with the
//! Chroma deltas:
//! - **sc-3836:** the distilled-guidance modulation generator — `ChromaCombinedTimestepTextProjEmbeddings`
//!   + `ChromaApproximator` → `pooled_temb [B, mod_index_len, inner]`.
//! - **sc-3837 (this slice):** the forward pass — `x_embedder`/`context_embedder`, RoPE over
//!   `cat(txt_ids, img_ids)`, the double/single blocks with **pruned adaLN** (modulation *sliced* from
//!   `pooled_temb`, no per-block linear), **MMDiT attention masking** (the 0/1 mask is added to the
//!   scores, the reference's literal behavior), QK-norm RMS eps **1e-6**, and the pruned `norm_out` +
//!   `proj_out`.
//!
//! The transformer runs f32 activations (parity is to the torch-`diffusers` reference; the cross-
//! backend f32 floor is ~1e-3, see the parity tests). The masked T5 encode that *builds* the
//! sequence mask is sc-3838; the generate path is sc-3839.

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, LinearFacts};
use mlx_gen::attention::{sdpa_budgeted_bhsd, AttentionPlan};
use mlx_gen::nn::{gated, gelu_tanh, silu};
/// Re-exported so the model's denoise loop can enable the shared `mx.compile` fusion of the DiT's
/// elementwise glue (adaLN modulate + gated residuals), matching FLUX.1/FLUX.2 (F-101/F-102).
/// [`CompileGlueGuard`] is the RAII form the production denoise binds so the toggle is restored on
/// drop (F-007) instead of leaking the render thread's setting into later work.
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};
use mlx_gen::qkv::{
    self, AttnPrepSpec, FusedQkvProjection, NormDtype, QkNormSpec, QkvPart, QkvSource, RopeDtype,
    RopeSpec, RopeStyle, RopeTables, RotationAxes, StreamOrder,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::{layer_norm, rms_norm};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};

use crate::config::ChromaTransformerConfig;

/// Sinusoid / RoPE frequency base (diffusers `get_timestep_embedding` `max_period` and
/// `FluxPosEmbed(theta=10000)`).
const MAX_PERIOD: f64 = 10000.0;
const ROPE_THETA: f32 = 10000.0;
/// RMSNorm epsilon for the Approximator norms — torch `nn.RMSNorm(hidden)` with `eps=None` resolves
/// to `torch.finfo(float32).eps` (the f32 path).
const APPROX_RMS_EPS: f32 = 1.192_092_9e-7;
/// QK-norm RMS epsilon. Chroma's `FluxAttention(eps=1e-6)` — **NOT** FLUX's 1e-5.
const QK_RMS_EPS: f32 = 1e-6;
/// AdaLayerNorm LayerNorm epsilon (all pruned norms + `norm_out`, `elementwise_affine=False`).
const LN_EPS: f32 = 1e-6;

// ============================ leaf helpers ============================

/// `get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift)` (diffusers),
/// in f32. `dim` even. `flip_sin_to_cos=True` ⇒ output order `[cos, sin]`. Delegates to the shared
/// [`mlx_gen::nn::timestep_sincos`] (F-016) — the FLUX-special-case form parameterized on
/// `max_period` + `downscale_freq_shift`. NOTE: the shared builder computes the exponent with MLX ops
/// (vs this port's prior host-f64 loop); the ~1e-7 shift is within the Chroma parity tolerance
/// (`approximator_parity` / `transformer_parity` gate it) and makes Chroma share FLUX's exact
/// on-device path.
fn timestep_embedding(timesteps: &Array, dim: usize, downscale_freq_shift: f64) -> Result<Array> {
    mlx_gen::nn::timestep_sincos(timesteps, dim, MAX_PERIOD, downscale_freq_shift)
}

/// A dense-or-packed `nn.Linear` (`[out, in]` weight + bias) wrapping the core [`AdaptableLinear`] —
/// so it can be quantized (sc-3841) and carry LoRA/LoKr adapters (sc-3842). The forward runs f32
/// activations over the bf16 (or quantized) weight; mlx promotes.
///
/// Loads through [`crate::quant::lin`], which **packed-detects** via `{prefix}.scales`: a
/// pre-quantized (Q4/Q8) turnkey's block Linears load as already-quantized bases directly (no dense
/// transient — sc-8777), while a dense snapshot (and the always-dense embedders/Approximator) load
/// dense exactly as before. Every Chroma Linear carries a bias, so `bias = true` throughout.
pub(crate) struct Lin(AdaptableLinear);

impl Lin {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self(crate::quant::lin(w, prefix, true)?))
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.0.forward(x)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.0.quantize(bits, None)
    }

    fn inner_mut(&mut self) -> &mut AdaptableLinear {
        &mut self.0
    }

    /// Unwrap to the bare [`AdaptableLinear`], for the q/k/v triples that now live inside a
    /// [`FusedQkvProjection`] (SC-18319). `Lin` adds no state over its inner linear, so this is the
    /// whole of it.
    fn into_inner(self) -> AdaptableLinear {
        self.0
    }
}

/// adaLN affine `normed·(1+scale) + shift`. `scale`/`shift` are `[B,1,inner]` (broadcast over seq).
/// Delegates to the shared [`mlx_gen::nn::modulate`] with `one_matches_scale=false` (strong-f32 `1`,
/// matching the previous hand-rolled affine bit-for-bit) so it fuses under `compile_glue` (F-102).
fn modulate(normed: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    mlx_gen::nn::modulate(normed, scale, shift, false)
}

/// The `j`-th modulation row of a `[B,K,inner]` slice, as `[B,1,inner]` (broadcastable over seq).
fn row(block: &Array, j: i32) -> Result<Array> {
    Ok(block.take_axis(Array::from_int(j), 1)?.expand_dims(1)?)
}

/// `len` contiguous modulation rows of `pooled_temb` from `start`, as `[B,len,inner]`.
fn rows(t: &Array, start: i32, len: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..start + len).collect();
    Ok(t.take_axis(Array::from_slice(&idx, &[len]), 1)?)
}

/// `len` contiguous sequence positions of `[B,S,inner]` from `start`, as `[B,len,inner]`.
fn seq_slice(t: &Array, start: i32, len: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..start + len).collect();
    Ok(t.take_axis(Array::from_slice(&idx, &[len]), 1)?)
}

// ============================ RoPE ============================

pub(crate) struct RopeTable {
    cos: Array,
    sin: Array,
}

/// FluxPosEmbed: per-axis sinusoid tables from position ids `[N,3]`, concatenated to `[N, head_dim/2]`.
/// Delegates to the shared [`mlx_gen::nn::rope_sincos_from_ids`] (F-015) — the same `omega =
/// theta^-(2k/dim)`, `out = pos·omega`, `cos`/`sin` the FLUX port uses (bit-exact: identical MLX ops).
fn build_rope(ids: &Array, axes: [usize; 3]) -> Result<RopeTable> {
    let (cos, sin) = mlx_gen::nn::rope_sincos_from_ids(
        ids,
        &[axes[0] as i32, axes[1] as i32, axes[2] as i32],
        ROPE_THETA,
    )?;
    Ok(RopeTable { cos, sin })
}

/// SC-18319 — Chroma's row of the shared knob table, in one place so the single-stream and joint
/// attentions provably select the same policy.
///
/// `rope` is `Some` only for the single-stream blocks, which rotate per stream; the joint blocks
/// pass `None` and rotate the **already-joined** sequence afterwards (knob 8's concat-then-RoPE arm,
/// via [`rotate_joint`]).
fn chroma_spec<'a>(
    heads: i32,
    head_dim: i32,
    norm_q: &'a Array,
    norm_k: &'a Array,
    rope: Option<&'a RopeTable>,
) -> AttnPrepSpec<'a> {
    let spec = AttnPrepSpec::new(heads, head_dim)
        .with_qk_norm(
            QkNormSpec::per_head(norm_q, norm_k, QK_RMS_EPS).with_dtype(NormDtype::PromoteToF32),
        )
        .with_rotation_axes(RotationAxes::HeadMajor);
    match rope {
        Some(r) => spec.with_rope(RopeSpec {
            style: RopeStyle::AdjacentPair,
            q: Some(RopeTables::new(&r.cos, &r.sin)),
            k: Some(RopeTables::new(&r.cos, &r.sin)),
            // Knob 12 — the removed `apply_rope_one` promoted and did NOT cast back. Chroma's
            // whole prologue already runs f32 (`NormDtype::PromoteToF32` above), so this is a
            // no-op here — stated rather than defaulted.
            dtype: RopeDtype::Promoted,
            ..RopeSpec::default()
        }),
        None => spec,
    }
}

/// Rotate an already-joined `[B, H, S, hd]` stream — the second half of knob 8's concat-then-RoPE.
fn rotate_joint(x: &Array, rope: &RopeTable) -> Result<Array> {
    qkv::apply_rope(
        x,
        RopeTables::new(&rope.cos, &rope.sin),
        RopeStyle::AdjacentPair,
        RotationAxes::HeadMajor,
        None,
        RopeDtype::Promoted,
    )
}

/// Scaled-dot-product attention over `[B,H,S,hd]` → `[B,S,inner]`. `mask` is the additive `[B,1,S,S]`
/// MMDiT mask (Chroma adds the 0/1 mask to the scores) or `None`.
///
/// Ladder rung 3 (SC-15520): `plan` is threaded down from the request. [`AttentionPlan::UNBOUNDED`]
/// — the default every unselected request carries — makes
/// [`sdpa_budgeted_bhsd`](mlx_gen::attention::sdpa_budgeted_bhsd) take its single-call fast path,
/// which is byte-for-byte the historical `scaled_dot_product_attention` call. A bounded plan splits
/// the query rows, each block attending over the **complete** k/v, and narrows the per-query
/// `[B,1,S,S]` mask onto each block; precision, scale, seed and schedule are untouched.
fn sdpa(
    q: &Array,
    k: &Array,
    v: &Array,
    hd: i32,
    mask: Option<&Array>,
    plan: AttentionPlan<'_>,
) -> Result<Array> {
    let b = q.shape()[0];
    let scale = (hd as f32).powf(-0.5);
    // `&Array` is taken as an *additive* mask (Chroma's 0/1 mask is added to the scores).
    let y = sdpa_budgeted_bhsd(q, k, v, scale, mask, plan)?;
    Ok(y.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, -1, q.shape()[1] * hd])?)
}

// ============================ embeddings + Approximator (sc-3836) ============================

/// `ChromaCombinedTimestepTextProjEmbeddings` — builds the Approximator input vector (parameter-free).
struct TimestepTextProj {
    num_channels: usize,
    mod_proj: Array,
}

impl TimestepTextProj {
    fn new(cfg: &ChromaTransformerConfig) -> Result<Self> {
        let num_channels = cfg.approximator_num_channels / 4;
        let n = cfg.mod_index_len();
        let idx: Vec<f32> = (0..n).map(|i| (i as f32) * 1000.0).collect();
        let idx = Array::from_slice(&idx, &[n as i32]);
        let mod_proj = timestep_embedding(&idx, 2 * num_channels, 0.0)?;
        Ok(Self {
            num_channels,
            mod_proj,
        })
    }

    /// `timestep` already scaled (`t*1000`), shape `[B]`. Returns `input_vec [B, mod_index_len, 4*nc]`.
    fn forward(&self, timestep: &Array) -> Result<Array> {
        let b = timestep.shape()[0];
        let n = self.mod_proj.shape()[0];
        let nc = 2 * self.num_channels as i32;
        let time = timestep_embedding(timestep, self.num_channels, 0.0)?;
        let zeros = Array::from_slice(&vec![0.0_f32; b as usize], &[b]);
        let guid = timestep_embedding(&zeros, self.num_channels, 0.0)?;
        let tg = concatenate_axis(&[time, guid], -1)?.reshape(&[b, 1, nc])?;
        let tg = broadcast_to(&tg, &[b, n, nc])?;
        let mp = broadcast_to(&self.mod_proj.reshape(&[1, n, nc])?, &[b, n, nc])?;
        Ok(concatenate_axis(&[tg, mp], -1)?)
    }
}

/// `ChromaApproximator` — `in_proj` then `n_layers` residual blocks
/// `x = x + linear_2(silu(linear_1(rms_norm(x))))`, then `out_proj`.
struct Approximator {
    in_proj: Lin,
    layers: Vec<(Lin, Lin)>,
    norms: Vec<Array>,
    out_proj: Lin,
}

impl Approximator {
    fn load(w: &Weights, cfg: &ChromaTransformerConfig) -> Result<Self> {
        let p = "distilled_guidance_layer";
        let mut layers = Vec::with_capacity(cfg.approximator_layers);
        let mut norms = Vec::with_capacity(cfg.approximator_layers);
        for i in 0..cfg.approximator_layers {
            layers.push((
                Lin::load(w, &format!("{p}.layers.{i}.linear_1"))?,
                Lin::load(w, &format!("{p}.layers.{i}.linear_2"))?,
            ));
            norms.push(w.require(&format!("{p}.norms.{i}.weight"))?.clone());
        }
        Ok(Self {
            in_proj: Lin::load(w, &format!("{p}.in_proj"))?,
            layers,
            norms,
            out_proj: Lin::load(w, &format!("{p}.out_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.in_proj.forward(x)?;
        for ((lin1, lin2), norm) in self.layers.iter().zip(self.norms.iter()) {
            let n = rms_norm(&x, norm, APPROX_RMS_EPS)?;
            let h = lin2.forward(&silu(&lin1.forward(&n)?)?)?;
            x = add(&x, &h)?;
        }
        self.out_proj.forward(&x)
    }

    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["in_proj"] => self.in_proj.inner_mut(),
            ["out_proj"] => self.out_proj.inner_mut(),
            ["layers", n, "linear_1"] => {
                self.layers.get_mut(n.parse::<usize>().ok()?)?.0.inner_mut()
            }
            ["layers", n, "linear_2"] => {
                self.layers.get_mut(n.parse::<usize>().ok()?)?.1.inner_mut()
            }
            _ => return None,
        })
    }
}

// ============================ blocks ============================

struct DoubleAttn {
    /// SC-18319 P4: the image stream's `to_q`/`to_k`/`to_v` behind one adapter/quant-aware packed
    /// matrix. All three read the SAME activation (`hidden`), which is the precondition for packing
    /// them; the text triple is a second, independent projection because it reads `encoder`.
    img_qkv: FusedQkvProjection,
    to_out: Lin,
    /// The text stream's `add_q_proj`/`add_k_proj`/`add_v_proj`, likewise packed.
    txt_qkv: FusedQkvProjection,
    to_add_out: Lin,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    heads: i32,
    head_dim: i32,
}

impl DoubleAttn {
    fn load(w: &Weights, p: &str, cfg: &ChromaTransformerConfig) -> Result<Self> {
        Ok(Self {
            img_qkv: FusedQkvProjection::new(
                Lin::load(w, &format!("{p}.to_q"))?.into_inner(),
                Lin::load(w, &format!("{p}.to_k"))?.into_inner(),
                Lin::load(w, &format!("{p}.to_v"))?.into_inner(),
            ),
            to_out: Lin::load(w, &format!("{p}.to_out.0"))?,
            txt_qkv: FusedQkvProjection::new(
                Lin::load(w, &format!("{p}.add_q_proj"))?.into_inner(),
                Lin::load(w, &format!("{p}.add_k_proj"))?.into_inner(),
                Lin::load(w, &format!("{p}.add_v_proj"))?.into_inner(),
            ),
            to_add_out: Lin::load(w, &format!("{p}.to_add_out"))?,
            norm_q: w.require(&format!("{p}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{p}.norm_k.weight"))?.clone(),
            norm_added_q: w.require(&format!("{p}.norm_added_q.weight"))?.clone(),
            norm_added_k: w.require(&format!("{p}.norm_added_k.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: cfg.attention_head_dim as i32,
        })
    }

    /// Joint attention. Returns `(image_attn [B,Si,inner], text_attn [B,St,inner])`. The concatenated
    /// sequence order is `[text, image]` (matches the mask order built in the forward).
    fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        rope: &RopeTable,
        mask: Option<&Array>,
        attention: AttentionPlan<'_>,
    ) -> Result<(Array, Array)> {
        let hd = self.head_dim;
        // SC-18319 — **knob 8's concat-then-RoPE arm**, and the reason that knob exists. Chroma joins
        // `[text, image]` FIRST (knob 11) and rotates the joint sequence with one table, where FLUX.1
        // rotates each stream and then concatenates. Both are expressed as a call-order choice over
        // the same two primitives: `prepare` with `RopeStyle::None`, then `join`, then `apply_rope`.
        let spec = chroma_spec(self.heads, hd, &self.norm_q, &self.norm_k, None);
        // SC-18319 P4 — one matmul per stream when the pack is engaged, three concatenated forwards
        // when it is not. `prepare` splits the packed result at the offsets a `Separate` source would
        // have carried, and a matmul's output rows are independent, so the two arms are bit-identical.
        let img = qkv::prepare(
            QkvSource::Packed(&self.img_qkv.forward_packed(hidden)?),
            &spec,
        )?;
        let txt_spec = chroma_spec(self.heads, hd, &self.norm_added_q, &self.norm_added_k, None);
        let txt = qkv::prepare(
            QkvSource::Packed(&self.txt_qkv.forward_packed(encoder)?),
            &txt_spec,
        )?;
        let joint = StreamOrder::TextFirst.join(&img, &txt)?;
        let q = rotate_joint(&joint.q, rope)?;
        let k = rotate_joint(&joint.k, rope)?;
        let out = sdpa(&q, &k, &joint.v, hd, mask, attention)?; // [B, S, inner]
        let st = encoder.shape()[1];
        let txt = seq_slice(&out, 0, st)?;
        let img = seq_slice(&out, st, hidden.shape()[1])?;
        Ok((self.to_out.forward(&img)?, self.to_add_out.forward(&txt)?))
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_qkv.quantize(bits, None)?;
        self.txt_qkv.quantize(bits, None)?;
        for l in [&mut self.to_out, &mut self.to_add_out] {
            l.quantize(bits)?;
        }
        Ok(())
    }

    /// Resolve a diffusers adapter sub-path (within `…attn.`) to its linear (sc-3842) — the
    /// **MUTATION** half. A q/k/v path goes through [`FusedQkvProjection::part_mut`], which unfuses
    /// first, so an adapter installed here can never be stranded behind a stale packed matrix.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["to_q"] => return self.img_qkv.part_mut(QkvPart::Q).ok(),
            ["to_k"] => return self.img_qkv.part_mut(QkvPart::K).ok(),
            ["to_v"] => return self.img_qkv.part_mut(QkvPart::V).ok(),
            ["to_out", "0"] => self.to_out.inner_mut(),
            ["add_q_proj"] => return self.txt_qkv.part_mut(QkvPart::Q).ok(),
            ["add_k_proj"] => return self.txt_qkv.part_mut(QkvPart::K).ok(),
            ["add_v_proj"] => return self.txt_qkv.part_mut(QkvPart::V).ok(),
            ["to_add_out"] => self.to_add_out.inner_mut(),
            _ => return None,
        })
    }

    /// The **PROBE** half (SC-18319): the six fused paths answer from
    /// [`FusedQkvProjection::part_facts`], reading the packed representation instead of dismantling
    /// it. `block_stream.rs`'s capture and verify walks hit EVERY path in every block, so without
    /// this a window scan would unfuse the whole stack.
    fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        match path {
            ["to_q"] => Some(self.img_qkv.part_facts(QkvPart::Q)),
            ["to_k"] => Some(self.img_qkv.part_facts(QkvPart::K)),
            ["to_v"] => Some(self.img_qkv.part_facts(QkvPart::V)),
            ["add_q_proj"] => Some(self.txt_qkv.part_facts(QkvPart::Q)),
            ["add_k_proj"] => Some(self.txt_qkv.part_facts(QkvPart::K)),
            ["add_v_proj"] => Some(self.txt_qkv.part_facts(QkvPart::V)),
            _ => self.adaptable_mut(path).map(|l| LinearFacts::of(l)),
        }
    }
}

struct FeedForward {
    lin1: Lin,
    lin2: Lin,
}

impl FeedForward {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            lin1: Lin::load(w, &format!("{p}.net.0.proj"))?,
            lin2: Lin::load(w, &format!("{p}.net.2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.lin2.forward(&gelu_tanh(&self.lin1.forward(x)?)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.lin1.quantize(bits)?;
        self.lin2.quantize(bits)
    }
}

/// The block-local adapter paths on a [`DoubleBlock`] — the enumeration the rung-4 stream captures
/// and replays over. Kept beside [`DoubleBlock::adaptable_mut`] and pinned against it in both
/// directions by `block_stream`'s `every_listed_adapter_path_resolves_and_nothing_else_does`: a path
/// silently dropped here means a streamed block loses that adapter with no error.
pub(crate) const DOUBLE_ADAPTER_PATHS: &[&str] = &[
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
];

/// The [`SingleBlock`] analogue of [`DOUBLE_ADAPTER_PATHS`].
pub(crate) const SINGLE_ADAPTER_PATHS: &[&str] = &[
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "proj_mlp",
    "proj_out",
];

/// One windowable Chroma sub-stack block, as the rung-4 stream sees it: something that can be
/// rebuilt from a snapshot view and whose adapter targets are enumerable.
///
/// A trait rather than two near-identical code paths, so the capture/replay/verify logic in
/// [`crate::block_stream`] is written once and cannot drift between the double and single stacks.
pub(crate) trait StreamBlock: Sized {
    /// The block-local dotted adapter paths this block type exposes.
    const ADAPTER_PATHS: &'static [&'static str];
    /// Rebuild block `index` from a snapshot view.
    fn from_view(view: &Weights, index: usize, cfg: &ChromaTransformerConfig) -> Result<Self>;
    /// Resolve a block-local dotted path to its adapter carrier — the **MUTATION** half.
    fn adapter_target(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear>;
    /// The **PROBE** half of the same block-local surface (SC-18319). The block stream's capture and
    /// verify walks visit every entry of [`ADAPTER_PATHS`](Self::ADAPTER_PATHS) on every window, so
    /// they must ask through here rather than through `adapter_target`, which unfuses.
    fn adapter_facts(&mut self, path: &[&str]) -> Option<LinearFacts>;
}

impl StreamBlock for DoubleBlock {
    const ADAPTER_PATHS: &'static [&'static str] = DOUBLE_ADAPTER_PATHS;

    fn from_view(view: &Weights, index: usize, cfg: &ChromaTransformerConfig) -> Result<Self> {
        Self::load(view, index, cfg)
    }

    fn adapter_target(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        self.adaptable_mut(path)
    }

    fn adapter_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        self.adaptable_facts(path)
    }
}

impl StreamBlock for SingleBlock {
    const ADAPTER_PATHS: &'static [&'static str] = SINGLE_ADAPTER_PATHS;

    fn from_view(view: &Weights, index: usize, cfg: &ChromaTransformerConfig) -> Result<Self> {
        Self::load(view, index, cfg)
    }

    fn adapter_target(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        self.adaptable_mut(path)
    }

    fn adapter_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        self.adaptable_facts(path)
    }
}

pub(crate) struct DoubleBlock {
    attn: DoubleAttn,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleBlock {
    pub(crate) fn load(w: &Weights, i: usize, cfg: &ChromaTransformerConfig) -> Result<Self> {
        let p = format!("transformer_blocks.{i}");
        Ok(Self {
            attn: DoubleAttn::load(w, &format!("{p}.attn"), cfg)?,
            ff: FeedForward::load(w, &format!("{p}.ff"))?,
            ff_context: FeedForward::load(w, &format!("{p}.ff_context"))?,
        })
    }

    /// `temb` is the 12-row modulation slice `[B,12,inner]` (`[:6]` image, `[6:]` text). Each stream's
    /// rows are `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
    fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        temb: &Array,
        rope: &RopeTable,
        mask: Option<&Array>,
        attention: AttentionPlan<'_>,
    ) -> Result<(Array, Array)> {
        let norm_hidden = modulate(
            &layer_norm(hidden, None, None, LN_EPS)?,
            &row(temb, 1)?,
            &row(temb, 0)?,
        )?;
        let norm_encoder = modulate(
            &layer_norm(encoder, None, None, LN_EPS)?,
            &row(temb, 7)?,
            &row(temb, 6)?,
        )?;

        let (attn_img, attn_txt) =
            self.attn
                .forward(&norm_hidden, &norm_encoder, rope, mask, attention)?;

        // image stream.
        let hidden = gated(hidden, &row(temb, 2)?, &attn_img)?;
        let nh = modulate(
            &layer_norm(&hidden, None, None, LN_EPS)?,
            &row(temb, 4)?,
            &row(temb, 3)?,
        )?;
        let hidden = gated(&hidden, &row(temb, 5)?, &self.ff.forward(&nh)?)?;

        // text stream.
        let encoder = gated(encoder, &row(temb, 8)?, &attn_txt)?;
        let ne = modulate(
            &layer_norm(&encoder, None, None, LN_EPS)?,
            &row(temb, 10)?,
            &row(temb, 9)?,
        )?;
        let encoder = gated(&encoder, &row(temb, 11)?, &self.ff_context.forward(&ne)?)?;

        Ok((encoder, hidden))
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        self.ff_context.quantize(bits)
    }

    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["attn", rest @ ..] => return self.attn.adaptable_mut(rest),
            ["ff", "net", "0", "proj"] => self.ff.lin1.inner_mut(),
            ["ff", "net", "2"] => self.ff.lin2.inner_mut(),
            ["ff_context", "net", "0", "proj"] => self.ff_context.lin1.inner_mut(),
            ["ff_context", "net", "2"] => self.ff_context.lin2.inner_mut(),
            _ => return None,
        })
    }

    /// SC-18319 — an intermediate hop to a fused leaf MUST forward the probe, or the `adaptable_mut`
    /// fallback takes over here and unfuses the attention below it.
    pub(crate) fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_facts(rest),
            _ => self.adaptable_mut(path).map(|l| LinearFacts::of(l)),
        }
    }
}

struct SingleAttn {
    /// SC-18319 P4: one packed q/k/v. The single-stream block is a true self-attention, so all three
    /// read the same `x`.
    qkv: FusedQkvProjection,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
}

impl SingleAttn {
    fn load(w: &Weights, p: &str, cfg: &ChromaTransformerConfig) -> Result<Self> {
        Ok(Self {
            qkv: FusedQkvProjection::new(
                Lin::load(w, &format!("{p}.to_q"))?.into_inner(),
                Lin::load(w, &format!("{p}.to_k"))?.into_inner(),
                Lin::load(w, &format!("{p}.to_v"))?.into_inner(),
            ),
            norm_q: w.require(&format!("{p}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{p}.norm_k.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: cfg.attention_head_dim as i32,
        })
    }

    fn forward(
        &self,
        x: &Array,
        rope: &RopeTable,
        mask: Option<&Array>,
        attention: AttentionPlan<'_>,
    ) -> Result<Array> {
        // SC-18319 — the shared prologue. Chroma's knob selection: separate q/k/v (knob 9), per-head
        // QK-RMSNorm computed in f32 with the whole stream (including `v`) promoted, adjacent-pair
        // rotation (knob 2) applied head-major, and a shared q/k table (knob 6 off).
        let heads = qkv::prepare(
            QkvSource::Packed(&self.qkv.forward_packed(x)?),
            &chroma_spec(
                self.heads,
                self.head_dim,
                &self.norm_q,
                &self.norm_k,
                Some(rope),
            ),
        )?;
        sdpa(&heads.q, &heads.k, &heads.v, self.head_dim, mask, attention)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.qkv.quantize(bits, None)
    }

    /// The MUTATION half — see [`DoubleAttn::adaptable_mut`].
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => self.qkv.part_mut(QkvPart::Q).ok(),
            ["to_k"] => self.qkv.part_mut(QkvPart::K).ok(),
            ["to_v"] => self.qkv.part_mut(QkvPart::V).ok(),
            _ => None,
        }
    }

    /// The PROBE half — see [`DoubleAttn::adaptable_facts`].
    fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        match path {
            ["to_q"] => Some(self.qkv.part_facts(QkvPart::Q)),
            ["to_k"] => Some(self.qkv.part_facts(QkvPart::K)),
            ["to_v"] => Some(self.qkv.part_facts(QkvPart::V)),
            _ => None,
        }
    }
}

pub(crate) struct SingleBlock {
    attn: SingleAttn,
    proj_mlp: Lin,
    proj_out: Lin,
}

impl SingleBlock {
    pub(crate) fn load(w: &Weights, i: usize, cfg: &ChromaTransformerConfig) -> Result<Self> {
        let p = format!("single_transformer_blocks.{i}");
        Ok(Self {
            attn: SingleAttn::load(w, &format!("{p}.attn"), cfg)?,
            proj_mlp: Lin::load(w, &format!("{p}.proj_mlp"))?,
            proj_out: Lin::load(w, &format!("{p}.proj_out"))?,
        })
    }

    /// `temb` is the 3-row modulation slice `[B,3,inner]` (shift, scale, gate). `hidden` is the joint
    /// `[text|image]` stream.
    fn forward(
        &self,
        hidden: &Array,
        temb: &Array,
        rope: &RopeTable,
        mask: Option<&Array>,
        attention: AttentionPlan<'_>,
    ) -> Result<Array> {
        let norm_hidden = modulate(
            &layer_norm(hidden, None, None, LN_EPS)?,
            &row(temb, 1)?,
            &row(temb, 0)?,
        )?;
        let mlp = gelu_tanh(&self.proj_mlp.forward(&norm_hidden)?)?;
        let attn = self.attn.forward(&norm_hidden, rope, mask, attention)?;
        let proj = self
            .proj_out
            .forward(&concatenate_axis(&[&attn, &mlp], 2)?)?;
        gated(hidden, &row(temb, 2)?, &proj)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.proj_mlp.quantize(bits)?;
        self.proj_out.quantize(bits)
    }

    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        Some(match path {
            ["attn", rest @ ..] => return self.attn.adaptable_mut(rest),
            ["proj_mlp"] => self.proj_mlp.inner_mut(),
            ["proj_out"] => self.proj_out.inner_mut(),
            _ => return None,
        })
    }

    /// SC-18319 — see [`DoubleBlock::adaptable_facts`].
    pub(crate) fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_facts(rest),
            _ => self.adaptable_mut(path).map(|l| LinearFacts::of(l)),
        }
    }
}

// ============================ the transformer ============================

pub struct ChromaTransformer {
    pub cfg: ChromaTransformerConfig,
    x_embedder: Lin,
    context_embedder: Lin,
    time_text_embed: TimestepTextProj,
    approximator: Approximator,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    proj_out: Lin,
    /// Ladder rung 4 (SC-15520): the reopenable snapshot description a windowed forward rebuilds
    /// blocks from. `None` on every ordinary load, which keeps the resident path byte-for-byte
    /// unchanged.
    block_stream: Option<crate::block_stream::ChromaBlockStream>,
}

impl ChromaTransformer {
    /// Load from a diffusers `transformer/` weight map. Validates the Chroma key surface + the
    /// pruned-adaLN invariant, then materializes the typed modules.
    pub fn from_weights(w: Weights, cfg: ChromaTransformerConfig) -> Result<Self> {
        // Pruned-adaLN invariant: Chroma blocks have NO `.norm*.linear` weights.
        if let Some(k) = w
            .keys()
            .find(|k| k.contains(".norm1.linear") || k.contains(".norm.linear"))
        {
            return Err(Error::Msg(format!(
                "chroma transformer: unexpected per-block modulation linear {k:?} — Chroma uses \
                 pruned adaLN (modulation comes from distilled_guidance_layer)"
            )));
        }

        let n_double = (0..)
            .take_while(|i| {
                w.get(&format!("transformer_blocks.{i}.attn.to_q.weight"))
                    .is_some()
            })
            .count();
        let n_single = (0..)
            .take_while(|i| {
                w.get(&format!("single_transformer_blocks.{i}.proj_out.weight"))
                    .is_some()
            })
            .count();
        if n_double != cfg.num_layers || n_single != cfg.num_single_layers {
            return Err(Error::Msg(format!(
                "chroma transformer: block counts {n_double} double / {n_single} single != config \
                 {} / {}",
                cfg.num_layers, cfg.num_single_layers
            )));
        }

        let double_blocks = (0..cfg.num_layers)
            .map(|i| DoubleBlock::load(&w, i, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..cfg.num_single_layers)
            .map(|i| SingleBlock::load(&w, i, &cfg))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            x_embedder: Lin::load(&w, "x_embedder")?,
            context_embedder: Lin::load(&w, "context_embedder")?,
            time_text_embed: TimestepTextProj::new(&cfg)?,
            approximator: Approximator::load(&w, &cfg)?,
            double_blocks,
            single_blocks,
            proj_out: Lin::load(&w, "proj_out")?,
            cfg,
            block_stream: None,
        })
    }

    /// Arm exact snapshot-backed reconstruction for both Chroma block stacks (rung 4).
    ///
    /// The caller must run [`Self::finalize_block_stream`] **after** every load-time transformation
    /// (quantization, adapters) has completed, so the stream captures the state the resident blocks
    /// actually ended up in rather than a second derivation of it.
    pub(crate) fn with_block_stream(mut self, source: mlx_gen::WeightsSource) -> Self {
        self.block_stream = Some(crate::block_stream::ChromaBlockStream::new(
            source, self.cfg,
        ));
        self
    }

    /// Capture the resident stacks' adapters into the armed stream, then evict both stacks. The
    /// embedders, Approximator, RoPE and `proj_out` remain resident.
    pub(crate) fn finalize_block_stream(&mut self) -> Result<()> {
        let Some(stream) = self.block_stream.as_mut() else {
            return Ok(());
        };
        stream.capture_adapters(&mut self.double_blocks, &mut self.single_blocks);
        let (double, single) = (stream.double_blocks(), stream.single_blocks());
        crate::block_stream::evict_resident_blocks(
            &mut self.double_blocks,
            &mut self.single_blocks,
            double,
            single,
        )
    }

    /// The resident block counts — `(0, 0)` once a stream has been finalized.
    #[doc(hidden)]
    pub fn resident_block_counts(&self) -> (usize, usize) {
        (self.double_blocks.len(), self.single_blocks.len())
    }

    /// Whether this transformer can rebuild its blocks on demand.
    #[doc(hidden)]
    pub fn is_streamable(&self) -> bool {
        self.block_stream.is_some()
    }

    /// Build the window plans for one denoise step, or `None` when rung 4 is unselected.
    pub(crate) fn block_window<'a>(
        &self,
        size: Option<usize>,
        cancel: &'a mlx_gen::CancelFlag,
    ) -> Result<Option<crate::block_stream::ChromaBlockWindow<'a>>> {
        let Some(size) = size else { return Ok(None) };
        let stream = self.block_stream.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "chroma: bounded transformer residency needs a snapshot-backed block stream — load \
                 with a staged request and LoadShape::DeferredMaterialization on a clean base route"
                    .to_owned(),
            )
        })?;
        Ok(Some(crate::block_stream::ChromaBlockWindow::new(
            stream.double_blocks(),
            stream.single_blocks(),
            size,
            cancel,
        )?))
    }

    /// Quantize the matmul-heavy block linears (double/single attention + FFN) to Q4/Q8 (sc-3841).
    /// The small/sensitive modules — `x_embedder`/`context_embedder`/`proj_out` and the
    /// distilled-guidance Approximator (which drives all modulation) — stay dense, mirroring the
    /// "quantize the big GEMMs" convention. T5/VAE are quantized separately by the loader (if at all).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        // Quantizing AFTER the block stream is armed would pack nothing: both stacks are evicted, so
        // the loops below are no-ops and every streamed block would be rebuilt dense from a snapshot
        // the caller believes it quantized. The production order (quantize, adapt, then arm) is
        // correct; this refuses the inverted one rather than documenting it (SC-15520).
        if self.block_stream.is_some() && self.double_blocks.is_empty() {
            return Err(Error::Unsupported(
                "chroma: cannot quantize after the block stream is armed — both stacks are evicted, \
                 so the block linears would be silently skipped. Quantize first, then arm the stream"
                    .to_owned(),
            ));
        }
        for b in &mut self.double_blocks {
            b.quantize(bits)?;
        }
        for b in &mut self.single_blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// `pooled_temb [B, mod_index_len, inner]` for a **raw** (unscaled) timestep `[B]`.
    pub fn pooled_temb(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            mlx_gen::array::scalar(1000.0),
        )?;
        self.approximator
            .forward(&self.time_text_embed.forward(&scaled)?)
    }

    /// The Chroma DiT forward.
    ///
    /// - `hidden [B, Si, in_channels]` — packed image latent tokens.
    /// - `encoder [B, St, joint_attention_dim]` — T5 prompt embeddings.
    /// - `timestep [B]` — raw denoise timestep (scaled `*1000` internally).
    /// - `img_ids [Si,3]` / `txt_ids [St,3]` — RoPE position ids.
    /// - `attention_mask [B, St+Si]` (0/1) or `None` — the **full-sequence** MMDiT mask in `[text,
    ///   image]` order. The 0/1 mask is added to the attention scores (the reference's behavior). The
    ///   mask that *builds* this from the T5 padding is sc-3838.
    ///
    /// Returns the predicted velocity `[B, Si, out_channels]`.
    ///
    /// Convenience wrapper that builds the step-invariant tensors (`pooled_temb`, the RoPE table, the
    /// `[B,1,S,S]` mask) and calls `forward_prepared`. The denoise loop prefers the prepared
    /// form so those tensors are computed once per step / per branch rather than per call (F-102).
    pub fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        timestep: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array> {
        let pooled = self.pooled_temb(timestep)?;
        let rope = self.build_rope_table(txt_ids, img_ids)?;
        let mask2d = Self::attention_mask2d(attention_mask)?;
        self.forward_prepared(
            hidden,
            encoder,
            &pooled,
            &rope,
            mask2d.as_ref(),
            AttentionPlan::UNBOUNDED,
            None,
        )
    }

    /// The RoPE table over `cat(txt_ids, img_ids)` — depends only on the token positions, so the
    /// denoise loop builds it once per branch instead of every step (F-102).
    pub(crate) fn build_rope_table(&self, txt_ids: &Array, img_ids: &Array) -> Result<RopeTable> {
        let ids = concatenate_axis(&[txt_ids, img_ids], 0)?;
        build_rope(&ids, self.cfg.axes_dims_rope)
    }

    /// `[B,S]` 0/1 mask → additive `[B,1,S,S] = m[:,None,None,:]·m[:,None,:,None]`. Depends only on the
    /// per-request padding, so the denoise loop builds it once per branch (F-102).
    pub(crate) fn attention_mask2d(attention_mask: Option<&Array>) -> Result<Option<Array>> {
        match attention_mask {
            Some(m) => {
                let m = m.as_dtype(Dtype::Float32)?;
                let b = m.shape()[0];
                let s = m.shape()[1];
                let a = m.reshape(&[b, 1, 1, s])?;
                let bt = m.reshape(&[b, 1, s, 1])?;
                Ok(Some(multiply(&a, &bt)?))
            }
            None => Ok(None),
        }
    }

    /// Run the MMDiT given the pre-built step-invariant tensors: `pooled` (the Approximator modulation
    /// table — shared by both CFG branches at a step), `rope`, and the additive `mask2d`. `hidden`
    /// (latents) and `encoder` (text) are per-branch. Bit-identical to [`Self::forward`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_prepared(
        &self,
        hidden: &Array,
        encoder: &Array,
        pooled: &Array,
        rope: &RopeTable,
        mask_ref: Option<&Array>,
        attention: AttentionPlan<'_>,
        window: Option<crate::block_stream::ChromaBlockWindow<'_>>,
    ) -> Result<Array> {
        let hidden = self.x_embedder.forward(hidden)?;
        let encoder = self.context_embedder.forward(encoder)?;

        let st = encoder.shape()[1];
        let n_single = self.cfg.num_single_layers as i32;
        let img_offset = 3 * n_single;
        let txt_offset = img_offset + 6 * self.cfg.num_layers as i32;
        // The modulation slice is derived from the block INDEX, so a windowed rewrite has to carry
        // `i` through the window range rather than from an enumeration of a resident vector.
        let double_temb = |i: usize| -> Result<Array> {
            let i = i as i32;
            let img = rows(pooled, img_offset + 6 * i, 6)?;
            let txt = rows(pooled, txt_offset + 6 * i, 6)?;
            concatenate_axis(&[&img, &txt], 1) // [B,12,inner]
                .map_err(Into::into)
        };

        let mut hidden = hidden;
        let mut encoder = encoder;
        match window {
            None => {
                if self.block_stream.is_some() && self.double_blocks.is_empty() {
                    return Err(Error::Unsupported(
                        "chroma: a deferred transformer requires an explicit block window"
                            .to_owned(),
                    ));
                }
                for (i, block) in self.double_blocks.iter().enumerate() {
                    let temb = double_temb(i)?;
                    let (e, h) =
                        block.forward(&hidden, &encoder, &temb, rope, mask_ref, attention)?;
                    encoder = e;
                    hidden = h;
                }
            }
            Some(window) => {
                let source = self.stream_source()?;
                if window.double.n_blocks() != source.double_blocks() {
                    return Err(Error::Msg(
                        "chroma: double block plan depth mismatch".to_owned(),
                    ));
                }
                let (e, h) = mlx_gen::block_residency::run_windowed(
                    &window.double,
                    window.cancel,
                    (encoder, hidden),
                    || source.open(),
                    |(mut encoder, mut hidden), view, range| {
                        for i in range {
                            let block = source.materialize_double(view, i)?;
                            let temb = double_temb(i)?;
                            (encoder, hidden) = block
                                .forward(&hidden, &encoder, &temb, rope, mask_ref, attention)
                                .map_err(|error| {
                                    Error::Msg(format!(
                                        "chroma block stream: double block {i} forward: {error}"
                                    ))
                                })?;
                        }
                        Ok((encoder, hidden))
                    },
                    // LOAD-BEARING: MLX is lazy, so the carried activation is an unevaluated graph
                    // node still referencing the window's weights. Dropping before forcing
                    // evaluation frees nothing and the bound silently does not hold.
                    |(encoder, hidden)| {
                        mlx_rs::transforms::eval([encoder, hidden]).map_err(Into::into)
                    },
                )?;
                encoder = e;
                hidden = h;
            }
        }

        let mut joint = concatenate_axis(&[&encoder, &hidden], 1)?; // [B, S, inner]
        match window {
            None => {
                for (i, block) in self.single_blocks.iter().enumerate() {
                    let temb = rows(pooled, 3 * i as i32, 3)?;
                    joint = block.forward(&joint, &temb, rope, mask_ref, attention)?;
                }
            }
            Some(window) => {
                let source = self.stream_source()?;
                if window.single.n_blocks() != source.single_blocks() {
                    return Err(Error::Msg(
                        "chroma: single block plan depth mismatch".to_owned(),
                    ));
                }
                joint = mlx_gen::block_residency::run_windowed(
                    &window.single,
                    window.cancel,
                    joint,
                    || source.open(),
                    |mut joint, view, range| {
                        for i in range {
                            let block = source.materialize_single(view, i)?;
                            let temb = rows(pooled, 3 * i as i32, 3)?;
                            joint = block
                                .forward(&joint, &temb, rope, mask_ref, attention)
                                .map_err(|error| {
                                    Error::Msg(format!(
                                        "chroma block stream: single block {i} forward: {error}"
                                    ))
                                })?;
                        }
                        Ok(joint)
                    },
                    |joint: &Array| mlx_rs::transforms::eval([joint]).map_err(Into::into),
                )?;
            }
        }

        // Drop the text tokens; pruned `norm_out` (shift, scale = pooled[-2:]); proj_out.
        let hidden = seq_slice(&joint, st, joint.shape()[1] - st)?;
        let n = self.cfg.mod_index_len() as i32;
        let no = rows(pooled, n - 2, 2)?;
        let hidden = modulate(
            &layer_norm(&hidden, None, None, LN_EPS)?,
            &row(&no, 1)?,
            &row(&no, 0)?,
        )?;
        self.proj_out.forward(&hidden)
    }

    fn stream_source(&self) -> Result<&crate::block_stream::ChromaBlockStream> {
        if !self.double_blocks.is_empty() || !self.single_blocks.is_empty() {
            return Err(Error::Msg(
                "chroma: a windowed forward ran against a transformer that still holds resident \
                 blocks — the bound would not hold"
                    .to_owned(),
            ));
        }
        self.block_stream
            .as_ref()
            .ok_or_else(|| Error::Unsupported("chroma: no snapshot-backed block stream".to_owned()))
    }

    /// Evidence hook (SC-15520): [`Self::forward`] under an explicit rung-3 attention plan.
    ///
    /// `mlx_gen_chroma::memory_strategy::ATTENTION_SUPPORT` is `false`, so the production path
    /// refuses every bounded-attention request — which is exactly why the *mechanism* needs a seam a
    /// harness can still reach. Without one, the measurement behind that `Missing` verdict could
    /// never be re-taken and the verdict would calcify into an unfalsifiable comment.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_attention_plan(
        &self,
        hidden: &Array,
        encoder: &Array,
        timestep: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        attention_mask: Option<&Array>,
        attention: AttentionPlan<'_>,
    ) -> Result<Array> {
        let pooled = self.pooled_temb(timestep)?;
        let rope = self.build_rope_table(txt_ids, img_ids)?;
        let mask2d = Self::attention_mask2d(attention_mask)?;
        self.forward_prepared(
            hidden,
            encoder,
            &pooled,
            &rope,
            mask2d.as_ref(),
            attention,
            None,
        )
    }

    /// Test hook: the Approximator input vector for a raw timestep `[B]` (pure elementwise — isolates
    /// the embedding build from the matmul floor).
    #[doc(hidden)]
    pub fn input_vec_for_tests(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            mlx_gen::array::scalar(1000.0),
        )?;
        self.time_text_embed.forward(&scaled)
    }
}

impl AdaptableHost for ChromaTransformer {
    /// Resolve a trained-file (diffusers/peft) dotted adapter path to its [`AdaptableLinear`].
    /// Covers the double/single block attention + FFN linears, the global embedders/`proj_out`, and
    /// the distilled-guidance Approximator (some community Chroma LoRAs train it).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", n, rest @ ..] => self
                .double_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["single_transformer_blocks", n, rest @ ..] => self
                .single_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["x_embedder"] => Some(self.x_embedder.inner_mut()),
            ["context_embedder"] => Some(self.context_embedder.inner_mut()),
            ["proj_out"] => Some(self.proj_out.inner_mut()),
            ["distilled_guidance_layer", rest @ ..] => self.approximator.adaptable_mut(rest),
            _ => None,
        }
    }

    /// SC-18319 — forward the probe down to the two block stacks; see
    /// the `DoubleBlock` twin below. Anything else falls through to the `adaptable_mut`
    /// delegation, which is what the trait default does and is correct for a plain linear.
    fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        match path {
            ["transformer_blocks", n, rest @ ..] => self
                .double_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_facts(rest),
            ["single_transformer_blocks", n, rest @ ..] => self
                .single_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_facts(rest),
            _ => self.adaptable_mut(path).map(|l| LinearFacts::of(l)),
        }
    }

    /// kohya `lora_unet_`-reachable targets: the block-indexed attention + FFN linears in trained-file
    /// naming. Globals (`x_embedder`/`context_embedder`/`proj_out`/`distilled_guidance_layer`) are
    /// excluded — they stay reachable via the dotted peft form (every path here must resolve via
    /// [`adaptable_mut`](Self::adaptable_mut); guarded by `tests/adapter_routing.rs`).
    fn adaptable_paths(&self) -> Vec<String> {
        const DOUBLE: [&str; 12] = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ];
        const SINGLE: [&str; 5] = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "proj_mlp",
            "proj_out",
        ];
        let mut out = Vec::new();
        for i in 0..self.double_blocks.len() {
            for leaf in DOUBLE {
                out.push(format!("transformer_blocks.{i}.{leaf}"));
            }
        }
        for i in 0..self.single_blocks.len() {
            for leaf in SINGLE {
                out.push(format!("single_transformer_blocks.{i}.{leaf}"));
            }
        }
        out
    }
}
