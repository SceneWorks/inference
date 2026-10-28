//! Shared Qwen3-style transformer blocks (sc-13334; hoisted into the audio commons by
//! sc-27270 / sc-27413).
//!
//! Several audio providers run Qwen3 decoder stacks with identical block math — GQA with per-head
//! q/k RMSNorm, half-split (NeoX) RoPE at `rope_theta`, and a SiLU gated MLP. MOSS-TTS-Realtime
//! uses it for both its backbone (`config.json.language_config`) and its local/depth transformer
//! (`config.json.local_config`); MOSS-TTSD's backbone (`language_model.layers`) is the identical
//! Qwen3 layer weight-for-weight. This module is the one verified implementation both MOSS crates
//! consume (each re-exports it as its `blocks` module, so the public paths are unchanged), and
//! [`TextEncoder`] — the stateless full-sequence stack — is the MOSS-SFX and ACE-Step Qwen3 text
//! encoder (sc-27270).
//!
//! Two attention forms share the same weights and math:
//! - [`Layer::forward`] — the **stateless full-sequence** attention: every position is recomputed
//!   from scratch. MOSS-TTS-Realtime's local/depth transformer uses this (its depth axis is short
//!   and re-embedded each frame), and it is the parity reference for the cached form.
//! - [`Layer::forward_cached`] — the **KV-cache** attention (sc-13417): keys/values for prior
//!   positions are read from a per-layer [`LayerKv`] slot and only the new position(s) are computed,
//!   turning the backbone AR step from O(seq) to O(1) amortized. Prefill runs the whole prompt with
//!   a causal mask (empty cache); each subsequent single-token step appends one position and attends
//!   maskless over the cache. This is a **pure latency optimization**: the cached path is
//!   byte-identical to the stateless recompute (same projections/norms/RoPE per position, same
//!   per-`(query, key)` dot products), proven by the MOSS backbones' parity tests.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use crate::ops::repeat_kv;
use candle_nn::{embedding, linear_b, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};

/// The per-block hyperparameters common to a Qwen3 decoder layer.
#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub attention_bias: bool,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &BlockConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.head_dim;
        Ok(Self {
            q_proj: linear_b(
                cfg.hidden_size,
                cfg.num_attention_heads * d,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * d,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * d,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(
                cfg.num_attention_heads * d,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            q_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: d,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = __model_proof_output!(self.q_proj.forward(x))?
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = __model_proof_output!(self.k_proj.forward(x))?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = __model_proof_output!(self.v_proj.forward(x))?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head q/k RMSNorm (over head_dim), then half-split RoPE — the HF Qwen3 order.
        let q = __model_proof_output!(self.q_norm.forward(&q.contiguous()?))?;
        let k = __model_proof_output!(self.k_norm.forward(&k.contiguous()?))?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;

        // GQA: expand kv heads to the query-head count.
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            att = att.broadcast_add(m)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((
            b,
            l,
            self.num_heads * self.head_dim,
        ))?;
        __model_proof_output!(self.o_proj.forward(&out))
    }

    /// KV-cache attention: identical math to [`Attention::forward`], but the keys/values of the
    /// positions preceding `x` are read from `slot` instead of being recomputed. `x` is this step's
    /// row(s) `[b, l, hidden]`; the trailing `new_positions` of them are genuinely new sequence
    /// positions to append to the cache (prefill: `new_positions == l`, `mask` = causal; step:
    /// `new_positions == 1`, `mask` = `None`). `cos`/`sin` must be the RoPE tables for **`x`'s
    /// absolute positions**, so a step at position `p` is roped exactly as position `p` in the full
    /// sequence.
    ///
    /// **The M=1 gemv trap (sc-13417).** Candle's CPU matmul takes a distinct gemv path when the
    /// left operand has a single row (`M == 1`) whose accumulation order — and therefore rounding —
    /// differs from the M ≥ 2 gemm path by ~1e-7; the recompute reference always runs the whole
    /// sequence (`M == prompt_len ≥ 2`). To stay **byte-identical** rather than merely close, a
    /// single-token step is driven with `l == 2` **duplicate** rows (see the MOSS backbones' `run_cached`):
    /// every matmul then takes the gemm path, which is bit-for-bit invariant to the row count, so the
    /// real (last) row equals the recompute's last row exactly. Only the last `new_positions` row is
    /// appended to the cache; the duplicate scratch row is dropped (matmul/softmax rows are
    /// independent, so it never perturbs the real row). With those inputs matched, `q·kᵀ`, softmax
    /// and `att·v` are per-`(query, key)` reductions whose ordering is independent of the row count.
    fn forward_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        slot: &mut LayerKv,
        new_positions: usize,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = __model_proof_output!(self.q_proj.forward(x))?
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = __model_proof_output!(self.k_proj.forward(x))?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = __model_proof_output!(self.v_proj.forward(x))?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head q/k RMSNorm then half-split RoPE at x's absolute positions — the HF Qwen3 order.
        let q = __model_proof_output!(self.q_norm.forward(&q.contiguous()?))?;
        let k = __model_proof_output!(self.k_norm.forward(&k.contiguous()?))?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;

        // Append only the trailing `new_positions` (already-RoPE'd) keys + raw values to the cache —
        // for a duplicated single-token step that is the last row; the leading scratch row(s) are not
        // real positions. Then attend over the full grown sequence. `k`/`v` here are the
        // pre-GQA-expansion [b, kv_heads, seq, head_dim] tensors.
        let k_new = k.narrow(2, l - new_positions, new_positions)?;
        let v_new = v.narrow(2, l - new_positions, new_positions)?;
        let (k, v) = slot.append(&k_new, &v_new)?;

        // GQA: expand kv heads to the query-head count.
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            att = att.broadcast_add(m)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((
            b,
            l,
            self.num_heads * self.head_dim,
        ))?;
        __model_proof_output!(self.o_proj.forward(&out))
    }
}

/// One decoder layer's KV-cache slot: the already-RoPE'd keys and raw values for every position seen
/// so far, laid out `[batch, kv_heads, seq, head_dim]` (the sequence axis grows each step). Empty
/// until the first append. Mirrors `candle-llm`'s `ContiguousKvCache` idiom —
/// keys stored post-RoPE, values raw, growing-concat along the sequence axis.
#[derive(Debug, Clone, Default)]
pub struct LayerKv {
    kv: Option<(Tensor, Tensor)>,
}

impl LayerKv {
    /// Append `keys`/`values` (each `[batch, kv_heads, step, head_dim]`) to this layer's cache and
    /// return the full cached `(keys, values)` to attend over, with the sequence axis grown.
    fn append(&mut self, keys: &Tensor, values: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let merged = match self.kv.take() {
            Some((pk, pv)) => (
                Tensor::cat(&[&pk, keys], 2)?,
                Tensor::cat(&[&pv, values], 2)?,
            ),
            None => (keys.clone(), values.clone()),
        };
        self.kv = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &BlockConfig, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            gate_proj: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_b(
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = __model_proof_output!(self.gate_proj.forward(x))?.silu()?;
        let up = __model_proof_output!(self.up_proj.forward(x))?;
        __model_proof_output!(self.down_proj.forward(&(gate * up)?))
    }
}

/// One Qwen3 decoder layer (pre-norm attention + pre-norm SiLU MLP, residual).
pub struct Layer {
    input_layernorm: RmsNorm,
    attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    pub fn new(cfg: &BlockConfig, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let h = self.attn.forward(
            &__model_proof_output!(self.input_layernorm.forward(x))?,
            cos,
            sin,
            mask,
        )?;
        let x = (x + h)?;
        let h = self.mlp.forward(&__model_proof_output!(self
            .post_attention_layernorm
            .forward(&x))?)?;
        x + h
    }

    /// KV-cache decoder layer: same pre-norm attention + pre-norm MLP residual math as
    /// [`Layer::forward`], but the attention reads prior keys/values from `slot` (the cache-bearing
    /// attention) instead of recomputing the prefix. Byte-identical to [`Layer::forward`] over the
    /// same positions.
    pub fn forward_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        slot: &mut LayerKv,
        new_positions: usize,
    ) -> CandleResult<Tensor> {
        let h = self.attn.forward_cached(
            &__model_proof_output!(self.input_layernorm.forward(x))?,
            cos,
            sin,
            mask,
            slot,
            new_positions,
        )?;
        let x = (x + h)?;
        let h = self.mlp.forward(&__model_proof_output!(self
            .post_attention_layernorm
            .forward(&x))?)?;
        x + h
    }
}

/// A stateless Qwen3 stack used as a **text encoder**: token embedding → `num_layers` [`Layer`]s
/// (full-sequence causal attention, no KV cache) → final RMSNorm.
///
/// Both MOSS-SFX (Qwen3-1.7B) and ACE-Step (Qwen3-Embedding-0.6B) feed the post-norm hidden
/// states to their DiTs and carried line-for-line copies of this stack (sc-27270). It is purpose
/// built rather than `candle_transformers::models::qwen3`: that generation stack's per-layer KV
/// caches accumulate across calls with no public reset on the hidden-state path, so a second
/// prompt would attend over the first prompt's keys. Encoding wants a stateless forward.
///
/// `vb` is the stack root holding `embed_tokens` / `layers.N.*` / `norm` (the bare `Qwen3Model`
/// layout; pass `vb.pp("model")` for a causal-LM checkpoint).
pub struct TextEncoder {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope_theta: f64,
    head_dim: usize,
    hidden_size: usize,
    device: Device,
}

impl TextEncoder {
    pub fn new(
        cfg: &BlockConfig,
        vocab_size: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let embed_tokens = embedding(vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let vb_l = vb.pp("layers");
        let layers = (0..num_layers)
            .map(|i| Layer::new(cfg, vb_l.pp(i)))
            .collect::<CandleResult<Vec<_>>>()?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope_theta: cfg.rope_theta,
            head_dim: cfg.head_dim,
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
        })
    }

    /// The hidden width of every returned state.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The embedding-layer token lookup only → `[1, len, hidden]`.
    pub fn embed(&self, ids: &[u32]) -> CandleResult<Tensor> {
        let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)?;
        __model_proof_output!(self.embed_tokens.forward(&input))
    }

    /// The full causal stack + final RMSNorm → `[1, len, hidden]` post-norm hidden states (the
    /// reference `hidden_states[-1]`).
    pub fn encode(&self, ids: &[u32]) -> CandleResult<Tensor> {
        let len = ids.len();
        let mut h = self.embed(ids)?;
        let (cos, sin) = rope_tables(&self.device, len, self.head_dim, self.rope_theta)?;
        let mask = if len > 1 {
            Some(causal_mask(&self.device, len, h.dtype())?)
        } else {
            None
        };
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, mask.as_ref())?;
        }
        __model_proof_output!(self.norm.forward(&h))
    }
}

// The half-split RoPE tables live in the audio commons' `ops` (sc-27260); re-exported here so the
// MOSS crates' `blocks::rope_tables*` paths are unchanged.
pub use crate::ops::{rope_tables, rope_tables_at};

/// A `[1, 1, len, len]` additive causal mask (`0` on/below the diagonal, `-inf` above).
pub fn causal_mask(device: &Device, len: usize, dtype: DType) -> CandleResult<Tensor> {
    prefix_causal_mask(device, 0, len, dtype)
}

/// The attention mask for prefilling `t` new positions onto a cache that already holds `offset`
/// prior positions (multi-turn warm-cache prefill, sc-14151) — shape `[1, 1, t, offset + t]`.
///
/// Row `i` (the new position at absolute index `offset + i`) may attend to: **every** prior cached
/// position (columns `0..offset`, all visible), and the new positions `offset..=offset + i` (causal
/// among the new block); positions `> offset + i` are masked with `-inf`. This is the reference's
/// `torch.cat([cached_mask, new_prefill_mask], dim=-1)` over the retained KV — so turn *N* attends
/// over turns `1..N-1` while staying causal within its own block. With `offset == 0` it is exactly
/// [`causal_mask`], so the single-turn / turn-0 prefill is byte-identical to before.
pub fn prefix_causal_mask(
    device: &Device,
    offset: usize,
    t: usize,
    dtype: DType,
) -> CandleResult<Tensor> {
    let cols = offset + t;
    let data: Vec<f32> = (0..t)
        .flat_map(|i| {
            (0..cols).map(move |j| {
                // Prior cached columns (j < offset) are always visible; among the new block a query
                // at absolute `offset + i` sees keys at absolute `j <= offset + i`.
                if j <= offset + i {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(data, (1, 1, t, cols), device)?.to_dtype(dtype)
}
