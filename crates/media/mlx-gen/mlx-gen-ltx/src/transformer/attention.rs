//! Attention, the feed-forward and the adaLN table helpers shared by the video and AV blocks.

use super::*;

thread_local! {
    /// Cache of the weightless-RMSNorm unit weight per `(dim, dtype)`. An all-ones `(dim,)` array is
    /// a constant, so building it once and cloning the cheap handle avoids the ~400 alloc+fill
    /// kernels a fresh `ones` per call cost every DiT forward, every denoise step (F-057; mirrors
    /// `Connector`'s cached `ones`). Keyed by dtype since the block runs f32 or bf16 activations.
    static ONES_CACHE: RefCell<HashMap<(i32, Dtype), Array>> = RefCell::new(HashMap::new());
}

/// `mx.fast.rms_norm(x, ones, eps)` — the block's weightless pre-norm (feature RMS over the last
/// axis). The all-ones weight is cached per `(dim, dtype)` (F-057), bit-identical to a fresh `ones`.
pub(super) fn rms_norm_noweight(x: &Array, eps: f32) -> Result<Array> {
    let dim = *x
        .shape()
        .last()
        .ok_or_else(|| Error::Msg("ltx rms_norm: zero-rank input has no last axis".into()))?;
    let dt = x.dtype();
    let ones = ONES_CACHE.with(|c| -> Result<Array> {
        if let Some(o) = c.borrow().get(&(dim, dt)) {
            return Ok(o.clone());
        }
        let o = Array::ones::<f32>(&[dim])?.as_dtype(dt)?;
        c.borrow_mut().insert((dim, dt), o.clone());
        Ok(o)
    })?;
    Ok(fast_rms_norm(x, &ones, eps)?)
}

/// Multi-head attention with q/k RMSNorm, optional SPLIT RoPE, optional per-head gating. Self-attn
/// when `context` is `None`; cross-attn otherwise. RoPE `(cos, sin)` applies to q **and** k (self-attn
/// only; cross-attn passes `pe = None`).
#[derive(Clone)]
pub(super) struct Attention {
    pub(super) to_q: Linear,
    pub(super) to_k: Linear,
    pub(super) to_v: Linear,
    pub(super) q_norm: Array,
    pub(super) k_norm: Array,
    pub(super) to_out: Linear,
    pub(super) gate: Option<Linear>,
    pub(super) heads: i32,
    pub(super) dim_head: i32,
    pub(super) eps: f32,
    /// sc-4942 — run the SDPA segment inside an `mlx::checkpoint` so its backward recomputes the
    /// decomposed attention (MLX has no fused SDPA backward) instead of retaining the per-layer seq²
    /// probability matrix. Training-only knob, set by [`set_sdpa_checkpoint`](Self::set_sdpa_checkpoint);
    /// `false` on the inference path (byte-identical to the pre-sc-4942 forward).
    pub(super) ckpt_sdpa: bool,
    /// Ladder rung 3 — bounded attention (sc-18797). [`AttentionBudget::UNBOUNDED`] is the load
    /// default and takes the single fused-SDPA call, byte-identical to the pre-rung forward; the
    /// chunk boundaries under a real budget come from the SHARED planner
    /// (`gen_core::attention_budget`) via `mlx_gen::attention`, never from arithmetic here.
    pub(super) attn_budget: AttentionBudget,
}

impl Attention {
    /// Load an attention with explicit `heads`/`dim_head` (the *inner* dims = `heads·dim_head`,
    /// which the q/k/v project to; cross-modal attns project a different query/context dim into the
    /// same inner). `eps` is the q/k-RMSNorm epsilon.
    pub(super) fn load(
        w: &Weights,
        prefix: &str,
        heads: i32,
        dim_head: i32,
        eps: f32,
        prec: Precision,
    ) -> Result<Self> {
        let gate = if w.get(&format!("{prefix}.to_gate_logits.weight")).is_some() {
            Some(Linear::load(w, &format!("{prefix}.to_gate_logits"), prec)?)
        } else {
            None
        };
        Ok(Self {
            to_q: Linear::load(w, &format!("{prefix}.to_q"), prec)?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"), prec)?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"), prec)?,
            q_norm: param(w, &format!("{prefix}.q_norm.weight"), prec)?,
            k_norm: param(w, &format!("{prefix}.k_norm.weight"), prec)?,
            to_out: Linear::load(w, &format!("{prefix}.to_out"), prec)?,
            gate,
            heads,
            dim_head,
            eps,
            ckpt_sdpa: false,
            attn_budget: AttentionBudget::UNBOUNDED,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-4942). Training-only — see `ckpt_sdpa`.
    pub(super) fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// Select ladder rung 3's score budget for this attention.
    pub(super) fn set_attention_budget(&mut self, budget: AttentionBudget) {
        self.attn_budget = budget;
    }

    /// Cast the attention's frozen weights (q/k/v/out projections, optional gate, q/k RMSNorm
    /// scales) to the training compute `dtype` in place (sc-4942 bf16 lever).
    pub(super) fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.to_q.cast_weights(dtype)?;
        self.to_k.cast_weights(dtype)?;
        self.to_v.cast_weights(dtype)?;
        self.to_out.cast_weights(dtype)?;
        if let Some(g) = &mut self.gate {
            g.cast_weights(dtype)?;
        }
        self.q_norm = to_dtype(&self.q_norm, dtype)?;
        self.k_norm = to_dtype(&self.k_norm, dtype)?;
        Ok(())
    }

    // The `(B, S, inner) → (B, H, S, head_dim)` head split that used to live here is now
    // `qkv::prepare`'s stages 3 and 5 (SC-18319) — the identical `reshape → transpose_axes`, with
    // the full-dim QK-norm placed before it and the rotation after it, exactly as below.

    /// `pe` rotates the query (and the key if `k_pe` is `None`); `k_pe` rotates the key separately
    /// (cross-modal: video-positioned q, audio-positioned k, or vice-versa). `pe == None` ⇒ no RoPE
    /// on either (text cross-attention). Mirrors `attention.py::Attention.__call__`.
    pub(super) fn forward(
        &self,
        x: &Array,
        context: Option<&Array>,
        mask: Option<&Array>,
        pe: Option<(&Array, &Array)>,
        k_pe: Option<(&Array, &Array)>,
    ) -> Result<Array> {
        let ctx = context.unwrap_or(x);

        // SC-18319 — the shared prologue. LTX is the family that forced two of the knobs to exist:
        //
        // * **knob 1's `FullDimPreSplit` arm** — `q_norm`/`k_norm` are RMSNorms over the whole
        //   `heads · dim_head` projection, applied BEFORE the head split, not per-head after it.
        //   Everything else in this tree except Wan normalizes per head, and the two reduce over
        //   different widths, so this is not a placement detail.
        // * **knob 6** — `k_pe` is an independent key position table (cross-modal attention
        //   positions the query on the video stream and the key on the audio stream, or vice
        //   versa); `k_pe == None` falls back to the query's table, exactly as before.
        //
        // Plus **knob 3** (`pe == None` ⇒ no rotation on either stream — text cross-attention) and
        // **knob 2's `HalvesPaired` arm**, which is `rope::apply_split_rotary_emb` verbatim: the
        // GPT-NeoX rotate-halves form over a **half**-width `[B, H, T, dim_head/2]` table through
        // the shared `nn::rope_rotate`, computed in f32 and cast back.
        //
        // **Adapters.** LTX carries its LoRA on its OWN stack rather than the shared
        // `AdaptableLinear` seam, so it is deliberately NOT routed through
        // `qkv::FusedQkvProjection`: `to_q`/`to_k`/`to_v` stay three separate `forward` calls and
        // every installed residual is applied exactly where it was before. The prologue below reads
        // the three projection OUTPUTS, so it cannot observe — let alone drop — an adapter.
        let spec = AttnPrepSpec::new(self.heads, self.dim_head)
            .with_qk_norm(QkNormSpec::full_dim_pre_split(
                &self.q_norm,
                &self.k_norm,
                self.eps,
            ))
            .with_rope(match pe {
                Some((cos, sin)) => {
                    let (kc, ks) = k_pe.unwrap_or((cos, sin));
                    RopeSpec {
                        style: RopeStyle::HalvesPaired,
                        q: Some(RopeTables::new(cos, sin)),
                        k: Some(RopeTables::new(kc, ks)),
                        // Knob 12 — `rope::apply_split_rotary_emb` promotes the stream and the
                        // tables to f32 and casts the RESULT back to the input dtype, so a bf16
                        // DiT (production) keeps a bf16 SDPA.
                        dtype: RopeDtype::RestoreInput,
                        ..RopeSpec::default()
                    }
                }
                None => RopeSpec::default(),
            })
            .with_rotation_axes(RotationAxes::HeadMajor);
        let heads = qkv::prepare(
            QkvSource::Separate {
                q: &self.to_q.forward(x)?,
                k: &self.to_k.forward(ctx)?,
                v: &self.to_v.forward(ctx)?,
            },
            &spec,
        )?;
        let (qh, kh, vh) = (heads.q, heads.k, heads.v);

        // Match the reference's Python `1.0 / math.sqrt(dim_head)` (f64 → f32), not `d^-0.5` in f32.
        let scale = (1.0f64 / (self.dim_head as f64).sqrt()) as f32;
        let out = if self.ckpt_sdpa {
            // sc-4942: checkpoint just the SDPA. q/k/v are the threaded inputs (grads to the QKV
            // projections — and their LoRA — flow through them); the f32 scale and the (constant,
            // non-differentiated) additive mask are captured. The backward recomputes the decomposed
            // attention for THIS layer alone, so the seq² probability matrix is a per-layer transient,
            // never 48× retained — the flash-backward surrogate every torch trainer gets free.
            let m = mask.cloned();
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                let o = match &m {
                    Some(m) => {
                        scaled_dot_product_attention(&inp[0], &inp[1], &inp[2], scale, m, None)?
                    }
                    None => {
                        scaled_dot_product_attention(&inp[0], &inp[1], &inp[2], scale, None, None)?
                    }
                };
                Ok(vec![o])
            });
            seg(&[qh, kh, vh])?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("ltx: checkpoint SDPA produced no output".into()))?
        } else if self.attn_budget.is_unbounded() {
            match mask {
                Some(m) => scaled_dot_product_attention(&qh, &kh, &vh, scale, m, None)?,
                None => scaled_dot_product_attention(&qh, &kh, &vh, scale, None, None)?,
            }
        } else {
            // Ladder rung 3 (sc-18797). On MLX the saving is NOT a bounded score tensor — `fast::sdpa`
            // is a fused Metal kernel that already streams them — it is the per-chunk `eval` cutting
            // the lazy graph. See `mlx_gen::attention`'s module docs; the Candle magnitude (-32% on
            // the denoise phase against -1.7% here) must NOT be carried across.
            mlx_gen::attention::sdpa_budgeted_bhsd(
                &qh,
                &kh,
                &vh,
                scale,
                mask,
                AttentionPlan::budgeted(self.attn_budget),
            )?
        };
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let inner = self.heads * self.dim_head;
        let mut out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, inner])?;

        if let Some(gate) = &self.gate {
            // Per-head gate: 2·sigmoid(logits) (zero-init → identity), broadcast over head_dim.
            let logits = gate.forward(x)?;
            let gates = multiply(&sigmoid(&logits)?, scalar(2.0).as_dtype(logits.dtype())?)?;
            let gates = gates.reshape(&[b, s, self.heads, 1])?;
            out = multiply(&out.reshape(&[b, s, self.heads, self.dim_head])?, &gates)?
                .reshape(&[b, s, inner])?;
        }
        self.to_out.forward(&out)
    }

    /// LoRA key→module map (diffusers/peft naming, post LTX normalization). `to_out.0`→`to_out`
    /// is done by the loader; `to_gate_logits` resolves only when the gated branch is present.
    pub(super) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out"] => Some(&mut self.to_out),
            ["to_gate_logits"] => self.gate.as_mut(),
            _ => None,
        }
    }

    pub(super) fn set_lora_pass(&self, pass: usize) {
        self.to_q.set_lora_pass(pass);
        self.to_k.set_lora_pass(pass);
        self.to_v.set_lora_pass(pass);
        self.to_out.set_lora_pass(pass);
        if let Some(g) = &self.gate {
            g.set_lora_pass(pass);
        }
    }
}

/// `proj_in → gelu(tanh) → proj_out`.
#[derive(Clone)]
pub(super) struct FeedForward {
    pub(super) proj_in: Linear,
    pub(super) proj_out: Linear,
}

impl FeedForward {
    /// `bias` is the caller's [`LtxConfig::ff_bias`]/`audio_ff_bias`/`connector_ff_bias` — `true`
    /// keeps both Linears biased (byte-identical to pre-sc-18758; this is 2.3 for all three, and 2.5
    /// for `audio_ff_bias`/`connector_ff_bias` — the real 2.5 header carries no `audio_ff_bias` key,
    /// so it stays the reference absent-key default `True`). `false` (2.5's **video** `ff_bias` only)
    /// means neither `proj_in` nor `proj_out` carries a bias; see reference `FeedForward.__init__`,
    /// which threads a single `bias` flag to both `net.0.proj` and `net.2`.
    pub(super) fn load(w: &Weights, prefix: &str, prec: Precision, bias: bool) -> Result<Self> {
        Ok(Self {
            proj_in: Linear::load_with_bias(w, &format!("{prefix}.proj_in"), prec, bias)?,
            proj_out: Linear::load_with_bias(w, &format!("{prefix}.proj_out"), prec, bias)?,
        })
    }

    /// Cast both projection weights to the training compute `dtype` (sc-4942 bf16 lever).
    pub(super) fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.proj_in.cast_weights(dtype)?;
        self.proj_out.cast_weights(dtype)
    }

    pub(super) fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out.forward(&gelu_ffn(&self.proj_in.forward(x)?)?)
    }

    /// LoRA targets: `ff.net.0.proj`→`ff.proj_in`, `ff.net.2`→`ff.proj_out` (renamed by the loader).
    pub(super) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["proj_in"] => Some(&mut self.proj_in),
            ["proj_out"] => Some(&mut self.proj_out),
            _ => None,
        }
    }

    pub(super) fn set_lora_pass(&self, pass: usize) {
        self.proj_in.set_lora_pass(pass);
        self.proj_out.set_lora_pass(pass);
    }
}

/// `table[row] + timestep_proj[row]` for `row ∈ [lo, hi)`. `table` is `(num_ada, dim)`; `timestep` is
/// `(B, S', num_ada·dim)`. Returns the `hi−lo` modulation tensors, each `(B, S', dim)`.
pub(super) fn ada_values(table: &Array, timestep: &Array, lo: i32, hi: i32) -> Result<Vec<Array>> {
    let num_ada = table.shape()[0];
    let dim = table.shape()[1];
    let ts = timestep.shape();
    let (b, s) = (ts[0], ts[1]);
    let ts4 = timestep.reshape(&[b, s, num_ada, dim])?;
    let mut out = Vec::with_capacity((hi - lo) as usize);
    for row in lo..hi {
        let trow = table.index_axis(row, 0)?.reshape(&[1, 1, dim])?;
        let tsrow = ts4.index_axis(row, 2)?;
        out.push(add(&trow, &tsrow)?);
    }
    Ok(out)
}

/// Index a single position `i` along `axis`, dropping that axis.
pub(super) trait IndexAxis {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array>;
}

impl IndexAxis for Array {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array> {
        Ok(self.take_axis(Array::from_int(i), axis)?)
    }
}
