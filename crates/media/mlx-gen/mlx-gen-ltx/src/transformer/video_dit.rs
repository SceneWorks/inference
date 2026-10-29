//! The video-only `LtxDiT` (the 2.3 trainer and parity surface) and the shared denoise helpers.

use super::*;

/// A per-block trainable LoRA target threaded into the gradient-checkpoint closure (sc-4942): the
/// block-local resolution path (e.g. `["attn1", "to_q"]`) and the factor-map keys for its A/B, so
/// `LtxDiT::forward_with_main_checkpointed` can pull each block's live factors out of the traced
/// `LoraParams` and re-install them inside the recompute segment. Built by the trainer from its
/// resolved targets.
pub struct BlockLoraRef {
    pub local: Vec<String>,
    pub a_key: String,
    pub b_key: String,
}

pub struct LtxDiT {
    pub(super) patchify_proj: Linear,
    pub(super) adaln: AdaLayerNormSingle,
    pub(super) prompt_adaln: Option<AdaLayerNormSingle>,
    pub(super) blocks: Vec<VideoBlock>,
    pub(super) scale_shift_table: Array, // (2, inner)
    pub(super) proj_out: Linear,
    /// `(1, inner_dim)` keyframe absolute-position marker (sc-18758), `Some` only when
    /// `cfg.use_keyframes_abs_pos_embedding` — the checkpoint carries no such tensor otherwise.
    pub(super) keyframes_embedding: Option<Array>,
    pub(super) cfg: LtxConfig,
    pub(super) prec: Precision,
    /// Per-stage SPLIT-RoPE table cache (F-048): the tables are constant across denoise steps.
    pub(super) rope_memo: RopeMemo,
    /// Monotonic source of per-stage RoPE epoch tokens (sc-7141); see [`Self::next_rope_epoch`].
    pub(super) rope_epoch: Cell<u64>,
}

impl LtxDiT {
    pub fn from_weights(w: &Weights, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let blocks = (0..cfg.num_layers)
            .map(|i| VideoBlock::load(w, &format!("transformer_blocks.{i}"), cfg, prec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patchify_proj: Linear::load(w, "patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "adaln_single", prec)?,
            prompt_adaln: if cfg.apply_gated_attention {
                Some(AdaLayerNormSingle::load(w, "prompt_adaln_single", prec)?)
            } else {
                None
            },
            blocks,
            scale_shift_table: param(w, "scale_shift_table", prec)?,
            proj_out: Linear::load(w, "proj_out", prec)?,
            // sc-18758: the checkpoint carries `keyframes_abs_pos_embedding` (1, inner_dim) iff the
            // config flag is set — `require`, not `get`, so a config that says `true` but a checkpoint
            // that omits the tensor is a load error, not a silent None.
            keyframes_embedding: cfg
                .use_keyframes_abs_pos_embedding
                .then(|| param(w, "keyframes_abs_pos_embedding", prec))
                .transpose()?,
            cfg: cfg.clone(),
            prec,
            rope_memo: RopeMemo::default(),
            rope_epoch: Cell::new(0),
        })
    }

    /// Issue a fresh per-stage RoPE epoch token (sc-7141). Call once at the top of a denoise loop and
    /// pass the returned token as `Some(epoch)` to every [`Self::forward`] in that loop: the per-stage
    /// SPLIT-RoPE tables then compute on the first step and hit the `RopeMemo` for the rest. The
    /// counter is monotonic and never reused, so a later stage's token can't collide with an earlier
    /// stage's cached tables (which would silently corrupt RoPE).
    pub fn next_rope_epoch(&self) -> u64 {
        let e = self.rope_epoch.get();
        self.rope_epoch.set(e.wrapping_add(1));
        e
    }

    /// Resolve a LoRA target (diffusers/peft dotted path, post LTX key normalization) to its
    /// [`Linear`], so the loader ([`crate::adapters`]) can install a residual onto it. The video-only
    /// surface: the 48 blocks' `attn{1,2}` + `ff` leaves, the two adaLN-single modules, and the
    /// global `patchify_proj`/`proj_out`. Audio / `av_ca` / `a2v` targets and the PixArt-spelled adaLN
    /// embedder (`linear_1/2`) resolve to `None` here → reported skipped, never silently dropped.
    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["patchify_proj"] => Some(&mut self.patchify_proj),
            ["proj_out"] => Some(&mut self.proj_out),
            ["adaln_single", rest @ ..] => self.adaln.adaptable_mut(rest),
            ["prompt_adaln_single", rest @ ..] => self.prompt_adaln.as_mut()?.adaptable_mut(rest),
            ["transformer_blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    /// Select the active distilled denoise pass for every installed LoRA residual (sc-2687 per-pass
    /// strength). The pipeline calls this before each stage; a no-adapter model is unaffected.
    pub fn set_lora_pass(&self, pass: usize) {
        self.patchify_proj.set_lora_pass(pass);
        self.proj_out.set_lora_pass(pass);
        self.adaln.set_lora_pass(pass);
        if let Some(p) = &self.prompt_adaln {
            p.set_lora_pass(pass);
        }
        for b in &self.blocks {
            b.set_lora_pass(pass);
        }
    }

    /// Toggle SDPA-segment gradient checkpointing across the whole block stack (sc-4942). The trainer
    /// arms it for every training forward (numerically identical to the retained backward; bounds the
    /// seq² attention retention to one layer's transient), and turns it OFF when whole-block
    /// checkpointing is on — the block recompute already covers attention, so nesting would recompute
    /// it twice for no memory win. Inference never calls this (the flag stays `false`).
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.blocks {
            b.set_sdpa_checkpoint(on);
        }
    }

    /// Cast the DiT to the `dtype` training compute precision in place — the sc-4942 bf16 lever. Casts
    /// every frozen leaf (patchify/adaLN/blocks/output) AND flips the activation mode via
    /// `Precision::with_compute_dtype`, so `preprocess` (which keys off
    /// `self.prec.dtype()`) runs the whole forward at the new dtype: bf16 halves the retained-activation
    /// working set (the dominant first-step cost on this Q-packed base) and matches the reference's own
    /// `quant_bf16` production-speed path. The trainable factors / loss / grads / optimizer stay f32
    /// (master-weights), installed at the compute dtype by the trainer after this cast. Destructive for
    /// a narrowing cast (f32→bf16): reload for f32. Inference never calls this (KEEP-F32 production).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.patchify_proj.cast_weights(dtype)?;
        self.proj_out.cast_weights(dtype)?;
        self.adaln.cast_weights(dtype)?;
        if let Some(p) = &mut self.prompt_adaln {
            p.cast_weights(dtype)?;
        }
        for b in &mut self.blocks {
            b.cast_weights(dtype)?;
        }
        self.scale_shift_table = to_dtype(&self.scale_shift_table, dtype)?;
        if let Some(kf) = &self.keyframes_embedding {
            self.keyframes_embedding = Some(to_dtype(kf, dtype)?);
        }
        self.prec = self.prec.with_compute_dtype(dtype);
        Ok(())
    }

    /// The DiT's current compute dtype (sc-4942), probed from the patchify projection. Lets the trainer
    /// detect a prior [`cast_weights`](Self::cast_weights) — the cast is destructive, so an f32 request
    /// after a bf16 cast must reload, not proceed.
    pub fn compute_dtype(&self) -> Dtype {
        self.patchify_proj.weight_dtype()
    }

    /// The preprocessor (mirrors the reference `TransformerArgsPreprocessor.prepare`): patchify_proj →
    /// adaLN-single timestep projection + prompt-adaLN → caption_projection (Identity, 2.3) → SPLIT
    /// RoPE tables. Shared by [`forward`](Self::forward) and [`block_hidden`](Self::block_hidden).
    pub(super) fn preprocess(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        positions: &Array,
        rope_epoch: Option<u64>,
    ) -> Result<Preprocessed> {
        let dt = self.prec.dtype();
        let b = latent.shape()[0];
        let inner = self.cfg.inner_dim();
        let coeff = self.cfg.adaln_embedding_coefficient;

        let x = self.patchify_proj.forward(&latent.as_dtype(dt)?)?;
        // sc-18758/sc-18789: the DFR keyframe-slot marker. This video-only (2.3) preprocess never
        // carries generated slots, so no mask — the AV token loops are the mask-threading path
        // (see `apply_keyframes_embedding`).
        let x = apply_keyframes_embedding(&x, self.keyframes_embedding.as_ref(), None)?;

        // adaLN-single timestep projection. The `× timestep_scale_multiplier` runs in the **input
        // dtype** (matching `denoise_av`, which feeds a latent-dtype timestep): the adaLN sinusoid
        // upcasts to f32 internally, but a bf16 timestep must round `bf16(σ·1000)` *first* — pre-
        // upcasting to f32 would change the high-frequency sinusoid phase (~33% velocity divergence
        // in the bf16 path). f32 input is unaffected (`f32(σ)·1000` either way).
        let mult = scalar(self.cfg.timestep_scale_multiplier as f32).as_dtype(timestep.dtype())?;
        let ts_flat = multiply(timestep, &mult)?.reshape(&[-1])?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, dt)?;
        let ts_emb = ts_emb.reshape(&[b, -1, coeff * inner])?;
        let emb_ts = emb_ts.reshape(&[b, -1, inner])?;

        // prompt-adaLN (gated family): one shared modulation per sample.
        let prompt_ts = match &self.prompt_adaln {
            Some(padaln) => {
                let src = if timestep.ndim() > 1 {
                    timestep.index_axis(0, 1)?.reshape(&[b, 1])?
                } else {
                    timestep.clone()
                };
                let src = multiply(&src, &mult)?.reshape(&[-1])?;
                let (pts, _) = padaln.forward(&src, dt)?;
                Some(pts.reshape(&[b, -1, 2 * inner])?)
            }
            None => None,
        };

        // caption_projection = Identity (2.3): context enters cross-attn as-is.
        let context = context.as_dtype(dt)?;

        // SPLIT RoPE from the position grid (f32 tables; the block casts per input dtype). Cached on
        // the per-stage `rope_epoch` (or `positions` content when `None`) — constant across the stage's
        // denoise steps, so computed once not per step (F-048 / sc-7141).
        let (cos, sin) = self.rope_memo.get_or_compute(positions, rope_epoch, || {
            precompute_split_freqs_cis(
                positions,
                inner,
                self.cfg.positional_embedding_theta,
                &self.cfg.positional_embedding_max_pos,
                self.cfg.num_attention_heads,
            )
        })?;

        Ok(Preprocessed {
            x,
            ts_emb,
            emb_ts,
            prompt_ts,
            context,
            cos,
            sin,
        })
    }

    /// Velocity forward.
    ///
    /// * `latent` — `(B, S, in_channels=128)` patchified latent tokens.
    /// * `timestep` — `(B, 1)` (or `(B,)`) per-sample sigma (T2V; broadcast over tokens).
    /// * `context` — `(B, ctx, inner)` text embeddings (connector output); `mask` its additive mask.
    /// * `positions` — `(B, 3, S, 2)` position grid (from [`crate::positions`]).
    /// * `rope_epoch` — per-stage RoPE cache token (sc-7141): `Some(epoch)` from a denoise loop (see
    ///   [`Self::next_rope_epoch`]) takes the O(1) memo fast path; `None` falls back to the
    ///   `positions`-content compare (trainer / direct callers, behavior unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
        positions: &Array,
        rope_epoch: Option<u64>,
    ) -> Result<Array> {
        let p = self.preprocess(latent, timestep, context, positions, rope_epoch)?;
        let mut h = p.x;
        for block in &self.blocks {
            h = block.forward(
                &h,
                &p.ts_emb,
                p.prompt_ts.as_ref(),
                &p.context,
                mask,
                &p.cos,
                &p.sin,
            )?;
        }
        self.output_head(&h, &p.emb_ts)
    }

    /// Training forward with **per-block gradient checkpointing** (sc-4942). Identical compute to
    /// [`forward`](Self::forward), but each of the 48 `transformer_blocks` runs inside an
    /// `mlx::checkpoint` segment whose explicit inputs are the block hidden state plus that block's
    /// trainable LoRA factors — so the reverse pass recomputes the block instead of retaining its
    /// activations (bounding the first-step working set), while gradients still flow to the LoRA params.
    /// The preprocessor and the output head run normally. The whole adapter surface of this trainer is
    /// in the blocks (`attn1`/`attn2`), so every trained target is threaded here.
    ///
    /// `params` is the live (traced) factor map; `block_targets[i]` lists block `i`'s trainable targets
    /// (block-local path + factor keys), in the order their factors are threaded as checkpoint inputs.
    /// Blocks with no trained targets still run checkpointed (hidden-only input) so the whole stack is
    /// uniformly recompute-on-backward.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_main_checkpointed(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
        positions: &Array,
        params: &LoraParams,
        block_targets: &[Vec<BlockLoraRef>],
        alpha: f32,
    ) -> Result<Array> {
        // `None`: content-keyed memo. Training positions are constant within a step (incl. the
        // checkpoint recompute), so the content compare hits; a per-step epoch would need threading
        // through the checkpoint closure for no inference benefit (sc-7141).
        let p = self.preprocess(latent, timestep, context, positions, None)?;
        let mut h = p.x;
        for (i, block) in self.blocks.iter().enumerate() {
            // Cheap clone (Arrays are refcounted): the closure must OWN its state because the backward
            // recompute runs after this method's frame is gone; a borrow of `self`/`p` would dangle.
            let mut blk = block.clone();
            let empty: Vec<BlockLoraRef> = Vec::new();
            let refs = block_targets.get(i).unwrap_or(&empty);
            let locals: Vec<Vec<String>> = refs.iter().map(|r| r.local.clone()).collect();
            let ts_emb = p.ts_emb.clone();
            let prompt_ts = p.prompt_ts.clone();
            let context_c = p.context.clone();
            let mask_c = mask.cloned();
            let cos = p.cos.clone();
            let sin = p.sin.clone();
            let alpha_c = alpha;

            // Threaded inputs: [hidden, a_0, b_0, a_1, b_1, ...] (raw `[r,in]`/`[out,r]` f32 factors).
            let mut inputs: Vec<Array> = Vec::with_capacity(1 + 2 * refs.len());
            inputs.push(h.clone());
            for r in refs {
                inputs.push(
                    params
                        .get(r.a_key.as_str())
                        .ok_or_else(|| {
                            mlx_gen::Error::Msg(format!("LoRA param missing: {}", r.a_key))
                        })?
                        .clone(),
                );
                inputs.push(
                    params
                        .get(r.b_key.as_str())
                        .ok_or_else(|| {
                            mlx_gen::Error::Msg(format!("LoRA param missing: {}", r.b_key))
                        })?
                        .clone(),
                );
            }

            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                // Reinstall the explicit-input factors with the SAME `(transpose, alpha/rank fold)`
                // `install_train_lora` applies — so the checkpointed block forward is numerically
                // identical to the installed-adapter path, and grads route to `inp`. Dtype-follow on the
                // hidden state (bf16 lever): the f32 factors must join the bf16 stream or every adapted
                // Linear re-promotes the block to f32. No-op in f32 mode; grads flow back f32 via the
                // astype VJP.
                let dt = inp[0].dtype();
                for (j, local) in locals.iter().enumerate() {
                    let a = inp[1 + 2 * j].t().as_dtype(dt)?; // [r,in] -> [in,r]
                    let rank = a.shape()[1] as f32;
                    let b = inp[2 + 2 * j]
                        .t() // [out,r] -> [r,out]
                        .multiply(scalar(alpha_c / rank))?
                        .as_dtype(dt)?;
                    let seg_refs: Vec<&str> = local.iter().map(String::as_str).collect();
                    blk.adaptable_mut(&seg_refs)
                        .ok_or_else(|| {
                            Exception::custom(format!(
                                "checkpoint LoRA target not found: {}",
                                local.join(".")
                            ))
                        })?
                        .set_train_lora(a, b);
                }
                let out = blk
                    .forward(
                        &inp[0],
                        &ts_emb,
                        prompt_ts.as_ref(),
                        &context_c,
                        mask_c.as_ref(),
                        &cos,
                        &sin,
                    )
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(vec![out])
            });
            h = seg(&inputs)?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("ltx: checkpoint segment produced no output".into()))?;
        }
        self.output_head(&h, &p.emb_ts)
    }

    /// Diagnostic: run the preprocessor + the first `n` blocks and return the hidden state (for the
    /// per-block bisection of the e2e residual). `n == blocks.len()` is the full pre-output hidden.
    #[doc(hidden)]
    pub fn block_hidden(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
        positions: &Array,
        n: usize,
    ) -> Result<Array> {
        let p = self.preprocess(latent, timestep, context, positions, None)?; // diagnostic: content-keyed
        let mut h = p.x;
        for block in self.blocks.iter().take(n) {
            h = block.forward(
                &h,
                &p.ts_emb,
                p.prompt_ts.as_ref(),
                &p.context,
                mask,
                &p.cos,
                &p.sin,
            )?;
        }
        Ok(h)
    }
}

/// The [`LtxDiT::preprocess`] outputs threaded into the block stack + output head.
pub(super) struct Preprocessed {
    pub(super) x: Array,
    pub(super) ts_emb: Array,
    pub(super) emb_ts: Array,
    pub(super) prompt_ts: Option<Array>,
    pub(super) context: Array,
    pub(super) cos: Array,
    pub(super) sin: Array,
}

impl LtxDiT {
    /// The output head in isolation (LayerNorm-affine-false → final scale-shift → proj_out), for the
    /// S3b bisection: feed the reference post-block hidden + embedded timestep, compare the velocity.
    pub fn output_head(&self, h: &Array, emb_ts: &Array) -> Result<Array> {
        let b = h.shape()[0];
        let inner = self.cfg.inner_dim();
        let table = self.scale_shift_table.reshape(&[1, 1, 2, inner])?;
        let ss = add(&table, &emb_ts.reshape(&[b, -1, 1, inner])?)?;
        let shift = ss.index_axis(0, 2)?;
        let scale = ss.index_axis(1, 2)?;
        let normed = layer_norm(h, None, None, self.cfg.norm_eps as f32)?;
        let out = modulate(&normed, &scale, &shift)?;
        self.proj_out.forward(&out)
    }
}

/// `denoised = latent − σ·velocity` (`to_denoised`): velocity → x₀.
pub fn to_denoised(latent: &Array, velocity: &Array, sigma: &Array) -> Result<Array> {
    Ok(subtract(latent, &multiply(velocity, sigma)?)?)
}

// ===================================================================================================
// AudioVideo DiT (sc-2684) — the dual-modality `BasicAVTransformerBlock` / `LTXModel`.
// ===================================================================================================

/// `positions[:, 0:1, :, :]` — the time axis as a `(B, 1, T, 2)` grid (the cross-modal RoPE input;
/// `MultiModalTransformerArgsPreprocessor.prepare`). For the audio grid (already `(B, 1, T, 2)`) this
/// is a no-op slice.
pub(super) fn time_axis(positions: &Array) -> Result<Array> {
    let sh = positions.shape();
    let (b, t) = (sh[0], sh[2]);
    Ok(positions
        .take_axis(Array::from_int(0), 1)? // (B, T, 2)
        .reshape(&[b, 1, t, 2])?)
}
