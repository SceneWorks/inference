//! The audio-video `AvDiT` inference transformer.

use super::*;

pub struct AvDiT {
    pub(super) video: Stream,
    pub(super) audio: Stream,
    pub(super) blocks: AvBlocks,
    /// Rung 4's window schedule. Unused by a resident stack. Carried as a `Cell` because `forward`
    /// takes `&self` and the selected window is a per-request value, not a load-time one — the
    /// selected value must be the executed one, or the calibration evidence describes a run that
    /// never happened.
    pub(super) block_plan: Cell<BlockPlan>,
    /// Checked at every window boundary by the shared driver, which reports a cancelled render as
    /// [`Error::Canceled`] rather than a generic failure.
    pub(super) cancel: CancelFlag,
    /// Monotonic source of per-stage RoPE epoch tokens (sc-7141); see [`Self::next_rope_epoch`]. One
    /// counter for the whole joint DiT — the same token keys all four stream memos (video/audio ×
    /// self/cross), which are each constant across a stage's denoise steps.
    pub(super) rope_epoch: Cell<u64>,
}

impl AvDiT {
    /// Build the two per-modality streams — the patchify/adaLN/output-head surface that stays
    /// resident under every load shape. Split out of [`Self::from_weights`] so the deferred
    /// constructor builds the **same** objects rather than a second transcription of them: a
    /// streamed AvDiT must differ from a resident one in the block axis alone, or the bit-identity
    /// AC would be comparing two different models.
    pub(super) fn streams(
        w: &Weights,
        cfg: &LtxConfig,
        prec: Precision,
    ) -> Result<(Stream, Stream)> {
        let video = Stream {
            patchify: Linear::load(w, "patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "adaln_single", prec)?,
            prompt_adaln: AdaLayerNormSingle::load(w, "prompt_adaln_single", prec)?,
            cross_ss_adaln: AdaLayerNormSingle::load(
                w,
                "av_ca_video_scale_shift_adaln_single",
                prec,
            )?,
            cross_gate_adaln: AdaLayerNormSingle::load(w, "av_ca_a2v_gate_adaln_single", prec)?,
            scale_shift_table: param(w, "scale_shift_table", prec)?,
            proj_out: Linear::load(w, "proj_out", prec)?,
            // sc-18758: video-stream-only marker; `require`d (not `get`) so a config that sets the
            // flag but a checkpoint that omits the tensor is a load error, not a silent None.
            keyframes_embedding: cfg
                .use_keyframes_abs_pos_embedding
                .then(|| param(w, "keyframes_abs_pos_embedding", prec))
                .transpose()?,
            inner: cfg.inner_dim(),
            heads: cfg.num_attention_heads,
            coeff: cfg.adaln_embedding_coefficient,
            self_max_pos: cfg.positional_embedding_max_pos.to_vec(),
            cross_inner: cfg.audio_cross_attention_dim,
            cross_max_pos: cfg.cross_pe_max_pos(),
            theta: cfg.positional_embedding_theta,
            ts_mult: cfg.timestep_scale_multiplier,
            av_ca_ts_mult: cfg.av_ca_timestep_scale_multiplier,
            eps: cfg.norm_eps as f32,
            prec,
            self_rope_memo: RopeMemo::default(),
            cross_rope_memo: RopeMemo::default(),
        };
        let audio = Stream {
            patchify: Linear::load(w, "audio_patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "audio_adaln_single", prec)?,
            prompt_adaln: AdaLayerNormSingle::load(w, "audio_prompt_adaln_single", prec)?,
            cross_ss_adaln: AdaLayerNormSingle::load(
                w,
                "av_ca_audio_scale_shift_adaln_single",
                prec,
            )?,
            cross_gate_adaln: AdaLayerNormSingle::load(w, "av_ca_v2a_gate_adaln_single", prec)?,
            scale_shift_table: param(w, "audio_scale_shift_table", prec)?,
            proj_out: Linear::load(w, "audio_proj_out", prec)?,
            // The reference never builds a keyframe marker for the audio stream (`_init_video` only).
            keyframes_embedding: None,
            inner: cfg.audio_inner_dim(),
            heads: cfg.audio_num_attention_heads,
            coeff: cfg.adaln_embedding_coefficient,
            self_max_pos: vec![cfg.audio_positional_embedding_max_pos],
            cross_inner: cfg.audio_cross_attention_dim,
            cross_max_pos: cfg.cross_pe_max_pos(),
            theta: cfg.positional_embedding_theta,
            ts_mult: cfg.timestep_scale_multiplier,
            av_ca_ts_mult: cfg.av_ca_timestep_scale_multiplier,
            eps: cfg.norm_eps as f32,
            prec,
            self_rope_memo: RopeMemo::default(),
            cross_rope_memo: RopeMemo::default(),
        };
        Ok((video, audio))
    }

    /// The historical eager load: every block materialized and retained.
    pub fn from_weights(w: &Weights, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let (video, audio) = Self::streams(w, cfg, prec)?;
        let blocks = (0..cfg.num_layers)
            .map(|i| AvBlock::load(w, &format!("transformer_blocks.{i}"), cfg, prec))
            .collect::<Result<Vec<_>>>()?;
        let plan = BlockPlan::resident(blocks.len().max(1))?;
        Ok(Self {
            video,
            audio,
            blocks: AvBlocks::Resident(blocks),
            block_plan: Cell::new(plan),
            cancel: CancelFlag::default(),
            rope_epoch: Cell::new(0),
        })
    }

    /// Rung 4's load: build the resident stream surface from `w`, hold **zero** blocks, and rebuild
    /// each one per window out of `stream` during the forward.
    ///
    /// `w` is still read for the patchify/adaLN/output-head tensors, which are small, are used on
    /// every forward, and would cost a re-read per window for no residency saving. The 48-block
    /// trunk — the part rung 4 exists to bound — is the only thing deferred.
    pub fn from_weights_streamed(
        w: &Weights,
        cfg: &LtxConfig,
        prec: Precision,
        stream: crate::block_stream::LtxBlockStream,
    ) -> Result<Self> {
        if stream.n_blocks() != cfg.num_layers as usize {
            return Err(Error::Msg(format!(
                "ltx: the block stream declares {} blocks but the config declares {} — a plan built \
                 from a desynchronized depth would silently skip or repeat layers",
                stream.n_blocks(),
                cfg.num_layers
            )));
        }
        let (video, audio) = Self::streams(w, cfg, prec)?;
        // Default to the tightest bound. A caller that selected a window carries it in the request
        // and calls `set_transformer_window`; defaulting to `resident()` here would make an
        // unconfigured streamed load silently run one all-covering window, which bounds nothing.
        let plan = BlockPlan::new(stream.n_blocks(), 1)?;
        Ok(Self {
            video,
            audio,
            blocks: AvBlocks::Streamed(stream),
            block_plan: Cell::new(plan),
            cancel: CancelFlag::default(),
            rope_epoch: Cell::new(0),
        })
    }

    /// Select rung 4's window size for the requests that follow.
    ///
    /// Rejected on a resident stack rather than ignored: silently accepting a window on a stack that
    /// cannot honour it is how a calibration record comes to describe a run that never happened.
    pub fn set_transformer_window(&self, window: usize) -> Result<()> {
        let AvBlocks::Streamed(stream) = &self.blocks else {
            return Err(Error::Unsupported(
                "ltx: a transformer window was selected on a resident block stack — rung 4 requires \
                 a deferred-materialization load"
                    .into(),
            ));
        };
        self.block_plan
            .set(BlockPlan::new(stream.n_blocks(), window)?);
        Ok(())
    }

    /// The window plan the next forward will execute. Reported so a caller can prove the selected
    /// window is the executed one.
    pub fn block_plan(&self) -> BlockPlan {
        self.block_plan.get()
    }

    /// Attach the request's cancel flag, checked at every window boundary.
    pub fn set_cancel(&mut self, cancel: CancelFlag) {
        self.cancel = cancel;
    }

    /// Select ladder rung 3's attention budget for the whole trunk.
    ///
    /// Both arms are load-bearing. On a resident stack the budget is written into the blocks that
    /// already exist; on a streamed stack there are no blocks yet, so it is recorded on the stream
    /// and replayed onto every block a window materializes. Setting only the first would leave the
    /// rung-3 + rung-4 composition — the one the cost-order default actually produces, since rung 4
    /// engages rung 3 — running unbounded attention with identical output.
    pub fn set_attention_budget(&mut self, budget: mlx_gen::attention::AttentionBudget) {
        match &mut self.blocks {
            AvBlocks::Resident(blocks) => {
                for block in blocks {
                    block.set_attention_budget(budget);
                }
            }
            AvBlocks::Streamed(stream) => stream.set_attention_budget(budget),
        }
    }

    /// Training-only attention-segment checkpoint switch over the complete resident AV trunk.
    /// Streamed trunks are inference-only and cannot carry trainable adapters.
    pub(crate) fn set_sdpa_checkpoint(&mut self, on: bool) {
        if let AvBlocks::Resident(blocks) = &mut self.blocks {
            for block in blocks {
                block.set_sdpa_checkpoint(on);
            }
        }
    }

    /// The budget the next forward will execute, read back from wherever it actually lives.
    pub fn attention_budget(&self) -> mlx_gen::attention::AttentionBudget {
        match &self.blocks {
            AvBlocks::Resident(blocks) => blocks
                .first()
                .map(|block| block.attention_budget())
                .unwrap_or(AttentionBudget::UNBOUNDED),
            AvBlocks::Streamed(stream) => stream.attention_budget(),
        }
    }

    /// Whether this AvDiT defers its block trunk. The loader-identity predicate: rung 4's failure is
    /// invisible in output, so a caller that needs to know *which* loader it got asks here rather
    /// than inferring it from a request flag it passed in.
    pub fn is_block_streamed(&self) -> bool {
        matches!(self.blocks, AvBlocks::Streamed(_))
    }

    /// Blocks held resident. `0` on a streamed stack — that is the entire rung.
    pub fn resident_blocks(&self) -> usize {
        self.blocks.resident_len()
    }

    /// Blocks the stack runs, streamed or not.
    pub fn num_blocks(&self) -> usize {
        self.blocks.n_blocks()
    }

    /// Issue a fresh per-stage RoPE epoch token (sc-7141). Call once at the top of a joint denoise loop
    /// and pass the returned token as `Some(epoch)` to every [`Self::forward`] in that loop, so all four
    /// stream RoPE tables compute on the first step and hit the `RopeMemo`s for the rest. Monotonic
    /// and never reused, so a later stage can't collide with an earlier stage's cached tables.
    pub fn next_rope_epoch(&self) -> u64 {
        let e = self.rope_epoch.get();
        self.rope_epoch.set(e.wrapping_add(1));
        e
    }

    /// Resolve a LoRA target (diffusers/peft dotted path, post LTX key normalization) to its
    /// [`Linear`] across the **full** AudioVideo surface (sc-2687): the per-block video/audio self +
    /// text attns, the two cross-modal attns, both stacks' feed-forwards, and every stream-global
    /// projection / adaLN-single (video + audio + the four `av_ca_*` cross modules). The module name
    /// itself selects the stream (`audio_*` / `av_ca_audio_*` / `av_ca_v2a_*` → audio). A target that
    /// names no module (e.g. an unnormalized PixArt-spelled adaLN embedder `linear_1/2`, or an
    /// out-of-range block) resolves to `None` → reported skipped, never silently dropped. The strict
    /// LTX-2.5 loader normalizes its official `linear_1/2` spellings before calling this resolver.
    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            // A streamed trunk holds no blocks to adapt. `LtxBlockStream::new` already refuses to
            // construct over a non-empty adapter set, so this arm is the second half of the same
            // guarantee rather than a silent drop: there is no block object here that an adapter
            // could be installed onto and then be rebuilt away by the next window.
            ["transformer_blocks", n, rest @ ..] => match &mut self.blocks {
                AvBlocks::Resident(blocks) => blocks
                    .get_mut(n.parse::<usize>().ok()?)?
                    .adaptable_mut(rest),
                AvBlocks::Streamed(_) => None,
            },
            // Video-stream globals.
            ["patchify_proj"] => Some(&mut self.video.patchify),
            ["proj_out"] => Some(&mut self.video.proj_out),
            ["adaln_single", rest @ ..] => self.video.adaln.adaptable_mut(rest),
            ["prompt_adaln_single", rest @ ..] => self.video.prompt_adaln.adaptable_mut(rest),
            ["av_ca_video_scale_shift_adaln_single", rest @ ..] => {
                self.video.cross_ss_adaln.adaptable_mut(rest)
            }
            ["av_ca_a2v_gate_adaln_single", rest @ ..] => {
                self.video.cross_gate_adaln.adaptable_mut(rest)
            }
            // Audio-stream globals.
            ["audio_patchify_proj"] => Some(&mut self.audio.patchify),
            ["audio_proj_out"] => Some(&mut self.audio.proj_out),
            ["audio_adaln_single", rest @ ..] => self.audio.adaln.adaptable_mut(rest),
            ["audio_prompt_adaln_single", rest @ ..] => self.audio.prompt_adaln.adaptable_mut(rest),
            ["av_ca_audio_scale_shift_adaln_single", rest @ ..] => {
                self.audio.cross_ss_adaln.adaptable_mut(rest)
            }
            ["av_ca_v2a_gate_adaln_single", rest @ ..] => {
                self.audio.cross_gate_adaln.adaptable_mut(rest)
            }
            _ => None,
        }
    }

    /// Select the active distilled denoise pass for every installed LoRA residual (sc-2687 per-pass
    /// strength) across both streams and all blocks. The pipeline calls this before each stage; a
    /// no-adapter model is unaffected.
    pub fn set_lora_pass(&self, pass: usize) {
        self.video.set_lora_pass(pass);
        self.audio.set_lora_pass(pass);
        // A streamed trunk carries no adapters (refused at `LtxBlockStream::new`), so there is no
        // per-pass strength to select on it.
        if let AvBlocks::Resident(blocks) = &self.blocks {
            for b in blocks {
                b.set_lora_pass(pass);
            }
        }
    }

    /// Ordinary, unperturbed joint velocity forward.
    ///
    /// This is the compatibility path for distilled inference. Guided dev
    /// evaluations opt into [`Self::forward_controlled`] explicitly.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        video_latent: &Array,
        video_timestep: &Array,
        video_context: &Array,
        video_mask: Option<&Array>,
        video_positions: &Array,
        audio_latent: &Array,
        audio_timestep: &Array,
        audio_context: &Array,
        audio_mask: Option<&Array>,
        audio_positions: &Array,
        video_keyframes_mask: Option<&Array>,
        rope_epoch: Option<u64>,
    ) -> Result<(Array, Array)> {
        self.forward_controlled(
            video_latent,
            video_timestep,
            video_context,
            video_mask,
            video_positions,
            audio_latent,
            audio_timestep,
            audio_context,
            audio_mask,
            audio_positions,
            video_keyframes_mask,
            rope_epoch,
            AvPerturbation::NONE,
        )
    }

    /// Controlled joint velocity forward for STG and modality guidance.
    ///
    /// * `*_latent` — `(B, S, in_channels)` patchified tokens (video 128, audio 128).
    /// * `*_timestep` — `(B, S)` per-token sigma.
    /// * `*_context` — text embeddings (video 4096, audio 2048); `*_mask` their additive masks.
    /// * `*_positions` — the position grids (video `(B,3,T,2)`, audio `(B,1,T,2)`).
    ///
    /// Returns `(video_velocity (B, S_v, 128), audio_velocity (B, S_a, 128))`.
    ///
    /// * `rope_epoch` — per-stage RoPE cache token (sc-7141): `Some(epoch)` from a denoise loop (see
    ///   [`Self::next_rope_epoch`]) takes the O(1) memo fast path for all four stream tables; `None`
    ///   falls back to the `positions`-content compare (behavior unchanged).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_controlled(
        &self,
        video_latent: &Array,
        video_timestep: &Array,
        video_context: &Array,
        video_mask: Option<&Array>,
        video_positions: &Array,
        audio_latent: &Array,
        audio_timestep: &Array,
        audio_context: &Array,
        audio_mask: Option<&Array>,
        audio_positions: &Array,
        video_keyframes_mask: Option<&Array>,
        rope_epoch: Option<u64>,
        perturbation: AvPerturbation,
    ) -> Result<(Array, Array)> {
        let vp = self.video.prepare(
            video_latent,
            video_timestep,
            video_context,
            video_positions,
            video_keyframes_mask,
            rope_epoch,
        )?;
        let ap = self.audio.prepare(
            audio_latent,
            audio_timestep,
            audio_context,
            audio_positions,
            None,
            rope_epoch,
        )?;
        let (vx0, ax0) = (vp.x.clone(), ap.x.clone());
        let (va, aa) = (vp.args(video_mask), ap.args(audio_mask));
        let (vx, ax) = match &self.blocks {
            AvBlocks::Resident(blocks) => {
                let (mut vx, mut ax) = (vx0, ax0);
                for (index, block) in blocks.iter().enumerate() {
                    let (nv, na) = block.forward_controlled(
                        &vx,
                        &ax,
                        &va,
                        &aa,
                        perturbation.attention_plan(index),
                    )?;
                    vx = nv;
                    ax = na;
                }
                (vx, ax)
            }
            AvBlocks::Streamed(stream) => {
                // Rung 4. The schedule is the SHARED driver — window arithmetic, loop order,
                // release discipline and the cancellation contract all live in
                // `gen_core::block_window` via MLX's binding. Only the "rebuild AvBlock n" step is
                // this family's, and it lives in `crate::block_stream`.
                let plan = self.block_plan.get();
                mlx_gen::block_residency::run_windowed(
                    &plan,
                    &self.cancel,
                    (vx0, ax0),
                    || stream.open(),
                    |(mut vx, mut ax), view, range| {
                        for index in range {
                            let block = stream.materialize(view, index)?;
                            let (nv, na) = block.forward_controlled(
                                &vx,
                                &ax,
                                &va,
                                &aa,
                                perturbation.attention_plan(index),
                            )?;
                            vx = nv;
                            ax = na;
                            // `block` drops here: a window holds `window_size` blocks, never the
                            // whole range's worth.
                        }
                        Ok((vx, ax))
                    },
                    // LOAD-BEARING on MLX: the carried activations are unevaluated graph nodes that
                    // still reference this window's weights, so dropping before forcing evaluation
                    // frees nothing — identical output, zero saving, silently.
                    |(vx, ax): &(Array, Array)| {
                        mlx_rs::transforms::eval([vx, ax]).map_err(Error::from)
                    },
                )?
            }
        };
        let v_vel = self.video.output_head(&vx, &vp.emb_ts)?;
        let a_vel = self.audio.output_head(&ax, &ap.emb_ts)?;
        Ok((v_vel, a_vel))
    }

    /// **Video-only** forward through the AV stack — the reference `LTXModel` called with
    /// `audio=None` (the DFR temporal-round tile denoise, `dfr_pipeline`'s `audio=None` stage
    /// call): the audio stream is skipped entirely and, per the reference block
    /// (`run_a2v = run_vx and audio is not None`), the cross-modal attentions do not run — each
    /// block reduces to video self-attention + text cross-attention + feed-forward.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_video_only(
        &self,
        video_latent: &Array,
        video_timestep: &Array,
        video_context: &Array,
        video_mask: Option<&Array>,
        video_positions: &Array,
        video_keyframes_mask: Option<&Array>,
        rope_epoch: Option<u64>,
    ) -> Result<Array> {
        self.forward_video_only_controlled(
            video_latent,
            video_timestep,
            video_context,
            video_mask,
            video_positions,
            video_keyframes_mask,
            rope_epoch,
            AvPerturbation::NONE,
        )
    }

    /// Video-only Dev validation forward with STG self-attention controls. Modality isolation is a
    /// no-op when audio is absent, while the supplied STG block mask remains fully executable.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_video_only_controlled(
        &self,
        video_latent: &Array,
        video_timestep: &Array,
        video_context: &Array,
        video_mask: Option<&Array>,
        video_positions: &Array,
        video_keyframes_mask: Option<&Array>,
        rope_epoch: Option<u64>,
        perturbation: AvPerturbation,
    ) -> Result<Array> {
        let vp = self.video.prepare(
            video_latent,
            video_timestep,
            video_context,
            video_positions,
            video_keyframes_mask,
            rope_epoch,
        )?;
        let va = vp.args(video_mask);
        let vx = match &self.blocks {
            AvBlocks::Resident(blocks) => {
                let mut vx = vp.x.clone();
                for (index, block) in blocks.iter().enumerate() {
                    vx = block.forward_video_only_controlled(
                        &vx,
                        &va,
                        perturbation.attention_plan(index).run_self,
                    )?;
                }
                vx
            }
            AvBlocks::Streamed(stream) => {
                let plan = self.block_plan.get();
                mlx_gen::block_residency::run_windowed(
                    &plan,
                    &self.cancel,
                    vp.x.clone(),
                    || stream.open(),
                    |mut vx, view, range| {
                        for index in range {
                            let block = stream.materialize(view, index)?;
                            vx = block.forward_video_only_controlled(
                                &vx,
                                &va,
                                perturbation.attention_plan(index).run_self,
                            )?;
                        }
                        Ok(vx)
                    },
                    |vx: &Array| mlx_rs::transforms::eval([vx]).map_err(Error::from),
                )?
            }
        };
        self.video.output_head(&vx, &vp.emb_ts)
    }

    /// Audio-only flexible-training forward. The video stream and both cross-modal attentions are
    /// skipped exactly as the upstream model does when `video=None`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_audio_only(
        &self,
        audio_latent: &Array,
        audio_timestep: &Array,
        audio_context: &Array,
        audio_mask: Option<&Array>,
        audio_positions: &Array,
        rope_epoch: Option<u64>,
    ) -> Result<Array> {
        let ap = self.audio.prepare(
            audio_latent,
            audio_timestep,
            audio_context,
            audio_positions,
            None,
            rope_epoch,
        )?;
        let aa = ap.args(audio_mask);
        let ax = match &self.blocks {
            AvBlocks::Resident(blocks) => {
                let mut ax = ap.x.clone();
                for block in blocks {
                    ax = block.forward_audio_only(&ax, &aa)?;
                }
                ax
            }
            AvBlocks::Streamed(stream) => {
                let plan = self.block_plan.get();
                mlx_gen::block_residency::run_windowed(
                    &plan,
                    &self.cancel,
                    ap.x.clone(),
                    || stream.open(),
                    |mut ax, view, range| {
                        for index in range {
                            let block = stream.materialize(view, index)?;
                            ax = block.forward_audio_only(&ax, &aa)?;
                        }
                        Ok(ax)
                    },
                    |ax: &Array| mlx_rs::transforms::eval([ax]).map_err(Error::from),
                )?
            }
        };
        self.audio.output_head(&ax, &ap.emb_ts)
    }
}
