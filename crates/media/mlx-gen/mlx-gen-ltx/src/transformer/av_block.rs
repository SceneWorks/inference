//! The audio-video block: per-stream preparation, cross-modal attention and the perturbation
//! plan.

use super::*;

/// One modality's non-block modules + dims — the video or audio half of the AV DiT. Carries the
/// patchify projection, adaLN-single (timestep → coeff·dim) + prompt-adaLN, the two cross-modal
/// adaLN-single modules (4-coeff scale-shift + 1-coeff gate), the output scale-shift table, and the
/// output projection, plus the dims that drive RoPE.
pub(super) struct Stream {
    pub(super) patchify: Linear,
    pub(super) adaln: AdaLayerNormSingle,
    pub(super) prompt_adaln: AdaLayerNormSingle,
    pub(super) cross_ss_adaln: AdaLayerNormSingle,
    pub(super) cross_gate_adaln: AdaLayerNormSingle,
    pub(super) scale_shift_table: Array, // (2, inner) output head
    pub(super) proj_out: Linear,
    /// `(1, inner)` keyframe absolute-position marker (sc-18758) — the **video** stream only
    /// (`_init_video`; the reference never builds this for the audio stream), `Some` only when
    /// `cfg.use_keyframes_abs_pos_embedding`.
    pub(super) keyframes_embedding: Option<Array>,
    pub(super) inner: i32,
    pub(super) heads: i32,
    pub(super) coeff: i32, // adaLN row count (9 gated)
    pub(super) self_max_pos: Vec<i32>,
    pub(super) cross_inner: i32, // audio_cross_attention_dim (2048) — the cross-modal RoPE inner
    pub(super) cross_max_pos: i32,
    pub(super) theta: f64,
    pub(super) ts_mult: i32,
    pub(super) av_ca_ts_mult: i32,
    pub(super) eps: f32,
    pub(super) prec: Precision,
    /// Per-stage caches for this stream's self-attention + cross-modal SPLIT-RoPE tables (F-048).
    pub(super) self_rope_memo: RopeMemo,
    pub(super) cross_rope_memo: RopeMemo,
}

/// The per-modality preprocessed args threaded into the block stack + output head.
pub(super) struct StreamPrep {
    pub(super) x: Array,
    pub(super) ts_emb: Array,
    pub(super) emb_ts: Array,
    pub(super) prompt_ts: Array,
    pub(super) context: Array,
    pub(super) cos: Array,
    pub(super) sin: Array,
    pub(super) cross_cos: Array,
    pub(super) cross_sin: Array,
    pub(super) cross_ss_ts: Array,
    pub(super) cross_gate_ts: Array,
}

/// Borrowed view of a [`StreamPrep`] passed to [`AvBlock::forward`].
pub(super) struct StreamArgs<'a> {
    pub(super) ts_emb: &'a Array,
    pub(super) prompt_ts: &'a Array,
    pub(super) context: &'a Array,
    pub(super) mask: Option<&'a Array>,
    pub(super) cos: &'a Array,
    pub(super) sin: &'a Array,
    pub(super) cross_cos: &'a Array,
    pub(super) cross_sin: &'a Array,
    pub(super) cross_ss_ts: &'a Array,
    pub(super) cross_gate_ts: &'a Array,
}

impl StreamPrep {
    pub(super) fn args<'a>(&'a self, mask: Option<&'a Array>) -> StreamArgs<'a> {
        StreamArgs {
            ts_emb: &self.ts_emb,
            prompt_ts: &self.prompt_ts,
            context: &self.context,
            mask,
            cos: &self.cos,
            sin: &self.sin,
            cross_cos: &self.cross_cos,
            cross_sin: &self.cross_sin,
            cross_ss_ts: &self.cross_ss_ts,
            cross_gate_ts: &self.cross_gate_ts,
        }
    }
}

impl Stream {
    /// `latent` `(B, S, in)`, per-token `timestep` `(B, S)`, text `context`, `positions` grid.
    /// Reproduces `TransformerArgsPreprocessor.prepare` + the multimodal cross-PE / cross-timesteps.
    pub(super) fn prepare(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        positions: &Array,
        keyframes_mask: Option<&Array>,
        rope_epoch: Option<u64>,
    ) -> Result<StreamPrep> {
        let dt = self.prec.dtype();
        let b = latent.shape()[0];
        let (inner, coeff) = (self.inner, self.coeff);

        let x = self.patchify.forward(&latent.as_dtype(dt)?)?;
        // sc-18758/sc-18789: the DFR keyframe-slot marker (video stream only;
        // `self.keyframes_embedding` is always `None` for the audio stream). The DFR token loops
        // thread a real `(B, S, 1)` mask marking generated-keyframe slot tokens; every other path
        // passes `None`, which keeps this an exact no-op.
        let x = apply_keyframes_embedding(&x, self.keyframes_embedding.as_ref(), keyframes_mask)?;

        // adaLN-single timestep projection (the `× ts_mult` runs in the input dtype; see the
        // video-only path's note — bf16 must round `bf16(σ·1000)` first).
        let mult = scalar(self.ts_mult as f32).as_dtype(timestep.dtype())?;
        let ts_flat = multiply(timestep, &mult)?.reshape(&[-1])?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, dt)?;
        let ts_emb = ts_emb.reshape(&[b, -1, coeff * inner])?;
        let emb_ts = emb_ts.reshape(&[b, -1, inner])?;

        // prompt-adaLN: one shared modulation per sample (timestep[:, :1]).
        let src = if timestep.ndim() > 1 {
            timestep.index_axis(0, 1)?.reshape(&[b, 1])?
        } else {
            timestep.clone()
        };
        let src = multiply(&src, &mult)?.reshape(&[-1])?;
        let (pts, _) = self.prompt_adaln.forward(&src, dt)?;
        let prompt_ts = pts.reshape(&[b, -1, 2 * inner])?;

        // Cross-modal scale-shift (4·dim) + gate (1·dim) timesteps. The gate timestep carries the
        // extra `av_ca_factor = av_ca_ts_mult / ts_mult` (1.0 for 2.3, an exact f32 no-op).
        let (cross_ss, _) = self.cross_ss_adaln.forward(&ts_flat, dt)?;
        let cross_ss_ts = cross_ss.reshape(&[b, -1, 4 * inner])?;
        let factor =
            scalar(self.av_ca_ts_mult as f32 / self.ts_mult as f32).as_dtype(ts_flat.dtype())?;
        let gate_in = multiply(&ts_flat, &factor)?;
        let (cross_gate, _) = self.cross_gate_adaln.forward(&gate_in, dt)?;
        let cross_gate_ts = cross_gate.reshape(&[b, -1, inner])?;

        // caption_projection = Identity (2.3): context enters cross-attn as-is.
        let context = context.as_dtype(dt)?;

        // Self-attention SPLIT RoPE (modality inner dim, modality max_pos). Both tables are constant
        // across the stage's denoise steps, so cache them on the per-stage `rope_epoch` (or `positions`
        // content when `None`) — computed once, not per step (F-048 / sc-7141; the cross table's
        // `time_axis(positions)` is derived from the same key).
        let (cos, sin) = self
            .self_rope_memo
            .get_or_compute(positions, rope_epoch, || {
                precompute_split_freqs_cis(
                    positions,
                    inner,
                    self.theta,
                    &self.self_max_pos,
                    self.heads,
                )
            })?;
        // Cross-modal SPLIT RoPE: the time axis only, at the cross inner dim (2048) / cross max_pos.
        let (cross_cos, cross_sin) =
            self.cross_rope_memo
                .get_or_compute(positions, rope_epoch, || {
                    precompute_split_freqs_cis(
                        &time_axis(positions)?,
                        self.cross_inner,
                        self.theta,
                        &[self.cross_max_pos],
                        self.heads,
                    )
                })?;

        Ok(StreamPrep {
            x,
            ts_emb,
            emb_ts,
            prompt_ts,
            context,
            cos,
            sin,
            cross_cos,
            cross_sin,
            cross_ss_ts,
            cross_gate_ts,
        })
    }

    /// Output head (LayerNorm-affine-false → final 2-row scale-shift → proj_out). Mirrors
    /// `LTXModel._process_output`.
    pub(super) fn output_head(&self, h: &Array, emb_ts: &Array) -> Result<Array> {
        let b = h.shape()[0];
        let table = self.scale_shift_table.reshape(&[1, 1, 2, self.inner])?;
        let ss = add(&table, &emb_ts.reshape(&[b, -1, 1, self.inner])?)?;
        let shift = ss.index_axis(0, 2)?;
        let scale = ss.index_axis(1, 2)?;
        let normed = layer_norm(h, None, None, self.eps)?;
        self.proj_out.forward(&modulate(&normed, &scale, &shift)?)
    }

    /// Select the active denoise pass for this stream's LoRA-targetable Linears (the patchify/out
    /// projections + the four adaLN-single modules). The `scale_shift_table`/gate params carry no
    /// Linear, so they are unaffected.
    pub(super) fn set_lora_pass(&self, pass: usize) {
        self.patchify.set_lora_pass(pass);
        self.proj_out.set_lora_pass(pass);
        self.adaln.set_lora_pass(pass);
        self.prompt_adaln.set_lora_pass(pass);
        self.cross_ss_adaln.set_lora_pass(pass);
        self.cross_gate_adaln.set_lora_pass(pass);
    }
}

/// `4·scale-shift + 1·gate` cross-modal adaLN values from the pre-split tables. Returns
/// `(scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate)` — the row layout of
/// `scale_shift_table_a2v_ca_{audio,video}` (`get_av_ca_ada_values`).
pub(super) fn av_ca_ada(
    ss_table: &Array,
    gate_table: &Array,
    ss_ts: &Array,
    gate_ts: &Array,
) -> Result<(Array, Array, Array, Array, Array)> {
    let ss = ada_values(ss_table, ss_ts, 0, 4)?;
    let g = ada_values(gate_table, gate_ts, 0, 1)?;
    Ok((
        ss[0].clone(),
        ss[1].clone(),
        ss[2].clone(),
        ss[3].clone(),
        g[0].clone(),
    ))
}

/// Attention controls for one AV-DiT evaluation.
///
/// STG perturbs only the two self-attention calls of selected blocks.  In
/// particular, it must not turn a selected block into a no-op: text attention,
/// cross-modal attention, and both feed-forward paths still run.  The modality
/// pass takes the complementary form and suppresses only the two inter-modality
/// attention calls at every block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AvPerturbation {
    pub(super) skip_self_mask: u64,
    pub(super) isolate_modalities: bool,
}

impl AvPerturbation {
    /// The ordinary, unperturbed AV-DiT evaluation.
    pub const NONE: Self = Self {
        skip_self_mask: 0,
        isolate_modalities: false,
    };

    /// STG evaluation: skip video and audio self-attention at precisely these
    /// transformer block indices.
    pub const fn stg(skip_self_blocks: &'static [usize]) -> Self {
        let mut mask = 0_u64;
        let mut index = 0;
        while index < skip_self_blocks.len() {
            let block = skip_self_blocks[index];
            if block < u64::BITS as usize {
                mask |= 1_u64 << block;
            }
            index += 1;
        }
        Self {
            skip_self_mask: mask,
            isolate_modalities: false,
        }
    }

    /// Runtime STG sibling used by typed validation overrides. LTX-2.5 has 48 blocks, so a u64
    /// mask preserves the cheap `Copy` perturbation value while accepting a request-owned block
    /// list instead of leaking it to manufacture a `'static` slice.
    pub fn stg_blocks(skip_self_blocks: &[usize]) -> Result<Self> {
        let mut mask = 0_u64;
        for &block in skip_self_blocks {
            if block >= u64::BITS as usize {
                return Err(Error::Msg(format!(
                    "LTX STG block {block} is outside the supported 0..{} range",
                    u64::BITS
                )));
            }
            mask |= 1_u64 << block;
        }
        Ok(Self {
            skip_self_mask: mask,
            isolate_modalities: false,
        })
    }

    /// A modality-isolated evaluation, retaining self/text/FF processing while
    /// suppressing both audio-to-video and video-to-audio attention calls.
    pub const fn modality_isolated() -> Self {
        Self {
            skip_self_mask: 0,
            isolate_modalities: true,
        }
    }

    #[must_use]
    pub(super) const fn attention_plan(self, block_index: usize) -> AvAttentionPlan {
        AvAttentionPlan {
            run_self: block_index >= u64::BITS as usize
                || self.skip_self_mask & (1_u64 << block_index) == 0,
            run_cross_modal: !self.isolate_modalities,
        }
    }
}

/// The four attention calls that an [`AvBlock`] may execute.  Text attention
/// and feed-forward are deliberately not represented because neither control
/// is allowed to suppress them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AvAttentionPlan {
    pub(super) run_self: bool,
    pub(super) run_cross_modal: bool,
}

/// One AudioVideo transformer block: the video stack + the audio stack + bidirectional cross-modal
/// attention (`BasicAVTransformerBlock`). Per-block order: video self+text-CA → audio self+text-CA →
/// cross-modal (a2v updates video, v2a updates audio) → video FF → audio FF.
pub(crate) struct AvBlock {
    // Video.
    pub(super) attn1: Attention,
    pub(super) attn2: Attention,
    pub(super) ff: FeedForward,
    pub(super) v_sst: Array, // (9, 4096)
    pub(super) v_pst: Array, // (2, 4096)
    // Audio.
    pub(super) a_attn1: Attention,
    pub(super) a_attn2: Attention,
    pub(super) a_ff: FeedForward,
    pub(super) a_sst: Array, // (9, 2048)
    pub(super) a_pst: Array, // (2, 2048)
    // Cross-modal.
    pub(super) a2v: Attention, // audio_to_video_attn (Q video, K/V audio)
    pub(super) v2a: Attention, // video_to_audio_attn (Q audio, K/V video)
    pub(super) ca_audio_ss: Array, // (4, 2048)
    pub(super) ca_audio_gate: Array, // (1, 2048)
    pub(super) ca_video_ss: Array, // (4, 4096)
    pub(super) ca_video_gate: Array, // (1, 4096)
    pub(super) eps: f32,
}

impl AvBlock {
    pub(crate) fn load(
        w: &Weights,
        prefix: &str,
        cfg: &LtxConfig,
        prec: Precision,
    ) -> Result<Self> {
        let eps = cfg.norm_eps as f32;
        let (vh, vdh) = (cfg.num_attention_heads, cfg.attention_head_dim);
        let (ah, adh) = (cfg.audio_num_attention_heads, cfg.audio_attention_head_dim);
        // Split a (5, dim) cross table into the 4-row scale-shift block + the 1-row gate.
        let split = |key: &str| -> Result<(Array, Array)> {
            let t = param(w, &format!("{prefix}.{key}"), prec)?;
            let ss = t.take_axis(Array::from_slice(&[0, 1, 2, 3], &[4]), 0)?;
            let gate = t.take_axis(Array::from_slice(&[4], &[1]), 0)?;
            Ok((ss, gate))
        };
        let (ca_audio_ss, ca_audio_gate) = split("scale_shift_table_a2v_ca_audio")?;
        let (ca_video_ss, ca_video_gate) = split("scale_shift_table_a2v_ca_video")?;
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), vh, vdh, eps, prec)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), vh, vdh, eps, prec)?,
            ff: FeedForward::load(w, &format!("{prefix}.ff"), prec, cfg.ff_bias)?,
            v_sst: param(w, &format!("{prefix}.scale_shift_table"), prec)?,
            v_pst: param(w, &format!("{prefix}.prompt_scale_shift_table"), prec)?,
            a_attn1: Attention::load(w, &format!("{prefix}.audio_attn1"), ah, adh, eps, prec)?,
            a_attn2: Attention::load(w, &format!("{prefix}.audio_attn2"), ah, adh, eps, prec)?,
            a_ff: FeedForward::load(w, &format!("{prefix}.audio_ff"), prec, cfg.audio_ff_bias)?,
            a_sst: param(w, &format!("{prefix}.audio_scale_shift_table"), prec)?,
            a_pst: param(w, &format!("{prefix}.audio_prompt_scale_shift_table"), prec)?,
            // Cross-modal attns run at the audio inner dim (heads 32 × head_dim 64 = 2048).
            a2v: Attention::load(
                w,
                &format!("{prefix}.audio_to_video_attn"),
                ah,
                adh,
                eps,
                prec,
            )?,
            v2a: Attention::load(
                w,
                &format!("{prefix}.video_to_audio_attn"),
                ah,
                adh,
                eps,
                prec,
            )?,
            ca_audio_ss,
            ca_audio_gate,
            ca_video_ss,
            ca_video_gate,
            eps,
        })
    }

    /// Self+text-CA for one modality (`run_vx`/`run_ax` body, sans FF): MSA (RoPE) → prompt-modulated
    /// text cross-attention. Returns the updated stream hidden.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn self_and_text(
        &self,
        x: &Array,
        attn1: &Attention,
        attn2: &Attention,
        sst: &Array,
        pst: &Array,
        a: &StreamArgs,
        run_self: bool,
    ) -> Result<Array> {
        let mut x = x.clone();
        if run_self {
            let msa = ada_values(sst, a.ts_emb, 0, 3)?;
            let norm = modulate(&rms_norm_noweight(&x, self.eps)?, &msa[1], &msa[0])?;
            let attn = attn1.forward(&norm, None, None, Some((a.cos, a.sin)), None)?;
            x = gated(&x, &attn, &msa[2])?;
        }

        let p = ada_values(pst, a.prompt_ts, 0, 2)?;
        let context = modulate(a.context, &p[1], &p[0])?;

        let ca = ada_values(sst, a.ts_emb, 6, 9)?;
        let norm_ca = modulate(&rms_norm_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = attn2.forward(&norm_ca, Some(&context), a.mask, None, None)?;
        x = gated(&x, &cross, &ca[2])?;
        Ok(x)
    }

    /// FeedForward (adaLN rows 3..6) for one modality.
    pub(super) fn feed_forward(
        &self,
        x: &Array,
        ff: &FeedForward,
        sst: &Array,
        ts_emb: &Array,
    ) -> Result<Array> {
        let mlp = ada_values(sst, ts_emb, 3, 6)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff_out = ff.forward(&norm)?;
        gated(x, &ff_out, &mlp[2])
    }

    /// Joint forward with the attention-level controls used by the dev sampler.
    /// This controls calls inside a block, never whether an entire block runs.
    pub(super) fn forward_controlled(
        &self,
        vx: &Array,
        ax: &Array,
        v: &StreamArgs,
        a: &StreamArgs,
        control: AvAttentionPlan,
    ) -> Result<(Array, Array)> {
        // Video / audio self-attention + text cross-attention.
        let mut vx = self.self_and_text(
            vx,
            &self.attn1,
            &self.attn2,
            &self.v_sst,
            &self.v_pst,
            v,
            control.run_self,
        )?;
        let mut ax = self.self_and_text(
            ax,
            &self.a_attn1,
            &self.a_attn2,
            &self.a_sst,
            &self.a_pst,
            a,
            control.run_self,
        )?;

        if control.run_cross_modal {
            // Cross-modal attention — both directions read the pre-update rms_norm snapshots.
            let vx_n3 = rms_norm_noweight(&vx, self.eps)?;
            let ax_n3 = rms_norm_noweight(&ax, self.eps)?;
            let (sca_a2v, sha_a2v, sca_v2a, sha_v2a, gate_v2a) = av_ca_ada(
                &self.ca_audio_ss,
                &self.ca_audio_gate,
                a.cross_ss_ts,
                a.cross_gate_ts,
            )?;
            let (scv_a2v, shv_a2v, scv_v2a, shv_v2a, gate_a2v) = av_ca_ada(
                &self.ca_video_ss,
                &self.ca_video_gate,
                v.cross_ss_ts,
                v.cross_gate_ts,
            )?;

            // Audio-to-Video: Q from video (video cross-PE), K/V from audio (audio cross-PE).
            let a2v = self.a2v.forward(
                &modulate(&vx_n3, &scv_a2v, &shv_a2v)?,
                Some(&modulate(&ax_n3, &sca_a2v, &sha_a2v)?),
                None,
                Some((v.cross_cos, v.cross_sin)),
                Some((a.cross_cos, a.cross_sin)),
            )?;
            vx = gated(&vx, &a2v, &gate_a2v)?;

            // Video-to-Audio: Q from audio (audio cross-PE), K/V from video (video cross-PE).
            let v2a = self.v2a.forward(
                &modulate(&ax_n3, &sca_v2a, &sha_v2a)?,
                Some(&modulate(&vx_n3, &scv_v2a, &shv_v2a)?),
                None,
                Some((a.cross_cos, a.cross_sin)),
                Some((v.cross_cos, v.cross_sin)),
            )?;
            ax = gated(&ax, &v2a, &gate_v2a)?;
        }

        // FeedForward.
        vx = self.feed_forward(&vx, &self.ff, &self.v_sst, v.ts_emb)?;
        ax = self.feed_forward(&ax, &self.a_ff, &self.a_sst, a.ts_emb)?;
        Ok((vx, ax))
    }

    /// The absent-audio block body (sc-18789): video self-attention + text cross-attention +
    /// feed-forward. Per the reference block's `run_a2v = run_vx and audio is not None`, the
    /// cross-modal attention (and its adaLN gate) is skipped entirely when the audio modality is
    /// absent — not run against a placeholder stream.
    pub(super) fn forward_video_only_controlled(
        &self,
        vx: &Array,
        v: &StreamArgs,
        run_self: bool,
    ) -> Result<Array> {
        let vx = self.self_and_text(
            vx,
            &self.attn1,
            &self.attn2,
            &self.v_sst,
            &self.v_pst,
            v,
            run_self,
        )?;
        self.feed_forward(&vx, &self.ff, &self.v_sst, v.ts_emb)
    }

    /// Audio-only sibling of [`Self::forward_video_only_controlled`]. With no video modality the cross-modal
    /// attentions are absent, while audio self/text attention and audio FF remain executable.
    pub(super) fn forward_audio_only(&self, ax: &Array, a: &StreamArgs) -> Result<Array> {
        let ax = self.self_and_text(
            ax,
            &self.a_attn1,
            &self.a_attn2,
            &self.a_sst,
            &self.a_pst,
            a,
            true,
        )?;
        self.feed_forward(&ax, &self.a_ff, &self.a_sst, a.ts_emb)
    }

    /// LoRA key→module map for one AV block: the video self/text attns + ff, the audio analogues, and
    /// the two cross-modal attns (`audio_to_video_attn`/`video_to_audio_attn`). `audio_ff.net.*` →
    /// `audio_ff.proj_*` is renamed by the loader before reaching here.
    pub(super) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["attn1", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["attn2", rest @ ..] => self.attn2.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            ["audio_attn1", rest @ ..] => self.a_attn1.adaptable_mut(rest),
            ["audio_attn2", rest @ ..] => self.a_attn2.adaptable_mut(rest),
            ["audio_ff", rest @ ..] => self.a_ff.adaptable_mut(rest),
            ["audio_to_video_attn", rest @ ..] => self.a2v.adaptable_mut(rest),
            ["video_to_audio_attn", rest @ ..] => self.v2a.adaptable_mut(rest),
            _ => None,
        }
    }

    pub(super) fn set_lora_pass(&self, pass: usize) {
        for attn in [
            &self.attn1,
            &self.attn2,
            &self.a_attn1,
            &self.a_attn2,
            &self.a2v,
            &self.v2a,
        ] {
            attn.set_lora_pass(pass);
        }
        self.ff.set_lora_pass(pass);
        self.a_ff.set_lora_pass(pass);
    }

    pub(super) fn set_sdpa_checkpoint(&mut self, on: bool) {
        for attention in [
            &mut self.attn1,
            &mut self.attn2,
            &mut self.a_attn1,
            &mut self.a_attn2,
            &mut self.a2v,
            &mut self.v2a,
        ] {
            attention.set_sdpa_checkpoint(on);
        }
    }

    /// Select ladder rung 3's score budget across **all six** attentions this block runs: the video
    /// and audio self+text attentions and both cross-modal attentions. Bounding only the video half
    /// would leave the audio branch's scores unbounded while the contract claimed the rung.
    pub(crate) fn set_attention_budget(&mut self, budget: AttentionBudget) {
        for attn in [
            &mut self.attn1,
            &mut self.attn2,
            &mut self.a_attn1,
            &mut self.a_attn2,
            &mut self.a2v,
            &mut self.v2a,
        ] {
            attn.set_attention_budget(budget);
        }
    }

    /// The budget this block's attentions carry. Read from the video self-attention; the six are
    /// written together by [`Self::set_attention_budget`], and
    /// `every_attention_in_a_block_carries_the_selected_budget` pins that they stay in step.
    pub(crate) fn attention_budget(&self) -> AttentionBudget {
        self.attn1.attn_budget
    }
}

/// The LTX-2.3 **AudioVideo** DiT (`LTXModel` with both stacks). Predicts `(video_velocity,
/// audio_velocity)` from the two latent token streams + shared text conditioning.
/// How the AvDiT's 48-block trunk is held — the intra-phase materialization axis
/// ([`LoadShape`](mlx_gen::LoadShape)), independent of phase-level component residency.
///
/// This enum, not a boolean flag, is what the forward branches on. That is deliberate: rung 4's
/// failure mode is output-invisible, so the streamed path must be **unrepresentable** alongside a
/// resident stack rather than merely unselected. `Streamed` holds zero blocks, which is the entire
/// rung — a variant that kept the `Vec` "just in case" would bound nothing while looking correct.
pub(crate) enum AvBlocks {
    /// The historical fast path: all `num_layers` blocks materialized and retained.
    Resident(Vec<AvBlock>),
    /// Rung 4: no blocks retained; each window rebuilds its own out of a fresh component view.
    Streamed(crate::block_stream::LtxBlockStream),
}

impl AvBlocks {
    /// Blocks actually held resident. `0` for a streamed stack — the property rung 4 buys.
    pub(crate) fn resident_len(&self) -> usize {
        match self {
            Self::Resident(blocks) => blocks.len(),
            Self::Streamed(_) => 0,
        }
    }

    /// Blocks the stack RUNS, which a streamed stack still does in full.
    pub(crate) fn n_blocks(&self) -> usize {
        match self {
            Self::Resident(blocks) => blocks.len(),
            Self::Streamed(stream) => stream.n_blocks(),
        }
    }
}
