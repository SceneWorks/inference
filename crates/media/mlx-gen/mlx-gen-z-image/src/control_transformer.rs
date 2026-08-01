//! Z-Image S3-DiT transformer with a VACE-style ControlNet branch (sc-2349 / sc-2257). Port of the
//! fork's `ZImageControlTransformer` (`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`, v2.1
//! config: `add_control_noise_refiner=True` / `add_control_noise_refiner_correctly=True`).
//!
//! On top of the parity-proven base [`ZImageTransformer`] (composed, not re-derived — its
//! submodules and `patchify`/`unpatchify` are reused via `pub(crate)`), this adds:
//!   - `control_all_x_embedder`: a `33·patch²·f_patch → dim` patch embedder for the VAE-encoded
//!     control context (control latent 16ch + mask 1ch + inpaint latent 16ch).
//!   - `control_noise_refiner` (2 blocks): a parallel control refiner whose hints inject into the
//!     base `noise_refiner` (image-length stage).
//!   - `control_layers` (15 blocks at base places 0,2,…,28): the main control stack whose hints
//!     inject into the matching base `layers` (unified image+caption stage).
//!
//! The control branch reuses the base image position ids / RoPE / padding (the control context
//! shares the image's spatial dims), so no separate alignment is needed. With `control_context =
//! None` the forward delegates to the base transformer verbatim; with `control_context_scale = 0`
//! the hints contribute zero — both reproduce the base output exactly (the parity self-check).

use mlx_rs::error::Exception;
use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::transforms::compile::compile;
use mlx_rs::Array;

use crate::control_transformer_block::ZImageControlBlock;
use crate::transformer::{apply_pad, row_indices, ZImageTransformer};
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::AttentionPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Cached step-invariant control-forward inputs: the base patchify metadata + RoPE freqs, plus the
/// embedded control context `c_emb` (which depends only on the constant `control_context` and the
/// image keep-mask, not the latent or timestep). Built once by
/// [`ZImageControlTransformer::prepare_control`] and reused for every denoise step (F-042), avoiding
/// the per-step re-patchify of the constant control context.
pub(crate) struct ControlPrepared {
    cap_tokens: Array,
    x_size: (i32, i32, i32),
    x_keep: Array,
    cap_keep: Array,
    img_pad: i32,
    x_freqs: Array,
    cap_freqs: Array,
    unified_freqs: Array,
    /// Control embedding, already padded + `expand_dims(0)` — the input to the control refiner.
    c_emb: Array,
}

/// Channel count of the VAE-encoded control context (16 control latent + 1 mask + 16 inpaint).
pub const CONTROL_IN_DIM: i32 = 33;
/// Base `layers` indices the 15 control layers inject into (the fork's `CONTROL_LAYERS_PLACES`).
pub(crate) const CONTROL_LAYERS_PLACES: [usize; 15] =
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
/// Base `noise_refiner` indices the 2 control refiner blocks inject into.
const CONTROL_REFINER_PLACES: [usize; 2] = [0, 1];

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

pub struct ZImageControlTransformer {
    base: ZImageTransformer,
    /// `control_all_x_embedder.{patch_size}-{f_patch_size}` — `33·p²·pf → dim`. Kept dense under
    /// Q8 (its in-features, e.g. 132, is not divisible by the group size 64).
    control_x_embedder: AdaptableLinear,
    /// The 15-block main control stack (injects into `base.layers` at `CONTROL_LAYERS_PLACES`).
    control_layers: Vec<ZImageControlBlock>,
    /// The 2-block control refiner (injects into `base.noise_refiner` at `CONTROL_REFINER_PLACES`).
    control_noise_refiner: Vec<ZImageControlBlock>,
    /// Ladder rung 4 (SC-15754) for the 15-block main control stack. The 2-block refiner is left
    /// resident: at ~97 MiB/block it is 2 blocks against the base stack's 30, so windowing it would
    /// buy ~6% of the stack's weights for another 2 re-reads per step.
    control_block_stream: Option<crate::block_stream::ZImageBlockStream>,
}

impl ZImageControlTransformer {
    /// Build from an already-loaded base transformer + the Fun-Controlnet-Union checkpoint
    /// (`control` Weights). The control keys (`control_all_x_embedder.*`, `control_layers.*`,
    /// `control_noise_refiner.*`) map 1:1 onto this tree. `prefix` is empty for a real checkpoint
    /// (un-prefixed keys) and e.g. `"w"` for the synthetic parity fixture.
    pub fn from_weights(base: ZImageTransformer, control: &Weights, prefix: &str) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let key = format!("{}-{}", base.cfg.patch_size, base.cfg.f_patch_size);
        let bcfg = crate::transformer_block::ZImageBlockConfig {
            dim: base.cfg.dim,
            n_heads: base.cfg.n_heads,
            norm_eps: base.cfg.norm_eps,
        };

        // Packed-detect (sc-8670): the control patch embedder loads packed from a pre-quantized
        // control snapshot or dense otherwise. In practice it stays dense in every tier — its
        // in-features (33·p²·pf, e.g. 132) is not divisible by the group size 64, so neither the
        // in-memory `quantize` nor the converter's shape guard packs it.
        let control_x_embedder =
            crate::quant::lin(control, &p(&format!("control_all_x_embedder.{key}")), true)?;

        let control_layers = (0..CONTROL_LAYERS_PLACES.len())
            .map(|i| {
                ZImageControlBlock::from_weights(
                    control,
                    &p(&format!("control_layers.{i}")),
                    bcfg,
                    i == 0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let control_noise_refiner = (0..CONTROL_REFINER_PLACES.len())
            .map(|i| {
                ZImageControlBlock::from_weights(
                    control,
                    &p(&format!("control_noise_refiner.{i}")),
                    bcfg,
                    i == 0,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            base,
            control_x_embedder,
            control_layers,
            control_noise_refiner,
            control_block_stream: None,
        })
    }

    /// Arm ladder rung 4 for the main control stack by recording where `control_layers` can be
    /// re-read from (SC-15754). Called by [`crate::loader::load_control_transformer`], which is the
    /// only place that knows the ControlNet checkpoint's source.
    pub fn with_control_block_stream(
        mut self,
        source: mlx_gen::WeightsSource,
        prefix: &str,
    ) -> Self {
        let base = if prefix.is_empty() {
            "control_layers".to_owned()
        } else {
            format!("{prefix}.control_layers")
        };
        // The SAME derivation the base arm uses, not a second spelling of it — a control block is a
        // base block plus two projections, so a divergence here would be silent.
        let cfg = self.base.cfg.block_cfg();
        self.control_block_stream = Some(crate::block_stream::ZImageBlockStream::new(
            source,
            base,
            cfg,
            CONTROL_LAYERS_PLACES.len(),
            true,
        ));
        self
    }

    /// Whether rung 4 can execute here — it needs BOTH stacks streamable, because a control forward
    /// runs both and the ladder bounds the transformer as a whole.
    pub(crate) fn can_stream_blocks(&self) -> bool {
        self.base.can_stream_blocks() && self.control_block_stream.is_some()
    }

    /// The control route's [`ZImageTransformer::block_window`]: same validation, plus the ControlNet
    /// stack's own source. Refusing when only one of the two stacks is streamable is deliberate — a
    /// half-bounded control forward would report a rung-4 run whose peak still carried 15 resident
    /// control blocks.
    pub(crate) fn control_block_window<'a>(
        &self,
        window_size: Option<usize>,
        cancel: &'a mlx_gen::CancelFlag,
    ) -> Result<Option<crate::block_stream::BlockWindow<'a>>> {
        if window_size.is_some() && !self.can_stream_blocks() {
            return Err(Error::Unsupported(
                "z-image control: bounded transformer residency needs a re-openable snapshot for \
                 BOTH the base DiT and the ControlNet checkpoint"
                    .to_owned(),
            ));
        }
        self.base.block_window(window_size, cancel)
    }

    /// See [`ZImageTransformer::capture_block_adapters`]. Only the base stack can carry adapters —
    /// this type's [`AdaptableHost`] routes every path to the composed base DiT.
    pub fn capture_block_adapters(&mut self) {
        self.base.capture_block_adapters();
    }

    /// Quantize to Q4/Q8 (group_size 64) — the base transformer plus every control block, but
    /// **not** the control patch embedder: its in-features (`33·p²·pf`, e.g. 132) is not divisible
    /// by 64, so `nn.quantize` leaves it dense (the fork's `d32454c` predicate). The fork applies
    /// base + control weights at full precision first, *then* quantizes the whole transformer;
    /// here the base load + this overlay are both dense, and this quantizes both together.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.base.quantize(bits)?;
        for block in &mut self.control_layers {
            block.quantize(bits)?;
        }
        for block in &mut self.control_noise_refiner {
            block.quantize(bits)?;
        }
        // Rung 4 replays the same quantization on each materialized control block.
        if let Some(stream) = self.control_block_stream.as_mut() {
            stream.set_quant_bits(bits);
        }
        // control_x_embedder intentionally left dense (in-features not divisible by 64).
        Ok(())
    }

    /// Dual-injection forward (port of `ZImageControlTransformer.__call__`).
    ///
    /// `x`: latent `(C, F, H, W)`; `cap_feats`: `(cap_len, cap_feat_dim)`; `timestep` in `[0,1]`;
    /// `control_context`: the `(33, F, H/8?, W/8?)` VAE-encoded control context (same spatial dims
    /// as the latent), or `None` to run the base transformer verbatim; `control_context_scale`
    /// weights every control hint. Returns the latent-shaped velocity `(C, F, H, W)`.
    pub fn forward(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        control_context: Option<&Array>,
        control_context_scale: f32,
    ) -> Result<Array> {
        self.forward_budgeted(
            x,
            timestep,
            cap_feats,
            control_context,
            control_context_scale,
            AttentionPlan::UNBOUNDED,
        )
    }

    /// [`forward`](Self::forward) under an explicit [`AttentionPlan`] — the public, DiT-level entry
    /// for ladder rung 3 (SC-15615) on the ControlNet route, so the equivalence and plan-threading
    /// tests can drive one whole dual-injection forward without standing up a denoise loop.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_budgeted(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        control_context: Option<&Array>,
        control_context_scale: f32,
        plan: AttentionPlan<'_>,
    ) -> Result<Array> {
        // No control context → identical to the base transformer (the fork's `control_context is
        // None` short-circuit). Delegating keeps the base path byte-for-byte the parity-proven one.
        match control_context {
            None => self.base.forward_budgeted(x, timestep, cap_feats, plan),
            Some(cc) => self.forward_control(
                x,
                timestep,
                cap_feats,
                cc,
                control_context_scale,
                None,
                plan,
            ),
        }
    }

    /// [`forward_budgeted`](Self::forward_budgeted) with a bounded transformer-residency window —
    /// the public, DiT-level entry for ladder rung 4 (SC-15754) on the ControlNet route, so the
    /// equivalence tests can drive one whole dual-injection forward without a denoise loop.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_windowed(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        control_context: Option<&Array>,
        control_context_scale: f32,
        plan: AttentionPlan<'_>,
        window_size: usize,
        cancel: &mlx_gen::CancelFlag,
    ) -> Result<Array> {
        let block_plan =
            mlx_gen::block_residency::BlockPlan::new(self.base.layers.len(), window_size)?;
        let window = crate::block_stream::BlockWindow {
            plan: block_plan,
            cancel,
        };
        match control_context {
            // No control context → the base transformer verbatim, windowed (the fork's short-circuit).
            None => {
                let sh = x.shape();
                let prep = self.base.prepare((sh[0], sh[1], sh[2], sh[3]), cap_feats)?;
                self.base
                    .forward_with_rungs(&prep, x, timestep, plan, Some(window))
            }
            Some(cc) => {
                let sh = x.shape();
                let prep = self.prepare_control((sh[0], sh[1], sh[2], sh[3]), cap_feats, cc)?;
                self.forward_with_control_rungs(
                    &prep,
                    x,
                    timestep,
                    control_context_scale,
                    None,
                    plan,
                    Some(window),
                )
            }
        }
    }

    /// Like [`forward`](Self::forward) (control path) but also returns the named per-stage
    /// intermediates, for stage-by-stage parity bisection vs the fork (`control_q8_bisect`). The
    /// returned output is identical to `forward`'s — both call the same `forward_control`.
    pub fn forward_capture(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        control_context: &Array,
        control_context_scale: f32,
    ) -> Result<(Array, Vec<(&'static str, Array)>)> {
        let mut stages = Vec::new();
        let v = self.forward_control(
            x,
            timestep,
            cap_feats,
            control_context,
            control_context_scale,
            Some(&mut stages),
            AttentionPlan::UNBOUNDED,
        )?;
        Ok((v, stages))
    }

    #[allow(clippy::too_many_arguments)]
    /// Build the [`ControlPrepared`] step-invariant inputs (base patchify metadata + RoPE freqs +
    /// the embedded constant control context) for a latent of shape `dims` and caption `cap_feats`
    /// under control context `cc`. A denoise loop builds this once and reuses it every step via
    /// [`forward_with_control`](Self::forward_with_control) (F-042).
    pub(crate) fn prepare_control(
        &self,
        dims: (i32, i32, i32, i32),
        cap_feats: &Array,
        cc: &Array,
    ) -> Result<ControlPrepared> {
        let cfg = &self.base.cfg;
        let meta = self.base.patchify_meta(dims, cap_feats)?;
        let x_freqs = self.base.rope.forward(&meta.x_pos_ids)?;
        let cap_freqs = self.base.rope.forward(&meta.cap_pos_ids)?;
        let unified_freqs = concatenate_axis(&[&x_freqs, &cap_freqs], 0)?;

        // Control context embedding (the constant the fork re-patchified every step). The control
        // patch embedder runs at the control context's dtype and the embedding is NOT recast —
        // exactly like the fork (`c_emb = control_all_x_embedder(c_tokens)`, no cast). The fork feeds
        // `control_context` as **f32** (only the latents + cap_feats are bf16), so the whole control
        // branch promotes to f32, and its f32 hints promote the bf16 image/caption stream to f32 when
        // added — i.e. the control path is **mixed precision** (bf16 base embeddings, f32 control
        // branch), NOT pure bf16. Reproducing that dtype flow is what matches the fork (sc-2720);
        // forcing the branch to bf16 diverges (~1.6% px>8 vs ~0.04%).
        //
        // (This GEMM was once forced to f32 to dodge the pmetal NAX 16-bit dense-GEMM miscompile — a
        // Metal compile-target bug fixed at the macOS-26.2 target, sc-2772. That forcing is gone; the
        // dtype now flows from the caller's `control_context`, which is f32, so the branch is f32
        // regardless — but it is no longer *forced*, and a bf16 control_context would run bf16.)
        let c_tokens = patchify_control(cc, cfg.patch_size, cfg.f_patch_size)?;
        let mut c_emb = self.control_x_embedder.forward(&c_tokens)?;
        c_emb = apply_pad(&c_emb, &meta.x_keep, &self.base.x_pad_token)?;
        let c_emb = c_emb.expand_dims(0)?;

        Ok(ControlPrepared {
            cap_tokens: meta.cap_tokens,
            x_size: meta.x_size,
            x_keep: meta.x_keep,
            cap_keep: meta.cap_keep,
            img_pad: meta.img_pad,
            x_freqs,
            cap_freqs,
            unified_freqs,
            c_emb,
        })
    }

    /// Single-shot control forward: build the [`ControlPrepared`] and run one step. Bit-identical to
    /// caching `prepare_control` across a denoise loop (the prep depends only on the loop-constant
    /// dims + caption + control context). `cap`, when `Some`, collects the named per-stage
    /// intermediates for the capture variant.
    #[allow(clippy::too_many_arguments)]
    fn forward_control(
        &self,
        x: &Array,
        timestep: f32,
        cap_feats: &Array,
        cc: &Array,
        control_context_scale: f32,
        cap: Option<&mut Vec<(&'static str, Array)>>,
        plan: AttentionPlan<'_>,
    ) -> Result<Array> {
        let sh = x.shape();
        let prep = self.prepare_control((sh[0], sh[1], sh[2], sh[3]), cap_feats, cc)?;
        self.forward_with_control_budgeted(&prep, x, timestep, control_context_scale, cap, plan)
    }

    /// Run the dual-injection control DiT for one denoise step against a cached [`ControlPrepared`].
    /// Only the latent values and timestep vary per step; the metadata, freqs, and control embedding
    /// come from `prep`. `cap`, when `Some`, records the named per-stage intermediates.
    /// [`forward_with_control`](Self::forward_with_control) with an explicit attention-score budget —
    /// ladder rung 3 (SC-15615) on the ControlNet route. The budget reaches **both** stacks: the base
    /// DiT's noise/context refiners and main layers, and the parallel control branch's own blocks
    /// (whose `base` block runs the identical attention over the same unified sequence). The
    /// before/after projections, hint injection, `control_context_scale`, and the dual-injection
    /// schedule are untouched.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_control_budgeted(
        &self,
        prep: &ControlPrepared,
        x: &Array,
        timestep: f32,
        control_context_scale: f32,
        cap: Option<&mut Vec<(&'static str, Array)>>,
        plan: AttentionPlan<'_>,
    ) -> Result<Array> {
        self.forward_with_control_rungs(prep, x, timestep, control_context_scale, cap, plan, None)
    }

    /// [`forward_with_control_budgeted`](Self::forward_with_control_budgeted) with ladder rung 4
    /// (SC-15754) layered on.
    ///
    /// **Both** stacks are bounded when `window` is `Some`: the 15-block parallel control stack that
    /// produces the hints, and the base DiT's 30 unified blocks that consume them. The dual-injection
    /// schedule itself is untouched — `CONTROL_LAYERS_PLACES`, the `before_proj` seed, the per-block
    /// `after_proj` hints and `control_context_scale` all behave exactly as on the resident path, so
    /// the residual injected into base layer *i* is the same value regardless of window size.
    ///
    /// The control hints are computed for the whole stack **before** the base loop runs (that is the
    /// fork's schedule, not a choice made here), so windowing the control stack bounds its *weights*
    /// while its 15 hint activations stay live either way. Rung 3 is what bounds those.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_control_rungs(
        &self,
        prep: &ControlPrepared,
        x: &Array,
        timestep: f32,
        control_context_scale: f32,
        mut cap: Option<&mut Vec<(&'static str, Array)>>,
        plan: AttentionPlan<'_>,
        window: Option<crate::block_stream::BlockWindow<'_>>,
    ) -> Result<Array> {
        macro_rules! record {
            ($name:expr, $a:expr) => {
                if let Some(c) = cap.as_deref_mut() {
                    c.push(($name, $a.clone()));
                }
            };
        }

        let cfg = &self.base.cfg;
        let t = Array::from_slice(&[timestep * cfg.t_scale], &[1]);
        let t_emb = self.base.t_embedder.forward(&t)?;
        record!("t_emb", t_emb);

        let x_tokens = self.base.patchify_image_tokens(x, prep.img_pad)?;
        record!("x_tokens", x_tokens); // pre-embed patchify output (the x-embedder input)

        // Image stream: embed → set padded positions to x_pad_token → (control refiner) → noise refiner.
        let mut x_emb = self.base.x_embedder.forward(&x_tokens)?;
        x_emb = apply_pad(&x_emb, &prep.x_keep, &self.base.x_pad_token)?;
        let mut x_emb = x_emb.expand_dims(0)?;
        record!("x_emb", x_emb);

        // Control refiner pass: reuse the cached control embedding, run the parallel control refiner,
        // collect the per-block hints + threaded state.
        record!("c_emb", prep.c_emb);
        let (refiner_hints, threaded_control) = self.run_control_blocks(
            &self.control_noise_refiner,
            prep.c_emb.clone(),
            &x_emb,
            &prep.x_freqs,
            &t_emb,
            plan,
        )?;
        record!("refiner_hint0", refiner_hints[0]);
        record!("refiner_hint1", refiner_hints[1]);
        record!("threaded", threaded_control);

        // Noise refiner (with control hints).
        for (i, layer) in self.base.noise_refiner.iter().enumerate() {
            x_emb = layer.forward_budgeted(&x_emb, &prep.x_freqs, &t_emb, plan)?;
            if let Some(n) = hint_index(&CONTROL_REFINER_PLACES, i) {
                x_emb = add_hint(&x_emb, &refiner_hints[n], control_context_scale)?;
            }
        }
        record!("x_refined", x_emb);

        // Caption stream: RMSNorm → linear → set padded to cap_pad_token → context refiner.
        let cap_normed = rms_norm(&prep.cap_tokens, &self.base.cap_norm_w, cfg.norm_eps)?;
        let mut cap_emb = self.base.cap_linear.forward(&cap_normed)?;
        cap_emb = apply_pad(&cap_emb, &prep.cap_keep, &self.base.cap_pad_token)?;
        let mut cap_emb = cap_emb.expand_dims(0)?;
        for layer in &self.base.context_refiner {
            cap_emb = layer.forward_budgeted(&cap_emb, &prep.cap_freqs, plan)?;
        }
        record!("cap_refined", cap_emb);

        // Unify image + caption.
        let x_len = x_emb.shape()[1];
        let unified = concatenate_axis(&[&x_emb, &cap_emb], 1)?;

        // Main control pass: thread the (refined) control state + caption through the 15 control
        // layers to produce the hints for the unified main loop.
        let control_unified = concatenate_axis(&[&threaded_control, &cap_emb], 1)?;
        let (main_hints, _) = match window {
            None => self.run_control_blocks(
                &self.control_layers,
                control_unified,
                &unified,
                &prep.unified_freqs,
                &t_emb,
                plan,
            )?,
            Some(window) => self.run_control_blocks_windowed(
                control_unified,
                &unified,
                &prep.unified_freqs,
                &t_emb,
                plan,
                window,
            )?,
        };
        record!("main_hint0", main_hints[0]);
        record!("main_hint_last", main_hints[main_hints.len() - 1]);

        // Main layers (with control hints).
        let mut unified = unified;
        match window {
            None => {
                for (i, layer) in self.base.layers.iter().enumerate() {
                    unified =
                        layer.forward_budgeted(&unified, &prep.unified_freqs, &t_emb, plan)?;
                    if let Some(n) = hint_index(&CONTROL_LAYERS_PLACES, i) {
                        unified = add_hint(&unified, &main_hints[n], control_context_scale)?;
                    }
                }
            }
            Some(window) => {
                unified = self.base.run_windowed_layers_with_hints(
                    unified,
                    &prep.unified_freqs,
                    &t_emb,
                    plan,
                    window,
                    &mut |state: &Array, i: usize| match hint_index(&CONTROL_LAYERS_PLACES, i) {
                        Some(n) => add_hint(state, &main_hints[n], control_context_scale),
                        None => Ok(state.clone()),
                    },
                )?;
            }
        }
        record!("unified_main", unified);

        // Final layer + unpatchify (only the real image tokens survive), negate the velocity.
        let unified = self.base.final_layer.forward(&unified, &t_emb)?;
        let embed_dim = unified.shape()[2];
        let head = unified
            .reshape(&[unified.shape()[1], embed_dim])?
            .take_axis(row_indices(x_len), 0)?;
        let out = self.base.unpatchify(&head, prep.x_size)?;
        Ok(out.multiply(scalar(-1.0))?)
    }

    /// Run a parallel control stack, returning `(per-block hints, threaded control state)`.
    /// Mirrors the VACE threading: block 0 seeds the branch via `before_proj(c) + x_base`; each
    /// block runs the base-block forward and emits `after_proj(c)` as its hint, passing the running
    /// control state `c` to the next block.
    fn run_control_blocks(
        &self,
        blocks: &[ZImageControlBlock],
        c: Array,
        x_base: &Array,
        freqs_cis: &Array,
        t_emb: &Array,
        plan: AttentionPlan<'_>,
    ) -> Result<(Vec<Array>, Array)> {
        let mut c = c;
        let mut hints = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            if i == 0 {
                let bp = block.before_proj().ok_or_else(|| {
                    Error::Msg("z-image control block 0 is missing before_proj".into())
                })?;
                c = add(&bp.forward(&c)?, x_base)?;
            }
            c = block.base.forward_budgeted(&c, freqs_cis, t_emb, plan)?;
            hints.push(block.after_proj().forward(&c)?);
        }
        Ok((hints, c))
    }

    /// [`run_control_blocks`](Self::run_control_blocks) for the main control stack under a residency
    /// window (SC-15754) — same VACE threading, same `before_proj` seed on block 0, same per-block
    /// `after_proj` hint, but each block is materialized from the ControlNet checkpoint and released
    /// with its window.
    ///
    /// The carried state a window must materialize before releasing is BOTH the threaded control
    /// state `c` **and** the hints produced so far: a hint is `after_proj(c)`, an unevaluated node
    /// still referencing that block's projection weights, so evaluating only `c` would leave the
    /// hints holding the window alive and the bound would silently not hold. `run_windowed`'s
    /// `materialize` hook receives the whole carried tuple, which is exactly why it is a closure over
    /// the state rather than a fixed `eval` of one array.
    fn run_control_blocks_windowed(
        &self,
        c: Array,
        x_base: &Array,
        freqs_cis: &Array,
        t_emb: &Array,
        plan: AttentionPlan<'_>,
        window: crate::block_stream::BlockWindow<'_>,
    ) -> Result<(Vec<Array>, Array)> {
        let stream = self.control_block_stream.as_ref().ok_or_else(|| {
            Error::Msg(
                "z-image control: bounded transformer residency was requested but the ControlNet \
                 checkpoint has no re-openable weights source"
                    .to_owned(),
            )
        })?;
        // The control stack is shorter than the base stack (15 vs 30), so it gets its own plan at the
        // same window size rather than the base plan — `BlockPlan::new` clamps a window larger than
        // the stack, so a 30-wide window degenerates to one control window rather than erroring.
        // The same invariant the base stack asserts (`run_windowed_layers_with_hints`): the plan, the
        // stream and the resident stack must describe the same blocks. Both derive from
        // `CONTROL_LAYERS_PLACES.len()` today so they cannot diverge — which is exactly why an
        // assertion is cheap, and why dropping it on one of the two arms is the asymmetry worth
        // closing rather than the risk worth taking.
        if stream.n_blocks() != self.control_layers.len() {
            return Err(Error::Msg(format!(
                "z-image control: the block stream covers {} blocks but the control stack has {}",
                stream.n_blocks(),
                self.control_layers.len()
            )));
        }
        let control_plan = mlx_gen::block_residency::BlockPlan::new(
            self.control_layers.len(),
            window.plan.window(),
        )?;
        let (hints, c) = mlx_gen::block_residency::run_windowed(
            &control_plan,
            window.cancel,
            (Vec::<Array>::with_capacity(self.control_layers.len()), c),
            || stream.open(),
            |(mut hints, mut c), view, range| {
                for i in range {
                    let block = stream.materialize_control(view, i)?;
                    if i == 0 {
                        let bp = block.before_proj().ok_or_else(|| {
                            Error::Msg("z-image control block 0 is missing before_proj".into())
                        })?;
                        c = add(&bp.forward(&c)?, x_base)?;
                    }
                    c = block.base.forward_budgeted(&c, freqs_cis, t_emb, plan)?;
                    hints.push(block.after_proj().forward(&c)?);
                }
                Ok((hints, c))
            },
            |(hints, c): &(Vec<Array>, Array)| {
                Ok(mlx_rs::transforms::eval(
                    hints.iter().chain(std::iter::once(c)),
                )?)
            },
        )?;
        Ok((hints, c))
    }
}

impl AdaptableHost for ZImageControlTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Adapters target the base DiT — the fork trains LoRA/LoKr on the base transformer, not the
        // control branch (`control_x_embedder` / `control_layers` / `control_noise_refiner` are
        // ControlNet weights, never adapter targets). Delegate the whole path to the composed base.
        self.base.adaptable_mut(path)
    }
}

/// `x + hint · scale`, broadcasting the scalar scale over `[1, seq, dim]`. Fused into one kernel
/// (multiply + add) when the sc-2963 glue toggle is on. The `scale` is passed as an array arg (not
/// baked into the trace), and dtype is preserved — bit-identical to the eager form. The f32 `hint`
/// promotes the bf16 base stream to f32 here exactly as before (the sc-2720 mixed-precision flow).
fn add_hint(x: &Array, hint: &Array, scale: f32) -> Result<Array> {
    let sc = scalar(scale);
    let f = |(x, h, sc): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(x, &multiply(h, sc)?)
    };
    if crate::compile_glue() {
        Ok(compile(f, true)((x, hint, &sc))?)
    } else {
        Ok(f((x, hint, &sc))?)
    }
}

/// The control-stack hint index for base-layer `i` (the fork's `*_mapping[i]`), or `None` when no
/// control block injects there.
fn hint_index(places: &[usize], i: usize) -> Option<usize> {
    places.iter().position(|&p| p == i)
}

/// Patchify the `(C=33, F, H, W)` control context into `(seq, p²·pf·33)` tokens, padded to a
/// multiple of 32 exactly like the base image patchify so the control sequence aligns 1:1 with the
/// image tokens (shared RoPE / pad mask). Padded rows are zeroed (the embedder output for them is
/// overwritten by `x_pad_token` in the forward, matching the base patchify's zero-pad convention).
fn patchify_control(cc: &Array, patch_size: i32, f_patch_size: i32) -> Result<Array> {
    let (pf, ph, pw) = (f_patch_size, patch_size, patch_size);
    let sh = cc.shape();
    let (c, f, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    // The control context is the `[control_latent(16) | mask(1) | inpaint(16)]` channel-concat; the
    // control patch embedder is built for exactly this width, so a mismatch is a caller bug, not a
    // silently-wrong run. Anchors `CONTROL_IN_DIM` as load-bearing.
    if c != CONTROL_IN_DIM {
        return Err(Error::Msg(format!(
            "z-image control_context must have {CONTROL_IN_DIM} channels, got {c}"
        )));
    }
    let (ft, ht, wt) = (f / pf, h / ph, w / pw);
    let tokens = cc
        .reshape(&[c, ft, pf, ht, ph, wt, pw])?
        .transpose_axes(&[1, 3, 5, 2, 4, 6, 0])?
        .reshape(&[ft * ht * wt, pf * ph * pw * c])?;
    let ori = ft * ht * wt;
    let pad = (-(ori as i64)).rem_euclid(32) as i32;
    crate::transformer::pad_rows(&tokens, pad)
}

#[cfg(test)]
mod sc2963 {
    use super::*;
    use mlx_rs::{random, Dtype};

    // sc-2720 guard: `add_hint` mixes a bf16 base stream `x` with an f32 `hint` — the f32 hint must
    // promote the result to f32 (the fork's mixed-precision flow), and compiled must equal eager.
    #[test]
    fn compiled_add_hint_mixed_precision_matches_eager() {
        let k = random::key(0).unwrap();
        let x = random::normal::<f32>(&[1, 16, 64], None, None, Some(&k))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap(); // bf16 base stream
        let hint = random::normal::<f32>(&[1, 16, 64], None, None, Some(&k)).unwrap(); // f32 control hint
        crate::set_compile_glue(false);
        let e = add_hint(&x, &hint, 0.8).unwrap();
        crate::set_compile_glue(true);
        let c = add_hint(&x, &hint, 0.8).unwrap();
        crate::set_compile_glue(false);
        assert_eq!(e.dtype(), Dtype::Float32, "f32 hint promotes bf16 → f32");
        assert_eq!(
            c.dtype(),
            Dtype::Float32,
            "compiled preserves the promotion"
        );
        let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(&c, &e).unwrap()).unwrap();
        let m = mlx_rs::ops::max(&d, None).unwrap().item::<f32>();
        assert_eq!(m, 0.0, "add_hint compiled vs eager");
    }
}
