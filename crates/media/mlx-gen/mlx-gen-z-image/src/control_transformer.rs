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
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::transforms::compile::{compile, compile_retained};
use mlx_rs::Array;

use crate::control_transformer_block::ZImageControlBlock;
use crate::transformer::{apply_pad, ZImageTransformer};
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
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
    pub(crate) c_emb: Array,
}

/// Channel count of the VAE-encoded control context (16 control latent + 1 mask + 16 inpaint).
pub const CONTROL_IN_DIM: i32 = 33;
/// Base `layers` indices the 15 control layers inject into (the fork's `CONTROL_LAYERS_PLACES`).
pub(crate) const CONTROL_LAYERS_PLACES: [usize; 15] =
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
/// Base `noise_refiner` indices the 2 control refiner blocks inject into.
pub(crate) const CONTROL_REFINER_PLACES: [usize; 2] = [0, 1];

const SITE_ADD_HINT: &str = "z_image::control_transformer::add_hint";

fn add_hint_impl(
    (x, hint, scale): (&Array, &Array, &Array),
) -> std::result::Result<Array, Exception> {
    add(x, &multiply(hint, scale)?)
}

thread_local! {
    static RETAINED_ADD_HINT: std::cell::RefCell<Option<mlx_gen::nn::RetainedTernary>> =
        const { std::cell::RefCell::new(None) };
}

fn retained_add_hint(args: (&Array, &Array, &Array)) -> std::result::Result<Array, Exception> {
    mlx_gen::nn::prepare_retained_compilation_thread();
    RETAINED_ADD_HINT.with(|slot| {
        slot.borrow_mut()
            .get_or_insert_with(|| {
                mlx_gen::nn::RetainedTernary::new(compile_retained(add_hint_impl, true))
            })
            .call(SITE_ADD_HINT, args)
    })
}

/// Exercise this module's production retained handle once for the release memory audit.
#[doc(hidden)]
pub fn exercise_retained_compile_inventory(input: &Array) -> Result<()> {
    let output = retained_add_hint((input, input, input))?;
    output.eval()?;
    drop(output);
    Ok(())
}

pub struct ZImageControlTransformer {
    pub(crate) base: ZImageTransformer,
    /// `control_all_x_embedder.{patch_size}-{f_patch_size}` — `33·p²·pf → dim`. Kept dense under
    /// Q8 (its in-features, e.g. 132, is not divisible by the group size 64).
    control_x_embedder: AdaptableLinear,
    /// The 15-block main control stack (injects into `base.layers` at `CONTROL_LAYERS_PLACES`).
    pub(crate) control_layers: Vec<ZImageControlBlock>,
    /// The 2-block control refiner (injects into `base.noise_refiner` at `CONTROL_REFINER_PLACES`).
    control_noise_refiner: Vec<ZImageControlBlock>,
    /// Ladder rung 4 (SC-15754) for the 15-block main control stack. The 2-block refiner is left
    /// resident: at ~97 MiB/block it is 2 blocks against the base stack's 30, so windowing it would
    /// buy ~6% of the stack's weights for another 2 re-reads per step.
    control_block_stream: Option<crate::block_stream::ZImageBlockStream>,
    /// The 15 main blocks were left lazy by a deferred-materialization load (`control_layers` holds
    /// unread handles; every forward streams them through `control_block_stream`). Such a load must
    /// not pack them at [`Self::quantize`]: packing would read and hold a packed copy of every
    /// block that no windowed forward uses (on iOS q8, ~3 GiB beside the per-block stream).
    deferred_main: bool,
    /// The load-time bits a deferred stack's lazy main blocks were NOT packed at: a forward that
    /// would run those resident blocks (no window) is refused rather than run dense.
    deferred_unpacked_bits: Option<i32>,
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
            deferred_main: false,
            deferred_unpacked_bits: None,
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
    /// The 2-block control refiner (always resident).
    pub(crate) fn control_noise_refiner(&self) -> &[ZImageControlBlock] {
        &self.control_noise_refiner
    }

    /// Mark the 15 main blocks as left lazy by a deferred-materialization load (see `deferred_main`).
    pub(crate) fn with_deferred_main_blocks(mut self) -> Self {
        self.deferred_main = true;
        self
    }

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
        // The ControlNet checkpoint is published dense only, so on a packed tier its resident blocks
        // are packed here, at load. iOS: one bounded quantum per block (its dense bytes are already
        // resident; the packed copies are checked before they exist and materialized before the
        // next block, so the dense source is released block by block). A deferred load's lazy main
        // blocks are not resident: each window packs its own block (the stream replays `bits`).
        let main: &mut [ZImageControlBlock] = if self.deferred_main {
            self.deferred_unpacked_bits = Some(bits);
            &mut []
        } else {
            &mut self.control_layers
        };
        for block in main.iter_mut().chain(self.control_noise_refiner.iter_mut()) {
            block.quantize_quantum(bits)?;
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
        let quanta = mlx_gen::ios_quantum::enabled();
        if quanta {
            mlx_gen::ios_quantum::before(
                crate::quantum::DIT_PREPARE,
                self.base.prepare_bytes(dims, cap_feats),
            )?;
        }
        let meta = self.base.patchify_meta(dims, cap_feats)?;
        let x_freqs = self.base.rope.forward(&meta.x_pos_ids)?;
        let cap_freqs = self.base.rope.forward(&meta.cap_pos_ids)?;
        let unified_freqs = concatenate_axis(&[&x_freqs, &cap_freqs], 0)?;
        if quanta {
            mlx_gen::ios_quantum::finish(
                crate::quantum::DIT_PREPARE,
                &[
                    &meta.cap_tokens,
                    &meta.x_keep,
                    &meta.cap_keep,
                    &x_freqs,
                    &cap_freqs,
                    &unified_freqs,
                ],
            )?;
            mlx_gen::ios_quantum::before(
                crate::quantum::CONTROL_EMBED,
                self.control_embed_bytes(cc, meta.img_pad),
            )?;
        }

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
        if quanta {
            mlx_gen::ios_quantum::finish(crate::quantum::CONTROL_EMBED, &[&c_emb])?;
        }

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

    /// The control embed quantum's counted price over control context `cc` (`[33, F, H, W]`): the
    /// patch tokens' transposed copy (and, when the count pads, the f32 zero rows and padded
    /// concatenation); `control_all_x_embedder`'s output and the pad merge's two products (`[N, D]`
    /// at their promoted widths); the dense embedder's stream-width copy.
    pub(crate) fn control_embed_bytes(&self, cc: &Array, img_pad: i32) -> u64 {
        use crate::quantum::{ab, u};
        let cfg = &self.base.cfg;
        let sh = cc.shape();
        let n0 =
            u((sh[1] / cfg.f_patch_size) * (sh[2] / cfg.patch_size) * (sh[3] / cfg.patch_size));
        let n = n0 + u(img_pad);
        let patch = u(cfg.f_patch_size * cfg.patch_size * cfg.patch_size * sh[0]);
        let cw = cc.item_size() as u64;
        let tokens = ab(cw, &[n0, patch])
            + if img_pad > 0 {
                ab(4, &[u(img_pad), patch]) + ab(4, &[n, patch])
            } else {
                0
            };
        let emb = crate::quantum::linear_out_dtype(&self.control_x_embedder, cc.dtype());
        tokens
            .saturating_add(ab(mlx_gen::ios_quantum::dtype_bytes(emb), &[n, u(cfg.dim)]))
            .saturating_add(
                2 * ab(
                    crate::quantum::pad_width(emb, &self.base.x_pad_token),
                    &[n, u(cfg.dim)],
                ),
            )
            .saturating_add(crate::quantum::promoted_weight_bytes(
                &[&self.control_x_embedder],
                cw.max(4),
            ))
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

        // iOS: every stage is one bounded quantum (see `ZImageTransformer::forward_with_rungs`);
        // pass-throughs off device.
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let backend = mlx_gen::ios_quantum::Backend::current;
        let quanta = mlx_gen::ios_quantum::enabled();
        let cfg = &self.base.cfg;
        if quanta {
            mlx_gen::ios_quantum::before(
                crate::quantum::DIT_EMBED,
                self.base.embed_bytes(x, prep.img_pad, backend()),
            )?;
        }
        let t = Array::from_slice(&[timestep * cfg.t_scale], &[1]);
        let t_emb = self.base.t_embedder.forward(&t)?;
        record!("t_emb", t_emb);

        let x_tokens = self.base.patchify_image_tokens(x, prep.img_pad)?;
        record!("x_tokens", x_tokens); // pre-embed patchify output (the x-embedder input)

        // Image stream: embed → set padded positions to x_pad_token → (control refiner) → noise refiner.
        let mut x_emb = self.base.x_embedder.forward(&x_tokens)?;
        x_emb = apply_pad(&x_emb, &prep.x_keep, &self.base.x_pad_token)?;
        let mut x_emb = x_emb.expand_dims(0)?;
        if quanta {
            mlx_gen::ios_quantum::finish(crate::quantum::DIT_EMBED, &[&t_emb, &x_emb])?;
        }
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

        // Noise refiner (with control hints, each added inside its block's quantum).
        for (i, layer) in self.base.noise_refiner.iter().enumerate() {
            let hint = hint_index(&CONTROL_REFINER_PLACES, i);
            x_emb = layer.forward_quantum(
                &x_emb,
                &prep.x_freqs,
                &t_emb,
                plan,
                hint.map_or(0, |_| add_hint_bytes(&x_emb)),
                |out| match hint {
                    Some(n) => add_hint(&out, &refiner_hints[n], control_context_scale),
                    None => Ok(out),
                },
            )?;
        }
        record!("x_refined", x_emb);

        // Caption stream: RMSNorm → linear → set padded to cap_pad_token → context refiner.
        let mut cap_emb = self
            .base
            .caption_quantum_parts(&prep.cap_tokens, &prep.cap_keep)?;
        for layer in &self.base.context_refiner {
            cap_emb = layer.forward_quantum(&cap_emb, &prep.cap_freqs, plan)?;
        }
        record!("cap_refined", cap_emb);

        // Unify image + caption.
        let x_len = x_emb.shape()[1];
        let unified = self.base.unify_quantum(&x_emb, &cap_emb)?;

        // Main control pass: thread the (refined) control state + caption through the 15 control
        // layers to produce the hints for the unified main loop.
        let control_unified = self.base.unify_quantum(&threaded_control, &cap_emb)?;
        let (main_hints, _) =
            match window {
                None if self.deferred_unpacked_bits.is_some() => return Err(Error::Unsupported(
                    "z-image control: a deferred-materialization load streams its control blocks \
                     (they are packed per window); run it windowed"
                        .to_owned(),
                )),
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
                    let hint = hint_index(&CONTROL_LAYERS_PLACES, i);
                    unified = layer.forward_quantum(
                        &unified,
                        &prep.unified_freqs,
                        &t_emb,
                        plan,
                        hint.map_or(0, |_| add_hint_bytes(&unified)),
                        |out| match hint {
                            Some(n) => add_hint(&out, &main_hints[n], control_context_scale),
                            None => Ok(out),
                        },
                    )?;
                }
            }
            Some(window) => {
                unified = self.base.run_windowed_layers_with_hints(
                    unified,
                    &prep.unified_freqs,
                    &t_emb,
                    plan,
                    window,
                    &|i: usize, state: &Array| {
                        hint_index(&CONTROL_LAYERS_PLACES, i).map_or(0, |_| add_hint_bytes(state))
                    },
                    &mut |state: &Array, i: usize| match hint_index(&CONTROL_LAYERS_PLACES, i) {
                        Some(n) => add_hint(state, &main_hints[n], control_context_scale),
                        None => Ok(state.clone()),
                    },
                )?;
            }
        }
        record!("unified_main", unified);

        // Final layer + unpatchify (only the real image tokens survive), negate the velocity.
        self.base
            .output_quantum(&unified, &t_emb, x_len, prep.x_size)
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
            let (next, hint) =
                control_block_quantum(block, i, c, x_base, freqs_cis, t_emb, plan, 0)?;
            c = next;
            hints.push(hint);
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
                    // The block's stored bytes (and its replayed packing) are read inside its
                    // quantum, which is priced with them.
                    let (block, read_bytes) = stream.materialize_control(view, i)?;
                    let (next, hint) = control_block_quantum(
                        &block, i, c, x_base, freqs_cis, t_emb, plan, read_bytes,
                    )?;
                    c = next;
                    hints.push(hint);
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

    fn adaptable_paths(&self) -> Vec<String> {
        self.base.adaptable_paths()
    }
}

/// One ControlNet block as one iOS quantum: (block 0) the `before_proj` seed added to the base
/// stream, the base block forward over the control state, and the block's `after_proj` hint.
/// Returns `(threaded control state, hint)`, both materialized on device. `extra_bytes` is a streamed
/// block's read (and replayed packing). Priced by [`ZImageControlBlock::quantum_scratch`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn control_block_quantum(
    block: &ZImageControlBlock,
    index: usize,
    c: Array,
    x_base: &Array,
    freqs_cis: &Array,
    t_emb: &Array,
    plan: AttentionPlan<'_>,
    extra_bytes: u64,
) -> Result<(Array, Array)> {
    let quanta = mlx_gen::ios_quantum::enabled();
    if quanta {
        let scratch = block.quantum_scratch(
            &c,
            x_base,
            freqs_cis,
            plan,
            mlx_gen::ios_quantum::Backend::current(),
        );
        mlx_gen::ios_quantum::before(
            crate::quantum::CONTROL_BLOCK,
            scratch.saturating_add(extra_bytes),
        )?;
    }
    let mut c = c;
    if index == 0 {
        let bp = block
            .before_proj()
            .ok_or_else(|| Error::Msg("z-image control block 0 is missing before_proj".into()))?;
        c = add(&bp.forward(&c)?, x_base)?;
    }
    let c = block.base.forward_budgeted(&c, freqs_cis, t_emb, plan)?;
    let hint = block.after_proj().forward(&c)?;
    if quanta {
        mlx_gen::ios_quantum::finish(crate::quantum::CONTROL_BLOCK, &[&c, &hint])?;
    }
    Ok((c, hint))
}

/// What [`add_hint`] adds to a block's quantum over the base stream `x`: the scaled hint and the
/// sum (`[1, L, D]`, f32 — the f32 hint promotes; one fused buffer under compiled glue).
pub(crate) fn add_hint_bytes(x: &Array) -> u64 {
    let sh = x.shape();
    let hidden = crate::quantum::tb(
        4,
        &[
            crate::quantum::u(sh[0]),
            crate::quantum::u(sh[1]),
            crate::quantum::u(sh[2]),
        ],
    );
    let units = if crate::compile_glue() { 1 } else { 2 };
    hidden * units + mlx_gen::ios_quantum::page_slack(units)
}

/// `x + hint · scale`, broadcasting the scalar scale over `[1, seq, dim]`. Fused into one kernel
/// (multiply + add) when the sc-2963 glue toggle is on. The `scale` is passed as an array arg (not
/// baked into the trace), and dtype is preserved — bit-identical to the eager form. The f32 `hint`
/// promotes the bf16 base stream to f32 here exactly as before (the sc-2720 mixed-precision flow).
fn add_hint(x: &Array, hint: &Array, scale: f32) -> Result<Array> {
    let sc = scalar(scale);
    if crate::compile_glue() {
        if mlx_gen::nn::retained_compilation_requested() {
            Ok(retained_add_hint((x, hint, &sc))?)
        } else {
            mlx_gen::diagnostics::record_compile(
                SITE_ADD_HINT,
                mlx_gen::diagnostics::CompileDisposition::OneShot,
            );
            Ok(compile(add_hint_impl, true)((x, hint, &sc))?)
        }
    } else {
        mlx_gen::diagnostics::record_fallback(SITE_ADD_HINT, "compiled_glue_disabled");
        Ok(add_hint_impl((x, hint, &sc))?)
    }
}

/// The control-stack hint index for base-layer `i` (the fork's `*_mapping[i]`), or `None` when no
/// control block injects there.
pub(crate) fn hint_index(places: &[usize], i: usize) -> Option<usize> {
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

#[cfg(all(test, target_os = "macos"))]
mod adapter_namespace_proofs {
    use super::*;
    use crate::transformer::ZImageTransformerConfig;
    use mlx_gen::adapters::Adapter;
    use mlx_gen::runtime::{AdapterKind, AdapterSpec};
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Write;

    const CHILD: &str = "ZIMAGE_CONTROL_ADAPTER_NAMESPACE_CPU_CHILD";
    const DONE: &str = "ZIMAGE_CONTROL_ADAPTER_NAMESPACE_CPU_DONE";
    const FQN: &str = "control_transformer::adapter_namespace_proofs::control_host_exposes_base_kohya_paths_and_strict_adapter_batch";

    struct Completion;
    impl Drop for Completion {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(
                    unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                    0,
                    "Control namespace CPU synchronization after fixture owners dropped"
                );
                std::fs::write(std::env::var(DONE).expect("fresh completion path"), b"ok")
                    .expect("write Control namespace completion");
            }
        }
    }

    type Shapes = BTreeMap<String, Vec<usize>>;
    fn linear(out: &mut Shapes, base: &str, rows: usize, cols: usize, bias: bool) {
        assert!(out
            .insert(format!("{base}.weight"), vec![rows, cols])
            .is_none());
        if bias {
            assert!(out.insert(format!("{base}.bias"), vec![rows]).is_none());
        }
    }
    fn block(out: &mut Shapes, prefix: &str, dim: usize, heads: usize, modulation: bool) {
        for path in [
            "attention.to_q",
            "attention.to_k",
            "attention.to_v",
            "attention.to_out.0",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        ] {
            linear(out, &format!("{prefix}.{path}"), dim, dim, false);
        }
        for (path, width) in [
            ("attention.norm_q.weight", dim / heads),
            ("attention.norm_k.weight", dim / heads),
            ("attention_norm1.weight", dim),
            ("attention_norm2.weight", dim),
            ("ffn_norm1.weight", dim),
            ("ffn_norm2.weight", dim),
        ] {
            assert!(out
                .insert(format!("{prefix}.{path}"), vec![width])
                .is_none());
        }
        if modulation {
            linear(
                out,
                &format!("{prefix}.adaLN_modulation.0"),
                4 * dim,
                dim,
                true,
            );
        }
    }

    /// Real file-backed model handles. Only adapter files carry small, NONZERO rank-one F32
    /// payloads; parent model files have sparse BF16 extents and are never materialized here.
    fn tensor_file(
        root: &std::path::Path,
        name: &str,
        shapes: &Shapes,
        adapter_payload: bool,
    ) -> std::path::PathBuf {
        let width: u64 = if adapter_payload { 4 } else { 2 };
        let mut offset = 0u64;
        let mut header = serde_json::Map::new();
        for (key, shape) in shapes {
            let bytes = shape
                .iter()
                .try_fold(width, |n, &dim| n.checked_mul(dim as u64))
                .unwrap();
            header.insert(
                key.clone(),
                serde_json::json!({"dtype": if adapter_payload { "F32" } else { "BF16" },
                "shape": shape, "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let path = root.join(format!("{name}.safetensors"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        if adapter_payload {
            for _ in 0..offset / 4 {
                file.write_all(&0.125_f32.to_le_bytes()).unwrap();
            }
        } else {
            file.set_len(8 + header.len() as u64 + offset).unwrap();
        }
        path
    }

    fn actual_control(
        root: &std::path::Path,
        cfg: &ZImageTransformerConfig,
    ) -> ZImageControlTransformer {
        let d = cfg.dim as usize;
        let patch_in =
            (cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.f_patch_size) as usize;
        let mut base = Shapes::new();
        linear(&mut base, "all_x_embedder.2-1", d, patch_in, true);
        base.insert(
            "cap_embedder.0.weight".into(),
            vec![cfg.cap_feat_dim as usize],
        );
        linear(
            &mut base,
            "cap_embedder.1",
            d,
            cfg.cap_feat_dim as usize,
            true,
        );
        linear(
            &mut base,
            "t_embedder.linear1",
            d,
            cfg.frequency_embedding_size as usize,
            true,
        );
        linear(&mut base, "t_embedder.linear2", d, d, true);
        linear(&mut base, "all_final_layer.2-1.linear", patch_in, d, true);
        linear(
            &mut base,
            "all_final_layer.2-1.adaLN_modulation.0",
            d,
            d,
            true,
        );
        base.insert("x_pad_token".into(), vec![d]);
        base.insert("cap_pad_token".into(), vec![d]);
        for (stack, count, modulation) in [
            ("layers", cfg.n_layers, true),
            ("noise_refiner", cfg.n_refiner_layers, true),
            ("context_refiner", cfg.n_refiner_layers, false),
        ] {
            for index in 0..count {
                block(
                    &mut base,
                    &format!("{stack}.{index}"),
                    d,
                    cfg.n_heads as usize,
                    modulation,
                );
            }
        }
        let source =
            Weights::from_file(tensor_file(root, "base-model-metadata", &base, false)).unwrap();
        // This is the production constructor, including its real small weightless RoPE constants.
        // Those constants are not fake carriers, not claimed allocation-free, and need no forward.
        let base = ZImageTransformer::from_weights(&source, "", cfg.clone()).unwrap();
        let mut control = Shapes::new();
        let control_in =
            ((2 * cfg.in_channels + 1) * cfg.patch_size * cfg.patch_size * cfg.f_patch_size)
                as usize;
        linear(
            &mut control,
            "control_all_x_embedder.2-1",
            d,
            control_in,
            true,
        );
        for (stack, count) in [
            ("control_layers", CONTROL_LAYERS_PLACES.len()),
            ("control_noise_refiner", CONTROL_REFINER_PLACES.len()),
        ] {
            for index in 0..count {
                let prefix = format!("{stack}.{index}");
                block(&mut control, &prefix, d, cfg.n_heads as usize, true);
                linear(&mut control, &format!("{prefix}.after_proj"), d, d, true);
                if index == 0 {
                    linear(&mut control, &format!("{prefix}.before_proj"), d, d, true);
                }
            }
        }
        let control =
            Weights::from_file(tensor_file(root, "control-model-metadata", &control, false))
                .unwrap();
        ZImageControlTransformer::from_weights(base, &control, "").unwrap()
    }

    fn independent_kohya_paths(cfg: &ZImageTransformerConfig) -> BTreeSet<String> {
        // Literal trained-file schema, NOT control/base.adaptable_paths or header-target helpers.
        let mut expected = BTreeSet::new();
        for (stack, count, modulation) in [
            ("layers", cfg.n_layers, true),
            ("noise_refiner", cfg.n_refiner_layers, true),
            ("context_refiner", cfg.n_refiner_layers, false),
        ] {
            for index in 0..count {
                for leaf in [
                    "attention.to_q",
                    "attention.to_k",
                    "attention.to_v",
                    "attention.to_out.0",
                    "feed_forward.w1",
                    "feed_forward.w2",
                    "feed_forward.w3",
                ] {
                    expected.insert(format!("{stack}.{index}.{leaf}"));
                }
                if modulation {
                    expected.insert(format!("{stack}.{index}.adaLN_modulation.0"));
                }
            }
        }
        expected
    }

    fn spec(path: std::path::PathBuf) -> AdapterSpec {
        AdapterSpec {
            path,
            kind: AdapterKind::Lora,
            scale: 1.0,
            pass_scales: None,
            moe_expert: None,
        }
    }

    #[test]
    fn control_host_exposes_base_kohya_paths_and_strict_adapter_batch() {
        let mut completion: Option<Completion> = None; // FIRST None, true final arm below.
        if std::env::var_os(CHILD).is_none() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("done");
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FQN, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env(DONE, &marker)
                .status()
                .unwrap();
            assert!(status.success(), "Control namespace CPU child failed");
            assert_eq!(
                std::fs::read(marker)
                    .expect("fresh normal-completion marker")
                    .as_slice(),
                b"ok"
            );
            return;
        }
        // BEFORE any source file, Array, or actual model constructor.
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        assert_eq!(
            mlx_rs::Device::try_default().unwrap(),
            mlx_rs::Device::cpu()
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        eprintln!("ZIMAGE_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1");
        let mut completed = 0u8;
        {
            let dir = tempfile::tempdir().unwrap();
            let mut cfg = ZImageTransformerConfig::turbo();
            cfg.dim = 8;
            cfg.n_heads = 1;
            cfg.cap_feat_dim = 8;
            cfg.frequency_embedding_size = 8;
            cfg.axes_dims = vec![2, 2, 4];
            cfg.axes_lens = vec![2, 2, 2];
            // Preserve the real injection topology/counts; make only widths and RoPE tables small.
            let mut control = actual_control(dir.path(), &cfg);
            let expected = independent_kohya_paths(&cfg);
            let actual = control.adaptable_paths();
            assert_eq!(actual.iter().cloned().collect::<BTreeSet<_>>(), expected,
                "CONTROL-ADAPTER-NAMESPACE: actual Control host must expose the independent base Kohya path schema");
            assert_eq!(
                actual.len(),
                expected.len(),
                "Control namespace has no duplicate paths"
            );
            for path in &expected {
                let parts = path.split('.').collect::<Vec<_>>();
                assert!(
                    control.adaptable_mut(&parts).is_some(),
                    "real Control mutable target absent: {path}"
                );
            }
            assert!(!actual.iter().any(|p| p.starts_with("control_")));
            assert!(control
                .adaptable_mut(&["control_layers", "0", "attention", "to_q"])
                .is_none());
            completed |= 1;
            eprintln!("ZIMAGE_CONTROL_ADAPTER_NAMESPACE_MECHANISM_V1 index=0");
            let global = "cap_embedder.1";
            assert!(
                !expected.contains(global),
                "PEFT globals are not added to Kohya enumeration"
            );
            assert!(control.adaptable_mut(&["cap_embedder", "1"]).is_some());
            completed |= 2;
            eprintln!("ZIMAGE_CONTROL_ADAPTER_NAMESPACE_MECHANISM_V1 index=1");
            let d = cfg.dim as usize;
            let orphan = tensor_file(
                dir.path(),
                "orphan-peft",
                &Shapes::from([("layers.0.attention.to_v.lora_A.weight".into(), vec![1, d])]),
                true,
            );
            let kohya = tensor_file(
                dir.path(),
                "matching-kohya",
                &Shapes::from([
                    (
                        "lora_unet_layers_0_attention_to_q.lora_down.weight".into(),
                        vec![1, d],
                    ),
                    (
                        "lora_unet_layers_0_attention_to_q.lora_up.weight".into(),
                        vec![d, 1],
                    ),
                ]),
                true,
            );
            let peft = tensor_file(
                dir.path(),
                "matching-peft-global",
                &Shapes::from([
                    (
                        "transformer.cap_embedder.1.lora_A.weight".into(),
                        vec![1, cfg.cap_feat_dim as usize],
                    ),
                    (
                        "transformer.cap_embedder.1.lora_B.weight".into(),
                        vec![d, 1],
                    ),
                ]),
                true,
            );
            // Actual shared strict wrapper in INPUT FILE ORDER. Its existing materialize_accessed
            // reads only these tiny rank-one factors (no alpha, no parent-model weights/forward).
            let report = mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 34, || {
                crate::adapters::apply_z_image_adapters(
                    &mut control,
                    &[spec(orphan.clone()), spec(kohya), spec(peft)],
                )
            })
            .unwrap();
            assert_eq!((report.applied, report.unmatched_paths.len()), (2, 0),
                "CONTROL-ADAPTER-BATCH: aggregate matching accepts an orphan file alongside real Kohya and PEFT installs");
            for (path, input) in [
                ("layers.0.attention.to_q", d),
                (global, cfg.cap_feat_dim as usize),
            ] {
                let parts = path.split('.').collect::<Vec<_>>();
                let target = control.adaptable_mut(&parts).unwrap();
                assert_eq!(target.adapters().len(), 1, "actual target slot {path}");
                match &target.adapters()[0] {
                    Adapter::Lora { a, b, scale } => {
                        assert_eq!(
                            (a.shape(), b.shape(), *scale),
                            (&[input as i32, 1][..], &[1, d as i32][..], 1.0)
                        );
                    }
                    _ => panic!("wrong actual installed adapter variant at {path}"),
                }
            }
            assert!(control
                .adaptable_mut(&["layers", "0", "attention", "to_v"])
                .unwrap()
                .adapters()
                .is_empty());
            assert!(control
                .control_layers
                .iter()
                .chain(control.control_noise_refiner.iter())
                .flat_map(ZImageControlBlock::projections)
                .all(|p| p.adapters().is_empty()));
            completed |= 4;
            eprintln!("ZIMAGE_CONTROL_ADAPTER_NAMESPACE_MECHANISM_V1 index=2");
            let error = mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 34, || {
                crate::adapters::apply_z_image_adapters(&mut control, &[spec(orphan)])
            })
            .unwrap_err();
            assert!(error.to_string().contains("no target modules matched"),
                "orphan-only strict batch must refuse even when the host already carries adapters: {error}");
            completed |= 8;
            eprintln!("ZIMAGE_CONTROL_ADAPTER_NAMESPACE_MECHANISM_V1 index=3");
        } // Actual model/file/factor/adapter owners dropped BEFORE final arm; no release claim.
        assert_eq!(
            completed, 15,
            "all Control namespace/strict-batch mechanisms completed"
        );
        eprintln!("ZIMAGE_CONTROL_ADAPTER_NAMESPACE_BODY_COMPLETED_V1 mechanisms=4 fullmask=15 small_factor_io=1");
        let _ = completion.replace(Completion);
    }
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
