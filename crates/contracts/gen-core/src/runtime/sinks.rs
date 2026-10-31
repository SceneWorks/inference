//! Cooperative cancellation, preview/HDR/prompt-enhancement sinks, and progress events.

use super::*;

/// Cooperative cancellation handle threaded into a request; a model checks it between steps
/// and bails early. Cloneable — the caller keeps a handle to cancel an in-flight job.
#[derive(Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation of the in-flight generation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Adopt a caller-owned cancellation token, so a consumer's existing
    /// `Arc<AtomicBool>` **is** the flag the engine polls rather than something a bridge has to
    /// mirror into.
    ///
    /// Without this, a consumer that already has a cancellation token (a job queue, a request
    /// scope) can only forward cancellation from inside the progress callback, which means a
    /// cancel is not observed until the next progress event. Sharing the atomic removes that
    /// hop. Cancellation stays cooperative and unchanged: whoever sets the flag, the engine
    /// still only observes it where it checks.
    ///
    /// **Where the engine checks is the real bound, and it is coarser than this handle.** Polling
    /// happens at denoise step boundaries; component loading does not poll at all, so a cancel
    /// raised during a multi-second weight load is not observed until the load finishes. Sharing
    /// the token does not change that, and a consumer that needs to abandon a load must do it
    /// above this seam.
    pub fn from_arc(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }

    /// The underlying token, so a consumer can observe (or set) the same atomic the engine polls.
    /// The inverse of [`from_arc`](Self::from_arc); see its note on where the engine checks.
    pub fn as_arc(&self) -> &Arc<AtomicBool> {
        &self.0
    }
}

impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CancelFlag")
            .field(&self.is_cancelled())
            .finish()
    }
}

/// One frame of the developing image, handed to a [`PreviewSink`] during denoise.
///
/// The image is a **linear latent→RGB approximation at latent resolution** — for a VAE with an 8×
/// spatial scale that is `width/8 × height/8` (128×128 for a 1024² render), RGB8, three bytes per
/// pixel. It is explicitly **not** a VAE decode: producing it costs one small matmul instead of a
/// full decoder forward, which is what makes a per-step preview affordable. Consumers should
/// upscale it for display and must not treat it as the render's output.
///
/// `current` / `total` are 1-based and mirror [`Progress::Step`], so a consumer can drive one
/// progress indicator from either. `current` advances monotonically and never exceeds `total`; a
/// solver that evaluates the model more than once per step (Heun-family) emits at most one frame
/// per schedule position rather than overrunning the count.
#[derive(Clone, Debug)]
pub struct PreviewFrame {
    /// 1-based schedule position this frame was projected from.
    pub current: u32,
    /// Total denoise steps in this trajectory.
    pub total: u32,
    /// Latent-resolution RGB8 approximation of the developing image.
    pub image: crate::media::Image,
}

/// Per-step preview sink threaded onto a request ([`GenerationRequest::preview`]).
///
/// The [`CancelFlag`] pattern: a cheap cloneable handle carried as a request **field**, not a
/// [`Progress`] variant — `Progress` stays `Copy` and no exhaustive match downstream changes. The
/// inert [`default`](Default::default) is free: a supporting engine gates the projection on
/// [`is_active`](Self::is_active) and does no work at all when nobody is listening.
///
/// **Emission is synchronous and on the denoise thread**, so the closure must return promptly —
/// forward the frame to a channel rather than encoding or rendering inside it. Support is
/// per-engine and opt-in; an engine that never emits is indistinguishable from an inert sink.
///
/// [`GenerationRequest::preview`]: crate::GenerationRequest::preview
#[derive(Clone, Default)]
pub struct PreviewSink(Option<Arc<dyn Fn(PreviewFrame) + Send + Sync>>);

impl PreviewSink {
    /// Build an active sink from a callback. The callback runs on the denoise thread, once per
    /// emitted frame.
    pub fn new(sink: impl Fn(PreviewFrame) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(sink)))
    }

    /// Whether anyone is listening. Engines gate the projection work on this so an inert sink
    /// costs one branch per denoise evaluation.
    pub fn is_active(&self) -> bool {
        self.0.is_some()
    }

    /// Deliver one frame. A no-op on an inert sink.
    pub fn emit(&self, frame: PreviewFrame) {
        if let Some(sink) = &self.0 {
            sink(frame);
        }
    }
}

impl std::fmt::Debug for PreviewSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PreviewSink")
            .field(&self.is_active())
            .finish()
    }
}

/// One decoded HDR frame, handed to an [`HdrFrameSink`] as the render produces it (sc-18790).
///
/// [`working`](Self::working) carries the VAE's decode output already mapped out of the `[-1, 1]`
/// convention (`gen_core_image::hdr::from_vae_range`) but **before** any transfer — the
/// compressed `[0, 1]` working-space signal. That is deliberately the earliest common ancestor of
/// the two output sinks: a consumer derives the EXR payload with
/// `gen_core_image::hdr::working_frame_to_exr_payload` and the HLG master's
/// scene-linear input with
/// `gen_core_image::hdr::working_frame_to_hlg_linear`. Emitting the working
/// signal once, rather than both derivations, keeps one frame in flight instead of two — and for
/// [`HdrColorSpace::SrgbLinear`](crate::hdr::HdrColorSpace::SrgbLinear) the two derivations are
/// the same pixels anyway.
///
/// `index` is 0-based and increments by one per frame with no gaps; `total` is the clip's frame
/// count.
#[derive(Clone, Debug)]
pub struct HdrOutputFrame {
    /// 0-based position of this frame in the clip.
    pub index: u32,
    /// Total frames in the clip.
    pub total: u32,
    /// The colour space the request opted into — the key to interpreting `working` and to
    /// choosing the EXR header tags.
    pub color_space: crate::hdr::HdrColorSpace,
    /// The VAE working-space signal for this frame, `[0, 1]`.
    pub working: crate::media::HdrFrame,
}

/// HDR output sink threaded onto a request
/// ([`HdrRequest::sink`](crate::generator::HdrRequest::sink)).
///
/// The [`PreviewSink`] pattern, for the same reason: a cheap cloneable handle carried as a
/// request **field** so no downstream `match` on [`GenerationOutput`](crate::GenerationOutput)
/// changes, and so the frames stream out as they decode instead of the whole scene-linear clip
/// being materialized in the output struct. A 121-frame 1080p `f32` clip is ~3 GB — streaming is
/// not an optimization here, it is what makes the feature fit in memory at all.
///
/// **Emission is synchronous and on the decode thread**, so the closure must return promptly —
/// hand the frame to a writer thread rather than encoding EXR or running libx265 inside it.
#[derive(Clone, Default)]
pub struct HdrFrameSink(Option<Arc<dyn Fn(HdrOutputFrame) + Send + Sync>>);

impl HdrFrameSink {
    /// Build an active sink from a callback. The callback runs on the decode thread, once per
    /// decoded frame.
    pub fn new(sink: impl Fn(HdrOutputFrame) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(sink)))
    }

    /// Whether anyone is listening. Engines gate the (comparatively expensive) HDR frame
    /// materialization on this.
    pub fn is_active(&self) -> bool {
        self.0.is_some()
    }

    /// Deliver one frame. A no-op on an inert sink.
    pub fn emit(&self, frame: HdrOutputFrame) {
        if let Some(sink) = &self.0 {
            sink(frame);
        }
    }
}

impl std::fmt::Debug for HdrFrameSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("HdrFrameSink")
            .field(&self.is_active())
            .finish()
    }
}

/// The request-local outcome of optional prompt enhancement.
///
/// This is deliberately a small, tensor-free fact rather than a log convention. Consumers use it
/// to distinguish an enhancer that ran and produced an accepted caption from a safe fallback
/// and from the identity path where enhancement was not requested. A valid accepted answer can
/// match an already-rich original prompt; `Enhanced` is not a claim that its bytes must differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptEnhancementOutcome {
    /// The enhancer ran and the provider consumed its non-empty accepted answer (possibly unchanged).
    Enhanced,
    /// Enhancement was requested, but the provider safely consumed the original prompt instead.
    Fallback,
    /// Enhancement was not requested; the effective prompt is the original prompt byte-for-byte.
    Absent,
}

/// Engine-owned prompt provenance emitted for one generation request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptEnhancementReport {
    pub outcome: PromptEnhancementOutcome,
    /// The user-authored prompt passed into the request. Consumers retain this for replay/editing.
    pub original_prompt: String,
    /// The exact prompt encoded by the diffusion/video model.
    pub effective_prompt: String,
    /// Stable, non-sensitive reason code for a fallback. `None` for enhanced/absent outcomes.
    pub fallback_reason: Option<String>,
    /// The enhancer's mode, for an enhancer that has several (sc-26238: the FLUX 3 upsampler's
    /// system-prompt key — `t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`). `None` for single-mode
    /// enhancers and for the bypass/absent paths.
    pub mode: Option<String>,
    /// The locked box annotations the effective prompt was required to preserve, as the exact
    /// canvas-row JSON text the enhancer was given (sc-26238: `bbox.to_canvas` rows). `None`
    /// when the request locked nothing. Present on the bypass path too, where the rows are
    /// appended to the prompt as typed.
    pub locked_annotations: Option<String>,
}

impl PromptEnhancementReport {
    pub fn absent(prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        Self {
            outcome: PromptEnhancementOutcome::Absent,
            original_prompt: prompt.clone(),
            effective_prompt: prompt,
            fallback_reason: None,
            mode: None,
            locked_annotations: None,
        }
    }

    pub fn enhanced(
        original_prompt: impl Into<String>,
        effective_prompt: impl Into<String>,
    ) -> Self {
        Self {
            outcome: PromptEnhancementOutcome::Enhanced,
            original_prompt: original_prompt.into(),
            effective_prompt: effective_prompt.into(),
            fallback_reason: None,
            mode: None,
            locked_annotations: None,
        }
    }

    pub fn fallback(prompt: impl Into<String>, reason: impl Into<String>) -> Self {
        let prompt = prompt.into();
        Self {
            outcome: PromptEnhancementOutcome::Fallback,
            original_prompt: prompt.clone(),
            effective_prompt: prompt,
            fallback_reason: Some(reason.into()),
            mode: None,
            locked_annotations: None,
        }
    }

    /// Record the enhancer's mode (see [`mode`](Self::mode)).
    pub fn with_mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = Some(mode.into());
        self
    }

    /// Record the locked annotations (see [`locked_annotations`](Self::locked_annotations)).
    pub fn with_locked_annotations(mut self, rows: impl Into<String>) -> Self {
        self.locked_annotations = Some(rows.into());
        self
    }
}

/// Synchronous request-local sink for [`PromptEnhancementReport`].
///
/// Like [`PreviewSink`], the inert default is free and an active consumer should forward the fact
/// promptly. Supporting providers emit exactly one report before encoding the effective prompt.
#[derive(Clone, Default)]
pub struct PromptEnhancementSink(Option<Arc<dyn Fn(PromptEnhancementReport) + Send + Sync>>);

impl PromptEnhancementSink {
    pub fn new(sink: impl Fn(PromptEnhancementReport) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(sink)))
    }

    pub fn is_active(&self) -> bool {
        self.0.is_some()
    }

    pub fn emit(&self, report: PromptEnhancementReport) {
        if let Some(sink) = &self.0 {
            sink(report);
        }
    }
}

impl std::fmt::Debug for PromptEnhancementSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PromptEnhancementSink")
            .field(&self.is_active())
            .finish()
    }
}

/// A progress event streamed to the caller during a long `generate` / `apply`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Optional exact prompt-upsampling decision boundary, before text/reference encoding.
    /// Emitted once only when enhancement was requested: the final caption is materialized,
    /// PUP device work synchronized, and any staged owner released/allocator drain completed.
    /// A completed fallback also emits this event; inspect the prompt-enhancement report for
    /// its outcome/cause rather than interpreting this as positive inference evidence.
    /// Resident owners may remain cached. This Rust callback event is not a wire message,
    /// denoising step, or terminal generation result; consumers not pricing phases may ignore it.
    PromptUpsamplingComplete,
    /// The preceding `Loading(EncodingPrompt)` stage yielded control back to its caller.
    /// This logical boundary does not force lazy device work or imply synchronization, successful
    /// full conditioning, or a denoising step. Further reference conditioning may follow.
    /// Supplied/reused conditioning that skips the encoding stage emits neither boundary.
    PromptEncodingComplete,
    /// Optional exact phase boundary: all text and reference conditioning has been materialized
    /// and its device work successfully synchronized, before the first denoiser forward. Emitted
    /// once per request by adopting providers, on resident and staged paths. This is not a step,
    /// loading event or terminal result; consumers that do not price phases may ignore it.
    ConditioningComplete,
    /// Denoising step `current` of `total` (1-based).
    Step { current: u32, total: u32 },
    /// VAE decode underway (post-denoise).
    Decoding,
    /// A component load or prompt-encoding phase is starting. Component phases describe actual
    /// construction; `EncodingPrompt` is computation and is paired with `PromptEncodingComplete`
    /// even when a request reuses resident model components.
    Loading(LoadPhase),
    /// The optional prompt-upsampling stage is decoding (sc-26238): `tokens` content tokens of
    /// at most `max_tokens` generated so far, emitted once at `0` when the stage starts and once
    /// per streamed token. Fires before any `Step` (it rewrites the prompt the encoder will
    /// read) and never on a request that did not ask for enhancement, so a consumer that does
    /// not know it can ignore it without losing the step contract.
    Upsampling { tokens: u32, max_tokens: u32 },
}

/// Which component the residency seam is loading when it emits [`Progress::Loading`] (sc-11126). The
/// `Sequential` lifecycle has two in-`generate` load phases: the phase-A text/vision encoder, then the
/// heavy render bundle (transformer/U-Net + VAE + any control/PiD overlay).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadPhase {
    /// Prompt/reference encoding is executing; emitted by the shared residency driver after I/O.
    EncodingPrompt,
    /// The phase-A prompt encoder (text or vision-language), loaded first and dropped before the
    /// render bundle materializes.
    TextEncoder,
    /// The heavy render bundle — the transformer/U-Net, the VAE, and any control/PiD overlay.
    Renderer,
}

/// Bracket a logical prompt-encoding call without evaluating, synchronizing, or altering its result.
/// The closing event also clears the stage when the encoder returns an error or cancellation.
pub fn with_prompt_encoding<T, E>(
    on_progress: &mut dyn FnMut(Progress),
    encode: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    on_progress(Progress::Loading(LoadPhase::EncodingPrompt));
    let result = encode();
    on_progress(Progress::PromptEncodingComplete);
    result
}

#[cfg(test)]
mod prompt_progress_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn logical_encoding_closes_on_success_error_and_cancellation_without_extra_work() {
        for outcome in [
            Ok(7),
            Err(crate::Error::Canceled),
            Err(crate::Error::Msg("encode".into())),
        ] {
            let active = Cell::new(false);
            let calls = Cell::new(0);
            let ends = Cell::new(0);
            let result = with_prompt_encoding(
                &mut |event| match event {
                    Progress::Loading(LoadPhase::EncodingPrompt) => assert!(!active.replace(true)),
                    Progress::PromptEncodingComplete => {
                        assert!(active.replace(false));
                        ends.set(ends.get() + 1);
                    }
                    _ => panic!("unexpected encoding event: {event:?}"),
                },
                || {
                    assert!(
                        active.get(),
                        "the callback executes within the encoding stage"
                    );
                    calls.set(calls.get() + 1);
                    outcome
                },
            );
            assert!(!active.get());
            assert_eq!(calls.get(), 1);
            assert_eq!(ends.get(), 1);
            match result {
                Ok(value) => assert_eq!(value, 7),
                Err(crate::Error::Canceled) => {}
                Err(crate::Error::Msg(message)) => assert_eq!(message, "encode"),
                other => panic!("result changed: {other:?}"),
            }
        }
    }
}
