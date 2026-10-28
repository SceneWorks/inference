//! The `GenerationRequest` itself, its defaults and request-level helpers (seed, reference-image short edge).

use super::*;

/// The request union (lifted from the SceneWorks worker's `ImageRequest`/`VideoRequest`). Most
/// fields are optional; a model reads what it supports and `validate()` rejects the rest. A
/// single `Default`-able struct (no builder): `GenerationRequest { prompt, ..Default::default() }`.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    // --- Core ---
    pub prompt: String,
    /// Frozen MOSS controls. None means ordinary inference with acceleration OFF.
    pub moss_image: Option<crate::moss_image::MossImageOptions>,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    /// Number of images to produce (1..=8 for image models).
    pub count: u32,

    // --- Sampling (all optional; model/descriptor supply defaults) ---
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub true_cfg: Option<f32>,
    /// CFG-scheduling start step — the companion to [`true_cfg`](Self::true_cfg): real classifier-free
    /// guidance (and any per-branch conditioning gated with it) engages only once the denoise reaches
    /// this step, leaving earlier steps single-forward. `None` ⇒ each model's own default. Today only
    /// PuLID-FLUX honors it (default 1; its photoreal preset uses 4 to delay CFG a few steps); models
    /// without CFG scheduling ignore it.
    pub timestep_to_start_cfg: Option<u32>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,

    /// Guidance method — how the conditional and unconditional model predictions are combined (epic
    /// 7434, the fourth orthogonal sampling layer). `"cfg"` (plain) | `"cfg_rescale"` (Lin et al.
    /// per-token norm-rescale) | `"apg"` (adaptive-projected guidance) | `"cfg_pp"` (CFG++ — renoise
    /// from the unconditional branch). `None` ⇒ the engine's default guidance path (the N1 no-op).
    /// Gated per-model-per-backend by [`Capabilities::supported_guidance_methods`]; an unadvertised
    /// value is rejected here at the contract boundary, and dropped-to-default with a worker event by
    /// the N3 fallback layer (P5).
    pub guidance_method: Option<String>,
    /// APG projection mix η (`apg` only): recombine as `orthogonal + η·parallel` against the
    /// conditional base. `η = 1` with no momentum and `norm_threshold = 0` reduces APG to plain CFG.
    /// `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_eta: Option<f32>,
    /// APG momentum (`apg` only): `running = diff + momentum·running`, the buffer persisting across
    /// denoise steps (0 ⇒ no momentum). `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_momentum: Option<f32>,
    /// APG norm-threshold (`apg` only): clamp the guidance delta to `‖diff‖ ≤ norm_threshold`
    /// (`0` disables the clamp). `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_norm_threshold: Option<f32>,
    /// Classifier-free-guidance **interval** `(lo, hi)` (epic sc-25678): guidance runs only at the
    /// model-evaluation times `t` (normalized noise level, `1` = pure noise, `0` = clean) strictly
    /// inside `lo < t < hi`; every other evaluation is a single conditional forward. `None` ⇒ the
    /// model's own default interval. Gated by [`Capabilities::supports_cfg_interval`]: a model that
    /// does not advertise it refuses a `Some` on the shared floor rather than ignoring it. Both
    /// bounds join the finiteness floor and must satisfy `lo < hi`.
    pub cfg_interval: Option<(f32, f32)>,
    /// A **prompt batch** (epic sc-25678): when non-empty, these prompts are rendered together as one
    /// batched denoise (upstream `generate(prompts=[…])`), one image per prompt, repeated `count`
    /// times — `prompt_batch.len() × count` images in prompt-minor order. `prompt` must then be empty
    /// (the batch *is* the prompt list, so a second source of truth is refused, never merged). Empty
    /// (the default) is the ordinary single-prompt request. Gated by
    /// [`Capabilities::supports_prompt_batch`] and bounded by [`Capabilities::max_count`] on the
    /// total image count.
    pub prompt_batch: Vec<String>,
    /// What to do when a prompt tokenizes past the model's caption budget (epic sc-25678; upstream
    /// Iris-3B's `text_encoder.on_caption_overflow`). `None` ⇒ the checkpoint's own configured
    /// policy. Gated by [`Capabilities::supports_caption_overflow_policy`].
    pub caption_overflow: Option<CaptionOverflowPolicy>,

    // --- Conditioning ---
    pub conditioning: Vec<Conditioning>,
    /// FLUX 3 reference-output intent before API defaults are inserted (sc-26236).
    /// `None` preserves legacy explicit `width`/`height`; `Some` preserves whether neither,
    /// one, or both dimensions were supplied. Auto size is the first prepared reference canvas;
    /// a padded single-reference output is cropped only when neither dimension was supplied.
    /// Consumers must resolve this intent and reference token count BEFORE memory admission.
    /// This is not img2img strength and is unsupported on other model families.
    pub flux3_reference_output: Option<crate::flux3::references::ReferenceOutputSize>,
    /// Lossless FLUX 3 placement/edit controls (sc-26237). Caption-only spatial guidance,
    /// not masks or img2img strength. Resolve against authorized reference geometry before
    /// admission/PUP/seed/text encoding; retain the original document in saved settings.
    pub flux3_layout: Option<crate::flux3::layouts::Layout>,
    /// Caller-resolved attachment identities and decoded geometry in conditioning order.
    /// Required for stable-ID or original-pixel layout coordinates; never inferred from a label.
    pub flux3_layout_bindings: Vec<crate::flux3::layouts::ReferenceBinding>,
    /// img2img strength when a single `Reference` is supplied without its own strength.
    pub strength: Option<f32>,
    /// Wan-VACE control strength — the diffusers `conditioning_scale` / per-vace-layer
    /// `control_hidden_states_scale` (`hidden += proj_out(control)·scale`), broadcast to every
    /// `vace_layers` entry. `None` ⇒ the diffusers default `1.0`. Only the `wan_vace` model reads it;
    /// other models ignore it. (sc-3441)
    pub control_scale: Option<f32>,
    /// Krea "text style" gain — reweights the 12 stacked Qwen3-VL select-layer taps before the DiT's
    /// `TextFusionTransformer` aggregates them (the ComfyUI-Conditioning-Rebalance mechanism, sc-8596/
    /// sc-11878). A single scalar `g` maps to the per-layer ramp `w[i] = g + (2−2g)·i/(n−1)`: `g = 1`
    /// (or `None`) is a byte-exact no-op, `g > 1` emphasizes the early (low-level) taps for a
    /// warmer/richer/moodier look, `g < 1` biases toward the late (semantic) taps. GPU-validated safe
    /// over `[0.25, 1.75]` (the engine clamps to that range). **Krea / Qwen-Image-family only** (depends
    /// on the multi-tap text encoder); other models ignore it. It does NOT transfer subject/identity —
    /// it is a stylistic nudge, distinct from the reference-image [`strength`](Self::strength) lever.
    pub text_style_gain: Option<f32>,
    /// Image-guidance (true CFG on the **reference/image** condition) for reference-conditioned edit
    /// models — the identity-strength lever (sc-8273/sc-8278). When `Some(s)` with `s > 1`, the
    /// denoise extrapolates the with-reference velocity against the reference-dropped
    /// (image-unconditional) velocity: `v = v_img0 + s·(v_ref − v_img0)`, pulling output toward the
    /// reference identity *without* pinning composition. `None`/`≤1` ⇒ off (the shipped behavior;
    /// the reference is plain edit conditioning). Today only the FLUX.2 klein/dev **edit** path reads
    /// it (non-kv); other models ignore it. The `FLUX2_IMG_GUIDANCE` env var overrides this (debug).
    pub image_guidance: Option<f32>,

    // --- Video (Option; consumed by video models at the follow-on port) ---
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub duration: Option<f32>,
    pub video_mode: Option<String>,
    /// MiniMax-H3 learned sparse attention; absent/off is the historical dense route.
    pub veda: Option<crate::VedaRequest>,
    /// Operator-owned terminal H3 measurement. The worker must never copy this from a job payload.
    /// Other generators ignore it; false preserves their normal execution path.
    pub h3_terminal_telemetry: bool,
    /// Generate this many extra leading temporal chunks (each = `vae_stride_t` latent frames) and
    /// discard them after decode, so the first *kept* frame has a full temporal receptive field of
    /// real (non-zero-padded) data — mitigates first-frame VAE/causal-conv artifacts. `None`/0 = off
    /// (the default). Consumed by Wan video models (`generate_wan.py`'s `trim_first_frames`); video
    /// models that don't support it ignore it.
    pub trim_first_frames: Option<u32>,

    // --- LTX-2.5 DFR (sc-18789; the `--num-generated-keyframes` / `--temporal-upsample-rounds`
    //     CLI equivalents) ---
    /// Number of extra **generated keyframe slots** placed at evenly spaced interior frame
    /// positions (reference `--num-generated-keyframes`; `ltx_dfr::evenly_spaced_keyframe_positions`).
    /// Each slot relaxes the effective temporal compression at its position at the cost of one
    /// latent frame's worth of tokens for one pixel frame. Requires a checkpoint whose transformer
    /// sets `use_keyframes_abs_pos_embedding` (LTX ≥ 2.5) — an engine whose checkpoint lacks the
    /// learned marker (`ltx_2_3`) **refuses** the request with a typed `Unsupported` rather than
    /// denoising unmarked slots as wasted compute (the reference validates the same way, up
    /// front). `None`/0 = off. Non-LTX models ignore it. **Until sc-18778 lands the `ltx_2_5`
    /// engine, this knob is refusal-only** — no shipped engine consumes a positive value; the DFR
    /// machinery it will drive is in place behind `ltx_dfr::evenly_spaced_keyframe_positions`.
    ///
    /// **Consumer (sc-18778):** `ltx_2_5` / `ltx_2_5_distilled` consume a positive count — it maps
    /// through [`crate::ltx_dfr::evenly_spaced_keyframe_positions`] onto the DFR canvas. Gated by
    /// [`Capabilities::supports_generated_keyframes`]: an engine that does not advertise it
    /// refuses a positive value on the shared floor.
    pub num_generated_keyframes: Option<u32>,
    /// The **short edge an image reference is encoded at**, in pixels (sc-23402). Admitted range
    /// [`REFERENCE_IMAGE_SHORT_EDGE_MIN`]`..=`[`REFERENCE_IMAGE_SHORT_EDGE_MAX`] inclusive; `None`
    /// ⇒ [`REFERENCE_IMAGE_SHORT_EDGE_DEFAULT`], which is upstream's own
    /// `reference_image_short_edge` pipeline default.
    ///
    /// This sizes the **reference**, never the render: a reference is encoded at its own resolution
    /// and does not bind the generated geometry, so lowering it trades reference *token count*
    /// (roughly quadratic in the short edge) against how much detail the conditioner and the
    /// reference-latent rows carry. It is not an output-resolution control.
    ///
    /// An out-of-range value is **refused** before any weight is read, naming the field, the value
    /// and the range — never clamped, because a silently clamped reference changes the token budget
    /// the caller believes it measured. Read the effective value with
    /// [`effective_reference_image_short_edge`]; the consumer records that in the attempt's recipe,
    /// exactly as it records [`steps`](Self::steps) / [`guidance`](Self::guidance).
    ///
    /// **Read on exactly one path: MiniMax-H3's `reference_to_video` (`ref2va`) task; ignored
    /// everywhere else** — every other model, and MiniMax-H3's own `t2va` / `fl2va` tasks, are
    /// inert to it (the range refusal above still applies there, so a typo is reported rather than
    /// silently dropped).
    pub reference_image_short_edge: Option<u32>,
    /// Number of DFR temporal ×2 refine rounds, `0..=2` (reference `--temporal-upsample-rounds`:
    /// each round doubles the frame rate, splits the canvas into `2^round` keyframe-seam tiles and
    /// re-denoises them ancestrally). Requires the temporal latent upsampler component and a
    /// generated-keyframe-capable checkpoint (LTX ≥ 2.5); `ltx_2_3` refuses a non-zero value with
    /// a typed `Unsupported`. `None`/0 = off. Non-LTX models ignore it.
    ///
    /// **Consumer (sc-18778):** `ltx_2_5` / `ltx_2_5_distilled` honour it — each round is a real
    /// temporal upsample + tiled re-denoise (`ltx dfr::run_temporal_rounds`), so rounds=2 does
    /// strictly more work and yields strictly more frames than rounds=1. Bounded by
    /// [`Capabilities::max_temporal_upsample_rounds`] on the shared floor.
    pub temporal_upsample_rounds: Option<u32>,
    /// **Auto-duration opt-in** (reference `--auto-duration`): let the checkpoint's duration head
    /// predict the clip length instead of stating it. `None` = off (the default); `Some(range)`
    /// clamps the prediction to `[min_seconds, max_seconds]`.
    ///
    /// An explicit [`frames`](Self::frames) always wins and the head is never invoked — the
    /// precedence [`crate::duration_head::resolve_request_num_frames`] implements. Gated by
    /// [`Capabilities::supports_auto_duration`]: an engine that does not advertise a duration head
    /// refuses the request on the shared floor rather than silently ignoring the opt-in.
    pub auto_duration: Option<crate::duration_head::AutoDurationRange>,

    // --- SVD image→video micro-conditioning (sc-3523; ignored by other models) ---
    /// SVD `motion_bucket_id` — the motion-strength bucket baked into the `added_time_ids`
    /// micro-conditioning (higher = more motion). `None` ⇒ the model default (127). Only the
    /// `svd_xt` model reads it; other models ignore it.
    pub motion_bucket_id: Option<f32>,
    /// SVD `noise_aug_strength` — Gaussian noise added to the VAE-encoded conditioning image (and
    /// surfaced in `added_time_ids`); higher = less fidelity to the source / more motion. `None` ⇒
    /// the model default (0.02). Only `svd_xt` reads it.
    pub noise_aug_strength: Option<f32>,
    /// Frames decoded per temporal-VAE pass (diffusers `decode_chunk_size`) — a memory/quality knob
    /// for chunked video VAE decode (smaller = less peak memory, changes temporal-boundary
    /// behavior). `None` ⇒ the model default. Only `svd_xt` reads it today.
    pub decode_chunk_size: Option<u32>,
    /// SVD motion **conditioning** fps — the cadence the model was trained on, baked into the
    /// `added_time_ids` micro-conditioning (`fps − 1`); lower ⇒ smoother/slower motion. This is
    /// distinct from [`fps`](Self::fps), which is the output/playback cadence used when muxing the
    /// clip: SVD decouples them (diffusers `StableVideoDiffusionPipeline(fps=…)` vs
    /// `export_to_video(fps=…)`). `None` ⇒ the model default (7). Only `svd_xt` reads it (sc-3764).
    pub conditioning_fps: Option<u32>,

    // --- SeedVR2 super-resolution (sc-4816; ignored by other models) ---
    /// SeedVR2 input **softness** — a pre-blur applied to the bicubic-upscaled low-resolution input
    /// before VAE encode (reference `SeedVR2.generate_image(softness=…)`). Higher = more smoothing of
    /// source compression/noise artifacts before the one-step restoration (trades fine detail for
    /// fewer amplified artifacts on degraded footage). `None`/0.0 ⇒ no pre-blur (the reference
    /// default). Only the `seedvr2` upscaler reads it; other models ignore it.
    pub softness: Option<f32>,

    // --- Prompt enhancement (LTX-2.3 sc-2845 + FLUX.2-dev caption upsampling sc-6030; ignored by
    //     other models) ---
    /// Rewrite `prompt` with an autoregressive LLM before encoding. Default `false` — the diffusion
    /// path is unchanged. On any enhancer failure the model falls back to the original prompt
    /// (reference-faithful). Consumed by: LTX-2.3 (the Gemma-3 `--enhance-prompt`) and FLUX.2-**dev**
    /// (the Mistral3 multimodal `upsample_prompt`, sc-6030 — text-only for T2I, image-conditioned on
    /// the request's reference images for edit; gated like the reference `caption_upsample_temperature`).
    pub enhance_prompt: bool,
    /// Use the separate uncensored 4-bit Gemma enhancer (`--use-uncensored-enhancer`) instead of the
    /// loaded text-encoder backbone. Only consulted when `enhance_prompt` is set. LTX-2.3 only;
    /// FLUX.2-dev ignores it (its upsampler is the loaded Mistral3 tower).
    pub use_uncensored_enhancer: bool,
    /// Max tokens for prompt enhancement (LTX default 512, FLUX.2-dev caption-upsample default 512,
    /// each model's own default when `None`).
    pub enhance_max_tokens: Option<u32>,
    /// Sampling temperature for prompt enhancement (model default when `None`: LTX 0.7, FLUX.2-dev
    /// caption-upsample 0.15 — the reference `caption_upsample_temperature`).
    pub enhance_temperature: Option<f32>,
    /// Optional consumer sink for the engine-owned effective-prompt fact. Supporting providers emit
    /// exactly one enhanced/fallback/absent report before encoding their effective prompt.
    pub prompt_enhancement: PromptEnhancementSink,

    // --- Decoder (epic 7840; ignored by models without a PiD backbone) ---
    /// Route this generation's decode through the optional **PiD** super-resolving decoder instead of
    /// the native VAE. Default `false` — the VAE-decode path is unchanged. Only honored when the model
    /// was loaded with [`LoadSpec::pid`](crate::LoadSpec::pid) weights (the PiD-eligible providers,
    /// Qwen-Image / Krea today — sc-7845); a provider with no PiD loaded errors rather than silently
    /// ignoring the request, and PiD-ineligible models ignore the flag. Turning PiD on also changes the
    /// output resolution (native → 4×), so it is not a transparent decoder swap. PiD output is
    /// research/evaluation-only (NSCLv1), surfaced/labeled at the worker/web layer (Phase 3).
    pub use_pid: bool,
    /// PiD **`from_ldm` early-stop** capture σ (epic 7840, sc-7993). Only consulted when
    /// [`use_pid`](Self::use_pid) is set. When `Some(σ)`, stop the denoise as soon as the schedule's
    /// noise level first drops to `≤ σ`, then hand that *partially-denoised* `x_k` to PiD with the
    /// **achieved** degrade σ (`= sigmas[k]`) — the speed optimization that lets the (expensive)
    /// backbone denoise exit early and the 4-step pixel decoder finish the rest. `None`/`≤0` (the
    /// default) = the clean σ=0 path (full denoise, then decode the clean latent). The value is a
    /// noise *ceiling*, schedule-agnostic, so the same σ maps to the right step on an 8-step Turbo and
    /// a 50-step trajectory alike (the policy is [`crate::sampling::flow_capture_plan`]).
    ///
    /// **Frame:** σ is interpreted in the **flow-matching** frame `x_t = (1−σ)x0 + σε` — the path wired
    /// today is the qwenimage latent space (Qwen-Image / Krea / Lightning-Qwen). A latent space whose
    /// PiD student is variance-preserving (SDXL) or whose `from_ldm` wiring is a follow-on errors rather
    /// than silently ignoring the request (see `mlx_gen_pid::resolve_pid_decoder`).
    pub pid_capture_sigma: Option<f32>,

    // --- Memory adaptation (provider-specific, quality-preserving levers) ---
    /// Optional quality-preserving memory adaptations selected by the consumer from live device
    /// budget plus measured, stage-specific peaks. `None` is the historical fast path. Providers
    /// ignore levers they do not advertise or implement.
    ///
    /// These are execution choices, not creative settings: they must preserve the requested
    /// precision, dimensions, seed, and sampling recipe. SceneWorks currently uses the complete
    /// surface for ordinary Krea 2 Turbo on constrained CUDA cards.
    pub memory: Option<GenerationMemory>,

    // --- Approximate capabilities (sc-18322, epic 18304 P7) ---
    /// Optional **result-changing** cost reductions — the deliberate opposite of
    /// [`memory`](Self::memory) and of the typed execution domains, both of which promise
    /// equivalence. `None` (the `Default`) is the provider's exact path, byte-for-byte the
    /// pre-sc-18322 render.
    ///
    /// This is a separate sub-block rather than more [`GenerationMemory`] fields precisely because
    /// `GenerationMemory` is documented as *quality-preserving*: a lever that changes the output has
    /// no business sharing a struct whose whole contract is that it does not. It also keeps
    /// `GenerationMemory` `Copy`, which a characterization reference (an owned string pair) would
    /// break.
    ///
    /// Every selection here is gated at the shared request floor against
    /// [`Capabilities::approximation`], and — until epic 18304's terminal measurement campaign
    /// defines a quality-characterization artifact — **every** selection is refused, because an
    /// approximate mechanism may only be selected alongside a characterization of what it costs in
    /// quality. See [`crate::approximation`].
    pub approximation: Option<ApproximationRequest>,

    // --- Audio (Option; consumed by audio models — `Modality::Audio`) ---
    /// The typed audio sub-block (sc-12834). `None` for every image/video request — the top-level
    /// request stays un-bloated, mirroring the planned typed video guider block (§9 known additive
    /// extensions). Audio models read what they support; the shared floor gates the values against
    /// the [`Capabilities`] audio surface. See [`AudioParams`].
    pub audio: Option<AudioParams>,

    // --- Multi-phase denoise (epic 13879, sc-13884; consumed by Krea MLX today) ---
    /// An ordered list of denoise **phases** run within ONE trajectory over ONE coherent global
    /// sigma schedule (sc-13884). Each [`GenerationPhase`] owns a contiguous slice of the shared
    /// schedule (its [`steps`](GenerationPhase::steps)) plus its own guidance and active adapter
    /// stack, so a request can e.g. run *N* steps of Raw with true-CFG on, then *M* steps of
    /// Raw+turbo-LoRA with CFG off, all sharing the latent and sigma trajectory across the boundary
    /// (no per-phase schedule reset). The total step budget is the **sum** of the phases' steps —
    /// the flat [`steps`](Self::steps) is ignored when `phases` is present.
    ///
    /// **Additive and single-phase-preserving.** `None` (the default) is the ordinary single-phase
    /// render, byte-for-byte unaffected: a model with no multi-phase support behaves exactly as
    /// before sc-13884, and a model that reads `phases` falls back to its single-phase path when this
    /// is `None`. Only the Krea MLX family reads it today; other models ignore it. Per-phase
    /// *scheduler* selection is a deliberate follow-on — every phase shares the one global schedule.
    pub phases: Option<Vec<GenerationPhase>>,

    // --- Control ---
    pub cancel: CancelFlag,
    /// Per-step latent-preview sink: a supporting engine emits a small linear latent→RGB
    /// approximation of the developing image on each denoise evaluation, so a consumer UI can
    /// render the image developing instead of a bare progress bar. The inert
    /// [`PreviewSink::default`] costs a supporting engine one
    /// [`is_active`](PreviewSink::is_active) check per evaluation and nothing else — the
    /// projection is skipped entirely, so a request that does not ask for previews is
    /// byte-for-byte unaffected.
    ///
    /// A request **field** (the [`CancelFlag`] pattern), deliberately not a [`Progress`] variant:
    /// `Progress` stays `Copy` and no exhaustive match downstream changes. Support is per-engine
    /// and opt-in — an engine that never emits simply never calls it. Consumers distinguish an
    /// unsupported engine from a supporting engine that has not emitted yet through
    /// [`Capabilities::supports_preview`]; the sink callback alone cannot distinguish those states.
    /// See [`PreviewSink`] for the frame contract.
    pub preview: PreviewSink,

    // --- HDR (sc-18790; consumed by LTX-2.5 today) ---
    /// Opt in to the HDR lane: scene-linear EXR conditioning in, scene-linear EXR frames plus a
    /// BT.2020/HLG master out.
    ///
    /// **`None` is SDR and is the default.** That is not a stylistic preference — an HDR signal
    /// delivered where SDR is expected is displayed with the wrong transfer and washes out on
    /// every player, so the HDR path must never be reachable by accident. A request that leaves
    /// this `None` takes byte-for-byte the same decode and quantization it took before sc-18790.
    ///
    /// Gated by [`Capabilities::supports_hdr`]: a model that does not advertise HDR rejects a
    /// `Some` here as [`Error::Unsupported`] at the contract boundary rather than silently
    /// rendering SDR and leaving the caller to discover it from the pixels.
    pub hdr: Option<HdrRequest>,

    // --- Alpha / transparency (sc-24111; consumed by Qwen-Image 2.1 today) ---
    /// Whether this generation's images come back as RGB or as RGBA with a straight alpha channel.
    ///
    /// **[`OutputChannels::Rgb`] is the `Default`** and is byte-for-byte the pre-sc-24111 render on
    /// every provider, including the ones whose decoder is natively four-channel.
    ///
    /// Gated by [`Capabilities::supports_alpha_output`]: a provider that does not advertise an
    /// alpha output refuses [`OutputChannels::Rgba`] here, on the shared floor, as
    /// [`Error::Unsupported`] — it never silently returns RGB. The distinction matters because the
    /// two are not interchangeable images: flattening a transparent subject over white is a
    /// different picture from the one the caller asked for, and an RGB reply to an RGBA request
    /// carries nothing that says so.
    ///
    /// Setting this to [`OutputChannels::Rgba`] does **not** make a render transparent. On
    /// Qwen-Image 2.1 transparency is prompt-driven (upstream ships no flag); this field only
    /// decides whether the decoder's alpha reaches the caller or is composited over white first.
    pub output_channels: OutputChannels,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            moss_image: None,
            negative_prompt: None,
            width: 1024,
            height: 1024,
            count: 1,
            seed: None,
            steps: None,
            guidance: None,
            true_cfg: None,
            timestep_to_start_cfg: None,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            guidance_method: None,
            guidance_eta: None,
            guidance_momentum: None,
            guidance_norm_threshold: None,
            cfg_interval: None,
            prompt_batch: Vec::new(),
            caption_overflow: None,
            conditioning: Vec::new(),
            flux3_reference_output: None,
            flux3_layout: None,
            flux3_layout_bindings: Vec::new(),
            strength: None,
            control_scale: None,
            text_style_gain: None,
            image_guidance: None,
            frames: None,
            fps: None,
            duration: None,
            video_mode: None,
            veda: None,
            h3_terminal_telemetry: false,
            trim_first_frames: None,
            num_generated_keyframes: None,
            reference_image_short_edge: None,
            temporal_upsample_rounds: None,
            auto_duration: None,
            motion_bucket_id: None,
            noise_aug_strength: None,
            decode_chunk_size: None,
            conditioning_fps: None,
            softness: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            prompt_enhancement: PromptEnhancementSink::default(),
            use_pid: false,
            pid_capture_sigma: None,
            memory: None,
            approximation: None,
            audio: None,
            phases: None,
            cancel: CancelFlag::default(),
            preview: PreviewSink::default(),
            // SDR. The one default that must never drift — see `GenerationRequest::hdr`.
            hdr: None,
            // RGB. Like `hdr`, a default that must never drift: an unasked-for alpha channel
            // reaching a consumer that indexes RGB triples is a corrupted image, not a bonus.
            output_channels: OutputChannels::Rgb,
        }
    }
}

impl GenerationRequest {
    /// Number of image-conditioning inputs represented by this request for memory-evidence
    /// geometry. Multi-image carriers contribute their flattened image count; reference (opaque
    /// **or** transparent), control/depth/mask carriers each contribute one. A keyframe is a
    /// distinct image input even though its placement is temporal; clips remain represented by the
    /// frame axis.
    pub fn image_reference_count(&self) -> u32 {
        self.conditioning.iter().fold(0_u32, |count, conditioning| {
            // **This match is deliberately wildcard-free** (sc-24111), for the same reason
            // `first_nonfinite_float` is: the `_ => 0` this replaces is exactly how
            // `ReferenceRgba` came to be priced at zero. A carrier scored zero here is admitted at
            // one `reference_count` and then refused at execution by both providers' request
            // scopes — see `memory_reference_count` below, whose whole contract is that admission
            // and execution agree carrier for carrier. A new variant now breaks the build here
            // until someone classifies it.
            let increment = match conditioning {
                // One image reference each. `ReferenceRgba` is an ordinary reference that kept its
                // alpha, so it is priced identically to `Reference` — same fit, same token cost.
                Conditioning::Reference { .. }
                | Conditioning::ReferenceRgba { .. }
                | Conditioning::Keyframe { .. }
                | Conditioning::Control { .. }
                | Conditioning::Depth { .. }
                | Conditioning::Mask { .. } => 1,
                // Multi-image carriers contribute their flattened image count.
                Conditioning::MultiReference { images } => {
                    u32::try_from(images.len()).unwrap_or(u32::MAX)
                }
                Conditioning::ReduxRefs { refs } => u32::try_from(refs.len()).unwrap_or(u32::MAX),
                // Carriers that are **not** still-image references: audio and video payloads, the
                // clip carriers (represented by the frame axis, not the image count), and the
                // tensor-free conversation history. Named rather than wildcarded.
                Conditioning::ReferenceAudio { .. }
                | Conditioning::ReferenceVideo { .. }
                | Conditioning::AudioEdit { .. }
                | Conditioning::AudioEditRegions { .. }
                | Conditioning::VoiceEmbedding { .. }
                | Conditioning::VideoClip { .. }
                | Conditioning::ControlClip { .. }
                | Conditioning::VideoSync { .. }
                | Conditioning::ConversationHistory { .. } => 0,
            };
            count.saturating_add(increment)
        })
    }

    /// Conceptual reference count used by memory evidence and request scopes. SCAIL-2's physical
    /// `Reference + Mask + ControlClip` list describes one character identity, not two images (or
    /// three generic conditioning entries); every other carrier retains the generic image count.
    ///
    /// This is the **execution-side** half of a pair: the admission side computes the same quantity
    /// from the request via
    /// [`wan_i2v_memory::geometry_from_request`](crate::wan_i2v_memory::geometry_from_request), and
    /// both request-scope cores reject a request whose `memory_reference_count()` differs from the
    /// admitted `MemoryGeometry::reference_count`. The two therefore have to agree carrier for
    /// carrier or the mode is unreachable by construction — admission and execution can never both
    /// pass. The temporal-carrier arms below exist to close exactly that gap:
    ///
    /// * `video_bridge` owns two immutable source boundaries — two `Keyframe`s (Wan2.2 Ti2V-5B) or
    ///   one masked `ControlClip` (Wan-VACE) — and generates only the middle. Those are execution
    ///   inputs, not conceptual image references, so a bridge cannot borrow a still-image /
    ///   first-last memory row. Admission already declares this as zero.
    /// * `extend_clip` carries exactly one source: a boundary `Keyframe` (Ti2V-5B, already counted
    ///   as one image) or the whole source clip as one masked `ControlClip` (VACE). The generic
    ///   image count scores the VACE carrier as zero while admission scores it as one, so the VACE
    ///   arm is spelled out here.
    pub fn memory_reference_count(&self) -> u32 {
        match self.video_mode.as_deref() {
            Some("animation" | "replacement") if self.scail2_animation_conditioning().is_ok() => 1,
            Some("video_bridge") => 0,
            Some("extend_clip")
                if matches!(
                    self.conditioning.as_slice(),
                    [Conditioning::ControlClip { .. }]
                ) =>
            {
                1
            }
            _ => self.image_reference_count(),
        }
    }

    /// The first request-supplied `Option<f32>` knob that is **non-finite** (NaN / ±Inf), returned as
    /// `(field, value)` — or `None` when every present float is finite. This is the single home of the
    /// finiteness floor (F-053 / F-001): a NaN/Inf on *any* float knob flows into the guidance /
    /// scheduler / conditioning math and silently poisons the whole denoise (garbage-as-success, no
    /// error), so it must be rejected at the contract boundary.
    ///
    /// The exhaustive destructuring below (no `..`) is deliberate and load-bearing: adding a field to
    /// [`GenerationRequest`] fails to compile *here*, forcing the author to decide whether the new knob
    /// is a float that must join the guard. New `Option<f32>` fields therefore inherit the finiteness
    /// check **by construction** instead of silently slipping past it — the recurring "the floor lags
    /// the request surface" regression this method exists to close.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            // Non-float fields: explicitly ignored, but named (no `..`) so a newly-added field breaks
            // the build here and the author must classify it.
            prompt: _,
            // MOSS sub-block floats join the floor through its exhaustive helper below.
            moss_image,
            negative_prompt: _,
            width: _,
            height: _,
            count: _,
            seed: _,
            steps: _,
            timestep_to_start_cfg: _,
            sampler: _,
            scheduler: _,
            guidance_method: _,
            // A string list and a unit enum (sc-25681): no floats to classify. `cfg_interval`'s two
            // bounds join the float floor below.
            prompt_batch: _,
            caption_overflow: _,
            // Optional integer output dimensions, not floating controls.
            flux3_reference_output: _,
            // Layout construction validates f64 coordinates, before conversion to native rows.
            flux3_layout: _,
            flux3_layout_bindings: _,
            cfg_interval,
            conditioning,
            frames: _,
            fps: _,
            video_mode: _,
            veda: _,
            h3_terminal_telemetry: _,
            trim_first_frames: _,
            // Integer DFR knobs (sc-18789): slot count + round count carry no floats.
            num_generated_keyframes: _,
            // sc-23402: an integer pixel extent, range-checked by
            // `validate_reference_image_short_edge` rather than by the float floor.
            reference_image_short_edge: _,
            temporal_upsample_rounds: _,
            // sc-18778: the auto-duration range's floats are validated at construction
            // (`AutoDurationRange::new` refuses non-finite / inverted / non-positive bounds), so
            // they never reach this sweep unchecked.
            auto_duration: _,
            decode_chunk_size: _,
            conditioning_fps: _,
            enhance_prompt: _,
            use_uncensored_enhancer: _,
            enhance_max_tokens: _,
            prompt_enhancement: _,
            use_pid: _,
            memory: _,
            // The approximation sub-block is float-free by construction (sc-18322): its policy is
            // step counts and its characterization reference is opaque strings. A float-bearing
            // approximate parameter — a quality/cost threshold, say — must join the floor, and this
            // named-not-`..` binding is what forces that decision when one arrives.
            approximation: _,
            cancel: _,
            preview: _,
            // Float-free by construction: an enum colour space plus a callback handle. A
            // float-bearing HDR knob (a diffuse-white signal or roll-off rate on the request,
            // say) must join the floor, and this named-not-`..` binding forces that decision.
            hdr: _,
            // An enum with two unit variants — no float to check (sc-24111).
            output_channels: _,
            // The audio sub-block carries its own floats — destructured below the flat knobs.
            audio,
            // The multi-phase list carries per-phase floats (guidance + adapter weights), checked
            // below the flat knobs (sc-13884). Named (no `..`) so a future float-bearing per-phase
            // control fails to compile here until it is classified into the floor.
            phases,
            // Every `Option<f32>` knob the floor owns.
            guidance,
            true_cfg,
            scheduler_shift,
            guidance_eta,
            guidance_momentum,
            guidance_norm_threshold,
            strength,
            control_scale,
            image_guidance,
            duration,
            motion_bucket_id,
            noise_aug_strength,
            softness,
            enhance_temperature,
            pid_capture_sigma,
            text_style_gain,
        } = self;
        if let Some(nonfinite) = moss_image.as_ref().and_then(|v| v.first_nonfinite_float()) {
            return Some(nonfinite);
        }
        let floats: [(&'static str, Option<f32>); 16] = [
            ("guidance", *guidance),
            ("true_cfg", *true_cfg),
            ("scheduler_shift", *scheduler_shift),
            ("guidance_eta", *guidance_eta),
            ("guidance_momentum", *guidance_momentum),
            ("guidance_norm_threshold", *guidance_norm_threshold),
            ("strength", *strength),
            ("control_scale", *control_scale),
            ("image_guidance", *image_guidance),
            ("duration", *duration),
            ("motion_bucket_id", *motion_bucket_id),
            ("noise_aug_strength", *noise_aug_strength),
            ("softness", *softness),
            ("enhance_temperature", *enhance_temperature),
            ("pid_capture_sigma", *pid_capture_sigma),
            ("text_style_gain", *text_style_gain),
        ];
        let interval: [(&'static str, Option<f32>); 2] = [
            ("cfg_interval.lo", cfg_interval.map(|(lo, _)| lo)),
            ("cfg_interval.hi", cfg_interval.map(|(_, hi)| hi)),
        ];
        for (name, v) in floats.into_iter().chain(interval) {
            if let Some(x) = v {
                if !x.is_finite() {
                    return Some((name, x));
                }
            }
        }
        // Audio sub-block floats (sc-12834): destructured exhaustively (no `..`) for the same
        // reason as the request itself — a new `AudioParams` float field fails to compile here
        // until it is classified into the floor.
        if let Some(AudioParams {
            voice: _,
            language: _,
            sample_rate: _,
            musical_key: _,
            lyrics: _,
            // The script carries no floats (text + opaque labels); named (no `..`) so a future
            // float-bearing per-segment control fails to compile here until it is classified.
            script: _,
            // Integer counts (sc-19382): no floats to classify.
            segments: _,
            max_new_tokens_per_segment: _,
            // A unit enum (sc-19378): no floats to classify.
            output_limiter: _,
            // A directory and a flag (sc-22994): no floats to classify.
            artifacts: _,
            target_duration,
            bpm,
            repetition_penalty,
            reference_region,
            song,
        }) = audio
        {
            // Symbolic-song token sampling (sc-22994): its three `f64` fields go through the same
            // non-finite check. Destructured without `..` so a new float-bearing field fails to
            // compile here until it is classified.
            if let Some(SongParams {
                planning: _,
                score: _,
                plan: _,
                cached_latents: _,
                decoder: _,
                // A flag and a text-only cover block (sc-22988): no floats to classify.
                plan_only: _,
                cover: _,
                score_sampling,
                semantic_sampling,
            }) = song
            {
                for (phase, sampling) in
                    [("score", score_sampling), ("semantic", semantic_sampling)]
                {
                    if let Some(TokenSampling {
                        temperature,
                        top_p,
                        repetition_penalty,
                        top_k: _,
                        penalty_window: _,
                        min_tokens: _,
                        max_tokens: _,
                    }) = sampling
                    {
                        for (field, v) in [
                            ("temperature", temperature),
                            ("top_p", top_p),
                            ("repetition_penalty", repetition_penalty),
                        ] {
                            if let Some(x) = v {
                                if !x.is_finite() {
                                    let name = match (phase, field) {
                                        ("score", "temperature") => {
                                            "audio.song.score_sampling.temperature"
                                        }
                                        ("score", "top_p") => "audio.song.score_sampling.top_p",
                                        ("score", _) => {
                                            "audio.song.score_sampling.repetition_penalty"
                                        }
                                        (_, "temperature") => {
                                            "audio.song.semantic_sampling.temperature"
                                        }
                                        (_, "top_p") => "audio.song.semantic_sampling.top_p",
                                        _ => "audio.song.semantic_sampling.repetition_penalty",
                                    };
                                    return Some((name, *x as f32));
                                }
                            }
                        }
                    }
                }
            }
            let audio_floats: [(&'static str, Option<f32>); 5] = [
                ("audio.target_duration", *target_duration),
                ("audio.bpm", *bpm),
                ("audio.repetition_penalty", *repetition_penalty),
                (
                    "audio.reference_region.start_secs",
                    reference_region.map(|r| r.start_secs),
                ),
                (
                    "audio.reference_region.end_secs",
                    reference_region.and_then(|r| r.end_secs),
                ),
            ];
            for (name, v) in audio_floats {
                if let Some(x) = v {
                    if !x.is_finite() {
                        return Some((name, x));
                    }
                }
            }
        }
        // Conditioning-carried floats the floor also owns (F-001): every numeric a `Conditioning`
        // variant carries flows into the same denoise / scheduler / mask math as the flat knobs.
        //
        // **This match is deliberately wildcard-free and guard-free** (sc-19571). It used to end in
        // `_ => {}`, and that one arm is the whole reason the floor lagged the request surface: the
        // exhaustive `Self { .. }` destructure above makes a new *field* break the build, but a new
        // float-bearing `Conditioning` **variant** compiled clean and slipped straight past. Four
        // did — `Keyframe.strength`, `VideoClip.strength`, `ControlClip.masking_strength` and every
        // `ReduxRefs` per-ref strength — each of them a `1 − strength` denoise mask or a blend
        // weight, i.e. exactly the math this method exists to protect. Guards are avoided for the
        // same reason: `match` exhaustiveness ignores them, so a guarded arm re-opens the hole.
        // Adding a variant now fails to compile here until its numerics are classified.
        for c in conditioning {
            match c {
                Conditioning::Control { scale, .. } => {
                    if let Some(s) = scale {
                        if !s.is_finite() {
                            return Some(("conditioning.control.scale", *s));
                        }
                    }
                }
                Conditioning::Reference { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference.strength", *s));
                        }
                    }
                }
                // The RGBA sibling of `Reference` (sc-24111): same strength field, same math.
                Conditioning::ReferenceRgba { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference_rgba.strength", *s));
                        }
                    }
                }
                Conditioning::ReferenceAudio { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference_audio.strength", *s));
                        }
                    }
                }
                // A reference clip's declared rate (sc-17149). Not a cosmetic label: it is the
                // divisor of the resample stride that picks which frames the model actually reads,
                // so a NaN propagates into a frame-index computation rather than into denoise math
                // — the same class of silent poisoning, one step earlier.
                Conditioning::ReferenceVideo { fps, .. } => {
                    if !fps.is_finite() {
                        return Some(("conditioning.reference_video.fps", *fps));
                    }
                }
                // The audio-edit strength and its region bounds all flow into the edit-window /
                // blend math; a NaN would silently poison the region conversion or the strength
                // gate (sc-12847).
                Conditioning::AudioEdit {
                    strength, region, ..
                } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.audio_edit.strength", *s));
                        }
                    }
                    if let Some(r) = region {
                        if !r.start_secs.is_finite() {
                            return Some((
                                "conditioning.audio_edit.region.start_secs",
                                r.start_secs,
                            ));
                        }
                        if let Some(end) = r.end_secs {
                            if !end.is_finite() {
                                return Some(("conditioning.audio_edit.region.end_secs", end));
                            }
                        }
                    }
                }
                // The multi-region carrier (sc-14549). Written as a **loop over every** region
                // rather than as more `if`-guarded arms, and that difference is the whole point:
                // the exhaustive destructuring above makes a new *field* break the build, but a
                // `Vec` defeats that mechanism entirely — a guard that reached only `regions[0]`
                // would compile, pass every pre-existing test, and let a NaN in region two flow
                // into the provider's mask rasterisation and poison it silently. That is exactly
                // the "the floor lags the request surface" regression this method exists to close,
                // so the testkit gate puts its bad value in region **two**.
                Conditioning::AudioEditRegions {
                    regions, strength, ..
                } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.audio_edit_regions.strength", *s));
                        }
                    }
                    for region in regions {
                        if !region.start_secs.is_finite() {
                            return Some((
                                "conditioning.audio_edit_regions.regions.start_secs",
                                region.start_secs,
                            ));
                        }
                        if let Some(end) = region.end_secs {
                            if !end.is_finite() {
                                return Some((
                                    "conditioning.audio_edit_regions.regions.end_secs",
                                    end,
                                ));
                            }
                        }
                    }
                }
                Conditioning::VoiceEmbedding { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.voice_embedding.strength", *s));
                        }
                    }
                }
                // sc-19571 — the four that the old `_ => {}` swallowed.
                //
                // A keyframe's `strength` is a `1 − strength` denoise mask on the pinned latent
                // frame AND the per-token diffusion timestep for that frame's tokens (Wan TI2V,
                // LTX). A NaN there does not merely mis-weight a pin: it makes the whole masked
                // frame's timestep NaN, which the denoise then multiplies into every step.
                Conditioning::Keyframe { strength, .. } => {
                    if !strength.is_finite() {
                        return Some(("conditioning.keyframe.strength", *strength));
                    }
                }
                // An in-context clip's `strength` is the same `1 − strength` mask on appended
                // conditioning tokens (LTX IC-LoRA, krea-realtime v2v).
                Conditioning::VideoClip { strength, .. } => {
                    if !strength.is_finite() {
                        return Some(("conditioning.video_clip.strength", *strength));
                    }
                }
                // `masking_strength` gates BOTH the pixel-space neutralization of the person region
                // and the mask-injection step count (`ceil(steps · masking_strength)`), so a NaN
                // reaches an integer step count as well as a blend weight.
                Conditioning::ControlClip {
                    masking_strength, ..
                } => {
                    if !masking_strength.is_finite() {
                        return Some((
                            "conditioning.control_clip.masking_strength",
                            *masking_strength,
                        ));
                    }
                }
                // Per-reference Redux weights. A **loop**, for the `AudioEditRegions` reason above:
                // checking only `refs[0]` would compile and pass, and a NaN in ref two would flow
                // into the conditioning blend unseen.
                Conditioning::ReduxRefs { refs } => {
                    for (_, strength) in refs {
                        if !strength.is_finite() {
                            return Some(("conditioning.redux_refs.strength", *strength));
                        }
                    }
                }
                // Float-free carriers: media and opaque labels only. Named rather than swallowed by
                // a wildcard so that adding a numeric to any of them breaks this match.
                Conditioning::MultiReference { images: _ } => {}
                Conditioning::Depth { image: _ } => {}
                Conditioning::Mask { image: _ } => {}
                Conditioning::VideoSync { frames: _ } => {}
                Conditioning::ConversationHistory { turns: _ } => {}
            }
        }
        // Multi-phase denoise floats (sc-13884): each phase's guidance and each phase-adapter weight
        // flow into the same guidance / adapter-scale math as the flat knobs, so a NaN/Inf must be
        // rejected here too rather than silently poisoning the phase's forward.
        if let Some(phases) = phases {
            for ph in phases {
                if let Some(g) = ph.guidance {
                    if !g.is_finite() {
                        return Some(("phases.guidance", g));
                    }
                }
                for pa in &ph.adapters {
                    if let Some(w) = pa.weight {
                        if !w.is_finite() {
                            return Some(("phases.adapter.weight", w));
                        }
                    }
                }
            }
        }
        None
    }

    /// Reject the request when any `Option<f32>` knob is non-finite (see
    /// [`first_nonfinite_float`](Self::first_nonfinite_float)). The shared home of the F-053 / F-001
    /// finiteness floor: [`Capabilities::validate_request`] calls it, and providers with a bespoke
    /// `validate` (e.g. flux1's IP-Adapter carve-out) call it directly so they inherit the guard too.
    pub fn ensure_finite_floats(&self) -> Result<()> {
        if let Some((field, value)) = self.first_nonfinite_float() {
            return Err(Error::Msg(format!("{field} must be finite (got {value})")));
        }
        Ok(())
    }

    /// All [`Conditioning::Keyframe`] inputs (first_last_frame / multi-keyframe), in request order.
    pub fn keyframes(&self) -> Vec<KeyframeRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::Keyframe {
                    image,
                    frame_idx,
                    strength,
                } => Some(KeyframeRef {
                    image,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// All [`Conditioning::VideoClip`] in-context clips (extend_clip / video_bridge), in request order.
    pub fn video_clips(&self) -> Vec<VideoClipRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::VideoClip {
                    frames,
                    frame_idx,
                    strength,
                } => Some(VideoClipRef {
                    frames,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// The replace_person masked control clip ([`Conditioning::ControlClip`]), if present. The first
    /// one wins (a request carries at most one person edit per generation).
    pub fn control_clip(&self) -> Option<ControlClipRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                mode,
            } => Some(ControlClipRef {
                frames,
                mask,
                masking_strength: *masking_strength,
                start_frame: *start_frame,
                mode: *mode,
            }),
            _ => None,
        })
    }

    /// Parse the exact ordered SCAIL-2 animation/replacement carrier. This deliberately does not infer the
    /// route from `conditioning.len() == 3`: every position and field is typed and crossed shapes
    /// fail closed before a provider configures or loads tensors.
    pub fn scail2_animation_conditioning(
        &self,
    ) -> crate::Result<Scail2AnimationConditioningRef<'_>> {
        let [Conditioning::Reference {
            image: character,
            strength: None,
        }, Conditioning::Mask {
            image: character_mask,
        }, Conditioning::ControlClip {
            frames,
            mask: driving_masks,
            masking_strength,
            start_frame,
            mode,
        }] = self.conditioning.as_slice()
        else {
            return Err(Error::Unsupported(
                "scail2 requires exactly ordered Reference(strength unset), Mask, and ControlClip conditioning"
                    .to_owned(),
            ));
        };
        if frames.is_empty() || driving_masks.len() != frames.len() {
            return Err(Error::Unsupported(format!(
                "scail2 requires one driving mask per non-empty driving frame ({} frames, {} masks)",
                frames.len(),
                driving_masks.len()
            )));
        }
        if *masking_strength != 1.0 || *start_frame != 0 || *mode != ReplacementMode::default() {
            return Err(Error::Unsupported(
                "scail2 ControlClip requires masking_strength=1, start_frame=0, and full-person mode"
                    .to_owned(),
            ));
        }
        if character.width == 0
            || character.height == 0
            || (character_mask.width, character_mask.height) != (character.width, character.height)
        {
            return Err(Error::Unsupported(
                "scail2 requires a non-empty character image and an exact-shape character mask"
                    .to_owned(),
            ));
        }
        let control_shape = (frames[0].width, frames[0].height);
        if control_shape.0 == 0
            || control_shape.1 == 0
            || frames.iter().zip(driving_masks).any(|(frame, mask)| {
                (frame.width, frame.height) != control_shape
                    || (mask.width, mask.height) != control_shape
            })
        {
            return Err(Error::Unsupported(
                "scail2 requires uniform non-empty driving frames with one exact-shape mask each"
                    .to_owned(),
            ));
        }
        Ok(Scail2AnimationConditioningRef {
            character,
            character_mask,
            driving_frames: frames,
            driving_masks,
        })
    }

    /// The prompted audio-edit conditioning ([`Conditioning::AudioEdit`]), if present. The first one
    /// wins (a request carries at most one source-audio edit per generation, mirroring
    /// [`control_clip`](Self::control_clip)). sc-12847.
    pub fn audio_edit(&self) -> Option<AudioEditRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::AudioEdit {
                audio,
                mode,
                region,
                strength,
            } => Some(AudioEditRef {
                audio,
                mode: *mode,
                region: *region,
                strength: *strength,
            }),
            _ => None,
        })
    }

    /// The **multi-region** audio-edit conditioning ([`Conditioning::AudioEditRegions`]), if present
    /// (sc-14549). The first one wins, exactly as [`audio_edit`](Self::audio_edit) does — a request
    /// carries at most one source-audio edit per generation, and a provider that wants to refuse a
    /// second carrier rather than silently take the first inspects `conditioning` directly (which is
    /// what Stable Audio 3 does).
    ///
    /// This is a **separate accessor** from [`audio_edit`](Self::audio_edit), which is left
    /// byte-identical: a single-region caller keeps its exact shape, and a provider that has not
    /// opted into multi-region never sees one because
    /// [`ConditioningKind::AudioEditRegions`] is a distinct kind the shared allowlist refuses.
    pub fn audio_edit_regions(&self) -> Option<AudioEditRegionsRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::AudioEditRegions {
                audio,
                mode,
                regions,
                strength,
            } => Some(AudioEditRegionsRef {
                audio,
                mode: *mode,
                regions,
                strength: *strength,
            }),
            _ => None,
        })
    }
}

/// The last value [`default_seed`] handed out in this process — the seam that makes two
/// back-to-back calls distinct even when they share a clock tick (sc-24114).
pub(super) static LAST_DEFAULT_SEED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Seed when a [`GenerationRequest`] omits one: nanos since the epoch (any nonzero value works —
/// this only sets which sample is drawn; a caller wanting reproducibility passes `req.seed`).
/// Shared by every generator (F-006).
///
/// **Never returns the same value twice in one process.** The clock alone does not guarantee
/// that: `SystemTime::now()` is coarser than a nanosecond on every platform this runs on, so two
/// unseeded requests resolved back to back could draw the identical seed and render the identical
/// sample. Each call therefore returns `max(now, last + 1)` through a compare-and-swap, which is
/// strictly increasing across calls and threads and lands back on the clock as soon as it moves
/// past the last value handed out.
pub fn default_seed() -> u64 {
    use std::sync::atomic::Ordering;
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        // Fall back to a nonzero value: 0 is the "no seed" sentinel a caller would pass to mean
        // "pick one", so the default must never itself be 0 (F-089).
        .unwrap_or(1);
    let mut last = LAST_DEFAULT_SEED.load(Ordering::Relaxed);
    loop {
        let candidate = now.max(last.wrapping_add(1)).max(1);
        match LAST_DEFAULT_SEED.compare_exchange_weak(
            last,
            candidate,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => return candidate,
            Err(observed) => last = observed,
        }
    }
}

/// Smallest admitted [`GenerationRequest::reference_image_short_edge`] (sc-23402).
///
/// The floor is a judgement, not a checkpoint constant: below roughly 1024 the reference stops
/// carrying the subject detail the conditioner and the reference-latent rows are there to supply,
/// and the saving is already ~4× the token count of the default.
pub const REFERENCE_IMAGE_SHORT_EDGE_MIN: u32 = 1024;

/// Largest admitted [`GenerationRequest::reference_image_short_edge`], equal to
/// [`REFERENCE_IMAGE_SHORT_EDGE_DEFAULT`] (sc-23402).
///
/// The knob only ever goes **downward** from upstream's default: upstream's value is the released
/// checkpoint's own rule (see `mlx_gen_minimax_h3::reference::REFERENCE_IMAGE_SHORT_EDGE` for the
/// citation), so admitting more than it would be inventing a regime nothing was conditioned on.
pub const REFERENCE_IMAGE_SHORT_EDGE_MAX: u32 = 2048;

/// [`GenerationRequest::reference_image_short_edge`] when the request omits one — upstream's own
/// `reference_image_short_edge` pipeline default (sc-23402).
pub const REFERENCE_IMAGE_SHORT_EDGE_DEFAULT: u32 = 2048;

/// The **effective** reference-image short edge for one request: the requested value, or
/// [`REFERENCE_IMAGE_SHORT_EDGE_DEFAULT`] when it omits one (sc-23402).
///
/// One resolver, shared by the engine that applies the value and by the consumer that records it in
/// the attempt's recipe, so the recorded number cannot drift from the rendered one. It resolves
/// *without* range-checking — [`validate_reference_image_short_edge`] is the refusal, and it runs
/// before any weight is read.
pub fn effective_reference_image_short_edge(req: &GenerationRequest) -> u32 {
    req.reference_image_short_edge
        .unwrap_or(REFERENCE_IMAGE_SHORT_EDGE_DEFAULT)
}

/// Refuse an out-of-range [`GenerationRequest::reference_image_short_edge`] (sc-23402).
///
/// **Fail closed, never clamp.** The value changes the reference token count, which is the whole
/// reason a caller sets it; a silently clamped request would report one budget and render another.
/// The message names the field, the value and the admitted range so the caller can tell a typo from
/// an unsupported control.
///
/// `id` is the model's descriptor id, as everywhere else on the request floor. A request that omits
/// the field validates vacuously, so this is inert for every caller that has not opted in.
pub fn validate_reference_image_short_edge(id: &str, req: &GenerationRequest) -> Result<()> {
    if let Some(edge) = req.reference_image_short_edge {
        if !(REFERENCE_IMAGE_SHORT_EDGE_MIN..=REFERENCE_IMAGE_SHORT_EDGE_MAX).contains(&edge) {
            return Err(Error::Msg(format!(
                "{id}: reference_image_short_edge must be in \
                 {REFERENCE_IMAGE_SHORT_EDGE_MIN}..={REFERENCE_IMAGE_SHORT_EDGE_MAX}, got {edge} \
                 (it sizes the REFERENCE, not the render — it is refused rather than clamped \
                 because clamping would change the reference token budget silently)"
            )));
        }
    }
    Ok(())
}
