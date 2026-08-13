//! SCAIL-2 provider: capability surface, registration, snapshot/config resolution, and the
//! [`Generator`] entrypoint.
//!
//! [`Generator::generate`] maps the [`GenerationRequest`] conditioning onto the SCAIL-2 inputs and
//! runs the live [`crate::generate()`] denoise pipeline: the primary **reference character** is a
//! [`Conditioning::Reference`] image paired with its color-coded [`Conditioning::Mask`]; the
//! **driving video + per-frame color masks** are a [`Conditioning::ControlClip`]; `video_mode ==
//! "replacement"` toggles the cross-identity `replace_flag` (else animation). Inference LoRA(s) from
//! [`LoadSpec::adapters`] (the Bias-Aware DPO refinement LoRA + a lightx2v step-distill lightning
//! LoRA, sc-5451) install onto the DiT as forward-time residuals. Multi-reference (extra characters,
//! each needing its own paired mask) awaits the sc-5583 request contract; the [`crate::generate()`]
//! core already supports extra characters via [`crate::CharacterRef`].

use std::path::PathBuf;

use mlx_gen::{
    default_seed, AdapterSpec, Capabilities, Conditioning, ConditioningKind, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Progress, Quant, Result, SizeFloor, WeightsSource,
};
use mlx_gen_wan::config::MAX_AREA_14B;
use mlx_gen_wan::pipeline::reject_over_area_dims;
use mlx_gen_wan::SolverKind;

use crate::config::Scail2Config;
use crate::generate::{align, CharacterRef, Scail2Job, DIM_ALIGN};

/// Default driving-segment window + clean-history overlap (upstream `scail.py` defaults).
const SEGMENT_LEN: usize = 81;
const SEGMENT_OVERLAP: usize = 5;
/// Upstream `generate()` sampler defaults: 40 steps, shift 5.0 (3.0 at 480p), guide 5.0, 16 fps.
const DEFAULT_STEPS: u32 = 40;
const DEFAULT_SHIFT: f32 = 5.0;
const DEFAULT_GUIDANCE: f32 = 5.0;
const DEFAULT_FPS: u32 = 16;

/// SceneWorks/engine model id. A still image is `num_frames == 1`.
pub const MODEL_ID: &str = "scail2_14b";

/// Stable identity + advertised capabilities for SCAIL-2 (Wan2.1-14B I2V end-to-end character
/// animation: reference image + driving video + color-coded masks → animated/identity-replaced video;
/// plain single-scale CFG; packed-token conditioning + per-source RoPE + CLIP image cross-attn).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "scail2",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // ADVERTISED, not merely implemented. `width`/`height == 0` means "size from the
            // driving-video frames", so the shared floor exempts that sentinel while applying the
            // provider's downstream envelope check. Explicit dimensions are different: the caller
            // chose them, so the same descriptor advertises the exact 32-pixel grid and the shared
            // floor rejects an off-grid request before a load (sc-16198).
            size_floor: SizeFloor::ResolvedDownstreamExplicitGrid {
                multiple: DIM_ALIGN,
            },
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Reference character image (Reference) + its color-coded segmentation mask (Mask); extra
            // characters (MultiReference, experimental); the driving video + its per-frame color masks
            // map to ControlClip.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::MultiReference,
                ConditioningKind::ControlClip,
            ],
            // Inference LoRA (the Bias-Aware DPO refinement LoRA + a lightx2v step-distill lightning
            // LoRA) installs as a forward-time residual over the (possibly Q4/Q8) base via the
            // family-agnostic loader — SCAIL-2 is Wan2.1-14B I2V, so a Wan-I2V LoRA resolves directly
            // (sc-5451). LoKr/LoHa ride the same residual path.
            supports_lora: true,
            supports_lokr: true,
            samplers: vec!["unipc", "dpm++"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: true,
            requires_sigma_shift: false,
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
            unconditionally_engages_staged_residency: false,
            supports_preview: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
        },
    }
}

/// The loaded SCAIL-2 model: resolved config + snapshot dir + optional load-time quant. The heavy
/// components (DiT / VAE / UMT5 / CLIP) are staged per-stage inside [`crate::generate()`].
pub struct Scail2 {
    descriptor: ModelDescriptor,
    config: Scail2Config,
    root: PathBuf,
    /// Q4/Q8 load-time quant (sc-5445) — applied to the DiT in [`crate::generate::generate`].
    quant: Option<Quant>,
    /// Inference LoRA(s) from [`LoadSpec::adapters`] (the Bias-Aware DPO / lightx2v lightning LoRA,
    /// sc-5451) — installed onto the DiT as forward-time residuals in [`crate::generate::generate`].
    adapters: Vec<AdapterSpec>,
}

/// Load SCAIL-2 from a converted MLX snapshot directory (`dit.safetensors` + `config.json` +
/// `Wan2.1_VAE.pth` + `umt5-xxl/` + the open-CLIP XLM-RoBERTa ViT-H/14 visual encoder), as published
/// to `SceneWorks/scail2-mlx`.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "scail2: expected a model directory (converted MLX snapshot), not a single file"
                    .into(),
            )),
        };
    if !root.exists() {
        return Err(Error::Msg(format!(
            "scail2: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    let config = Scail2Config::from_model_dir(&root)?;
    Ok(Box::new(Scail2 {
        descriptor: descriptor(),
        config,
        root,
        quant: spec.quantize,
        adapters: spec.adapters.clone(),
    }))
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_generators! { pub(crate) const REGISTRATION = descriptor => load }

mlx_gen::impl_generator!(Scail2 {
    validate: |s, req| {
        s.descriptor
            .capabilities
            .validate_request(s.descriptor.id, req)?;
        // The RESOLVED geometry, which the shared floor above cannot see (sc-16167). Only the `0x0`
        // sentinel needs the driving clip to resolve; an EXPLICIT size is checkable with no clip at
        // all, and must be — candle's `validate` area-checks `req` unconditionally, so gating the
        // whole thing behind a clip would leave the pre-flight divergent whenever the conditioning is
        // not attached yet. `None` only when the sentinel has nothing to resolve against, which `run`
        // reports as its own missing-ControlClip error rather than as a geometry one.
        resolve_pre_flight_size(req).map_or(Ok(()), |(w, h)| {
            reject_unrenderable_geometry(&s.descriptor.capabilities, req, w, h)
        })
    },
    generate: run,
});

/// The geometry the pipeline will actually render: the request's `width`/`height` where non-zero,
/// else the driving clip's first frame (SCAIL-2's "match the driving video" convention). Per axis,
/// matching [`Scail2::run`] — only the zero axis is filled from the frame.
fn resolve_target_size(req: &GenerationRequest, first: &Image) -> (u32, u32) {
    (
        if req.width > 0 {
            req.width
        } else {
            first.width
        },
        if req.height > 0 {
            req.height
        } else {
            first.height
        },
    )
}

/// [`resolve_target_size`] for the pre-flight, where the driving clip may not be attached yet:
/// `None` exactly when the request needs one and has none.
///
/// A half-sentinel (`0x512`) still needs the clip, and cannot reach here anyway — the shared floor
/// range-checks it, since only `0x0` is exempt.
fn resolve_pre_flight_size(req: &GenerationRequest) -> Option<(u32, u32)> {
    if req.width > 0 && req.height > 0 {
        return Some((req.width, req.height));
    }
    req.control_clip()
        .and_then(|c| c.frames.first())
        .map(|first| resolve_target_size(req, first))
}

/// Refuse a geometry SCAIL-2 does not advertise, **before** the render — the second half of the
/// [`SizeFloor::ResolvedDownstream`] contract (sc-16167).
///
/// The shared floor deliberately exempts the `0x0` sentinel from its size-range check, because `0`
/// means "resolve from the driving video" and range-checking it would reject the convention. That
/// exemption promises the provider re-checks what it resolved to; until this function existed, no
/// provider did, so an auto-sized request was bounded by nothing at all: a 4K driving clip resolved
/// to `3840x2160` against declared bounds of `32..=1280` and rendered. Nothing downstream would have
/// caught it — [`crate::generate::align`] only snaps to the [`DIM_ALIGN`] lattice, so 4K becomes
/// `3840x2144`, not a working resolution. On a 14B DiT with a packed conditioning sequence over 2x
/// the plain token count, that is minutes of GPU time ending in an OOM instead of a fast refusal.
///
/// Both bounds are checked, because neither implies the other:
///
/// * the **range**, per edge, against the descriptor's own `min_size`/`max_size` — read from
///   [`Capabilities`] rather than re-typed here, so the check can never drift from the advertisement
///   a weights-free consumer holds;
/// * the **area**, via the shared [`reject_over_area_dims`], because `max_size` alone bounds each
///   edge independently — `1280x1280` is 1.64 Mpx with both edges inside `max_size`. That half is
///   what the candle twin has enforced since sc-11215 and MLX never did (F-090's edge, epic 11146's
///   "class-wide sweeps have edges"), so it bites on an **explicit** size too, not only the sentinel.
///
/// The range half, by contrast, is new **only** on the sentinel path: an explicit out-of-range size
/// was already refused by the shared floor above, which exempts `0x0` and nothing else.
///
/// # Explicit-grid and sentinel parity
///
/// The **area** halves now agree: both measure the lattice-aligned geometry — what actually renders.
/// candle used to measure `req.width * req.height` raw, so a request between the two, e.g. `1280x730`
/// (934 400 raw, 901 120 once `730` floors to `704`), was refused there and accepted here; sc-16197
/// moved `candle-gen-scail2`'s area helper onto its render lattice.
///
/// sc-16198 then removed the observable explicit-request band: both backends advertise the 32-pixel
/// grid and reject an off-grid explicit size before either area calculation. MLX still measures
/// lattice-aligned geometry here because this function also checks a size resolved from the driving
/// clip, whose source-media geometry may be off-grid.
///
/// sc-16199 subsequently brought the Candle sibling onto this same safe sentinel contract: both
/// backends now resolve `0x0` from the driving clip and bound the resolved geometry before rendering.
///
/// The paired [`reject_off_grid`](mlx_gen_wan::pipeline::reject_off_grid) from the `model_vace.rs`
/// site cannot be applied to this *resolved* geometry unchanged: it would refuse an ordinary
/// `640x360` driving clip that renders at `640x352`, even though the caller never chose that size.
/// The asymmetry is now explicit in
/// [`SizeFloor::ResolvedDownstreamExplicitGrid`](mlx_gen::gen_core::SizeFloor::ResolvedDownstreamExplicitGrid):
/// explicit sizes are exact-or-rejected; resolved source-media sizes retain the alignment.
///
/// Deliberately a refusal, not a downscale. Silently rendering a geometry the caller did not ask for
/// is the exact shape epic 15448 exists to eliminate, and `resolve_capped_dims`' silent refit was
/// already removed from the wan family for that reason (sc-12308). The message instead **names** the
/// largest in-envelope geometry at the source aspect ([`suggest_in_envelope`]), so the caller re-sends
/// with an explicit `width`/`height` and keeps the same driving clip — `run` resizes the driving
/// frames to the target either way. That suggestion is a statement about what this engine will
/// *accept*, not a measured memory fit.
///
/// sc-15807 is building the typed carrier for exactly this shape (inference#333, draft at time of
/// writing): `Error::GeometryRefused { reason, requested_width, requested_height, alternative:
/// Option<(u32, u32)> }`. [`suggest_in_envelope`] already returns that `Option<(u32, u32)>`, so the
/// upgrade here is swapping the [`Error::Msg`] below for that variant — the refusal itself, and the
/// decision to refuse rather than downscale, do not change. What changes is that the alternative
/// stops being prose a caller has to parse.
fn reject_unrenderable_geometry(
    caps: &Capabilities,
    req: &GenerationRequest,
    width: u32,
    height: u32,
) -> Result<()> {
    // Which half of the request produced this geometry, so the caller knows where to change it.
    let origin = if req.width == 0 && req.height == 0 {
        "resolved from the driving video"
    } else {
        "requested"
    };
    let range = width < caps.min_size
        || width > caps.max_size
        || height < caps.min_size
        || height > caps.max_size;
    let area = reject_over_area_dims(MODEL_ID, width, height, DIM_ALIGN, DIM_ALIGN, MAX_AREA_14B);
    if !range && area.is_ok() {
        return Ok(());
    }
    let reason = if range {
        format!(
            "each edge must be within {}..={}",
            caps.min_size, caps.max_size
        )
    } else {
        // The area the gate measured, not `width × height` — those differ off-lattice, and quoting
        // the raw product would send the caller looking for an off-by-one that isn't there. Every
        // sibling refusal reports the measured area the same way (`mlx-gen-wan/src/pipeline.rs`).
        let (aw, ah) = (
            width / DIM_ALIGN * DIM_ALIGN,
            height / DIM_ALIGN * DIM_ALIGN,
        );
        format!(
            "the rendered {aw}×{ah} is {} px, over the max area {MAX_AREA_14B} px",
            aw as usize * ah as usize
        )
    };
    let advice = match suggest_in_envelope(width, height, caps.min_size, caps.max_size) {
        Some((sw, sh)) => format!(
            "pass an explicit width/height — the largest geometry this engine accepts at this aspect \
             is {sw}×{sh} (the same driving clip is resized to whatever target you set)"
        ),
        None => "pass an explicit width/height inside the advertised range".to_string(),
    };
    Err(Error::Msg(format!(
        "{MODEL_ID}: {width}×{height} ({origin}) is outside this model's advertised size envelope — \
         {reason}; {advice}"
    )))
}

/// The largest geometry inside SCAIL-2's advertised envelope that holds `w`/`h`'s aspect ratio and
/// lands on the [`DIM_ALIGN`] lattice the pipeline renders on — the actionable half of
/// [`reject_unrenderable_geometry`]'s message.
///
/// A descending walk of the lattice rather than the obvious `w · min(max_size/w, √(max_area/w·h))`,
/// because that form is a float round-trip feeding a **truncating** `as u32`: whenever the product
/// lands at `x.999…` rather than exactly `x`, the truncation drops a whole [`DIM_ALIGN`] step and the
/// suggestion comes out one lattice narrower than the right answer. Whether a given ratio trips it
/// depends on `f64` rounding — not a property worth depending on. The walk is at most
/// `max_size / DIM_ALIGN` (40) iterations of integer arithmetic and cannot drift at all.
///
/// At each candidate width the aspect-derived height is tried **rounded up** to the lattice before
/// rounded down, taking the larger of the two whenever it still fits. That tie-break is what keeps
/// the suggestion on a bucket a user recognises: a `1080x1920` phone clip yields `704x1280`, which is
/// a resolution the model manifest actually advertises, where preferring the rounded-down `704x1248`
/// would name a geometry on no menu entry anywhere in the product. Landing off every advertised
/// bucket is precisely the failure sc-12308 exists to prevent, and it would be perverse to reproduce
/// it in the message telling the caller how to avoid it.
///
/// The widest candidate is the **source** snapped down onto the lattice, never `max_size`: a
/// suggestion that upscales past the driving clip is not an answer to "this is too big". A source
/// below `min_size` therefore yields no candidate and returns `None`, which is honest — no
/// in-envelope geometry matches its aspect at or below its own size — and the caller falls back to
/// the generic "inside the advertised range" advice. `None` also covers a degenerate zero edge,
/// which has no aspect to preserve.
fn suggest_in_envelope(w: u32, h: u32, min_size: u32, max_size: u32) -> Option<(u32, u32)> {
    if w == 0 || h == 0 {
        return None;
    }
    let step = u64::from(DIM_ALIGN);
    let clamp = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    // `min_size` can never be below one lattice step here, so the `-=` below cannot underflow.
    let floor = min_size.max(DIM_ALIGN);
    let mut sw = clamp(u64::from(w.min(max_size)) / step * step);
    while sw >= floor {
        // The exact aspect target `sw·h/w`, taken to the lattice both ways. Integer throughout: a
        // float round-trip through a truncating `as u32` can drop a whole step on an `x.999…`.
        let (num, den) = (u64::from(sw) * u64::from(h), u64::from(w));
        let up = clamp(num.div_ceil(den * step) * step);
        let down = clamp(num / den / step * step);
        for sh in [up, down] {
            if sh >= min_size && sh <= max_size && sw as usize * sh as usize <= MAX_AREA_14B {
                return Some((sw, sh));
            }
        }
        sw -= DIM_ALIGN;
    }
    None
}

/// The first conditioning input matching `f`.
fn find_conditioning<'a, T>(
    req: &'a GenerationRequest,
    f: impl Fn(&'a Conditioning) -> Option<T>,
) -> Option<T> {
    req.conditioning.iter().find_map(f)
}

impl Scail2 {
    /// Map the request conditioning onto a [`Scail2Job`] and run the denoise pipeline.
    fn run(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Self-validate the shared floor first (F-158). `impl_generator!`'s `generate` does NOT call
        // `validate`, so a direct `Generator::generate` on scail2 (the only provider that skipped
        // this) otherwise bypassed count, sampler membership, conditioning allowlist, and the
        // F-053 finiteness guard — `guidance: Some(NAN)` would NaN-poison a multi-minute render into
        // garbage-as-success. Every other provider re-validates at the top of its generate impl.
        //
        // scail2 supports a "match the driving-video size" convention (`width`/`height == 0` → resolved
        // from the driving frames below), which the floor's size-range check would wrongly reject. That
        // exemption used to be hand-rolled here as `if req.width > 0 && req.height > 0 { … } else {
        // validate_request_skip_size }`; it is now ADVERTISED as
        // `SizeFloor::ResolvedDownstreamExplicitGrid` on the descriptor, so this single call applies
        // exactly that policy — and a weights-free consumer holding these `Capabilities` computes
        // the same verdict this line does, instead of having to guess which entry point the provider
        // happens to call. Every other floor check is unchanged and still fires on the auto-size
        // path: count/frame caps, sampler membership, the conditioning allowlist, support gating,
        // and the F-053 finiteness guard.
        //
        // The advertised floor is also **stricter** than the branch it replaces, deliberately. `0 > 0`
        // is false, so the old condition sent a HALF-sentinel (`0x512`, `4096x0`) down the
        // size-skipping path: the non-zero axis carried a real, checkable size and was never
        // range-checked, and `run` then took it verbatim into `Scail2Job` (only the zero axis is filled
        // from the driving frame below). An explicit `0x4096` against declared bounds of 32..=1280 was
        // accepted and rendered, with nothing downstream to catch it — `min_size`/`max_size` appear
        // nowhere in this crate outside `descriptor()`. `ResolvedDownstream` treats only
        // `width == 0 && height == 0` as the sentinel, which is what the convention actually means:
        // a half-sentinel is a malformed request, not "resolve from the driving media".
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        let reference = find_conditioning(req, |c| match c {
            Conditioning::Reference { image, .. } => Some(image),
            _ => None,
        })
        .ok_or_else(|| Error::Msg("scail2: a Reference character image is required".into()))?;
        let ref_mask = find_conditioning(req, |c| match c {
            Conditioning::Mask { image } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            Error::Msg(
                "scail2: a Mask (the reference character's color-coded segmentation mask) is required"
                    .into(),
            )
        })?;
        let driving = req.control_clip().ok_or_else(|| {
            Error::Msg(
                "scail2: a ControlClip (driving video frames + per-frame color masks) is required"
                    .into(),
            )
        })?;

        // Explicit geometry is already on the advertised 32-pixel grid. A `0x0` sentinel first
        // resolves to the driving frame's native size, then is aligned below while that source-media
        // origin is still known.
        let first: &Image = driving
            .frames
            .first()
            .ok_or_else(|| Error::Msg("scail2: the ControlClip has no driving frames".into()))?;
        let (resolved_width, resolved_height) = resolve_target_size(req, first);
        // The second half of `SizeFloor::ResolvedDownstream` (sc-16167). `impl_generator!`'s
        // `generate` does not call `validate`, so this must run here and not only there — the same
        // reason the shared floor is re-checked at the top of this function.
        reject_unrenderable_geometry(
            &self.descriptor.capabilities,
            req,
            resolved_width,
            resolved_height,
        )?;
        // Preserve the source-media origin distinction exactly once: a sentinel-resolved geometry
        // may be floored to the render lattice, while an explicit off-grid request has already been
        // rejected by the advertised floor above. `Scail2Job` itself accepts exact geometry only.
        let (width, height) = if req.width == 0 && req.height == 0 {
            (align(resolved_width) as u32, align(resolved_height) as u32)
        } else {
            (resolved_width, resolved_height)
        };

        let neg = req.negative_prompt.clone().unwrap_or_default();
        let job = Scail2Job {
            prompt: &req.prompt,
            negative_prompt: &neg,
            width,
            height,
            reference: CharacterRef {
                image: reference,
                mask: ref_mask,
            },
            additional: Vec::new(),
            driving_frames: driving.frames,
            driving_masks: driving.mask,
            replace_flag: req.video_mode.as_deref() == Some("replacement"),
            seed: req.seed.unwrap_or_else(default_seed),
            steps: req.steps.unwrap_or(DEFAULT_STEPS) as usize,
            shift: req.scheduler_shift.unwrap_or(DEFAULT_SHIFT),
            guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE),
            sampler: SolverKind::from_name(req.sampler.as_deref().unwrap_or("unipc")),
            fps: req.fps.unwrap_or(DEFAULT_FPS),
            segment_len: SEGMENT_LEN,
            segment_overlap: SEGMENT_OVERLAP,
        };
        crate::generate::generate(
            &self.root,
            &self.config,
            &job,
            self.quant,
            &self.adapters,
            &req.cancel,
            on_progress,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::ReplacementMode;

    /// A weights-free [`Scail2`] whose [`Scail2::run`] hits the shared floor before any load. Sound
    /// because every check under test runs before `root` is touched, and a *passing* request errors
    /// further in regardless — on the missing conditioning, the undecodable stub pixels, or finally
    /// `root` itself. Used to prove the floor rejects an out-of-surface request on the auto-size path
    /// without weights. Positive cases therefore assert the *specific* later error they should reach
    /// ([`CLEARED_THE_FLOOR`] / [`CLEARED_THE_GEOMETRY_GATE`]), never a bare `is_err`.
    fn unloaded() -> Scail2 {
        Scail2 {
            descriptor: descriptor(),
            config: Scail2Config::default(),
            root: PathBuf::from("/nonexistent-scail2-snapshot"),
            quant: None,
            adapters: Vec::new(),
        }
    }

    /// F-158: with `width == height == 0` (scail2's "match the driving-video size" sentinel), `run`
    /// routes through `validate_request_skip_size` — so the whole shared floor except the size-range
    /// check still fires. An oversized count and an unadvertised sampler are both rejected on the
    /// auto-size path, before any weight load.
    #[test]
    fn floor_fires_on_auto_size_path() {
        let m = unloaded();
        let mut noop = |_: Progress| {};

        // Oversized count (max_count == 1) — rejected even though dims are the 0x0 auto sentinel.
        let bad_count = GenerationRequest {
            width: 0,
            height: 0,
            count: 4,
            ..Default::default()
        };
        let err = m
            .run(&bad_count, &mut noop)
            .expect_err("oversized count must be rejected on the auto-size path");
        assert!(
            err.to_string().contains("count"),
            "expected a count-range rejection, got: {err}"
        );

        // Unadvertised sampler (scail2 advertises only `unipc` / `dpm++`); count == 1 so the sampler is
        // the failing check.
        let bad_sampler = GenerationRequest {
            width: 0,
            height: 0,
            count: 1,
            sampler: Some("euler".into()),
            ..Default::default()
        };
        let err = m
            .run(&bad_sampler, &mut noop)
            .expect_err("unadvertised sampler must be rejected on the auto-size path");
        assert!(
            err.to_string().contains("sampler"),
            "expected an unsupported-sampler rejection, got: {err}"
        );
    }

    /// The first conditioning lookup `run` performs after the shared floor. Reaching it is proof the
    /// floor **passed** — which is what makes the positive cases below non-vacuous: every weights-free
    /// `run` ends in an error, so `is_err()` alone would assert nothing at all.
    const CLEARED_THE_FLOOR: &str = "a Reference character image is required";

    /// F-158 read the other way round: an **explicit** size is still range-checked, and only the full
    /// `0x0` sentinel is exempt.
    ///
    /// This is the integration-level gate for the regression that motivated advertising
    /// [`SizeFloor::ResolvedDownstream`](mlx_gen::gen_core::SizeFloor::ResolvedDownstream). Reading
    /// that variant as a blanket opt-out deletes SCAIL-2's explicit-size rejection outright, and
    /// nothing downstream catches it: `min_size`/`max_size` appear nowhere in this crate outside
    /// [`descriptor`], so an out-of-range explicit size flows straight into [`Scail2Job`] and is
    /// rendered. gen-core's unit tests pin [`Capabilities`] in isolation; this pins that SCAIL-2's own
    /// `generate` — the thing a caller actually invokes — still gets the rejection.
    ///
    /// The half-sentinel rows are a behaviour **change**, not a re-assertion. The hand-rolled branch
    /// this replaced tested `req.width > 0 && req.height > 0`, so `0x4096` took the size-skipping path
    /// and its explicit 4096 height was accepted against declared bounds of 32..=1280.
    ///
    /// Weights-free: `run` validates before it touches `self.root`, so a rejected request never loads.
    #[test]
    fn run_range_checks_explicit_sizes_and_exempts_only_the_full_sentinel() {
        let m = unloaded();
        let mut noop = |_: Progress| {};

        for (w, h, why) in [
            (16, 16, "below min_size"),
            (4096, 4096, "above max_size"),
            (
                0,
                4096,
                "half-sentinel: the height is explicit and above max_size",
            ),
            (
                4096,
                0,
                "half-sentinel: the width is explicit and above max_size",
            ),
            (
                0,
                16,
                "half-sentinel: the height is explicit and below min_size",
            ),
        ] {
            let req = GenerationRequest {
                width: w,
                height: h,
                ..Default::default()
            };
            let err = m
                .run(&req, &mut noop)
                .expect_err("an out-of-range explicit size must never reach the pipeline")
                .to_string();
            assert!(
                err.contains("outside supported range"),
                "{w}x{h} ({why}) must be rejected by the advertised size range, got: {err}"
            );
        }

        for (w, h, why) in [
            (0, 0, "the resolve-from-the-driving-video sentinel"),
            (512, 512, "an in-range explicit size"),
            (32, 1280, "the inclusive bounds themselves"),
        ] {
            let req = GenerationRequest {
                width: w,
                height: h,
                ..Default::default()
            };
            let err = m
                .run(&req, &mut noop)
                .expect_err("no request renders without weights")
                .to_string();
            assert!(
                err.contains(CLEARED_THE_FLOOR),
                "{w}x{h} ({why}) must clear the size floor and fail later on missing conditioning, \
                 got: {err}"
            );
        }
    }

    #[test]
    fn explicit_off_grid_size_is_refused() {
        let m = unloaded();
        for (width, height) in [(1280, 730), (730, 1280)] {
            let req = GenerationRequest {
                width,
                height,
                count: 1,
                ..Default::default()
            };

            let err = Generator::validate(&m, &req)
                .expect_err("an explicit off-grid size must be refused before generation")
                .to_string();
            assert!(
                err.contains("multiples of 32") && err.contains(&format!("{width}×{height}")),
                "the refusal must name the required grid and requested size, got: {err}"
            );
        }
        assert_eq!(
            descriptor()
                .capabilities
                .size_floor
                .explicit_size_multiple(),
            Some(DIM_ALIGN)
        );
    }

    /// All three size verdicts a caller can obtain must be the **same** verdict.
    ///
    /// The point of putting `size_floor` on [`Capabilities`] is that a consumer can type-check a
    /// request before paying for a multi-gigabyte load and get the provider's *real* answer. That
    /// holds only while three separately-reachable paths agree, and until this change two of them
    /// did not:
    ///
    /// 1. the published [`descriptor`]'s capabilities — what a consumer has **before** any load;
    /// 2. [`Generator::validate`] — the pre-flight on a loaded provider (`impl_generator!` routes it
    ///    to the plain [`Capabilities::validate_request`], and always has);
    /// 3. the floor `generate` runs on itself — the only one that can actually stop a render.
    ///
    /// (2) and (3) disagreed about the sentinel: `validate` range-checked `0x0` and said **no**,
    /// while `generate` hand-rolled the exemption and rendered it. And (1) and (3) disagreed about
    /// half-sentinels: the hand-rolled predicate `req.width > 0 && req.height > 0` is *looser* than
    /// the advertisement, so `0x512` and `4096x0` skipped the size check entirely while a consumer
    /// reading the descriptor was told they would be rejected. Those rows go red if the branch comes
    /// back — which is the regression class this test exists for.
    ///
    /// Deliberately NOT covered here: a wrong `size_floor` **value** on the descriptor. Flip it to
    /// `RangeChecked` and all three move together — they read one field — so this stays green and
    /// [`run_range_checks_explicit_sizes_and_exempts_only_the_full_sentinel`] is what fails. The two
    /// are complementary: that one pins the absolute behaviour, this one pins the coupling.
    #[test]
    fn every_reachable_size_verdict_is_the_same_verdict() {
        let m = unloaded();
        let mut noop = |_: Progress| {};
        let published = descriptor();

        for (w, h) in [
            (0, 0),
            (512, 512),
            (16, 16),
            (4096, 4096),
            (0, 512),
            (512, 0),
            (32, 1280),
            (1280, 730),
            (730, 1280),
        ] {
            let req = GenerationRequest {
                width: w,
                height: h,
                ..Default::default()
            };
            // (1) what a weights-free consumer computes from the published descriptor,
            let advertised_ok = published
                .capabilities
                .validate_request(MODEL_ID, &req)
                .is_ok();
            // (2) what the loaded provider's own pre-flight says,
            let validate_ok = Generator::validate(&m, &req).is_ok();
            // (3) and what the floor inside `generate` actually enforces. `run` always errors without
            // weights, so "the floor accepted it" is read off the message, not off `is_ok`.
            let enforced_ok = m
                .run(&req, &mut noop)
                .expect_err("no request renders without weights")
                .to_string()
                .contains(CLEARED_THE_FLOOR);
            assert_eq!(
                (advertised_ok, validate_ok),
                (enforced_ok, enforced_ok),
                "{w}x{h}: descriptor accepted={advertised_ok}, Generator::validate \
                 accepted={validate_ok}, the floor inside generate accepted={enforced_ok} — a \
                 caller that trusts either pre-flight would be wrong about this request",
            );
        }
    }

    // ---------------------------------------------------------------------------------------
    // sc-16167: the RESOLVED geometry is bounded.
    // ---------------------------------------------------------------------------------------

    /// An `w`x`h` image with an empty pixel buffer. Sound for these tests because the geometry gate
    /// reads `width`/`height` only and runs before any pixel decode — materializing 4K RGB buffers
    /// (~25 MB each) would test nothing. A request that *clears* the gate then dies on these stub
    /// pixels inside `crate::generate::generate` (`image_to_chw` decodes all four inputs before the
    /// first file read), which is what [`CLEARED_THE_GEOMETRY_GATE`] matches — the snapshot dir is
    /// never reached.
    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: Vec::new(),
        }
    }

    /// A single-frame driving clip at `w`x`h`.
    fn clip(w: u32, h: u32) -> Conditioning {
        Conditioning::ControlClip {
            frames: vec![img(w, h)],
            mask: vec![img(w, h)],
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        }
    }

    /// The full SCAIL-2 conditioning set: reference character + its color mask + a `w`x`h` driving
    /// clip. `run` looks all three up **before** it resolves the geometry, so a request missing any
    /// of them dies on the lookup and never reaches the gate — which would make every assertion
    /// about the gate inside `generate` vacuous.
    fn conditioning(w: u32, h: u32) -> Vec<Conditioning> {
        vec![
            Conditioning::Reference {
                image: img(64, 64),
                strength: None,
            },
            Conditioning::Mask { image: img(64, 64) },
            clip(w, h),
        ]
    }

    /// An auto-sized request (`0x0`) whose driving clip is `w`x`h`.
    fn auto_sized(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            width: 0,
            height: 0,
            count: 1,
            conditioning: conditioning(w, h),
            ..Default::default()
        }
    }

    /// The substring that identifies this gate's refusal, distinct from the shared floor's
    /// "outside supported range".
    const REFUSED: &str = "advertised size envelope";

    /// The error a request reaches once it has cleared the geometry gate — the stub pixel buffers
    /// failing to decode inside `crate::generate::generate`. The positive counterpart of
    /// [`REFUSED`], and the reason the accepting cases below are not vacuous: every weights-free
    /// `run` errors, so "did not produce the refusal" alone would also pass if the request had died
    /// *earlier* (a missing Reference, a floor rejection) and never reached the gate at all.
    const CLEARED_THE_GEOMETRY_GATE: &str = "image pixel buffer";

    /// The story's headline case: a 4K driving clip on the `0x0` sentinel resolves to `3840x2160`
    /// against declared bounds of `32..=1280` and used to render. The shared floor cannot catch it —
    /// [`SizeFloor::ResolvedDownstream`] exempts the sentinel by design, and `align()` only snaps 4K
    /// to `3840x2144`, so nothing downstream bounded it either.
    ///
    /// Both the `Generator::validate` pre-flight and the floor inside `generate` must refuse, because
    /// `impl_generator!`'s `generate` does not call `validate` — a gate on only one of them leaves
    /// the other path unbounded.
    #[test]
    fn auto_size_refuses_an_over_envelope_driving_clip() {
        let m = unloaded();
        let mut noop = |_: Progress| {};

        for (w, h, why) in [
            (3840, 2160, "4K — 8.29 Mpx, 9x the area cap"),
            (
                1920,
                1080,
                "1080p — 2.07 Mpx, both over the cap and over max_size",
            ),
            (
                2560,
                720,
                "at the area cap but the long edge is over max_size",
            ),
        ] {
            let req = auto_sized(w, h);
            let pre = Generator::validate(&m, &req)
                .expect_err("the pre-flight must refuse an over-envelope resolved geometry")
                .to_string();
            assert!(
                pre.contains(REFUSED) && pre.contains(&format!("{w}×{h}")),
                "{w}x{h} ({why}) must be refused by validate naming the resolved size, got: {pre}"
            );
            assert!(
                pre.contains("resolved from the driving video"),
                "{w}x{h} ({why}): the refusal must say the size came from the clip, got: {pre}"
            );

            let ran = m
                .run(&req, &mut noop)
                .expect_err("the render path must refuse it too")
                .to_string();
            assert!(
                ran.contains(REFUSED),
                "{w}x{h} ({why}) must be refused inside generate, not only by validate, got: {ran}"
            );
        }
    }

    /// The complement, and what stops the gate from being "refuse everything". A clip inside the
    /// envelope resolves and proceeds.
    ///
    /// Two rows carry weight beyond "an ordinary size passes":
    ///
    /// * `1280x720` — the family's canonical 720p, `921_600` px **as typed**, which is
    ///   `MAX_AREA_14B` exactly. It passes for a reason worth pinning: the cap is measured on the
    ///   geometry that actually renders, and `720` floors onto the `DIM_ALIGN` lattice to `704`, so
    ///   the measured area is `901_120`. A gate that judged the raw request instead would sit one
    ///   pixel of slack away from refusing the model's own headline resolution.
    /// * `960x960` — both edges already on the lattice, so the measured area *is* `921_600`. That is
    ///   the row that pins the cap as a strict `>`; relaxing it to `>=` refuses this and only this.
    #[test]
    fn auto_size_accepts_an_in_envelope_driving_clip() {
        let m = unloaded();
        let mut noop = |_: Progress| {};

        for (w, h, why) in [
            (832, 480, "the model's default bucket"),
            (
                640,
                360,
                "an ordinary off-grid source clip whose resolved geometry is aligned downstream",
            ),
            (1280, 704, "the widest advertised bucket"),
            (
                1280,
                720,
                "canonical 720p — at the cap as typed, under it once lattice-aligned",
            ),
            (
                960,
                960,
                "EXACTLY at MAX_AREA_14B once aligned — the cap is a strict >",
            ),
            (480, 832, "portrait"),
        ] {
            let req = auto_sized(w, h);
            Generator::validate(&m, &req)
                .unwrap_or_else(|e| panic!("{w}x{h} ({why}) must clear the pre-flight, got: {e}"));
            // `run` errors regardless (no weights), so the assertion is the POSITIVE marker of
            // having got past the gate — not merely the absence of its message, which a request
            // that died earlier would also satisfy.
            let ran = m
                .run(&req, &mut noop)
                .expect_err("no request renders without weights")
                .to_string();
            assert!(
                ran.contains(CLEARED_THE_GEOMETRY_GATE),
                "{w}x{h} ({why}) must clear the geometry gate inside generate, got: {ran}"
            );
        }
    }

    /// The gate judges the **rendered** geometry, so it covers an explicit size too — closing a
    /// candle/MLX divergence that has nothing to do with the sentinel.
    ///
    /// `1280x1280` has both edges inside `max_size` and so clears the shared range check, but it is
    /// 1.64 Mpx — 1.8x the area cap. The candle twin has rejected it since sc-11215
    /// (`candle-gen-scail2/src/pipeline.rs`'s `reject_over_area`); MLX had no area check at all, so
    /// one manifest entry meant two different things per backend.
    #[test]
    fn explicit_over_area_is_refused_even_though_both_edges_are_in_range() {
        let m = unloaded();
        let over = GenerationRequest {
            width: 1280,
            height: 1280,
            count: 1,
            conditioning: conditioning(832, 480),
            ..Default::default()
        };
        // The shared range check passes it — proof this test is not just re-asserting the floor.
        descriptor()
            .capabilities
            .validate_request(MODEL_ID, &over)
            .expect("1280x1280 is within min_size..=max_size on both edges");

        let err = Generator::validate(&m, &over)
            .expect_err("1280x1280 is 1.8x the area cap and must be refused")
            .to_string();
        assert!(
            err.contains(REFUSED) && err.contains(&MAX_AREA_14B.to_string()),
            "expected an area refusal naming the cap, got: {err}"
        );
        assert!(
            err.contains("requested"),
            "an explicit size must not be reported as resolved from the clip, got: {err}"
        );

        // And with NO conditioning at all. Only the `0x0` sentinel needs the clip to resolve, so
        // gating the whole check behind one would leave the pre-flight blind exactly when a consumer
        // uses it as intended — type-checking a geometry before assembling multi-megabyte
        // conditioning. candle's `validate` area-checks unconditionally; this is that parity.
        let bare = GenerationRequest {
            width: 1280,
            height: 1280,
            count: 1,
            ..Default::default()
        };
        let err = Generator::validate(&m, &bare)
            .expect_err("an explicit over-area size must be refused with no clip attached")
            .to_string();
        assert!(
            err.contains(REFUSED),
            "expected the area refusal without conditioning, got: {err}"
        );
    }

    /// The refusal has to be actionable, so the suggested geometry must itself be one the gate would
    /// accept — a suggestion that would be refused in turn is worse than none.
    ///
    /// The expected pairs are pinned exactly, not merely bounds-checked. Three rows carry an argument
    /// beyond "it fits":
    ///
    /// * `1920x1080` and `3840x2160` both land on `1280x704`, and `1080x1920` on `704x1280` — all
    ///   resolutions the model manifest actually advertises. The round-up-before-round-down tie-break
    ///   is what buys the portrait row: rounding the aspect target down instead gives `704x1248`,
    ///   in-envelope but on no menu entry in the product.
    /// * `1280x1280` lands on `960x960`, `921_600` px — the cap exactly, so this row also pins the
    ///   cap as a strict `>`.
    #[test]
    fn the_suggested_geometry_is_itself_in_envelope() {
        let caps = descriptor().capabilities;
        for (w, h, want) in [
            (1920, 1080, Some((1280, 704))),
            (3840, 2160, Some((1280, 704))),
            (1280, 1280, Some((960, 960))),
            (2560, 720, Some((1280, 384))),
            (1080, 1920, Some((704, 1280))),
            // Below `min_size`: no in-envelope geometry at or under the source, so no suggestion
            // rather than an invented upscale.
            (16, 16, None),
            (0, 0, None),
        ] {
            let got = suggest_in_envelope(w, h, caps.min_size, caps.max_size);
            assert_eq!(got, want, "suggestion for {w}x{h}");

            if let Some((sw, sh)) = got {
                let req = GenerationRequest {
                    width: sw,
                    height: sh,
                    count: 1,
                    ..Default::default()
                };
                // It must clear BOTH the shared floor and this gate.
                caps.validate_request(MODEL_ID, &req).unwrap_or_else(|e| {
                    panic!("suggestion {sw}x{sh} for {w}x{h} fails the shared floor: {e}")
                });
                reject_unrenderable_geometry(&caps, &req, sw, sh).unwrap_or_else(|e| {
                    panic!("suggestion {sw}x{sh} for {w}x{h} would itself be refused: {e}")
                });
                assert_eq!(
                    (sw % DIM_ALIGN, sh % DIM_ALIGN),
                    (0, 0),
                    "suggestion {sw}x{sh} for {w}x{h} is off the {DIM_ALIGN} lattice"
                );
                assert!(
                    sw <= w && sh <= h,
                    "suggestion {sw}x{sh} upscales the {w}x{h} source"
                );
            }
        }
    }

    /// The refusal must name a way forward. An over-envelope clip whose aspect has an in-envelope
    /// geometry gets it named; one that does not falls back to the generic advice instead of
    /// suggesting nothing at all.
    #[test]
    fn the_refusal_names_a_usable_alternative() {
        let m = unloaded();
        let err = Generator::validate(&m, &auto_sized(3840, 2160))
            .expect_err("4K must be refused")
            .to_string();
        assert!(
            err.contains("1280×704"),
            "the refusal must name the largest in-envelope geometry, got: {err}"
        );

        // A clip below `min_size` on both edges has no in-envelope geometry at or under its own
        // size, so the message carries the generic advice rather than an invented upscale.
        let tiny = Generator::validate(&m, &auto_sized(16, 16))
            .expect_err("a 16x16 clip is below min_size and must be refused")
            .to_string();
        assert!(
            tiny.contains("inside the advertised range") && !tiny.contains("largest advertised"),
            "a sub-min clip must fall back to the generic advice, got: {tiny}"
        );
    }

    /// `Generator::validate` and the floor inside `generate` must agree about the **resolved**
    /// geometry, exactly as [`every_reachable_size_verdict_is_the_same_verdict`] pins them for the
    /// requested one. A consumer that pre-flights with `validate` and then calls `generate` must not
    /// get two different answers — and `impl_generator!` routes them through different code (its
    /// `generate` does not call `validate`), so this coupling is not free: dropping the gate from
    /// either call site alone makes this red.
    ///
    /// Deliberately NOT covered here, same blind spot its sibling documents: deleting the gate from
    /// **both** sites leaves them agreeing on "accept everything" and this stays green.
    /// [`auto_size_refuses_an_over_envelope_driving_clip`] is what fails then. The two are
    /// complementary — that one pins the absolute behaviour, this one pins the coupling.
    #[test]
    fn validate_and_generate_agree_on_the_resolved_geometry() {
        let m = unloaded();
        let mut noop = |_: Progress| {};

        for (w, h) in [
            (832, 480),
            (1280, 720),
            (1280, 736),
            (1920, 1080),
            (3840, 2160),
            (1312, 480),
        ] {
            let req = auto_sized(w, h);
            let validate_ok = Generator::validate(&m, &req).is_ok();
            let generate_ok = !m
                .run(&req, &mut noop)
                .expect_err("no request renders without weights")
                .to_string()
                .contains(REFUSED);
            assert_eq!(
                validate_ok, generate_ok,
                "{w}x{h}: validate accepted={validate_ok} but the gate inside generate \
                 accepted={generate_ok}",
            );
        }
    }
}
