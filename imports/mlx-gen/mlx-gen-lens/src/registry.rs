//! `LensGenerator` — the [`mlx_gen::Generator`] impl wiring the Lens pipeline ([`crate::pipeline`])
//! into `mlx_gen`'s registry under **two** ids (sc-3173):
//!
//! - **`lens_turbo`** — the distilled turbo variant: **4 steps, guidance 1.0** (≈ no CFG).
//! - **`lens`** — the base variant: **20 steps, CFG 5.0**.
//!
//! Both ids share the identical crate/architecture/weights tree and differ **only** in their default
//! `num_steps` / `guidance_scale` (the reference ships them as separate model cards with the same
//! arch). A request's explicit `steps` / `guidance` still override the per-id default.
//!
//! **Surface.** This is a pure **T2I** generator: no img2img / ControlNet / IP conditioning (none
//! exists in the Lens port). **LoRA + LoKr** merge into the DiT's joint-attention projections at load
//! (sc-3174 — inference consumption; native-MLX *training* is [`crate::training`], sc-5148). The dense path is bf16; the `Fp32`
//! precision override is honored. **Q4/Q8** quantize the gpt-oss encoder's MoE experts (sc-3172 —
//! the ~38 GB / 20 B-param bulk → ~12 GB) **and** the DiT's linears (sc-3175) at load.
//!
//! **Registration mechanism:** the two `inventory::submit!`s below are collected by `mlx_gen`'s
//! `inventory::collect!` at *link* time, so they activate whenever a consumer (the worker, or this
//! crate's own test binary) links `mlx-gen-lens`. The core `mlx-gen` crate does **not** depend on the
//! model crates (by design); there is no root-crate dependency to add.

use std::path::Path;

use mlx_rs::{Array, Dtype};

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, Capabilities, Error,
    GenerationOutput, GenerationRequest, Generator, LatentDecoder, LoadSpec, Modality,
    ModelDescriptor, OffloadPolicy, Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_gen_flux2::model::PID_BACKBONE;
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};

use crate::pipeline::{LensHeavy, LensText, DEFAULT_DATE, VAE_SCALE_FACTOR};

/// Registry id — the distilled turbo variant.
pub const MODEL_ID_TURBO: &str = "lens_turbo";
/// Registry id — the base variant.
pub const MODEL_ID_BASE: &str = "lens";

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`) baked into the loaded generator.
#[derive(Clone, Copy)]
struct Defaults {
    id: &'static str,
    steps: u32,
    guidance: f32,
}

// The step/guidance numbers are the single source of truth in [`crate::schedule`] (`TURBO`/`BASE`);
// the registry just re-tags them with the model id.
const TURBO_DEFAULTS: Defaults = Defaults {
    id: MODEL_ID_TURBO,
    steps: crate::schedule::TURBO.num_steps as u32,
    guidance: crate::schedule::TURBO.guidance_scale,
};
const BASE_DEFAULTS: Defaults = Defaults {
    id: MODEL_ID_BASE,
    steps: crate::schedule::BASE.num_steps as u32,
    guidance: crate::schedule::BASE.guidance_scale,
};

/// Lens' identity + capabilities for `id` — constructible without loading weights (registry
/// introspection). Advertises the wired + parity-proven surface: T2I with negative-prompt /
/// guidance CFG, no conditioning, LoRA + LoKr (DiT joint-attention, sc-3174), and Q4/Q8 load-time
/// quant (gpt-oss MoE experts sc-3172 + DiT linears sc-3175).
fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "lens",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // The norm-rescaled CFG path is always present; turbo simply defaults guidance to 1.0.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![], // pure T2I — no img2img / control / IP in the Lens port
            // sc-3174: LoRA + LoKr merge into the DiT's joint-attention projections at load.
            supports_lora: true,
            supports_lokr: true,
            // epic 7114 sc-7305: advertise the curated sampler/scheduler menu (mirrors the candle Lens
            // adoption) so the per-generation knobs route through the unified `Sampler<MlxLatentOps>` +
            // `FlowModelSampling`. The legacy native aliases stay valid for old recipes; both N3-fall
            // back to the default (`flow_match_euler` → euler, `flow_match` → the native empirical-μ
            // schedule), so they never hard-fail a generation.
            samplers: {
                let mut s = curated_sampler_names();
                s.push("flow_match_euler");
                s
            },
            schedulers: {
                let mut s = curated_scheduler_names();
                s.push("flow_match");
                s
            },
            // Buckets span 736..2080 (all ÷16); allow any ÷16 size in a sane range.
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2080,
            max_count: 8,
            mac_only: true,
            // Q4/Q8 quantize the gpt-oss encoder's MoE experts (sc-3172 — the ~38 GB / 20 B-param
            // bulk → ~12 GB) and the DiT's linears (sc-3175) at load.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            // The Lens schedule computes its own empirical-μ shift internally (not a loader hint).
            requires_sigma_shift: false,
        },
    }
}

/// Public descriptor accessors (used by the registry submits + tests).
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(MODEL_ID_TURBO)
}
pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(MODEL_ID_BASE)
}

/// A loaded, dispatchable Lens generator: the variant's descriptor & sampling defaults + the
/// component-residency strategy (epic 10834 Phase 1, sc-11030). Both `lens` and `lens_turbo` share
/// this and differ only in the baked sampling defaults.
pub struct LensGenerator {
    descriptor: ModelDescriptor,
    defaults: Defaults,
    /// Component-residency strategy, selected from [`LoadSpec::offload_policy`]. `Resident` (default)
    /// holds the gpt-oss text encoder + DiT + VAE warm for the whole job and across jobs; `Sequential`
    /// holds only the [`LoadSpec`] and re-loads per generation in phase order (encode → **drop the text
    /// encoder** → denoise/decode), bounding peak unified memory to `max(text-encoder, DiT+VAE)` — the
    /// gpt-oss encoder is the dominant footprint, so lens-turbo drops ~13.1 GB (46%).
    residency: Residency,
}

/// The heavy-component residency for a [`LensGenerator`] (sc-11030). See [`LensGenerator::residency`].
enum Residency {
    /// Every component loaded once at [`load_with`] and held (today's warm-cache path). `generate`
    /// borrows these. Boxed so this heavy variant doesn't bloat every `Sequential` handle
    /// (`clippy::large_enum_variant`).
    Resident(Box<LensResident>),
    /// Only the [`LoadSpec`] is held; each `generate` re-loads the phases in order and frees them after,
    /// so peak memory is `max(text-encoder, DiT+VAE)` and nothing stays resident across jobs. The
    /// per-phase loaders rebuild byte-identical components to the `Resident` path.
    Sequential(Box<LoadSpec>),
}

/// The warm-resident Lens components (the phase-A [`LensText`] dropped first under `Sequential`, plus
/// the heavy render bundle). Split so the `Resident` and `Sequential` paths hand the render body the
/// exact same [`LensHeavyRef`] borrow.
struct LensResident {
    text: LensText,
    heavy: LensHeavyOwned,
}

/// The heavy render-phase components (the DiT + VAE via [`LensHeavy`], plus the optional PiD decoder) —
/// everything but the text encoder. Owned by the `Resident` components or by a `Sequential` generate.
struct LensHeavyOwned {
    heavy: LensHeavy,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7847): loaded when the spec carries
    /// `LoadSpec::pid`. `Some` → a `req.use_pid` generation decodes through the `flux2` student (4× SR).
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body runs identically whether
/// they are held resident or were just loaded by the `Sequential` path.
struct LensHeavyRef<'a> {
    heavy: &'a LensHeavy,
    pid: Option<&'a PidEngine>,
}

impl LensHeavyOwned {
    fn as_ref(&self) -> LensHeavyRef<'_> {
        LensHeavyRef {
            heavy: &self.heavy,
            pid: self.pid.as_ref(),
        }
    }
}

/// Build a [`LensGenerator`] from a [`LoadSpec`] with the given per-variant defaults.
///
/// `spec.weights` is a `microsoft/Lens-Turbo` (or `microsoft/Lens`) snapshot dir (the diffusers
/// multi-component tree). Dense runs **bf16**; `Precision::Fp32` loads the tight-gate f32 path.
/// `spec.quantize` (Q4/Q8) quantizes the encoder's MoE experts at load (sc-3172); `spec.adapters`
/// (LoRA/LoKr) merge into the DiT (sc-3174). `control` / `ip_adapter` are not part of the Lens port.
///
/// Component residency (epic 10834 Phase 1, sc-11030): `Resident` (default) builds every phase now and
/// holds it warm; `Sequential` keeps only the spec and re-loads per generate in phase order (encode →
/// drop the text encoder → denoise/decode) to bound peak memory to `max(text-encoder, DiT+VAE)`. Both
/// use the same per-phase loaders, so the components are byte-identical.
fn load_with(spec: &LoadSpec, defaults: Defaults) -> Result<Box<dyn Generator>> {
    let residency = match spec.offload_policy {
        OffloadPolicy::Resident => {
            let (root, dtype) = resolve_root(spec)?;
            let text = load_text_phase(spec, &root, dtype)?;
            let heavy = load_heavy_phase(spec, &root, dtype)?;
            Residency::Resident(Box::new(LensResident { text, heavy }))
        }
        OffloadPolicy::Sequential => {
            // Validate the spec up front (fail fast, same as `Resident`); the heavy build is deferred
            // to each generate.
            resolve_root(spec)?;
            Residency::Sequential(Box::new(spec.clone()))
        }
    };
    Ok(Box::new(LensGenerator {
        descriptor: descriptor_for(defaults.id),
        defaults,
        residency,
    }))
}

/// Snapshot-dir + precision→dtype resolution (rejecting a single-file source / unsupported overlays),
/// shared by the `Resident` build and the `Sequential` per-phase loaders (sc-11030).
fn resolve_root(spec: &LoadSpec) -> Result<(std::path::PathBuf, Dtype)> {
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(Error::Msg(
            "lens: ControlNet / IP-Adapter conditioning is not part of the Lens port".into(),
        ));
    }
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "lens: expects a Lens snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok((root, dtype))
}

/// Load the text-encode phase — the gpt-oss encoder dropped first under `Sequential`. `spec.quantize`
/// quantizes the encoder's MoE experts at load (sc-3172).
fn load_text_phase(spec: &LoadSpec, root: &Path, dtype: Dtype) -> Result<LensText> {
    LensText::load(root, dtype, spec.quantize)
}

/// Load the heavy render phase — DiT (+ LoRA/LoKr merge, then Q4/Q8) + VAE + the optional PiD overlay —
/// everything but the text encoder. Factored so `Sequential` loads these AFTER the encoder is dropped.
/// The DiT quantizes **after** any adapter merge (sc-3175 — adapters are forward-time residuals over
/// the quantized base); the components are byte-identical to the `Resident` composition.
fn load_heavy_phase(spec: &LoadSpec, root: &Path, dtype: Dtype) -> Result<LensHeavyOwned> {
    let mut heavy = LensHeavy::load(root, dtype)?;
    if !spec.adapters.is_empty() {
        heavy.apply_adapters(&spec.adapters)?;
    }
    if let Some(q) = spec.quantize {
        heavy.quantize_dit(q)?;
    }
    // PiD decoder overlay (epic 7840, sc-7847): load the shared `flux2` student + Gemma once.
    let pid = spec
        .pid
        .as_ref()
        .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
        .transpose()?;
    Ok(LensHeavyOwned { heavy, pid })
}

mlx_gen::impl_generator!(LensGenerator {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

impl LensGenerator {
    /// The rich-`Result` body behind [`Generator::validate`].
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(self.defaults.id, &self.descriptor.capabilities, req)?;
        Ok(())
    }

    /// Text-encode the prompt + negative per the residency (sc-11030). `Resident` borrows the warm
    /// gpt-oss encoder (byte-identical to the pre-sc-11030 calls); `Sequential` loads it, encodes,
    /// forces materialization (`eval`), then DROPS it + `clear_cache()` so its ~13 GB frees before the
    /// DiT loads.
    fn encode(&self, req: &GenerationRequest, negative: &str) -> Result<(Vec<Array>, Array)> {
        let encode_with = |text: &LensText| {
            text.encode_prompt(&req.prompt, negative, DEFAULT_DATE, Some(&req.cancel))
        };
        match &self.residency {
            Residency::Resident(c) => encode_with(&c.text),
            Residency::Sequential(spec) => {
                let (root, dtype) = resolve_root(spec)?;
                let text = load_text_phase(spec, &root, dtype)?;
                let (features, mask) = encode_with(&text)?;
                // MLX is lazy — materialize NOW while `text` is alive, else `features`/`mask` keep the
                // encoder weights referenced through the graph and the drop frees nothing (cf. Wan's
                // `encode_text_staged` / Z-Image's `encode`).
                let mut to_eval: Vec<&Array> = features.iter().collect();
                to_eval.push(&mask);
                mlx_rs::transforms::eval(to_eval)?;
                drop(text);
                mlx_rs::memory::clear_cache();
                Ok((features, mask))
            }
        }
    }

    /// Load the heavy render phase (DiT + VAE + PiD) for a `Sequential` job — after [`Self::encode`]
    /// dropped the text encoder — or `None` under `Resident` (already held). Kept separate from
    /// [`Self::heavy`] so the owned bundle outlives the render-body borrow.
    fn load_seq_heavy(&self) -> Result<Option<LensHeavyOwned>> {
        match &self.residency {
            Residency::Resident(_) => Ok(None),
            Residency::Sequential(spec) => {
                let (root, dtype) = resolve_root(spec)?;
                Ok(Some(load_heavy_phase(spec, &root, dtype)?))
            }
        }
    }

    /// Borrow the heavy render components: the warm bundle under `Resident`, or the just-loaded
    /// `seq_heavy` under `Sequential`. The render body is written once against this borrow.
    fn heavy<'a>(&'a self, seq_heavy: &'a Option<LensHeavyOwned>) -> LensHeavyRef<'a> {
        match (&self.residency, seq_heavy) {
            (Residency::Resident(c), _) => c.heavy.as_ref(),
            (_, Some(owned)) => owned.as_ref(),
            (Residency::Sequential(_), None) => {
                unreachable!("Sequential residency always loads seq_heavy before rendering")
            }
        }
    }

    /// The rich-`Result` body behind [`Generator::generate`]: map the request onto the residency,
    /// looping `count` with per-image seeds and streaming step/decode progress.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate_impl(req)?;

        let steps = req.steps.unwrap_or(self.defaults.steps) as usize;
        let guidance = req.guidance.unwrap_or(self.defaults.guidance);
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let total = steps as u32;
        let latent_h = (req.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (req.width / VAE_SCALE_FACTOR) as usize;

        // Phase A: prompt → embeds (sc-11030). Under `Sequential` this loads the gpt-oss encoder,
        // encodes, forces materialization, then DROPS it + `clear_cache()` so its ~13 GB frees before
        // the DiT/VAE load below — the peak-bounding win. Encoding once (deterministic, no RNG draw) is
        // byte-identical to the pre-sc-11030 per-image re-encode (the init noise reseeds per image
        // inside `render`). Under `Resident` it borrows the warm encoder. See `Self::encode`.
        let (encoder_features, encoder_mask) = self.encode(req, negative)?;

        // Establish the heavy render components (DiT + VAE + PiD). `Resident` borrows the warm bundle;
        // `Sequential` loads it NOW — after the encoder was dropped — and frees it when the job ends.
        // The render body below runs identically for both residencies.
        let seq_heavy = self.load_seq_heavy()?;
        let heavy = self.heavy(&seq_heavy);

        // PiD decode overlay (epic 7840, sc-7847) + `from_ldm` early-stop (sc-8048): one decoder serves
        // the whole count loop (same prompt). Errors if `req.use_pid` but the model wasn't loaded with
        // `LoadSpec::pid`; `None` (the default) → the byte-exact native Flux.2 VAE path. Lens is
        // `vp_frame=false` (schedule σ *is* the degrade σ) and pure T2I (`start_step = 0`); resolve the
        // plan against the SAME descending schedule `render` runs. `None` capture → full schedule.
        let sigmas =
            heavy
                .heavy
                .resolve_sigmas(latent_h, latent_w, steps, req.scheduler.as_deref());
        let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, 0);
        let keep = (keep < sigmas.len()).then_some(keep);
        let pid_decoder = resolve_pid_decoder_at_sigma(
            heavy.pid,
            req,
            base_seed,
            self.defaults.id,
            capture_sigma,
        )?;
        let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            // The one render body (sc-11030): the same `LensHeavy::render` for both residencies, so a
            // Sequential job (encoder already dropped) is byte-identical to Resident. The reasoner
            // (sc-3176) is a standalone struct-API opt-in; the registry path leaves it off.
            let image = heavy.heavy.render(
                &encoder_features,
                &encoder_mask,
                latent_h,
                latent_w,
                steps,
                guidance,
                // epic 7114 sc-7305: per-generation curated sampler/scheduler (N3 fallback inside the
                // unified framework; the worker also pre-normalizes unadvertised names).
                req.sampler.as_deref(),
                req.scheduler.as_deref(),
                seed,
                keep,
                pid_ref,
                &req.cancel,
                &mut |cur| {
                    on_progress(Progress::Step {
                        current: cur as u32,
                        total,
                    });
                    // F-106: `render` decodes immediately after the final step (it exposes only a step
                    // callback, not a Progress sink), so emit `Decoding` when the last step lands —
                    // BEFORE the VAE/PiD decode.
                    if cur as u32 >= total {
                        on_progress(Progress::Decoding);
                    }
                },
            )?;
            images.push(image);
        }
        // Sequential (sc-11030): free the DiT/VAE/PiD working set now that every image is rendered, then
        // `clear_cache()` to return the pages to the OS. `heavy` (a struct of borrows) is unused past
        // the loop, so NLL has ended its borrow of `seq_heavy`; dropping the owned bundle frees the
        // components before `clear_cache()`. Resident is a no-op (`seq_heavy` None).
        let was_sequential = seq_heavy.is_some();
        drop(seq_heavy);
        if was_sequential {
            mlx_rs::memory::clear_cache();
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation (unit-testable without loaded weights).
pub(crate) fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    // Shared capability contract: count/size range, negative_prompt/guidance/true_cfg, sampler,
    // scheduler, conditioning kinds.
    caps.validate_request(id, req)?;

    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    // The Flux.2 VAE + DiT patchify downsample by 16; non-multiple-of-16 dims mismatch latent shapes.
    if !req.width.is_multiple_of(VAE_SCALE_FACTOR) || !req.height.is_multiple_of(VAE_SCALE_FACTOR) {
        return Err(Error::Msg(format!(
            "{id}: width/height must be multiples of {VAE_SCALE_FACTOR} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

// Thin id-binding loaders: each pins the variant defaults onto `load_with`, so they can't be a
// plain `load` path. They return the crate's rich `Result`; `register_generators!` adds the
// `gen_core::Result` bridge (epic 3720) and emits each `inventory::submit!`.
fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, TURBO_DEFAULTS)
}
fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, BASE_DEFAULTS)
}

mlx_gen::register_generators! {
    descriptor_turbo => load_turbo,
    descriptor_base => load_base,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_are_lens() {
        for (d, id, steps, g) in [
            (descriptor_turbo(), MODEL_ID_TURBO, 4u32, 1.0f32),
            (descriptor_base(), MODEL_ID_BASE, 20, 5.0),
        ] {
            assert_eq!(d.id, id);
            assert_eq!(d.family, "lens");
            assert_eq!(d.modality, Modality::Image);
            assert!(d.capabilities.supports_guidance);
            assert!(d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_true_cfg);
            assert!(d.capabilities.conditioning.is_empty());
            // sc-3174: LoRA + LoKr merge into the DiT joint-attention projections at load.
            assert!(d.capabilities.supports_lora);
            assert!(d.capabilities.supports_lokr);
            // sc-3172: encoder MoE experts quantize to Q4/Q8 at load.
            assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
            // sc-7305: the curated sampler/scheduler menu is advertised (the unified framework) with the
            // legacy native aliases retained — both backends (mlx + candle) now expose the same menu.
            assert!(d.capabilities.samplers.contains(&"euler"));
            assert!(d.capabilities.samplers.contains(&"dpmpp_2m"));
            assert!(d.capabilities.samplers.contains(&"uni_pc"));
            assert!(d.capabilities.samplers.contains(&"flow_match_euler"));
            assert!(d.capabilities.schedulers.contains(&"karras"));
            assert!(d.capabilities.schedulers.contains(&"exponential"));
            assert!(d.capabilities.schedulers.contains(&"flow_match"));
            // The defaults are exercised end-to-end in the e2e test; assert the constants here.
            let def = if id == MODEL_ID_TURBO {
                TURBO_DEFAULTS
            } else {
                BASE_DEFAULTS
            };
            assert_eq!((def.steps, def.guidance), (steps, g));
        }
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        // The `inventory::submit!`s are linked into this test binary, so `mlx_gen::load` resolves
        // both ids (and fails on the bogus weights dir) — proving registration without the snapshot.
        for id in [MODEL_ID_TURBO, MODEL_ID_BASE] {
            let spec = LoadSpec {
                weights: WeightsSource::Dir("/nonexistent/lens".into()),
                quantize: None,
                precision: Precision::Bf16,
                control: None,
                ip_adapter: None,
                adapters: Vec::new(),
                extra_controls: Vec::new(),
                pid: None,
                identity: None,
                text_encoder: None,
                offload_policy: Default::default(),
            };
            let err = match mlx_gen::load(id, &spec) {
                Ok(_) => panic!("bogus weights dir must fail to load"),
                Err(e) => e.to_string(),
            };
            assert!(
                !err.contains("no generator registered"),
                "{id} should resolve in the registry; got: {err}"
            );
        }
    }

    #[test]
    fn load_rejects_unsupported_overlays_not_quant() {
        let base = LoadSpec {
            weights: WeightsSource::Dir("/nonexistent/lens".into()),
            quantize: None,
            precision: Precision::Bf16,
            control: None,
            ip_adapter: None,
            adapters: Vec::new(),
            extra_controls: Vec::new(),
            pid: None,
            identity: None,
            text_encoder: None,
            offload_policy: Default::default(),
        };
        // A ControlNet overlay is rejected (not part of the Lens port) — the message names it, before
        // any weights load.
        let with_control = LoadSpec {
            control: Some(WeightsSource::Dir("/nonexistent/cn".into())),
            ..base.clone()
        };
        let err = match load_with(&with_control, TURBO_DEFAULTS) {
            Ok(_) => panic!("control must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("not part of the Lens port"), "got: {err}");

        // Quantize is NOT rejected (sc-3172) — it proceeds to the load and fails only on the bogus
        // weights dir, never with an "unsupported" message.
        let quant = LoadSpec {
            quantize: Some(Quant::Q8),
            ..base.clone()
        };
        let err = match load_with(&quant, TURBO_DEFAULTS) {
            Ok(_) => panic!("bogus weights dir must fail to load"),
            Err(e) => e.to_string(),
        };
        assert!(
            !err.contains("quantization") && !err.contains("not part of"),
            "quantize must be accepted (sc-3172); got: {err}"
        );
    }

    #[test]
    fn validate_rejects_bad_inputs() {
        let caps = descriptor_turbo().capabilities;
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &ok).is_ok());

        let empty = GenerationRequest {
            prompt: "".into(),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &empty).is_err());

        let zero_steps = GenerationRequest {
            steps: Some(0),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &zero_steps).is_err());

        let bad_dims = GenerationRequest {
            width: 1000, // not ÷16
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &bad_dims).is_err());
    }
}
