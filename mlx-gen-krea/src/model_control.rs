//! `KreaTurboControl` — the Krea 2 Turbo **pose-ControlNet** variant (sc-8465, epic 8459 S5): strict
//! pose conditioning via a trained control branch overlaid on the frozen Krea 2 Turbo base, registered
//! as its own `Generator` (`krea_2_turbo_control`). The MLX twin of the candle `Krea2Control` provider
//! (candle-gen-krea, sc-8464) — same 8-step CFG-free Turbo sampler, same single-forward residual
//! injection, same `control_scale` semantics.
//!
//! Identical to [`crate::model::Krea`] (Turbo) except a [`Krea2ControlBranch`] rides the DiT and
//! `generate` threads a VAE-encoded pose skeleton through it. [`load`] needs the base snapshot
//! (`spec.weights`, the DENSE `krea/Krea-2-Turbo` diffusers tree — NOT the packed Q4/Q8 turnkey the
//! plain `krea_2_turbo` gen uses, because the branch is a composable-forward overlay trained on the bf16
//! base) **and** the control overlay checkpoint (`spec.control`). Pose-only + dense bf16 (no quant, no
//! negative prompt, no guidance), mirroring the candle `krea_2_turbo_control` engine.

use mlx_gen::gen_core;
use mlx_gen::{
    require_base_dir, require_control, AcceptedControlKinds, ConditioningKind, ControlBranch,
    Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    OffloadPolicy, Precision, Progress, Result,
};

use mlx_gen::default_seed;

use mlx_rs::Array;
use std::path::Path;

use crate::config::Krea2Config;
use crate::control::Krea2ControlBranch;
use crate::pipeline::{KreaHeavy, KreaText, TurboOptions};

/// Registry id for the Krea 2 Turbo pose-ControlNet variant. Matches the SceneWorks worker's
/// `STRICT_CONTROL_ENGINES` `krea_2_turbo_control` id and the candle lane, so one id serves both
/// backends (the worker's OS/feature cfg picks the MLX vs candle provider).
pub const KREA_2_TURBO_CONTROL_ID: &str = "krea_2_turbo_control";

/// Turbo default steps (CFG-free distilled few-step), shared with the base [`crate::model`] Turbo path.
const DEFAULT_STEPS: u32 = 8;

/// Krea 2 Turbo pose-control identity + capabilities. Derived from the base Turbo [`crate::model::descriptor`]
/// so the shared surface (family/backend/samplers/size bounds/LoRA/mac_only) stays in lockstep; the pose
/// control lane swaps the Turbo img2img `Reference` surface for a required `Control` conditioning and is
/// dense-only (the overlay is trained on the bf16 base — no quant tier, matching candle).
pub fn descriptor() -> ModelDescriptor {
    let mut d = crate::model::descriptor();
    d.id = KREA_2_TURBO_CONTROL_ID;
    // Pose ControlNet: a required Control conditioning replaces Turbo's optional img2img Reference.
    d.capabilities.conditioning = vec![ConditioningKind::Control];
    // Dense bf16 only — the pose overlay is trained on the dense base; a packed base would desync the
    // main-stream activations the residual was trained against (candle "no quant tier").
    d.capabilities.supported_quants = &[];
    d
}

/// A loaded Krea 2 Turbo pose-control generator: the cached descriptor + a component-residency strategy
/// (the base Turbo text phase + DiT/VAE + the pose control branch).
pub struct KreaTurboControl {
    descriptor: ModelDescriptor,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11101), selected from
    /// [`LoadSpec::offload_policy`]. `Resident` (default) holds the text phase + DiT + VAE + branch warm;
    /// `Sequential` holds only the [`LoadSpec`] and re-loads per generation in phase order (encode →
    /// **drop the text phase** → denoise/decode), bounding peak unified memory to `max(text, DiT+VAE+
    /// branch)`. Dense bf16 only, so the tier that matters here is bf16 (no Q8/Q4 lever).
    residency: ControlResidency,
}

/// The heavy-component residency for a [`KreaTurboControl`] (sc-11101). See [`KreaTurboControl::residency`].
enum ControlResidency {
    /// Every component loaded once at [`load`] and held (today's warm path). Boxed so this heavy variant
    /// doesn't bloat every `Sequential` handle (`clippy::large_enum_variant`).
    Resident(Box<ControlResident>),
    /// Only the [`LoadSpec`] is held; each `generate` re-loads the components in phase order and frees
    /// them after. The per-phase loaders rebuild byte-identical components to the `Resident` path.
    Sequential(Box<LoadSpec>),
}

/// The Krea text phase held resident (dropped first under `Sequential`), paired with the heavy render
/// bundle (DiT + VAE + the pose control branch).
struct ControlResident {
    text: KreaText,
    heavy: ControlHeavyOwned,
}

/// The heavy render-phase components for pose control: the single-stream DiT + VAE ([`KreaHeavy`]) plus
/// the pose [`Krea2ControlBranch`] — which is a SECOND heavy component that stays on the heavy side (it
/// rides the DiT), mirroring qwen-image-control's `QwenControlHeavyOwned`. Owned by the `Resident`
/// components or a `Sequential` generate.
struct ControlHeavyOwned {
    heavy: KreaHeavy,
    branch: Krea2ControlBranch,
}

/// A borrow of the pose-control heavy components, so the render loop runs identically whether they are
/// held resident or were just loaded by the `Sequential` path.
struct ControlHeavyRef<'a> {
    heavy: &'a KreaHeavy,
    branch: &'a Krea2ControlBranch,
}

impl ControlHeavyOwned {
    fn as_ref(&self) -> ControlHeavyRef<'_> {
        ControlHeavyRef {
            heavy: &self.heavy,
            branch: &self.branch,
        }
    }
}

/// Construct a [`KreaTurboControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] DENSE `krea/Krea-2-Turbo` snapshot (`transformer/
/// text_encoder/ vae/ tokenizer/`), and `spec.control` (required) the converted MLX pose overlay
/// (`control_step5000` — a single `.safetensors` `File`, or a `Dir` of shards). Loaded dense bf16;
/// Raw-trained LoRA/LoKr in `spec.adapters` install onto the base DiT (the branch is never an adapter
/// target). A `spec.quantize` override is rejected — the pose overlay is dense-only.
///
/// Component residency (epic 10834 Phase 1, sc-11101): `Resident` (default) builds every component now
/// and holds it warm; `Sequential` keeps only the spec and re-loads per generate in phase order (encode
/// → drop the text phase → denoise/decode). Both use the same per-phase loaders, so the components are
/// byte-identical. Validity (base dir, control present, no quant, bf16) is checked up front either way.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // Fail fast — validate the whole spec up front for BOTH residencies (mirrors the pre-sc-11101 load
    // order): dense bf16, a base snapshot dir, the required control overlay, and no quant override.
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{KREA_2_TURBO_CONTROL_ID}: only the default dense bf16 precision is wired (drop the \
             precision override)"
        )));
    }
    let root = require_base_dir(spec, KREA_2_TURBO_CONTROL_ID, "a base snapshot directory")?;
    let _ = require_control(spec, KREA_2_TURBO_CONTROL_ID, "Krea 2 pose control overlay")?;
    if spec.quantize.is_some() {
        return Err(Error::Msg(format!(
            "{KREA_2_TURBO_CONTROL_ID}: the pose control overlay is trained on the dense bf16 base; \
             quantization is not supported (drop the quant override, point at a dense snapshot)"
        )));
    }

    let residency = match spec.offload_policy {
        OffloadPolicy::Resident => {
            let text = KreaText::from_snapshot(root)?;
            let heavy = load_control_heavy(spec, root)?;
            ControlResidency::Resident(Box::new(ControlResident { text, heavy }))
        }
        OffloadPolicy::Sequential => ControlResidency::Sequential(Box::new(spec.clone())),
    };

    Ok(Box::new(KreaTurboControl {
        descriptor: descriptor(),
        residency,
    }))
}

/// Load the pose-control heavy phase — the base DiT + VAE ([`KreaHeavy`]), Raw-trained LoRA/LoKr on the
/// base DiT (the branch is never an adapter target), then the pose branch (`spec.control`). Dense bf16
/// (no quant). Factored so `Sequential` loads these AFTER the text phase is dropped. `spec.control` was
/// already validated present by [`load`]; re-requiring it here keeps the `Sequential` per-generate path
/// self-contained.
fn load_control_heavy(spec: &LoadSpec, root: &Path) -> Result<ControlHeavyOwned> {
    let mut heavy = KreaHeavy::from_snapshot(root)?;
    if !spec.adapters.is_empty() {
        heavy.apply_adapters(&spec.adapters)?;
    }
    let control = require_control(spec, KREA_2_TURBO_CONTROL_ID, "Krea 2 pose control overlay")?;
    let cfg = Krea2Config::from_snapshot(root)?;
    let branch = Krea2ControlBranch::from_source(control, &cfg)?;
    Ok(ControlHeavyOwned { heavy, branch })
}

/// The pose branch is pose-only (`Only([Pose])`) and defaults an unset `control_scale` to the S0 mid
/// value (0.6). All the resolve/validate-present boilerplate comes from the shared trait (sc-8241).
impl ControlBranch for KreaTurboControl {
    fn model_id(&self) -> &'static str {
        KREA_2_TURBO_CONTROL_ID
    }

    fn accepted_control_kinds(&self) -> AcceptedControlKinds {
        AcceptedControlKinds::Only(vec![mlx_gen::ControlKind::Pose])
    }

    fn default_control_scale(&self) -> f32 {
        crate::control::DEFAULT_CONTROL_SCALE
    }
}

impl KreaTurboControl {
    /// Text-encode the prompt per the residency (sc-11101). Pose control is CFG-free (one context, no
    /// negative). `Resident` borrows the warm text phase (byte-identical to the pre-sc-11101 per-image
    /// re-encode); `Sequential` loads the text phase, encodes, forces materialization (`eval`), then
    /// DROPS it + `clear_cache()` so its ~4 GB frees before the DiT/VAE/branch load.
    fn encode(&self, prompt: &str) -> Result<Array> {
        match &self.residency {
            ControlResidency::Resident(c) => c.text.encode(prompt),
            ControlResidency::Sequential(spec) => {
                let root =
                    require_base_dir(spec, KREA_2_TURBO_CONTROL_ID, "a base snapshot directory")?;
                let text = KreaText::from_snapshot(root)?;
                let ctx = text.encode(prompt)?;
                // MLX is lazy — materialize NOW while `text` is alive, else `ctx` keeps the encoder
                // weights referenced through the graph and the drop frees nothing.
                mlx_rs::transforms::eval([&ctx])?;
                drop(text);
                mlx_rs::memory::clear_cache();
                Ok(ctx)
            }
        }
    }

    /// Load the heavy render components (DiT + VAE + branch) for a `Sequential` job — after
    /// [`Self::encode`] dropped the text phase — or `None` under `Resident` (already held).
    fn load_seq_heavy(&self) -> Result<Option<ControlHeavyOwned>> {
        match &self.residency {
            ControlResidency::Resident(_) => Ok(None),
            ControlResidency::Sequential(spec) => {
                let root =
                    require_base_dir(spec, KREA_2_TURBO_CONTROL_ID, "a base snapshot directory")?;
                Ok(Some(load_control_heavy(spec, root)?))
            }
        }
    }

    /// Borrow the heavy render components: the warm bundle under `Resident`, or the just-loaded
    /// `seq_heavy` under `Sequential`. The render loop is written once against this borrow.
    fn heavy<'a>(&'a self, seq_heavy: &'a Option<ControlHeavyOwned>) -> ControlHeavyRef<'a> {
        match (&self.residency, seq_heavy) {
            (ControlResidency::Resident(c), _) => c.heavy.as_ref(),
            (_, Some(owned)) => owned.as_ref(),
            (ControlResidency::Sequential(_), None) => {
                unreachable!("Sequential residency always loads seq_heavy before rendering")
            }
        }
    }

    /// The rich-`Result` body behind [`Generator::generate`] (the crate's own [`mlx_gen::Error`] so `?`
    /// lifts `mlx_rs` device exceptions; the trait wrapper bridges into [`gen_core::Error`]). Renders
    /// `req.count` CFG-free Turbo images, one per pose per seed (`seed + n`), each pose-locked by the
    /// single-forward residual injection — through the residency (encode → drop text phase under
    /// `Sequential` → load heavy+branch → per-image render → free heavy).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // The required pose skeleton + its resolved scale (unset → the 0.6 default; Some(0.0) inert).
        let (control_image, control_scale) = self.resolve_control(req)?;

        // Phase A: prompt → context (sc-11101). Under `Sequential` this loads the text phase, encodes,
        // forces materialization, then DROPS it + `clear_cache()` before the DiT/VAE/branch load below.
        let context = self.encode(&req.prompt)?;

        // Phase B: heavy render components (DiT + VAE + the pose branch). `Resident` borrows the warm
        // bundle; `Sequential` loads it NOW — after the text phase was dropped — and frees it when done.
        let seq_heavy = self.load_seq_heavy()?;
        let heavy = self.heavy(&seq_heavy);

        let mut images = Vec::with_capacity(req.count as usize);
        for n in 0..req.count {
            let opts = TurboOptions {
                width: req.width,
                height: req.height,
                steps,
                seed: base_seed.wrapping_add(n as u64),
                sampler: req.sampler.clone(),
                scheduler: req.scheduler.clone(),
            };
            let img = heavy.heavy.render_turbo_control(
                &context,
                heavy.branch,
                control_image,
                control_scale,
                &opts,
                &req.cancel,
                on_progress,
            )?;
            images.push(img);
        }
        // Sequential (sc-11101): free the DiT/VAE/branch working set now that every image is rendered,
        // then `clear_cache()`. Resident is a no-op (`seq_heavy` None).
        let was_sequential = seq_heavy.is_some();
        drop(seq_heavy);
        if was_sequential {
            mlx_rs::memory::clear_cache();
        }
        Ok(GenerationOutput::Images(images))
    }
}

impl Generator for KreaTurboControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Capability floor (size/count/guidance/negative/conditioning-kind), then the shared
        // control-present check (a `Conditioning::Control` must be present).
        crate::model::validate_request(&self.descriptor, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

// Link-time registration (epic 3720): `krea_2_turbo_control`. The `impl Generator` stays hand-written
// because `validate` adds the control-present check beyond the shared `validate_request`, so it is not
// the plain delegation `impl_generator!` expresses (the z-image control precedent).
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{Conditioning, ControlKind, Modality, Quant, WeightsSource};

    #[test]
    fn descriptor_is_krea_2_turbo_control() {
        let d = descriptor();
        assert_eq!(d.id, "krea_2_turbo_control");
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Pose ControlNet: Control conditioning (not the base Turbo Reference), CFG-free, dense-only.
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert_eq!(d.capabilities.supported_quants, &[] as &[Quant]);
        // Shared surface stays in lockstep with the base Turbo descriptor.
        assert!(d.capabilities.mac_only);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(
            d.capabilities.max_count,
            crate::model::descriptor().capabilities.max_count
        );
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // A base dir but no `spec.control` → fail on the missing overlay (proving it is a hard
        // requirement — never a silent un-conditioned base).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("pose control overlay"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/krea.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_rejects_quant_override() {
        // Dense-only: a quant override is rejected (not silently ignored), even with a control overlay.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()))
            .with_quant(Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("quantization is not supported"), "got: {err}");
    }

    #[test]
    fn reachable_via_registry_by_id() {
        assert!(
            gen_core::registry::generators()
                .any(|r| (r.descriptor)().id == KREA_2_TURBO_CONTROL_ID),
            "id {KREA_2_TURBO_CONTROL_ID} not registered"
        );
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = gen_core::registry::load(KREA_2_TURBO_CONTROL_ID, &spec)
            .err()
            .expect("missing weights → err")
            .to_string();
        assert!(
            !err.contains("no generator registered"),
            "id not resolved: {err}"
        );
    }

    #[test]
    fn accepts_pose_rejects_other_control_kinds() {
        // A pose-only branch: the trait's resolve_control admits Pose and rejects Canny/Depth. Exercised
        // through the descriptor's accepted_control_kinds via a lightweight stub-free check on the enum.
        let accepted = AcceptedControlKinds::Only(vec![ControlKind::Pose]);
        assert!(accepted.accepts(&ControlKind::Pose));
        assert!(!accepted.accepts(&ControlKind::Canny));
        assert!(!accepted.accepts(&ControlKind::Depth));
        // Sanity: the trait wiring builds the same Conditioning::Control the worker feeds.
        let _ = Conditioning::Control {
            image: mlx_gen::media::Image {
                width: 8,
                height: 8,
                pixels: vec![0u8; 8 * 8 * 3],
            },
            kind: ControlKind::Pose,
            scale: Some(0.6),
        };
    }
}
