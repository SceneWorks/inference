//! Contract conformance for [`gen_core::Trainer`] providers — the training analog of the
//! [`Generator`](crate::conformance) suite (epic 3720, sc-4895). It exercises the behavioral
//! guarantees the [`Trainer`] contract promises but cannot express in the type system: typed
//! cancellation, `TrainingProgress` monotonicity, capability honesty, and registry discoverability.
//!
//! ## Trainer-specific cancellation semantics
//!
//! Unlike a generator (where any cancel → `Canceled`), a trainer that has already completed ≥1
//! optimizer step on cancel returns a **partial** `Ok` — a legitimately-trained adapter with
//! `TrainingOutput.steps < config.steps` (the documented "stopped early" result). The **typed
//! `Err(Error::Canceled)`** contract therefore covers cancellation *before any step runs*:
//!
//! * cancel tripped during dataset caching ⇒ the cache loop breaks, and (depending on whether any
//!   item was cached first) either the empty-cache disambiguation or the `steps_run == 0` guard
//!   returns `Canceled` — never a valid-looking identity adapter (F-040);
//! * a pre-cancelled request ⇒ the cache loop breaks on the first item, and the empty-cache
//!   disambiguation returns `Canceled`.
//!
//! The check drives both paths. Because `train` is `&mut self` and several families are single-use
//! (LTX/Wan free their text encoder after caching), the cancellation check takes a `make` closure
//! and constructs a **fresh** trainer per sub-check rather than sharing one instance.

use std::cell::Cell;
use std::path::PathBuf;

use gen_core::{
    Error, NetworkType, ResolutionBucket, Trainer, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest,
};

/// Parameters for a conformance run. Keep `config.steps` and the dataset tiny — the suite trains a
/// real (if minimal) run for the progress check, so the macOS-lane cost is dominated by it.
#[derive(Clone, Debug)]
pub struct TrainerProfile {
    /// The dataset. The progress check trains over it, so the real lane must point these at actual
    /// images; the Linux stub never reads them. Keep it to 1–2 items.
    pub items: Vec<TrainingItem>,
    /// Hyperparameters. The progress check asserts `TrainingProgress::Training.total == config.steps`
    /// and that the run completes `config.steps`, so keep `steps` small (2) and `save_every` at 0.
    pub config: TrainingConfig,
    /// Where the (final/checkpoint) adapter is written. The cancellation checks assert nothing is
    /// written here; the progress check writes one cheap adapter.
    pub output_dir: PathBuf,
    /// Output adapter file name.
    pub file_name: String,
}

impl TrainerProfile {
    /// The cheapest generally-valid profile: a 2-step run at a 64px bucket, rank 8, no intermediate
    /// checkpoints, over the supplied dataset. `output_dir` is where a passing progress run writes
    /// its one adapter (use a temp dir).
    pub fn cheap(items: Vec<TrainingItem>, output_dir: PathBuf) -> Self {
        Self {
            items,
            config: TrainingConfig {
                rank: 8,
                alpha: 8.0,
                learning_rate: 1e-3,
                steps: 2,
                resolution: 64,
                save_every: 0,
                seed: 7,
                ..Default::default()
            },
            output_dir,
            file_name: "conformance_lora.safetensors".to_owned(),
        }
    }
}

/// Build the in-capability request the positive checks train from, with a fresh cancel flag.
fn base_request(profile: &TrainerProfile) -> TrainingRequest {
    TrainingRequest {
        items: profile.items.clone(),
        config: profile.config.clone(),
        output_dir: profile.output_dir.clone(),
        file_name: profile.file_name.clone(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

/// **Validate honesty.** A declared, in-capability request is accepted; an empty dataset is
/// rejected (the universal floor); and a network type the descriptor does **not** advertise is
/// rejected — all by `validate()`, before any expensive work.
pub fn check_trainer_validate(t: &dyn Trainer, profile: &TrainerProfile) -> Result<(), String> {
    let desc = t.descriptor();
    let id = desc.id;

    // Positive: a declared request using a supported network type must be accepted. Prefer LoRA if
    // supported (every family does), else LoKr.
    let mut ok = base_request(profile);
    ok.config.network_type = if desc.supports_lora {
        NetworkType::Lora
    } else {
        NetworkType::Lokr
    };
    t.validate(&ok).map_err(|e| {
        format!(
            "validate-honesty[{id}]: the declared cheap request was rejected by validate(): {e}"
        )
    })?;

    // Negative: an empty dataset must be rejected before any work.
    let mut empty = base_request(profile);
    empty.items.clear();
    if t.validate(&empty).is_ok() {
        return Err(format!(
            "validate-honesty[{id}]: an empty dataset was accepted by validate()"
        ));
    }

    // Negative: a network type the descriptor does not advertise must be rejected.
    if !desc.supports_lokr {
        let mut lokr = base_request(profile);
        lokr.config.network_type = NetworkType::Lokr;
        if t.validate(&lokr).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a LoKr request was accepted by validate() despite \
                 supports_lokr == false"
            ));
        }
    }
    if !desc.supports_lora {
        let mut lora = base_request(profile);
        lora.config.network_type = NetworkType::Lora;
        if t.validate(&lora).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a LoRA request was accepted by validate() despite \
                 supports_lora == false"
            ));
        }
    }

    // Negative (F-006): a control-branch request on a trainer that does NOT advertise
    // `supports_control` must be rejected by `validate()` — not silently trained as a plain adapter
    // (F-055). The shared `validate_control_request` floor enforces this; assert the trainer routes
    // through it. (A control-capable trainer is exempt — it should accept a well-formed control
    // request; there are none shipped today.)
    if !desc.supports_control {
        let mut ctrl = ok.clone();
        ctrl.config.control_type = Some("pose".to_owned());
        if t.validate(&ctrl).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a control-branch request (control_type set) was accepted by \
                 validate() despite supports_control == false — it must be rejected, not silently \
                 trained as a plain adapter (F-006/F-055)"
            ));
        }
    }

    // Negative (F-006, sc-14056): a full-base-fine-tune request on a trainer that does NOT advertise
    // `supports_full_finetune` must be rejected by `validate()` — not silently trained as a LoRA
    // adapter (F-055), which would hand the caller a small adapter where they asked for a fine-tuned
    // base checkpoint. The shared `validate_full_finetune_request` floor enforces this; assert the
    // trainer routes through it. (A full-tune-capable trainer is exempt — it should accept.)
    if !desc.supports_full_finetune {
        let mut full = ok.clone();
        full.config.full_finetune = true;
        if t.validate(&full).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a full base fine-tune request (full_finetune == true) was \
                 accepted by validate() despite supports_full_finetune == false — it must be \
                 rejected, not silently trained as a LoRA adapter (F-006/F-055)"
            ));
        }
    }

    // Epic 2123 (sc-24826/sc-24827) technique honesty: a technique the descriptor does not declare
    // must be refused by `validate()` with a typed `Unsupported` — never silently ignored (E3). A
    // declared technique must be accepted on the plain adapter request, and weight/gradient noise
    // must still be refused for a full base fine-tune (E5). The shared
    // `validate_training_techniques` floor enforces all three; assert the trainer routes through it.
    check_technique_validate(t, &ok)?;
    check_resolution_buckets_validate(t, &ok)?;

    // Negative (sc-24161): instruction-edit datasets. A trainer that does NOT advertise
    // `max_reference_images` must refuse an edit dataset — never silently train a text-to-image
    // adapter on the edit targets (F-055). An edit-capable trainer must refuse an item carrying one
    // reference more than its advertised cap. The shared `validate_edit_request` floor enforces
    // both; assert the trainer routes through it. The reference paths are never read — the floor
    // runs before any file I/O.
    let cap = desc.max_reference_images as usize;
    let mut edit = ok.clone();
    let refs = if cap == 0 { 1 } else { cap + 1 };
    for item in &mut edit.items {
        item.reference_image_paths = vec![item.image_path.clone(); refs];
    }
    // The refusal must be the RIGHT one, not any error: a capability gap stays a typed
    // `Unsupported` (the worker gates on the variant), and a cap refusal names the cap.
    match (t.validate(&edit), cap) {
        (Ok(()), 0) => Err(format!(
            "validate-honesty[{id}]: an instruction-edit dataset (items with reference images) was \
             accepted by validate() despite max_reference_images == 0 — it must be rejected, not \
             silently trained as a text-to-image adapter (F-055)"
        )),
        (Ok(()), _) => Err(format!(
            "validate-honesty[{id}]: an edit item with {refs} reference images was accepted by \
             validate() despite max_reference_images == {cap}"
        )),
        (Err(Error::Unsupported(_)), 0) => Ok(()),
        (Err(other), 0) => Err(format!(
            "validate-honesty[{id}]: an instruction-edit dataset on a trainer with \
             max_reference_images == 0 must be refused with a typed Error::Unsupported, got \
             {other:?}"
        )),
        (Err(e), _) if e.to_string().contains(&format!("at most {cap}")) => Ok(()),
        (Err(other), _) => Err(format!(
            "validate-honesty[{id}]: an edit item with {refs} reference images must be refused \
             naming the cap (\"at most {cap}\"), got {other:?}"
        )),
    }
}

/// The suggested upstream weight-noise strength, used as the "technique on" probe value.
const WEIGHT_NOISE_PROBE_SIGMA: f32 = 0.0125;
/// The suggested upstream gradient-noise eta (sc-24827), used as the "technique on" probe value.
const GRADIENT_NOISE_PROBE_ETA: f32 = 0.01;

/// One optional training technique the honesty / refusal checks probe (epic 2123): its display
/// name, the config knob it turns on, and the descriptor flag that declares it.
struct TechniqueProbe {
    name: &'static str,
    knob: &'static str,
    enable: fn(&mut TrainingRequest),
    declared: fn(&gen_core::TrainingTechniques) -> bool,
}

const TECHNIQUE_PROBES: &[TechniqueProbe] = &[
    TechniqueProbe {
        name: "weight_noise",
        knob: "weight_noise_sigma",
        enable: |r| r.config.weight_noise_sigma = WEIGHT_NOISE_PROBE_SIGMA,
        declared: |t| t.weight_noise,
    },
    TechniqueProbe {
        name: "gradient_noise",
        knob: "gradient_noise_eta",
        enable: |r| r.config.gradient_noise_eta = GRADIENT_NOISE_PROBE_ETA,
        declared: |t| t.gradient_noise,
    },
];

/// Technique half of [`check_trainer_validate`] (sc-24826 weight noise, sc-24827 gradient noise) —
/// `ok` is the accepted base request. For each probed technique: an undeclared one must be refused
/// by `validate()` with a typed `Unsupported`, a declared one accepted on the plain adapter request,
/// and either one refused for a full base fine-tune (both are adapter-only, E5).
fn check_technique_validate(t: &dyn Trainer, ok: &TrainingRequest) -> Result<(), String> {
    let desc = t.descriptor();
    let id = desc.id;
    for probe in TECHNIQUE_PROBES {
        let (name, knob) = (probe.name, probe.knob);
        let declared = (probe.declared)(&desc.techniques);
        let mut on = ok.clone();
        (probe.enable)(&mut on);
        match (t.validate(&on), declared) {
            (Ok(()), false) => {
                return Err(format!(
                    "technique-honesty[{id}]: a {name} request ({knob} > 0) was accepted by \
                     validate() despite techniques.{name} == false — it must be refused, not \
                     silently ignored (epic 2123 E3)"
                ))
            }
            (Err(Error::Unsupported(_)), false) | (Ok(()), true) => {}
            (Err(other), false) => {
                return Err(format!(
                    "technique-honesty[{id}]: an unsupported {name} request must be refused with \
                     a typed Error::Unsupported, got {other:?}"
                ))
            }
            (Err(e), true) => {
                return Err(format!(
                    "technique-honesty[{id}]: a {name} request was rejected by validate() despite \
                     techniques.{name} == true: {e}"
                ))
            }
        }
        let mut full = on;
        full.config.full_finetune = true;
        if t.validate(&full).is_ok() {
            return Err(format!(
                "technique-honesty[{id}]: {name} combined with a full base fine-tune was accepted \
                 by validate() — it must touch adapter factors only (epic 2123 E5)"
            ));
        }
    }
    Ok(())
}

/// The "technique on" resolution-bucket probe (sc-2127): the profile's own resolution at two
/// repeats plus twice that resolution at one — a real two-bucket mix that stays as cheap as the
/// profile allows.
fn probe_buckets(config: &TrainingConfig) -> Vec<ResolutionBucket> {
    // On the trainers' latent stride, so the probe is a well-formed list whatever the profile.
    let stride = gen_core::RESOLUTION_BUCKET_STRIDE;
    let base = (config.resolution / stride * stride).max(stride);
    vec![
        ResolutionBucket {
            resolution: base,
            repeats: 2,
        },
        ResolutionBucket {
            resolution: base * 2,
            repeats: 1,
        },
    ]
}

/// Resolution-bucket half of [`check_trainer_validate`] (sc-2127) — `ok` is the accepted base
/// request. Undeclared ⇒ typed `Unsupported`; declared ⇒ accepted; a zero repeat count is refused
/// either way (the floor's malformed-list check).
fn check_resolution_buckets_validate(t: &dyn Trainer, ok: &TrainingRequest) -> Result<(), String> {
    let desc = t.descriptor();
    let id = desc.id;
    let mut bucketed = ok.clone();
    bucketed.config.resolution_buckets = probe_buckets(&ok.config);
    match (t.validate(&bucketed), desc.techniques.resolution_buckets) {
        (Ok(()), false) => {
            return Err(format!(
                "technique-honesty[{id}]: a resolution_buckets request was accepted by validate() \
                 despite techniques.resolution_buckets == false — it must be refused, not silently \
                 trained at one resolution (epic 2123 E3)"
            ))
        }
        (Err(Error::Unsupported(_)), false) | (Ok(()), true) => {}
        (Err(other), false) => {
            return Err(format!(
                "technique-honesty[{id}]: an unsupported resolution_buckets request must be \
                 refused with a typed Error::Unsupported, got {other:?}"
            ))
        }
        (Err(e), true) => {
            return Err(format!(
                "technique-honesty[{id}]: a resolution_buckets request was rejected by validate() \
                 despite techniques.resolution_buckets == true: {e}"
            ))
        }
    }
    let mut zero = bucketed;
    zero.config.resolution_buckets[0].repeats = 0;
    if t.validate(&zero).is_ok() {
        return Err(format!(
            "technique-honesty[{id}]: a resolution bucket with repeats == 0 was accepted by \
             validate()"
        ));
    }
    Ok(())
}

/// **Technique refusal at the `train` entry point** (epic 2123 E3, sc-24826/sc-24827/sc-2127). A
/// caller that skips `validate` and calls `train` directly with a technique the trainer does not
/// declare (weight noise, gradient noise, resolution buckets) must get a typed
/// `Err(Error::Unsupported)` **before training starts** — no `Caching`/`Training`/`Saving` event, so
/// nothing is loaded, cached or written. Each undeclared technique is probed on a fresh trainer; a
/// trainer that declares every probed technique passes vacuously (its positive path is covered by
/// [`check_trainer_progress`] / [`check_trainer_bucketed_progress`]).
pub fn check_trainer_technique_refusal(
    make: &dyn Fn() -> Box<dyn Trainer>,
    profile: &TrainerProfile,
) -> Result<(), String> {
    for probe in TECHNIQUE_PROBES {
        if (probe.declared)(&make().descriptor().techniques) {
            continue;
        }
        let mut req = base_request(profile);
        (probe.enable)(&mut req);
        refuse_at_train(
            make,
            req,
            &format!("{} > 0", probe.knob),
            &format!("techniques.{}", probe.name),
        )?;
    }
    if !make().descriptor().techniques.resolution_buckets {
        let mut req = base_request(profile);
        req.config.resolution_buckets = probe_buckets(&req.config);
        refuse_at_train(
            make,
            req,
            "resolution_buckets set",
            "techniques.resolution_buckets",
        )?;
    }
    Ok(())
}

/// One undeclared-technique probe of [`check_trainer_technique_refusal`].
fn refuse_at_train(
    make: &dyn Fn() -> Box<dyn Trainer>,
    req: TrainingRequest,
    knob: &str,
    flag: &str,
) -> Result<(), String> {
    let mut t = make();
    let id = t.descriptor().id;
    let mut started = false;
    let result = t.train(&req, &mut |p| {
        if matches!(
            p,
            TrainingProgress::Caching { .. }
                | TrainingProgress::Training { .. }
                | TrainingProgress::Saving
        ) {
            started = true;
        }
    });
    match result {
        Err(Error::Unsupported(_)) if !started => Ok(()),
        Err(Error::Unsupported(_)) => Err(format!(
            "technique-refusal[{id}]: train() refused {knob} only after training had started \
             (caching/training/saving progress was emitted) — refuse before any work (E3)"
        )),
        Ok(out) => Err(format!(
            "technique-refusal[{id}]: train() ran {} steps with {knob} despite {flag} == false — \
             the knob was silently ignored (E3)",
            out.steps
        )),
        Err(other) => Err(format!(
            "technique-refusal[{id}]: train() with an unsupported request ({knob}) must return a \
             typed Err(Error::Unsupported), got {other:?}"
        )),
    }
}

/// **Bucketed progress** (sc-2127). A trainer that declares
/// [`resolution_buckets`](gen_core::TrainingTechniques::resolution_buckets) must complete a
/// two-bucket run end to end: `Caching` still counts dataset items (`1..=items.len()`, every bucket
/// of an item is cached under its one event) and `Training` still counts `1..=config.steps`. An
/// undeclared trainer passes vacuously (its refusal is [`check_trainer_technique_refusal`]).
pub fn check_trainer_bucketed_progress(
    make: &dyn Fn() -> Box<dyn Trainer>,
    profile: &TrainerProfile,
) -> Result<(), String> {
    let mut t = make();
    if !t.descriptor().techniques.resolution_buckets {
        return Ok(());
    }
    let id = t.descriptor().id;
    let mut req = base_request(profile);
    req.config.resolution_buckets = probe_buckets(&req.config);
    let mut caching: Vec<(u32, u32)> = Vec::new();
    let mut training: Vec<(u32, u32)> = Vec::new();
    let out = t
        .train(&req, &mut |p| match p {
            TrainingProgress::Caching { current, total } => caching.push((current, total)),
            TrainingProgress::Training { step, total, .. } => training.push((step, total)),
            _ => {}
        })
        .map_err(|e| format!("bucketed-progress[{id}]: train() failed on a two-bucket run: {e}"))?;
    check_monotone(id, "Caching", &caching, profile.items.len() as u32)?;
    check_monotone(id, "Training", &training, profile.config.steps)?;
    if out.steps != profile.config.steps {
        return Err(format!(
            "bucketed-progress[{id}]: TrainingOutput.steps ({}) != config.steps ({})",
            out.steps, profile.config.steps
        ));
    }
    Ok(())
}

/// **Progress.** A completed (uncancelled) run streams `TrainingProgress::Caching` over exactly
/// `1..=items.len()` and `TrainingProgress::Training` over exactly `1..=config.steps` (monotone,
/// complete, constant `total`), and `TrainingOutput.steps == config.steps`.
pub fn check_trainer_progress(t: &mut dyn Trainer, profile: &TrainerProfile) -> Result<(), String> {
    let id = t.descriptor().id;
    let req = base_request(profile);
    let mut caching: Vec<(u32, u32)> = Vec::new();
    let mut training: Vec<(u32, u32)> = Vec::new();
    let out = t
        .train(&req, &mut |p| match p {
            TrainingProgress::Caching { current, total } => caching.push((current, total)),
            TrainingProgress::Training { step, total, .. } => training.push((step, total)),
            _ => {}
        })
        .map_err(|e| format!("progress[{id}]: train() failed on the cheap request: {e}"))?;

    check_monotone(id, "Caching", &caching, profile.items.len() as u32)?;
    check_monotone(id, "Training", &training, profile.config.steps)?;

    if out.steps != profile.config.steps {
        return Err(format!(
            "progress[{id}]: TrainingOutput.steps ({}) != config.steps ({}) on an uncancelled run",
            out.steps, profile.config.steps
        ));
    }
    Ok(())
}

/// Shared monotonicity assertion for a `(current, total)` event stream: `total` constant and equal
/// to `expected_total`, `current` exactly `1..=expected_total`.
fn check_monotone(
    id: &str,
    band: &str,
    events: &[(u32, u32)],
    expected_total: u32,
) -> Result<(), String> {
    if events.is_empty() {
        return Err(format!(
            "progress[{id}]: train() emitted no TrainingProgress::{band} events"
        ));
    }
    let total = events[0].1;
    if let Some((c, t)) = events.iter().find(|(_, t)| *t != total) {
        return Err(format!(
            "progress[{id}]: {band}.total changed mid-run ({total} then {t} at current={c})"
        ));
    }
    let observed: Vec<u32> = events.iter().map(|(c, _)| *c).collect();
    let expected: Vec<u32> = (1..=total).collect();
    if observed != expected {
        return Err(format!(
            "progress[{id}]: {band}.current must be exactly 1..={total} (monotone, complete, no \
             repeats); got {observed:?}"
        ));
    }
    if total != expected_total {
        return Err(format!(
            "progress[{id}]: {band}.total ({total}) != the expected count ({expected_total})"
        ));
    }
    Ok(())
}

/// **Cancellation.** Cancelling before any optimizer step runs makes `train` return the **typed**
/// `Err(Error::Canceled)` (not a stringified `Msg`) and write **no** adapter (no `Saving` event).
/// Two paths are exercised against fresh trainers:
///
/// 1. **pre-cancelled** — the cache loop breaks on the first item, empty-cache disambiguation;
/// 2. **cancel during caching** — tripped at the first `Caching` event, so ≥1 item caches but the
///    training loop breaks before step 1 (`steps_run == 0` guard).
pub fn check_trainer_cancellation(
    make: &dyn Fn() -> Box<dyn Trainer>,
    profile: &TrainerProfile,
) -> Result<(), String> {
    // Path 1: a request that is already cancelled when train() is called.
    {
        let mut t = make();
        let id = t.descriptor().id;
        let req = base_request(profile);
        req.cancel.cancel();
        let mut saved = false;
        let result = t.train(&req, &mut |p| {
            if matches!(p, TrainingProgress::Saving) {
                saved = true;
            }
        });
        classify_cancel(id, "pre-cancelled", result, saved)?;
    }

    // Path 2: cancellation tripped at the first Caching event (≥1 item cached, then the training
    // loop breaks before any step → the steps_run == 0 guard).
    {
        let mut t = make();
        let id = t.descriptor().id;
        let req = base_request(profile);
        let cancel = req.cancel.clone();
        let tripped = Cell::new(false);
        let mut saved = false;
        let result = t.train(&req, &mut |p| match p {
            TrainingProgress::Caching { .. } => {
                if !tripped.get() {
                    cancel.cancel();
                    tripped.set(true);
                }
            }
            TrainingProgress::Saving => saved = true,
            _ => {}
        });
        if !tripped.get() {
            return Err(format!(
                "cancellation[{id}]: no TrainingProgress::Caching was emitted, so mid-caching \
                 cancellation could not be exercised (a trainer must report caching progress)"
            ));
        }
        classify_cancel(id, "cancel-during-caching", result, saved)?;
    }
    Ok(())
}

/// Turn a cancelled `train` result into a pass/fail verdict: it must be the typed `Canceled` and
/// must not have emitted `Saving`.
fn classify_cancel(
    id: &str,
    path: &str,
    result: gen_core::Result<gen_core::TrainingOutput>,
    saved: bool,
) -> Result<(), String> {
    match result {
        Ok(out) => Err(format!(
            "cancellation[{id}/{path}]: train() returned Ok ({} steps) despite cancellation before \
             any step; it must return Err(Error::Canceled) and write no adapter (F-040)",
            out.steps
        )),
        Err(Error::Canceled) if saved => Err(format!(
            "cancellation[{id}/{path}]: returned Canceled but emitted TrainingProgress::Saving — a \
             cancelled-before-any-step run must not write an adapter (F-040)"
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "cancellation[{id}/{path}]: must return the typed Err(Error::Canceled) on cancel, got \
             {other:?} — a stringified Error::Msg breaks the typed-cancellation contract (sc-4895)"
        )),
    }
}

/// **Registry round-trip.** The trainer's descriptor `id` is present in the explicit registry
/// supplied by the caller.
pub fn check_trainer_registry(
    registry: &gen_core::ProviderRegistry,
    t: &dyn Trainer,
) -> Result<(), String> {
    let id = t.descriptor().id;
    if registry
        .trainers()
        .any(|registration| (registration.descriptor)().id == id)
    {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found in the explicit provider registry (gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// Run the full trainer conformance suite. `make` constructs a fresh trainer (it is invoked several
/// times — once for the validate/registry pair, once for the progress run, once per cancellation
/// path, once per undeclared-technique refusal probe, and once for the bucketed run of a
/// bucket-capable trainer — because `train` is `&mut self` and several families are single-use).
/// Panics with every failure aggregated.
pub fn trainer_conformance(make: impl Fn() -> Box<dyn Trainer>, profile: &TrainerProfile) {
    let mut failures: Vec<String> = Vec::new();

    // Capture the descriptor id here (F-059) so
    // the aggregated-failure panic below doesn't reload a multi-GB trainer a fourth time just to name
    // it — a failing conformance run otherwise pays an extra multi-minute load (or a flaky reload
    // replaces the panic message entirely).
    let id;
    {
        let t = make();
        id = t.descriptor().id;
        if let Err(e) = check_trainer_validate(t.as_ref(), profile) {
            failures.push(e);
        }
    }

    // progress trains a fresh instance to completion.
    {
        let mut t = make();
        if let Err(e) = check_trainer_progress(t.as_mut(), profile) {
            failures.push(e);
        }
    }

    if let Err(e) = check_trainer_cancellation(&make, profile) {
        failures.push(e);
    }

    if let Err(e) = check_trainer_technique_refusal(&make, profile) {
        failures.push(e);
    }

    if let Err(e) = check_trainer_bucketed_progress(&make, profile) {
        failures.push(e);
    }

    if !failures.is_empty() {
        panic!(
            "gen-core trainer conformance FAILED for `{id}` (gen-core {}):\n  - {}",
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
