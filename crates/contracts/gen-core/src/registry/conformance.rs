//! Descriptor conformance and identity checks the builder and testkit share.

use super::*;

/// An identifier-shaped registry string: non-empty lowercase `a-z0-9` with `_`/`-`/`.`/`/`
/// separators — the shape every shipped id/family/backend uses (`z_image_turbo`, `image-embed`,
/// `mlx`, and HF-repo-style captioner ids like `fancyfeast/llama-joycaption-beta-one-hf-llava`).
/// Rejects whitespace/uppercase/unicode, which would break worker payload routing and log grepping.
pub(super) fn is_registry_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.' | '/')
        })
}

/// Push an error for every malformed identity field (shared by all descriptor kinds).
pub(super) fn check_identity(errs: &mut Vec<String>, ctx: &str, fields: &[(&str, &str)]) {
    for (name, value) in fields {
        if !is_registry_ident(value) {
            errs.push(format!(
                "{ctx}: {name} {value:?} is not a valid registry identifier \
                 (non-empty lowercase [a-z0-9_.-/])"
            ));
        }
    }
}

/// Push an error for empty/whitespace/duplicate entries in a descriptor's curated name list
/// (samplers / schedulers / guidance methods).
pub(super) fn check_name_list(errs: &mut Vec<String>, ctx: &str, list_name: &str, names: &[&str]) {
    for (i, n) in names.iter().enumerate() {
        if n.is_empty() || n.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: {list_name}[{i}] {n:?} is empty or contains whitespace"
            ));
        }
        if names[..i].contains(n) {
            errs.push(format!("{ctx}: duplicate {list_name} entry {n:?}"));
        }
    }
}

/// The weights-free invariants a generator [`ModelDescriptor`] must satisfy — everything checkable
/// from `(registration.descriptor)()` alone, with no model load (sc-9098, F-009):
///
/// - `id` / `family` / `backend` are non-empty registry identifiers,
/// - `max_count ≥ 1`, and `1 ≤ min_size ≤ max_size` for the visual modalities (a `Default` 0 bound
///   rejects every request with a confusing "out of range 0..=0" — the F-084 footgun, enforced here
///   for *every* linked visual descriptor rather than only when a request happens to reach
///   `validate_request`); the size range is **skipped for `Modality::Audio`**, whose generators emit
///   a track with no width/height and leave the bounds at the unused 0 (matching the size-skipping
///   `validate_request_audio` floor, sc-12834/sc-13314),
/// - any explicit-size grid multiple advertised by [`SizeFloor`](crate::SizeFloor) is non-zero and
///   no larger than the visual descriptor's `max_size`,
/// - `samplers` / `schedulers` / `supported_guidance_methods` entries are non-empty, whitespace-free
///   and duplicate-free (name *shape* only — resolvability is per-engine: several families advertise
///   native sampler names alongside the gen-core curated set),
/// - `conditioning` is duplicate-free, and the video-frame kinds
///   ([`Keyframe`](ConditioningKind::Keyframe) / [`VideoClip`](ConditioningKind::VideoClip) /
///   [`ControlClip`](ConditioningKind::ControlClip) / [`VideoSync`](ConditioningKind::VideoSync) /
///   [`ReferenceVideo`](ConditioningKind::ReferenceVideo)) are
///   not advertised by `Image`-modality models — an `Image` model cannot consume video frames (the LTX
///   clip kinds ride `Video`/`Both`; the `VideoSync` Foley condition rides a `Modality::Audio`
///   video→audio model, sc-13436; the `ReferenceVideo` motion reference rides a video model whose
///   *references* need not share the output modality, sc-17149).
///
/// Returns one message per violation (empty = conformant). Public so a provider's own tests can
/// target a single descriptor; [`ProviderRegistry::descriptor_conformance_errors`] sweeps a catalog.
pub fn model_descriptor_errors(d: &ModelDescriptor) -> Vec<String> {
    let mut errs = Vec::new();
    let ctx = format!("generator '{}'", d.id);
    check_identity(
        &mut errs,
        &ctx,
        &[("id", d.id), ("family", d.family), ("backend", d.backend)],
    );
    if let Some(contract) = d.encoder_contract {
        if let Err(error) = contract.validate_definition() {
            errs.push(format!("{ctx}: {error}"));
        }
    }
    let caps = &d.capabilities;
    if caps.max_count == 0 {
        errs.push(format!(
            "{ctx}: max_count is 0 — every request would be rejected"
        ));
    }
    // Size bounds only mean something for the visual modalities. A `Modality::Audio` generator emits
    // a `GenerationOutput::Audio` track with no width/height, so its `min_size`/`max_size` are unused
    // and left at the natural `Default` 0 — exactly the fields `validate_request_audio` skips the
    // range check for (sc-12834). Enforcing the visual `1 <= min_size <= max_size` floor on an audio
    // descriptor would force a nominal placeholder bound purely to satisfy the sweep (the sc-13314
    // wart), so the check is skipped for `Audio` and stays strict for `Image`/`Video`/`Both`, which
    // genuinely carry a spatial size range.
    if d.modality != Modality::Audio {
        if caps.min_size == 0 || caps.max_size == 0 {
            errs.push(format!(
                "{ctx}: min_size={} max_size={} — size bounds left at the Default 0",
                caps.min_size, caps.max_size
            ));
        } else if caps.min_size > caps.max_size {
            errs.push(format!(
                "{ctx}: min_size {} > max_size {}",
                caps.min_size, caps.max_size
            ));
        }
    }
    if let Some(multiple) = caps.size_floor.explicit_size_multiple() {
        if multiple == 0 {
            errs.push(format!(
                "{ctx}: explicit-size multiple is 0 — grid validation would be undefined"
            ));
        } else if d.modality != Modality::Audio && multiple > caps.max_size {
            errs.push(format!(
                "{ctx}: explicit-size multiple {multiple} > max_size {} — no explicit size can pass",
                caps.max_size
            ));
        }
    }
    // The advertised step surface must be satisfiable (sc-19559). Each of these declares a
    // constraint that refuses EVERY step count, which the shared floor would then enforce on every
    // request — the descriptor mistake is silent otherwise, because `Unconstrained` (the common
    // case) never reaches here.
    match &caps.supported_steps {
        StepSupport::Unconstrained => {}
        StepSupport::Exact(counts) => {
            if counts.is_empty() {
                errs.push(format!(
                    "{ctx}: supported_steps is an EMPTY exact menu — no step count would be \
                     admitted; use StepSupport::Unconstrained for 'no constraint'"
                ));
            }
            if counts.contains(&0) {
                errs.push(format!(
                    "{ctx}: supported_steps advertises 0 steps, which the shared floor always \
                     refuses (an explicit 0 renders undenoised noise)"
                ));
            }
            for (i, c) in counts.iter().enumerate() {
                if counts[..i].contains(c) {
                    errs.push(format!("{ctx}: duplicate supported step count {c}"));
                }
            }
        }
        StepSupport::Range { min, max } => {
            if *min == 0 {
                errs.push(format!(
                    "{ctx}: supported_steps range starts at 0, which the shared floor always \
                     refuses (an explicit 0 renders undenoised noise)"
                ));
            }
            if min > max {
                errs.push(format!(
                    "{ctx}: supported_steps range {min}..={max} is empty — no step count would be \
                     admitted"
                ));
            }
        }
    }
    check_name_list(&mut errs, &ctx, "sampler", &caps.samplers);
    check_name_list(&mut errs, &ctx, "scheduler", &caps.schedulers);
    check_name_list(
        &mut errs,
        &ctx,
        "guidance_method",
        &caps.supported_guidance_methods,
    );
    for (i, k) in caps.conditioning.iter().enumerate() {
        if caps.conditioning[..i].contains(k) {
            errs.push(format!("{ctx}: duplicate conditioning kind {k:?}"));
        }
        let is_video_kind = matches!(
            k,
            ConditioningKind::Keyframe
                | ConditioningKind::VideoClip
                | ConditioningKind::ControlClip
                | ConditioningKind::VideoSync
                | ConditioningKind::ReferenceVideo
        );
        if is_video_kind && d.modality == Modality::Image {
            errs.push(format!(
                "{ctx}: advertises video conditioning {k:?} but modality is Image"
            ));
        }
    }
    // Multi-turn path-A consistency (sc-14150): `supports_conversation_history` is the opt-in flag,
    // but the request carrier is a `Conditioning::ConversationHistory` gated by the conditioning
    // allowlist — so a descriptor that sets the flag without also advertising the kind would set the
    // shared floor's keyed check to pass while the allowlist still rejects every conversation (the
    // "flag on, kind missing" footgun). Cross-check them here so a provider cannot half-wire path A.
    if caps.supports_conversation_history
        && !caps
            .conditioning
            .contains(&ConditioningKind::ConversationHistory)
    {
        errs.push(format!(
            "{ctx}: supports_conversation_history is set but ConditioningKind::ConversationHistory \
             is not in `conditioning` — path-A requests would be rejected by the allowlist"
        ));
    }
    if let Some(space) = d.denoiser_output_latent_space {
        let validation = space.validation();
        if validation.zero_channels {
            errs.push(format!("{ctx}: latent-space channel count is 0"));
        }
        if validation.zero_spatial_compression {
            errs.push(format!(
                "{ctx}: latent-space spatial compression is {}x{} — both factors must be non-zero",
                space.spatial_compression.height, space.spatial_compression.width
            ));
        }
        if let crate::latent::LatentPatchLayout::Packed {
            patch_height,
            patch_width,
        } = space.patch_layout
        {
            if validation.zero_packed_patch {
                errs.push(format!(
                    "{ctx}: packed latent patch is {patch_height}x{patch_width} — both factors must be non-zero"
                ));
            }
        }
        match space.normalization {
            crate::latent::LatentNormalization::Affine { .. } => {
                let (scale, shift) = space
                    .normalization
                    .affine_values()
                    .expect("matched affine normalization");
                if validation.invalid_affine {
                    errs.push(format!(
                        "{ctx}: affine latent normalization has invalid scale={scale:?} shift={shift:?}"
                    ));
                }
            }
            crate::latent::LatentNormalization::PerChannel(stats) => {
                if validation.per_channel_count_mismatch {
                    errs.push(format!(
                        "{ctx}: latent space declares {} channels but normalization {:?} hashes {}",
                        space.channels, stats.identity, stats.channels
                    ));
                }
                if validation.invalid_per_channel_metadata {
                    errs.push(format!(
                        "{ctx}: per-channel latent normalization must have a non-empty, whitespace-free identity and non-zero content hash"
                    ));
                }
            }
            crate::latent::LatentNormalization::LearnedPerChannel { .. } => {
                if validation.invalid_learned_identity {
                    errs.push(format!(
                        "{ctx}: learned latent normalization identity must be non-empty and whitespace-free"
                    ));
                }
            }
            crate::latent::LatentNormalization::Identity => {}
        }
    }
    // Required components (sc-13658): the weights-free advertisement of the named model components a
    // consumer must provision (see `ModelDescriptor::required_components`). Each declared id must be a
    // non-empty, whitespace-free registry token, and the set must be duplicate-free — a blank or
    // repeated id would be an unstageable / ambiguous `LoadSpec::components` key. `&[]` (the shipped
    // value for every image/video provider and every single-file audio model) is trivially
    // conformant.
    check_name_list(&mut errs, &ctx, "required_component", d.required_components);
    errs
}

/// Push duplicate-id errors for one registry kind.
pub(super) fn check_unique_ids(errs: &mut Vec<String>, kind: &str, ids: &[&str]) {
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            errs.push(format!(
                "{kind} id '{id}' is registered more than once (first-wins shadows the rest)"
            ));
        }
    }
}

/// Weights-free descriptor-level conformance sweep over one explicit provider catalog (sc-9098,
/// F-009): generators through [`model_descriptor_errors`], plus identity
/// and capability-bound checks and per-kind id uniqueness for trainers, captioners, transforms and
/// image/text/voice embedders. No `load` is ever called, so it runs by default (no weights, no Metal) —
/// each provider crate invokes it from a default test, giving every cataloged id at least
/// descriptor-level coverage; behavioral conformance (progress/cancel/seed) stays weights-gated in
/// the `gen-core-testkit` suite.
///
/// Returns one message per violation (empty = conformant).
// One `&[Registration]` slice per provider kind, so the arg count tracks the number of kinds (10 as
// of the sc-12851 audio-embedder kind) rather than any avoidable coupling — the alternative is a
// throwaway "all registrations" struct that adds no clarity.
#[allow(clippy::too_many_arguments)]
pub(super) fn descriptor_conformance_errors_for(
    generator_registrations: &[ModelRegistration],
    transform_registrations: &[TransformRegistration],
    audio_transform_registrations: &[AudioTransformRegistration],
    trainer_registrations: &[TrainerRegistration],
    captioner_registrations: &[CaptionerRegistration],
    transcriber_registrations: &[TranscriberRegistration],
    image_embedder_registrations: &[ImageEmbedderRegistration],
    text_embedder_registrations: &[TextEmbedderRegistration],
    voice_embedder_registrations: &[VoiceEmbedderRegistration],
    audio_embedder_registrations: &[AudioEmbedderRegistration],
) -> Vec<String> {
    let mut errs = Vec::new();

    let gen_descs: Vec<ModelDescriptor> = generator_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &gen_descs {
        errs.extend(model_descriptor_errors(d));
    }
    let gen_ids: Vec<&str> = gen_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "generator", &gen_ids);

    let trainer_descs: Vec<TrainerDescriptor> = trainer_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &trainer_descs {
        let ctx = format!("trainer '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
    }
    let trainer_ids: Vec<&str> = trainer_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "trainer", &trainer_ids);

    let cap_descs: Vec<CaptionerDescriptor> = captioner_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &cap_descs {
        let ctx = format!("captioner '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let c = &d.capabilities;
        if c.min_image_size == 0 || c.max_image_size < c.min_image_size {
            errs.push(format!(
                "{ctx}: image-size bounds incoherent (min {} max {})",
                c.min_image_size, c.max_image_size
            ));
        }
        if c.max_new_tokens == 0 {
            errs.push(format!(
                "{ctx}: max_new_tokens is 0 — no caption could be produced"
            ));
        }
    }
    let cap_ids: Vec<&str> = cap_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "captioner", &cap_ids);

    // Transcribers (sc-12850): identity, capability-bound coherence (a non-zero token ceiling and a
    // positive max clip duration — the audio twin of the captioner's max_new_tokens/image-size
    // checks), and id uniqueness.
    let asr_descs: Vec<TranscriberDescriptor> = transcriber_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &asr_descs {
        let ctx = format!("transcriber '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let c = &d.capabilities;
        if c.max_new_tokens == 0 {
            errs.push(format!(
                "{ctx}: max_new_tokens is 0 — no transcript could be produced"
            ));
        }
        if !c.max_audio_seconds.is_finite() || c.max_audio_seconds <= 0.0 {
            errs.push(format!(
                "{ctx}: max_audio_seconds is {} — no audio could be accepted",
                c.max_audio_seconds
            ));
        }
    }
    let asr_ids: Vec<&str> = asr_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "transcriber", &asr_ids);

    let tf_descs: Vec<TransformDescriptor> = transform_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &tf_descs {
        let ctx = format!("transform '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
    }
    let tf_ids: Vec<&str> = tf_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "transform", &tf_ids);

    // Audio transforms (sc-12839): identity, a kind/stem-count coherence check (a separator must
    // advertise ≥ 2 stems; the single-output kinds advertise 0), and id uniqueness.
    let atf_descs: Vec<AudioTransformDescriptor> = audio_transform_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &atf_descs {
        let ctx = format!("audio transform '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let caps = &d.capabilities;
        match caps.kind {
            AudioTransformKind::StemSeparation if caps.stem_count < 2 => errs.push(format!(
                "{ctx}: StemSeparation advertises stem_count {} (a separator must produce ≥ 2 stems)",
                caps.stem_count
            )),
            AudioTransformKind::VoiceConversion | AudioTransformKind::SuperResolution
                if caps.stem_count != 0 =>
            {
                errs.push(format!(
                    "{ctx}: {:?} advertises stem_count {} — only StemSeparation produces stems",
                    caps.kind, caps.stem_count
                ))
            }
            _ => {}
        }
    }
    let atf_ids: Vec<&str> = atf_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "audio transform", &atf_ids);

    let ie_descs: Vec<ImageEmbedderDescriptor> = image_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    let te_descs: Vec<TextEmbedderDescriptor> = text_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    // Audio embedders (sc-12851) carry a joint audio-text `space` exactly like image/text
    // embedders, so they run through the same identity + non-zero-dim + non-empty-space check.
    let ae_descs: Vec<AudioEmbedderDescriptor> = audio_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for (ctx_kind, id, family, backend, dim, space) in ie_descs
        .iter()
        .map(|d| {
            (
                "image embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        })
        .chain(te_descs.iter().map(|d| {
            (
                "text embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        }))
        .chain(ae_descs.iter().map(|d| {
            (
                "audio embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        }))
    {
        let ctx = format!("{ctx_kind} '{id}'");
        check_identity(
            &mut errs,
            &ctx,
            &[("id", id), ("family", family), ("backend", backend)],
        );
        if dim == 0 {
            errs.push(format!("{ctx}: embedding_dim is 0"));
        }
        if space.is_empty() {
            errs.push(format!("{ctx}: embedding space is empty"));
        }
    }
    let ie_ids: Vec<&str> = ie_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "image embedder", &ie_ids);
    let te_ids: Vec<&str> = te_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "text embedder", &te_ids);
    let ae_ids: Vec<&str> = ae_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "audio embedder", &ae_ids);

    // Voice embedders (sc-12838) carry no embedding `space` (they are the audio-identity sibling of
    // the face embedder, not a cross-encoder cosine space) — check identity, a non-zero dim, and
    // id uniqueness only.
    let ve_descs: Vec<VoiceEmbedderDescriptor> = voice_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &ve_descs {
        let ctx = format!("voice embedder '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        if d.embedding_dim == 0 {
            errs.push(format!("{ctx}: embedding_dim is 0"));
        }
    }
    let ve_ids: Vec<&str> = ve_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "voice embedder", &ve_ids);

    errs
}
