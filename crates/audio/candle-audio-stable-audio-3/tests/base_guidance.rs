//! What the `-base` registrations are actually for: real classifier-free guidance and a real
//! negative prompt (sc-14546).
//!
//! # The claim this file had to correct before it could gate anything
//!
//! sc-14546 was written on the premise that the base checkpoints are "the only checkpoints where
//! `cfg_scale` and `negative_prompt` do anything". At the level of the ported graph that is false:
//! [`candle_audio_stable_audio_3::dit`]'s guided forward takes its batch-1 shortcut when
//! `cfg_scale == 1.0` and on no other condition, so at any other guidance the identical batch-2
//! CFG / APG / rescale path runs on a post-trained checkpoint too. Upstream's "no effect on
//! post-trained checkpoints" is a statement about what distillation did to the weights.
//!
//! What *is* true, and what this file pins, is that `1.0` is the crate's post-trained default and
//! `7.0` is the base default — so a default post-trained request genuinely ignores its negative
//! prompt and a default base request genuinely does not. Both halves are asserted, in the same run,
//! against the same checkpoint.
//!
//! # What the divergence assertion proves, and what it does not
//!
//! "A negative prompt changes the output" is only evidence if the change is bigger than the
//! implementation's own numerical disagreement with the reference. So the divergence gate measures
//! that disagreement first: it replays the frozen-Torch guidance oracle
//! (`docs/migration/sa3-sampler-reference/guidance.safetensors`) on this exact checkpoint, device
//! and dtype, records the largest per-element gap against the reference, and then requires the
//! negative-prompt divergence — measured in the same latent space, from the same initial noise, at
//! the same single Euler step — to exceed it.
//!
//! Be honest about the strength of that comparison. The floor is an *agreement* quantity (order
//! `1e-3`); the divergence is a *signal* quantity (order `1e2`). The only alternative it
//! distinguishes is a divergence of exactly zero — negative threading deleted outright. A
//! mis-wired-but-non-zero negative branch — conditional and unconditional halves swapped, the
//! guidance weight applied as `g` where it should be `g - 1`, the negative attention mask dropped —
//! diverges just as loudly and would sail through it. No frozen-Torch artifact covers the explicit
//! negative path (`guidance.safetensors` carries only `None`-negative renders), so parity cannot
//! close that hole either.
//!
//! What closes it is the recomposition identity in
//! [`the_guided_latents_are_exactly_the_cfg_recomposition_of_their_own_two_branches`]: with
//! `apg_scale = 0`, `cfg_norm_threshold = 0` and `scale_phi = 0`, `guided_prediction` reduces to
//! `uncond + g * (cond - uncond)` in v-space, and a single Euler step is affine in the model
//! output, so the guided latents must equal `L(N) + g * (L(P) - L(N))` where `L(P)` and `L(N)` are
//! the batch-1 `cfg_scale = 1.0` steps on the positive and the *masked* negative embeddings. That
//! prediction is compared against three named mis-wirings of the same two branches, so it fails on
//! each of them.

use std::path::{Path, PathBuf};

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device, Tensor, D};
use candle_audio_stable_audio_3::dit::{Guidance, StableAudio3Dit};
use candle_audio_stable_audio_3::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::sampler::{padding_mask, sample_dit, InjectedNoise, SamplerKind};
use candle_audio_stable_audio_3::t5gemma::T5GemmaConditioner;
use candle_audio_stable_audio_3::weights::SnapshotLayout;
use candle_audio_stable_audio_3::Variant;
use candle_nn::VarBuilder;

/// The oracle prefix in `guidance.safetensors`, and the snapshot env var, for the one checkpoint the
/// frozen guidance artifact covers on the base side.
const GUIDANCE_ORACLE_PREFIX: &str = "small-music-base";
const GUIDANCE_ORACLE_ENV: &str = "SA3_SMALL_MUSIC_BASE_SNAPSHOT";

/// The guidance scale every case in `guidance.safetensors` was frozen at.
const GUIDANCE_ORACLE_SCALE: f64 = 2.5;

/// How much a CFG recombination amplifies an independent per-branch error.
///
/// `uncond + g·(cond − uncond) = g·cond − (g − 1)·uncond`, so an error of `ε` in each of the two
/// branch predictions moves the guided result by up to `(2g − 1)·ε`. Both quantities this file
/// compares are guided-space disagreements carrying that factor — the frozen-Torch floor at
/// [`GUIDANCE_ORACLE_SCALE`], the recomposition residual at the base default — so converting one
/// into a bound on the other means rescaling by the ratio of their factors.
///
/// Be exact about what that yields: a principled cross-scale bound calibrated on three backends —
/// Metal, CUDA and CPU — not a derivation. Only the first two are still enforced: since sc-14546
/// [`device`] honours `SA3_TEST_CUDA`, so `sa3-base-identity-{metal,cuda}` pin the Metal and CUDA
/// points and no lane measures CPU any more. The CPU point is why the rescaling exists (the
/// unscaled floor failed there), and the CUDA residual `3.524780e-3` likewise exceeds its own
/// unscaled floor `3.471375e-3`, so the unscaled bound would have failed on both non-Metal
/// backends. Two caveats say why it is not stronger than that. First, the two quantities do not
/// measure the same error — the floor is this implementation's disagreement with frozen Torch, the
/// residual is batch-2-versus-batch-1 disagreement inside this implementation — so they are related
/// by the shared `2g − 1` recombination by analogy, not by identity. Second,
/// [`frozen_torch_agreement_floor`] is a `max` over three cases, two of which (`apg_scale = 1.0`
/// and the blended rescale) do not recombine as `2g − 1` at all, so attributing factor 4 to the
/// floor is loose in an unspecified direction. What *is* asserted is only that the scaling is
/// fixed, derived from the two guidance scales, and device-portable — not tuned until the numbers
/// passed.
fn cfg_error_amplification(guidance: f64) -> f64 {
    2.0 * guidance - 1.0
}

/// The crate's three-way real-weight device selector, identical to `same_oracle.rs` and
/// `chunked_oracle.rs`.
///
/// Honouring `SA3_TEST_METAL` alone would silently run every gate in this file — and the
/// frozen-Torch agreement floor they derive, which is only meaningful as a statement about *this*
/// device and dtype — on `Device::Cpu` inside `sa3-base-identity-cuda`, which sets `SA3_TEST_CUDA`.
/// A requested backend that is unavailable is a hard failure, never a fallback.
fn device() -> Device {
    if std::env::var_os("SA3_TEST_METAL").is_some() {
        Device::new_metal(0).expect("SA3_TEST_METAL requested but Metal is unavailable")
    } else if std::env::var_os("SA3_TEST_CUDA").is_some() {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).expect("SA3_TEST_CUDA requested but CUDA is unavailable")
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("SA3_TEST_CUDA requires --features cuda")
        }
    } else {
        Device::Cpu
    }
}

fn snapshot(env: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    )
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
    values(&left.broadcast_sub(right).unwrap().abs().unwrap())
        .into_iter()
        .fold(0.0, f32::max)
}

fn cosine(actual: &Tensor, expected: &Tensor) -> f64 {
    let actual = values(actual);
    let expected = values(expected);
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in actual.iter().zip(&expected) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
    }
    dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

/// The two shipped reference artifacts this file reads.
fn reference(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/migration")
        .join(relative)
}

/// A single Euler step from the frozen oracle's own initial noise, at `guidance`.
///
/// Every fixed input — the schedule, the `seconds_total` scalar, the local conditioning, the
/// padding mask — matches `tests/sampler_oracle.rs`'s frozen guidance replay exactly, so a latent
/// measured here is directly comparable to the agreement bound measured there.
#[allow(clippy::too_many_arguments)]
fn one_euler_step(
    model: &StableAudio3Dit,
    initial: &Tensor,
    positive: &Tensor,
    negative: Option<&Tensor>,
    negative_mask: Option<&Tensor>,
    seconds: &Tensor,
    local: &Tensor,
    mask: &Tensor,
    guidance: Guidance,
) -> Tensor {
    let mut noise = InjectedNoise::new(Vec::new());
    let output = sample_dit(
        model,
        SamplerKind::Euler,
        initial,
        &candle_audio_stable_audio_3::sampler::Schedule::Shared(vec![0.5, 0.0]),
        positive,
        negative,
        negative_mask,
        seconds,
        local,
        Some(mask),
        guidance,
        &mut noise,
        false,
    )
    .expect("single Euler step");
    assert_eq!(output.model_calls, 1);
    assert_eq!(
        output.noise_draws, 0,
        "Euler must not draw the per-step random tensors Pingpong needs"
    );
    output.latents
}

/// This implementation's own largest per-element disagreement with the frozen-Torch guidance oracle
/// on this checkpoint, device and dtype, across all three frozen guidance variants.
///
/// Everything is fixed to the artifact's own frame — the oracle's initial noise, its projected
/// prompt, the same schedule, the same single Euler step — so the returned number is a pure
/// numerics quantity in the same latent space as everything else this file measures.
#[allow(clippy::too_many_arguments)]
fn frozen_torch_agreement_floor(
    model: &StableAudio3Dit,
    device: &Device,
    initial: &Tensor,
    oracle_prompt: &Tensor,
    seconds: &Tensor,
    local: &Tensor,
    mask: &Tensor,
) -> f32 {
    let guidance_oracle = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[reference("sa3-sampler-reference/guidance.safetensors")],
            DType::F32,
            device,
        )
        .unwrap()
    };
    let mut floor = 0.0f32;
    for (name, guidance) in [
        (
            "vanilla",
            Guidance {
                cfg_scale: GUIDANCE_ORACLE_SCALE,
                apg_scale: 0.0,
                ..Guidance::default()
            },
        ),
        (
            "apg",
            Guidance {
                cfg_scale: GUIDANCE_ORACLE_SCALE,
                apg_scale: 1.0,
                ..Guidance::default()
            },
        ),
        (
            "blended_rescaled",
            Guidance {
                cfg_scale: GUIDANCE_ORACLE_SCALE,
                apg_scale: 0.5,
                cfg_norm_threshold: 0.25,
                scale_phi: 0.3,
            },
        ),
    ] {
        let actual = one_euler_step(
            model,
            initial,
            oracle_prompt,
            None,
            None,
            seconds,
            local,
            mask,
            guidance,
        );
        let expected = guidance_oracle
            .get_unchecked(&format!("{GUIDANCE_ORACLE_PREFIX}.{name}.final"))
            .unwrap();
        let cosine = cosine(&actual, &expected);
        let gap = max_abs_diff(&actual, &expected);
        eprintln!("{GUIDANCE_ORACLE_PREFIX}.{name}: cosine={cosine:.9} max_abs={gap:.9}");
        assert!(
            cosine >= 0.999,
            "{GUIDANCE_ORACLE_PREFIX}.{name} lost frozen-Torch parity: cosine={cosine}"
        );
        floor = floor.max(gap);
    }
    assert!(
        floor > 0.0,
        "a zero agreement gap means the oracle comparison did not run"
    );
    floor
}

/// The whole gate, in one model load.
///
/// 1. Replay the three frozen guidance variants at `cfg_scale = 2.5` and record the largest
///    per-element disagreement with `guidance.safetensors`. That is the noise floor.
/// 2. At `cfg_scale = 1.0`, run with and without a negative prompt and require the latents to be
///    **bit-identical**. This is the honest control: the DiT's batch-1 shortcut means a negative
///    prompt cannot matter here, and a test that "proved" otherwise would be proving a bug.
/// 3. At `cfg_scale = 7.0` — the base default — run the same pair and require the divergence to
///    exceed the floor from step 1.
///
/// What this discriminates, precisely: delete the negative-prompt threading and step 3 measures
/// exactly 0.0; invert the batch-1 shortcut and step 2 fails. What it does **not** discriminate is a
/// negative branch that is threaded but wired wrongly — the divergence is a signal-scale quantity
/// and the floor is a noise-scale one, so any non-zero mis-wiring clears it just as easily as the
/// correct one. That case is
/// [`the_guided_latents_are_exactly_the_cfg_recomposition_of_their_own_two_branches`]'s job.
#[test]
#[ignore = "requires the pinned small-music-base snapshot and real batch-2 CFG forwards"]
fn negative_prompt_divergence_at_the_base_default_exceeds_the_frozen_torch_agreement_floor() {
    let device = device();
    let layout = SnapshotLayout::from_dir(&snapshot(GUIDANCE_ORACLE_ENV)).unwrap();
    let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();

    let p0 = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[reference(
                "sa3-reference/small-music-base-reference.safetensors",
            )],
            DType::F32,
            &device,
        )
        .unwrap()
    };
    let initial = p0.get_unchecked("sampler_initial_noise").unwrap();
    let oracle_prompt = p0.get_unchecked("t5_projected_padded").unwrap();
    let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
    let local = Tensor::zeros((1, 257, 16), DType::F32, &device).unwrap();
    let mask = padding_mask(&[12], 16, &device).unwrap();

    // ---- step 1: the frozen-Torch agreement floor, on this checkpoint/device/dtype ----
    let floor = frozen_torch_agreement_floor(
        &model,
        &device,
        &initial,
        &oracle_prompt,
        &seconds,
        &local,
        &mask,
    );

    // ---- steps 2 and 3: real text, real negative prompt ----
    let conditioner = T5GemmaConditioner::from_layout(&layout, &device).unwrap();
    let positive = conditioner
        .encode(&["A beautiful piano arpeggio grows into a grand cinematic climax".to_owned()])
        .unwrap();
    let negative = conditioner
        .encode(&["harsh digital clipping, distorted noise, speech".to_owned()])
        .unwrap();

    let render = |cfg_scale: f64, with_negative: bool| {
        one_euler_step(
            &model,
            &initial,
            &positive.embeddings,
            with_negative.then_some(&negative.embeddings),
            with_negative.then_some(&negative.attention_mask),
            &seconds,
            &local,
            &mask,
            Guidance {
                cfg_scale,
                apg_scale: 0.0,
                ..Guidance::default()
            },
        )
    };

    // Step 2: at guidance 1.0 the negative prompt must be a *no-op*, bit for bit.
    let unguided_without = render(1.0, false);
    let unguided_with = render(1.0, true);
    assert_eq!(
        values(&unguided_without),
        values(&unguided_with),
        "at cfg_scale = 1.0 the guided forward takes its batch-1 branch, so a negative prompt \
         cannot change a single element; if this ever differs the shortcut has been broken"
    );

    // Step 3: at the base default it must change the latents by more than the numerics can explain.
    let guided_without = render(Variant::SmallMusicBase.default_guidance(), false);
    let guided_with = render(Variant::SmallMusicBase.default_guidance(), true);
    let divergence = max_abs_diff(&guided_without, &guided_with);
    let margin = divergence / floor;
    eprintln!(
        "negative-prompt divergence at cfg_scale={}: max_abs={divergence:.9} \
         frozen_torch_agreement_floor={floor:.9} margin={margin:.2}x",
        Variant::SmallMusicBase.default_guidance()
    );
    assert!(
        divergence > floor,
        "a negative prompt at the base default changed the latents by {divergence}, which is inside \
         this implementation's own {floor} disagreement with frozen Torch — that is not evidence \
         the negative conditioning is wired at all"
    );
    // No second, larger multiple of the floor is asserted. `margin` is reported because it is worth
    // reading — but a "must be at least Nx" bound would be a number chosen by taste, and it would
    // buy nothing: the two quantities are orders of magnitude apart, so every failure mode this
    // comparison can see is already caught by `divergence > floor`. The strength that a bigger
    // multiple only *looks* like it adds comes from the recomposition identity below instead.

    // ...and the guided renders must differ from the unguided ones, or the guidance scale itself is
    // not reaching the model.
    assert!(max_abs_diff(&guided_without, &unguided_without) > floor);
}

/// The gate that a mis-wired-but-non-zero negative branch cannot pass.
///
/// With `apg_scale = 0`, `cfg_norm_threshold = 0` and `scale_phi = 0`, `dit::guided_prediction`
/// reduces algebraically to `uncond + g * (cond - uncond)` in v-space — the denoised-space detour it
/// takes cancels exactly — and a single Euler step is affine in the model output. So the guided
/// latents are pinned to an identity, not merely to an inequality:
///
/// ```text
/// L_guided(P, N, g) == L(N) + g * (L(P) - L(N))
/// ```
///
/// where `L(X)` is the same single Euler step run at `cfg_scale = 1.0` — the batch-1 branch — on
/// prompt `X`, and `N` is the negative embedding **after** the attention-mask multiply that
/// `forward_guided_impl` applies before batching. The only difference between the two sides is
/// batch-2 versus batch-1 numerics.
///
/// The identity is checked against three named mis-wirings of the very same two branches, each of
/// which diverges from the no-negative render just as loudly as the correct one and therefore passes
/// `negative_prompt_divergence_at_the_base_default_exceeds_the_frozen_torch_agreement_floor`:
///
/// - **swapped halves** — `narrow(0, 0, batch)` and `narrow(0, batch, batch)` exchanged, i.e.
///   `L(P) + g * (L(N) - L(P))`;
/// - **off-by-one weight** — `(cfg_scale - 1.0)` written as `cfg_scale`, i.e.
///   `L(N) + (g - 1) * (L(P) - L(N))`;
/// - **dropped negative mask** — the raw negative embedding used instead of the masked one.
///
/// The bound is the same frozen-Torch agreement floor the divergence gate derives, measured in the
/// same latent space on the same device and dtype, read at this test's guidance scale through
/// [`cfg_error_amplification`]: the correct recomposition must land inside it and every mis-wiring
/// must land outside it. The rescaling is not slack bought to make the gate pass — the closest
/// mis-wiring still overshoots the bound by more than 6,690x — it is what makes the bound
/// device-portable, and it was added because the unscaled floor failed on CPU (residual `5.05e-3`,
/// floor `4.03e-3`) while passing comfortably on Metal.
#[test]
#[ignore = "requires the pinned small-music-base snapshot and real batch-2 CFG forwards"]
fn the_guided_latents_are_exactly_the_cfg_recomposition_of_their_own_two_branches() {
    let device = device();
    let layout = SnapshotLayout::from_dir(&snapshot(GUIDANCE_ORACLE_ENV)).unwrap();
    let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();

    let p0 = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[reference(
                "sa3-reference/small-music-base-reference.safetensors",
            )],
            DType::F32,
            &device,
        )
        .unwrap()
    };
    let initial = p0.get_unchecked("sampler_initial_noise").unwrap();
    let oracle_prompt = p0.get_unchecked("t5_projected_padded").unwrap();
    let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
    let local = Tensor::zeros((1, 257, 16), DType::F32, &device).unwrap();
    let mask = padding_mask(&[12], 16, &device).unwrap();

    let step = |prompt: &Tensor, negative: Option<(&Tensor, &Tensor)>, cfg_scale: f64| {
        one_euler_step(
            &model,
            &initial,
            prompt,
            negative.map(|(embeddings, _)| embeddings),
            negative.map(|(_, mask)| mask),
            &seconds,
            &local,
            &mask,
            Guidance {
                cfg_scale,
                apg_scale: 0.0,
                ..Guidance::default()
            },
        )
    };

    // The same agreement floor the divergence gate derives, from the same helper.
    let floor = frozen_torch_agreement_floor(
        &model,
        &device,
        &initial,
        &oracle_prompt,
        &seconds,
        &local,
        &mask,
    );

    let conditioner = T5GemmaConditioner::from_layout(&layout, &device).unwrap();
    let positive = conditioner
        .encode(&["A beautiful piano arpeggio grows into a grand cinematic climax".to_owned()])
        .unwrap();
    let negative = conditioner
        .encode(&["harsh digital clipping, distorted noise, speech".to_owned()])
        .unwrap();
    // Exactly what `forward_guided_impl` builds before it concatenates the batch.
    let masked_negative = negative
        .embeddings
        .broadcast_mul(
            &negative
                .attention_mask
                .unsqueeze(D::Minus1)
                .unwrap()
                .to_dtype(negative.embeddings.dtype())
                .unwrap(),
        )
        .unwrap();
    assert!(
        max_abs_diff(&masked_negative, &negative.embeddings) > 0.0,
        "the encoded negative prompt is fully valid, so the mask multiply is a no-op and the \
         dropped-mask mis-wiring below would be indistinguishable from the correct wiring"
    );

    let guidance = Variant::SmallMusicBase.default_guidance();
    let guided = step(
        &positive.embeddings,
        Some((&negative.embeddings, &negative.attention_mask)),
        guidance,
    );
    let branch_positive = step(&positive.embeddings, None, 1.0);
    let branch_negative = step(&masked_negative, None, 1.0);
    let branch_negative_unmasked = step(&negative.embeddings, None, 1.0);

    // `low + weight * (high - low)`, in f32 on the test device.
    let recompose = |low: &Tensor, high: &Tensor, weight: f64| {
        (low + ((high - low).unwrap() * weight).unwrap()).unwrap()
    };

    let correct = recompose(&branch_negative, &branch_positive, guidance);
    let residual = max_abs_diff(&guided, &correct);
    let mis_wirings = [
        (
            "swapped conditional/unconditional halves",
            recompose(&branch_positive, &branch_negative, guidance),
        ),
        (
            "guidance weight applied as g-1 instead of g",
            recompose(&branch_negative, &branch_positive, guidance - 1.0),
        ),
        (
            "negative attention mask dropped",
            recompose(&branch_negative_unmasked, &branch_positive, guidance),
        ),
    ];

    // The floor is a guided-space disagreement measured at the oracle's own guidance scale; this
    // residual is one measured at the base default. Both carry the CFG amplification factor of
    // their own scale, so the floor is read at this test's scale before it is used as a bound. On
    // Metal that is 3.685e-3 -> 1.198e-2; on CPU 4.028e-3 -> 1.309e-2. The correct recomposition
    // measures 6.1e-5 (Metal) / 5.1e-3 (CPU) and the closest mis-wiring measures 87.75, so the
    // rescaling is what makes the bound device-portable, not what makes the gate pass.
    let bound = (floor as f64 * cfg_error_amplification(guidance)
        / cfg_error_amplification(GUIDANCE_ORACLE_SCALE)) as f32;
    eprintln!(
        "cfg recomposition at g={guidance}: residual={residual:.9} \
         frozen_torch_agreement_floor={floor:.9} rescaled_bound={bound:.9}"
    );
    for (name, prediction) in &mis_wirings {
        eprintln!(
            "  mis-wiring `{name}`: residual={:.9}",
            max_abs_diff(&guided, prediction)
        );
    }

    assert!(
        residual <= bound,
        "the guided latents are {residual} away from `L(N) + {guidance} * (L(P) - L(N))`, which is \
         outside the {bound} this implementation's own {floor} disagreement with frozen Torch \
         explains at this guidance scale — the batch-2 CFG path is not recomposing the two branches \
         it actually evaluated"
    );
    for (name, prediction) in &mis_wirings {
        let mis_wired = max_abs_diff(&guided, prediction);
        assert!(
            mis_wired > bound,
            "the `{name}` mis-wiring predicts the guided latents to within {mis_wired}, inside the \
             {bound} numerics bound — this gate cannot tell it apart from the correct wiring"
        );
    }
}

/// The same two facts, end to end through the registered provider and on decoded PCM.
///
/// The test above works in latent space with hand-encoded conditioning, which proves the sampler and
/// DiT are wired but says nothing about `GenerationRequest::negative_prompt` reaching them. This one
/// goes through `provider_registry().load(...)` and compares `Vec<f32>` PCM, so it covers the whole
/// request path — validation, `synthesis_parameters`, the conditioner call in `pipeline::sample`,
/// and the decode.
///
/// Cheap on purpose: 0.25 s at two explicit steps. The expensive default-operating-point render is
/// `tests/provider.rs`'s job.
#[test]
#[ignore = "requires the pinned small-music-base snapshot"]
fn the_request_negative_prompt_is_inert_at_guidance_one_and_material_at_the_base_default() {
    let generator = candle_audio_stable_audio_3::provider_registry()
        .expect("provider registry")
        .load(
            Variant::SmallMusicBase.model_id(),
            &LoadSpec::new(WeightsSource::Dir(snapshot(GUIDANCE_ORACLE_ENV))),
        )
        .expect("strict registered variant-bound load");

    let render = |guidance: f32, negative: Option<&str>| {
        let request = GenerationRequest {
            prompt: "A beautiful piano arpeggio grows into a grand cinematic climax".into(),
            negative_prompt: negative.map(str::to_owned),
            seed: Some(42),
            steps: Some(2),
            sampler: Some("euler".into()),
            guidance: Some(guidance),
            audio: Some(AudioParams {
                target_duration: Some(0.25),
                sample_rate: Some(44_100),
                ..Default::default()
            }),
            ..Default::default()
        };
        match generator
            .generate(&request, &mut |_| {})
            .expect("generation")
        {
            GenerationOutput::Audio(track) => track.samples,
            other => panic!("expected audio, got {other:?}"),
        }
    };

    let negative = "harsh digital clipping, distorted noise, speech";
    let unguided_without = render(1.0, None);
    let unguided_with = render(1.0, Some(negative));
    assert_eq!(
        unguided_without, unguided_with,
        "at guidance 1.0 — the post-trained default — a negative prompt must not change one PCM \
         sample; the DiT never evaluates the negative branch"
    );

    let default_guidance = Variant::SmallMusicBase.default_guidance() as f32;
    let guided_without = render(default_guidance, None);
    let guided_with = render(default_guidance, Some(negative));
    assert_ne!(
        guided_without, guided_with,
        "at the base default guidance ({default_guidance}) the negative prompt must reach the model"
    );
    let peak = guided_without
        .iter()
        .zip(&guided_with)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("provider-level negative-prompt PCM divergence at guidance {default_guidance}: max_abs={peak:.9}");

    // The discriminating control the byte-identity assertion needs: this generator is not simply
    // returning the same buffer for every request.
    assert_ne!(unguided_without, guided_without);
}

/// Where `50` and `7.0` come from, read out of the pinned snapshots themselves.
///
/// The base defaults are a **product** choice — upstream's Python and CLI entry points default to
/// 8 steps at guidance 1 for every checkpoint, and only Stability's Gradio app varies them per
/// model — but they are not invented here. Each `model_config.json` carries a `training.demo` block
/// that records the operating point the checkpoint was demoed at, and the two halves of the family
/// disagree there exactly the way the shipped defaults do:
///
/// | | `demo_steps` | `demo_cfg_scales` |
/// |---|---|---|
/// | post-trained | 8 | `[1]` |
/// | `-base` | 50 | `[2, 4, 7]` |
///
/// `training.*` is never read at inference — the crate's config parser ignores it — so this test
/// reads the raw JSON. It is what stops the two constants from being unattributable numbers.
#[test]
#[ignore = "requires all six pinned snapshots"]
fn the_shipped_configs_record_the_operating_point_each_half_of_the_family_defaults_to() {
    let envs = [
        (Variant::SmallMusic, "SA3_SMALL_MUSIC_SNAPSHOT"),
        (Variant::SmallSfx, "SA3_SMALL_SFX_SNAPSHOT"),
        (Variant::Medium, "SA3_MEDIUM_SNAPSHOT"),
        (Variant::SmallMusicBase, "SA3_SMALL_MUSIC_BASE_SNAPSHOT"),
        (Variant::SmallSfxBase, "SA3_SMALL_SFX_BASE_SNAPSHOT"),
        (Variant::MediumBase, "SA3_MEDIUM_BASE_SNAPSHOT"),
    ];
    for (variant, env) in envs {
        let path = snapshot(env).join("model_config.json");
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let demo = &config["training"]["demo"];
        let steps = demo["demo_steps"].as_u64().unwrap_or_else(|| {
            panic!(
                "{}: no training.demo.demo_steps in {}",
                variant.model_id(),
                path.display()
            )
        }) as usize;
        let scales = demo["demo_cfg_scales"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: no training.demo.demo_cfg_scales", variant.model_id()))
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let largest = scales.iter().copied().fold(f64::MIN, f64::max);
        eprintln!(
            "{}: demo_steps={steps} demo_cfg_scales={scales:?} -> shipped defaults {} / {}",
            variant.model_id(),
            variant.default_steps(),
            variant.default_guidance()
        );
        assert_eq!(
            steps,
            variant.default_steps(),
            "{}: the shipped default step count must be the one the checkpoint's own config \
             records; if Stability re-publishes a different `demo_steps` this is where it surfaces",
            variant.model_id()
        );
        assert_eq!(
            largest,
            variant.default_guidance(),
            "{}: the shipped default guidance must be the largest `demo_cfg_scales` entry",
            variant.model_id()
        );
    }
}
