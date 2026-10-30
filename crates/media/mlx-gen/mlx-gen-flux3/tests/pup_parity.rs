//! Prompt-upsampler parity on the committed miniature Qwen3.5 VLM (sc-26238): the four modes
//! through the real MLX engine against the oracle's greedy run (`tools/dump_flux3_pup_golden.py`
//! — a seeded `Qwen3_5ForConditionalGeneration` with the pinned chat template), the generator's
//! pre-generation hook end to end (bypass identity, provenance, progress ordering, the typed
//! refusal without the component), and the package-level refusals.
//!
//! The greedy ids are compared where the oracle's top-1/top-2 logit margin is wide enough that
//! a dtype-matched MLX-vs-torch rounding difference cannot flip the argmax; a mismatch on a narrow step
//! ends the comparison (the trajectories legitimately diverge from there), and at least
//! `MIN_MATCHED` steps must have matched. An earlier ambiguous text step is accepted only after
//! its complete 12-step teacher-forced certificate passes at least `MIN_MATCHED` wide exact checks.
//! Candidate validation retains the same margin; a tie is never pinned to a CPU-specific id.

use std::sync::{Arc, Mutex};

use gen_core_flux3::{
    pup,
    pup::{
        plan_text, PromptUpsampler, UpsampleMode, UpsampleOptions, UpsamplePlan,
        UPSAMPLER_COMPONENT,
    },
};
use mlx_gen::gen_core::core_llm::{Channel, StreamEvent};
use mlx_gen::gen_core::{
    core_llm, Error as CoreError, Progress, PromptEnhancementOutcome, PromptEnhancementReport,
    PromptEnhancementSink,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, WeightsSource,
};
use mlx_gen_flux3::Flux3PromptUpsampler;

use crate::common::{fixture, fixtures, host_f32, host_i32, tiny_package};

/// The oracle's margin above which the argmax is held bit-exact (logit units; the tiny model's
/// logits are O(1)).
const MARGIN: f32 = 0.02;
/// Steps that must have matched before a narrow-margin divergence is accepted.
const MIN_MATCHED: usize = 3;

fn tiny_snapshot() -> std::path::PathBuf {
    fixtures().join("tiny-pup-snapshot")
}

fn golden() -> Weights {
    fixture("flux3_pup_mlx_bf16_golden.safetensors")
}

fn load_upsampler() -> Flux3PromptUpsampler {
    Flux3PromptUpsampler::load(&tiny_snapshot()).expect("the miniature upsampler loads")
}

fn meta(g: &Weights, key: &str) -> String {
    g.metadata(key)
        .unwrap_or_else(|| panic!("golden metadata {key}"))
        .to_string()
}

/// The oracle's reference pixels of one case, in attachment order.
fn golden_references(g: &Weights, case: &str) -> Vec<core_llm::ImageRef> {
    let n: usize = meta(g, &format!("n_refs_{case}")).parse().unwrap();
    (0..n)
        .map(|i| {
            let t = g.get(&format!("ref_{case}_{i}")).expect("golden reference");
            let shape = t.shape();
            let (h, w) = (shape[0] as u32, shape[1] as u32);
            let pixels: Vec<u8> = host_i32(t).into_iter().map(|v| v as u8).collect();
            core_llm::ImageRef::new(w, h, pixels).unwrap()
        })
        .collect()
}

/// The inputs of one oracle case (`plan_cases` of the dump script).
fn case_inputs(case: &str) -> (&'static str, Option<Vec<pup::BoxSpec>>) {
    const BALLOONS: &str = r#"[{"id": "obj_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]}, {"id": "obj_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]}]"#;
    const CACTUS: &str = r#"[{"id": "obj_1", "desc": "a small potted cactus", "from": "target", "src_bbox": null, "tgt_bbox": [0.55, 0.62, 0.85, 0.85]}]"#;
    match case {
        "t2i" => ("a red bicycle leaning against a brick wall", None),
        "t2i_bbox" => (
            " two hot-air balloons over the sea\n",
            Some(pup::BoxSpec::parse_list(BALLOONS).unwrap()),
        ),
        "i2i" => ("put a red bandana around the dog's neck", None),
        "i2i_bbox" => (
            "add a small potted cactus on the table",
            Some(pup::BoxSpec::parse_list(CACTUS).unwrap()),
        ),
        other => panic!("unknown case {other}"),
    }
}

fn plan_for(up: &Flux3PromptUpsampler, g: &Weights, case: &str) -> UpsamplePlan {
    let (prompt, boxes) = case_inputs(case);
    let references = golden_references(g, case);
    let (mode, user_text, locked) = plan_text(prompt, boxes.as_deref(), references.len()).unwrap();
    assert_eq!(
        mode.system_key(),
        meta(g, &format!("system_key_{case}")),
        "{case}: mode"
    );
    let user_text = if mode.uses_references() {
        pup::i2i_scaffold(&user_text, locked.is_some())
    } else {
        user_text
    };
    let mut plan = UpsamplePlan {
        mode,
        original_prompt: prompt.to_string(),
        system_prompt: up.package().prompts().get(mode).to_string(),
        user_text,
        locked,
        references,
        max_new_tokens: 0,
    };
    let steps: u32 = meta(g, "greedy_steps").parse().unwrap();
    plan.max_new_tokens = up.package().budget(&plan, Some(steps)).unwrap();
    assert_eq!(plan.max_new_tokens, steps);
    plan
}

/// Stream the plan through the engine, returning the generated ids and the prompt length.
fn run_engine(
    up: &Flux3PromptUpsampler,
    plan: &UpsamplePlan,
) -> (Vec<i32>, core_llm::TextLlmOutput) {
    let request = plan.request(&UpsampleOptions::default(), core_llm::CancelFlag::new());
    let mut ids = Vec::new();
    let out = up
        .engine()
        .generate(&request, &mut |event| {
            if let StreamEvent::Token {
                id,
                channel: Channel::Content,
                ..
            } = event
            {
                ids.push(id as i32);
            }
        })
        .expect("the miniature engine generates");
    (ids, out)
}

/// Compare the public Content projection, not raw HF generation ids. The independent raw CPU
/// decoder gate is candle-llm/tests/flux3_pup_decoder.rs; never delete protocol ids from its oracle.
fn assert_greedy_parity(
    up: &Flux3PromptUpsampler,
    g: &Weights,
    case: &str,
    got: &[i32],
    raw: &[i32],
    margins: &[f32],
    output_and_certificate: (&core_llm::TextLlmOutput, Option<usize>),
) {
    let (output, teacher_forced_wide) = output_and_certificate;
    assert_eq!(raw.len(), margins.len());
    let tokenizer = up.package().tokenizer();
    let special = tokenizer.constraint_decode_table().special;
    let stops: Vec<u32> = pup::STOP_TOKENS
        .iter()
        .map(|token| {
            let ids = tokenizer.encode(token, false).unwrap();
            assert_eq!(ids.len(), 1, "the frozen EOS marker is one tokenizer id");
            ids[0]
        })
        .collect();
    let raw_count = raw
        .iter()
        .position(|&id| stops.contains(&(id as u32)))
        .unwrap_or(raw.len());
    let visible: Vec<_> = raw[..raw_count]
        .iter()
        .zip(&margins[..raw_count])
        .enumerate()
        .filter(|(_, (id, _))| !special.contains(&(**id as u32)))
        .map(|(index, (&id, &margin))| (index, id, margin))
        .collect();
    let mut matched = 0usize;
    for (index, &(raw_index, expected, margin)) in visible.iter().enumerate() {
        let actual = *got.get(index).unwrap_or_else(|| {
            panic!("{case}: public stream omitted visible token {index} (raw step {raw_index})")
        });
        if margin == 0.0 {
            let scores = g.get(&format!("step_scores_{case}")).unwrap();
            let vocab = scores.shape()[1] as usize;
            let rows = host_f32(scores);
            gen_core_testkit::assert_exact_greedy_tie(
                &rows[raw_index * vocab..(raw_index + 1) * vocab],
                actual as usize,
            );
            assert!(
                matched >= MIN_MATCHED
                    || teacher_forced_wide.is_some_and(|wide| wide >= MIN_MATCHED),
                "{case}: exact tie requires sufficient prefix or complete teacher-forced coverage"
            );
            eprintln!("{case}: validated exact tie at raw step {raw_index}; teacher-forced gate owns suffix");
            return;
        }
        if actual == expected {
            matched += 1;
            continue;
        }
        assert!(margin < MARGIN,
            "{case}: visible step {index}, raw step {raw_index} differs ({actual} vs {expected}) on wide margin {margin:.4}");
        let scores = g.get(&format!("step_scores_{case}")).unwrap();
        let vocab = scores.shape()[1] as usize;
        let rows = host_f32(scores);
        let row = &rows[raw_index * vocab..(raw_index + 1) * vocab];
        assert!(
            row[expected as usize] - row[actual as usize] <= MARGIN,
            "{case}: public candidate outside the existing narrow margin"
        );
        assert!(
            matched >= MIN_MATCHED || teacher_forced_wide.is_some_and(|wide| wide >= MIN_MATCHED),
            "{case}: only {matched} prefix matches and no sufficient teacher-forced certificate"
        );
        eprintln!(
            "{case}: narrow-margin divergence at raw step {raw_index} after {matched} matches"
        );
        return; // Existing rounding policy: later trajectories need not have the same EOS/tail.
    }
    assert!(matched >= MIN_MATCHED);
    assert_eq!(got.len(), visible.len(), "{case}: exact public token count");
    assert_eq!(
        output.usage.generated_tokens as usize, raw_count,
        "{case}: suppressed protocol ids still count toward the raw decode budget; EOS does not"
    );
    let reason = if raw_count < raw.len() {
        core_llm::FinishReason::Stop
    } else {
        core_llm::FinishReason::Length
    };
    assert_eq!(
        output.finish_reason,
        Some(reason),
        "{case}: source EOS/budget boundary"
    );
}

#[test]
fn the_golden_was_dumped_from_the_pinned_upsampler() {
    let g = golden();
    assert_eq!(meta(&g, "pup_revision"), pup::UPSAMPLER_REVISION);
    assert_eq!(meta(&g, "presence_penalty_scope"), "generated_only");
    for (key, expected) in [
        ("oracle_precision", "mlx_mixed_bf16_f32"),
        ("language_compute_dtype", "bfloat16"),
        ("vision_compute_dtype", "float32"),
        ("recurrence_dtype", "float32"),
        ("conv_cache_dtype", "bfloat16"),
        ("kv_cache_dtype", "bfloat16"),
        ("rope_frequency_dtype", "float32"),
        ("ordinary_norm_multiplier", "bf16_load_folded"),
        (
            "cast_guard_negative_controls",
            "whole_model_float32,whole_model_bfloat16,whole_language_module_bfloat16,unfolded_norm",
        ),
    ] {
        assert_eq!(meta(&g, key), expected, "{key}");
    }
    for key in [
        "reference_torch_version",
        "reference_transformers_version",
        "reference_python_version",
    ] {
        assert!(
            !meta(&g, key).is_empty(),
            "recorded reference version: {key}"
        );
    }
    for mode in UpsampleMode::ALL {
        assert_eq!(
            meta(&g, &format!("presence_penalty_{}", mode.system_key()))
                .parse::<f32>()
                .unwrap(),
            mode.presence_penalty()
        );
    }
    assert_eq!(
        meta(&g, "code_commit"),
        gen_core_flux3::UPSTREAM_CODE_REVISION
    );
    // The miniature snapshot is the package layout: every required file, the Qwen3.5 config.
    pup::UpsamplePackage::check(&tiny_snapshot()).unwrap();
    use sha2::Digest;
    for (path, key) in [
        (
            fixtures().join("flux3_pup_golden.safetensors"),
            "f32_oracle_sha256",
        ),
        (
            tiny_snapshot().join("model.safetensors"),
            "checkpoint_sha256",
        ),
    ] {
        assert_eq!(
            format!("{:x}", sha2::Sha256::digest(std::fs::read(path).unwrap())),
            meta(&g, key)
        );
    }
    let tok = std::fs::read(tiny_snapshot().join("tokenizer.json")).unwrap();
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(&tok)),
        meta(&g, "tokenizer_sha256")
    );
}

/// Verify the complete raw trajectory before certifying an ambiguous public-stream prefix.
fn teacher_forced_text_certificate(g: &Weights, case: &str) -> usize {
    use mlx_llm::models::{Qwen35Config, Qwen35Model};
    use mlx_llm::primitives::Weights as LlmWeights;
    use mlx_rs::Array;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tiny_snapshot().join("config.json")).unwrap())
            .unwrap();
    let weights = LlmWeights::from_dir(tiny_snapshot()).unwrap();
    let model = Qwen35Model::from_weights(
        &weights,
        "model.language_model",
        Qwen35Config::from_json(&config).unwrap(),
    )
    .unwrap();
    let prompt = host_i32(g.get(&format!("ids_{case}")).unwrap());
    let expected = host_i32(g.get(&format!("generated_{case}")).unwrap());
    assert_eq!(
        expected.len(),
        12,
        "{case}: complete teacher-forced trajectory"
    );
    let reference = g.get(&format!("step_scores_{case}")).unwrap();
    let vocab = reference.shape()[1] as usize;
    let reference = host_f32(reference);
    assert_eq!(reference.len(), expected.len() * vocab);
    let mut cache = model.new_cache();
    let mut input = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
    let mut offset = 0;
    let mut generated = std::collections::BTreeSet::new();
    let mut wide = 0;
    for (step, &id) in expected.iter().enumerate() {
        let mut actual = host_f32(&model.decode_logits(&input, &mut cache, offset).unwrap());
        assert_eq!(actual.len(), vocab);
        // The independent host projection applies presence once to GENERATED ids only.
        for &token in &generated {
            actual[token as usize] -= pup::PRESENCE_PENALTY_T2I;
        }
        if gen_core_testkit::assert_teacher_forced_logit_step(
            &format!("{case}: teacher-forced raw step {step}"),
            id as usize,
            &reference[step * vocab..(step + 1) * vocab],
            &actual,
            MARGIN,
        ) {
            wide += 1;
        }
        generated.insert(id);
        offset += input.shape()[1];
        input = Array::from_slice(&[id], &[1, 1]);
    }
    assert!(wide >= MIN_MATCHED, "{case}: only {wide} wide exact checks");
    wide
}

#[test]
fn text_modes_match_the_oracle_greedy_run() {
    let up = load_upsampler();
    let g = golden();
    for case in ["t2i", "t2i_bbox"] {
        let plan = plan_for(&up, &g, case);
        assert_eq!(
            up.package().render(&plan).unwrap(),
            meta(&g, &format!("chat_text_{case}")),
            "{case}: chat text"
        );
        let want_ids = host_i32(g.get(&format!("ids_{case}")).unwrap());
        let rendered = up.package().render(&plan).unwrap();
        let actual_ids: Vec<i32> = up
            .package()
            .tokenizer()
            .encode(&rendered, false)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect();
        assert_eq!(
            actual_ids, want_ids,
            "{case}: complete package-tokenized prompt"
        );
        assert_eq!(
            up.package().prompt_tokens(&plan).unwrap(),
            want_ids.len(),
            "{case}: prompt length"
        );
        let certificate = teacher_forced_text_certificate(&g, case);
        let (got, output) = run_engine(&up, &plan);
        assert_eq!(
            output.usage.prompt_tokens as usize,
            want_ids.len(),
            "{case}: the engine tokenized the same prompt length"
        );
        let want = host_i32(g.get(&format!("generated_{case}")).unwrap());
        let margins = host_f32(g.get(&format!("margins_{case}")).unwrap());
        assert_greedy_parity(
            &up,
            &g,
            case,
            &got,
            &want,
            &margins,
            (&output, Some(certificate)),
        );
    }
}

#[test]
fn editing_modes_match_the_oracle_greedy_run_with_the_same_reference_pixels() {
    let up = load_upsampler();
    let g = golden();
    for case in ["i2i", "i2i_bbox"] {
        let plan = plan_for(&up, &g, case);
        assert!(!plan.references.is_empty());
        assert_eq!(
            up.package().render(&plan).unwrap(),
            meta(&g, &format!("chat_text_{case}")),
            "{case}: chat text"
        );
        let want_ids = host_i32(g.get(&format!("ids_{case}")).unwrap());
        // The host-side budget and the engine agree on the expanded prompt length.
        assert_eq!(
            up.package().prompt_tokens(&plan).unwrap(),
            want_ids.len(),
            "{case}: expanded prompt length"
        );
        let (got, output) = run_engine(&up, &plan);
        assert_eq!(
            output.usage.prompt_tokens as usize,
            want_ids.len(),
            "{case}: the engine expanded the image tokens as the processor does"
        );
        let want = host_i32(g.get(&format!("generated_{case}")).unwrap());
        let margins = host_f32(g.get(&format!("margins_{case}")).unwrap());
        assert_greedy_parity(&up, &g, case, &got, &want, &margins, (&output, None));
    }
}

#[test]
fn the_stage_end_to_end_reports_the_effective_caption_and_streams_progress() {
    let up = load_upsampler();
    let g = golden();
    let req = GenerationRequest {
        prompt: "a red bicycle leaning against a brick wall".into(),
        enhance_prompt: true,
        enhance_max_tokens: Some(meta(&g, "greedy_steps").parse().unwrap()),
        ..Default::default()
    };
    let mut events = Vec::new();
    let out = mlx_gen_flux3::pup::effective_prompt(
        Some(&up),
        &req,
        &mlx_gen_flux3::pup::text_input(&req),
        &mut |p| events.push(p),
    )
    .unwrap();
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Enhanced);
    assert_eq!(out.report.mode.as_deref(), Some("t2i"));
    assert_eq!(out.report.original_prompt, req.prompt);
    assert_eq!(out.report.effective_prompt, out.caption);
    assert!(!out.caption.is_empty());
    assert!(matches!(
        events.first(),
        Some(Progress::Upsampling { tokens: 0, .. })
    ));
    assert!(events.len() >= 2, "{events:?}");
    assert!(events
        .iter()
        .all(|e| matches!(e, Progress::Upsampling { .. })));
}

#[test]
fn the_generator_runs_the_stage_before_encoding_and_refuses_without_the_component() {
    // Without the component: `enhance_prompt` is a typed refusal, a plain request still renders.
    let base = LoadSpec::new(WeightsSource::Dir(tiny_package()))
        .with_offload_policy(OffloadPolicy::Resident);
    let plain = mlx_gen_flux3::provider_registry()
        .unwrap()
        .load("flux3_klein", &base)
        .unwrap();
    let mut req = GenerationRequest {
        prompt: "a red fox in the snow".into(),
        width: 80,
        height: 64,
        steps: Some(1),
        seed: Some(3),
        enhance_prompt: true,
        enhance_max_tokens: Some(6),
        ..Default::default()
    };
    let err = plain.generate(&req, &mut |_| {}).unwrap_err();
    assert!(
        matches!(&err, CoreError::Unsupported(m) if m.contains(UPSAMPLER_COMPONENT)),
        "{err}"
    );

    // With the component: the stage runs first, its report is emitted, then the usual steps.
    let mut spec = base.clone();
    spec.components.insert(
        UPSAMPLER_COMPONENT.to_string(),
        WeightsSource::Dir(tiny_snapshot()),
    );
    let gen = mlx_gen_flux3::provider_registry()
        .unwrap()
        .load("flux3_klein", &spec)
        .unwrap();
    let reports: Arc<Mutex<Vec<PromptEnhancementReport>>> = Arc::default();
    let sink = reports.clone();
    req.prompt_enhancement = PromptEnhancementSink::new(move |r| sink.lock().unwrap().push(r));
    let mut events = Vec::new();
    let out = gen.generate(&req, &mut |p| events.push(p)).unwrap();
    assert!(matches!(out, GenerationOutput::Images(ref images) if images.len() == 1));
    let reports = reports.lock().unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].outcome, PromptEnhancementOutcome::Enhanced);
    assert_eq!(reports[0].mode.as_deref(), Some("t2i"));
    assert_ne!(reports[0].effective_prompt, reports[0].original_prompt);
    let first_step = events
        .iter()
        .position(|e| matches!(e, Progress::Step { .. }))
        .expect("steps");
    let last_upsample = events
        .iter()
        .rposition(|e| matches!(e, Progress::Upsampling { .. }))
        .expect("upsampling");
    assert!(
        last_upsample < first_step,
        "upsampling precedes the denoise: {events:?}"
    );
    assert!(matches!(events[0], Progress::Upsampling { tokens: 0, .. }));

    // Bypass through the same loaded generator: identity, reported as `Absent`, no stage events.
    let reports2: Arc<Mutex<Vec<PromptEnhancementReport>>> = Arc::default();
    let sink = reports2.clone();
    let bypass = GenerationRequest {
        enhance_prompt: false,
        enhance_max_tokens: None,
        prompt_enhancement: PromptEnhancementSink::new(move |r| sink.lock().unwrap().push(r)),
        ..req.clone()
    };
    let mut events = Vec::new();
    gen.generate(&bypass, &mut |p| events.push(p)).unwrap();
    let reports2 = reports2.lock().unwrap();
    assert_eq!(reports2[0].outcome, PromptEnhancementOutcome::Absent);
    assert_eq!(reports2[0].effective_prompt, "a red fox in the snow");
    assert!(events
        .iter()
        .all(|e| !matches!(e, Progress::Upsampling { .. })));

    // Cancellation raised during the stage is the typed `Canceled`, before any step.
    let cancel = CancelFlag::new();
    let c2 = cancel.clone();
    let cancelled = GenerationRequest {
        cancel,
        ..req.clone()
    };
    let mut events = Vec::new();
    let err = gen
        .generate(&cancelled, &mut |p| {
            events.push(p);
            if matches!(p, Progress::Upsampling { tokens: 1, .. }) {
                c2.cancel();
            }
        })
        .unwrap_err();
    assert!(matches!(err, CoreError::Canceled), "{err}");
    assert!(
        events
            .iter()
            .all(|e| matches!(e, Progress::Upsampling { .. })),
        "{events:?}"
    );
}

#[test]
fn a_component_that_is_not_the_upsampler_is_refused_at_load() {
    // S1's miniature Qwen3-VL text encoder is a VLM snapshot too, but not the upsampler
    // (`model_type: qwen3_vl`): the package check names the mismatch before any weight loads.
    let te = tiny_package().join("text_encoder");
    let err = Flux3PromptUpsampler::load(&te).unwrap_err();
    assert!(err.to_string().contains("prompt_upsampler"), "{err}");
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_package()));
    spec.components
        .insert(UPSAMPLER_COMPONENT.to_string(), WeightsSource::Dir(te));
    assert!(mlx_gen_flux3::provider_registry()
        .unwrap()
        .load("flux3_klein", &spec)
        .is_err());
    // The mode contract: an explicit editing mode with no references is a typed refusal.
    let up = load_upsampler();
    let input = pup::UpsamplerInput {
        prompt: "p".into(),
        references: Vec::new(),
        boxes: None,
        mode: UpsampleMode::Editing,
        bypass: false,
    };
    let err = pup::upsample(
        Some(&up as &dyn PromptUpsampler),
        &input,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, CoreError::Unsupported(m) if m.contains("would run t2i")),
        "{err}"
    );
}
