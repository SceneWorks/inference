//! FLUX 3 prompt-upsampler contract tests (sc-26238) against the oracle goldens dumped by
//! `crates/media/mlx-gen/tools/dump_flux3_pup_golden.py` from the frozen `pup.py` / `bbox.py`
//! and the pinned package's own tokenizer + chat template: the four modes' plans and rendered
//! chat text (with and without locked boxes, detected tables, finished answers), `reconcile`,
//! `detect` / `split` / `is_finished`, the bypass caption, `to_canvas` refusals, the reference
//! size table, the Qwen `smart_resize` token table, the reference policy pixels, and — through a
//! scripted `TextLlm` — the stage itself: bypass identity, effective-caption provenance, locked
//! annotation preservation, cancellation and progress, and the typed refusals.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use gen_core::core_llm::{
    self, Channel, FinishReason, ImageRef, StreamEvent, TextLlm, TextLlmCapabilities,
    TextLlmDescriptor, TextLlmOutput, TextLlmRequest, Usage,
};
use gen_core::flux3::pup::{
    self, bypass_caption, decide, detect, dumps, is_finished, loads, plan_text, policy_reference,
    reconcile, reference_size, split, to_canvas, upsample, vision_tokens, BoxSpec, JsonValue,
    PromptUpsampler, ReferenceImage, SystemPrompts, UpsampleDecision, UpsampleMode,
    UpsampleOptions, UpsamplePackage, UpsamplePlan, UpsamplerInput, IMAGE_PLACEHOLDER,
    MAX_MODEL_LEN, MAX_REFERENCES, MAX_TOKENS, STOP_TOKENS, TOP_K, TOP_P, UPSAMPLER_COMPONENT,
};
use gen_core::{CancelFlag, Error, Progress, PromptEnhancementOutcome};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flux3_pup")
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixtures().join("golden.json")).expect("golden.json");
    serde_json::from_str(&text).expect("golden.json parses")
}

fn tiny_package() -> UpsamplePackage {
    UpsamplePackage::open(&fixtures().join("tiny-pup-package")).expect("tiny package opens")
}

fn s(v: &Value) -> &str {
    v.as_str()
        .unwrap_or_else(|| panic!("string golden, got {v}"))
}

fn boxes_of(v: &Value) -> Option<Vec<BoxSpec>> {
    v.as_str()
        .map(|json| BoxSpec::parse_list(json).expect("golden boxes parse"))
}

fn rows_of(json: &str) -> Vec<JsonValue> {
    loads(json)
        .expect("golden rows parse")
        .as_array()
        .expect("golden rows are a list")
        .to_vec()
}

/// A gray reference of the given size.
fn gray(width: u32, height: u32) -> ReferenceImage {
    ReferenceImage::Rgb8 {
        width,
        height,
        pixels: vec![128; (width * height * 3) as usize],
    }
}

// ---------------------------------------------------------------------------------------------
// Oracle goldens
// ---------------------------------------------------------------------------------------------

#[test]
fn the_golden_was_dumped_from_the_pinned_upstream() {
    let g = golden();
    assert_eq!(
        s(&g["upstream"]["code_commit"]),
        gen_core::flux3::UPSTREAM_CODE_REVISION
    );
    assert_eq!(s(&g["upstream"]["pup_revision"]), pup::UPSAMPLER_REVISION);
    assert_eq!(
        s(&g["upstream"]["source_sha256"]["src/flux3/pup.py"]),
        pup::PUP_PY_SHA256
    );
    assert_eq!(
        s(&g["upstream"]["source_sha256"]["src/flux3/bbox.py"]),
        pup::BBOX_PY_SHA256
    );
    assert_eq!(
        s(&g["upstream"]["package_sha256"]["system_prompts.json"]),
        pup::SYSTEM_PROMPTS_SHA256
    );
    // The sampling policy the constants carry is upstream's.
    let sampling = &g["sampling"];
    assert_eq!(
        sampling["temperature"].as_f64().unwrap() as f32,
        pup::TEMPERATURE
    );
    assert_eq!(sampling["top_p"].as_f64().unwrap() as f32, TOP_P);
    assert_eq!(sampling["top_k"].as_u64().unwrap() as usize, TOP_K);
    assert_eq!(sampling["min_p"].as_f64().unwrap() as f32, pup::MIN_P);
    assert_eq!(sampling["max_tokens"].as_u64().unwrap() as u32, MAX_TOKENS);
    assert_eq!(
        sampling["presence_penalty"]["t2i"].as_f64().unwrap() as f32,
        pup::PRESENCE_PENALTY_T2I
    );
    assert_eq!(
        sampling["presence_penalty"]["i2i"].as_f64().unwrap() as f32,
        pup::PRESENCE_PENALTY_I2I
    );
    assert_eq!(
        sampling["max_model_len"].as_u64().unwrap() as usize,
        MAX_MODEL_LEN
    );
    assert_eq!(
        sampling["max_references"].as_u64().unwrap() as usize,
        MAX_REFERENCES
    );
    assert_eq!(
        sampling["reference_multiple"].as_u64().unwrap() as u32,
        pup::REFERENCE_MULTIPLE
    );
    assert_eq!(
        sampling["max_aspect_ratio"].as_f64().unwrap(),
        pup::MAX_ASPECT_RATIO
    );
    assert_eq!(
        sampling["jpeg_quality"].as_u64().unwrap() as u8,
        pup::JPEG_QUALITY
    );
    let stops: Vec<&str> = sampling["stop_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(s)
        .collect();
    assert_eq!(stops, STOP_TOKENS);
    for (k, v) in sampling["max_pixels_by_refs"].as_object().unwrap() {
        assert_eq!(
            pup::max_pixels_by_refs(k.parse().unwrap()),
            Some(v.as_u64().unwrap()),
            "{k} refs"
        );
    }
    assert_eq!(s(&g["scaffolds"]["plain"]), pup::I2I_SCAFFOLD_PLAIN);
    assert_eq!(s(&g["scaffolds"]["locked"]), pup::I2I_SCAFFOLD_LOCKED);
}

#[test]
fn plan_cases_match_bbox_plan_and_the_rendered_chat_text() {
    let g = golden();
    let package = tiny_package();
    let prompts = SystemPrompts::builtin();
    let cases = g["plan"].as_array().unwrap();
    assert!(
        cases.len() >= 11,
        "the four modes with and without locks, detected and finished tables"
    );
    let mut keys_seen = std::collections::BTreeSet::new();
    for case in cases {
        let name = s(&case["name"]);
        let prompt = s(&case["prompt"]);
        let boxes = boxes_of(&case["boxes_json"]);
        let n = case["n_images"].as_u64().unwrap() as usize;
        let (mode, user_text, locked) = plan_text(prompt, boxes.as_deref(), n).unwrap();
        assert_eq!(mode.system_key(), s(&case["system_key"]), "{name}: mode");
        keys_seen.insert(mode.system_key());
        let scaffolded = if n > 0 {
            pup::i2i_scaffold(&user_text, locked.is_some())
        } else {
            user_text.clone()
        };
        assert_eq!(scaffolded, s(&case["user_text"]), "{name}: user text");
        match (&locked, case["locked"].as_str()) {
            (Some(rows), Some(want)) => assert_eq!(
                dumps(&JsonValue::Array(rows.clone())),
                want,
                "{name}: locked rows"
            ),
            (None, None) => {}
            (got, want) => panic!("{name}: locked {got:?} vs {want:?}"),
        }
        let has_boxes = boxes.as_deref().is_some_and(|b| !b.is_empty());
        assert_eq!(
            !has_boxes && is_finished(prompt),
            case["finished"].as_bool().unwrap(),
            "{name}: finished"
        );
        assert_eq!(
            bypass_caption(prompt, boxes.as_deref()).unwrap(),
            s(&case["bypass"]),
            "{name}: bypass caption"
        );

        // The rendered chat text through the pinned template, references as placeholders.
        let plan = UpsamplePlan {
            mode,
            original_prompt: prompt.to_string(),
            system_prompt: prompts.get(mode).to_string(),
            user_text: scaffolded,
            locked,
            references: (0..n)
                .map(|_| ImageRef::new(1, 1, vec![0, 0, 0]).unwrap())
                .collect(),
            max_new_tokens: MAX_TOKENS,
        };
        let chat = package.render(&plan).unwrap();
        assert_eq!(chat, s(&case["chat_text"]), "{name}: chat text");
        assert_eq!(
            chat.matches(IMAGE_PLACEHOLDER).count(),
            n,
            "{name}: one placeholder per reference"
        );
        assert!(
            chat.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{name}: thinking is off"
        );
        let tiny_ids = package.tokenizer().encode(&chat, false).unwrap();
        assert_eq!(
            tiny_ids.len() as u64,
            case["chat_tokens_tiny"].as_u64().unwrap(),
            "{name}: tiny token count"
        );
        assert!(
            case["chat_tokens_real"].as_u64().unwrap() > 100,
            "{name}: the real tokenizer count is recorded"
        );
    }
    assert_eq!(keys_seen.len(), 4, "every mode is exercised: {keys_seen:?}");
}

#[test]
fn reconcile_cases_match_bbox_reconcile() {
    let g = golden();
    let cases = g["reconcile"].as_array().unwrap();
    assert!(cases.len() >= 8);
    for case in cases {
        let name = s(&case["name"]);
        let locked = rows_of(s(&case["locked_json"]));
        let got = reconcile(s(&case["text"]), &locked).unwrap();
        assert_eq!(got, s(&case["expect"]), "{name}");
    }
}

#[test]
fn detect_split_and_is_finished_match_bbox_py() {
    let g = golden();
    let cases = g["detect"].as_array().unwrap();
    assert!(cases.len() >= 12);
    for case in cases {
        let prompt = s(&case["prompt"]);
        let (caption, rows) = split(prompt);
        assert_eq!(
            caption,
            s(&case["split"]["caption"]),
            "split caption of {prompt:?}"
        );
        assert_eq!(
            rows.map(|r| dumps(&JsonValue::Array(r))),
            case["split"]["rows_json"].as_str().map(str::to_string),
            "split rows of {prompt:?}"
        );
        let found = detect(prompt);
        match (&found, &case["expect"]) {
            (None, Value::Null) => {}
            (Some((p, rows)), want) => {
                assert_eq!(p, s(&want["prompt"]), "detect prompt of {prompt:?}");
                assert_eq!(
                    dumps(&JsonValue::Array(rows.clone())),
                    s(&want["rows_json"]),
                    "detect rows of {prompt:?}"
                );
            }
            (got, want) => panic!("detect({prompt:?}) = {got:?}, upstream {want}"),
        }
        assert_eq!(
            is_finished(prompt),
            case["finished"].as_bool().unwrap(),
            "is_finished({prompt:?})"
        );
    }
}

#[test]
fn bypass_captions_and_to_canvas_refusals_match_upstream() {
    let g = golden();
    for case in g["bypass"].as_array().unwrap() {
        let boxes = boxes_of(&case["boxes_json"]);
        assert_eq!(
            bypass_caption(s(&case["prompt"]), boxes.as_deref()).unwrap(),
            s(&case["expect"])
        );
    }
    let errors = g["to_canvas_errors"].as_array().unwrap();
    assert!(errors.len() >= 3);
    for case in errors {
        let boxes = BoxSpec::parse_list(s(&case["boxes_json"])).unwrap();
        let err = to_canvas(&boxes).unwrap_err();
        let want = s(&case["error"]);
        // Upstream's message, minus its Python list repr of the offending box.
        let stem = want.split(", got").next().unwrap();
        assert!(
            err.to_string().contains(stem),
            "{err} should carry {want:?}"
        );
    }
}

#[test]
fn reference_size_and_vision_token_tables_match_upstream() {
    let g = golden();
    let table = g["reference_size"].as_array().unwrap();
    assert!(table.len() >= 20);
    for row in table {
        let (w, h, n) = (
            row["w"].as_u64().unwrap() as u32,
            row["h"].as_u64().unwrap() as u32,
            row["n"].as_u64().unwrap() as usize,
        );
        match (reference_size(w, h, n), row.get("expect")) {
            (Ok((gw, gh)), Some(want)) => {
                let want: Vec<u32> = want
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(vec![gw, gh], want, "reference_size({w}, {h}, {n})");
            }
            (Err(e), None) => assert!(e.to_string().contains(s(&row["error"])), "{w}x{h}/{n}: {e}"),
            (got, want) => panic!("reference_size({w}, {h}, {n}) = {got:?}, upstream {want:?}"),
        }
    }
    for row in g["vision_tokens"].as_array().unwrap() {
        let (w, h) = (
            row["w"].as_u64().unwrap() as u32,
            row["h"].as_u64().unwrap() as u32,
        );
        assert_eq!(
            vision_tokens(w, h).unwrap() as u64,
            row["tokens"].as_u64().unwrap(),
            "smart_resize tokens of {w}x{h}"
        );
    }
}

/// Byte-exact end to end: Pillow's RGB conversion (alpha over white), its LANCZOS resample, and
/// libjpeg's quality-92 round trip (4:2:0, ISLOW DCTs, fancy upsampling) as Pillow runs it.
#[test]
fn policy_reference_matches_pillow_bit_for_bit() {
    let bytes = std::fs::read(fixtures().join("policy_reference.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let names: Vec<String> = st
        .names()
        .into_iter()
        .filter_map(|n| n.strip_prefix("in_").map(str::to_string))
        .collect();
    assert!(names.len() >= 6, "{names:?}");
    let u8s = |key: &str| -> (Vec<usize>, Vec<u8>) {
        let t = st.tensor(key).unwrap_or_else(|_| panic!("{key}"));
        assert_eq!(t.dtype(), safetensors::Dtype::U8, "{key}");
        (t.shape().to_vec(), t.data().to_vec())
    };
    let mut lossy_cases = 0usize;
    for name in &names {
        let (shape, input) = u8s(&format!("in_{name}"));
        let (h, w, c) = (shape[0] as u32, shape[1] as u32, shape[2]);
        let n_refs: usize = name.rsplit("_n").next().unwrap().parse().unwrap();
        let image = match c {
            3 => ReferenceImage::Rgb8 {
                width: w,
                height: h,
                pixels: input,
            },
            4 => ReferenceImage::Rgba8 {
                width: w,
                height: h,
                pixels: input,
            },
            other => panic!("{name}: {other} channels"),
        };
        let rgb = image.to_rgb8().unwrap().2;
        if c == 4 {
            let (_, want) = u8s(&format!("rgb_{name}"));
            assert_eq!(
                rgb, want,
                "{name}: RGB conversion (alpha over white) is Pillow's"
            );
        }
        let (oshape, out) = u8s(&format!("out_{name}"));
        let (th, tw) = (oshape[0] as u32, oshape[1] as u32);
        assert_eq!(
            reference_size(w, h, n_refs).unwrap(),
            (tw, th),
            "{name}: size budget"
        );
        let resized_key = format!("resized_{name}");
        if st.names().iter().any(|n| **n == resized_key) {
            let (_, resized) = u8s(&format!("resized_{name}"));
            let ours: Vec<u8> = if (tw, th) == (w, h) {
                rgb.clone()
            } else {
                gen_core::imageops::resize_lanczos_u8(
                    &rgb,
                    h as usize,
                    w as usize,
                    th as usize,
                    tw as usize,
                )
                .unwrap()
                .into_iter()
                .map(|v| v.round().clamp(0.0, 255.0) as u8)
                .collect()
            };
            assert_eq!(
                ours, resized,
                "{name}: LANCZOS resample is Pillow's, bit for bit"
            );
        }
        let got = policy_reference(&image, n_refs).unwrap();
        assert_eq!((got.height, got.width), (th, tw), "{name}: output size");
        let diffs: Vec<u8> = got
            .pixels
            .iter()
            .zip(&out)
            .map(|(a, b)| a.abs_diff(*b))
            .collect();
        let max = diffs.iter().copied().max().unwrap();
        let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
        assert_eq!(
            max, 0,
            "{name}: JPEG round trip differs from Pillow's (mean |delta| {mean:.3}, max {max})"
        );
        // The codec step is load-bearing: the fixture's output differs from its resampled input.
        let pre: Vec<u8> = if (tw, th) == (w, h) {
            rgb.clone()
        } else {
            Vec::new()
        };
        if !pre.is_empty() && pre != out {
            lossy_cases += 1;
        }
    }
    assert!(lossy_cases > 0, "no case exercised a lossy JPEG step");
}

// ---------------------------------------------------------------------------------------------
// The stage over a scripted engine
// ---------------------------------------------------------------------------------------------

/// A `TextLlm` that streams a scripted answer token by token (one token per whitespace-split
/// word), records the request it saw, and honours cancellation between tokens.
struct Scripted {
    descriptor: TextLlmDescriptor,
    answer: String,
    seen: Mutex<Vec<TextLlmRequest>>,
}

impl Scripted {
    fn new(answer: &str) -> Self {
        Self {
            descriptor: TextLlmDescriptor {
                id: "scripted".into(),
                family: "test".into(),
                backend: "none".into(),
                capabilities: TextLlmCapabilities {
                    supports_system_prompt: true,
                    supports_vision: true,
                    supports_thinking: true,
                    ..Default::default()
                },
            },
            answer: answer.to_string(),
            seen: Mutex::new(Vec::new()),
        }
    }
}

impl TextLlm for Scripted {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, _req: &TextLlmRequest) -> core_llm::Result<()> {
        Ok(())
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> core_llm::Result<TextLlmOutput> {
        self.seen.lock().unwrap().push(req.clone());
        if req.cancel.is_cancelled() {
            return Err(core_llm::Error::Canceled);
        }
        let mut text = String::new();
        let mut generated = 0u32;
        let mut finish = FinishReason::Stop;
        for (i, word) in self.answer.split_inclusive(' ').enumerate() {
            if generated >= req.max_new_tokens {
                finish = FinishReason::Length;
                break;
            }
            if req.cancel.is_cancelled() {
                finish = FinishReason::Cancelled;
                break;
            }
            text.push_str(word);
            generated += 1;
            on_event(StreamEvent::Token {
                id: i as u32,
                text: word.to_string(),
                index: i,
                channel: Channel::Content,
            });
        }
        let usage = Usage {
            prompt_tokens: 10,
            generated_tokens: generated,
        };
        on_event(StreamEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            text,
            usage,
            finish_reason: Some(finish),
            ..Default::default()
        })
    }
}

struct FakeUpsampler {
    package: UpsamplePackage,
    engine: Scripted,
}

impl FakeUpsampler {
    fn new(answer: &str) -> Self {
        Self {
            package: tiny_package(),
            engine: Scripted::new(answer),
        }
    }
}

impl PromptUpsampler for FakeUpsampler {
    fn package(&self) -> &UpsamplePackage {
        &self.package
    }

    fn generate(
        &self,
        plan: &UpsamplePlan,
        options: &UpsampleOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<String> {
        pup::run_text_llm(&self.engine, plan, options, cancel, on_progress)
    }
}

fn input(
    prompt: &str,
    mode: UpsampleMode,
    references: Vec<ReferenceImage>,
    boxes: Option<&str>,
) -> UpsamplerInput {
    UpsamplerInput {
        prompt: prompt.to_string(),
        references,
        boxes: boxes.map(|b| BoxSpec::parse_list(b).unwrap()),
        mode,
        bypass: false,
    }
}

const BALLOONS: &str = r#"[{"id": "obj_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]}, {"id": "obj_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]}]"#;
const CACTUS: &str = r#"[{"id": "obj_1", "desc": "a small potted cactus", "from": "target", "src_bbox": null, "tgt_bbox": [0.55, 0.62, 0.85, 0.85]}]"#;

#[test]
fn text_mode_runs_with_the_t2i_prompt_and_reports_the_effective_caption() {
    let up = FakeUpsampler::new(
        "Full-color outdoor photograph of a red bicycle leaning against a brick building.",
    );
    let inp = input("a red bicycle", UpsampleMode::Text, vec![], None);
    let mut events = Vec::new();
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |p| events.push(p),
    )
    .unwrap();
    assert_eq!(
        out.caption,
        "Full-color outdoor photograph of a red bicycle leaning against a brick building."
    );
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Enhanced);
    assert_eq!(out.report.original_prompt, "a red bicycle");
    assert_eq!(out.report.effective_prompt, out.caption);
    assert_eq!(out.report.mode.as_deref(), Some("t2i"));
    assert_eq!(out.report.locked_annotations, None);
    assert_eq!(out.report.fallback_reason, None);
    // Progress: one `0` then one per token, bounded by the budget.
    assert_eq!(
        events[0],
        Progress::Upsampling {
            tokens: 0,
            max_tokens: MAX_TOKENS
        }
    );
    let words = out.caption.split_inclusive(' ').count() as u32;
    assert_eq!(events.len() as u32, words + 1);
    assert_eq!(
        events[events.len() - 1],
        Progress::Upsampling {
            tokens: words,
            max_tokens: MAX_TOKENS
        }
    );
    assert!(events.windows(2).all(|w| matches!((w[0], w[1]), (Progress::Upsampling { tokens: a, .. }, Progress::Upsampling { tokens: b, .. }) if b == a + 1)));
    // The engine saw upstream's request shape.
    let seen = up.engine.seen.lock().unwrap();
    let req = &seen[0];
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[0].role, core_llm::Role::System);
    assert_eq!(req.messages[0].text_content(), SystemPrompts::builtin().t2i);
    assert_eq!(req.messages[1].text_content(), "a red bicycle");
    assert!(!req.messages[1].has_image());
    assert_eq!(req.sampling.temperature, 0.0);
    assert_eq!(req.sampling.top_p, TOP_P);
    assert_eq!(req.sampling.top_k, TOP_K);
    assert_eq!(req.sampling.presence_penalty, 1.5);
    assert!(
        req.sampling.presence_penalty_generated_only,
        "vLLM excludes the input prompt from presence penalties"
    );
    assert_eq!(req.sampling.repetition_penalty, 1.0);
    assert_eq!(req.max_new_tokens, MAX_TOKENS);
    assert_eq!(req.thinking, core_llm::ThinkingMode::Disabled);
    assert_eq!(req.stop, STOP_TOKENS);
    assert_eq!(req.seed, None);
}

#[test]
fn editing_modes_attach_the_policy_references_and_scaffold_the_instruction() {
    let up = FakeUpsampler::new(
        "Change the dog's collar to a red bandana, everything else stays identical.",
    );
    let inp = input(
        "put a red bandana on the dog",
        UpsampleMode::Editing,
        vec![gray(70, 50)],
        None,
    );
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.report.mode.as_deref(), Some("i2i"));
    let seen = up.engine.seen.lock().unwrap();
    let req = &seen[0];
    assert_eq!(req.messages[0].text_content(), SystemPrompts::builtin().i2i);
    let user = &req.messages[1];
    assert_eq!(user.content.len(), 2, "one image then the scaffold");
    match &user.content[0] {
        core_llm::Content::Image(img) => assert_eq!(
            (img.width, img.height),
            (64, 64),
            "70x50 under the 1-ref budget rounds to 64x64"
        ),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        user.text_content(),
        pup::i2i_scaffold("put a red bandana on the dog", false)
    );
    assert_eq!(
        req.sampling.presence_penalty, 0.0,
        "i2i has no presence penalty"
    );
    drop(seen);

    // Editing with boxes: the locked scaffold, the `i2i_bbox` prompt, two references in order.
    let up = FakeUpsampler::new("Add a cactus <obj_1>. [{\"id\": \"obj_1\", \"from\": \"target\", \"src_bbox\": null, \"tgt_bbox\": [550, 620, 850, 850], \"desc\": \"a cactus in a terracotta pot\"}]");
    let inp = input(
        "add a cactus on the table",
        UpsampleMode::EditingWithBoxes,
        vec![gray(40, 60), gray(64, 32)],
        Some(CACTUS),
    );
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.report.mode.as_deref(), Some("i2i_bbox"));
    let locked = r#"[{"id": "obj_1", "from": "target", "src_bbox": null, "tgt_bbox": [550, 620, 850, 850], "desc": "a small potted cactus"}]"#;
    assert_eq!(out.report.locked_annotations.as_deref(), Some(locked));
    assert_eq!(
        out.caption,
        "Add a cactus <obj_1>. [{\"id\": \"obj_1\", \"from\": \"target\", \"src_bbox\": null, \"tgt_bbox\": [550, 620, 850, 850], \"desc\": \"a cactus in a terracotta pot\"}]",
        "the locked row is kept verbatim and adopts the model's desc"
    );
    let seen = up.engine.seen.lock().unwrap();
    let user = &seen[0].messages[1];
    assert_eq!(user.content.iter().filter(|c| c.is_image()).count(), 2);
    let text = user.text_content();
    assert!(text.starts_with("Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: ORIGINAL PROMPT:\nadd a cactus on the table\n\nLOCKED BOUNDING-BOX ANNOTATIONS ("), "{text}");
    assert!(text.ends_with(&format!(":\n{locked}\nDecide the edit, imagine the target image, and output the annotation record.")), "{text}");
}

#[test]
fn text_with_boxes_locks_the_table_and_reconciles_the_answer() {
    let answer = "Full-color outdoor photograph of two balloons: a striped balloon <balloon_1> and a small blue balloon <obj_2> above a lighthouse <lighthouse_1>. [{\"id\": \"balloon_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"a red and yellow striped balloon\"}, {\"id\": \"obj_2\", \"bbox\": [200, 620, 420, 800], \"desc\": \"a small blue balloon\"}, {\"id\": \"lighthouse_1\", \"bbox\": [600, 300, 1000, 500], \"desc\": \"a white lighthouse\"}]";
    let up = FakeUpsampler::new(answer);
    let inp = input(
        "two hot-air balloons over the sea",
        UpsampleMode::TextWithBoxes,
        vec![],
        Some(BALLOONS),
    );
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    let locked = r#"[{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red and yellow striped hot-air balloon"}, {"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a small blue hot-air balloon"}]"#;
    assert_eq!(out.report.mode.as_deref(), Some("t2i_bbox"));
    assert_eq!(out.report.locked_annotations.as_deref(), Some(locked));
    assert_eq!(
        out.caption,
        "Full-color outdoor photograph of two balloons: a striped balloon <obj_1> and a small blue balloon <obj_2> above a lighthouse <lighthouse_1>. [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"a red and yellow striped balloon\"}, {\"id\": \"obj_2\", \"bbox\": [200, 620, 420, 800], \"desc\": \"a small blue balloon\"}, {\"id\": \"lighthouse_1\", \"bbox\": [600, 300, 1000, 500], \"desc\": \"a white lighthouse\"}]"
    );
    // Every locked box survives with its coordinates; the citation was renamed to the lock's id.
    assert!(out.caption.contains("<obj_1>") && !out.caption.contains("<balloon_1>"));
    assert!(
        out.caption.contains("[80, 60, 480, 380]") && out.caption.contains("[200, 620, 420, 800]")
    );
    // The engine was given the locked input and the t2i_bbox prompt.
    let seen = up.engine.seen.lock().unwrap();
    assert_eq!(
        seen[0].messages[0].text_content(),
        SystemPrompts::builtin().t2i_bbox
    );
    assert_eq!(
        seen[0].messages[1].text_content(),
        pup::upsampling_input("two hot-air balloons over the sea", &rows_of(locked))
    );
    assert_eq!(seen[0].sampling.presence_penalty, 1.5);
}

#[test]
fn bypass_preserves_the_prompt_and_appends_a_passed_table() {
    let up = FakeUpsampler::new("never called");
    let mut inp = input("  as typed \n", UpsampleMode::Text, vec![], None);
    inp.bypass = true;
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| panic!("no progress on bypass"),
    )
    .unwrap();
    assert_eq!(out.caption, "  as typed \n");
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Absent);
    assert_eq!(out.report.effective_prompt, "  as typed \n");
    assert_eq!(out.report.mode, None);
    assert!(
        up.engine.seen.lock().unwrap().is_empty(),
        "bypass never touches the engine"
    );

    let mut inp = input(
        "  two balloons \n",
        UpsampleMode::TextWithBoxes,
        vec![],
        Some(BALLOONS),
    );
    inp.bypass = true;
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    let table = r#"[{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red and yellow striped hot-air balloon"}, {"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a small blue hot-air balloon"}]"#;
    assert_eq!(
        out.caption,
        format!("  two balloons {table}"),
        "generate.py's no-pup caption"
    );
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Absent);
    assert_eq!(out.report.original_prompt, "  two balloons \n");
    assert_eq!(out.report.locked_annotations.as_deref(), Some(table));

    // Bypass without any upsampler loaded is the same identity path (no package needed).
    let out = upsample(
        None,
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.caption, format!("  two balloons {table}"));
}

#[test]
fn a_finished_answer_skips_the_upsampler_as_a_fallback() {
    let up = FakeUpsampler::new("never called");
    let finished = "a red balloon <obj_1> [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"b\"}]";
    let inp = input(finished, UpsampleMode::TextWithBoxes, vec![], None);
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.caption, finished);
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Fallback);
    assert_eq!(
        out.report.fallback_reason.as_deref(),
        Some("finished_answer")
    );
    assert!(up.engine.seen.lock().unwrap().is_empty());
    // The same table with an uncited id is a request: detected boxes lock it and the model runs.
    let unfinished =
        "a red balloon [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"b\"}]";
    let up = FakeUpsampler::new("A red balloon <obj_1>. [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"a red balloon\"}]");
    let inp = input(unfinished, UpsampleMode::TextWithBoxes, vec![], None);
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Enhanced);
    assert_eq!(out.caption, "A red balloon <obj_1>. [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"a red balloon\"}]");
}

#[test]
fn typed_refusals_name_their_cause() {
    let up = FakeUpsampler::new("x");
    let opts = UpsampleOptions::default();
    let cancel = CancelFlag::new();
    // Missing upsampler with enhancement requested.
    let err = upsample(
        None,
        &input("p", UpsampleMode::Text, vec![], None),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains(UPSAMPLER_COMPONENT) && m.contains(pup::UPSAMPLER_REPO)),
        "{err}"
    );
    // Mode / reference mismatch, both ways.
    let err = upsample(
        Some(&up),
        &input("p", UpsampleMode::Text, vec![gray(64, 64)], None),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("mode t2i") && m.contains("would run i2i")),
        "{err}"
    );
    let err = upsample(
        Some(&up),
        &input("p", UpsampleMode::Editing, vec![], None),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("mode i2i") && m.contains("would run t2i")),
        "{err}"
    );
    let err = upsample(
        Some(&up),
        &input("p", UpsampleMode::Text, vec![], Some(BALLOONS)),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("would run t2i_bbox")),
        "{err}"
    );
    // A table detected in the prompt also counts as boxes.
    let detected = "a <obj_1> [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380]}]";
    let err = upsample(
        Some(&up),
        &input(detected, UpsampleMode::Editing, vec![gray(64, 64)], None),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("would run i2i_bbox")),
        "{err}"
    );
    // Too many references.
    let refs: Vec<ReferenceImage> = (0..=MAX_REFERENCES).map(|_| gray(32, 32)).collect();
    let err = upsample(
        Some(&up),
        &input("p", UpsampleMode::Editing, refs, None),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("at most 10 references (got 11)")),
        "{err}"
    );
    // A malformed box.
    let err = upsample(
        Some(&up),
        &input(
            "p",
            UpsampleMode::TextWithBoxes,
            vec![],
            Some(r#"[{"bbox": [0.1, 0.2]}]"#),
        ),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("[y_min, x_min, y_max, x_max]")),
        "{err}"
    );
    // Over-long input: no room left in the window.
    let long = "balloon ".repeat(MAX_MODEL_LEN);
    let err = upsample(
        Some(&up),
        &input(&long, UpsampleMode::Text, vec![], None),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("leaving no room to generate")),
        "{err}"
    );
    assert!(
        up.engine.seen.lock().unwrap().is_empty(),
        "no refusal reached the engine"
    );
    // An empty answer.
    let up = FakeUpsampler::new("   ");
    let err = upsample(
        Some(&up),
        &input("p", UpsampleMode::Text, vec![], None),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Msg(m) if m.ends_with("empty generation")),
        "{err}"
    );
}

#[test]
fn the_window_budget_clips_the_decode_length_and_the_request_overrides_apply() {
    let up = FakeUpsampler::new("short answer");
    let package = tiny_package();
    // ~6000 tiny tokens of prompt leave less than MAX_TOKENS in the 8192 window.
    let long = "balloon ".repeat(6000);
    let inp = input(&long, UpsampleMode::Text, vec![], None);
    let UpsampleDecision::Run(plan) = decide(&inp, &package, &UpsampleOptions::default()).unwrap()
    else {
        panic!("a run")
    };
    let tokens = package.prompt_tokens(&plan).unwrap();
    assert!(tokens > 6000 && tokens < MAX_MODEL_LEN, "{tokens}");
    assert_eq!(plan.max_new_tokens as usize, MAX_MODEL_LEN - tokens);
    // An explicit smaller budget wins; a larger one is clipped.
    let opts = UpsampleOptions {
        temperature: Some(0.3),
        max_tokens: Some(2),
        seed: Some(9),
    };
    let UpsampleDecision::Run(plan) = decide(&inp, &package, &opts).unwrap() else {
        panic!("a run")
    };
    assert_eq!(plan.max_new_tokens, 2);
    let req = plan.request(&opts, core_llm::CancelFlag::new());
    assert_eq!(req.sampling.temperature, 0.3);
    assert_eq!(req.seed, Some(9));
    assert_eq!(req.max_new_tokens, 2);
    // The engine honours the budget: the scripted answer is truncated to two tokens.
    let out = upsample(Some(&up), &inp, &opts, &CancelFlag::new(), &mut |_| {}).unwrap();
    assert_eq!(out.caption, "short answer");
    let out = upsample(
        Some(&FakeUpsampler::new("one two three four")),
        &inp,
        &opts,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.caption, "one two");
    // Reference token expansion counts into the window.
    let inp = input("p", UpsampleMode::Editing, vec![gray(1024, 1024)], None);
    let UpsampleDecision::Run(plan) = decide(&inp, &package, &UpsampleOptions::default()).unwrap()
    else {
        panic!("a run")
    };
    let with_image = package.prompt_tokens(&plan).unwrap();
    let text_only = package
        .tokenizer()
        .encode(&package.render(&plan).unwrap(), false)
        .unwrap()
        .len();
    assert_eq!(
        with_image,
        text_only + vision_tokens(1024, 1024).unwrap() - 1
    );
    assert_eq!(vision_tokens(1024, 1024).unwrap(), 1024);
}

#[test]
fn cancellation_routes_from_the_generator_flag_into_the_engine() {
    // Pre-cancelled: nothing runs.
    let up = FakeUpsampler::new("one two three four five six");
    let cancel = CancelFlag::new();
    cancel.cancel();
    let err = upsample(
        Some(&up),
        &input("p", UpsampleMode::Text, vec![], None),
        &UpsampleOptions::default(),
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(err, Error::Canceled));
    assert!(up.engine.seen.lock().unwrap().is_empty());
    // Mid-stream: cancel from the progress callback after the second token; the engine's own
    // flag trips on the next poll, the run ends as `Canceled` and no caption is produced.
    let cancel = CancelFlag::new();
    let seen_tokens = Arc::new(Mutex::new(Vec::new()));
    let log = seen_tokens.clone();
    let c2 = cancel.clone();
    let err = upsample(
        Some(&up),
        &input("p", UpsampleMode::Text, vec![], None),
        &UpsampleOptions::default(),
        &cancel,
        &mut |p| {
            if let Progress::Upsampling { tokens, .. } = p {
                log.lock().unwrap().push(tokens);
                if tokens == 2 {
                    c2.cancel();
                }
            }
        },
    )
    .unwrap_err();
    assert!(matches!(err, Error::Canceled), "{err}");
    let seen = seen_tokens.lock().unwrap();
    assert_eq!(
        seen.as_slice(),
        &[0, 1, 2],
        "the decode stopped within one token of the cancel"
    );
    let requests = up.engine.seen.lock().unwrap();
    assert!(
        requests[0].cancel.is_cancelled(),
        "the engine flag was tripped"
    );
}

#[test]
fn a_reference_beyond_the_aspect_limit_falls_back_to_the_prompt_as_typed() {
    let up = FakeUpsampler::new("never called");
    let inp = input("p", UpsampleMode::Editing, vec![gray(2010, 10)], None);
    let out = upsample(
        Some(&up),
        &inp,
        &UpsampleOptions::default(),
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.caption, "p");
    assert_eq!(out.report.outcome, PromptEnhancementOutcome::Fallback);
    assert_eq!(
        out.report.fallback_reason.as_deref(),
        Some("reference_aspect_ratio")
    );
    assert!(up.engine.seen.lock().unwrap().is_empty());
}

#[test]
fn package_checks_are_typed() {
    let dir = tempfile::tempdir().unwrap();
    let err = UpsamplePackage::check(dir.path()).unwrap_err();
    assert!(
        err.to_string()
            .contains("missing config.json, model.safetensors, tokenizer.json"),
        "{err}"
    );
    for name in pup::UPSAMPLER_REQUIRED_FILES {
        std::fs::write(dir.path().join(name), b"{}").unwrap();
    }
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3_vl", "vision_config": {}}"#,
    )
    .unwrap();
    let err = UpsamplePackage::check(dir.path()).unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("model_type is \"qwen3_vl\"")),
        "{err}"
    );
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3_5"}"#,
    )
    .unwrap();
    let err = UpsamplePackage::check(dir.path()).unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(m) if m.contains("no vision_config")),
        "{err}"
    );
    // The tiny package is the real layout and opens; its prompts are the shipped file.
    let package = tiny_package();
    assert_eq!(package.prompts(), SystemPrompts::builtin());
    assert!(package.dir().join("system_prompts.json").is_file());
}

#[test]
fn optional_pup_license_identity_is_separate_and_frozen() {
    let row =
        gen_core::resolve_component(gen_core::MEDIA_COMPONENT_LICENSES, "flux3_image_klein_pup")
            .expect("the optional upsampler has its own shared license row");
    assert_eq!(
        row.source_url,
        "https://huggingface.co/black-forest-labs/flux-3-image-klein-pup"
    );
    assert_eq!(row.declared, "apache-2.0");
    assert_eq!(row.family, "apache-2-0");
    assert!(!row.gated);
    assert_eq!(
        pup::UPSAMPLER_REPO,
        "black-forest-labs/flux-3-image-klein-pup"
    );
    assert_eq!(
        pup::UPSAMPLER_REVISION,
        "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed"
    );
}

#[test]
fn native_request_options_preserve_absence_and_frozen_producer_budget() {
    let package = tiny_package();
    let mut request = gen_core::GenerationRequest {
        prompt: "a fox".to_owned(),
        enhance_prompt: true,
        ..Default::default()
    };
    let inp = input(&request.prompt, UpsampleMode::Text, vec![], None);
    for requested in [None, Some(2), Some(3_584)] {
        request.enhance_max_tokens = requested;
        let options = UpsampleOptions::from_generation_request(&request);
        assert_eq!(options.max_tokens, requested);
        assert_eq!(options.temperature, None);
        assert_eq!(options.seed, request.seed);
        let UpsampleDecision::Run(plan) = decide(&inp, &package, &options).unwrap() else {
            panic!("enabled request must plan a run");
        };
        assert_eq!(plan.max_new_tokens, requested.unwrap_or(3_584));
    }
    request.enhance_prompt = false;
    request.enhance_max_tokens = None;
    let mut bypass = inp;
    bypass.bypass = true;
    let UpsampleDecision::Bypass(outcome) = decide(
        &bypass,
        &package,
        &UpsampleOptions::from_generation_request(&request),
    )
    .unwrap() else {
        panic!("disabled request must bypass");
    };
    assert_eq!(outcome.caption, request.prompt);
}
