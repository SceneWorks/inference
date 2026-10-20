//! The same four frozen upstream prompt-upsampler fixtures through the real native Candle VLM.

use candle_gen::gen_core::core_llm::{self, Channel, StreamEvent};
use candle_gen::gen_core::flux3::pup::{self, PromptUpsampler, UpsampleOptions, UpsamplePlan};
use candle_gen_flux3::pup::Flux3PromptUpsampler;

use crate::common::{fixture, fixtures, host_f32, host_i32, Fixture};

fn plan_for(up: &Flux3PromptUpsampler, g: &Fixture, case: &str) -> UpsamplePlan {
    const BALLOONS: &str = r#"[{"id": "obj_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]}, {"id": "obj_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]}]"#;
    const CACTUS: &str = r#"[{"id": "obj_1", "desc": "a small potted cactus", "from": "target", "src_bbox": null, "tgt_bbox": [0.55, 0.62, 0.85, 0.85]}]"#;
    let (prompt, boxes) = match case {
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
        _ => panic!("unknown oracle case"),
    };
    let n: usize = g.meta(&format!("n_refs_{case}")).parse().unwrap();
    let references = (0..n)
        .map(|i| {
            let tensor = g.require(&format!("ref_{case}_{i}"));
            let shape = tensor.dims();
            core_llm::ImageRef::new(
                shape[1] as u32,
                shape[0] as u32,
                host_i32(tensor).into_iter().map(|v| v as u8).collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let (mode, mut user_text, locked) =
        pup::plan_text(prompt, boxes.as_deref(), references.len()).unwrap();
    assert_eq!(mode.system_key(), g.meta(&format!("system_key_{case}")));
    if mode.uses_references() {
        user_text = pup::i2i_scaffold(&user_text, locked.is_some());
    }
    let mut plan = UpsamplePlan {
        mode,
        original_prompt: prompt.into(),
        system_prompt: up.package().prompts().get(mode).to_string(),
        user_text,
        locked,
        references,
        max_new_tokens: 0,
    };
    let steps = g.meta("greedy_steps").parse().unwrap();
    plan.max_new_tokens = up.package().budget(&plan, Some(steps)).unwrap();
    plan
}

/// Derive the public Content-token view independently from the fixture tokenizer and frozen
/// stop strings. The oracle records raw decoder IDs, including protocol tokens and final EOS;
/// those are consumed by the native provider, not exposed as answer tokens. Ordinary caption
/// and box-annotation tokens must survive unchanged. Raw decoder parity is separately covered
/// by the shared Candle LLM teacher-forced regression.
fn content_projection(up: &Flux3PromptUpsampler, raw: &[i32], margins: &[f32]) -> Vec<(i32, f32)> {
    let tokenizer = up.package().tokenizer();
    let special = tokenizer.constraint_decode_table().special;
    let stops: std::collections::HashSet<u32> = pup::STOP_TOKENS
        .iter()
        .map(|s| {
            let ids = tokenizer.encode(s, false).unwrap();
            assert_eq!(
                ids.len(),
                1,
                "frozen stop strings are single-token protocol markers"
            );
            ids[0]
        })
        .collect();
    let mut visible = Vec::new();
    for (&id, &margin) in raw.iter().zip(margins) {
        if stops.contains(&(id as u32)) {
            break;
        }
        if !special.contains(&(id as u32)) {
            visible.push((id, margin));
        }
    }
    visible
}

fn native_case(case: &str) {
    let up = Flux3PromptUpsampler::load(&fixtures().join("tiny-pup-snapshot")).unwrap();
    let golden = fixture("flux3_pup_golden.safetensors");
    assert_eq!(golden.meta("pup_revision"), pup::UPSAMPLER_REVISION);
    assert_eq!(golden.meta("presence_penalty_scope"), "generated_only");
    let plan = plan_for(&up, &golden, case);
    assert_eq!(
        up.package().render(&plan).unwrap(),
        golden.meta(&format!("chat_text_{case}"))
    );
    let prompt_ids = host_i32(golden.require(&format!("ids_{case}")));
    assert_eq!(up.package().prompt_tokens(&plan).unwrap(), prompt_ids.len());
    let request = plan.request(&UpsampleOptions::default(), core_llm::CancelFlag::new());
    let mut got = Vec::new();
    let report = up
        .engine()
        .generate(&request, &mut |event| {
            if let StreamEvent::Token {
                id,
                channel: Channel::Content,
                ..
            } = event
            {
                got.push(id as i32);
            }
        })
        .unwrap();
    assert_eq!(
        report.usage.prompt_tokens as usize,
        prompt_ids.len(),
        "{case}: tokenization"
    );
    let expected = host_i32(golden.require(&format!("generated_{case}")));
    let margins = host_f32(golden.require(&format!("margins_{case}")));
    // Identical oracle policy to the MLX twin: never accept a wide-margin argmax mismatch,
    // and never call a wholly divergent trajectory parity. No tolerance widened for Candle.
    let mut matched = 0;
    let visible = content_projection(&up, &expected, &margins);
    for (i, &(want, margin)) in visible.iter().enumerate() {
        let actual = got
            .get(i)
            .unwrap_or_else(|| panic!("{case}: stopped at token {i}"));
        if *actual == want {
            matched += 1;
        } else {
            assert!(
                margin < 0.02,
                "{case}: wide-margin mismatch at {i}: {actual} vs {want}, margin {margin}"
            );
            break;
        }
    }
    assert!(
        matched >= 3,
        "{case}: only {matched} oracle tokens matched: {got:?} vs {expected:?}"
    );
    if matched == visible.len() {
        assert_eq!(
            got.len(),
            visible.len(),
            "{case}: no unexpected public tokens"
        );
        let stops: Vec<_> = pup::STOP_TOKENS
            .iter()
            .flat_map(|s| up.package().tokenizer().encode(s, false).unwrap())
            .collect();
        let raw_prefix = expected
            .iter()
            .position(|id| stops.contains(&(*id as u32)))
            .unwrap_or(expected.len());
        let stopped = raw_prefix < expected.len();
        // Native usage counts sampled protocol tokens, but its terminal EOS is detected before
        // the generated-token counter and is not counted (the HF raw oracle includes it).
        assert_eq!(
            report.usage.generated_tokens as usize, raw_prefix,
            "{case}: native raw token accounting"
        );
        assert_eq!(
            report.finish_reason,
            Some(if stopped {
                core_llm::FinishReason::Stop
            } else {
                core_llm::FinishReason::Length
            }),
            "{case}: EOS versus budget completion"
        );
    }
}

#[test]
fn projection_keeps_ordinary_caption_and_locked_box_tokens() {
    let up = Flux3PromptUpsampler::load(&fixtures().join("tiny-pup-snapshot")).unwrap();
    let tokenizer = up.package().tokenizer();
    // Locked JSON is reimposed after the VLM finishes; it is not passed through a protocol-token
    // filter. In particular the miniature word vocabulary's [UNK] punctuation cannot erase it.
    let golden = fixture("flux3_pup_golden.safetensors");
    let plan = plan_for(&up, &golden, "t2i_bbox");
    let locked = plan.locked.clone().unwrap();
    let finished = plan.finish("red bicycle").unwrap();
    let (_, recovered) = pup::detect(&finished.caption).expect("locked annotation block survives");
    assert_eq!(recovered, locked);
    assert!(finished
        .caption
        .contains(&pup::dumps(&pup::JsonValue::Array(locked))));
    let caption = tokenizer
        .encode("red bicycle", false)
        .unwrap()
        .into_iter()
        .map(|v| v as i32)
        .collect::<Vec<_>>();
    assert_eq!(
        content_projection(&up, &caption, &vec![1.0; caption.len()])
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        caption
    );
}

#[test]
fn native_text_matches_the_shared_oracle() {
    native_case("t2i");
}
#[test]
fn native_text_with_boxes_matches_the_shared_oracle() {
    native_case("t2i_bbox");
}
#[test]
fn native_edit_matches_the_shared_oracle() {
    native_case("i2i");
}
#[test]
fn native_edit_with_boxes_matches_the_shared_oracle() {
    native_case("i2i_bbox");
}
