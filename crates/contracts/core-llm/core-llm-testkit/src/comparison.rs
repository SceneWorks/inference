//! Native, tensor-neutral measurement records for matched model comparisons.
//!
//! This deliberately reports task scores rather than declaring a model-quality threshold. It is a
//! small diagnostic suite, not a reproduction of a vendor's aggregate benchmark. Callers must seal
//! these raw records together with model inventories, runtime identity and external memory samples.

use core_llm::{
    Channel, Content, ImageRef, Message, MtpMode, Role, Sampling, StreamEvent, TextLlm,
    TextLlmRequest, ThinkingMode, VideoRef,
};
use serde_json::{json, Value};
use std::time::Instant;

/// Observable answer oracle, kept with the raw input so scoring can be independently repeated.
#[derive(Clone, Debug)]
pub enum AnswerOracle {
    /// Trim surrounding whitespace, retaining exact answer contents.
    Exact(String),
    /// Parse the entire answer as JSON, without silently stripping prose or code fences.
    Json(Value),
    /// Color words in order, collapsing consecutive repetitions (for repeated video frames).
    Colors(Vec<String>),
    /// One structured function call with exactly these arguments.
    Tool { name: String, arguments: Value },
}

impl AnswerOracle {
    fn as_json(&self) -> Value {
        match self {
            Self::Exact(text) => json!({"kind":"exact", "value":text}),
            Self::Json(value) => json!({"kind":"json", "value":value}),
            Self::Colors(colors) => json!({"kind":"ordered_colors", "value":colors}),
            Self::Tool { name, arguments } => {
                json!({"kind":"tool", "name":name, "arguments":arguments})
            }
        }
    }

    fn matches(&self, output: &core_llm::TextLlmOutput) -> bool {
        match self {
            Self::Exact(expected) => output.text.trim() == expected,
            Self::Json(expected) => {
                serde_json::from_str::<Value>(&output.text).ok().as_ref() == Some(expected)
            }
            Self::Colors(expected) => {
                let mut colors = output
                    .text
                    .to_lowercase()
                    .split(|c: char| !c.is_alphabetic())
                    .filter(|word| {
                        ["red", "blue", "green", "yellow", "black", "white"].contains(word)
                    })
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                colors.dedup();
                &colors == expected
            }
            Self::Tool { name, arguments } => {
                output.tool_calls.len() == 1
                    && output.tool_calls[0].name == *name
                    && Value::Object(output.tool_calls[0].arguments.clone()) == *arguments
            }
        }
    }
}

/// One fixed workload and its scoring rule. Every model in a comparison group gets this request.
#[derive(Clone, Debug)]
pub struct ComparisonCase {
    /// Stable case identity.
    pub id: String,
    /// Separate quality category; never average heterogeneous tasks without retaining these rows.
    pub category: String,
    /// Exact native request.
    pub request: TextLlmRequest,
    /// Independent observable answer.
    pub oracle: AnswerOracle,
}

fn image_json(image: &ImageRef) -> Value {
    json!({"width":image.width, "height":image.height, "rgb8":image.pixels})
}

/// Lossless request evidence, including actual decoded media rather than only a descriptive label.
pub fn request_evidence(request: &TextLlmRequest) -> Value {
    let messages = request.messages.iter().map(|message| {
        let content = message.content.iter().map(|part| match part {
            Content::Text(text) => json!({"type":"text", "text":text}),
            Content::Image(image) => json!({"type":"image", "image":image_json(image)}),
            Content::Video(video) => json!({"type":"video", "timestamps":video.timestamps,
                "frames":video.frames.iter().map(image_json).collect::<Vec<_>>()}),
            Content::Audio(audio) => json!({"type":"audio", "sample_rate":audio.sample_rate, "samples":audio.samples}),
        }).collect::<Vec<_>>();
        json!({"role":message.role.as_str(), "content":content, "thinking":message.thinking,
            "tool_calls":message.tool_calls.iter().map(|call|json!({"name":call.name,"arguments":call.arguments})).collect::<Vec<_>>()})
    }).collect::<Vec<_>>();
    let mtp = match request.mtp {
        MtpMode::Off => json!({"mode":"off"}),
        MtpMode::Auto => json!({"mode":"auto"}),
        MtpMode::Enabled { draft_tokens } => json!({"mode":"enabled", "draft_tokens":draft_tokens}),
    };
    json!({"messages":messages, "max_new_tokens":request.max_new_tokens, "seed":request.seed,
        "sampling":{"temperature":request.sampling.temperature,"top_p":request.sampling.top_p,
            "top_k":request.sampling.top_k,"presence_penalty":request.sampling.presence_penalty,
            "repetition_penalty":request.sampling.repetition_penalty,
            "repetition_context":request.sampling.repetition_context},
        "thinking":format!("{:?}",request.thinking),
        "reasoning_effort":request.reasoning_effort.map(|effort|effort.as_str()),
        "preserve_thinking":request.preserve_thinking,"mtp":mtp,
        "constraint":request.constraint.as_ref().map(|value|format!("{value:?}")),
        "stop":request.stop,
        "tools":request.tools.iter().map(|tool|tool.to_template_json()).collect::<Vec<_>>()})
}

/// Execute a case once, retaining every token event and any failure. A failure is never a skip.
///
/// First-token time starts before validation; prefill/decode come exclusively from synchronized
/// native instrumentation. Absent native phase measurements make evidence incomplete; a run that
/// emits no token events (for example, only a structured tool call) has explicitly unavailable TTFT.
pub fn measure_case(provider: &dyn TextLlm, case: &ComparisonCase) -> Value {
    let mut record = json!({"schema_version":1,"case_id":case.id,"category":case.category,
        "request":request_evidence(&case.request),"oracle":case.oracle.as_json()});
    let started = Instant::now();
    let mut first_token = None;
    let mut events = Vec::new();
    let mut text = String::new();
    let mut thinking = String::new();
    let mut terminal = Vec::new();
    let mut event_order_valid = true;
    let mut previous_index = None;
    let result = provider.validate(&case.request).and_then(|()|provider.generate(&case.request, &mut |event| {
        let seconds = started.elapsed().as_secs_f64();
        match event {
            StreamEvent::Token { id, text: delta, index, channel } => {
                event_order_valid &= terminal.is_empty() && previous_index.is_none_or(|previous| index > previous);
                previous_index = Some(index);
                first_token.get_or_insert(seconds);
                match channel { Channel::Content => text.push_str(&delta), Channel::Thinking => thinking.push_str(&delta) }
                events.push(json!({"event":"token","seconds":seconds,"id":id,"index":index,
                    "channel":format!("{channel:?}"),"text":delta}));
            }
            StreamEvent::Done { finish_reason, usage } => {
                terminal.push((finish_reason,usage));
                events.push(json!({"event":"done","seconds":seconds,"finish_reason":format!("{finish_reason:?}"),
                    "prompt_tokens":usage.prompt_tokens,"generated_tokens":usage.generated_tokens}));
            }
        }
    }));
    record["total_seconds"] = json!(started.elapsed().as_secs_f64());
    record["time_to_first_token_seconds"] = json!(first_token);
    record["time_to_first_token_unavailable_reason"] =
        json!(first_token.is_none().then_some("no_token_event_emitted"));
    record["events"] = json!(events);
    match result {
        Err(error) => {
            record["status"] = json!("failed");
            record["error"] = json!(error.to_string());
            record["evidence_complete"] = json!(false);
        }
        Ok(output) => {
            let stream_ok = event_order_valid
                && text == output.text
                && thinking == output.thinking.as_deref().unwrap_or("")
                && terminal.len() == 1
                && Some(terminal[0].0) == output.finish_reason
                && terminal[0].1 == output.usage;
            record["status"] = json!("completed");
            let quality_passed = case.oracle.matches(&output);
            record["quality_passed"] = json!(quality_passed);
            let reasoning_observed = case.request.thinking != ThinkingMode::Enabled
                || output
                    .thinking
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty());
            let mtp_observed = !matches!(case.request.mtp, MtpMode::Enabled { .. })
                || output
                    .mtp
                    .is_some_and(|stats| stats.proposed_tokens > 0 && stats.target_forwards > 0);
            record["functional_acceptance_passed"] =
                json!(quality_passed && stream_ok && reasoning_observed && mtp_observed);
            record["stream_contract_passed"] = json!(stream_ok);
            record["evidence_complete"] = json!(stream_ok && output.timings.is_some());
            record["output"] = json!({"text":output.text,"thinking":output.thinking,
                "tool_calls":output.tool_calls.iter().map(|call|json!({"name":call.name,"arguments":call.arguments})).collect::<Vec<_>>(),
                "prompt_tokens":output.usage.prompt_tokens,"generated_tokens":output.usage.generated_tokens,
                "finish_reason":output.finish_reason.map(|value|format!("{value:?}")),
                "mtp":output.mtp.map(|value|json!({"proposed_tokens":value.proposed_tokens,
                    "accepted_tokens":value.accepted_tokens,"target_forwards":value.target_forwards}))});
            record["prefill_seconds"] =
                json!(output.timings.map(|timings| timings.prefill.as_secs_f64()));
            record["decode_seconds"] =
                json!(output.timings.map(|timings| timings.decode.as_secs_f64()));
        }
    }
    record
}

/// Prove that `preserve_thinking` changes native history serialization without asking the model
/// to disclose hidden reasoning. The requests differ only in the template control; both must run
/// through the provider, stream a benign exact answer, and report exact prompt-token usage. A
/// larger preserved prompt is the observable native proof that assistant reasoning history was
/// retained.
pub fn measure_preserve_thinking(provider: &dyn TextLlm, case: &ComparisonCase) -> Value {
    let mut preserved_case = case.clone();
    preserved_case.request.preserve_thinking = Some(true);
    let mut stripped_case = case.clone();
    stripped_case.request.preserve_thinking = Some(false);

    let preserved = measure_case(provider, &preserved_case);
    let stripped = measure_case(provider, &stripped_case);
    let preserved_request = request_evidence(&preserved_case.request);
    let stripped_request = request_evidence(&stripped_case.request);
    let history_coverage_passed = preserved_case.request.messages.len() >= 3
        && preserved_case.request.messages == stripped_case.request.messages
        && preserved_case.request.messages.iter().any(|message| {
            message.role == Role::Assistant
                && message
                    .thinking
                    .as_deref()
                    .is_some_and(|thinking| !thinking.trim().is_empty())
        });
    let preserved_prompt_tokens = preserved["output"]["prompt_tokens"].as_u64();
    let stripped_prompt_tokens = stripped["output"]["prompt_tokens"].as_u64();
    let additional_preserved_tokens = preserved_prompt_tokens
        .zip(stripped_prompt_tokens)
        .and_then(|(preserved, stripped)| preserved.checked_sub(stripped));
    let token_proof_passed = additional_preserved_tokens.is_some_and(|delta| delta > 0);
    let evidence_complete =
        preserved["evidence_complete"] == true && stripped["evidence_complete"] == true;
    let quality_passed = preserved["quality_passed"] == true && stripped["quality_passed"] == true;
    let stream_contract_passed =
        preserved["stream_contract_passed"] == true && stripped["stream_contract_passed"] == true;
    let functional_acceptance_passed = evidence_complete
        && quality_passed
        && stream_contract_passed
        && history_coverage_passed
        && token_proof_passed;

    json!({
        "schema_version": 1,
        "case_id": case.id,
        "category": case.category,
        "request": {
            "preserved": preserved_request,
            "stripped": stripped_request,
        },
        "oracle": {
            "kind": "preserve_thinking_history",
            "preserved": preserved["oracle"].clone(),
            "stripped": stripped["oracle"].clone(),
        },
        "status": if evidence_complete { "completed" } else { "incomplete" },
        "evidence_complete": evidence_complete,
        "quality_passed": quality_passed,
        "functional_acceptance_passed": functional_acceptance_passed,
        "stream_contract_passed": stream_contract_passed,
        "history_coverage_passed": history_coverage_passed,
        "prompt_token_proof": {
            "preserved_prompt_tokens": preserved_prompt_tokens,
            "stripped_prompt_tokens": stripped_prompt_tokens,
            "additional_preserved_tokens": additional_preserved_tokens,
            "passed": token_proof_passed,
        },
        "time_to_first_token_seconds": preserved["time_to_first_token_seconds"].clone(),
        "time_to_first_token_unavailable_reason": preserved["time_to_first_token_unavailable_reason"].clone(),
        "total_seconds": preserved["total_seconds"].as_f64().unwrap_or(0.0)
            + stripped["total_seconds"].as_f64().unwrap_or(0.0),
        "prefill_seconds": preserved["prefill_seconds"].as_f64().unwrap_or(0.0)
            + stripped["prefill_seconds"].as_f64().unwrap_or(0.0),
        "decode_seconds": preserved["decode_seconds"].as_f64().unwrap_or(0.0)
            + stripped["decode_seconds"].as_f64().unwrap_or(0.0),
        "events": {
            "preserved": preserved["events"].clone(),
            "stripped": stripped["events"].clone(),
        },
        "output": preserved["output"].clone(),
        "paired_steps": {
            "preserved": preserved,
            "stripped": stripped,
        },
    })
}

/// Execute a complete tool exchange: the model must emit the call, the harness feeds a result tied
/// to that emitted call, and the model must then produce the final answer.
pub fn measure_tool_roundtrip(provider: &dyn TextLlm, case: &ComparisonCase) -> Value {
    let initial = measure_case(provider, case);
    let calls = initial["output"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if initial["evidence_complete"] != true || calls.len() != 1 {
        let mut failed = initial.clone();
        failed["quality_passed"] = json!(false);
        failed["functional_acceptance_passed"] = json!(false);
        failed["functional_failure"] =
            json!("initial generation did not produce one complete structured tool call");
        failed["roundtrip_steps"] = json!([initial]);
        return failed;
    }
    let call = &calls[0];
    let name = call["name"].as_str().unwrap_or_default().to_owned();
    let arguments = call["arguments"].as_object().cloned().unwrap_or_default();
    let city = arguments
        .get("city")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let mut follow_up = text_case(
        "tool_roundtrip_follow_up",
        "tool_roundtrip_acceptance",
        "Use the completed tool result and reply with only the integer temperature.",
        "18",
    );
    follow_up.request.tools = case.request.tools.clone();
    follow_up.request.messages = case.request.messages.clone();
    let mut assistant = Message::assistant(initial["output"]["text"].as_str().unwrap_or_default())
        .with_tool_calls(vec![core_llm::ToolCall::new(name, arguments)]);
    if let Some(thinking) = initial["output"]["thinking"].as_str() {
        assistant = assistant.with_thinking(thinking);
    }
    follow_up.request.messages.extend([
        assistant,
        Message::text(
            Role::Tool,
            json!({"city":city,"temperature_c":18}).to_string(),
        ),
        Message::user("Use the completed tool result and reply with only the integer temperature."),
    ]);
    let final_step = measure_case(provider, &follow_up);
    let evidence_complete =
        initial["evidence_complete"] == true && final_step["evidence_complete"] == true;
    let quality_passed = initial["quality_passed"] == true && final_step["quality_passed"] == true;
    json!({
        "schema_version": 1,
        "case_id": case.id,
        "category": case.category,
        "request": {
            "initial": initial["request"].clone(),
            "follow_up": final_step["request"].clone(),
        },
        "status": if evidence_complete { "completed" } else { "incomplete" },
        "evidence_complete": evidence_complete,
        "quality_passed": quality_passed,
        "functional_acceptance_passed": initial["functional_acceptance_passed"] == true
            && final_step["functional_acceptance_passed"] == true,
        "oracle": {"kind":"tool_roundtrip", "initial":initial["oracle"].clone(), "final":final_step["oracle"].clone()},
        "stream_contract_passed": initial["stream_contract_passed"] == true
            && final_step["stream_contract_passed"] == true,
        "time_to_first_token_seconds": initial["time_to_first_token_seconds"].clone(),
        "time_to_first_token_unavailable_reason": initial["time_to_first_token_unavailable_reason"].clone(),
        "total_seconds": initial["total_seconds"].as_f64().unwrap_or(0.0)
            + final_step["total_seconds"].as_f64().unwrap_or(0.0),
        "prefill_seconds": initial["prefill_seconds"].as_f64().unwrap_or(0.0)
            + final_step["prefill_seconds"].as_f64().unwrap_or(0.0),
        "decode_seconds": initial["decode_seconds"].as_f64().unwrap_or(0.0)
            + final_step["decode_seconds"].as_f64().unwrap_or(0.0),
        "events": {
            "initial": initial["events"].clone(),
            "follow_up": final_step["events"].clone(),
        },
        "output": final_step["output"].clone(),
        "roundtrip_steps": [initial, final_step],
    })
}

fn text_case(id: &str, category: &str, prompt: &str, answer: &str) -> ComparisonCase {
    let mut request = TextLlmRequest::new(vec![Message::user(prompt)], 128);
    request.sampling = Sampling::greedy();
    request.seed = Some(23935);
    request.thinking = ThinkingMode::Disabled;
    ComparisonCase {
        id: id.into(),
        category: category.into(),
        request,
        oracle: AnswerOracle::Exact(answer.into()),
    }
}

fn solid(color: [u8; 3]) -> ImageRef {
    ImageRef::new(64, 64, color.repeat(64 * 64)).expect("fixed RGB fixture")
}

/// Fixed, small diagnostic tasks. All rows use the same greedy/no-thinking budget across models.
/// This is intentionally labeled a diagnostic workload, never vendor benchmark reproduction.
pub fn diagnostic_cases() -> Vec<ComparisonCase> {
    let mut cases = vec![
        text_case("arithmetic","math","Calculate 17 times 23. Reply with only the integer.","391"),
        text_case("logic","reasoning","Four sealed boxes are labeled A, B, C, and D. The key is not in A. If the key is in B, then it is also in C, which is impossible because exactly one box contains it. The key is not in D. Which box contains the key? Reply with only the letter.","C"),
        text_case("ordering","instructions","Sort these names alphabetically: Cy, Ava, Bo. Reply with only the comma-separated names, with no spaces.","Ava,Bo,Cy"),
        text_case("code","code","What integer does the expression sum(n*n for n in range(1,6)) return? Reply with only the integer.","55"),
    ];
    let mut structured = text_case(
        "json",
        "structured",
        "Return exactly one JSON object with key sum and integer value equal to 3 plus 4.",
        "",
    );
    structured.request.constraint = Some(core_llm::Constraint::Json);
    structured.oracle = AnswerOracle::Json(json!({"sum":7}));
    cases.push(structured);
    let mut tool = text_case(
        "tool",
        "tools",
        "Use lookup_weather to look up the weather for Paris.",
        "",
    );
    tool.request.tools = vec![core_llm::ToolSpec::new(
        "lookup_weather",
        "Look up weather for a city",
        json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
    )];
    tool.oracle = AnswerOracle::Tool {
        name: "lookup_weather".into(),
        arguments: json!({"city":"Paris"}),
    };
    cases.push(tool);
    let mut multi_turn = text_case(
        "multi_turn",
        "conversation",
        "What project codename did I give you? Reply with only the codename.",
        "ORCHID-9",
    );
    multi_turn.request.messages = vec![
        Message::system("Retain exact user-provided identifiers across turns."),
        Message::user("The project codename is ORCHID-9."),
        Message::assistant("Understood."),
        Message::user("What project codename did I give you? Reply with only the codename."),
    ];
    cases.push(multi_turn);
    for (id, colors) in [
        ("image", vec![[255, 0, 0]]),
        ("multi_image", vec![[255, 0, 0], [0, 0, 255]]),
    ] {
        let mut case = text_case(id,"vision","Name the solid color in each image, in image order. Use only color names separated by commas.","");
        let mut content = colors
            .iter()
            .map(|color| Content::Image(solid(*color)))
            .collect::<Vec<_>>();
        content.extend(case.request.messages.remove(0).content);
        case.request.messages = vec![Message {
            role: Role::User,
            content,
            thinking: None,
            tool_calls: vec![],
        }];
        case.oracle = AnswerOracle::Colors(if colors.len() == 1 {
            vec!["red".into()]
        } else {
            vec!["red".into(), "blue".into()]
        });
        cases.push(case);
    }
    for (id, colors, names) in [
        ("video_forward", [[255, 0, 0], [0, 0, 255]], ["red", "blue"]),
        ("video_reverse", [[0, 0, 255], [255, 0, 0]], ["blue", "red"]),
    ] {
        let mut case = text_case(id,"video","Name the first color and then the last color in this video. Use only color names separated by a comma.","");
        let video = VideoRef::new(
            vec![
                solid(colors[0]),
                solid(colors[0]),
                solid(colors[1]),
                solid(colors[1]),
            ],
            vec![0.0, 0.5, 1.0, 1.5],
        )
        .expect("fixed video");
        case.request.messages[0]
            .content
            .insert(0, Content::Video(video));
        case.oracle = AnswerOracle::Colors(names.iter().map(|value| (*value).into()).collect());
        cases.push(case);
    }
    cases
}

/// Capability acceptance cases that are selected only for models advertising the corresponding
/// controls. They stay outside [`diagnostic_cases`] so the matched comparison workload remains
/// byte-for-byte equal across the parent, compressed model, and baseline.
pub fn capability_acceptance_cases() -> Vec<ComparisonCase> {
    let diagnostics = diagnostic_cases();
    let by_id = |id: &str| {
        diagnostics
            .iter()
            .find(|case| case.id == id)
            .cloned()
            .unwrap_or_else(|| panic!("missing fixed diagnostic case {id}"))
    };
    let mut cases = Vec::new();
    for (id, effort) in [
        ("reasoning_low", core_llm::ReasoningEffort::Low),
        ("reasoning_medium", core_llm::ReasoningEffort::Medium),
        ("reasoning_xhigh", core_llm::ReasoningEffort::XHigh),
    ] {
        let mut case = by_id("arithmetic");
        case.id = id.into();
        case.category = "reasoning_acceptance".into();
        case.request.thinking = ThinkingMode::Enabled;
        case.request.reasoning_effort = Some(effort);
        case.request.max_new_tokens = 512;
        cases.push(case);
    }

    let mut preserve = text_case(
        "preserve_thinking",
        "reasoning_history_acceptance",
        "Reply with only OK.",
        "OK",
    );
    preserve.request.messages = vec![
        Message::user("Acknowledge this setup for a later turn."),
        Message::assistant("Acknowledged.")
            .with_thinking("This private history marker must stay in serialized reasoning."),
        Message::user("Reply with only OK."),
    ];
    preserve.request.thinking = ThinkingMode::Disabled;
    preserve.request.preserve_thinking = Some(true);
    preserve.request.max_new_tokens = 32;
    cases.push(preserve);

    let weather = core_llm::ToolSpec::new(
        "lookup_weather",
        "Look up weather for a city",
        json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
    );
    let mut tool_roundtrip = text_case(
        "tool_roundtrip",
        "tool_roundtrip_acceptance",
        "Use lookup_weather to look up the weather for Paris.",
        "",
    );
    tool_roundtrip.request.tools = vec![weather];
    tool_roundtrip.oracle = AnswerOracle::Tool {
        name: "lookup_weather".into(),
        arguments: json!({"city":"Paris"}),
    };
    cases.push(tool_roundtrip);

    let mut json_thinking = by_id("json");
    json_thinking.id = "json_thinking".into();
    json_thinking.category = "structured_reasoning_acceptance".into();
    json_thinking.request.thinking = ThinkingMode::Enabled;
    json_thinking.request.reasoning_effort = Some(core_llm::ReasoningEffort::Medium);
    json_thinking.request.max_new_tokens = 512;
    cases.push(json_thinking);

    let mut mtp = by_id("arithmetic");
    mtp.id = "mtp_greedy".into();
    mtp.category = "mtp_acceptance".into();
    mtp.request.mtp = MtpMode::Enabled { draft_tokens: 3 };
    cases.push(mtp);

    let mut mtp_json = by_id("json");
    mtp_json.id = "mtp_json".into();
    mtp_json.category = "mtp_structured_acceptance".into();
    mtp_json.request.mtp = MtpMode::Enabled { draft_tokens: 3 };
    mtp_json.request.thinking = ThinkingMode::Enabled;
    mtp_json.request.reasoning_effort = Some(core_llm::ReasoningEffort::Medium);
    mtp_json.request.max_new_tokens = 512;
    cases.push(mtp_json);

    for (source, id, category) in [
        ("image", "mtp_image", "mtp_vision_acceptance"),
        ("video_forward", "mtp_video", "mtp_video_acceptance"),
    ] {
        let mut case = by_id(source);
        case.id = id.into();
        case.category = category.into();
        case.request.mtp = MtpMode::Enabled { draft_tokens: 3 };
        cases.push(case);
    }
    cases
}

/// Controlled context-growth case. Repetitions are not claimed to be tokens: actual prompt-token
/// counts must come from the provider's returned usage and the declared context limit is recorded.
pub fn context_case(repetitions: usize) -> ComparisonCase {
    let padding = "A robin rests near the old oak tree. ".repeat(repetitions);
    text_case(&format!("context_{repetitions}"),"context",
        &format!("{padding}The secret identifier is NEBULA-47.{padding}\nWhat is the secret identifier? Reply with only the identifier."),"NEBULA-47")
}

/// Required-env entrypoint for an explicitly selected ignored backend test. Each invocation loads
/// one model once, so a process-boundary memory monitor can attribute residency to that model.
/// The callback supplies backend-native memory counters; an empty object is not a memory pass.
pub fn run_environment(
    load: impl FnOnce(&core_llm::LoadSpec) -> core_llm::Result<Box<dyn TextLlm>>,
    mut memory: impl FnMut() -> Value,
) -> Result<(), String> {
    fn required(name: &str) -> Result<String, String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("required comparison environment {name} is missing"))
    }
    let model_path = required("BONSAI_COMPARISON_MODEL_PATH")?;
    let model_id = required("BONSAI_COMPARISON_MODEL_ID")?;
    let revision = required("BONSAI_COMPARISON_MODEL_REVISION")?;
    let runtime_sha = required("BONSAI_COMPARISON_RUNTIME_SHA")?;
    for (name, value) in [("model revision", &revision), ("runtime SHA", &runtime_sha)] {
        if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("{name} must be an immutable 40-character commit"));
        }
    }
    let output_path = std::path::PathBuf::from(required("BONSAI_COMPARISON_OUTPUT")?);
    if output_path.exists() {
        return Err("comparison output already exists; use a fresh run directory".into());
    }
    let mut cases = diagnostic_cases();
    for repetitions in [64, 512, 2048] {
        cases.push(context_case(repetitions));
    }
    cases.extend(capability_acceptance_cases());
    let selected: Vec<String> = serde_json::from_str(&required("BONSAI_COMPARISON_CASES")?)
        .map_err(|error| format!("comparison cases must be a JSON string array: {error}"))?;
    if selected.is_empty()
        || selected
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != selected.len()
    {
        return Err("comparison cases must be nonempty and unique".into());
    }
    let mut selected_cases = Vec::new();
    for id in &selected {
        selected_cases.push(
            cases
                .iter()
                .find(|case| case.id == *id)
                .cloned()
                .ok_or_else(|| format!("unknown comparison case {id}"))?,
        );
    }
    let mut report = json!({"schema_version":1,"suite":"bonsai-native-diagnostic-v1",
        "model_id":model_id,"model_revision":revision,"model_path":model_path,
        "runtime_sha":runtime_sha,"case_ids":selected,"status":"running",
        "claims":{"vendor_benchmark_reproduction":false,
            "vendor_quality_retention":false,
            "scope":"fixed native diagnostic cases only"},
        "native_memory_before_load":memory()});
    let start = Instant::now();
    let loaded = load(&core_llm::LoadSpec::dense(&model_path));
    report["load_seconds"] = json!(start.elapsed().as_secs_f64());
    let result = match loaded {
        Err(error) => {
            report["status"] = json!("load_failed");
            report["error"] = json!(error.to_string());
            Err(error.to_string())
        }
        Ok(provider) => {
            let descriptor = provider.descriptor();
            let caps = &descriptor.capabilities;
            report["provider"] = json!({"id":descriptor.id,"family":descriptor.family,"backend":descriptor.backend,
                "max_context_tokens":caps.max_context_tokens,"supports_vision":caps.supports_vision,
                "supports_video":caps.supports_video,"supports_thinking":caps.supports_thinking,
                "supports_reasoning_effort":caps.supports_reasoning_effort,
                "reasoning_efforts":caps.reasoning_efforts.iter().map(|e|e.as_str()).collect::<Vec<_>>(),
                "model_sampling_defaults":caps.model_sampling_defaults.map(|d|json!({
                    "thinking":{"temperature":d.thinking.temperature,"top_p":d.thinking.top_p,
                        "top_k":d.thinking.top_k,"presence_penalty":d.thinking.presence_penalty},
                    "non_thinking":{"temperature":d.non_thinking.temperature,"top_p":d.non_thinking.top_p,
                        "top_k":d.non_thinking.top_k,"presence_penalty":d.non_thinking.presence_penalty}})),
                "supports_preserve_thinking":caps.supports_preserve_thinking,
                "mtp":caps.mtp.map(|mtp|json!({"max_draft_tokens":mtp.max_draft_tokens,"recommended_draft_tokens":mtp.recommended_draft_tokens}))});
            report["native_memory_after_load"] = memory();
            let memory_override = "SCENEWORKS_LLM_AVAILABLE_MEMORY_BYTES";
            let previous_override = std::env::var_os(memory_override);
            // SAFETY: this explicitly selected ignored entrypoint runs with one test thread and no
            // provider work in parallel. The prior process value is restored before any other case.
            unsafe { std::env::set_var(memory_override, "1") };
            let resource_case = &selected_cases[0];
            let resource_record = measure_case(provider.as_ref(), resource_case);
            match previous_override {
                Some(value) => {
                    // SAFETY: same single-threaded scope described above.
                    unsafe { std::env::set_var(memory_override, value) };
                }
                None => {
                    // SAFETY: same single-threaded scope described above.
                    unsafe { std::env::remove_var(memory_override) };
                }
            }
            let resource_rejected = resource_record["status"] == "failed"
                && resource_record["error"].as_str().is_some_and(|error| {
                    error.contains("request requires an estimated")
                        && error
                            .contains("bytes of native workspace but only 1 bytes are available")
                });
            report["resource_admission"] = json!({
                "evidence_complete": resource_rejected,
                "paired_case_id": resource_case.id,
                "available_memory_override_bytes": 1,
                "record": resource_record,
            });
            let mut records = Vec::new();
            for case in &selected_cases {
                let mut record = if case.id == "tool_roundtrip" {
                    measure_tool_roundtrip(provider.as_ref(), case)
                } else if case.id == "preserve_thinking" {
                    measure_preserve_thinking(provider.as_ref(), case)
                } else {
                    measure_case(provider.as_ref(), case)
                };
                record["native_memory_after_case"] = memory();
                records.push(record);
            }
            // Bind the low-budget rejection to the identical workload's actual expanded token
            // count at the ordinary operational budget, rather than claiming bytes are tokens.
            let paired_prompt_tokens = records[0]["output"]["prompt_tokens"].as_u64();
            let within_context = paired_prompt_tokens.is_some_and(|tokens| {
                tokens
                    .checked_add(u64::from(resource_case.request.max_new_tokens))
                    .is_some_and(|total| total <= caps.max_context_tokens as u64)
            });
            report["resource_admission"]["paired_prompt_tokens"] = json!(paired_prompt_tokens);
            report["resource_admission"]["declared_context_tokens"] =
                json!(caps.max_context_tokens);
            report["resource_admission"]["architecturally_within_context"] = json!(within_context);
            report["resource_admission"]["evidence_complete"] =
                json!(resource_rejected && within_context);
            // Exercise the provider's real context admission boundary without estimating token
            // count from text repetitions. A one-turn prompt plus a generation budget equal to the
            // declared complete window must be rejected before model execution because their sum
            // cannot fit. Successful context rows above retain the provider's exact prompt count.
            let context_admission = if caps.max_context_tokens == 0
                || caps.max_context_tokens > u32::MAX as usize
            {
                json!({"evidence_complete":false,"reason":"provider has no representable context limit"})
            } else {
                let mut probe = text_case(
                    "context_admission",
                    "context_admission",
                    "Reply with only OK.",
                    "OK",
                );
                probe.request.max_new_tokens = caps.max_context_tokens as u32;
                let record = measure_case(provider.as_ref(), &probe);
                let rejected = record["status"] == "failed"
                    && record["error"]
                        .as_str()
                        .is_some_and(|error| error.to_lowercase().contains("context"));
                json!({"evidence_complete":rejected,"declared_context_tokens":caps.max_context_tokens,
                    "requested_max_new_tokens":probe.request.max_new_tokens,"record":record})
            };
            let complete = records
                .iter()
                .all(|record| record["evidence_complete"] == true)
                && context_admission["evidence_complete"] == true
                && report["resource_admission"]["evidence_complete"] == true;
            report["cases"] = json!(records);
            report["context_admission"] = context_admission;
            report["status"] = json!(if complete { "completed" } else { "incomplete" });
            drop(provider);
            report["native_memory_after_unload"] = memory();
            if complete {
                Ok(())
            } else {
                Err("one or more native cases failed or lack required phase/stream evidence".into())
            }
        }
    };
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .map_err(|error| format!("create comparison evidence: {error}"))?;
    serde_json::to_writer_pretty(file, &report)
        .map_err(|error| format!("write comparison evidence: {error}"))?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{FinishReason, GenerationTimings, TextLlmDescriptor, TextLlmOutput, Usage};
    use std::sync::Mutex;
    use std::time::Duration;

    struct Stub {
        descriptor: TextLlmDescriptor,
        timings: bool,
        broken: bool,
        fail: bool,
    }
    impl TextLlm for Stub {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
        }
        fn validate(&self, _: &TextLlmRequest) -> core_llm::Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _: &TextLlmRequest,
            emit: &mut dyn FnMut(StreamEvent),
        ) -> core_llm::Result<TextLlmOutput> {
            if self.fail {
                return Err(core_llm::Error::InvalidRequest("fixture failure".into()));
            }
            let usage = Usage {
                prompt_tokens: 10,
                generated_tokens: 1,
            };
            emit(StreamEvent::Token {
                id: 7,
                text: if self.broken { "wrong" } else { "391" }.into(),
                index: 0,
                channel: Channel::Content,
            });
            emit(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage,
            });
            Ok(TextLlmOutput {
                text: "391".into(),
                usage,
                finish_reason: Some(FinishReason::Stop),
                timings: self.timings.then_some(GenerationTimings {
                    prefill: Duration::from_millis(4),
                    decode: Duration::from_millis(2),
                }),
                ..Default::default()
            })
        }
    }
    fn stub() -> Stub {
        Stub {
            descriptor: TextLlmDescriptor {
                id: "fixture".into(),
                family: "fixture".into(),
                backend: "cpu".into(),
                capabilities: Default::default(),
            },
            timings: true,
            broken: false,
            fail: false,
        }
    }

    #[test]
    fn missing_timing_and_broken_stream_cannot_be_complete() {
        let case = &diagnostic_cases()[0];
        assert_eq!(measure_case(&stub(), case)["evidence_complete"], true);
        let mut provider = stub();
        provider.timings = false;
        let output = provider.generate(&case.request, &mut |_| {}).unwrap();
        assert!(
            output.timings.is_none(),
            "the provider output itself must report missing native phase timings"
        );
        let record = measure_case(&provider, case);
        assert_eq!(record["evidence_complete"], false);
        provider.timings = true;
        provider.broken = true;
        assert_eq!(
            measure_case(&provider, case)["stream_contract_passed"],
            false
        );
        provider.broken = false;
        provider.fail = true;
        assert_eq!(measure_case(&provider, case)["status"], "failed");
    }
    #[test]
    fn quality_failure_is_retained_without_invented_acceptance_threshold() {
        let mut case = diagnostic_cases().remove(0);
        case.oracle = AnswerOracle::Exact("different".into());
        let record = measure_case(&stub(), &case);
        assert_eq!(record["evidence_complete"], true);
        assert_eq!(record["quality_passed"], false);
        assert_eq!(record["output"]["text"], "391");
    }
    #[test]
    fn frozen_cases_retain_discriminating_media_order_and_raw_pixels() {
        let cases = diagnostic_cases();
        let forward = cases
            .iter()
            .find(|case| case.id == "video_forward")
            .unwrap();
        let reverse = cases
            .iter()
            .find(|case| case.id == "video_reverse")
            .unwrap();
        assert_ne!(
            request_evidence(&forward.request),
            request_evidence(&reverse.request)
        );
        let request = request_evidence(&forward.request);
        assert_eq!(
            request["messages"][0]["content"][0]["frames"][0]["rgb8"][0],
            255
        );
        assert_eq!(
            request["messages"][0]["content"][0]["timestamps"],
            json!([0.0, 0.5, 1.0, 1.5])
        );
    }

    struct ToolStub {
        descriptor: TextLlmDescriptor,
        requests: Mutex<Vec<TextLlmRequest>>,
    }

    impl TextLlm for ToolStub {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
        }

        fn validate(&self, _: &TextLlmRequest) -> core_llm::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            request: &TextLlmRequest,
            emit: &mut dyn FnMut(StreamEvent),
        ) -> core_llm::Result<TextLlmOutput> {
            self.requests.lock().unwrap().push(request.clone());
            let usage = Usage {
                prompt_tokens: 10,
                generated_tokens: 1,
            };
            let mut output = TextLlmOutput {
                usage,
                finish_reason: Some(FinishReason::Stop),
                timings: Some(GenerationTimings {
                    prefill: Duration::from_millis(1),
                    decode: Duration::from_millis(1),
                }),
                ..Default::default()
            };
            if request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool)
            {
                output.text = "18".into();
                emit(StreamEvent::Token {
                    id: 18,
                    text: "18".into(),
                    index: 0,
                    channel: Channel::Content,
                });
            } else {
                let mut arguments = serde_json::Map::new();
                arguments.insert("city".into(), json!("Paris"));
                output.tool_calls = vec![core_llm::ToolCall::new("lookup_weather", arguments)];
            }
            emit(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage,
            });
            Ok(output)
        }
    }

    #[test]
    fn tool_roundtrip_uses_the_models_call_before_final_generation() {
        let provider = ToolStub {
            descriptor: stub().descriptor,
            requests: Mutex::new(Vec::new()),
        };
        let case = capability_acceptance_cases()
            .into_iter()
            .find(|case| case.id == "tool_roundtrip")
            .unwrap();
        let record = measure_tool_roundtrip(&provider, &case);
        assert_eq!(record["evidence_complete"], true);
        assert_eq!(record["quality_passed"], true);
        assert_eq!(record["output"]["text"], "18");
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[1]
            .messages
            .iter()
            .any(|message| message.role == Role::Assistant && !message.tool_calls.is_empty()));
        assert!(requests[1]
            .messages
            .iter()
            .any(|message| message.role == Role::Tool));
    }

    #[test]
    fn correct_answers_cannot_hide_ignored_reasoning_or_mtp_controls() {
        let cases = capability_acceptance_cases();
        for id in [
            "reasoning_low",
            "reasoning_medium",
            "reasoning_xhigh",
            "mtp_greedy",
        ] {
            let case = cases.iter().find(|case| case.id == id).unwrap();
            let record = measure_case(&stub(), case);
            assert_eq!(record["quality_passed"], true);
            assert_eq!(record["evidence_complete"], true);
            assert_eq!(record["functional_acceptance_passed"], false, "{id}");
        }
        let tool = cases
            .iter()
            .find(|case| case.id == "tool_roundtrip")
            .unwrap();
        let record = measure_tool_roundtrip(&stub(), tool);
        assert_eq!(record["evidence_complete"], true);
        assert_eq!(record["functional_acceptance_passed"], false);
    }

    struct PreserveStub {
        descriptor: TextLlmDescriptor,
        honor_control: bool,
    }

    impl TextLlm for PreserveStub {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
        }

        fn validate(&self, _: &TextLlmRequest) -> core_llm::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            request: &TextLlmRequest,
            emit: &mut dyn FnMut(StreamEvent),
        ) -> core_llm::Result<TextLlmOutput> {
            let preserved = request.preserve_thinking == Some(true) && self.honor_control;
            let usage = Usage {
                prompt_tokens: if preserved { 14 } else { 10 },
                generated_tokens: 1,
            };
            emit(StreamEvent::Token {
                id: 1,
                text: "OK".into(),
                index: 0,
                channel: Channel::Content,
            });
            emit(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage,
            });
            Ok(TextLlmOutput {
                text: "OK".into(),
                usage,
                finish_reason: Some(FinishReason::Stop),
                timings: Some(GenerationTimings {
                    prefill: Duration::from_millis(1),
                    decode: Duration::from_millis(1),
                }),
                ..Default::default()
            })
        }
    }

    #[test]
    fn preserve_thinking_acceptance_uses_paired_native_prompt_token_proof() {
        let case = capability_acceptance_cases()
            .into_iter()
            .find(|case| case.id == "preserve_thinking")
            .unwrap();
        let provider = PreserveStub {
            descriptor: stub().descriptor,
            honor_control: true,
        };
        let record = measure_preserve_thinking(&provider, &case);
        assert_eq!(record["status"], "completed");
        assert_eq!(record["quality_passed"], true);
        assert_eq!(record["stream_contract_passed"], true);
        assert_eq!(record["history_coverage_passed"], true);
        assert_eq!(record["prompt_token_proof"]["preserved_prompt_tokens"], 14);
        assert_eq!(record["prompt_token_proof"]["stripped_prompt_tokens"], 10);
        assert_eq!(
            record["prompt_token_proof"]["additional_preserved_tokens"],
            4
        );
        assert_eq!(record["prompt_token_proof"]["passed"], true);
        assert_eq!(record["functional_acceptance_passed"], true);
        assert_eq!(record["request"]["preserved"]["preserve_thinking"], true);
        assert_eq!(record["request"]["stripped"]["preserve_thinking"], false);
        assert_eq!(record["paired_steps"]["preserved"]["output"]["text"], "OK");
        assert_eq!(record["paired_steps"]["stripped"]["output"]["text"], "OK");

        let ignored = PreserveStub {
            descriptor: stub().descriptor,
            honor_control: false,
        };
        let ignored_record = measure_preserve_thinking(&ignored, &case);
        assert_eq!(ignored_record["evidence_complete"], true);
        assert_eq!(ignored_record["quality_passed"], true);
        assert_eq!(ignored_record["prompt_token_proof"]["passed"], false);
        assert_eq!(ignored_record["functional_acceptance_passed"], false);
    }

    #[test]
    fn capability_acceptance_cases_cover_reasoning_json_mtp_and_media() {
        let cases = capability_acceptance_cases();
        let ids = cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "reasoning_low",
                "reasoning_medium",
                "reasoning_xhigh",
                "preserve_thinking",
                "tool_roundtrip",
                "json_thinking",
                "mtp_greedy",
                "mtp_json",
                "mtp_image",
                "mtp_video",
            ]
        );
        for (id, effort) in [
            ("reasoning_low", core_llm::ReasoningEffort::Low),
            ("reasoning_medium", core_llm::ReasoningEffort::Medium),
            ("reasoning_xhigh", core_llm::ReasoningEffort::XHigh),
        ] {
            let case = cases.iter().find(|case| case.id == id).unwrap();
            assert_eq!(case.request.thinking, ThinkingMode::Enabled);
            assert_eq!(case.request.reasoning_effort, Some(effort));
        }
        let preserve = cases
            .iter()
            .find(|case| case.id == "preserve_thinking")
            .unwrap();
        assert_eq!(preserve.request.preserve_thinking, Some(true));
        assert_eq!(preserve.request.thinking, ThinkingMode::Disabled);
        assert_eq!(
            preserve.oracle.as_json(),
            json!({"kind":"exact","value":"OK"})
        );
        assert!(preserve
            .request
            .messages
            .iter()
            .any(|message| message.thinking.is_some()));
        for id in ["mtp_greedy", "mtp_json", "mtp_image", "mtp_video"] {
            assert!(matches!(
                cases.iter().find(|case| case.id == id).unwrap().request.mtp,
                MtpMode::Enabled { draft_tokens: 3 }
            ));
        }
        let json = cases
            .iter()
            .find(|case| case.id == "json_thinking")
            .unwrap();
        assert_eq!(json.request.thinking, ThinkingMode::Enabled);
        assert!(json.request.constraint.is_some());
        let mtp_json = cases.iter().find(|case| case.id == "mtp_json").unwrap();
        assert!(matches!(mtp_json.request.mtp, MtpMode::Enabled { .. }));
        assert_eq!(mtp_json.request.thinking, ThinkingMode::Enabled);
        assert!(mtp_json.request.constraint.is_some());
        assert!(cases
            .iter()
            .find(|case| case.id == "mtp_image")
            .unwrap()
            .request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|content| matches!(content, Content::Image(_))));
        assert!(cases
            .iter()
            .find(|case| case.id == "mtp_video")
            .unwrap()
            .request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|content| matches!(content, Content::Video(_))));
    }
}
