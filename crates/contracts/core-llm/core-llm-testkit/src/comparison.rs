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
            "top_k":request.sampling.top_k,"repetition_penalty":request.sampling.repetition_penalty,
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
            record["quality_passed"] = json!(case.oracle.matches(&output));
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
        text_case("logic","reasoning","All squares are rectangles. Does that imply all rectangles are squares? Reply with only No or Yes.","No"),
        text_case("ordering","instructions","Sort these names alphabetically: Cy, Ava, Bo. Reply with only the comma-separated names, with no spaces.","Ava,Bo,Cy"),
        text_case("python","code","What integer does Python sum(n*n for n in range(1,6)) return? Reply with only the integer.","55"),
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
    let mut reasoning = cases[0].clone();
    reasoning.id = "reasoning_low".into();
    reasoning.category = "reasoning_enabled".into();
    reasoning.request.thinking = ThinkingMode::Enabled;
    reasoning.request.reasoning_effort = Some(core_llm::ReasoningEffort::Low);
    reasoning.request.max_new_tokens = 512;
    cases.push(reasoning);
    let mut mtp = cases[0].clone();
    mtp.id = "mtp_greedy".into();
    mtp.category = "mtp".into();
    mtp.request.mtp = MtpMode::Enabled { draft_tokens: 3 };
    cases.push(mtp);
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
        "vendor_benchmark_reproduction":false,"native_memory_before_load":memory()});
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
            let mut records = Vec::new();
            for case in &selected_cases {
                let mut record = measure_case(provider.as_ref(), case);
                record["native_memory_after_case"] = memory();
                records.push(record);
            }
            let complete = records
                .iter()
                .all(|record| record["evidence_complete"] == true);
            report["cases"] = json!(records);
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
}
