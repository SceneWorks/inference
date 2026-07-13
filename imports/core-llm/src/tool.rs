//! Tool ("function") calling: the request-side tool offer, the parsed / round-trip tool call, and a
//! streaming parser that lifts a model's `<tool_call>` blocks out of its answer text.
//!
//! Mirrors the `transformers` tool-calling contract: a request carries `tools` (offered functions)
//! that the chat template renders into the prompt, and the model replies with tool calls the host
//! parses back into structure. Like reasoning segmentation ([`crate::thinking`]) this runs on the
//! *decoded text* (the `<tool_call>` markers need not align to token boundaries), so it is
//! backend-neutral host policy and is fed the same incremental detokenized deltas a provider already
//! produces — specifically the answer-channel ([`Channel::Content`](crate::Channel)) text, after
//! reasoning has been split off.
//!
//! Two on-the-wire call formats are recognised inside a `<tool_call>…</tool_call>` block:
//! - **Qwen3.6 XML** — `<function=name><parameter=key>\nvalue\n</parameter>…</function>`, the format
//!   the Qwen3.6 chat template instructs the model to emit.
//! - **JSON / Hermes** — `{"name": …, "arguments": {…}}`, the Qwen2.5 / Hermes convention.
//!
//! Argument values emitted by the XML form are raw text; they are coerced to their declared
//! JSON-Schema type (number / integer / boolean / array / object → JSON-parsed; string / unknown →
//! kept verbatim) when the calling [`ToolSpec`] is known, so the parsed result is genuinely typed.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::thinking::max_partial_suffix;

/// A function offered to the model (request input). Rendered into a chat template's `tools` context
/// as the OpenAI function-tool JSON the model is trained on (see [`to_template_json`]).
///
/// [`to_template_json`]: ToolSpec::to_template_json
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    /// The function name the model calls.
    pub name: String,
    /// A natural-language description of what the function does and when to use it.
    pub description: String,
    /// JSON-Schema for the call arguments — typically an `{"type":"object","properties":{…}}` object
    /// schema. Carried as raw JSON so the contract stays free of a schema type; it is rendered into
    /// the prompt verbatim and consulted to type-coerce parsed arguments.
    pub parameters: Value,
}

impl ToolSpec {
    /// Construct a tool spec.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// The `{"type":"function","function":{"name","description","parameters"}}` JSON a chat template
    /// renders (matching `transformers` `tools=`). Key order is insertion-faithful (serde_json
    /// `preserve_order`) so the rendered tool section matches the reference byte-for-byte.
    pub fn to_template_json(&self) -> Value {
        let mut func = Map::new();
        func.insert("name".into(), Value::String(self.name.clone()));
        func.insert("description".into(), Value::String(self.description.clone()));
        func.insert("parameters".into(), self.parameters.clone());
        let mut obj = Map::new();
        obj.insert("type".into(), Value::String("function".into()));
        obj.insert("function".into(), Value::Object(func));
        Value::Object(obj)
    }

    /// The declared JSON-Schema `type` of each named parameter, for output coercion. Reads
    /// `parameters.properties.<name>.type`; a missing or non-object schema yields an empty map (so
    /// every argument is then treated as a string).
    fn param_types(&self) -> HashMap<String, String> {
        let mut out = HashMap::new();
        if let Some(props) = self.parameters.get("properties").and_then(Value::as_object) {
            for (k, schema) in props {
                if let Some(t) = schema.get("type").and_then(Value::as_str) {
                    out.insert(k.clone(), t.to_string());
                }
            }
        }
        out
    }
}

/// A tool / function call: the model's output (parsed from a `<tool_call>` block by
/// [`ToolCallSegmenter`]) and the multi-turn input dual — a prior assistant turn's call, carried on
/// [`Message::tool_calls`](crate::Message::tool_calls) and re-rendered by the chat template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCall {
    /// The called function's name.
    pub name: String,
    /// The call arguments as an ordered name→value map. Parsed-output values are coerced to their
    /// declared schema type when known, else kept as the raw emitted string; insertion order is
    /// preserved (serde_json `preserve_order`) so a round-tripped call re-renders faithfully.
    pub arguments: Map<String, Value>,
}

impl ToolCall {
    /// Construct a tool call.
    pub fn new(name: impl Into<String>, arguments: Map<String, Value>) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    /// The `{"type":"function","function":{"name","arguments"}}` JSON a chat template renders for a
    /// prior assistant turn's tool call (the round-trip dual of parsing).
    pub fn to_template_json(&self) -> Value {
        let mut func = Map::new();
        func.insert("name".into(), Value::String(self.name.clone()));
        func.insert("arguments".into(), Value::Object(self.arguments.clone()));
        let mut obj = Map::new();
        obj.insert("type".into(), Value::String("function".into()));
        obj.insert("function".into(), Value::Object(func));
        Value::Object(obj)
    }
}

/// The marker that opens a tool-call block.
const OPEN: &str = "<tool_call>";
/// The marker that closes a tool-call block.
const CLOSE: &str = "</tool_call>";

/// A streaming parser that lifts `<tool_call>…</tool_call>` blocks out of a model's answer text.
///
/// Construct with the request's [`ToolSpec`]s (for argument type coercion), then feed each decoded
/// answer-channel delta to [`push`](ToolCallSegmenter::push): it returns the plain-content runs to
/// stream (the answer with tool-call blocks removed) and accumulates the parsed [`ToolCall`]s,
/// recoverable with [`take_calls`](ToolCallSegmenter::take_calls). It holds back the longest suffix
/// that could still begin the next `<tool_call>` marker, so a marker split across deltas is matched.
/// Call [`flush`](ToolCallSegmenter::flush) once when generation ends to release any held-back tail.
///
/// It starts outside a block and toggles on each fully-seen marker (`<tool_call>` while outside,
/// `</tool_call>` while inside), so it handles several calls in one stream; a `</tool_call>` seen
/// outside a block is ordinary content.
#[derive(Clone, Debug)]
pub struct ToolCallSegmenter {
    /// Declared parameter types per tool name, for output coercion.
    schemas: HashMap<String, HashMap<String, String>>,
    /// Whether we are currently inside a `<tool_call>` block.
    in_call: bool,
    /// Decoded content received but not yet emitted — the tail that might still begin `<tool_call>`.
    pending: String,
    /// The accumulated body of the current `<tool_call>` block (between the markers).
    body: String,
    /// Parsed calls, in order.
    calls: Vec<ToolCall>,
}

impl ToolCallSegmenter {
    /// Build a segmenter that coerces arguments against `tools`' declared schemas. Pass `&[]` to keep
    /// every argument as a raw string (no schema-typed coercion).
    pub fn new(tools: &[ToolSpec]) -> Self {
        let schemas = tools.iter().map(|t| (t.name.clone(), t.param_types())).collect();
        Self {
            schemas,
            in_call: false,
            pending: String::new(),
            body: String::new(),
            calls: Vec::new(),
        }
    }

    /// Whether the segmenter is currently inside a `<tool_call>` block.
    pub fn in_call(&self) -> bool {
        self.in_call
    }

    /// Feed the next decoded answer-channel `delta`. Returns the plain-content runs now safe to emit,
    /// in order, with tool-call blocks removed. A single delta may cross block boundaries (so it can
    /// produce several content runs) or only extend a partial marker / a block body (none). Empty
    /// runs are never returned.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        if self.in_call {
            self.body.push_str(delta);
        } else {
            self.pending.push_str(delta);
        }
        let mut out: Vec<String> = Vec::new();
        loop {
            if self.in_call {
                // Inside a block: consume up to the close marker, parse the body, return to content.
                let Some(pos) = self.body.find(CLOSE) else {
                    break; // still accumulating this block's body
                };
                let block = self.body[..pos].to_string();
                let rest = self.body[pos + CLOSE.len()..].to_string();
                self.body.clear();
                if let Some(call) = self.parse_block(&block) {
                    self.calls.push(call);
                }
                // Text after the close belongs to content; re-scan it for a following block.
                self.pending.push_str(&rest);
                self.in_call = false;
                continue;
            }
            // Outside a block: emit content up to the open marker, then enter the block.
            if let Some(pos) = self.pending.find(OPEN) {
                if pos > 0 {
                    out.push(self.pending[..pos].to_string());
                }
                let rest = self.pending[pos + OPEN.len()..].to_string();
                self.pending.clear();
                self.body.push_str(&rest);
                self.in_call = true;
                continue;
            }
            // No full open marker: hold back the longest suffix that is a proper prefix of it (it
            // might complete on a later delta); emit everything before it as content.
            let hold = max_partial_suffix(&self.pending, OPEN);
            let safe = self.pending.len() - hold;
            if safe > 0 {
                out.push(self.pending[..safe].to_string());
                self.pending.drain(..safe);
            }
            break;
        }
        out
    }

    /// Release any held-back tail when generation finishes. A held partial-`<tool_call>` prefix that
    /// never completed is content; an unterminated block (the model stopped mid-call) is parsed
    /// best-effort, and if that fails its raw text is surfaced as content so nothing is silently lost.
    /// Call once, at end of generation. Returns the content runs (possibly empty).
    pub fn flush(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if self.in_call {
            let body = std::mem::take(&mut self.body);
            if let Some(call) = self.parse_block(&body) {
                self.calls.push(call);
            } else if !body.is_empty() {
                out.push(format!("{OPEN}{body}"));
            }
            self.in_call = false;
        }
        if !self.pending.is_empty() {
            out.push(std::mem::take(&mut self.pending));
        }
        out
    }

    /// The parsed calls so far.
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// Take the parsed calls, leaving the segmenter holding none.
    pub fn take_calls(&mut self) -> Vec<ToolCall> {
        std::mem::take(&mut self.calls)
    }

    /// Parse a `<tool_call>` block body (markers already stripped): the JSON / Hermes form when it
    /// begins with `{`, else the Qwen3.6 XML form. Returns `None` for an empty / unparsable body.
    fn parse_block(&self, block: &str) -> Option<ToolCall> {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            return None;
        }
        let call = if trimmed.starts_with('{') {
            parse_json_call(trimmed)?
        } else {
            parse_xml_call(trimmed)?
        };
        Some(self.coerce(call))
    }

    /// Coerce each raw-string argument to its declared schema type (number / integer / boolean /
    /// array / object → JSON-parsed; string / unknown / unknown-tool → kept as-is). A value that does
    /// not parse as the declared type is left as the raw string rather than dropped.
    fn coerce(&self, mut call: ToolCall) -> ToolCall {
        let Some(types) = self.schemas.get(&call.name) else {
            return call;
        };
        for (k, v) in call.arguments.iter_mut() {
            let Value::String(s) = v else { continue };
            let ty = types.get(k).map(String::as_str).unwrap_or("string");
            if ty != "string" {
                if let Ok(parsed) = serde_json::from_str::<Value>(s.trim()) {
                    *v = parsed;
                }
            }
        }
        call
    }
}

/// Parse the Qwen3.6 XML call body: `<function=NAME>` then zero or more
/// `<parameter=KEY>VALUE</parameter>`, then `</function>`. Each value is taken verbatim between its
/// tags and trimmed of the surrounding whitespace the template wraps it in. Returns `None` if there
/// is no `<function=…>` opening with a non-empty name.
fn parse_xml_call(body: &str) -> Option<ToolCall> {
    const FN: &str = "<function=";
    const PARAM: &str = "<parameter=";
    const PARAM_END: &str = "</parameter>";
    let fstart = body.find(FN)? + FN.len();
    let nend = body[fstart..].find('>')? + fstart;
    let name = body[fstart..nend].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut arguments = Map::new();
    let mut rest = &body[nend + 1..];
    while let Some(ps) = rest.find(PARAM) {
        let kstart = ps + PARAM.len();
        let kend = rest[kstart..].find('>')? + kstart;
        let key = rest[kstart..kend].trim().to_string();
        let vstart = kend + 1;
        let vend = rest[vstart..].find(PARAM_END)? + vstart;
        let value = rest[vstart..vend].trim().to_string();
        arguments.insert(key, Value::String(value));
        rest = &rest[vend + PARAM_END.len()..];
    }
    Some(ToolCall { name, arguments })
}

/// Parse a JSON / Hermes call body: `{"name": "...", "arguments": {...}}` (arguments may also be a
/// JSON-encoded string, or absent). Returns `None` if it isn't a JSON object with a string `name`.
fn parse_json_call(body: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments") {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) => serde_json::from_str::<Value>(s)
            .ok()
            .and_then(|x| x.as_object().cloned())
            .unwrap_or_default(),
        _ => Map::new(),
    };
    Some(ToolCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> ToolSpec {
        ToolSpec::new(
            "get_weather",
            "Get the weather for a city",
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"}
                },
                "required": ["location"]
            }),
        )
    }

    /// Drive a segmenter over deltas; return the concatenated streamed content and the parsed calls.
    fn run(tools: &[ToolSpec], deltas: &[&str]) -> (String, Vec<ToolCall>) {
        let mut seg = ToolCallSegmenter::new(tools);
        let mut content = String::new();
        for d in deltas {
            for c in seg.push(d) {
                content.push_str(&c);
            }
        }
        for c in seg.flush() {
            content.push_str(&c);
        }
        (content, seg.take_calls())
    }

    #[test]
    fn to_template_json_is_openai_function_shape_in_order() {
        let v = weather_tool().to_template_json();
        // Insertion order preserved: type before function, name before description before parameters.
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"location":{"type":"string"},"days":{"type":"integer"},"metric":{"type":"boolean"}},"required":["location"]}}}"#
        );
    }

    #[test]
    fn xml_call_in_one_delta_parses_and_strips() {
        let block = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = run(&[weather_tool()], &[block]);
        assert_eq!(content, "Let me check.\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments.get("location"), Some(&json!("Paris")));
    }

    #[test]
    fn xml_arguments_coerced_to_schema_types() {
        let block = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n</function>\n</tool_call>";
        let (_content, calls) = run(&[weather_tool()], &[block]);
        let args = &calls[0].arguments;
        assert_eq!(args.get("location"), Some(&json!("Paris")), "string stays string");
        assert_eq!(args.get("days"), Some(&json!(3)), "integer coerced");
        assert_eq!(args.get("metric"), Some(&json!(true)), "boolean coerced");
        // Argument order is insertion order, not alphabetical.
        let keys: Vec<&String> = args.keys().collect();
        assert_eq!(keys, vec!["location", "days", "metric"]);
    }

    #[test]
    fn without_schema_arguments_stay_strings() {
        let block = "<tool_call>\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let (_c, calls) = run(&[], &[block]);
        assert_eq!(calls[0].arguments.get("days"), Some(&json!("3")));
    }

    #[test]
    fn block_split_across_deltas_byte_by_byte() {
        // The whole call arrives one byte per delta; no marker byte may leak into content.
        let block = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";
        let deltas: Vec<String> = block.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = deltas.iter().map(String::as_str).collect();
        let (content, calls) = run(&[weather_tool()], &refs);
        assert_eq!(content, "", "all markup is consumed, nothing leaks to content");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments.get("location"), Some(&json!("Paris")));
    }

    #[test]
    fn json_hermes_form_parses() {
        let block = r#"<tool_call>{"name": "get_weather", "arguments": {"location": "Paris", "days": 2}}</tool_call>"#;
        let (content, calls) = run(&[weather_tool()], &[block]);
        assert_eq!(content, "");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments.get("location"), Some(&json!("Paris")));
        assert_eq!(calls[0].arguments.get("days"), Some(&json!(2)), "JSON args keep their type");
    }

    #[test]
    fn multiple_calls_in_one_stream() {
        let block = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nRome\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = run(&[weather_tool()], &[block]);
        assert_eq!(content.trim(), "", "only the inter-call separator is content");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments.get("location"), Some(&json!("Paris")));
        assert_eq!(calls[1].arguments.get("location"), Some(&json!("Rome")));
    }

    #[test]
    fn no_tool_call_is_all_content() {
        let (content, calls) = run(&[weather_tool()], &["The weather in Paris is sunny."]);
        assert_eq!(content, "The weather in Paris is sunny.");
        assert!(calls.is_empty());
    }

    #[test]
    fn stray_close_marker_outside_block_is_content() {
        // A bare `</tool_call>` that never opened is ordinary text, not a transition.
        let (content, calls) = run(&[weather_tool()], &["see </tool_call> here"]);
        assert_eq!(content, "see </tool_call> here");
        assert!(calls.is_empty());
    }

    #[test]
    fn partial_open_that_is_not_a_marker_is_flushed_as_content() {
        // "<tool" looks like the start of "<tool_call>"; generation ends → it is content.
        let (content, calls) = run(&[weather_tool()], &["pick a <tool"]);
        assert_eq!(content, "pick a <tool");
        assert!(calls.is_empty());
    }

    #[test]
    fn unterminated_block_surfaces_raw_text_on_flush() {
        // The model opened a block then stopped before a parsable function — no call, no lost text.
        let (content, calls) = run(&[weather_tool()], &["<tool_call>\noops "]);
        assert_eq!(content, "<tool_call>\noops ");
        assert!(calls.is_empty());
    }

    #[test]
    fn unterminated_but_parsable_block_yields_a_call_on_flush() {
        // A complete function with no closing `</tool_call>` (model hit the token budget) still parses.
        let (content, calls) = run(
            &[weather_tool()],
            &["<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>"],
        );
        assert_eq!(content, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments.get("location"), Some(&json!("Paris")));
    }

    #[test]
    fn content_before_and_no_content_after() {
        // Reasoning/content may precede the call (Qwen3.6 allows natural language before, not after).
        let (content, calls) = run(
            &[weather_tool()],
            &[
                "I'll look that up for you. ",
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(content, "I'll look that up for you. ");
        assert_eq!(calls[0].arguments.get("location"), Some(&json!("Tokyo")));
    }
}
