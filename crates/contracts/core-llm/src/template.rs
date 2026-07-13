//! Chat-template rendering: a conversation → the model's prompt string.
//!
//! [`ChatTemplate`] is the host-policy seam. This module ships the common **typed** templates
//! ([`Llama3Template`], [`ChatMlTemplate`]) and a [`JinjaChatTemplate`] (story 7164) that renders a
//! model's own `chat_template` from `tokenizer_config.json` using `minijinja` — the same Jinja2
//! semantics `transformers.apply_chat_template` uses. Templates render the text of each turn; the
//! vision image-placeholder splice is the backend VLM path's concern (story 7157).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::{Environment, Value};
use serde_json::{Map as JsonMap, Value as Json};

use crate::error::{Error, Result};
use crate::message::Message;
use crate::tool::ToolSpec;

/// Options for a chat-template render. Extensible carrier for the standard chat-template kwargs
/// beyond the conversation itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderOptions<'a> {
    /// Append the opening of an assistant turn so the model continues as the assistant.
    pub add_generation_prompt: bool,
    /// The `enable_thinking` chat-template kwarg: `None` omits it (template default — the Jinja
    /// `is defined` test is false), `Some(true)`/`Some(false)` request reasoning on/off. Maps from
    /// [`TextLlmRequest::enable_thinking_kwarg`](crate::TextLlmRequest::enable_thinking_kwarg).
    pub enable_thinking: Option<bool>,
    /// Tools / functions offered to the model. Threaded into the template's `tools` context
    /// (matching `transformers` `tools=`); empty ⇒ the context is omitted, so a template's `if tools`
    /// test is false and the render is byte-identical to a no-tools render.
    pub tools: &'a [ToolSpec],
}

impl<'a> RenderOptions<'a> {
    /// Options that append a generation prompt and leave the thinking mode at the template default.
    pub fn generation() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: None,
            tools: &[],
        }
    }

    /// Set the `enable_thinking` kwarg (builder style).
    pub fn with_enable_thinking(mut self, enable_thinking: Option<bool>) -> Self {
        self.enable_thinking = enable_thinking;
        self
    }

    /// Set the offered `tools` (builder style).
    pub fn with_tools(mut self, tools: &'a [ToolSpec]) -> Self {
        self.tools = tools;
        self
    }
}

/// Renders a conversation into a single prompt string the tokenizer then encodes.
pub trait ChatTemplate {
    /// Render `messages`. When `add_generation_prompt` is set, append the opening of an assistant
    /// turn so the model continues as the assistant.
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String>;

    /// Render with the full [`RenderOptions`] (chat-template kwargs). The default ignores any kwarg
    /// a simple typed template has no notion of (e.g. `enable_thinking`, `tools`) and delegates to
    /// [`render`](ChatTemplate::render); [`JinjaChatTemplate`] overrides it to thread the kwargs
    /// into the Jinja context.
    fn render_with(&self, messages: &[Message], opts: &RenderOptions<'_>) -> Result<String> {
        self.render(messages, opts.add_generation_prompt)
    }
}

/// The Llama-3 instruct chat format.
#[derive(Clone, Copy, Debug, Default)]
pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let mut out = String::from("<|begin_of_text|>");
        for m in messages {
            out.push_str("<|start_header_id|>");
            out.push_str(m.role.as_str());
            out.push_str("<|end_header_id|>\n\n");
            out.push_str(&m.text_content());
            out.push_str("<|eot_id|>");
        }
        if add_generation_prompt {
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        Ok(out)
    }
}

/// The ChatML format (`<|im_start|>role … <|im_end|>`), used by Qwen and others.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChatMlTemplate;

impl ChatTemplate for ChatMlTemplate {
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let mut out = String::new();
        for m in messages {
            out.push_str("<|im_start|>");
            out.push_str(m.role.as_str());
            out.push('\n');
            out.push_str(&m.text_content());
            out.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
        }
        Ok(out)
    }
}

/// Renders a model's own Jinja `chat_template` (from `tokenizer_config.json`), matching
/// `transformers.apply_chat_template`.
///
/// The template sees the standard HF context: `messages` (a list of `{role, content}` maps),
/// `add_generation_prompt`, and `bos_token` / `eos_token`. Python str methods (`.strip()`, …) work
/// via pycompat, `raise_exception(msg)` errors the render, and `strftime_now(fmt)` supplies the date
/// some templates inject. The environment uses `trim_blocks` + `lstrip_blocks` like transformers.
#[derive(Clone, Debug)]
pub struct JinjaChatTemplate {
    source: String,
    bos_token: String,
    eos_token: String,
}

impl JinjaChatTemplate {
    /// From a raw `chat_template` string (no special tokens).
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            bos_token: String::new(),
            eos_token: String::new(),
        }
    }

    /// From a `chat_template` string plus the model's BOS/EOS token strings.
    pub fn with_tokens(
        source: impl Into<String>,
        bos_token: impl Into<String>,
        eos_token: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
        }
    }

    /// Build from a parsed `tokenizer_config.json`. Handles `chat_template` as a plain string or as
    /// an array of `{name, template}` (picks `default`, else the first), and `bos_token`/`eos_token`
    /// as a plain string or an `{"content": …}` added-token object.
    pub fn from_tokenizer_config(config: &serde_json::Value) -> Result<Self> {
        let source = extract_chat_template(config)
            .ok_or_else(|| Error::Msg("tokenizer_config.json has no chat_template".to_string()))?;
        let bos_token = extract_token(config.get("bos_token")).unwrap_or_default();
        let eos_token = extract_token(config.get("eos_token")).unwrap_or_default();
        Ok(Self::with_tokens(source, bos_token, eos_token))
    }

    /// Read and parse `tokenizer_config.json` from a path.
    pub fn from_tokenizer_config_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("parse tokenizer_config.json: {e}")))?;
        Self::from_tokenizer_config(&v)
    }

    /// The raw template source.
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl ChatTemplate for JinjaChatTemplate {
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        self.render_with(
            messages,
            &RenderOptions {
                add_generation_prompt,
                enable_thinking: None,
                tools: &[],
            },
        )
    }

    fn render_with(&self, messages: &[Message], opts: &RenderOptions<'_>) -> Result<String> {
        let mut env = Environment::new();
        // Match transformers' Jinja environment (ImmutableSandboxedEnvironment, trim/lstrip blocks).
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", self.source.clone())
            .map_err(jinja_err)?;
        let tmpl = env.get_template("chat").map_err(jinja_err)?;

        // Each message is a JSON object so a turn can carry structured fields (`tool_calls`) beyond
        // the flat `role`/`content` strings — the same shape `transformers` passes the template.
        let msgs: Vec<Json> = messages
            .iter()
            .map(|m| {
                let mut map = JsonMap::new();
                map.insert("role".into(), Json::String(m.role.as_str().to_string()));
                map.insert("content".into(), Json::String(m.text_content()));
                // A turn's prior reasoning is exposed under both standard keys so a reasoning
                // model's template finds it: `reasoning_content` (Qwen3 / DeepSeek) and `thinking`.
                // Inserted only when present, so a template's `reasoning_content is string` test is
                // false otherwise (the model's own retention/stripping policy then applies).
                if let Some(thinking) = &m.thinking {
                    map.insert("reasoning_content".into(), Json::String(thinking.clone()));
                    map.insert("thinking".into(), Json::String(thinking.clone()));
                }
                // An assistant turn's tool calls, in the `{type, function:{name, arguments}}` shape a
                // tool template re-renders. Inserted only when present, so a `message.tool_calls`
                // truthiness test is false otherwise.
                if !m.tool_calls.is_empty() {
                    let calls: Vec<Json> = m.tool_calls.iter().map(|c| c.to_template_json()).collect();
                    map.insert("tool_calls".into(), Json::Array(calls));
                }
                Json::Object(map)
            })
            .collect();

        // Build the context as a map so `enable_thinking` can be *omitted* entirely when the kwarg
        // is `None` — a template's `enable_thinking is defined` test must then be false, matching
        // `transformers` omitting it from `chat_template_kwargs`. Inserting an explicit
        // `Value::UNDEFINED` would not read as undefined to that test.
        let mut ctx: BTreeMap<&str, Value> = BTreeMap::new();
        ctx.insert("messages", Value::from_serialize(&msgs));
        ctx.insert("add_generation_prompt", Value::from(opts.add_generation_prompt));
        ctx.insert("bos_token", Value::from(self.bos_token.clone()));
        ctx.insert("eos_token", Value::from(self.eos_token.clone()));
        if let Some(enable_thinking) = opts.enable_thinking {
            ctx.insert("enable_thinking", Value::from(enable_thinking));
        }
        // Offered tools in the OpenAI function shape the template renders (`tool | tojson`). Inserted
        // only when non-empty, so a template's `if tools` test is false on a no-tools render (the
        // result is then byte-identical to before this field existed).
        if !opts.tools.is_empty() {
            let tools: Vec<Json> = opts.tools.iter().map(|t| t.to_template_json()).collect();
            ctx.insert("tools", Value::from_serialize(&tools));
        }
        tmpl.render(ctx).map_err(jinja_err)
    }
}

fn jinja_err(e: minijinja::Error) -> Error {
    Error::Msg(format!("chat template render: {e}"))
}

/// Pull the chat-template source from a tokenizer config (string or `[{name, template}]`).
fn extract_chat_template(config: &serde_json::Value) -> Option<String> {
    match config.get("chat_template")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(entries) => {
            let pick = entries
                .iter()
                .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
                .or_else(|| entries.first())?;
            pick.get("template").and_then(|t| t.as_str()).map(String::from)
        }
        _ => None,
    }
}

/// Pull a special-token string (plain string or `{"content": …}` added-token object).
fn extract_token(value: Option<&serde_json::Value>) -> Option<String> {
    match value? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

/// `strftime_now(fmt)`: format the current UTC date. Supports the specifiers HF templates use
/// (`%Y %y %m %d %e %b %B %%`).
fn strftime_now(fmt: String) -> std::result::Result<Value, minijinja::Error> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    Ok(Value::from(format_date(&fmt, y, m, d)))
}

/// Civil date (year, month 1..=12, day 1..=31) from days since the Unix epoch (Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn format_date(fmt: &str, y: i64, m: u32, d: u32) -> String {
    const ABBR: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const FULL: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December",
    ];
    let mut out = String::new();
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('b') => out.push_str(ABBR[(m - 1) as usize]),
            Some('B') => out.push_str(FULL[(m - 1) as usize]),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};

    #[test]
    fn llama3_renders_turns_and_generation_prompt() {
        let msgs = vec![Message::system("be brief"), Message::user("hi")];
        let s = Llama3Template.render(&msgs, true).unwrap();
        assert!(s.starts_with("<|begin_of_text|>"));
        assert!(s.contains("<|start_header_id|>system<|end_header_id|>\n\nbe brief<|eot_id|>"));
        assert!(s.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
        assert!(s.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn chatml_renders_turns() {
        let msgs = vec![Message::text(Role::User, "hi")];
        let s = ChatMlTemplate.render(&msgs, true).unwrap();
        assert_eq!(s, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    // The actual SmolLM2 chat_template (ChatML), `\n` escapes as in the model's tokenizer_config.
    const SMOLLM2_CHATML: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    const LLAMA3: &str = "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";

    const MISTRAL: &str = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}";

    #[test]
    fn jinja_renders_chatml_byte_correct() {
        let t = JinjaChatTemplate::new(SMOLLM2_CHATML);
        let msgs = vec![Message::system("be brief"), Message::user("hi")];
        let out = t.render(&msgs, true).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn jinja_chatml_injects_default_system_when_absent() {
        // First message is a user turn → the template prepends its default system block.
        let t = JinjaChatTemplate::new(SMOLLM2_CHATML);
        let out = t.render(&[Message::user("hi")], false).unwrap();
        assert!(out.starts_with("<|im_start|>system\nYou are a helpful AI assistant named SmolLM"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>\n"));
    }

    #[test]
    fn jinja_renders_llama3_with_bos() {
        let t = JinjaChatTemplate::with_tokens(LLAMA3, "<|begin_of_text|>", "<|eot_id|>");
        let out = t.render(&[Message::user("hi")], true).unwrap();
        assert_eq!(
            out,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn jinja_renders_mistral_multiturn() {
        let t = JinjaChatTemplate::with_tokens(MISTRAL, "<s>", "</s>");
        let msgs = vec![
            Message::user("hi"),
            Message::assistant("hello"),
            Message::user("bye"),
        ];
        let out = t.render(&msgs, false).unwrap();
        assert_eq!(out, "<s>[INST] hi [/INST]hello</s>[INST] bye [/INST]");
    }

    #[test]
    fn jinja_raise_exception_errors() {
        let t = JinjaChatTemplate::new(
            "{% if messages[0]['role'] == 'assistant' %}{{ raise_exception('bad first role') }}{% endif %}ok",
        );
        assert!(t.render(&[Message::assistant("oops")], false).is_err());
        // A valid first role renders fine.
        assert_eq!(t.render(&[Message::user("hi")], false).unwrap(), "ok");
    }

    #[test]
    fn jinja_renders_tool_role() {
        let t = JinjaChatTemplate::new(
            "{% for m in messages %}{% if m['role'] == 'tool' %}{{ '<tool>' + m['content'] + '</tool>' }}{% else %}{{ m['role'] + ':' + m['content'] }}{% endif %}{% endfor %}",
        );
        let msgs = vec![
            Message::user("q"),
            Message {
                role: Role::Tool,
                content: vec![crate::message::Content::Text("result".into())],
                thinking: None,
                tool_calls: Vec::new(),
            },
        ];
        assert_eq!(t.render(&msgs, false).unwrap(), "user:q<tool>result</tool>");
    }

    #[test]
    fn jinja_pycompat_str_methods() {
        let t = JinjaChatTemplate::new("{{ messages[0]['content'].strip().upper() }}");
        assert_eq!(t.render(&[Message::user("  hi  ")], false).unwrap(), "HI");
    }

    fn weather_tool() -> crate::tool::ToolSpec {
        crate::tool::ToolSpec::new(
            "get_weather",
            "Get the weather",
            serde_json::json!({"type":"object","properties":{"location":{"type":"string"}}}),
        )
    }

    // The Qwen3.6 tools branch, verbatim (the `<tools>` block + the `tool | tojson` loop). Renders
    // only when `tools` is a non-empty iterable — which is exactly when our context inserts it.
    const QWEN36_TOOLS_BRANCH: &str = "{%- if tools and tools is iterable and tools is not mapping %}{{- '<tools>' }}{%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}{{- '\\n</tools>' }}{%- else %}NO_TOOLS{%- endif %}";

    #[test]
    fn jinja_renders_tools_section_when_offered() {
        let t = JinjaChatTemplate::new(QWEN36_TOOLS_BRANCH);
        let tools = [weather_tool()];
        let out = t
            .render_with(&[Message::user("hi")], &RenderOptions::generation().with_tools(&tools))
            .unwrap();
        // The tool renders in the OpenAI function shape, keys in insertion order (preserve_order).
        assert_eq!(
            out,
            "<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}}}}}\n</tools>"
        );
    }

    #[test]
    fn jinja_tools_section_omitted_when_none() {
        // No tools offered ⇒ the `if tools` branch is false (we omit the context entirely), so the
        // render is identical to before this field existed.
        let t = JinjaChatTemplate::new(QWEN36_TOOLS_BRANCH);
        assert_eq!(t.render(&[Message::user("hi")], false).unwrap(), "NO_TOOLS");
        assert_eq!(
            t.render_with(&[Message::user("hi")], &RenderOptions::generation())
                .unwrap(),
            "NO_TOOLS"
        );
    }

    // The Qwen3.6 assistant tool_calls branch, verbatim: unwrap `.function`, emit the `<function=…>`
    // / `<parameter=…>` XML, string args as-is and non-string args via tojson.
    const QWEN36_TOOLCALL_BRANCH: &str = "{%- for message in messages %}{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{%- if tool_call.arguments is defined %}{%- for args_name, args_value in tool_call.arguments|items %}{{- '<parameter=' + args_name + '>\\n' }}{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}{{- args_value }}{{- '\\n</parameter>\\n' }}{%- endfor %}{%- endif %}{{- '</function>\\n</tool_call>' }}{%- endfor %}{%- endif %}{%- endfor %}";

    #[test]
    fn jinja_round_trips_assistant_tool_calls_as_xml() {
        let t = JinjaChatTemplate::new(QWEN36_TOOLCALL_BRANCH);
        let mut args = serde_json::Map::new();
        args.insert("location".into(), serde_json::json!("Paris")); // string → rendered as-is
        args.insert("days".into(), serde_json::json!(3)); // non-string → rendered via tojson
        let call = crate::tool::ToolCall::new("get_weather", args);
        let msgs = [Message::assistant("").with_tool_calls(vec![call])];
        let out = t.render(&msgs, false).unwrap();
        assert_eq!(
            out,
            "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
        );
    }

    #[test]
    #[ignore = "needs the real Qwen3.6 tokenizer_config via MLX_LLM_QWEN35_MODEL"]
    fn qwen36_real_template_renders_tools_and_calls() {
        // Faithfulness against the *actual* Qwen3.6 chat template (not a stand-in): a request that
        // offers a tool renders the `<tools>` section + the call format instructions, and a prior
        // assistant tool_call turn re-renders as the `<function=…>` / `<parameter=…>` XML.
        let dir = std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL");
        let t = JinjaChatTemplate::from_tokenizer_config_file(format!("{dir}/tokenizer_config.json"))
            .expect("load real Qwen3.6 chat template");

        let tools = [weather_tool()];
        let offered = t
            .render_with(&[Message::user("weather in Paris?")], &RenderOptions::generation().with_tools(&tools))
            .unwrap();
        assert!(offered.contains("<tools>"), "tools section: {offered}");
        assert!(
            offered.contains("{\"type\":\"function\",\"function\":{\"name\":\"get_weather\""),
            "tool json: {offered}"
        );
        assert!(offered.contains("<tool_call>"), "call-format instructions: {offered}");

        let mut args = serde_json::Map::new();
        args.insert("location".into(), serde_json::json!("Paris"));
        let call = crate::tool::ToolCall::new("get_weather", args);
        let round = t
            .render(
                &[
                    Message::user("weather in Paris?"),
                    Message::assistant("").with_tool_calls(vec![call]),
                ],
                false,
            )
            .unwrap();
        assert!(
            round.contains("<function=get_weather>\n<parameter=location>\nParis\n</parameter>"),
            "round-trip tool_call XML: {round}"
        );
    }

    #[test]
    #[ignore = "needs the real Qwen3.6 tokenizer_config via MLX_LLM_QWEN35_MODEL"]
    fn qwen36_real_template_thinking_reasoning_and_vision() {
        // Faithfulness against the *actual* Qwen3.6 chat template (not a stand-in), the host-policy
        // audit (sc-7631): ChatML structure, the `enable_thinking` no-think branch, the prior-turn
        // reasoning-retention policy, and how image content is (not) rendered. The companion
        // `qwen36_real_template_renders_tools_and_calls` covers the tool-injection / call round-trip.
        use crate::message::{Content, ImageRef};

        let dir = std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL");
        let t = JinjaChatTemplate::from_tokenizer_config_file(format!("{dir}/tokenizer_config.json"))
            .expect("load real Qwen3.6 chat template");

        // ChatML structure: a user turn renders as an `<|im_start|>user … <|im_end|>` block.
        let chatml = t.render(&[Message::user("Hello")], false).unwrap();
        assert!(
            chatml.contains("<|im_start|>user\nHello<|im_end|>\n"),
            "ChatML user block: {chatml}"
        );

        // Thinking generation-prompt branch (the no-think reachability the provider relies on):
        // Disabled injects the model's closed empty think block; Auto/Enabled open `<think>\n` so the
        // model produces its own. (Generation prompt is the tail of the rendered string.)
        let msgs = [Message::user("What is 2+2?")];
        let disabled = t
            .render_with(&msgs, &RenderOptions::generation().with_enable_thinking(Some(false)))
            .unwrap();
        assert!(
            disabled.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "disabled = closed empty think block: {disabled}"
        );
        let auto = t.render_with(&msgs, &RenderOptions::generation()).unwrap();
        assert!(
            auto.ends_with("<|im_start|>assistant\n<think>\n"),
            "auto opens <think>: {auto}"
        );
        let enabled = t
            .render_with(&msgs, &RenderOptions::generation().with_enable_thinking(Some(true)))
            .unwrap();
        assert!(
            enabled.ends_with("<|im_start|>assistant\n<think>\n"),
            "enabled opens <think>: {enabled}"
        );

        // Reasoning-retention policy: the latest assistant turn re-emits its `reasoning_content` (from
        // Message::thinking); once a newer user turn follows, the template strips the prior reasoning.
        let latest = [
            Message::user("What is 2+2?"),
            Message::assistant("2+2 is 4.").with_thinking("The user asks 2+2; that is 4."),
        ];
        let rendered = t.render(&latest, false).unwrap();
        assert!(
            rendered.contains("<think>\nThe user asks 2+2; that is 4.\n</think>\n\n2+2 is 4."),
            "latest-turn reasoning must be re-rendered: {rendered}"
        );
        let prior = [
            Message::user("What is 2+2?"),
            Message::assistant("2+2 is 4.").with_thinking("The user asks 2+2; that is 4."),
            Message::user("And 3+3?"),
        ];
        let rendered = t.render_with(&prior, &RenderOptions::generation()).unwrap();
        assert!(
            rendered.contains("<|im_start|>assistant\n2+2 is 4.<|im_end|>"),
            "prior assistant content kept: {rendered}"
        );
        assert!(
            !rendered.contains("The user asks 2+2"),
            "prior-turn reasoning must be stripped by Qwen3.6's policy: {rendered}"
        );

        // Vision handling — the host-policy finding. The contract flattens a turn's content to text
        // (`Message::text_content` drops image blocks), so the template's native vision-counting
        // branch (which only fires for *structured* content lists, emitting `<|image_pad|>`) does NOT
        // run: an image block renders as nothing, just the accompanying text.
        let with_image = Message {
            role: Role::User,
            content: vec![
                Content::Image(ImageRef::new(2, 2, vec![0u8; 12]).unwrap()),
                Content::text("describe this"),
            ],
            thinking: None,
            tool_calls: Vec::new(),
        };
        let img_rendered = t.render(std::slice::from_ref(&with_image), false).unwrap();
        assert!(img_rendered.contains("describe this"), "text kept: {img_rendered}");
        assert!(
            !img_rendered.contains("<|image_pad|>"),
            "the template does NOT insert vision tokens for flattened content — the provider \
             substitutes the placeholder text instead: {img_rendered}"
        );

        // The actual vision path: the provider substitutes the Qwen-VL placeholder *text* before
        // rendering (one `<|image_pad|>` per image, expanded to the patch count after tokenizing), and
        // the real template passes that text through verbatim via its `content is string` branch.
        let substituted = t
            .render(
                &[Message::user(
                    "<|vision_start|><|image_pad|><|vision_end|>describe this",
                )],
                false,
            )
            .unwrap();
        assert!(
            substituted.contains("<|vision_start|><|image_pad|><|vision_end|>describe this"),
            "provider-substituted vision placeholder renders verbatim: {substituted}"
        );
    }

    /// Locate the cached `Qwen/Qwen3-VL-8B-Instruct` snapshot (rev 0c351dd0) for the real
    /// chat-template oracle. `MLX_LLM_QWEN3VL_MODEL` / `QWEN3VL_SNAPSHOT` override; otherwise the
    /// default HF cache path. `None` ⇒ the gated test self-skips cleanly (the snapshot is present in
    /// CI for sc-8077).
    #[cfg(test)]
    fn qwen3vl_snapshot_dir() -> Option<std::path::PathBuf> {
        for var in ["MLX_LLM_QWEN3VL_MODEL", "QWEN3VL_SNAPSHOT"] {
            if let Ok(path) = std::env::var(var) {
                let path = std::path::PathBuf::from(path);
                if path.exists() {
                    return Some(path);
                }
            }
        }
        let home = std::env::var("HOME").ok()?;
        let path = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        );
        path.exists().then_some(path)
    }

    #[test]
    fn qwen3vl_real_template_chatml_tools_and_vision() {
        // Faithfulness against the *actual* Qwen3-VL-8B-Instruct chat template (sc-8077), mirroring
        // the Qwen3.6 host-policy audit (sc-7631) for the qwen3_vl VLM. Self-skips when the snapshot
        // is absent. Asserts: ChatML structure, the `add_generation_prompt` tail, the tools section,
        // and the vision finding (flattened content drops image blocks → no native vision token; the
        // provider's substituted placeholder text passes through verbatim).
        use crate::message::{Content, ImageRef, Role, VideoRef};
        let Some(dir) = qwen3vl_snapshot_dir() else {
            eprintln!("skipping qwen3vl_real_template_chatml_tools_and_vision: Qwen3-VL-8B snapshot \
                       not present (set MLX_LLM_QWEN3VL_MODEL or QWEN3VL_SNAPSHOT)");
            return;
        };
        let t = JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json"))
            .expect("load real Qwen3-VL chat template");

        // The Qwen3-VL-8B-Instruct template is NOT a thinking template (no `enable_thinking` /
        // `reasoning_content` gating) — its generation prompt is a plain `<|im_start|>assistant\n`,
        // never a primed `<think>`. This is exactly what the provider's capability detection keys on.
        assert!(
            !t.source().contains("enable_thinking"),
            "Qwen3-VL-8B-Instruct template must not gate enable_thinking (supports_thinking=false)"
        );
        // It DOES render tool calls (the `<tools>` section / `tool_call` blocks), which is what the
        // provider's `supports_tools` detection keys on.
        assert!(
            t.source().contains("tool_call") && t.source().contains("<tools>"),
            "Qwen3-VL template must render tools (supports_tools=true)"
        );

        // ChatML structure + the generation-prompt tail.
        let chatml = t.render(&[Message::user("Hello")], true).unwrap();
        assert!(chatml.contains("<|im_start|>user\nHello<|im_end|>\n"), "ChatML user block: {chatml}");
        assert!(chatml.ends_with("<|im_start|>assistant\n"), "generation prompt tail: {chatml}");

        // Tools section: offering a tool renders the `<tools>` block with the OpenAI function json.
        let tools = [weather_tool()];
        let offered = t
            .render_with(&[Message::user("weather in Paris?")], &RenderOptions::generation().with_tools(&tools))
            .unwrap();
        assert!(offered.contains("<tools>"), "tools section: {offered}");
        assert!(
            offered.contains("{\"type\":\"function\",\"function\":{\"name\":\"get_weather\""),
            "tool json: {offered}"
        );

        // Vision handling (the host-policy finding): the contract flattens content to text
        // (`Message::text_content` drops image blocks), so the template's native vision branch does
        // NOT emit `<|image_pad|>` for a structured image block — only the accompanying text renders.
        let with_image = Message {
            role: Role::User,
            content: vec![
                Content::Image(ImageRef::new(2, 2, vec![0u8; 12]).unwrap()),
                Content::text("describe this"),
            ],
            thinking: None,
            tool_calls: Vec::new(),
        };
        let img_rendered = t.render(std::slice::from_ref(&with_image), false).unwrap();
        assert!(img_rendered.contains("describe this"), "text kept: {img_rendered}");
        assert!(
            !img_rendered.contains("<|image_pad|>"),
            "flattened content must NOT insert vision tokens — the provider substitutes the \
             placeholder text instead: {img_rendered}"
        );

        // The actual vision path: the provider substitutes the Qwen-VL placeholder *text* before
        // rendering, and the real template passes it through verbatim via its `content is string`
        // branch (one `<|image_pad|>` per image, expanded to the patch count after tokenizing).
        let substituted = t
            .render(
                &[Message::user("<|vision_start|><|image_pad|><|vision_end|>describe this")],
                false,
            )
            .unwrap();
        assert!(
            substituted.contains("<|vision_start|><|image_pad|><|vision_end|>describe this"),
            "provider-substituted vision placeholder renders verbatim: {substituted}"
        );

        // Video flattens the same way: a `Content::Video` block renders as nothing; the provider
        // substitutes the Text–Timestamp-Alignment placeholder text before rendering.
        let frame = ImageRef::new(2, 2, vec![0u8; 12]).unwrap();
        let with_video = Message {
            role: Role::User,
            content: vec![
                Content::Video(VideoRef::new(vec![frame.clone(), frame], vec![0.0, 0.5]).unwrap()),
                Content::text("what happens"),
            ],
            thinking: None,
            tool_calls: Vec::new(),
        };
        let vid_rendered = t.render(std::slice::from_ref(&with_video), false).unwrap();
        assert!(vid_rendered.contains("what happens"), "video-turn text kept: {vid_rendered}");
        assert!(
            !vid_rendered.contains("<|video_pad|>"),
            "flattened video content must NOT insert video tokens — the provider substitutes the \
             timestamp+placeholder text instead: {vid_rendered}"
        );

        // The provider-substituted video placeholder (per-frame `<{t} seconds>` timestamp + a vision
        // block wrapping `<|video_pad|>`) renders verbatim through the `content is string` branch —
        // the Text–Timestamp-Alignment shape `Qwen3VLProcessor.replace_video_token` emits.
        let vid_substituted = t
            .render(
                &[Message::user(
                    "<0.0 seconds><|vision_start|><|video_pad|><|vision_end|>\
                     <0.5 seconds><|vision_start|><|video_pad|><|vision_end|>what happens",
                )],
                false,
            )
            .unwrap();
        assert!(
            vid_substituted.contains("<0.0 seconds><|vision_start|><|video_pad|><|vision_end|>"),
            "provider-substituted video timestamp+placeholder renders verbatim: {vid_substituted}"
        );
    }

    #[test]
    fn jinja_from_tokenizer_config_extracts_template_and_tokens() {
        let cfg = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}{{ m['role'] + ':' + m['content'] + eos_token }}{% endfor %}",
            "bos_token": "<s>",
            "eos_token": { "content": "</s>", "lstrip": false }
        });
        let t = JinjaChatTemplate::from_tokenizer_config(&cfg).unwrap();
        assert_eq!(t.render(&[Message::user("hi")], false).unwrap(), "<s>user:hi</s>");
    }

    #[test]
    fn jinja_chat_template_array_form() {
        let cfg = serde_json::json!({
            "chat_template": [
                { "name": "tool_use", "template": "TOOLS" },
                { "name": "default", "template": "{{ messages[0]['content'] }}" }
            ]
        });
        let t = JinjaChatTemplate::from_tokenizer_config(&cfg).unwrap();
        assert_eq!(t.render(&[Message::user("hi")], false).unwrap(), "hi");
    }

    // The Qwen3 generation-prompt branch, verbatim: a no-think (`enable_thinking=false`) request
    // injects an empty think block; otherwise the model produces its own.
    const QWEN3_THINK_TAIL: &str = "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}";

    #[test]
    fn jinja_enable_thinking_drives_qwen3_nothink_branch() {
        let t = JinjaChatTemplate::new(QWEN3_THINK_TAIL);
        let msgs = [Message::user("hi")];

        // Auto (kwarg omitted) ⇒ `is defined` is false ⇒ no injection. This is what plain
        // `render` / the pre-fix behavior produced, and is why no-think was previously unreachable.
        let auto = t.render_with(&msgs, &RenderOptions::generation()).unwrap();
        assert_eq!(auto, "<|im_start|>assistant\n");
        assert_eq!(t.render(&msgs, true).unwrap(), auto, "render == render_with(Auto)");

        // Enabled ⇒ defined and true ⇒ branch (which only fires on `is false`) does not inject.
        let enabled = t
            .render_with(&msgs, &RenderOptions::generation().with_enable_thinking(Some(true)))
            .unwrap();
        assert_eq!(enabled, "<|im_start|>assistant\n");

        // Disabled ⇒ defined and false ⇒ the empty think block is injected (no-think now reachable).
        let disabled = t
            .render_with(&msgs, &RenderOptions::generation().with_enable_thinking(Some(false)))
            .unwrap();
        assert_eq!(disabled, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[test]
    fn jinja_message_thinking_is_exposed_as_reasoning_content() {
        // A turn's `thinking` is visible to the template as `reasoning_content` (and `thinking`);
        // absent when None, so the template's `is string` gate is false.
        let probe = JinjaChatTemplate::new(
            "{% for m in messages %}{% if m.reasoning_content is string %}RC={{ m.reasoning_content }};{% else %}RC=none;{% endif %}{% if m.thinking is string %}T={{ m.thinking }};{% endif %}{% endfor %}",
        );
        let msgs = [
            Message::user("q"),
            Message::assistant("A").with_thinking("R"),
        ];
        assert_eq!(probe.render(&msgs, false).unwrap(), "RC=none;RC=R;T=R;");
    }

    // Mirrors Qwen3's assistant handling: reasoning is re-emitted from `reasoning_content` only for
    // the most recent turn; an earlier turn's reasoning is dropped (the template owns the policy —
    // the contract only carries the field).
    const REASONING_RETENTION: &str = "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' }}{% if m.reasoning_content is string and loop.last %}{{ '<think>\\n' + m.reasoning_content + '\\n</think>\\n\\n' }}{% endif %}{{ m.content + '<|im_end|>\\n' }}{% endfor %}";

    #[test]
    fn jinja_reasoning_kept_for_latest_turn() {
        let t = JinjaChatTemplate::new(REASONING_RETENTION);
        let msgs = [Message::user("q1"), Message::assistant("A1").with_thinking("R1")];
        let out = t.render(&msgs, false).unwrap();
        assert!(out.contains("<|im_start|>assistant\n<think>\nR1\n</think>\n\nA1<|im_end|>"), "{out}");
    }

    #[test]
    fn jinja_prior_turn_reasoning_is_stripped_by_template_policy() {
        // The same assistant turn, now followed by a newer user turn, renders without its reasoning —
        // carrying `thinking` does not force-inject it; the template decides.
        let t = JinjaChatTemplate::new(REASONING_RETENTION);
        let msgs = [
            Message::user("q1"),
            Message::assistant("A1").with_thinking("R1"),
            Message::user("q2"),
        ];
        let out = t.render(&msgs, false).unwrap();
        assert!(out.contains("<|im_start|>assistant\nA1<|im_end|>"), "{out}");
        assert!(!out.contains("<think>"), "prior-turn reasoning must be stripped: {out}");
    }

    #[test]
    #[ignore = "needs a real Qwen3 tokenizer_config via MLX_LLM_QWEN3_MODEL"]
    fn qwen3_real_template_round_trips_reasoning() {
        // Faithfulness against the *actual* Qwen3 chat template (not a stand-in): carrying
        // Message::thinking re-renders the reasoning for the latest assistant turn and the template's
        // own policy strips it once a newer user turn follows.
        let dir = std::env::var("MLX_LLM_QWEN3_MODEL").expect("set MLX_LLM_QWEN3_MODEL");
        let t = JinjaChatTemplate::from_tokenizer_config_file(format!("{dir}/tokenizer_config.json"))
            .expect("load real Qwen3 chat template");

        // Assistant is the most recent turn → its reasoning is re-emitted from reasoning_content.
        let latest = [
            Message::user("What is 2+2?"),
            Message::assistant("2+2 is 4.").with_thinking("The user asks 2+2; that is 4."),
        ];
        let rendered = t.render(&latest, false).unwrap();
        assert!(
            rendered.contains("<think>\nThe user asks 2+2; that is 4.\n</think>\n\n2+2 is 4."),
            "latest-turn reasoning must be re-rendered: {rendered}"
        );

        // A newer user turn follows → Qwen3 strips the prior assistant turn's reasoning.
        let prior = [
            Message::user("What is 2+2?"),
            Message::assistant("2+2 is 4.").with_thinking("The user asks 2+2; that is 4."),
            Message::user("And 3+3?"),
        ];
        let rendered = t.render_with(&prior, &RenderOptions::generation()).unwrap();
        assert!(rendered.contains("<|im_start|>assistant\n2+2 is 4.<|im_end|>"), "{rendered}");
        assert!(
            !rendered.contains("The user asks 2+2"),
            "prior-turn reasoning must be stripped by Qwen3's template: {rendered}"
        );
    }

    #[test]
    fn jinja_strftime_now_matches_formatter() {
        let t = JinjaChatTemplate::new("Today: {{ strftime_now('%d %b %Y') }}");
        let out = t.render(&[Message::user("x")], false).unwrap();
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
        assert_eq!(out, format!("Today: {}", format_date("%d %b %Y", y, m, d)));
    }

    #[test]
    fn civil_date_known_value() {
        // 2021-01-01 is day 18628 since the Unix epoch.
        assert_eq!(civil_from_days(18628), (2021, 1, 1));
        assert_eq!(format_date("%Y-%m-%d", 2021, 1, 1), "2021-01-01");
        assert_eq!(format_date("%d %b %Y", 2026, 6, 21), "21 Jun 2026");
    }
}
