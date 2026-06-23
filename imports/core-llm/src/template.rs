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

use crate::error::{Error, Result};
use crate::message::Message;

/// Options for a chat-template render. Extensible carrier for the standard chat-template kwargs
/// beyond the conversation itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderOptions {
    /// Append the opening of an assistant turn so the model continues as the assistant.
    pub add_generation_prompt: bool,
    /// The `enable_thinking` chat-template kwarg: `None` omits it (template default — the Jinja
    /// `is defined` test is false), `Some(true)`/`Some(false)` request reasoning on/off. Maps from
    /// [`TextLlmRequest::enable_thinking_kwarg`](crate::TextLlmRequest::enable_thinking_kwarg).
    pub enable_thinking: Option<bool>,
}

impl RenderOptions {
    /// Options that append a generation prompt and leave the thinking mode at the template default.
    pub fn generation() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: None,
        }
    }

    /// Set the `enable_thinking` kwarg (builder style).
    pub fn with_enable_thinking(mut self, enable_thinking: Option<bool>) -> Self {
        self.enable_thinking = enable_thinking;
        self
    }
}

/// Renders a conversation into a single prompt string the tokenizer then encodes.
pub trait ChatTemplate {
    /// Render `messages`. When `add_generation_prompt` is set, append the opening of an assistant
    /// turn so the model continues as the assistant.
    fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String>;

    /// Render with the full [`RenderOptions`] (chat-template kwargs). The default ignores any kwarg
    /// a simple typed template has no notion of (e.g. `enable_thinking`) and delegates to
    /// [`render`](ChatTemplate::render); [`JinjaChatTemplate`] overrides it to thread the kwargs
    /// into the Jinja context.
    fn render_with(&self, messages: &[Message], opts: &RenderOptions) -> Result<String> {
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
            },
        )
    }

    fn render_with(&self, messages: &[Message], opts: &RenderOptions) -> Result<String> {
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

        let msgs: Vec<BTreeMap<&str, String>> = messages
            .iter()
            .map(|m| {
                let mut map = BTreeMap::new();
                map.insert("role", m.role.as_str().to_string());
                map.insert("content", m.text_content());
                // A turn's prior reasoning is exposed under both standard keys so a reasoning
                // model's template finds it: `reasoning_content` (Qwen3 / DeepSeek) and `thinking`.
                // Inserted only when present, so a template's `reasoning_content is string` test is
                // false otherwise (the model's own retention/stripping policy then applies).
                if let Some(thinking) = &m.thinking {
                    map.insert("reasoning_content", thinking.clone());
                    map.insert("thinking", thinking.clone());
                }
                map
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
            },
        ];
        assert_eq!(t.render(&msgs, false).unwrap(), "user:q<tool>result</tool>");
    }

    #[test]
    fn jinja_pycompat_str_methods() {
        let t = JinjaChatTemplate::new("{{ messages[0]['content'].strip().upper() }}");
        assert_eq!(t.render(&[Message::user("  hi  ")], false).unwrap(), "HI");
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
