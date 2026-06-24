//! The `core-llm` provider: a generic Llama model exposed through the backend-neutral contract.
//!
//! This is the candle-llm half of story 7237 — it implements [`core_llm::TextLlm`] by wrapping the
//! [`CausalLm`] decoder, a [`core_llm::Tokenizer`], and a chat template, driving the internal
//! streaming decode loop and translating its token events into contract [`StreamEvent`]s (with
//! incremental detokenization). It registers into [`core_llm::registry`] under the id `candle-llama`.
//! Passing the `core-llm-testkit` conformance suite as a *second, independent* backend is what
//! de-provisionalizes the contract.

use std::cell::OnceCell;
use std::path::Path;

use candle_core::{Device, Tensor};
use core_llm::{
    Channel, ChatTemplate, Constraint, ConstraintDecodeTable, Error as CoreError,
    FinishReason as CoreFinish, JinjaChatTemplate, JsonConstraint, Llama3Template, LoadSpec,
    Quantize, RenderOptions, Result as CoreResult, Sampling, StreamEvent as CoreEvent, TextLlm,
    TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, TextLlmRequest, ThinkingSegmenter,
    Tokenizer, ToolCallSegmenter, Usage,
};

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_with, ConstraintMask, Decode, FinishReason, GenerationConfig, StreamEvent,
};
use crate::device::select_device;
use crate::gguf::GgufCheckpoint;
use crate::models::{CausalLm, Qwen35Config, Qwen35Model};
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{KvCache, Weights};

/// The registry id of this provider.
pub const PROVIDER_ID: &str = "candle-llama";

/// The loaded decoder behind the provider: the generic Llama-family [`CausalLm`] or the Qwen3.6
/// hybrid [`Qwen35Model`] (DeltaNet linear attention + gated full attention), both driven through the
/// shared [`Decode`] loop. Dispatched by [`Architecture`] at load.
enum Decoder {
    Causal(CausalLm),
    Qwen35(Qwen35Model),
}

impl Decode for Decoder {
    fn make_cache(&self) -> Box<dyn KvCache> {
        match self {
            Decoder::Causal(m) => m.make_cache(),
            Decoder::Qwen35(m) => m.make_cache(),
        }
    }

    fn device(&self) -> &Device {
        match self {
            Decoder::Causal(m) => m.device(),
            Decoder::Qwen35(m) => m.device(),
        }
    }

    fn step(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Tensor> {
        match self {
            Decoder::Causal(m) => m.step(input_ids, cache, offset),
            Decoder::Qwen35(m) => m.step(input_ids, cache, offset),
        }
    }
}

impl Decoder {
    fn is_quantized(&self) -> bool {
        match self {
            Decoder::Causal(m) => m.is_quantized(),
            Decoder::Qwen35(m) => m.is_quantized(),
        }
    }
}

/// A generic Llama provider implementing [`core_llm::TextLlm`].
pub struct LlamaProvider {
    descriptor: TextLlmDescriptor,
    model: Decoder,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    stop_tokens: Vec<i32>,
    /// Cached per-vocab decode table for constrained decoding — built once (it decodes the whole
    /// vocabulary) on the first JSON-constrained request, then reused.
    constraint_table: OnceCell<ConstraintDecodeTable>,
}

impl LlamaProvider {
    /// Load a provider from `spec.source`: either a `*.gguf` file (loaded directly via Candle's
    /// native GGUF reader, story 7254) or an HF snapshot directory (config.json + tokenizer.json +
    /// shards). Either way the decoder architecture is dispatched (Llama / Mistral / Qwen3) and the
    /// projections are optionally quantized on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        let requested = spec.quantize.map(|q| match q {
            Quantize::Q4 => QuantSpec::q4(),
            Quantize::Q8 => QuantSpec::q8(),
        });
        let device = select_device().map_err(to_core)?;
        if crate::gguf::is_gguf_path(&spec.source) {
            Self::load_gguf(Path::new(&spec.source), &device, requested)
        } else {
            Self::load_dir(Path::new(&spec.source), &device, requested)
        }
    }

    /// Load from an HF snapshot directory (config.json + tokenizer.json + safetensors shards).
    ///
    /// `requested` is an explicit load-time quantization (`spec.quantize`); when it is `None` the
    /// snapshot's own persisted `quantization` block (written by the [`prepare`](crate::prepare)
    /// writer) is honored, so a `LoadSpec::dense` of a prepared Q4/Q8 snapshot loads quantized.
    fn load_dir(dir: &Path, device: &Device, requested: Option<QuantSpec>) -> CoreResult<Self> {
        let cfg_value = read_json(dir, "config.json")
            .ok_or_else(|| CoreError::Load(format!("read config.json in {}", dir.display())))?;
        let arch = Architecture::from_config(&cfg_value).map_err(to_core)?;
        let weights = Weights::from_dir(dir, device).map_err(to_core)?;
        let (model, mut descriptor) = if arch == Architecture::Qwen35 {
            // Qwen3.6 hybrid decoder: its own config, the VLM-nested `model.language_model` prefix, and
            // a top-level untied `lm_head`. (The 27B VLM-wrapped checkpoint carries a `vision_config`;
            // the text path ignores it — vision is a follow-on story.)
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for_qwen35(&qcfg);
            let m =
                Qwen35Model::from_weights_with(&weights, "model.language_model", qcfg, requested)
                    .map_err(to_core)?;
            (Decoder::Qwen35(m), descriptor)
        } else {
            let cfg = ModelConfig::from_dir(dir).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let quant = requested.or(cfg.quantization);
            let m = CausalLm::from_weights_with(&weights, "", cfg, quant).map_err(to_core)?;
            (Decoder::Causal(m), descriptor)
        };
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let stop_tokens = eos_token_ids(dir);
        let (template, supports_thinking, supports_tools) = load_chat_template(dir);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        Ok(Self {
            descriptor,
            model,
            tokenizer,
            template,
            stop_tokens,
            constraint_table: OnceCell::new(),
        })
    }

    /// Load a single `*.gguf` checkpoint directly into the decoder. The tokenizer prefers a sibling
    /// `tokenizer.json`, falling back to a reconstruction from the GGUF's embedded tokenizer
    /// metadata; the chat template prefers a sibling `tokenizer_config.json`, then the GGUF's own
    /// `chat_template`, then the typed Llama-3 default.
    fn load_gguf(path: &Path, device: &Device, requested: Option<QuantSpec>) -> CoreResult<Self> {
        let ck = GgufCheckpoint::open(path, device).map_err(to_core)?;
        let mut descriptor = descriptor_for(&ck.config);
        let quant = requested.or(ck.config.quantization);
        // GGUF is the dense Llama-family path only (no hybrid Qwen3.6 GGUF remap).
        let model = Decoder::Causal(
            CausalLm::from_weights_with(&ck.weights, "", ck.config.clone(), quant)
                .map_err(to_core)?,
        );

        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let sibling_tokenizer = dir.join("tokenizer.json");
        let tokenizer = if sibling_tokenizer.is_file() {
            Tokenizer::from_file(sibling_tokenizer)?
        } else {
            ck.tokenizer_from_metadata().map_err(to_core)?
        };

        let stop_tokens = if ck.stop_tokens.is_empty() {
            eos_token_ids(dir)
        } else {
            ck.stop_tokens.clone()
        };

        let (template, supports_thinking, supports_tools) = gguf_chat_template(dir, &ck);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        Ok(Self {
            descriptor,
            model,
            tokenizer,
            template,
            stop_tokens,
            constraint_table: OnceCell::new(),
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// Assemble a provider from already-loaded parts with a default Llama-3 template (used by tests
    /// and converters that don't have a `tokenizer_config.json`).
    pub fn from_parts(model: CausalLm, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        Self {
            descriptor: provider_descriptor(),
            model: Decoder::Causal(model),
            tokenizer,
            template: Box::new(Llama3Template),
            stop_tokens,
            constraint_table: OnceCell::new(),
        }
    }
}

/// Adapts a `core_llm::JsonConstraint` to the engine's [`ConstraintMask`] decode seam.
struct JsonMask<'a>(JsonConstraint<'a>);

impl ConstraintMask for JsonMask<'_> {
    fn allowed(&mut self) -> &[bool] {
        self.0.allowed()
    }
    fn accept(&mut self, token: i32) {
        self.0.accept(token as u32);
    }
}

/// Use the model's own Jinja `chat_template` (from `tokenizer_config.json`) when present; otherwise
/// fall back to the typed Llama-3 template. Also reports two template-gated capabilities, detected
/// from the source (not the family, matching the transformers convention):
/// - **thinking** — the template gates an `enable_thinking` kwarg (the Qwen3, … convention; story
///   7707). The Llama-3 fallback never reasons.
/// - **tools** — the template renders tool calls (its source mentions `tool_call`), so it has a
///   `tools` section and the model emits parseable `<tool_call>` blocks (story 7636). Covers the
///   Qwen3.6 XML and the Qwen2.5/Hermes JSON tool templates alike.
fn load_chat_template(dir: &Path) -> (Box<dyn ChatTemplate>, bool, bool) {
    match JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json")) {
        Ok(t) => {
            let supports_thinking = t.source().contains("enable_thinking");
            let supports_tools = t.source().contains("tool_call");
            (Box::new(t), supports_thinking, supports_tools)
        }
        Err(_) => (Box::new(Llama3Template), false, false),
    }
}

/// Pick a chat template for a GGUF load: a sibling `tokenizer_config.json` first, then the GGUF's
/// own embedded `chat_template` metadata, then the typed Llama-3 default. Also reports
/// `supports_thinking` (the chosen template's source gates `enable_thinking`) and `supports_tools`
/// (its source renders tool calls — it mentions `tool_call`).
fn gguf_chat_template(dir: &Path, ck: &GgufCheckpoint) -> (Box<dyn ChatTemplate>, bool, bool) {
    if let Ok(t) = JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json"))
    {
        let supports_thinking = t.source().contains("enable_thinking");
        let supports_tools = t.source().contains("tool_call");
        return (Box::new(t), supports_thinking, supports_tools);
    }
    if let Some(src) = &ck.chat_template {
        let supports_thinking = src.contains("enable_thinking");
        let supports_tools = src.contains("tool_call");
        let bos = ck.bos_token.clone().unwrap_or_default();
        let eos = ck.eos_token.clone().unwrap_or_default();
        return (
            Box::new(JinjaChatTemplate::with_tokens(src.clone(), bos, eos)),
            supports_thinking,
            supports_tools,
        );
    }
    (Box::new(Llama3Template), false, false)
}

/// Whether a rendered prompt ends with an **unclosed** `<think>` block — i.e. the chat template
/// opened reasoning in the prompt (a Qwen3-style thinking/auto generation prompt) so the model
/// generates inside it. True iff the last `<think>` occurs after the last `</think>` (or there is no
/// close), so the segmenter is primed into the Thinking channel.
fn prompt_opens_thinking(prompt: &str) -> bool {
    match prompt.rfind("<think>") {
        None => false,
        Some(open) => prompt.rfind("</think>").is_none_or(|close| open > close),
    }
}

/// Run a piece of answer-channel text through the tool-call segmenter when active, returning the
/// plain-content runs to stream (tool-call blocks lifted out + parsed into [`ToolCallSegmenter`]).
/// With no segmenter the text passes straight through, so the non-tools path is byte-identical to
/// before.
fn tool_pieces(seg: &mut Option<ToolCallSegmenter>, text: &str) -> Vec<String> {
    match seg {
        Some(ts) => ts.push(text),
        None => vec![text.to_string()],
    }
}

/// Emit one answer-channel content `piece` as a [`Channel::Content`] token event with the gap-free
/// `emit_index`, accumulating it into `streamed`. Shared by the streaming loop and the
/// end-of-generation tails; `*emit_index` / `*last_id` advance only when text is actually emitted, so
/// the contract's token index stays gap-free across stripped reasoning markers and lifted-out
/// tool-call blocks.
fn emit_content(
    piece: String,
    id: u32,
    streamed: &mut String,
    emit_index: &mut usize,
    last_id: &mut u32,
    on_event: &mut dyn FnMut(CoreEvent),
) {
    streamed.push_str(&piece);
    *last_id = id;
    on_event(CoreEvent::Token {
        id,
        text: piece,
        index: *emit_index,
        channel: Channel::Content,
    });
    *emit_index += 1;
}

impl TextLlm for LlamaProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
    ) -> CoreResult<TextLlmOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled); // typed pre-inference cancel
        }

        // Render the conversation and tokenize. The template already includes BOS, so encode without
        // auto special tokens. `enable_thinking` flows into the template kwarg so a no-think
        // (Disabled) request injects the model's closed `<think></think>` generation prompt; Auto
        // omits the kwarg (template default).
        let prompt = self.template.render_with(
            &req.messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.enable_thinking_kwarg(),
                tools: &req.tools,
            },
        )?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let prompt_len = prompt_ids.len();

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: map_sampling(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };

        // Structured-output constraint: build a JSON mask over the cached decode table.
        let mut json_mask = match req.constraint {
            Some(Constraint::Json) => {
                let table = self
                    .constraint_table
                    .get_or_init(|| self.tokenizer.constraint_decode_table());
                Some(JsonMask(JsonConstraint::new(
                    table,
                    self.stop_tokens.iter().map(|&i| i as u32),
                )))
            }
            None => None,
        };

        // A reasoning segmenter when the model advertises a thinking mode: it splits the decoded
        // stream into `<think>…</think>` reasoning vs answer (markers stripped) across the Thinking /
        // Content channels. `None` otherwise, so a non-thinking provider stays on the original
        // single-channel path (byte-identical streaming).
        let thinking_active = self.descriptor.capabilities.supports_thinking;
        let mut segmenter = thinking_active.then(ThinkingSegmenter::default);
        // Some chat templates open the reasoning block *in the prompt* (e.g. a Qwen3 generation
        // prompt ending `…<|im_start|>assistant\n<think>\n`), so the model generates inside the block
        // and only emits the closing `</think>`. Prime the segmenter into the Thinking channel by
        // feeding it that already-rendered opening marker (stripped, emits nothing); a Disabled
        // request renders a *closed* `<think></think>`, so this correctly does not prime.
        if let Some(seg) = segmenter.as_mut() {
            if prompt_opens_thinking(&prompt) {
                let _ = seg.push("<think>");
            }
        }
        // A tool-call segmenter when the request offers tools and the model's template renders them:
        // it lifts `<tool_call>` blocks out of the answer channel (markup excluded from the streamed
        // text) and parses them into structured calls (story 7636). `None` otherwise, so a no-tools
        // request flows straight through `tool_pieces` unchanged.
        let tools_active = self.descriptor.capabilities.supports_tools && !req.tools.is_empty();
        let mut tool_seg = tools_active.then(|| ToolCallSegmenter::new(&req.tools));
        // Reasoning text (Thinking channel) and the answer (Content channel), accumulated as the
        // segmenter releases each span; the answer becomes the result text when thinking is active.
        let mut thinking_buf = String::new();
        let mut streamed = String::new();
        // Contract token index over *emitted* events, not the raw decode step — stripped
        // `<think>`/`</think>` marker tokens produce no event, so this stays gap-free (and equals the
        // step in the common one-delta-per-token, non-thinking case).
        let mut emit_index = 0usize;
        let mut last_id = 0u32; // id of the last emitted token, for the flushed-tail events

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix). The
        // segmenter (when active) splits each delta into reasoning vs answer.
        let tokenizer = &self.tokenizer;
        let out = {
            let mut acc: Vec<u32> = Vec::new();
            let mut shown = 0usize;
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, step } = ev {
                    let id = id as u32;
                    acc.push(id);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if text.len() > shown {
                            let delta = text[shown..].to_string();
                            shown = text.len();
                            match segmenter.as_mut() {
                                Some(seg) => {
                                    for span in seg.push(&delta) {
                                        match span.channel {
                                            // Reasoning streams straight out (markers already stripped).
                                            Channel::Thinking => {
                                                thinking_buf.push_str(&span.text);
                                                last_id = id;
                                                on_event(CoreEvent::Token {
                                                    id,
                                                    text: span.text,
                                                    index: emit_index,
                                                    channel: Channel::Thinking,
                                                });
                                                emit_index += 1;
                                            }
                                            // Answer text → tool segmenter (lifts out tool-call
                                            // blocks) → emit.
                                            Channel::Content => {
                                                for piece in tool_pieces(&mut tool_seg, &span.text)
                                                {
                                                    emit_content(
                                                        piece,
                                                        id,
                                                        &mut streamed,
                                                        &mut emit_index,
                                                        &mut last_id,
                                                        &mut *on_event,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                None if tool_seg.is_some() => {
                                    // No reasoning split, but tools are active: route the whole delta
                                    // through the tool segmenter (the answer is the only channel).
                                    for piece in tool_pieces(&mut tool_seg, &delta) {
                                        emit_content(
                                            piece,
                                            id,
                                            &mut streamed,
                                            &mut emit_index,
                                            &mut last_id,
                                            &mut *on_event,
                                        );
                                    }
                                }
                                None => {
                                    // Neither reasoning nor tools: the original single-channel path,
                                    // byte-identical to before either feature existed (raw `step`
                                    // index, no streamed accumulation).
                                    on_event(CoreEvent::Token {
                                        id,
                                        text: delta,
                                        index: step,
                                        channel: Channel::Content,
                                    });
                                }
                            }
                        }
                    }
                }
            };
            let constraint = json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask);
            generate_with(
                &self.model,
                &prompt_ids,
                &config,
                &req.cancel,
                &mut sink,
                constraint,
            )
            .map_err(to_core)?
        };

        // End-of-generation tails, in pipeline order. First the reasoning segmenter's held-back
        // partial marker (it turned out not to begin a marker) as current-channel text — reasoning
        // straight out, answer through the tool segmenter; then the tool segmenter's own tail (a held
        // partial `<tool_call>` / an unterminated block surfaced as content).
        if let Some(seg) = segmenter.as_mut() {
            for span in seg.flush() {
                match span.channel {
                    Channel::Thinking => {
                        thinking_buf.push_str(&span.text);
                        on_event(CoreEvent::Token {
                            id: last_id,
                            text: span.text,
                            index: emit_index,
                            channel: Channel::Thinking,
                        });
                        emit_index += 1;
                    }
                    Channel::Content => {
                        for piece in tool_pieces(&mut tool_seg, &span.text) {
                            emit_content(
                                piece,
                                last_id,
                                &mut streamed,
                                &mut emit_index,
                                &mut last_id,
                                &mut *on_event,
                            );
                        }
                    }
                }
            }
        }
        if let Some(ts) = tool_seg.as_mut() {
            for piece in ts.flush() {
                emit_content(
                    piece,
                    last_id,
                    &mut streamed,
                    &mut emit_index,
                    &mut last_id,
                    &mut *on_event,
                );
            }
        }

        // Result text: the streamed answer when thinking or tools are active (either means the
        // streamed channel is the authoritative answer, with reasoning / tool-call markup removed);
        // otherwise the original decode-all-tokens path (byte-identical to the no-feature case).
        // Reasoning and tool calls, if the model produced any, are reported separately (their markup
        // excluded from `text`).
        let text = if thinking_active || tools_active {
            streamed
        } else {
            let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&gen_u32, true)?
        };
        let thinking = (!thinking_buf.is_empty()).then_some(thinking_buf);
        let tool_calls = tool_seg.map(|mut ts| ts.take_calls()).unwrap_or_default();
        let finish = map_finish(out.finish_reason);
        let usage = Usage {
            prompt_tokens: prompt_len as u32,
            generated_tokens: out.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            text,
            thinking,
            tool_calls,
            usage,
            finish_reason: Some(finish),
        })
    }
}

/// The descriptor for the `candle-llama` provider (constructible without loading weights; used for
/// link-time registration and registry discovery).
pub fn provider_descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.to_string(),
        family: "llama".to_string(),
        backend: "candle".to_string(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            // Text-only today; the VLM path flips this on for a vision provider.
            supports_vision: false,
            // No controllable reasoning mode yet (a separate story); the contract requires an
            // explicit enable-thinking request to be rejected, which validate_request enforces.
            supports_thinking: false,
            // Weightless default: conservative. The load path flips this on when the loaded model's
            // chat template renders tool calls (story 7636).
            supports_tools: false,
            // JSON-constrained decoding.
            supported_constraints: vec![Constraint::Json],
        },
    }
}

/// A descriptor reflecting a *loaded* model: family from the dispatched architecture and the context
/// length from `config.json`. (Quantization state is reported via [`LlamaProvider::is_quantized`].)
fn descriptor_for(cfg: &ModelConfig) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = cfg.architecture.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d
}

/// A descriptor for a loaded Qwen3.6 (`qwen3_5`) hybrid decoder. The context length comes from the
/// [`Qwen35Config`] (which `ModelConfig` does not represent). Text-only here; the vision path is a
/// follow-on story.
fn descriptor_for_qwen35(cfg: &Qwen35Config) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = Architecture::Qwen35.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d
}

/// Resolve the stop-token ids for a snapshot directory. Prefers `generation_config.json` (HF's
/// canonical "how to generate" source — where models like Qwen3.6 put the turn-end ids; its
/// top-level `config.json` `eos_token_id` is null), then `config.json` (top-level, then the nested
/// `text_config` of a VLM wrapper), then the Llama-3 defaults. Each `eos_token_id` may be a single
/// int or an array.
pub fn eos_token_ids(dir: &Path) -> Vec<i32> {
    let fallback = vec![128001, 128008, 128009]; // <|end_of_text|>, <|eom_id|>, <|eot_id|>
    if let Some(ids) = read_json(dir, "generation_config.json")
        .as_ref()
        .and_then(|v| parse_token_ids(v.get("eos_token_id")))
    {
        return ids;
    }
    if let Some(v) = read_json(dir, "config.json") {
        if let Some(ids) = parse_token_ids(v.get("eos_token_id"))
            .or_else(|| parse_token_ids(v.get("text_config").and_then(|t| t.get("eos_token_id"))))
        {
            return ids;
        }
    }
    fallback
}

/// Read and parse a JSON file in `dir`, or `None` if absent / malformed.
fn read_json(dir: &Path, name: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join(name)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Parse an `eos_token_id`-style field — a single int or an array of ints — into a non-empty id list.
fn parse_token_ids(v: Option<&serde_json::Value>) -> Option<Vec<i32>> {
    match v? {
        serde_json::Value::Number(n) => n.as_i64().map(|x| vec![x as i32]),
        serde_json::Value::Array(a) => {
            let ids: Vec<i32> = a
                .iter()
                .filter_map(|x| x.as_i64().map(|x| x as i32))
                .collect();
            (!ids.is_empty()).then_some(ids)
        }
        _ => None,
    }
}

fn map_sampling(s: &Sampling) -> SamplingParams {
    SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        repetition_penalty: s.repetition_penalty,
        repetition_context: s.repetition_context,
    }
}

fn map_finish(f: FinishReason) -> CoreFinish {
    match f {
        FinishReason::StopToken => CoreFinish::Stop,
        FinishReason::MaxTokens => CoreFinish::Length,
        FinishReason::Cancelled => CoreFinish::Cancelled,
    }
}

/// Bridge an engine error into the contract error, preserving the typed cancellation / capability /
/// load variants (do not stringify those).
pub(crate) fn to_core(e: crate::Error) -> CoreError {
    match e {
        crate::Error::Canceled => CoreError::Canceled,
        crate::Error::Unsupported(m) => CoreError::Unsupported(m),
        crate::Error::MissingTensor(m) => CoreError::Load(format!("missing tensor: {m}")),
        crate::Error::Config(m) => CoreError::Load(m),
        crate::Error::Io(e) => CoreError::Io(e),
        other => CoreError::backend(other),
    }
}

// Register `candle-llama` into core-llm's provider registry at link time.
inventory::submit! {
    core_llm::TextLlmRegistration {
        descriptor: provider_descriptor,
        load: load_registered,
        can_load,
    }
}

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `candle-llama` provider serve the model at
/// `spec.source`? A `*.gguf` file is a text-model container this provider loads directly, so it is
/// accepted by extension. Otherwise this reads **only** `config.json` and runs the same
/// [`Architecture::from_config`] dispatch the loader uses (Llama / Mistral / Qwen2 / Qwen3 /
/// Qwen2-MoE / Gemma2 / GLM-4 / DeepSeek-V2 / Phi-3) — it never opens a safetensors shard, so
/// `core-llm`'s `load_for_model` can resolve a provider by model without loading weights. A
/// multimodal snapshot (a `vision_config` block — including a VLM whose `model_type` substring-
/// matches a text family, e.g. `mllama`) is declined so the vision provider claims it instead.
pub fn can_load(spec: &LoadSpec) -> bool {
    if crate::gguf::is_gguf_path(&spec.source) {
        return true;
    }
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() {
        dir.join("config.json")
    } else {
        dir.to_path_buf()
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    // A multimodal snapshot (a `vision_config` block) is normally declined so a vision provider claims
    // it — EXCEPT Qwen3.6 (`qwen3_5`), which this provider serves as a **text** model: its 27B/35B
    // checkpoints are VLM-wrapped, but the hybrid text decoder stands alone (vision is a follow-on
    // story). So a `vision_config` config is declined only when it is not a qwen3_5.
    let arch = Architecture::from_config(&v);
    if v.get("vision_config").is_some() && !matches!(arch, Ok(Architecture::Qwen35)) {
        return false;
    }
    arch.is_ok()
}

#[cfg(test)]
mod tests {
    use super::prompt_opens_thinking;

    #[test]
    fn prompt_opens_thinking_matches_template_modes() {
        // A Qwen3-style thinking/auto generation prompt opens the block and leaves it unclosed.
        assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
        // A no-think (Disabled) prompt renders a closed `<think></think>`.
        assert!(!prompt_opens_thinking(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        // A prior closed reasoning turn followed by a fresh open block still opens.
        assert!(prompt_opens_thinking(
            "<think>\nold\n</think>\n\nq<|im_start|>assistant\n<think>\n"
        ));
        // No reasoning markers at all (a non-thinking template).
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
    }
}
