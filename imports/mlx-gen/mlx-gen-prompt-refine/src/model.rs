//! Prompt-refine provider registration and the text-in / text-out generation path.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use mlx_rs::Array;

use mlx_gen::gen_core::{
    self, TextLlm, TextLlmDescriptor, TextLlmFinishReason, TextLlmOutput, TextLlmRequest,
};
use mlx_gen::registry::TextLlmRegistration;
use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{default_seed, Error, LoadSpec, Progress, Result, WeightsSource};

use crate::llama::{sample_token, LlamaConfig, LlamaModel, SplitMix64};
use crate::prompt::{
    build_chat_text, capabilities, BEGIN_OF_TEXT_TOKEN_ID, DEFAULT_MAX_CONTEXT_TOKENS,
    END_OF_TEXT_TOKEN_ID, EOT_TOKEN_ID, PROMPT_REFINE_FAMILY, PROMPT_REFINE_ID,
};

/// The loaded model + tokenizer + resolved stop tokens (cached after the first `generate`).
struct Engine {
    model: LlamaModel,
    tokenizer: TextTokenizer,
    stop_ids: Vec<i32>,
}

impl Engine {
    fn load(root: &Path) -> Result<Self> {
        let cfg_path = root.join("config.json");
        // Config carries the Llama-3.2 dims, GQA, and the rope_scaling block.
        let cfg = LlamaConfig::from_json(&cfg_path)?;
        let weights = Weights::from_dir(root)?;
        let model = LlamaModel::from_weights(&weights, "", cfg)?;

        let tokenizer = TextTokenizer::from_file(
            root.join("tokenizer.json"),
            TokenizerConfig {
                max_length: DEFAULT_MAX_CONTEXT_TOKENS,
                pad_token_id: END_OF_TEXT_TOKEN_ID,
                chat_template: ChatTemplate::None, // the template is hand-assembled in `prompt`
                pad_to_max_length: false,
            },
        )
        .map_err(|e| Error::Msg(format!("prompt_refine: load tokenizer: {e}")))?;

        Ok(Self {
            stop_ids: stop_ids_from_config(&cfg_path)?,
            model,
            tokenizer,
        })
    }
}

/// Generation stop tokens: the config's `eos_token_id` (single or array) unioned with the Llama-3
/// `<|eot_id|>` and `<|end_of_text|>` (instruct models stop on `<|eot_id|>`, which some `config.json`s
/// omit from `eos_token_id`).
fn stop_ids_from_config(path: &Path) -> Result<Vec<i32>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Msg(format!("prompt_refine: read {}: {e}", path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("prompt_refine: parse config.json: {e}")))?;
    let mut ids: Vec<i32> = Vec::new();
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            if let Some(x) = n.as_i64() {
                ids.push(x as i32);
            }
        }
        Some(serde_json::Value::Array(a)) => {
            for e in a {
                if let Some(x) = e.as_i64() {
                    ids.push(x as i32);
                }
            }
        }
        _ => {}
    }
    for must in [EOT_TOKEN_ID, END_OF_TEXT_TOKEN_ID] {
        if !ids.contains(&must) {
            ids.push(must);
        }
    }
    Ok(ids)
}

/// The MLX prompt-refine text-LLM provider. Lazily loads weights on the first `generate` and caches
/// them (the loaded `Engine` holds MLX `Array`s, which are neither `Send` nor `Sync`, so it lives
/// behind the `Mutex` rather than an `Arc`; MLX is single-threaded, so holding the lock across a
/// generation is fine — there is no concurrent `generate` on one provider).
pub struct PromptRefiner {
    descriptor: TextLlmDescriptor,
    root: PathBuf,
    engine: Mutex<Option<Engine>>,
}

impl PromptRefiner {
    /// The rich-`Result` body behind [`TextLlm::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts `mlx_rs` device exceptions transparently; the
    /// trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn run(
        &self,
        req: &TextLlmRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<TextLlmOutput> {
        let mut guard = self
            .engine
            .lock()
            .expect("prompt_refine engine cache mutex poisoned");
        if guard.is_none() {
            *guard = Some(Engine::load(&self.root)?);
        }
        let engine = guard.as_ref().expect("engine loaded above");

        // Build the Llama-3 chat text (system + user → assistant prompt), map to ids without
        // auto-specials (the template carries the header/eot tokens as literal strings), then prepend
        // the <|begin_of_text|> BOS id.
        let chat = build_chat_text(&req.system, &req.prompt);
        let encoded = engine
            .tokenizer
            .encode_ids(&chat, false)
            .map_err(|e| Error::Msg(format!("prompt_refine: tokenize: {e}")))?;
        let mut all_tokens: Vec<i32> = Vec::with_capacity(encoded.len() + 1);
        all_tokens.push(BEGIN_OF_TEXT_TOKEN_ID);
        all_tokens.extend(encoded.iter().copied());

        // Sampling: temperature <= 0 → greedy argmax (seed unused); else top-p nucleus. Seed is
        // caller-pinned or a fresh per-call draw, so a fixed seed reproduces the rewrite.
        let mut rng = SplitMix64::new(req.sampling.seed.unwrap_or_else(default_seed));
        let temperature = req.sampling.temperature;
        let top_p = req.sampling.top_p;

        let mut cache = engine.model.new_cache();
        let total = req.sampling.max_new_tokens;
        let mut generated: Vec<i32> = Vec::new();
        let mut index_pos = 0usize;
        let mut finish = TextLlmFinishReason::MaxTokens;
        for step in 0..total {
            // Cooperative cancel between tokens → return the partial reply marked Cancelled (the
            // pre-inference already-cancelled case is the typed Err in `generate`). Pulling logits to
            // the host for sampling forces evaluation each step, so this check is effective despite
            // MLX's lazy graph (cancel-lifecycle gotcha).
            if req.cancel.is_cancelled() {
                finish = TextLlmFinishReason::Cancelled;
                break;
            }
            // With the KV cache, feed the whole prompt on step 0 and one token thereafter.
            let (context_size, context_index) = if step > 0 {
                (1usize, index_pos)
            } else {
                (all_tokens.len(), 0usize)
            };
            let ctxt = &all_tokens[all_tokens.len() - context_size..];
            let input = Array::from_slice(ctxt, &[1, ctxt.len() as i32]);
            let logits = engine
                .model
                .decode_logits(&input, &mut cache, context_index as i32)?;
            index_pos += ctxt.len();

            let next = sample_token(&logits, temperature, top_p, &mut rng)?;
            all_tokens.push(next);
            if engine.stop_ids.contains(&next) {
                finish = TextLlmFinishReason::StopToken;
                break;
            }
            generated.push(next);
            // 1-based step count = tokens emitted so far (monotone, ≤ total).
            on_progress(Progress::Step {
                current: generated.len() as u32,
                total,
            });
        }

        let gen_u32: Vec<u32> = generated.iter().map(|&id| id as u32).collect();
        let text = engine
            .tokenizer
            .decode(&gen_u32, true)
            .map(|t| t.trim().to_owned())
            .map_err(|e| Error::Msg(format!("prompt_refine: detokenize: {e}")))?;

        Ok(TextLlmOutput {
            text,
            generated_tokens: Some(generated.len() as u32),
            finish_reason: Some(finish),
        })
    }
}

impl TextLlm for PromptRefiner {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(PROMPT_REFINE_ID, req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<TextLlmOutput> {
        self.validate(req)?;
        // An already-cancelled request returns the typed `Canceled` before any inference (or weight
        // load) runs — the TextLlm pre-inference cancellation contract (sc-5500).
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        self.run(req, on_progress).map_err(Into::into)
    }
}

/// The prompt-refine text-LLM descriptor (MLX backend; mac-only).
pub fn descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROMPT_REFINE_ID,
        family: PROMPT_REFINE_FAMILY,
        backend: "mlx",
        capabilities: capabilities(),
    }
}

/// Construct a lazy MLX prompt-refine provider. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a Llama-3.2-3B-Instruct snapshot (`config.json`, `tokenizer.json`,
/// `model-*.safetensors`). Adapters / quantization are rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn TextLlm>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "prompt_refine expects a snapshot directory (config.json, tokenizer.json, \
                 model-*.safetensors), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "mlx prompt-refine does not support LoRA/LoKr".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "mlx prompt-refine does not support on-the-fly quantization".into(),
        ));
    }
    Ok(Box::new(PromptRefiner {
        descriptor: descriptor(),
        root,
        engine: Mutex::new(None),
    }))
}

inventory::submit! {
    TextLlmRegistration { descriptor, load }
}

/// Force-link hook (keeps the `inventory::submit!` registration from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::registry;
    use mlx_gen::runtime::{AdapterKind, AdapterSpec};

    #[test]
    fn descriptor_advertises_prompt_refine_surface() {
        let d = descriptor();
        assert_eq!(d.id, PROMPT_REFINE_ID);
        assert_eq!(d.family, "llama");
        assert_eq!(d.backend, "mlx");
        assert!(d.capabilities.supports_system_prompt);
        assert!(d.capabilities.mac_only);
        assert_eq!(
            d.capabilities.max_new_tokens,
            crate::prompt::MAX_NEW_TOKENS_CAP
        );
    }

    #[test]
    fn registers_and_resolves_as_mlx_textllm() {
        // Lazy load: a nonexistent dir still resolves (weights are only touched at generate time).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry::load_textllm(PROMPT_REFINE_ID, &spec).expect("registered");
        assert_eq!(t.descriptor().id, PROMPT_REFINE_ID);
        assert_eq!(t.descriptor().backend, "mlx");
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_rejects_adapters() {
        let spec = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&spec).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn validate_rejects_empty_prompt_and_overlong_tokens() {
        let p = PromptRefiner {
            descriptor: descriptor(),
            root: "/nonexistent".into(),
            engine: Mutex::new(None),
        };
        // empty prompt
        assert!(p.validate(&TextLlmRequest::default()).is_err());
        // max_new_tokens over the advertised cap
        let mut req = TextLlmRequest {
            prompt: "rewrite this".to_owned(),
            ..Default::default()
        };
        req.sampling.max_new_tokens = crate::prompt::MAX_NEW_TOKENS_CAP + 1;
        assert!(p.validate(&req).is_err());
    }

    #[test]
    fn stop_ids_union_eos_and_llama3_turn_tokens() {
        let dir = std::env::temp_dir().join("mlx_gen_prompt_refine_stop_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"eos_token_id": [128001, 128008, 128009]}"#).unwrap();
        let ids = stop_ids_from_config(&path).unwrap();
        assert!(ids.contains(&128001));
        assert!(ids.contains(&128008));
        assert!(ids.contains(&EOT_TOKEN_ID)); // 128009
        let _ = std::fs::remove_file(&path);

        // A single-int eos still unions in <|eot_id|> + <|end_of_text|>.
        std::fs::write(&path, r#"{"eos_token_id": 2}"#).unwrap();
        let ids = stop_ids_from_config(&path).unwrap();
        assert!(ids.contains(&2));
        assert!(ids.contains(&EOT_TOKEN_ID));
        assert!(ids.contains(&END_OF_TEXT_TOKEN_ID));
        let _ = std::fs::remove_file(&path);
    }
}
