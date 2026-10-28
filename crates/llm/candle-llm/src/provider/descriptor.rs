//! [`LlamaProvider`](super::LlamaProvider)'s weightless surface (sc-27302): the descriptor and its
//! speculative depths, the explicit registration, the structural `can_load*` probes, and the
//! snapshot JSON readers (EOS ids, sampling defaults) they share.

use super::*;

/// The deepest speculative proposal any proposer runs on Candle (sc-24438, epic sc-24432 E4):
/// drafts per verify step, so the verify forward carries at most `1 + 7 = 8` token rows.
///
/// Why 7: two decode kernels serve at most 8 token rows per launch, and a verify of up to 8 rows
/// stays on them — one launch per projection, a cost flat in the row count, and the plain step's
/// numerics:
///
/// * [`NVFP4_GEMV_MAX_ROWS`](candle_quant_kernels::NVFP4_GEMV_MAX_ROWS) (8) — the fused NVFP4
///   decode GEMV (sc-24136), the W4A16 kernel every plain decode step of an NVFP4 checkpoint runs
///   on unquantized bf16 activations. A 9-row verify is refused by the GEMV and runs the cuBLASLt
///   W4A4 GEMM instead — pad to 16 rows, quantize the activation with a host sync — so it is both
///   slower per step and numerically a different product from the plain step greedy speculation
///   must reproduce.
/// * GGML's **MMVQ** — the quantized mat-vec kernel a Q4/Q8 (GGML-block) checkpoint's decode step
///   runs, which candle's CUDA backend dispatches for batches of 1–8 rows only
///   (`candle-core/src/quantized/cuda.rs`, `QCudaStorage::fwd`); a 9-row verify moves to the MMQ
///   tile kernel (or the dequantize + GEMM fallback), the same cliff.
///
/// A bf16 checkpoint's cuBLAS GEMM has no such cliff; it is capped at the same depth by policy, so
/// a depth means the same thing on every Candle checkpoint, every proposer and weight format.
pub const SPECULATIVE_MAX_DEPTH: u32 = candle_quant_kernels::NVFP4_GEMV_MAX_ROWS as u32 - 1;

/// The deepest prompt-lookup proposal the provider runs: [`SPECULATIVE_MAX_DEPTH`]. The bound is
/// the provider's, not the proposer's (an n-gram match can be longer): every extra depth also adds
/// a static-KV overshoot position and — on the qwen3_5 hybrid — one more recurrent checkpoint per
/// linear layer (`K + 2` in the ring), all priced by admission; a copied span past it gains
/// little over the next verify step starting a fresh lookup.
pub const PROMPT_LOOKUP_MAX_DEPTH: u32 = SPECULATIVE_MAX_DEPTH;

/// The deepest MTP proposal a Qwen3.8 head runs: [`SPECULATIVE_MAX_DEPTH`] (each MTP draft also
/// costs one sequential predictor forward, so the verify width is the binding bound, not the
/// head). Advertised on both the legacy `mtp` field and the per-proposer list (sc-24438).
pub const MTP_MAX_DEPTH: u32 = SPECULATIVE_MAX_DEPTH;

/// The per-proposer depths [`Speculative::Auto`](core_llm::Speculative::Auto) runs and every
/// advertisement recommends for a model on `device`: its row of the decode-defaults table (E5,
/// [`crate::device::decode_defaults`]).
pub(super) fn recommended_depths(device: &Device) -> &'static core_llm::RecommendedDepths {
    &crate::device::decode_defaults(device).recommended_depths
}

/// The prompt-lookup advertisement every `candle-llama` decoder carries (the shared
/// [`core_llm::prompt_lookup_capabilities`] at [`PROMPT_LOOKUP_MAX_DEPTH`], recommending
/// `backend`'s defaults-table depth): both decoder families (the llama-family `CausalLm` and the
/// qwen3_5 hybrid) run through the step-seam engine, so the proposer needs nothing from the
/// checkpoint.
pub(super) fn prompt_lookup_capabilities(backend: core_llm::DecodeBackend) -> ProposerCapabilities {
    core_llm::prompt_lookup_capabilities(
        PROMPT_LOOKUP_MAX_DEPTH,
        &backend.defaults().recommended_depths,
    )
}

/// The descriptor for the `candle-llama` provider (constructible without loading weights; used for
/// explicit catalog composition and inspection). Without a device, its recommended depths are the
/// defaults-table row of the device a load would open (CUDA in a `cuda` build, else Metal or the
/// CPU, as [`select_device`] chooses).
pub fn provider_descriptor() -> TextLlmDescriptor {
    provider_descriptor_on(crate::device::decode_backend_for(cfg!(feature = "cuda")))
}

/// [`provider_descriptor`] for a model decoding on `backend`'s row of the defaults table.
pub(super) fn provider_descriptor_on(backend: core_llm::DecodeBackend) -> TextLlmDescriptor {
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
            // Text-only candle-llama accepts no video content.
            supports_video: false,
            // Weightless default: conservative. The load path flips this on for a Gemma 4
            // checkpoint that actually ships an audio projector (sc-18772).
            supports_audio: false,
            // No controllable reasoning mode yet (a separate story); the contract requires an
            // explicit enable-thinking request to be rejected, which validate_request enforces.
            supports_thinking: false,
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            model_sampling_defaults: None,
            supports_preserve_thinking: false,
            // Weightless default: conservative. The load path flips this on when the loaded model's
            // chat template renders tool calls (story 7636).
            supports_tools: false,
            mtp: None,
            // Prompt lookup runs on every decoder this provider loads (sc-24433); a checkpoint's
            // MTP head is advertised at load through the legacy `mtp` field.
            speculative: vec![prompt_lookup_capabilities(backend)],
            // JSON-constrained decoding.
            supported_constraints: vec![ConstraintKind::Json],
        },
    }
}

/// A descriptor reflecting a *loaded* model: family from the dispatched architecture and the context
/// length from `config.json`. (Quantization state is reported via [`LlamaProvider::is_quantized`].)
pub(super) fn descriptor_for(cfg: &ModelConfig, device: &Device) -> TextLlmDescriptor {
    let mut d = provider_descriptor_on(crate::device::decode_backend(device));
    d.family = cfg.architecture.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d
}

/// A descriptor for a loaded Qwen3.6 (`qwen3_5`) hybrid decoder. The context length comes from the
/// [`Qwen35Config`] (which `ModelConfig` does not represent). Text-only here; the vision path is a
/// follow-on story.
pub(super) fn descriptor_for_qwen35(cfg: &Qwen35Config, device: &Device) -> TextLlmDescriptor {
    let mut d = provider_descriptor_on(crate::device::decode_backend(device));
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
pub(super) fn read_json(dir: &Path, name: &str) -> Option<serde_json::Value> {
    let text = crate::__model_input_reader!(
        std::fs::read_to_string(dir.join(name)),
        "crates/llm/candle-llm/src/provider.rs::read_json::std::fs::read_to_string@267613",
        ArtifactInput,
        dir.join(name),
        |__model_input_path| std::fs::read_to_string(__model_input_path)
    )
    .ok()?;
    serde_json::from_str(&text).ok()
}

/// Parse an `eos_token_id`-style field — a single int or an array of ints — into a non-empty id list.
pub(super) fn parse_token_ids(v: Option<&serde_json::Value>) -> Option<Vec<i32>> {
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

/// Official Bonsai source presets exposed as discovery metadata. The card also explicitly sets
/// `min_p = 0.0` in both modes; that disabled no-op needs no sampler field or runtime operator.
pub(super) fn bonsai_sampling_defaults() -> ModelSamplingDefaults {
    ModelSamplingDefaults {
        thinking: Sampling {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            presence_penalty_generated_only: false,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
        non_thinking: Sampling {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            presence_penalty: 1.5,
            presence_penalty_generated_only: false,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
    }
}

/// Ordinary registration used by explicit runtime bundles.
pub const REGISTRATION: core_llm::TextLlmRegistration = core_llm::TextLlmRegistration {
    descriptor: provider_descriptor,
    load: load_registered,
    can_load,
    // Per-snapshot vision probe: the static descriptor reports `supports_vision=false` (most
    // snapshots are text-only), but a Qwen3.6 / Qwen3-VL checkpoint with a `vision_config` IS
    // vision-capable — this provider loads its ViT tower alongside the decoder. The probe lets a
    // vision-required model-first load resolve it without reading weights.
    weightless_vision: Some(can_load_vision),
    // Per-snapshot audio probe (sc-18772): Gemma 4 unified is the one architecture served here with
    // an audio path, and the same registration also serves text-only checkpoints, so the static
    // descriptor stays `supports_audio=false` and this probe carries the per-snapshot truth.
    weightless_audio: Some(can_load_audio),
};

pub(super) fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}

/// Weightless per-snapshot vision probe (core-llm `weightless_vision`): does this provider serve the
/// snapshot at `spec.source` *with* vision? True for a Qwen3.6 (`qwen3_5`) / Qwen3-VL (`qwen3_vl`)
/// HF checkpoint carrying a `vision_config` (the ViT tower loads alongside the decoder), and for a
/// Gemma 4 unified checkpoint carrying a `vision_config` + `image_token_id` (its encoder-free vision
/// embedder loads alongside the decoder, sc-18772). Reads only `config.json` — never a weight shard.
/// GGUF is declined (candle's GGUF path is dense text-only).
pub fn can_load_vision(spec: &LoadSpec) -> bool {
    if crate::gguf::is_gguf_path(&spec.source) {
        return crate::prism_checkpoint::PrismGgufCheckpoint::is_prism(Path::new(&spec.source))
            .unwrap_or(false)
            && spec.projector_source.as_deref().is_some_and(|path| {
                crate::prism_vision_gguf::PrismVisionGguf::is_qwen3vl_merger(Path::new(path))
            });
    }
    let Some(v) = probe_config(spec) else {
        return false;
    };
    if v.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
        return spec.projector_source.is_none()
            && v.get("vision_config").is_some()
            && v.pointer("/components/vision").and_then(Value::as_bool) == Some(true);
    }
    let qwen_vl = v.get("vision_config").is_some()
        && matches!(
            Architecture::from_config(&v),
            Ok(Architecture::Qwen35) | Ok(Architecture::Qwen3Vl)
        );
    // Gemma 4 is deliberately absent: its vision path is loaded but unvalidated and therefore
    // unadvertised (see `gemma4_vision_is_validated`). The probe must agree with the loaded
    // descriptor, or a model-first vision-required load would resolve here and then be rejected.
    qwen_vl
}

/// Weightless per-snapshot **audio** probe (core-llm `weightless_audio`, sc-18772): does this
/// provider serve the snapshot at `spec.source` *with* audio? Gemma 4 unified is the only
/// architecture served here that has an audio path; like vision it is per-snapshot, so the static
/// descriptor stays `supports_audio=false` and this probe is what lets a model-first audio-required
/// load resolve here. Reads only `config.json`; GGUF is declined (dense text-only).
pub fn can_load_audio(spec: &LoadSpec) -> bool {
    let Some(v) = probe_config(spec) else {
        return false;
    };
    gemma4_multimodal(&v, "audio_config", "audio_token_id")
}

/// Read and parse a snapshot's `config.json` for a weightless probe, or `None` when the source is a
/// GGUF path (no `config.json`, and candle's GGUF path is dense text-only) or is unreadable.
pub(super) fn probe_config(spec: &LoadSpec) -> Option<serde_json::Value> {
    if crate::gguf::is_gguf_path(&spec.source) {
        return None;
    }
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() {
        dir.join("config.json")
    } else {
        dir.to_path_buf()
    };
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<serde_json::Value>(&text).ok()
}

/// Whether `v` is a Gemma 4 checkpoint declaring the named front-end block (`vision_config` /
/// `audio_config`) **and** the token id that front-end splices at.
///
/// The token-id check is not redundant with the block: a config carrying a `vision_config` but no
/// `image_token_id` has no row to splice image features into, so advertising vision for it would be
/// exactly the advertised-but-absent case. The load path refuses such a config for the same reason,
/// which keeps probe and loader agreeing.
pub(super) fn gemma4_multimodal(v: &serde_json::Value, block: &str, token_key: &str) -> bool {
    if v.get(block).is_none() {
        return false;
    }
    if !matches!(Architecture::from_config(v), Ok(a) if a.is_gemma4()) {
        return false;
    }
    v.get(token_key).and_then(|x| x.as_i64()).is_some()
}

/// Weightless model-first probe (story 7406): can the `candle-llama` provider serve the model at
/// `spec.source`?
///
/// For a `*.gguf` file this reads **only** the GGUF header/metadata (never a tensor block, via
/// [`gguf_architecture`](crate::gguf::gguf_architecture)) and accepts it iff its `general.architecture`
/// is one the native GGUF loader can reconstruct ([`gguf_arch_to_hf`](crate::gguf::gguf_arch_to_hf):
/// `llama`/`mistral`, `qwen3`, and `gemma4`); an unsupported or non-LLM GGUF (`bert`, a `clip`
/// mmproj, …) is declined so `load_for_model` returns a clean `Unsupported` rather than routing it
/// here to fail at load (story 7420, replacing the earlier extension-only accept).
///
/// Otherwise this reads **only** `config.json` and runs the same [`Architecture::from_config`] dispatch
/// the loader uses (Llama / Mistral / Qwen2 / Qwen3 / Qwen2-MoE / Gemma2 / Gemma 4 / GLM-4 /
/// DeepSeek-V2 / Phi-3) — it never opens a safetensors shard, so `core-llm`'s `load_for_model` can
/// resolve a provider by model without loading weights. A multimodal snapshot (a `vision_config`
/// block — including a VLM whose `model_type` substring-matches a text family, e.g. `mllama`) is
/// declined so the vision provider claims it instead, **except** the families this provider serves
/// end-to-end: Qwen3.6, Qwen3-VL, and Gemma 4 unified.
pub fn can_load(spec: &LoadSpec) -> bool {
    if crate::gguf::is_gguf_path(&spec.source) {
        let prism = crate::prism_checkpoint::PrismGgufCheckpoint::is_prism(Path::new(&spec.source))
            .unwrap_or(false);
        if prism {
            return spec.projector_source.as_deref().is_none_or(|path| {
                crate::prism_vision_gguf::PrismVisionGguf::is_qwen3vl_merger(Path::new(path))
            });
        }
        if spec.projector_source.is_some() {
            return false;
        }
        // Confirm the GGUF's architecture from its header alone (weightless) — accept iff the loader
        // can actually reconstruct it. A `.gguf` that is missing/corrupt or names an unsupported arch
        // resolves to `None` and is declined.
        return crate::gguf::gguf_architecture(&spec.source)
            .as_deref()
            .and_then(crate::gguf::gguf_arch_to_hf)
            .is_some();
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
    if v.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
        return spec.projector_source.is_none();
    }
    // A multimodal snapshot (a `vision_config` block) is normally declined so a vision provider claims
    // it — EXCEPT the families this provider serves directly: Qwen3.6 (`qwen3_5`) and Qwen3-VL
    // (`qwen3_vl`), whose checkpoints are VLM-wrapped but whose matching ViT tower loads here, and
    // Gemma 4 unified, whose encoder-free vision embedder and audio projector likewise load here
    // (sc-18772). Gemma 4 is the reason this is not a Qwen-only list: `google/gemma-4-12B-it` carries
    // a `vision_config`, so the old rule declined the general-purpose Gemma 4 LLM outright.
    let arch = Architecture::from_config(&v);
    let serves_multimodal = matches!(arch, Ok(Architecture::Qwen35) | Ok(Architecture::Qwen3Vl))
        || matches!(arch, Ok(a) if a.is_gemma4());
    if v.get("vision_config").is_some() && !serves_multimodal {
        return false;
    }
    arch.is_ok()
}
