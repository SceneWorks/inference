//! The `core-llm` provider: a generic Llama model exposed through the backend-neutral contract.
//!
//! This is the mlx-llm half of story 7154 — it implements [`core_llm::TextLlm`] by wrapping the
//! [`CausalLm`] decoder, a [`core_llm::Tokenizer`], and a chat template, driving the internal
//! streaming decode loop and translating its token events into contract [`StreamEvent`]s (with
//! incremental detokenization). It registers into [`core_llm::registry`] under the id `mlx-llama`.
//!
//! The chat template is the model's own `chat_template` from `tokenizer_config.json` (rendered via
//! `core_llm::JinjaChatTemplate`, story 7164), falling back to the typed [`Llama3Template`] when a
//! snapshot ships no `tokenizer_config.json`.

use std::cell::{OnceCell, RefCell};
use std::path::Path;
use std::time::Instant;

use core_llm::{
    AudioRef, Channel, ChatTemplate, Constraint, ConstraintDecodeTable, ConstraintKind, Content,
    Error as CoreError, FinishReason as CoreFinish, ImageRef, IncrementalDetok, JinjaChatTemplate,
    JsonConstraint, Llama3Template, LlmMemoryGeometry, LoadSpec, Message, ModelSamplingDefaults,
    ProposerCapabilities, ProposerKind, Quantize, ReasoningEffort, RenderOptions,
    Result as CoreResult, Role, Sampling, SpeculativePlan, SpeculativeProposer, StopMatcher,
    StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput,
    TextLlmRequest, ThinkingMode, ThinkingSegmenter, Tokenizer, ToolCall, ToolCallSegmenter, Usage,
    VideoRef,
};

use core_llm::DraftReport;

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_batch, generate_speculative, generate_with_observer, prefill_restored, BatchRequest,
    Boundary, CancelFlag, ConstraintMask, Decode, DraftModelProposer, DynCacheTarget,
    EngineOptions, FinishReason, GenerationConfig, GenerationOutput, MtpProposer, NgramProposer,
    NoProposer, PrefixCache, PrefixPrefill, PrefixStats, Proposer, Qwen35MtpMultimodalPrompt,
    RewindableConstraintMask, SpeculativePrompt, SpeculativeRun, SpeculativeTarget, StreamEvent,
};
use crate::error::to_core;
use crate::image::{Qwen35ImageProcessor, Qwen35Preprocess};
use crate::models::gemma4_mm;
use crate::models::{
    CausalLm, Gemma4Layout, Gemma4Mm, Gemma4MmConfig, Qwen35Cache, Qwen35Config, Qwen35Model,
    Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::attention::{prefill_attention_tile_bytes, SDPA_MAX_FUSED_QLEN};
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{dtype_bytes, input_ids, Weights};
use crate::prism::PrismMlxPack;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

/// The registry id of this provider.
pub const PROVIDER_ID: &str = "mlx-llama";

enum CapturedCacheLifecycle {
    Allocation(&'static str, &'static str, &'static str, u64),
    Snapshot(u64, u64, u64, u64),
    Release(&'static str, &'static str, u64),
    PackedEvidence(Box<crate::primitives::PackedCacheEvidence>),
    DenseFallback(String, String),
    DenseReconstruction(u64),
    CompressedStorage(crate::primitives::CompressedCacheStorage),
}

#[derive(Default)]
struct CacheLifecycleCapture {
    events: Vec<CapturedCacheLifecycle>,
}

impl CacheLifecycleCapture {
    fn replay(self, observer: &mut dyn crate::campaign::Observer) {
        for event in self.events {
            match event {
                CapturedCacheLifecycle::Allocation(kind, role, lifetime, bytes) => {
                    observer.allocation_event(kind, role, lifetime, bytes);
                }
                CapturedCacheLifecycle::Snapshot(bytes, tokens, capacity, element_bytes) => {
                    observer.cache_snapshot(bytes, tokens, capacity, element_bytes);
                }
                CapturedCacheLifecycle::Release(kind, role, bytes) => {
                    observer.release_event(kind, role, bytes);
                }
                CapturedCacheLifecycle::PackedEvidence(evidence) => {
                    observer.packed_cache_evidence(&evidence);
                }
                CapturedCacheLifecycle::DenseFallback(operation, reason) => {
                    observer.dense_fallback(&operation, &reason);
                }
                CapturedCacheLifecycle::DenseReconstruction(bytes) => {
                    observer.dense_reconstruction(bytes);
                }
                CapturedCacheLifecycle::CompressedStorage(storage) => {
                    observer.compressed_storage(&storage);
                }
            }
        }
    }
}

impl crate::campaign::Observer for CacheLifecycleCapture {
    fn phase(&mut self, _name: &'static str) {}

    fn allocation(&mut self, role: &'static str, lifetime: &'static str, bytes: u64) {
        self.events.push(CapturedCacheLifecycle::Allocation(
            role, role, lifetime, bytes,
        ));
    }

    fn allocation_event(
        &mut self,
        kind: &'static str,
        role: &'static str,
        lifetime: &'static str,
        bytes: u64,
    ) {
        self.events.push(CapturedCacheLifecycle::Allocation(
            kind, role, lifetime, bytes,
        ));
    }

    fn cache_snapshot(&mut self, bytes: u64, tokens: u64, capacity: u64, element_bytes: u64) {
        self.events.push(CapturedCacheLifecycle::Snapshot(
            bytes,
            tokens,
            capacity,
            element_bytes,
        ));
    }

    fn release_event(&mut self, kind: &'static str, role: &'static str, bytes: u64) {
        self.events
            .push(CapturedCacheLifecycle::Release(kind, role, bytes));
    }

    fn packed_cache_evidence(&mut self, evidence: &crate::primitives::PackedCacheEvidence) {
        self.events
            .push(CapturedCacheLifecycle::PackedEvidence(Box::new(
                evidence.clone(),
            )));
    }

    fn dense_fallback(&mut self, operation: &str, reason: &str) {
        self.events.push(CapturedCacheLifecycle::DenseFallback(
            operation.into(),
            reason.into(),
        ));
    }

    fn dense_reconstruction(&mut self, bytes: u64) {
        self.events
            .push(CapturedCacheLifecycle::DenseReconstruction(bytes));
    }

    fn compressed_storage(&mut self, storage: &crate::primitives::CompressedCacheStorage) {
        self.events
            .push(CapturedCacheLifecycle::CompressedStorage(*storage));
    }
}

/// Campaign-only compressed decoder (SC-20676 compressed rows): every cache it makes is the
/// session's compressed representation bound to its retained fused reader. A selection the model
/// refuses before mutation stays dense and is recorded as an explicit, reasoned fallback rather
/// than silently replacing the fused path.
struct PackedCampaignDecoder<'a> {
    model: &'a CausalLm,
    arm: &'a crate::campaign::CompressedKvArm,
    selection_fallbacks: RefCell<Vec<String>>,
}

impl Decode for PackedCampaignDecoder<'_> {
    fn make_cache(&self) -> Box<dyn KvCache> {
        let selection = self.arm.select_cache(self.model);
        // Every refused selection keeps its own reason, including a packed cache whose reader
        // binding failed: that cache's later update fallback would otherwise hide why.
        if let crate::primitives::CacheRoute::DenseFallback { reason } = selection.route() {
            self.selection_fallbacks.borrow_mut().push(reason.clone());
        }
        selection.into_cache()
    }

    fn step(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Array> {
        self.model.step(input_ids, cache, offset)
    }
}

/// [`LlamaProvider::campaign_steady_decode`] over an already-tokenized context. A compressed arm's
/// measurement must run wholly on its fused compressed reader (selected cache, no fallback, no
/// dense reconstruction): a timing that silently decoded dense would carry a compressed label.
fn campaign_steady_decode_on(
    model: &Decoder,
    prompt_ids: &[i32],
    tokens: usize,
    stop_tokens: &[i32],
    compressed: Option<&crate::campaign::CompressedKvArm>,
) -> CoreResult<crate::campaign::SteadyDecodeMeasurement> {
    let measured = campaign_forced_decode_on(
        model,
        prompt_ids,
        tokens,
        stop_tokens,
        compressed,
        None,
        false,
    )?;
    Ok(crate::campaign::SteadyDecodeMeasurement {
        prompt_tokens: prompt_ids.len() as u64,
        generated_tokens: measured.tokens.len() as u64,
        timed_tokens: measured.timed_tokens,
        decode_ms: measured.decode_ms,
        forced_stop_tokens: measured.forced_stop_tokens,
    })
}

/// One fixed-length greedy decode through every stop token on a fresh cache of the session's
/// representation, optionally teacher-forced on `teacher_forced` and scored on its stream with
/// `score` (see [`crate::decode::forced_greedy_decode`]). A compressed arm must run wholly on its
/// fused compressed reader or the call fails closed.
fn campaign_forced_decode_on(
    model: &Decoder,
    prompt_ids: &[i32],
    tokens: usize,
    stop_tokens: &[i32],
    compressed: Option<&crate::campaign::CompressedKvArm>,
    teacher_forced: Option<&[i32]>,
    score: bool,
) -> CoreResult<crate::decode::ForcedDecode> {
    let packed = match (compressed, model) {
        (None, _) => None,
        (Some(arm), Decoder::Causal(causal)) => Some(PackedCampaignDecoder {
            model: causal,
            arm,
            selection_fallbacks: RefCell::new(Vec::new()),
        }),
        (Some(_), Decoder::Qwen35(_)) => {
            return Err(CoreError::Unsupported(
                "compressed campaign steady decode requires the causal decoder".into(),
            ))
        }
    };
    let decoder: &dyn Decode = match &packed {
        Some(packed) => packed,
        None => model,
    };
    let mut cache = decoder.make_cache();
    let measured = crate::decode::forced_greedy_decode(
        decoder,
        cache.as_mut(),
        prompt_ids,
        tokens,
        stop_tokens,
        teacher_forced,
        score,
        &mut |_| {},
    );
    let evidence = cache.packed_evidence();
    cache.reset().map_err(to_core)?;
    let measured = measured.map_err(to_core)?;
    if let Some(packed) = &packed {
        let fused = packed.selection_fallbacks.borrow().is_empty()
            && evidence.is_some_and(|evidence| {
                evidence.accepted_direct_calls > 0
                    && evidence.fallback_reasons.is_empty()
                    && !evidence.dense_active
                    && evidence.full_cache_dequantizations == 0
                    && evidence.failed_dispatches == 0
            });
        if !fused {
            return Err(CoreError::Load(
                "compressed forced decode did not run wholly on the fused compressed reader".into(),
            ));
        }
    }
    Ok(measured)
}

/// SC-20669 dense noise-floor control: the dense model teacher-forced on `stream` and scored, as
/// [`campaign_forced_decode_on`] does it, except that all but the last prompt token are prefilled
/// in `prefill_chunk`-token steps. Every value is exact bf16 dense attention over the same K/V; only
/// the computation order differs (projection row tiling, attention over the cached prefix instead
/// of in-step K/V). Its agreement with — and likelihood of — the one-shot dense continuation is the
/// dense arm's own numerical floor for the frozen greedy and perplexity thresholds.
fn campaign_chunked_prefill_decode_on(
    model: &Decoder,
    prompt_ids: &[i32],
    stream: &[i32],
    stop_tokens: &[i32],
    prefill_chunk: usize,
) -> CoreResult<crate::decode::ForcedDecode> {
    if prefill_chunk == 0 || prompt_ids.len() < 2 {
        return Err(CoreError::InvalidRequest(
            "chunked prefill needs a positive chunk and a prompt of two or more tokens".into(),
        ));
    }
    let mut cache = chunked_prefill_cache(model, prompt_ids.len() + stream.len())?;
    let prefilled = prompt_ids.len() - 1;
    let measured = (|| -> crate::error::Result<crate::decode::ForcedDecode> {
        chunked_prefill(model, &mut cache, &prompt_ids[..prefilled], prefill_chunk)?;
        crate::decode::forced_greedy_decode_from(
            model,
            &mut cache,
            prompt_ids,
            prefilled,
            stream.len(),
            stop_tokens,
            Some(stream),
            true,
            &mut |_| {},
        )
    })();
    cache.reset().map_err(to_core)?;
    measured.map_err(to_core)
}

/// The chunked control's dense cache: ONE allocation covering the prompt and the forced stream.
/// The decoder's default cache grows by concatenation as chunks arrive, so every chunk of a long
/// prompt retired a whole-history buffer into MLX's freed-buffer cache (run 37021783368: the
/// 130k-token llama row's 64 growths reached a 73.8 GB footprint against a 68 GiB cap). Sized
/// once, the chunked control holds what a one-shot dense run holds: one dense K/V history.
fn chunked_prefill_cache(model: &Decoder, positions: usize) -> CoreResult<ContiguousKvCache> {
    let block = i32::try_from(positions.max(1))
        .map_err(|_| CoreError::InvalidRequest("chunked prefill length overflows i32".into()))?;
    match model {
        Decoder::Causal(model) => Ok(ContiguousKvCache::with_block_tokens(
            model.config().num_layers,
            block,
        )),
        Decoder::Qwen35(_) => Err(CoreError::Unsupported(
            "the chunked-prefill noise-floor control needs the causal decoder".into(),
        )),
    }
}

/// Prefill `prompt_ids` into `cache` in `prefill_chunk`-token steps, evaluating each step and
/// releasing MLX's freed-buffer cache after it, so no chunk's transients outlive it.
fn chunked_prefill(
    model: &Decoder,
    cache: &mut dyn KvCache,
    prompt_ids: &[i32],
    prefill_chunk: usize,
) -> crate::error::Result<()> {
    for start in (0..prompt_ids.len()).step_by(prefill_chunk.max(1)) {
        let end = (start + prefill_chunk).min(prompt_ids.len());
        let offset = cache.offset();
        let logits = model.step(
            &crate::primitives::nn::input_ids(&prompt_ids[start..end]),
            cache,
            offset,
        )?;
        logits.eval()?;
        drop(logits);
        mlx_rs::memory::clear_cache();
    }
    Ok(())
}

/// The cache record of one prompt-cache turn: whether the lookup since `before` hit, and how many
/// prompt tokens it reused.
fn prompt_cache_turn(
    before: crate::decode::PrefixStats,
    store: &crate::decode::PrefixCache,
    prompt_ids: &[i32],
) -> CoreResult<crate::campaign::PromptCacheTurn> {
    let after = store.stats();
    if after.lookups != before.lookups + 1 {
        return Err(CoreError::Load(
            "multi-turn prompt cache: a turn must perform exactly one prompt-cache lookup".into(),
        ));
    }
    let cache_hit = after.hits > before.hits;
    let reused = after.reused_prefix_tokens - before.reused_prefix_tokens;
    Ok(crate::campaign::PromptCacheTurn {
        prompt_tokens: prompt_ids.len() as u64,
        prompt_sha256: crate::campaign::token_stream_sha256(prompt_ids),
        cache_hit,
        reused_prefix_tokens: reused as u64,
    })
}

/// The loaded decoder, dispatched by architecture. The generic softmax-attention decoders share
/// [`CausalLm`]; Qwen3.6 (`qwen3_5`) is the hybrid linear-attention/full-attention decoder. Both
/// implement [`Decode`], so the generation loop is identical.
enum Decoder {
    Causal(Box<CausalLm>),
    Qwen35(Box<Qwen35Model>),
}

/// The identity of a snapshot's weight files for [`CausalLm::cache_fingerprint`] (sc-20688
/// review): each `*.safetensors` file's name, size and modification time, in name order, so
/// weights replaced at the same path never share a stored compressed prefix with the old ones.
fn snapshot_weights_identity(dir: &Path) -> String {
    let mut files = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .map(|entry| {
            let metadata = entry.metadata().ok();
            let modified = metadata
                .as_ref()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |since| since.as_nanos());
            format!(
                "{}:{}:{modified}",
                entry.file_name().to_string_lossy(),
                metadata.map_or(0, |metadata| metadata.len())
            )
        })
        .collect::<Vec<_>>();
    files.sort();
    files.join(";")
}

/// The KV cache one production generation runs on (sc-20679), decided before any K/V mutation.
enum KvPlan {
    /// The SC-20671 campaign observer path: it selects its own explicit arm and is not a product
    /// generation, so it reports no [`core_llm::KvCacheReport`].
    Unreported,
    /// Dense, with the report (and its reason) the output carries.
    Dense(core_llm::KvCacheReport),
    /// The qualified compressed format and the provider's retained fused reader.
    Compressed {
        format: core_llm::KvCompressionFormat,
        reader: crate::primitives::CompiledKernelHandle,
    },
}

impl Decode for Decoder {
    fn make_cache(&self) -> Box<dyn KvCache> {
        match self {
            Decoder::Causal(m) => m.make_cache(),
            Decoder::Qwen35(m) => m.make_cache(),
        }
    }

    fn step(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Array> {
        match self {
            Decoder::Causal(m) => m.step(input_ids, cache, offset),
            Decoder::Qwen35(m) => m.step(input_ids, cache, offset),
        }
    }
}

impl Decoder {
    fn memory_geometry(&self) -> LlmMemoryGeometry {
        let (query_heads, kv_heads, head_dim, layers, hidden, intermediate, vocab, recurrent) =
            match self {
                Decoder::Causal(m) => {
                    let c = m.config();
                    (
                        c.num_heads,
                        c.num_kv_heads,
                        c.head_dim,
                        c.num_layers,
                        c.hidden_size,
                        c.intermediate_size,
                        c.vocab_size,
                        0,
                    )
                }
                Decoder::Qwen35(m) => {
                    let c = m.config();
                    (
                        c.num_heads,
                        c.num_kv_heads,
                        c.head_dim,
                        c.num_layers,
                        c.hidden_size,
                        c.intermediate_size,
                        c.vocab_size,
                        (c.num_layers as u64)
                            .saturating_mul(c.linear_num_value_heads as u64)
                            .saturating_mul(c.linear_value_head_dim as u64)
                            .saturating_mul(
                                (c.linear_key_head_dim + c.linear_conv_kernel_dim) as u64,
                            )
                            .saturating_mul(4),
                    )
                }
            };
        let (compute, prism) = match self {
            Decoder::Causal(m) => (m.compute_dtype(), false),
            Decoder::Qwen35(m) => (m.compute_dtype(), m.is_prism()),
        };
        LlmMemoryGeometry {
            query_heads: query_heads.max(0) as u64,
            kv_heads: kv_heads.max(0) as u64,
            head_dim: head_dim.max(0) as u64,
            layers: layers as u64,
            element_bytes: priced_compute_element_bytes(compute, prism),
            score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
            hidden_size: hidden as u64,
            intermediate_size: intermediate as u64,
            vocab_size: vocab as u64,
            recurrent_bytes: recurrent,
        }
    }

    /// Bytes a forward's promoted weight copies hold at once ([`promoted_weight_bytes`]), when
    /// the decoder's activations are wider than its weights ([`CausalLm::activations_promote`]);
    /// `Some(0)` otherwise.
    fn promoted_request_bytes(&self) -> Option<u64> {
        match self {
            Decoder::Causal(m) => match m.promoted_weight_elements() {
                Some((head, largest)) => promoted_weight_bytes(head, largest),
                None => Some(0),
            },
            Decoder::Qwen35(_) => Some(0),
        }
    }

    /// Select the request-memory contract matching the complete decoder implementation.
    fn workspace_contract(&self) -> MlxWorkspaceContract<'_> {
        match self {
            Decoder::Qwen35(m) if m.config().moe.is_none() => MlxWorkspaceContract::Qwen35 {
                config: m.config(),
                prism: m.is_prism(),
            },
            // The MoE expert gather/scatter lifetime differs from the dense path below and has no
            // complete-model allocator calibration yet. Retain the safe eager estimate rather than
            // applying a dense-Qwen envelope to a different execution graph.
            Decoder::Qwen35(_) => MlxWorkspaceContract::Eager,
            Decoder::Causal(m) => {
                let c = m.config();
                if c.attn_logit_softcap.is_none() && !c.is_mla() && c.gemma4.is_none() {
                    MlxWorkspaceContract::Chunked
                } else {
                    MlxWorkspaceContract::Eager
                }
            }
        }
    }

    fn is_quantized(&self) -> bool {
        match self {
            Decoder::Causal(m) => m.is_quantized(),
            Decoder::Qwen35(m) => m.is_quantized(),
        }
    }

    /// The decoder as the backend-neutral multimodal seam. Both backbones implement [`VlmDecode`]
    /// (the Qwen3.6 hybrid and the generic Qwen3-VL causal decoder), so the provider drives the
    /// image/video prefill + decode through this one trait object rather than forking on the concrete
    /// decoder type.
    fn as_vlm(&self) -> &dyn VlmDecode {
        match self {
            Decoder::Causal(m) => m.as_ref(),
            Decoder::Qwen35(m) => m.as_ref(),
        }
    }
}

/// The Qwen-VL vision side of the provider: the ViT tower, the image preprocessor, and the
/// multimodal token ids needed to expand placeholders and assign M-RoPE positions. Present when the
/// loaded `qwen3_5` (Qwen3.6) or `qwen3_vl` (Qwen3-VL) checkpoint carries `model.visual.*`. The two
/// share the identical Qwen3-VL ViT tower (`Qwen3VLVisionModel == Qwen35VisionModel`); only the
/// decoder prefill differs ([`Decoder::Qwen35`] vs [`Decoder::Causal`]).
struct Qwen35Vision {
    tower: Qwen35VisionModel,
    processor: Qwen35ImageProcessor,
    image_token_id: i32,
    /// The `<|video_pad|>` placeholder token id (151656 for Qwen3-VL) — the per-frame video
    /// placeholder the processor expands to `frame_seqlen` copies.
    video_token_id: i32,
    spatial_merge_size: i32,
}

impl Qwen35Vision {
    fn estimate_workspace(
        &self,
        images: &[&ImageRef],
        videos: &[&VideoRef],
    ) -> CoreResult<(usize, u64)> {
        let cfg = self.tower.config();
        let patch = cfg.patch_size.max(1) as usize;
        let merge = self.spatial_merge_size.max(1) as usize;
        let temporal = self.processor.temporal_patch_size.max(1);
        let mut tokens = 0usize;
        let mut pixels = 0u64;
        for image in images {
            let (h, w) = self
                .processor
                .smart_resize(image.height as usize, image.width as usize)
                .map_err(|e| to_core(e.into()))?;
            tokens = tokens
                .checked_add((h / patch) * (w / patch) / (merge * merge))
                .ok_or_else(|| {
                    CoreError::InvalidRequest("visual token geometry overflow".into())
                })?;
            pixels = pixels.saturating_add((h as u64).saturating_mul(w as u64).saturating_mul(12));
        }
        for video in videos {
            let frame = video
                .frames
                .first()
                .ok_or_else(|| CoreError::InvalidRequest("video has no frames".into()))?;
            let t = video.frames.len().div_ceil(temporal);
            let (h, w) = self
                .processor
                .smart_resize_with(
                    frame.height as usize,
                    frame.width as usize,
                    t,
                    self.processor.video_min_pixels,
                    self.processor.video_max_pixels,
                )
                .map_err(|e| to_core(e.into()))?;
            tokens = tokens
                .checked_add(t * (h / patch) * (w / patch) / (merge * merge))
                .ok_or_else(|| {
                    CoreError::InvalidRequest("visual token geometry overflow".into())
                })?;
            pixels = pixels.saturating_add(
                (t as u64)
                    .saturating_mul(temporal as u64)
                    .saturating_mul(h as u64)
                    .saturating_mul(w as u64)
                    .saturating_mul(12),
            );
        }
        let activations = (tokens as u64)
            .checked_mul(cfg.hidden_size.max(0) as u64)
            .and_then(|v| {
                v.checked_mul((cfg.depth + cfg.deepstack_visual_indexes.len() + 2) as u64)
            })
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| CoreError::InvalidRequest("visual workspace overflow".into()))?;
        // The tower attends over unmerged patches, not the merged language tokens.
        // Pricing all frames together is conservative even when attention is frame-local.
        let patches = (tokens as u64)
            .checked_mul(merge as u64)
            .and_then(|v| v.checked_mul(merge as u64))
            .ok_or_else(|| CoreError::InvalidRequest("visual patch geometry overflow".into()))?;
        let attention = patches
            .checked_mul(patches)
            .and_then(|v| v.checked_mul(cfg.num_heads.max(0) as u64))
            .and_then(|v| v.checked_mul(12))
            .ok_or_else(|| {
                CoreError::InvalidRequest("visual attention workspace overflow".into())
            })?;
        let mlp = patches
            .checked_mul(cfg.intermediate_size.max(0) as u64)
            .and_then(|v| v.checked_mul(12))
            .ok_or_else(|| CoreError::InvalidRequest("visual MLP workspace overflow".into()))?;
        let required = pixels
            .checked_add(activations)
            .and_then(|v| v.checked_add(attention))
            .and_then(|v| v.checked_add(mlp))
            .ok_or_else(|| CoreError::InvalidRequest("visual workspace overflow".into()))?;
        Ok((tokens, required))
    }

    /// Encode one image to its merged patch rows `[n_tokens, hidden]` (the merger output is already
    /// the language hidden size — no separate projector), the per-tap **DeepStack** feature sets
    /// (each `[n_tokens, hidden]`, one per `deepstack_visual_indexes` tap), plus the image's
    /// `grid_thw` (`[1, h, w]` in patch units). `n_tokens = (grid_h/merge)·(grid_w/merge)` is the
    /// placeholder expansion count.
    fn encode(&self, img: &ImageRef) -> CoreResult<(Array, Vec<Array>, [i32; 3])> {
        let (pixels, grid) = self
            .processor
            .preprocess(&img.pixels, img.width as usize, img.height as usize)
            .map_err(to_core)?;
        let out = self
            .tower
            .forward_with_deepstack(&pixels, &grid)
            .map_err(to_core)?;
        Ok((out.pooler_output, out.deepstack_features, grid[0]))
    }

    /// Encode one **video** (sampled frames) to its merged patch rows `[grid_t·n_per_frame, hidden]`,
    /// the per-tap DeepStack features, and the `video_grid_thw` (`[grid_t, h, w]`). The ViT tower is
    /// modality-agnostic — it processes the `grid_t` temporal patches as a block-diagonal-masked
    /// frame sequence exactly like multiple images — so this reuses `forward_with_deepstack`. The
    /// per-frame timestamp tokens are rendered separately (Text–Timestamp Alignment); here we only
    /// produce the visual features and the grid.
    fn encode_video(&self, video: &VideoRef) -> CoreResult<(Array, Vec<Array>, [i32; 3])> {
        let frames: Vec<(&[u8], usize, usize)> = video
            .frames
            .iter()
            .map(|f| (f.pixels.as_slice(), f.width as usize, f.height as usize))
            .collect();
        let (pixels, grid) = self.processor.preprocess_video(&frames).map_err(to_core)?;
        let out = self
            .tower
            .forward_with_deepstack(&pixels, &[grid])
            .map_err(to_core)?;
        Ok((out.pooler_output, out.deepstack_features, grid))
    }
}

/// The prepared multimodal prefill: the image-token-expanded prompt ids, the decoder input embeds
/// with image features spliced in, the interleaved M-RoPE position rows + delta, the per-position
/// visual mask (`true` at image-token positions), and the per-tap DeepStack feature sets fused into
/// the first decoder layers.
struct MultimodalPrefill {
    expanded_ids: Vec<i32>,
    embeds: Array,
    positions: (Vec<i32>, Vec<i32>, Vec<i32>, i32),
    visual_pos_mask: Vec<bool>,
    deepstack: Vec<Array>,
}

/// Gemma 4's prepared multimodal prefill: the marker-expanded prompt ids and the decoder input
/// embeds with vision / audio feature rows spliced onto the soft-token positions.
///
/// Deliberately *not* [`MultimodalPrefill`]: Gemma 4 has no M-RoPE, no positional compression, and
/// no DeepStack taps, so there is no `mrope_delta` to shift the continuation by and no per-position
/// visual mask to build. Reusing the Qwen-VL struct would mean carrying four fields that mean
/// nothing here.
struct Gemma4Prefill {
    expanded_ids: Vec<i32>,
    embeds: Array,
}

/// Whether Gemma 4's vision path has been validated end-to-end, and may therefore be advertised.
///
/// **`false`** (sc-18772). The front-end is implemented and its pieces are unit-tested, but its
/// end-to-end behaviour could not be demonstrated on real weights, and two of its steps had to be
/// chosen without a reference implementation to check against (upstream `transformers` with
/// `Gemma4Unified` is not available offline):
///
/// * the patch-grid tiling — how an arbitrary aspect ratio maps onto a patch grid within the
///   280-soft-token budget, and
/// * the resampling of the 1120-entry positional table onto that grid.
///
/// Measured on `google/gemma-4-12B-it` at Q4 on CPU, through this provider:
///
/// * a text-only prompt through the SAME provider instance answers cleanly ("The sun dipped below
///   the horizon, bleeding hues of molten gold ...");
/// * audio conditioning answers cleanly and discriminates content (a 440 Hz tone is described as
///   "a low-frequency, rhythmic thumping or pulsing", silence as "I cannot hear anything");
/// * an image-conditioned prompt degenerates into a `thought` loop and, where it emits anything
///   else, describes the input as **audio**. The span itself is correct — 261 prompt tokens for a
///   17-token question is exactly `boi + 11x22 soft tokens + eoi` — so the framing and expansion
///   are right and the *features* are wrong. Both plausible patch flattening orders (row-major and
///   the channel-major Conv2d equivalent) were tried on real weights and fail the same way.
///
/// Advertising vision on that evidence is exactly the advertised-but-absent failure this story
/// forbids, so an image-carrying request is REJECTED by the capability gate rather than answered
/// from its text. The embedder and its host-side geometry stay in the tree, unadvertised and
/// unreachable, so the next attempt starts from them and from the real-weight leg that must pass
/// before this returns `true` (`gemma4_answers_about_an_image` in the breadth suite).
fn gemma4_vision_is_validated() -> bool {
    false
}

/// Gemma 4's loaded multimodal front-ends.
struct Gemma4Runtime {
    mm: Gemma4Mm,
}

/// Collect the image blocks of a conversation, in order.
fn collect_images(messages: &[Message]) -> Vec<&ImageRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Image(img) => Some(img),
                Content::Text(_) | Content::Video(_) | Content::Audio(_) => None,
            })
        })
        .collect()
}

/// Collect the video blocks of a conversation, in order.
fn collect_videos(messages: &[Message]) -> Vec<&VideoRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Video(v) => Some(v),
                Content::Text(_) | Content::Image(_) | Content::Audio(_) => None,
            })
        })
        .collect()
}

/// Collect the audio blocks of a conversation, in order.
fn collect_audio(messages: &[Message]) -> Vec<&AudioRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Audio(a) => Some(a),
                Content::Text(_) | Content::Image(_) | Content::Video(_) => None,
            })
        })
        .collect()
}

fn request_media_workspace_bytes(messages: &[Message]) -> CoreResult<u64> {
    messages.iter().try_fold(0u64, |total, message| {
        message.content.iter().try_fold(total, |total, content| {
            let bytes = match content {
                Content::Image(image) => (image.pixels.len() as u64).saturating_mul(4),
                Content::Video(video) => video.frames.iter().fold(0u64, |sum, frame| {
                    sum.saturating_add((frame.pixels.len() as u64).saturating_mul(4))
                }),
                Content::Audio(audio) => (audio.samples.len() as u64).saturating_mul(16),
                Content::Text(_) => 0,
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| CoreError::InvalidRequest("media workspace overflow".into()))
        })
    })
}

/// Gemma 4's rendered image marker — the single token its chat template emits for an image part,
/// which the processor (here, [`LlamaProvider::prepare_gemma4`]) expands into
/// `boi` + N soft tokens + `eoi`.
const GEMMA4_IMAGE_MARKER: &str = "<|image|>";
/// Gemma 4's rendered audio marker, expanded to `boa` + M soft tokens + `eoa`.
const GEMMA4_AUDIO_MARKER: &str = "<|audio|>";

/// Replace each image/audio block with Gemma 4's marker text so the (content-free) core-llm chat
/// template contract renders exactly what the model's own Jinja template renders for a typed image /
/// audio part. Video is refused rather than silently rendered as an image: Gemma 4 declares a
/// `video_token_id`, but this provider ships no frame-sampling path for it, and quietly dropping the
/// frames would answer a question about a video from its text alone.
fn substitute_gemma4_placeholders(messages: &[Message]) -> CoreResult<Vec<Message>> {
    messages
        .iter()
        .map(|m| {
            let content = m
                .content
                .iter()
                .map(|c| match c {
                    Content::Image(_) => Ok(Content::text(GEMMA4_IMAGE_MARKER)),
                    Content::Audio(_) => Ok(Content::text(GEMMA4_AUDIO_MARKER)),
                    Content::Text(t) => Ok(Content::Text(t.clone())),
                    Content::Video(_) => Err(CoreError::Unsupported(
                        "[mlx-llama] Gemma 4: video input is not supported by this provider (no \
                         frame-sampling path); send frames as individual images"
                            .to_string(),
                    )),
                })
                .collect::<CoreResult<Vec<_>>>()?;
            Ok(Message {
                role: m.role,
                content,
                thinking: m.thinking.clone(),
                tool_calls: m.tool_calls.clone(),
            })
        })
        .collect()
}

/// Compute the Qwen3-VL **merged per-frame timestamps** for a sampled video, mirroring
/// `Qwen3VLProcessor._calculate_timestamps`. The vision encoder folds `temporal_patch_size` frames
/// into one temporal patch, so the per-sample timestamps are padded up to a multiple of
/// `temporal_patch_size` (repeating the last) and then **averaged within each temporal patch**,
/// yielding one timestamp per emitted vision frame (`grid_t = padded / temporal_patch_size`). These
/// are the `<{t:.1f} seconds>` values the Text–Timestamp-Alignment placeholder uses.
fn merged_frame_timestamps(timestamps: &[f32], temporal_patch_size: usize) -> Vec<f32> {
    let tps = temporal_patch_size.max(1);
    let mut ts: Vec<f32> = timestamps.to_vec();
    if ts.is_empty() {
        return ts;
    }
    while !ts.len().is_multiple_of(tps) {
        ts.push(*ts.last().unwrap());
    }
    (0..ts.len())
        .step_by(tps)
        .map(|i| (ts[i] + ts[i + tps - 1]) / 2.0)
        .collect()
}

/// The Text–Timestamp-Alignment placeholder text for one video: per merged frame, a
/// `<{t:.1f} seconds>` timestamp tag followed by `<|vision_start|><|video_pad|><|vision_end|>`
/// (exactly `Qwen3VLProcessor.replace_video_token`). The single `<|video_pad|>` per frame is expanded
/// to `frame_seqlen` copies after tokenizing.
fn video_placeholder_text(video: &VideoRef, temporal_patch_size: usize) -> String {
    let merged = merged_frame_timestamps(&video.timestamps, temporal_patch_size);
    let mut out = String::new();
    for t in merged {
        out.push_str(&format!("<{t:.1} seconds>"));
        out.push_str("<|vision_start|><|video_pad|><|vision_end|>");
    }
    out
}

/// Replace each image/video block with its Qwen-VL placeholder text so the (text-only) chat template
/// renders the vision framing verbatim. Images become `<|vision_start|><|image_pad|><|vision_end|>`
/// (one `image_pad`, expanded to the per-image patch count after tokenizing); videos become the
/// per-frame Text–Timestamp-Alignment string `<{t} seconds><|vision_start|><|video_pad|><|vision_end|>`
/// (one `video_pad` per frame, each expanded to `frame_seqlen` after tokenizing). Keeps the core-llm
/// template contract image/video-free.
fn substitute_vision_placeholders(
    messages: &[Message],
    temporal_patch_size: usize,
) -> CoreResult<Vec<Message>> {
    const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    messages
        .iter()
        .map(|m| {
            let content = m
                .content
                .iter()
                .map(|c| -> CoreResult<Content> {
                    match c {
                        Content::Image(_) => Ok(Content::text(IMAGE_PLACEHOLDER)),
                        Content::Video(v) => {
                            v.validate().map_err(CoreError::InvalidRequest)?;
                            Ok(Content::text(video_placeholder_text(v, temporal_patch_size)))
                        }
                        Content::Text(t) => Ok(Content::Text(t.clone())),
                        // The Qwen-VL path has no audio projector, and this provider's
                        // `supports_audio` is false for every Qwen checkpoint, so `validate`
                        // rejects an audio-carrying request before substitution. Erroring here
                        // rather than dropping the block means that if that invariant ever breaks,
                        // the request fails loudly instead of being answered from its text alone.
                        Content::Audio(_) => Err(CoreError::Unsupported(
                            "[mlx-llama] the Qwen-VL path carries no audio; an audio block reached \
                             placeholder substitution, which the capability gate should have rejected"
                                .to_string(),
                        )),
                    }
                })
                .collect::<CoreResult<Vec<_>>>()?;
            Ok(Message {
                role: m.role,
                content,
                thinking: m.thinking.clone(),
                tool_calls: m.tool_calls.clone(),
            })
        })
        .collect()
}

/// A generic Llama provider implementing [`core_llm::TextLlm`].
pub struct LlamaProvider {
    descriptor: TextLlmDescriptor,
    /// Architecture parsed from the loaded snapshot. Campaign receipts use this product-owned
    /// identity instead of trusting the matrix row's caller-authored family label.
    architecture: Architecture,
    campaign_family: Option<&'static str>,
    /// The compressed-KV qualification family (sc-20679). Only a snapshot load names it, and only
    /// for a checkpoint whose config is a measured row's exact architecture
    /// ([`core_llm::qualified_kv_model_family`], sc-20688 review); [`Self::from_parts`] cannot
    /// tell a Llama checkpoint from a Mistral or dense Qwen2 one sharing its decoder, so it has
    /// none.
    kv_family: Option<core_llm::KvModelFamily>,
    model: Decoder,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    tool_call_format: Option<ToolCallFormat>,
    stop_tokens: Vec<i32>,
    /// Cached per-vocab decode table for constrained decoding — built once (it decodes the whole
    /// vocabulary) on the first JSON-constrained request, then reused.
    constraint_table: OnceCell<ConstraintDecodeTable>,
    /// The Qwen3.6 vision tower + preprocessor, present iff this is a `qwen3_5` checkpoint carrying
    /// `model.visual.*`. Drives the image path in [`LlamaProvider::generate`].
    vision: Option<Qwen35Vision>,
    /// Gemma 4's encoder-free vision embedder and/or audio projector, present iff this is a Gemma 4
    /// checkpoint that actually ships them (sc-18772). Independent of [`vision`](Self::vision):
    /// Gemma 4 does not use the Qwen-VL ViT/M-RoPE/DeepStack machinery at all.
    gemma4: Option<Gemma4Runtime>,
    /// Campaign-only prefix cache; ordinary serving never consults this state.
    campaign_prefix_cache: RefCell<Option<crate::decode::PrefixCache>>,
    /// At most two OWNED stream handles: load CPU and actual last execution.
    /// Fencing these orders execution; it does not evict model/cache ownership.
    execution_fence: core_llm::text_llm::OwnedExecutionFence<mlx_rs::Stream>,
    /// The chat template's `date_string` kwarg. `None` (every production load) keeps the
    /// template's own dating — `strftime_now`, the current date, on Llama 3.2. Only
    /// [`Self::load_for_campaign`] pins it ([`crate::campaign::CAMPAIGN_CHAT_TEMPLATE_DATE_STRING`])
    /// so a campaign's rendered prompts never depend on the day the row runs.
    campaign_template_date: Option<&'static str>,
    /// The retained fused compressed-KV reader (sc-20679), built on the first request the
    /// qualification table admits and reused by every later one; `Err` is the build failure each
    /// such request then reports as [`core_llm::KvCacheFallbackReason::ReaderUnavailable`].
    kv_reader: OnceCell<Result<crate::primitives::CompiledKernelHandle, String>>,
    /// The loaded checkpoint's name for compressed-cache keying (its load source; empty for a
    /// provider assembled from parts, which then never shares compressed pages).
    kv_model_name: String,
    /// Dense Prism `vision_tower.*` tensors retained verbatim for the native multimodal adapter.
    /// Text loading must not discard them merely because sc-23937 constructs only the decoder.
    _prism_vision_weights: Option<Weights>,
    /// The cross-turn prefix cache (sc-24437): one per loaded model, every decoder family, under
    /// the byte budget [`load`](Self::load) admitted.
    prefix: RefCell<PrefixCache>,
    /// The resident draft model for `draft_model` speculation (sc-24436), present iff the load
    /// named a compatible one ([`LoadSpec::draft_source`]); `draft_model` is advertised exactly
    /// then. A second decoder with its own cache per request, driven as a proposer by the engine.
    draft: Option<ResidentDraft>,
    /// What the load settled: the requested weight format, what became of a named draft
    /// (resident, or refused with the reason), the prefix cache's admitted budget, and every other
    /// optional accelerator — the companion MTP head (sc-24444) — requested but not attached,
    /// named in `fallbacks`. Read through [`TextLlm::load_report`].
    load_report: core_llm::LoadReport,
    /// The speculative option a request that leaves it unset runs with (E5, sc-24446): the MLX
    /// defaults-table row ([`core_llm::defaults::MLX`]) unless
    /// [`set_speculative_default`](LlamaProvider::set_speculative_default) overrides it.
    speculative_default: core_llm::Speculative,
}

// A failing load unwinds partial arrays before this earlier-declared guard.
// Fence both retained streams on that failure, without replacing its primary error.
struct LoadFailureFence<F: FnOnce() -> CoreResult<()>> {
    fence: Option<F>,
}
impl<F: FnOnce() -> CoreResult<()>> LoadFailureFence<F> {
    fn new(fence: F) -> Self {
        Self { fence: Some(fence) }
    }
    fn disarm(&mut self) {
        self.fence = None;
    }
}
impl<F: FnOnce() -> CoreResult<()>> Drop for LoadFailureFence<F> {
    fn drop(&mut self) {
        if let Some(fence) = self.fence.take() {
            if let Err(error) = fence() {
                eprintln!("mlx-llama partial-load synchronization failed; original load error preserved, no reclamation claimed: {error}");
            }
        }
    }
}

/// A draft model resident beside its target (sc-24436).
struct ResidentDraft {
    /// The draft decoder.
    model: Decoder,
    /// Leading draft ids it may propose — its tokenizer's tokens
    /// ([`core_llm::draft_compatibility`]); a padding row never becomes a draft.
    proposable: usize,
    /// The target's logits width, which the draft's logits are shaped to.
    width: usize,
    /// The draft's own context window (`0`: unbounded): a request reaching past it runs `auto`
    /// instead ([`core_llm::fit_draft_context`], E2).
    context: usize,
}

/// What a load does with its named draft model (sc-24436), decided before any weight is read.
enum DraftPlan {
    /// No draft named.
    None,
    /// Admitted beside the target: load it from this spec.
    Load(LoadSpec),
    /// Refused before loading (unpriceable, or no room beside the target).
    Refused(DraftReport),
}

impl DraftPlan {
    /// Price a named draft beside the target (E7): its load estimate is admitted together with
    /// the target's. `Err` only when the target alone does not fit — a draft never fails the load
    /// (E2); a draft that cannot be priced or does not fit beside the target is refused by name.
    fn admit(spec: &LoadSpec, target_required: u64, available: u64) -> CoreResult<Self> {
        let Some(source) = spec.draft_source.as_deref() else {
            core_llm::admit_load_memory(target_required, available)?;
            return Ok(DraftPlan::None);
        };
        let draft_spec = draft_load_spec(spec, source);
        let draft_required = match crate::load_memory::required_bytes(&draft_spec) {
            Ok(bytes) => bytes,
            Err(e) => {
                core_llm::admit_load_memory(target_required, available)?;
                return Ok(DraftPlan::Refused(DraftReport::refused(
                    source,
                    core_llm::draft_unpriced_refusal(e),
                )));
            }
        };
        Ok(
            match core_llm::admit_draft_load(target_required, draft_required, available)? {
                None => DraftPlan::Load(draft_spec),
                Some(why) => DraftPlan::Refused(DraftReport::refused(source, why)),
            },
        )
    }
}

/// The load spec of a named draft: its own source, text-only, at the target's load-time tier
/// (a quantization tier covers every model the load makes resident) and naming no draft itself.
fn draft_load_spec(spec: &LoadSpec, source: &str) -> LoadSpec {
    LoadSpec {
        source: source.to_string(),
        projector_source: None,
        quantize: spec.quantize,
        cuda_graphs: spec.cuda_graphs,
        // A draft keeps no cross-turn prefix cache of its own (sc-24437), and no companion head.
        prefix_cache_bytes: Some(0),
        draft_source: None,
        mtp_head_source: None,
    }
}

/// The same Qwen3.5/3.8 decoder appears under `model.language_model` in VLM snapshots and
/// directly under `model` in text-only finetunes. Reject ambiguous or incomplete layouts.
fn qwen35_dense_prefix(has_key: impl Fn(&str) -> bool) -> CoreResult<&'static str> {
    match (
        has_key("model.language_model.embed_tokens.weight"),
        has_key("model.embed_tokens.weight"),
    ) {
        (true, false) => Ok("model.language_model"),
        (false, true) => Ok("model"),
        (false, false) => Err(CoreError::Load(
            "qwen3_5 checkpoint has no wrapped or flat decoder embeddings".into(),
        )),
        (true, true) => Err(CoreError::Load(
            "qwen3_5 checkpoint has both wrapped and flat decoder embeddings".into(),
        )),
    }
}

/// The MLX projection format of a load-time quantization tier. NVFP4 is a CUDA sm_120
/// capability (sc-24135): refused by name, never substituted.
pub(crate) fn quant_spec(quantize: Quantize) -> CoreResult<QuantSpec> {
    match quantize {
        Quantize::Q4 => Ok(QuantSpec::q4()),
        Quantize::Q8 => Ok(QuantSpec::q8()),
        Quantize::Nvfp4 => Err(CoreError::Unsupported(
            "nvfp4: NVFP4 projections need a CUDA device with compute capability >= sm_120; the \
             MLX backend has no NVFP4 GEMM"
                .into(),
        )),
    }
}

/// Materialize a lazily built decoder group by group ([`Weights::materialize_groups`]) and
/// refuse the load if any source it read went unconsumed (sc-24446): such a source is held
/// resident beside the model, outside what `load_memory::required_bytes` prices — an
/// enumeration gap in the decoder's `param_groups`, never a property of the checkpoint.
fn materialize_decoder(weights: &mut Weights, groups: &[Vec<Array>]) -> CoreResult<()> {
    let report = weights.materialize_groups(groups).map_err(to_core)?;
    if report.leftover > 0 {
        return Err(CoreError::Load(format!(
            "load materialization: {} source tensor(s) the decoder read were consumed by none of \
             its arrays; the load would hold them outside its admitted bound",
            report.leftover
        )));
    }
    Ok(())
}

impl LlamaProvider {
    fn new_execution_fence() -> core_llm::text_llm::OwnedExecutionFence<mlx_rs::Stream> {
        core_llm::text_llm::OwnedExecutionFence::new(
            mlx_rs::Stream::cpu(),
            mlx_rs::Stream::task_local_or_default(),
        )
    }

    fn note_execution_stream(&self) -> crate::Result<()> {
        self.execution_fence.before_execution(
            mlx_rs::Stream::task_local_or_default(),
            crate::primitives::synchronize_stream,
        )
    }

    /// Frozen SC-20671 family identity for architectures that implement the campaign's contiguous
    /// cache controls. Other architectures fail closed instead of being mislabeled as Llama/Qwen.
    pub(crate) fn campaign_family(&self) -> CoreResult<&'static str> {
        self.campaign_family.ok_or_else(|| {
            CoreError::Unsupported(format!(
                "SC-20671 does not support loaded architecture {:?} as Llama or Qwen",
                self.architecture
            ))
        })
    }

    pub(crate) fn campaign_context_window(&self) -> CoreResult<u64> {
        let Decoder::Causal(model) = &self.model else {
            return Err(CoreError::Unsupported(
                "SC-20671 requires a causal decoder context window".into(),
            ));
        };
        u64::try_from(model.config().max_position_embeddings)
            .ok()
            .filter(|tokens| *tokens >= 1_024)
            .ok_or_else(|| {
                CoreError::Load(
                    "SC-20671 requires max_position_embeddings >= 1024 in loaded config".into(),
                )
            })
    }

    pub(crate) fn campaign_prompt_tokens(&self, prompt: &str) -> CoreResult<u64> {
        u64::try_from(self.tokenizer.encode(prompt, false)?.len())
            .map_err(|_| CoreError::Load("campaign prompt token count overflows u64".into()))
    }

    /// The `context_band` payload of at most `target` tokens (the coordinate's
    /// [`crate::campaign::family_band_target`]), which must fit the loaded window.
    pub(crate) fn campaign_context_band_measurement(
        &self,
        context_band: &str,
        target: u64,
    ) -> CoreResult<(String, u64, u64)> {
        if target > self.campaign_context_window()? {
            return Err(CoreError::Load(format!(
                "SC-20671 {context_band} target {target} exceeds the loaded window"
            )));
        }
        self.campaign_band_payload(context_band, target)
    }

    /// The multi-turn prompt-cache fixture's payload (contract v4): the band payload capped by
    /// [`crate::campaign::multi_turn_payload_target`], so turn 2 plus the full turn-2 forced
    /// continuation fits the native window. The row's coordinate operations keep the full band.
    pub(crate) fn campaign_multi_turn_payload(
        &self,
        context_band: &str,
        band: u64,
    ) -> CoreResult<(String, u64, u64)> {
        let context_window = self.campaign_context_window()?;
        let target = crate::campaign::multi_turn_payload_target(context_window, band)
            .map_err(CoreError::Load)?;
        self.campaign_band_payload(context_band, target)
    }

    /// The largest `context_band` payload of at most `target` tokens.
    fn campaign_band_payload(
        &self,
        context_band: &str,
        target: u64,
    ) -> CoreResult<(String, u64, u64)> {
        let header = format!("SC20671-CONTEXT-BAND-{context_band}");
        let mut lower = 0usize;
        let mut upper = usize::try_from(target)
            .map_err(|_| CoreError::Load("campaign context target overflows usize".into()))?;
        while lower < upper {
            let midpoint = lower + (upper - lower).div_ceil(2);
            let candidate = format!("{header}{}", " context".repeat(midpoint));
            if self.campaign_prompt_tokens(&candidate)? <= target {
                lower = midpoint;
            } else {
                upper = midpoint - 1;
            }
        }
        let payload = format!("{header}{}", " context".repeat(lower));
        let observed_tokens = self.campaign_prompt_tokens(&payload)?;
        if observed_tokens < target / 2 || observed_tokens > target {
            return Err(CoreError::Load(format!(
                "context band {context_band} produced {observed_tokens} tokens for target {target}"
            )));
        }
        Ok((payload, target, observed_tokens))
    }

    /// Loaded decoder geometry for the receipt producer.  This is crate-private so a campaign
    /// cannot substitute JSON-provided head/layer values for the actual provider configuration.
    /// SC-20677 K/V capture: the loaded causal decoder the campaign decodes through, and the
    /// snapshot's own tokenizer. Only campaign families (Llama/Qwen3) are accepted.
    pub(crate) fn campaign_causal_decoder(&self) -> CoreResult<(&CausalLm, &Tokenizer)> {
        self.campaign_family()?;
        match &self.model {
            Decoder::Causal(model) => Ok((model, &self.tokenizer)),
            Decoder::Qwen35(_) => Err(CoreError::Unsupported(
                "SC-20677 K/V capture requires the causal campaign decoder".into(),
            )),
        }
    }

    pub(crate) fn campaign_geometry(&self) -> crate::campaign::ProductGeometry {
        let (query_heads, kv_heads, head_dimension, layers) = match &self.model {
            Decoder::Causal(model) => {
                let config = model.config();
                (
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.num_layers,
                )
            }
            Decoder::Qwen35(model) => {
                let config = model.config();
                (
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.num_layers,
                )
            }
        };
        crate::campaign::ProductGeometry {
            query_heads: query_heads.max(0) as u64,
            kv_heads: kv_heads.max(0) as u64,
            head_dimension: head_dimension.max(0) as u64,
            layers: layers as u64,
            // The product observer fills this from the first retained MLX key/value arrays.
            element_bytes: 0,
        }
    }

    /// Exercise real contiguous-cache prefix reuse on the loaded causal decoder.  Hybrid Qwen3.6
    /// has a distinct cache contract and is rejected here rather than being mislabeled as a
    /// successful contiguous-cache observation.
    pub(crate) fn campaign_prefix_reuse(&self, prompt: &str) -> CoreResult<u64> {
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        if ids.len() < 2 {
            return Err(CoreError::InvalidRequest(
                "campaign prefix-reuse prompt needs at least two tokens".into(),
            ));
        }
        let Decoder::Causal(model) = &self.model else {
            return Err(CoreError::Unsupported(
                "campaign prefix reuse is not implemented for the hybrid Qwen3.6 cache".into(),
            ));
        };
        let config = GenerationConfig {
            max_new_tokens: 1,
            seed: Some(0),
            ..Default::default()
        };
        let cancel = crate::decode::CancelFlag::new();
        let mut cache_slot = self.campaign_prefix_cache.borrow_mut();
        let cache =
            cache_slot.get_or_insert_with(|| crate::decode::PrefixCache::with_budget(u64::MAX));
        let mut sink = |_| {};
        crate::decode::generate_cached(model, &ids, &config, &cancel, &mut sink, cache)
            .map_err(to_core)?;
        crate::decode::generate_cached(model, &ids, &config, &cancel, &mut sink, cache)
            .map_err(to_core)?;
        if cache.stats().hits == 0 {
            return Err(CoreError::Load(
                "campaign prefix reuse did not produce a cache hit".into(),
            ));
        }
        u64::try_from(cache.stats().hits)
            .map_err(|_| CoreError::Load("campaign prefix hit count overflow".into()))
    }

    /// Seed the provider-owned prefix cache before a campaign opens its phase-local MLX peak
    /// window.  The observed cache-hit dispatch is deliberately separate so seed allocations and
    /// compile work cannot establish the peak used as measured prefill evidence.
    pub(crate) fn campaign_seed_prefix_reuse(&self, prompt: &str) -> CoreResult<f64> {
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        if ids.len() < 2 {
            return Err(CoreError::InvalidRequest(
                "campaign prefix-reuse prompt needs at least two tokens".into(),
            ));
        }
        let Decoder::Causal(model) = &self.model else {
            return Err(CoreError::Unsupported(
                "campaign prefix reuse is not implemented for the hybrid Qwen3.6 cache".into(),
            ));
        };
        let config = GenerationConfig {
            max_new_tokens: 1,
            seed: Some(0),
            ..Default::default()
        };
        let cancel = crate::decode::CancelFlag::new();
        let mut cache_slot = self.campaign_prefix_cache.borrow_mut();
        if cache_slot.is_some() {
            return Err(CoreError::InvalidRequest(
                "campaign prefix cache was already seeded".into(),
            ));
        }
        let mut cache = crate::decode::PrefixCache::with_budget(u64::MAX);
        let mut sink = |_| {};
        let dispatch_started = std::time::Instant::now();
        crate::decode::generate_cached(model, &ids, &config, &cancel, &mut sink, &mut cache)
            .map_err(to_core)?;
        let dispatch_elapsed_ms = dispatch_started.elapsed().as_secs_f64() * 1_000.0;
        if !dispatch_elapsed_ms.is_finite() || dispatch_elapsed_ms <= 0.0 {
            return Err(CoreError::Load(
                "campaign prefix seed produced no positive dispatch duration".into(),
            ));
        }
        *cache_slot = Some(cache);
        Ok(dispatch_elapsed_ms)
    }

    /// Execute only the cache-hit half of the real prefix-reuse path with campaign observation
    /// attached. The caller must seed before resetting the phase-local peak; this method proves a
    /// new hit and emitted token before its output can be used as coordinate evidence. A
    /// `compressed` arm imports the reused prefix into its compressed cache.
    pub(crate) fn campaign_prefix_reuse_observed(
        &self,
        prompt: &str,
        observer: &mut dyn crate::campaign::Observer,
        compressed: Option<&crate::campaign::CompressedKvArm>,
    ) -> CoreResult<(GenerationOutput, u64, u64)> {
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        if ids.len() < 2 {
            return Err(CoreError::InvalidRequest(
                "campaign prefix-reuse prompt needs at least two tokens".into(),
            ));
        }
        let Decoder::Causal(model) = &self.model else {
            return Err(CoreError::Unsupported(
                "campaign prefix reuse is not implemented for the hybrid Qwen3.6 cache".into(),
            ));
        };
        let config = GenerationConfig {
            max_new_tokens: 1,
            seed: Some(0),
            ..Default::default()
        };
        let cancel = crate::decode::CancelFlag::new();
        let mut cache_slot = self.campaign_prefix_cache.borrow_mut();
        let cache = cache_slot.as_mut().ok_or_else(|| {
            CoreError::InvalidRequest("campaign prefix cache was not seeded".into())
        })?;
        let before_hits = cache.stats().hits;
        let mut emitted = 0usize;
        let output = crate::decode::prefix::generate_cached_with_observer(
            model,
            &ids,
            &config,
            &cancel,
            &mut |event| emitted += usize::from(matches!(event, StreamEvent::Token { .. })),
            cache,
            None,
            None,
            Some(observer),
            compressed,
        )
        .map_err(to_core)?;
        let hits = cache.stats().hits;
        if hits <= before_hits || emitted == 0 {
            return Err(CoreError::Load(
                "campaign prefix reuse did not produce an observed cache hit and token".into(),
            ));
        }
        Ok((
            output,
            u64::try_from(hits)
                .map_err(|_| CoreError::Load("campaign prefix hit count overflow".into()))?,
            u64::try_from(ids.len())
                .map_err(|_| CoreError::Load("campaign prefix token count overflow".into()))?,
        ))
    }

    /// Drop campaign-only shared-prefix ownership before a post-request release sample. Ordinary
    /// serving has no access to this cache; campaign workers must not let it retain MLX arrays and
    /// then claim that request-scoped cache memory was released.
    /// The stop tokens every product generation of this provider ends on.
    pub(crate) fn campaign_stop_tokens(&self) -> &[i32] {
        &self.stop_tokens
    }

    pub(crate) fn campaign_release_cache_state(&self) {
        self.campaign_prefix_cache.borrow_mut().take();
    }

    /// Exercise the actual synchronous MLX batch decoder for the baseline's supported-batch arm.
    /// This is not emulated by serial `TextLlm` requests.
    pub(crate) fn campaign_supported_batch(&self, prompt: &str, batch: usize) -> CoreResult<u64> {
        if batch < 2 {
            return Err(CoreError::InvalidRequest(
                "campaign supported batch requires at least two rows".into(),
            ));
        }
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        let Decoder::Causal(model) = &self.model else {
            return Err(CoreError::Unsupported(
                "campaign supported batch is unavailable for the hybrid Qwen3.6 decoder".into(),
            ));
        };
        let requests = (0..batch)
            .map(|lane| BatchRequest {
                prompt_ids: ids.clone(),
                sampling: SamplingParams::default(),
                seed: Some(lane as u64),
                max_new_tokens: 2,
                stop_tokens: self.stop_tokens.clone(),
            })
            .collect::<Vec<_>>();
        let cancel = CancelFlag::new();
        let mut emitted = 0usize;
        let outputs = generate_batch(model, &requests, &cancel, &mut |_, event| {
            emitted += usize::from(matches!(event, StreamEvent::Token { .. }));
        })
        .map_err(to_core)?;
        if outputs.len() != batch || emitted == 0 {
            return Err(CoreError::InvalidRequest(
                "campaign batch produced no product tokens".into(),
            ));
        }
        Ok(outputs.len() as u64)
    }

    /// Run the actual batched decoder while the receipt observer is attached to its prefill and
    /// decode path.  Ordinary scheduling continues to call [`Self::campaign_supported_batch`].
    pub(crate) fn campaign_supported_batch_observed(
        &self,
        prompt: &str,
        batch: usize,
        observer: &mut dyn crate::campaign::Observer,
    ) -> CoreResult<(Vec<GenerationOutput>, u64)> {
        if batch < 2 {
            return Err(CoreError::InvalidRequest(
                "campaign supported batch requires at least two rows".into(),
            ));
        }
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        let Decoder::Causal(model) = &self.model else {
            return Err(CoreError::Unsupported(
                "campaign supported batch is unavailable for the hybrid Qwen3.6 decoder".into(),
            ));
        };
        let requests = (0..batch)
            .map(|lane| BatchRequest {
                prompt_ids: ids.clone(),
                sampling: SamplingParams::default(),
                seed: Some(lane as u64),
                max_new_tokens: 2,
                stop_tokens: self.stop_tokens.clone(),
            })
            .collect::<Vec<_>>();
        let cancel = CancelFlag::new();
        let mut emitted = 0usize;
        let outputs = crate::decode::batch::generate_batch_with_observer(
            model,
            &requests,
            &cancel,
            &mut |_, event| emitted += usize::from(matches!(event, StreamEvent::Token { .. })),
            Some(observer),
        )
        .map_err(to_core)?;
        if outputs.len() != batch || emitted == 0 {
            return Err(CoreError::InvalidRequest(
                "campaign batch produced no observed product tokens".into(),
            ));
        }
        Ok((
            outputs,
            u64::try_from(ids.len())
                .map_err(|_| CoreError::Load("campaign batch token count overflow".into()))?,
        ))
    }

    /// Deliberately cancel after the first emitted product token, proving the decoder's cooperative
    /// cleanup path rather than recording a pre-cancelled no-op request. This uses a dedicated
    /// no-tools prompt: a valid tool-only response is intentionally lifted out of the content
    /// stream, so reusing a structured-output fixture would never trigger a content-token cancel.
    pub(crate) fn campaign_cancel_after_first_token(
        &self,
        observer: &mut dyn crate::campaign::Observer,
        packed: Option<&crate::campaign::CompressedKvArm>,
    ) -> CoreResult<()> {
        let mut request = campaign_cancellation_probe_request();
        let cancel = crate::decode::CancelFlag::new();
        request.cancel = cancel.clone();
        let mut sink = |event: CoreEvent| {
            if matches!(event, CoreEvent::Token { .. }) {
                cancel.cancel();
            }
        };
        let mut captured = CacheLifecycleCapture::default();
        let output = self.generate_inner(&request, &mut sink, Some(&mut captured), packed, None)?;
        if output.finish_reason != Some(CoreFinish::Cancelled) {
            return Err(CoreError::Load(
                "campaign cancellation did not finish as cancelled".into(),
            ));
        }
        observer.phase("cancellation-cleanup");
        captured.replay(observer);
        Ok(())
    }

    /// SC-20671 steady-decode timing: prefill the raw `prompt` (the row's context) into a fresh
    /// cache of the session's representation and greedily decode exactly `tokens` ids through any
    /// stop token (see [`crate::decode::forced_greedy_decode`]). No observer is attached: this runs
    /// outside every coordinate's memory attribution, and its request-scoped cache is reset and
    /// MLX's buffer cache released before it returns.
    pub(crate) fn campaign_steady_decode(
        &self,
        prompt: &str,
        tokens: usize,
        compressed: Option<&crate::campaign::CompressedKvArm>,
    ) -> CoreResult<crate::campaign::SteadyDecodeMeasurement> {
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        let budget = u32::try_from(tokens)
            .map_err(|_| CoreError::InvalidRequest("steady-decode length overflows".into()))?;
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            ids.len(),
            budget,
        )?;
        let measured =
            campaign_steady_decode_on(&self.model, &ids, tokens, &self.stop_tokens, compressed);
        mlx_rs::memory::clear_cache();
        measured
    }

    /// SC-20671 forced continuation (compressed rows): on the session's representation, prefill the
    /// raw `prompt` and greedily decode `min(tokens, context window - prompt)` ids through every
    /// stop token. With `teacher_forced`, the decode is forced on that stream instead and the
    /// session's argmax at each position is returned (its length is the stream's). Either way the
    /// session's probability of every stream token is returned beside its choices, so both arms'
    /// likelihoods are measured on the same tokens. Runs outside every coordinate's memory
    /// attribution on a request-scoped cache.
    pub(crate) fn campaign_forced_continuation(
        &self,
        prompt: &str,
        tokens: usize,
        compressed: Option<&crate::campaign::CompressedKvArm>,
        teacher_forced: Option<&[i32]>,
    ) -> CoreResult<crate::campaign::ScoredContinuation> {
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        let window = usize::try_from(self.campaign_context_window()?)
            .map_err(|_| CoreError::Load("context window overflows usize".into()))?;
        let length = match teacher_forced {
            Some(forced) => forced.len(),
            None => tokens.min(window.saturating_sub(ids.len())),
        };
        let budget = u32::try_from(length)
            .map_err(|_| CoreError::InvalidRequest("forced continuation overflows".into()))?;
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            ids.len(),
            budget,
        )?;
        let measured = campaign_forced_decode_on(
            &self.model,
            &ids,
            length,
            &self.stop_tokens,
            compressed,
            teacher_forced,
            true,
        );
        mlx_rs::memory::clear_cache();
        let measured = measured?;
        Ok(crate::campaign::ScoredContinuation {
            choices: measured.tokens,
            stream_probabilities: measured.stream_probabilities,
        })
    }

    /// SC-20669 dense noise-floor control (see [`campaign_chunked_prefill_decode_on`]): the dense
    /// session teacher-forced on `stream` over the raw `prompt` prefilled in `prefill_chunk`-token
    /// steps, scored on that stream. Refused on a compressed session's provider path by
    /// construction: it always decodes the dense cache.
    pub(crate) fn campaign_chunked_prefill_continuation(
        &self,
        prompt: &str,
        stream: &[i32],
        prefill_chunk: usize,
    ) -> CoreResult<crate::campaign::ScoredContinuation> {
        let ids = self
            .tokenizer
            .encode(prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        let budget = u32::try_from(stream.len())
            .map_err(|_| CoreError::InvalidRequest("forced continuation overflows".into()))?;
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            ids.len(),
            budget,
        )?;
        let measured = campaign_chunked_prefill_decode_on(
            &self.model,
            &ids,
            stream,
            &self.stop_tokens,
            prefill_chunk,
        );
        mlx_rs::memory::clear_cache();
        let measured = measured?;
        Ok(crate::campaign::ScoredContinuation {
            choices: measured.tokens,
            stream_probabilities: measured.stream_probabilities,
        })
    }

    /// Load a provider from a snapshot directory (config.json + tokenizer.json + shards). Dispatches
    /// the decoder architecture from `config.json` (Llama / Mistral / Qwen3) and optionally
    /// quantizes the projections on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        Self::load_inner(spec, false)
    }

    /// Campaign-only load which evaluates every tensor consumed by the product model constructor.
    /// Ordinary serving keeps MLX's lazy-load behavior; the sealed memory campaign needs an exact
    /// parameter-only materialization boundary before it samples `weights-loaded`.
    pub(crate) fn load_for_campaign(spec: &LoadSpec) -> CoreResult<Self> {
        Self::load_inner(spec, true).map(Self::pin_campaign_template_date)
    }

    /// Fix the chat template's `date_string` to the campaign constant, so every campaign render
    /// (turn 1, turn 2, fixtures) is the same prompt on any day.
    fn pin_campaign_template_date(mut self) -> Self {
        self.campaign_template_date = Some(crate::campaign::CAMPAIGN_CHAT_TEMPLATE_DATE_STRING);
        self
    }

    fn load_inner(spec: &LoadSpec, materialize_campaign_weights: bool) -> CoreResult<Self> {
        if spec.projector_source.is_some()
            && Path::new(&spec.source).extension().and_then(|v| v.to_str()) != Some("gguf")
        {
            return Err(CoreError::Unsupported(
                "[mlx-llama] projector_source is only valid for a separable Prism GGUF model; \
                 safetensors snapshots carry their vision tower in the snapshot"
                    .into(),
            ));
        }
        // sc-24135: NVFP4 is refused by name before admission, so a memory refusal cannot mask it.
        let quant = spec.quantize.map(quant_spec).transpose()?;
        let required = crate::load_memory::required_bytes(spec)?;
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            load_memory_budget()?,
        )?;
        let draft = DraftPlan::admit(spec, required, available)?;
        let mut admitted = match &draft {
            DraftPlan::Load(draft_spec) => {
                required.saturating_add(crate::load_memory::required_bytes(draft_spec)?)
            }
            DraftPlan::None | DraftPlan::Refused(_) => required,
        };
        // E7: a companion head is resident weights; it is admitted on top of the target and any
        // admitted draft, and a head that does not fit is refused by name while the target still
        // loads (E2).
        let mut fallbacks = Vec::new();
        let mtp_head = spec.mtp_head_source.as_deref().and_then(|source| {
            admit_companion_head(Path::new(source), admitted, available)
                .map_err(|reason| fallbacks.push(reason))
                .ok()
        });
        if let Some((_, head_bytes)) = mtp_head {
            admitted = admitted.saturating_add(head_bytes);
        }
        // The cross-turn prefix cache's budget (sc-24437, E7): what the load asked for, clamped to
        // the headroom this admission leaves beside the target, any admitted draft and any
        // admitted companion head, so the cache can never push the load past it.
        let prefix_budget = core_llm::prefix_cache_budget(
            core_llm::DecodeBackend::Mlx,
            spec.prefix_cache_bytes,
            admitted,
            available,
        );

        let load_streams = Self::new_execution_fence();
        let mut failed_load_fence = LoadFailureFence::new(move || {
            load_streams
                .synchronize(crate::primitives::synchronize_stream)
                .map_err(to_core)
        });
        let mut provider = Self::load_admitted(spec, quant, materialize_campaign_weights)?
            .with_prefix_budget(prefix_budget);
        provider.load_report.requested = spec.quantize;
        // The load's own fallbacks (a refused companion head) join what the decoder's load named
        // (a configured native MTP head the snapshot does not carry).
        provider.load_report.fallbacks.splice(0..0, fallbacks);
        provider.attach_mtp_head(mtp_head.map(|(head, _)| head));
        provider.load_report.record_prefix_budget(
            core_llm::DecodeBackend::Mlx,
            spec.prefix_cache_bytes,
            prefix_budget,
        );
        match draft {
            DraftPlan::None => {}
            DraftPlan::Refused(report) => provider.load_report.record_draft(report),
            DraftPlan::Load(draft_spec) => provider.attach_draft(&draft_spec),
        }
        failed_load_fence.disarm();
        Ok(provider)
    }

    /// Load the draft `spec` names beside this target and keep it only if it can propose for it
    /// ([`core_llm::draft_compatibility`]): resident — `draft_model` advertised — or refused
    /// with the reason in the load report, the target unaffected either way (sc-24436, E2).
    fn attach_draft(&mut self, spec: &LoadSpec) {
        let source = spec.source.clone();
        let target_logits = self.model.memory_geometry().vocab_size as usize;
        // The tokenizer is checked before any draft weight is touched when the draft ships one.
        let early = core_llm::draft_tokenizer_refusal(&self.tokenizer, Path::new(&source));
        let outcome = match early {
            Some(why) => Err(why),
            None => match spec.quantize.map(quant_spec).transpose() {
                Err(e) => Err(core_llm::draft_refusal(e)),
                Ok(quant) => match Self::load_admitted(spec, quant, false) {
                    Err(e) => Err(core_llm::draft_load_refusal(e)),
                    Ok(draft) => core_llm::draft_compatibility(
                        &self.tokenizer,
                        target_logits,
                        &draft.tokenizer,
                        draft.model.memory_geometry().vocab_size as usize,
                    )
                    .map(|proposable| ResidentDraft {
                        proposable,
                        width: target_logits,
                        context: draft.descriptor.capabilities.max_context_tokens,
                        model: draft.model,
                    }),
                },
            },
        };
        let (draft, report) = core_llm::settle_draft(
            source,
            outcome,
            &mut self.descriptor.capabilities,
            &mut self.load_report.fallbacks,
            recommended_depths(),
        );
        self.draft = draft;
        self.load_report.draft = Some(report);
    }

    /// [`load`](Self::load) after admission: the target decoder, its tokenizer, template and
    /// multimodal front-ends, with no draft. `materialize_campaign_weights`: evaluate every tensor
    /// the constructors consumed before returning (the campaign's parameter-only boundary).
    fn load_admitted(
        spec: &LoadSpec,
        quant: Option<QuantSpec>,
        materialize_campaign_weights: bool,
    ) -> CoreResult<Self> {
        let dir = Path::new(&spec.source);
        if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
            return Self::load_prism_gguf(spec, dir);
        }
        // Read config.json once to dispatch the architecture: the hybrid Qwen3.6 (`qwen3_5`) decoder
        // has its own config/weights path (and `ModelConfig` deliberately rejects it).
        let cfg_value = read_config_value(dir)?;
        let arch = Architecture::from_config(&cfg_value).map_err(|e| to_core(e.into()))?;
        let text_config = cfg_value.get("text_config").unwrap_or(&cfg_value);
        let architecture_name = text_config
            .get("architectures")
            .and_then(|value| value.as_array())
            .and_then(|values| values.first())
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let model_type = text_config
            .get("model_type")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        // The shared (sc-20683) family rule, so Candle names the same checkpoint the same family.
        let campaign_family =
            core_llm::kv_model_family(arch.family(), &architecture_name, &model_type)
                .map(crate::kv_policy::campaign_family);
        let mut weights = Weights::from_dir(dir).map_err(to_core)?;
        let is_prism =
            cfg_value.get("model_type").and_then(|v| v.as_str()) == Some("prism_hadamard_qwen35");
        if is_prism && quant.is_some() {
            return Err(CoreError::Load(
                "Prism snapshots are already packed 2-bit and reject load-time Q4/Q8".into(),
            ));
        }

        let mut prism_vision_weights = None;
        // What the decoder's own load could not attach (E2): a configured native MTP head the
        // snapshot does not carry, or a variant this runtime does not run.
        let mut load_fallbacks = Vec::new();
        let (model, mut descriptor) = if arch == Architecture::Qwen35 {
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let mut descriptor = descriptor_for_qwen35(&qcfg);
            let m = if is_prism {
                let pack = PrismMlxPack::from_dir(dir, &cfg_value, &weights).map_err(to_core)?;
                descriptor.family = "prism_hadamard_qwen35".into();
                let model =
                    Qwen35Model::build_prism_lazy(&weights, qcfg, &pack).map_err(to_core)?;
                materialize_decoder(&mut weights, &model.param_groups())?;
                if pack.has_vision_tower {
                    let keys = weights
                        .keys()
                        .filter(|key| key.starts_with("vision_tower."))
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    let retained = keys
                        .into_iter()
                        .filter_map(|key| weights.get(&key).cloned().map(|value| (key, value)))
                        .collect();
                    prism_vision_weights = Some(Weights::from_map(retained));
                }
                model
            } else {
                // The text decoder nests under `model.language_model` in the VLM-wrapped checkpoint.
                let prefix = qwen35_dense_prefix(|key| weights.contains(key))?;
                let model =
                    Qwen35Model::build_lazy(&weights, prefix, qcfg, quant).map_err(to_core)?;
                // Group by group: read, verify, convert, release (sc-24446) — the order
                // `load_memory::required_bytes` prices.
                materialize_decoder(&mut weights, &model.param_groups())?;
                model
            };
            // A configured head that was not built is named and not advertised: the loaded
            // config (its MTP layers cleared) settles the advertisement.
            if let Some(why) = m.mtp_fallback() {
                load_fallbacks.push(why.to_string());
                descriptor.capabilities = descriptor_for_qwen35(m.config()).capabilities;
            }
            (Decoder::Qwen35(Box::new(m)), descriptor)
        } else {
            let cfg = ModelConfig::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let m = CausalLm::build_lazy(&weights, "", cfg, quant).map_err(to_core)?;
            materialize_decoder(&mut weights, &m.param_groups())?;
            let m = m.with_weights_identity(snapshot_weights_identity(dir));
            (Decoder::Causal(Box::new(m)), descriptor)
        };

        // Qwen-VL vision: load the ViT tower when the checkpoint carries `model.visual.*` (a wrapped
        // VLM) and the config exposes a `vision_config`. Covers Qwen3.6 (`qwen3_5`) and Qwen3-VL
        // (`qwen3_vl`), which share the identical Qwen3-VL ViT tower. Absent → a text-only checkpoint.
        let vision_prefix = if weights.contains("model.visual.patch_embed.proj.weight") {
            Some("model.visual")
        } else if is_prism && weights.contains("vision_tower.patch_embed.proj.weight") {
            Some("vision_tower")
        } else {
            None
        };
        let vision = if let (true, Some(prefix)) = (
            (arch == Architecture::Qwen35 || arch == Architecture::Qwen3Vl)
                && cfg_value.get("vision_config").is_some(),
            vision_prefix,
        ) {
            let vcfg = Qwen35VisionConfig::from_json(&cfg_value).map_err(to_core)?;
            let tower = if is_prism {
                Qwen35VisionModel::from_mlx_weights(&weights, prefix, vcfg.clone())
            } else {
                Qwen35VisionModel::from_weights(&weights, prefix, vcfg.clone())
            }
            .map_err(to_core)?;
            let image_token_id = cfg_value
                .get("image_token_id")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .unwrap_or(248056);
            // Video tokens (Qwen3-VL): `video_token_id` (`<|video_pad|>`, 151656) plus the
            // vision_start/end framing the Text–Timestamp-Alignment substitution emits per frame. A
            // checkpoint without `video_token_id` in its config does not advertise video.
            let video_token_id = cfg_value
                .get("video_token_id")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32);
            descriptor.capabilities.supports_vision = true;
            descriptor.capabilities.supports_video = video_token_id.is_some();
            Some(Qwen35Vision {
                tower,
                processor: Qwen35ImageProcessor::default(),
                image_token_id,
                // Fall back to the canonical Qwen3-VL id when absent so the field is always valid;
                // `supports_video` already gates whether the video path is reachable.
                video_token_id: video_token_id.unwrap_or(151656),
                spatial_merge_size: vcfg.spatial_merge_size,
            })
        } else {
            None
        };

        // Gemma 4 multimodal (sc-18772): load whichever front-ends the checkpoint actually ships.
        // The capability flags are set from the LOADED tensors, never from the config alone — a
        // `vision_config` with no `model.vision_embedder.*` yields `None` and stays unadvertised,
        // which is the difference between declaring a capability and having one.
        let gemma4 = if arch.is_gemma4() {
            let layout = Gemma4Layout::detect(&weights).map_err(to_core)?;
            let processor = read_json(dir, "processor_config.json");
            let mm_cfg =
                Gemma4MmConfig::from_json(&cfg_value, processor.as_ref()).map_err(to_core)?;
            let mm = Gemma4Mm::from_weights(&weights, layout, mm_cfg).map_err(to_core)?;
            // Audio tracks the loaded tensors; vision does not — see `gemma4_vision_is_validated`.
            descriptor.capabilities.supports_vision =
                mm.vision.is_some() && gemma4_vision_is_validated();
            descriptor.capabilities.supports_audio = mm.audio.is_some();
            mm.audio.is_some().then_some(Gemma4Runtime { mm })
        } else {
            None
        };

        let tokenizer = crate::__model_input_reader!(
            Tokenizer::from_file(dir.join("tokenizer.json")),
            "crates/llm/mlx-llm/src/provider.rs::load_admitted::Tokenizer::from_file@87862",
            Tokenizer,
            dir.join("tokenizer.json"),
            |__model_input_path| Tokenizer::from_file(__model_input_path)
        )?;
        let stop_tokens = eos_token_ids(dir);
        let (
            template,
            supports_thinking,
            supports_reasoning_effort,
            supports_preserve_thinking,
            supports_tools,
            tool_call_format,
        ) = load_chat_template(dir);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_reasoning_effort = supports_reasoning_effort;
        if supports_reasoning_effort {
            descriptor.capabilities.reasoning_efforts = if is_prism {
                vec![ReasoningEffort::XHigh, ReasoningEffort::Medium]
            } else {
                vec![
                    ReasoningEffort::XHigh,
                    ReasoningEffort::Medium,
                    ReasoningEffort::Low,
                ]
            };
        }
        if is_prism {
            descriptor.capabilities.model_sampling_defaults = Some(bonsai_sampling_defaults());
        }
        descriptor.capabilities.supports_preserve_thinking = supports_preserve_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        if materialize_campaign_weights {
            weights.materialize_accessed().map_err(to_core)?;
        }
        Ok(Self {
            descriptor,
            architecture: arch,
            campaign_family,
            // sc-20688 review: only the measured architectures plan as a table family.
            kv_family: core_llm::qualified_kv_model_family(arch.family(), &cfg_value),
            model,
            tokenizer,
            template,
            tool_call_format,
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision,
            gemma4,
            campaign_prefix_cache: RefCell::new(None),
            execution_fence: LlamaProvider::new_execution_fence(),
            campaign_template_date: None,
            kv_reader: OnceCell::new(),
            kv_model_name: spec.source.clone(),
            _prism_vision_weights: prism_vision_weights,
            prefix: RefCell::new(PrefixCache::with_budget(0)),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport {
                fallbacks: load_fallbacks,
                ..core_llm::LoadReport::default()
            },
        })
    }

    /// Attach an admitted companion MTP head (sc-24444). It attaches only to a Qwen3.5/3.8-family
    /// target (dense `qwen3_5` or Prism/Bonsai) without a native head, and only when its geometry
    /// matches; anything else is a named load fallback and the target stays exactly as loaded.
    fn attach_mtp_head(&mut self, head: Option<&Path>) {
        let Some(head) = head else { return };
        let outcome = match &mut self.model {
            Decoder::Qwen35(model) => model
                .attach_companion_mtp(head)
                .map(|()| speculative_max_depth([qwen35_attention_geometry(model.config())]))
                .map_err(|e| format!("mtp_head: {e}")),
            Decoder::Causal(_) => Err(format!(
                "{}; this model's family is `{}`",
                core_llm::COMPANION_MTP_FAMILY_REFUSAL,
                self.descriptor.family
            )),
        };
        match outcome {
            // The same backend-true depth a native head advertises (sc-24438): a companion head
            // runs the identical one-layer predictor through the identical verify path.
            Ok(max_depth) => self
                .descriptor
                .capabilities
                .advertise_mtp(max_depth, recommended_depths().mtp),
            Err(reason) => self
                .load_report
                .fallbacks
                .push(core_llm::companion_head_fallback(&reason, head)),
        }
    }

    fn load_prism_gguf(spec: &LoadSpec, path: &Path) -> CoreResult<Self> {
        if spec.quantize.is_some() {
            return Err(CoreError::Load(
                "Prism GGUF is already packed and rejects load-time Q4/Q8".into(),
            ));
        }
        let file = crate::gguf::GgufFile::open(path).map_err(to_core)?;
        let loaded = crate::prism_gguf::load(&file).map_err(to_core)?;
        let qcfg = Qwen35Config::from_json(&loaded.config).map_err(to_core)?;
        let mut descriptor = descriptor_for_qwen35(&qcfg);
        descriptor.family = "prism_hadamard_qwen35".into();
        let (thinking, reasoning, preserve, tools) = loaded.template_capabilities;
        descriptor.capabilities.supports_thinking = thinking;
        descriptor.capabilities.supports_reasoning_effort = reasoning;
        if reasoning {
            descriptor.capabilities.reasoning_efforts =
                vec![ReasoningEffort::XHigh, ReasoningEffort::Medium];
        }
        descriptor.capabilities.model_sampling_defaults = Some(bonsai_sampling_defaults());
        descriptor.capabilities.supports_preserve_thinking = preserve;
        descriptor.capabilities.supports_tools = tools;
        let model = Qwen35Model::from_prism_weights(&loaded.weights, qcfg, &loaded.pack)
            .map_err(to_core)?;
        let vision = match spec.projector_source.as_deref() {
            Some(source) => {
                let projector = crate::gguf::GgufFile::open(source).map_err(to_core)?;
                let loaded_vision = crate::prism_vision_gguf::load(&projector).map_err(to_core)?;
                let tower = Qwen35VisionModel::from_weights(
                    &loaded_vision.weights,
                    "vision_tower",
                    loaded_vision.config.clone(),
                )
                .map_err(to_core)?;
                descriptor.capabilities.supports_vision = true;
                descriptor.capabilities.supports_video = true;
                Some(Qwen35Vision {
                    tower,
                    processor: Qwen35ImageProcessor::default(),
                    image_token_id: 248056,
                    video_token_id: 248057,
                    spatial_merge_size: loaded_vision.config.spatial_merge_size,
                })
            }
            None => None,
        };
        Ok(Self {
            descriptor,
            architecture: Architecture::Qwen35,
            campaign_family: None,
            kv_family: None,
            model: Decoder::Qwen35(Box::new(model)),
            tokenizer: loaded.tokenizer,
            template: loaded.template,
            tool_call_format: tools.then_some(ToolCallFormat::Tagged),
            stop_tokens: loaded.stop_tokens,
            constraint_table: OnceCell::new(),
            vision,
            gemma4: None,
            campaign_prefix_cache: RefCell::new(None),
            execution_fence: LlamaProvider::new_execution_fence(),
            campaign_template_date: None,
            kv_reader: OnceCell::new(),
            kv_model_name: spec.source.clone(),
            _prism_vision_weights: None,
            prefix: RefCell::new(PrefixCache::with_budget(
                core_llm::defaults::MLX.prefix_cache_bytes,
            )),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport::default(),
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// The cross-turn prefix cache's byte budget (sc-24437) — what the load admitted for it.
    pub fn prefix_cache_budget(&self) -> u64 {
        self.prefix.borrow().budget_bytes()
    }

    /// Bytes the prefix cache holds now (never more than [`prefix_cache_budget`](Self::prefix_cache_budget)).
    pub fn prefix_cache_resident_bytes(&self) -> u64 {
        self.prefix.borrow().resident_bytes()
    }

    /// The prefix cache's cumulative reuse accounting.
    pub fn prefix_cache_stats(&self) -> PrefixStats {
        self.prefix.borrow().stats()
    }

    /// Settle the prefix cache budget the load admitted: an empty cache of `budget` bytes, and
    /// the load report naming it.
    fn with_prefix_budget(mut self, budget: u64) -> Self {
        self.prefix.replace(PrefixCache::with_budget(budget));
        self.load_report.prefix_cache_bytes = Some(budget);
        self
    }

    /// The speculative option a request that leaves it unset runs with on this provider: the MLX
    /// defaults-table default ([`core_llm::DecodeDefaults::speculative`], epic sc-24432 E5)
    /// unless [`set_speculative_default`](Self::set_speculative_default) replaced it.
    pub fn speculative_default(&self) -> core_llm::Speculative {
        self.speculative_default
    }

    /// Replace the speculative option an unset request runs with on this provider (a deployment
    /// choice, e.g. a server flag); a request's own `speculative` / legacy `mtp` still wins.
    pub fn set_speculative_default(&mut self, speculative: core_llm::Speculative) {
        self.speculative_default = speculative;
    }

    /// Whether a Prism VLM's dense vision tensors were retained for the multimodal adapter.
    pub fn has_deferred_prism_vision(&self) -> bool {
        self._prism_vision_weights.is_some()
    }

    /// Assemble a provider from already-loaded parts with a default Llama-3 template (used by tests
    /// and converters that don't have a `tokenizer_config.json`).
    pub fn from_parts(model: CausalLm, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        let architecture = model.config().architecture;
        let campaign_family = match architecture {
            Architecture::Llama => Some("llama"),
            Architecture::Qwen3 => Some("qwen"),
            _ => None,
        };
        Self {
            descriptor: {
                let mut d = provider_descriptor();
                d.capabilities.speculative = vec![prompt_lookup_capabilities(
                    speculative_max_depth(causal_attention_geometries(model.config())),
                )];
                d
            },
            architecture,
            campaign_family,
            kv_family: None,
            model: Decoder::Causal(Box::new(model)),
            tokenizer,
            template: Box::new(Llama3Template),
            tool_call_format: None,
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision: None,
            gemma4: None,
            campaign_prefix_cache: RefCell::new(None),
            execution_fence: LlamaProvider::new_execution_fence(),
            campaign_template_date: None,
            kv_reader: OnceCell::new(),
            kv_model_name: String::new(),
            _prism_vision_weights: None,
            prefix: RefCell::new(PrefixCache::with_budget(
                core_llm::defaults::MLX.prefix_cache_bytes,
            )),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport::default(),
        }
    }

    /// Build the multimodal prefill: encode each visual (image or video) in **document order**
    /// (preprocess → ViT → merged rows), expand the rendered `image_pad` / `video_pad` placeholders to
    /// the per-visual / per-frame token counts, splice the features into the token embeds, and compute
    /// the interleaved M-RoPE 3-D positions over the image **and** video grids. `prompt_ids` is the
    /// tokenized prompt (one `image_token_id` per image; one `video_token_id` per frame from the
    /// Text–Timestamp-Alignment placeholders). `messages` is the *original* (un-substituted)
    /// conversation, walked to recover the visual order.
    /// Build Gemma 4's multimodal prefill: encode every image and audio clip in **document order**,
    /// expand the rendered `<|image|>` / `<|audio|>` markers into their framed soft-token spans, and
    /// splice the feature rows onto the matching placeholder positions.
    ///
    /// Simpler than the Qwen-VL path in every dimension that matters here: no ViT, no M-RoPE, no
    /// DeepStack. The spans are ordinary positions in a causal 1-D sequence, so the continuation
    /// needs no position shift.
    fn prepare_gemma4(
        &self,
        prompt_ids: &[i32],
        messages: &[Message],
    ) -> CoreResult<Gemma4Prefill> {
        let rt = self.gemma4.as_ref().ok_or_else(|| {
            CoreError::Load("gemma 4: provider has no multimodal front-end".into())
        })?;
        let cfg = &rt.mm.cfg;

        // Images: one feature block and one soft-token count per image, in document order.
        let images = collect_images(messages);
        let mut image_feats: Vec<Array> = Vec::with_capacity(images.len());
        let mut image_counts: Vec<usize> = Vec::with_capacity(images.len());
        if !images.is_empty() {
            let tower = rt.mm.vision.as_ref().ok_or_else(|| {
                CoreError::Unsupported(
                    "[mlx-llama] Gemma 4: this checkpoint ships no vision embedder, so it cannot \
                     be conditioned on an image"
                        .to_string(),
                )
            })?;
            let vcfg = tower.config().clone();
            for img in &images {
                let (gh, gw) = gemma4_mm::soft_token_grid(
                    img.width as usize,
                    img.height as usize,
                    vcfg.max_soft_tokens,
                );
                let (tw, th) = (gw * vcfg.patch_pixels, gh * vcfg.patch_pixels);
                // Resize to an exact patch multiple through the PIL-matching bicubic path, then
                // patchify. `resample: 3` in the shipped processor config is PIL BICUBIC.
                // NOTE the argument order: `resize_bicubic_u8` takes HEIGHT before WIDTH, and
                // returns f32 samples still on the 0..255 scale (the rescale to 0..1 happens in
                // `patchify`). Passing width first would transpose every non-square image.
                let resized = crate::image::resize_bicubic_u8(
                    &img.pixels,
                    img.height as usize,
                    img.width as usize,
                    th,
                    tw,
                )
                .map_err(|e| to_core(e.into()))?;
                let flat =
                    gemma4_mm::patchify(&resized, tw, th, vcfg.patch_pixels).map_err(to_core)?;
                let patches = gemma4_mm::patch_array(&flat, vcfg.patch_elems()).map_err(to_core)?;
                let feats = tower.forward(&patches, (gh, gw)).map_err(to_core)?;
                image_counts.push(gh * gw);
                image_feats.push(feats);
            }
        }

        // Audio: one feature block and one soft-token count per clip, in document order.
        let clips = collect_audio(messages);
        let mut audio_feats: Vec<Array> = Vec::with_capacity(clips.len());
        let mut audio_counts: Vec<usize> = Vec::with_capacity(clips.len());
        if !clips.is_empty() {
            let proj = rt.mm.audio.as_ref().ok_or_else(|| {
                CoreError::Unsupported(
                    "[mlx-llama] Gemma 4: this checkpoint ships no audio projector, so it cannot \
                     be conditioned on audio"
                        .to_string(),
                )
            })?;
            let acfg = proj.config().clone();
            for clip in &clips {
                // The projector's framing is defined in samples at a fixed rate; resampling behind
                // the caller's back would silently change what the model hears.
                if clip.sample_rate != acfg.sample_rate {
                    return Err(CoreError::InvalidRequest(format!(
                        "[mlx-llama] Gemma 4 audio expects {} Hz mono PCM, got {} Hz; resample \
                         before sending",
                        acfg.sample_rate, clip.sample_rate
                    )));
                }
                let framed = gemma4_mm::audio_frames(
                    &clip.samples,
                    acfg.samples_per_token,
                    acfg.max_soft_tokens,
                )
                .map_err(to_core)?;
                let frames =
                    gemma4_mm::frame_array(&framed, acfg.samples_per_token).map_err(to_core)?;
                let feats = proj.forward(&frames).map_err(to_core)?;
                audio_counts.push(framed.len() / acfg.samples_per_token);
                audio_feats.push(feats);
            }
        }

        // Expand each marker into `begin` + count soft tokens + `end`. Images first, then audio;
        // each pass touches only its own marker id, so interleaved order is preserved.
        let expanded = gemma4_mm::expand_framed_placeholders(
            prompt_ids,
            cfg.image_token_id,
            cfg.boi_token_id,
            cfg.eoi_token_id,
            &image_counts,
        )
        .map_err(to_core)?;
        let expanded = gemma4_mm::expand_framed_placeholders(
            &expanded,
            cfg.audio_token_id,
            cfg.boa_token_id,
            cfg.eoa_token_id,
            &audio_counts,
        )
        .map_err(to_core)?;

        // Splice. Each modality is placed on its own token id, so the two calls cannot collide.
        let model = match &self.model {
            Decoder::Causal(m) => m,
            Decoder::Qwen35(_) => {
                return Err(CoreError::Load(
                    "gemma 4: the multimodal path requires the generic causal decoder".into(),
                ))
            }
        };
        let mut embeds = model
            .embed_input_ids(&input_ids(&expanded))
            .map_err(to_core)?;
        if !image_feats.is_empty() {
            let all = gemma4_mm::concat_features(&image_feats).map_err(to_core)?;
            embeds = model
                .splice_vision_features(&embeds, &expanded, &all, &[cfg.image_token_id])
                .map_err(to_core)?;
        }
        if !audio_feats.is_empty() {
            let all = gemma4_mm::concat_features(&audio_feats).map_err(to_core)?;
            embeds = model
                .splice_vision_features(&embeds, &expanded, &all, &[cfg.audio_token_id])
                .map_err(to_core)?;
        }

        Ok(Gemma4Prefill {
            expanded_ids: expanded,
            embeds,
        })
    }

    fn prepare_multimodal(
        &self,
        prompt_ids: &[i32],
        messages: &[Message],
    ) -> CoreResult<MultimodalPrefill> {
        let vision = self.vision.as_ref().ok_or_else(|| {
            CoreError::Load("qwen-vl vision: provider has no vision tower".into())
        })?;

        // Walk the conversation in document order; encode each visual once, in order, so the
        // concatenated feature buffer lines up one-to-one with the visual placeholder spans of the
        // (image+video) prompt. Image placeholders expand to one count; a video expands to `grid_t`
        // per-frame counts (`frame_seqlen` each), in frame order.
        let mut feats: Vec<Array> = Vec::new();
        let mut image_counts: Vec<usize> = Vec::new();
        let mut video_counts: Vec<usize> = Vec::new();
        let mut image_grids: Vec<[i32; 3]> = Vec::new();
        let mut video_grids: Vec<[i32; 3]> = Vec::new();
        let mut deepstack_by_tap: Vec<Vec<Array>> = Vec::new();
        let merge = vision.spatial_merge_size;

        let mut push_deepstack = |deepstack: Vec<Array>| -> CoreResult<()> {
            if deepstack_by_tap.is_empty() {
                deepstack_by_tap.resize_with(deepstack.len(), Vec::new);
            }
            if deepstack.len() != deepstack_by_tap.len() {
                return Err(CoreError::Load(format!(
                    "qwen-vl vision: inconsistent DeepStack tap count {} != {}",
                    deepstack.len(),
                    deepstack_by_tap.len()
                )));
            }
            for (tap, feature) in deepstack.into_iter().enumerate() {
                deepstack_by_tap[tap].push(feature);
            }
            Ok(())
        };

        for m in messages {
            for c in &m.content {
                match c {
                    Content::Image(img) => {
                        let (f, deepstack, grid) = vision.encode(img)?;
                        image_counts.push(f.shape()[0] as usize);
                        image_grids.push(grid);
                        feats.push(f);
                        push_deepstack(deepstack)?;
                    }
                    Content::Video(video) => {
                        let (f, deepstack, grid) = vision.encode_video(video)?;
                        // One placeholder count per frame: `frame_seqlen = (h/merge)·(w/merge)`.
                        let [gt, gh, gw] = grid;
                        let frame_seqlen = ((gh / merge) * (gw / merge)) as usize;
                        for _ in 0..gt {
                            video_counts.push(frame_seqlen);
                        }
                        video_grids.push(grid);
                        feats.push(f);
                        push_deepstack(deepstack)?;
                    }
                    Content::Text(_) => {}
                    // Rejected by `validate` long before here: this provider reports
                    // `supports_audio=false` for every Qwen-VL checkpoint. Refuse rather than skip,
                    // so a broken gate cannot answer an audio question from the text alone.
                    Content::Audio(_) => {
                        return Err(CoreError::Unsupported(
                            "[mlx-llama] the Qwen-VL prefill carries no audio; an audio block \
                             reached it, which the capability gate should have rejected"
                                .to_string(),
                        ))
                    }
                }
            }
        }

        // Expand both placeholder tokens to their per-occurrence counts. Each call only touches its
        // own token, so order across the two is preserved and the result interleaves correctly.
        let img_id = vision.image_token_id;
        let vid_id = vision.video_token_id;
        let expanded =
            crate::models::qwen35::expand_vision_placeholders(prompt_ids, img_id, &image_counts)
                .map_err(to_core)?;
        let expanded =
            crate::models::qwen35::expand_vision_placeholders(&expanded, vid_id, &video_counts)
                .map_err(to_core)?;
        let visual_pos_mask: Vec<bool> = expanded
            .iter()
            .map(|&id| id == img_id || id == vid_id)
            .collect();

        let refs: Vec<&Array> = feats.iter().collect();
        let all_features = match refs.as_slice() {
            [one] => (*one).clone(),
            many => concatenate_axis(many, 0).map_err(|e| to_core(e.into()))?,
        };
        let mut deepstack = Vec::with_capacity(deepstack_by_tap.len());
        for by_visual in deepstack_by_tap {
            let refs: Vec<&Array> = by_visual.iter().collect();
            deepstack.push(match refs.as_slice() {
                [one] => (*one).clone(),
                many => concatenate_axis(many, 0).map_err(|e| to_core(e.into()))?,
            });
        }

        // Embed the expanded ids, splice in the vision features (image+video placeholder rows), and
        // compute interleaved-M-RoPE positions over both image and video grids — through the shared
        // `VlmDecode` seam, identical for whichever decoder powers this VLM (Qwen3.6 hybrid or
        // Qwen3-VL generic-causal).
        let placeholders = [img_id, vid_id];
        let model = self.model.as_vlm();
        let embeds = model
            .embed_input_ids(&input_ids(&expanded))
            .map_err(to_core)?;
        let spliced = model
            .splice_vision_features(&embeds, &expanded, &all_features, &placeholders)
            .map_err(to_core)?;
        let positions = model
            .mrope_positions_mm(&expanded, &image_grids, img_id, &video_grids, vid_id, merge)
            .map_err(to_core)?;

        Ok(MultimodalPrefill {
            expanded_ids: expanded,
            embeds: spliced,
            positions,
            visual_pos_mask,
            deepstack,
        })
    }
}

fn campaign_cancellation_probe_request() -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::text(
            Role::User,
            "Reply with exactly these words: alpha beta gamma delta epsilon zeta eta theta.",
        )],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    }
}

/// Adapts a `core_llm::JsonConstraint` to the engine's [`ConstraintMask`] decode seam.
struct JsonMask<'a> {
    inner: JsonConstraint<'a>,
    table: &'a ConstraintDecodeTable,
    stop_ids: Vec<u32>,
    accepted: Vec<i32>,
    initial_reasoning: bool,
    reasoning: bool,
    reasoning_tail: String,
    allow: Vec<bool>,
}

impl<'a> JsonMask<'a> {
    fn new(
        table: &'a ConstraintDecodeTable,
        stop_ids: impl IntoIterator<Item = u32>,
        reasoning: bool,
    ) -> Self {
        let stop_ids = stop_ids.into_iter().collect::<Vec<_>>();
        Self {
            inner: JsonConstraint::new(table, stop_ids.iter().copied()),
            table,
            stop_ids,
            accepted: Vec::new(),
            initial_reasoning: reasoning,
            reasoning,
            reasoning_tail: String::new(),
            allow: vec![false; table.pieces.len()],
        }
    }

    fn accept_reasoning_piece(&mut self, piece: &str) {
        self.reasoning_tail.push_str(piece);
        if let Some(end) = self.reasoning_tail.find("</think>") {
            let answer = self.reasoning_tail[end + "</think>".len()..].to_string();
            self.reasoning = false;
            self.reasoning_tail.clear();
            let accepted = self.inner.accept_text(&answer);
            debug_assert!(accepted);
            return;
        }
        let keep = (1.."</think>".len())
            .rev()
            .find(|&n| self.reasoning_tail.ends_with(&"</think>"[..n]))
            .unwrap_or(0);
        if self.reasoning_tail.len() > keep {
            self.reasoning_tail
                .drain(..self.reasoning_tail.len() - keep);
        }
    }
}

impl ConstraintMask for JsonMask<'_> {
    fn allowed(&mut self) -> &[bool] {
        if !self.reasoning {
            return self.inner.allowed();
        }
        for (id, slot) in self.allow.iter_mut().enumerate() {
            if self.stop_ids.contains(&(id as u32)) {
                *slot = false;
                continue;
            }
            let mut candidate = self.reasoning_tail.clone();
            candidate.push_str(&self.table.pieces[id]);
            *slot = candidate
                .find("</think>")
                .is_none_or(|end| self.inner.allows_text(&candidate[end + "</think>".len()..]));
        }
        &self.allow
    }
    fn accept(&mut self, token: i32) {
        if self.reasoning {
            let piece = self
                .table
                .pieces
                .get(token as usize)
                .cloned()
                .unwrap_or_default();
            self.accept_reasoning_piece(&piece);
        } else {
            self.inner.accept(token as u32);
        }
        self.accepted.push(token);
    }
}

impl RewindableConstraintMask for JsonMask<'_> {
    fn checkpoint(&self) -> usize {
        self.accepted.len()
    }

    fn rewind(&mut self, checkpoint: usize) {
        // Nothing was accepted since the checkpoint: the state is already there, and a rebuild +
        // replay of every accepted token would make each constrained step O(n).
        if checkpoint == self.accepted.len() {
            return;
        }
        self.accepted.truncate(checkpoint);
        self.inner = JsonConstraint::new(self.table, self.stop_ids.iter().copied());
        self.reasoning_tail.clear();
        self.reasoning = self.initial_reasoning;
        let accepted = self.accepted.clone();
        for token in accepted {
            self.accept(token);
        }
        self.accepted.truncate(checkpoint);
    }
}

/// Use the model's own Jinja `chat_template` (from `tokenizer_config.json`, story 7164) when
/// present; otherwise fall back to the typed Llama-3 template. Also reports two template-gated
/// capabilities, detected from the source (not the family, matching the transformers convention):
/// - **thinking** — the template gates an `enable_thinking` kwarg (sc-7585).
/// - **tools** — the template renders either tagged tool-call blocks or the bare JSON format shipped
///   by Llama 3.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolCallFormat {
    Tagged,
    BareJson,
}

fn tool_call_format(source: &str) -> Option<ToolCallFormat> {
    if source.contains("<tool_call>") {
        Some(ToolCallFormat::Tagged)
    } else if source.contains("tool_call")
        && source.contains("respond with JSON for a function call")
    {
        Some(ToolCallFormat::BareJson)
    } else {
        None
    }
}

fn load_chat_template(
    dir: &Path,
) -> (
    Box<dyn ChatTemplate>,
    bool,
    bool,
    bool,
    bool,
    Option<ToolCallFormat>,
) {
    // The sidecar `chat_template.jinja` wins over the embedded key — see `sidecar_chat_template`.
    if let Some(t) = sidecar_chat_template(dir) {
        let supports_thinking = t.source().contains("enable_thinking");
        let supports_reasoning_effort = t.source().contains("reasoning_effort");
        let supports_preserve_thinking = t.source().contains("preserve_thinking");
        let supports_tools = t.source().contains("tool_call");
        let format = tool_call_format(t.source());
        return (
            Box::new(t),
            supports_thinking,
            supports_reasoning_effort,
            supports_preserve_thinking,
            supports_tools,
            format,
        );
    }
    match JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json")) {
        Ok(t) => {
            let supports_thinking = t.source().contains("enable_thinking");
            let supports_reasoning_effort = t.source().contains("reasoning_effort");
            let supports_preserve_thinking = t.source().contains("preserve_thinking");
            let supports_tools = t.source().contains("tool_call");
            let format = tool_call_format(t.source());
            (
                Box::new(t),
                supports_thinking,
                supports_reasoning_effort,
                supports_preserve_thinking,
                supports_tools,
                format,
            )
        }
        Err(_) => (Box::new(Llama3Template), false, false, false, false, None),
    }
}

/// Match the text-only `generate_inner` rendering/tokenization path without loading weights.
/// Campaign preflight uses this before MLX admission; the product still revalidates at dispatch.
pub(crate) fn campaign_preflight_request_tokens(
    snapshot: &Path,
    request: &TextLlmRequest,
) -> CoreResult<u64> {
    let tokenizer = Tokenizer::from_file(snapshot.join("tokenizer.json"))?;
    let (template, ..) = load_chat_template(snapshot);
    let prompt = template.render_with(
        &request.messages,
        &RenderOptions {
            add_generation_prompt: true,
            enable_thinking: request.enable_thinking_kwarg(),
            reasoning_effort: request.reasoning_effort,
            preserve_thinking: request.preserve_thinking,
            tools: &request.tools,
            date_string: Some(crate::campaign::CAMPAIGN_CHAT_TEMPLATE_DATE_STRING),
        },
    )?;
    u64::try_from(tokenizer.encode(&prompt, false)?.len())
        .ok()
        .and_then(|tokens| tokens.checked_add(u64::from(request.max_new_tokens)))
        .ok_or_else(|| {
            CoreError::InvalidRequest("campaign rendered request token count overflow".into())
        })
}

pub(crate) fn campaign_preflight_cancellation_tokens(snapshot: &Path) -> CoreResult<u64> {
    campaign_preflight_request_tokens(snapshot, &campaign_cancellation_probe_request())
}

/// The modern HF layout ships the chat template as a **separate `chat_template.jinja`** beside
/// `tokenizer_config.json` rather than inside it (transformers writes it that way for newer
/// releases — `google/gemma-4-12B-it` is one, and its `tokenizer_config.json` carries no
/// `chat_template` key at all).
///
/// Reading only the embedded key means such a snapshot silently falls back to the typed Llama-3
/// default: the prompt still renders, the model still generates, and every assertion about shapes
/// and lengths still passes — it just answers badly, because it is being addressed in a chat format
/// it was never trained on. Prefer the sidecar file, then the embedded key, then the default.
///
/// BOS/EOS still come from `tokenizer_config.json`; the sidecar carries only the template body.
fn sidecar_chat_template(dir: &Path) -> Option<JinjaChatTemplate> {
    let source = crate::__model_input_reader!(
        std::fs::read_to_string(dir.join("chat_template.jinja")),
        "crates/llm/mlx-llm/src/provider.rs::sidecar_chat_template::std::fs::read_to_string@120781",
        ArtifactInput,
        dir.join("chat_template.jinja"),
        |__model_input_path| std::fs::read_to_string(__model_input_path)
    )
    .ok()?;
    if source.trim().is_empty() {
        return None;
    }
    let (bos, eos) = tokenizer_special_tokens(dir);
    Some(JinjaChatTemplate::with_tokens(source, bos, eos))
}

/// `(bos_token, eos_token)` strings from `tokenizer_config.json`, empty when absent. Each may be a
/// bare string or an `AddedToken` object carrying `content`.
fn tokenizer_special_tokens(dir: &Path) -> (String, String) {
    let Some(v) = read_json(dir, "tokenizer_config.json") else {
        return (String::new(), String::new());
    };
    let token = |key: &str| -> String {
        match v.get(key) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Object(o)) => o
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string(),
            _ => String::new(),
        }
    };
    (token("bos_token"), token("eos_token"))
}

/// Run a piece of answer-channel text through the tool-call segmenter when active, returning the
/// plain-content runs to stream (tool-call blocks lifted out + parsed). With no segmenter the text
/// passes straight through, so the non-tools path is byte-identical to before.
fn tool_pieces(seg: &mut Option<ToolCallSegmenter>, text: &str) -> Vec<String> {
    match seg {
        Some(ts) => ts.push(text),
        None => vec![text.to_string()],
    }
}

/// Parse the complete bare JSON tool-call form emitted by Llama 3.2. This is deliberately an
/// end-of-generation operation: unlike tagged formats, raw JSON has no streaming boundary that can
/// distinguish a tool call from ordinary answer text until the document is complete.
fn bare_json_tool_calls(text: &str, tools: &[core_llm::ToolSpec]) -> Vec<ToolCall> {
    let mut parser = ToolCallSegmenter::new(tools);
    let wrapped = format!("<tool_call>{text}</tool_call>");
    let mut remainder = parser.push(&wrapped).concat();
    remainder.push_str(&parser.flush().concat());
    let calls = parser.take_calls();
    if remainder.trim().is_empty() {
        calls
    } else {
        Vec::new()
    }
}

/// Push one content piece through the stop matcher and emit the released text as a Content token
/// event. Shared by the streaming loop and the end-of-generation tails. `*last_id` / `*emit_index`
/// advance only when text is actually emitted, so the contract's token index stays gap-free across
/// stripped markers and lifted-out tool-call blocks.
#[allow(clippy::too_many_arguments)]
fn emit_content(
    piece: &str,
    id: u32,
    stop_matcher: &mut StopMatcher,
    streamed: &mut String,
    emit_index: &mut usize,
    last_id: &mut u32,
    halt: &std::cell::Cell<bool>,
    on_event: &mut dyn FnMut(CoreEvent),
) {
    let chunk = stop_matcher.push(piece);
    if !chunk.emit.is_empty() {
        streamed.push_str(&chunk.emit);
        *last_id = id;
        on_event(CoreEvent::Token {
            id,
            text: chunk.emit,
            index: *emit_index,
            channel: Channel::Content,
        });
        *emit_index += 1;
    }
    if chunk.stop {
        halt.set(true);
    }
}

/// One generation's token-to-text pipeline (shared by the single-request decode and, sc-20681,
/// the batched one): incremental detokenization (holding back lossy U+FFFD placeholders so a
/// multi-byte character split across BPE tokens streams intact — sc-12452), the thinking
/// segmenter (reasoning vs answer), the tool-call segmenter (lifts `<tool_call>` blocks out of the
/// answer) and the stop matcher (trims a stop string and trips `halt`), emitting contract token
/// events as text is released.
struct TextPipeline<'a> {
    tokenizer: &'a Tokenizer,
    acc: Vec<u32>,
    detok: IncrementalDetok,
    segmenter: Option<ThinkingSegmenter>,
    tool_seg: Option<ToolCallSegmenter>,
    stop_matcher: StopMatcher,
    /// Content emitted as the matchers release it.
    streamed: String,
    /// Reasoning text, accumulated from the Thinking channel (sc-7585).
    thinking_buf: String,
    /// Contract token index: a running counter over *emitted* events, not the raw decode step —
    /// detok hold-backs and stripped `<think>`/`</think>` marker tokens produce no event, so this
    /// stays gap-free (and equals the step in the common one-delta-per-token case).
    emit_index: usize,
    /// Id of the last emitted token, for flushed-tail events.
    last_id: u32,
}

impl<'a> TextPipeline<'a> {
    fn new(
        tokenizer: &'a Tokenizer,
        segmenter: Option<ThinkingSegmenter>,
        tool_seg: Option<ToolCallSegmenter>,
        stop_matcher: StopMatcher,
    ) -> Self {
        Self {
            tokenizer,
            acc: Vec::new(),
            detok: IncrementalDetok::new(),
            segmenter,
            tool_seg,
            stop_matcher,
            streamed: String::new(),
            thinking_buf: String::new(),
            emit_index: 0,
            last_id: 0,
        }
    }

    /// One generated token: re-decode the running sequence, emit the new suffix through the
    /// segmenters and the stop matcher.
    fn push(&mut self, id: u32, halt: &std::cell::Cell<bool>, on_event: &mut dyn FnMut(CoreEvent)) {
        self.acc.push(id);
        let Ok(text) = self.tokenizer.decode(&self.acc, true) else {
            return;
        };
        let Some(delta) = self.detok.push(&text) else {
            return;
        };
        let delta = delta.to_string();
        match self.segmenter.as_mut() {
            Some(seg) => {
                for span in seg.push(&delta) {
                    match span.channel {
                        Channel::Thinking => {
                            self.thinking_buf.push_str(&span.text);
                            self.last_id = id;
                            on_event(CoreEvent::Token {
                                id,
                                text: span.text,
                                index: self.emit_index,
                                channel: Channel::Thinking,
                            });
                            self.emit_index += 1;
                        }
                        Channel::Content => {
                            // Answer text → tool segmenter (lifts out tool-call blocks) → stop
                            // matcher → emit.
                            for piece in tool_pieces(&mut self.tool_seg, &span.text) {
                                emit_content(
                                    &piece,
                                    id,
                                    &mut self.stop_matcher,
                                    &mut self.streamed,
                                    &mut self.emit_index,
                                    &mut self.last_id,
                                    halt,
                                    &mut *on_event,
                                );
                            }
                        }
                    }
                }
            }
            None => {
                for piece in tool_pieces(&mut self.tool_seg, &delta) {
                    emit_content(
                        &piece,
                        id,
                        &mut self.stop_matcher,
                        &mut self.streamed,
                        &mut self.emit_index,
                        &mut self.last_id,
                        halt,
                        &mut *on_event,
                    );
                }
            }
        }
    }

    /// End-of-generation tails, in pipeline order. First the thinking segmenter's held-back
    /// partial marker (it turned out not to begin a marker) as current-channel text — reasoning
    /// straight out, answer through the tool segmenter; then the tool segmenter's own tail (held
    /// partial `<tool_call>` / an unterminated block surfaced as content); then the stop matcher's
    /// held-back partial stop. Returns the streamed answer, the reasoning and the tool segmenter.
    fn finish(
        mut self,
        halt: &std::cell::Cell<bool>,
        stop_active: bool,
        on_event: &mut dyn FnMut(CoreEvent),
    ) -> (String, String, Option<ToolCallSegmenter>) {
        if let Some(seg) = self.segmenter.as_mut() {
            for span in seg.flush() {
                match span.channel {
                    Channel::Thinking => {
                        self.thinking_buf.push_str(&span.text);
                        on_event(CoreEvent::Token {
                            id: self.last_id,
                            text: span.text,
                            index: self.emit_index,
                            channel: Channel::Thinking,
                        });
                        self.emit_index += 1;
                    }
                    Channel::Content => {
                        for piece in tool_pieces(&mut self.tool_seg, &span.text) {
                            let id = self.last_id;
                            emit_content(
                                &piece,
                                id,
                                &mut self.stop_matcher,
                                &mut self.streamed,
                                &mut self.emit_index,
                                &mut self.last_id,
                                halt,
                                &mut *on_event,
                            );
                        }
                    }
                }
            }
        }
        if let Some(ts) = self.tool_seg.as_mut() {
            for piece in ts.flush() {
                let id = self.last_id;
                emit_content(
                    &piece,
                    id,
                    &mut self.stop_matcher,
                    &mut self.streamed,
                    &mut self.emit_index,
                    &mut self.last_id,
                    halt,
                    &mut *on_event,
                );
            }
        }
        // If generation ended for any reason other than a stop string, flush the stop matcher's
        // held-back tail (a partial stop-prefix that never completed) — real output to
        // stream/return.
        if stop_active && !halt.get() {
            let tail = self.stop_matcher.flush();
            if !tail.is_empty() {
                self.streamed.push_str(&tail);
                on_event(CoreEvent::Token {
                    id: self.last_id,
                    text: tail,
                    index: self.emit_index,
                    channel: Channel::Content,
                });
            }
        }
        (self.streamed, self.thinking_buf, self.tool_seg)
    }
}

impl TextLlm for LlamaProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn synchronize_before_release(&self) -> CoreResult<()> {
        self.execution_fence
            .synchronize(crate::primitives::synchronize_stream)
            .map_err(to_core)
    }

    fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn load_report(&self) -> Option<core_llm::LoadReport> {
        Some(self.load_report.clone())
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
    ) -> CoreResult<TextLlmOutput> {
        self.generate_inner(req, on_event, None, None, None)
    }

    /// Continuous batching of the requests the batched decode serves (sc-20681): plain text
    /// generations on the causal decoder (no media, JSON constraint, tools or stop strings),
    /// decoded together by [`crate::decode::generate_continuous_kv`] — each with its own
    /// compressed-KV opt-in and report and its own cancellation. The compressed KV cache is
    /// qualified for single-sequence decodes only, so every request decoding beside another runs
    /// dense and reports [`core_llm::KvCacheFallbackReason::BatchedDecode`], exactly as the
    /// single-request plan reports a batch (sc-20688 review). Every other request, and any request
    /// the batch's memory admission does not fit, runs on its own through [`TextLlm::generate`]
    /// afterwards.
    fn generate_batch(
        &self,
        reqs: &[TextLlmRequest],
        on_event: &mut dyn FnMut(usize, CoreEvent),
    ) -> Vec<CoreResult<TextLlmOutput>> {
        let mut results: Vec<Option<CoreResult<TextLlmOutput>>> =
            (0..reqs.len()).map(|_| None).collect();
        let batchable = (0..reqs.len())
            .filter(|&i| self.batchable(&reqs[i]))
            .collect::<Vec<_>>();
        if batchable.len() >= 2 {
            for (i, result) in self.generate_text_batch(reqs, &batchable, on_event) {
                results[i] = Some(result);
            }
        }
        results
            .into_iter()
            .enumerate()
            .map(|(i, result)| {
                result.unwrap_or_else(|| self.generate(&reqs[i], &mut |event| on_event(i, event)))
            })
            .collect()
    }
}

/// Tokens per compressed page of the provider's batched decode.
const BATCH_PAGE_TOKENS: usize = 64;

impl LlamaProvider {
    /// Whether `req` is a plain, non-speculative text generation the batched decode serves exactly
    /// as [`TextLlm::generate`] would (sc-20681).
    fn batchable(&self, req: &TextLlmRequest) -> bool {
        matches!(self.model, Decoder::Causal(_))
            && collect_images(&req.messages).is_empty()
            && collect_videos(&req.messages).is_empty()
            && collect_audio(&req.messages).is_empty()
            && req.constraint.is_none()
            && req.tools.is_empty()
            && req.stop.is_empty()
            // The batched decode has no proposer: a request that resolves to speculation runs on
            // its own through the engine, never a silent plain decode (epic sc-24432 E2).
            && req.speculative_or(self.speculative_default) == core_llm::Speculative::Off
    }

    /// Decode the requests `indices` of `reqs` together; their results (a request rejected
    /// before decoding included). Requests left out — memory admission did not fit them, or only
    /// one was left to decode — get no result here and run on their own.
    fn generate_text_batch(
        &self,
        reqs: &[TextLlmRequest],
        indices: &[usize],
        on_event: &mut dyn FnMut(usize, CoreEvent),
    ) -> Vec<(usize, CoreResult<TextLlmOutput>)> {
        let Decoder::Causal(model) = &self.model else {
            return Vec::new();
        };
        let mut results = Vec::new();
        let mut admitted: Vec<(usize, String, Vec<i32>)> = Vec::new();
        let available = core_llm::operational_memory_override().and_then(|operational| {
            core_llm::effective_memory_budget(core_llm::available_host_memory_bytes(), operational)
        });
        let mut required_total = 0_u64;
        for &i in indices {
            let req = &reqs[i];
            let prepared = (|| -> CoreResult<Option<(String, Vec<i32>)>> {
                self.validate(req)?;
                if req.cancel.is_cancelled() {
                    return Err(CoreError::Canceled);
                }
                let (prompt, prompt_ids) = self.render_prompt(req, &req.messages)?;
                let max_context = self.descriptor.capabilities.max_context_tokens;
                validate_context_window(max_context, prompt_ids.len(), req.max_new_tokens)?;
                let required = estimate_mlx_request_bytes(
                    prompt_ids.len(),
                    req.max_new_tokens,
                    self.model.memory_geometry(),
                    0,
                    0,
                    self.model.workspace_contract(),
                )
                .ok_or_else(|| {
                    CoreError::InvalidRequest("request memory estimate overflow".into())
                })?;
                let available = available
                    .as_ref()
                    .map_err(|error| CoreError::InvalidRequest(error.to_string()))?;
                // The batch is admitted request by request against the running total; one that
                // does not fit beside the others runs alone later, under its own admission.
                let total = required_total.saturating_add(required);
                if core_llm::admit_request_memory_with_geometry(
                    prompt_ids.len(),
                    req.max_new_tokens,
                    max_context,
                    total,
                    *available,
                )
                .is_err()
                {
                    return Ok(None);
                }
                required_total = total;
                Ok(Some((prompt, prompt_ids)))
            })();
            match prepared {
                Ok(Some((prompt, prompt_ids))) => admitted.push((i, prompt, prompt_ids)),
                Ok(None) => {}
                Err(error) => results.push((i, Err(error))),
            }
        }
        // A lone survivor (the others cancelled before start, rejected, or not admitted beside
        // it) is not a batch: it gets no result here and runs on its own through
        // `TextLlm::generate`, the measured single-request path — compressed on the contiguous
        // cache where it qualifies, never on the paged cache (sc-20688 review).
        if admitted.len() < 2 {
            return results;
        }
        if let Err(error) = self.note_execution_stream() {
            let message = error.to_string();
            results.extend(
                admitted
                    .into_iter()
                    .map(|(index, _, _)| (index, Err(to_core(crate::Error::Msg(message.clone()))))),
            );
            return results;
        }

        let thinking_active = self.descriptor.capabilities.supports_thinking;
        let halts = admitted
            .iter()
            .map(|_| std::cell::Cell::new(false))
            .collect::<Vec<_>>();
        let mut pipelines = admitted
            .iter()
            .map(|(_, prompt, _)| {
                let mut segmenter = thinking_active.then(ThinkingSegmenter::default);
                if let Some(seg) = segmenter.as_mut() {
                    if prompt_opens_thinking(prompt) {
                        let _ = seg.push("<think>");
                    }
                }
                TextPipeline::new(
                    &self.tokenizer,
                    segmenter,
                    None,
                    StopMatcher::new(Vec::new()),
                )
            })
            .collect::<Vec<_>>();
        let batch = admitted
            .iter()
            .map(|(i, _, prompt_ids)| {
                let req = &reqs[*i];
                crate::decode::BatchRequest {
                    prompt_ids: prompt_ids.clone(),
                    sampling: SamplingParams::from(&req.sampling),
                    seed: req.seed,
                    max_new_tokens: req.max_new_tokens as usize,
                    stop_tokens: self.stop_tokens.clone(),
                }
            })
            .collect::<Vec<_>>();
        let cancels = admitted
            .iter()
            .map(|(i, _, _)| reqs[*i].cancel.clone())
            .collect::<Vec<_>>();
        let policies = admitted
            .iter()
            .map(|(i, _, _)| reqs[*i].kv_compression)
            .collect::<Vec<_>>();
        // The product's batched decode never arms the per-sequence paged compressed experiment
        // (sc-20688 review), so it builds no fused reader, page pool or prefix store: every
        // request decodes dense with its reason.
        let kv = crate::decode::ContinuousKv {
            policy: core_llm::KvCompressionPolicy::Off,
            family: self.kv_family,
            page_tokens: BATCH_PAGE_TOKENS,
            reader: None,
            prefix: None,
            pool: None,
            model_identity: &self.kv_model_name,
            cancels: &cancels,
            policies: &policies,
            // The product path never arms the per-sequence paged compressed experiment.
            #[cfg(test)]
            experimental_per_sequence_compression: false,
        };
        let config = crate::decode::ContinuousConfig {
            max_batch: batch.len(),
            block_size: 16,
            exactness: crate::decode::BatchExactness::Throughput,
        };
        let outputs = crate::decode::generate_continuous_kv(
            model,
            &batch,
            &config,
            kv,
            &crate::decode::CancelFlag::new(),
            &mut |b, event| {
                if let StreamEvent::Token { id, .. } = event {
                    let index = admitted[b].0;
                    pipelines[b].push(id as u32, &halts[b], &mut |e| on_event(index, e));
                }
            },
        );
        let outputs = match outputs {
            Ok(outputs) => outputs,
            Err(error) => {
                let message = error.to_string();
                let mut error = Some(to_core(error));
                for (i, _, _) in &admitted {
                    let error = error
                        .take()
                        .unwrap_or_else(|| CoreError::Msg(message.clone()));
                    results.push((*i, Err(error)));
                }
                return results;
            }
        };
        for (((index, _, prompt_ids), (pipeline, halt)), out) in admitted
            .iter()
            .zip(pipelines.into_iter().zip(&halts))
            .zip(outputs)
        {
            let (streamed, thinking_buf, _) =
                pipeline.finish(halt, false, &mut |e| on_event(*index, e));
            // The single-request rule: the streamed answer when thinking is active, else the
            // decode of every generated token.
            let text = if thinking_active {
                Ok(streamed)
            } else {
                let ids = out
                    .output
                    .tokens
                    .iter()
                    .map(|&t| t as u32)
                    .collect::<Vec<_>>();
                self.tokenizer.decode(&ids, true)
            };
            let text = match text {
                Ok(text) => text,
                Err(error) => {
                    results.push((*index, Err(error)));
                    continue;
                }
            };
            let finish = map_finish(out.output.finish_reason);
            let usage = Usage {
                prompt_tokens: prompt_ids.len() as u32,
                generated_tokens: out.output.tokens.len() as u32,
            };
            on_event(
                *index,
                CoreEvent::Done {
                    finish_reason: finish,
                    usage,
                },
            );
            results.push((
                *index,
                Ok(TextLlmOutput {
                    text,
                    thinking: (!thinking_buf.is_empty()).then_some(thinking_buf),
                    tool_calls: Vec::new(),
                    usage,
                    mtp: None,
                    timings: None,
                    decode: None,
                    kv_cache: Some(out.kv_cache),
                    finish_reason: Some(finish),
                }),
            ));
        }
        results
    }

    /// Campaign-only entrypoint. The observer is never installed on ordinary production calls.
    /// `packed` selects the compressed arm: the observed decode then runs on the packed
    /// group-affine cache with the retained fused reader (SC-20676 compressed rows).
    pub(crate) fn generate_observed(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
        observer: &mut dyn crate::campaign::Observer,
        packed: Option<&crate::campaign::CompressedKvArm>,
    ) -> CoreResult<TextLlmOutput> {
        self.generate_inner(req, on_event, Some(observer), packed, None)
    }

    /// Render `messages` with `req`'s template options and tokenize. The template already includes
    /// BOS, so encode without auto special tokens. `enable_thinking` (sc-7585) flows into the
    /// template kwarg so a no-think (Disabled) request injects the model's empty `<think></think>`
    /// generation prompt; Auto omits the kwarg (template default).
    fn render_prompt(
        &self,
        req: &TextLlmRequest,
        messages: &[Message],
    ) -> CoreResult<(String, Vec<i32>)> {
        let prompt = self.template.render_with(
            messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.enable_thinking_kwarg(),
                // Bonsai's official model card does not recommend `low` as an effective distinct
                // level, so it is omitted from the selectable capability list. The frozen template
                // still accepts and renders `low`; preserve that compatibility input verbatim.
                reasoning_effort: req.reasoning_effort,
                preserve_thinking: req.preserve_thinking,
                tools: &req.tools,
                date_string: self.campaign_template_date,
            },
        )?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        Ok((prompt, prompt_ids))
    }

    /// SC-20671 multi-turn prompt-cache fixture, turn 1: the request runs on the product
    /// prompt-cache path into a fresh `store` (always a miss) and its `prompt + answer` K/V is
    /// stored there for turn 2. Turn 1 is dense in every arm: the provider prefix store holds
    /// shared prefixes as dense K/V, and only a compressed cache hit imports one. Returns the
    /// answer, the turn's cache record, and its rendered prompt ids.
    fn campaign_turn_one(
        &self,
        turn1: &TextLlmRequest,
        store: &mut crate::decode::PrefixCache,
    ) -> CoreResult<(TextLlmOutput, crate::campaign::PromptCacheTurn, Vec<i32>)> {
        let (_, ids) = self.render_prompt(turn1, &turn1.messages)?;
        let mut unobserved = CacheLifecycleCapture::default();
        let before = store.stats();
        let output = self.generate_inner(
            turn1,
            &mut |_| {},
            Some(&mut unobserved),
            None,
            Some(&mut *store),
        )?;
        let turn = prompt_cache_turn(before, store, &ids)?;
        if turn.cache_hit || store.len() != 1 {
            return Err(CoreError::Load(
                "multi-turn prompt cache: turn 1 must miss a fresh store and store its K/V".into(),
            ));
        }
        Ok((output, turn, ids))
    }

    /// Turn 2 of the multi-turn fixture: turn 1's conversation, its answer, then `follow_up`.
    fn campaign_turn_two_request(
        turn1: &TextLlmRequest,
        answer: &TextLlmOutput,
        follow_up: &str,
    ) -> TextLlmRequest {
        let mut turn2 = turn1.clone();
        turn2
            .messages
            .push(Message::text(Role::Assistant, answer.text.clone()));
        turn2.messages.push(Message::text(Role::User, follow_up));
        turn2
    }

    /// Turn 2 must be served by a prompt-cache hit reusing at least the prompt it shares with
    /// turn 1 (fail closed otherwise: the fixture would measure a cold prefill).
    fn require_turn_two_hit(
        turn: &crate::campaign::PromptCacheTurn,
        turn1_ids: &[i32],
        turn2_ids: &[i32],
    ) -> CoreResult<()> {
        let shared = turn1_ids
            .iter()
            .zip(turn2_ids)
            .take_while(|(left, right)| left == right)
            .count()
            .min(turn2_ids.len().saturating_sub(1)) as u64;
        if !turn.cache_hit || turn.reused_prefix_tokens == 0 || turn.reused_prefix_tokens < shared {
            return Err(CoreError::Load(format!(
                "multi-turn prompt cache: turn 2 was not served by a prompt-cache hit over its \
                 shared prefix (hit={}, reused={}, shared={shared})",
                turn.cache_hit, turn.reused_prefix_tokens
            )));
        }
        Ok(())
    }

    /// SC-20671 multi-turn prompt-cache fixture on the product prompt-cache path. Turn 1 runs the
    /// request and stores its K/V ([`Self::campaign_turn_one`]); turn 2 appends the answer and
    /// `follow_up` and is served by the cache hit — a dense seed, or, with `packed`, the prefix
    /// imported into the compressed cache — with `observer` attached. Turn 2's output and both
    /// turns' cache records are returned; a turn 2 that missed is refused.
    pub(crate) fn campaign_multi_turn_observed(
        &self,
        turn1: &TextLlmRequest,
        follow_up: &str,
        on_event: &mut dyn FnMut(CoreEvent),
        observer: &mut dyn crate::campaign::Observer,
        packed: Option<&crate::campaign::CompressedKvArm>,
    ) -> CoreResult<(TextLlmOutput, crate::campaign::MultiTurnPromptCacheTurns)> {
        let mut store = crate::decode::PrefixCache::with_budget(u64::MAX);
        let (answer, turn1_cache, turn1_ids) = self.campaign_turn_one(turn1, &mut store)?;
        let turn2 = Self::campaign_turn_two_request(turn1, &answer, follow_up);
        let (_, turn2_ids) = self.render_prompt(&turn2, &turn2.messages)?;
        let before = store.stats();
        let output =
            self.generate_inner(&turn2, on_event, Some(observer), packed, Some(&mut store))?;
        let turn2_cache = prompt_cache_turn(before, &store, &turn2_ids)?;
        Self::require_turn_two_hit(&turn2_cache, &turn1_ids, &turn2_ids)?;
        drop(store);
        mlx_rs::memory::clear_cache();
        Ok((
            output,
            crate::campaign::MultiTurnPromptCacheTurns {
                turn1: turn1_cache,
                turn2: turn2_cache,
            },
        ))
    }

    /// The multi-turn fixture's turn-2 forced continuation (SC-20671 contract v4): turn 1 exactly
    /// as [`Self::campaign_multi_turn_observed`] runs it, then turn 2's prompt served by the same
    /// prompt-cache hit (dense seed, or with `packed` the compressed import) and greedily decoded
    /// for `min(tokens, context window - turn 2 prompt)` ids through every stop token. With
    /// `teacher_forced`, the decode is forced on that stream and the session's argmax at every
    /// position is returned. A compressed pass must run wholly on the fused reader.
    pub(crate) fn campaign_multi_turn_forced_continuation(
        &self,
        turn1: &TextLlmRequest,
        follow_up: &str,
        tokens: usize,
        packed: Option<&crate::campaign::CompressedKvArm>,
        teacher_forced: Option<&[i32]>,
    ) -> CoreResult<(Vec<i32>, crate::campaign::MultiTurnPromptCacheTurns)> {
        self.campaign_multi_turn_continuation(
            turn1,
            follow_up,
            tokens,
            packed,
            teacher_forced,
            false,
        )
        .map(|(scored, turns)| (scored.choices, turns))
    }

    /// [`Self::campaign_multi_turn_forced_continuation`], also scoring the session's probability
    /// of every turn-2 stream token (the SC-20669 dense multi-turn noise-floor control).
    pub(crate) fn campaign_multi_turn_scored_continuation(
        &self,
        turn1: &TextLlmRequest,
        follow_up: &str,
        tokens: usize,
        teacher_forced: Option<&[i32]>,
    ) -> CoreResult<(
        crate::campaign::ScoredContinuation,
        crate::campaign::MultiTurnPromptCacheTurns,
    )> {
        self.campaign_multi_turn_continuation(turn1, follow_up, tokens, None, teacher_forced, true)
    }

    fn campaign_multi_turn_continuation(
        &self,
        turn1: &TextLlmRequest,
        follow_up: &str,
        tokens: usize,
        packed: Option<&crate::campaign::CompressedKvArm>,
        teacher_forced: Option<&[i32]>,
        score: bool,
    ) -> CoreResult<(
        crate::campaign::ScoredContinuation,
        crate::campaign::MultiTurnPromptCacheTurns,
    )> {
        let Decoder::Causal(model) = &self.model else {
            return Err(CoreError::Unsupported(
                "the multi-turn prompt-cache fixture requires the causal decoder's prefix cache"
                    .into(),
            ));
        };
        let mut store = crate::decode::PrefixCache::with_budget(u64::MAX);
        let (answer, turn1_cache, turn1_ids) = self.campaign_turn_one(turn1, &mut store)?;
        let turn2 = Self::campaign_turn_two_request(turn1, &answer, follow_up);
        let (_, turn2_ids) = self.render_prompt(&turn2, &turn2.messages)?;
        let window = usize::try_from(self.campaign_context_window()?)
            .map_err(|_| CoreError::Load("context window overflows usize".into()))?;
        // The fixture is sized so turn 2 leaves the full continuation; never shorten it.
        if teacher_forced.is_none() && window.saturating_sub(turn2_ids.len()) < tokens {
            return Err(CoreError::InvalidRequest(format!(
                "multi-turn turn 2 ({} tokens) leaves fewer than the {tokens}-token forced \
                 continuation in the {window}-token window",
                turn2_ids.len()
            )));
        }
        let length = teacher_forced.map_or(tokens, <[i32]>::len);
        let budget = u32::try_from(length)
            .map_err(|_| CoreError::InvalidRequest("forced continuation overflows".into()))?;
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            turn2_ids.len(),
            budget,
        )?;
        let before = store.stats();
        let measured = crate::decode::prefix::forced_cached_decode(
            model,
            &turn2_ids,
            &mut store,
            packed,
            length,
            &self.stop_tokens,
            teacher_forced,
            score,
        );
        let turn2_cache = prompt_cache_turn(before, &store, &turn2_ids);
        drop(store);
        mlx_rs::memory::clear_cache();
        let measured = measured.map_err(to_core)?;
        let turn2_cache = turn2_cache?;
        Self::require_turn_two_hit(&turn2_cache, &turn1_ids, &turn2_ids)?;
        if packed.is_some() {
            let fused = measured.fallbacks.is_empty()
                && measured.packed_evidence.as_ref().is_some_and(|evidence| {
                    evidence.accepted_direct_calls > 0
                        && evidence.fallback_reasons.is_empty()
                        && !evidence.dense_active
                        && evidence.full_cache_dequantizations == 0
                        && evidence.failed_dispatches == 0
                });
            if !fused {
                return Err(CoreError::Load(
                    "compressed multi-turn forced continuation did not run wholly on the fused \
                     compressed reader"
                        .into(),
                ));
            }
        }
        Ok((
            crate::campaign::ScoredContinuation {
                choices: measured.decode.tokens,
                stream_probabilities: measured.decode.stream_probabilities,
            },
            crate::campaign::MultiTurnPromptCacheTurns {
                turn1: turn1_cache,
                turn2: turn2_cache,
            },
        ))
    }

    /// Decide the KV cache of one product generation with the shared [`core_llm::plan_kv_cache`]
    /// (sc-20683): the qualification table (the request's opt-in, the `batch`, this model's table
    /// family, the `context_tokens` prefilled before decode and the final context after up to
    /// `max_new_tokens` more), then the request shape, then the shared geometry stage over the
    /// decoder's attention geometry, then this backend's reader stage — the retained fused reader.
    /// Every refusal is a dense plan with its reason.
    fn plan_kv_cache(
        &self,
        policy: core_llm::KvCompressionPolicy,
        context_tokens: usize,
        max_new_tokens: u32,
        batch: u64,
        multimodal: bool,
    ) -> KvPlan {
        use core_llm::KvCacheFallbackReason as Reason;
        let unsupported_request = if multimodal {
            Some("multimodal prefill splices embeddings outside the compressed cache".to_string())
        } else if matches!(self.model, Decoder::Qwen35(_)) {
            Some("the hybrid recurrent decoder has no compressed cache".to_string())
        } else {
            None
        };
        let request = core_llm::KvCacheRequest {
            policy,
            family: self.kv_family,
            context_tokens: u64::try_from(context_tokens).unwrap_or(u64::MAX),
            max_new_tokens: u64::from(max_new_tokens),
            batch,
            unsupported_request,
            // The hybrid decoder is refused by the request-shape stage before geometry is read.
            geometry: match &self.model {
                Decoder::Causal(model) => crate::kv_policy::attention_geometry(model.config()),
                _ => core_llm::KvAttentionGeometry::default(),
            },
        };
        let plan = core_llm::plan_kv_cache(request, |row| {
            let bits = crate::kv_policy::packed_code_bits(row.format);
            match self
                .kv_reader
                .get_or_init(|| crate::kv_policy::group_affine_reader(bits))
            {
                Ok(reader) if reader.code_bits() == bits => Ok(reader.clone()),
                Ok(reader) => Err((
                    Reason::ReaderUnavailable,
                    format!(
                        "the retained reader reads {}-bit codes, not {}",
                        reader.code_bits().bits(),
                        row.format.id()
                    ),
                )),
                Err(error) => Err((Reason::ReaderUnavailable, error.clone())),
            }
        });
        match plan {
            core_llm::KvCachePlan::Compressed {
                qualification,
                reader,
            } => KvPlan::Compressed {
                format: qualification.format,
                reader,
            },
            core_llm::KvCachePlan::Dense(report) => KvPlan::Dense(report),
        }
    }

    /// Native-memory estimate request admission prices (sc-20682): the K/V term at the
    /// compressed `format`'s bytes ([`crate::kv_policy::compressed_request_kv_bytes`]) when the
    /// request's plan runs compressed, dense otherwise.
    #[cfg(test)]
    fn admission_estimate(
        &self,
        format: Option<core_llm::KvCompressionFormat>,
        prompt_tokens: usize,
        max_new_tokens: u32,
        vision_workspace: u64,
        mtp_width: u32,
    ) -> CoreResult<u64> {
        // One pricing for every request (sc-24446 merge): the plain route, or the MTP head's verify
        // width, through the speculative estimate.
        let route = match usize::try_from(mtp_width).unwrap_or(usize::MAX) {
            0 => SpeculativeRoute::Plain,
            width => SpeculativeRoute::Mtp { width },
        };
        let estimate = self.speculative_request_bytes(
            route,
            format,
            prompt_tokens,
            max_new_tokens,
            vision_workspace,
        );
        estimate.ok_or_else(|| CoreError::InvalidRequest("request memory estimate overflow".into()))
    }

    fn generate_inner(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
        observer: Option<&mut dyn crate::campaign::Observer>,
        packed: Option<&crate::campaign::CompressedKvArm>,
        prefix_cache: Option<&mut crate::decode::PrefixCache>,
    ) -> CoreResult<TextLlmOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled); // typed pre-inference cancel
        }
        // The fused-primitive routes the whole request builds — the vision tower and a prefill
        // run outside the engine included — are what its report names (E3).
        let fused_start = crate::primitives::fused::fused_tally();

        // Multimodal (Qwen-VL + image/video content): replace image blocks with the Qwen-VL image
        // placeholder (`<|vision_start|><|image_pad|><|vision_end|>`) and video blocks with the
        // per-frame Text–Timestamp-Alignment placeholder string
        // (`<{t} seconds><|vision_start|><|video_pad|><|vision_end|>` ×frames), so the (vision-free)
        // chat template renders the framing verbatim. The visuals are encoded + spliced after
        // tokenizing, in document order. Text-only requests are unchanged.
        let temporal_patch = self
            .vision
            .as_ref()
            .map(|v| v.processor.temporal_patch_size)
            .unwrap_or(2);
        let (images, videos): (Vec<&ImageRef>, Vec<&VideoRef>) = match &self.vision {
            Some(_) => (collect_images(&req.messages), collect_videos(&req.messages)),
            None => (Vec::new(), Vec::new()),
        };
        let multimodal = !images.is_empty() || !videos.is_empty();
        // Gemma 4 multimodal (sc-18772): its own marker substitution and prefill, taken whenever
        // this is a Gemma 4 checkpoint with front-ends AND the request actually carries a visual or
        // a clip. `validate` has already rejected a modality this checkpoint does not ship.
        let gemma4_mm_request = self.gemma4.is_some()
            && (!collect_images(&req.messages).is_empty()
                || !collect_audio(&req.messages).is_empty());
        // The compressed arm is wired only through the observed text decode; any other route would
        // silently run dense under a compressed label.
        if packed.is_some() && (observer.is_none() || multimodal || gemma4_mm_request) {
            return Err(CoreError::Unsupported(
                "compressed campaign decode supports only observed text generation".into(),
            ));
        }
        if prefix_cache.is_some() && (observer.is_none() || multimodal || gemma4_mm_request) {
            return Err(CoreError::Unsupported(
                "campaign prompt-cache decode supports only observed text generation".into(),
            ));
        }
        let substituted;
        let messages: &[Message] = if gemma4_mm_request {
            substituted = substitute_gemma4_placeholders(&req.messages)?;
            &substituted
        } else if multimodal {
            substituted = substitute_vision_placeholders(&req.messages, temporal_patch)?;
            &substituted
        } else {
            &req.messages
        };

        let (prompt, prompt_ids) = self.render_prompt(req, messages)?;

        // MLX uses unified memory. Admit from a fresh host availability snapshot before visual
        // preprocessing or model execution, using pure request geometry for visual expansion.
        let (visual_tokens, vision_workspace) = match &self.vision {
            Some(vision) if multimodal => vision.estimate_workspace(&images, &videos)?,
            _ => (0, 0),
        };
        let vision_workspace = vision_workspace
            .checked_add(request_media_workspace_bytes(&req.messages)?)
            .ok_or_else(|| CoreError::InvalidRequest("media workspace overflow".into()))?;
        let replaced_visual_placeholders =
            self.vision
                .as_ref()
                .filter(|_| multimodal)
                .map_or(0, |vision| {
                    prompt_ids
                        .iter()
                        .filter(|&&id| id == vision.image_token_id || id == vision.video_token_id)
                        .count()
                });
        let admitted_prompt = prompt_ids
            .len()
            .checked_sub(replaced_visual_placeholders)
            .and_then(|n| n.checked_add(visual_tokens))
            .ok_or_else(|| CoreError::InvalidRequest("expanded prompt geometry overflow".into()))?;
        // The tokenized prompt plus pure visual expansion geometry is known before any native
        // preprocessing. Reject architectural overflow before consulting transient capacity so a
        // valid context error cannot be masked by the machine's current memory pressure.
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            admitted_prompt,
            req.max_new_tokens,
        )?;
        // Pure validation/admission geometry above has scheduled no native work.
        // On a stream change, finish the previous execution BEFORE cache mutation
        // or new preprocessing; unchanged-stream generation adds no barrier.
        self.note_execution_stream().map_err(to_core)?;
        // The backend-neutral speculative resolution (core-llm, sc-24433) against what this
        // provider advertises, then the proposer this request's shape can run on the engine
        // (sc-24434). Anything less than the request asked for is named in the report's
        // `fallbacks` (epic sc-24432 E2) — never a silent downgrade, never a failure. A request
        // that leaves the option unset runs the MLX defaults-table default (E5, sc-24446).
        // The engine also gets the option itself: under `auto` it monitors the proposer it runs
        // against the plain loop it would fall back to, and demotes one measurably slower than
        // plain decoding (sc-24446, E5); an explicit `{proposer, depth}` runs as asked.
        let speculative_mode = req.speculative_or(self.speculative_default);
        let resolution =
            core_llm::resolve_speculative(speculative_mode, &self.descriptor.capabilities);
        let mut fallbacks: Vec<String> = resolution.fallback.into_iter().collect();
        let route = self.speculative_route(resolution.plan, gemma4_mm_request, &mut fallbacks);
        // A `draft_model` request reaching past the draft's own context window runs `auto`
        // instead, by name (sc-24436, E2).
        let route = self.fit_draft_route(
            route,
            admitted_prompt,
            req.max_new_tokens,
            gemma4_mm_request,
            &mut fallbacks,
        );

        // sc-20679: the request's KV cache, decided against the qualification table before any
        // K/V mutation — and, since sc-20682, before admission, so the estimate prices the cache
        // the request will actually run on. Only the plain text decode below runs a compressed
        // plan; its prompt is the tokenized prompt (no visual expansion on that path). The
        // compressed cache runs every engine route the causal decoder has — the pipelined loop,
        // prompt lookup and a draft model — rolling a verify back exactly (sc-20681).
        // A campaign receipt request (an observer attached) measures the plain loop; its output
        // reports no decode path and no KV plan of its own.
        let observed = observer.is_some();
        let kv_plan = if observed {
            KvPlan::Unreported
        } else {
            self.plan_kv_cache(
                req.kv_compression,
                admitted_prompt,
                req.max_new_tokens,
                1,
                multimodal || gemma4_mm_request,
            )
        };
        let compressed_format = match &kv_plan {
            KvPlan::Compressed { format, .. } => Some(*format),
            KvPlan::Unreported | KvPlan::Dense(_) => None,
        };

        // The cross-turn prefix cache (sc-24437) keys on token ids, which cannot tell two images
        // (or clips) behind the same placeholder ids apart — a multimodal prompt neither reads nor
        // feeds it, and the report says so. The hybrid decoder snapshots at the end of the
        // rendered conversation, the prefix the next chat turn extends.
        let (mut prefix_path, mut prefix_reason) = prefix_path_for(
            self.prefix.borrow().budget_bytes() > 0,
            multimodal,
            gemma4_mm_request,
        );
        // Only a request the cache serves reads (restores) or feeds (stores) it — `off` and
        // `bypassed` never touch it. A compressed request reads it — a dense hit is imported into
        // its compressed cache — but never feeds it: the store holds dense K/V, and a compressed
        // cache never re-enters it (sc-20681), so it takes no snapshot.
        let prefix_route = prefix_path == "miss";
        let prefix_store = prefix_route && compressed_format.is_none();
        let prefix_boundary = match &self.model {
            Decoder::Qwen35(_) if prefix_store => {
                self.conversation_boundary(messages, req, &prompt_ids)
            }
            _ => None,
        };
        let snapshot_bytes = if prefix_store {
            self.prefix_snapshot_bytes(prefix_boundary, route.is_mtp())
                .ok_or_else(|| {
                    CoreError::InvalidRequest("prefix snapshot estimate overflow".into())
                })?
        } else {
            0
        };
        // The request's native-memory estimate: the K/V term at the compressed format's bytes
        // when the plan runs compressed (sc-20682), dense otherwise — plus what its speculative
        // route holds beside it (verify overshoot, a draft model's own request, sc-24436/E7).
        let estimate = |format| {
            self.speculative_request_bytes(
                route,
                format,
                admitted_prompt,
                req.max_new_tokens,
                vision_workspace,
            )
            .ok_or_else(|| CoreError::InvalidRequest("request memory estimate overflow".into()))
        };
        let required = estimate(compressed_format)?;
        // What a compressed generation that turns dense part-way grows to (its transition is
        // admitted against this, sc-20682).
        let dense_final_kv = u64::try_from(admitted_prompt)
            .ok()
            .and_then(|prompt| prompt.checked_add(u64::from(req.max_new_tokens)))
            .and_then(|total| {
                crate::kv_policy::dense_final_kv_bytes(
                    self.model.memory_geometry().kv_shape(),
                    total,
                )
            });
        // A compressed request looks its dense prefix up before admission, so the reclaim below
        // never evicts the entry it is about to import (the restored cache shares the entry's
        // buffers: it holds them, no copy).
        let mut compressed_prefix = if prefix_route && compressed_format.is_some() {
            self.prefix
                .borrow_mut()
                .restore::<ContiguousKvCache>(&prompt_ids, false)
                .map_err(to_core)?
        } else {
            None
        };
        let keeps_hit = compressed_prefix.is_some();
        let reclaim = |required: u64, available: u64| {
            let mut prefix = self.prefix.borrow_mut();
            if keeps_hit {
                prefix.reclaim_for_keeping_latest(required, available)
            } else {
                prefix.reclaim_for(required, available)
            }
        };
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            core_llm::operational_memory_override()?,
        )?;
        // Held prefix-cache entries are reclaimable memory (sc-24437, E7): a request that would
        // not fit evicts them least-recently-used first, and the snapshot it would leave behind
        // is admitted with it — or not taken, so caching never makes a request fail and never
        // takes memory admission did not grant.
        let prefix_admission = if prefix_store {
            self.prefix
                .borrow_mut()
                .admit(required, snapshot_bytes, available)
        } else {
            core_llm::PrefixAdmission {
                available: reclaim(required, available),
                snapshot: false,
            }
        };
        let keep_prefix = prefix_admission.snapshot;
        if prefix_store && !keep_prefix {
            prefix_reason = Some(core_llm::PREFIX_NOT_ADMITTED);
        }
        if prefix_route && compressed_format.is_some() {
            prefix_reason = Some(core_llm::PREFIX_COMPRESSED_NOT_STORED);
        }
        core_llm::admit_request_memory_with_geometry(
            admitted_prompt,
            req.max_new_tokens,
            self.descriptor.capabilities.max_context_tokens,
            if keep_prefix {
                required.saturating_add(snapshot_bytes)
            } else {
                required
            },
            prefix_admission.available,
        )?;
        let prefix_boundary = prefix_boundary.filter(|_| keep_prefix);
        // A compressed selection the model refuses runs dense from the start: the request was
        // admitted at the compressed price, so it is admitted again at the dense price before any
        // K/V exists (sc-20682), held prefix entries reclaimable as above.
        let admit_dense = || -> CoreResult<()> {
            let required = estimate(None)?;
            let available = core_llm::effective_memory_budget(
                core_llm::available_host_memory_bytes(),
                core_llm::operational_memory_override()?,
            )?;
            let available = reclaim(required, available);
            core_llm::admit_request_memory_with_geometry(
                admitted_prompt,
                req.max_new_tokens,
                self.descriptor.capabilities.max_context_tokens,
                required,
                available,
            )
        };

        // Encode + splice the visuals and compute M-RoPE positions (the placeholder-expanded prompt
        // becomes the effective sequence). `None` on the text-only path.
        let qwen_prefill_started = (multimodal && !gemma4_mm_request).then(Instant::now);
        let mm = if multimodal && !gemma4_mm_request {
            Some(self.prepare_multimodal(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let g4 = if gemma4_mm_request {
            Some(self.prepare_gemma4(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let prompt_len = mm
            .as_ref()
            .map(|m| m.expanded_ids.len())
            .or_else(|| g4.as_ref().map(|m| m.expanded_ids.len()))
            .unwrap_or(prompt_ids.len());
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            prompt_len,
            req.max_new_tokens,
        )?;
        // A Gemma 4 prompt's soft-token expansion is known only now: a `draft_model` route checks
        // the draft's context window again against the effective prompt (E2). Nothing has been
        // allocated for the route yet; the priced draft is only an over-estimate.
        let route = self.fit_draft_route(
            route,
            prompt_len,
            req.max_new_tokens,
            gemma4_mm_request,
            &mut fallbacks,
        );

        let mut kv_report = match &kv_plan {
            KvPlan::Dense(report) => Some(report.clone()),
            KvPlan::Unreported | KvPlan::Compressed { .. } => None,
        };

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: SamplingParams::from(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };

        let mut prefix_hit = 0usize;

        // Structured-output constraint (story 7166): build a JSON mask over the cached decode table.
        let constraint_starts_in_reasoning =
            self.descriptor.capabilities.supports_thinking && prompt_opens_thinking(&prompt);
        let mut json_mask = match req.constraint {
            Some(Constraint::Json) => {
                let table = self
                    .constraint_table
                    .get_or_init(|| self.tokenizer.constraint_decode_table());
                Some(JsonMask::new(
                    table,
                    self.stop_tokens.iter().map(|&i| i as u32),
                    constraint_starts_in_reasoning,
                ))
            }
            None => None,
        };

        // Request `stop` strings (story 7349): a backend-neutral matcher over the decoded text.
        // Matching is in the detokenization seam below (not the token-id loop) because a stop string
        // need not align to a token boundary. When no stops are requested the matcher is a
        // transparent pass-through, so streaming output stays byte-identical to before.
        let stop_matcher = StopMatcher::new(req.stop.iter().cloned());
        let stop_active = !stop_matcher.is_empty();
        // A single-threaded latch the detok sink trips on a stop hit; the decode loop reads it after
        // each token via `should_stop` and halts with `FinishReason::Stopped`.
        let halt = std::cell::Cell::new(false);

        // A reasoning segmenter when the model advertises a thinking mode: it splits the decoded
        // stream into `<think>…</think>` reasoning vs answer (markers stripped). `None` otherwise, so
        // a non-thinking provider stays on the original single-channel path.
        let thinking_active = self.descriptor.capabilities.supports_thinking;
        let mut segmenter = thinking_active.then(ThinkingSegmenter::default);
        // Some chat templates open the reasoning block *in the prompt* — e.g. Qwen3.6 ends the
        // generation prompt with `<|im_start|>assistant\n<think>\n`, so the model generates inside
        // the block and only emits the closing `</think>`. Prime the segmenter into the Thinking
        // channel by feeding it that already-rendered opening marker (stripped, so it emits nothing);
        // otherwise the reasoning would be misclassified as answer. Disabled mode renders a *closed*
        // `<think>\n\n</think>`, so this correctly does not prime.
        if let Some(seg) = segmenter.as_mut() {
            if constraint_starts_in_reasoning {
                let _ = seg.push("<think>");
                debug_assert!(seg.in_thinking());
            }
        }

        // A tool-call segmenter when the request offers tools and the model's template renders them:
        // it lifts `<tool_call>` blocks out of the answer channel (markup excluded from the streamed
        // text) and parses them into structured calls (sc-7636). `None` otherwise, so a no-tools
        // request flows straight through `tool_pieces` unchanged.
        let tools_active = self.tool_call_format.is_some() && !req.tools.is_empty();
        let tool_seg = matches!(self.tool_call_format, Some(ToolCallFormat::Tagged))
            .then(|| ToolCallSegmenter::new(&req.tools));

        // Whether the stream has delivered a token yet. The segmenter, the tool segmenter and the
        // stop matcher can hold the first engine tokens back (a thinking model's opening marker
        // emits nothing), so the engine keeps its look-ahead out of the way until the first
        // delivered token is out — the time to first token never waits behind one (sc-24446).
        let delivered_any = std::cell::Cell::new(false);
        let mut tracked = |event: CoreEvent| {
            if matches!(event, CoreEvent::Token { .. }) {
                delivered_any.set(true);
            }
            on_event(event);
        };
        let on_event: &mut dyn FnMut(CoreEvent) = &mut tracked;
        let delivered = || delivered_any.get();

        // Drive the engine; translate token-id events to contract text-delta events through the
        // token-to-text pipeline (see `TextPipeline`): the streamed answer is the result text when
        // stop strings, thinking or tools are active.
        let tokenizer = &self.tokenizer;
        let mut pipeline = TextPipeline::new(tokenizer, segmenter, tool_seg, stop_matcher);
        let mut run = {
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, .. } = ev {
                    pipeline.push(id as u32, &halt, &mut *on_event);
                }
            };
            let should_stop = || halt.get();
            let should_stop_opt = stop_active.then_some(&should_stop as &dyn Fn() -> bool);
            let width = route.width();
            match (&mm, &g4) {
                // Qwen-VL multimodal: prefill the spliced embeds with interleaved M-RoPE +
                // DeepStack fusion, then decode the continuation (RoPE shifted by `mrope_delta`)
                // through the engine — the MTP head seeds from the fused embeddings.
                (Some(m), _) => {
                    let (t, h, w, delta) = &m.positions;
                    let pos = [t.as_slice(), h.as_slice(), w.as_slice()];
                    let clock = Some(
                        qwen_prefill_started
                            .expect("Qwen multimodal preparation starts the prefill clock"),
                    );
                    match &self.model {
                        Decoder::Qwen35(model) => {
                            let model: &Qwen35Model = model;
                            let mtp_prompt = Qwen35MtpMultimodalPrompt {
                                input_ids: &m.expanded_ids,
                                embeddings: &m.embeds,
                                positions: pos,
                                visual_pos_mask: &m.visual_pos_mask,
                                deepstack: &m.deepstack,
                                continuation_delta: *delta,
                            };
                            let mut cache = model.new_cache();
                            let (mut proposer, logits, hidden) = if route.is_mtp() {
                                let (hidden, logits) = model
                                    .prefill_hidden_and_last_logits_from_embeds_with_deepstack(
                                        &m.embeds,
                                        pos,
                                        &mut cache,
                                        &m.visual_pos_mask,
                                        &m.deepstack,
                                    )
                                    .map_err(to_core)?;
                                let proposer: Box<dyn Proposer<Qwen35Model>> =
                                    Box::new(MtpProposer::multimodal(&mtp_prompt));
                                (proposer, logits, Some(hidden))
                            } else {
                                let logits = model
                                    .prefill_with_deepstack(
                                        &m.embeds,
                                        pos,
                                        &mut cache,
                                        &m.visual_pos_mask,
                                        &m.deepstack,
                                    )
                                    .map_err(to_core)?;
                                (route.plain_proposer(self.draft.as_ref()), logits, None)
                            };
                            generate_speculative(
                                model,
                                &mut *proposer,
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden,
                                    history: &m.expanded_ids,
                                    position_delta: *delta,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                EngineOptions {
                                    constraint: json_mask
                                        .as_mut()
                                        .map(|m| m as &mut dyn RewindableConstraintMask),
                                    should_stop: should_stop_opt,
                                    delivered: Some(&delivered),
                                    prefill_clock: clock,
                                    speculative_mode,
                                    ..EngineOptions::default()
                                },
                            )
                        }
                        Decoder::Causal(model) => {
                            let model: &CausalLm = model;
                            let mut cache = model.new_cache();
                            let logits = model
                                .prefill_with_deepstack(
                                    &m.embeds,
                                    pos,
                                    &mut cache,
                                    &m.visual_pos_mask,
                                    &m.deepstack,
                                )
                                .map_err(to_core)?;
                            generate_speculative(
                                model,
                                &mut *route.plain_proposer(self.draft.as_ref()),
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden: None,
                                    history: &m.expanded_ids,
                                    position_delta: *delta,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                EngineOptions {
                                    constraint: json_mask
                                        .as_mut()
                                        .map(|m| m as &mut dyn RewindableConstraintMask),
                                    should_stop: should_stop_opt,
                                    delivered: Some(&delivered),
                                    prefill_clock: clock,
                                    speculative_mode,
                                    ..EngineOptions::default()
                                },
                            )
                        }
                    }
                }
                // Gemma 4 multimodal: prefill the spliced embeds on ordinary causal 1-D positions
                // (no M-RoPE, so no position shift for the continuation), then decode through the
                // engine against the unwrapped decoder.
                (None, Some(m)) => {
                    let model: &CausalLm =
                        match &self.model {
                            Decoder::Causal(c) => c,
                            Decoder::Qwen35(_) => return Err(CoreError::Load(
                                "gemma 4: the multimodal path requires the generic causal decoder"
                                    .into(),
                            )),
                        };
                    let mut cache = model.new_cache();
                    let logits = model
                        .decode_logits_from_embeds(&m.embeds, &mut cache, 0)
                        .map_err(to_core)?;
                    generate_speculative(
                        model,
                        &mut *route.plain_proposer(self.draft.as_ref()),
                        SpeculativePrompt::Prefilled {
                            cache: &mut cache,
                            logits,
                            hidden: None,
                            history: &m.expanded_ids,
                            position_delta: 0,
                        },
                        &config,
                        width,
                        &req.cancel,
                        &mut sink,
                        EngineOptions {
                            constraint: json_mask
                                .as_mut()
                                .map(|m| m as &mut dyn RewindableConstraintMask),
                            should_stop: should_stop_opt,
                            delivered: Some(&delivered),
                            prefill_clock: None,
                            speculative_mode,
                            ..EngineOptions::default()
                        },
                    )
                }
                // The campaign receipt producer (epic sc-20669): the observed plain decode — the
                // measurement's own loop, on a dense or compressed (`packed`) cache, through the
                // campaign prompt-cache store when it passes one. Never installed on a production
                // request, and never run under a speculative route.
                (None, None) if observed => {
                    let observer = observer.expect("matched Some");
                    if !matches!(route, SpeculativeRoute::Plain) {
                        return Err(CoreError::Unsupported(
                            "campaign observer measures only the plain decode".into(),
                        ));
                    }
                    let packed_decoder = match (packed, &self.model) {
                        (None, _) => None,
                        (Some(arm), Decoder::Causal(model)) => Some(PackedCampaignDecoder {
                            model,
                            arm,
                            selection_fallbacks: RefCell::new(Vec::new()),
                        }),
                        (Some(_), Decoder::Qwen35(_)) => {
                            return Err(CoreError::Unsupported(
                                "compressed campaign decode requires the causal decoder".into(),
                            ))
                        }
                    };
                    let decoder: &dyn Decode = match &packed_decoder {
                        Some(packed) => packed,
                        None => &self.model,
                    };
                    let output = match (prefix_cache, &self.model) {
                        // The product prompt-cache path (SC-20671 multi-turn fixture): a hit
                        // seeds the dense cache or imports into the packed one.
                        (Some(store), Decoder::Causal(model)) => {
                            crate::decode::prefix::generate_cached_with_observer(
                                model,
                                &prompt_ids,
                                &config,
                                &req.cancel,
                                &mut sink,
                                store,
                                json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                                should_stop_opt,
                                Some(&mut *observer),
                                packed,
                            )
                        }
                        (Some(_), Decoder::Qwen35(_)) => {
                            return Err(CoreError::Unsupported(
                                "the campaign prompt cache requires the causal decoder".into(),
                            ))
                        }
                        (None, _) => generate_with_observer(
                            decoder,
                            &prompt_ids,
                            &config,
                            &req.cancel,
                            &mut sink,
                            json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                            should_stop_opt,
                            Some(&mut *observer),
                        ),
                    }
                    .map_err(to_core)?;
                    for reason in packed_decoder
                        .map(|packed| packed.selection_fallbacks.into_inner())
                        .unwrap_or_default()
                    {
                        observer.dense_fallback("cache-selection", &reason);
                    }
                    Ok(SpeculativeRun::unmeasured(output, 0))
                }
                // Text: prefill on top of the longest prefix the cross-turn cache restores
                // (sc-24437), decode through the engine, then keep the request's state for the
                // next turn — a softmax KV cache whole, the hybrid's snapshot at the conversation
                // boundary (with the MTP head's state there when it ran).
                (None, None) => {
                    let options = EngineOptions {
                        constraint: json_mask
                            .as_mut()
                            .map(|m| m as &mut dyn RewindableConstraintMask),
                        should_stop: should_stop_opt,
                        delivered: Some(&delivered),
                        prefill_clock: Some(Instant::now()),
                        speculative_mode,
                        ..EngineOptions::default()
                    };
                    match &self.model {
                        // A compressed plan (sc-20679): the compressed cache is selected before
                        // any K/V mutation and its evidence read before it is reset, and the
                        // engine runs on it like on any cache — the pipelined loop, or a
                        // speculative route whose verify steps it rolls back exactly (sc-20681).
                        Decoder::Causal(model) if compressed_format.is_some() => {
                            let model: &CausalLm = model;
                            let KvPlan::Compressed { format, reader } = &kv_plan else {
                                unreachable!("a compressed format comes from a compressed plan")
                            };
                            let (cache, refused) = crate::kv_policy::select_compressed_cache(
                                model,
                                reader.clone(),
                                prompt_ids.len(),
                                crate::kv_policy::dense_transition_admission(
                                    dense_final_kv.ok_or_else(|| {
                                        CoreError::InvalidRequest(
                                            "request memory estimate overflow".into(),
                                        )
                                    })?,
                                    core_llm::RequestResourceExhausted {
                                        prompt_tokens: admitted_prompt,
                                        max_new_tokens: req.max_new_tokens,
                                        max_context_tokens: self
                                            .descriptor
                                            .capabilities
                                            .max_context_tokens,
                                        required_bytes: 0,
                                        available_bytes: 0,
                                    },
                                ),
                            );
                            if refused.is_some() {
                                admit_dense()?;
                            }
                            let mut cache: Box<dyn KvCache + '_> = cache;
                            // A dense prefix hit (looked up before admission) is imported into the
                            // compressed cache by quantize-on-append — never a reconstruction —
                            // from views of the stored entry's own buffers, no dense copy; a cache
                            // that declines it prefills the whole prompt instead, by name
                            // (sc-20681).
                            let mut reused = 0usize;
                            if let Some(restored) = compressed_prefix.take() {
                                let layers = restored.cache.share_live().map_err(to_core)?;
                                let imported = match layers {
                                    Some(layers) => {
                                        cache.import_prefix(&layers).map_err(to_core)?
                                    }
                                    None => false,
                                };
                                if imported {
                                    reused = restored.reused;
                                } else {
                                    prefix_reason =
                                        Some(core_llm::PREFIX_COMPRESSED_IMPORT_DECLINED);
                                }
                            }
                            if req.cancel.is_cancelled() {
                                return Err(CoreError::Canceled);
                            }
                            let logits = model
                                .decode_logits(
                                    &crate::primitives::input_ids(&prompt_ids[reused..]),
                                    cache.as_mut(),
                                    reused as i32,
                                )
                                .map_err(to_core)?;
                            // Measured, not looked up: the positions the prefill did not feed.
                            prefix_hit = reused;
                            if prefix_hit > 0 {
                                prefix_path = "hit";
                            }
                            let run = generate_speculative(
                                &DynCacheTarget(model),
                                &mut *route.plain_proposer(self.draft.as_ref()),
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden: None,
                                    history: &prompt_ids,
                                    position_delta: 0,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                options,
                            );
                            let report = run.as_ref().ok().map(|_| {
                                crate::kv_policy::compressed_report(
                                    *format,
                                    refused,
                                    cache.as_ref(),
                                )
                            });
                            let reset = cache.reset();
                            let run = run.map_err(to_core)?;
                            reset.map_err(to_core)?;
                            kv_report = report.transpose().map_err(to_core)?;
                            Ok(run)
                        }
                        Decoder::Causal(model) => {
                            let model: &CausalLm = model;
                            let restored = if prefix_route {
                                self.prefix
                                    .borrow_mut()
                                    .restore::<ContiguousKvCache>(&prompt_ids, false)
                                    .map_err(to_core)?
                            } else {
                                None
                            };
                            let PrefixPrefill {
                                mut cache,
                                logits,
                                fed_tokens,
                                ..
                            } = prefill_restored(
                                model,
                                restored,
                                &prompt_ids,
                                None,
                                false,
                                &req.cancel,
                            )
                            .map_err(to_core)?;
                            // Measured, not looked up: the positions the prefill did not feed.
                            prefix_hit = prompt_ids.len() - fed_tokens;
                            if prefix_hit > 0 {
                                prefix_path = "hit";
                            }
                            let run = generate_speculative(
                                model,
                                &mut *route.plain_proposer(self.draft.as_ref()),
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden: None,
                                    history: &prompt_ids,
                                    position_delta: 0,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                options,
                            );
                            if let (Ok(run), true) = (&run, keep_prefix) {
                                self.prefix
                                    .borrow_mut()
                                    .store_run(&prompt_ids, run, cache, None, None)
                                    .map_err(to_core)?;
                            }
                            run
                        }
                        Decoder::Qwen35(model) => {
                            let model: &Qwen35Model = model;
                            let mtp = route.is_mtp();
                            let restored = if prefix_route {
                                self.prefix
                                    .borrow_mut()
                                    .restore::<Qwen35Cache>(&prompt_ids, mtp)
                                    .map_err(to_core)?
                            } else {
                                None
                            };
                            let PrefixPrefill {
                                mut cache,
                                logits,
                                hidden,
                                fed_tokens,
                                mtp: resume,
                                boundary,
                                ..
                            } = prefill_restored(
                                model,
                                restored,
                                &prompt_ids,
                                prefix_boundary,
                                mtp,
                                &req.cancel,
                            )
                            .map_err(to_core)?;
                            // Measured, not looked up: the positions the prefill did not feed.
                            prefix_hit = prompt_ids.len() - fed_tokens;
                            if prefix_hit > 0 {
                                prefix_path = "hit";
                            }
                            let mut head = mtp.then(|| {
                                MtpProposer::new()
                                    .resume_from(resume)
                                    .capture_at(boundary.as_ref().map(Boundary::len))
                            });
                            let mut plain = route.plain_proposer(self.draft.as_ref());
                            let proposer: &mut dyn Proposer<Qwen35Model> = match head.as_mut() {
                                Some(head) => head,
                                None => &mut *plain,
                            };
                            let run = generate_speculative(
                                model,
                                proposer,
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden,
                                    history: &prompt_ids,
                                    position_delta: 0,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                options,
                            );
                            if let (Ok(run), true) = (&run, keep_prefix) {
                                let captured = head.as_mut().and_then(MtpProposer::take_captured);
                                self.prefix
                                    .borrow_mut()
                                    .store_run(&prompt_ids, run, cache, boundary, captured)
                                    .map_err(to_core)?;
                            }
                            run
                        }
                    }
                }
            }
            .map_err(to_core)?
        };
        let timing = run.take_timer();
        let out = run.output;
        let stats = run.stats;
        let mut report = run.report;
        // The request's resolution fallbacks join the engine's measured report (E2/E3), and so
        // does what the prefix cache restored. The engine counts a caller's prefill as its one
        // prefill forward: the boundary snapshot is taken inside it (sc-24446).
        report.fallbacks = fallbacks;
        report.prefix_hit_tokens = prefix_hit as u64;
        report.prefix_cache = core_llm::PathReport {
            path: prefix_path.into(),
            reason: prefix_reason.map(str::to_string),
        };
        report.fused_primitives = crate::primitives::fused::fused_tally()
            .since(&fused_start)
            .path_report();

        // End-of-generation tails (thinking marker, tool block, stop prefix), in pipeline order.
        let (streamed, thinking_buf, tool_seg) =
            pipeline.finish(&halt, stop_active, &mut *on_event);
        // Result text: the streamed answer when stop strings, thinking, or tools are active (any of
        // which means the streamed channel is the authoritative answer with markup removed);
        // otherwise the original decode-all-tokens path (byte-identical to before). Reasoning and
        // tool calls, if the model produced any, are reported separately (markup excluded from text).
        // `streamed` accumulates only `IncrementalDetok`-released deltas, so it carries no
        // transient U+FFFD placeholders; a character truncated by end-of-generation is dropped
        // rather than surfaced as U+FFFD (sc-12452).
        let mut text = if stop_active || thinking_active || tools_active {
            streamed
        } else {
            let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&gen_u32, true)?
        };
        let thinking = (!thinking_buf.is_empty()).then_some(thinking_buf);
        let mut tool_calls = tool_seg.map(|mut ts| ts.take_calls()).unwrap_or_default();
        if matches!(self.tool_call_format, Some(ToolCallFormat::BareJson)) && tools_active {
            let parsed = bare_json_tool_calls(&text, &req.tools);
            if !parsed.is_empty() {
                tool_calls = parsed;
                text.clear();
            }
        }
        let finish = map_finish(out.finish_reason);
        let usage = Usage {
            prompt_tokens: prompt_len as u32,
            generated_tokens: out.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        let timings = timing.map(|timer| timer.finish());
        Ok(TextLlmOutput {
            timings,
            text,
            thinking,
            tool_calls,
            usage,
            mtp: (report.proposer == ProposerKind::Mtp).then(|| core_llm::MtpStats {
                proposed_tokens: u32::try_from(stats.proposed).unwrap_or(u32::MAX),
                accepted_tokens: u32::try_from(stats.accepted).unwrap_or(u32::MAX),
                target_forwards: u32::try_from(stats.forwards).unwrap_or(u32::MAX),
            }),
            decode: (!observed).then_some(report),
            finish_reason: Some(finish),
            kv_cache: kv_report,
        })
    }
}

// Why a multimodal request never reads or feeds the cross-turn prefix cache (sc-24437): the
// shared reason both backends name (E8).
#[cfg(test)]
use core_llm::PREFIX_MULTIMODAL_BYPASS;

/// The cross-turn prefix cache's part in a request before any lookup (sc-24437): `off` when the
/// load settled a zero budget, `bypassed` (with the reason) for a multimodal prompt — Qwen-VL
/// (`multimodal`) or Gemma 4 (`gemma4_mm`) — whose token ids cannot tell two images or clips apart,
/// else `miss` (the lookup may still turn it into a `hit`). Only a `miss` request reads or feeds
/// the cache.
fn prefix_path_for(
    prefix_on: bool,
    multimodal: bool,
    gemma4_mm: bool,
) -> (&'static str, Option<&'static str>) {
    core_llm::prefix_path_before_lookup(prefix_on, multimodal || gemma4_mm)
}

/// The ceiling on any speculative proposal MLX runs (sc-24438, epic sc-24432 E4): drafts per
/// verify step, so the verify forward carries at most `1 + 7 = 8` query rows. A loaded model
/// advertises its own, possibly lower, depth — [`speculative_max_depth`] over its attention
/// geometry — and the weightless [`provider_descriptor`] advertises this ceiling.
///
/// Why 8 rows: the attention primitive's `MLX_SDPA_VECTOR_MAX_QLEN` is the widest query MLX
/// 0.32's single-pass **vector** SDPA kernel takes — its decode kernel, the one every plain
/// `q_len = 1` step runs. A verify it serves in one call costs one fused call per attention layer,
/// no score matrix, the same kernel as the plain step it must agree with token for token. Past it
/// the verify moves to the fused **full** kernel (head dims 64/80/128), whose query blocks are
/// sized for prefill, or — for head dims only the vector kernel serves, like the Qwen35 hybrid's
/// 256 — to a second vector-kernel tile per layer: the per-step verify cost stops being flat in
/// the depth exactly where speculation needs it flat. sc-24442 removed the old 8-row cap on
/// *fused* SDPA; this is the width the decode kernel serves, not that cap. On the Qwen35 hybrid
/// every extra depth also adds a checkpoint-ring slot per linear layer (sc-24435).
pub const SPECULATIVE_MAX_DEPTH: u32 =
    crate::primitives::attention::MLX_SDPA_VECTOR_MAX_QLEN as u32 - 1;

/// The deepest speculative proposal (prompt lookup and MTP alike) a model whose attention layers
/// have these `(q_heads, kv_heads, qk_head_dim, v_head_dim)` geometries runs on MLX (sc-24438,
/// E4): one less than the narrowest `vector_verify_rows` among them, so a max-depth verify is one
/// decode-kernel call in **every** attention layer.
///
/// MLX's vector kernel serves `q_len × gqa <= 32`, so a GQA group wider than 4 narrows it:
/// Qwen3.8-27B (24q/4kv/hd 256, gqa 6) verifies 5 rows in one call → depth 4; Qwen3.5/3.6-35B-A3B
/// (16q/2kv/hd 256, gqa 8) 4 rows → depth 3; a gqa-4 Llama 8 rows → depth 7
/// ([`SPECULATIVE_MAX_DEPTH`]). A geometry the vector kernel does not serve is bounded at 8 rows.
/// Never below 1: a gqa-32 group (one vector row) still speculates one draft, in two calls.
pub fn speculative_max_depth(geometries: impl IntoIterator<Item = (i32, i32, i32, i32)>) -> u32 {
    geometries
        .into_iter()
        .map(|(hq, hkv, qd, vd)| crate::primitives::attention::vector_verify_rows(hq, hkv, qd, vd))
        .min()
        .map_or(SPECULATIVE_MAX_DEPTH, |rows| (rows - 1).max(1) as u32)
}

/// Every attention geometry `(q_heads, kv_heads, qk_head_dim, v_head_dim)` a causal decoder's
/// layers hand `sdpa`: per layer type (Gemma 4's sliding and full layers differ), MLA's
/// head-expanded `qk_nope + qk_rope` / `v_head_dim` keys.
fn causal_attention_geometries(cfg: &ModelConfig) -> Vec<(i32, i32, i32, i32)> {
    if let Some(mla) = &cfg.mla {
        return vec![(
            cfg.num_heads,
            cfg.num_heads,
            mla.q_head_dim(),
            mla.v_head_dim,
        )];
    }
    let mut geometries: Vec<_> = (0..cfg.num_layers.max(1))
        .map(|i| {
            let a = cfg.layer_attention(i);
            (cfg.num_heads, a.num_kv_heads, a.head_dim, a.head_dim)
        })
        .collect();
    geometries.sort_unstable();
    geometries.dedup();
    geometries
}

/// The attention geometry of the Qwen35 hybrid's full-attention layers — the only layers that run
/// `sdpa` (the Gated DeltaNet layers are recurrent) — which its MTP predictor layer shares.
fn qwen35_attention_geometry(cfg: &Qwen35Config) -> (i32, i32, i32, i32) {
    (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.head_dim)
}

/// The per-proposer depths [`Speculative::Auto`](core_llm::Speculative::Auto) runs and every MLX
/// advertisement recommends: the MLX row of the decode-defaults table (E5,
/// [`core_llm::defaults`]).
fn recommended_depths() -> &'static core_llm::RecommendedDepths {
    &core_llm::DecodeBackend::Mlx.defaults().recommended_depths
}

/// The prompt-lookup advertisement every `mlx-llama` decoder carries, at the model's
/// [`speculative_max_depth`] (the shared [`core_llm::prompt_lookup_capabilities`], recommending the
/// MLX row's depth).
fn prompt_lookup_capabilities(max_depth: u32) -> ProposerCapabilities {
    core_llm::prompt_lookup_capabilities(max_depth, recommended_depths())
}

/// The proposer a request runs on the engine, after its resolved plan meets the request's shape
/// (sc-24434, sc-24436). `width` is the drafts per verify step (`0` for plain decoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpeculativeRoute {
    Plain,
    Mtp { width: usize },
    PromptLookup { width: usize },
    DraftModel { width: usize },
}

impl SpeculativeRoute {
    fn width(self) -> usize {
        match self {
            SpeculativeRoute::Plain => 0,
            SpeculativeRoute::Mtp { width }
            | SpeculativeRoute::PromptLookup { width }
            | SpeculativeRoute::DraftModel { width } => width,
        }
    }

    fn is_mtp(self) -> bool {
        matches!(self, SpeculativeRoute::Mtp { .. })
    }

    /// The proposer for any target: prompt lookup, the resident `draft` model, or none. (MTP
    /// needs the Qwen35 target.)
    fn plain_proposer<'a, T: SpeculativeTarget + 'a>(
        self,
        draft: Option<&'a ResidentDraft>,
    ) -> Box<dyn Proposer<T> + 'a> {
        match (self, draft) {
            (SpeculativeRoute::PromptLookup { .. }, _) => Box::new(NgramProposer::default()),
            (SpeculativeRoute::DraftModel { width: drafts }, Some(resident)) => {
                let (proposable, width) = (resident.proposable, resident.width);
                match &resident.model {
                    Decoder::Causal(draft) => Box::new(
                        DraftModelProposer::new(&**draft, drafts).with_vocab(proposable, width),
                    ),
                    Decoder::Qwen35(draft) => Box::new(
                        DraftModelProposer::new(&**draft, drafts).with_vocab(proposable, width),
                    ),
                }
            }
            // `speculative_route` routes `draft_model` only with a resident draft.
            (SpeculativeRoute::DraftModel { .. }, None)
            | (SpeculativeRoute::Plain | SpeculativeRoute::Mtp { .. }, _) => Box::new(NoProposer),
        }
    }
}

impl LlamaProvider {
    /// Where the rendered conversation ends and the generation prompt begins, when that is a
    /// strict token prefix of `prompt_ids` (sc-24437): the point the hybrid decoder's prefix-cache
    /// snapshot is taken at, because the next chat turn re-renders these messages and extends
    /// them (whatever the template later does to this turn's reply). `None` when the template
    /// renders the messages differently without a generation prompt.
    fn conversation_boundary(
        &self,
        messages: &[Message],
        req: &TextLlmRequest,
        prompt_ids: &[i32],
    ) -> Option<usize> {
        let rendered = self
            .template
            .render_with(
                messages,
                &RenderOptions {
                    add_generation_prompt: false,
                    enable_thinking: req.enable_thinking_kwarg(),
                    reasoning_effort: req.reasoning_effort,
                    preserve_thinking: req.preserve_thinking,
                    tools: &req.tools,
                    // The date the request's own render used (the campaign's pinned one, or the
                    // template's default), so the conversation is a prefix of the prompt.
                    date_string: self.campaign_template_date,
                },
            )
            .ok()?;
        let ids = self.tokenizer.encode(&rendered, false).ok()?;
        let len = ids.len();
        (len > 0
            && len < prompt_ids.len()
            && ids.iter().zip(prompt_ids).all(|(&a, &b)| a as i32 == b))
        .then_some(len)
    }

    /// An upper bound on the bytes of the prefix-cache snapshot a text request takes during its
    /// run (sc-24437), which its admission adds: the hybrid's whole cache at the conversation
    /// `boundary` (and the MTP head's state there with `mtp`). A softmax decoder keeps the
    /// request's own cache after the run — memory the request already holds — so it takes none.
    fn prefix_snapshot_bytes(&self, boundary: Option<usize>, mtp: bool) -> Option<u64> {
        match (&self.model, boundary) {
            (Decoder::Qwen35(m), Some(b)) => m.config().prefix_snapshot_bytes(b, mtp),
            _ => Some(0),
        }
    }

    /// The request's admission price under its speculative route. Prompt lookup and a draft
    /// model's verify roll back without a forward or a copy on both decoders — a softmax decoder
    /// by truncation, the Qwen35 hybrid through its DeltaNet checkpoint ring (sc-24435) — so they
    /// pay their verify overshoot
    /// (`width` more KV positions and `width` more hidden + logit rows) and, on the hybrid, the
    /// ring ([`Qwen35Model::checkpoint_ring_bytes`]). MTP keeps the predictor-cache pricing
    /// (`width` in the estimate) plus the ring; the ring replaced the step-start snapshot, so the
    /// hybrid's recurrent state is charged once, never as clone + replay copies.
    fn speculative_request_bytes(
        &self,
        route: SpeculativeRoute,
        compressed: Option<core_llm::KvCompressionFormat>,
        prompt_tokens: usize,
        max_new_tokens: u32,
        vision_workspace: u64,
    ) -> Option<u64> {
        let mut geometry = self.model.memory_geometry();
        if let (Decoder::Qwen35(m), width @ 1..) = (&self.model, route.width()) {
            geometry.recurrent_bytes = geometry
                .recurrent_bytes
                .checked_add(m.checkpoint_ring_bytes(width)?)?;
        }
        let (priced_width, overshoot) = match route {
            SpeculativeRoute::PromptLookup { width } | SpeculativeRoute::DraftModel { width } => {
                let width = u64::try_from(width).ok()?;
                let kv_position = geometry
                    .layers
                    .checked_mul(geometry.kv_heads)?
                    .checked_mul(geometry.head_dim)?
                    .checked_mul(geometry.element_bytes)?
                    .checked_mul(2)?;
                let rows = geometry
                    .hidden_size
                    .checked_add(geometry.vocab_size)?
                    .checked_mul(geometry.element_bytes)?;
                // The verify step writes `width` positions past the committed ones; the caches
                // round *that* up to a block. What the target estimate already priced: its own
                // padding over prompt + generation, plus the `width` rows below.
                let committed = u64::try_from(prompt_tokens)
                    .ok()?
                    .checked_add(u64::from(max_new_tokens))?;
                let block_extra = kv_overshoot_padding(committed, width)?;
                (
                    0,
                    width
                        .checked_mul(kv_position.checked_add(rows)?)?
                        .checked_add(block_extra.checked_mul(kv_position)?)?,
                )
            }
            other => (u32::try_from(other.width()).ok()?, 0),
        };
        // The K/V term at the compressed format's bytes when the request runs compressed
        // (sc-20682), dense otherwise. A verify's overshoot rows stay dense on either cache (the
        // compressed cache keeps a step's rows unquantized until the rollback, sc-20681).
        let target = match compressed {
            None => estimate_mlx_request_bytes(
                prompt_tokens,
                max_new_tokens,
                geometry,
                vision_workspace,
                priced_width,
                self.model.workspace_contract(),
            )?,
            Some(format) => {
                let prompt = u64::try_from(prompt_tokens).ok()?;
                let total = prompt.checked_add(u64::from(max_new_tokens))?;
                let kv = crate::kv_policy::compressed_request_kv_bytes(
                    format, &geometry, prompt, total,
                )?;
                estimate_mlx_compressed_request_bytes(
                    prompt_tokens,
                    max_new_tokens,
                    geometry,
                    vision_workspace,
                    priced_width,
                    self.model.workspace_contract(),
                    kv,
                )?
            }
        }
        .checked_add(overshoot)?
        .checked_add(self.model.promoted_request_bytes()?)?;
        // The draft model's own request (E7, sc-24436): its prefill of the same prompt and a
        // cache that grows over the whole run plus the provisional drafts a step writes past the
        // committed tokens — and, when the draft is the hybrid, the checkpoint ring its proposal
        // window holds (sc-24435), armed for the same `width`: no snapshot, no replay.
        let draft = match (route, self.draft.as_ref().map(|d| &d.model)) {
            (SpeculativeRoute::DraftModel { width }, Some(draft)) => {
                let mut geometry = draft.memory_geometry();
                if let Decoder::Qwen35(m) = draft {
                    geometry.recurrent_bytes = geometry
                        .recurrent_bytes
                        .checked_add(m.checkpoint_ring_bytes(width)?)?;
                }
                estimate_mlx_request_bytes(
                    prompt_tokens,
                    max_new_tokens.checked_add(u32::try_from(width).ok()?)?,
                    geometry,
                    0,
                    0,
                    draft.workspace_contract(),
                )?
                .checked_add(draft.promoted_request_bytes()?)?
            }
            _ => 0,
        };
        target.checked_add(draft)
    }

    /// The resident draft's context window (`0`: no draft, or an unbounded one).
    fn draft_context(&self) -> usize {
        self.draft.as_ref().map_or(0, |draft| draft.context)
    }

    /// A `draft_model` route whose request — `prompt_tokens` + `max_new_tokens` and a step's
    /// overshoot — outruns the resident draft's context window runs what `auto` resolves to
    /// instead, the reason named in `fallbacks` ([`core_llm::fit_draft_context`], sc-24436 E2).
    /// Any other route is returned unchanged.
    fn fit_draft_route(
        &self,
        route: SpeculativeRoute,
        prompt_tokens: usize,
        max_new_tokens: u32,
        gemma4_mm_request: bool,
        fallbacks: &mut Vec<String>,
    ) -> SpeculativeRoute {
        let SpeculativeRoute::DraftModel { width } = route else {
            return route;
        };
        let fit = core_llm::fit_draft_context(
            core_llm::SpeculativeResolution {
                plan: SpeculativePlan::Run {
                    proposer: SpeculativeProposer::DraftModel,
                    depth: width as u32,
                },
                fallback: None,
            },
            &self.descriptor.capabilities,
            self.draft_context(),
            prompt_tokens,
            max_new_tokens,
        );
        match fit.fallback {
            None => route,
            Some(why) => {
                fallbacks.push(why);
                self.speculative_route(fit.plan, gemma4_mm_request, fallbacks)
            }
        }
    }

    /// Route a resolved speculative plan onto what this request can run, naming every downgrade
    /// in `fallbacks` (E2): MTP needs a loaded Qwen3.8 predictor and a text or Qwen-VL prompt, and
    /// `draft_model` a resident draft (sc-24436) — it runs on every prompt shape, the draft
    /// proposing over the effective prompt ids.
    fn speculative_route(
        &self,
        plan: SpeculativePlan,
        gemma4_mm_request: bool,
        fallbacks: &mut Vec<String>,
    ) -> SpeculativeRoute {
        match plan {
            SpeculativePlan::Off => SpeculativeRoute::Plain,
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::Mtp,
                depth,
            } => {
                let head = matches!(&self.model, Decoder::Qwen35(m) if m.has_mtp());
                if head && !gemma4_mm_request {
                    SpeculativeRoute::Mtp {
                        width: depth as usize,
                    }
                } else {
                    fallbacks.push(if head {
                        "speculative: `mtp` has no Qwen predictor for a Gemma 4 multimodal \
                         prompt; decoded without a proposer"
                            .into()
                    } else {
                        "speculative: `mtp` has no loaded MTP predictor on this model; decoded \
                         without a proposer"
                            .into()
                    });
                    SpeculativeRoute::Plain
                }
            }
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::PromptLookup,
                depth,
            } => SpeculativeRoute::PromptLookup {
                width: depth as usize,
            },
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::DraftModel,
                depth,
            } if self.draft.is_some() => SpeculativeRoute::DraftModel {
                width: depth as usize,
            },
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::DraftModel,
                ..
            } => {
                fallbacks.push(core_llm::DRAFT_MODEL_NOT_LOADED.into());
                SpeculativeRoute::Plain
            }
        }
    }
}

/// The descriptor for the `mlx-llama` provider (constructible without loading weights; used for
/// explicit catalog composition and inspection).
pub fn provider_descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.to_string(),
        family: "llama".to_string(),
        backend: "mlx".to_string(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            // Text-only today; the VLM path (sc-7157) flips this on for a vision provider.
            supports_vision: false,
            // Weightless default: conservative. The load path flips this on for a Qwen3-VL checkpoint
            // whose config carries a `video_token_id` (sc-8081).
            supports_video: false,
            // Weightless default: conservative. The load path flips this on for a Gemma 4 checkpoint
            // that actually ships an audio projector (sc-18772).
            supports_audio: false,
            // Weightless default: conservative. The load path (descriptor_for + load) flips this on
            // when the loaded model's own chat template gates an `enable_thinking` kwarg (sc-7585).
            supports_thinking: false,
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            model_sampling_defaults: None,
            supports_preserve_thinking: false,
            // Weightless default: conservative. The load path flips this on when the loaded model's
            // chat template renders tool calls (sc-7636).
            supports_tools: false,
            mtp: None,
            // Every decoder this provider loads runs the speculative engine (sc-24434), so prompt
            // lookup needs nothing from the checkpoint. A Qwen3.8 MTP head is advertised on the
            // legacy `mtp` field by the load path.
            speculative: vec![prompt_lookup_capabilities(SPECULATIVE_MAX_DEPTH)],
            // JSON-constrained decoding (sc-7166).
            supported_constraints: vec![ConstraintKind::Json],
        },
    }
}

/// A descriptor reflecting a *loaded* model: family from the dispatched architecture and the context
/// length from `config.json`. (Quantization state is reported via [`LlamaProvider::is_quantized`].)
fn descriptor_for(cfg: &ModelConfig) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = cfg.architecture.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d.capabilities.speculative = vec![prompt_lookup_capabilities(speculative_max_depth(
        causal_attention_geometries(cfg),
    ))];
    d
}

/// The loaded-model descriptor for the hybrid Qwen3.6 (`qwen3_5`) decoder (parsed via
/// [`Qwen35Config`], which `ModelConfig` does not represent).
fn descriptor_for_qwen35(cfg: &Qwen35Config) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = Architecture::Qwen35.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    // Both proposers at the full-attention layers' backend-true depth (sc-24438); a configured
    // head — dense or sparse-MoE predictor layer — on both the legacy and the per-proposer field.
    let max_depth = speculative_max_depth([qwen35_attention_geometry(cfg)]);
    d.capabilities.speculative = vec![prompt_lookup_capabilities(max_depth)];
    if cfg.mtp_num_hidden_layers > 0 {
        d.capabilities
            .advertise_mtp(max_depth, recommended_depths().mtp);
    }
    d
}

#[cfg(test)]
thread_local! {
    /// A test's operational load budget on this thread ([`with_load_budget`]); the process-wide
    /// [`core_llm::AVAILABLE_MEMORY_OVERRIDE`] would leak into every concurrently loading test.
    static LOAD_BUDGET: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with load admission capped at `budget` bytes on this thread.
#[cfg(test)]
pub(crate) fn with_load_budget<R>(budget: u64, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            LOAD_BUDGET.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(LOAD_BUDGET.with(|b| b.replace(Some(budget))));
    f()
}

/// The operational budget load admission caps measured availability with
/// ([`core_llm::operational_memory_override`]).
fn load_memory_budget() -> CoreResult<Option<u64>> {
    #[cfg(test)]
    if let Some(budget) = LOAD_BUDGET.with(std::cell::Cell::get) {
        return Ok(Some(budget));
    }
    core_llm::operational_memory_override()
}

/// Admit a companion MTP head's resident bytes on top of what the load already admitted (the
/// target and any admitted draft, E7), returning the head and its bytes. `Err` is the named load
/// fallback (E2): the head is unreadable, or those bytes plus the head's exceed the budget.
fn admit_companion_head(
    head: &Path,
    target_bytes: u64,
    available: u64,
) -> Result<(&Path, u64), String> {
    // The shared rule Candle admits by, in MLX's one allocation domain (E8).
    core_llm::admit_companion_head(
        head,
        crate::load_memory::companion_head_bytes(head),
        [("unified memory", target_bytes, available)],
    )
}

/// Read and parse `config.json` from a snapshot directory into a JSON value (used to dispatch the
/// architecture before constructing the architecture-specific config).
fn read_config_value(dir: &Path) -> CoreResult<serde_json::Value> {
    let text = crate::__model_input_reader!(
        std::fs::read_to_string(dir.join("config.json")),
        "crates/llm/mlx-llm/src/provider.rs::read_config_value::std::fs::read_to_string@232605",
        ArtifactInput,
        dir.join("config.json"),
        |__model_input_path| std::fs::read_to_string(__model_input_path)
    )
    .map_err(|e| CoreError::Load(format!("read {}: {e}", dir.join("config.json").display())))?;
    serde_json::from_str(&text)
        .map_err(|e| CoreError::Load(format!("parse {}: {e}", dir.join("config.json").display())))
}

/// Read `eos_token_id` (int or array) from `config.json`; falls back to the Llama-3 stop ids.
pub fn eos_token_ids(dir: &Path) -> Vec<i32> {
    let fallback = vec![128001, 128008, 128009]; // <|end_of_text|>, <|eom_id|>, <|eot_id|>

    // Prefer `generation_config.json` — HF's canonical "how to generate" source, where models put
    // the *generation* EOS set (Qwen3.6's `<|im_end|>` turn-end lives only here, not in config.json).
    if let Some(ids) = read_json(dir, "generation_config.json")
        .as_ref()
        .and_then(|v| parse_token_ids(v.get("eos_token_id")))
    {
        return ids;
    }
    // Otherwise fall back to `config.json` — top-level, then the VLM-nested `text_config`.
    if let Some(v) = read_json(dir, "config.json") {
        if let Some(ids) = parse_token_ids(v.get("eos_token_id"))
            .or_else(|| parse_token_ids(v.get("text_config").and_then(|t| t.get("eos_token_id"))))
        {
            return ids;
        }
    }
    fallback
}

/// Whether a rendered prompt ends with an **unclosed** `<think>` block — i.e. the chat template
/// opened reasoning in the prompt (Qwen3.6's thinking/auto generation prompt) so the model generates
/// inside it. True iff the last `<think>` occurs after the last `</think>` (or there is no close).
fn prompt_opens_thinking(prompt: &str) -> bool {
    match prompt.rfind("<think>") {
        None => false,
        Some(open) => prompt.rfind("</think>").is_none_or(|close| open > close),
    }
}

/// Read and parse a JSON file from a snapshot dir, or `None` if missing/invalid.
fn read_json(dir: &Path, name: &str) -> Option<serde_json::Value> {
    let text = crate::__model_input_reader!(
        std::fs::read_to_string(dir.join(name)),
        "crates/llm/mlx-llm/src/provider.rs::read_json::std::fs::read_to_string@234545",
        ArtifactInput,
        dir.join(name),
        |__model_input_path| std::fs::read_to_string(__model_input_path)
    )
    .ok()?;
    serde_json::from_str(&text).ok()
}

/// Parse an `eos_token_id`-style field — a single int or an array of ints — into a non-empty id list.
fn parse_token_ids(v: Option<&serde_json::Value>) -> Option<Vec<i32>> {
    let ids = match v? {
        serde_json::Value::Number(n) => vec![n.as_i64()? as i32],
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_i64().map(|x| x as i32))
            .collect(),
        _ => return None,
    };
    (!ids.is_empty()).then_some(ids)
}

#[derive(Clone, Copy)]
enum MlxWorkspaceContract<'a> {
    Eager,
    Chunked,
    Qwen35 {
        config: &'a Qwen35Config,
        prism: bool,
    },
}

/// MLX permits ten completed command buffers to remain in flight and can be building the next
/// buffer before applying backpressure. Price that current buffer plus the in-flight set.
pub(crate) const MLX_EVAL_BUFFER_WINDOW: u64 = 11;
/// The largest command buffer MLX 0.32 encodes before committing it, in bytes of its ops'
/// inputs and outputs (`max_mb_per_buffer`: 40 MB on base / Pro / phone GPUs, 50 MB on Max and
/// Ultra; `MLX_MAX_MB_PER_BUFFER` is never set by this runtime). A buffer may exceed it by the one
/// op that crosses it.
const MLX_MAX_BUFFER_BYTES: u64 = 50 * 1024 * 1024;

/// The most ops MLX 0.32 encodes into one command buffer before committing it
/// (`max_ops_per_buffer`: 20–50 by GPU family; `MLX_MAX_OPS_PER_BUFFER` is never set here).
const MLX_MAX_OPS_PER_BUFFER: u64 = 50;

/// What a forward's promoted weight copies can hold at once when the decoder's activations are
/// `f32` against BF16 weights (a Gemma GeGLU whose role the activation-dtype policy keeps on
/// `f32`, [`CausalLm::activations_promote`]; sc-24446).
///
/// Each dense matmul materializes an `f32` copy of its weight (a quantized matmul, of its scales
/// and biases), held until the command buffer that consumes it completes. The LM head's copy —
/// the vocabulary-wide one, 2.4 GB on Gemma 2 and 4.0 GB on the 262K-token Gemma 4 — is priced
/// whole; the decoder's copies are bounded by MLX's evaluation window: at most
/// [`MLX_EVAL_BUFFER_WINDOW`] command buffers are alive (ten committed plus the one encoding),
/// each holding at most [`MLX_MAX_BUFFER_BYTES`] plus the one op that crosses it — at most the
/// largest promoted projection, `largest_elements`.
fn promoted_weight_bytes(head_elements: u64, largest_elements: u64) -> Option<u64> {
    const F32: u64 = 4;
    let per_buffer = largest_elements
        .checked_mul(F32)?
        .checked_add(MLX_MAX_BUFFER_BYTES)?;
    head_elements
        .checked_mul(F32)?
        .checked_add(per_buffer.checked_mul(MLX_EVAL_BUFFER_WINDOW)?)
}

/// Apple-Silicon Metal allocations are rounded to 16-KiB VM pages. Gated DeltaNet retains one
/// independently allocated output row per prompt token until its final concatenate.
const MLX_ALLOCATION_PAGE_BYTES: u64 = 16 * 1024;
/// The recurrence creates five one-element index buffers per step and evaluates every 256 steps.
///
/// These `QWEN35_RECURRENCE_*` terms price the op-by-op recurrence the frozen allocator peaks were
/// measured on. Since sc-24443 the GPU runs the fused Metal kernel instead. Its prompt-sized
/// buffers are its f32 `y` output and the row-contiguous copies MLX makes of non-contiguous `q`,
/// `k` and `v` inputs — `2·Hk·Dk + 2·Hv·Dv` elements per token, never more than the
/// `2·Hv·Dk + 2·Hv·Dv` priced here (`Hv ≥ Hk`). Past `KERNEL_MAX_STEPS` tokens the per-dispatch
/// input gathers replace those copies and the final concatenate of the per-dispatch outputs adds
/// one more `Hv·Dv`, still within the priced width while `Hv·Dv ≤ 2·(Hv − Hk)·Dk` (Qwen3.6-27B:
/// `Hv = 3·Hk`, `Dk = Dv`). Its recurrent state is fixed-size, covered by the 256 × 5 state-array
/// term, and it allocates neither the per-step index buffers nor the page-padded per-token `y`
/// rows priced below. On the GPU these terms are therefore a conservative upper bound for those
/// geometries (the op path remains the CPU-stream route for short runs).
const QWEN35_RECURRENCE_EVAL_CHUNK: u64 = 256;
const QWEN35_RECURRENCE_INDEX_BUFFERS: u64 = 5;
/// Each lazy recurrence step creates five state-shaped arrays: the decayed state, the state-key
/// product, the delta outer product, the updated state, and the updated-state/query product. None
/// can be released before that chunk's explicit evaluation.
const QWEN35_RECURRENCE_STATE_ARRAYS_PER_STEP: u64 = 5;

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(0u64, |sum, value| sum.checked_add(value))
}

fn checked_product(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(1u64, |product, value| product.checked_mul(value))
}

fn round_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

/// Extra complete-decoder workspace above the generic chunked-attention estimate for dense
/// Qwen3.5. Every term corresponds to a prompt-sized tensor retained by the Rust graph or by the
/// pinned MLX evaluator; all tensor elements are charged at four bytes even when the runtime value
/// is BF16.
fn estimate_qwen35_workspace_extra_bytes(
    prompt_tokens: usize,
    config: &Qwen35Config,
    prism: bool,
) -> Option<u64> {
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let nonnegative = |value: i32| u64::try_from(value).ok();
    let hidden = nonnegative(config.hidden_size)?;
    let intermediate = nonnegative(config.intermediate_size)?;
    let query_heads = nonnegative(config.num_heads)?;
    let head_dim = nonnegative(config.head_dim)?;
    let key_heads = nonnegative(config.linear_num_key_heads)?;
    let value_heads = nonnegative(config.linear_num_value_heads)?;
    let key_head_dim = nonnegative(config.linear_key_head_dim)?;
    let value_head_dim = nonnegative(config.linear_value_head_dim)?;
    let conv_kernel = nonnegative(config.linear_conv_kernel_dim)?;
    let key_width = key_heads.checked_mul(key_head_dim)?;
    let value_width = value_heads.checked_mul(value_head_dim)?;
    let conv_width = key_width.checked_mul(2)?.checked_add(value_width)?;
    let attention_width = query_heads.checked_mul(head_dim)?;

    // PrismLinear rotates each projection input independently. The full-attention block has the
    // wider simultaneous input-width sum; the linear block uses two mixer input rotations plus its
    // output rotation. Four buffers per input width covers the unfused cast -> sign multiply ->
    // Hadamard -> cast chain (F32 intermediates plus the BF16 result, no donation) — the path a
    // block the fused kernel does not cover still takes. The fused kernel (sc-24444) writes only
    // the output buffer, so this stays an upper bound on the Bonsai path.
    let packed_rotation_elements = if prism {
        let linear_widths = checked_sum([hidden.checked_mul(4)?, value_width, intermediate])?;
        let attention_widths =
            checked_sum([hidden.checked_mul(5)?, attention_width, intermediate])?;
        linear_widths.max(attention_widths).checked_mul(4)?
    } else {
        0
    };

    // The ops recurrence expands Q/K from key heads to value heads, casts Q/K/V to F32, and keeps
    // every F32 Y row until concatenate. Count two Q/K tensors plus V and Y.
    let recurrence_elements = checked_sum([
        value_heads.checked_mul(key_head_dim)?.checked_mul(2)?,
        value_width.checked_mul(2)?,
    ])?;

    // Bound the lazy linear-mixer graph through the four-term depthwise convolution, normalized
    // Q/K, gates, and projection outputs. `(2*K + 3)` covers K products, K-1 additions, the
    // concatenated input, convolution output, and activation output.
    let mixer_elements = checked_sum([
        conv_width.checked_mul(conv_kernel.checked_mul(2)?.checked_add(3)?)?,
        value_width,
        key_width.checked_mul(2)?,
        value_heads.checked_mul(2)?,
    ])?;

    // Completed Metal command buffers retain primitive inputs until their callbacks run. Bound one
    // largest prompt-sized output for each possible in-flight/current buffer.
    let largest_output = [
        hidden,
        intermediate,
        conv_width,
        attention_width.checked_mul(2)?,
        value_width,
    ]
    .into_iter()
    .max()?;
    let evaluator_elements = largest_output.checked_mul(MLX_EVAL_BUFFER_WINDOW)?;

    let per_token_elements = checked_sum([
        packed_rotation_elements,
        recurrence_elements,
        mixer_elements,
        evaluator_elements,
    ])?;
    let prompt_workspace = checked_product([prompt, per_token_elements, 4])?;
    // The shared tiled estimate prices one attention tile (F32 scores). MLX can retain the same
    // tile's buffers for the rest of its evaluator window, so price those additional tiles here.
    let kv_heads = nonnegative(config.num_kv_heads)?;
    let attention_window =
        prefill_attention_tile_bytes(prompt, query_heads, kv_heads, head_dim, 4)?
            .checked_mul(MLX_EVAL_BUFFER_WINDOW.checked_sub(1)?)?;

    // Each retained recurrence row owns a separate allocator buffer; price its VM-page rounding.
    let y_row_bytes = value_width.checked_mul(4)?;
    let row_padding = round_up(y_row_bytes, MLX_ALLOCATION_PAGE_BYTES)?
        .checked_sub(y_row_bytes)?
        .checked_mul(prompt)?;
    // Index buffers are created eagerly for all five slices in the current recurrence chunk.
    let recurrence_steps = prompt.min(QWEN35_RECURRENCE_EVAL_CHUNK);
    let index_buffers = checked_product([
        recurrence_steps,
        QWEN35_RECURRENCE_INDEX_BUFFERS,
        MLX_ALLOCATION_PAGE_BYTES,
    ])?;
    // `gated_delta_recurrence` builds a complete lazy graph until its 256-step `eval`. Price every
    // state-shaped node in that graph plus the input state. The shared base estimate separately
    // prices the retained recurrent caches for all decoder layers.
    let recurrence_state_arrays = recurrence_steps
        .checked_mul(QWEN35_RECURRENCE_STATE_ARRAYS_PER_STEP)?
        .checked_add(1)?;
    let recurrent_state_buffers =
        checked_product([value_width, key_head_dim, 4, recurrence_state_arrays])?;

    checked_sum([
        prompt_workspace,
        attention_window,
        row_padding,
        index_buffers,
        recurrent_state_buffers,
    ])
}

/// Scalar width of the attention scores, additive mask and softmax weights priced by the request
/// estimate: `sdpa_eager` upcasts all three to F32 for a BF16 decoder.
const EAGER_SCORE_ELEMENT_BYTES: u64 = dtype_bytes(Dtype::Float32);

/// Scalar width the request estimate prices K/V, decoder activations and logits at.
///
/// Since sc-20671 every decoder runs these in its compute dtype, whatever dtype the snapshot
/// stores its quantized scales in. The Prism/Bonsai path is held at the F32 width: its F16 scales
/// promoted the whole decoder to F32 before sc-20671, and its frozen allocator peaks (the
/// `qwen35_prism_estimate_covers_frozen_allocator_peaks…` floors) were measured on that path. At
/// the compute width the 27-token floor is no longer covered, so the width stays until those
/// peaks are re-taken on the BF16 path at epic end (SC-20671 dense-baseline contract doc).
fn priced_compute_element_bytes(compute: Dtype, prism: bool) -> u64 {
    if prism {
        dtype_bytes(Dtype::Float32)
    } else {
        dtype_bytes(compute)
    }
}

/// The shared tiled estimate with the attention term priced from the tile `sdpa` actually runs for
/// this geometry ([`prefill_attention_tile_bytes`]): a `rows × k_len` mask slice for fused
/// full-kernel blocks, a per-head score/mask/softmax set for row chunks (sc-20676).
fn estimate_routed_request_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
) -> Option<u64> {
    let attention = prefill_attention_tile_bytes(
        u64::try_from(prompt_tokens).ok()?,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.score_element_bytes,
    )?;
    core_llm::estimate_tiled_request_bytes_with_recurrent_copies(
        prompt_tokens,
        max_new_tokens,
        geometry,
        vision_workspace_bytes,
        mtp_width,
        attention,
        // The recurrent term once: the hybrid rolls back through its checkpoint ring, whose bytes
        // the caller folds into `geometry.recurrent_bytes`, and is never cloned (sc-24435).
        1,
    )
}

/// [`estimate_mlx_request_bytes`] for a request the provider serves from the compressed KV cache
/// (sc-20682): `compressed_kv_bytes` replaces the dense K/V term of the generic fused-attention
/// estimate. Only that contract's decoders can run compressed (the eager contract is exactly the
/// soft-cap/latent/shared-K/V geometry the fused reader refuses, and the hybrid decoder has no
/// compressed cache), so any other contract keeps its dense estimate, which is never smaller.
fn estimate_mlx_compressed_request_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    contract: MlxWorkspaceContract<'_>,
    compressed_kv_bytes: u64,
) -> Option<u64> {
    match contract {
        MlxWorkspaceContract::Chunked => {
            let attention = prefill_attention_tile_bytes(
                u64::try_from(prompt_tokens).ok()?,
                geometry.query_heads,
                geometry.kv_heads,
                geometry.head_dim,
                geometry.score_element_bytes,
            )?;
            core_llm::estimate_tiled_request_bytes_with_kv_bytes(
                prompt_tokens,
                geometry,
                vision_workspace_bytes,
                mtp_width,
                attention,
                1,
                compressed_kv_bytes,
            )?
            .checked_add(mlx_runtime_request_bytes(
                prompt_tokens,
                u64::try_from(prompt_tokens)
                    .ok()?
                    .checked_add(u64::from(max_new_tokens))?,
                u64::from(mtp_width),
                u64::try_from(prompt_tokens)
                    .ok()?
                    .min(SDPA_MAX_FUSED_QLEN as u64),
                geometry,
                // The compressed format's own bytes already round to its growth blocks.
                false,
            )?)
        }
        MlxWorkspaceContract::Eager | MlxWorkspaceContract::Qwen35 { .. } => {
            estimate_mlx_request_bytes(
                prompt_tokens,
                max_new_tokens,
                geometry,
                vision_workspace_bytes,
                mtp_width,
                contract,
            )
        }
    }
}

/// Select the estimate matching the complete decoder execution graph. Generic fused attention uses
/// the shared tiled estimate. Dense Qwen3.5 adds its F32 recurrence, packed-Hadamard, allocator, and
/// lazy-evaluator lifetimes; eager/otherwise-unbounded implementations retain the quadratic model.
fn estimate_mlx_request_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    contract: MlxWorkspaceContract<'_>,
) -> Option<u64> {
    let decoder = estimate_mlx_decoder_bytes(
        prompt_tokens,
        max_new_tokens,
        geometry,
        vision_workspace_bytes,
        mtp_width,
        contract,
    )?;
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let total = prompt.checked_add(u64::from(max_new_tokens))?;
    let runtime = match contract {
        // The hybrid's contract already prices MLX's evaluation window and allocator rounding from
        // its own frozen probes; only the driver's wake is added.
        MlxWorkspaceContract::Qwen35 { .. } => MLX_REQUEST_WAKE_BYTES,
        MlxWorkspaceContract::Eager => mlx_runtime_request_bytes(
            prompt_tokens,
            total,
            u64::from(mtp_width),
            prompt,
            geometry,
            true,
        )?,
        MlxWorkspaceContract::Chunked => mlx_runtime_request_bytes(
            prompt_tokens,
            total,
            u64::from(mtp_width),
            prompt.min(SDPA_MAX_FUSED_QLEN as u64),
            geometry,
            true,
        )?,
    };
    // Block growth (sc-20682 review): at each block growth the MLX dense cache holds every layer's
    // pre-growth buffer beside its successor until the evaluator releases it (a dense decode across
    // growth blocks peaked 1.30x the shared estimate). The generic runtime term already prices the
    // block padding itself, so it adds only those retired buffers; the hybrid's frozen probes price
    // neither, so it is charged the whole growth (padding and retired buffers).
    let shape = geometry.kv_shape();
    let final_kv = crate::kv_policy::dense_final_kv_bytes(shape, total)?;
    let priced_kv = match contract {
        MlxWorkspaceContract::Qwen35 { .. } => shape.dense_bytes(total)?,
        MlxWorkspaceContract::Eager | MlxWorkspaceContract::Chunked => {
            let block = u64::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).ok()?;
            shape.dense_bytes(round_up(total, block)?.max(block))?
        }
    };
    checked_sum([decoder, runtime, final_kv.checked_sub(priced_kv)?])
}

/// The `phys_footprint` a request re-acquires from the Metal driver after the process has idled
/// ([`crate::load_memory::MLX_DRIVER_WAKE_BYTES`]), charged once per request.
const MLX_REQUEST_WAKE_BYTES: u64 = crate::load_memory::MLX_DRIVER_WAKE_BYTES;

/// K/V positions a request's caches allocate beyond `positions`: [`ContiguousKvCache`] grows in
/// whole [`KV_BLOCK_TOKENS`](crate::primitives::kv_cache::KV_BLOCK_TOKENS)-position blocks.
fn kv_block_padding(positions: u64) -> Option<u64> {
    let block = u64::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).ok()?;
    round_up(positions, block)?.checked_sub(positions)
}

/// Block positions a `width`-position verify overshoot past `committed` positions allocates
/// beyond `committed`'s own block padding and the `width` positions themselves.
fn kv_overshoot_padding(committed: u64, width: u64) -> Option<u64> {
    let block = u64::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).ok()?;
    let with = round_up(committed.checked_add(width)?, block)?;
    let without = round_up(committed, block)?.checked_add(width)?;
    Some(with.saturating_sub(without))
}

/// What MLX's runtime holds for a generic decoder's request beyond the decoder's own tensors
/// (sc-24446), which the shared estimates do not see — measured on the one-token requests of the
/// sc-24446 probes (35–87 MB MLX active on Llama 3.2 1B, Qwen3-1.7B and Qwen3-8B, against 6–9 MB
/// estimated before):
///
/// * **KV block padding**: the caches hold whole blocks over every position the request can write
///   — the `committed` prompt and generation plus a `verify_width` step past them (an MTP verify;
///   the prompt-lookup / draft overshoot is added by its route).
/// * **In-flight layer temporaries.** MLX keeps every op's inputs until the command buffer that
///   ran it completes, and runs ahead by up to [`MLX_EVAL_BUFFER_WINDOW`] buffers, each holding at
///   most [`MLX_MAX_BUFFER_BYTES`] plus the one op that crosses it. The shared estimate prices one
///   layer's activations; the rest in flight is every other layer's, capped by that window —
///   whose crossing op can be a whole `prompt × inter` MLP activation or an `attention_rows`-row
///   score block (eager: the full `prompt × prompt` per head).
/// * **Allocation rounding**: a page per op output in the window.
/// * **The driver's wake** ([`MLX_REQUEST_WAKE_BYTES`]).
fn mlx_runtime_request_bytes(
    prompt_tokens: usize,
    committed: u64,
    verify_width: u64,
    attention_rows: u64,
    geometry: LlmMemoryGeometry,
    dense_block_padding: bool,
) -> Option<u64> {
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let kv_position = checked_product([
        geometry.layers,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.element_bytes,
        2,
    ])?;
    // Every allocated position past the committed ones: the verify width the step writes beyond
    // them and — on a dense block-allocated cache — the rest of their last block.
    let block_padding = if dense_block_padding {
        kv_block_padding(committed.checked_add(verify_width)?)?
    } else {
        0
    };
    let kv_padding = checked_sum([verify_width, block_padding])?.checked_mul(kv_position)?;
    let scores = checked_product([
        prompt,
        attention_rows,
        geometry.query_heads,
        geometry.element_bytes,
    ])?;
    let mlp = checked_product([prompt, geometry.intermediate_size, geometry.element_bytes])?;
    let per_layer = checked_sum([
        checked_product([
            prompt,
            checked_sum([
                geometry.intermediate_size.checked_mul(3)?,
                geometry.hidden_size.checked_mul(8)?,
            ])?,
            geometry.element_bytes,
        ])?,
        scores.checked_mul(3)?,
    ])?;
    let window = MLX_EVAL_BUFFER_WINDOW
        .checked_mul(checked_sum([MLX_MAX_BUFFER_BYTES, mlp.max(scores)])?)?;
    let in_flight = geometry
        .layers
        .saturating_sub(1)
        .checked_mul(per_layer)?
        .min(window);
    // Each in-flight buffer's op outputs are page-rounded allocations, however small.
    let rounding = checked_product([
        MLX_EVAL_BUFFER_WINDOW,
        MLX_MAX_OPS_PER_BUFFER,
        MLX_ALLOCATION_PAGE_BYTES,
    ])?;
    checked_sum([kv_padding, in_flight, rounding, MLX_REQUEST_WAKE_BYTES])
}

/// The decoder's own request tensors under its workspace contract: the shared estimates with the
/// attention term priced from the tile `sdpa` actually runs (sc-20676,
/// [`estimate_routed_request_bytes`]), plus the Qwen3.5 hybrid's recurrence workspace.
fn estimate_mlx_decoder_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    contract: MlxWorkspaceContract<'_>,
) -> Option<u64> {
    match contract {
        // The recurrent term once: the only MLX decoder with recurrent state (the Qwen35 hybrid,
        // eager for MoE) rolls back through its checkpoint ring, whose bytes the caller folds into
        // `geometry.recurrent_bytes`, and is never cloned; a causal decoder holds none.
        MlxWorkspaceContract::Eager => core_llm::estimate_request_bytes_with_recurrent_copies(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace_bytes,
            mtp_width,
            1,
        ),
        MlxWorkspaceContract::Chunked => estimate_routed_request_bytes(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace_bytes,
            mtp_width,
        ),
        MlxWorkspaceContract::Qwen35 { config, prism } => {
            let base = estimate_routed_request_bytes(
                prompt_tokens,
                max_new_tokens,
                geometry,
                vision_workspace_bytes,
                mtp_width,
            )?;
            base.checked_add(estimate_qwen35_workspace_extra_bytes(
                prompt_tokens,
                config,
                prism,
            )?)
        }
    }
}

fn validate_context_window(
    cap: usize,
    prompt_tokens: usize,
    max_new_tokens: u32,
) -> CoreResult<()> {
    if cap == 0 {
        return Ok(());
    }
    let total = prompt_tokens
        .checked_add(max_new_tokens as usize)
        .ok_or_else(|| CoreError::InvalidRequest("prompt + generation length overflow".into()))?;
    if total > cap {
        return Err(CoreError::InvalidRequest(format!(
            "expanded prompt ({prompt_tokens} tokens) + requested generation ({max_new_tokens}) exceeds context window {cap}"
        )));
    }
    Ok(())
}

fn bonsai_sampling_defaults() -> ModelSamplingDefaults {
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

fn map_finish(f: FinishReason) -> CoreFinish {
    match f {
        // An EOS *id* and a host stop condition (a request `stop` string) are both `Stop` per the
        // contract / OpenAI semantics.
        FinishReason::StopToken | FinishReason::Stopped => CoreFinish::Stop,
        FinishReason::MaxTokens => CoreFinish::Length,
        FinishReason::Cancelled => CoreFinish::Cancelled,
    }
}

/// Ordinary registration used by explicit runtime bundles.
pub const REGISTRATION: core_llm::TextLlmRegistration = core_llm::TextLlmRegistration {
    descriptor: provider_descriptor,
    load: load_registered,
    can_load,
    weightless_vision: Some(weightless_vision),
    weightless_audio: Some(weightless_audio),
};

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `mlx-llama` provider serve the snapshot at
/// `spec.source`? Reads **only** `config.json` and runs the same [`Architecture::from_config`]
/// dispatch the loader uses — it never opens a safetensors shard, so `core-llm`'s `load_for_model`
/// can resolve a provider by model without loading weights. A VLM wrapper (one carrying a
/// `vision_config`) is served **text-only** here unless a dedicated vision provider claims it: a
/// LLaVA snapshot is ceded to `mlx-joycaption`, while a Qwen-VL (`qwen3_5`) wrapper is claimed here
/// (its nested text decoder) so it no longer misroutes to JoyCaption (sc-7626).
pub fn can_load(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let language_is_prism = crate::gguf::GgufFile::open(dir).ok().is_some_and(|g| {
            g.meta_str("general.architecture") == Some("qwen35")
                && g.meta("prism.hadamard.version").is_some()
        });
        if !language_is_prism {
            return false;
        }
        return spec.projector_source.as_deref().is_none_or(|source| {
            crate::gguf::GgufFile::open(source)
                .ok()
                .is_some_and(|projector| crate::prism_vision_gguf::validate(&projector).is_ok())
        });
    }
    if spec.projector_source.is_some() {
        return false;
    }
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
    can_load_value(&v)
}

/// Pure architecture/routing decision over a parsed `config.json` (split out for unit testing without
/// a snapshot on disk).
fn can_load_value(v: &serde_json::Value) -> bool {
    // A VLM wrapper carries a `vision_config`. Cede it to the dedicated vision provider when that
    // provider claims it (LLaVA → `mlx-joycaption`); otherwise serve the nested text decoder
    // text-only (a Qwen-VL `qwen3_5` wrapper).
    if v.get("vision_config").is_some() && crate::joycaption::can_load_value(v) {
        return false;
    }
    Architecture::from_config(v).is_ok()
}

/// **Weightless** per-snapshot vision probe (sc-8077): does `mlx-llama` serve the snapshot at
/// `spec.source` *with* vision? Reads only `config.json` for safetensors snapshots, or the GGUF
/// language and explicitly-associated projector headers (never their tensor payloads), mirroring
/// [`can_load`]. It drives core-llm's pre-load capability gate so a *model-first* vision-required
/// load (`load_for_model_with(spec, with_vision())`) resolves a Qwen-VL wrapper here.
///
/// This is necessary because the provider's STATIC [`provider_descriptor`] must report
/// `supports_vision=false` (the one registration also serves plain text-only checkpoints; vision is
/// only flipped on post-load when the loaded checkpoint carries `model.visual.*`). Without a
/// per-snapshot probe a genuine Qwen3-VL snapshot would be rejected at the gate (the story-D gap).
pub fn weightless_vision(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
        return spec.projector_source.is_some() && can_load(spec);
    }
    if spec.projector_source.is_some() {
        return false;
    }
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
    weightless_vision_value(&v)
}

/// Pure per-snapshot vision decision over a parsed `config.json` (split out for unit testing). True
/// iff this provider both *claims* the snapshot ([`can_load_value`]) AND the snapshot is a Qwen-VL
/// wrapper that loads its vision tower: a `vision_config` is present and the architecture dispatches
/// to a Qwen-VL decoder (`qwen3_vl` or the Qwen3.6 `qwen3_5` hybrid) — exactly the post-load
/// condition that flips `supports_vision` on in [`LlamaProvider::load`]. A plain text checkpoint (no
/// `vision_config`) or a ceded LLaVA snapshot returns false.
fn weightless_vision_value(v: &serde_json::Value) -> bool {
    if !can_load_value(v) || v.get("vision_config").is_none() {
        return false;
    }
    matches!(
        Architecture::from_config(v),
        Ok(Architecture::Qwen3Vl) | Ok(Architecture::Qwen35)
    )
    // Gemma 4 is deliberately absent: its vision path is loaded but unvalidated and therefore
    // unadvertised (see `gemma4_vision_is_validated`). The probe must agree with the loaded
    // descriptor, or a model-first vision-required load would resolve here and then be rejected.
}

/// **Weightless** per-snapshot audio probe (sc-18772): does `mlx-llama` serve the snapshot at
/// `spec.source` *with* audio? The audio analogue of [`weightless_vision`] — reads only
/// `config.json`, never a weight shard.
///
/// Gemma 4 unified is the only architecture this provider serves that has an audio path at all, and
/// like vision it is per-snapshot: the same registration also serves text-only checkpoints, so the
/// static descriptor stays `supports_audio=false` and this probe is what lets a model-first
/// audio-required load (`load_for_model_with(spec, with_audio())`) resolve here.
pub fn weightless_audio(spec: &LoadSpec) -> bool {
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
    weightless_audio_value(&v)
}

/// Pure per-snapshot audio decision over a parsed `config.json` (split out for unit testing).
fn weightless_audio_value(v: &serde_json::Value) -> bool {
    gemma4_multimodal(v, "audio_config", "audio_token_id")
}

/// Whether this provider claims `v` AND it is a Gemma 4 checkpoint declaring the named front-end
/// block (`vision_config` / `audio_config`) plus the token id that front-end splices at.
///
/// The token-id check is not redundant with the block: a config carrying a `vision_config` but no
/// `image_token_id` cannot be conditioned on an image at all (there is no row to splice into), so
/// advertising vision for it would be exactly the advertised-but-absent case. The load path refuses
/// such a config for the same reason, which keeps probe and loader agreeing.
fn gemma4_multimodal(v: &serde_json::Value, block: &str, token_key: &str) -> bool {
    if !can_load_value(v) || v.get(block).is_none() {
        return false;
    }
    if !matches!(Architecture::from_config(v), Ok(a) if a.is_gemma4()) {
        return false;
    }
    v.get(token_key).and_then(|x| x.as_i64()).is_some()
}

#[cfg(test)]
pub(crate) mod tests;
