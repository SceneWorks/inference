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
    MtpMode, Quantize, ReasoningEffort, RenderOptions, Result as CoreResult, Role, Sampling,
    StopMatcher, StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor,
    TextLlmOutput, TextLlmRequest, ThinkingMode, ThinkingSegmenter, Tokenizer, ToolCall,
    ToolCallSegmenter, Usage, VideoRef,
};

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_batch, generate_from_prefill, generate_from_prefill_with_timings,
    generate_qwen35_mtp_multimodal_with_timings, generate_qwen35_mtp_with_timings,
    generate_with_observer, generate_with_timings, generate_with_timings_on, BatchRequest,
    CancelFlag, ConstraintMask, Decode, FinishReason, GenerationConfig, GenerationOutput,
    Qwen35MtpMultimodalPrompt, RewindableConstraintMask, StreamEvent,
};
use crate::image::Qwen35ImageProcessor;
use crate::models::gemma4_mm;
use crate::models::{
    CausalLm, Gemma4Layout, Gemma4Mm, Gemma4MmConfig, Qwen35Config, Qwen35Model,
    Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::attention::prefill_attention_tile_bytes;
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
                .map_err(to_core)?;
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
                .map_err(to_core)?;
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

/// A [`Decode`] wrapper that shifts the RoPE offset by a constant `delta` for the post-prompt
/// continuation of a multimodal decode. Image tokens compress the position cursor, so the text
/// positions that follow the prompt are `cache_len + mrope_delta`, not `cache_len`; the new tokens
/// are text, so a single shifted 1-D position is the correct (interleaved-)M-RoPE position. Drives
/// either backbone through [`VlmDecode`]'s [`Decode`] supertrait — each decoder downcasts its own
/// cache inside `step`, so no concrete-type fork is needed here.
struct Shifted<'a> {
    model: &'a dyn VlmDecode,
    delta: i32,
}

impl Decode for Shifted<'_> {
    fn make_cache(&self) -> Box<dyn KvCache> {
        self.model.make_cache()
    }

    fn step(
        &self,
        ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Array> {
        self.model.step(ids, cache, offset + self.delta)
    }
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
    /// The compressed-KV qualification family (sc-20679). Only a snapshot load names it, from the
    /// same architecture-name check as `campaign_family`; [`Self::from_parts`] cannot tell a
    /// Llama checkpoint from a Mistral or dense Qwen2 one sharing its decoder, so it has none.
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
    /// The retained fused compressed-KV reader (sc-20679), built on the first request the
    /// qualification table admits and reused by every later one; `Err` is the build failure each
    /// such request then reports as [`core_llm::KvCacheFallbackReason::ReaderUnavailable`].
    kv_reader: OnceCell<Result<crate::primitives::CompiledKernelHandle, String>>,
    /// Dense Prism `vision_tower.*` tensors retained verbatim for the native multimodal adapter.
    /// Text loading must not discard them merely because sc-23937 constructs only the decoder.
    _prism_vision_weights: Option<Weights>,
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

impl LlamaProvider {
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

    pub(crate) fn campaign_context_band_measurement(
        &self,
        context_band: &str,
    ) -> CoreResult<(String, u64, u64)> {
        let context_window = self.campaign_context_window()?;
        let target = crate::campaign::context_band_target(context_window, context_band)
            .map_err(CoreError::Load)?;
        self.campaign_band_payload(context_band, target)
    }

    /// The multi-turn prompt-cache fixture's payload (contract v4): the band payload capped by
    /// [`crate::campaign::multi_turn_payload_target`], so turn 2 plus the full turn-2 forced
    /// continuation fits the native window. The row's coordinate operations keep the full band.
    pub(crate) fn campaign_multi_turn_payload(
        &self,
        context_band: &str,
    ) -> CoreResult<(String, u64, u64)> {
        let context_window = self.campaign_context_window()?;
        let band = crate::campaign::context_band_target(context_window, context_band)
            .map_err(CoreError::Load)?;
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
        let cache = cache_slot.get_or_insert_with(|| crate::decode::PrefixCache::new(2));
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
        let mut cache = crate::decode::PrefixCache::new(2);
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
        Self::load_inner(spec, true)
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
        let quant = spec
            .quantize
            .map(|q| match q {
                Quantize::Q4 => Ok(QuantSpec::q4()),
                Quantize::Q8 => Ok(QuantSpec::q8()),
                // sc-24135: NVFP4 is a CUDA sm_120 capability; refuse by name, never substitute —
                // before admission, so a memory refusal cannot mask it.
                Quantize::Nvfp4 => Err(CoreError::Unsupported(
                    "nvfp4: NVFP4 projections need a CUDA device with compute capability >= \
                     sm_120; the MLX backend has no NVFP4 GEMM"
                        .into(),
                )),
            })
            .transpose()?;
        let required = crate::load_memory::required_bytes(spec)?;
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            core_llm::operational_memory_override()?,
        )?;
        core_llm::admit_request_memory(required, available)?;

        let dir = Path::new(&spec.source);
        if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
            return Self::load_prism_gguf(spec, dir);
        }
        // Read config.json once to dispatch the architecture: the hybrid Qwen3.6 (`qwen3_5`) decoder
        // has its own config/weights path (and `ModelConfig` deliberately rejects it).
        let cfg_value = read_config_value(dir)?;
        let arch = Architecture::from_config(&cfg_value).map_err(to_core)?;
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
        let weights = Weights::from_dir(dir).map_err(to_core)?;
        let is_prism =
            cfg_value.get("model_type").and_then(|v| v.as_str()) == Some("prism_hadamard_qwen35");
        if is_prism && quant.is_some() {
            return Err(CoreError::Load(
                "Prism snapshots are already packed 2-bit and reject load-time Q4/Q8".into(),
            ));
        }

        let mut prism_vision_weights = None;
        let (model, mut descriptor) = if arch == Architecture::Qwen35 {
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let mut descriptor = descriptor_for_qwen35(&qcfg);
            let m = if is_prism {
                let pack = PrismMlxPack::from_dir(dir, &cfg_value, &weights).map_err(to_core)?;
                descriptor.family = "prism_hadamard_qwen35".into();
                let model =
                    Qwen35Model::from_prism_weights(&weights, qcfg, &pack).map_err(to_core)?;
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
                Qwen35Model::from_weights_with(&weights, prefix, qcfg, quant).map_err(to_core)?
            };
            (Decoder::Qwen35(Box::new(m)), descriptor)
        } else {
            let cfg = ModelConfig::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let m = CausalLm::from_weights_with(&weights, "", cfg, quant).map_err(to_core)?;
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

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
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
            kv_family: crate::kv_policy::family_for(campaign_family),
            model,
            tokenizer,
            template,
            tool_call_format,
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision,
            gemma4,
            campaign_prefix_cache: RefCell::new(None),
            kv_reader: OnceCell::new(),
            _prism_vision_weights: prism_vision_weights,
        })
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
            kv_reader: OnceCell::new(),
            _prism_vision_weights: None,
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
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
            descriptor: provider_descriptor(),
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
            kv_reader: OnceCell::new(),
            _prism_vision_weights: None,
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
                .map_err(to_core)?;
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
    let source = std::fs::read_to_string(dir.join("chat_template.jinja")).ok()?;
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
        self.generate_inner(req, on_event, None, None, None)
    }
}

impl LlamaProvider {
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
        let mut store = crate::decode::PrefixCache::new(2);
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
        let mut store = crate::decode::PrefixCache::new(2);
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
    /// `max_new_tokens` more), then the request shape, then this backend's reader stage — the
    /// decoder's attention geometry and the retained fused reader. Every refusal is a dense plan
    /// with its reason.
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
        };
        let plan = core_llm::plan_kv_cache(request, |row| {
            let Decoder::Causal(model) = &self.model else {
                // The request-shape stage refuses the hybrid decoder before this stage runs.
                return Err((
                    Reason::UnsupportedRequest,
                    "the hybrid recurrent decoder has no compressed cache".into(),
                ));
            };
            if let Some(refusal) = crate::kv_policy::geometry_refusal(model.config()) {
                return Err((Reason::UnsupportedGeometry, refusal));
            }
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
    fn admission_estimate(
        &self,
        format: Option<core_llm::KvCompressionFormat>,
        prompt_tokens: usize,
        max_new_tokens: u32,
        vision_workspace: u64,
        mtp_width: u32,
    ) -> CoreResult<u64> {
        let geometry = self.model.memory_geometry();
        let contract = self.model.workspace_contract();
        let estimate = match format {
            None => estimate_mlx_request_bytes(
                prompt_tokens,
                max_new_tokens,
                geometry,
                vision_workspace,
                mtp_width,
                contract,
            ),
            Some(format) => u64::try_from(prompt_tokens)
                .ok()
                .and_then(|prompt| prompt.checked_add(u64::from(max_new_tokens)))
                .and_then(|total| {
                    crate::kv_policy::compressed_request_kv_bytes(format, &geometry, total)
                })
                .and_then(|kv| {
                    estimate_mlx_compressed_request_bytes(
                        prompt_tokens,
                        max_new_tokens,
                        geometry,
                        vision_workspace,
                        mtp_width,
                        contract,
                        kv,
                    )
                }),
        };
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

        // sc-20679: the request's KV cache, decided against the qualification table before any
        // K/V mutation — and, since sc-20682, before admission, so the estimate prices the cache
        // the request will actually run on. Only the plain text decode below runs a compressed
        // plan; its prompt is the tokenized prompt (no visual expansion on that path).
        let kv_plan = if observer.is_some() {
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
        let mtp_width = match req.mtp {
            MtpMode::Off => 0,
            MtpMode::Auto => self
                .descriptor
                .capabilities
                .mtp
                .map_or(0, |c| c.recommended_draft_tokens),
            MtpMode::Enabled { draft_tokens } => draft_tokens,
        };
        let geometry = self.model.memory_geometry();
        let compressed_format = match &kv_plan {
            KvPlan::Compressed { format, .. } => Some(*format),
            KvPlan::Unreported | KvPlan::Dense(_) => None,
        };
        let estimate = |format| {
            self.admission_estimate(
                format,
                admitted_prompt,
                req.max_new_tokens,
                vision_workspace,
                mtp_width,
            )
        };
        let required = estimate(compressed_format)?;
        // What a compressed generation that turns dense part-way grows to (its transition is
        // admitted against this, sc-20682).
        let dense_final_kv = u64::try_from(admitted_prompt)
            .ok()
            .and_then(|prompt| prompt.checked_add(u64::from(req.max_new_tokens)))
            .and_then(|total| crate::kv_policy::dense_final_kv_bytes(geometry.kv_shape(), total));
        let admit = |required: u64| -> CoreResult<()> {
            let available = core_llm::effective_memory_budget(
                core_llm::available_host_memory_bytes(),
                core_llm::operational_memory_override()?,
            )?;
            core_llm::admit_request_memory_with_geometry(
                admitted_prompt,
                req.max_new_tokens,
                self.descriptor.capabilities.max_context_tokens,
                required,
                available,
            )
        };
        admit(required)?;

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

        let mut kv_report = match &kv_plan {
            KvPlan::Dense(report) => Some(report.clone()),
            KvPlan::Unreported | KvPlan::Compressed { .. } => None,
        };

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: map_sampling(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };

        let mut mtp_draft_tokens = match req.mtp {
            MtpMode::Off => None,
            MtpMode::Auto => self
                .descriptor
                .capabilities
                .mtp
                .map(|caps| caps.recommended_draft_tokens as usize),
            MtpMode::Enabled { draft_tokens } => Some(draft_tokens as usize),
        };
        // Gemma 4 has no Qwen predictor. Qwen multimodal prompts use the fused-embedding MTP route
        // below, including their explicit three-axis positions and continuation delta.
        if mtp_draft_tokens.is_some() && gemma4_mm_request {
            if matches!(req.mtp, MtpMode::Enabled { .. }) {
                return Err(CoreError::Unsupported(
                    "[mlx-llama] native Qwen MTP is unavailable for Gemma 4 prompts".into(),
                ));
            }
            mtp_draft_tokens = None;
        }

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
        let mut stop_matcher = StopMatcher::new(req.stop.iter().cloned());
        let stop_active = !stop_matcher.is_empty();
        // A single-threaded latch the detok sink trips on a stop hit; the decode loop reads it after
        // each token via `should_stop` and halts with `FinishReason::Stopped`.
        let halt = std::cell::Cell::new(false);
        // Content emitted as the matchers release it — the result text when stop strings or thinking
        // are active (the plain path still decodes all tokens, byte-identical to before).
        let mut streamed = String::new();
        // Reasoning text, accumulated from the Thinking channel (sc-7585).
        let mut thinking_buf = String::new();
        // Contract token index: a running counter over *emitted* events, not the raw decode step —
        // detok hold-backs and stripped `<think>`/`</think>` marker tokens produce no event, so this
        // stays gap-free (and equals the step in the common one-delta-per-token case).
        let mut emit_index = 0usize;
        let mut last_id = 0u32; // id of the last emitted token, for flushed-tail events

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
        let mut tool_seg = matches!(self.tool_call_format, Some(ToolCallFormat::Tagged))
            .then(|| ToolCallSegmenter::new(&req.tools));

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix). The
        // `IncrementalDetok` guard holds back lossy U+FFFD placeholders so a multi-byte character
        // split across BPE tokens streams intact (and never panics a mid-char slice) — sc-12452.
        // The segmenter (when active) splits each delta into reasoning vs answer; answer text then
        // feeds the stop matcher so a stop string is trimmed and halts generation.
        let tokenizer = &self.tokenizer;
        let (out, mtp_stats, timing) = {
            let mut acc: Vec<u32> = Vec::new();
            let mut detok = IncrementalDetok::new();
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, .. } = ev {
                    let id = id as u32;
                    acc.push(id);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if let Some(delta) = detok.push(&text) {
                            let delta = delta.to_string();
                            match segmenter.as_mut() {
                                Some(seg) => {
                                    for span in seg.push(&delta) {
                                        match span.channel {
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
                                            Channel::Content => {
                                                // Answer text → tool segmenter (lifts out tool-call
                                                // blocks) → stop matcher → emit.
                                                for piece in tool_pieces(&mut tool_seg, &span.text)
                                                {
                                                    emit_content(
                                                        &piece,
                                                        id,
                                                        &mut stop_matcher,
                                                        &mut streamed,
                                                        &mut emit_index,
                                                        &mut last_id,
                                                        &halt,
                                                        &mut *on_event,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {
                                    for piece in tool_pieces(&mut tool_seg, &delta) {
                                        emit_content(
                                            &piece,
                                            id,
                                            &mut stop_matcher,
                                            &mut streamed,
                                            &mut emit_index,
                                            &mut last_id,
                                            &halt,
                                            &mut *on_event,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            };
            let should_stop = || halt.get();
            let should_stop_opt = stop_active.then_some(&should_stop as &dyn Fn() -> bool);
            match &mm {
                // Multimodal: prefill the spliced embeds with interleaved M-RoPE + DeepStack fusion,
                // then decode the continuation (text positions shifted by `mrope_delta`) through the
                // shared loop — one path for either backbone via the `VlmDecode` seam.
                Some(m) => {
                    let (t, h, w, delta) = &m.positions;
                    let pos = [t.as_slice(), h.as_slice(), w.as_slice()];
                    if let (Decoder::Qwen35(model), Some(num_draft)) =
                        (&self.model, mtp_draft_tokens)
                    {
                        let prompt = Qwen35MtpMultimodalPrompt {
                            input_ids: &m.expanded_ids,
                            embeddings: &m.embeds,
                            positions: pos,
                            visual_pos_mask: &m.visual_pos_mask,
                            deepstack: &m.deepstack,
                            continuation_delta: *delta,
                        };
                        let (timed, stats) = generate_qwen35_mtp_multimodal_with_timings(
                            model,
                            &prompt,
                            &config,
                            num_draft,
                            &req.cancel,
                            &mut sink,
                            json_mask
                                .as_mut()
                                .map(|m| m as &mut dyn RewindableConstraintMask),
                            should_stop_opt,
                            qwen_prefill_started
                                .expect("Qwen multimodal preparation starts the prefill clock"),
                        )
                        .map_err(to_core)?;
                        (timed.output, Some(stats), Some(timed.timer))
                    } else {
                        let model = self.model.as_vlm();
                        let mut cache = model.make_cache();
                        let first = model
                            .prefill_with_deepstack(
                                &m.embeds,
                                pos,
                                cache.as_mut(),
                                &m.visual_pos_mask,
                                &m.deepstack,
                            )
                            .map_err(to_core)?;
                        let shifted = Shifted {
                            model,
                            delta: *delta,
                        };
                        let timed = generate_from_prefill_with_timings(
                            &shifted,
                            cache.as_mut(),
                            first,
                            m.expanded_ids.clone(),
                            &config,
                            &req.cancel,
                            &mut sink,
                            json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                            should_stop_opt,
                            qwen_prefill_started
                                .expect("Qwen multimodal preparation starts the prefill clock"),
                        )
                        .map_err(to_core)?;
                        (timed.output, None, Some(timed.timer))
                    }
                }
                // Gemma 4 multimodal: prefill the spliced embeds on ordinary causal 1-D positions
                // (no M-RoPE, so no position shift for the continuation), then decode through the
                // shared loop against the unwrapped decoder.
                None => match &g4 {
                    Some(m) => {
                        let model = match &self.model {
                            Decoder::Causal(c) => c,
                            Decoder::Qwen35(_) => {
                                return Err(CoreError::Load(
                                    "gemma 4: the multimodal path requires the generic causal \
                                     decoder"
                                        .into(),
                                ))
                            }
                        };
                        let mut cache = model.new_cache();
                        let first = model
                            .decode_logits_from_embeds(&m.embeds, &mut cache, 0)
                            .map_err(to_core)?;
                        let out = generate_from_prefill(
                            &self.model,
                            &mut cache,
                            first,
                            m.expanded_ids.clone(),
                            &config,
                            &req.cancel,
                            &mut sink,
                            json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                            should_stop_opt,
                        )
                        .map_err(to_core)?;
                        (out, None, None)
                    }
                    None => {
                        if let Some(observer) = observer {
                            if mtp_draft_tokens.is_some() {
                                return Err(CoreError::Unsupported(
                                    "campaign observer cannot measure MTP decode".into(),
                                ));
                            }
                            let packed_decoder = match (packed, &self.model) {
                                (None, _) => None,
                                (Some(arm), Decoder::Causal(model)) => {
                                    Some(PackedCampaignDecoder {
                                        model,
                                        arm,
                                        selection_fallbacks: RefCell::new(Vec::new()),
                                    })
                                }
                                (Some(_), Decoder::Qwen35(_)) => {
                                    return Err(CoreError::Unsupported(
                                        "compressed campaign decode requires the causal decoder"
                                            .into(),
                                    ))
                                }
                            };
                            let decoder: &dyn Decode = match &packed_decoder {
                                Some(packed) => packed,
                                None => &self.model,
                            };
                            let output = match (prefix_cache, &self.model) {
                                // The product prompt-cache path (SC-20671 multi-turn fixture):
                                // a hit seeds the dense cache or imports into the packed one.
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
                                        "the campaign prompt cache requires the causal decoder"
                                            .into(),
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
                            (output, None, None)
                        } else {
                            match (&self.model, mtp_draft_tokens) {
                                (Decoder::Qwen35(model), Some(num_draft)) => {
                                    let (timed, stats) = generate_qwen35_mtp_with_timings(
                                        model,
                                        &prompt_ids,
                                        &config,
                                        num_draft,
                                        &req.cancel,
                                        &mut sink,
                                        json_mask
                                            .as_mut()
                                            .map(|m| m as &mut dyn RewindableConstraintMask),
                                        should_stop_opt,
                                    )
                                    .map_err(to_core)?;
                                    (timed.output, Some(stats), Some(timed.timer))
                                }
                                (Decoder::Causal(model), _) => {
                                    if let KvPlan::Compressed { format, reader } = &kv_plan {
                                        // The compressed cache is selected before any K/V
                                        // mutation and its evidence read before it is reset.
                                        let (mut cache, refused) =
                                            crate::kv_policy::select_compressed_cache(
                                                model,
                                                reader.clone(),
                                                prompt_ids.len(),
                                                crate::kv_policy::dense_transition_admission(
                                                    dense_final_kv.ok_or_else(|| {
                                                        CoreError::InvalidRequest(
                                                            "request memory estimate overflow"
                                                                .into(),
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
                                        // A refused selection runs dense from the start: the
                                        // request was admitted at the compressed price, so it
                                        // is admitted again at the dense price before any K/V
                                        // exists (sc-20682).
                                        if refused.is_some() {
                                            admit(estimate(None)?)?;
                                        }
                                        let timed = generate_with_timings_on(
                                            &self.model,
                                            cache.as_mut(),
                                            &prompt_ids,
                                            &config,
                                            &req.cancel,
                                            &mut sink,
                                            json_mask
                                                .as_mut()
                                                .map(|m| m as &mut dyn ConstraintMask),
                                            should_stop_opt,
                                        );
                                        let report = timed.as_ref().ok().map(|_| {
                                            crate::kv_policy::compressed_report(
                                                *format,
                                                refused,
                                                cache.as_ref(),
                                            )
                                        });
                                        let reset = cache.reset();
                                        let timed = timed.map_err(to_core)?;
                                        reset.map_err(to_core)?;
                                        kv_report = report.transpose().map_err(to_core)?;
                                        (timed.output, None, Some(timed.timer))
                                    } else {
                                        let timed = generate_with_timings(
                                            &self.model,
                                            &prompt_ids,
                                            &config,
                                            &req.cancel,
                                            &mut sink,
                                            json_mask
                                                .as_mut()
                                                .map(|m| m as &mut dyn ConstraintMask),
                                            should_stop_opt,
                                        )
                                        .map_err(to_core)?;
                                        (timed.output, None, Some(timed.timer))
                                    }
                                }
                                (Decoder::Qwen35(_), None) => {
                                    let timed = generate_with_timings(
                                        &self.model,
                                        &prompt_ids,
                                        &config,
                                        &req.cancel,
                                        &mut sink,
                                        json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                                        should_stop_opt,
                                    )
                                    .map_err(to_core)?;
                                    (timed.output, None, Some(timed.timer))
                                }
                            }
                        }
                    }
                },
            }
        };

        // End-of-generation tails, in pipeline order. First the thinking segmenter's held-back
        // partial marker (it turned out not to begin a marker) as current-channel text — reasoning
        // straight out, answer through the tool segmenter; then the tool segmenter's own tail (held
        // partial `<tool_call>` / an unterminated block surfaced as content); then the stop matcher's
        // held-back partial stop.
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
                                &piece,
                                last_id,
                                &mut stop_matcher,
                                &mut streamed,
                                &mut emit_index,
                                &mut last_id,
                                &halt,
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
                    &piece,
                    last_id,
                    &mut stop_matcher,
                    &mut streamed,
                    &mut emit_index,
                    &mut last_id,
                    &halt,
                    &mut *on_event,
                );
            }
        }
        // If generation ended for any reason other than a stop string, flush the stop matcher's
        // held-back tail (a partial stop-prefix that never completed) — real output to stream/return.
        if stop_active && !halt.get() {
            let tail = stop_matcher.flush();
            if !tail.is_empty() {
                streamed.push_str(&tail);
                on_event(CoreEvent::Token {
                    id: last_id,
                    text: tail,
                    index: emit_index,
                    channel: Channel::Content,
                });
            }
        }

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
            mtp: mtp_stats.map(|stats| core_llm::MtpStats {
                proposed_tokens: u32::try_from(stats.proposed).unwrap_or(u32::MAX),
                accepted_tokens: u32::try_from(stats.accepted).unwrap_or(u32::MAX),
                target_forwards: u32::try_from(stats.forwards).unwrap_or(u32::MAX),
            }),
            decode: None,
            finish_reason: Some(finish),
            kv_cache: kv_report,
        })
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
    d
}

/// The loaded-model descriptor for the hybrid Qwen3.6 (`qwen3_5`) decoder (parsed via
/// [`Qwen35Config`], which `ModelConfig` does not represent).
fn descriptor_for_qwen35(cfg: &Qwen35Config) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = Architecture::Qwen35.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    if cfg.mtp_num_hidden_layers > 0 {
        d.capabilities.mtp = Some(core_llm::MtpCapabilities {
            max_draft_tokens: u32::MAX,
            recommended_draft_tokens: 3,
        });
    }
    d
}

/// Read and parse `config.json` from a snapshot directory into a JSON value (used to dispatch the
/// architecture before constructing the architecture-specific config).
fn read_config_value(dir: &Path) -> CoreResult<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join("config.json"))
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
    let text = std::fs::read_to_string(dir.join(name)).ok()?;
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

fn map_sampling(s: &Sampling) -> SamplingParams {
    SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        presence_penalty: s.presence_penalty,
        repetition_penalty: s.repetition_penalty,
        repetition_context: s.repetition_context,
    }
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
/// Apple-Silicon Metal allocations are rounded to 16-KiB VM pages. Gated DeltaNet retains one
/// independently allocated output row per prompt token until its final concatenate.
const MLX_ALLOCATION_PAGE_BYTES: u64 = 16 * 1024;
/// The recurrence creates five one-element index buffers per step and evaluates every 256 steps.
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

    // PrismLinear builds cast -> sign multiply -> Hadamard -> cast graphs independently for each
    // projection. The full-attention block has the wider simultaneous input-width sum; the linear
    // block uses two mixer input rotations plus its output rotation. Four buffers per input width
    // safely covers the F32 chain and the BF16 result even when donation is unavailable.
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
        if mtp_width > 0 { 3 } else { 1 },
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
                if mtp_width > 0 { 3 } else { 1 },
                compressed_kv_bytes,
            )
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
    match contract {
        MlxWorkspaceContract::Eager => core_llm::estimate_request_bytes(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace_bytes,
            mtp_width,
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
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
        non_thinking: Sampling {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            presence_penalty: 1.5,
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

/// Bridge an engine error into the contract error, preserving the typed cancellation / capability
/// variants (do not stringify those).
fn to_core(e: crate::Error) -> CoreError {
    match e {
        crate::Error::Canceled => CoreError::Canceled,
        crate::Error::Unsupported(m) => CoreError::Unsupported(m),
        crate::Error::MissingTensor(m) => CoreError::Load(format!("missing tensor: {m}")),
        crate::Error::Config(m) => CoreError::Load(m),
        crate::Error::ResourceExhausted(evidence) => CoreError::RequestResourceExhausted(evidence),
        crate::Error::Io(e) => CoreError::Io(e),
        other => CoreError::backend(other),
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
mod tests {
    use super::*;
    use serde_json::json;

    fn admit_any_transition() -> crate::primitives::DenseTransitionAdmission {
        crate::primitives::DenseTransitionAdmission::new(|_| Ok(()))
    }

    #[derive(Default)]
    struct RecordingLifecycleObserver(Vec<(&'static str, u64)>);

    impl crate::campaign::Observer for RecordingLifecycleObserver {
        fn phase(&mut self, _name: &'static str) {}

        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}

        fn cache_snapshot(
            &mut self,
            bytes: u64,
            _tokens: u64,
            _capacity: u64,
            _element_bytes: u64,
        ) {
            self.0.push(("persistent", bytes));
        }

        fn release_event(&mut self, _kind: &'static str, _role: &'static str, bytes: u64) {
            self.0.push(("released", bytes));
        }
    }

    #[test]
    fn cancellation_capture_replays_persistent_then_release_without_allocation_aliasing() {
        let mut capture = CacheLifecycleCapture::default();
        crate::campaign::Observer::cache_snapshot(&mut capture, 64, 8, 8, 4);
        crate::campaign::Observer::release_event(&mut capture, "cache_release", "cache", 64);
        assert!(matches!(
            capture.events.first(),
            Some(CapturedCacheLifecycle::Snapshot(64, 8, 8, 4))
        ));
        let mut observer = RecordingLifecycleObserver::default();
        capture.replay(&mut observer);
        assert_eq!(observer.0, vec![("persistent", 64), ("released", 64)]);
    }

    #[test]
    fn campaign_cancellation_probe_is_bounded_content_without_tools_or_thinking() {
        let request = campaign_cancellation_probe_request();
        assert_eq!(request.messages.len(), 1);
        assert!(request.tools.is_empty());
        assert!(request.stop.is_empty());
        assert!(request.constraint.is_none());
        assert_eq!(request.thinking, ThinkingMode::Disabled);
        assert_eq!(request.max_new_tokens, 16);
        assert!(request.sampling.is_greedy());
    }

    #[test]
    fn qwen35_dense_checkpoint_selects_one_decoder_root() {
        let select = |keys: &[&str]| qwen35_dense_prefix(|key| keys.contains(&key));
        assert_eq!(
            select(&["model.language_model.embed_tokens.weight"]).unwrap(),
            "model.language_model"
        );
        assert_eq!(
            select(&["model.embed_tokens.weight", "mtp.fc.weight"]).unwrap(),
            "model"
        );
        assert!(select(&[]).is_err());
        assert!(select(&[
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight"
        ])
        .is_err());
    }

    fn frozen_dense_qwen35_config() -> Qwen35Config {
        Qwen35Config::from_json(&json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "intermediate_size": 17_408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248_320,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4
        }))
        .unwrap()
    }

    #[test]
    fn fused_request_estimate_tracks_chunked_attention_and_last_row_logits() {
        // Frozen Qwen3.8 parent geometry. This prompt size reproduces the campaign's long-context
        // scale: eager prompt-squared scores dominate hundreds of GB, while MLX runs eight query
        // rows per fused call and projects one final row to the vocabulary.
        let geometry = LlmMemoryGeometry {
            query_heads: 40,
            kv_heads: 4,
            head_dim: 128,
            layers: 64,
            element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, false),
            score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: 0,
        };
        let fused =
            estimate_mlx_request_bytes(29_600, 128, geometry, 0, 0, MlxWorkspaceContract::Chunked)
                .unwrap();
        let eager =
            estimate_mlx_request_bytes(29_600, 128, geometry, 0, 0, MlxWorkspaceContract::Eager)
                .unwrap();
        assert!(fused < 36_000_000_000, "bounded MLX peak: {fused}");
        assert!(eager > 400_000_000_000, "quadratic eager peak: {eager}");
        assert_eq!(
            eager,
            core_llm::estimate_request_bytes(29_600, 128, geometry, 0, 0).unwrap(),
            "non-fused paths retain the shared fail-closed estimate"
        );
    }

    /// The shared tiled estimate with the attention term `sdpa`'s routing implies for `geometry`.
    fn routed_estimate(prompt: usize, new_tokens: u32, geometry: LlmMemoryGeometry) -> u64 {
        let attention = prefill_attention_tile_bytes(
            prompt as u64,
            geometry.query_heads,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.score_element_bytes,
        )
        .unwrap();
        core_llm::estimate_tiled_request_bytes_with_recurrent_copies(
            prompt, new_tokens, geometry, 0, 0, attention, 1,
        )
        .unwrap()
    }

    /// sc-20676: the chunked contract prices the tile `sdpa` actually runs — a `rows × k_len` mask
    /// slice for a full-kernel head dim, a per-head score set over `rows · gqa ≤ 32` rows otherwise
    /// (Phi-3's head dim 96 used to be priced as 8-row tiles while running one unfused call).
    #[test]
    fn chunked_contract_prices_the_routed_attention_tile() {
        let with_head_dim = |head_dim, query_heads, kv_heads| LlmMemoryGeometry {
            query_heads,
            kv_heads,
            head_dim,
            layers: 32,
            element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, false),
            score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
            hidden_size: 3072,
            intermediate_size: 8192,
            vocab_size: 32_064,
            recurrent_bytes: 0,
        };
        let prompt = 16_384usize;
        for (head_dim, query_heads, kv_heads) in [(128, 24, 8), (96, 32, 32), (256, 16, 2)] {
            let geometry = with_head_dim(head_dim, query_heads, kv_heads);
            let estimate = estimate_mlx_request_bytes(
                prompt,
                16,
                geometry,
                0,
                0,
                MlxWorkspaceContract::Chunked,
            )
            .unwrap();
            assert_eq!(
                estimate,
                routed_estimate(prompt, 16, geometry),
                "head_dim {head_dim}"
            );
        }
        // The full-kernel block's mask slice (16k × 2048 × 4 B) outweighs the old 8-row tile.
        let block = routed_estimate(prompt, 16, with_head_dim(128, 24, 8));
        let old_tile = core_llm::estimate_chunked_request_bytes(
            prompt,
            16,
            with_head_dim(128, 24, 8),
            0,
            0,
            8,
        )
        .unwrap();
        assert!(block > old_tile);
    }

    #[test]
    fn fused_request_estimate_is_checked_and_preserves_mtp_and_media_costs() {
        let geometry = LlmMemoryGeometry {
            query_heads: 8,
            kv_heads: 2,
            head_dim: 64,
            layers: 4,
            element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, false),
            score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
            hidden_size: 256,
            intermediate_size: 512,
            vocab_size: 1024,
            recurrent_bytes: 4096,
        };
        let plain =
            estimate_mlx_request_bytes(128, 16, geometry, 0, 0, MlxWorkspaceContract::Chunked)
                .unwrap();
        let media = estimate_mlx_request_bytes(
            128,
            16,
            geometry,
            123_456,
            0,
            MlxWorkspaceContract::Chunked,
        )
        .unwrap();
        let mtp =
            estimate_mlx_request_bytes(128, 16, geometry, 0, 3, MlxWorkspaceContract::Chunked)
                .unwrap();
        assert_eq!(media - plain, 123_456);
        assert!(mtp > plain);
        assert!(estimate_mlx_request_bytes(
            usize::MAX,
            u32::MAX,
            geometry,
            0,
            3,
            MlxWorkspaceContract::Chunked,
        )
        .is_none());
    }

    #[test]
    fn qwen35_prism_estimate_covers_frozen_allocator_peaks_and_rejects_remote_long_context() {
        let config = frozen_dense_qwen35_config();
        let geometry = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            layers: 64,
            element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, true),
            score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: 207_618_048,
        };
        let contract = MlxWorkspaceContract::Qwen35 {
            config: &config,
            prism: true,
        };
        let arithmetic = estimate_mlx_request_bytes(27, 128, geometry, 0, 0, contract).unwrap();
        let context_64 = estimate_mlx_request_bytes(1_187, 128, geometry, 0, 0, contract).unwrap();
        let context_512 = estimate_mlx_request_bytes(9_251, 128, geometry, 0, 0, contract).unwrap();
        let context_2048 =
            estimate_mlx_request_bytes(36_899, 128, geometry, 0, 0, contract).unwrap();

        // Frozen local probe at 02ed7e958: arithmetic raised the process-global allocator peak to
        // 9,321,652,712 from a captured post-load active 8,602,141,760 (a 719,510,952-byte lower
        // bound). Subtracting the stable post-request residency 8,548,422,952 gives 773,229,760;
        // price that stricter observed bound even though it is not an immediately-pre-request
        // sample. Frozen campaign 35461924246 supplies the two longer-context deltas below.
        assert!(arithmetic >= 773_229_760, "estimate: {arithmetic}");
        assert!(context_64 >= 4_091_010_468, "estimate: {context_64}");
        assert!(context_512 >= 22_885_373_536, "estimate: {context_512}");
        // The Prism hold (see `priced_compute_element_bytes`) exceeds the compute-width estimate
        // by exactly the K/V, activation and last-row logit terms at the width difference; every
        // other term is width-independent.
        let compute_width = LlmMemoryGeometry {
            element_bytes: dtype_bytes(Dtype::Bfloat16),
            ..geometry
        };
        let delta = geometry.element_bytes - compute_width.element_bytes;
        for (prompt, estimate) in [
            (27_u64, arithmetic),
            (1_187, context_64),
            (9_251, context_512),
            (36_899, context_2048),
        ] {
            let g = geometry;
            let kv = (prompt + 128) * g.layers * g.kv_heads * g.head_dim * delta * 2;
            let activations = prompt * (g.intermediate_size * 3 + g.hidden_size * 8) * delta
                + g.vocab_size * delta;
            let at_compute_width =
                estimate_mlx_request_bytes(prompt as usize, 128, compute_width, 0, 0, contract)
                    .unwrap();
            assert_eq!(
                estimate - at_compute_width,
                kv + activations,
                "prompt {prompt}"
            );
        }
        assert!(arithmetic < context_64 && context_64 < context_512 && context_512 < context_2048);
        assert!(
            core_llm::admit_request_memory(context_2048, 47_922_610_176).is_err(),
            "the 36,899-token request must not be admitted on the failed remote's observed budget"
        );
    }

    #[test]
    fn qwen35_workspace_estimate_is_checked_and_keeps_ordinary_requests_available() {
        let config = frozen_dense_qwen35_config();
        let geometry = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            layers: 64,
            element_bytes: priced_compute_element_bytes(Dtype::Bfloat16, true),
            score_element_bytes: EAGER_SCORE_ELEMENT_BYTES,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: 207_618_048,
        };
        let contract = MlxWorkspaceContract::Qwen35 {
            config: &config,
            prism: true,
        };
        // A one-token request exercises the pre-evaluation recurrence branch without applying a
        // 256-step floor. A normal 128-token prompt remains admitted on an 8-GB machine.
        let scalar = estimate_mlx_request_bytes(1, 1, geometry, 0, 0, contract).unwrap();
        let ordinary = estimate_mlx_request_bytes(128, 128, geometry, 0, 0, contract).unwrap();
        for (prompt, new_tokens, estimate) in [(1, 1, scalar), (128, 128, ordinary)] {
            assert_eq!(
                estimate,
                routed_estimate(prompt, new_tokens, geometry)
                    + estimate_qwen35_workspace_extra_bytes(prompt, &config, true).unwrap(),
                "prompt {prompt}"
            );
        }
        assert!(scalar < ordinary);
        assert!(core_llm::admit_request_memory(ordinary, 8_000_000_000).is_ok());
        assert!(
            estimate_mlx_request_bytes(usize::MAX, u32::MAX, geometry, u64::MAX, 3, contract,)
                .is_none()
        );
    }

    #[test]
    fn reasoning_boundary_commits_same_token_json_in_release_builds() {
        let table = core_llm::ConstraintDecodeTable {
            pieces: vec!["</think>{}".into(), String::new(), "{".into()],
            special: [1].into_iter().collect(),
        };
        let mut mask = JsonMask::new(&table, [1], true);
        assert!(mask.allowed()[0]);
        mask.accept(0);
        assert!(
            mask.allowed()[1],
            "same-token answer must commit even with debug assertions off"
        );
        assert!(
            !mask.allowed()[2],
            "completed answer cannot begin a second JSON value"
        );
        let mark = mask.checkpoint();
        mask.rewind(mark);
        assert!(
            mask.allowed()[1],
            "MTP rewind must replay phase and same-token JSON"
        );
    }

    #[test]
    fn json_constraint_waits_for_split_reasoning_close_and_rewinds_phase() {
        let table = ConstraintDecodeTable {
            pieces: vec![
                "reason".into(),
                "</thi".into(),
                "nk>".into(),
                "{".into(),
                "x".into(),
                "</think>x".into(),
                String::new(),
            ],
            special: [6].into_iter().collect(),
        };
        let mut mask = JsonMask::new(&table, [6], true);
        assert!(mask.allowed()[0]);
        assert!(
            !mask.allowed()[5],
            "invalid same-token JSON suffix is masked"
        );
        assert!(!mask.allowed()[6], "EOS cannot end an open reasoning block");
        mask.accept(0);
        let checkpoint = mask.checkpoint();
        mask.accept(1);
        mask.accept(2);
        assert!(mask.allowed()[3], "JSON begins only after split </think>");
        assert!(!mask.allowed()[4]);
        mask.rewind(checkpoint);
        assert!(mask.allowed()[1], "rewind restores the reasoning phase");
        assert!(!mask.allowed()[6]);

        let mut disabled = JsonMask::new(&table, [6], false);
        assert!(disabled.allowed()[3], "disabled thinking starts in JSON");
        assert!(!disabled.allowed()[0]);
    }

    #[test]
    fn reasoning_controls_are_advertised_only_when_template_names_them() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ enable_thinking }} {{ tool_call }}",
        )
        .unwrap();
        let (_, thinking, effort, preserve, tools, _) = load_chat_template(dir.path());
        assert!(thinking);
        assert!(!effort);
        assert!(!preserve);
        assert!(tools);

        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ enable_thinking }} {{ reasoning_effort }} {{ preserve_thinking }}",
        )
        .unwrap();
        let (_, thinking, effort, preserve, tools, _) = load_chat_template(dir.path());
        assert!(thinking);
        assert!(effort);
        assert!(preserve);
        assert!(!tools);
    }

    #[test]
    fn frozen_qwen38_generation_config_uses_both_official_stop_tokens() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../docs/reference/qwen38/generation_config.json"
            )),
        )
        .unwrap();

        assert_eq!(eos_token_ids(dir.path()), vec![248046, 248044]);
    }

    fn qwen36_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5", "depth": 27 }
        })
    }
    fn llava_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
            "text_config": { "model_type": "llama" },
            "vision_config": { "model_type": "siglip_vision_model" }
        })
    }
    fn qwen3vl_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": { "model_type": "qwen3_vl_text" },
            "vision_config": { "model_type": "qwen3_vl", "depth": 27 }
        })
    }

    #[test]
    fn text_provider_claims_qwen36_but_cedes_llava() {
        // Qwen3.6 (Qwen-VL wrapper): claimed by the text provider, served text-only.
        assert!(can_load_value(&qwen36_wrapper()));
        // Qwen3-VL (Qwen-VL wrapper): also claimed by `mlx-llama` (its nested standard Qwen3 decoder).
        assert!(can_load_value(&qwen3vl_wrapper()));
        // LLaVA: ceded to the JoyCaption vision provider.
        assert!(!can_load_value(&llava_wrapper()));
        // Plain text models (no vision_config) are unaffected.
        assert!(can_load_value(
            &json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })
        ));
        assert!(can_load_value(
            &json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })
        ));
    }

    #[test]
    fn tool_format_distinguishes_tagged_and_llama_bare_json() {
        assert_eq!(
            tool_call_format("emit <tool_call>...</tool_call>"),
            Some(ToolCallFormat::Tagged)
        );
        assert_eq!(
            tool_call_format(
                "message.tool_calls; please respond with JSON for a function call exactly"
            ),
            Some(ToolCallFormat::BareJson)
        );
        assert_eq!(tool_call_format("message.tool_calls only"), None);

        let tools = [core_llm::ToolSpec::new(
            "record_baseline_fact",
            "Record a fact",
            json!({
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"]
            }),
        )];
        let calls = bare_json_tool_calls(
            r#"{"name":"record_baseline_fact","parameters":{"fact":"SC20671 structured fixture"}}"#,
            &tools,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "record_baseline_fact");
        assert_eq!(
            calls[0].arguments.get("fact"),
            Some(&json!("SC20671 structured fixture"))
        );
        assert!(bare_json_tool_calls("ordinary answer", &tools).is_empty());
    }

    #[test]
    fn weightless_vision_advertises_qwen_vl_wrappers_only() {
        // The sc-8077 weightless vision gate: `mlx-llama` advertises vision (pre-load, config.json
        // only) for a Qwen-VL wrapper so a model-first vision-required load resolves here.
        // Qwen3-VL: a vision_config + qwen3_vl arch ⇒ vision-capable.
        assert!(weightless_vision_value(&qwen3vl_wrapper()));
        // Qwen3.6 hybrid (qwen3_5) Qwen-VL wrapper: likewise vision-capable.
        assert!(weightless_vision_value(&qwen36_wrapper()));
        // A LLaVA snapshot is ceded to JoyCaption (can_load=false here) ⇒ NOT advertised, so a
        // Qwen3-VL load can never be mistaken for / misrouted via this provider's vision path.
        assert!(!weightless_vision_value(&llava_wrapper()));
        // Plain text checkpoints (no vision_config) ⇒ NOT vision-capable (the static descriptor
        // already reports supports_vision=false; the probe must not flip it on).
        assert!(!weightless_vision_value(
            &json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })
        ));
        assert!(!weightless_vision_value(
            &json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })
        ));
    }

    /// Locate a staged Qwen3-VL-8B-Instruct snapshot for the chat-template oracle from the explicit
    /// passed-in `QWEN3VL_SNAPSHOT` env path. Inference never self-fetches or derives a cache location
    /// (epic 13657). `None` ⇒ the gated tests self-skip cleanly (CI sets the var for this story).
    fn qwen3vl_snapshot_dir() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(std::env::var("QWEN3VL_SNAPSHOT").ok()?);
        path.exists().then_some(path)
    }

    /// Build the representative chat-message sets the oracle pins (text-only, system+user, single
    /// image, multi-image/mixed, multi-turn). Image content is a `Content::Image` (a 1×1 black pixel
    /// placeholder); the byte-match exercises the same `substitute_image_placeholders` path the
    /// provider uses, so the rendered single `<|image_pad|>` per image must match HF.
    fn oracle_messages(case: &str) -> Vec<Message> {
        let img = || Content::Image(ImageRef::new(1, 1, vec![0, 0, 0]).unwrap());
        let user = |content: Vec<Content>| Message {
            role: core_llm::Role::User,
            content,
            thinking: None,
            tool_calls: Vec::new(),
        };
        let sys = |t: &str| Message {
            role: core_llm::Role::System,
            content: vec![Content::text(t)],
            thinking: None,
            tool_calls: Vec::new(),
        };
        let asst = |t: &str| Message {
            role: core_llm::Role::Assistant,
            content: vec![Content::text(t)],
            thinking: None,
            tool_calls: Vec::new(),
        };
        match case {
            "text_only" => vec![user(vec![Content::text("What is the capital of France?")])],
            "system_user_text" => {
                vec![
                    sys("You are a helpful assistant."),
                    user(vec![Content::text("Hello!")]),
                ]
            }
            "single_image" => vec![user(vec![img(), Content::text("Describe this image.")])],
            "multi_image_mixed" => vec![user(vec![
                Content::text("Compare:"),
                img(),
                Content::text("and"),
                img(),
                Content::text("please."),
            ])],
            "multi_turn" => vec![
                user(vec![Content::text("Hi")]),
                asst("Hello there!"),
                user(vec![Content::text("How are you?")]),
            ],
            other => panic!("unknown oracle case {other}"),
        }
    }

    /// Byte-match oracle (sc-8075 AC #1): the engine's chat-template + image-placeholder + tokenize
    /// path must reproduce the pinned HF `apply_chat_template` prompt token ids **exactly** for the
    /// representative messages — text-only, single-image, and multi-image/mixed. The single
    /// `<|image_pad|>` (151655) per image is what the processor emits *before* patch-count expansion;
    /// these fixtures pin that pre-expansion prompt. Self-skips cleanly when the snapshot is absent.
    #[test]
    fn chat_template_byte_matches_hf_processor() {
        let Some(dir) = qwen3vl_snapshot_dir() else {
            eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
            return;
        };
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "models/testdata/qwen3vl_chat_template_oracle.json"
        ))
        .expect("parse chat-template oracle");

        // Sanity: the dispatch and config see Qwen3-VL (and parse the nested 256K-context text decoder).
        let cfg_value = read_config_value(&dir).expect("read config.json");
        assert_eq!(
            Architecture::from_config(&cfg_value).unwrap(),
            Architecture::Qwen3Vl,
            "snapshot must dispatch to Qwen3-VL"
        );
        let cfg = ModelConfig::from_json(&cfg_value).expect("parse Qwen3-VL text config");
        assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
        assert_eq!(cfg.max_position_embeddings, 262144, "256K context");

        let (template, _, _, _, _, _) = load_chat_template(&dir);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        for (case, expected) in oracle["cases"].as_object().unwrap() {
            let want: Vec<i32> = expected["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            // Drive the exact provider path: substitute image content with the Qwen-VL placeholder,
            // render the model's own Jinja template (add_generation_prompt), then tokenize without
            // auto special tokens.
            let messages = oracle_messages(case);
            let substituted = substitute_vision_placeholders(&messages, 2)
                .expect("the oracle cases carry no audio, so substitution cannot fail");
            let prompt = template
                .render_with(
                    &substituted,
                    &RenderOptions {
                        add_generation_prompt: true,
                        enable_thinking: None,
                        reasoning_effort: None,
                        preserve_thinking: None,
                        tools: &[],
                    },
                )
                .expect("render");
            assert_eq!(
                prompt,
                expected["text"].as_str().unwrap(),
                "case {case}: rendered prompt string must byte-match HF"
            );
            let got: Vec<i32> = tokenizer
                .encode(&prompt, false)
                .expect("encode")
                .into_iter()
                .map(|id| id as i32)
                .collect();
            assert_eq!(
                got, want,
                "case {case}: tokenized prompt ids must byte-match HF processor"
            );
        }
    }

    #[test]
    fn prompt_opens_thinking_matches_template_modes() {
        // Qwen3.6 thinking/auto generation prompt: opens the block, leaves it unclosed.
        assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
        // Disabled mode renders a *closed* empty block: must not prime.
        assert!(!prompt_opens_thinking(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        // A prior closed reasoning turn followed by a fresh open block still opens.
        assert!(prompt_opens_thinking(
            "<think>\nold\n</think>\n\nq<|im_start|>assistant\n<think>\n"
        ));
        // No reasoning markers at all (non-thinking template).
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
    }

    // --- Qwen3-VL video Text–Timestamp-Alignment oracle (tools/gen_qwen3vl_video_oracle.py) --------

    fn qwen3vl_video_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("models/testdata/qwen3vl_video_oracle.json")).unwrap()
    }

    /// **Merged per-frame timestamps match `Qwen3VLProcessor._calculate_timestamps`.** Given the
    /// sampled `frames_indices` + `fps`, the per-temporal-patch averaged timestamps must equal the HF
    /// reference exactly — the values that feed the `<{t:.1f} seconds>` Text–Timestamp-Alignment tags.
    #[test]
    fn video_merged_timestamps_match_hf_reference() {
        let j = qwen3vl_video_oracle();
        let fps = j["fps"].as_f64().unwrap() as f32;
        let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
        let indices: Vec<f32> = j["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        // Per-sample timestamps are `idx / fps` (matching the reference, which averages these).
        let per_sample: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
        let got = merged_frame_timestamps(&per_sample, temporal);
        let want: Vec<f32> = j["merged_timestamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(got.len(), want.len(), "merged timestamp count vs HF");
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "merged timestamp {g} vs HF {w}");
        }
    }

    #[test]
    fn invalid_public_video_timestamps_fail_before_prompt_rendering() {
        let frame = || ImageRef::new(1, 1, vec![0, 0, 0]).unwrap();
        for timestamps in [vec![0.0, f32::NAN], vec![0.5, 0.25], vec![-0.1, 0.0]] {
            let video = VideoRef {
                frames: vec![frame(), frame()],
                timestamps,
            };
            let messages = vec![Message {
                role: core_llm::Role::User,
                content: vec![Content::Video(video)],
                thinking: None,
                tool_calls: Vec::new(),
            }];
            let error = substitute_vision_placeholders(&messages, 2)
                .expect_err("invalid timestamp sequence accepted");
            assert!(error.to_string().contains("timestamp"));
        }
        let count_mismatch = vec![Message {
            role: core_llm::Role::User,
            content: vec![Content::Video(VideoRef {
                frames: vec![frame(), frame()],
                timestamps: vec![0.0],
            })],
            thinking: None,
            tool_calls: Vec::new(),
        }];
        assert!(substitute_vision_placeholders(&count_mismatch, 2)
            .unwrap_err()
            .to_string()
            .contains("one timestamp per frame"));
    }

    #[test]
    fn expanded_text_image_video_and_mtp_budgets_share_the_context_gate() {
        for expanded_prompt in [48usize, 52, 60] {
            validate_context_window(64, expanded_prompt, (64 - expanded_prompt) as u32).unwrap();
            let error =
                validate_context_window(64, expanded_prompt, (64 - expanded_prompt + 1) as u32)
                    .expect_err("one-token context overflow accepted");
            assert!(error.to_string().contains("exceeds context window 64"));
        }
        assert!(validate_context_window(usize::MAX, usize::MAX, 1)
            .unwrap_err()
            .to_string()
            .contains("overflow"));
    }

    /// **The per-frame placeholder string matches `Qwen3VLProcessor.replace_video_token` (collapsed
    /// form).** The engine emits **one** `<|video_pad|>` per frame and expands it to `frame_seqlen`
    /// copies after tokenizing (exactly the image path's pattern, where the chat template renders a
    /// single `<|image_pad|>`). The reference `replace_video_token` writes the *already-expanded*
    /// string (`frame_seqlen` `<|video_pad|>` per frame). Collapsing each consecutive `<|video_pad|>`
    /// run of the reference string to one token must yield exactly [`video_placeholder_text`]: same
    /// `<{t:.1f} seconds>` Text–Timestamp-Alignment tags, same per-frame vision framing.
    #[test]
    fn video_placeholder_string_matches_hf_reference() {
        let j = qwen3vl_video_oracle();
        let fps = j["fps"].as_f64().unwrap() as f32;
        let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
        let indices: Vec<f32> = j["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        // Build a synthetic VideoRef with one 1x1 frame per sampled index carrying its `idx/fps`
        // timestamp (the frame pixels are irrelevant to the placeholder string).
        let frames: Vec<ImageRef> = indices
            .iter()
            .map(|_| ImageRef::new(1, 1, vec![0, 0, 0]).unwrap())
            .collect();
        let timestamps: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
        let video = VideoRef::new(frames, timestamps).unwrap();
        let got = video_placeholder_text(&video, temporal);

        // Collapse the reference string's `<|video_pad|>` runs to a single token per frame.
        let pad = "<|video_pad|>";
        let mut collapsed = j["placeholder_text"].as_str().unwrap().to_string();
        while collapsed.contains(&format!("{pad}{pad}")) {
            collapsed = collapsed.replace(&format!("{pad}{pad}"), pad);
        }
        assert_eq!(
            got, collapsed,
            "collapsed Text–Timestamp-Alignment placeholder string must byte-match HF replace_video_token"
        );
        // The timestamp tags themselves must appear verbatim (the core of Text–Timestamp Alignment).
        for t in j["merged_timestamps"].as_array().unwrap() {
            let tag = format!("<{:.1} seconds>", t.as_f64().unwrap());
            assert!(
                got.contains(&tag),
                "placeholder must carry the `{tag}` timestamp tag: {got}"
            );
        }
    }

    /// **The per-frame `<|video_pad|>` expansion matches the HF id stream.** Tokenizing the reference
    /// placeholder string yields one `<|video_pad|>` per frame; expanding each to `frame_seqlen`
    /// copies (the merged patch count the ViT emits per frame) must reproduce the exact id stream the
    /// processor produces — same per-frame vision framing, same timestamp tokens, same counts. This is
    /// the video analogue of `expand_vision_placeholders` for images, but with `grid_t` runs.
    #[test]
    fn video_token_expansion_matches_hf_reference() {
        let j = qwen3vl_video_oracle();
        let expanded_hf: Vec<i32> = j["expanded_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let grid_t = j["grid_t"].as_u64().unwrap() as usize;
        let frame_seqlen = j["frame_seqlen"].as_u64().unwrap() as usize;
        let expected_video_tokens = j["expected_video_tokens"].as_u64().unwrap() as usize;

        // The merged-token count per frame, and total, must agree with the HF processor.
        assert_eq!(
            grid_t * frame_seqlen,
            expected_video_tokens,
            "total video tokens vs HF"
        );
        assert_eq!(
            expanded_hf.iter().filter(|&&x| x == vid).count(),
            expected_video_tokens,
            "video tokens in HF id stream"
        );

        // Reconstruct the *raw* (pre-expansion) ids: collapse each consecutive `<|video_pad|>` run
        // back to a single placeholder. The HF stream has `grid_t` such runs (one per frame), each of
        // `frame_seqlen` tokens; collapsing recovers one `<|video_pad|>` per frame.
        let mut raw = Vec::new();
        let mut i = 0usize;
        let mut runs = 0usize;
        while i < expanded_hf.len() {
            if expanded_hf[i] == vid {
                raw.push(vid);
                runs += 1;
                while i < expanded_hf.len() && expanded_hf[i] == vid {
                    i += 1;
                }
            } else {
                raw.push(expanded_hf[i]);
                i += 1;
            }
        }
        assert_eq!(runs, grid_t, "one <|video_pad|> run per frame (grid_t)");

        // Expanding each per-frame placeholder to `frame_seqlen` reproduces the HF id stream exactly.
        let counts = vec![frame_seqlen; grid_t];
        let expanded =
            crate::models::qwen35::expand_vision_placeholders(&raw, vid, &counts).unwrap();
        assert_eq!(expanded, expanded_hf, "expanded video ids vs HF processor");
    }

    /// **The video M-RoPE positions over the oracle grid are well-formed and per-frame-reset.** Feed
    /// the expanded video id stream + the `video_grid_thw` through `mrope_positions_mm`: the temporal
    /// row must reset to the frame's cursor at each frame (Qwen3-VL's synthetic time axis splits each
    /// `[t,h,w]` into `t` per-frame `[1,h,w]` blocks). The HF-pinned exact-row check lives in
    /// `qwen35::qwen3vl_mrope_video_matches_hf_reference`; here we confirm the provider's video grid
    /// drives the same path consistently.
    #[test]
    fn video_mrope_positions_split_frames() {
        let j = qwen3vl_video_oracle();
        let expanded_hf: Vec<i32> = j["expanded_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let img = j["video_token_id"].as_i64().unwrap() as i32 - 1; // a distinct unused image id
        let g = j["video_grid_thw"].as_array().unwrap()[0]
            .as_array()
            .unwrap();
        let grid = [
            g[0].as_i64().unwrap() as i32,
            g[1].as_i64().unwrap() as i32,
            g[2].as_i64().unwrap() as i32,
        ];
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (t, h, w, _delta) = crate::models::deepstack::mrope_positions_mm(
            &expanded_hf,
            &[],
            img,
            &[grid],
            vid,
            merge,
        )
        .unwrap();
        assert_eq!(t.len(), expanded_hf.len());
        // Each frame's video tokens share one temporal index (gt = 1 per frame after the split), and
        // the two frames sit at *different* temporal positions (the cursor advances between them).
        let frame_temporals: Vec<i32> = expanded_hf
            .iter()
            .zip(&t)
            .filter_map(|(&id, &tt)| (id == vid).then_some(tt))
            .collect();
        let distinct: std::collections::BTreeSet<i32> = frame_temporals.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            grid[0] as usize,
            "one distinct temporal index per frame"
        );
        // h/w spans are bounded by the per-frame grid (h/merge, w/merge).
        let max_w = (grid[2] / merge) - 1;
        let frame_ws: Vec<i32> = expanded_hf
            .iter()
            .zip(&w)
            .zip(&t)
            .filter_map(|((&id, &ww), &tt)| (id == vid).then_some(ww - tt))
            .collect();
        assert!(
            frame_ws.iter().all(|&rel| (0..=max_w).contains(&rel)),
            "w within per-frame grid"
        );
        let _ = h;
    }

    /// Tiny synthetic Llama whose head dimension the packed Metal reader supports.
    fn tiny_packed_capable_model() -> CausalLm {
        tiny_causal_model(2, 1, 64)
    }

    /// Tiny synthetic two-layer Llama (hidden 128) with the given attention geometry.
    fn tiny_causal_model(heads: i32, kv_heads: i32, head_dim: i32) -> CausalLm {
        use crate::primitives::sampler::{SplitMix64, TokenRng};
        let cfg = crate::config::ModelConfig {
            hidden_size: 128,
            intermediate_size: 64,
            num_layers: 2,
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            architecture: crate::config::Architecture::Llama,
            max_position_embeddings: 0,
            quantization: None,
            moe: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            query_pre_attn_scalar: None,
            partial_rotary_factor: 1.0,
            mla: None,
            yarn: None,
            mrope_section: None,
            gemma4: None,
        };
        let mut rng = SplitMix64::new(0x5c20676);
        let mut randn = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
            Array::from_slice(&data, shape)
        };
        let (h, v, inter) = (cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size);
        let (qd, kvd) = (
            cfg.num_heads * cfg.head_dim,
            cfg.num_kv_heads * cfg.head_dim,
        );
        let mut m = std::collections::HashMap::new();
        m.insert("model.embed_tokens.weight".to_string(), randn(&[v, h]));
        m.insert(
            "model.norm.weight".into(),
            Array::ones::<f32>(&[h]).unwrap(),
        );
        m.insert("lm_head.weight".into(), randn(&[v, h]));
        for i in 0..cfg.num_layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            m.insert(
                p("input_layernorm.weight"),
                Array::ones::<f32>(&[h]).unwrap(),
            );
            m.insert(
                p("post_attention_layernorm.weight"),
                Array::ones::<f32>(&[h]).unwrap(),
            );
            m.insert(p("self_attn.q_proj.weight"), randn(&[qd, h]));
            m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, h]));
            m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, h]));
            m.insert(p("self_attn.o_proj.weight"), randn(&[h, qd]));
            m.insert(p("mlp.gate_proj.weight"), randn(&[inter, h]));
            m.insert(p("mlp.up_proj.weight"), randn(&[inter, h]));
            m.insert(p("mlp.down_proj.weight"), randn(&[h, inter]));
        }
        CausalLm::from_weights(&Weights::from_map(m), "", cfg).unwrap()
    }

    /// sc-20671: admission prices K/V, activations and logits at the width the decoder actually
    /// caches (its compute dtype), and only the eager score term at F32. The Prism/Bonsai path is
    /// held at F32 until its frozen allocator peaks are re-taken.
    #[test]
    fn memory_geometry_prices_kv_at_the_cached_width() {
        let decoder = Decoder::Causal(Box::new(tiny_packed_capable_model()));
        let geometry = decoder.memory_geometry();
        let Decoder::Causal(model) = &decoder else {
            unreachable!()
        };
        let mut cache = model.new_cache();
        model
            .decode_logits(&input_ids(&[1, 2, 3]), &mut cache, 0)
            .unwrap();
        assert_eq!(cache.element_bytes().unwrap(), Some(geometry.element_bytes));
        assert_eq!(geometry.element_bytes, dtype_bytes(model.compute_dtype()));
        assert_eq!(
            geometry.score_element_bytes,
            dtype_bytes(Dtype::Float32),
            "sdpa_eager scores stay F32"
        );
        assert_eq!(
            priced_compute_element_bytes(Dtype::Bfloat16, true),
            dtype_bytes(Dtype::Float32),
            "Prism admission holds the pre-fix width until its peaks are re-taken"
        );
    }

    #[derive(Default)]
    struct CompressedArmCapture {
        snapshots: Vec<u64>,
        releases: Vec<u64>,
        reconstructions: usize,
        evidence: Vec<crate::primitives::PackedCacheEvidence>,
        fallbacks: Vec<(String, String)>,
        storage_tokens: Vec<u64>,
    }

    impl crate::campaign::Observer for CompressedArmCapture {
        fn phase(&mut self, _name: &'static str) {}
        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
        fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _capacity: u64, _element: u64) {
            self.snapshots.push(bytes);
        }
        fn release_event(&mut self, _kind: &'static str, _role: &'static str, bytes: u64) {
            self.releases.push(bytes);
        }
        fn dense_reconstruction(&mut self, _bytes: u64) {
            self.reconstructions += 1;
        }
        fn packed_cache_evidence(&mut self, evidence: &crate::primitives::PackedCacheEvidence) {
            self.evidence.push(evidence.clone());
        }
        fn dense_fallback(&mut self, operation: &str, reason: &str) {
            self.fallbacks.push((operation.into(), reason.into()));
        }
        fn compressed_storage(&mut self, storage: &crate::primitives::CompressedCacheStorage) {
            self.storage_tokens.push(storage.tokens);
        }
    }

    /// A compressed prefix hit imports the reused dense prefix into the arm's compressed cache by
    /// quantize-on-append: the suffix prefill and decode run on the fused reader over the imported
    /// history, nothing is reconstructed or recorded as a fallback, and the compressed cache never
    /// re-enters the dense prefix store. A refused compressed selection keeps the dense seed and
    /// records its reason instead.
    #[cfg(target_os = "macos")]
    #[test]
    fn compressed_prefix_hit_imports_the_reused_prefix_into_the_compressed_cache() {
        let model = tiny_packed_capable_model();
        // Longer than one 32-token group, so the import packs a K group and keeps a residual. The
        // ids stay inside the 32-token vocabulary: MLX's embedding gather is not bounds-checked,
        // so an out-of-vocabulary id reads whatever the allocator last left past the table and
        // the logits then depend on which test ran before.
        let prompt = (0..40).map(|i| i % 31 + 1).collect::<Vec<i32>>();
        let vocab = model.config().vocab_size;
        assert!(prompt.iter().all(|id| (0..vocab).contains(id)));
        let config = GenerationConfig {
            max_new_tokens: 2,
            seed: Some(0),
            ..Default::default()
        };
        let cancel = CancelFlag::new();
        let seeded_store = || {
            let mut store = crate::decode::PrefixCache::new(2);
            crate::decode::generate_cached(
                &model,
                &prompt,
                &config,
                &cancel,
                &mut |_| {},
                &mut store,
            )
            .unwrap();
            assert_eq!(store.len(), 1);
            store
        };

        let arm = crate::campaign::CompressedKvMethod::GroupAffine
            .arm()
            .unwrap();
        let mut store = seeded_store();
        let mut capture = CompressedArmCapture::default();
        let output = crate::decode::prefix::generate_cached_with_observer(
            &model,
            &prompt,
            &config,
            &cancel,
            &mut |_| {},
            &mut store,
            None,
            None,
            Some(&mut capture),
            Some(&arm),
        )
        .unwrap();
        assert_eq!(output.tokens.len(), 2);
        assert_eq!(store.stats().hits, 1);
        assert!(capture.fallbacks.is_empty(), "{:?}", capture.fallbacks);
        let [evidence] = capture.evidence.as_slice() else {
            panic!("the compressed cache must export evidence once");
        };
        assert!(evidence.accepted_direct_calls > 0);
        assert!(evidence.fallback_reasons.is_empty() && !evidence.dense_active);
        assert_eq!(evidence.full_cache_dequantizations, 0);
        assert_eq!(capture.reconstructions, 0);
        // 39 imported prefix tokens (a whole-prompt match recomputes the last) plus the suffix.
        assert_eq!(capture.storage_tokens.first(), Some(&40));
        assert_eq!(store.len(), 1, "the compressed cache is not stored back");

        let refused = crate::campaign::CompressedKvArm::with_reader(
            crate::campaign::CompressedKvMethod::GroupAffine,
            crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
                crate::primitives::OpaqueCompiledKernel::new(
                    "sc20676-prefix-refused",
                    "cpu",
                    0,
                    std::sync::Arc::new(()),
                ),
            )),
        );
        let mut store = seeded_store();
        let mut capture = CompressedArmCapture::default();
        crate::decode::prefix::generate_cached_with_observer(
            &model,
            &prompt,
            &config,
            &cancel,
            &mut |_| {},
            &mut store,
            None,
            None,
            Some(&mut capture),
            Some(&refused),
        )
        .unwrap();
        assert!(
            matches!(capture.fallbacks.as_slice(), [(operation, _)] if operation == "cache-selection"),
            "{:?}",
            capture.fallbacks
        );
        assert!(
            capture.evidence.is_empty(),
            "the dense seed carries the request"
        );
    }

    /// Prefill logits, packed evidence and fallbacks of one observed generation.
    #[derive(Default)]
    struct PrefillLogitsCapture {
        prefill: Vec<f32>,
        evidence: Vec<crate::primitives::PackedCacheEvidence>,
        fallbacks: Vec<(String, String)>,
    }

    impl crate::campaign::Observer for PrefillLogitsCapture {
        fn phase(&mut self, _name: &'static str) {}
        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
        fn logits(&mut self, stage: &'static str, values: &[f32]) {
            if stage == "prefill" {
                self.prefill = values.to_vec();
            }
        }
        fn packed_cache_evidence(&mut self, evidence: &crate::primitives::PackedCacheEvidence) {
            self.evidence.push(evidence.clone());
        }
        fn dense_fallback(&mut self, operation: &str, reason: &str) {
            self.fallbacks.push((operation.into(), reason.into()));
        }
    }

    /// Multi-turn prompt-cache reuse on the 8-bit compressed arm agrees with dense (SC-20671
    /// `multiTurnPromptCache` diagnosis). Turn 2 extends turn 1's prompt and answer; the dense
    /// store holds turn 1. The dense hit, the compressed hit (prefix imported by quantize-on-append
    /// at the reused offset) and a cold compressed turn 2 must choose the same tokens, with turn-2
    /// prefill logits within 8-bit rounding of the dense hit. A misplaced, dropped or re-quantized
    /// imported prefix moves the compressed logits by far more than that.
    #[cfg(target_os = "macos")]
    #[test]
    fn compressed_8bit_multi_turn_prompt_cache_reuse_agrees_with_dense() {
        let model = tiny_packed_capable_model();
        let vocab = model.config().vocab_size;
        // Turn 1 spans two 32-token groups plus a residual; ids stay inside the vocabulary.
        let turn1 = (0..70).map(|i| (i * 7) % 31 + 1).collect::<Vec<i32>>();
        let config = GenerationConfig {
            max_new_tokens: 6,
            seed: Some(0),
            ..Default::default()
        };
        let cancel = CancelFlag::new();
        let arm = crate::campaign::CompressedKvMethod::GroupAffine8
            .arm()
            .unwrap();
        let turn1_store = || {
            let mut store = crate::decode::PrefixCache::new(2);
            let out = crate::decode::generate_cached(
                &model,
                &turn1,
                &config,
                &cancel,
                &mut |_| {},
                &mut store,
            )
            .unwrap();
            (store, out.tokens)
        };
        let (_, answer) = turn1_store();
        let mut turn2 = turn1.clone();
        turn2.extend_from_slice(&answer);
        turn2.extend((0..9).map(|i| (i * 5) % 31 + 1));
        assert!(turn2.iter().all(|id| (0..vocab).contains(id)));

        let turn2_run =
            |store: &mut crate::decode::PrefixCache,
             compressed: Option<&crate::campaign::CompressedKvArm>| {
                let mut capture = PrefillLogitsCapture::default();
                let out = crate::decode::prefix::generate_cached_with_observer(
                    &model,
                    &turn2,
                    &config,
                    &cancel,
                    &mut |_| {},
                    store,
                    None,
                    None,
                    Some(&mut capture),
                    compressed,
                )
                .unwrap();
                (out.tokens, capture)
            };
        let (mut dense_store, _) = turn1_store();
        let (dense_tokens, dense) = turn2_run(&mut dense_store, None);
        let (mut hit_store, _) = turn1_store();
        let (hit_tokens, hit) = turn2_run(&mut hit_store, Some(&arm));
        let mut cold_store = crate::decode::PrefixCache::new(2);
        let (cold_tokens, cold) = turn2_run(&mut cold_store, Some(&arm));

        assert_eq!(dense_store.stats().hits, 1);
        assert_eq!(
            hit_store.stats().hits,
            1,
            "turn 2 must reuse turn 1's prefix"
        );
        assert_eq!(cold_store.stats().hits, 0);
        for (label, capture) in [("hit", &hit), ("cold", &cold)] {
            assert!(
                capture.fallbacks.is_empty(),
                "{label}: {:?}",
                capture.fallbacks
            );
            assert!(
                !capture.evidence.is_empty()
                    && capture
                        .evidence
                        .iter()
                        .all(|e| e.accepted_direct_calls > 0 && !e.dense_active),
                "{label} must decode on the fused reader"
            );
        }
        let max_abs = |a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            a.iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        let scale = dense.prefill.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(scale > 0.0);
        let hit_error = max_abs(&hit.prefill, &dense.prefill) / scale;
        let cold_error = max_abs(&cold.prefill, &dense.prefill) / scale;
        eprintln!(
            "mtpc: 8-bit turn-2 prefill relative max error hit {hit_error} cold {cold_error}"
        );
        assert!(
            hit_error < 0.02 && cold_error < 0.02,
            "8-bit turn-2 prefill logits drifted from dense: hit {hit_error}, cold {cold_error}"
        );
        assert_eq!(hit_tokens, dense_tokens, "compressed reuse vs dense turn 2");
        assert_eq!(cold_tokens, dense_tokens, "compressed cold vs dense turn 2");
    }

    /// A tiny packed-capable on-disk snapshot (head dim 64, 32-word vocabulary, a 2048-token
    /// window, no reachable stop token) the provider loads like a real one.
    fn tiny_packed_capable_snapshot() -> tempfile::TempDir {
        tiny_snapshot(json!({"architectures": ["LlamaForCausalLM"]}), 2048, false)
    }

    /// [`tiny_packed_capable_snapshot`] with the config's architecture `identity` keys, a
    /// `window`-token context, and (for Qwen3) unit per-head q/k RMSNorm weights.
    fn tiny_snapshot(identity: serde_json::Value, window: u64, qk_norm: bool) -> tempfile::TempDir {
        tiny_snapshot_with(identity, window, qk_norm, SnapshotGains::DEFAULT)
    }

    /// Weight gains of a [`tiny_snapshot_with`] fixture over its unit-scale random draws.
    #[derive(Clone, Copy, Debug)]
    struct SnapshotGains {
        /// Scale of the query/key projections (of the q/k RMSNorm weights on a Qwen3 fixture):
        /// sharper attention.
        qk: f32,
        /// Scale of the value/output projections: attention's share of the residual.
        vo: f32,
        /// Weight of the embeddings in the output head: how strongly the current token decides
        /// the next one.
        head_align: f32,
    }

    impl SnapshotGains {
        const DEFAULT: Self = Self {
            qk: 1.0,
            vo: 1.0,
            head_align: 4.0,
        };
    }

    /// [`tiny_snapshot`] with explicit weight `gains`; the random draws are the same for every
    /// gain, so [`SnapshotGains::DEFAULT`] reproduces [`tiny_snapshot`] exactly.
    fn tiny_snapshot_with(
        identity: serde_json::Value,
        window: u64,
        qk_norm: bool,
        gains: SnapshotGains,
    ) -> tempfile::TempDir {
        use crate::primitives::sampler::{SplitMix64, TokenRng};
        let dir = tempfile::tempdir().unwrap();
        let mut vocab = serde_json::Map::new();
        for (id, word) in ["<unk>", "<|", "|>", "user", "assistant", "eot_id"]
            .into_iter()
            .map(String::from)
            .chain((6..32).map(|id| format!("w{id}")))
            .enumerate()
        {
            vocab.insert(word, json!(id));
        }
        let tokenizer = json!({
            "version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" }, "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" },
        });
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let mut config = json!({
            "hidden_size": 128,
            "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64, "vocab_size": 32, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": false,
            "max_position_embeddings": window, "eos_token_id": 99,
        });
        for (key, value) in identity.as_object().unwrap() {
            config[key] = value.clone();
        }
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let mut rng = SplitMix64::new(0x5c20671);
        let mut randn = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
            Array::from_slice(&data, shape)
        };
        let mut arrays = vec![
            (
                "model.embed_tokens.weight".to_string(),
                randn(&[32, 128]) * 4.0f32,
            ),
            (
                "model.norm.weight".into(),
                Array::ones::<f32>(&[128]).unwrap(),
            ),
        ];
        // A head partly aligned with the embeddings gives decisive next-token margins (a random
        // head over 32 words leaves bf16-rounding ties that flip even dense against dense), while
        // the attention path still moves the argmax (2-bit K/V flips positions).
        let head = &arrays[0].1 * gains.head_align + randn(&[32, 128]);
        arrays.push(("lm_head.weight".into(), head));
        // The identity may deepen the decoder (`num_hidden_layers`); every layer gets weights.
        for i in 0..config["num_hidden_layers"].as_u64().unwrap() {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            arrays.extend([
                (
                    p("input_layernorm.weight"),
                    Array::ones::<f32>(&[128]).unwrap(),
                ),
                (
                    p("post_attention_layernorm.weight"),
                    Array::ones::<f32>(&[128]).unwrap(),
                ),
                (p("self_attn.q_proj.weight"), randn(&[128, 128]) * gains.qk),
                (p("self_attn.k_proj.weight"), randn(&[64, 128]) * gains.qk),
                (p("self_attn.v_proj.weight"), randn(&[64, 128]) * gains.vo),
                (p("self_attn.o_proj.weight"), randn(&[128, 128]) * gains.vo),
                (p("mlp.gate_proj.weight"), randn(&[64, 128])),
                (p("mlp.up_proj.weight"), randn(&[64, 128])),
                (p("mlp.down_proj.weight"), randn(&[128, 64])),
            ]);
            if qk_norm {
                arrays.extend([
                    (
                        p("self_attn.q_norm.weight"),
                        Array::ones::<f32>(&[64]).unwrap() * gains.qk,
                    ),
                    (
                        p("self_attn.k_norm.weight"),
                        Array::ones::<f32>(&[64]).unwrap() * gains.qk,
                    ),
                ]);
            }
        }
        let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();
        dir
    }

    /// SC-20671 contract v4 multi-turn prompt-cache fixture on the product provider path, at
    /// 8-bit. Turn 1 runs and stores its K/V; turn 2 (turn 1's conversation, answer and a
    /// follow-up) is served by a prompt-cache hit in both arms — the compressed arm importing the
    /// prefix — and the compressed arm, teacher-forced on the dense turn-2 forced continuation,
    /// agrees at >= 0.999 of 1024 positions. A turn 2 the cache did not serve is refused.
    #[cfg(target_os = "macos")]
    #[test]
    fn multi_turn_prompt_cache_fixture_agrees_at_8_bit_and_requires_a_turn_two_hit() {
        let snapshot = tiny_packed_capable_snapshot();
        let provider = LlamaProvider::load_for_campaign(&core_llm::LoadSpec::dense(
            snapshot.path().to_string_lossy().to_string(),
        ))
        .unwrap();
        let words = (0..70)
            .map(|i| format!("w{}", (i * 7) % 26 + 6))
            .collect::<Vec<_>>()
            .join(" ");
        let turn1 = TextLlmRequest {
            messages: vec![Message::text(Role::User, words)],
            sampling: core_llm::Sampling::greedy(),
            max_new_tokens: 8,
            seed: Some(0),
            ..Default::default()
        };
        let follow_up = "w7 w9 w11 w13 w15 w17 w19 w21 w23";
        let arm = crate::campaign::CompressedKvMethod::GroupAffine8
            .arm()
            .unwrap();

        let (reference, dense_turns) = provider
            .campaign_multi_turn_forced_continuation(&turn1, follow_up, 1024, None, None)
            .unwrap();
        assert_eq!(reference.len(), 1024, "every stop token is decoded through");
        // SC-20669 dense multi-turn noise-floor control: the same flow scored. The dense reference
        // reproduces the unscored stream, and a dense session teacher-forced on it through the
        // same cache-hit path scores every turn-2 position.
        let (scored, scored_turns) = provider
            .campaign_multi_turn_scored_continuation(&turn1, follow_up, 1024, None)
            .unwrap();
        assert_eq!(scored.choices, reference);
        assert_eq!(scored.stream_probabilities.len(), 1024);
        assert_eq!(scored_turns, dense_turns);
        let (forced, forced_turns) = provider
            .campaign_multi_turn_scored_continuation(&turn1, follow_up, 1024, Some(&reference))
            .unwrap();
        assert_eq!(forced.choices, reference);
        assert_eq!(forced.stream_probabilities, scored.stream_probabilities);
        assert_eq!(forced_turns, dense_turns);
        let (choices, compressed_turns) = provider
            .campaign_multi_turn_forced_continuation(
                &turn1,
                follow_up,
                1024,
                Some(&arm),
                Some(&reference),
            )
            .unwrap();
        dense_turns.validate("dense").unwrap();
        assert!(
            dense_turns.turn2.reused_prefix_tokens >= dense_turns.turn1.prompt_tokens,
            "turn 2 reuses all of turn 1's prompt: {dense_turns:?}"
        );
        assert_eq!(compressed_turns, dense_turns, "the same flow in both arms");
        let evidence =
            crate::campaign::multi_turn_forced_continuation_evidence(&reference, &choices).unwrap();
        eprintln!(
            "mtpc v4: 8-bit turn-2 teacher-forced agreement {} ({}/{}; flips {:?})",
            evidence.agreement, evidence.matches, evidence.tokens, evidence.first_flip_positions
        );
        assert!(
            evidence.agreement >= crate::campaign::COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN,
            "{evidence:?}"
        );

        // The observed fixture: both arms' turn 2 is the same prompt-cache hit.
        let observed = |packed: Option<&crate::campaign::CompressedKvArm>| {
            let mut capture = CompressedArmCapture::default();
            let (output, turns) = provider
                .campaign_multi_turn_observed(&turn1, follow_up, &mut |_| {}, &mut capture, packed)
                .unwrap();
            assert!(capture.fallbacks.is_empty(), "{:?}", capture.fallbacks);
            (output.text, turns, capture.evidence)
        };
        let (dense_text, dense_observed, dense_evidence) = observed(None);
        let (compressed_text, compressed_observed, compressed_evidence) = observed(Some(&arm));
        assert_eq!(dense_observed, dense_turns);
        assert_eq!(compressed_observed, dense_turns);
        assert!(dense_evidence.is_empty(), "the dense arm stays dense");
        assert!(
            !compressed_evidence.is_empty()
                && compressed_evidence
                    .iter()
                    .all(|e| e.accepted_direct_calls > 0 && !e.dense_active),
            "turn 2 decodes on the fused reader over the imported prefix"
        );
        assert!(!dense_text.is_empty() && !compressed_text.is_empty());

        // The 2048-token window cannot hold turn 2 of a ~1100-token turn 1 plus the 1024-token
        // continuation: refused before decoding, never shortened.
        let long_turn1 = TextLlmRequest {
            messages: vec![Message::text(
                Role::User,
                (0..1_100)
                    .map(|i| format!("w{}", (i * 5) % 26 + 6))
                    .collect::<Vec<_>>()
                    .join(" "),
            )],
            ..turn1.clone()
        };
        let error = provider
            .campaign_multi_turn_forced_continuation(&long_turn1, follow_up, 1024, None, None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("fewer than the 1024-token forced continuation"),
            "{error}"
        );

        // A turn 2 the cache did not serve is refused.
        let mut store = crate::decode::PrefixCache::new(2);
        let (answer, _, turn1_ids) = provider.campaign_turn_one(&turn1, &mut store).unwrap();
        let turn2 = LlamaProvider::campaign_turn_two_request(&turn1, &answer, follow_up);
        let (_, turn2_ids) = provider.render_prompt(&turn2, &turn2.messages).unwrap();
        let missed = crate::campaign::PromptCacheTurn {
            prompt_tokens: turn2_ids.len() as u64,
            prompt_sha256: crate::campaign::token_stream_sha256(&turn2_ids),
            cache_hit: false,
            reused_prefix_tokens: 0,
        };
        let error = LlamaProvider::require_turn_two_hit(&missed, &turn1_ids, &turn2_ids)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("not served by a prompt-cache hit"),
            "{error}"
        );
    }

    /// A packed cache whose retained reader fails to bind still records the selection's own
    /// reason rather than only the later generic dense-attention fallback.
    #[cfg(target_os = "macos")]
    #[test]
    fn compressed_campaign_decoder_records_a_refused_reader_binding() {
        let model = tiny_packed_capable_model();
        let reader = crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
            crate::primitives::OpaqueCompiledKernel::new(
                "sc20676rx-refused",
                "cpu",
                0,
                std::sync::Arc::new(()),
            ),
        ));
        let arm = crate::campaign::CompressedKvArm::with_reader(
            crate::campaign::CompressedKvMethod::GroupAffine,
            reader,
        );
        let decoder = PackedCampaignDecoder {
            model: &model,
            arm: &arm,
            selection_fallbacks: RefCell::new(Vec::new()),
        };
        let cache = decoder.make_cache();
        assert!(
            cache.packed_evidence().is_some(),
            "a refused binding still yields the packed cache"
        );
        let fallbacks = decoder.selection_fallbacks.into_inner();
        assert!(
            matches!(fallbacks.as_slice(), [reason] if reason.contains("packed reader rejected before mutation")),
            "{fallbacks:?}"
        );
    }

    /// The compressed campaign decoder runs the whole observed generation (prefill and decode) on
    /// the packed cache through the fused reader: no fallback, no dense reconstruction, and the
    /// persistent KV it reports is the packed storage it later releases.
    #[cfg(target_os = "macos")]
    #[test]
    fn compressed_campaign_decoder_keeps_prefill_and_decode_on_the_fused_reader() {
        let model = tiny_packed_capable_model();
        let arm = crate::campaign::CompressedKvMethod::GroupAffine
            .arm()
            .unwrap();
        let decoder = PackedCampaignDecoder {
            model: &model,
            arm: &arm,
            selection_fallbacks: RefCell::new(Vec::new()),
        };
        let mut capture = CompressedArmCapture::default();
        let config = GenerationConfig {
            max_new_tokens: 4,
            seed: Some(0),
            ..Default::default()
        };
        let output = crate::decode::generate_with_observer(
            &decoder,
            &[1, 2, 3, 4, 5],
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            Some(&mut capture),
        )
        .unwrap();
        assert_eq!(output.tokens.len(), 4);
        assert!(decoder.selection_fallbacks.borrow().is_empty());
        let [evidence] = capture.evidence.as_slice() else {
            panic!("one compressed cache must export evidence once");
        };
        // Two layers: one fused prefill call plus three fused decode steps each.
        assert_eq!(evidence.accepted_direct_calls, 2 * 4);
        assert!(evidence.fallback_reasons.is_empty() && !evidence.dense_active);
        assert_eq!(evidence.full_cache_dequantizations, 0);
        assert_eq!(capture.reconstructions, 0);
        assert!(!capture.snapshots.is_empty());
        assert_eq!(capture.releases, vec![*capture.snapshots.last().unwrap()]);
    }

    /// SC-20671 prefill attribution through the real decode loop, at 2- and 4-bit. An empty
    /// cache's first multi-row step attends on dense SDPA, so no output reads the packed store;
    /// the cache must still materialize the store it reports at the step's commit. Left lazy, the
    /// prefill-peak sample holds the step's dense K/V instead of the store, and the campaign's
    /// `baseline + persistent KV` floor fails whenever that dense K/V is smaller than the
    /// block-allocated store (Mac2 A2 qwen-short 4-bit: 80 prompt tokens, active growth
    /// 9,504,184 B for a 12,845,056 B store; 2-bit passed only because its 9,175,040 B store
    /// equals the 80-token dense K/V).
    #[cfg(target_os = "macos")]
    #[test]
    fn compressed_prefill_sample_holds_the_reported_packed_store() {
        #[derive(Default)]
        struct PrefillCapture {
            phase: Option<&'static str>,
            prefill_active: Option<u64>,
            prefill_kv: u64,
            element_bytes: u64,
        }
        impl crate::campaign::Observer for PrefillCapture {
            fn phase(&mut self, name: &'static str) {
                self.phase = Some(name);
                if name == "prefill-peak" {
                    self.prefill_active = Some(mlx_rs::memory::get_active_memory() as u64);
                }
            }
            fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
            fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _capacity: u64, element: u64) {
                if self.phase == Some("prefill-peak") {
                    self.prefill_kv = self.prefill_kv.max(bytes);
                    self.element_bytes = element;
                }
            }
        }
        let model = tiny_packed_capable_model();
        // 40 tokens: past the fused-SDPA row limit (the dense first step), one flushed 32-token
        // group and an 8-token residual.
        let prompt = (0..40).map(|i| (i % 31) + 1).collect::<Vec<i32>>();
        let config = GenerationConfig {
            max_new_tokens: 2,
            seed: Some(0),
            ..Default::default()
        };
        for method in crate::campaign::CompressedKvMethod::ALL {
            let arm = method.arm().unwrap();
            let decoder = PackedCampaignDecoder {
                model: &model,
                arm: &arm,
                selection_fallbacks: RefCell::new(Vec::new()),
            };
            let run = |observer: Option<&mut dyn crate::campaign::Observer>| {
                crate::decode::generate_with_observer(
                    &decoder,
                    &prompt,
                    &config,
                    &CancelFlag::new(),
                    &mut |_| {},
                    None,
                    None,
                    observer,
                )
                .unwrap();
            };
            // Warm the reader and the lazily created weights so the baseline is steady. Tests run
            // one at a time (`.cargo/config.toml`), so the process-global counter is ours.
            run(None);
            let baseline = mlx_rs::memory::get_active_memory() as u64;
            let mut capture = PrefillCapture::default();
            run(Some(&mut capture));
            assert!(
                decoder.selection_fallbacks.borrow().is_empty(),
                "{method:?} selected the packed cache"
            );
            // The failing scenario: the prompt's dense K + V (2 layers, 1 KV head, D64, at the
            // cache's element width) is smaller than the reported store.
            let dense_prompt_kv = 2 * 2 * 40 * 64 * capture.element_bytes;
            assert!(
                dense_prompt_kv < capture.prefill_kv,
                "{method:?}: dense {dense_prompt_kv} B vs store {} B",
                capture.prefill_kv
            );
            let growth = capture.prefill_active.unwrap().saturating_sub(baseline);
            assert!(
                growth >= capture.prefill_kv,
                "{method:?}: prefill-peak active growth {growth} B does not hold the reported \
                 packed store {} B",
                capture.prefill_kv
            );
        }
    }

    /// SC-20671 storage reconciliation through the real decode loop, at 2- and 4-bit: a prompt
    /// that crosses a 256-token block boundary leaves a pending residual, and the decode that
    /// follows (including a group flush) stays inside the last block, so the device share is a
    /// plateau. The coordinate storage the campaign observer keeps must be the plateau's latest
    /// instant: its device share is the largest persistent snapshot and its length the largest
    /// live length (the Mac2 32k single-shot row recorded the prefill instant, 32836 tokens,
    /// against a kvLength of 32843).
    #[cfg(target_os = "macos")]
    #[test]
    fn compressed_coordinate_storage_reconciles_across_a_block_plateau() {
        #[derive(Default)]
        struct StorageCapture {
            snapshots: Vec<(u64, u64)>,
            storage: Vec<crate::primitives::CompressedCacheStorage>,
        }
        impl crate::campaign::Observer for StorageCapture {
            fn phase(&mut self, _name: &'static str) {}
            fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
            fn cache_snapshot(&mut self, bytes: u64, tokens: u64, _capacity: u64, _element: u64) {
                self.snapshots.push((bytes, tokens));
            }
            fn compressed_storage(&mut self, storage: &crate::primitives::CompressedCacheStorage) {
                self.storage.push(*storage);
            }
        }
        let model = tiny_packed_capable_model();
        // 300 prompt tokens: one full block plus 44 (a 12-token residual after 32-token groups).
        let prompt = (0..300).map(|i| (i % 31) + 1).collect::<Vec<i32>>();
        for method in crate::campaign::CompressedKvMethod::ALL {
            let arm = method.arm().unwrap();
            let decoder = PackedCampaignDecoder {
                model: &model,
                arm: &arm,
                selection_fallbacks: RefCell::new(Vec::new()),
            };
            let mut capture = StorageCapture::default();
            let config = GenerationConfig {
                max_new_tokens: 30,
                seed: Some(0),
                ..Default::default()
            };
            crate::decode::generate_with_observer(
                &decoder,
                &prompt,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
                Some(&mut capture),
            )
            .unwrap();
            let mut observer = crate::campaign::ProductObserver::new();
            observer.begin_coordinate_operation();
            for storage in &capture.storage {
                crate::campaign::Observer::compressed_storage(&mut observer, storage);
            }
            observer.end_coordinate_operation();
            let peak = observer.coordinate_storage_peak().unwrap();
            let persistent = capture
                .snapshots
                .iter()
                .map(|(bytes, _)| *bytes)
                .max()
                .unwrap();
            let kv_length = capture
                .snapshots
                .iter()
                .map(|(_, tokens)| *tokens)
                .max()
                .unwrap();
            // The scenario is the failing one: several instants share the peak device share.
            let plateau = capture
                .storage
                .iter()
                .filter(|storage| storage.device_bytes() == persistent)
                .map(|storage| storage.tokens)
                .collect::<Vec<_>>();
            assert!(
                plateau.len() > 1 && plateau[0] < kv_length,
                "{method:?} {plateau:?}"
            );
            assert!(
                kv_length > 300 && kv_length < 512,
                "{method:?} decode stays in block two"
            );
            crate::campaign::coordinate_storage_reconciles(
                peak.device_code_bytes,
                Some(peak.device_bytes()),
                peak.tokens,
                persistent,
                kv_length,
            )
            .unwrap_or_else(|error| panic!("{method:?}: {error}"));
        }
    }

    /// SC-20671 steady decode on a tiny model: both arms decode exactly the fixed length through
    /// stop tokens (every vocabulary id is declared one), and a compressed arm whose reader is
    /// refused fails closed instead of timing a dense decode under the compressed label.
    #[cfg(target_os = "macos")]
    #[test]
    fn campaign_steady_decode_is_fixed_length_on_both_arms_and_refuses_a_dense_compressed_arm() {
        let model = Decoder::Causal(Box::new(tiny_packed_capable_model()));
        let every_id = (0..32).collect::<Vec<i32>>();
        let prompt = [1, 2, 3, 4, 5];
        let arm = crate::campaign::CompressedKvMethod::GroupAffine
            .arm()
            .unwrap();
        for compressed in [None, Some(&arm)] {
            let measured =
                campaign_steady_decode_on(&model, &prompt, 8, &every_id, compressed).unwrap();
            assert_eq!(measured.prompt_tokens, 5);
            assert_eq!(measured.generated_tokens, 8);
            assert_eq!(measured.timed_tokens, 7);
            assert_eq!(
                measured.forced_stop_tokens, 8,
                "every token was a forced stop"
            );
        }
        let refused = crate::campaign::CompressedKvArm::with_reader(
            crate::campaign::CompressedKvMethod::GroupAffine,
            crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
                crate::primitives::OpaqueCompiledKernel::new(
                    "sc20671-refused",
                    "cpu",
                    0,
                    std::sync::Arc::new(()),
                ),
            )),
        );
        let error = campaign_steady_decode_on(&model, &prompt, 8, &every_id, Some(&refused))
            .unwrap_err()
            .to_string();
        assert!(error.contains("fused compressed reader"), "{error}");
    }

    /// SC-20671 forced continuation on a tiny model: the dense arm continues greedily through
    /// every stop token; teacher-forcing the dense arm on its own continuation reproduces it at
    /// every position; the compressed arm is teacher-forced on the dense stream wholly on its fused
    /// reader; a position forced off the compressed arm's choice is a recorded flip; and a refused
    /// compressed reader fails closed.
    #[cfg(target_os = "macos")]
    #[test]
    fn campaign_forced_continuation_teacher_forces_the_compressed_arm_on_the_dense_stream() {
        let model = Decoder::Causal(Box::new(tiny_packed_capable_model()));
        let every_id = (0..32).collect::<Vec<i32>>();
        let prompt = [1, 2, 3, 4, 5];
        let arm = crate::campaign::CompressedKvMethod::GroupAffine
            .arm()
            .unwrap();
        let scored = |compressed, stream: Option<&[i32]>| {
            campaign_forced_decode_on(&model, &prompt, 64, &every_id, compressed, stream, true)
        };
        let forced = |compressed, stream: Option<&[i32]>| {
            scored(compressed, stream).map(|decode| decode.tokens)
        };
        let reference = forced(None, None).unwrap();
        assert_eq!(reference.len(), 64, "every stop token is decoded through");
        assert_eq!(forced(None, Some(&reference)).unwrap(), reference);
        // Scored on the same stream, the dense arm teacher-forced on its own continuation has
        // exactly its free-running likelihood; the compressed arm scores those same tokens.
        let dense = scored(None, None).unwrap().stream_probabilities;
        assert_eq!(dense.len(), 64);
        assert_eq!(
            scored(None, Some(&reference)).unwrap().stream_probabilities,
            dense
        );
        let compressed = scored(Some(&arm), Some(&reference))
            .unwrap()
            .stream_probabilities;
        assert_eq!(compressed.len(), 64);
        assert!(compressed
            .iter()
            .all(|probability| *probability > 0.0 && *probability <= 1.0));
        // The dense noise-floor control decodes the same stream after a chunked prefill: an exact
        // dense computation in another order, so it agrees with the one-shot continuation and
        // scores the same tokens (here identically: the tiny model's chunks are exact).
        for chunk in [1, 2, 3, 4, 64] {
            let control =
                campaign_chunked_prefill_decode_on(&model, &prompt, &reference, &every_id, chunk)
                    .unwrap();
            assert_eq!(control.tokens, reference, "chunk {chunk}");
            assert_eq!(control.stream_probabilities.len(), 64);
            let likelihood = crate::campaign::StreamLikelihood::from_probabilities(
                &dense,
                &control.stream_probabilities,
            )
            .unwrap();
            assert!(
                (likelihood.candidate - likelihood.reference).abs() < 1e-3,
                "chunk {chunk}: {likelihood:?}"
            );
        }
        assert!(
            campaign_chunked_prefill_decode_on(&model, &prompt, &reference, &every_id, 0).is_err()
        );
        // Run 37021783368: the control's cache is ONE allocation for the prompt and stream, so a
        // prompt longer than the default 256-position block retires no growth buffer (the
        // default cache retired one per growth into MLX's buffer cache, past the row's cap).
        let long_prompt = (0..600).map(|index| index % 31).collect::<Vec<_>>();
        let mut cache = chunked_prefill_cache(&model, long_prompt.len() + 64).unwrap();
        chunked_prefill(&model, &mut cache, &long_prompt, 64).unwrap();
        assert_eq!(cache.offset(), 600);
        assert!(
            cache
                .events()
                .iter()
                .all(|event| event.operation != "dense_block_growth_retired_buffer"),
            "the chunked control retired a dense growth buffer"
        );
        let mut grown = match &model {
            Decoder::Causal(model) => model.new_cache(),
            Decoder::Qwen35(_) => unreachable!(),
        };
        chunked_prefill(&model, &mut grown, &long_prompt, 64).unwrap();
        assert!(
            grown
                .events()
                .iter()
                .any(|event| event.operation == "dense_block_growth_retired_buffer"),
            "the default cache grows (the defect the sized cache removes)"
        );
        let choices = forced(Some(&arm), Some(&reference)).unwrap();
        let evidence = crate::campaign::forced_continuation_evidence(&reference, &choices).unwrap();
        assert_eq!(evidence.tokens, 64);
        assert_eq!(evidence.matches + evidence.flip_count, 64);
        // Force position 10 off the compressed arm's own choice there: a guaranteed flip.
        let mut off = reference.clone();
        off[10] = (choices[10] + 1) % 32;
        let off_choices = forced(Some(&arm), Some(&off)).unwrap();
        assert_eq!(
            off_choices[..10],
            choices[..10],
            "the prefix before 10 is unchanged"
        );
        let miss = crate::campaign::forced_continuation_evidence(&off, &off_choices).unwrap();
        assert!(miss.first_flip_positions.contains(&10), "{miss:?}");
        assert!(miss.agreement < 1.0);
        assert!(forced(Some(&arm), Some(&reference[..63])).is_err());
        let refused = crate::campaign::CompressedKvArm::with_reader(
            crate::campaign::CompressedKvMethod::GroupAffine,
            crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
                crate::primitives::OpaqueCompiledKernel::new(
                    "sc20671-refused",
                    "cpu",
                    0,
                    std::sync::Arc::new(()),
                ),
            )),
        );
        let error = forced(Some(&refused), Some(&reference))
            .unwrap_err()
            .to_string();
        assert!(error.contains("fused compressed reader"), "{error}");
    }

    // ---- sc-20679: production compressed KV ----

    /// One greedy product generation over `words` user words at `policy`, through the
    /// production entry point (`TextLlm::generate`).
    fn kv_generate(
        provider: &LlamaProvider,
        words: usize,
        policy: core_llm::KvCompressionPolicy,
        max_new_tokens: u32,
    ) -> TextLlmOutput {
        kv_generate_words(provider, words, (7, 26), policy, max_new_tokens).0
    }

    /// [`kv_generate`] over the word pattern `w{(i · step) % modulo + 6}`, also returning the
    /// rendered prompt ids and the generated token ids as the stream emitted them.
    fn kv_generate_words(
        provider: &LlamaProvider,
        words: usize,
        (step, modulo): (usize, usize),
        policy: core_llm::KvCompressionPolicy,
        max_new_tokens: u32,
    ) -> (TextLlmOutput, Vec<i32>, Vec<i32>) {
        let text = (0..words)
            .map(|i| format!("w{}", (i * step) % modulo + 6))
            .collect::<Vec<_>>()
            .join(" ");
        let request = TextLlmRequest {
            messages: vec![Message::text(Role::User, text)],
            sampling: core_llm::Sampling::greedy(),
            max_new_tokens,
            seed: Some(0),
            kv_compression: policy,
            ..Default::default()
        };
        let (_, prompt_ids) = provider.render_prompt(&request, &request.messages).unwrap();
        let mut stream = Vec::new();
        let output = provider
            .generate(&request, &mut |event| {
                if let CoreEvent::Token { id, .. } = event {
                    stream.push(id as i32);
                }
            })
            .unwrap();
        (output, prompt_ids, stream)
    }

    fn load_tiny(snapshot: &tempfile::TempDir) -> LlamaProvider {
        LlamaProvider::load(&core_llm::LoadSpec::dense(
            snapshot.path().to_string_lossy().to_string(),
        ))
        .unwrap()
    }

    /// Gains that make the tiny fixture's generation depend on its context: sharp attention whose
    /// output dominates the residual, and a head only weakly tied to the current token. With the
    /// default gains every prompt decodes the same single token, which no reader could get wrong.
    /// Sharper attention (qk ≥ 3) amplifies 8-bit key rounding past the parity bound, so the
    /// gains stay in the band where 8 bits holds and 2/4 bits do not.
    const CONTEXT_SENSITIVE_GAINS: SnapshotGains = SnapshotGains {
        qk: 2.0,
        vo: 3.0,
        head_align: 1.0,
    };

    /// AC3: a request at the family's qualified minimum context runs wholly on the fused
    /// compressed reader through the production entry point, and every token it emits is the
    /// dense model's choice at that position (its dense logit within 8-bit rounding of the dense
    /// maximum) — on a fixture whose dense output demonstrably depends on the context. The same
    /// provider keeps a short request, and an un-opted request, dense with their reasons.
    fn assert_compressed_matches_dense(
        identity: serde_json::Value,
        qk_norm: bool,
        family: core_llm::KvModelFamily,
    ) {
        use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
        let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
            .iter()
            .find(|row| row.family == family)
            .unwrap();
        let min = usize::try_from(row.min_context_tokens).unwrap();
        let snapshot = tiny_snapshot_with(
            identity,
            row.min_context_tokens + 1024,
            qk_norm,
            CONTEXT_SENSITIVE_GAINS,
        );
        let provider = load_tiny(&snapshot);
        const NEW: u32 = 24;
        const PATTERN: (usize, usize) = (7, 26);

        let dense = kv_generate(&provider, min, Policy::Off, NEW);
        let (compressed, prompt_ids, stream) =
            kv_generate_words(&provider, min, PATTERN, Policy::Qualified, NEW);

        // Sensitivity control: the dense output is context-dependent and varied, so a reader
        // that corrupts the history changes what it should emit.
        let control = kv_generate_words(&provider, min, (5, 13), Policy::Off, NEW).0;
        assert_ne!(control.text, dense.text, "the fixture ignores its context");
        let distinct = |text: &str| {
            text.split_whitespace()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        };
        assert!(distinct(&dense.text) >= 4, "{}", dense.text);
        assert!(distinct(&compressed.text) >= 4, "{}", compressed.text);

        // Per-step parity on the stream the production compressed generation emitted: the dense
        // model, teacher-forced on that stream, scores every emitted token within 8-bit rounding
        // of its best. With every logit within 0.02 of the dense range (the logit-parity bound),
        // a near-tie the compressed cache resolves the other way costs at most two such errors:
        // 0.04. Measured ≤ 0.004 here at 8 bits; 4-bit codes reach 0.15 and 2-bit 0.3–0.55.
        assert_eq!(stream.len(), NEW as usize);
        let Decoder::Causal(model) = &provider.model else {
            panic!("a qualified family is a causal decoder");
        };
        let mut dense_cache = model.make_cache();
        let rows = forced_logits(
            model,
            dense_cache.as_mut(),
            &prompt_ids,
            &stream[..stream.len() - 1],
        );
        let range = rows
            .iter()
            .flatten()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        for (position, (logits, &token)) in rows.iter().zip(&stream).enumerate() {
            let best = logits.iter().copied().fold(f32::MIN, f32::max);
            let gap = (best - logits[token as usize]) / range;
            assert!(
                gap <= 0.04,
                "position {position}: compressed emitted {token}, {gap} of the dense logit \
                 range below the dense choice"
            );
        }
        assert!(
            u64::from(compressed.usage.prompt_tokens) >= row.min_context_tokens,
            "{:?}",
            compressed.usage
        );
        assert_eq!(compressed.usage, dense.usage);
        assert_eq!(compressed.usage.generated_tokens, NEW);
        assert_eq!(
            dense.kv_cache,
            Some(core_llm::KvCacheReport::dense(Reason::PolicyDisabled, None))
        );
        let report = compressed.kv_cache.clone().unwrap();
        assert!(report.ran_compressed(), "{report:?}");
        assert_eq!(report.format, Some(row.format));
        assert_eq!(report.format_version, core_llm::KV_CACHE_FORMAT_VERSION);
        // Every decode step after the prefill reads both layers through the fused reader.
        assert!(
            report.counters.fused_attention_calls >= 2 * u64::from(NEW - 1),
            "{report:?}"
        );
        assert_eq!(report.counters.full_cache_dequantizations, 0);
        assert_eq!(report.counters.dense_fallback_events, 0);
        // Retained compressed bytes stay below the dense K/V of the same tokens
        // (2 layers × K,V × 1 KV head × head dim 64 × bf16).
        let tokens = u64::from(compressed.usage.total_tokens());
        assert!(report.counters.compressed_cache_bytes > 0);
        assert!(
            report.counters.compressed_cache_bytes < tokens * 2 * 2 * 64 * 2,
            "{report:?}"
        );
        // The same provider keeps a short request dense: short contexts decode slower compressed.
        let short = kv_generate(&provider, 64, Policy::Qualified, 8);
        assert_eq!(
            short.kv_cache,
            Some(core_llm::KvCacheReport::dense(
                Reason::BelowMinimumContext,
                None
            ))
        );
        assert_eq!(
            short.text,
            kv_generate(&provider, 64, Policy::Off, 8).text,
            "a refused request generates exactly the dense output"
        );
    }

    /// Per-position logits of `model` prefilling `prompt` on `cache` and then teacher-forced on
    /// `stream`, one token per step: the prefill's last position, then one row per forced token.
    fn forced_logits(
        model: &CausalLm,
        cache: &mut dyn KvCache,
        prompt: &[i32],
        stream: &[i32],
    ) -> Vec<Vec<f32>> {
        let host = |logits: Array| {
            let logits = logits.as_dtype(Dtype::Float32).unwrap();
            logits.eval().unwrap();
            logits.as_slice::<f32>().to_vec()
        };
        let mut rows = vec![host(model.step(&input_ids(prompt), cache, 0).unwrap())];
        for (i, &token) in stream.iter().enumerate() {
            let offset = (prompt.len() + i) as i32;
            rows.push(host(
                model.step(&input_ids(&[token]), cache, offset).unwrap(),
            ));
        }
        rows
    }

    /// AC3 logit parity: on the cache the production plan selects for a request at the family's
    /// qualified minimum context, every teacher-forced decode position's logits stay within 8-bit
    /// rounding of the dense cache's. Returns the largest error relative to the dense logit range.
    fn compressed_logit_error(
        snapshot: &tempfile::TempDir,
        family: core_llm::KvModelFamily,
    ) -> f32 {
        let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
            .iter()
            .find(|row| row.family == family)
            .unwrap();
        let provider = load_tiny(snapshot);
        let Decoder::Causal(model) = &provider.model else {
            panic!("a qualified family is a causal decoder");
        };
        let prompt = (0..row.min_context_tokens)
            .map(|i| ((i * 7) % 26 + 6) as i32)
            .collect::<Vec<_>>();
        let stream = (0..40).map(|i| (i * 5) % 26 + 6).collect::<Vec<i32>>();
        let KvPlan::Compressed { format, reader } = provider.plan_kv_cache(
            core_llm::KvCompressionPolicy::Qualified,
            prompt.len(),
            stream.len() as u32,
            1,
            false,
        ) else {
            panic!("the qualified context must plan compressed");
        };
        let (mut cache, refused) = crate::kv_policy::select_compressed_cache(
            model,
            reader,
            prompt.len(),
            admit_any_transition(),
        );
        assert_eq!(refused, None);
        let compressed = forced_logits(model, cache.as_mut(), &prompt, &stream);
        let report = crate::kv_policy::compressed_report(format, None, cache.as_ref()).unwrap();
        assert!(report.ran_compressed(), "{report:?}");
        assert_eq!(
            report.counters.fused_attention_calls,
            2 * stream.len() as u64
        );
        let mut dense_cache = model.make_cache();
        let dense = forced_logits(model, dense_cache.as_mut(), &prompt, &stream);
        let range = dense
            .iter()
            .flatten()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        assert!(range > 0.0);
        dense
            .iter()
            .zip(&compressed)
            .flat_map(|(dense, compressed)| dense.iter().zip(compressed))
            .map(|(dense, compressed)| (dense - compressed).abs() / range)
            .fold(0.0f32, f32::max)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn compressed_kv_decode_logits_match_dense_within_8_bit_rounding() {
        for (identity, qk_norm, family) in [
            (
                json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
                false,
                core_llm::KvModelFamily::Llama,
            ),
            (
                json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
                true,
                core_llm::KvModelFamily::Qwen3,
            ),
        ] {
            let snapshot = tiny_snapshot(identity, 40_960, qk_norm);
            // Measured ~0.009 at 8 bits (bf16 kernel rounding included) against ~0.3 for the same
            // cache at 2 bits; 0.02 is the bound the SC-20671 8-bit prompt-cache test uses.
            let error = compressed_logit_error(&snapshot, family);
            eprintln!("sc-20679 {family:?}: K8V8 decode logit error {error}");
            assert!(
                error < 0.02,
                "{family:?}: K8V8 decode logits drifted {error}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn qwen3_compressed_kv_matches_dense_at_its_qualified_context() {
        assert_compressed_matches_dense(
            json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
            true,
            core_llm::KvModelFamily::Qwen3,
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn llama_compressed_kv_matches_dense_at_its_qualified_context() {
        assert_compressed_matches_dense(
            json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
            false,
            core_llm::KvModelFamily::Llama,
        );
    }

    /// AC1: a model whose family has no qualification row stays dense when opted in, and says so;
    /// so does every non-text path the plan refuses before any cache exists.
    #[test]
    fn unqualified_models_stay_dense_with_their_reason() {
        use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
        // Mistral shares the Llama decoder but is not the family the evidence measured.
        let snapshot = tiny_snapshot(
            json!({"architectures": ["MistralForCausalLM"], "model_type": "mistral"}),
            2048,
            false,
        );
        let provider = load_tiny(&snapshot);
        let output = kv_generate(&provider, 64, Policy::Qualified, 4);
        assert_eq!(
            output.kv_cache,
            Some(core_llm::KvCacheReport::dense(
                Reason::UnqualifiedModel,
                None
            ))
        );
        // The plan refuses a multimodal request on a qualified family before any cache exists.
        let qwen = load_tiny(&tiny_snapshot(
            json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
            40_960,
            true,
        ));
        let KvPlan::Dense(report) = qwen.plan_kv_cache(Policy::Qualified, 20_000, 0, 1, true)
        else {
            panic!("a multimodal request must plan dense");
        };
        assert_eq!(report.fallback, Some(Reason::UnsupportedRequest));
        assert!(matches!(
            qwen.plan_kv_cache(Policy::Qualified, 20_000, 64, 1, false),
            KvPlan::Compressed { .. }
        ));
        assert!(matches!(
            qwen.plan_kv_cache(Policy::Off, 20_000, 64, 1, false),
            KvPlan::Dense(report) if report.fallback == Some(Reason::PolicyDisabled)
        ));
        // The plan bounds the final context, not just the prompt: a fit-boundary prompt whose
        // budget runs past the evidenced 40 960-token window stays dense.
        assert!(matches!(
            qwen.plan_kv_cache(Policy::Qualified, 40_448, 512, 1, false),
            KvPlan::Compressed { .. }
        ));
        assert!(matches!(
            qwen.plan_kv_cache(Policy::Qualified, 40_448, 513, 1, false),
            KvPlan::Dense(report) if report.fallback == Some(Reason::AboveQualifiedContext)
        ));
    }

    /// AC1 at the production entry point: the request's token budget counts against the
    /// qualified range. A Qwen3 prompt that fits the evidenced 40 960-token window runs compressed
    /// only while prompt + `max_new_tokens` stays inside it.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_final_context_bounds_a_production_compressed_request() {
        use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
        let provider = load_tiny(&tiny_snapshot(
            json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
            41_472,
            true,
        ));
        // The chat template's own tokens, so the prompt lands 8 tokens inside the window.
        let one_word = TextLlmRequest {
            messages: vec![Message::text(Role::User, "w6")],
            ..Default::default()
        };
        let overhead = provider
            .render_prompt(&one_word, &one_word.messages)
            .unwrap()
            .1
            .len()
            - 1;
        let words = 40_960 - overhead - 8;
        let (inside, prompt_ids, _) =
            kv_generate_words(&provider, words, (7, 26), Policy::Qualified, 1);
        let headroom = u32::try_from(40_960 - prompt_ids.len()).unwrap();
        assert_eq!(headroom, 8);
        assert!(inside.kv_cache.unwrap().ran_compressed());
        let fits = kv_generate(&provider, words, Policy::Qualified, headroom);
        assert!(fits.kv_cache.unwrap().ran_compressed());
        let past = kv_generate(&provider, words, Policy::Qualified, headroom + 1);
        assert_eq!(
            past.kv_cache,
            Some(core_llm::KvCacheReport::dense(
                Reason::AboveQualifiedContext,
                None
            ))
        );
    }

    /// A provider assembled from parts cannot tell Llama from a Mistral or dense Qwen2 checkpoint
    /// sharing its decoder, so it never qualifies as a table family.
    #[test]
    fn a_provider_from_parts_is_never_a_qualified_family() {
        use core_llm::{KvCacheFallbackReason as Reason, KvCompressionPolicy as Policy};
        let snapshot = tiny_packed_capable_snapshot();
        let tokenizer = Tokenizer::from_file(snapshot.path().join("tokenizer.json")).unwrap();
        // A Mistral-shaped decoder: `Architecture::Llama`, exactly what `from_parts` receives.
        let model = tiny_packed_capable_model();
        assert_eq!(model.config().architecture, Architecture::Llama);
        let provider = LlamaProvider::from_parts(model, tokenizer, vec![99]);
        assert!(matches!(
            provider.plan_kv_cache(Policy::Qualified, 40_000, 64, 1, false),
            KvPlan::Dense(report) if report.fallback == Some(Reason::UnqualifiedModel)
        ));
    }

    /// sc-20682: on the generic fused-attention contract the compressed estimate differs from the
    /// dense one by exactly the K/V term (dense K/V out, the format-derived compressed bytes in);
    /// a contract that cannot run compressed keeps its dense estimate.
    #[test]
    fn compressed_admission_replaces_only_the_kv_term() {
        let geometry = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 8,
            head_dim: 128,
            layers: 28,
            element_bytes: 2,
            score_element_bytes: 4,
            hidden_size: 3072,
            intermediate_size: 8192,
            vocab_size: 128_256,
            recurrent_bytes: 0,
        };
        let format = core_llm::KvCompressionFormat::GroupAffineK8V8;
        let (prompt, new) = (32_768_usize, 512_u32);
        let total = prompt as u64 + u64::from(new);
        let compressed_kv =
            crate::kv_policy::compressed_request_kv_bytes(format, &geometry, total).unwrap();
        let dense_kv = geometry.kv_shape().dense_bytes(total).unwrap();
        assert!(compressed_kv < dense_kv);
        let dense =
            estimate_mlx_request_bytes(prompt, new, geometry, 0, 0, MlxWorkspaceContract::Chunked)
                .unwrap();
        let compressed = estimate_mlx_compressed_request_bytes(
            prompt,
            new,
            geometry,
            0,
            0,
            MlxWorkspaceContract::Chunked,
            compressed_kv,
        )
        .unwrap();
        assert_eq!(dense - compressed, dense_kv - compressed_kv);
        assert_eq!(
            estimate_mlx_compressed_request_bytes(
                prompt,
                new,
                geometry,
                0,
                0,
                MlxWorkspaceContract::Eager,
                compressed_kv,
            ),
            estimate_mlx_request_bytes(prompt, new, geometry, 0, 0, MlxWorkspaceContract::Eager)
        );
    }

    /// A mid-generation admission refusal (a compressed cache's refused dense transition) reaches
    /// the contract as the typed resource error, not a backend failure.
    #[test]
    fn a_typed_engine_refusal_is_the_contract_resource_error() {
        let evidence = core_llm::RequestResourceExhausted {
            prompt_tokens: 3,
            max_new_tokens: 2,
            max_context_tokens: 9,
            required_bytes: 8,
            available_bytes: 7,
        };
        assert!(matches!(
            to_core(crate::Error::ResourceExhausted(evidence)),
            CoreError::RequestResourceExhausted(mapped) if mapped == evidence
        ));
    }

    /// Restores the operational memory budget when a test that pins it ends.
    struct MemoryBudget;

    impl MemoryBudget {
        fn pin(bytes: u64) -> Self {
            std::env::set_var(core_llm::AVAILABLE_MEMORY_OVERRIDE, bytes.to_string());
            Self
        }
    }

    impl Drop for MemoryBudget {
        fn drop(&mut self) {
            std::env::remove_var(core_llm::AVAILABLE_MEMORY_OVERRIDE);
        }
    }

    /// An 8-bit reader for the K8V8 identity whose backend the cache refuses to bind.
    #[derive(Debug)]
    struct UnbindableReader;

    impl crate::primitives::RetainedPackedKernel for UnbindableReader {
        fn cache_identity(&self) -> &str {
            crate::primitives::PACKED_METAL_B8_IDENTITY
        }
        fn backend(&self) -> &str {
            "cpu"
        }
        fn retained_host_bytes_estimate(&self) -> usize {
            0
        }
        fn dispatch(
            &self,
            _args: &crate::primitives::PackedAttentionArgs<'_>,
        ) -> crate::error::Result<Array> {
            Err(crate::error::Error::Msg("never dispatched".into()))
        }
        fn code_bits(&self) -> crate::primitives::PackedCodeBits {
            crate::primitives::PackedCodeBits::Eight
        }
    }

    /// sc-20682 AC2 through the production entry point: under a budget that holds the compressed
    /// estimate but not the dense one, the qualified request is admitted and runs compressed,
    /// while the same request un-opted is refused at the dense price; a request whose selection
    /// then refuses the compressed cache is admitted again at the dense price before it runs.
    #[cfg(target_os = "macos")]
    #[test]
    fn admission_prices_compressed_kv_only_when_the_request_runs_it() {
        use core_llm::{KvCompressionFormat as Format, KvCompressionPolicy as Policy};
        let row = core_llm::KV_COMPRESSION_QUALIFICATIONS
            .iter()
            .find(|row| row.family == core_llm::KvModelFamily::Qwen3)
            .unwrap();
        let words = usize::try_from(row.min_context_tokens).unwrap();
        // As deep as the evidence model (28 layers): the compressed cache's per-layer transients
        // amortize over the depth, so a two-layer toy prices above dense.
        let snapshot = tiny_snapshot(
            json!({
                "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                "num_hidden_layers": 28,
            }),
            row.min_context_tokens + 1024,
            true,
        );
        let provider = load_tiny(&snapshot);
        const NEW: u32 = 8;
        let prompt = kv_generate(&provider, words, Policy::Off, NEW)
            .usage
            .prompt_tokens as usize;
        let estimate = |format| {
            provider
                .admission_estimate(format, prompt, NEW, 0, 0)
                .unwrap()
        };
        let (compressed, dense) = (estimate(Some(Format::GroupAffineK8V8)), estimate(None));
        assert!(compressed < dense, "{compressed} vs {dense}");

        let budget = MemoryBudget::pin(compressed);
        let output = kv_generate(&provider, words, Policy::Qualified, NEW);
        assert!(output.kv_cache.as_ref().unwrap().ran_compressed());
        let refuse = |provider: &LlamaProvider, policy| {
            let request = TextLlmRequest {
                messages: vec![Message::text(
                    Role::User,
                    (0..words)
                        .map(|i| format!("w{}", (i * 7) % 26 + 6))
                        .collect::<Vec<_>>()
                        .join(" "),
                )],
                sampling: core_llm::Sampling::greedy(),
                max_new_tokens: NEW,
                seed: Some(0),
                kv_compression: policy,
                ..Default::default()
            };
            match provider.generate(&request, &mut |_| {}) {
                Err(CoreError::RequestResourceExhausted(evidence)) => evidence,
                other => panic!("expected a dense-priced refusal, got {other:?}"),
            }
        };
        let evidence = refuse(&provider, Policy::Off);
        assert_eq!(
            (evidence.required_bytes, evidence.available_bytes),
            (dense, compressed)
        );

        let unbindable = load_tiny(&snapshot);
        #[allow(clippy::arc_with_non_send_sync)]
        let reader =
            crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(UnbindableReader));
        assert!(unbindable.kv_reader.set(Ok(reader)).is_ok());
        let evidence = refuse(&unbindable, Policy::Qualified);
        assert_eq!(evidence.required_bytes, dense);
        drop(budget);
        let output = kv_generate(&unbindable, words, Policy::Qualified, NEW);
        let report = output.kv_cache.unwrap();
        assert_eq!(
            report.fallback,
            Some(core_llm::KvCacheFallbackReason::ReaderUnavailable)
        );
    }

    /// sc-20683 AC2: the cross-backend compressed-KV conformance table
    /// (`core_llm_testkit::kv_policy_cases`) through MLX's production plan, with the fused reader:
    /// the same table Candle's plan passes without one. The family each provider plans with is the
    /// shared `core_llm::kv_model_family` of its loaded config.
    #[test]
    fn mlx_kv_plan_conforms_to_the_cross_backend_policy_table() {
        use core_llm::KvModelFamily;
        use core_llm_testkit::{kv_policy_conformance, KvBackendDecision, KvReader};
        let load = |identity, qk_norm| load_tiny(&tiny_snapshot(identity, 2048, qk_norm));
        let llama = load(
            json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
            false,
        );
        let qwen = load(
            json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
            true,
        );
        // Mistral loads through the Llama decoder but is not a table family.
        let mistral = load(
            json!({"architectures": ["MistralForCausalLM"], "model_type": "mistral"}),
            false,
        );
        for (provider, family) in [
            (&llama, Some(KvModelFamily::Llama)),
            (&qwen, Some(KvModelFamily::Qwen3)),
            (&mistral, None),
        ] {
            assert_eq!(provider.kv_family, family);
        }
        kv_policy_conformance(KvReader::Fused, |case| {
            let provider = match case.family {
                Some(KvModelFamily::Llama) => &llama,
                Some(KvModelFamily::Qwen3) => &qwen,
                None => &mistral,
            };
            match provider.plan_kv_cache(
                case.policy,
                usize::try_from(case.context_tokens).unwrap(),
                case.max_new_tokens,
                case.batch,
                case.multimodal,
            ) {
                KvPlan::Compressed { format, .. } => KvBackendDecision::Compressed(format),
                KvPlan::Dense(report) => KvBackendDecision::Dense(report),
                KvPlan::Unreported => panic!("a product plan always reports"),
            }
        });
    }

    /// A reader that binds to the K8V8 cache and faults on every dispatch.
    #[derive(Debug)]
    struct FaultingReader;

    impl crate::primitives::RetainedPackedKernel for FaultingReader {
        fn cache_identity(&self) -> &str {
            crate::primitives::PACKED_METAL_B8_IDENTITY
        }
        fn backend(&self) -> &str {
            "mlx-metal"
        }
        fn retained_host_bytes_estimate(&self) -> usize {
            0
        }
        fn dispatch(
            &self,
            _args: &crate::primitives::PackedAttentionArgs<'_>,
        ) -> crate::error::Result<Array> {
            Err(crate::error::Error::Msg("injected dispatch fault".into()))
        }
        fn code_bits(&self) -> crate::primitives::PackedCodeBits {
            crate::primitives::PackedCodeBits::Eight
        }
    }

    /// AC1: a compressed generation that the cache moves to dense part-way, or whose reader the
    /// selection refuses, reports the dense outcome with its reason — never a compressed label.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_reader_fault_or_refusal_is_reported_as_a_dense_fallback() {
        use core_llm::{KvCacheFallbackReason as Reason, KvCompressionFormat as Format};
        let model = tiny_packed_capable_model();
        let prompt = (0..40).map(|i| i % 31 + 1).collect::<Vec<i32>>();
        let config = GenerationConfig {
            max_new_tokens: 4,
            seed: Some(0),
            ..Default::default()
        };
        let run = |reader: crate::primitives::CompiledKernelHandle| {
            let (mut cache, refused) = crate::kv_policy::select_compressed_cache(
                &model,
                reader,
                prompt.len(),
                admit_any_transition(),
            );
            generate_with_timings_on(
                &model,
                cache.as_mut(),
                &prompt,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
            )
            .unwrap();
            crate::kv_policy::compressed_report(Format::GroupAffineK8V8, refused, cache.as_ref())
                .unwrap()
        };

        #[allow(clippy::arc_with_non_send_sync)]
        let faulting =
            crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(FaultingReader));
        let report = run(faulting);
        assert_eq!(report.format, Some(Format::GroupAffineK8V8));
        assert_eq!(report.fallback, Some(Reason::RuntimeFallback), "{report:?}");
        assert!(!report.ran_compressed());
        assert!(report.counters.dense_fallback_events > 0);
        assert!(
            report
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("dispatch")),
            "{report:?}"
        );

        let refused = crate::primitives::CompiledKernelHandle::new(std::sync::Arc::new(
            crate::primitives::OpaqueCompiledKernel::new(
                "sc20679-refused",
                "cpu",
                0,
                std::sync::Arc::new(()),
            ),
        ));
        let report = run(refused);
        assert_eq!(report.format, None);
        assert_eq!(
            report.fallback,
            Some(Reason::ReaderUnavailable),
            "{report:?}"
        );
        assert!(report.detail.is_some());

        let healthy =
            crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight)
                .unwrap();
        assert!(run(healthy).ran_compressed());
    }

    /// AC2: no dense K/V mirror survives an attention call on the compressed path. After the
    /// prefill and after every decode step, the compressed cache's dense fallback owns no K/V
    /// (handle accounting), and MLX active memory above the pre-request baseline is the compressed
    /// representation's own device bytes (byte accounting) — far below what any one layer's dense
    /// K or V of the same tokens would add.
    #[cfg(target_os = "macos")]
    #[test]
    fn no_dense_kv_mirror_survives_a_compressed_attention_call() {
        use mlx_rs::memory;
        // Four K/V heads of 128 make one layer's dense K (1.5 MB at 1536 bf16 tokens) dwarf the
        // accounting slack.
        let model = tiny_causal_model(4, 4, 128);
        let reader =
            crate::kv_policy::group_affine_reader(crate::primitives::PackedCodeBits::Eight)
                .unwrap();
        let prompt = (0..1536).map(|i| i % 31 + 1).collect::<Vec<i32>>();
        let step = |cache: &mut dyn KvCache, ids: &[i32], offset: usize| {
            let logits = model.step(&input_ids(ids), cache, offset as i32).unwrap();
            logits.eval().unwrap();
        };
        // Warm every lazily retained model and reader state on a throwaway cache first.
        {
            let (mut warm, refused) = crate::kv_policy::select_compressed_cache(
                &model,
                reader.clone(),
                prompt.len(),
                admit_any_transition(),
            );
            assert_eq!(refused, None);
            step(warm.as_mut(), &prompt, 0);
            step(warm.as_mut(), &[3], prompt.len());
            warm.reset().unwrap();
        }
        memory::clear_cache();
        let baseline = memory::get_active_memory() as u64;

        let (mut cache, refused) = crate::kv_policy::select_compressed_cache(
            &model,
            reader,
            prompt.len(),
            admit_any_transition(),
        );
        assert_eq!(refused, None);
        const SLACK: u64 = 256 * 1024;
        let check = |cache: &dyn KvCache, tokens: u64| {
            let dense = cache
                .compressed_dense_fallback()
                .expect("the compressed cache owns an explicit dense fallback");
            assert_eq!(
                dense.retained_snapshot().unwrap(),
                None,
                "the dense fallback holds K/V after an attention call"
            );
            assert!(dense.events().is_empty(), "{:?}", dense.events());
            let evidence = cache.packed_evidence().unwrap();
            assert!(!evidence.dense_active && evidence.fallback_reasons.is_empty());
            assert_eq!(evidence.full_cache_dequantizations, 0);
            let storage = cache.compressed_storage().unwrap().unwrap();
            assert_eq!(storage.tokens, tokens);
            // One layer's dense K of the same tokens: what the smallest surviving mirror adds.
            let one_dense_tensor = tokens * 4 * 128 * storage.element_bytes;
            assert!(
                one_dense_tensor > 4 * SLACK,
                "the fixture must expose a mirror"
            );
            let growth = (memory::get_active_memory() as u64).saturating_sub(baseline);
            assert!(
                growth <= storage.device_bytes() + SLACK,
                "{tokens} tokens: active memory grew {growth} B; the compressed cache retains \
                 {} B (a dense K or V of one layer is {one_dense_tensor} B)",
                storage.device_bytes()
            );
        };
        step(cache.as_mut(), &prompt, 0);
        check(cache.as_ref(), prompt.len() as u64);
        // Decode steps cross the next 32-token group boundary and fill its residual.
        for i in 0..40 {
            let offset = prompt.len() + i;
            step(cache.as_mut(), &[(i as i32) % 31 + 1], offset);
            check(cache.as_ref(), offset as u64 + 1);
        }
        let evidence = cache.packed_evidence().unwrap();
        assert!(evidence.accepted_direct_calls >= 2 * 40, "{evidence:?}");
        cache.reset().unwrap();
    }
}
