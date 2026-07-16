//! Generic Llama-family causal decoder, config-dispatched across architectures (Llama / Mistral,
//! Qwen3, Phi-3, Qwen2-MoE, Gemma-2, GLM-4, DeepSeek-V2).
//!
//! The Candle port of `mlx-llm`'s `CausalLm`, modelled alongside `candle-gen-sensenova`'s
//! hand-rolled Qwen3 stack. One block shape covers the family: self-attention is either grouped-query
//! attention (with optional per-head q/k RMSNorm for Qwen3) or Multi-head Latent Attention (DeepSeek's
//! low-rank KV path); the FFN is a dense gated MLP or a sparse Mixture-of-Experts bank; norms are the
//! Llama pre-norm or the Gemma-2/GLM-4 4-norm sandwich. Projections are held behind [`Projection`] so
//! a model can be quantized on load. The forward is `&self`; the KV cache is the only mutable state,
//! threaded in as `&mut dyn KvCache`.
//!
//! Shapes are batch-capable (`[batch, seq, …]`). `head_dim` is taken from config and may differ from
//! `hidden_size / num_heads` (e.g. Qwen3-0.6B: hidden 1024, 16 heads, head_dim 128). Compute runs in
//! the device's [`compute_dtype`] (bf16 on GPU, f32 on CPU).

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};

use crate::config::{Architecture, ModelConfig};
use crate::device::compute_dtype;
use crate::error::Result;
use crate::models::deepstack::{self, deepstack_fused_decoder_layers, MropePositions};
use crate::primitives::attention::{sdpa, AttnMask};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, gelu, rms_norm, silu, soft_cap};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{repeat_kv, ContiguousKvCache, PagedKvCache, Weights};

/// A loaded causal decoder.
pub struct CausalLm {
    embed_tokens: Tensor,
    layers: Vec<LlamaLayer>,
    norm: Tensor,
    lm_head: Linear,
    rope: Rope,
    cfg: ModelConfig,
    dtype: DType,
    device: Device,
    /// Per-layer compute device — every entry equals `device` for a single-device model, and differs
    /// when the model is **pipeline-sharded** across GPUs (contiguous layer blocks per device). Drives
    /// the cross-device hidden-state hand-off in the decoder loop; `Tensor::to_device` is a no-op clone
    /// when a layer already sits on the running device, so the single-device path stays zero-cost.
    layer_devices: Vec<Device>,
    quantized: bool,
    /// Gemma scales token embeddings by √hidden; `None` ⇒ no scaling.
    embed_scale: Option<f64>,
    /// Gemma-2 final-logit soft-cap; `None` ⇒ no cap.
    final_softcap: Option<f32>,
}

impl CausalLm {
    /// Build from a loaded checkpoint (dense). `prefix` is the weight-key prefix (`""` for a plain
    /// `*ForCausalLM`, e.g. `"language_model"` for a VLM-nested decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ModelConfig) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the attention/MLP projections on load.
    /// Embeddings, the LM head, and norms always stay dense. The compute dtype is the device default
    /// ([`compute_dtype`] — bf16 on GPU, f32 on CPU); use [`CausalLm::from_weights_dtype`] to pick it.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: ModelConfig,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        Self::from_weights_dtype(w, prefix, cfg, quant, compute_dtype(w.device()))
    }

    /// Like [`CausalLm::from_weights_with`] but with an explicit dense compute `dtype` — the
    /// f16-vs-bf16 knob for CUDA dtype perf tuning (story 7263). Dequantized projections accumulate in
    /// this dtype too. Passing a dtype the backend can't run (e.g. f16 on CPU) surfaces at forward time.
    pub fn from_weights_dtype(
        w: &Weights,
        prefix: &str,
        cfg: ModelConfig,
        quant: Option<QuantSpec>,
        dtype: DType,
    ) -> Result<Self> {
        let device = w.device().clone();
        // The Qwen3-VL VLM wrapper nests the decoder under `model.language_model.*` (embeddings,
        // norm, `layers.{i}.*`) with the untied `lm_head.weight` at the checkpoint root; a plain
        // `*ForCausalLM` keeps the historical `[{prefix}.]model.*` / `[{prefix}.]lm_head.weight`
        // layout. `decoder_root` carries the right stem so the per-key suffixes below are uniform.
        let vlm_nested = cfg.architecture.is_qwen3_vl();
        let decoder_root = if vlm_nested {
            "model.language_model".to_string()
        } else {
            join(prefix, "model")
        };
        let p = |suffix: &str| join(&decoder_root, suffix);
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        let proj = |key: String| -> Result<Projection> { Projection::load(req(key)?, quant) };
        // Like `proj`, but also loads a sibling `.bias` when present (Qwen2 attention carries q/k/v
        // bias; Llama / Qwen3 / Phi-3 do not).
        let proj_b = |wkey: String| -> Result<Projection> {
            let stem = wkey.strip_suffix(".weight").unwrap_or(&wkey);
            let bkey = format!("{stem}.bias");
            let bias = if w.contains(&bkey) {
                Some(req(bkey)?)
            } else {
                None
            };
            Projection::load_with_bias(req(wkey)?, bias, quant)
        };
        // Gemma's norms are `(1 + weight)`; fold the +1 into the stored weight so the standard
        // `rms_norm` applies it. (Llama / Qwen3 norm weights are used verbatim.)
        let gemma = cfg.architecture.is_gemma2();
        let norm_w = |key: String| -> Result<Tensor> {
            let t = req(key)?;
            if gemma {
                Ok(t.affine(1.0, 1.0)?)
            } else {
                Ok(t)
            }
        };

        let embed_tokens = req(p("embed_tokens.weight"))?;
        let norm = norm_w(p("norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.clone(), None)
        } else {
            let head_key = if vlm_nested {
                "lm_head.weight".to_string()
            } else {
                join(prefix, "lm_head.weight")
            };
            Linear::new(req(head_key)?, None)
        };

        let qk_norm = cfg.has_qk_norm();
        let groups = cfg.groups() as usize;
        let num_heads = cfg.num_heads as usize;
        let num_kv_heads = cfg.num_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let scale = cfg.attn_scale();
        let eps = cfg.rms_norm_eps as f64;
        // Phi-3 fuses q‖k‖v into one `qkv_proj` and gate‖up into one `gate_up_proj`; the row spans the
        // split slices below carve out (each `[out, hidden]`, so the split is along axis 0).
        let qd = num_heads * head_dim;
        let kvd = num_kv_heads * head_dim;
        let inter = cfg.intermediate_size as usize;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |suffix: &str| join(&decoder_root, &format!("layers.{i}.{suffix}"));

            // Attention: Multi-head Latent Attention (DeepSeek-V2) or grouped-query attention. MLA's
            // low-rank q/kv projections are wholly distinct from GQA's q/k/v, so it is its own path.
            let attn = if cfg.architecture.is_mla() {
                Attention::Mla(MlaAttention::load(w, lp, &cfg, dtype, quant)?)
            } else {
                let (q_norm, k_norm) = if qk_norm {
                    (
                        Some(req(lp("self_attn.q_norm.weight"))?),
                        Some(req(lp("self_attn.k_norm.weight"))?),
                    )
                } else {
                    (None, None)
                };
                // A packed `qkv_proj` (Phi-3, no bias) is split into q/k/v, else the separate
                // `q_proj`/`k_proj`/`v_proj` are loaded directly (with q/k/v bias for Qwen2).
                let (q, k, v) = {
                    let packed = lp("self_attn.qkv_proj.weight");
                    if w.contains(&packed) {
                        let qkv = req(packed)?; // [qd + 2*kvd, hidden]
                        (
                            Projection::load(qkv.narrow(0, 0, qd)?.contiguous()?, quant)?,
                            Projection::load(qkv.narrow(0, qd, kvd)?.contiguous()?, quant)?,
                            Projection::load(qkv.narrow(0, qd + kvd, kvd)?.contiguous()?, quant)?,
                        )
                    } else {
                        (
                            proj_b(lp("self_attn.q_proj.weight"))?,
                            proj_b(lp("self_attn.k_proj.weight"))?,
                            proj_b(lp("self_attn.v_proj.weight"))?,
                        )
                    }
                };
                Attention::Gqa(LlamaAttention {
                    q,
                    k,
                    v,
                    o: proj_b(lp("self_attn.o_proj.weight"))?,
                    q_norm,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    groups,
                    eps,
                    softcap: cfg.attn_logit_softcap,
                    rope_interleaved: cfg.architecture.rope_interleaved(),
                })
            };

            // Feed-forward: a sparse Mixture-of-Experts bank or a dense MLP. DeepSeek keeps its leading
            // `first_k_dense_replace` layers dense even though the model is MoE. Gemma uses GeGLU
            // (gelu); everything else SwiGLU (silu).
            let moe_layer = cfg.moe.filter(|m| i >= m.first_k_dense_replace);
            let ffn = if let Some(moe) = moe_layer {
                let mut experts = Vec::with_capacity(moe.num_experts);
                for e in 0..moe.num_experts {
                    let ep = |s: &str| lp(&format!("mlp.experts.{e}.{s}"));
                    experts.push(LlamaMlp {
                        gate: proj(ep("gate_proj.weight"))?,
                        up: proj(ep("up_proj.weight"))?,
                        down: proj(ep("down_proj.weight"))?,
                        gelu: false,
                    });
                }
                // Shared expert key stem: DeepSeek packs `n_shared_experts` into `mlp.shared_experts`
                // (plural, ungated); Qwen2-MoE has a single `mlp.shared_expert` gated by a sigmoid.
                let shared_stem = if w.contains(&lp("mlp.shared_experts.gate_proj.weight")) {
                    "mlp.shared_experts"
                } else {
                    "mlp.shared_expert"
                };
                let shared_gate_key = lp("mlp.shared_expert_gate.weight");
                Ffn::Moe(MoeMlp {
                    router: req(lp("mlp.gate.weight"))?, // [num_experts, hidden]
                    experts,
                    shared: LlamaMlp {
                        gate: proj(lp(&format!("{shared_stem}.gate_proj.weight")))?,
                        up: proj(lp(&format!("{shared_stem}.up_proj.weight")))?,
                        down: proj(lp(&format!("{shared_stem}.down_proj.weight")))?,
                        gelu: false,
                    },
                    shared_gate: if w.contains(&shared_gate_key) {
                        Some(req(shared_gate_key)?) // [1, hidden]
                    } else {
                        None
                    },
                    experts_per_tok: moe.num_experts_per_tok,
                    norm_topk_prob: moe.norm_topk_prob,
                    routed_scaling_factor: moe.routed_scaling_factor,
                })
            } else {
                // Dense MLP; Phi-3 fuses gate‖up into one weight, split along axis 0.
                let (gate, up) = {
                    let packed = lp("mlp.gate_up_proj.weight");
                    if w.contains(&packed) {
                        let gu = req(packed)?; // [2*inter, hidden]
                        (
                            Projection::load(gu.narrow(0, 0, inter)?.contiguous()?, quant)?,
                            Projection::load(gu.narrow(0, inter, inter)?.contiguous()?, quant)?,
                        )
                    } else {
                        (
                            proj(lp("mlp.gate_proj.weight"))?,
                            proj(lp("mlp.up_proj.weight"))?,
                        )
                    }
                };
                Ffn::Dense(LlamaMlp {
                    gate,
                    up,
                    down: proj(lp("mlp.down_proj.weight"))?,
                    gelu: gemma,
                })
            };

            // Gemma-2 / GLM-4 wrap the block in a 4-norm "sandwich" (pre+post for both attn and MLP);
            // the Llama shape has only the two pre-norms. The norm key names differ per family.
            let (post_attn_key, pre_ff_key, post_ff_key) = match cfg.architecture {
                Architecture::Glm4 => (
                    "post_self_attn_layernorm",
                    "post_attention_layernorm",
                    "post_mlp_layernorm",
                ),
                // Gemma-2 (and the default fallback for the post-attention norm name).
                _ => (
                    "post_attention_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                ),
            };
            let (pre_ff_ln, post_ff_ln) = if cfg.architecture.is_sandwich() {
                (
                    Some(norm_w(lp(&format!("{pre_ff_key}.weight")))?),
                    Some(norm_w(lp(&format!("{post_ff_key}.weight")))?),
                )
            } else {
                (None, None)
            };

            layers.push(LlamaLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp(&format!("{post_attn_key}.weight")))?,
                pre_ff_ln,
                post_ff_ln,
                attn,
                ffn,
                eps,
            });
        }

        let rope = cfg.build_rope();
        // The per-layer device is wherever that layer's weights landed — equal to `device` for a
        // normal load, distinct blocks when the `Weights` were placed by a sharded loader.
        let layer_devices: Vec<Device> =
            layers.iter().map(|l| l.input_ln.device().clone()).collect();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            dtype,
            device,
            layer_devices,
            quantized: quant.is_some(),
            embed_scale: gemma.then(|| (cfg.hidden_size as f64).sqrt()),
            final_softcap: cfg.final_logit_softcap,
            cfg,
        })
    }

    /// Load a plain (`*ForCausalLM`) decoder from `dir`, **pipeline-sharded** across `devices` in
    /// contiguous layer blocks, computing in `dtype` — for a model too large to fit on any single GPU
    /// (e.g. splitting across 2×24GB cards). Layer block `b` of `L` layers goes on `devices[b]`, the
    /// token embeddings + first input on `devices[0]`, the final norm + LM head on the last device, and
    /// the decoder hands the hidden state across each boundary. The sharded [`Weights`] loader streams
    /// each file through host memory, so no single GPU ever holds more than its own shard. Dense only —
    /// quantize-on-load is not combined with sharding (use one *or* the other to fit). `devices` must be
    /// non-empty; a single-element slice is an ordinary single-GPU load.
    pub fn from_dir_sharded(
        dir: impl AsRef<std::path::Path>,
        cfg: ModelConfig,
        dtype: DType,
        devices: &[Device],
    ) -> Result<Self> {
        if devices.is_empty() {
            return Err(crate::error::Error::Msg(
                "from_dir_sharded: needs at least one device".into(),
            ));
        }
        let plan = shard_plan(devices, cfg.num_layers);
        let w = Weights::from_dir_sharded(dir, devices[0].clone(), plan)?;
        Self::from_weights_dtype(&w, "", cfg, None, dtype)
    }

    /// The devices this model's layers live on, in layer order (all equal unless pipeline-sharded).
    pub fn layer_devices(&self) -> &[Device] {
        &self.layer_devices
    }

    /// The model config.
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Whether the projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// The device the model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A fresh contiguous KV cache sized for this model.
    pub fn new_cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.cfg.num_layers)
    }

    /// A fresh single-sequence [`PagedKvCache`] sized for this model, backed by its own
    /// `block_size`-token block pool — the ragged-batch / prefix-sharing cache (story 7257). Drive it
    /// through [`generate_with_cache`](crate::decode::generate_with_cache); pack concurrent sequences
    /// as separate caches over a shared [`BlockPool`](crate::primitives::BlockPool).
    pub fn new_paged_cache(&self, block_size: usize) -> PagedKvCache {
        PagedKvCache::new(self.cfg.num_layers, block_size)
    }

    /// The engine's compute dtype for this model (bf16 on GPU, f32 on CPU) — the batched decode reads
    /// it to match its additive attention mask to the score dtype.
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// Build per-row RoPE `(cos, sin)` tables for a `[rows, cols]` grid of absolute positions
    /// (row-major flat `positions`, length `rows * cols`) — the **per-sequence** position tables the
    /// batched decode (story 7255) feeds [`CausalLm::decode_logits_masked`]. Each is
    /// `[rows, cols, head_dim]` in the compute dtype.
    pub fn rope_tables(&self, positions: &[i32], rows: i32, cols: i32) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rope.cos_sin_at(positions, self.dtype, &self.device)?; // [1, rows*cols, hd]
        let hd = self.rope.dim();
        Ok((
            cos.reshape((rows as usize, cols as usize, hd))?,
            sin.reshape((rows as usize, cols as usize, hd))?,
        ))
    }

    /// Embed token ids `[batch, seq]` (u32) → `[batch, seq, hidden]`. Gemma scales the embeddings by
    /// √hidden.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let e = embed(&self.embed_tokens, input_ids)?;
        match self.embed_scale {
            Some(s) => Ok(e.affine(s, 0.0)?),
            None => Ok(e),
        }
    }

    /// Run a forward step over token ids and return logits for the **last** position only,
    /// `[batch, vocab]`. `offset` is the position of the first input token (number of cached
    /// positions).
    pub fn decode_logits(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Like [`CausalLm::decode_logits`] but from pre-computed input embeddings — the hook the VLM
    /// path uses to splice image features before the decoder.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let s = input_embeds.dim(1)? as i32;
        // Single-sequence / uniform batch: positions [offset, offset+s) shared across the batch, with
        // an implicit bottom-right causal mask; cos/sin `[1, s, head_dim]` broadcast over the batch.
        let (cos, sin) = self.rope.cos_sin(s, offset, self.dtype, &self.device)?;
        self.forward_to_last_logits(input_embeds, cache, &cos, &sin, AttnMask::Causal)
    }

    /// Embed token ids `[1, S]` → `[1, S, hidden]` in the compute dtype — the Qwen3-VL multimodal
    /// splice point (image/video-token rows are overwritten with the vision tower's merged features).
    pub fn embed_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed(input_ids)?.to_dtype(self.dtype)?)
    }

    /// Replace every row of `embeds` `[1, S, hidden]` whose id is any of `placeholder_tokens` with the
    /// next `vision_features` row, in sequence order — the mixed image+video splice (delegates to the
    /// shared `deepstack::splice_vision_features`).
    pub fn splice_vision_features(
        &self,
        embeds: &Tensor,
        input_ids: &[i32],
        vision_features: &Tensor,
        placeholder_tokens: &[i32],
    ) -> Result<Tensor> {
        deepstack::splice_vision_features(embeds, input_ids, vision_features, placeholder_tokens)
    }

    /// Interleaved-M-RoPE 3-D position rows + `mrope_delta` for an image-only prompt (the
    /// `get_rope_index` port, B=1); see [`Self::mrope_positions_mm`] for image+video.
    pub fn mrope_positions(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            &[],
            image_token_id,
            spatial_merge_size,
        )
    }

    /// The full image **and** video interleaved-M-RoPE entry (see the shared
    /// `deepstack::mrope_positions_mm`).
    #[allow(clippy::too_many_arguments)]
    pub fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    /// Prefill precomputed `embeds` `[1, S, hidden]` (text embeds with vision features spliced in)
    /// using **interleaved M-RoPE** from the explicit 3-D `positions` (temporal/height/width rows,
    /// each length `S`) **and DeepStack feature fusion**: after decoder layer `i` (for
    /// `i < deepstack.len()`) the `i`-th tapped/merged ViT feature set is added to the visual-token
    /// rows (`visual_pos_mask`). Returns last-position logits `[1, vocab]`. The Qwen3-VL prefill seam
    /// (`Qwen3VLTextModel.forward` + `_deepstack_process`); with all three position rows equal and an
    /// empty `deepstack` it is bit-identical to a plain 1-D-RoPE prefill.
    pub fn decode_logits_from_embeds_mrope_deepstack(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Tensor],
    ) -> Result<Tensor> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            self.dtype,
            &self.device,
        )?;
        let h0 = embeds.to_dtype(self.dtype)?;
        let (b, s, _) = h0.dims3()?;
        // The hidden state + RoPE tables follow each layer onto its device (a no-op clone for a
        // single-device model); DeepStack features are moved onto the running device by the fusion.
        let mut cur = h0.device().clone();
        let mut cos_d = cos.clone();
        let mut sin_d = sin.clone();
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            self.layers.len(),
            |i, h| {
                let dev = &self.layer_devices[i];
                let h = if cur.same_device(dev) {
                    h.clone()
                } else {
                    cos_d = cos.to_device(dev)?;
                    sin_d = sin.to_device(dev)?;
                    cur = dev.clone();
                    h.to_device(dev)?
                };
                self.layers[i].forward(&h, &cos_d, &sin_d, AttnMask::Causal, cache, i)
            },
        )?;
        let last_h = h.narrow(1, s - 1, 1)?.contiguous()?;
        let logits = self.project_logits(&last_h)?;
        Ok(logits.reshape((b, self.cfg.vocab_size as usize))?)
    }

    /// Batched forward over a **left-padded** `[batch, seq]` step with **per-sequence** RoPE positions
    /// and an explicit additive attention mask — the decode primitive the dynamic-batch scheduler
    /// (story 7255) runs each step.
    ///
    /// `input_ids` is `[batch, seq]` (u32); `cos`/`sin` are `[batch, seq, head_dim]` (per-row
    /// positions, e.g. from [`CausalLm::rope_tables`]); `mask` is an additive
    /// `[batch, 1, seq, k_total]` score mask (`0` keep, large-negative block) covering left-padding +
    /// causality. Returns logits for the **last column** `[batch, vocab]` — left-padding right-aligns
    /// every row's last real token to that column, so one slice serves the whole batch.
    pub fn decode_logits_masked(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.forward_to_last_logits(&embeds, cache, cos, sin, AttnMask::Additive(mask))
    }

    /// **Per-sequence** batched decode step for iteration-level continuous batching (story 7347,
    /// `Throughput` mode): the embeddings / projections / MLP / lm_head are **batched** over the active
    /// sequences, but attention runs **per-sequence** — each row attends only its own sequence's real
    /// KV in `caches[i]` (its own paged cache, gathered to true length), with no padding mask. Returns
    /// the **last column** logits `[batch, vocab]`.
    ///
    /// `input_ids` is `[batch, seq]` (u32); `positions` are the per-row absolute RoPE positions (length
    /// `batch * seq`, row-major — for a decode step `seq == 1`, one position per sequence at its own
    /// offset). `caches.len()` must equal the batch. Unlike [`CausalLm::decode_logits`] run per row,
    /// the batched projections here are **not** bit-identical to the batch-1 logits — the batched matmul
    /// is not M-invariant on a GPU, so a row *tracks* its batch-1 run only to sub-ULP (the documented
    /// Throughput tradeoff that buys the weight-read amortization). The bit-exact continuous path runs
    /// each sequence through [`CausalLm::decode_logits`] on its own cache instead.
    pub fn decode_logits_per_seq(
        &self,
        input_ids: &Tensor,
        caches: &mut [&mut PagedKvCache],
        positions: &[i32],
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        if caches.len() != b {
            return Err(crate::error::Error::Msg(format!(
                "decode_logits_per_seq: {} caches for a batch of {b}",
                caches.len()
            )));
        }
        if positions.len() != b * s {
            return Err(crate::error::Error::Msg(format!(
                "decode_logits_per_seq: {} positions for a {b}x{s} step",
                positions.len()
            )));
        }
        // Story 7485: a batched **prefill** wave's caches are fresh (never written), so `reserve_step`
        // would fail for lack of a pool store; seed it from the model's cfg head shape + compute dtype
        // + device (idempotent — a no-op on every decode step, where the lanes already prefilled). The
        // shared pool means seeding via any one cache initializes it for the whole wave.
        if let Some(c) = caches.first() {
            c.ensure_pool_store(
                self.cfg.num_kv_heads as usize,
                self.cfg.head_dim as usize,
                self.dtype,
                &self.device,
            )?;
        }
        // Story 7453: reserve this step's positions for every sequence (advance the block tables), then
        // build the **one** gather index spanning all sequences' pooled token slots — the per-layer
        // attention does a single `index_select`, not an O(N) per-sequence gather.
        for c in caches.iter_mut() {
            c.reserve_step(s)?;
        }
        let plan = ThroughputGather::build(
            caches,
            self.cfg.num_kv_heads as usize,
            self.cfg.head_dim as usize,
            &self.device,
        )?;
        let (cos, sin) = self.rope_tables(positions, b as i32, s as i32)?;
        let mut h = self.embed(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_per_seq(&h, &cos, &sin, caches, &plan, i)?;
        }
        let last_h = h.narrow(1, s - 1, 1)?.contiguous()?; // [b, 1, hidden]
        let logits = self.project_logits(&last_h)?; // [b, 1, vocab]
        Ok(logits.reshape((b, self.cfg.vocab_size as usize))?)
    }

    /// Run a forward step over token ids and return logits for **every** position, `[batch, seq,
    /// vocab]`. The all-positions forward speculative decoding (stories 7259 / 7260) runs to verify
    /// `[cur, draft₁ … draftₖ]` in one pass — `logits[.., i, ..]` predicts the token after position
    /// `i`. `offset` is the position of the first input token (number of cached positions); the mask
    /// is the implicit bottom-right causal one.
    pub fn decode_logits_all(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        let s = embeds.dim(1)? as i32;
        let (cos, sin) = self.rope.cos_sin(s, offset, self.dtype, &self.device)?;
        let h = self.run_decoder_stack(&embeds, cache, &cos, &sin, AttnMask::Causal)?;
        self.project_logits(&h) // [b, s, vocab]
    }

    /// Run the decoder stack over `input_embeds` with the given RoPE tables and attention mask, and
    /// project the **last column** to logits `[batch, vocab]`. The shared core of the single and
    /// batched forwards: they differ only in how `cos`/`sin` and `mask` are built.
    fn forward_to_last_logits(
        &self,
        input_embeds: &Tensor,
        cache: &mut dyn KvCache,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
    ) -> Result<Tensor> {
        let (b, s, _) = input_embeds.dims3()?;
        let h = self.run_decoder_stack(input_embeds, cache, cos, sin, mask)?;
        let last_h = h.narrow(1, s - 1, 1)?.contiguous()?; // [b, 1, hidden]
        let logits = self.project_logits(&last_h)?; // [b, 1, vocab]
        Ok(logits.reshape((b, self.cfg.vocab_size as usize))?)
    }

    /// Run the decoder stack, returning the final hidden states `[batch, seq, hidden]` (pre-norm /
    /// pre-`lm_head`). Shared by the last-position and all-position projections.
    fn run_decoder_stack(
        &self,
        input_embeds: &Tensor,
        cache: &mut dyn KvCache,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
    ) -> Result<Tensor> {
        let mut h = input_embeds.clone();
        // The hidden state and the RoPE tables follow each layer onto its device; an explicit additive
        // mask (batched decode) is carried across too. All `to_device`s are no-op clones for a model
        // whose layers share one device, so the common path pays nothing.
        let mut cur = h.device().clone();
        let mut cos_d = cos.clone();
        let mut sin_d = sin.clone();
        let mut mask_d: Option<Tensor> = match mask {
            AttnMask::Additive(m) => Some(m.clone()),
            _ => None,
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let dev = &self.layer_devices[i];
            if !cur.same_device(dev) {
                h = h.to_device(dev)?;
                cos_d = cos.to_device(dev)?;
                sin_d = sin.to_device(dev)?;
                if let Some(m) = &mask_d {
                    mask_d = Some(m.to_device(dev)?);
                }
                cur = dev.clone();
            }
            let layer_mask = match mask {
                AttnMask::Causal => AttnMask::Causal,
                AttnMask::None => AttnMask::None,
                AttnMask::Additive(_) => AttnMask::Additive(mask_d.as_ref().unwrap()),
            };
            h = layer.forward(&h, &cos_d, &sin_d, layer_mask, cache, i)?;
        }
        Ok(h)
    }

    /// Final RMSNorm + `lm_head` (+ Gemma-2 logit soft-cap) over hidden states `[batch, n, hidden]`,
    /// giving logits `[batch, n, vocab]`. `n` is `1` for the last-position forward, `seq` for the
    /// all-positions one.
    fn project_logits(&self, h: &Tensor) -> Result<Tensor> {
        let normed = rms_norm(h, &self.norm, self.cfg.rms_norm_eps as f64)?;
        // When sharded, the final norm sits on the last shard but the LM head may live elsewhere
        // (tied embeddings stay on the first shard); move the small hidden state to the head's device.
        let normed = normed.to_device(self.lm_head.weight().device())?;
        let logits = self.lm_head.forward(&normed)?;
        // Gemma-2 soft-caps the final logits.
        match self.final_softcap {
            Some(c) => Ok(soft_cap(&logits, c)?),
            None => Ok(logits),
        }
    }
}

/// Which shard a given layer index belongs to when `num_layers` are split into `num_shards`
/// **contiguous** blocks: layer `i` → `min(i·num_shards/num_layers, num_shards-1)`. Pure integer math
/// (no devices) so the split is unit-testable; [`shard_plan`] uses it to place layer weights.
fn shard_for_layer(layer: usize, num_shards: usize, num_layers: usize) -> usize {
    (layer * num_shards / num_layers.max(1)).min(num_shards.saturating_sub(1))
}

/// A key→device placement that splits a plain decoder's `num_layers` transformer blocks contiguously
/// across `devices` (see `shard_for_layer`), with the token embeddings (and any other non-layer
/// weight) on `devices[0]` and the final norm + LM head on the last device. This is the layout
/// [`CausalLm::from_dir_sharded`] feeds to [`Weights::from_dir_sharded`]. `devices` must be non-empty.
pub fn shard_plan(devices: &[Device], num_layers: usize) -> impl Fn(&str) -> Device + '_ {
    let last = devices.len() - 1;
    move |key: &str| -> Device {
        if let Some(rest) = key.strip_prefix("model.layers.") {
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                return devices[shard_for_layer(idx, devices.len(), num_layers)].clone();
            }
        }
        if key.starts_with("model.norm") || key.starts_with("lm_head") {
            return devices[last].clone();
        }
        // Token embeddings and anything else start on the home device.
        devices[0].clone()
    }
}

impl crate::decode::Decode for CausalLm {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn step(&self, input_ids: &Tensor, cache: &mut dyn KvCache, offset: i32) -> Result<Tensor> {
        self.decode_logits(input_ids, cache, offset)
    }
}

impl crate::models::VlmDecode for CausalLm {
    fn embed_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        CausalLm::embed_input_ids(self, input_ids)
    }

    fn splice_vision_features(
        &self,
        embeds: &Tensor,
        input_ids: &[i32],
        vision_features: &Tensor,
        placeholder_tokens: &[i32],
    ) -> Result<Tensor> {
        CausalLm::splice_vision_features(self, embeds, input_ids, vision_features, placeholder_tokens)
    }

    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        CausalLm::mrope_positions_mm(
            self,
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    fn prefill_with_deepstack(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Tensor],
    ) -> Result<Tensor> {
        // The generic decoder's cache is already the trait-object form — no downcast needed.
        self.decode_logits_from_embeds_mrope_deepstack(
            embeds,
            positions,
            cache,
            visual_pos_mask,
            deepstack,
        )
    }
}

/// The per-step plan for the continuous `Throughput` decode's **batched** paged attention (stories
/// 7453 + 7467): the single gather index over every active sequence's pooled token slots, the matching
/// fused-write slot index, and the cumulative key offsets the varlen kernel (and the eager fallback's
/// per-sequence split) read. Built once per step — block tables do not change across layers — and
/// reused by every layer's `forward_per_seq`.
struct ThroughputGather {
    /// Concatenated per-sequence token slots `[Σ lₖ]` (u32) — the one `index_select` index over the
    /// shared pool's per-layer KV tensor.
    index: Tensor,
    /// The fused-write target-slot index `[Σ s, n_kv_heads, head_dim]` (story 7467): each sequence's
    /// newly-reserved slots (in batch order) broadcast across the head/dim columns, so one
    /// [`BlockPool::scatter_write`](crate::primitives::BlockPool::scatter_write) lands this step's whole
    /// batch of new K/V in place — the write-side analogue of `index`.
    write_index: Tensor,
    /// Cumulative key offsets `[b + 1]`: sequence `i` owns gathered rows `cu_k[i] .. cu_k[i+1]`.
    cu_k: Vec<u32>,
    /// Longest cached key run (the varlen kernel's `max_k`). Only the `flash-attn` varlen path reads
    /// it; the eager fallback splits by `cu_k` and never needs it.
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    max_k: usize,
}

impl ThroughputGather {
    /// Concatenate every sequence's pooled token slots (in batch order) into the one gather index and
    /// the matching fused-write index, with the cumulative key offsets. `n_kv_heads`/`head_dim` size the
    /// write index's broadcast columns (the GQA values, uniform across the Throughput-eligible layers).
    /// The caches must already have reserved this step's positions.
    fn build(
        caches: &[&mut PagedKvCache],
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let cols = n_kv_heads * head_dim;
        let mut index: Vec<u32> = Vec::new();
        // The write index repeats each new-token slot across the `cols` head/dim columns the scatter
        // preserves (`scatter_set` needs the index shaped exactly like its source).
        let mut wslots: Vec<u32> = Vec::new();
        let mut cu_k = Vec::with_capacity(caches.len() + 1);
        cu_k.push(0u32);
        let mut acc = 0u32;
        let mut max_k = 0usize;
        for c in caches.iter() {
            let ts = c.token_slots();
            index.extend_from_slice(ts);
            acc += ts.len() as u32;
            cu_k.push(acc);
            max_k = max_k.max(ts.len());
            for &slot in c.new_token_slots() {
                for _ in 0..cols {
                    wslots.push(slot);
                }
            }
        }
        let total_new = wslots.len() / cols;
        let index = Tensor::from_vec(index, (acc as usize,), device)?;
        let write_index = Tensor::from_vec(wslots, (total_new, n_kv_heads, head_dim), device)?;
        Ok(Self {
            index,
            write_index,
            cu_k,
            max_k,
        })
    }
}

/// One transformer block. Pre-norm by default (Llama / Qwen / Phi); Gemma-2 adds the post-attention
/// and post-feedforward norms ([`LlamaLayer::pre_ff_ln`] / [`LlamaLayer::post_ff_ln`] are `Some`) for
/// its 4-norm "sandwich" residual.
struct LlamaLayer {
    /// Pre-attention norm.
    input_ln: Tensor,
    /// Llama: the MLP pre-norm. Gemma-2: the post-attention norm.
    post_ln: Tensor,
    /// Gemma-2 only: the MLP pre-norm (the post-attention residual is normed by `input`/`post_ln`).
    pre_ff_ln: Option<Tensor>,
    /// Gemma-2 only: the post-feedforward norm applied to the MLP output before the residual add.
    post_ff_ln: Option<Tensor>,
    attn: Attention,
    ffn: Ffn,
    eps: f64,
}

impl LlamaLayer {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let attn = self.attn.forward(
            &rms_norm(x, &self.input_ln, self.eps)?,
            cos,
            sin,
            mask,
            cache,
            layer_idx,
        )?;
        self.combine_ffn(x, &attn)
    }

    /// Per-sequence attention variant of [`LlamaLayer::forward`] (stories 7347 + 7453): the norms and
    /// MLP are batched as usual, but attention runs over each sequence's own paged cache — gathered for
    /// the whole batch in one `index_select` per the `gather` plan.
    fn forward_per_seq(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        caches: &[&mut PagedKvCache],
        gather: &ThroughputGather,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let attn = self.attn.forward_per_seq(
            &rms_norm(x, &self.input_ln, self.eps)?,
            cos,
            sin,
            caches,
            gather,
            layer_idx,
        )?;
        self.combine_ffn(x, &attn)
    }

    /// The residual + MLP half shared by both forwards: the Llama pre-norm, or the Gemma-2 4-norm
    /// sandwich when `pre_ff_ln`/`post_ff_ln` are set. `x` is the block input, `attn` the attention out.
    fn combine_ffn(&self, x: &Tensor, attn: &Tensor) -> Result<Tensor> {
        match (&self.pre_ff_ln, &self.post_ff_ln) {
            // Gemma-2 sandwich: post-norm the attention output and the MLP output before each add.
            (Some(pre_ff), Some(post_ff)) => {
                let attn = rms_norm(attn, &self.post_ln, self.eps)?;
                let h = x.broadcast_add(&attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, pre_ff, self.eps)?)?;
                let ffn = rms_norm(&ffn, post_ff, self.eps)?;
                Ok(h.broadcast_add(&ffn)?)
            }
            // Llama pre-norm: `post_ln` is the MLP pre-norm.
            _ => {
                let h = x.broadcast_add(attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
                Ok(h.broadcast_add(&ffn)?)
            }
        }
    }
}

/// A layer's feed-forward network: a dense SwiGLU MLP, or a sparse Mixture-of-Experts bank.
enum Ffn {
    Dense(LlamaMlp),
    Moe(MoeMlp),
}

impl Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

/// A layer's self-attention: grouped-query attention (Llama family) or Multi-head Latent Attention
/// (DeepSeek-V2). Both consume the same RoPE tables and additive mask and write into the same
/// [`KvCache`] seam, so the surrounding block is identical.
enum Attention {
    Gqa(LlamaAttention),
    Mla(MlaAttention),
}

impl Attention {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        match self {
            Attention::Gqa(a) => a.forward(x, cos, sin, mask, cache, layer_idx),
            Attention::Mla(a) => a.forward(x, cos, sin, mask, cache, layer_idx),
        }
    }

    /// Per-sequence (paged) attention for the continuous-batching Throughput path (story 7347).
    /// Implemented for grouped-query attention; MLA (DeepSeek-V2) has no per-sequence variant yet, so
    /// the Throughput mode is unavailable for it — the bit-exact `Exact` mode covers MLA (it runs each
    /// sequence through the ordinary `forward`).
    fn forward_per_seq(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        caches: &[&mut PagedKvCache],
        gather: &ThroughputGather,
        layer_idx: usize,
    ) -> Result<Tensor> {
        match self {
            Attention::Gqa(a) => a.forward_per_seq(x, cos, sin, caches, gather, layer_idx),
            Attention::Mla(_) => Err(crate::error::Error::Msg(
                "continuous-batching Throughput mode is not supported for MLA (DeepSeek-V2); use the \
                 Exact mode"
                    .into(),
            )),
        }
    }
}

/// Grouped-query attention with RoPE and optional per-head q/k RMSNorm (Qwen3).
struct LlamaAttention {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    groups: usize,
    eps: f64,
    /// Gemma-2 attention-score soft-cap; `None` ⇒ no cap.
    softcap: Option<f32>,
    /// Whether RoPE uses the interleaved (GPT-J) pairing (GLM-4).
    rope_interleaved: bool,
}

impl LlamaAttention {
    /// Project `x` `[b, s, hidden]` into attention-layout `(q, k, v)` — `q` `[b, heads, s, head_dim]`,
    /// `k`/`v` `[b, kv_heads, s, head_dim]` — applying the qkv projections, optional Qwen3 per-head
    /// q/k RMSNorm, RoPE, and the transpose into head-major layout. The shared front half of both the
    /// masked (batched) and per-sequence attention paths; nothing here touches the cache or attends.
    fn project(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        // Project, then split into heads in [b, s, heads, head_dim] layout.
        let mut q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let mut k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;

        // Qwen3 per-head q/k RMSNorm over the head_dim axis, before RoPE.
        if let Some(qn) = &self.q_norm {
            q = rms_norm(&q, qn, self.eps)?;
        }
        if let Some(kn) = &self.k_norm {
            k = rms_norm(&k, kn, self.eps)?;
        }

        // RoPE on q,k (cos/sin broadcast over the head axis), then -> [b, heads, s, head_dim].
        let q = apply_rope(&q, cos, sin, self.rope_interleaved)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_rope(&k, cos, sin, self.rope_interleaved)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        Ok((q, k, v))
    }

    /// Project the attended output `[b, heads, s, head_dim]` back to `[b, s, hidden]` through `o`. The
    /// shared back half of both attention paths.
    fn output(&self, attn: &Tensor) -> Result<Tensor> {
        let (b, _nh, s, _hd) = attn.dims4()?;
        let out =
            attn.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.num_heads * self.head_dim))?;
        self.o.forward(&out)
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project(x, cos, sin)?;
        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        let k_all = repeat_kv(&k_all, self.groups)?;
        let v_all = repeat_kv(&v_all, self.groups)?;
        let out = sdpa(&q, &k_all, &v_all, self.scale, self.softcap, mask)?; // [b, heads, s, head_dim]
        self.output(&out)
    }

    /// Per-sequence attention (stories 7347 + 7351 + 7453): the projection is **batched** over all
    /// rows, then each row attends only its own sequence's real KV in `caches[i]` — no padding mask, no
    /// cross-row mixing. `caches.len()` must equal the batch. The projections / MLP / lm_head around
    /// this stay batched, which is where the throughput comes from (and why the surrounding logits only
    /// *track* batch-1 to sub-ULP — the batched matmul isn't M-invariant).
    ///
    /// Stories 7453 + 7467 make both the write and the gather batched: this step's new keys/values are
    /// scattered into the shared pool with **one** in-place `scatter_set` per side (the `gather` plan's
    /// `write_index`, no per-sequence loop), then **every** active sequence's KV is gathered in **one**
    /// `index_select` over the pool's per-layer tensor (the `gather` plan's index), already in the varlen
    /// kernel's `[Σ lₖ, kvh, hd]` layout — no per-sequence `squeeze`/`transpose`/`cat`. With
    /// `--features flash-attn` that ragged KV feeds **one**
    /// [`try_flash_attn_varlen`](crate::primitives::attention::try_flash_attn_varlen) call (one kernel
    /// launch instead of N). The eager per-sequence SDPA loop is the fallback for everything varlen
    /// cannot serve (CPU/f32, Gemma-2 soft-cap, or the feature off), splitting the same ragged gather
    /// per sequence — byte-identical to each sequence's batch-1 attention.
    fn forward_per_seq(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        caches: &[&mut PagedKvCache],
        gather: &ThroughputGather,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project(x, cos, sin)?; // [b,H,s,hd], [b,kvh,s,hd], [b,kvh,s,hd]
        let (b, _h, s, _hd) = q.dims4()?;
        let (kvh, hd) = (self.num_kv_heads, self.head_dim);

        // Story 7467: reshape this step's new K/V to token-major `[Σ s, kvh, hd]` (batch order) and
        // scatter every sequence's tokens into their pooled slots with **one** in-place `scatter_set`
        // per side — replacing the `O(N)` per-sequence `slice_set` loop (the residual launch-latency
        // cost sc-7453 left). The `write_index` (built once per step) maps each token-major row to its
        // physical pool slot.
        let total_new = b * s;
        let k_tm = k
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_new, kvh, hd))?;
        let v_tm = v
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_new, kvh, hd))?;
        caches[0]
            .pool()
            .borrow()
            .scatter_write(layer_idx, &gather.write_index, &k_tm, &v_tm)?;

        // One gather over every sequence's pooled token slots → ragged [Σ lₖ, kvh, hd] for K and V.
        let (k_rag, v_rag) = caches[0].pool().borrow().gather(layer_idx, &gather.index)?;

        // Batched attention over all sequences in one ragged varlen kernel when eligible; otherwise
        // the eager per-sequence SDPA loop (CPU/f32, soft-cap, or no `flash-attn` feature).
        #[cfg(feature = "flash-attn")]
        if let Some(out) = crate::primitives::attention::try_flash_attn_varlen(
            &q,
            &k_rag,
            &v_rag,
            &gather.cu_k,
            gather.max_k,
            self.scale,
            self.softcap,
        )? {
            return self.output(&out);
        }

        // Eager fallback: slice each sequence's rows out of the ragged gather, rebuild head-major
        // [1, kvh, lₖ, hd], GQA-expand, and run the stock causal SDPA — identical to its batch-1 path.
        let mut outs = Vec::with_capacity(b);
        for i in 0..b {
            let start = gather.cu_k[i] as usize;
            let li = gather.cu_k[i + 1] as usize - start;
            let to_hm = |t: &Tensor| -> Result<Tensor> {
                Ok(t.narrow(0, start, li)?
                    .reshape((1, li, kvh, hd))?
                    .transpose(1, 2)?
                    .contiguous()?)
            };
            let k_all = repeat_kv(&to_hm(&k_rag)?, self.groups)?;
            let v_all = repeat_kv(&to_hm(&v_rag)?, self.groups)?;
            let qi = q.narrow(0, i, 1)?; // [1, H, s, hd]
            outs.push(sdpa(
                &qi,
                &k_all,
                &v_all,
                self.scale,
                self.softcap,
                AttnMask::Causal,
            )?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let out = Tensor::cat(&refs, 0)?; // [b, heads, s, head_dim]
        self.output(&out)
    }
}

/// Multi-head Latent Attention (DeepSeek-V2).
///
/// Instead of projecting full per-head keys/values, MLA down-projects the input to a small shared
/// latent (`kv_a_proj_with_mqa` → `kv_a_layernorm`, width `kv_lora_rank`) plus a single shared rotary
/// key sub-vector (`k_pe`, MQA-style). The latent is up-projected (`kv_b_proj`) to per-head content
/// keys (`k_nope`) and values. Queries split the same way: a content part (`q_nope`) and a rotary
/// part (`q_pe`) — from a full `q_proj`, or a low-rank `q_a_proj` → norm → `q_b_proj` when the model
/// has a query LoRA. RoPE rotates only the `qk_rope_head_dim` sub-vectors; the per-head key is
/// `[k_nope ‖ k_pe]` and the query `[q_nope ‖ q_pe]`, attended at `q_head_dim = qk_nope + qk_rope`.
///
/// This is the **correctness-first** materialized form: it reconstructs full per-head K (`q_head_dim`)
/// and V (`v_head_dim`) and caches them like ordinary attention, so the existing [`KvCache`] and
/// [`sdpa`] are reused unchanged (the latent-caching "absorbed" optimization is a later throughput
/// concern). Heads are full MHA here (no GQA expansion).
struct MlaAttention {
    /// Full query projection (when there is no query LoRA — DeepSeek-V2-Lite).
    q_proj: Option<Projection>,
    /// Query LoRA down-projection (`q_a_proj`), present iff `q_lora_rank` is set.
    q_a_proj: Option<Projection>,
    /// RMSNorm over the query latent (`q_a_layernorm`).
    q_a_layernorm: Option<Tensor>,
    /// Query LoRA up-projection (`q_b_proj`).
    q_b_proj: Option<Projection>,
    /// Shared KV down-projection with the MQA rotary key (`kv_a_proj_with_mqa`) →
    /// `[kv_lora_rank ‖ qk_rope_head_dim]`.
    kv_a_proj: Projection,
    /// RMSNorm over the KV latent (`kv_a_layernorm`).
    kv_a_layernorm: Tensor,
    /// KV up-projection (`kv_b_proj`) → per-head `[qk_nope_head_dim ‖ v_head_dim]`.
    kv_b_proj: Projection,
    /// Output projection over the concatenated per-head values.
    o_proj: Projection,
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    scale: f32,
    eps: f64,
}

impl MlaAttention {
    fn load(
        w: &Weights,
        lp: impl Fn(&str) -> String,
        cfg: &ModelConfig,
        dtype: DType,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        let mla = cfg
            .mla
            .expect("MLA config present for a DeepSeek-V2 decoder");
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        let proj = |key: String| -> Result<Projection> { Projection::load(req(key)?, quant) };

        // Query: a low-rank `q_a → norm → q_b` when the model has a query LoRA, else a full `q_proj`.
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) =
            if w.contains(&lp("self_attn.q_a_proj.weight")) {
                (
                    None,
                    Some(proj(lp("self_attn.q_a_proj.weight"))?),
                    Some(req(lp("self_attn.q_a_layernorm.weight"))?),
                    Some(proj(lp("self_attn.q_b_proj.weight"))?),
                )
            } else {
                (Some(proj(lp("self_attn.q_proj.weight"))?), None, None, None)
            };

        Ok(Self {
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj: proj(lp("self_attn.kv_a_proj_with_mqa.weight"))?,
            kv_a_layernorm: req(lp("self_attn.kv_a_layernorm.weight"))?,
            kv_b_proj: proj(lp("self_attn.kv_b_proj.weight"))?,
            o_proj: proj(lp("self_attn.o_proj.weight"))?,
            num_heads: cfg.num_heads as usize,
            qk_nope_head_dim: mla.qk_nope_head_dim as usize,
            qk_rope_head_dim: mla.qk_rope_head_dim as usize,
            v_head_dim: mla.v_head_dim as usize,
            kv_lora_rank: mla.kv_lora_rank as usize,
            scale: cfg.attn_scale(),
            eps: cfg.rms_norm_eps as f64,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let nh = self.num_heads;
        let (nope, rope, vhd) = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        );
        let qhd = nope + rope; // per-head q/k dim attended over

        // Query → [b, s, nh, qhd], split into content (nope) and rotary (rope) parts.
        let q = match (&self.q_proj, &self.q_a_proj) {
            (Some(qp), _) => qp.forward(x)?,
            (None, Some(qa)) => {
                let c = qa.forward(x)?;
                let c = rms_norm(&c, self.q_a_layernorm.as_ref().unwrap(), self.eps)?;
                self.q_b_proj.as_ref().unwrap().forward(&c)?
            }
            _ => unreachable!("MLA query has either q_proj or q_a/q_b"),
        };
        let q = q.reshape((b, s, nh, qhd))?;
        let q_nope = q.narrow(3, 0, nope)?.contiguous()?;
        let q_pe = q.narrow(3, nope, rope)?.contiguous()?;

        // Shared KV latent + the single MQA rotary key.
        let kv = self.kv_a_proj.forward(x)?; // [b, s, kv_lora_rank + rope]
        let compressed = kv.narrow(2, 0, self.kv_lora_rank)?.contiguous()?;
        let k_pe = kv
            .narrow(2, self.kv_lora_rank, rope)?
            .reshape((b, s, 1, rope))?
            .contiguous()?; // shared across heads
        let compressed = rms_norm(&compressed, &self.kv_a_layernorm, self.eps)?;
        // Up-project to per-head content keys and values: [b, s, nh, nope + vhd].
        let kv = self
            .kv_b_proj
            .forward(&compressed)?
            .reshape((b, s, nh, nope + vhd))?;
        let k_nope = kv.narrow(3, 0, nope)?.contiguous()?;
        let value = kv.narrow(3, nope, vhd)?.contiguous()?;

        // RoPE the rotary sub-vectors (interleaved); broadcast the shared key over heads.
        let q_pe = apply_rope(&q_pe, cos, sin, true)?;
        let k_pe = apply_rope(&k_pe, cos, sin, true)?;
        let k_pe = k_pe.broadcast_as((b, s, nh, rope))?.contiguous()?;

        // Assemble full per-head q/k, then [b, nh, s, *] for attention.
        let q = Tensor::cat(&[&q_nope, &q_pe], 3)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = Tensor::cat(&[&k_nope, &k_pe], 3)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = value.transpose(1, 2)?.contiguous()?;

        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        let out = sdpa(&q, &k_all, &v_all, self.scale, None, mask)?; // [b, nh, s, v_head_dim]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, nh * vhd))?;
        self.o_proj.forward(&out)
    }
}

/// A gated MLP: SwiGLU (`silu`) by default, or GeGLU (`gelu`, the Gemma activation) when `gelu`.
struct LlamaMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
    gelu: bool,
}

impl LlamaMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let g = if self.gelu { gelu(&g)? } else { silu(&g)? };
        let up = self.up.forward(x)?;
        self.down.forward(&(g * up)?)
    }
}

/// A sparse Mixture-of-Experts feed-forward (Qwen2-MoE, DeepSeek-V2): a softmax router over `experts`
/// (top-k per token) plus an always-on `shared` expert. Correctness-first — each expert runs **only
/// on its routed tokens** (gathered, then scatter-added back), so the active compute scales with
/// `experts_per_tok`, not the full bank. Top-k selection is done on the host (Candle has no fused
/// top-k); `n_group`/`topk_group` group-limited routing (DeepSeek-V2-236B / V3) is not modelled —
/// the verification model (V2-Lite) uses plain greedy top-k.
struct MoeMlp {
    /// Router weight `[num_experts, hidden]`.
    router: Tensor,
    experts: Vec<LlamaMlp>,
    shared: LlamaMlp,
    /// Shared-expert sigmoid gate `[1, hidden]` (Qwen2-MoE); `None` ⇒ the shared expert is added
    /// ungated (DeepSeek-V2).
    shared_gate: Option<Tensor>,
    experts_per_tok: usize,
    norm_topk_prob: bool,
    /// Multiplier on the (un-normalized) routed weights — DeepSeek's `routed_scaling_factor`; `1.0`
    /// for Qwen2-MoE. Ignored when `norm_topk_prob` (the weights are renormalized instead).
    routed_scaling_factor: f32,
}

impl MoeMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let t = b * s;
        let dtype = x.dtype();
        let device = x.device();
        let xf = x.reshape((t, h))?;

        // Router probabilities (computed in f32 for a stable top-k), pulled to host.
        let logits = xf.matmul(&self.router.t()?)?; // [t, E]
        let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let probs = probs.to_vec2::<f32>()?; // [t][E]
        let num_experts = self.experts.len();
        let k = self.experts_per_tok.min(num_experts).max(1);

        // Invert the per-token top-k into per-expert (token, weight) lists.
        let mut routed: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_experts];
        for (ti, row) in probs.iter().enumerate() {
            let mut idx: Vec<usize> = (0..num_experts).collect();
            idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
            let top = &idx[..k];
            // Renormalize the top-k weights to sum to 1, or (when not normalizing) apply the routed
            // scaling factor — matching the reference gate's two branches.
            let (denom, post_scale) = if self.norm_topk_prob {
                let sum = top
                    .iter()
                    .map(|&e| row[e])
                    .sum::<f32>()
                    .max(f32::MIN_POSITIVE);
                (sum, 1.0)
            } else {
                (1.0, self.routed_scaling_factor)
            };
            for &e in top {
                routed[e].push((ti as u32, row[e] / denom * post_scale));
            }
        }

        // Each expert runs on just its tokens; scatter the weighted outputs back.
        let mut out = Tensor::zeros((t, h), dtype, device)?;
        for (e, toks) in routed.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let idx = Tensor::from_vec(
                toks.iter().map(|&(ti, _)| ti).collect::<Vec<u32>>(),
                (n,),
                device,
            )?;
            let wts = Tensor::from_vec(
                toks.iter().map(|&(_, w)| w).collect::<Vec<f32>>(),
                (n, 1),
                device,
            )?
            .to_dtype(dtype)?;
            let xe = xf.index_select(&idx, 0)?; // [n, h]
            let ye = self.experts[e].forward(&xe)?.broadcast_mul(&wts)?; // [n, h]
            out = out.index_add(&idx, &ye, 0)?;
        }

        // Always-on shared expert: Qwen2 gates it by sigmoid(x · shared_gateᵀ); DeepSeek packs several
        // shared experts into one MLP and adds them ungated.
        let shared = self.shared.forward(&xf)?;
        let shared = match &self.shared_gate {
            Some(g) => {
                let sg = candle_nn::ops::sigmoid(&xf.matmul(&g.t()?)?)?; // [t, 1]
                shared.broadcast_mul(&sg)?
            }
            None => shared,
        };
        Ok((out + shared)?.reshape((b, s, h))?)
    }
}

/// Join a key prefix and suffix (`""` prefix ⇒ the suffix verbatim).
fn join(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_handles_empty_prefix() {
        assert_eq!(join("", "model.norm.weight"), "model.norm.weight");
        assert_eq!(
            join("language_model", "model.norm.weight"),
            "language_model.model.norm.weight"
        );
    }

    #[test]
    fn shard_split_is_contiguous_and_balanced() {
        // 28 layers across 2 shards: a clean 14 / 14 split, monotonic, last layer on the last shard.
        let s = |i| shard_for_layer(i, 2, 28);
        assert_eq!(
            (0..28)
                .map(s)
                .collect::<Vec<_>>()
                .iter()
                .filter(|&&x| x == 0)
                .count(),
            14
        );
        assert_eq!(s(0), 0);
        assert_eq!(s(13), 0);
        assert_eq!(s(14), 1);
        assert_eq!(s(27), 1);
        // Monotonic non-decreasing (contiguous blocks, never interleaved).
        assert!((1..28).all(|i| s(i) >= s(i - 1)));
        // Uneven split (3 shards, 10 layers) still lands every layer in range and the last on shard 2.
        assert!((0..10).all(|i| shard_for_layer(i, 3, 10) < 3));
        assert_eq!(shard_for_layer(9, 3, 10), 2);
    }

    #[test]
    fn shard_plan_places_embeddings_layers_and_head() {
        let devs = [Device::Cpu, Device::Cpu];
        let plan = shard_plan(&devs, 28);
        // Pure routing check: every key resolves without panicking, layer keys parse the index.
        assert!(plan("model.embed_tokens.weight").is_cpu());
        assert!(plan("model.layers.0.self_attn.q_proj.weight").is_cpu());
        assert!(plan("model.layers.27.mlp.gate_proj.weight").is_cpu());
        assert!(plan("model.norm.weight").is_cpu());
        assert!(plan("lm_head.weight").is_cpu());
    }

    /// **sc-7477 per-step decode phase decomposition.** Reconstructs **one** decode step for both the
    /// continuous `Throughput` path ([`CausalLm::decode_logits_per_seq`]) and the `generate_batch`
    /// decode path ([`CausalLm::decode_logits_masked`]) at batch `N`, over a uniform cached context of
    /// length `L`, and sync-brackets every phase — embed, RoPE-table build, the per-step index/mask
    /// **build**, and per layer: input-norm, q/k/v projection, the attention core (scatter-write +
    /// gather + varlen kernel for continuous; cache-cat + masked SDPA for the batch path), o-projection,
    /// MLP — plus the final norm + lm_head. The non-attention phases are the **same batched ops on the
    /// same shapes** in both paths, so the per-step gap must live in the attention core + the index/mask
    /// build; this attributes it, and the `attention / total` ratio per model explains the
    /// SmolLM2-vs-Qwen3 asymmetry.
    ///
    /// Calls the **real** layer methods (`project`/`output`/`combine_ffn`/`scatter_write`/`gather`/the
    /// varlen kernel / `sdpa`) in the real forwards' order, so the breakdown does not drift from the
    /// production path. Each phase is `synchronize()`-bracketed, so the columns are
    /// launch-latency-attributed and the *sum* over-counts the real pipelined step (the realized
    /// end-to-end throughput is `attention_bottleneck_bound`); the cross-path *comparison* is the signal.
    /// Needs `--features flash-attn` + CUDA + `CANDLE_LLM_TEST_MODEL` / `CANDLE_LLM_QWEN3_MODEL`.
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "sc-7477 phase bench; needs CUDA + CANDLE_LLM_TEST_MODEL / CANDLE_LLM_QWEN3_MODEL"]
    fn decode_step_phase_breakdown_on_cuda() {
        use crate::config::ModelConfig;
        use crate::primitives::attention::{repeat_kv, sdpa, try_flash_attn_varlen};
        use crate::primitives::kv_cache::KvCache;
        use crate::primitives::{BlockPool, PagedKvCache, Weights};
        use std::time::Instant;

        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: no CUDA device");
                return;
            }
        };
        const N: usize = 16; // occupancy where the gap is largest
        const L: usize = 60; // prior cached length (uniform); the step attends L+1
        let iters = 30usize;
        let warmup = 8usize;

        for env in ["CANDLE_LLM_TEST_MODEL", "CANDLE_LLM_QWEN3_MODEL"] {
            let Ok(dir) = std::env::var(env) else {
                eprintln!("skip {env}: unset");
                continue;
            };
            if dir.is_empty() {
                eprintln!("skip {env}: empty");
                continue;
            }
            let cfg = ModelConfig::from_dir(&dir).unwrap();
            let model = CausalLm::from_weights(&Weights::from_dir(&dir, &device).unwrap(), "", cfg)
                .unwrap();
            let dt = model.dtype;
            let nkv = model.cfg.num_kv_heads as usize;
            let hd = model.cfg.head_dim as usize;
            let nl = model.cfg.num_layers;
            println!(
                "\n[{env}] N={N} L={L} (attends {}), layers={nl}, H={} KVH={nkv} D={hd}, dtype={dt:?}",
                L + 1,
                model.cfg.num_heads
            );

            // Bounded bf16/f16 dummy of arbitrary shape (values irrelevant to kernel timing).
            let mk = |dims: &[usize], phase: f64| -> Tensor {
                let n: usize = dims.iter().product();
                Tensor::arange(0f32, n as f32, &device)
                    .unwrap()
                    .reshape(dims)
                    .unwrap()
                    .affine(0.013, phase)
                    .unwrap()
                    .cos()
                    .unwrap()
                    .to_dtype(dt)
                    .unwrap()
            };

            let ids = Tensor::from_vec(vec![5u32; N], (N, 1), &device).unwrap();
            let positions: Vec<i32> = (0..N).map(|_| L as i32).collect();
            let us = |s: f64| s / iters as f64 * 1e6;
            let sync = || device.synchronize().unwrap();

            // ===================== Continuous Throughput path =====================
            // N paged caches over one pool, prefilled to L, with this step's +1 token reserved so the
            // gather index spans L+1 per sequence. State is held fixed across iters (re-scatter to the
            // same slots, re-gather the same index) so every iter measures the step at the same L.
            let pool = BlockPool::new(16);
            let mut pcaches: Vec<PagedKvCache> = (0..N)
                .map(|_| PagedKvCache::with_pool(pool.clone(), nl))
                .collect();
            for c in pcaches.iter_mut() {
                for layer in 0..nl {
                    c.update(
                        layer,
                        &mk(&[1, nkv, L, hd], 0.1),
                        &mk(&[1, nkv, L, hd], 0.2),
                    )
                    .unwrap();
                }
            }
            for c in pcaches.iter_mut() {
                c.reserve_step(1).unwrap();
            }

            let (mut c_embed, mut c_rope, mut c_build, mut c_iln, mut c_proj) =
                (0f64, 0f64, 0f64, 0f64, 0f64);
            let (mut c_write, mut c_gather, mut c_kernel, mut c_oproj, mut c_mlp, mut c_head) =
                (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
            // Sub-task 4: split build into host-loop (Vec construction) vs from_vec (H2D copy).
            let (mut c_build_host, mut c_build_copy) = (0f64, 0f64);

            {
                let crefs: Vec<&mut PagedKvCache> = pcaches.iter_mut().collect();
                for it in 0..(warmup + iters) {
                    let rec = it >= warmup;

                    sync();
                    let t = Instant::now();
                    let mut h = model.embed(&ids).unwrap();
                    sync();
                    let t_e = Instant::now();
                    let (cos, sin) = model.rope_tables(&positions, N as i32, 1).unwrap();
                    sync();
                    let t_r = Instant::now();

                    // build: the per-step gather + fused-write index (host loops + H2D copies).
                    let cols = nkv * hd;
                    let mut index: Vec<u32> = Vec::new();
                    let mut wslots: Vec<u32> = Vec::new();
                    let mut cu_k = vec![0u32];
                    let mut acc = 0u32;
                    let mut max_k = 0usize;
                    for c in crefs.iter() {
                        let ts = c.token_slots();
                        index.extend_from_slice(ts);
                        acc += ts.len() as u32;
                        cu_k.push(acc);
                        max_k = max_k.max(ts.len());
                        for &slot in c.new_token_slots() {
                            for _ in 0..cols {
                                wslots.push(slot);
                            }
                        }
                    }
                    let total_new = wslots.len() / cols;
                    let t_bh = Instant::now(); // host loops done; below is the H2D copy
                    let index = Tensor::from_vec(index, (acc as usize,), &device).unwrap();
                    let write_index =
                        Tensor::from_vec(wslots, (total_new, nkv, hd), &device).unwrap();
                    sync();
                    let t_b = Instant::now();

                    for (li, layer) in model.layers.iter().enumerate() {
                        let a = match &layer.attn {
                            Attention::Gqa(a) => a,
                            Attention::Mla(_) => unreachable!("test models are GQA"),
                        };
                        sync();
                        let l0 = Instant::now();
                        let xn = rms_norm(&h, &layer.input_ln, layer.eps).unwrap();
                        sync();
                        let l1 = Instant::now();
                        let (q, k, v) = a.project(&xn, &cos, &sin).unwrap();
                        sync();
                        let l2 = Instant::now();
                        let k_tm = k
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((total_new, nkv, hd))
                            .unwrap();
                        let v_tm = v
                            .transpose(1, 2)
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((total_new, nkv, hd))
                            .unwrap();
                        crefs[0]
                            .pool()
                            .borrow()
                            .scatter_write(li, &write_index, &k_tm, &v_tm)
                            .unwrap();
                        sync();
                        let l3 = Instant::now();
                        let (k_rag, v_rag) = crefs[0].pool().borrow().gather(li, &index).unwrap();
                        sync();
                        let l4 = Instant::now();
                        let out = try_flash_attn_varlen(
                            &q, &k_rag, &v_rag, &cu_k, max_k, a.scale, a.softcap,
                        )
                        .unwrap()
                        .expect("varlen-eligible");
                        sync();
                        let l5 = Instant::now();
                        let attn = a.output(&out).unwrap();
                        sync();
                        let l6 = Instant::now();
                        h = layer.combine_ffn(&h, &attn).unwrap();
                        sync();
                        let l7 = Instant::now();
                        if rec {
                            c_iln += (l1 - l0).as_secs_f64();
                            c_proj += (l2 - l1).as_secs_f64();
                            c_write += (l3 - l2).as_secs_f64();
                            c_gather += (l4 - l3).as_secs_f64();
                            c_kernel += (l5 - l4).as_secs_f64();
                            c_oproj += (l6 - l5).as_secs_f64();
                            c_mlp += (l7 - l6).as_secs_f64();
                        }
                    }
                    // The norm + lm_head phase (head) is measured cleanly in the standalone block below.
                    let last = h.narrow(1, 0, 1).unwrap().contiguous().unwrap();
                    let _logits = model.project_logits(&last).unwrap();
                    sync();
                    if rec {
                        c_embed += (t_e - t).as_secs_f64();
                        c_rope += (t_r - t_e).as_secs_f64();
                        c_build += (t_b - t_r).as_secs_f64();
                        c_build_host += (t_bh - t_r).as_secs_f64();
                        c_build_copy += (t_b - t_bh).as_secs_f64();
                    }
                }
            }

            // Norm + lm_head on a [N,1,hidden] hidden state (identical in both paths).
            {
                let hh = mk(&[N, 1, model.cfg.hidden_size as usize], 0.3);
                for it in 0..(warmup + iters) {
                    sync();
                    let t = Instant::now();
                    let _ = model.project_logits(&hh).unwrap();
                    sync();
                    if it >= warmup {
                        c_head += t.elapsed().as_secs_f64();
                    }
                }
            }

            let c_attn = c_build + c_write + c_gather + c_kernel;
            let c_total = c_embed
                + c_rope
                + c_build
                + c_iln
                + c_proj
                + c_write
                + c_gather
                + c_kernel
                + c_oproj
                + c_mlp
                + c_head;
            println!("  -- continuous Throughput per-step (sync-bracketed, sum over {nl} layers):");
            println!(
                "     embed {:6.1} | rope {:6.1} | build {:6.1} (host {:5.1}+copy {:5.1}) | \
                 iln {:6.1} | qkv-proj {:7.1} | write {:6.1} | gather {:6.1} | kernel {:7.1} | \
                 o-proj {:7.1} | mlp {:8.1} | norm+head {:6.1} | TOTAL {:8.1} us",
                us(c_embed),
                us(c_rope),
                us(c_build),
                us(c_build_host),
                us(c_build_copy),
                us(c_iln),
                us(c_proj),
                us(c_write),
                us(c_gather),
                us(c_kernel),
                us(c_oproj),
                us(c_mlp),
                us(c_head),
                us(c_total),
            );
            println!(
                "     attention (build+write+gather+kernel) {:7.1} us = {:.0}% of step",
                us(c_attn),
                100.0 * c_attn / c_total
            );

            // ===================== generate_batch decode path =====================
            // One dense [N,nkv,L,hd] cached K/V per layer (reused), catted with the step's new token
            // and attended by one masked SDPA — exactly decode_logits_masked's per-step attention.
            let k_cached = mk(&[N, nkv, L, hd], 0.4);
            let v_cached = mk(&[N, nkv, L, hd], 0.5);
            let (mut b_embed, mut b_rope, mut b_build, mut b_iln, mut b_proj) =
                (0f64, 0f64, 0f64, 0f64, 0f64);
            let (mut b_cat, mut b_kernel, mut b_oproj, mut b_mlp) = (0f64, 0f64, 0f64, 0f64);
            for it in 0..(warmup + iters) {
                let rec = it >= warmup;
                sync();
                let t = Instant::now();
                let mut h = model.embed(&ids).unwrap();
                sync();
                let t_e = Instant::now();
                let (cos, sin) = model.rope_tables(&positions, N as i32, 1).unwrap();
                sync();
                let t_r = Instant::now();
                // build: the [N,1,1,L+1] additive decode mask (host loop + H2D + dtype cast),
                // mirroring batch::decode_mask. Uniform lengths -> every key attendable (no left-pad).
                let kt = L + 1;
                let pad_len = 0i32; // uniform lengths: no left-pad region to block
                let mut mh = Vec::with_capacity(N * kt);
                for _lane in 0..N {
                    for j in 0..kt {
                        mh.push(if (j as i32) >= pad_len {
                            0.0f32
                        } else {
                            -1e30f32
                        });
                    }
                }
                let mask = Tensor::from_vec(mh, (N, 1, 1, kt), &device)
                    .unwrap()
                    .to_dtype(dt)
                    .unwrap();
                sync();
                let t_b = Instant::now();
                for layer in model.layers.iter() {
                    let a = match &layer.attn {
                        Attention::Gqa(a) => a,
                        Attention::Mla(_) => unreachable!(),
                    };
                    sync();
                    let l0 = Instant::now();
                    let xn = rms_norm(&h, &layer.input_ln, layer.eps).unwrap();
                    sync();
                    let l1 = Instant::now();
                    let (q, k, v) = a.project(&xn, &cos, &sin).unwrap();
                    sync();
                    let l2 = Instant::now();
                    let k_all = Tensor::cat(&[&k_cached, &k], 2).unwrap();
                    let v_all = Tensor::cat(&[&v_cached, &v], 2).unwrap();
                    let k_all = repeat_kv(&k_all, a.groups).unwrap();
                    let v_all = repeat_kv(&v_all, a.groups).unwrap();
                    sync();
                    let l3 = Instant::now();
                    let out = sdpa(
                        &q,
                        &k_all,
                        &v_all,
                        a.scale,
                        a.softcap,
                        AttnMask::Additive(&mask),
                    )
                    .unwrap();
                    sync();
                    let l4 = Instant::now();
                    let attn = a.output(&out).unwrap();
                    sync();
                    let l5 = Instant::now();
                    h = layer.combine_ffn(&h, &attn).unwrap();
                    sync();
                    let l6 = Instant::now();
                    if rec {
                        b_iln += (l1 - l0).as_secs_f64();
                        b_proj += (l2 - l1).as_secs_f64();
                        b_cat += (l3 - l2).as_secs_f64();
                        b_kernel += (l4 - l3).as_secs_f64();
                        b_oproj += (l5 - l4).as_secs_f64();
                        b_mlp += (l6 - l5).as_secs_f64();
                    }
                }
                let last = h.narrow(1, 0, 1).unwrap().contiguous().unwrap();
                let _ = model.project_logits(&last).unwrap();
                sync();
                if rec {
                    b_embed += (t_e - t).as_secs_f64();
                    b_rope += (t_r - t_e).as_secs_f64();
                    b_build += (t_b - t_r).as_secs_f64();
                }
            }
            let b_attn = b_build + b_cat + b_kernel;
            let b_total = b_embed
                + b_rope
                + b_build
                + b_iln
                + b_proj
                + b_cat
                + b_kernel
                + b_oproj
                + b_mlp
                + c_head; // norm+head identical to continuous; reuse the clean measurement
            println!("  -- generate_batch decode per-step (sync-bracketed, sum over {nl} layers):");
            println!(
                "     embed {:6.1} | rope {:6.1} | build {:6.1} | iln {:6.1} | qkv-proj {:7.1} | \
                 cat {:6.1} | sdpa {:7.1} | o-proj {:7.1} | mlp {:8.1} | norm+head {:6.1} | \
                 TOTAL {:8.1} us",
                us(b_embed),
                us(b_rope),
                us(b_build),
                us(b_iln),
                us(b_proj),
                us(b_cat),
                us(b_kernel),
                us(b_oproj),
                us(b_mlp),
                us(c_head),
                us(b_total),
            );
            println!(
                "     attention (build+cat+sdpa) {:7.1} us = {:.0}% of step",
                us(b_attn),
                100.0 * b_attn / b_total
            );
            println!(
                "  -- GAP: cont {:.1} - batch {:.1} = {:.1} us/step ({:.0}% slower); \
                 attn-core delta {:.1} us (cont {:.1} - batch {:.1})",
                us(c_total),
                us(b_total),
                us(c_total - b_total),
                100.0 * (c_total / b_total - 1.0),
                us(c_attn - b_attn),
                us(c_attn),
                us(b_attn),
            );
        }
    }
}
