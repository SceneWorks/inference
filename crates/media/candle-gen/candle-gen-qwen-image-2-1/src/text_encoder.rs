//! Qwen3-VL text conditioning for Qwen-Image 2.1 — the port of
//! `QwenImage21Pipeline._get_qwen_prompt_embeds` for the text-only (T2I) path.
//!
//! The prompt is rendered into the **raw** T2I template (not `apply_chat_template`; upstream notes
//! the two tokenize differently and the checkpoint expects this one):
//!
//! ```text
//! <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! tokenized with the snapshot's own `processor/tokenizer.json`, run through the Qwen3-VL **language
//! tower**, and the hidden state of the **last decoder layer before the final RMSNorm** is taken
//! (upstream neutralises `norm` with a forward hook because transformers ≥ 5.0 ties
//! `hidden_states[-1]` to the normalised `last_hidden_state`). The leading system-role tokens are
//! then dropped: upstream derives that count by tokenizing the system message through the chat
//! template, and [`system_prompt_drop_count`] derives it the same way from the loaded tokenizer
//! instead of hard-coding it.
//!
//! For a text-only prompt the three mRoPE position streams (`mrope_section [24, 20, 20]`,
//! interleaved) are all the plain token index, so the rotary embedding is exactly 1-D RoPE at
//! `rope_theta = 5e6`. The MLX twin reuses `mlx_gen_z_image::text_encoder::EncoderLayer` for this;
//! candle's Z-Image Qwen3 decoder is a private vendored module (`candle_gen_z_image::packed_te`,
//! and its block is additionally pinned to Z-Image's layer[-2]/`model.` conventions), so the same
//! pre-norm decoder block is ported here rather than reached across a crate boundary — the forward
//! math is the stock Qwen3 one (per-head `q_norm`/`k_norm`, bias-free GQA, HF half-split RoPE,
//! SwiGLU). The DeepStack visual injection and the vision tower engage only with condition images
//! — see *The image-conditioned path* below.
//!
//! Activations run at the dtype the tower's `VarBuilder` was built at — the backend's
//! [`crate::loader::compute_dtype`]: the checkpoint's own bf16 on CUDA/Metal, exactly as upstream
//! runs the encoder, and f32 on the CPU parity lane (sc-24114). Every table this module builds
//! (rotary, causal mask, M-RoPE) is cast to that dtype; the DiT rounds the result to its own compute
//! dtype at `txt_in` either way.
//!
//! # The image-conditioned path (sc-24110)
//!
//! With condition images the template becomes [`prompt_template_ti2i`] — one
//! `<imageN><|vision_start|><|image_pad|><|vision_end|>` group per reference, in order, ahead of
//! the prompt — and the full Qwen3-VL multimodal path engages:
//!
//! * each `<|image_pad|>` placeholder is **expanded** to one token per merged 2x2 vision patch,
//!   exactly as `Qwen3VLProcessor` does when it sees `pixel_values`;
//! * the ViT tower ([`candle_llm::models::Qwen35VisionModel`]) encodes the references and its
//!   merged rows are **spliced** into the image-token positions;
//! * positions become **interleaved M-RoPE** (`mrope_section`) instead of plain 1-D RoPE — with a
//!   text-only prompt all three rows are the token index, so the T2I path is bit-unchanged;
//! * **DeepStack**: the tower's tapped features are added to the visual rows after each of the
//!   first `deepstack_visual_indexes.len()` decoder layers.
//!
//! The returned [`TextConditioning`] carries the image-token mask alongside the hidden states —
//! the DiT needs it to know which joint positions its condition latents occupy.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_gen::candle_nn::{
    ops::softmax_last_dim, rms_norm, rotary_emb, Embedding, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::{CandleError as Error, Result};
use candle_llm::models::deepstack::{
    add_visual_features, mrope_positions_mm, splice_vision_features,
};
use candle_llm::models::Qwen35VisionModel;
use candle_llm::primitives::{apply_rope, Rope};

use crate::config::{TextEncoderConfig, VisionConfig, SYSTEM_PROMPT};
use crate::quant::{guard_dense, QLinear, GROUP_SIZE};
use crate::reference::PreparedReference;

/// A bias-less, packed-detecting `[out, in]` projection — the seam that makes a pre-quantized
/// Qwen-Image 2.1 tier installable on candle. Qwen3-VL text attention and its SwiGLU are bias-free
/// (the config reader refuses `attention_bias: true`), and every decoder `Linear` in an installed
/// tier is packed at [`GROUP_SIZE`], so this is the only width the detect loader ever needs.
fn lin(in_dim: usize, out_dim: usize, vb: &VarBuilder, base: &str) -> Result<QLinear> {
    Ok(QLinear::linear_detect_gs(
        in_dim, out_dim, vb, base, false, GROUP_SIZE,
    )?)
}

/// The system-role prefix of the T2I template — exactly what upstream tokenizes to derive
/// `_drop_idx`.
pub fn system_prefix() -> String {
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
}

/// The full raw T2I template for `prompt` (`QwenImage21Pipeline.prompt_template_t2i`). An empty
/// prompt is rendered as a single space, as upstream does ("Qwen has no bos token").
pub fn prompt_template(prompt: &str) -> String {
    let prompt = if prompt.is_empty() { " " } else { prompt };
    format!(
        "{}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}

/// The literal `<|image_pad|>` placeholder — one per reference in the rendered template, which
/// the processor then expands to one token per merged vision patch.
pub const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";

/// The full raw **image-conditioned** template for `prompt` and `count` references
/// (`QwenImage21Pipeline.prompt_template_ti2i` with `_get_qwen_prompt_embeds`' multi-image
/// expansion). Note the single leading space before the second and later groups, which upstream's
/// `replace += f" <image{i}>…"` introduces and which changes the tokenization.
pub fn prompt_template_ti2i(prompt: &str, count: usize) -> String {
    let prompt = if prompt.is_empty() { " " } else { prompt };
    let mut groups = String::new();
    for i in 1..=count {
        if i > 1 {
            groups.push(' ');
        }
        groups.push_str(&format!(
            "<image{i}><|vision_start|>{IMAGE_PAD_TOKEN}<|vision_end|>"
        ));
    }
    format!(
        "{}<|im_start|>user\n{groups}{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}

/// The `<|image_pad|>` id for the loaded tokenizer — upstream's
/// `processor.tokenizer.encode("<|image_pad|>")[0]`, derived rather than read from
/// `config.json`'s `image_token_id` (the miniature parity snapshot's tokenizer maps the same
/// literal to a different id, and upstream trusts the tokenizer).
pub fn image_pad_token_id(tokenizer: &TextTokenizer) -> Result<i32> {
    let ids = tokenizer.encode_ids(IMAGE_PAD_TOKEN, false)?;
    match ids.first() {
        Some(&id) if ids.len() == 1 => Ok(id),
        _ => Err(Error::Msg(format!(
            "qwen_image_2_1: the tokenizer encodes `{IMAGE_PAD_TOKEN}` to {ids:?}, not a single \
             id; condition images cannot be placed in the prompt"
        ))),
    }
}

/// How many leading template tokens to drop from the hidden states: the tokenized system-role
/// prefix, derived from the loaded tokenizer so it tracks the snapshot rather than a constant.
pub fn system_prompt_drop_count(tokenizer: &TextTokenizer) -> Result<usize> {
    let ids = tokenizer.encode_ids(&system_prefix(), true)?;
    if ids.is_empty() {
        return Err(Error::Msg(
            "qwen_image_2_1: the tokenizer encodes the system prefix to nothing".into(),
        ));
    }
    Ok(ids.len())
}

/// Precomputed 1-D RoPE table (`θ = rope_theta`), half-split as HF/Qwen3 applies it.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    /// The table is computed in f32 and cast to the tower's compute `dtype` once, so a bf16 tower
    /// rotates bf16 queries with a bf16 table (candle's rope kernel wants one dtype throughout).
    fn new(
        head_dim: usize,
        theta: f32,
        max_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let half = inv.len();
        let inv = Tensor::from_vec(inv, (1, half), device)?;
        let t = Tensor::arange(0u32, max_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_len, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// Apply to `q`/`k` shaped `[B, H, L, D]`.
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let len = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, len)?;
        let sin = self.sin.narrow(0, 0, len)?;
        Ok((
            rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, kv_heads, len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv_heads, n_rep, len, head_dim))?
        .reshape((b, kv_heads * n_rep, len, head_dim))
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    kv_groups: usize,
    head_dim: usize,
    rotary: Arc<Rotary>,
}

impl Attention {
    fn new(cfg: &TextEncoderConfig, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self> {
        let (heads, kv_heads, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        // Packed-detect: an installed q8/q4 tier binds the packed triple straight into the
        // quantized weight; a dense snapshot takes the plain `candle_nn::Linear` path unchanged.
        let qkv = |out: usize, name: &str| lin(cfg.hidden_size, out, &vb, name);
        // The per-head RMSNorms are the leaves that stay dense in every tier.
        for name in ["q_norm", "k_norm"] {
            guard_dense(&vb, name)?;
        }
        Ok(Self {
            q_proj: qkv(heads * head_dim, "q_proj")?,
            k_proj: qkv(kv_heads * head_dim, "k_proj")?,
            v_proj: qkv(kv_heads * head_dim, "v_proj")?,
            o_proj: lin(heads * head_dim, cfg.hidden_size, &vb, "o_proj")?,
            q_norm: rms_norm(head_dim, cfg.rms_norm_eps as f64, vb.pp("q_norm"))?,
            k_norm: rms_norm(head_dim, cfg.rms_norm_eps as f64, vb.pp("k_norm"))?,
            heads,
            kv_heads,
            kv_groups: heads / kv_heads.max(1),
            head_dim,
            rotary,
        })
    }

    /// `rope` is `None` for the plain 1-D path (the text-to-image route, unchanged) and
    /// `Some((cos, sin))` — each `[1, L, head_dim]`, the NeoX `cat(freqs, freqs)` layout — for the
    /// interleaved M-RoPE tables the image-conditioned path builds.
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (b, len, _) = x.dims3()?;
        let shape = |t: Tensor, heads: usize| -> candle_core::Result<Tensor> {
            t.reshape((b, len, heads, self.head_dim))?.transpose(1, 2)
        };
        let q = shape(self.q_proj.forward(x)?, self.heads)?;
        let k = shape(self.k_proj.forward(x)?, self.kv_heads)?;
        let v = shape(self.v_proj.forward(x)?, self.kv_heads)?;
        // Per-head RMSNorm (Qwen3).
        let q =
            self.q_norm
                .forward(&q.flatten(0, 2)?)?
                .reshape((b, self.heads, len, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.kv_heads,
            len,
            self.head_dim,
        ))?;
        let (q, k) = match rope {
            // The 1-D table precomputed at load: byte-for-byte the text-to-image path.
            None => self.rotary.apply(&q, &k)?,
            // `apply_rope` wants `[B, L, H, D]`; both conventions are the same NeoX half-split.
            Some((cos, sin)) => {
                let rotate = |t: &Tensor| -> Result<Tensor> {
                    let t = t.transpose(1, 2)?.contiguous()?;
                    let t = apply_rope(&t, cos, sin, false).map_err(from_llm)?;
                    Ok(t.transpose(1, 2)?.contiguous()?)
                };
                (rotate(&q)?, rotate(&k)?)
            }
        };
        let k = repeat_kv(k, self.kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let ctx = attend(&q, &k, &v, mask, scale, candle_gen::ATTN_SCORES_BUDGET)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, len, self.heads * self.head_dim))?;
        Ok(self.o_proj.forward(&ctx)?)
    }
}

/// The scores → context step of [`Attention::forward`], through the F-003 i32-overflow guard
/// ([`candle_gen::sdpa_budgeted_bhsd`]) with the additive `[1, 1, L, L]` causal `mask`.
///
/// This used to be a plain `matmul · scale + mask → softmax → matmul` on the argument that
/// [`crate::loader::MAX_PROMPT_TOKENS`] (4096) bounds the sequence, so the widest scores tensor
/// was `32 · 4096² ≈ 5.4e8` elements. That cap bounds only the **tokenizer**: the image-conditioned
/// route ([`QwenImage21TextEncoder::encode_conditioning`]) expands each `<|image_pad|>` to one
/// token per merged vision patch *after* tokenizing — 1,024 per reference fitted to 1024² — so the
/// advertised ten-reference request runs this tower at ≈10.3k tokens and `32 · 10.3k² ≈ 3.4e9`
/// score elements, past `i32::MAX`. candle's CUDA softmax kernel indexes `row · ncols + col` as an
/// `int`, so that single pass wrapped negative and faulted (`CUDA_ERROR_ILLEGAL_ADDRESS`, the
/// sc-24114 ten-reference evidence run). The guard chunks the query rows past the budget; below it
/// — every text-to-image prompt, and up to six 1024²-fitted references — it is the same single
/// pass as before (see `chunked_and_unchunked_text_attention_agree` and
/// `ten_reference_conditioning_overflows_i32_and_chunks`).
fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
    budget: usize,
) -> Result<Tensor> {
    Ok(candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        Some(mask),
        softmax_last_dim,
        budget,
    )?)
}

struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl Mlp {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: lin(h, i, &vb, "gate_proj")?,
            up_proj: lin(h, i, &vb, "up_proj")?,
            down_proj: lin(i, h, &vb, "down_proj")?,
        })
    }

    /// SwiGLU: `down(silu(gate(x)) · up(x))`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.gate_proj.forward(x)?.silu()? * self.up_proj.forward(x)?)?;
        Ok(self.down_proj.forward(&gated)?)
    }
}

/// One pre-norm Qwen3 decoder block.
struct DecoderLayer {
    attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TextEncoderConfig, rotary: Arc<Rotary>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps as f64,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps as f64,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let h = self
            .attn
            .forward(&self.input_layernorm.forward(x)?, mask, rope)?;
        let x = (x + h)?;
        let h = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?;
        Ok((&x + h)?)
    }
}

/// The Qwen3-VL language tower as Qwen-Image 2.1 conditions on it: token embedding → N pre-norm
/// decoder layers → the last layer's **un-normalised** hidden states.
pub struct QwenImage21TextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    hidden_size: usize,
    device: Device,
    /// The dtype the weights were materialized at and the activations run at
    /// ([`crate::loader::compute_dtype`] on the production path).
    dtype: DType,
    /// Interleaved M-RoPE geometry for the image-conditioned path.
    mrope: Rope,
    mrope_section: [usize; 3],
    /// The ViT tower + its host processor geometry. `None` on a snapshot that ships no
    /// `vision_config` / `model.visual.*`, which makes the reference route a typed refusal
    /// instead of a wrong render.
    vision: Option<(Qwen35VisionModel, VisionConfig)>,
}

/// The conditioning one prompt produces: the hidden states the DiT's `txt_in` consumes, and —
/// for the image-conditioned path — which of those positions are condition-image slots.
pub struct TextConditioning {
    /// `[1, L, hidden]` at the tower's compute dtype, system prefix already dropped.
    pub hidden: Tensor,
    /// `L` flags, `true` at `<|image_pad|>` positions (all `false` for text-to-image). Each flag
    /// stands for a 2×2 group of condition latents in the joint sequence.
    pub image_pad_mask: Vec<bool>,
}

impl TextConditioning {
    /// Rows the DiT's `txt_in` actually consumes — everything that is not a condition-image slot.
    pub fn text_len(&self) -> usize {
        self.image_pad_mask.iter().filter(|m| !**m).count()
    }
}

impl QwenImage21TextEncoder {
    /// Build from a Qwen3-VL checkpoint. `prefix` is the language-tower prefix —
    /// `model.language_model` for the released `Qwen3VLForConditionalGeneration` layout.
    pub fn new(cfg: &TextEncoderConfig, vb: VarBuilder, prefix: &str) -> Result<Self> {
        let tower = prefix
            .split('.')
            .filter(|part| !part.is_empty())
            .fold(vb.clone(), |vb, part| vb.pp(part));
        // The token embedding stays dense in every installable tier (`crate::quant`), so it is
        // read as floats and guarded against a packed sibling rather than packed-detected.
        guard_dense(&tower, "embed_tokens")?;
        guard_dense(&tower, "norm")?;
        let embed_tokens = candle_gen::candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            tower.pp("embed_tokens"),
        )?;
        // The compute dtype is the `VarBuilder`'s: whatever the loader materialized the weights at
        // is what the activations run at (bf16 on CUDA/Metal like upstream, f32 on CPU).
        let dtype = vb.dtype();
        let rotary = Arc::new(Rotary::new(
            cfg.head_dim,
            cfg.rope_theta,
            crate::loader::MAX_PROMPT_TOKENS,
            dtype,
            vb.device(),
        )?);
        let vb_layers = tower.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_layers.pp(i))?);
        }
        // The final norm (`…norm.weight`) is deliberately NOT loaded — the conditioning is the last
        // layer's output before it.
        Ok(Self {
            embed_tokens,
            layers,
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
            dtype,
            mrope: Rope::standard(cfg.head_dim as i32, cfg.rope_theta),
            mrope_section: cfg.mrope_section,
            vision: None,
        })
    }

    /// The dtype the tower's weights were materialized at and its activations run at.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Attach the Qwen3-VL **vision** tower this snapshot ships (`model.visual.*`), which the
    /// reference route needs. Without it [`Self::encode_conditioning`] refuses any request that
    /// carries condition images.
    pub fn with_vision(mut self, tower: Qwen35VisionModel, cfg: VisionConfig) -> Self {
        self.vision = Some((tower, cfg));
        self
    }

    /// `true` when the loaded snapshot carried a vision tower.
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// The vision geometry, for callers that need the snapshot's `output_resolution` / processor
    /// before preparing references.
    pub fn vision_config(&self) -> Option<&VisionConfig> {
        self.vision.as_ref().map(|(_, cfg)| cfg)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The additive causal mask for a `len`-token prompt (prompts are never padded, so the
    /// attention mask is all-ones and the block is purely causal).
    fn causal_mask(&self, len: usize) -> Result<Tensor> {
        causal_mask(len, self.dtype, &self.device)
    }

    /// `input_ids` `[1, L]` (u32) → the last decoder layer's hidden states `[1, L, hidden]` at
    /// [`Self::dtype`], **before** the final norm.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_traced(input_ids, false).map(|(h, _)| h)
    }

    /// [`Self::forward`] that also returns the embedding output and every layer's output
    /// (`embed`, `layer_{i}`) when `trace` is set — the localisation seam the parity tests read.
    pub fn forward_traced(
        &self,
        input_ids: &Tensor,
        trace: bool,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let (_, len) = input_ids.dims2()?;
        let mut stages = Vec::new();
        let mut x = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        if trace {
            stages.push(("embed".to_string(), x.clone()));
        }
        let mask = self.causal_mask(len)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mask, None)?;
            if trace {
                stages.push((format!("layer_{i}"), x.clone()));
            }
        }
        Ok((x, stages))
    }

    /// Prompt → conditioning `[1, L − drop, hidden]` at [`Self::dtype`]: render the template,
    /// tokenize, run the tower, drop the `drop` system-prefix tokens.
    pub fn encode_prompt(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        drop: usize,
    ) -> Result<Tensor> {
        let tokens = tokenizer.tokenize_preformatted(&prompt_template(prompt))?;
        if tokens.ids.len() <= drop {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: prompt tokenized to {} tokens, not more than the {drop} template tokens to drop",
                tokens.ids.len()
            )));
        }
        let ids = input_ids(&tokens.ids, &self.device)?;
        let hidden = self.forward(&ids)?;
        let total = hidden.dim(1)?;
        Ok(hidden.i((.., drop..total, ..))?.contiguous()?)
    }

    /// Prompt (+ ordered condition images) → [`TextConditioning`].
    ///
    /// With `references` empty this is [`Self::encode_prompt`] with an all-`false` mask — the
    /// text-to-image path, unchanged. With references it renders [`prompt_template_ti2i`],
    /// expands each `<|image_pad|>` placeholder to that reference's merged-patch count, runs the
    /// ViT tower, splices its merged rows into the image positions, switches the decoder onto
    /// interleaved M-RoPE and fuses the DeepStack taps — the port of `_get_qwen_prompt_embeds`'
    /// `image is not None` branch.
    pub fn encode_conditioning(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        drop: usize,
        references: &[PreparedReference],
    ) -> Result<TextConditioning> {
        if references.is_empty() {
            let hidden = self.encode_prompt(tokenizer, prompt, drop)?;
            let len = hidden.dim(1)?;
            return Ok(TextConditioning {
                hidden,
                image_pad_mask: vec![false; len],
            });
        }
        let (tower, vision_cfg) = self.vision.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "qwen_image_2_1: this snapshot ships no Qwen3-VL vision tower \
                 (`text_encoder/config.json` has no `vision_config`, or `model.visual.*` is \
                 missing), so reference images cannot be conditioned on. Reference conditioning \
                 needs the tower: upstream encodes every condition image as vision context for the \
                 text encoder as well as VAE latents for the DiT."
                    .to_string(),
            )
        })?;
        let merge = vision_cfg.processor.merge_size.max(1) as i32;
        let image_token = image_pad_token_id(tokenizer)?;

        // 1. Render + tokenize the image-conditioned template, then expand each single
        //    `<|image_pad|>` placeholder into one token per merged vision patch — what
        //    `Qwen3VLProcessor` does when it is handed `pixel_values`.
        let text = prompt_template_ti2i(prompt, references.len());
        let tokens = tokenizer.tokenize_preformatted(&text)?;
        let placeholders = tokens.ids.iter().filter(|&&id| id == image_token).count();
        if placeholders != references.len() {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: the image-conditioned template tokenized to {placeholders} \
                 `{IMAGE_PAD_TOKEN}` placeholders for {} reference images; the snapshot's \
                 tokenizer does not render the upstream template",
                references.len()
            )));
        }
        let mut ids: Vec<i32> = Vec::with_capacity(tokens.ids.len());
        let mut reference_cursor = 0usize;
        for &id in &tokens.ids {
            if id == image_token {
                let slots = references[reference_cursor].vision_slots();
                ids.extend(std::iter::repeat_n(id, slots));
                reference_cursor += 1;
            } else {
                ids.push(id);
            }
        }
        if ids.len() <= drop {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: the image-conditioned prompt tokenized to {} tokens, not more \
                 than the {drop} template tokens to drop",
                ids.len()
            )));
        }

        // 2. The ViT tower, per reference in order; merged rows and DeepStack taps concatenate in
        //    the same order the placeholders appear.
        let mut merged: Vec<Tensor> = Vec::with_capacity(references.len());
        let mut taps: Vec<Vec<Tensor>> = Vec::new();
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(references.len());
        for reference in references {
            let out = tower
                .forward_with_deepstack(&reference.pixel_values, &[reference.grid_thw])
                .map_err(from_llm)?;
            if taps.is_empty() {
                taps = vec![Vec::with_capacity(references.len()); out.deepstack_features.len()];
            } else if taps.len() != out.deepstack_features.len() {
                return Err(Error::Msg(
                    "qwen_image_2_1: the vision tower returned a different number of DeepStack \
                     taps for two references"
                        .into(),
                ));
            }
            // The tower ran at its own (loader-cast) dtype; both halves of the one encoder
            // run at the language tower's compute dtype from here on.
            for (slot, feature) in taps.iter_mut().zip(out.deepstack_features) {
                slot.push(feature.to_dtype(self.dtype)?);
            }
            merged.push(out.pooler_output.to_dtype(self.dtype)?);
            grids.push(reference.grid_thw);
        }
        let join = |parts: &[Tensor]| -> Result<Tensor> {
            let refs: Vec<&Tensor> = parts.iter().collect();
            Ok(Tensor::cat(&refs, 0)?)
        };
        let vision_features = join(&merged)?;
        let deepstack: Vec<Tensor> = taps
            .iter()
            .map(|parts| join(parts))
            .collect::<Result<Vec<_>>>()?;

        // 3. Embed, splice the vision rows in, build interleaved M-RoPE positions.
        let input_ids = input_ids(&ids, &self.device)?;
        let embeds = self
            .embed_tokens
            .forward(&input_ids)?
            .to_dtype(self.dtype)?;
        let embeds = splice_vision_features(&embeds, &ids, &vision_features, &[image_token])
            .map_err(from_llm)?;
        let (t_row, h_row, w_row, _) =
            mrope_positions_mm(&ids, &grids, image_token, &[], i32::MIN, merge)
                .map_err(from_llm)?;
        let (cos, sin) = self
            .mrope
            .mrope_interleaved_cos_sin(
                [&t_row, &h_row, &w_row],
                self.mrope_section,
                self.dtype,
                &self.device,
            )
            .map_err(from_llm)?;

        // 4. Decoder layers with DeepStack fusion on the first `deepstack.len()` of them.
        let visual_pos_mask: Vec<bool> = ids.iter().map(|&id| id == image_token).collect();
        let mask = self.causal_mask(ids.len())?;
        let mut x = embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mask, Some((&cos, &sin)))?;
            if let Some(feature) = deepstack.get(i) {
                x = add_visual_features(&x, &visual_pos_mask, feature).map_err(from_llm)?;
            }
        }

        let total = x.dim(1)?;
        Ok(TextConditioning {
            hidden: x.i((.., drop..total, ..))?.contiguous()?,
            image_pad_mask: visual_pos_mask[drop..].to_vec(),
        })
    }
}

/// The additive `[1, 1, len, len]` causal mask: `0` at and below the diagonal, `-inf` above.
fn causal_mask(len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..len)
        .flat_map(|i| (0..len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Ok(Tensor::from_vec(mask, (1, 1, len, len), device)?.to_dtype(dtype)?)
}

/// `candle-llm`'s error into this crate's. `CandleError` has no `Unsupported` variant (unlike the
/// MLX twin's), so every non-tensor failure keeps its message under `Msg`.
fn from_llm(e: candle_llm::Error) -> Error {
    match e {
        candle_llm::Error::Candle(c) => Error::Candle(c),
        other => Error::Msg(format!("qwen_image_2_1 vision: {other}")),
    }
}

/// `[1, L]` u32 input ids on `device` from gen-core's host-side token vector.
pub fn input_ids(ids: &[i32], device: &Device) -> Result<Tensor> {
    let host: Vec<u32> = ids.iter().map(|&id| id.max(0) as u32).collect();
    let len = host.len();
    Ok(Tensor::from_vec(host, (1, len), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_upstream_shape() {
        let t = prompt_template("a fox");
        assert_eq!(
            t,
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(prompt_template("").contains("user\n <|im_end|>"));
        assert!(t.starts_with(&system_prefix()));
    }

    /// The ten-reference contract at the production tower: the image-conditioned sequence is
    /// past the tokenizer cap the old single pass relied on, its scores tensor is past
    /// `i32::MAX`, and the shipped budget chunks it — while the capped text-to-image prompt stays
    /// a single pass (`block == len`), byte-identical to before.
    #[test]
    fn ten_reference_conditioning_overflows_i32_and_chunks() {
        use crate::config::{
            TextEncoderConfig, IMAGE_TOKENS_PER_SLOT, MAX_REFERENCE_IMAGES, OUTPUT_RESOLUTION,
            VAE_SCALE_FACTOR,
        };
        use candle_gen::attention::attention_budget_from_usize;

        let heads = TextEncoderConfig::production().num_attention_heads;
        let slots_per_reference =
            ((OUTPUT_RESOLUTION / VAE_SCALE_FACTOR) as usize).pow(2) / IMAGE_TOKENS_PER_SLOT;
        assert_eq!(slots_per_reference, 1024);
        // Image slots alone — the template's own tokens only add to it.
        let len = MAX_REFERENCE_IMAGES * slots_per_reference;
        assert!(len > crate::loader::MAX_PROMPT_TOKENS);
        let scores = (heads * len * len) as u64;
        assert!(scores > i32::MAX as u64, "{scores}");

        let plan = |len: usize| {
            attention_budget_from_usize(candle_gen::ATTN_SCORES_BUDGET)
                .query_block_rows((heads * len) as u64, len as u64) as usize
        };
        assert!(
            plan(len) < len,
            "ten references must chunk: block {}",
            plan(len)
        );
        assert!((heads * plan(len) * len) as u64 <= i32::MAX as u64);
        let capped = crate::loader::MAX_PROMPT_TOKENS;
        assert_eq!(
            plan(capped),
            capped,
            "the text-to-image cap stays a single pass"
        );
    }

    /// The guarded path at a forced budget agrees with the un-chunked pass on random tensors
    /// under the tower's own causal mask — the per-query `[1, 1, L, L]` mask must be narrowed
    /// with its query chunk. Tolerance, not equality: a different GEMM `M` changes the
    /// accumulation order.
    #[test]
    fn chunked_and_unchunked_text_attention_agree() {
        use candle_gen::attention::attention_budget_from_usize;

        let (heads, len, head_dim) = (4usize, 13usize, 8usize);
        let dev = Device::Cpu;
        let fill = |seed: usize| -> Tensor {
            let n = heads * len * head_dim;
            let v: Vec<f32> = (0..n)
                .map(|i| (((i * 37 + seed * 11) % 97) as f32 / 97.0) - 0.5)
                .collect();
            Tensor::from_vec(v, (1, heads, len, head_dim), &dev).unwrap()
        };
        let (q, k, v) = (fill(0), fill(1), fill(2));
        let mask = causal_mask(len, DType::F32, &dev).unwrap();
        let scale = 1.0 / (head_dim as f64).sqrt();

        let single = attend(&q, &k, &v, &mask, scale, usize::MAX).unwrap();
        // Budgets forcing one-row chunks and ragged three-row chunks (13 = 3·4 + 1).
        for rows in [1usize, 3] {
            let budget = heads * len * rows;
            assert_eq!(
                attention_budget_from_usize(budget)
                    .query_block_rows((heads * len) as u64, len as u64) as usize,
                rows
            );
            let chunked = attend(&q, &k, &v, &mask, scale, budget).unwrap();
            let diff = (&single - &chunked)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff <= 1e-5, "rows {rows}: max |Δ| = {diff}");
        }
    }
}
