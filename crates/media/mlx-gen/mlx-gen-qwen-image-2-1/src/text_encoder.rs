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
//! template, and [`system_prompt_drop_count`] derives it the same way from the loaded tokenizer (14
//! for the released tokenizer, pinned by the ignored real-weight test) instead of hard-coding it.
//!
//! For a text-only prompt the three mRoPE position streams (`mrope_section [24, 20, 20]`,
//! interleaved) are all the plain token index, so the rotary embedding is exactly 1-D RoPE at
//! `rope_theta = 5e6` — which is why the decoder block is **reused** from Z-Image's Qwen3 port
//! ([`mlx_gen_z_image::text_encoder::EncoderLayer`]: per-head `q_norm`/`k_norm`, bias-free GQA,
//! HF half-split RoPE, SwiGLU) rather than ported a third time. The DeepStack visual injection and
//! the vision tower engage only with condition images — see *The image-conditioned path* below.
//!
//! Activations run f32 over the bf16 weight store (the sibling text encoders' policy); the DiT
//! rounds the result to its own compute dtype at `txt_in`.
//!
//! # The image-conditioned path (sc-24110)
//!
//! With condition images the template becomes `prompt_template_ti2i` — one
//! `<imageN><|vision_start|><|image_pad|><|vision_end|>` group per reference, in order, ahead of
//! the prompt — and the full Qwen3-VL multimodal path engages:
//!
//! * each `<|image_pad|>` placeholder is **expanded** to one token per merged 2×2 vision patch,
//!   exactly as `Qwen3VLProcessor` does when it sees `pixel_values`;
//! * the ViT tower ([`mlx_llm::models::Qwen3VLVisionModel`]) encodes the references and its merged
//!   rows are **spliced** into the image-token positions;
//! * positions become **interleaved M-RoPE** (`mrope_section`) instead of plain 1-D RoPE — with a
//!   text-only prompt all three rows are the token index, so the T2I path is bit-unchanged;
//! * **DeepStack**: the tower's tapped features are added to the visual rows after each of the
//!   first `deepstack_visual_indexes.len()` decoder layers.
//!
//! The returned [`TextConditioning`] carries the image-token mask alongside the hidden states —
//! the DiT needs it to know which joint positions its condition latents occupy.

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_z_image::text_encoder::EncoderLayer;
use mlx_llm::models::deepstack::{add_visual_features, mrope_positions_mm, splice_vision_features};
use mlx_llm::models::Qwen3VLVisionModel;
use mlx_llm::primitives::Rope;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};

use crate::config::{TextEncoderConfig, VisionConfig, SYSTEM_PROMPT};
use crate::reference::PreparedReference;

/// Group size for a pre-quantized (packed) Qwen3 tower — the codebase-wide default (64).
/// Single-sourced from [`crate::quant::GROUP_SIZE`]: the converter writes tiers at this group
/// size and the packed-detect loaders must read them back at the same one.
const GROUP_SIZE: i32 = crate::quant::GROUP_SIZE;

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

/// The Qwen3-VL language tower as Qwen-Image 2.1 conditions on it: token embedding → N pre-norm
/// decoder layers → the last layer's **un-normalised** hidden states.
pub struct QwenImage21TextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<EncoderLayer>,
    rope: TextRope,
    hidden_size: usize,
    /// The SwiGLU intermediate width — the second of the two `Linear` input widths
    /// [`Self::quantize`] has to be able to cover at [`GROUP_SIZE`].
    intermediate_size: usize,
    /// Interleaved M-RoPE geometry for the image-conditioned path.
    mrope: Rope,
    mrope_section: [usize; 3],
    /// The ViT tower + its host processor geometry. `None` on a snapshot that ships no
    /// `vision_config` / `model.visual.*`, which makes the reference route a typed refusal
    /// instead of a wrong render.
    vision: Option<(Qwen3VLVisionModel, VisionConfig)>,
}

/// The conditioning one prompt produces: the hidden states the DiT's `txt_in` consumes, and —
/// for the image-conditioned path — which of those positions are condition-image slots.
pub struct TextConditioning {
    /// `[1, L, hidden]`, f32, system prefix already dropped.
    pub hidden: Array,
    /// `L` flags, `true` at `<|image_pad|>` positions (all `false` for text-to-image). Each flag
    /// stands for a 2×2 group of condition latents in the joint sequence.
    pub image_pad_mask: Vec<bool>,
}

impl TextConditioning {
    /// Vision slots per reference, in order — the run lengths of `image_pad_mask`, split by the
    /// per-reference slot counts the caller supplies.
    pub fn text_len(&self) -> usize {
        self.image_pad_mask.iter().filter(|m| !**m).count()
    }
}

impl QwenImage21TextEncoder {
    /// Build from a Qwen3-VL checkpoint. `prefix` is the language-tower prefix —
    /// `model.language_model` for the released `Qwen3VLForConditionalGeneration` layout.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &TextEncoderConfig) -> Result<Self> {
        let join = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };
        let embed_tokens = mlx_gen::quant::embedding(w, &join("embed_tokens"), GROUP_SIZE)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::from_weights(
                w,
                &join(&format!("layers.{i}")),
                cfg.num_attention_heads as i32,
                cfg.num_key_value_heads as i32,
                cfg.head_dim as i32,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            rope: TextRope::new(cfg.head_dim as i32, cfg.rope_theta),
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            mrope: Rope::standard(cfg.head_dim as i32, cfg.rope_theta),
            mrope_section: cfg.mrope_section,
            vision: None,
        })
    }

    /// Attach the Qwen3-VL **vision** tower this snapshot ships (`model.visual.*`), which the
    /// reference route needs. Without it [`Self::encode_conditioning`] refuses any request that
    /// carries condition images.
    pub fn with_vision(mut self, tower: Qwen3VLVisionModel, cfg: VisionConfig) -> Self {
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

    /// The SwiGLU intermediate width — the other `Linear` input width a tier has to cover.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// `true` iff both decoder `Linear` input widths are multiples of [`crate::quant::GROUP_SIZE`] — i.e. this
    /// geometry can be affine-quantized at the one group size a Qwen-Image 2.1 tier may declare
    /// ([`crate::quant`]). The released Qwen3 tower (4096 / 12288) can; the miniature parity
    /// snapshot (32 / 64) cannot.
    pub fn is_group_aligned(&self) -> bool {
        self.hidden_size.is_multiple_of(GROUP_SIZE as usize)
            && self.intermediate_size.is_multiple_of(GROUP_SIZE as usize)
    }

    /// Quantize every decoder `Linear` to Q4/Q8 at [`crate::quant::GROUP_SIZE`] (the embedding and the norms stay
    /// dense — see [`crate::quant`]'s per-component table), returning whether anything was packed.
    ///
    /// A geometry [`Self::is_group_aligned`] rejects is left **dense** rather than packed at a
    /// second group size: a tier declares exactly one `quantization.group_size`, and a component
    /// packed at another would be decoded at the wrong bit-width on reload. This mirrors
    /// [`crate::transformer::QwenImage21Transformer::quantize`]'s own width fallback, minus its
    /// group-32 arm, which a shippable tier cannot use.
    pub fn quantize(&mut self, bits: i32) -> Result<bool> {
        if !self.is_group_aligned() {
            return Ok(false);
        }
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(true)
    }

    /// `input_ids` `[1, L]` (i32) + `attention_mask` `[1, L]` → the last decoder layer's hidden
    /// states `[1, L, hidden]`, f32, **before** the final norm.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        self.forward_traced(input_ids, attention_mask, false)
            .map(|(hidden, _)| hidden)
    }

    /// [`Self::forward`] that also returns the embedding output and every layer's output
    /// (`embed`, `layer_{i}`) when `trace` is set — the localisation seam the parity tests read.
    pub fn forward_traced(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        trace: bool,
    ) -> Result<(Array, Vec<(String, Array)>)> {
        let shape = input_ids.shape();
        let (b, s) = (shape[0], shape[1]);
        let mut stages = Vec::new();
        let mut x = self
            .embed_tokens
            .forward(input_ids)?
            .as_dtype(Dtype::Float32)?;
        if trace {
            stages.push(("embed".to_string(), x.clone()));
        }
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &cos, &sin, &mask)?;
            if trace {
                stages.push((format!("layer_{i}"), x.clone()));
            }
        }
        Ok((x, stages))
    }

    /// Prompt → conditioning `[1, L − drop, hidden]` (f32): render the template, tokenize, run the
    /// tower, drop the `drop` system-prefix tokens.
    pub fn encode_prompt(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        drop: usize,
    ) -> Result<Array> {
        let text = prompt_template(prompt);
        let tokens = tokenizer.tokenize_preformatted(&text)?;
        if tokens.ids.len() <= drop {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: prompt tokenized to {} tokens, not more than the {drop} template tokens to drop",
                tokens.ids.len()
            )));
        }
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tokens);
        let hidden = self.forward(&input_ids, &attention_mask)?;
        let total = hidden.shape()[1];
        Ok(hidden.index((.., drop as i32..total, ..)))
    }

    /// Prompt (+ ordered condition images) → [`TextConditioning`].
    ///
    /// With `references` empty this is [`Self::encode_prompt`] with an all-`false` mask — the
    /// text-to-image path, unchanged. With references it renders
    /// [`prompt_template_ti2i`], expands each `<|image_pad|>` placeholder to that reference's
    /// merged-patch count, runs the ViT tower, splices its merged rows into the image positions,
    /// switches the decoder onto interleaved M-RoPE and fuses the DeepStack taps — the port of
    /// `_get_qwen_prompt_embeds`' `image is not None` branch.
    pub fn encode_conditioning(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        drop: usize,
        references: &[PreparedReference],
    ) -> Result<TextConditioning> {
        if references.is_empty() {
            let hidden = self.encode_prompt(tokenizer, prompt, drop)?;
            let len = hidden.shape()[1] as usize;
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
                    .into(),
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
        let mut merged: Vec<Array> = Vec::with_capacity(references.len());
        let mut taps: Vec<Vec<Array>> = Vec::new();
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
            for (slot, feature) in taps.iter_mut().zip(out.deepstack_features) {
                slot.push(feature.as_dtype(Dtype::Float32)?);
            }
            merged.push(out.pooler_output.as_dtype(Dtype::Float32)?);
            grids.push(reference.grid_thw);
        }
        let join = |parts: &[Array]| -> Result<Array> {
            let refs: Vec<&Array> = parts.iter().collect();
            Ok(concatenate_axis(&refs, 0)?)
        };
        let vision_features = join(&merged)?;
        let deepstack: Vec<Array> = taps
            .iter()
            .map(|parts| join(parts))
            .collect::<Result<Vec<_>>>()?;

        // 3. Embed, splice the vision rows in, build interleaved M-RoPE positions.
        let seq = ids.len() as i32;
        let input_ids = Array::from_slice(&ids, &[1, seq]);
        let embeds = self
            .embed_tokens
            .forward(&input_ids)?
            .as_dtype(Dtype::Float32)?;
        let hidden = self.hidden_size as i32;
        let embeds = splice_vision_features(
            &embeds,
            &ids,
            &vision_features,
            &[image_token],
            hidden,
            Dtype::Float32,
        )
        .map_err(from_llm)?;
        let (t_row, h_row, w_row, _) =
            mrope_positions_mm(&ids, &grids, image_token, &[], i32::MIN, merge)
                .map_err(from_llm)?;
        let (cos, sin) = self
            .mrope
            .mrope_interleaved_cos_sin([&t_row, &h_row, &w_row], self.mrope_section, Dtype::Float32)
            .map_err(from_llm)?;

        // 4. Decoder layers with DeepStack fusion on the first `deepstack.len()` of them.
        let visual_pos_mask: Vec<bool> = ids.iter().map(|&id| id == image_token).collect();
        let attention_mask = Array::from_slice(&vec![1i32; ids.len()], &[1, seq]);
        let mask = build_mask(&attention_mask, 1, seq)?;
        let mut x = embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &cos, &sin, &mask)?;
            if let Some(feature) = deepstack.get(i) {
                x = add_visual_features(&x, &visual_pos_mask, feature).map_err(from_llm)?;
            }
        }

        let total = x.shape()[1];
        Ok(TextConditioning {
            hidden: x.index((.., drop as i32..total, ..)),
            image_pad_mask: visual_pos_mask[drop..].to_vec(),
        })
    }
}

fn from_llm(e: mlx_llm::Error) -> Error {
    match e {
        mlx_llm::Error::Unsupported(m) => Error::Unsupported(m),
        mlx_llm::Error::MissingTensor(k) => Error::MissingTensor(k),
        other => Error::Msg(format!("qwen_image_2_1 vision: {other}")),
    }
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
}
