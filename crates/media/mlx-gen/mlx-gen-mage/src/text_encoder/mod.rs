//! Qwen3-VL-4B text encoder — **owned by sc-14038** (the vision tower by sc-14048).
//!
//! Port of `_vendor/mage_flow/models/modules/text_encoder.py` — the **embedding forward only**
//! (see "Explicitly out of scope" below). Entry points: [`load()`] for a published snapshot, then
//! [`MageTextEncoder::encode`] for prompt bodies, or [`MageTextEncoder::encode_packed_ids`] when
//! the caller builds its own token ids.
//!
//! ## What conditioning actually is
//!
//! **The FINAL (36th) decoder hidden state, AFTER the model's final RMSNorm**, with the first
//! [`DROP_IDX_GEN`](crate::config::DROP_IDX_GEN) (gen) /
//! [`DROP_IDX_EDIT`](crate::config::DROP_IDX_EDIT) (edit) system-prompt tokens dropped
//! (`text_encoder.py:83-84`, `:156`, `:290`, `:560`; `models/utils.py:55`, `:64`).
//!
//! This **corrects** the epic's original z-image-borrowed assumption of a penultimate layer, and
//! the difference is not subtle — measured against the committed golden on the real 4.1B
//! checkpoint, the penultimate candidate is off by max_abs 10433.29 and the final *pre*-norm
//! candidate by 4225.29, while the post-norm final layer is bit-exact
//! (`_vendor/MAGE_FLOW_GAPS.md` GAP 1). `output_hidden_states: False` (`:521`) means the
//! penultimate layer is not even materialised on this path.
//!
//! The pooled `vec` the reference also produces is discarded by the DiT
//! (`models/mage_flow.py:116`), so no pooled vector is needed. Conditioning enters the DiT as
//! `RMSNorm(2560, eps=1e-6) → Linear(2560 → 3072)`.
//!
//! Shapes: [`crate::config::QwenVlTextConfig::mage_flow`] plus
//! [`TE_RMS_NORM_EPS`](crate::config::TE_RMS_NORM_EPS),
//! [`TE_ROPE_THETA`](crate::config::TE_ROPE_THETA) and the two verbatim chat templates
//! ([`PROMPT_TEMPLATE_GEN`](crate::config::PROMPT_TEMPLATE_GEN) /
//! [`PROMPT_TEMPLATE_EDIT`](crate::config::PROMPT_TEMPLATE_EDIT)).
//!
//! ## Explicitly out of scope (sc-14105 decision)
//!
//! The reference's mandatory, fail-closed content-moderation gate is **not** ported: no `lm_head`,
//! no KV-cache decode loop, no sampling, no `.generate()`, no `set_output_mode` switch, no
//! `CONTENT_FILTER_*` prompts, no `FilterVerdict`, no `make_refusal_image`. The MLX encoder runs in
//! embedding mode permanently — there is no mode to switch.
//!
//! **Do not book a memory saving for omitting `lm_head`.** `tie_word_embeddings: true` and the
//! shipped checkpoint contains no `lm_head` tensor at all — the tied matrix *is*
//! `embed_tokens.weight` (`[151936, 2560]` bf16 = 0.778 GB), which the embedding forward needs
//! regardless. The saving is latency, not bytes.
//!
//! ## Tolerance — measured
//!
//! Thirty-six bf16 decoder layers accumulate real cross-backend drift (the same reference on MPS
//! instead of CPU moves the tensor by mean_rel ≈ 2.7e-2), so the parity gate against
//! `mage_flow_te_golden.safetensors` is a tolerance, never an equality. **This port measures
//! `max_abs 1.523` / `mean_rel 2.18e-2`** against the committed CPU golden on the real 4.1B
//! checkpoint — i.e. at the oracle's own bf16 noise floor — while the penultimate candidate is
//! 10503.13 and the un-normed final layer 4309.12. `tests/te_parity_real_weights.rs` carries the
//! full sensitivity table and the reasoning behind the gate values.
//!
//! ## Prompt length
//!
//! Truncate the **templated** prompt at [`max_prompt_tokens`](crate::config::max_prompt_tokens)
//! (`drop_idx`), i.e. 2082 gen / 2112 edit — [`TXT_MAX_LENGTH`](crate::config::TXT_MAX_LENGTH)
//! **plus** the tokens that are about to be dropped (`pipeline.py:225-228`). Do not read the
//! reference's `ModelConfig` dataclass default (4096, `mage_flow.py:31`) — `load_from_repo`
//! overrides it with the published 2048 (`pipeline.py:745`) — and do not forget the `+ drop_idx`
//! term. Every boundary golden uses a short prompt, so **neither mistake shows up in a parity
//! test**; that is why the value is a constant here rather than a transcription.
//!
//! ## Weight loading
//!
//! There is no shared `loader.rs` (see the decision note in `lib.rs`): [`load()`]/[`load_tokenizer`]
//! for this component live **inside this directory**, so the concurrent VAE and DiT ports never
//! touch a file this story owns.
//!
//! ## Module map
//!
//! | module | port of |
//! | --- | --- |
//! | [`rope`] | `Qwen3VLTextRotaryEmbedding` — interleaved 3-axis M-RoPE |
//! | [`attention`] | `Qwen3VLTextAttention` (as re-bound by `qwen3_patch_forward`) |
//! | [`mlp`] | `Qwen3VLTextMLP` — SwiGLU |
//! | [`layer`] | `Qwen3VLTextDecoderLayer` |
//! | [`encoder`] | `Qwen3VLTextModel.forward` (patched, `cu_seqlens`-packed) |
//! | [`prompt`] | `PROMPT_TEMPLATE` + the truncation/`drop_idx` policy |
//! | [`encode`] | the `TextEncoder` wrapper + `_encode_texts_packed` |
//! | [`mod@load`] | `TextEncoder.__init__`'s `from_pretrained` half |
//!
//! ## Seam for the vision tower (sc-14048)
//!
//! Editing adds the Qwen3-VL vision tower **around** this LM; nothing here is restructured. The
//! three hooks it needs are already public — [`Qwen3VlTextEncoder::embed`] (splice merged image
//! features over the `<|image_pad|>` run), [`Qwen3VlTextEncoder::layers`] +
//! [`Qwen3VlDecoderLayer::forward`] (drive the stack and inject the deepstack features into **LM
//! layers `0..deepstack.len()`**, additively, **only over the `<|image_pad|>` run** — *not* at
//! `deepstack_visual_indexes`, which are vision-tower **extraction** indices; see the detailed note
//! in [`encoder`]), and [`Qwen3VlTextEncoder::final_norm`] — plus
//! [`MageTextEncoder::encode_packed_ids`], which takes caller-built ids so the placeholder run can
//! be expanded to the merged-token count, and [`MRopePositions`], which already carries three
//! independent axes. The tower itself need not be written from scratch: `mlx_gen_boogu::VisionTower`
//! with a `VisionConfig` of `depth 24 / hidden 1024 → out 2560 / patch 16 / spatial_merge 2 /
//! deepstack [5, 11, 17]` is exactly Mage's published `vision_config` (it is already reused that way
//! by `mlx-gen-krea`, whose Qwen3-VL-4B is the same tower).

pub mod attention;
pub mod encode;
pub mod encoder;
pub mod layer;
pub mod load;
pub mod mlp;
pub mod prompt;
pub mod rope;

pub use attention::Qwen3VlAttention;
pub use encode::{Conditioning, MageTextEncoder};
pub use encoder::{cu_seqlens_from_lens, Qwen3VlTextEncoder};
pub use layer::Qwen3VlDecoderLayer;
pub use load::{
    load, load_lm, load_multimodal, load_tokenizer, mage_vision_config, verify_text_config,
    COMPONENT_DIR, LM_PREFIX,
};
pub use mlp::Qwen3VlMlp;
pub use prompt::{edit_body, PromptKind, EDIT_IMAGE_PLACEHOLDER};
pub use rope::{mrope_cos_sin, MRopePositions};

pub(crate) use load::{embedding, lin};

/// Join a module prefix with a leaf name, tolerating an empty prefix (so a flat test fixture and a
/// real `model.language_model.layers.{i}` tree both resolve without a stray leading dot).
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
