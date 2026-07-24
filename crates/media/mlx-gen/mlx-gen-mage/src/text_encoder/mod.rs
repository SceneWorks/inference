//! Qwen3-VL-4B text encoder — **owned by sc-14038** (the vision tower by sc-14048).
//!
//! Port of `_vendor/mage_flow/models/modules/text_encoder.py`. Add submodules under this directory
//! (`attention.rs`, `layer.rs`, `mlp.rs`, `encoder.rs`, …, mirroring
//! `mlx-gen-z-image/src/text_encoder/`); nothing outside this directory needs to change, which is
//! what keeps this story parallel with the VAE and DiT ports.
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
//! ## Tolerance
//!
//! Thirty-six bf16 decoder layers accumulate real cross-backend drift (the same reference on MPS
//! instead of CPU moves the tensor by mean_rel ≈ 2.7e-2), so the parity gate against
//! `mage_flow_te_golden.safetensors` must be a tolerance, never an equality.
