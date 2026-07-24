//! Joint (dual-stream) attention for the NR-MMDiT — **owned by sc-14040**.
//!
//! Port of `Attention` + `MageDoubleStreamAttnProcessor`
//! (`_vendor/mage_flow/models/modules/mage_layers.py:212-514`):
//!
//! - separate `to_q`/`to_k`/`to_v` per stream with bias ([`crate::config::QKV_BIAS`]), plus
//!   `add_q_proj`/`add_k_proj`/`add_v_proj` for the text stream;
//! - **QK-RMSNorm on both streams** before the rotation;
//! - msrope applied to the **image** q/k only (`:421-422`) — the text stream is never rotated,
//!   matching the published `apply_text_rotary_emb: false`;
//! - concatenation order **`[text, image]`**, `causal=false` (`:424`, `:490`), default softmax
//!   scale, with per-sample isolation expressed as varlen `cu_seqlens` rather than a block-diagonal
//!   mask (MLX has no varlen flash kernel — the port picks its own equivalent, and must keep the
//!   isolation exact because native-resolution packing puts unrelated samples in one sequence).
//!
//! Rotation convention is adjacent-pair complex (`view_as_complex` over `[..., -1, 2]`, `:15-21`),
//! so the table from [`crate::rope_embedder`] is consumed as (cos, sin) pairs over adjacent lanes,
//! **not** the half-split convention FLUX/Qwen use.
