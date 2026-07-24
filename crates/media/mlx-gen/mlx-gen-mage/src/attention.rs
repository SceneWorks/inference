//! Joint (dual-stream) attention for the NR-MMDiT — **owned by sc-14040**.
//!
//! Port of `Attention` + `MageDoubleStreamAttnProcessor`
//! (`_vendor/mage_flow/models/modules/mage_layers.py:212-514`):
//!
//! - separate `to_q`/`to_k`/`to_v` per stream with bias ([`crate::config::QKV_BIAS`]), plus
//!   `add_q_proj`/`add_k_proj`/`add_v_proj` for the text stream;
//! - **QK-RMSNorm on both streams** before the rotation;
//! - msrope applied to the **image** q/k only (`:421-422`) — the text stream is never rotated,
//!   which is what [`crate::config::APPLY_TEXT_ROTARY_EMB`] pins. The config reader now *rejects*
//!   a checkpoint that flips it, rather than running this port against a model that expects
//!   rotated text;
//! - order **`[text, image]`** ([`crate::config::TEXT_STREAM_FIRST`]), `causal=false` (`:424`,
//!   `:490`), default softmax scale. The reference expresses that order as scatter offsets rather
//!   than a `cat`: text lands at each sample's start and image is shifted by that sample's text
//!   length (`:466-467`; the `torch.cat` at `:425-427` is commented out). Per-sample isolation is
//!   varlen `cu_seqlens`, not a block-diagonal mask (MLX has no varlen flash kernel — the port
//!   picks its own equivalent, and must keep the isolation exact because native-resolution packing
//!   puts unrelated samples in one sequence).
//!
//! Rotation convention is adjacent-pair complex (`view_as_complex` over `[..., -1, 2]`, `:15-21`),
//! so the table from [`crate::rope_embedder`] is consumed as (cos, sin) pairs over adjacent lanes,
//! **not** the half-split convention FLUX/Qwen use.
