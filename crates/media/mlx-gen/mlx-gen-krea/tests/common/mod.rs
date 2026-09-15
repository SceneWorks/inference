#![allow(dead_code)]
//! Cross-backend fixture geometry for the Krea 2 parity goldens (sc-19496).
//!
//! Every file under `tests/fixtures/` here — `dit_golden.safetensors`, `rope_golden.safetensors`,
//! `single_block_golden.safetensors`, `te_golden.safetensors`, `text_fusion_golden.safetensors` and
//! `variant5_native_keys.txt` — is committed **byte-identical** to the file `candle-gen-krea` commits
//! under the same name. Both lanes therefore load the same bytes through their own hand-typed
//! geometry, so a drift in either config leaves both lanes internally consistent and both parity
//! suites green while the two backends compare tensors dumped at one shape against a model built at
//! another. Nothing could see that: the two crates cannot import each other, because `mlx-gen-*`
//! builds on macOS only.
//!
//! The numbers the two lanes must agree on are declared here as `SHARED_FIXTURE_*` constants, and the
//! constructors below build from them. `check_cross_backend_geometry` in `scripts/check-workspace.py`
//! compares every `SHARED_FIXTURE_*` declaration under this crate's `tests/` against the candle
//! crate's, by name set and by value — so the module docs that used to assert "the SAME fixtures" in
//! prose now describe something enforced.
//!
//! `mod common;` is compiled into every including test binary, so only a subset is used by any one of
//! them.

use mlx_gen_krea::{Krea2Config, KreaTeConfig};

// --- DiT fixtures: `tools/dump_krea_dit_golden.py` -----------------------------------------------

/// DiT attention heads.
pub const SHARED_FIXTURE_DIT_HEADS: i32 = 4;
/// DiT key/value heads — GQA, strictly fewer than [`SHARED_FIXTURE_DIT_HEADS`].
pub const SHARED_FIXTURE_DIT_KV_HEADS: i32 = 2;
/// DiT per-head width. `mmdit` derives the 3-axis RoPE dims `[8, 12, 12]` from this 32.
pub const SHARED_FIXTURE_DIT_HEAD_DIM: i32 = 32;
/// DiT residual width.
pub const SHARED_FIXTURE_DIT_HIDDEN: i32 = 128;
/// Text-fusion attention heads (both the layerwise and the refiner blocks).
pub const SHARED_FIXTURE_DIT_TXT_HEADS: i32 = 2;
/// Norm epsilon shared by the block, text-fusion and full-DiT fixtures.
pub const SHARED_FIXTURE_DIT_EPS: f32 = 1e-5;
/// DiT latent channels.
pub const SHARED_FIXTURE_DIT_IN_CHANNELS: usize = 16;
/// DiT patch size.
pub const SHARED_FIXTURE_DIT_PATCH_SIZE: usize = 2;
/// Single-stream depth.
pub const SHARED_FIXTURE_DIT_NUM_LAYERS: usize = 2;
/// SwiGLU inner width — documentary: the loader reads the real inner dims off the weights.
pub const SHARED_FIXTURE_DIT_INTERMEDIATE_SIZE: usize = 384;
/// The 3-axis interleaved RoPE dims, derived by `mmdit` from head_dim 32.
pub const SHARED_FIXTURE_DIT_AXES_DIMS_ROPE: [usize; 3] = [8, 12, 12];
/// RoPE base, fixed at 1000 by the dump.
pub const SHARED_FIXTURE_DIT_ROPE_THETA: f32 = 1000.0;
/// Timestep-embedding width.
pub const SHARED_FIXTURE_DIT_TIMESTEP_EMBED_DIM: usize = 64;
/// Stacked text-encoder layers the fusion aggregates over.
pub const SHARED_FIXTURE_DIT_NUM_TEXT_LAYERS: usize = 3;
/// Layer-axis text blocks.
pub const SHARED_FIXTURE_DIT_NUM_LAYERWISE_TEXT_BLOCKS: usize = 2;
/// Token-axis refiner text blocks.
pub const SHARED_FIXTURE_DIT_NUM_REFINER_TEXT_BLOCKS: usize = 2;
/// Text-fusion residual width.
pub const SHARED_FIXTURE_DIT_TEXT_HIDDEN_DIM: usize = 64;
/// Text-fusion feed-forward width.
pub const SHARED_FIXTURE_DIT_TEXT_INTERMEDIATE_SIZE: usize = 256;
/// Text-fusion attention heads (the `Krea2Config` field).
pub const SHARED_FIXTURE_DIT_TEXT_NUM_ATTENTION_HEADS: usize = 2;
/// Text-fusion key/value heads.
pub const SHARED_FIXTURE_DIT_TEXT_NUM_KV_HEADS: usize = 2;

/// Caption length the `rope_golden` table was built at (`meta = [n_tok, ht, wt, ax0, ax1, ax2]`).
pub const SHARED_FIXTURE_ROPE_CAP_LEN: usize = 5;
/// Latent-grid height the `rope_golden` table was built at.
pub const SHARED_FIXTURE_ROPE_GRID_H: usize = 4;
/// Latent-grid width the `rope_golden` table was built at.
pub const SHARED_FIXTURE_ROPE_GRID_W: usize = 4;

// --- Text-encoder fixture: `tools/dump_krea_te_golden.py` ----------------------------------------

/// TE residual width. `candle-gen-krea`'s `KreaTeConfig` has no `hidden_size` field — the loader
/// takes it as an argument — so this constant is what the two lanes agree on.
pub const SHARED_FIXTURE_TE_HIDDEN_SIZE: i32 = 64;
/// TE decoder depth.
pub const SHARED_FIXTURE_TE_NUM_LAYERS: i32 = 6;
/// TE attention heads.
pub const SHARED_FIXTURE_TE_NUM_HEADS: i32 = 4;
/// TE key/value heads (GQA).
pub const SHARED_FIXTURE_TE_NUM_KV_HEADS: i32 = 2;
/// TE per-head width — decoupled from `hidden_size`, as in the shipped Qwen3-VL-4B.
pub const SHARED_FIXTURE_TE_HEAD_DIM: i32 = 32;
/// TE RMSNorm epsilon.
pub const SHARED_FIXTURE_TE_RMS_NORM_EPS: f32 = 1e-6;
/// TE RoPE base.
pub const SHARED_FIXTURE_TE_ROPE_THETA: f32 = 5_000_000.0;
/// The hidden-state layers stacked into the DiT's `context`.
pub const SHARED_FIXTURE_TE_SELECT_HIDDEN: [usize; 2] = [2, 4];
/// Template-prefix tokens dropped off the front of the stacked context.
pub const SHARED_FIXTURE_TE_PREFIX_TOKENS: usize = 3;
/// The `<|image_pad|>` id, the shipped Qwen3-VL value.
pub const SHARED_FIXTURE_TE_IMAGE_TOKEN_ID: i32 = 151_655;
/// The production per-axis (T/H/W) MRoPE frequency counts.
///
/// `mrope_section` is read only on the image-grounded path (`forward_with_images`); the text-only
/// [`KreaTextEncoder::forward`] both lanes' `te_golden` tests drive builds a plain 1-D RoPE table and
/// never looks at it. This lane used to declare a reduced `[16, 0, 0]` here, saturated to head_dim/2,
/// under a comment explaining that text-only MRoPE collapses to the T axis — true, but a second
/// spelling of a production constant that nothing held equal to the candle lane's. Both lanes now
/// carry the shipped `[24, 20, 20]`, which at head_dim 32 puts the same 16 frequencies on the same
/// axis, so no test's behaviour changes.
pub const SHARED_FIXTURE_TE_MROPE_SECTION: [i32; 3] = [24, 20, 20];

/// The TE fixture's feed-forward width. `candle-gen-krea` derives this from the loaded weights rather
/// than declaring it, so it is not a shared claim and carries no `SHARED_FIXTURE_` name — a one-sided
/// `SHARED_FIXTURE_*` declaration is itself a `check_cross_backend_geometry` failure.
const TE_INTERMEDIATE_SIZE: i32 = 128;

/// Tiny DiT config matching `tools/dump_krea_dit_golden.py::dump_dit`.
///
/// Built from the `SHARED_FIXTURE_DIT_*` constants, which `check_cross_backend_geometry` holds equal
/// to the candle lane's — the two lanes load byte-identical fixture bytes and must agree about their
/// shape before either compares a tensor.
pub fn tiny_dit_config() -> Krea2Config {
    Krea2Config {
        in_channels: SHARED_FIXTURE_DIT_IN_CHANNELS,
        patch_size: SHARED_FIXTURE_DIT_PATCH_SIZE,
        hidden_size: SHARED_FIXTURE_DIT_HIDDEN as usize,
        num_attention_heads: SHARED_FIXTURE_DIT_HEADS as usize,
        num_kv_heads: SHARED_FIXTURE_DIT_KV_HEADS as usize,
        attention_head_dim: SHARED_FIXTURE_DIT_HEAD_DIM as usize,
        num_layers: SHARED_FIXTURE_DIT_NUM_LAYERS,
        intermediate_size: SHARED_FIXTURE_DIT_INTERMEDIATE_SIZE,
        norm_eps: SHARED_FIXTURE_DIT_EPS,
        axes_dims_rope: SHARED_FIXTURE_DIT_AXES_DIMS_ROPE,
        rope_theta: SHARED_FIXTURE_DIT_ROPE_THETA,
        timestep_embed_dim: SHARED_FIXTURE_DIT_TIMESTEP_EMBED_DIM,
        num_text_layers: SHARED_FIXTURE_DIT_NUM_TEXT_LAYERS,
        num_layerwise_text_blocks: SHARED_FIXTURE_DIT_NUM_LAYERWISE_TEXT_BLOCKS,
        num_refiner_text_blocks: SHARED_FIXTURE_DIT_NUM_REFINER_TEXT_BLOCKS,
        text_hidden_dim: SHARED_FIXTURE_DIT_TEXT_HIDDEN_DIM,
        text_intermediate_size: SHARED_FIXTURE_DIT_TEXT_INTERMEDIATE_SIZE,
        text_num_attention_heads: SHARED_FIXTURE_DIT_TEXT_NUM_ATTENTION_HEADS,
        text_num_kv_heads: SHARED_FIXTURE_DIT_TEXT_NUM_KV_HEADS,
    }
}

/// Tiny TE config matching `tools/dump_krea_te_golden.py`.
pub fn tiny_te_config() -> KreaTeConfig {
    KreaTeConfig {
        hidden_size: SHARED_FIXTURE_TE_HIDDEN_SIZE,
        num_layers: SHARED_FIXTURE_TE_NUM_LAYERS,
        num_heads: SHARED_FIXTURE_TE_NUM_HEADS,
        num_kv_heads: SHARED_FIXTURE_TE_NUM_KV_HEADS,
        head_dim: SHARED_FIXTURE_TE_HEAD_DIM,
        intermediate_size: TE_INTERMEDIATE_SIZE,
        rms_norm_eps: SHARED_FIXTURE_TE_RMS_NORM_EPS,
        rope_theta: SHARED_FIXTURE_TE_ROPE_THETA,
        select_hidden: SHARED_FIXTURE_TE_SELECT_HIDDEN.to_vec(),
        prefix_tokens: SHARED_FIXTURE_TE_PREFIX_TOKENS,
        image_token_id: SHARED_FIXTURE_TE_IMAGE_TOKEN_ID,
        mrope_section: SHARED_FIXTURE_TE_MROPE_SECTION,
    }
}
