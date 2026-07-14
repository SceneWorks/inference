//! # mlx-gen-mochi
//!
//! Native-Rust / MLX inference edges for **Mochi 1** (`genmo/mochi-1-preview`, Apache-2.0) — a
//! T5-XXL-conditioned MMDiT text-to-video model with an asymmetric 3-D causal-conv VAE (6× temporal,
//! 8× spatial). This crate (story A2) scaffolds the model's **I/O edges**:
//!
//!  - the **text encoder** — the reused [`mlx_gen_flux::T5TextEncoder`] run *with* the tokenizer
//!    padding mask (Mochi's `_get_t5_prompt_embeds`), plus the vendored t5-v1.1-xxl tokenizer;
//!  - the **AsymmVAE decoder** — a faithful port of `AutoencoderKLMochi`'s decode path (attention-free
//!    mid-blocks, `MochiUpBlock3D` depth-to-space unpatchify, `CogVideoXCausalConv3d` replicate-pad,
//!    per-frame chunked GroupNorm), gated by the A1 real-weight goldens.
//!
//! The DiT transformer itself lands in a later story (A3/A4); this crate deliberately stops at the
//! edges so each component is parity-gated in isolation against the A1 goldens.

pub mod config;
pub mod text_encoder;
pub mod tokenizer;

pub use config::{MochiConfig, MochiVaeConfig};
pub use text_encoder::{encode_prompt, load_t5_encoder, MochiTextConditioning};
pub use tokenizer::{load_tokenizer, load_tokenizer_with_max_len};
