//! # mlx-gen-lens
//!
//! The **Microsoft Lens / Lens-Turbo** provider crate for [`mlx-gen`](mlx_gen) — a native-MLX port
//! of the turbo-distilled text-to-image model that today runs only in a transformers-5 Python
//! sidecar venv (`/opt/lens-venv`). See epic 3164.
//!
//! Lens-Turbo is three components:
//!
//! 1. **Text encoder — gpt-oss-20b** (`GptOssForCausalLM`, subclassed `LensGptOssEncoder`): a
//!    24-layer **MoE** LLM (hidden 2880, 64 query / 8 KV heads, 32 experts top-4) with **attention
//!    sinks**, **alternating sliding/full attention**, **YaRN RoPE**, and clamped-SwiGLU experts.
//!    Used *encoder-only* — forward to layer 23, capture hidden states at layers `[5, 11, 17, 23]`,
//!    no LM head / KV cache / generation. The 32 expert stacks are MXFP4 in the checkpoint; the
//!    attention/router/embedding modules stay dense bf16 (`modules_to_not_convert`).
//! 2. **Denoising DiT — 48-layer dual-stream MMDiT** (`LensTransformer2DModel`): a near-twin of the
//!    `mlx-gen-qwen-image` MMDiT with a multi-layer text-feature front-end.
//! 3. **VAE — `AutoencoderKLFlux2`** — already ported in [`mlx_gen_flux2`]; only a thin Lens decode
//!    shim is new.
//!
//! ## Status
//!
//! Under construction (epic 3164). This slice (**sc-3165**) ships the crate scaffold, the [`config`]
//! parser for the gpt-oss text encoder, and the gpt-oss **attention core** ([`text_encoder::gpt_oss`])
//! — GQA + learned attention sinks + alternating sliding/full causal masks + YaRN RoPE + RMSNorm,
//! validated single-layer against the reference `transformers.models.gpt_oss` forward. The MoE
//! feed-forward + full decoder-layer assembly (sc-3166), the multi-layer hidden capture (sc-3171),
//! the weight conversion / quant (sc-3172), the DiT (sc-3168), VAE shim (sc-3169), scheduler
//! (sc-3170), and the generate/e2e integration (sc-3173) land in the following stories.

pub mod config;
pub mod text_encoder;
