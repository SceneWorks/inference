//! # mlx-gen-anima
//!
//! The **Anima** provider crate for [`mlx-gen`](mlx_gen) — circlestone-labs' anime text-to-image model
//! (epic 10512). Three variants share **one architecture** and differ only in the DiT weights file:
//! - **`anima_base`** — the base model (30 steps, CFG 4.5),
//! - **`anima_aesthetic`** — the aesthetic fine-tune (30 steps, CFG 4.5),
//! - **`anima_turbo`** — the merged CFG-free few-step student (10 steps, CFG 1.0).
//!
//! ## Architecture (verified against the real `circlestone-labs/Anima` checkpoint)
//! - **DiT** — the **Cosmos-Predict2** `CosmosTransformer3DModel` (`Cosmos-2.0-Diffusion-2B-Text2Image`):
//!   28 layers, hidden 2048 (16 heads × 128), patch `(1,2,2)`, adaLN-LoRA 256, 3-axis NTK RoPE
//!   `rope_scale (1,4,4)`, `concat_padding_mask` ⇒ **17-channel** patch-embed input
//!   ([`transformer::CosmosDiT`]). Net-new port from diffusers `transformer_cosmos.py`.
//! - **Text conditioner** — the **`AnimaTextConditioner`** (bundled in the DiT file under
//!   `net.llm_adapter.*`): `nn.Embedding(32128, 1024)` over T5 token ids as learned query tokens →
//!   6 × [self-attn → cross-attn into Qwen3 states → GELU MLP] → out_proj + RMSNorm, right-padded to
//!   **512** ([`conditioner::AnimaTextConditioner`]). Net-new port from `condition_embedder_anima.py`.
//! - **Text encoder** — **Qwen3-0.6B base** (`last_hidden_state`), reusing z-image's Qwen3 decoder
//!   block ([`text_encoder::AnimaQwen3`]).
//! - **VAE** — the **Qwen-Image** `AutoencoderKLQwenImage`, reusing `mlx_gen_qwen_image::QwenVae`
//!   ([`vae`]).
//! - **Scheduler** — `FlowMatchEulerDiscreteScheduler` static `shift=3.0`, `sigmas = linspace(1, 1/N, N)`
//!   ([`pipeline::anima_sigmas`]); default solver the recommended `er_sde` (sc-10519).
//!
//! Tokenization is dual: Qwen2 BPE (no BOS/EOS, pad 151643) for the encoder, T5 SentencePiece (with
//! EOS) for the conditioner's query tokens ([`tokenizer`]).

pub mod adapters;
pub mod conditioner;
pub mod config;
pub mod convert;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod prompt_weight;
pub mod rope;
pub mod text_encoder;
pub mod tokenizer;
pub mod training;
pub mod transformer;
pub mod vae;

pub use adapters::{apply_anima_adapters, AnimaAdapterHost};
pub use conditioner::AnimaTextConditioner;
pub use config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
pub use convert::{is_dit_quant_target, quantize_anima_dit};
pub use loader::{load_conditioning_at_dtype, split_anima_keys, AnimaComponents};
pub use model::{
    descriptor_aesthetic, descriptor_base, descriptor_turbo, load_aesthetic, load_base, load_turbo,
    Anima,
};
pub use pipeline::{anima_sigmas, AnimaPipeline, GenOptions, DEFAULT_SAMPLER};
pub use prompt_weight::{parse_prompt_weights, strip_prompt_weights};
pub use text_encoder::AnimaQwen3;
pub use training::{
    trainer_descriptor_aesthetic, trainer_descriptor_base, trainer_descriptor_turbo, AnimaTrainer,
};
pub use transformer::CosmosDiT;
pub use vae::{load_vae, QwenVae};
