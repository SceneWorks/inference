//! # mlx-gen-prompt-refine
//!
//! A text-in / text-out **instruction-LLM** provider for [`mlx-gen`](mlx_gen) — the native MLX
//! (Apple Silicon) implementation of the gen-core [`TextLlm`](mlx_gen::gen_core::TextLlm) contract
//! (sc-5552, the MLX twin of `candle-gen-prompt-refine`). It runs **Llama-3.2-3B-Instruct** (the
//! abliterated prompt-refine checkpoint by default) to rewrite a user's prompt, replacing the
//! worker's Python `prompt_refine.py` (`AutoModelForCausalLM` + `model.generate`).
//!
//! MLX has no shared Llama decoder (candle gets one from `candle_transformers`), so this crate
//! vendors a small **config-driven** Llama-3.2 decoder ([`llama`]) modeled on the parity-proven
//! JoyCaption MLX path (`mlx_gen::caption::joycaption::language`): it reads the snapshot's
//! `config.json` (dims, GQA, the Llama-3 `rope_scaling` block) and handles **tied embeddings**
//! (Llama-3.2-1B/3B share the LM head with `embed_tokens`, so there is no separate `lm_head.weight`).
//! The Llama-3 chat template is hand-assembled in [`prompt`] (the `tokenizers` crate applies no
//! template), and decoding is a standard KV-cached greedy/top-p sampling loop.
//!
//! **Generic by contract.** This is a plain instruction LLM: the caller supplies the `system` message
//! (the prompt-rewrite rules + the model's prompt guide) and the `user` prompt, and gets back the raw
//! model text. The product-specific prompt assembly and any output cleanup (`<think>` stripping,
//! fence/quote trimming) live at the caller's edge — the worker — keeping this a reusable seam.
//! Registered as id `"prompt_refine"`, `backend = "mlx"`, `mac_only = true`.

pub mod llama;
pub mod model;
pub mod prompt;

pub use model::{descriptor, force_link, load, PromptRefiner};
