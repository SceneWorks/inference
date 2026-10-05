//! The shared native **training** harness (epic 5164 / sc-5165) — the candle twin of
//! `mlx_gen::train`. The family provider crates (sdxl/z-image/wan/lens) implement the backend-neutral
//! [`gen_core::Trainer`](crate::gen_core::train::Trainer) on top of these primitives, mirroring how
//! the MLX `*-gen-{family}/training.rs` twins build on the shared MLX harness.
//!
//! The pieces, and the candle realities that shape them:
//!
//! - [`lora`] — the trainable adapter seam. candle is **eager** and candle-transformers' `nn::Linear`
//!   holds a frozen `Tensor` with no hook, so a LoRA residual cannot be a precomputed weight delta
//!   (it would materialize once and never see optimizer updates). Instead [`lora::LoraLinear`] wraps a
//!   frozen base `Linear` and adds the residual *in the forward*, with the adapter factors held as
//!   `Var`-backed tensors (storage-shared clones) — so `Var::set` from the optimizer is seen by the
//!   next forward with no re-install, and `loss.backward()` attributes grads to the underlying `Var`s.
//!   Adapter factors / loss / grads / optimizer state stay f32 (master-weights); only the frozen base
//!   and the activation stream follow the train dtype.
//!
//! - [`dataset`] — resolution bucketing + image → VAE-input tensor (decode/crop/resize/normalize).
//! - [`schedule`] — the gen-core LR schedule (constant/linear/cosine + warmup), re-exported.
//! - [`optim`] — the full optimizer set (`adamw`/`adam`/`rose`/`prodigy`) stepping factor `Var`s from
//!   a `GradStore`, plus grad-norm clipping (candle ships neither Rose/Prodigy nor clipping).
//! - [`gradient_checkpoint`] — manual segmented vector-Jacobian product, the candle stand-in for
//!   `mx.checkpoint` (candle has no activation-checkpointing primitive). Trades one extra forward for
//!   `O(boundary)` instead of `O(all activations)` peak memory — all but required to fit SDXL LoRA
//!   training on smaller cards.
//! - [`checkpoint`] — intermediate-adapter file naming (the *file* kind of checkpoint, distinct from
//!   [`gradient_checkpoint`]).
//!
//! The per-family training loop (cache latents/embeddings → noised forward → loss → backward → clip →
//! step) lives in each provider crate's `Trainer`, built from these primitives.
pub mod checkpoint;
pub mod dataset;
pub mod flow_match;
// Deterministic formula weights for the sc-24833 latent-perceptual parity tests.
pub mod formula;
pub mod gradient_checkpoint;
pub mod lora;
/// The shared inference-side adapter-merge skeleton (sc-8998 / F-018): the format-parsing +
/// merge-report machinery the provider crates' `adapters.rs` hand-copied around the [`lora`] delta
/// reconstruction. Family-specific key→module resolution stays in each crate.
pub mod merge;
pub mod optim;
// The shared decoded-x0 perceptual auxiliary-loss path (epic 2123 E8) — Candle twin of
// `mlx_gen::train::perceptual`.
pub mod perceptual;
pub mod schedule;
// The sc-24833 latent-space perceptual losses (E-LatentLPIPS, FLUX.2 VAE anchor).
pub mod latent_lpips;
pub mod vae_anchor;
// TAESD-family tiny differentiable x0 decoders for the perceptual path.
pub mod tae;
