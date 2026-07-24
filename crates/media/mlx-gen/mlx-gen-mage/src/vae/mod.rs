//! Mage-VAE — the one-step 128-channel / 16× codec — **owned by sc-14039**.
//!
//! Port of `_vendor/mage_flow/models/modules/mage_vae.py`. Add submodules under this directory
//! (encoder / decoder / shared blocks); nothing outside it needs to change, which keeps this story
//! parallel with the text-encoder and DiT ports.
//!
//! ## Shape
//!
//! [`LATENT_CHANNELS`](crate::config::LATENT_CHANNELS) 128,
//! [`VAE_DOWNSAMPLE_FACTOR`](crate::config::VAE_DOWNSAMPLE_FACTOR) 16, fully convolutional, no
//! global attention. State-dict prefixes: `student.dconv_encoder.*` for the encoder,
//! `dconv_denoiser` / `y_embedder.decoder` for the decoder.
//!
//! ## Encode is a single forward at t = 0
//!
//! `z_t = zeros[B, z_ch, H/16, W/16]`, `t = zeros`, `out = dconv_encoder.forward_pred(z_t, t, x)`;
//! `mean = out[:, :128]`, `logvar = out[:, 128:256].clamp(-20, 10)` (`:598-623`). Because the adaLN
//! modulation depends only on `t` and `t` is always 0, the reference precomputes it and frees the
//! adaLN MLPs (`_replace_adaln_with_const`, `:343-374`) — worth replicating, ~37M params.
//!
//! ## Decode
//!
//! One-step `_DConvDenoiser` + a CoD pixel-diffusion decoder → RGB in `[-1, 1]` (`:626-633`).
//!
//! ## Two traps
//!
//! 1. **No latent scale or shift** ([`LATENT_SCALE_SHIFT`](crate::config::LATENT_SCALE_SHIFT) is
//!    `None`): latents feed `img_in = Linear(128 → 3072)` raw. There is no `scaling_factor` /
//!    `shift_factor` / `latents_mean` / `latents_std` anywhere in the reference or the published
//!    configs. Do not import a FLUX/SD-style constant.
//! 2. **The published `vae/config.json sample_posterior: false` is not what runs.**
//!    `ModelConfig.vae_sample_posterior` defaults to `true` (`models/mage_flow.py:35`) and
//!    `load_from_repo` never overrides it, so the edit path *samples* the posterior, seeded off the
//!    global RNG (`pipeline.py:499`). The golden therefore gates the moments (`enc_mean` /
//!    `enc_logvar`) and the port applies its own RNG. Watermark detection uses the mean.
//!
//! ## Weight loading
//!
//! There is no shared `loader.rs` (see the decision note in `lib.rs`): put this component's `load`
//! **inside this directory**, so the concurrent text-encoder and DiT ports never touch a file this
//! story owns.
