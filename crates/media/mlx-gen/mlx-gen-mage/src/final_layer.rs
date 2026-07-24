//! Output head — **owned by sc-14040**.
//!
//! `AdaLayerNormContinuous(hidden_size, hidden_size, elementwise_affine=False, eps=1e-6)` followed
//! by `proj_out = Linear(hidden_size → patch_size² · out_channels, bias=True)`
//! (`_vendor/mage_flow/models/mage_flow.py:90-91`, `:147-152`;
//! `models/modules/mage_layers.py:668-724`).
//!
//! `AdaLayerNormContinuous` is `SiLU → Linear → (scale, shift)` applied to a non-affine LayerNorm.
//! Only the **image** stream reaches the head; the text stream is dropped after the last block.
//! With `patch_size == 1` there is no unpatchify — the head emits one 128-channel latent cell per
//! token, and `unpack` (`models/utils.py:36`) only reshapes `(h w) c → c h w` at
//! `ceil(height/16) × ceil(width/16)`.
