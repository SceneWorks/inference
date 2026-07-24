//! One dual-stream NR-MMDiT block — **owned by sc-14040**.
//!
//! Port of `MageFlowTransformerBlock` (`_vendor/mage_flow/models/modules/mage_layers.py:515-667`).
//! All [`MageFlowConfig::depth`](crate::config::MageFlowConfig::depth) blocks are dual-stream:
//! [`crate::config::DEPTH_SINGLE_BLOCKS`] is 0, so there is no single-stream tail (unlike FLUX).
//!
//! Per stream: adaLN modulation `SiLU → Linear(dim, 6·dim)` producing
//! `(shift, scale, gate) × 2`, over `LayerNorm(elementwise_affine=False, eps=1e-6)`; then the joint
//! attention ([`crate::attention`]) and the gelu-approximate FFN ([`crate::feed_forward`]), each
//! with a gated residual.
//!
//! **Trap:** the modulation broadcast uses `repeat_interleave` with an int32 `repeats` tensor
//! (`:566`) — the op that makes the reference unrunnable on torch MPS. It only fires when the pack
//! carries ≥2 segments, which is why a `cfg <= 1` MPS run completes and silently produces
//! garbage-adjacent output. Irrelevant to MLX numerics, but it explains why the goldens are
//! CPU-dumped and why `MAGE_DEVICE=cpu` is mandatory when regenerating them.
