//! Concrete model decoders built on the [`crate::primitives`].
//!
//! The generic Llama decoder uses an **immutable `&self` forward** and a `from_weights`
//! constructor, so a single loaded model can be shared and driven concurrently in the batch
//! dimension later. The only mutable state in a forward pass is the KV cache, threaded in as
//! `&mut dyn KvCache`.

pub(crate) mod deepstack;
pub mod llama;
pub mod qwen35;
pub mod qwen35_vision;
pub mod siglip;

use candle_core::Tensor;

use crate::decode::stream::Decode;
use crate::error::Result;
use crate::primitives::kv_cache::KvCache;

pub use deepstack::MropePositions;
pub use llama::{shard_plan, CausalLm};
pub use qwen35::{Qwen35Cache, Qwen35Config, Qwen35Model};
pub use qwen35_vision::{Qwen35VisionConfig, Qwen35VisionModel, Qwen35VisionOutput};
pub use siglip::{
    select_vision_feature, SiglipVisionConfig, SiglipVisionOutput, SiglipVisionTower,
};

/// The backend-neutral multimodal seam over a loaded decoder. Both Qwen-VL backbones — the Qwen3.6
/// hybrid linear/full-attention decoder ([`Qwen35Model`]) and the generic Qwen3-VL causal decoder
/// ([`CausalLm`]) — implement it, so the provider drives the image/video prefill + decode through
/// one trait object (`as_vlm`) rather than forking on the concrete decoder type (sc-8080, mirroring
/// mlx-llm's sc-8314 `VlmDecode` refactor). The [`Decode`] supertrait provides the post-prompt
/// continuation step (each decoder downcasts its own cache inside `step`).
pub trait VlmDecode: Decode {
    /// Embed token ids `[1, S]` → `[1, S, hidden]` in the compute dtype — the splice point where the
    /// multimodal path overwrites placeholder rows with the vision tower's merged patch features.
    fn embed_input_ids(&self, input_ids: &Tensor) -> Result<Tensor>;

    /// Replace every row of `embeds` `[1, S, hidden]` whose id is any of `placeholder_tokens`
    /// (`<|image_pad|>` / `<|video_pad|>`) with the next `vision_features` `[num_vision_tokens,
    /// hidden]` row, in sequence order — the mixed image+video splice.
    fn splice_vision_features(
        &self,
        embeds: &Tensor,
        input_ids: &[i32],
        vision_features: &Tensor,
        placeholder_tokens: &[i32],
    ) -> Result<Tensor>;

    /// Interleaved-M-RoPE 3-D position rows (temporal/height/width) + the `mrope_delta`, computed
    /// over the image **and** video grids (the `get_rope_index` port).
    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions>;

    /// Prefill precomputed `embeds` `[1, S, hidden]` (with vision features spliced in) using
    /// interleaved M-RoPE from the explicit 3-D `positions` **and** DeepStack feature fusion: after
    /// decoder layer `i` (for `i < deepstack.len()`) the `i`-th tapped/merged ViT feature set is
    /// added to the `visual_pos_mask` rows. Returns last-position logits `[1, vocab]`.
    fn prefill_with_deepstack(
        &self,
        embeds: &Tensor,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Tensor],
    ) -> Result<Tensor>;
}
