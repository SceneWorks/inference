use mlx_gen::Result;
use mlx_gen_flux::{T5Sublayer, T5TextEncoder};

/// The production-length SC-16462 sensitivity sweep selected the final block's FFN as the smallest
/// source-precision carve-out that best preserves Chroma's active T5 conditioning across the two
/// render prompts and the empty true-CFG prompt.
pub const DENSE_FFN_BLOCK: usize = 23;

const DENSE_FFN_PREFIX: &str = "encoder.block.23.layer.1.DenseReluDense.";

/// Apply the same T5 policy represented by [`should_pack_linear`] to a dense in-memory encoder.
pub(crate) fn quantize_linears(t5: &mut T5TextEncoder, bits: i32) -> Result<()> {
    t5.quantize_linears_except(bits, DENSE_FFN_BLOCK, T5Sublayer::FeedForward)
}

/// Return whether a T5 weight base belongs in the packed artifact.
pub(crate) fn should_pack_linear(base: &str) -> bool {
    base != "shared"
        && !base.ends_with("SelfAttention.relative_attention_bias")
        && !base.starts_with(DENSE_FFN_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_ffn_prefix_tracks_the_selected_block() {
        assert_eq!(
            DENSE_FFN_PREFIX,
            format!("encoder.block.{DENSE_FFN_BLOCK}.layer.1.DenseReluDense.")
        );
    }
}
