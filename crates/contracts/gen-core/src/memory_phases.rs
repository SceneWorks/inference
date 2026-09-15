//! Provider-owned facts for phase-specific memory estimates. These describe execution and
//! loaded tensor bytes; they contain no fitted coefficients or claims of measured memory.

/// Components kept live by the staged pipeline after conditioning has been released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagedWeightSchedule {
    /// Denoiser and decoder share one heavy bundle, including throughout decode.
    TwoStage,
    /// The denoiser is dropped before decoding begins.
    ThreeStage,
}

/// The exact streamable portion of one component. Stacks run sequentially and each window is
/// evaluated and dropped before the next one is materialized. Everything outside the stacks
/// remains live, including embeddings, projections, normalizations and convolutional trunks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamedWeightFacts {
    pub resident_bytes: u64,
    pub stacks: Vec<Vec<u64>>,
}

impl StreamedWeightFacts {
    /// Peak component weights for the actual aligned window schedule, rather than dividing the
    /// entire component by a block count. A missing/zero window is not a streaming request.
    pub fn peak_bytes(&self, window: Option<u32>) -> u64 {
        let Some(window) = window.filter(|window| *window > 0) else {
            return self
                .stacks
                .iter()
                .flatten()
                .fold(self.resident_bytes, |total, bytes| {
                    total.saturating_add(*bytes)
                });
        };
        let largest_window = self
            .stacks
            .iter()
            .flat_map(|stack| stack.chunks(window as usize))
            .map(|chunk| {
                chunk
                    .iter()
                    .fold(0_u64, |total, bytes| total.saturating_add(*bytes))
            })
            .max()
            .unwrap_or(0);
        self.resident_bytes.saturating_add(largest_window)
    }
}

/// The decoder operation that a selected tile bounds. Layer-wise tiling retains full-resolution
/// feature maps; whole-tail tiling retains the output/blend buffers and runs the tail per crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderTilingRealization {
    LayerwiseConvolution,
    WholeTail,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderWorkspaceFacts {
    pub tiling: DecoderTilingRealization,
    /// Decoder compute width, independent of the denoiser's activation dtype.
    pub activation_dtype_width: u32,
    /// Tail channel widths, in execution order from the latent toward the output.
    pub channels: Vec<u32>,
    /// First residual input widths, including wider full-image upsample intermediates.
    pub input_channels: Vec<u32>,
    /// Output pixels per feature-map pixel on each spatial axis, corresponding to `channels`.
    pub spatial_divisors: Vec<u32>,
}

/// An executable architecture with retained phase measurements. This describes the provider's
/// constructors, not a catalog model name; fine-tunes using those constructors share the facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImagePipelineArchitecture {
    SdxlUnetWithDualClip,
    /// ZImage transformer uses fused SDPA; its live tensor workspace grows with image tokens.
    ZImageDit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPhaseFacts {
    pub architecture: Option<ImagePipelineArchitecture>,
    pub staged_weights: StagedWeightSchedule,
    /// Absent means unknown, never zero resident weights.
    pub transformer_stream: Option<StreamedWeightFacts>,
    pub decoder_workspace: Option<DecoderWorkspaceFacts>,
}

impl MemoryPhaseFacts {
    /// Declare only known phase ownership. Unknown block/decoder workspaces remain conservative.
    pub const fn staged(staged_weights: StagedWeightSchedule) -> Self {
        Self {
            architecture: None,
            staged_weights,
            transformer_stream: None,
            decoder_workspace: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unequal_stacks_keep_the_trunk_and_only_one_aligned_window() {
        let facts = StreamedWeightFacts {
            resident_bytes: 100,
            stacks: vec![vec![10, 40, 20], vec![30, 15]],
        };
        assert_eq!(facts.peak_bytes(None), 215);
        assert_eq!(facts.peak_bytes(Some(0)), 215);
        assert_eq!(facts.peak_bytes(Some(1)), 140);
        assert_eq!(facts.peak_bytes(Some(2)), 150);
        assert_eq!(facts.peak_bytes(Some(10)), 170);
    }
}
