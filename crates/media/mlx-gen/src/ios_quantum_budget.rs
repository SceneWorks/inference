//! iOS diagnostic quantum accounting (SANA/Gemma, and the generic transformer/attention/conv
//! floors Anima, Iris and SDXL price their quanta with). These are conservative tensor scratch floors,
//! not a total-process or Metal-driver upper bound. They never authorize production admission.

pub const PROCESS_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

fn product(values: &[u64]) -> u64 {
    values.iter().fold(1u64, |a, b| a.saturating_mul(*b))
}
fn sum(values: &[u64]) -> u64 {
    values.iter().fold(0u64, |a, b| a.saturating_add(*b))
}

/// Eight score-sized buffers cover scale/softcap/mask/softmax conversions; 24 hidden,
/// eight MLP and twelve QKV buffers cover norm, residual, gated activation and GQA copies.
/// Also price full f32 projection copies, even when packed kernels do not need them.
pub fn gemma_layer_bytes(b: u64, l: u64, h: u64, intermediate: u64, heads: u64, hd: u64) -> u64 {
    transformer_layer_bytes(b, l, h, intermediate, heads, hd)
}

/// One full-attention transformer layer of any text encoder (Gemma, Qwen3, CLIP, T5-style): the
/// Gemma accounting above, which already prices `L x L` scores, every hidden/MLP/QKV copy and
/// full f32 projection copies.
pub fn transformer_layer_bytes(
    b: u64,
    l: u64,
    h: u64,
    intermediate: u64,
    heads: u64,
    hd: u64,
) -> u64 {
    sum(&[
        product(&[8, b, heads, l, l, 4]),
        product(&[24, b, l, h, 4]),
        product(&[8, b, l, intermediate, 4]),
        product(&[12, b, l, heads, hd, 4]),
        product(&[4, h, heads, hd, 4]),
        product(&[3, h, intermediate, 4]),
    ])
}

/// Actual configured geometry of one SANA transformer block, derived from its loaded tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SanaBlockGeometry {
    pub batch: u64,
    pub tokens: u64,
    /// Block model width (`heads * head_dim`).
    pub dim: u64,
    pub caption_tokens: u64,
    /// GLUMBConv expanded channels (`2 * int(mlp_ratio * dim)`).
    pub ff_channels: u64,
    /// Cross-attention heads whose `[B, H, N, M]` f32 score tensor is materialized.
    pub cross_heads: u64,
    /// Elements of every GLUMBConv weight and bias (inverted, depthwise, point).
    pub conv_weight_elements: u64,
}

/// Full F32 copies of every GLUMBConv weight/bias, ADDITIONAL to the resident (possibly F16) copy.
pub fn sana_promoted_conv_bytes(conv_weight_elements: u64) -> u64 {
    product(&[4, conv_weight_elements])
}

/// Promoted convolution copies, eight attention-projection copies, eight score-sized
/// softmax buffers, 32 expanded-activation buffers and eight caption-sized buffers. Per-block
/// materialization prevents multiplying this scratch by the stack depth.
pub fn sana_block_bytes(g: SanaBlockGeometry) -> u64 {
    sum(&[
        sana_promoted_conv_bytes(g.conv_weight_elements),
        product(&[8, g.dim, g.dim, 4]),
        product(&[32, g.batch, g.tokens, g.ff_channels, 4]),
        product(&[8, g.batch, g.cross_heads, g.tokens, g.caption_tokens, 4]),
        product(&[8, g.batch, g.caption_tokens, g.dim, 4]),
    ])
}

/// Latent-sized elementwise work BETWEEN denoiser forwards (classifier-free-guidance combine and
/// the sampler update). It is lazy, so it materializes inside the NEXT guarded quantum, whose
/// scratch carries this allowance: CFG combine holds both branch predictions, their difference,
/// the guidance-scaled term and the sum (5 buffers); the sampler update holds the previous sample,
/// the model output, the scaled step, the noise and the new sample (5); plus up to 6 dtype-cast
/// copies = 16 f32 latent-sized buffers.
pub fn sampler_step_bytes(latent_elements: u64) -> u64 {
    product(&[16, latent_elements, 4])
}

/// Rotary tables built for `positions` at `head_dim`: f32 cos and sin plus their broadcast copies.
pub fn rope_table_bytes(positions: u64, head_dim: u64) -> u64 {
    product(&[4, positions, head_dim, 4])
}

/// `copies` f32 buffers of `elements` each: a small elementwise pass (mask multiply, positional
/// add, concatenation) folded into the guarded quantum that materializes it.
pub fn elementwise_bytes(elements: u64, copies: u64) -> u64 {
    product(&[copies, elements, 4])
}

/// A DC-AE substage: 32 full activation-sized f32 buffers, and a conservative 64*C*C
/// f32 weight/copy allowance covering GLUMBConv (12*C*C), QKV, multiscale and conv scratch.
/// Upsample callers price the DOUBLED spatial extent before constructing that graph.
pub fn dcae_stage_bytes(b: u64, h: u64, w: u64, c: u64) -> u64 {
    conv_stage_bytes(b, h, w, c)
}

/// Any convolutional VAE/UNet substage (resnet, attention-in-conv, up/downsample) at its OUTPUT
/// spatial extent and widest channel count: the DC-AE accounting above. 32 activation-sized f32
/// buffers cover group-norm/activation/conv/residual copies; 64*C*C covers promoted 3x3 weights.
pub fn conv_stage_bytes(b: u64, h: u64, w: u64, c: u64) -> u64 {
    sum(&[product(&[32, b, h, w, c, 4]), product(&[64, c, c, 4])])
}

/// Actual geometry of one attention block of a DiT or UNet transformer, from loaded tensors and the
/// request (never production constants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBlockGeometry {
    pub batch: u64,
    /// Image/latent tokens attended over.
    pub tokens: u64,
    /// Block model width.
    pub dim: u64,
    /// Attention heads whose `[B, H, rows, keys]` f32 scores may be materialized.
    pub heads: u64,
    /// Cross-attention context tokens (0 for a self-attention-only block).
    pub context_tokens: u64,
    /// MLP hidden width.
    pub ff_hidden: u64,
    /// Query rows scored at once: `tokens` unchunked, the chunk size under a bounded-attention plan.
    pub query_rows: u64,
    /// Elements of every weight in the block, priced as full f32 copies (dequantize/promotion).
    pub weight_elements: u64,
}

/// Promoted f32 weight copies, 24 hidden-sized buffers (norm/modulation/QKV/residual), eight MLP
/// buffers, eight score-sized buffers per query chunk over self AND cross keys, and eight
/// context-sized projections. A floor for one materialized block, not a process bound.
pub fn attention_block_bytes(g: AttentionBlockGeometry) -> u64 {
    sum(&[
        product(&[4, g.weight_elements]),
        product(&[24, g.batch, g.tokens, g.dim, 4]),
        product(&[8, g.batch, g.tokens, g.ff_hidden, 4]),
        product(&[
            8,
            g.batch,
            g.heads,
            g.query_rows,
            g.tokens.saturating_add(g.context_tokens),
            4,
        ]),
        product(&[8, g.batch, g.context_tokens, g.dim, 4]),
    ])
}
pub fn check_headroom(
    available: Option<u64>,
    scratch: u64,
    reserve: u64,
) -> Result<(), &'static str> {
    let required = scratch
        .checked_add(reserve)
        .ok_or("iOS quantum accounting overflow")?;
    match available {
        Some(bytes) if bytes >= required => Ok(()),
        Some(_) => Err("iOS quantum refused before construction: insufficient current OS headroom"),
        None => Err("iOS quantum refused before construction: OS headroom unavailable"),
    }
}
pub fn guarded<T, E: From<&'static str>>(
    available: Option<u64>,
    scratch: u64,
    reserve: u64,
    build: impl FnOnce() -> Result<T, E>,
    materialize: impl FnOnce(&T) -> Result<(), E>,
    release: impl FnOnce(),
) -> Result<T, E> {
    check_headroom(available, scratch, reserve).map_err(E::from)?;
    let value = build()?;
    materialize(&value)?;
    release();
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn unavailable_insufficient_and_overflow_refuse_before_construction() {
        for available in [None, Some(0), Some(99)] {
            let constructed = Cell::new(false);
            let result = guarded::<(), &'static str>(
                available,
                64,
                36,
                || {
                    constructed.set(true);
                    Ok(())
                },
                |_| Ok(()),
                || {},
            );
            assert!(result.is_err());
            assert!(!constructed.get());
        }
        assert!(check_headroom(Some(u64::MAX), u64::MAX, 1).is_err());
        assert!(check_headroom(Some(100), 64, 36).is_ok());
    }

    #[test]
    fn layer_barrier_and_release_precede_the_next_layer() {
        let lazy_graph_depth = Cell::new(0);
        let releases = Cell::new(0);
        for _ in 0..26 {
            guarded::<(), &'static str>(
                Some(100),
                64,
                36,
                || {
                    lazy_graph_depth.set(lazy_graph_depth.get() + 1);
                    Ok(())
                },
                |_| {
                    assert_eq!(lazy_graph_depth.get(), 1);
                    lazy_graph_depth.set(0);
                    Ok(())
                },
                || {
                    assert_eq!(lazy_graph_depth.get(), 0);
                    releases.set(releases.get() + 1);
                },
            )
            .unwrap();
        }
        assert_eq!(releases.get(), 26);
        assert_eq!(lazy_graph_depth.get(), 0);
    }

    #[test]
    fn failed_materialization_never_returns_an_unevaluated_result() {
        let result = guarded::<(), &'static str>(
            Some(100),
            64,
            36,
            || Ok(()),
            |_| Err("eval failed"),
            || {},
        );
        assert_eq!(result, Err("eval failed"));
    }

    fn sana_geometry(dim: u64, ratio_num: u64, ratio_den: u64, heads: u64) -> SanaBlockGeometry {
        let hidden = dim * ratio_num / ratio_den;
        SanaBlockGeometry {
            batch: 1,
            tokens: 64,
            dim,
            caption_tokens: 300,
            ff_channels: 2 * hidden,
            cross_heads: heads,
            conv_weight_elements: 2 * hidden * dim
                + 2 * hidden
                + 2 * hidden * 9
                + 2 * hidden
                + dim * hidden,
        }
    }

    #[test]
    fn f16_convolution_promotion_is_charged_as_full_f32_scratch_not_half_storage() {
        let production = sana_geometry(2240, 5, 2, 20);
        assert_eq!(
            sana_promoted_conv_bytes(production.conv_weight_elements),
            4 * production.conv_weight_elements
        );
        assert!(sana_block_bytes(production) >= 4 * production.conv_weight_elements);
        let larger = SanaBlockGeometry {
            tokens: 1024,
            ..production
        };
        assert!(sana_block_bytes(larger) > sana_block_bytes(production));
    }

    #[test]
    fn sana_floor_uses_configured_ffn_expansion_and_cross_heads() {
        // Non-production geometry: mlp_ratio 3 and seven heads. A formula hardcoding 5*dim
        // or 20 heads cannot satisfy both exact deltas.
        let base = sana_geometry(96, 3, 1, 7);
        let more_heads = SanaBlockGeometry {
            cross_heads: 8,
            ..base
        };
        assert_eq!(
            sana_block_bytes(more_heads) - sana_block_bytes(base),
            8 * base.tokens * base.caption_tokens * 4
        );
        let wider_ffn = SanaBlockGeometry {
            ff_channels: base.ff_channels + 2,
            ..base
        };
        assert_eq!(
            sana_block_bytes(wider_ffn) - sana_block_bytes(base),
            32 * base.tokens * 2 * 4
        );
        let expected = 4 * base.conv_weight_elements
            + 8 * 96 * 96 * 4
            + 32 * 64 * 576 * 4
            + 8 * 7 * 64 * 300 * 4
            + 8 * 300 * 96 * 4;
        assert_eq!(sana_block_bytes(base), expected);
    }

    #[test]
    fn gemma_scores_use_padded_tokens_and_do_not_shrink_with_image_geometry() {
        let floor = gemma_layer_bytes(1, 506, 2304, 9216, 8, 256);
        assert!(floor >= 8 * 506 * 506 * 4 + 506 * 9216 * 4);
        assert!(gemma_layer_bytes(1, 507, 2304, 9216, 8, 256) > floor);
        assert!(gemma_layer_bytes(u64::MAX, 506, 2304, 9216, 8, 256) == u64::MAX);
    }

    #[test]
    fn attention_floor_prices_chunked_scores_context_and_weights_exactly() {
        let base = AttentionBlockGeometry {
            batch: 1,
            tokens: 64,
            dim: 32,
            heads: 4,
            context_tokens: 7,
            ff_hidden: 96,
            query_rows: 64,
            weight_elements: 1000,
        };
        let expected =
            4 * 1000 + 24 * 64 * 32 * 4 + 8 * 64 * 96 * 4 + 8 * 4 * 64 * 71 * 4 + 8 * 7 * 32 * 4;
        assert_eq!(attention_block_bytes(base), expected);
        let chunked = AttentionBlockGeometry {
            query_rows: 16,
            ..base
        };
        assert_eq!(
            attention_block_bytes(base) - attention_block_bytes(chunked),
            8 * 4 * (64 - 16) * 71 * 4
        );
        let bigger = AttentionBlockGeometry {
            tokens: 4096,
            query_rows: 4096,
            ..base
        };
        assert!(attention_block_bytes(bigger) > attention_block_bytes(base));
        assert_eq!(
            attention_block_bytes(AttentionBlockGeometry {
                batch: u64::MAX,
                ..base
            }),
            u64::MAX
        );
        assert_eq!(
            transformer_layer_bytes(1, 77, 768, 3072, 12, 64),
            gemma_layer_bytes(1, 77, 768, 3072, 12, 64)
        );
        assert_eq!(conv_stage_bytes(1, 8, 8, 4), dcae_stage_bytes(1, 8, 8, 4));
    }

    #[test]
    fn small_step_allowances_are_shape_derived_and_saturating() {
        // 1x4x64x64 latent: 16 f32 buffers.
        assert_eq!(sampler_step_bytes(4 * 64 * 64), 16 * 4 * 64 * 64 * 4);
        assert_eq!(rope_table_bytes(4096, 128), 4 * 4096 * 128 * 4);
        assert_eq!(elementwise_bytes(1000, 3), 12_000);
        assert_eq!(sampler_step_bytes(u64::MAX), u64::MAX);
    }

    #[test]
    fn dcae_tiles_and_channels_control_the_floor_without_wrapping() {
        assert!(dcae_stage_bytes(1, 128, 128, 128) >= 32 * 128 * 128 * 128 * 4);
        assert!(dcae_stage_bytes(1, 256, 256, 128) > dcae_stage_bytes(1, 128, 128, 128));
        assert_eq!(dcae_stage_bytes(u64::MAX, 128, 128, 128), u64::MAX);
    }
}
