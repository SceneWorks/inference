//! SANA/Gemma iOS diagnostic quantum accounting. These are conservative tensor scratch floors,
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
    sum(&[
        product(&[8, b, heads, l, l, 4]),
        product(&[24, b, l, h, 4]),
        product(&[8, b, l, intermediate, 4]),
        product(&[12, b, l, heads, hd, 4]),
        product(&[4, h, heads, hd, 4]),
        product(&[3, h, intermediate, 4]),
    ])
}

/// Price all three GLUMBConv f32 copies while the ORIGINAL F16 weights remain resident,
/// eight attention projection copies, score softmax scratch and 32 activation-sized buffers.
/// Per-block materialization prevents multiplying this scratch by the 20-block stack.
pub fn sana_promoted_conv_bytes(dim: u64) -> u64 {
    let expanded = dim.saturating_mul(5);
    sum(&[
        product(&[4, expanded, dim]),
        product(&[4, expanded / 2, dim]),
        product(&[4, 9, expanded]),
    ])
}

pub fn sana_block_bytes(b: u64, tokens: u64, dim: u64, caption_tokens: u64) -> u64 {
    let expanded = dim.saturating_mul(5);
    sum(&[
        sana_promoted_conv_bytes(dim),
        product(&[8, dim, dim, 4]),
        product(&[32, b, tokens, expanded, 4]),
        product(&[8, b, 20, tokens, caption_tokens, 4]),
        product(&[8, b, caption_tokens, dim, 4]),
    ])
}

/// A DC-AE substage: 32 full activation-sized f32 buffers, and a conservative 64*C*C
/// f32 weight/copy allowance covering GLUMBConv (12*C*C), QKV, multiscale and conv scratch.
/// Upsample callers price the DOUBLED spatial extent before constructing that graph.
pub fn dcae_stage_bytes(b: u64, h: u64, w: u64, c: u64) -> u64 {
    sum(&[product(&[32, b, h, w, c, 4]), product(&[64, c, c, 4])])
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

    #[test]
    fn f16_convolution_promotion_is_charged_as_full_f32_scratch_not_half_storage() {
        let dim = 2240;
        // Both original F16 weights remain resident; their FULL promoted F32 copies are additional.
        let conv_copies = 4 * (5 * dim * dim + (5 * dim / 2) * dim + 9 * 5 * dim);
        assert_eq!(sana_promoted_conv_bytes(dim), conv_copies);
        assert!(sana_block_bytes(1, 64, dim, 300) >= conv_copies);
        assert!(sana_block_bytes(1, 1024, dim, 300) > sana_block_bytes(1, 64, dim, 300));
    }

    #[test]
    fn gemma_scores_use_padded_tokens_and_do_not_shrink_with_image_geometry() {
        let floor = gemma_layer_bytes(1, 506, 2304, 9216, 8, 256);
        assert!(floor >= 8 * 506 * 506 * 4 + 506 * 9216 * 4);
        assert!(gemma_layer_bytes(1, 507, 2304, 9216, 8, 256) > floor);
        assert!(gemma_layer_bytes(u64::MAX, 506, 2304, 9216, 8, 256) == u64::MAX);
    }

    #[test]
    fn dcae_tiles_and_channels_control_the_floor_without_wrapping() {
        assert!(dcae_stage_bytes(1, 128, 128, 128) >= 32 * 128 * 128 * 128 * 4);
        assert!(dcae_stage_bytes(1, 256, 256, 128) > dcae_stage_bytes(1, 128, 128, 128));
        assert_eq!(dcae_stage_bytes(u64::MAX, 128, 128, 128), u64::MAX);
    }
}
