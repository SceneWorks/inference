//! Shared Perturbed-Attention Guidance (PAG) capabilities and tensor-free helpers for
//! Qwen-Image 2.1 (`qwen_image_2_1` and `qwen_image_2_1_turbo`) across MLX and Candle.

use std::borrow::Cow;
use std::ops::Range;

use gen_core::{Error, PagCapability, PagResolved, Result, DEFAULT_PAG_MAX_SCALE};

/// Number of single-stream transformer blocks in Qwen-Image 2.1.
pub const QWEN_IMAGE_2_1_NUM_BLOCKS: u32 = 32;
/// Curated default block index perturbed when [`gen_core::PagConfig::layers`] is omitted.
pub const QWEN_IMAGE_2_1_DEFAULT_LAYER: u32 = 16;

/// Shared PAG capability advertised by `qwen_image_2_1` on MLX and Candle.
pub fn qwen_image_2_1_pag_capability() -> PagCapability {
    PagCapability {
        num_blocks: QWEN_IMAGE_2_1_NUM_BLOCKS,
        block_domain: Cow::Borrowed("blocks"),
        default_layers: vec![QWEN_IMAGE_2_1_DEFAULT_LAYER],
        default_start: 0.0,
        default_stop: 1.0,
        recommended_scale: 1.5,
        max_scale: DEFAULT_PAG_MAX_SCALE,
    }
}

/// Shared PAG capability advertised by `qwen_image_2_1_turbo` (8-step distilled) on MLX and Candle.
pub fn qwen_image_2_1_turbo_pag_capability() -> PagCapability {
    PagCapability {
        num_blocks: QWEN_IMAGE_2_1_NUM_BLOCKS,
        block_domain: Cow::Borrowed("blocks"),
        default_layers: vec![QWEN_IMAGE_2_1_DEFAULT_LAYER],
        default_start: 0.0,
        default_stop: 0.75,
        recommended_scale: 1.2,
        max_scale: DEFAULT_PAG_MAX_SCALE,
    }
}

/// Map a descending flow-match `sigma` onto its zero-based step index in `0..sigmas.len() - 1`.
///
/// Matches the outer-step progress convention used by `candle_gen::sampler` (`sigmas` descending to
/// trailing `0.0`), clamping to the final active step when a multi-evaluation solver queries at a
/// smaller intermediate sigma.
pub fn pag_step_index_for_sigma(sigmas: &[f32], sigma: f32) -> usize {
    let total_steps = sigmas.len().saturating_sub(1);
    if total_steps == 0 {
        return 0;
    }
    sigmas[..total_steps]
        .iter()
        .filter(|&&s| s > sigma)
        .count()
        .min(total_steps - 1)
}

/// Build the per-block perturbation mask for `(step_index, total_steps)` when PAG is active on this
/// step (`step_scale > 0.0`), or `None` when the step does not run a perturbed pass.
pub fn pag_perturb_block_flags(
    pag: Option<&PagResolved>,
    num_blocks: usize,
    step_index: usize,
    total_steps: usize,
) -> Option<Vec<bool>> {
    let pag = pag?;
    if pag.step_scale(step_index, total_steps) <= 0.0 {
        return None;
    }
    Some(
        (0..num_blocks)
            .map(|idx| pag.perturbs_block(idx as u32, step_index, total_steps))
            .collect(),
    )
}

/// Validate and return the trailing `TargetImage` token range `[prefix_len, total_len)` inside
/// Qwen-Image 2.1's single-stream `[Text, RefImage0, ..., TargetImage]` sequence.
pub fn pag_joint_target_range(prefix_len: usize, total_len: usize) -> Result<Range<usize>> {
    if prefix_len >= total_len {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: PAG target range requires prefix_len ({prefix_len}) < total_len ({total_len})"
        )));
    }
    Ok(prefix_len..total_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{resolve_pag, PagConfig, PagPolicyErrorKind};

    #[test]
    fn capabilities_are_self_consistent_and_distinct_between_base_and_turbo() {
        let base = qwen_image_2_1_pag_capability();
        let turbo = qwen_image_2_1_turbo_pag_capability();
        base.validate_definition().unwrap();
        turbo.validate_definition().unwrap();

        assert_eq!(base.num_blocks, 32);
        assert_eq!(base.block_domain, "blocks");
        assert_eq!(base.default_layers, vec![16]);
        assert_eq!(base.default_start, 0.0);
        assert_eq!(base.default_stop, 1.0);
        assert_eq!(base.recommended_scale, 1.5);
        assert_eq!(base.max_scale, 15.0);

        assert_eq!(turbo.num_blocks, 32);
        assert_eq!(turbo.block_domain, "blocks");
        assert_eq!(turbo.default_layers, vec![16]);
        assert_eq!(turbo.default_start, 0.0);
        assert_eq!(turbo.default_stop, 0.75);
        assert_eq!(turbo.recommended_scale, 1.2);
        assert_eq!(turbo.max_scale, 15.0);
    }

    #[test]
    fn resolve_pag_defaults_and_zero_scale_noop() {
        let base = qwen_image_2_1_pag_capability();
        let turbo = qwen_image_2_1_turbo_pag_capability();

        let cfg = PagConfig {
            scale: 1.5,
            ..Default::default()
        };
        let resolved_base = resolve_pag(Some(&base), "qwen_image_2_1", Some(&cfg))
            .unwrap()
            .expect("positive scale resolves");
        assert_eq!(resolved_base.layers, vec![16]);
        assert_eq!(resolved_base.start, 0.0);
        assert_eq!(resolved_base.stop, 1.0);
        assert_eq!(resolved_base.active_steps(40), 40);

        let resolved_turbo = resolve_pag(Some(&turbo), "qwen_image_2_1_turbo", Some(&cfg))
            .unwrap()
            .expect("positive scale resolves");
        assert_eq!(resolved_turbo.layers, vec![16]);
        assert_eq!(resolved_turbo.start, 0.0);
        assert_eq!(resolved_turbo.stop, 0.75);
        assert_eq!(resolved_turbo.active_steps(8), 6);

        let zero_cfg = PagConfig {
            scale: 0.0,
            layers: Some(vec![16]),
            ..Default::default()
        };
        assert!(resolve_pag(Some(&base), "qwen_image_2_1", Some(&zero_cfg))
            .unwrap()
            .is_none());

        let oob_cfg = PagConfig {
            scale: 0.0,
            layers: Some(vec![32]),
            ..Default::default()
        };
        let err = resolve_pag(Some(&base), "qwen_image_2_1", Some(&oob_cfg)).unwrap_err();
        assert_eq!(err.kind(), PagPolicyErrorKind::Invalid);

        let report = resolved_turbo.execution_report(8);
        report.validate_with_capability(&turbo).unwrap();
    }

    #[test]
    fn step_index_and_block_flags_and_target_range() {
        let sigmas = [1.0f32, 0.8, 0.5, 0.2, 0.0];
        assert_eq!(pag_step_index_for_sigma(&sigmas, 1.0), 0);
        assert_eq!(pag_step_index_for_sigma(&sigmas, 0.8), 1);
        assert_eq!(pag_step_index_for_sigma(&sigmas, 0.5), 2);
        assert_eq!(pag_step_index_for_sigma(&sigmas, 0.2), 3);
        assert_eq!(pag_step_index_for_sigma(&sigmas, 0.0), 3);
        assert_eq!(pag_step_index_for_sigma(&[], 1.0), 0);

        let turbo = qwen_image_2_1_turbo_pag_capability();
        let resolved = resolve_pag(
            Some(&turbo),
            "qwen_image_2_1_turbo",
            Some(&PagConfig {
                scale: 1.2,
                layers: Some(vec![10, 16]),
                ..Default::default()
            }),
        )
        .unwrap()
        .unwrap();
        // 8 steps with default_stop 0.75 -> steps 0..6 active, steps 6..8 inactive.
        let active = pag_perturb_block_flags(Some(&resolved), 32, 5, 8).unwrap();
        assert_eq!(active.len(), 32);
        assert!(active[10] && active[16]);
        assert!(!active[0] && !active[15]);
        assert!(pag_perturb_block_flags(Some(&resolved), 32, 6, 8).is_none());
        assert!(pag_perturb_block_flags(None, 32, 0, 8).is_none());

        assert_eq!(pag_joint_target_range(120, 376).unwrap(), 120..376);
        assert!(pag_joint_target_range(376, 376).is_err());
    }
}
