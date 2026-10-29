use super::*;

const GIB: u64 = 1 << 30;

/// The planned decode is exactly the product planner's decision at the same safe budget.
#[test]
fn planned_decode_follows_the_product_planner() {
    for (vae, frames) in [(WanVae::VAE_TILING, 36), (Wan22Vae::VAE_TILING, 33)] {
        let single =
            video_decode_peak_bytes_for_vae_and_tiling(vae, 768, 512, frames, None).unwrap();
        // A budget the single pass fits: single pass, priced at the single-pass cost.
        let roomy = planned_video_decode(vae, 768, 512, frames, 200 * GIB).unwrap();
        assert_eq!(roomy.mode, PlannedDecodeMode::SinglePass);
        assert_eq!(roomy.working_set_bytes, single);
        // A smaller budget tiles: the chosen tile's cost, under the safe budget, below single.
        let tight = planned_video_decode(vae, 768, 512, frames, 40 * GIB).unwrap();
        assert_eq!(tight.mode, PlannedDecodeMode::Tiled);
        assert!((tight.safe_budget_gib - 34.0).abs() < 1e-9);
        assert!(tight.working_set_bytes < single);
        assert!(tight.working_set_bytes as f64 <= tight.safe_budget_gib * GIB as f64);
        let (h, w, f) = (512, 768, frames as i32);
        let expected = if vae == WanVae::VAE_TILING {
            plan_z16_tiling(h, w, f, tight.safe_budget_gib)
        } else {
            plan_vae22_tiling(h, w, f, tight.safe_budget_gib, true)
        }
        .unwrap()
        .unwrap();
        assert_eq!(
            tight.working_set_bytes,
            video_decode_peak_bytes_for_vae_and_tiling(vae, 768, 512, frames, Some(&expected))
                .unwrap()
        );
        // No tile fits: over budget, priced at the conservative single pass.
        let starved = planned_video_decode(vae, 768, 512, frames, GIB / 4).unwrap();
        assert_eq!(starved.mode, PlannedDecodeMode::OverBudget);
        assert_eq!(starved.working_set_bytes, single);
    }
}
