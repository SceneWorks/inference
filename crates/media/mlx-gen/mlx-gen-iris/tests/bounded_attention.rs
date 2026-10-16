//! Tiny, CPU-only parity for the request-selected kernel (no model weights).
use mlx_gen::{AttentionBudget, AttentionPlan};
use mlx_gen_iris::nn::attention_budgeted;
use mlx_rs::{Array, Dtype};

#[test]
fn bounded_attention_preserves_gqa_mha_and_query_masks() {
    crate::common::cpu_process(
        "bounded_attention::bounded_attention_preserves_gqa_mha_and_query_masks",
        run,
    );
}

fn run() {
    for (heads, kv_heads) in [(4, 1), (4, 2), (4, 4)] {
        let q = Array::from_slice(
            &(0..7 * heads * 8)
                .map(|i| (i as f32 * 0.13).sin())
                .collect::<Vec<_>>(),
            &[1, 7, heads, 8],
        );
        let k = Array::from_slice(
            &(0..7 * kv_heads * 8)
                .map(|i| (i as f32 * 0.17).cos())
                .collect::<Vec<_>>(),
            &[1, 7, kv_heads, 8],
        );
        let v = k.multiply(Array::from_f32(0.7)).unwrap();
        let mask = Array::from_slice(
            &(0..49)
                .map(|i| if i % 7 > i / 7 { -10000.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            &[1, 1, 7, 7],
        );
        for mask in [None, Some(&mask)] {
            let full =
                attention_budgeted(&q, &k, &v, Dtype::Float32, mask, AttentionPlan::UNBOUNDED)
                    .unwrap();

            let bounded = attention_budgeted(
                &q,
                &k,
                &v,
                Dtype::Float32,
                mask,
                AttentionPlan::budgeted(AttentionBudget::from_score_elements(28, true)),
            )
            .unwrap();
            let error = full
                .subtract(&bounded)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(error < 2e-5, "{heads}/{kv_heads}: {error}");
            // A cancelled bounded call must enter the chunk path, not silently run unbounded.
            let cancel = mlx_gen::CancelFlag::new();
            cancel.cancel();
            assert!(matches!(
                attention_budgeted(
                    &q,
                    &k,
                    &v,
                    Dtype::Float32,
                    mask,
                    AttentionPlan::budgeted(AttentionBudget::from_score_elements(28, true))
                        .with_cancel(&cancel)
                ),
                Err(mlx_gen::Error::Canceled)
            ));
            assert!(attention_budgeted(
                &q,
                &k,
                &v,
                Dtype::Float32,
                mask,
                AttentionPlan::UNBOUNDED.with_cancel(&cancel)
            )
            .is_ok());
        }
    }
}

#[test]
fn tiny_full_backbone_bounded_matches_unbounded() {
    crate::common::cpu_process(
        "bounded_attention::tiny_full_backbone_bounded_matches_unbounded",
        full_backbone,
    );
}

fn full_backbone() {
    let config = crate::common::tiny_config();
    let golden = crate::common::fixture("iris_dit_golden.safetensors");
    let flat = crate::common::host_i32(golden.require("y_mask").unwrap());
    let mask: Vec<Vec<i32>> = flat.chunks(flat.len() / 2).map(|m| m.to_vec()).collect();
    let text = mlx_gen_iris::TextBatch {
        states: golden.require("y").unwrap(),
        mask: &mask,
    };
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let dit =
            mlx_gen_iris::load_backbone(&crate::common::tiny_backbone(), &config, dtype).unwrap();
        let x = golden.require("x").unwrap();
        let t = golden.require("t").unwrap();
        let full = dit.forward(x, t, &text).unwrap();
        let bounded = dit
            .forward_budgeted(x, t, &text, AttentionBudget::from_score_elements(32, true))
            .unwrap();
        crate::common::assert_close(
            "bounded full Iris",
            &bounded,
            &full,
            if dtype == Dtype::Float32 { 2e-4 } else { 0.03 },
        );
    }
}
