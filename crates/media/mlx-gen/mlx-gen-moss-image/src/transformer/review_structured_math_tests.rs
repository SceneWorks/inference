use super::author_fixture_tests::*;
use super::*;
use mlx_gen::gen_core::{moss_image::MossImageVariant, moss_image_training as contract};
use std::collections::{BTreeMap, HashMap};
fn values(rows: usize, cols: usize, phase: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i as f32 + 1.) * 0.31 + phase).sin() * 0.4)
        .collect()
}
fn product(a: &[f32], rows: usize, rank: usize, b: &[f32], cols: usize) -> Vec<f64> {
    (0..rows)
        .flat_map(|r| {
            (0..cols).map(move |c| {
                (0..rank)
                    .map(|k| a[r * rank + k] as f64 * b[k * cols + c] as f64)
                    .sum()
            })
        })
        .collect()
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn raw_structured_full_and_decomposed_factors_match_independent_kronecker_math() {
    let _device = crate::cpu_test_device();
    for ((a, c), (b, d)) in [((2usize, 3usize), (4usize, 2usize)), ((4, 2), (2, 3))] {
        for decomposed in [false, true] {
            let mut group = BTreeMap::new();
            let mut effective = Vec::new();
            for (name, rows, cols, phase) in [("lokr_w1", a, c, 0.2), ("lokr_w2", b, d, 0.7)] {
                if decomposed {
                    let rank = 2;
                    let left = values(rows, rank, phase);
                    let right = values(rank, cols, phase + 0.4);
                    effective.push(product(&left, rows, rank, &right, cols));
                    group.insert(
                        format!("{name}_a"),
                        Array::from_slice(&left, &[rows as i32, rank as i32]),
                    );
                    group.insert(
                        format!("{name}_b"),
                        Array::from_slice(&right, &[rank as i32, cols as i32]),
                    );
                } else {
                    let full = values(rows, cols, phase);
                    effective.push(full.iter().map(|v| *v as f64).collect());
                    group.insert(
                        name.into(),
                        Array::from_slice(&full, &[rows as i32, cols as i32]),
                    );
                }
            }
            let input = values(6, c * d, 1.3);
            let x = Array::from_slice(&input, &[2, 3, (c * d) as i32]);
            let scale = -0.75f32;
            let lokr =
                crate::adapters::Lokr::from_group(&group, &[(a * b) as i32, (c * d) as i32], scale)
                    .unwrap();
            let output = lokr.residual(&x).unwrap().expect("nonzero explicit scale");
            assert_eq!(output.shape(), &[2, 3, (a * b) as i32]);
            let actual = crate::pipeline::logical_f32_readback(&output).unwrap();
            for row in 0..6 {
                for p in 0..a {
                    for q in 0..b {
                        let expected: f64 = (0..c)
                            .flat_map(|r| (0..d).map(move |s| (r, s)))
                            .map(|(r, s)| {
                                input[row * c * d + r * d + s] as f64
                                    * effective[0][p * c + r]
                                    * effective[1][q * d + s]
                                    * scale as f64
                            })
                            .sum();
                        let error = (actual[row * a * b + p * b + q] as f64 - expected).abs();
                        assert!(
                            error < 1e-6 + expected.abs() * 1e-5,
                            "Kronecker axis/order/scale mismatch: {error}, decomposed={decomposed}"
                        );
                    }
                }
            }
            let zero =
                crate::adapters::Lokr::from_group(&group, &[(a * b) as i32, (c * d) as i32], 0.)
                    .unwrap();
            assert!(zero.residual(&x).unwrap().is_none());
        }
    }
}
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn real_loader_uses_nondefault_rank_alpha_and_user_strength_on_the_projection() {
    let _device = crate::cpu_test_device();
    let fixture = fixture("ordinary");
    let geometry = geometry(&fixture);
    let mut metadata = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
    metadata.insert("networkType".into(), "lokr".into());
    metadata.insert("rank".into(), "4".into());
    metadata.insert("alpha".into(), "6".into());
    let metadata: HashMap<_, _> = metadata.into_iter().collect();
    let w1 = Array::from_slice(&[0.5f32; 4], &[2, 2]);
    let w2 = Array::from_slice(&[0.25f32; 64], &[8, 8]);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("scaled.safetensors");
    Array::save_safetensors(
        [
            ("layers.0.proj.q.lokr_w1", &w1),
            ("layers.0.proj.q.lokr_w2", &w2),
        ],
        Some(&metadata),
        &path,
    )
    .unwrap();
    let x = Array::from_slice(&(1..=16).map(|v| v as f32).collect::<Vec<_>>(), &[1, 1, 16]);
    for strength in [1f32, -2.] {
        let mut model = Transformer::load(
            &weights(&fixture["weights"], Dtype::Float32),
            geometry.clone(),
        )
        .unwrap();
        crate::adapters::apply(
            &mut model,
            MossImageVariant::Base,
            &[mlx_gen::runtime::AdapterSpec::new(
                path.clone(),
                strength,
                mlx_gen::runtime::AdapterKind::Lokr,
            )],
        )
        .unwrap();
        let residual = model.blocks[0].projections[0].lokr[0]
            .residual(&x)
            .unwrap()
            .unwrap();
        let expected = (1..=16).sum::<i32>() as f32 * 0.5 * 0.25 * (6. / 4.) * strength;
        assert_eq!(
            crate::pipeline::logical_f32_readback(&residual).unwrap(),
            vec![expected; 16]
        );
    }
    // Mirrored Candle/MLX parity oracle: rank=2, global alpha=4, no target alpha
    // MUST produce residual 17 at strength 1, not the unity-fallback value 8.5.
    // Per-target alpha overrides global; malformed native rank is a refusal.
    let a = Array::from_slice(&[0.125f32; 32], &[2, 16]);
    let b = Array::from_slice(&[0.25f32; 32], &[16, 2]);
    for (native_rank, target_alpha, expected_alpha) in
        [(2, None, 4.), (2, Some(1f32), 1.), (3, None, 4.)]
    {
        let mut metadata = contract::metadata(MossImageVariant::Base, contract::SAMPLER);
        metadata.extend([
            ("networkType".into(), "lora".into()),
            ("rank".into(), native_rank.to_string()),
            ("alpha".into(), "4".into()),
        ]);
        let metadata: HashMap<_, _> = metadata.into_iter().collect();
        let mut tensors = vec![
            ("layers.0.proj.q.lora_A.weight", a.clone()),
            ("layers.0.proj.q.lora_B.weight", b.clone()),
        ];
        if let Some(alpha) = target_alpha {
            tensors.push(("layers.0.proj.q.alpha", Array::from_slice(&[alpha], &[1])));
        }
        let file = temp.path().join("lora-alpha.safetensors");
        Array::save_safetensors(tensors.iter().map(|(k, v)| (*k, v)), Some(&metadata), &file)
            .unwrap();
        for strength in [1f32, -2.] {
            let mut model = Transformer::load(
                &weights(&fixture["weights"], Dtype::Float32),
                geometry.clone(),
            )
            .unwrap();
            let result = crate::adapters::apply(
                &mut model,
                MossImageVariant::Base,
                &[mlx_gen::runtime::AdapterSpec::new(
                    file.clone(),
                    strength,
                    mlx_gen::runtime::AdapterKind::Lora,
                )],
            );
            if native_rank != 2 {
                assert!(
                    result.is_err(),
                    "LoRA tensor/native metadata rank mismatch accepted"
                );
                continue;
            }
            result.unwrap();
            let Adapter::Lora { a, b, scale } =
                &model.blocks[0].projections[0].custom.adapters()[0]
            else {
                panic!("actual loaded LoRA missing")
            };
            let residual = self::scale(
                &linear_fp32(&linear_fp32(&x, &a.t()).unwrap(), &b.t()).unwrap(),
                *scale,
            )
            .unwrap();
            let expected = 136. * 0.125 * 0.25 * 2. * (expected_alpha / 2.) * strength;
            assert_eq!(crate::pipeline::logical_f32_readback(&residual).unwrap(), vec![expected;16],
                    "native LoRA global alpha fallback/target precedence differs from Candle parity oracle");
        }
    }
}
