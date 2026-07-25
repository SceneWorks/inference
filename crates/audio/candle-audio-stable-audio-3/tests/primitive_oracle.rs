//! Actual-checkpoint parity against the independently locked sc-14536 primitive oracle.

use std::path::{Path, PathBuf};

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_stable_audio_3::config::{FeedForwardConfig, NormConfig, NormType, QkNorm};
use candle_audio_stable_audio_3::pretransform::PatchedPretransform;
use candle_audio_stable_audio_3::softnorm::SoftNorm;
use candle_audio_stable_audio_3::transformer::{
    sliding_window_additive_mask, Attention, AttentionMasks, FeedForward, LayerScale, MemoryTokens,
    Norm, RotaryEmbedding, TransformerBlock, TransformerBlockMasks,
};
use candle_audio_stable_audio_3::weight_norm::wn_conv1d;
use candle_nn::{Conv1dConfig, Module, VarBuilder};

fn snapshot(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to the pinned snapshot"))
}

fn mmap(path: &Path) -> VarBuilder<'static> {
    // Safety: these tests require immutable pinned snapshots for their entire process lifetime.
    unsafe {
        VarBuilder::from_mmaped_safetensors(&[path.to_path_buf()], DType::F32, &Device::Cpu)
            .unwrap()
    }
}

fn oracle() -> VarBuilder<'static> {
    mmap(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/migration/sa3-primitives-reference/primitives.safetensors"),
    )
}

fn metric(name: &str, actual: &Tensor, expected: &Tensor, max_abs_limit: f32) {
    let actual = actual
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let expected = expected
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(actual.len(), expected.len(), "{name}");
    let mut dot = 0f64;
    let mut aa = 0f64;
    let mut bb = 0f64;
    let mut max_abs = 0f32;
    for (&a, &b) in actual.iter().zip(&expected) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
    }
    let cosine = dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE);
    eprintln!("{name}: cosine={cosine:.9}, max_abs={max_abs:.9}");
    assert!(cosine >= 0.9999, "{name}: cosine {cosine}");
    assert!(
        max_abs <= max_abs_limit,
        "{name}: max_abs {max_abs} > {max_abs_limit}"
    );
}

fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
    left.to_dtype(DType::F32)
        .unwrap()
        .broadcast_sub(&right.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn frozen_upstream_missing_branches_match() {
    let expected = oracle();

    let layer_norm = Norm::load(
        NormType::LayerNorm,
        4,
        &NormConfig {
            fix_scale: true,
            force_fp32: true,
            eps: 1e-5,
        },
        expected.pp("branch_ln"),
    )
    .unwrap();
    let ln_x = expected
        .get((1, 2, 4), "branch_ln_x")
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let ln_output = layer_norm.forward(&ln_x).unwrap();
    assert_eq!(ln_output.dtype(), DType::F16);
    metric(
        "layer_norm_fix_scale_force_fp32",
        &ln_output,
        &expected.get((1, 2, 4), "branch_ln_output").unwrap(),
        0.0,
    );

    let rms_norm = Norm::load(
        NormType::RmsNorm,
        4,
        &NormConfig {
            fix_scale: true,
            force_fp32: true,
            eps: 1e-5,
        },
        expected.pp("branch_rms"),
    )
    .unwrap();
    let rms_x = expected
        .get((1, 2, 4), "branch_rms_x")
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let rms_output = rms_norm.forward(&rms_x).unwrap();
    assert_eq!(rms_output.dtype(), DType::F16);
    metric(
        "rms_norm_fix_scale_force_fp32",
        &rms_output,
        &expected.get((1, 2, 4), "branch_rms_output").unwrap(),
        0.0,
    );
    let rms_without_force_fp32 = Norm::load(
        NormType::RmsNorm,
        4,
        &NormConfig {
            fix_scale: true,
            force_fp32: false,
            eps: 1e-5,
        },
        expected.pp("branch_rms"),
    )
    .unwrap()
    .forward(&rms_x)
    .unwrap();
    assert!(
        max_abs_diff(&rms_output, &rms_without_force_fp32) > 1e-4,
        "force_fp32 mutation must be observable on the locked f16 input"
    );

    let layer_scale = LayerScale::load(4, expected.pp("branch_scale")).unwrap();
    metric(
        "layer_scale",
        &layer_scale
            .forward(&expected.get((1, 1, 4), "branch_scale_x").unwrap())
            .unwrap(),
        &expected.get((1, 1, 4), "branch_scale_output").unwrap(),
        0.0,
    );

    let cross = Attention::load(
        4,
        2,
        Some(4),
        QkNorm::None,
        1e-6,
        true,
        false,
        false,
        expected.pp("branch_cross"),
    )
    .unwrap();
    let cross_output = cross
        .forward(
            &expected.get((1, 3, 4), "branch_cross_x").unwrap(),
            Some(&expected.get((1, 4, 4), "branch_cross_context").unwrap()),
            None,
            None,
            Some(&expected.get((1, 4), "branch_cross_padding").unwrap()),
            Some(&expected.get((1, 1, 3, 4), "branch_cross_additive").unwrap()),
        )
        .unwrap();
    metric(
        "differential_cross_padding_additive",
        &cross_output,
        &expected.get((1, 3, 4), "branch_cross_output").unwrap(),
        1e-5,
    );

    let qk_ln = Attention::load(
        4,
        2,
        None,
        QkNorm::Ln,
        1e-6,
        false,
        false,
        false,
        expected.pp("branch_qk_ln"),
    )
    .unwrap();
    let qk_ln_output = qk_ln
        .forward(
            &expected.get((1, 3, 4), "branch_qk_ln_x").unwrap(),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    metric(
        "attention_qk_layer_norm_weight_bias",
        &qk_ln_output,
        &expected.get((1, 3, 4), "branch_qk_ln_output").unwrap(),
        1e-5,
    );
    let qk_norm_removed = Attention::load(
        4,
        2,
        None,
        QkNorm::None,
        1e-6,
        false,
        false,
        false,
        expected.pp("branch_qk_ln"),
    )
    .unwrap()
    .forward(
        &expected.get((1, 3, 4), "branch_qk_ln_x").unwrap(),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(
        max_abs_diff(&qk_ln_output, &qk_norm_removed) > 1e-4,
        "removing qk LayerNorm must fail the locked oracle"
    );

    let sliding = sliding_window_additive_mask(4, 4, 1, 1, DType::F32, &Device::Cpu).unwrap();
    let values = sliding.to_vec2::<f32>().unwrap();
    assert!(values[0][2].is_infinite() && values[0][2].is_sign_negative());
    assert_eq!(values[1][0..3], [0.0, 0.0, 0.0]);
    assert!(values[3][1].is_infinite() && values[3][1].is_sign_negative());
}

#[test]
#[ignore = "requires four explicitly provisioned pinned SA3 snapshots"]
fn actual_checkpoints_match_locked_primitive_oracle() {
    let expected = oracle();
    let small_root = mmap(&snapshot("SA3_SMALL_MUSIC_SNAPSHOT").join("model.safetensors"));
    let medium_root = mmap(&snapshot("SA3_MEDIUM_SNAPSHOT").join("model.safetensors"));
    let same_s_root = mmap(&snapshot("SA3_SAME_S_SNAPSHOT").join("model.safetensors"));
    let same_l_root = mmap(&snapshot("SA3_SAME_L_SNAPSHOT").join("model.safetensors"));

    let small_x = expected.get((1, 4, 1024), "small_x").unwrap();
    let small_context = expected.get((1, 3, 1024), "small_context").unwrap();
    let small_global = expected.get((1, 6144), "small_global").unwrap();
    let small_local = expected.get((1, 4, 257), "small_local").unwrap();
    let small_padding = expected.get((1, 4), "small_padding").unwrap();
    let small_rope = expected.get((4, 32), "small_rope").unwrap();
    let small_cfg = FeedForwardConfig {
        mult: 4.0,
        ..Default::default()
    };
    let rms = NormConfig {
        fix_scale: false,
        force_fp32: true,
        eps: 1e-5,
    };
    let small = TransformerBlock::load(
        1024,
        64,
        Some(1024),
        NormType::RmsNorm,
        &rms,
        QkNorm::Rms,
        1e-6,
        false,
        false,
        &small_cfg,
        true,
        true,
        Some(257),
        false,
        small_root.pp("model.model.transformer.layers.0"),
    )
    .unwrap();
    let actual = small
        .forward(
            &small_x,
            Some(&small_context),
            Some(&small_global),
            Some(&small_local),
            Some(&small_rope),
            None,
            TransformerBlockMasks {
                self_attention: AttentionMasks {
                    key_padding: Some(&small_padding),
                    additive: None,
                },
                cross_attention: AttentionMasks::default(),
            },
        )
        .unwrap();
    metric(
        "small_block",
        &actual,
        &expected.get((1, 4, 1024), "small_block").unwrap(),
        2e-3,
    );

    // Mutation-sensitive assembled-block proof: self and cross masks are independently wired.
    let self_additive = sliding_window_additive_mask(4, 4, 0, 0, DType::F32, &Device::Cpu).unwrap();
    let cross_padding = Tensor::from_vec(vec![1f32, 0.0, 1.0], (1, 3), &Device::Cpu).unwrap();
    let cross_additive = Tensor::from_vec(
        vec![
            0f32,
            f32::NEG_INFINITY,
            -0.5,
            0.0,
            f32::NEG_INFINITY,
            -0.5,
            0.0,
            f32::NEG_INFINITY,
            -0.5,
            0.0,
            f32::NEG_INFINITY,
            -0.5,
        ],
        (1, 1, 4, 3),
        &Device::Cpu,
    )
    .unwrap();
    let self_masked = small
        .forward(
            &small_x,
            Some(&small_context),
            Some(&small_global),
            Some(&small_local),
            Some(&small_rope),
            None,
            TransformerBlockMasks {
                self_attention: AttentionMasks {
                    key_padding: Some(&small_padding),
                    additive: Some(&self_additive),
                },
                cross_attention: AttentionMasks::default(),
            },
        )
        .unwrap();
    let cross_masked = small
        .forward(
            &small_x,
            Some(&small_context),
            Some(&small_global),
            Some(&small_local),
            Some(&small_rope),
            None,
            TransformerBlockMasks {
                self_attention: AttentionMasks {
                    key_padding: Some(&small_padding),
                    additive: None,
                },
                cross_attention: AttentionMasks {
                    key_padding: Some(&cross_padding),
                    additive: Some(&cross_additive),
                },
            },
        )
        .unwrap();
    assert!(max_abs_diff(&actual, &self_masked) > 1e-4);
    assert!(max_abs_diff(&actual, &cross_masked) > 1e-4);
    assert!(max_abs_diff(&self_masked, &cross_masked) > 1e-4);

    let memory = MemoryTokens::load(64, 1024, small_root.pp("model.model.transformer")).unwrap();
    let (with_memory, memory_mask) = memory.prepend(&small_x, Some(&small_padding)).unwrap();
    metric(
        "memory_prepend",
        &with_memory,
        &expected.get((1, 68, 1024), "small_memory_prepend").unwrap(),
        0.0,
    );
    metric(
        "memory_mask",
        &memory_mask.unwrap(),
        &expected.get((1, 68), "small_memory_mask").unwrap(),
        0.0,
    );
    metric(
        "memory_trim",
        &memory.trim(&with_memory).unwrap(),
        &small_x,
        0.0,
    );

    let medium = Attention::load(
        1536,
        64,
        None,
        QkNorm::Rms,
        1e-6,
        true,
        false,
        false,
        medium_root.pp("model.model.transformer.layers.0.self_attn"),
    )
    .unwrap();
    let medium_x = expected.get((1, 3, 1536), "medium_x").unwrap();
    let medium_out = medium
        .forward(
            &medium_x,
            None,
            Some(&expected.get((3, 32), "medium_rope").unwrap()),
            None,
            Some(&expected.get((1, 3), "medium_padding").unwrap()),
            None,
        )
        .unwrap();
    metric(
        "medium_differential_attention",
        &medium_out,
        &expected.get((1, 3, 1536), "medium_attention").unwrap(),
        4e-4,
    );

    let dyt_cfg = NormConfig::default();
    let same_s_ff = FeedForwardConfig {
        mult: 3.0,
        ..Default::default()
    };
    let same_s = TransformerBlock::load(
        768,
        64,
        None,
        NormType::Dyt,
        &dyt_cfg,
        QkNorm::Dyt,
        1e-6,
        true,
        false,
        &same_s_ff,
        false,
        false,
        None,
        false,
        same_s_root.pp("encoder.layers.0.transformers.0"),
    )
    .unwrap();
    let same_s_x = expected.get((1, 5, 768), "same_s_x").unwrap();
    let same_s_rope = RotaryEmbedding::new(32, &Device::Cpu)
        .unwrap()
        .frequencies(5)
        .unwrap();
    let same_s_out = same_s
        .forward(
            &same_s_x,
            None,
            None,
            None,
            Some(&same_s_rope),
            None,
            TransformerBlockMasks::default(),
        )
        .unwrap();
    metric(
        "same_s_block",
        &same_s_out,
        &expected.get((1, 5, 768), "same_s_block").unwrap(),
        4e-4,
    );

    let same_l_cfg = FeedForwardConfig {
        mult: 3.0,
        sinusoidal: true,
        zero_init_output: false,
        ..Default::default()
    };
    let same_l = FeedForward::load(
        1536,
        &same_l_cfg,
        same_l_root.pp("decoder.layers.3.transformers.5.ff"),
    )
    .unwrap();
    let same_l_out = same_l
        .forward(&expected.get((1, 2, 1536), "same_l_x").unwrap())
        .unwrap();
    metric(
        "same_l_sin_ff",
        &same_l_out,
        &expected.get((1, 2, 1536), "same_l_sin_ff").unwrap(),
        2e-4,
    );

    let wn = wn_conv1d(
        512,
        768,
        1,
        true,
        Conv1dConfig::default(),
        same_s_root.pp("encoder.layers.0.mapping"),
    )
    .unwrap();
    let wn_out = wn
        .forward(&expected.get((1, 512, 7), "wn_x").unwrap())
        .unwrap();
    metric(
        "wn_conv1d",
        &wn_out,
        &expected.get((1, 768, 7), "wn_output").unwrap(),
        2e-4,
    );

    let patch = PatchedPretransform::new(2, 256).unwrap();
    let patch_encoded = patch
        .encode(&expected.get((1, 2, 259), "patch_x").unwrap())
        .unwrap();
    metric(
        "patch_encode",
        &patch_encoded,
        &expected.get((1, 512, 2), "patch_encoded").unwrap(),
        0.0,
    );
    metric(
        "patch_decode",
        &patch.decode(&patch_encoded).unwrap(),
        &expected.get((1, 2, 512), "patch_decoded").unwrap(),
        0.0,
    );

    let soft = SoftNorm::load(256, 0, true, true, same_s_root.pp("bottleneck")).unwrap();
    let soft_encoded = soft
        .encode(&expected.get((1, 256, 5), "soft_x").unwrap())
        .unwrap();
    metric(
        "softnorm_encode",
        &soft_encoded,
        &expected.get((1, 256, 5), "soft_encoded").unwrap(),
        1e-6,
    );
    for (training, name) in [(false, "soft_eval"), (true, "soft_train")] {
        let decoded = soft
            .decode_with_noise(
                &soft_encoded,
                training,
                Some(&expected.get((1, 256, 5), "soft_noise").unwrap()),
                None,
            )
            .unwrap();
        metric(
            name,
            &decoded,
            &expected.get((1, 256, 5), name).unwrap(),
            1e-6,
        );
    }
}
