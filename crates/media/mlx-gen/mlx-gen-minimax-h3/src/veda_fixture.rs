//! Device fixtures replace only weight-heavy conditioner/decoder sources. The actual Generator,
//! native packed DiT, learned predictor, scheduler, latent assembly and outcome path execute.
use crate::dit::{MiniMaxH3Dit, MiniMaxH3DitConfig};
use mlx_gen::gen_core::{GenerationOutput, GenerationRequest, Generator, LoadShape, VedaRequest};
use mlx_gen::{weights::Weights, Result};
use mlx_rs::{Array, Dtype};

#[derive(Clone, Default)]
pub(crate) struct Fixture {
    pub predictor: bool,
    pub fail_self_test: bool,
}
fn config() -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig {
        num_attention_heads: 4,
        attention_head_dim: 128,
        hidden_size: 64,
        num_layers: 2,
        num_refiner_layers: 2,
        ffn_dim: 32,
        text_dim: 40,
        freq_dim: 16,
        time_embed_hidden_dim: 64,
        time_embed_dim: 48,
        rope_freq_dim: 2,
        ..Default::default()
    }
}
pub(crate) fn dit() -> Result<MiniMaxH3Dit> {
    let mut w = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/dit_block.safetensors"
    ))?;
    for prefix in ["src.", "in.", "out.", "layout."] {
        w.remove_prefix(prefix);
    }
    // Widen only attention fixture weights to the production D128 kernel shape. The committed
    // fixture's nonzero channels remain, with zero projection padding and unit norm gains.
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.contains(".attn."))
        .map(str::to_owned)
        .collect();
    for key in keys {
        let value = w.require(&key)?.clone();
        let widened = if key.ends_with("norm_q.weight") || key.ends_with("norm_k.weight") {
            mlx_rs::ops::concatenate_axis(&[&value, &mlx_rs::ops::ones::<f32>(&[104])?], 0)?
        } else if key.ends_with("to_q.weight")
            || key.ends_with("to_k.weight")
            || key.ends_with("to_v.weight")
        {
            mlx_rs::ops::concatenate_axis(
                &[
                    &value.reshape(&[4, 24, 64])?,
                    &mlx_rs::ops::zeros::<f32>(&[4, 104, 64])?,
                ],
                1,
            )?
            .reshape(&[512, 64])?
        } else if key.ends_with("to_out.0.weight") {
            mlx_rs::ops::concatenate_axis(
                &[
                    &value.reshape(&[64, 4, 24])?,
                    &mlx_rs::ops::zeros::<f32>(&[64, 4, 104])?,
                ],
                2,
            )?
            .reshape(&[64, 512])?
        } else {
            value
        };
        w.insert(key, widened);
    }
    MiniMaxH3Dit::from_weights(&mut w, &config(), Dtype::Float32)
}
pub(crate) fn context() -> Array {
    Array::from_slice(
        &(0..6 * 40)
            .map(|i| (i % 13) as f32 / 13.0 - 0.5)
            .collect::<Vec<_>>(),
        &[1, 6, 40],
    )
}
pub(crate) fn decode_video(
    req: &GenerationRequest,
    latents: &Array,
) -> Result<Vec<mlx_gen::media::Image>> {
    use sha2::{Digest, Sha256};
    let flat = latents.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
    mlx_rs::transforms::eval([&flat])?;
    let mut hash = Sha256::new();
    for x in flat.as_slice::<f32>() {
        hash.update(x.to_le_bytes());
    }
    let digest = hash.finalize();
    Ok(vec![
        mlx_gen::media::Image {
            width: 1,
            height: 1,
            pixels: digest[..3].to_vec()
        };
        req.frames.unwrap_or(124) as usize
    ])
}
pub(crate) fn decode_audio(latents: &Array) -> Result<mlx_gen::media::AudioTrack> {
    let flat = latents.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
    mlx_rs::transforms::eval([&flat])?;
    Ok(mlx_gen::media::AudioTrack {
        samples: (0..330_666)
            .map(|i| flat.as_slice::<f32>()[i % 32])
            .collect(),
        sample_rate: 32000,
        channels: 2,
        ..Default::default()
    })
}
/// Tiny genuine FP8/scaled/BF16-loaded predictor, with nonzero learned weights.
pub(crate) fn predictor() -> crate::veda::contract::Bundle {
    use serde_json::json;
    let heads = 4usize;
    let dim = 128usize;
    let layers = 2usize;
    let mut metadata = serde_json::Map::new();
    for (k, v) in [
        ("format", "miowtion-veda-predictor-v1"),
        ("num_layers", "2"),
        ("num_heads", "4"),
        ("head_dim", "128"),
        ("keep_ratio", "0.1"),
        ("dtype", "float8_e4m3fn"),
    ] {
        metadata.insert(k.into(), json!(v));
    }
    metadata.insert("plans".into(),json!(json!({"canonical":{"geometry":"16x9_t37","grid":[37,24,42],"shapes":["8x4x4","2x8x8"],"head_shape":[[0,0,1,1],[1,1,0,0]]}}).to_string()));
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!(metadata));
    let mut bytes = Vec::new();
    for layer in 0..layers {
        for name in ["proj_q", "proj_k"] {
            let key = format!("layers.{layer}.{name}");
            let start = bytes.len();
            bytes.extend((0..heads * 3 * dim * dim).map(|i| {
                if (i + layer) % 5 == 0 {
                    0x9b
                } else {
                    0x1a
                }
            }));
            header.insert(key.clone(),json!({"dtype":"F8_E4M3","shape":[heads,3*dim,dim],"data_offsets":[start,bytes.len()]}));
            let start = bytes.len();
            for h in 0..heads {
                bytes.extend((0.03 + h as f32 * 0.005).to_le_bytes());
            }
            header.insert(
                format!("{key}.__scale"),
                json!({"dtype":"F32","shape":[heads],"data_offsets":[start,bytes.len()]}),
            );
        }
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(bytes);
    let meta = mlx_gen::gen_core::weightsmeta::CheckpointMeta::from_bytes(file).unwrap();
    crate::veda::contract::Bundle::from_meta(&meta, layers, heads, dim).unwrap()
}
#[test]
#[ignore = "requires the exclusive local Metal token"]
fn native_generator_learned_sparse_off_and_failure_paths() {
    let mut generator = crate::model::tests::generator_with(
        "/fixture/transformer",
        LoadShape::EagerMaterialization,
    );
    generator.fixture = Some(Fixture {
        predictor: true,
        ..Default::default()
    });
    let mut request = GenerationRequest {
        prompt: "bounded walking path".into(),
        width: 1344,
        height: 768,
        frames: Some(124),
        seed: Some(25769),
        steps: Some(1),
        ..Default::default()
    };
    let run = |g: &crate::model::MiniMaxH3, r: &GenerationRequest| {
        let out = g.generate(r, &mut |_| {}).unwrap();
        let GenerationOutput::Video { frames, fps, audio } = out else {
            panic!("video output")
        };
        assert_eq!(frames.len(), 124);
        assert_eq!(fps, 24);
        assert!(audio.is_some());
        frames
    };
    // A configured but missing optional component must not be touched by omitted/off settings.
    generator.veda_component = Some(std::path::PathBuf::from(
        "/missing/veda-predictor.safetensors",
    ));
    let dense = run(&generator, &request);
    let off_report = generator.veda_execution_report().unwrap();
    assert!(!off_report.requested);
    assert_eq!(off_report.predictor_load_attempts, 0);
    assert_eq!(off_report.workspace_bytes, 0);
    request.veda = Some(VedaRequest::default());
    assert_eq!(run(&generator, &request), dense);
    assert_eq!(
        generator
            .veda_execution_report()
            .unwrap()
            .predictor_load_attempts,
        0
    );
    request.veda = Some(VedaRequest {
        enabled: true,
        ..Default::default()
    });
    let sparse = run(&generator, &request);
    let report = generator.veda_execution_report().unwrap();
    assert_eq!(report.sparse_calls, 2);
    assert!(report.predictor_score_calls > 0);
    assert!(report.selected_token_pairs < report.dense_token_pairs);
    assert!(report.exact_plan);
    assert!(report.fallback_reason.is_none());
    assert_eq!(report.kernel.as_deref(), Some(crate::veda::KERNEL));
    assert_ne!(
        sparse, dense,
        "learned sparse path must affect output assembly"
    );
    generator.fixture.as_mut().unwrap().predictor = false;
    assert_eq!(run(&generator, &request), dense);
    let missing = generator.veda_execution_report().unwrap();
    assert_eq!(missing.sparse_calls, 0);
    assert_eq!(missing.dense_calls, 2);
    assert!(missing
        .fallback_reason
        .unwrap()
        .contains("managed component"));
    generator.fixture.as_mut().unwrap().predictor = true;
    generator.fixture.as_mut().unwrap().fail_self_test = true;
    assert_eq!(run(&generator, &request), dense);
    let failed = generator.veda_execution_report().unwrap();
    assert_eq!(failed.sparse_calls, 0);
    assert_eq!(failed.dense_calls, 2);
    assert!(failed
        .fallback_reason
        .unwrap()
        .contains("self-test forced failure"));
    request.veda = None;
    assert_eq!(run(&generator, &request), dense);
    assert!(generator
        .veda_execution_report()
        .unwrap()
        .fallback_reason
        .is_none());
    println!("S1 native Generator: canonical1344x768/124f, 2 fixture DiT blocks, learned score calls={}, selected/dense={}/{}, dense/off identity, missing/self-test fallbacks verified",report.predictor_score_calls,report.selected_token_pairs,report.dense_token_pairs);
}
#[test]
#[ignore = "requires the exclusive local Metal token"]
fn selected_kernel_actual_device_smoke() {
    crate::veda::device_self_test().unwrap();
}

#[test]
fn predictor_fixture_strict_loader_and_bf16_storage() {
    let p = predictor();
    assert_eq!((p.layers, p.heads, p.dim), (2, 4, 128));
    assert!(p.q.iter().flatten().any(|&v| v != 0));
    let proj = p.projection(0, &[0], true);
    assert!(proj.iter().all(|x| x.to_bits() & 0xffff == 0));
    assert_eq!(p.choose([37, 24, 42]).name, "16x9_t37");
    assert_eq!(p.choose([37, 42, 24]).name, "16x9_t37_T");
}
