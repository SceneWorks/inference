//! sc-18773 — the candle twin of `mlx-gen-ltx/tests/upsampler_parity.rs` for the two shipped
//! LTX-2.5 `LatentUpsampler` checkpoints: spatial ×2 and temporal ×2.
//!
//! The goldens are the **same committed fixtures** the MLX side consumes
//! (`mlx-gen-ltx/tests/fixtures/ltx25_{spatial,temporal}_upsampler_golden.safetensors`, from
//! `mlx-gen/tools/dump_ltx25_upsampler_goldens.py` against upstream `ltx_core.model.upsampler` at
//! `Lightricks/LTX-2` @ `d1511477`, v1.2.0). `tests/vae_encode_parity.rs` and
//! `tests/connector_parity.rs` establish that cross-backend golden-reuse convention; it applies
//! here for the same reason — one reference, one fixture, two ports.
//!
//! Deliberately **not** CUDA-gated (same reasoning as `ltx_2_5_vae_real_weights.rs`): the upsampler
//! is convolution + group norm with no bf16 matmul, the goldens are float32, and the geometry is
//! small enough to run in seconds on a CPU. The device used is reported in the test output.
//!
//! Run:
//! ```text
//! LTX25_UPSAMPLER_DIR=<snapshot>/latent_upscale_models \
//!   cargo test -p candle-gen-ltx --release --test integration upsampler_parity:: -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::ltx_checkpoint::{LatentUpsamplerConfig, LatentUpsamplerMode};
use candle_gen_ltx::upsampler::LatentUpsampler;

const SPATIAL: &str = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
const TEMPORAL: &str = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors";

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/mlx-gen-ltx/tests/fixtures")
        .join(name)
}

/// The LTX-2.5 `latent_upscale_models/` directory.
///
/// **Required**, and a hard failure when unset or incomplete: a real-weight gate that silently
/// passes when the weights are absent gates nothing. `#[ignore]` is the only opt-out.
fn upsampler_dir() -> PathBuf {
    let dir =
        PathBuf::from(std::env::var_os("LTX25_UPSAMPLER_DIR").expect(
            "LTX25_UPSAMPLER_DIR must point at the LTX-2.5 latent_upscale_models/ directory",
        ));
    for name in [SPATIAL, TEMPORAL] {
        assert!(
            dir.join(name).is_file(),
            "LTX25_UPSAMPLER_DIR={} does not hold {name}",
            dir.display()
        );
    }
    dir
}

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

fn load(checkpoint: &str, expected: LatentUpsamplerMode, device: &Device) -> LatentUpsampler {
    let path = upsampler_dir().join(checkpoint);
    // f32, matching the golden: the shipped bf16 weights upcast losslessly. `from_checkpoint` is
    // the production constructor, so this exercises the same cross-check production runs.
    let up =
        LatentUpsampler::from_checkpoint(&path, DType::F32, device).expect("build LatentUpsampler");
    assert_eq!(up.mode(), expected, "{checkpoint}");

    // The bare-root config the split-checkpoint loader (sc-18757) reads must agree with the
    // structure the weights imply — two independent authorities on the same fact.
    let config = LatentUpsamplerConfig::from_file(&path).expect("bare upsampler config");
    assert_eq!(config.mode().unwrap(), expected, "{checkpoint} config");
    up.assert_matches_config(&config)
        .expect("weights and config agree");
    up
}

type Golden = std::collections::HashMap<String, Tensor>;

fn golden(name: &str, d: &Device) -> Golden {
    candle_gen::candle_core::safetensors::load(golden_path(name), d)
        .expect("golden (run mlx-gen/tools/dump_ltx25_upsampler_goldens.py)")
}

fn tensor(g: &Golden, key: &str) -> Tensor {
    g.get(key)
        .unwrap_or_else(|| panic!("golden has no {key}"))
        .to_dtype(DType::F32)
        .expect("f32")
}

fn stats(got: &Tensor, want: &Tensor) -> (f32, f32, f32) {
    let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got.len(), want.len());
    let mut peak_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    let mut peak_ref = 0f32;
    for (a, b) in got.iter().zip(&want) {
        let d = (a - b).abs();
        peak_abs = peak_abs.max(d);
        sum_abs += d as f64;
        sum_ref += b.abs() as f64;
        peak_ref = peak_ref.max(b.abs());
    }
    (
        peak_abs,
        peak_abs / peak_ref.max(1e-12),
        (sum_abs / sum_ref.max(1e-12)) as f32,
    )
}

/// `upsample_latents` — un-normalize, upsample, re-normalize, exactly as upstream `upsample_video`.
fn upsample_latents(up: &LatentUpsampler, latent: &Tensor, mean: &Tensor, std: &Tensor) -> Tensor {
    let channels = mean.elem_count();
    let mean = mean.reshape((1, channels, 1, 1, 1)).unwrap();
    let std = std.reshape((1, channels, 1, 1, 1)).unwrap();
    let unnorm = latent
        .broadcast_mul(&std)
        .unwrap()
        .broadcast_add(&mean)
        .unwrap();
    let out = up.forward(&unnorm).expect("upsampler forward");
    out.broadcast_sub(&mean)
        .unwrap()
        .broadcast_div(&std)
        .unwrap()
}

fn golden_parity(up: &LatentUpsampler, name: &str, label: &str, d: &Device) -> Tensor {
    let g = golden(name, d);
    let latent = tensor(&g, "latent");
    let mean = tensor(&g, "latent_mean");
    let std = tensor(&g, "latent_std");
    let want = tensor(&g, "output");

    let got = upsample_latents(up, &latent, &mean, &std);
    assert_eq!(got.dims(), want.dims(), "{label} output shape");
    let (peak_abs, peak_rel, mean_rel) = stats(&got, &want);
    eprintln!(
        "{label} [{d:?}] peak_abs = {peak_abs:.3e} peak_rel = {peak_rel:.3e} \
         mean_rel = {mean_rel:.3e} shape={:?}",
        got.dims()
    );
    // f32 on both sides through the same op graph: anything above ~1e-3 is a real divergence, not
    // accumulation. The absolute bound is stated too, so a golden that collapsed toward zero could
    // not make the relative figures look good.
    assert!(mean_rel < 1e-3, "{label} mean_rel {mean_rel:.3e} too high");
    assert!(peak_rel < 5e-3, "{label} peak_rel {peak_rel:.3e} too high");
    assert!(peak_abs < 5e-3, "{label} peak_abs {peak_abs:.3e} too high");
    got
}

#[test]
#[ignore = "sc-18773: needs the gated LTX-2.5 spatial latent upscaler (1.00 GB)"]
fn ltx_2_5_spatial_upsampler_matches_reference() {
    let d = device();
    let up = load(SPATIAL, LatentUpsamplerMode::Spatial2x, &d);
    let got = golden_parity(
        &up,
        "ltx25_spatial_upsampler_golden.safetensors",
        "ltx25 spatial upsampler",
        &d,
    );
    assert_eq!(got.dims(), &[1, 128, 3, 12, 12]);
}

#[test]
#[ignore = "sc-18773: needs the gated LTX-2.5 temporal latent upscaler (0.26 GB)"]
fn ltx_2_5_temporal_upsampler_matches_reference() {
    let d = device();
    let up = load(TEMPORAL, LatentUpsamplerMode::Temporal2x, &d);
    let got = golden_parity(
        &up,
        "ltx25_temporal_upsampler_golden.safetensors",
        "ltx25 temporal upsampler",
        &d,
    );
    // 9 latent frames in, 2·9−1 = 17 out, spatial untouched.
    assert_eq!(got.dims(), &[1, 128, 17, 6, 6]);
}

#[test]
#[ignore = "sc-18773: needs the gated LTX-2.5 temporal latent upscaler (0.26 GB)"]
fn ltx_2_5_temporal_upsampler_reproduces_the_reference_frame_counts() {
    let d = device();
    let up = load(TEMPORAL, LatentUpsamplerMode::Temporal2x, &d);
    let g = golden("ltx25_temporal_upsampler_golden.safetensors", &d);
    // Rows of `[frames_in, frames_out]` MEASURED on the upstream module by the dump script, so the
    // frame rule is checked against upstream rather than against our own arithmetic.
    let counts = g.get("temporal_frame_counts").expect("frame counts");
    assert_eq!(counts.dim(0).unwrap(), 3, "three frame-count probes");
    let flat = counts
        .flatten_all()
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    for row in flat.chunks(2) {
        let (frames_in, frames_out) = (row[0] as usize, row[1] as usize);
        assert_eq!(frames_in % 8, 1, "probe {frames_in} is an LTX latent size");
        assert_eq!(
            up.output_frames(frames_in).unwrap(),
            frames_out,
            "reference says {frames_in} -> {frames_out}"
        );
        assert_eq!(
            frames_out % 8,
            1,
            "{frames_in} -> {frames_out} breaks n % 8 == 1"
        );
    }
}

// -------------------------------------------------------------------------------------------------
// Frame-count rule on the real forward path, with synthetic weights (no checkpoint needed)
// -------------------------------------------------------------------------------------------------

const IN: usize = 8;
const MID: usize = 64;

/// A deterministic, small-magnitude probe — no RNG, so the tensors are identical on every machine.
fn ramp(dims: &[usize], device: &Device) -> Tensor {
    let n: usize = dims.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017_3).sin() * 0.05)
        .collect();
    Tensor::from_vec(values, dims, device).expect("ramp")
}

/// A minimal `LatentUpsampler` checkpoint in memory: `in_channels` 8, `mid_channels` 64 (two
/// channels per 32-group norm), one `ResBlock` per stage. Structurally identical to the shipped
/// files, small enough to forward many geometries in a unit test.
fn synthetic_upsampler(temporal: bool, device: &Device) -> LatentUpsampler {
    let vb = VarBuilder::from_tensors(synthetic_map(temporal, device), DType::F32, device);
    LatentUpsampler::load(vb).expect("synthetic upsampler")
}

fn synthetic_map(temporal: bool, device: &Device) -> std::collections::HashMap<String, Tensor> {
    let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    let conv3 = |map: &mut std::collections::HashMap<String, Tensor>,
                 prefix: &str,
                 out: usize,
                 inp: usize| {
        map.insert(
            format!("{prefix}.weight"),
            ramp(&[out, inp, 3, 3, 3], device),
        );
        map.insert(format!("{prefix}.bias"), ramp(&[out], device));
    };
    conv3(&mut map, "initial_conv", MID, IN);
    conv3(&mut map, "final_conv", IN, MID);
    for stem in ["res_blocks.0", "post_upsample_res_blocks.0"] {
        conv3(&mut map, &format!("{stem}.conv1"), MID, MID);
        conv3(&mut map, &format!("{stem}.conv2"), MID, MID);
    }
    for prefix in [
        "initial_norm".to_string(),
        "res_blocks.0.norm1".to_string(),
        "res_blocks.0.norm2".to_string(),
        "post_upsample_res_blocks.0.norm1".to_string(),
        "post_upsample_res_blocks.0.norm2".to_string(),
    ] {
        map.insert(format!("{prefix}.weight"), ramp(&[MID], device));
        map.insert(format!("{prefix}.bias"), ramp(&[MID], device));
    }
    if temporal {
        map.insert(
            "upsampler.0.weight".into(),
            ramp(&[MID * 2, MID, 3, 3, 3], device),
        );
        map.insert("upsampler.0.bias".into(), ramp(&[MID * 2], device));
    } else {
        map.insert(
            "upsampler.0.weight".into(),
            ramp(&[MID * 4, MID, 3, 3], device),
        );
        map.insert("upsampler.0.bias".into(), ramp(&[MID * 4], device));
    }
    map
}

/// A checkpoint missing a whole res-block stage must be **refused**, not run shallower.
///
/// `while contains_tensor(stem.i)` returns an empty `Vec` for an absent stage, so without the floor
/// a truncated file loads happily and silently runs a different network — a regression against the
/// hard-coded `(0..4).map(ResBlock::load)` this loader replaced.
#[test]
fn a_truncated_res_block_stage_is_refused() {
    let d = Device::Cpu;
    // `LatentUpsampler` is not `Debug`, so `expect_err` is unavailable.
    fn load_err(vb: VarBuilder, why: &str) -> String {
        match LatentUpsampler::load(vb) {
            Ok(_) => panic!("{why}"),
            Err(e) => e.to_string(),
        }
    }
    for stem in ["res_blocks", "post_upsample_res_blocks"] {
        let mut map = synthetic_map(true, &d);
        map.retain(|k, _| !k.starts_with(&format!("{stem}.")));
        let vb = VarBuilder::from_tensors(map, DType::F32, &d);
        let err = load_err(vb, "a stage-less checkpoint must not load");
        assert!(err.contains(stem), "error must name the stem: {err}");
        assert!(
            err.contains("no residual blocks"),
            "error must say the stage is empty: {err}"
        );
    }

    // Stages that are both present but of different lengths are equally impossible upstream.
    let mut map = synthetic_map(true, &d);
    for prefix in ["conv1", "conv2", "norm1", "norm2"] {
        for suffix in ["weight", "bias"] {
            let src = format!("res_blocks.0.{prefix}.{suffix}");
            let value = map.get(&src).expect("synthetic block key").clone();
            map.insert(format!("res_blocks.1.{prefix}.{suffix}"), value);
        }
    }
    let vb = VarBuilder::from_tensors(map, DType::F32, &d);
    let err = load_err(vb, "lopsided stages must not load");
    assert!(err.contains("res_blocks has 2"), "{err}");
    assert!(err.contains("post_upsample_res_blocks has 1"), "{err}");
}

/// Serialize a synthetic upsampler to a real `.safetensors` file, optionally stamping a
/// `__metadata__["config"]` — the only way to exercise the *file*-taking constructor without a
/// gated 1 GB checkpoint.
fn write_synthetic_checkpoint(path: &std::path::Path, temporal: bool, config: Option<&str>) {
    let map = synthetic_map(temporal, &Device::Cpu);
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let mut header = String::from("{");
    if let Some(config) = config {
        header.push_str(&format!(
            "\"__metadata__\":{{\"config\":{}}},",
            serde_json::json!(config)
        ));
    }
    let mut blob: Vec<u8> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        let t = &map[*key];
        let values = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let start = blob.len();
        for v in &values {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "{}:{{\"dtype\":\"F32\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
            serde_json::json!(key),
            t.dims(),
            start,
            blob.len()
        ));
    }
    header.push('}');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&blob);
    std::fs::write(path, bytes).expect("write synthetic checkpoint");
}

/// `from_checkpoint` — the constructor every production site uses — must run the config
/// cross-check, not just load the tensors. Loading through `load` alone lets the weight rank
/// silently win over a config that disagrees, which is what the two authorities exist to catch.
#[test]
fn from_checkpoint_runs_the_config_cross_check_on_the_production_path() {
    let d = Device::Cpu;
    let dir = tempfile::tempdir().unwrap();
    const TRUTH: &str = r#"{"_class_name":"LatentUpsampler","in_channels":8,"mid_channels":64,
        "num_blocks_per_stage":1,"dims":3,"spatial_upsample":false,"temporal_upsample":true}"#;

    let ok = dir.path().join("ok.safetensors");
    write_synthetic_checkpoint(&ok, true, Some(TRUTH));
    let up = LatentUpsampler::from_checkpoint(&ok, DType::F32, &d).expect("agreeing config loads");
    assert_eq!(up.mode(), LatentUpsamplerMode::Temporal2x);

    // No `__metadata__` at all (every SceneWorks-converted LTX-2.3 tree): still loads.
    let bare = dir.path().join("bare.safetensors");
    write_synthetic_checkpoint(&bare, true, None);
    assert_eq!(
        LatentUpsampler::from_checkpoint(&bare, DType::F32, &d)
            .expect("an unstamped checkpoint still loads")
            .mode(),
        LatentUpsamplerMode::Temporal2x
    );

    for (what, config) in [
        (
            "Spatial2x",
            TRUTH
                .replace(
                    r#""temporal_upsample":true"#,
                    r#""temporal_upsample":false"#,
                )
                .replace(r#""spatial_upsample":false"#, r#""spatial_upsample":true"#),
        ),
        (
            "mid_channels",
            TRUTH.replace(r#""mid_channels":64"#, r#""mid_channels":512"#),
        ),
        (
            "num_blocks_per_stage",
            TRUTH.replace(r#""num_blocks_per_stage":1"#, r#""num_blocks_per_stage":4"#),
        ),
    ] {
        let path = dir.path().join("bad.safetensors");
        write_synthetic_checkpoint(&path, true, Some(&config));
        let err = match LatentUpsampler::from_checkpoint(&path, DType::F32, &d) {
            Ok(_) => panic!("{what}: a disagreeing config must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(what), "{what}: {err}");
    }
}

/// The declared config must agree with the loaded structure on **every** field the two both know,
/// not just on the mode: a config that says 1024 mid channels over a 64-channel file is describing
/// a different checkpoint.
#[test]
fn assert_matches_config_compares_block_count_and_channels_not_just_the_mode() {
    let up = synthetic_upsampler(true, &Device::Cpu);
    let truth = LatentUpsamplerConfig {
        in_channels: IN as u64,
        mid_channels: MID as u64,
        num_blocks_per_stage: 1,
        dims: 3,
        spatial_upsample: false,
        temporal_upsample: true,
        spatial_scale: 1.0,
        rational_resampler: false,
    };
    up.assert_matches_config(&truth)
        .expect("the synthetic file's own config agrees");

    for (what, config) in [
        (
            "num_blocks_per_stage",
            LatentUpsamplerConfig {
                num_blocks_per_stage: 4,
                ..truth
            },
        ),
        (
            "in_channels",
            LatentUpsamplerConfig {
                in_channels: 128,
                ..truth
            },
        ),
        (
            "mid_channels",
            LatentUpsamplerConfig {
                mid_channels: 512,
                ..truth
            },
        ),
    ] {
        let err = up
            .assert_matches_config(&config)
            .expect_err("mismatch must be refused")
            .to_string();
        assert!(err.contains(what), "{what}: {err}");
    }
}

#[test]
fn temporal_upsampling_produces_2n_minus_1_frames_and_keeps_n_mod_8_eq_1() {
    let d = Device::Cpu;
    let up = synthetic_upsampler(true, &d);
    assert_eq!(up.mode(), LatentUpsamplerMode::Temporal2x);
    // Edge sizes: the single-frame floor (a still), the smallest multi-frame LTX latent, and two
    // larger ones. Every input satisfies `n % 8 == 1`, so every output must too.
    for frames in [1usize, 9, 17, 25] {
        let out = up
            .forward(&ramp(&[1, IN, frames, 3, 3], &d))
            .expect("temporal forward");
        let got = out.dim(2).unwrap();
        assert_eq!(got, 2 * frames - 1, "{frames} latent frames -> {got}");
        assert_eq!(got % 8, 1, "{frames} -> {got} breaks n % 8 == 1");
        // Spatial dims are untouched by the temporal branch.
        assert_eq!(&out.dims()[3..], &[3, 3]);
        assert_eq!(out.dim(1).unwrap(), IN, "channels round-trip");
    }
}

#[test]
fn spatial_upsampling_doubles_hw_and_leaves_the_frame_count_alone() {
    let d = Device::Cpu;
    let up = synthetic_upsampler(false, &d);
    assert_eq!(up.mode(), LatentUpsamplerMode::Spatial2x);
    for frames in [1usize, 9, 17] {
        let out = up
            .forward(&ramp(&[1, IN, frames, 3, 3], &d))
            .expect("spatial forward");
        assert_eq!(out.dims(), &[1, IN, frames, 6, 6], "{frames} frames");
        assert_eq!(up.output_frames(frames).unwrap(), frames);
    }
}
