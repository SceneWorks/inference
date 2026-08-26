//! S4 spatial-upsampler parity vs the reference `upsample_latents` (sc-2679 S4).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `upsampler.safetensors` (~1 GB). The committed
//! golden (`tests/fixtures/ltx_upsampler_golden.safetensors`, from
//! `tools/dump_ltx_upsampler_golden.py`) holds the reference **bf16** `upsample_latents` I/O over a
//! synthetic latent; this test loads the SAME bf16 weights and checks the Rust `upsample_latents`
//! reproduces the output.
//!
//! The upsampler is pure dense (conv + group-norm, no quantized ops), run bf16 to match the
//! production path — every op is the same mlx op at the same dtype, so the gate is tight. Honors
//! "divergence is not rounding": a >1% gap here would be a real bug.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test integration upsampler_parity:: -- --ignored --nocapture`
//!
//! # LTX-2.5 (sc-18773)
//!
//! The 2.5 half gates **both** shipped `LatentUpsampler` checkpoints — spatial ×2 and temporal ×2 —
//! against `tools/dump_ltx25_upsampler_goldens.py`, which dumps upstream
//! `ltx_core.model.upsampler`'s own `upsample_video` at `Lightricks/LTX-2` @ `d1511477` (v1.2.0).
//! Those goldens are **float32** (the shipped bf16 weights upcast losslessly on both sides), so the
//! gate is a correctness check, not a bf16 rounding check, and the same fixture serves the candle
//! twin (`candle-gen-ltx/tests/upsampler_parity.rs`) unchanged. The 2.3 golden above stays bf16 —
//! it gates the bf16 production path and must not move.
//!
//! Run:
//! `LTX25_UPSAMPLER_DIR=<snapshot>/latent_upscale_models cargo test -p mlx-gen-ltx --test
//! integration upsampler_parity:: -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::ltx_checkpoint::{LatentUpsamplerConfig, LatentUpsamplerMode};
use mlx_gen::weights::Weights;
use mlx_gen_ltx::upsampler::{upsample_latents, LatentUpsampler};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_upsampler_golden.safetensors"
);

const SPATIAL_2_5: &str = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
const TEMPORAL_2_5: &str = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors";
const SPATIAL_2_5_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx25_spatial_upsampler_golden.safetensors"
);
const TEMPORAL_2_5_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx25_temporal_upsampler_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let denom = max_op(abs(f32(want)).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// `Σ|Δ| / Σ|ref|`.
fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), f32(want)).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(f32(want)).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

/// `max|Δ|` — the absolute error, kept alongside the relative figures so the gate never rests on a
/// scale-invariant statistic alone.
fn peak_abs(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    max_op(&diff, None).unwrap().item::<f32>()
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 upsampler.safetensors (~1 GB)"]
fn upsampler_matches_reference() {
    let dir = base_dir();
    // Same constructor `model.rs` uses for this exact file.
    let up = LatentUpsampler::from_checkpoint(dir.join("upsampler.safetensors"))
        .expect("build LatentUpsampler");

    let g = Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx_upsampler_golden.py)");
    let latent = g.require("latent").unwrap();
    let mean = g.require("latent_mean").unwrap();
    let std = g.require("latent_std").unwrap();
    let want = g.require("output").unwrap();

    let got = upsample_latents(latent, &up, mean, std).expect("upsample_latents");
    assert_eq!(got.shape(), want.shape(), "output shape");
    let (pr, mr) = (peak_rel(&got, want), mean_rel(&got, want));
    eprintln!(
        "upsampler peak_rel = {pr:.3e} mean_rel = {mr:.3e} shape={:?}",
        got.shape()
    );
    // **Bit-identical**, not "close": the golden is mlx's own bf16 `upsample_latents` output and
    // this port replays the same mlx ops at the same dtype in the same order, so there is no step
    // that could round differently. Measured 2026-08-25 against
    // `SceneWorks/ltx-2.3-mlx @ 01df27d3 q8/upsampler.safetensors`: `peak_rel = mean_rel = 0.0`
    // exactly. A tolerance here would let a real op substitution (a different eps, a reassociated
    // reduction, a dropped f32 island) pass as rounding.
    assert_eq!(mr, 0.0, "upsampler mean_rel {mr:.3e} is not bit-identical");
    assert_eq!(pr, 0.0, "upsampler peak_rel {pr:.3e} is not bit-identical");
}

// =================================================================================================
// LTX-2.5 — both shipped upsamplers (sc-18773)
// =================================================================================================

/// The directory holding the LTX-2.5 `latent_upscale_models/` component checkpoints.
///
/// **Required**, and a hard failure when unset or incomplete: a real-weight gate that silently
/// passes when the weights are absent gates nothing. `#[ignore]` is the only opt-out.
fn upsampler_dir_2_5() -> std::path::PathBuf {
    let dir =
        std::path::PathBuf::from(std::env::var_os("LTX25_UPSAMPLER_DIR").expect(
            "LTX25_UPSAMPLER_DIR must point at the LTX-2.5 latent_upscale_models/ directory",
        ));
    for name in [SPATIAL_2_5, TEMPORAL_2_5] {
        assert!(
            dir.join(name).is_file(),
            "LTX25_UPSAMPLER_DIR={} does not hold {name}",
            dir.display()
        );
    }
    dir
}

/// Load a 2.5 upsampler at **f32** (the golden's dtype: the shipped bf16 weights upcast losslessly)
/// and cross-check the structure read from the weights against the config the file declares.
fn load_2_5(checkpoint: &str, expected: LatentUpsamplerMode) -> LatentUpsampler {
    let path = upsampler_dir_2_5().join(checkpoint);
    // `from_checkpoint` is the production constructor: it runs the config cross-check itself, so
    // this exercises the same path production takes rather than a test-only variant.
    LatentUpsampler::from_checkpoint(&path).expect("from_checkpoint");

    let mut w = Weights::from_file(&path).expect("2.5 upsampler weights");
    w.cast_all(Dtype::Float32).expect("cast to f32");
    let up = LatentUpsampler::from_weights(&w).expect("build LatentUpsampler");
    assert_eq!(up.mode(), expected, "{checkpoint}");

    // The bare-root config the split-checkpoint loader (sc-18757) reads must agree with the
    // structure the weights imply — two independent authorities on the same fact.
    let config = LatentUpsamplerConfig::from_file(&path).expect("bare upsampler config");
    assert_eq!(config.mode().unwrap(), expected, "{checkpoint} config");
    up.assert_matches_config(&config)
        .expect("weights and config agree");
    up
}

fn golden_parity(up: &LatentUpsampler, golden_path: &str, label: &str) -> Array {
    let g = Weights::from_file(golden_path)
        .expect("golden (run tools/dump_ltx25_upsampler_goldens.py)");
    let latent = f32(g.require("latent").unwrap());
    let mean = f32(g.require("latent_mean").unwrap());
    let std = f32(g.require("latent_std").unwrap());
    let want = f32(g.require("output").unwrap());

    let got = upsample_latents(&latent, up, &mean, &std).expect("upsample_latents");
    assert_eq!(got.shape(), want.shape(), "{label} output shape");
    let (pa, pr, mr) = (
        peak_abs(&got, &want),
        peak_rel(&got, &want),
        mean_rel(&got, &want),
    );
    eprintln!(
        "{label} peak_abs = {pa:.3e} peak_rel = {pr:.3e} mean_rel = {mr:.3e} shape={:?}",
        got.shape()
    );
    // f32 on both sides through the same op graph: anything above ~1e-3 is a real divergence, not
    // accumulation. The absolute bound is stated too, so a golden that collapsed toward zero could
    // not make the relative figures look good.
    assert!(mr < 1e-3, "{label} mean_rel {mr:.3e} too high");
    assert!(pr < 5e-3, "{label} peak_rel {pr:.3e} too high");
    assert!(pa < 5e-3, "{label} peak_abs {pa:.3e} too high");
    got
}

#[test]
#[ignore = "sc-18773: needs the gated LTX-2.5 spatial latent upscaler (1.00 GB)"]
fn ltx_2_5_spatial_upsampler_matches_reference() {
    let up = load_2_5(SPATIAL_2_5, LatentUpsamplerMode::Spatial2x);
    let got = golden_parity(&up, SPATIAL_2_5_GOLDEN, "ltx25 spatial upsampler");
    // Spatial is the frame-preserving branch: `H,W` double, `F` does not move.
    assert_eq!(got.shape(), &[1, 128, 3, 12, 12]);
}

#[test]
#[ignore = "sc-18773: needs the gated LTX-2.5 temporal latent upscaler (0.26 GB)"]
fn ltx_2_5_temporal_upsampler_matches_reference() {
    let up = load_2_5(TEMPORAL_2_5, LatentUpsamplerMode::Temporal2x);
    let got = golden_parity(&up, TEMPORAL_2_5_GOLDEN, "ltx25 temporal upsampler");
    // 9 latent frames in, 2·9−1 = 17 out, spatial untouched.
    assert_eq!(got.shape(), &[1, 128, 17, 6, 6]);
}

#[test]
#[ignore = "sc-18773: needs the gated LTX-2.5 temporal latent upscaler (0.26 GB)"]
fn ltx_2_5_temporal_upsampler_reproduces_the_reference_frame_counts() {
    let up = load_2_5(TEMPORAL_2_5, LatentUpsamplerMode::Temporal2x);
    let g = Weights::from_file(TEMPORAL_2_5_GOLDEN).expect("temporal golden");
    // Rows of `[frames_in, frames_out]` MEASURED on the upstream module by the dump script, so the
    // frame rule is checked against upstream rather than against our own arithmetic.
    let counts = g.require("temporal_frame_counts").unwrap();
    let rows = counts.shape()[0];
    assert_eq!(rows, 3, "golden records three frame-count probes");
    let flat: Vec<i32> = counts
        .reshape(&[rows * 2])
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
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

/// A deterministic, small-magnitude probe — no RNG, so the tensors are identical on every machine.
fn ramp(shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017_3).sin() * 0.05)
        .collect();
    Array::from_slice(&values, shape)
}

/// A minimal `LatentUpsampler` checkpoint in memory: `in_channels` 8, `mid_channels` 64 (two
/// channels per 32-group norm), one `ResBlock` per stage. Small enough to forward many geometries
/// in a unit test, structurally identical to the shipped files.
fn synthetic_upsampler(temporal: bool) -> LatentUpsampler {
    LatentUpsampler::from_weights(&Weights::from_map(synthetic_map(temporal)))
        .expect("synthetic upsampler")
}

fn synthetic_map(temporal: bool) -> std::collections::HashMap<String, Array> {
    const IN: i32 = 8;
    const MID: i32 = 64;
    fn conv3(map: &mut std::collections::HashMap<String, Array>, prefix: &str, out: i32, inp: i32) {
        map.insert(format!("{prefix}.weight"), ramp(&[out, inp, 3, 3, 3]));
        map.insert(format!("{prefix}.bias"), ramp(&[out]));
    }
    fn norm(map: &mut std::collections::HashMap<String, Array>, prefix: &str, channels: i32) {
        map.insert(format!("{prefix}.weight"), ramp(&[channels]));
        map.insert(format!("{prefix}.bias"), ramp(&[channels]));
    }
    let mut map: std::collections::HashMap<String, Array> = std::collections::HashMap::new();
    conv3(&mut map, "initial_conv", MID, IN);
    norm(&mut map, "initial_norm", MID);
    conv3(&mut map, "final_conv", IN, MID);
    for stem in ["res_blocks.0", "post_upsample_res_blocks.0"] {
        conv3(&mut map, &format!("{stem}.conv1"), MID, MID);
        conv3(&mut map, &format!("{stem}.conv2"), MID, MID);
        norm(&mut map, &format!("{stem}.norm1"), MID);
        norm(&mut map, &format!("{stem}.norm2"), MID);
    }
    if temporal {
        // Conv3d mid -> 2*mid: the temporal branch.
        map.insert("upsampler.0.weight".into(), ramp(&[MID * 2, MID, 3, 3, 3]));
        map.insert("upsampler.0.bias".into(), ramp(&[MID * 2]));
    } else {
        // Conv2d mid -> 4*mid: the spatial branch.
        map.insert("upsampler.0.weight".into(), ramp(&[MID * 4, MID, 3, 3]));
        map.insert("upsampler.0.bias".into(), ramp(&[MID * 4]));
    }
    map
}

/// A checkpoint missing a whole res-block stage must be **refused**, not run shallower.
///
/// `while contains(stem.i)` returns an empty `Vec` for an absent stage, so without the floor a
/// truncated file loads happily and silently runs a different network against real weights.
#[test]
fn a_truncated_res_block_stage_is_refused() {
    // `LatentUpsampler` is not `Debug`, so `expect_err` is unavailable.
    fn load_err(map: std::collections::HashMap<String, Array>, why: &str) -> String {
        match LatentUpsampler::from_weights(&Weights::from_map(map)) {
            Ok(_) => panic!("{why}"),
            Err(e) => e.to_string(),
        }
    }
    for stem in ["res_blocks", "post_upsample_res_blocks"] {
        let mut map = synthetic_map(true);
        map.retain(|k, _| !k.starts_with(&format!("{stem}.")));
        let err = load_err(map, "a stage-less checkpoint must not load");
        assert!(err.contains(stem), "error must name the stem: {err}");
        assert!(
            err.contains("no residual blocks"),
            "error must say the stage is empty: {err}"
        );
    }

    // Stages that are both present but of different lengths are equally impossible upstream.
    let mut map = synthetic_map(true);
    for prefix in ["conv1", "conv2", "norm1", "norm2"] {
        for suffix in ["weight", "bias"] {
            let src = format!("res_blocks.0.{prefix}.{suffix}");
            let value = map.get(&src).expect("synthetic block key").clone();
            map.insert(format!("res_blocks.1.{prefix}.{suffix}"), value);
        }
    }
    let err = load_err(map, "lopsided stages must not load");
    assert!(err.contains("res_blocks has 2"), "{err}");
    assert!(err.contains("post_upsample_res_blocks has 1"), "{err}");
}

/// Serialize a synthetic upsampler to a real `.safetensors` file, optionally stamping a
/// `__metadata__["config"]` — the only way to exercise the *file*-taking constructor without a
/// gated 1 GB checkpoint.
fn write_synthetic_checkpoint(path: &std::path::Path, temporal: bool, config: Option<&str>) {
    let map = synthetic_map(temporal);
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
        let a = &map[*key];
        let values: Vec<f32> = a.as_slice::<f32>().to_vec();
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
            a.shape().to_vec(),
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
/// cross-check, not just load the tensors. Loading through `from_weights` alone lets the weight
/// rank silently win over a config that disagrees, which is what this file's two authorities exist
/// to catch.
#[test]
fn from_checkpoint_runs_the_config_cross_check_on_the_production_path() {
    let dir = tempfile::tempdir().unwrap();
    const TRUTH: &str = r#"{"_class_name":"LatentUpsampler","in_channels":8,"mid_channels":64,
        "num_blocks_per_stage":1,"dims":3,"spatial_upsample":false,"temporal_upsample":true}"#;

    // Agreeing config: loads.
    let ok = dir.path().join("ok.safetensors");
    write_synthetic_checkpoint(&ok, true, Some(TRUTH));
    let up = LatentUpsampler::from_checkpoint(&ok).expect("agreeing config loads");
    assert_eq!(up.mode(), LatentUpsamplerMode::Temporal2x);

    // No `__metadata__` at all (every SceneWorks-converted LTX-2.3 tree): still loads.
    let bare = dir.path().join("bare.safetensors");
    write_synthetic_checkpoint(&bare, true, None);
    assert_eq!(
        LatentUpsampler::from_checkpoint(&bare)
            .expect("an unstamped checkpoint still loads")
            .mode(),
        LatentUpsamplerMode::Temporal2x
    );

    // Each disagreement must be refused *by the path constructor*.
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
        let err = match LatentUpsampler::from_checkpoint(&path) {
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
    let up = synthetic_upsampler(true);
    let truth = LatentUpsamplerConfig {
        in_channels: 8,
        mid_channels: 64,
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

/// `PixelShuffle1d(2)` over an odd channel count would otherwise reshape into a wrong-sized view.
#[test]
fn the_temporal_pixel_shuffle_refuses_an_indivisible_channel_count() {
    let mut map = synthetic_map(true);
    // 129 output channels: rank 5 (still the temporal branch) but not divisible by the shuffle
    // factor 2.
    map.insert("upsampler.0.weight".into(), ramp(&[129, 64, 3, 3, 3]));
    map.insert("upsampler.0.bias".into(), ramp(&[129]));
    let up = LatentUpsampler::from_weights(&Weights::from_map(map)).expect("loads");
    let err = up
        .forward(&ramp(&[1, 8, 9, 3, 3]))
        .expect_err("odd channel count must be refused")
        .to_string();
    assert!(err.contains("PixelShuffle1d(2)"), "{err}");
    assert!(err.contains("129"), "{err}");
}

/// The spatial twin of the guard above (the candle port has both).
#[test]
fn the_spatial_pixel_shuffle_refuses_an_indivisible_channel_count() {
    let mut map = synthetic_map(false);
    // 257 output channels: rank 4 (still the spatial branch) but not divisible by 2².
    map.insert("upsampler.0.weight".into(), ramp(&[257, 64, 3, 3]));
    map.insert("upsampler.0.bias".into(), ramp(&[257]));
    let up = LatentUpsampler::from_weights(&Weights::from_map(map)).expect("loads");
    let err = match up.forward(&ramp(&[1, 8, 9, 3, 3])) {
        Ok(_) => panic!("an indivisible channel count must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("PixelShuffle(2)"), "{err}");
    assert!(err.contains("257"), "{err}");
}

#[test]
fn temporal_upsampling_produces_2n_minus_1_frames_and_keeps_n_mod_8_eq_1() {
    let up = synthetic_upsampler(true);
    assert_eq!(up.mode(), LatentUpsamplerMode::Temporal2x);
    // Edge sizes: the single-frame floor (a still), the smallest multi-frame LTX latent, and two
    // larger ones. Every input satisfies `n % 8 == 1`, so every output must too.
    for frames in [1i32, 9, 17, 25] {
        let latent = ramp(&[1, 8, frames, 3, 3]);
        let out = up.forward(&latent).expect("temporal forward");
        let got = out.shape()[2];
        assert_eq!(
            got,
            2 * frames - 1,
            "{frames} latent frames -> {got}, expected {}",
            2 * frames - 1
        );
        assert_eq!(got % 8, 1, "{frames} -> {got} breaks n % 8 == 1");
        // Spatial dims are untouched by the temporal branch.
        assert_eq!(&out.shape()[3..], &[3, 3]);
        assert_eq!(out.shape()[1], 8, "channels round-trip through final_conv");
    }
}

#[test]
fn spatial_upsampling_doubles_hw_and_leaves_the_frame_count_alone() {
    let up = synthetic_upsampler(false);
    assert_eq!(up.mode(), LatentUpsamplerMode::Spatial2x);
    for frames in [1i32, 9, 17] {
        let out = up
            .forward(&ramp(&[1, 8, frames, 3, 3]))
            .expect("spatial forward");
        assert_eq!(out.shape(), &[1, 8, frames, 6, 6], "{frames} frames");
        assert_eq!(up.output_frames(frames as usize).unwrap(), frames as usize);
    }
}
