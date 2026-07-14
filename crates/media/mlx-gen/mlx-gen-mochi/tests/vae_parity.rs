//! A2 AsymmVAE-decoder parity for Mochi 1 (sc-11985).
//!
//! Two tiers:
//!  - a **committed, non-ignored** CI-green test that builds a tiny random-weight decoder (channels
//!    divisible by GroupNorm's 32, 1 resnet/stage) via the real `from_weights` + forward path and
//!    asserts the decode output shape (temporal 6×, spatial 8×, drop-5-frames) and determinism — no
//!    model weights needed.
//!  - an **`#[ignore]`d** real-weight test that loads the bf16 VAE from `$MOCHI_SNAPSHOT`, feeds the
//!    golden's teacher-forced `denormalized_latents` through the decoder, and checks the decoded
//!    `video` reproduces `mochi_vae_golden.safetensors` (pixel space).
//!
//! Run the real-weight gate:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p mlx-gen-mochi --test vae_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, mean, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mochi::{MochiVaeConfig, MochiVaeDecoder};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mochi_vae_golden.safetensors"
);

fn snapshot_dir() -> std::path::PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// `max|got − want| / max|want|` — peak relative error.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let diff = abs(subtract(&got, &want).unwrap()).unwrap();
    max_abs(&diff) / max_abs(&want).max(1e-12)
}

// ---------------------------------------------------------------- CI-green (no weights)

/// Deterministic small "random" fill for the synthetic weights (bounded so 5 stages of
/// GroupNorm+conv stay well-conditioned).
fn rnd(shape: &[i32], seed: u64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            (((i as u64).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 * 0.000_001).sin()
                * 0.05
        })
        .collect();
    Array::from_slice(&data, shape)
}

/// A tiny, GroupNorm-valid decoder config: 32-wide stages, 1 resnet each, real expansions.
fn tiny_cfg() -> MochiVaeConfig {
    MochiVaeConfig {
        latent_channels: 12,
        out_channels: 3,
        decoder_block_out_channels: vec![32, 32, 32, 32],
        layers_per_block: vec![1, 1, 1, 1, 1],
        temporal_expansions: vec![1, 2, 3],
        spatial_expansions: vec![2, 2, 2],
        latents_mean: vec![0.0; 12],
        latents_std: vec![1.0; 12],
        scaling_factor: 1.0,
    }
}

/// Insert a `MochiResnetBlock3D`'s weights (norm1/2 identity-ish affine, small random convs) at `pfx`.
fn insert_resnet(w: &mut Weights, pfx: &str, ch: i32, seed: u64) {
    for (j, norm) in ["norm1", "norm2"].iter().enumerate() {
        w.insert(
            format!("{pfx}.{norm}.norm_layer.weight"),
            Array::ones::<f32>(&[ch]).unwrap(),
        );
        w.insert(
            format!("{pfx}.{norm}.norm_layer.bias"),
            Array::zeros::<f32>(&[ch]).unwrap(),
        );
        let _ = j;
    }
    for (j, conv) in ["conv1", "conv2"].iter().enumerate() {
        w.insert(
            format!("{pfx}.{conv}.conv.weight"),
            rnd(&[ch, ch, 3, 3, 3], seed + j as u64 * 7 + 1),
        );
        w.insert(
            format!("{pfx}.{conv}.conv.bias"),
            Array::zeros::<f32>(&[ch]).unwrap(),
        );
    }
}

/// Build the full synthetic decoder weight map for [`tiny_cfg`].
fn synthetic_weights(cfg: &MochiVaeConfig) -> Weights {
    let mut w = Weights::empty();
    let c_last = cfg.decoder_block_out_channels[cfg.decoder_block_out_channels.len() - 1] as i32;
    let c_first = cfg.decoder_block_out_channels[0] as i32;
    let lat = cfg.latent_channels as i32;

    // conv_in: plain 1x1x1 conv (latent -> c_last).
    w.insert("decoder.conv_in.weight", rnd(&[c_last, lat, 1, 1, 1], 10));
    w.insert(
        "decoder.conv_in.bias",
        Array::zeros::<f32>(&[c_last]).unwrap(),
    );

    // block_in.
    insert_resnet(&mut w, "decoder.block_in.resnets.0", c_last, 100);

    // up_blocks.
    let n = cfg.decoder_block_out_channels.len();
    let k = cfg.temporal_expansions.len();
    for i in 0..(n - 1) {
        let in_ch = cfg.decoder_block_out_channels[n - 1 - i] as i32;
        let out_ch = cfg.decoder_block_out_channels[n - 2 - i] as i32;
        let t = cfg.temporal_expansions[k - 1 - i] as i32;
        let s = cfg.spatial_expansions[k - 1 - i] as i32;
        let pfx = format!("decoder.up_blocks.{i}");
        insert_resnet(
            &mut w,
            &format!("{pfx}.resnets.0"),
            in_ch,
            200 + i as u64 * 13,
        );
        let proj_out = out_ch * t * s * s;
        w.insert(
            format!("{pfx}.proj.weight"),
            rnd(&[proj_out, in_ch], 300 + i as u64 * 13),
        );
        w.insert(
            format!("{pfx}.proj.bias"),
            Array::zeros::<f32>(&[proj_out]).unwrap(),
        );
    }

    // block_out.
    insert_resnet(&mut w, "decoder.block_out.resnets.0", c_first, 400);

    // proj_out: c_first -> out_channels.
    w.insert(
        "decoder.proj_out.weight",
        rnd(&[cfg.out_channels as i32, c_first], 500),
    );
    w.insert(
        "decoder.proj_out.bias",
        Array::zeros::<f32>(&[cfg.out_channels as i32]).unwrap(),
    );
    w
}

#[test]
fn synthetic_decode_shape_and_determinism() {
    let cfg = tiny_cfg();
    let w = synthetic_weights(&cfg);
    let dec = MochiVaeDecoder::from_weights(&w, &cfg).expect("build synthetic decoder");

    // Teacher-forced latent [B=1, C=12, T_lat=2, H_lat=4, W_lat=4].
    let latent = rnd(&[1, 12, 2, 4, 4], 42);
    let v1 = dec.decode_denormalized(&latent).expect("decode 1");
    let v2 = dec.decode_denormalized(&latent).expect("decode 2");

    // Output: temporal (2-1)*6+1 = 7 frames, spatial 4*8 = 32, out_channels = 3.
    assert_eq!(v1.shape(), &[1, 3, 7, 32, 32], "decode output shape");
    // Determinism: identical across two runs.
    let d = max_abs(&subtract(&v1, &v2).unwrap());
    assert_eq!(d, 0.0, "decode must be deterministic");
    // Sanity: finite output (no NaN/Inf blow-up through the stages).
    assert!(
        max_abs(&v1).is_finite(),
        "decode produced non-finite values"
    );
}

// ------------------------------------------------------------- real-weight golden gate

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (vae bf16 safetensors) + tools/golden/mochi_vae_golden.safetensors"]
fn vae_decode_matches_golden() {
    let root = snapshot_dir();
    let dec = mlx_gen_mochi::load_vae_decoder(&root).expect("load vae decoder");
    let g = Weights::from_file(GOLDEN).expect("vae golden");

    // Teacher-forced: feed the golden's de-normalized latent straight into the decoder.
    let denorm = g.require("denormalized_latents").unwrap();
    let video = dec.decode_denormalized(denorm).expect("decode");

    let want = g.require("video").unwrap();
    assert_eq!(video.shape(), want.shape(), "decoded video shape");

    // The Mochi AsymmVAE is numerically UNSTABLE in bf16 (decoder intermediate activations reach
    // O(100), outside bf16's precise range), so a bf16 decode produces a video far outside [-1, 1].
    // A valid (f32) golden video is ~[-1, 1]; if the provisioned golden is out of range it was dumped
    // in bf16 and is NOT a valid parity target — surface that explicitly rather than gate against it
    // (sc-11985 finding; the A1 dump script has been fixed to decode the VAE in f32).
    let want_range = max_abs(want);
    assert!(
        want_range < 1.1,
        "golden `video` range is ±{want_range:.2} — this is a bf16 decode (the Mochi AsymmVAE is \
         f32-only). Re-provision the golden with the f32-fixed tools/dump_mochi_golden.py \
         (`--stage vae`) before running this gate."
    );

    let pr = peak_rel(&video, want);
    let diff = abs(subtract(
        video.as_dtype(Dtype::Float32).unwrap(),
        want.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap())
    .unwrap();
    let max_px = max_abs(&diff);
    let mean_px = mean(&diff, None).unwrap().item::<f32>();
    eprintln!("VAE decode: peak_rel {pr:.3e}  max_px {max_px:.3e}  mean_px {mean_px:.3e}  (video ~[-1,1])");

    // Cross-impl pixel-space tolerance: MLX f32 decode vs the reference's f32 decode over ~19 resnet
    // blocks + causal convs — a real bug (wrong unpatchify, pad mode, groupnorm, or frame-drop)
    // diverges by O(1) in pixel space. Measured against a fresh torch-CPU f32 reference: max_px 1.5e-3,
    // mean_px 3.0e-4 (Metal-vs-torch f32 accumulation noise). Generous bounds around that.
    assert!(
        mean_px < 5e-3,
        "VAE mean pixel error {mean_px:.3e} too high"
    );
    assert!(max_px < 5e-2, "VAE max pixel error {max_px:.3e} too high");
}
