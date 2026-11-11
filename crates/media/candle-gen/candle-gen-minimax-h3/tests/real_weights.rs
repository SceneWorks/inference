//! sc-17154: real-weight smokes for the candle MiniMax-H3 VAE decode paths.
//!
//! These are `#[ignore]`d — they need the ~9.7 GB `vae/` and ~0.6 GB `audio_vae/` components of a
//! `MiniMaxAI/MiniMax-H3` snapshot. Point `MINIMAX_H3_SNAPSHOT` at the snapshot root (the directory
//! holding `vae/`) and run:
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root> \
//!   cargo test -p candle-gen-minimax-h3 --test integration real_weights:: -- --ignored --nocapture
//! ```
//!
//! On the self-hosted Windows CUDA runner (sc-18677) the snapshot is provisioned at
//! `E:\huggingface\hub\models--MiniMaxAI--MiniMax-H3\snapshots\939557dc319dd91227e30195a763f272ba7f8765`.
//! Everything loads at **f32** so the numbers are comparable with the committed-fixture lane. That
//! is FREE rather than expensive, which earlier revisions of this note had backwards: both
//! components ship F32 already — `vae/` is 703 of 703 tensors F32 and `audio_vae/` 1087 of 1087,
//! measured from the published headers — so `DType::F32` is a no-op cast, not a doubling. The
//! decode halves these tests load are **9.03 GiB** (585 `decoder.*` + `post_quant_conv.*` tensors)
//! and **0.242 GiB** (914 `dec_in_proj.*` + `decoder.*`), where this file used to claim ~19.4 GB
//! and ~1.2 GB. The bf16 component in this checkpoint is `transformer/`, which none of these load.
//! Several of the tests below read only safetensors headers and cost no weight I/O at all.
//!
//! **A skipped run must not look like a passing one.** An `#[ignore]`d test that returns early when
//! its input is missing prints `ok` in 0.00s, which reads exactly like success. Every test here
//! therefore *asserts* on the snapshot (see `common::snapshot`) and then asserts on evidence that
//! the model actually executed — the published tensor count, the decoded shape, and the fact that
//! the output is finite and non-constant. None of those can hold unless real weights were read.

use crate::common;

use std::collections::BTreeSet;

use common::{flat, l2_norm, read_snapshot_file, rel, safetensors_keys, snapshot, std_dev};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::loader::sorted_safetensors;
use candle_gen::testkit::{tensor_cosine_f32_sums, Denom};
use candle_gen_minimax_h3::{
    kaiser_sinc_filter1d, MiniMaxH3AudioVae, MiniMaxH3AudioVaeConfig, MiniMaxH3VaeConfig,
    MiniMaxH3VideoVae, TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};

/// Total tensors in the published `vae/` component.
const VAE_TENSOR_COUNT: usize = 703;
/// Of those, the encode half: 116 `encoder.*` + 2 `quant_conv.*` (sc-19008).
const VAE_ENCODE_TENSORS: usize = 118;

/// Peak-relative bound for the real-weight encode against the official diffusers implementation.
///
/// **Derived from this arm's own measured floor, not inherited.** Both sides run f32 over the same
/// published bytes, so this is a round-off comparison over a 12-resnet conv stack — but the
/// reference is torch and this is candle, with different reduction orders in every convolution, so
/// it cannot be held to the committed fixture suite's 1e-5 either.
///
/// **Measured on TWO devices** (sc-19455), three runs each, `--release`, f32 on both sides:
///
/// | probe | mean (CPU / Metal) | std (CPU / Metal) |
/// |---|---|---|
/// | `keyframe` (1 frame, tiled at 320 px) | 1.355e-5 / 1.787e-5 | 5.690e-6 / 4.243e-6 |
/// | `clip` (17 frames, untiled) | 6.949e-6 / 6.412e-6 | 1.137e-5 / 1.309e-5 |
/// | `multiclip` (20 frames, padded to 2 clips) | 4.899e-6 / 6.446e-6 | 1.730e-5 / **1.848e-5** |
///
/// Each device is **bit-deterministic**: three runs apiece reproduced every figure to all four
/// printed digits, so the within-device spread is zero and the only spread that exists is the
/// between-device one. The floor over both is **1.848e-5**, and this bound is **10.8x** it.
///
/// **What a device change actually costs — measured, not hedged.** An earlier revision of this note
/// justified the headroom by saying the figures were CPU-only while "this arm also runs on the
/// `--features cuda` Windows runner (sc-18677)". Both halves needed correcting. The reach claim is
/// **false**: no lane selects this test. The Windows CUDA job in `real-weights.yml` runs seven
/// header gates, three decode smokes and the vram probe, and its own header lists this test as not
/// covered, blocked on the `MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE` operator artifact that has only
/// ever been produced on `nax-macos`. The cost itself is now observed rather than reserved against:
/// swapping CPU → Metal — a real backend change, different reduction orders in every convolution —
/// moved the binding quantity by only **1.07x**, and no single probe by more than **1.32x**. So the
/// bound clears the measured envelope of a device swap by ~8x, which is why it is left where
/// sc-19008 put it rather than tightened. A per-device bound would be false precision at a 1.07x
/// separation: one bound weakens neither device.
///
/// ⚠️ **THE CUDA ARM IS UNMEASURED, AND THAT IS A DECISION RATHER THAN AN OVERSIGHT** (sc-19455).
/// The two columns above are the only two devices this test has ever run on. CUDA is not among
/// them, no lane executes this test at all, and **none is being added**: wiring one would put a
/// real-weight comparison on every run of a lane, which is the cost the "no new gates" direction is
/// deliberately removing (the same ruling stripped `cuda_quant_smoke`'s wiring in sc-19545). So the
/// honest statement is the one recorded here — *nothing measures this on CUDA* — rather than a
/// number implied by a bound that was never taken there. Do not read the 10.8x headroom as a CUDA
/// claim; read it as a two-device measurement plus a 1.07x observed cost for one backend swap.
///
/// If that ever needs closing, it is small and it is not a research task: place the ~5.4 MB
/// reference (`crates/media/mlx-gen/tools/dump_minimax_h3_video_vae_encode_real.py`, run against
/// the snapshot on a box that has torch) on the Windows runner, point
/// `MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE` at it, and add one `call :run_one` line for this test to
/// the existing `candle-minimax-h3` job. The snapshot side is already done — `vars`
/// `CANDLE_MINIMAX_H3_SNAPSHOT` **does** exist (set 2026-08-15), so that job's own header comment
/// claiming "the variable does not exist on this repository" is stale; the encode reference is the
/// only genuine blocker left.
///
/// **The `std`-amplification story is half right, and the wrong half was load-bearing.** `std` is
/// `exp(0.5 * logvar)` (`vae_encoder.rs`), and against a *systematic* drift it really does amplify:
/// a 1e-3 relative drift injected at `DiagonalGaussian::from_parameters` comes back as 1.000e-3 on
/// every `mean` probe but **1.42e-3 … 1.87e-3** on the `std` probes, on both devices. The `std`
/// probes do also set the floor — `multiclip.std` is the largest residual on CPU and on Metal
/// alike. What does **not** hold is the inference drawn from those two facts, that `std` is
/// therefore the quantity most likely to move under a device change. Measured, it is not: the two
/// largest movers CPU → Metal are `keyframe.mean` and `multiclip.mean` (1.32x each), while
/// `keyframe.std` *fell*, to 0.75x. A reduction-order change is incoherent round-off, not a
/// coherent multiplicative drift in `logvar`, so it does not propagate through the exponential the
/// way an injected drift does. Expect a device change to move the `mean` probes at least as much.
///
/// What this bound is *not* is the 2e-2 house value it replaced (sc-19008 review): that sat 1155x
/// above the measurement and so stated a convention in the voice of a derivation. The band between
/// the two is not hypothetical — the 1e-3 drift above measures 1.862e-3 (CPU) / 1.866e-3 (Metal) at
/// its worst, trips this bound on **both** devices at the first probe it reaches, and **passes** at
/// 2e-2. If a third device ever lands above this, the fix is to re-derive from a measurement taken
/// there, not to restore a convention.
///
/// Every defect class this epic has actually shipped is still orders clear either way: the sc-18740
/// half-swap at 0.86-0.99, a symmetric downsampler pad at 1.8, a global GroupNorm at 1.6, an
/// interleaved-vs-contiguous GroupNorm grouping at 1.1, an untiled encode at ~5e-1, a front pad at
/// 6.9e-1, and a first-frame-instead-of-last repeat at 2.8e-1.
const REAL_WEIGHT_ENCODE_TOL: f32 = 2e-4;
/// Total tensors in the published `audio_vae/` component (encode + decode).
const AUDIO_TENSOR_COUNT: usize = 1087;
/// Of those, the decode half this crate ports.
const AUDIO_DECODE_TENSORS: usize = 914;

/// Where real-weight decodes run. On a `--features cuda` build this is the GPU; on the CPU lane it
/// is the CPU, and either way the dtype is f32 so the residuals are comparable with the
/// committed-fixture suite.
fn device() -> Device {
    candle_gen::default_device().expect("a candle device")
}

fn shard_keys(dir: &std::path::Path, label: &str) -> BTreeSet<String> {
    let files = sorted_safetensors(dir, label).expect("safetensors shards");
    assert!(!files.is_empty(), "{}: no shards", dir.display());
    let mut keys = BTreeSet::new();
    for file in &files {
        for k in safetensors_keys(file) {
            assert!(keys.insert(k.clone()), "duplicate tensor {k} across shards");
        }
    }
    keys
}

/// `name -> dtype` over a component's shard headers, read without any weight I/O.
///
/// Used by the DiT checks to turn `crate::dit::heads`' **mixed-precision** claim — twelve of the 17
/// top-level tensors ship float32 while everything else ships bfloat16 — into a property asserted
/// against the real checkpoint. That claim is why `LinearBias` loads at the *stored* dtype rather
/// than taking one as a parameter, and nothing in the all-f32 fixture can check it.
fn shard_dtypes(dir: &std::path::Path, label: &str) -> std::collections::BTreeMap<String, String> {
    let files = sorted_safetensors(dir, label).expect("safetensors shards");
    let mut out = std::collections::BTreeMap::new();
    for file in &files {
        let mut f = std::fs::File::open(file).expect("open shard");
        let mut len = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut len).expect("header length");
        let hlen = u64::from_le_bytes(len) as usize;
        let mut header = vec![0u8; hlen];
        std::io::Read::read_exact(&mut f, &mut header).expect("header");
        let json: serde_json::Value = serde_json::from_slice(&header).expect("header json");
        for (k, v) in json.as_object().expect("header object") {
            if k == "__metadata__" {
                continue;
            }
            out.insert(k.clone(), v["dtype"].as_str().expect("dtype").to_string());
        }
    }
    out
}

/// `name -> shape` over a component's shard headers, read without any weight I/O.
///
/// The key-set proof cannot see a tensor read at the wrong *level*; this is what turns the
/// encoder's declared geometry into an assertion against the published bytes (sc-19008).
fn shard_shapes(
    dir: &std::path::Path,
    label: &str,
) -> std::collections::BTreeMap<String, Vec<usize>> {
    let files = sorted_safetensors(dir, label).expect("safetensors shards");
    let mut out = std::collections::BTreeMap::new();
    for file in &files {
        let mut f = std::fs::File::open(file).expect("open shard");
        let mut len = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut len).expect("header length");
        let hlen = u64::from_le_bytes(len) as usize;
        let mut header = vec![0u8; hlen];
        std::io::Read::read_exact(&mut f, &mut header).expect("header");
        let json: serde_json::Value = serde_json::from_slice(&header).expect("header json");
        for (k, v) in json.as_object().expect("header object") {
            if k == "__metadata__" {
                continue;
            }
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|d| d.as_u64().expect("dim") as usize)
                .collect();
            out.insert(k.clone(), shape);
        }
    }
    out
}

// =============================================================================================
// Video VAE
// =============================================================================================

/// **The declared key set must be EXACTLY the published checkpoint's — all 703, both halves.**
/// Reads only the safetensors headers, so it costs no weight I/O.
///
/// This is the exhaustive-mapping proof against the real model rather than against the tiny
/// fixture: a tensor the loader never reads would encode or decode to something plausible but
/// wrong, and a tensor the loader *asks for* that the checkpoint lacks would fail at load with no
/// clue which of 703 names was invented.
///
/// The assertion is set equality in **both directions** — the port declares nothing the checkpoint
/// does not have, and the checkpoint has nothing the port does not consume — and the two halves
/// are additionally asserted to partition that set at 585 / 118 (sc-19008 raised the declared half
/// from 585 to the whole file).
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn declared_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let published = shard_keys(&root.join("vae"), "minimax-h3 vae");
    assert_eq!(
        published.len(),
        VAE_TENSOR_COUNT,
        "the published vae/ component should carry {VAE_TENSOR_COUNT} tensors"
    );

    let cfg = MiniMaxH3VaeConfig::from_diffusers_json(&read_snapshot_file(
        &root.join("vae").join("config.json"),
    ))
    .expect("parse vae/config.json");
    assert_eq!(cfg, MiniMaxH3VaeConfig::default(), "the shipped geometry");

    let declared: BTreeSet<String> = MiniMaxH3VideoVae::tensor_names(&cfg).into_iter().collect();
    assert_eq!(
        declared.len(),
        VAE_TENSOR_COUNT,
        "the declared set must name each tensor exactly once"
    );

    // Set equality, reported per direction so a failure says WHICH names are wrong.
    let unmapped: Vec<&String> = published.difference(&declared).collect();
    assert!(
        unmapped.is_empty(),
        "published vae/ tensors the port never consumes: {unmapped:?}"
    );
    let invented: Vec<&String> = declared.difference(&published).collect();
    assert!(
        invented.is_empty(),
        "the port declares tensors the published vae/ does not have: {invented:?}"
    );
    assert_eq!(declared, published);

    // …and the two halves partition it.
    let encode: BTreeSet<String> = published
        .iter()
        .filter(|k| k.starts_with("encoder.") || k.starts_with("quant_conv."))
        .cloned()
        .collect();
    assert_eq!(encode.len(), VAE_ENCODE_TENSORS);
    let declared_encode: BTreeSet<String> = MiniMaxH3VideoVae::encode_tensor_names(&cfg)
        .into_iter()
        .collect();
    assert_eq!(
        declared_encode, encode,
        "the declared ENCODE key set must be exactly the published encoder.* / quant_conv.* half"
    );
    let declared_decode: BTreeSet<String> = MiniMaxH3VideoVae::decode_tensor_names(&cfg)
        .into_iter()
        .collect();
    assert_eq!(declared_decode.len(), VAE_TENSOR_COUNT - VAE_ENCODE_TENSORS);
    assert_eq!(
        &declared_decode | &declared_encode,
        published,
        "the two halves must cover the published index with nothing left over"
    );
    assert!(
        (&declared_decode & &declared_encode).is_empty(),
        "the two halves must be disjoint"
    );

    // The encode half's own structure, against the real index rather than the fixture's: four
    // downsamplers (levels 0..3), none on 4 or 5, and a residual projection exactly where the
    // width changes (levels 1, 3, 5).
    let has = |k: &str| published.contains(k);
    for level in 0..4 {
        assert!(
            has(&format!(
                "encoder.down_blocks.{level}.downsamplers.0.conv.weight"
            )),
            "level {level} downsamples"
        );
    }
    for level in [4, 5] {
        assert!(
            !published
                .iter()
                .any(|k| k.starts_with(&format!("encoder.down_blocks.{level}.downsamplers"))),
            "level {level} carries no downsampler in the published checkpoint"
        );
    }
    for level in [1, 3, 5] {
        assert!(
            has(&format!(
                "encoder.down_blocks.{level}.resnets.0.conv_shortcut.weight"
            )),
            "level {level} changes width"
        );
    }
    for level in [0, 2, 4] {
        assert!(
            !published
                .iter()
                .any(|k| k.contains(&format!("down_blocks.{level}.resnets.0.conv_shortcut"))),
            "level {level} keeps its width"
        );
    }
    println!(
        "video vae: {} published tensors, {} declared and consumed ({} decode + {} encode); \
         nothing unmapped, nothing invented",
        published.len(),
        declared.len(),
        declared_decode.len(),
        declared_encode.len()
    );
}

/// Every declared encoder tensor's **shape** must be the one the loader's geometry implies.
///
/// The key-set proof above is necessary and not sufficient: a port could name all 118 keys and
/// still read them at the wrong level (a 512-wide conv where the checkpoint has 1024), which loads
/// cleanly in candle only because `Weights` hands back whatever is stored. Reads headers only.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn declared_encoder_shapes_match_the_published_checkpoint() {
    let root = snapshot();
    let dir = root.join("vae");
    let cfg =
        MiniMaxH3VaeConfig::from_diffusers_json(&read_snapshot_file(&dir.join("config.json")))
            .expect("parse vae/config.json");
    let shapes = shard_shapes(&dir, "minimax-h3 vae");

    // The geometry, restated from the config so the expectations are derived and not typed twice.
    let mut want: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    let c0 = cfg.block_out_channels[0];
    want.insert(
        "encoder.conv_in.weight".into(),
        vec![c0, cfg.in_channels, 3, 3, 3],
    );
    want.insert("encoder.conv_in.bias".into(), vec![c0]);
    for level in 0..cfg.num_encoder_levels() {
        let out = cfg.block_out_channels[level];
        for j in 0..cfg.layers_per_block {
            let inp = if j == 0 {
                cfg.encoder_level_in_channels(level)
            } else {
                out
            };
            let p = format!("encoder.down_blocks.{level}.resnets.{j}");
            want.insert(format!("{p}.norm1.weight"), vec![inp]);
            want.insert(format!("{p}.norm1.bias"), vec![inp]);
            want.insert(format!("{p}.conv1.weight"), vec![out, inp, 3, 3, 3]);
            want.insert(format!("{p}.conv1.bias"), vec![out]);
            want.insert(format!("{p}.norm2.weight"), vec![out]);
            want.insert(format!("{p}.norm2.bias"), vec![out]);
            want.insert(format!("{p}.conv2.weight"), vec![out, out, 3, 3, 3]);
            want.insert(format!("{p}.conv2.bias"), vec![out]);
            if inp != out {
                want.insert(format!("{p}.conv_shortcut.weight"), vec![out, inp, 1, 1, 1]);
                want.insert(format!("{p}.conv_shortcut.bias"), vec![out]);
            }
        }
        if cfg.encoder_level_has_downsampler(level) {
            let p = format!("encoder.down_blocks.{level}.downsamplers.0.conv");
            want.insert(format!("{p}.weight"), vec![out, out, 3, 3, 3]);
            want.insert(format!("{p}.bias"), vec![out]);
        }
    }
    let last = *cfg.block_out_channels.last().expect("levels");
    want.insert("encoder.norm_out.weight".into(), vec![last]);
    want.insert("encoder.norm_out.bias".into(), vec![last]);
    let params = 2 * cfg.latent_channels;
    want.insert(
        "encoder.conv_out.weight".into(),
        vec![params, last, 3, 3, 3],
    );
    want.insert("encoder.conv_out.bias".into(), vec![params]);
    want.insert("quant_conv.weight".into(), vec![params, params, 1, 1, 1]);
    want.insert("quant_conv.bias".into(), vec![params]);

    assert_eq!(
        want.len(),
        VAE_ENCODE_TENSORS,
        "the derived expectation must cover all {VAE_ENCODE_TENSORS} encode tensors"
    );
    let declared: BTreeSet<String> = MiniMaxH3VideoVae::encode_tensor_names(&cfg)
        .into_iter()
        .collect();
    assert_eq!(
        want.keys().cloned().collect::<BTreeSet<_>>(),
        declared,
        "the shape table and the declared key set must describe the same tensors"
    );

    for (key, expected) in &want {
        let got = shapes
            .get(key)
            .unwrap_or_else(|| panic!("the published vae/ has no `{key}`"));
        assert_eq!(
            got, expected,
            "{key}: published shape vs the port's geometry"
        );
    }
    // Every 3x3x3 kernel really is 3x3x3 — a 1x3x3 would make the whole temporal-causality
    // discussion vacuous, and `conv_shortcut` really is the pointwise 1x1x1.
    let kernel = |k: &str| shapes[k][2..].to_vec();
    assert_eq!(kernel("encoder.conv_in.weight"), vec![3, 3, 3]);
    assert_eq!(
        kernel("encoder.down_blocks.0.downsamplers.0.conv.weight"),
        vec![3, 3, 3]
    );
    assert_eq!(
        kernel("encoder.down_blocks.1.resnets.0.conv_shortcut.weight"),
        vec![1, 1, 1]
    );
    println!(
        "video vae encode half: all {} declared shapes match the published checkpoint",
        want.len()
    );
}

/// Load the real 36-layer decoder and decode a small latent end to end.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); 9.03 GiB of all-F32 decode-half parameters"]
fn real_weight_decode_produces_a_plausible_video() {
    let root = snapshot();
    let dev = device();
    let vae = MiniMaxH3VideoVae::load_decode_only(&root, &dev, DType::F32)
        .expect("load the real video VAE");
    let cfg = vae.config().clone();
    assert_eq!(cfg.num_layers, 36, "the real 36-layer transformer decoder");
    assert_eq!(cfg.dim(), 2048);

    // 5 temporal tokens of a 2x3 latent grid -> 17 frames at 32x48 px.
    let (t, h, w) = (5usize, 2usize, 3usize);
    let n = cfg.latent_channels * t * h * w;
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 0.8).collect();
    let latent = Tensor::from_vec(vals, (1, cfg.latent_channels, t, h, w), &dev).expect("latent");

    let video = vae.decode(&latent).expect("decode");
    assert_eq!(
        video.dims(),
        &[1, 3, 17, h * 16, w * 16],
        "16x spatial / 4x temporal upsampling with the clip_length-17 chunk plan"
    );

    let px = flat(&video);
    assert!(
        px.iter().all(|v| v.is_finite()),
        "decode produced non-finite pixels"
    );
    let spread = std_dev(&video);
    println!(
        "real video decode: {:?}, std {spread:.4}, |max| {:.4}",
        video.dims(),
        px.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );
    assert!(
        spread > 1e-3,
        "the decode is ~constant ({spread:.3e}); real weights would not produce that"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_decode_produces_a_plausible_video");
}

/// **The gate a self-generated checksum cannot be.** Decode the *same latent* the official
/// diffusers `AutoencoderKLMiniMaxH3` decoded, from the *same published bytes*, and compare
/// numerically.
///
/// Why the smoke above cannot do this job: it asserts `std` and finiteness, which any
/// plausible-but-wrong port also satisfies — the sc-18740 gate/value half-swap changed neither.
/// Reproducing an independent implementation's output on real weights is the only assertion that
/// can catch a layout error, because layout errors are invisible to shape, magnitude and checksum.
///
/// Generate the reference with the MLX lane's `tools/dump_minimax_h3_video_vae_real.py` (a few
/// hundred KB, deliberately not committed) and point `MINIMAX_H3_VIDEO_VAE_REFERENCE` at it. Like
/// every test in this file it **asserts** rather than skipping, so a missing reference cannot read
/// as a pass.
#[test]
#[ignore = "needs a real snapshot + a reference decode (MINIMAX_H3_VIDEO_VAE_REFERENCE)"]
fn real_weight_decode_matches_the_official_diffusers_vae() {
    let root = snapshot();
    let raw = std::env::var("MINIMAX_H3_VIDEO_VAE_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_REFERENCE must point at a reference decode produced by \
         tools/dump_minimax_h3_video_vae_real.py. This test asserts rather than skips so a missing \
         reference cannot be mistaken for a pass."
    );
    let reference = common::Golden::load(&raw);
    for key in ["in.latent", "out.video"] {
        assert!(
            reference.has(key),
            "{raw} has no `{key}`; it is not a MiniMax-H3 real-weight reference decode"
        );
    }

    let dev = device();
    let vae = MiniMaxH3VideoVae::load_decode_only(&root, &dev, DType::F32)
        .expect("load the real video VAE");
    let latent = reference
        .tensor("in.latent")
        .to_device(&dev)
        .expect("latent");
    let want = reference.tensor("out.video");
    let got = vae
        .decode(&latent)
        .expect("decode")
        .to_device(&Device::Cpu)
        .expect("to cpu");

    assert_eq!(got.dims(), want.dims(), "decoded shape");
    let (peak, mean) = rel(&got, &want);
    let cos = tensor_cosine_f32_sums(&got, &want, Denom::FloorAt(1e-12));
    println!(
        "real-weight decode vs diffusers AutoencoderKLMiniMaxH3: peak rel {peak:.3e}, mean rel \
         {mean:.3e}, cosine {cos:.6}"
    );
    // The reference runs bf16 or f32 depending on how it was dumped; 2e-2 is the bar the MLX lane
    // uses for the same comparison, and the sc-18740 half-swap sits at 0.86-0.99 — two orders
    // clear of it either way.
    assert!(
        peak < 2e-2,
        "the candle decode diverges from the official diffusers VAE by {peak:.3e}; a gated-FFN \
         half-swap or a QKV mis-split lands here (sc-18740)"
    );
    assert!(
        std_dev(&want) > 1e-3,
        "the reference decode is ~constant; it would gate nothing"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_decode_matches_the_official_diffusers_vae");
}

// ---------------------------------------------------------------------------------------------
// The encode half (sc-19008)
// ---------------------------------------------------------------------------------------------

/// **The gate a self-generated checksum cannot be, for the encode half.** Encode the *same
/// pixels* the official diffusers `AutoencoderKLMiniMaxH3` encoded, from the *same published
/// bytes*, and compare numerically.
///
/// This is the assertion the committed tiny fixture cannot make. That fixture is a 32-channel,
/// four-level toy at `norm_num_groups == block_out_channels`, which makes every GroupNorm an
/// *instance* norm; the shipped encoder runs 32 groups over 128…1024 channels with six levels,
/// two resnets each, and its `conv_shortcut` at levels 1/3/5 rather than 3. A port can reproduce
/// the toy exactly and still index the real stack wrong.
///
/// It is also the only place the shipped **256/64 spatial tiling** runs on the real weights: the
/// keyframe golden is 320 px, so the reference genuinely tiled it, and the tiled result differs
/// from an untiled one by far more than this bound.
///
/// The three probes cover three different paths, and the third exists because the other two — and
/// every committed fixture, all of which are 1, 5 or exactly 17 frames — leave it dark:
///
/// | probe | frames | what only it reaches |
/// |---|---|---|
/// | `keyframe` | 1 | the single-frame short circuit, and the 256/64 spatial tiling at 320 px |
/// | `clip` | 17 | the temporal strides and the `token_drop` tail trim, untiled |
/// | `multiclip` | 20 | the **frame-repeat pad** to a multiple of `clip_length` and the clip-by-clip concatenation |
///
/// 20 is deliberately ragged: it pads by 14 to 34 = 2 clips. A front pad, a zero pad, or a
/// `token_drop` applied per-clip instead of once at the tail all yield the identical *shape*, so
/// the shape assertions below cannot separate them and only the reference values can (sc-19008
/// review).
///
/// **20 rather than any other ragged count**, because `token_drop` otherwise hides the pad: it
/// keeps only the first 2 of the final clip's 5 latent frames, which reach back over clip-local
/// pixels `0 ..= pad_reach` only, so a final clip with more than `pad_reach` real frames pads
/// entirely inside the dropped tail. At 25 frames, repeating the *first* frame instead of the last
/// leaves the encode **bit-identical** — measured, not reasoned. The assertion below derives
/// `pad_reach` from the shipped config so it cannot silently stop holding.
///
/// Generate the reference with the MLX lane's `tools/dump_minimax_h3_video_vae_encode_real.py`
/// (a few MB, deliberately not committed) and point `MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE` at it.
/// Like every test in this file it **asserts** rather than skipping, so a missing reference cannot
/// read as a pass.
#[test]
#[ignore = "needs a real snapshot + a reference encode (MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE)"]
fn real_weight_encode_matches_the_official_diffusers_vae() {
    let root = snapshot();
    let raw = std::env::var("MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE must point at a reference encode produced by \
         tools/dump_minimax_h3_video_vae_encode_real.py. This test asserts rather than skips so a \
         missing reference cannot be mistaken for a pass."
    );
    let reference = common::Golden::load(&raw);
    // Rule 3: only a golden from the CONVERTED checkpoint path can validate this loader.
    assert_eq!(
        reference.meta("provenance"),
        Some("converted-checkpoint"),
        "the reference must come from `AutoencoderKLMiniMaxH3`, not the MiniMax source modules"
    );
    assert_eq!(
        reference.meta("reference"),
        Some("diffusers.AutoencoderKLMiniMaxH3")
    );
    assert_eq!(reference.meta("half"), Some("encode"));
    // The reference must have run the SHIPPED tile geometry, not some other one.
    assert_eq!(
        reference.meta("tile_sample_min_size").map(str::to_string),
        Some(TILE_SAMPLE_MIN_SIZE.to_string()),
        "the reference tiled at a different size than this port does"
    );
    assert_eq!(
        reference
            .meta("tile_sample_min_overlap")
            .map(str::to_string),
        Some(TILE_SAMPLE_MIN_OVERLAP.to_string())
    );

    let dev = device();
    let vae = MiniMaxH3VideoVae::load(&root, &dev, DType::F32).expect("load the real video VAE");
    assert!(
        vae.can_encode(),
        "`load` must materialize the encode half — the published vae/ always ships it"
    );
    let cfg = vae.config().clone();
    assert_eq!(cfg.num_encoder_levels(), 6, "the real six-level encoder");
    assert_eq!(cfg.block_out_channels, vec![128, 256, 256, 512, 512, 1024]);

    let mut worst = 0.0f32;
    for name in ["keyframe", "clip", "multiclip"] {
        let pixels = reference
            .tensor(&format!("in.{name}.pixels"))
            .to_device(&dev)
            .expect("pixels");
        let want_mean = reference.tensor(&format!("out.{name}.mean"));
        let want_std = reference.tensor(&format!("out.{name}.std"));

        let posterior = vae.encode(&pixels).expect("encode");
        let got_mean = posterior
            .mean()
            .to_device(&Device::Cpu)
            .expect("mean to cpu");
        let got_std = posterior.std().to_device(&Device::Cpu).expect("std to cpu");

        assert_eq!(got_mean.dims(), want_mean.dims(), "{name}: posterior shape");
        // The compression is `pixel / ratio` with NO cropping, on the real 16x stack.
        assert_eq!(
            got_mean.dims()[4],
            pixels.dims()[4] / cfg.patch_size,
            "{name}: the encoder is cropping rather than compressing"
        );

        for (label, got, want) in [
            ("mean", &got_mean, &want_mean),
            ("std", &got_std, &want_std),
        ] {
            let (peak, mean) = rel(got, want);
            let cos = tensor_cosine_f32_sums(got, want, Denom::FloorAt(1e-12));
            println!(
                "real-weight encode {name}.{label} vs diffusers AutoencoderKLMiniMaxH3: peak rel \
                 {peak:.3e}, mean rel {mean:.3e}, cosine {cos:.7}"
            );
            // **Gated on the peak relative max-abs-diff**, never on the cosine printed beside it:
            // sc-18740's half-swap held cosine at 0.73-0.78 while being wrong by 0.86-0.99 here.
            assert!(
                peak < REAL_WEIGHT_ENCODE_TOL,
                "the candle encode of `{name}.{label}` diverges from the official diffusers VAE \
                 by {peak:.3e}; a wrong pad, a global GroupNorm, an un-tiled encode or a \
                 mis-levelled conv lands here"
            );
            worst = worst.max(peak);
        }
        assert!(
            std_dev(&want_mean) > 1e-3,
            "{name}: the reference posterior is ~constant; it would gate nothing"
        );
    }
    // A keyframe is ONE latent frame; a 17-frame clip is more. The short circuit is live on real
    // weights, not only in the fixture.
    assert_eq!(reference.shape("out.keyframe.mean")[2], 1);
    assert!(reference.shape("out.clip.mean")[2] > 1);

    // The multiclip probe has to actually be ragged, or the two branches it exists to gate are
    // never entered and the comparison above is a third copy of the `clip` probe. Both facts are
    // read off the reference and the shipped config rather than restated as literals.
    let clip_length = usize::try_from(cfg.clip_length).expect("a positive clip_length");
    let frames = reference.shape("in.multiclip.pixels")[2];
    assert!(
        !frames.is_multiple_of(clip_length),
        "the multiclip probe is {frames} frames, a whole number of {clip_length}-frame clips; it \
         never reaches the frame-repeat pad"
    );
    let clips = frames.div_ceil(clip_length);
    assert!(
        clips > 1,
        "the multiclip probe pads to {clips} clip(s); it never reaches the clip-by-clip \
         concatenation"
    );
    // `clips` chunks of `ceil(clip_length / patch_size_t)` latent frames, `token_drop` removed ONCE
    // at the tail. A per-clip drop, or a concat that kept only the last chunk, lands elsewhere.
    let per_clip = clip_length.div_ceil(cfg.patch_size_t);
    let drop = usize::try_from(cfg.token_drop).expect("a non-negative token_drop");
    // **And the repeated frames must be observable.** `token_drop` removes all but the first
    // `per_clip - drop` latent frames of the final clip, and those reach back over clip-local
    // pixels `0 ..= (per_clip - drop - 1) * patch_size_t` only. A ragged count whose final clip
    // carries more real frames than that pads *entirely inside the dropped tail*: measured, at 25
    // frames repeating the FIRST frame instead of the LAST leaves the encode bit-identical, so a
    // 25-frame probe would gate the pad's position but not its content. 20 leaves 3.
    let pad_reach = (per_clip - drop - 1) * cfg.patch_size_t;
    let tail_real = frames % clip_length;
    assert!(
        tail_real <= pad_reach,
        "the multiclip probe leaves {tail_real} real frames in its final clip, past the \
         {pad_reach}-pixel reach of the latents that survive token_drop; every repeated frame \
         would be dropped and the probe could not tell a last-frame repeat from any other"
    );
    assert_eq!(
        reference.shape("out.multiclip.mean")[2],
        clips * per_clip - drop,
        "a {frames}-frame encode is {clips} clips of {per_clip} latent frames less the {drop} \
         dropped once at the tail"
    );
    println!(
        "real-weight encode: worst peak-relative {worst:.3e} (bound {REAL_WEIGHT_ENCODE_TOL:.0e})"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_encode_matches_the_official_diffusers_vae");
}

/// **Spatial tiling is not free on the real stack.** The keyframe reference is 320 px, so the
/// official encode tiled it; encoding the same pixels untiled must give a measurably different
/// answer, or this port's agreement with the reference above would say nothing about whether it
/// tiles at all.
#[test]
#[ignore = "needs a real snapshot + a reference encode (MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE)"]
fn real_weight_tiling_changes_the_encode() {
    let root = snapshot();
    let raw = std::env::var("MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE must point at a reference encode; this test \
         asserts rather than skips"
    );
    let reference = common::Golden::load(&raw);
    let pixels_shape = reference.shape("in.keyframe.pixels");
    assert!(
        pixels_shape[3] > TILE_SAMPLE_MIN_SIZE,
        "the keyframe golden is {}px and must exceed the {TILE_SAMPLE_MIN_SIZE}px tile, or the \
         reference did not tile it",
        pixels_shape[3]
    );

    let dev = device();
    let vae = MiniMaxH3VideoVae::load(&root, &dev, DType::F32).expect("load the real video VAE");
    let pixels = reference
        .tensor("in.keyframe.pixels")
        .to_device(&dev)
        .expect("pixels");

    let tiled = vae.encode_clip(&pixels).expect("tiled encode");
    let untiled = vae
        .encode_clip_tiled(&pixels, 4096, TILE_SAMPLE_MIN_OVERLAP)
        .expect("untiled encode");
    assert_eq!(
        tiled.dims(),
        untiled.dims(),
        "tiling must not change the latent SHAPE — only shape-blind gates would notice if it did"
    );
    let (delta, _) = rel(&untiled, &tiled);
    println!("real-weight tiled vs untiled keyframe encode: peak-rel {delta:.3e}");
    assert!(
        delta > 1e-2,
        "an untiled encode of a {}px canvas agrees with the tiled one to {delta:.3e}; the \
         reference comparison cannot be said to gate the tiling",
        pixels_shape[3]
    );

    // …and the tiled result is the one that matches the reference.
    let want = reference.tensor("out.keyframe.mean");
    let got = tiled
        .to_device(&Device::Cpu)
        .expect("to cpu")
        .narrow(1, 0, want.dims()[1])
        .expect("mean half");
    let (peak, _) = rel(&got, &want);
    assert!(
        peak < REAL_WEIGHT_ENCODE_TOL,
        "the TILED encode is the one the reference produced, but it differs by {peak:.3e}"
    );

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_tiling_changes_the_encode"
    );
}

/// The multi-chunk path runs on real weights: 12 tokens span two chunks joined by the 5-frame
/// cross-fade, so this exercises the blend and the chunk-advance arithmetic that the single-chunk
/// smoke above never reaches.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); 9.03 GiB of all-F32 decode-half parameters"]
fn real_weight_multi_chunk_decode_blends_the_seam() {
    let root = snapshot();
    let dev = device();
    let vae = MiniMaxH3VideoVae::load_decode_only(&root, &dev, DType::F32)
        .expect("load the real video VAE");
    let g = vae.geometry();
    assert_eq!(g.tokens_chunk_size, 5);
    assert_eq!(g.frame_overlap, 5, "the cross-faded seam");

    let (t, h, w) = (12usize, 2usize, 2usize);
    let cfg = vae.config();
    let n = cfg.latent_channels * t * h * w;
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.21).cos() * 0.7).collect();
    let latent = Tensor::from_vec(vals, (1, cfg.latent_channels, t, h, w), &dev).expect("latent");

    let video = vae.decode(&latent).expect("decode");
    assert_eq!(
        video.dims(),
        &[1, 3, 39, h * 16, w * 16],
        "12 tokens decode to 39 frames across two chunks"
    );
    assert!(flat(&video).iter().all(|v| v.is_finite()));

    // A cross-faded seam must not be a visible discontinuity: the frame-to-frame delta across the
    // seam should be the same order as elsewhere. Frames 17..22 carry the blend.
    let frame = |i: usize| flat(&video.narrow(2, i, 1).expect("frame"));
    let delta = |a: usize, b: usize| {
        frame(a)
            .iter()
            .zip(frame(b).iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let seam = delta(16, 17).max(delta(21, 22));
    let interior = delta(5, 6).max(delta(30, 31));
    println!("real multi-chunk decode: seam delta {seam:.4}, interior delta {interior:.4}");
    assert!(
        seam < interior * 10.0,
        "the chunk seam ({seam:.4}) is an order louder than ordinary frame-to-frame motion \
         ({interior:.4}); the cross-fade is mis-aligned"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_multi_chunk_decode_blends_the_seam");
}

// =============================================================================================
// Audio VAE
// =============================================================================================

/// The reference's own `from_pretrained` inputs, under `FL2VA/audio_vae/`.
fn audio_source_dir(root: &std::path::Path) -> std::path::PathBuf {
    let dir = root.join("FL2VA").join("audio_vae");
    assert!(
        dir.join("config.json").is_file(),
        "{} has no config.json; the snapshot is missing the FL2VA audio_vae source bundle",
        dir.display()
    );
    dir
}

/// The published documents must parse to exactly [`MiniMaxH3AudioVaeConfig::default`], and the
/// diffusers-repackaged root config must agree with them.
///
/// This is the check that the constructor kwargs really come from `metadata.json` / `config.yaml`
/// rather than from a hardcoded table that happens to match: change either file and this fails. It
/// reads no weights.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn published_audio_configs_reproduce_the_declared_geometry() {
    let root = snapshot();
    let src = audio_source_dir(&root);
    let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
        &read_snapshot_file(&src.join("config.json")),
        &read_snapshot_file(&src.join("config.yaml")),
        &read_snapshot_file(&src.join("metadata.json")),
    )
    .expect("parse the FL2VA audio source triple");
    assert_eq!(
        cfg,
        MiniMaxH3AudioVaeConfig::default(),
        "the published documents must reproduce the declared config exactly"
    );

    cfg.cross_check_diffusers_json(&read_snapshot_file(
        &root.join("audio_vae").join("config.json"),
    ))
    .expect("the repackaged root config must agree with the source triple");

    println!(
        "audio vae: sample_rate {}, latent_dim {}, hop {}, {} upsample stages",
        cfg.sample_rate,
        cfg.bigvgan.num_mels,
        cfg.hop_length(),
        cfg.bigvgan.num_upsamples()
    );
}

/// The declared decode-path key set must be EXACTLY the published checkpoint's, minus the encode
/// half — asserted against BOTH published weight files, whose tensor names must also agree with
/// each other. Reads only the safetensors headers, so it costs no weight I/O.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn declared_audio_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let repackaged = shard_keys(&root.join("audio_vae"), "minimax-h3 audio_vae");
    let source = shard_keys(&audio_source_dir(&root), "minimax-h3 FL2VA audio_vae");
    assert_eq!(
        repackaged, source,
        "the two published audio VAE forms must carry identical tensor names (the conversion is \
         an identity — sc-18740's audit result)"
    );
    assert_eq!(repackaged.len(), AUDIO_TENSOR_COUNT);

    let declared: BTreeSet<String> =
        MiniMaxH3AudioVae::tensor_names(&MiniMaxH3AudioVaeConfig::default())
            .into_iter()
            .collect();
    assert_eq!(declared.len(), AUDIO_DECODE_TENSORS);
    assert!(
        declared.is_subset(&repackaged),
        "declared audio tensors absent from the checkpoint: {:?}",
        declared.difference(&repackaged).collect::<Vec<_>>()
    );

    // The remainder is the encode half, and it is exactly what the module docs claim.
    let unread: Vec<&String> = repackaged.difference(&declared).collect();
    assert_eq!(unread.len(), AUDIO_TENSOR_COUNT - AUDIO_DECODE_TENSORS);
    assert!(
        unread.iter().all(|k| k.starts_with("encoder.")
            || k.starts_with("mean_proj.")
            || k.starts_with("logs_proj.")
            || k.starts_with("pre_block.")),
        "an unread tensor is outside the encode half: {unread:?}"
    );
    println!(
        "audio vae: {} published tensors, {} declared and consumed, {} encode-half",
        repackaged.len(),
        declared.len(),
        unread.len()
    );
}

/// Load the real ~605 MB audio VAE and decode a stereo latent end to end.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); 0.242 GiB of all-F32 decode-half parameters"]
fn real_weight_audio_decode_produces_a_plausible_stereo_track() {
    let root = snapshot();
    let dev = device();
    let cfg = MiniMaxH3AudioVaeConfig::default();
    let files =
        sorted_safetensors(&root.join("audio_vae"), "minimax-h3 audio_vae").expect("audio shards");
    let w = candle_gen::Weights::from_files_filtered(
        &files,
        &dev,
        DType::F32,
        &["dec_in_proj.", "decoder."],
    )
    .expect("load the audio decode half");
    let vae = MiniMaxH3AudioVae::from_weights(&w, &cfg, &dev, DType::F32)
        .expect("build the real audio VAE");

    // 40 latent tokens = 1.0 s of 32 kHz stereo.
    let tokens = 40usize;
    let n = 2 * cfg.latent_channels * tokens;
    let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin() * 0.9).collect();
    let z = Tensor::from_vec(vals, (1, 2, cfg.latent_channels, tokens), &dev).expect("latent");

    let track = vae.decode_audio_track(&z).expect("decode_audio_track");
    assert_eq!(track.sample_rate, 32_000);
    assert_eq!(track.channels, 2);
    assert!(track.stems.is_empty());
    assert_eq!(
        track.samples.len(),
        tokens * 800 * 2,
        "40 tokens at 40 Hz = 1 s of interleaved stereo at 32 kHz"
    );
    assert!(track.samples.iter().all(|s| s.is_finite()));
    assert!(track.samples.iter().all(|s| (-1.0..=1.0).contains(s)));

    let mean = track.samples.iter().sum::<f32>() / track.samples.len() as f32;
    let spread = (track
        .samples
        .iter()
        .map(|s| (s - mean) * (s - mean))
        .sum::<f32>()
        / track.samples.len() as f32)
        .sqrt();
    let saturated = track
        .samples
        .iter()
        .filter(|s| s.abs() >= 1.0 - 1e-6)
        .count() as f32
        / track.samples.len() as f32;
    println!(
        "real audio decode: {} samples, std {spread:.4}, {:.2}% clamped",
        track.samples.len(),
        saturated * 100.0
    );
    assert!(spread > 1e-3, "the waveform is ~silent ({spread:.3e})");

    // The two channels must be genuinely different — a mono-duplicating port would pass every
    // shape and finiteness assertion above.
    let gap = (0..tokens * 800)
        .map(|t| (track.samples[2 * t] - track.samples[2 * t + 1]).abs())
        .fold(0.0f32, f32::max);
    println!("  L-vs-R peak gap {gap:.4}");
    assert!(
        gap > 1e-3,
        "the two decoded channels are near-identical ({gap:.3e})"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_audio_decode_produces_a_plausible_stereo_track");
}

/// The Kaiser-sinc taps the port DERIVES must reproduce the ones the real checkpoint STORES.
///
/// The loader reads the stored buffers, so without this the derivation would only ever be checked
/// against the tiny fixture. Reads a handful of 12-float buffers, so it costs no meaningful I/O.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT)"]
fn stored_kaiser_filters_match_the_derivation_on_real_weights() {
    let root = snapshot();
    let files =
        sorted_safetensors(&root.join("audio_vae"), "minimax-h3 audio_vae").expect("audio shards");
    let w = candle_gen::Weights::from_files_filtered(
        &files,
        &Device::Cpu,
        DType::F32,
        &[
            "decoder.activation_post.",
            "decoder.resblocks.0.activations.0.",
        ],
    )
    .expect("load the stored filters");

    let derived = kaiser_sinc_filter1d(0.25, 0.3, 12, &Device::Cpu).expect("derive");
    let mut checked = 0usize;
    for key in [
        "decoder.activation_post.upsample.filter",
        "decoder.activation_post.downsample.lowpass.filter",
        "decoder.resblocks.0.activations.0.upsample.filter",
        "decoder.resblocks.0.activations.0.downsample.lowpass.filter",
    ] {
        let stored = w.require(key).expect(key);
        assert_eq!(stored.dims(), &[1, 1, 12], "{key} is not a 12-tap buffer");
        let (peak, _) = rel(&derived, &stored);
        println!("  {key}: peak rel {peak:.3e}");
        assert!(
            peak < 1e-5,
            "{key} disagrees with `kaiser_sinc_filter1d` by {peak:.3e}; the derivation and the \
             shipped buffers describe different filters"
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "every filter must have been read");
}

// =============================================================================================
// DiT (sc-17155)
// =============================================================================================

/// Tensors the published `transformer/` partition carries: `50 · 12 + 21 + 17`.
const DIT_TENSOR_COUNT: usize = 638;

/// **The DiT's exhaustive-mapping proof against the real checkpoint.** Reads only the safetensors
/// headers, so it costs no weight I/O and no GPU — the same class as
/// `declared_tensor_names_match_the_published_checkpoint` above.
///
/// This is the one check the committed fixture structurally cannot give. `dit_block.safetensors` is
/// dumped at **2** layers and 2 refiner layers; the shipped model has **50** and 2. So the fixture
/// can prove the per-block name pattern and the top-level set, but it cannot prove the stack is
/// enumerated to the right depth, nor that the published partition holds nothing outside the three
/// declared groups. A tensor the loader never reads still produces plausible output.
///
/// It also re-verifies against real bytes the finding `crate::dit::qkv` documents as a *contract*:
/// the published checkpoint ships **no** fused QKV, because the conversion already split it. That
/// claim is what justifies `DitAttention` applying no transform at all, and it is invisible in the
/// port.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); headers only, no weight I/O"]
fn declared_dit_tensor_names_match_the_published_checkpoint() {
    use candle_gen_minimax_h3::{MiniMaxH3Dit, MiniMaxH3DitConfig};

    let root = snapshot();
    let dir = root.join("transformer");
    let published = shard_keys(&dir, "minimax-h3 transformer");
    assert_eq!(
        published.len(),
        DIT_TENSOR_COUNT,
        "the published transformer/ partition should carry {DIT_TENSOR_COUNT} tensors"
    );

    let cfg =
        MiniMaxH3DitConfig::from_diffusers_json(&read_snapshot_file(&dir.join("config.json")))
            .expect("parse transformer/config.json");
    assert_eq!(
        cfg,
        MiniMaxH3DitConfig::default(),
        "the shipped geometry — 50 layers, hidden 5376, 56 heads x 128"
    );

    let declared: BTreeSet<String> = MiniMaxH3Dit::names(&cfg).into_iter().collect();
    assert_eq!(
        declared, published,
        "the declared DiT key set must be exactly the published one — nothing unread, nothing \
         invented"
    );

    // The three groups partition it at the real depth, not the fixture's.
    let blocks = published
        .iter()
        .filter(|k| k.starts_with("transformer_blocks."))
        .count();
    let refiner = published
        .iter()
        .filter(|k| k.starts_with("token_refiner."))
        .count();
    assert_eq!(
        (blocks, refiner, published.len() - blocks - refiner),
        (600, 21, 17)
    );

    // `crate::dit::qkv`'s contract, against the real index: the conversion already split the fused
    // projection, so nothing named `qkv_proj` / `to_qkv` survives and the port applies no transform.
    let fused: Vec<&String> = published
        .iter()
        .filter(|k| k.contains("qkv_proj") || k.contains("to_qkv"))
        .collect();
    assert!(
        fused.is_empty(),
        "the published DiT must ship no fused QKV; found {fused:?}"
    );
    for i in 0..cfg.num_layers {
        for part in ["to_q", "to_k", "to_v"] {
            assert!(published.contains(&format!("transformer_blocks.{i}.attn.{part}.weight")));
        }
    }

    // **The mixed-precision claim, against the real bytes.** `crate::dit::heads` says twelve of the
    // 17 top-level tensors ship float32 and everything else bfloat16, and that is why `LinearBias`
    // loads at the STORED dtype instead of taking one as a parameter. The all-f32 fixture cannot
    // check it; a port that cast the timestep MLP to bf16 would round the one tensor every AdaLN
    // projection in the model reads, biasing all 50 blocks identically at every step.
    let dtypes = shard_dtypes(&dir, "minimax-h3 transformer");
    let f32_names: BTreeSet<&String> = dtypes
        .iter()
        .filter(|(_, d)| d.as_str() == "F32")
        .map(|(k, _)| k)
        .collect();
    assert_eq!(f32_names.len(), 12, "twelve float32 tensors: {f32_names:?}");
    for prefix in [
        "proj_in.",
        "audio_proj_in.",
        "time_embedder.",
        "proj_out.",
        "audio_proj_out.",
    ] {
        assert!(
            f32_names.iter().any(|k| k.starts_with(prefix)),
            "{prefix} should be among the float32-kept modules"
        );
    }
    assert!(
        f32_names
            .iter()
            .all(|k| !k.starts_with("transformer_blocks.") && !k.starts_with("token_refiner.")),
        "the float32 set is entirely top-level; the block stack is bfloat16"
    );
    assert_eq!(
        dtypes.values().filter(|d| d.as_str() == "BF16").count(),
        DIT_TENSOR_COUNT - 12
    );

    println!(
        "dit: {} published tensors = {blocks} block + {refiner} refiner + {} top-level; 0 fused \
         QKV; all {} declared names consumed; dtypes 12 F32 (top-level only) + {} BF16",
        published.len(),
        published.len() - blocks - refiner,
        declared.len(),
        DIT_TENSOR_COUNT - 12
    );
}

/// The AdaLN eviction's headline number, computed from the **real** shard headers rather than from
/// the config arithmetic: `50 × (96768·2688 + 96768)` at the published dtype.
///
/// Header-only, so it costs no weight I/O. It is what turns 26_020_915_200 B from a number in a doc
/// comment into a property of the checkpoint on disk — `dit::adaln`'s unit test asserts the
/// arithmetic, and this asserts the arithmetic describes the real tensors.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); headers only, no weight I/O"]
fn the_adaln_projections_are_the_documented_twenty_six_gigabytes() {
    use candle_gen_minimax_h3::MiniMaxH3DitConfig;

    let root = snapshot();
    let dir = root.join("transformer");
    let cfg =
        MiniMaxH3DitConfig::from_diffusers_json(&read_snapshot_file(&dir.join("config.json")))
            .expect("parse transformer/config.json");

    let published = shard_keys(&dir, "minimax-h3 transformer");
    let adaln: Vec<&String> = published
        .iter()
        .filter(|k| k.contains(".adaln_proj.linear."))
        .collect();
    assert_eq!(
        adaln.len(),
        cfg.num_layers * 2,
        "one weight + one bias per block"
    );

    // Parameters, from the published geometry the config above was just asserted to carry.
    let per_block = cfg.adaln_out_features() * cfg.time_embed_dim + cfg.adaln_out_features();
    let params = per_block * cfg.num_layers;
    assert_eq!(per_block, 260_209_152, "per-block AdaLN parameters");
    assert_eq!(params, 13_010_457_600, "13.01 B over the 50-block stack");
    // At bf16 — and that dtype is READ from the shard headers rather than assumed, because the
    // whole 26.02 GB figure is `params × bytes-per-element` and a wrong dtype halves or doubles it.
    let dtypes = shard_dtypes(&dir, "minimax-h3 transformer");
    for key in &adaln {
        assert_eq!(
            dtypes.get(*key).map(String::as_str),
            Some("BF16"),
            "{key} must ship bfloat16 for the 26.02 GB arithmetic to hold"
        );
    }
    assert_eq!(params * 2, 26_020_915_200, "26.02 GB at bf16");

    println!(
        "adaln: {} tensors over {} blocks, {per_block} params/block, {params} total = {} GB at \
         bf16 — the eviction's headline number",
        adaln.len(),
        cfg.num_layers,
        (params * 2) as f64 / 1e9
    );
}

// ---------------------------------------------------------------------------------------------
// Spatial tiling on real weights (sc-18786)
// ---------------------------------------------------------------------------------------------

/// **The gate for sc-18786.** `AutoencoderKLMiniMaxH3` ships with `use_tiling = True` (256 px
/// tiles, 64 px minimum overlap) for both halves, and upstream states the consequence outright:
/// MiniMax-H3 was released with tiling enabled and *the released frames are the blended-tile ones,
/// so disabling tiling changes the output*. sc-17154 decoded the whole canvas in one pass.
///
/// # Why the sc-18740 reference could not catch this
///
/// `tools/dump_minimax_h3_video_vae_real.py` decodes a 4x4 latent to 64x64 px and calls
/// `disable_tiling()` first, because at that canvas the two paths are bit-identical anyway. It is
/// inert at its own geometry by construction, so its residual is genuine but blind here.
///
/// This test uses a second reference (`mlx-gen/tools/dump_minimax_h3_video_vae_tiling.py` — one
/// artifact serves both lanes) at a **512x320** canvas: 3 tile rows x 2 tile columns, non-square so
/// a transposed plan is not accidentally correct, with a genuine interior row that a 2x2 grid never
/// exercises. The generator measured the reference's own tiled-vs-untiled separation there at
/// **rel-max-abs 6.470e-1** — the size of the defect, a 65 % error rather than a rounding
/// difference — and it is re-derived below rather than trusted.
///
/// The MLX `i32` write cap does not apply to this lane (`gen-core::tiling::MAX_WRITABLE_ELEMS` is
/// documented as MLX-only; candle uses its own tensor library), and the reference itself is torch.
#[test]
#[ignore = "needs a real snapshot + a tiling reference (MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE)"]
fn real_weight_tiled_decode_matches_the_official_diffusers_vae() {
    use candle_gen_minimax_h3::spatial_tiling::{SpatialTiling, TilePlan};

    let root = snapshot();
    let raw = std::env::var("MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE must point at the output of \
         mlx-gen/tools/dump_minimax_h3_video_vae_tiling.py. This test asserts rather than skips so \
         a missing reference cannot be mistaken for a pass."
    );
    let r = common::Golden::load(&raw);
    assert_eq!(
        r.meta("reference").unwrap_or_default(),
        "diffusers.AutoencoderKLMiniMaxH3",
        "the reference must come from the official converted-checkpoint class"
    );
    for key in ["in.latent", "out.video.tiled", "out.video.untiled"] {
        assert!(
            r.has(key),
            "{raw} has no `{key}`; it is not a tiling reference"
        );
    }

    // The reference records what the shipped model actually does. Pin it here rather than trusting
    // this crate's own constants — the only way the port's defaults are gated against the model
    // instead of against themselves.
    let shipped = r.meta("shipped_tiling").unwrap_or_default().to_string();
    for expected in [
        "\"use_tiling\": true",
        "\"tile_sample_min_height\": 256",
        "\"tile_sample_min_width\": 256",
        "\"tile_sample_min_overlap_height\": 64",
        "\"tile_sample_min_overlap_width\": 64",
    ] {
        assert!(
            shipped.contains(expected),
            "the shipped VAE no longer reports {expected}; recorded defaults were {shipped}"
        );
    }
    let defaults = SpatialTiling::default();
    assert!(defaults.enabled, "this port must tile by default too");
    assert_eq!((defaults.tile_height, defaults.tile_width), (256, 256));
    assert_eq!((defaults.overlap_height, defaults.overlap_width), (64, 64));

    let dev = device();
    let vae = MiniMaxH3VideoVae::load_decode_only(&root, &dev, DType::F32)
        .expect("load the real video VAE");
    let latent = r.tensor("in.latent").to_device(&dev).expect("latent");
    let want = r.tensor("out.video.tiled");
    let want_untiled = r.tensor("out.video.untiled");

    // The canvas must genuinely tile in BOTH axes, or this test proves nothing at all.
    let ls = latent.dims().to_vec();
    let (lat_h, lat_w) = (ls[3], ls[4]);
    let ratio = vae.config().patch_size;
    let rows = TilePlan::split(lat_h * ratio, 256, 64, ratio).unwrap();
    let cols = TilePlan::split(lat_w * ratio, 256, 64, ratio).unwrap();
    assert!(
        rows.len() > 1 && cols.len() > 1,
        "the reference canvas is {}x{} px = {} rows x {} cols; it does not cross a tile boundary \
         on both axes and cannot gate spatial tiling",
        lat_h * ratio,
        lat_w * ratio,
        rows.len(),
        cols.len()
    );
    let plan = r.meta("tile_plan").unwrap_or_default().to_string();
    assert!(
        plan.contains(&format!("{:?}", rows.starts))
            && plan.contains(&format!("{:?}", cols.starts)),
        "the port's tile starts (rows {:?}, cols {:?}) are not the reference's: {plan}",
        rows.starts,
        cols.starts
    );

    let got = vae
        .decode(&latent)
        .expect("decode")
        .to_device(&Device::Cpu)
        .expect("to cpu");
    assert_eq!(got.dims(), want.dims(), "decoded shape");

    // **The pre-fix behaviour, measured on this same port.** sc-17154 decoded the whole canvas in
    // one pass, which is exactly what `disable_tiling()` selects. Running it here turns "the new
    // code is necessary" from a claim into an assertion: if tiling were a no-op, or if the tiled
    // and untiled paths converged, `before` would pass the same gate `after` does and this test
    // would be proving nothing.
    let before = {
        let mut off = vae.clone();
        off.disable_tiling();
        let out = off
            .decode(&latent)
            .expect("untiled decode")
            .to_device(&Device::Cpu)
            .expect("to cpu");
        rel(&out, &want).0
    };

    let (against_tiled, mean_tiled) = rel(&got, &want);
    let (against_untiled, _) = rel(&got, &want_untiled);
    let (reference_separation, _) = rel(&want_untiled, &want);
    println!(
        "real-weight TILED decode ({} rows x {} cols at {}x{} px) vs diffusers \
         AutoencoderKLMiniMaxH3: BEFORE (untiled, sc-17154) {before:.3e} -> AFTER (tiled) \
         {against_tiled:.3e} (mean {mean_tiled:.3e}), both vs the TILED reference. Our tiled \
         decode vs the UNTILED reference {against_untiled:.3e}; the reference's own tiled/untiled \
         separation {reference_separation:.3e}",
        rows.len(),
        cols.len(),
        lat_h * ratio,
        lat_w * ratio,
    );

    // (0) The single-pass decode this story replaced **fails** this gate. Without this the whole
    // test could pass on a canvas where tiling happened not to matter.
    assert!(
        before > 2e-2,
        "an UNTILED decode is within {before:.3e} of the tiled reference, so this canvas cannot \
         distinguish the pre-sc-18786 behaviour from the fix"
    );
    assert!(
        against_tiled < before / 10.0,
        "tiling only improved the residual from {before:.3e} to {against_tiled:.3e}; that is not \
         the reference's geometry"
    );

    // (1) The canvas actually separates the two implementations. Without this the rest is vacuous.
    assert!(
        reference_separation > 1e-2,
        "the reference's own tiled and untiled decodes differ by only {reference_separation:.3e} \
         at this canvas; regenerate the reference at a canvas that crosses a tile boundary"
    );
    // (2) We match the TILED reference — the released frames. 2e-2 is this file's bar for the
    // real-weight comparison (the reference may be dumped at bf16 or f32).
    assert!(
        against_tiled < 2e-2,
        "the tiled decode diverges from the official diffusers VAE by {against_tiled:.3e}; gate on \
         rel-max-abs, never on norm or cosine (sc-18740)"
    );
    // (3) …and are decisively closer to it than to the untiled one, so a regression back to a
    // single-pass decode fails here rather than merely loosening a tolerance.
    assert!(
        against_untiled > 1e-2,
        "the decode is within {against_untiled:.3e} of the UNTILED reference too, so this test \
         cannot tell the two paths apart"
    );
    assert!(
        std_dev(&want) > 1e-3,
        "the reference decode is ~constant; it would gate nothing"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_tiled_decode_matches_the_official_diffusers_vae");
}

/// The mirror assertion, on real weights: **below one tile the tiled and untiled paths agree
/// exactly**, which is what keeps sc-17154's sub-tile fixtures valid across this change. The
/// reference asserts the same thing on its side (`subtile_delta_max_abs == 0`).
#[test]
#[ignore = "needs a real snapshot + a tiling reference (MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE)"]
fn real_weight_tiling_is_inert_below_one_tile() {
    let root = snapshot();
    let raw = std::env::var("MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE must point at the output of \
         mlx-gen/tools/dump_minimax_h3_video_vae_tiling.py"
    );
    let r = common::Golden::load(&raw);
    assert_eq!(
        r.meta("subtile_delta_max_abs").unwrap_or_default(),
        "0.000000e+00",
        "the reference itself no longer finds tiling inert below one tile"
    );

    let dev = device();
    let vae = MiniMaxH3VideoVae::load_decode_only(&root, &dev, DType::F32)
        .expect("load the real video VAE");
    let latent = r
        .tensor("in.latent.subtile")
        .to_device(&dev)
        .expect("latent");
    let want = r.tensor("out.video.subtile.tiled");
    let s = latent.dims().to_vec();
    let ratio = vae.config().patch_size;
    assert!(
        s[3] * ratio <= 256 && s[4] * ratio <= 256,
        "the sub-tile control is not below one tile"
    );

    let tiled = vae
        .decode(&latent)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let untiled = {
        let mut off = vae.clone();
        off.disable_tiling();
        off.decode(&latent)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
    };
    assert_eq!(
        flat(&tiled),
        flat(&untiled),
        "below one tile the two paths must be BIT-identical"
    );

    let (peak, _) = rel(&tiled, &want);
    println!("real-weight SUB-TILE parity: rel-max-abs={peak:.3e} (tiling inert, delta 0)");
    assert!(peak < 2e-2, "the sub-tile decode diverges by {peak:.3e}");

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::real_weight_tiling_is_inert_below_one_tile"
    );
}

// ── SW-7089: MiniMax-H3-Image 2-D VAE decoder ────────────────────────────────────────────────────

/// Peak-relative max-abs-diff bound for the image VAE against diffusers, at f32 — the bar the
/// video decode gate uses for the same comparison.
const IMAGE_VAE_TOL: f32 = 2e-2;

/// The committed producer whose output the image gate accepts (it lives with the MLX tools).
const IMAGE_VAE_PRODUCER: &[u8] =
    include_bytes!("../../../mlx-gen/tools/dump_minimax_h3_image_vae_real.py");

fn image_snapshot() -> std::path::PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3-Image snapshot root (the \
         dir holding `vae_diffusers/`). This test is #[ignore]d and asserts rather than skips."
    );
    let path = std::path::PathBuf::from(raw);
    assert!(
        path.join("vae_diffusers").join("config.json").is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} has no vae_diffusers/config.json",
        path.display()
    );
    path
}

/// AC1 (candle): the 2-D decoder, tiled and untiled, against the official diffusers
/// `AutoencoderKLMiniMaxH3Image` on the published `vae_diffusers/` bytes. The reference must carry
/// the committed producer's SHA-256.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot + a reference decode (MINIMAX_H3_IMAGE_VAE_REFERENCE)"]
fn image_vae_decode_matches_the_official_diffusers_vae() {
    use candle_gen_minimax_h3::spatial_tiling::SpatialTiling;
    use candle_gen_minimax_h3::MiniMaxH3ImageVae;
    use sha2::{Digest, Sha256};

    let root = image_snapshot();
    let raw = std::env::var("MINIMAX_H3_IMAGE_VAE_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_VAE_REFERENCE must point at the output of \
         tools/dump_minimax_h3_image_vae_real.py; this test asserts rather than skips."
    );
    let reference = common::Golden::load(&raw);
    assert_eq!(
        reference.meta("reference").unwrap_or_default(),
        "diffusers.AutoencoderKLMiniMaxH3Image",
        "the reference must come from the official image VAE class"
    );
    let pinned = format!("{:x}", Sha256::digest(IMAGE_VAE_PRODUCER));
    assert_eq!(
        reference.meta("producer_sha256").unwrap_or_default(),
        pinned,
        "the reference was produced by a different dump_minimax_h3_image_vae_real.py than the \
         committed one; re-produce it"
    );

    let dev = device();
    let vae = MiniMaxH3ImageVae::load(&root, &dev, DType::F32).expect("load the image VAE");
    assert!(
        vae.tiling().enabled,
        "the shipped image VAE tiles by default"
    );
    let decode = |vae: &MiniMaxH3ImageVae, key: &str| -> Tensor {
        let latent = reference.tensor(key).to_device(&dev).expect("latent");
        vae.decode(&latent)
            .expect("decode")
            .to_device(&Device::Cpu)
            .expect("to cpu")
    };

    let got = decode(&vae, "in.small");
    let want = reference.tensor("out.small");
    assert_eq!(got.dims(), want.dims(), "small decode shape");
    let (small_rel, _) = rel(&got, &want);

    let got_tiled = decode(&vae, "in.tiled");
    let want_tiled = reference.tensor("out.tiled");
    assert_eq!(got_tiled.dims(), &[1, 3, 384, 640], "tiled decode shape");
    let (tiled_rel, _) = rel(&got_tiled, &want_tiled);

    let untiled = vae.clone().with_tiling(SpatialTiling::disabled());
    let got_untiled = decode(&untiled, "in.tiled");
    let want_untiled = reference.tensor("out.untiled");
    let (untiled_rel, _) = rel(&got_untiled, &want_untiled);
    let (branch_gap, _) = rel(&want_tiled, &want_untiled);

    println!(
        "image VAE vs diffusers AutoencoderKLMiniMaxH3Image (bound {IMAGE_VAE_TOL:.0e}): small \
         {small_rel:.3e}, tiled {tiled_rel:.3e}, untiled {untiled_rel:.3e}; reference tiled-vs-untiled \
         gap {branch_gap:.3e}"
    );
    assert!(
        small_rel < IMAGE_VAE_TOL,
        "small decode diverges by {small_rel:.3e}"
    );
    assert!(
        tiled_rel < IMAGE_VAE_TOL,
        "tiled decode diverges by {tiled_rel:.3e}"
    );
    assert!(
        untiled_rel < IMAGE_VAE_TOL,
        "untiled decode diverges by {untiled_rel:.3e}"
    );
    assert!(
        branch_gap > IMAGE_VAE_TOL,
        "the reference's tiled and untiled decodes differ by only {branch_gap:.3e}, so this \
         canvas does not distinguish the branches"
    );
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_vae_decode_matches_the_official_diffusers_vae"
    );
}

/// AC2 (candle/CUDA): at 16 MP (a 256x256 latent -> 4096x4096 px) the shipped 256 / 64 tiled
/// decode — the family's BoundedDecode — peaks below the untiled decode, measured as the CUDA
/// pool's concurrent-live high-water reset before each run. Off CUDA there is no device peak to
/// read, so the test refuses rather than passing on nothing.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT) and CUDA; 16 MP decode twice"]
fn image_vae_tiled_decode_lowers_the_16mp_peak() {
    #[cfg(not(feature = "cuda"))]
    panic!("the 16 MP peak comparison reads the CUDA pool high-water; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        use candle_gen_minimax_h3::spatial_tiling::SpatialTiling;
        use candle_gen_minimax_h3::MiniMaxH3ImageVae;

        let root = image_snapshot();
        let dev = device();
        assert!(dev.is_cuda(), "the peak comparison needs the CUDA device");
        let vae = MiniMaxH3ImageVae::load(&root, &dev, DType::F32).expect("load the image VAE");
        let n = 24 * 256 * 256;
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let latents = Tensor::from_vec(vals, (1, 24, 256, 256), &dev).expect("latents");
        let measure = |label: &str, vae: &MiniMaxH3ImageVae| -> u64 {
            dev.synchronize().expect("sync");
            let pool = candle_gen::cuda_mempool::MemPool::device_current(0)
                .expect("a CUDA memory pool on device 0");
            pool.reset_high_water();
            let started = std::time::Instant::now();
            let out = vae.decode(&latents).expect("decode");
            dev.synchronize().expect("sync");
            let peak = pool.used_high().expect("pool high-water");
            assert_eq!(out.dims(), &[1, 3, 4096, 4096], "{label} decode shape");
            let mean = out
                .abs()
                .and_then(|t| t.mean_all())
                .and_then(|t| t.to_scalar::<f32>())
                .expect("mean");
            assert!(
                mean.is_finite() && mean > 0.0,
                "{label} decode is degenerate: {mean}"
            );
            println!(
                "16 MP {label} decode: peak {:.3} GiB in {:.1} s",
                peak as f64 / (1u64 << 30) as f64,
                started.elapsed().as_secs_f64()
            );
            peak
        };
        let tiled_peak = measure("tiled 256/64", &vae);
        let untiled_vae = vae.clone().with_tiling(SpatialTiling::disabled());
        let untiled_peak = measure("untiled", &untiled_vae);
        assert!(
            tiled_peak < untiled_peak,
            "the bounded (tiled) decode peaked at {tiled_peak} B, not below the untiled \
             {untiled_peak} B"
        );
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_vae_tiled_decode_lowers_the_16mp_peak"
        );
    }
}

// ── SW-7091: MiniMax-H3-Image DiT and text-to-image ──────────────────────────────────────────────

// The DiT gate asserts the family bounds (requirement E9, `gen_core::h3_image`), the same metric and
// values as the MLX gate: rel-L2 and cosine of the velocity against diffusers' float32-activation
// forward over the same bf16 weights. Peak and mean errors are printed as diagnostics only.
use candle_gen::gen_core::h3_image::dit_sweep::{self, RowRegions, SweepRow};
use candle_gen::gen_core::h3_image::{DIT_PARITY_MAX_REL_L2, DIT_PARITY_MIN_COSINE};

/// `(||got - want|| / ||want||, cosine)` over the flattened tensors, in f32.
fn image_dit_rel_l2_cosine(got: &Tensor, want: &Tensor) -> (f32, f32) {
    let (g, w) = (flat(got), flat(want));
    assert_eq!(g.len(), w.len(), "length mismatch");
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let diff: Vec<f32> = g.iter().zip(&w).map(|(a, b)| a - b).collect();
    let dot: f32 = g.iter().zip(&w).map(|(a, b)| a * b).sum();
    (norm(&diff) / norm(&w), dot / (norm(&g) * norm(&w)))
}

/// The committed producer whose output the DiT gate accepts.
const IMAGE_DIT_PRODUCER: &[u8] =
    include_bytes!("../../../mlx-gen/tools/dump_minimax_h3_image_dit_real.py");

/// One diagnostic line for the image DiT gate: a named stage's peak- and mean-relative error and
/// rel-L2 against the reference's own tensor. Printed, never asserted — the bounds stay on the
/// velocity alone.
fn image_dit_stage_line(step: usize, stage: &str, got: &Tensor, want: &Tensor) -> String {
    let got = got
        .to_dtype(DType::F32)
        .and_then(|t| t.to_device(&Device::Cpu))
        .expect("stage to host");
    assert_eq!(got.dims(), want.dims(), "step {step} {stage} shape");
    let (peak, mean) = rel(&got, want);
    let l2 = l2_norm(&(&got - want).expect("diff")) / l2_norm(want).max(1e-12);
    format!(
        "image DiT step {step} stage {stage:<16} peak {peak:.3e} mean {mean:.3e} rel-L2 {l2:.3e}"
    )
}

/// The snapshot root holding `transformer_diffusers/` (the `minimax-h3-image-t2i` row).
fn image_dit_snapshot() -> std::path::PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3-Image snapshot root; this \
         test is #[ignore]d and asserts rather than skips."
    );
    let path = std::path::PathBuf::from(raw);
    assert!(
        path.join("transformer_diffusers")
            .join("config.json")
            .is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} has no transformer_diffusers/config.json",
        path.display()
    );
    path
}

/// AC1 + AC2 (candle): the image DiT loads from `transformer_diffusers/` with exactly the 644
/// declared tensors and no audio tensor, and its step-0 output (head 0) and step-3 output (head 3)
/// match diffusers' `MiniMaxH3ImageTransformer2DModel` on the published weights, over the exact
/// inputs of the reference SW-7090's macOS job produces (`dit-reference.safetensors`, shared by
/// both backends; float32 activations over the bf16 weights). Step 0 is asserted on the noise
/// rows; step 3 on the sampler's own `x_3` (`diag.traj.rows3`, reached by the reference from those
/// noise rows). The timesteps this crate derives are held to the producer's.
///
/// Before asserting, every step prints its error at each stage the reference recorded (`temb`,
/// `norm_out` shift/scale, block 0's AdaLN tables, the residual stream after the `diag_layers`,
/// `norm_out`'s output). Also printed, never asserted: step 3 on the step-0 noise rows (an input
/// no sampler shows the model at `t = 0.25`), and both asserted velocities against the
/// reference's own bf16-compute forward (`diag.bf16.*`).
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (transformer_diffusers/) + a reference (MINIMAX_H3_IMAGE_DIT_REFERENCE); 66 GB bf16 DiT"]
fn image_dit_first_step_matches_the_official_diffusers_transformer() {
    use candle_gen_minimax_h3::dit::{ImageDit, ImageDitTrace, MiniMaxH3Dit, MiniMaxH3DitConfig};
    use candle_gen_minimax_h3::{ImageLayout, ImageVelocity, IMAGE_DIT_PARTITION};
    use gen_core_h3::h3_schedule::SigmaSchedule;
    use sha2::{Digest, Sha256};

    let root = image_dit_snapshot();
    let raw = std::env::var("MINIMAX_H3_IMAGE_DIT_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_DIT_REFERENCE must point at the output of \
         tools/dump_minimax_h3_image_dit_real.py; this test asserts rather than skips."
    );
    let reference = common::Golden::load(&raw);
    assert_eq!(
        reference.meta("reference").unwrap_or_default(),
        "diffusers.MiniMaxH3ImageTransformer2DModel"
    );
    assert_eq!(
        reference.meta("producer_sha256").unwrap_or_default(),
        format!("{:x}", Sha256::digest(IMAGE_DIT_PRODUCER)),
        "the reference was produced by a different dump_minimax_h3_image_dit_real.py; re-produce it"
    );
    assert_eq!(
        reference.meta("compute_dtype").unwrap_or_default(),
        "float32",
        "the asserted reference must be computed with float32 activations"
    );

    // Exhaustive mapping: the partition carries exactly the declared names — no audio tensor.
    let dir = root.join(IMAGE_DIT_PARTITION);
    let cfg = MiniMaxH3DitConfig::from_image_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("config"),
    )
    .expect("image config");
    let declared: BTreeSet<String> = MiniMaxH3Dit::names(&cfg).into_iter().collect();
    assert_eq!(shard_keys(&dir, "transformer_diffusers"), declared);
    assert!(!declared.iter().any(|n| n.contains("audio")));

    assert_eq!(reference.meta("steps").unwrap_or_default(), "0,3");
    let num_text: usize = reference.meta("num_text").expect("meta").parse().unwrap();
    let hw = reference.meta("latent_hw").expect("meta").to_string();
    let (lh, lw) = hw.split_once('x').expect("HxW");
    let (lh, lw): (usize, usize) = (lh.parse().unwrap(), lw.parse().unwrap());
    let layout = ImageLayout::text_to_image(num_text, lh, lw, cfg.patch_size).expect("layout");

    let schedule = SigmaSchedule::h3_image_distilled().expect("schedule");
    let want_t: Vec<f32> = reference
        .meta("timesteps")
        .expect("meta")
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    assert_eq!(
        want_t,
        vec![schedule.timesteps()[0], schedule.timesteps()[3]],
        "the timesteps must be the released sampler's float32 values"
    );

    let dev = device();
    let dit = MiniMaxH3Dit::load_from_dir(&dir, &dev, DType::BF16).expect("load the image DiT");
    assert!(!dit.config().has_audio());
    let context = reference
        .tensor("in.context")
        .to_dtype(DType::BF16)
        .and_then(|t| t.to_device(&dev))
        .expect("context");
    let rows = reference.tensor("in.rows").to_device(&dev).expect("rows");
    let mut model = ImageDit::new(dit, &layout, &context, schedule.timesteps()).expect("ImageDit");
    assert!(model.released_bytes() > 0, "adaln_proj was evicted");
    let diag_layers: Vec<usize> = reference
        .meta("diag_layers")
        .expect("the reference records its diag_layers")
        .split(',')
        .map(|k| k.parse().unwrap())
        .collect();
    let host = |t: &Tensor| -> Tensor {
        t.to_dtype(DType::F32)
            .and_then(|t| t.to_device(&Device::Cpu))
            .expect("to host")
    };
    let run = |model: &mut ImageDit, step: usize| -> Tensor {
        let mut trace = ImageDitTrace::new(&diag_layers);
        let got = host(
            &model
                .forward_traced(step, &rows, &mut trace)
                .expect("forward"),
        );
        let d = |name: &str| reference.tensor(&format!("diag.step{step}.{name}"));
        let g = model
            .adaln_cache()
            .schedule()
            .global_timestep_index(step, 0)
            .expect("timestep row") as usize;
        let mods = model
            .adaln_cache()
            .modulation(0)
            .expect("block 0 modulation");
        let block0 = Tensor::stack(
            &mods
                .tables()
                .map(|t| t.narrow(0, g * 3, 3).expect("rows"))
                .collect::<Vec<_>>(),
            0,
        )
        .expect("stack");
        let norm_out = model.norm_out_modulation();
        let mut lines = vec![
            image_dit_stage_line(
                step,
                "temb",
                &model.temb().narrow(0, g, 1).expect("row"),
                &d("temb"),
            ),
            image_dit_stage_line(
                step,
                "norm_out.shift",
                &norm_out.shift.narrow(0, g, 1).expect("row"),
                &d("norm_out.shift"),
            ),
            image_dit_stage_line(
                step,
                "norm_out.scale",
                &norm_out.scale.narrow(0, g, 1).expect("row"),
                &d("norm_out.scale"),
            ),
            image_dit_stage_line(step, "block0.adaln", &block0, &d("block0.adaln")),
        ];
        for (layer, hidden) in &trace.hidden {
            lines.push(image_dit_stage_line(
                step,
                &format!("hidden.block{layer}"),
                hidden,
                &d(&format!("hidden.block{layer}")),
            ));
        }
        lines.push(image_dit_stage_line(
            step,
            "pre_head",
            trace.pre_head.as_ref().expect("the trace records norm_out"),
            &d("pre_head"),
        ));
        lines.push(image_dit_stage_line(
            step,
            "velocity",
            &got,
            &reference.tensor(&format!("out.step{step}")),
        ));
        for line in lines {
            println!("{line}");
        }
        got
    };
    let got0 = run(&mut model, 0);
    let got3 = run(&mut model, 3);
    // Asserted step 3: the sampler's own x_3, reached by the reference from the same noise rows.
    let traj_rows = reference
        .tensor("diag.traj.rows3")
        .to_device(&dev)
        .expect("trajectory rows");
    let traj = host(&model.forward(3, &traj_rows).expect("on-trajectory forward"));
    let traj_want = reference.tensor("diag.traj.out3");
    println!(
        "{}",
        image_dit_stage_line(3, "on-trajectory", &traj, &traj_want)
    );
    // The reference's own rounding: the same two asserted inputs at bf16 compute.
    let want0 = reference.tensor("out.step0");
    let bf16_step0 = reference.tensor("diag.bf16.step0");
    let bf16_traj = reference.tensor("diag.bf16.traj.out3");
    for (step, stage, got, want) in [
        (0, "ref bf16 vs f32", &bf16_step0, &want0),
        (3, "ref bf16 vs f32", &bf16_traj, &traj_want),
        (0, "vs ref bf16", &got0, &bf16_step0),
        (3, "traj vs ref bf16", &traj, &bf16_traj),
    ] {
        println!("{}", image_dit_stage_line(step, stage, got, want));
    }
    let want3 = reference.tensor("out.step3");
    assert_eq!(got0.dims(), want0.dims());
    let (p0, m0) = rel(&got0, &want0);
    let (pt, mt) = rel(&traj, &traj_want);
    let (p3, m3) = rel(&got3, &want3);
    let (l0, c0) = image_dit_rel_l2_cosine(&got0, &want0);
    let (lt, ct) = image_dit_rel_l2_cosine(&traj, &traj_want);
    let (l3, c3) = image_dit_rel_l2_cosine(&got3, &want3);
    let (between, _) = image_dit_rel_l2_cosine(&want3, &want0);
    let (cross, _) = image_dit_rel_l2_cosine(&got3, &want0);
    println!(
        "image DiT vs diffusers (family bounds rel-L2 {DIT_PARITY_MAX_REL_L2:.0e}, cosine {DIT_PARITY_MIN_COSINE}): \
         step0 rel-L2 {l0:.3e} cosine {c0:.6}; step3 on-trajectory rel-L2 {lt:.3e} cosine {ct:.6}; \
         reference step0 vs step3 rel-L2 {between:.3e}; candle step3 vs reference step0 {cross:.3e}"
    );
    println!(
        "image DiT peak/mean diagnostics (not asserted): step0 peak {p0:.3e} mean {m0:.3e}; step3 \
         on-trajectory peak {pt:.3e} mean {mt:.3e}"
    );
    println!(
        "image DiT step 3 off-trajectory (step-0 noise rows at t = 0.25; diagnostic, not \
         asserted): rel-L2 {l3:.3e} cosine {c3:.6} peak {p3:.3e} mean {m3:.3e}"
    );
    for (what, l2, cos) in [("step 0", l0, c0), ("step 3 on-trajectory", lt, ct)] {
        assert!(
            l2 <= DIT_PARITY_MAX_REL_L2,
            "{what} diverges: rel-L2 {l2:.3e}"
        );
        assert!(
            cos >= DIT_PARITY_MIN_COSINE,
            "{what} cosine {cos:.6} below {DIT_PARITY_MIN_COSINE}"
        );
    }
    assert!(
        between > DIT_PARITY_MAX_REL_L2 && cross > DIT_PARITY_MAX_REL_L2,
        "steps 0 and 3 must be distinguishable for the head check to mean anything"
    );
    assert_eq!(model.forwards(), 3);
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_dit_first_step_matches_the_official_diffusers_transformer"
    );
}

/// AC1 (candle/CUDA): the registered `minimax_h3_image` generator renders a 2048 x 2048 bf16
/// image end to end — conditioner, six-step DiT, tiled 2-D VAE decode — through the registry's own
/// load path. The image is written to `MINIMAX_H3_IMAGE_RENDER_OUT` when set.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT), MiniMax-H3's text encoder (MINIMAX_H3_SNAPSHOT) and CUDA; full 2048x2048 render"]
fn image_text_to_image_renders_2048_on_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the 2048x2048 render runs the 66 GB DiT on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        use candle_gen::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};

        let root = image_dit_snapshot();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        // The conditioner: this snapshot's own `text_encoder/` when present, else MiniMax-H3's
        // byte-identical copy staged as the `text_encoder` component.
        if !root.join("text_encoder").is_dir() {
            let h3 = std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default();
            assert!(
                !h3.is_empty(),
                "the image snapshot has no text_encoder/; MINIMAX_H3_SNAPSHOT must point at a \
                 MiniMax-H3 snapshot holding text_encoder/ shards 1-12"
            );
            spec.components.insert(
                "text_encoder".into(),
                WeightsSource::Dir(std::path::PathBuf::from(h3).join("text_encoder")),
            );
        }
        let generator = candle_gen_minimax_h3::image_model::load_image(&spec)
            .expect("load the minimax_h3_image generator");
        assert_eq!(generator.descriptor().id, "minimax_h3_image");
        let req = GenerationRequest {
            prompt: "A red lighthouse on a rocky coast at dusk, waves breaking, warm light in the \
                     lantern room, photographed with a 35mm lens"
                .into(),
            width: 2048,
            height: 2048,
            seed: Some(7091),
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let mut steps = Vec::new();
        let output = generator
            .generate(&req, &mut |p| {
                if let candle_gen::gen_core::Progress::Step { current, .. } = p {
                    steps.push(current)
                }
            })
            .expect("render");
        assert_eq!(steps, vec![1, 2, 3, 4, 5, 6], "one forward per fixed step");
        let GenerationOutput::Images(images) = output else {
            panic!("an image generator must return images")
        };
        assert_eq!(images.len(), 1);
        let img = &images[0];
        assert_eq!((img.width, img.height), (2048, 2048));
        let n = img.pixels.len() as f64;
        let mean = img.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
        let var = img
            .pixels
            .iter()
            .map(|&p| (f64::from(p) - mean).powi(2))
            .sum::<f64>()
            / n;
        println!(
            "2048x2048 render in {:.1} s: mean {mean:.1}, std {:.1}",
            started.elapsed().as_secs_f64(),
            var.sqrt()
        );
        assert!(
            var.sqrt() > 10.0 && (5.0..250.0).contains(&mean),
            "degenerate render: mean {mean:.1}, std {:.1}",
            var.sqrt()
        );
        if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_RENDER_OUT") {
            image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
                .expect("rgb")
                .save(&out)
                .expect("write the render");
            println!("wrote {out}");
        }
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_text_to_image_renders_2048_on_cuda"
        );
    }
}

// ── SW-7092: MiniMax-H3-Image edit (1-16 reference images) ──────────────────────────────────────

/// The committed producer whose output the edit gates accept.
const IMAGE_EDIT_PRODUCER: &[u8] =
    include_bytes!("../../../mlx-gen/tools/dump_minimax_h3_image_edit_real.py");

/// Peak-relative bound for the f32 reference encode. The posterior mean is the f32 encoder's
/// (the video encode bound is 2e-4); the latent is after a float16 round, where one flipped
/// half-precision ulp near the peak alone is ~1e-3 of it.
const IMAGE_EDIT_MEAN_TOL: f32 = 2e-4;
const IMAGE_EDIT_LATENT_TOL: f32 = 2e-3;

fn image_edit_reference() -> common::Golden {
    use sha2::{Digest, Sha256};
    let raw = std::env::var("MINIMAX_H3_IMAGE_EDIT_REFERENCE").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_EDIT_REFERENCE must point at the output of \
         tools/dump_minimax_h3_image_edit_real.py; this test asserts rather than skips."
    );
    let reference = common::Golden::load(&raw);
    assert_eq!(
        reference.meta("producer_sha256").unwrap_or_default(),
        format!("{:x}", Sha256::digest(IMAGE_EDIT_PRODUCER)),
        "the reference was produced by a different dump_minimax_h3_image_edit_real.py; re-produce it"
    );
    assert_eq!(reference.meta("num_references").unwrap_or_default(), "3");
    reference
}

/// Print the per-block sweep of one traced step-0 forward against the reference's
/// `diag.sweep.{f32,bf16}.block{k}` (see `gen_core::h3_image::dit_sweep`). Diagnostic only.
fn image_edit_block_sweep(
    reference: &common::Golden,
    trace: &candle_gen_minimax_h3::dit::ImageDitTrace,
    layers: &[usize],
    regions: RowRegions,
) {
    println!(
        "sweep (step 0): per block, rel-L2 vs the f32 reference, vs the bf16 reference, and the \
         reference's own bf16-vs-f32 drift — all rows, then text / reference / generated rows"
    );
    let mut rows = Vec::with_capacity(layers.len());
    for &layer in layers {
        let (_, hidden) = trace
            .hidden
            .iter()
            .find(|(l, _)| *l == layer)
            .unwrap_or_else(|| panic!("the trace recorded block {layer}"));
        let (f32_key, bf16_key) = (
            format!("diag.sweep.f32.block{layer}"),
            format!("diag.sweep.bf16.block{layer}"),
        );
        assert_eq!(
            hidden.dims(),
            reference.shape(&f32_key),
            "sweep block {layer} shape"
        );
        assert_eq!(
            reference.dtype(&bf16_key),
            "BF16",
            "sweep block {layer} bf16 dtype"
        );
        let width = *hidden.dims().last().expect("hidden width");
        let row = SweepRow::new(
            layer,
            &flat(hidden),
            &reference.f32(&f32_key),
            &reference.f32(&bf16_key),
            width,
            regions,
        )
        .expect("sweep row");
        println!("{row}");
        rows.push(row);
    }
    for line in dit_sweep::summary(&rows) {
        println!("{line}");
    }
}

/// AC2 (candle): the `edit` presentation built with the snapshot's own tokenizer is the processor
/// template's (`Picture i: ` + vision block per reference, then the prompt, no chat template), id
/// for id and tag for tag, as diffusers' `MiniMaxH3ImageEditTextEncoderStep` builds it.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT (tokenizer/) + MINIMAX_H3_IMAGE_EDIT_REFERENCE; tokenizer only"]
fn image_edit_presentation_matches_the_processor_template() {
    let root = image_dit_snapshot();
    let reference = image_edit_reference();
    let counts: Vec<usize> = reference
        .meta("vision_tokens")
        .expect("meta")
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    let prompt = reference.meta("prompt").expect("meta").to_string();
    let tok = candle_gen_minimax_h3::text_encoder::MiniMaxH3Tokenizer::from_snapshot(&root)
        .expect("tokenizer");
    let (ids, tags) = tok.image_edit_ids(&prompt, &counts).expect("presentation");
    let want_ids: Vec<u32> = reference
        .f32("in.token_ids")
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let want_tags: Vec<u32> = reference
        .f32("in.token_tags")
        .into_iter()
        .map(|v| v as u32)
        .collect();
    println!(
        "edit presentation: {} rows, vision blocks {counts:?}",
        ids.len()
    );
    assert_eq!(
        ids, want_ids,
        "token ids differ from the reference presentation"
    );
    assert_eq!(
        tags, want_tags,
        "row tags differ from the reference presentation"
    );
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_edit_presentation_matches_the_processor_template"
    );
}

/// AC1 (candle, reference latents): the f32 image VAE encode of the three references matches
/// diffusers' posterior mean, and the released latent recipe matches its latent.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT (vae_diffusers/) + MINIMAX_H3_IMAGE_EDIT_REFERENCE"]
fn image_edit_reference_latents_match_the_official_diffusers_vae() {
    use candle_gen_minimax_h3::{reference_vae_pixels, MiniMaxH3ImageVaeEncoder};

    let root = image_dit_snapshot();
    let reference = image_edit_reference();
    let dev = device();
    let encoder = MiniMaxH3ImageVaeEncoder::load(&root, &dev).expect("encoder");
    let mut worst = (0f32, 0f32);
    for i in 0..3 {
        let shape = reference.shape(&format!("ref{i}.image"));
        let image = candle_gen::gen_core::Image {
            width: shape[1] as u32,
            height: shape[0] as u32,
            pixels: reference
                .f32(&format!("ref{i}.image"))
                .into_iter()
                .map(|v| v as u8)
                .collect(),
        };
        let posterior = encoder
            .encode(&reference_vae_pixels(&image, &dev).expect("pixels"))
            .expect("encode");
        let want_mean = reference.tensor(&format!("ref{i}.mean"));
        let mean = posterior
            .mean()
            .reshape(want_mean.dims())
            .and_then(|t| t.to_device(&Device::Cpu))
            .expect("mean");
        let noise = reference
            .tensor(&format!("ref{i}.posterior_noise"))
            .unsqueeze(2)
            .and_then(|t| t.to_device(&dev))
            .expect("noise");
        let latent = encoder
            .reference_latent(&posterior, &noise, (2, 2))
            .expect("latent")
            .to_device(&Device::Cpu)
            .expect("latent to host");
        let want_latent = reference.tensor(&format!("ref{i}.latent"));
        assert_eq!(
            latent.dims(),
            want_latent.dims(),
            "reference {i} latent shape"
        );
        let (pm, _) = rel(&mean, &want_mean);
        let (pl, _) = rel(&latent, &want_latent);
        println!(
            "reference {i} ({}x{} px): posterior mean peak-rel {pm:.3e} (bound \
             {IMAGE_EDIT_MEAN_TOL:.0e}), latent peak-rel {pl:.3e} (bound {IMAGE_EDIT_LATENT_TOL:.0e})",
            image.width, image.height
        );
        worst = (worst.0.max(pm), worst.1.max(pl));
    }
    assert!(
        worst.0 < IMAGE_EDIT_MEAN_TOL,
        "posterior mean diverges: {:.3e}",
        worst.0
    );
    assert!(
        worst.1 < IMAGE_EDIT_LATENT_TOL,
        "reference latent diverges: {:.3e}",
        worst.1
    );
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_edit_reference_latents_match_the_official_diffusers_vae"
    );
}

/// AC1 (candle): a 3-reference edit's step-0 (head 0) and step-3 (head 3) DiT outputs match
/// diffusers' `MiniMaxH3ImageTransformer2DModel` over `[text | 3 references | 512² image]`,
/// computed by the reference with float32 activations over the bf16 weights (`compute_dtype`), as
/// the `text2image` gate's reference is. The reference rows are built here from the reference's
/// latents and draws and must equal the producer's before the DiT runs; the layout and the held
/// `0.999` row class are this crate's.
///
/// Step 0 is asserted on the packed rows; step 3 on the sampler's own `x_3` (`diag.traj.rows3`: the
/// generated rows walked through steps 0-2 by the reference, the reference rows unchanged). Printed,
/// never asserted: every stage the reference recorded (both timestep rows of `temb`, `norm_out`,
/// block 0's AdaLN tables, the residual stream after the `diag_layers`, `norm_out`'s output), the
/// velocity error over the reference rows and the generated rows separately, step 3 on the step-0
/// rows, and both asserted velocities against the reference's own bf16-compute forward.
///
/// Also printed at step 0: the per-block sweep (`diag.sweep.*`, `gen_core::h3_image::dit_sweep`)
/// — after every block in `sweep_layers`, the error against the f32 and the bf16 reference and the
/// reference's own bf16-vs-f32 drift, over all rows and per row class, with the largest reference
/// activation and the largest error located — then the first block where this backend departs
/// from diffusers' bf16 compute.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT (transformer_diffusers/) + MINIMAX_H3_IMAGE_EDIT_REFERENCE; 66 GB bf16 DiT"]
fn image_edit_first_step_matches_the_official_diffusers_transformer() {
    use candle_gen_minimax_h3::dit::{ImageDit, ImageDitTrace, MiniMaxH3Dit, MiniMaxH3DitConfig};
    use candle_gen_minimax_h3::{
        reference_condition_rows, ImageLayout, ImageVelocity, IMAGE_DIT_PARTITION,
    };
    use gen_core_h3::h3_schedule::SigmaSchedule;

    let root = image_dit_snapshot();
    let reference = image_edit_reference();
    assert_eq!(
        reference.meta("compute_dtype").unwrap_or_default(),
        "float32",
        "the asserted edit reference must be computed with float32 activations"
    );
    let dir = root.join(IMAGE_DIT_PARTITION);
    let cfg = MiniMaxH3DitConfig::from_image_diffusers_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("config"),
    )
    .expect("image config");
    let tags: Vec<u32> = reference
        .f32("in.token_tags")
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let hw = reference.meta("latent_hw").expect("meta").to_string();
    let (lh, lw) = hw.split_once('x').expect("HxW");
    let (lh, lw): (usize, usize) = (lh.parse().unwrap(), lw.parse().unwrap());
    let latents: Vec<Tensor> = (0..3)
        .map(|i| reference.tensor(&format!("ref{i}.latent")))
        .collect();
    let aug: Vec<Tensor> = (0..3)
        .map(|i| reference.tensor(&format!("ref{i}.aug_noise")))
        .collect();
    let condition = reference_condition_rows(&latents, &aug, cfg.patch_size).expect("rows");
    let rows = Tensor::cat(&[&condition, &reference.tensor("in.target_rows")], 1).expect("cat");
    let (rows_gap, _) = rel(&rows, &reference.tensor("in.rows"));
    assert!(
        rows_gap < 1e-6,
        "packed reference rows differ: {rows_gap:.3e}"
    );
    let grids: Vec<(usize, usize)> = latents.iter().map(|l| (l.dims()[2], l.dims()[3])).collect();
    let layout = ImageLayout::edit(&tags, &grids, lh, lw, cfg.patch_size).expect("layout");
    assert_eq!(
        layout.num_reference_rows().to_string(),
        reference.meta("num_reference_rows").expect("meta")
    );
    let num_ref = layout.num_reference_rows();
    let num_packed = layout.num_packed_image_rows();
    let sweep_regions = RowRegions {
        text: layout.num_text_tokens(),
        reference: num_ref,
        generated: num_packed - num_ref,
    };

    let schedule = SigmaSchedule::h3_image_distilled().expect("schedule");
    let want_t: Vec<f32> = reference
        .meta("timesteps")
        .expect("meta")
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    assert_eq!(
        want_t,
        vec![schedule.timesteps()[0], schedule.timesteps()[3]],
        "the timesteps must be the released sampler's float32 values"
    );
    let diag_layers: Vec<usize> = reference
        .meta("diag_layers")
        .expect("the reference records its diag_layers")
        .split(',')
        .map(|k| k.parse().unwrap())
        .collect();
    assert_eq!(reference.meta("sweep_step").unwrap_or_default(), "0");
    let sweep_layers: Vec<usize> = reference
        .meta("sweep_layers")
        .expect("the reference records its sweep_layers")
        .split(',')
        .map(|k| k.parse().unwrap())
        .collect();
    let dev = device();
    let dit = MiniMaxH3Dit::load_from_dir(&dir, &dev, DType::BF16).expect("load the image DiT");
    let context = reference
        .tensor("in.context")
        .to_dtype(DType::BF16)
        .and_then(|t| t.to_device(&dev))
        .expect("context");
    let rows = rows.to_device(&dev).expect("rows");
    let mut model = ImageDit::new(dit, &layout, &context, schedule.timesteps()).expect("ImageDit");
    let host = |t: &Tensor| -> Tensor {
        t.to_dtype(DType::F32)
            .and_then(|t| t.to_device(&Device::Cpu))
            .expect("to host")
    };
    // The reference's tables hold one row block per DISTINCT timestep of the step, ascending: the
    // step's timestep (row class 0), then the reference rows' 0.999 (row class 1).
    let blocks = |model: &ImageDit, table: &Tensor, step: usize, rows: usize| -> Tensor {
        let parts: Vec<Tensor> = (0..2)
            .map(|class| {
                let g = model
                    .adaln_cache()
                    .schedule()
                    .global_timestep_index(step, class)
                    .expect("timestep row") as usize;
                table.narrow(0, g * rows, rows).expect("rows")
            })
            .collect();
        Tensor::cat(&parts, 0).expect("cat")
    };
    let regions = |step: usize, what: &str, got: &Tensor, want: &Tensor| {
        for (name, lo, hi) in [
            ("reference rows", 0, num_ref),
            ("generated rows", num_ref, num_packed),
        ] {
            println!(
                "{}",
                image_dit_stage_line(
                    step,
                    &format!("{what} {name}"),
                    &got.narrow(1, lo, hi - lo).expect("region"),
                    &want.narrow(1, lo, hi - lo).expect("region"),
                )
            );
        }
    };
    let run = |model: &mut ImageDit, step: usize| -> Tensor {
        let mut layers = diag_layers.clone();
        if step == 0 {
            layers.extend(&sweep_layers);
        }
        let mut trace = ImageDitTrace::new(&layers);
        let got = host(
            &model
                .forward_traced(step, &rows, &mut trace)
                .expect("forward"),
        );
        let d = |name: &str| reference.tensor(&format!("diag.step{step}.{name}"));
        let mods = model
            .adaln_cache()
            .modulation(0)
            .expect("block 0 modulation");
        let block0 = Tensor::stack(
            &mods
                .tables()
                .map(|t| blocks(model, t, step, 3))
                .collect::<Vec<_>>(),
            0,
        )
        .expect("stack");
        let norm_out = model.norm_out_modulation();
        let mut lines = vec![
            image_dit_stage_line(
                step,
                "temb",
                &blocks(model, model.temb(), step, 1),
                &d("temb"),
            ),
            image_dit_stage_line(
                step,
                "norm_out.shift",
                &blocks(model, &norm_out.shift, step, 1),
                &d("norm_out.shift"),
            ),
            image_dit_stage_line(
                step,
                "norm_out.scale",
                &blocks(model, &norm_out.scale, step, 1),
                &d("norm_out.scale"),
            ),
            image_dit_stage_line(step, "block0.adaln", &block0, &d("block0.adaln")),
        ];
        for (layer, hidden) in trace.hidden.iter().filter(|(l, _)| diag_layers.contains(l)) {
            lines.push(image_dit_stage_line(
                step,
                &format!("hidden.block{layer}"),
                hidden,
                &d(&format!("hidden.block{layer}")),
            ));
        }
        lines.push(image_dit_stage_line(
            step,
            "pre_head",
            trace.pre_head.as_ref().expect("the trace records norm_out"),
            &d("pre_head"),
        ));
        let want = reference.tensor(&format!("out.step{step}"));
        lines.push(image_dit_stage_line(step, "velocity", &got, &want));
        for line in lines {
            println!("{line}");
        }
        regions(step, "velocity", &got, &want);
        if step == 0 {
            image_edit_block_sweep(&reference, &trace, &sweep_layers, sweep_regions);
        }
        got
    };
    let got0 = run(&mut model, 0);
    let got3 = run(&mut model, 3);
    // Asserted step 3: the sampler's own x_3, reached by the reference from the same rows.
    let traj_rows = reference
        .tensor("diag.traj.rows3")
        .to_device(&dev)
        .expect("trajectory rows");
    let traj = host(&model.forward(3, &traj_rows).expect("on-trajectory forward"));
    let traj_want = reference.tensor("diag.traj.out3");
    println!(
        "{}",
        image_dit_stage_line(3, "on-trajectory", &traj, &traj_want)
    );
    regions(3, "on-trajectory", &traj, &traj_want);
    // The reference's own rounding: the same two asserted inputs at bf16 compute.
    let want0 = reference.tensor("out.step0");
    let bf16_step0 = reference.tensor("diag.bf16.step0");
    let bf16_traj = reference.tensor("diag.bf16.traj.out3");
    for (step, stage, got, want) in [
        (0, "ref bf16 vs f32", &bf16_step0, &want0),
        (3, "ref bf16 vs f32", &bf16_traj, &traj_want),
        (0, "vs ref bf16", &got0, &bf16_step0),
        (3, "traj vs ref bf16", &traj, &bf16_traj),
    ] {
        println!("{}", image_dit_stage_line(step, stage, got, want));
    }
    let want3 = reference.tensor("out.step3");
    assert_eq!(got0.dims(), want0.dims());
    // The family bound both backends assert (`gen_core::h3_image`), as the MLX edit gate does.
    let (p0, m0) = rel(&got0, &want0);
    let (pt, mt) = rel(&traj, &traj_want);
    let (p3, m3) = rel(&got3, &want3);
    let (l0, c0) = image_dit_rel_l2_cosine(&got0, &want0);
    let (lt, ct) = image_dit_rel_l2_cosine(&traj, &traj_want);
    let (l3, c3) = image_dit_rel_l2_cosine(&got3, &want3);
    let (between, _) = image_dit_rel_l2_cosine(&want3, &want0);
    println!(
        "image edit DiT vs diffusers (family bounds rel-L2 {DIT_PARITY_MAX_REL_L2:.0e}, cosine \
         {DIT_PARITY_MIN_COSINE}): step0 rel-L2 {l0:.3e} cosine {c0:.6}; step3 on-trajectory \
         rel-L2 {lt:.3e} cosine {ct:.6}; reference step0 vs step3 rel-L2 {between:.3e}"
    );
    println!(
        "image edit DiT peak/mean diagnostics (not asserted): step0 peak {p0:.3e} mean {m0:.3e}; \
         step3 on-trajectory peak {pt:.3e} mean {mt:.3e}"
    );
    println!(
        "image edit DiT step 3 off-trajectory (step-0 rows at t = 0.25; diagnostic, not \
         asserted): rel-L2 {l3:.3e} cosine {c3:.6} peak {p3:.3e} mean {m3:.3e}"
    );
    for (what, l2, cos) in [("step 0", l0, c0), ("step 3 on-trajectory", lt, ct)] {
        assert!(
            l2 <= DIT_PARITY_MAX_REL_L2,
            "{what} diverges: rel-L2 {l2:.3e}"
        );
        assert!(
            cos >= DIT_PARITY_MIN_COSINE,
            "{what} cosine {cos:.6} below {DIT_PARITY_MIN_COSINE}"
        );
    }
    assert!(
        between > DIT_PARITY_MAX_REL_L2,
        "steps 0 and 3 must be distinguishable"
    );
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_edit_first_step_matches_the_official_diffusers_transformer"
    );
}

/// The registered `minimax_h3_image` generator runs a 3-reference `edit` end to end on CUDA at a
/// requested 1024x768 — vision tower + grounded conditioner, the reference VAE encodes, six steps
/// over `[text | references | image]`, the tiled decode — and returns the requested size. Written
/// to `MINIMAX_H3_IMAGE_EDIT_RENDER_OUT` when set.
#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT, MiniMax-H3's text encoder incl. shard 14 (MINIMAX_H3_SNAPSHOT) and CUDA; full edit render"]
fn image_edit_renders_three_references_on_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the edit render runs the 66 GB DiT on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        use candle_gen::gen_core::{
            Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
        };

        let root = image_dit_snapshot();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        if !root.join("text_encoder").is_dir() {
            let h3 =
                std::path::PathBuf::from(std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default());
            assert!(
                h3.join("text_encoder")
                    .join("model-00014-of-00014.safetensors")
                    .is_file(),
                "MINIMAX_H3_SNAPSHOT={} must hold MiniMax-H3's text_encoder/ including shard 14 \
                 (the vision tower)",
                h3.display()
            );
            spec.components.insert(
                "text_encoder".into(),
                WeightsSource::Dir(h3.join("text_encoder")),
            );
        }
        let generator = candle_gen_minimax_h3::image_model::load_image(&spec)
            .expect("load the minimax_h3_image generator");
        let reference = |w: u32, h: u32, k: u32| Image {
            width: w,
            height: h,
            pixels: (0..w * h)
                .flat_map(|p| {
                    let (x, y) = (p % w, p / w);
                    [
                        ((x * 255) / w) as u8,
                        ((y * 255) / h) as u8,
                        (((x / (8 + k)) + (y / (8 + k))) % 2 * 200) as u8,
                    ]
                })
                .collect(),
        };
        let req = GenerationRequest {
            prompt: "Place the checkered tile from <Picture 1> on the floor of the room in \
                     <Picture 2>, with the colours of <Picture 3>"
                .into(),
            width: 1024,
            height: 768,
            seed: Some(7092),
            conditioning: vec![
                Conditioning::Reference {
                    image: reference(640, 480, 0),
                    strength: None,
                },
                Conditioning::MultiReference {
                    images: vec![reference(512, 768, 3), reference(1200, 1100, 6)],
                },
            ],
            ..Default::default()
        };
        generator.validate(&req).expect("validate");
        let started = std::time::Instant::now();
        let mut steps = Vec::new();
        let output = generator
            .generate(&req, &mut |p| {
                if let candle_gen::gen_core::Progress::Step { current, .. } = p {
                    steps.push(current)
                }
            })
            .expect("render");
        assert_eq!(steps, vec![1, 2, 3, 4, 5, 6]);
        let GenerationOutput::Images(images) = output else {
            panic!("an image generator must return images")
        };
        let img = &images[0];
        assert_eq!(
            (img.width, img.height),
            (1024, 768),
            "the size is the request's"
        );
        let n = img.pixels.len() as f64;
        let mean = img.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
        let std = (img
            .pixels
            .iter()
            .map(|&p| (f64::from(p) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        println!(
            "3-reference edit at 1024x768 in {:.1} s: mean {mean:.1}, std {std:.1}",
            started.elapsed().as_secs_f64()
        );
        assert!(std > 10.0, "degenerate render: std {std:.1}");
        if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_EDIT_RENDER_OUT") {
            image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
                .expect("rgb")
                .save(&out)
                .expect("write the render");
            println!("wrote {out}");
        }
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_edit_renders_three_references_on_cuda"
        );
    }
}

// ---- SW-7093: the q8 / q4 DiT tiers (candle/CUDA) ---------------------------------------------

/// The converter's output: `<root>/{q8,q4}/transformer_diffusers/` and
/// `<root>/{q8,q4}/pdd_fused_heads.safetensors` (`MINIMAX_H3_IMAGE_TIERS_ROOT`), written by the
/// `minimax_h3_image_prequant` example.
#[cfg(feature = "cuda")]
fn image_tiers_root() -> std::path::PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_TIERS_ROOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_TIERS_ROOT must point at the converter's output; this test is #[ignore]d \
         and asserts rather than skips."
    );
    std::path::PathBuf::from(raw)
}

/// Audit one converted tier by its safetensors index and headers: exactly the published 644 tensor
/// names plus a `.scales`/`.biases` pair per packed one, the six `pdd_proj_out.{i}` heads and the
/// I/O projections dense, the blocks and AdaLN packed, and the heads file a byte copy.
#[cfg(feature = "cuda")]
fn audit_image_tier(label: &str, bits: i64) {
    let tier = image_tiers_root().join(label);
    let dir = tier.join("transformer_diffusers");
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&config).unwrap();
    assert_eq!(cfg["quantization"]["bits"], bits, "{label} marker bits");
    assert_eq!(cfg["quantization"]["group_size"], 64);
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diffusion_pytorch_model.safetensors.index.json"))
            .unwrap(),
    )
    .unwrap();
    let keys: BTreeSet<String> = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let unpacked: BTreeSet<String> = keys
        .iter()
        .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
        .cloned()
        .collect();
    assert_eq!(unpacked.len(), 644, "{label}: the published tensor count");
    assert!(!unpacked.iter().any(|n| n.contains("audio")));
    for i in 0..6 {
        assert!(
            unpacked.contains(&format!("pdd_proj_out.{i}.weight")),
            "{label}: head {i} missing"
        );
        assert!(
            !keys.contains(&format!("pdd_proj_out.{i}.scales")),
            "{label}: pdd_proj_out.{i} was packed"
        );
    }
    for dense in [
        "proj_in",
        "context_embedder",
        "norm_out.linear",
        "time_embedder.linear_1",
    ] {
        assert!(
            !keys.contains(&format!("{dense}.scales")),
            "{label}: {dense} was packed"
        );
    }
    assert!(keys.contains("transformer_blocks.0.attn.to_q.scales"));
    assert!(keys.contains("transformer_blocks.49.adaln_proj.linear.scales"));
    let heads = image_dit_snapshot().join("pdd_fused_heads.safetensors");
    if heads.is_file() {
        assert_eq!(
            std::fs::read(tier.join("pdd_fused_heads.safetensors")).unwrap(),
            std::fs::read(&heads).unwrap(),
            "{label}: pdd_fused_heads.safetensors must be a byte copy"
        );
    }
}

/// Audit the converted tier, then render 2048 x 2048 through the registry's load path at it. The text encoder is
/// MiniMax-H3's own tier artifact (`MINIMAX_H3_TIER_SNAPSHOT/{tier}/text_encoder`), never a copy
/// under the image repository. A CUDA out-of-memory error fails the test loudly (the card may be
/// shared); the packed render needs less than bf16.
#[cfg(feature = "cuda")]
fn render_image_tier(label: &str, quant: candle_gen::gen_core::Quant) {
    use candle_gen::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};

    // Audit the converted index and heads before spending a render on them.
    audit_image_tier(label, if label == "q8" { 8 } else { 4 });

    let te =
        std::path::PathBuf::from(std::env::var("MINIMAX_H3_TIER_SNAPSHOT").unwrap_or_default())
            .join(label)
            .join("text_encoder");
    assert!(
        te.join("config.json").is_file(),
        "MINIMAX_H3_TIER_SNAPSHOT must hold MiniMax-H3's {label}/text_encoder/ (no {})",
        te.join("config.json").display()
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(image_dit_snapshot()));
    spec.quantize = Some(quant);
    spec.components.insert(
        "transformer_diffusers".into(),
        WeightsSource::Dir(image_tiers_root().join(label).join("transformer_diffusers")),
    );
    spec.components
        .insert("text_encoder".into(), WeightsSource::Dir(te));
    let generator = candle_gen_minimax_h3::image_model::load_image(&spec)
        .expect("load the minimax_h3_image generator at the packed tier");
    let req = GenerationRequest {
        prompt: "A red lighthouse on a rocky coast at dusk, waves breaking, warm light in the \
                 lantern room, photographed with a 35mm lens"
            .into(),
        width: 2048,
        height: 2048,
        seed: Some(7093),
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let mut steps = Vec::new();
    let output = generator
        .generate(&req, &mut |p| {
            if let candle_gen::gen_core::Progress::Step { current, .. } = p {
                steps.push(current)
            }
        })
        .unwrap_or_else(|e| panic!("{label} render failed (a CUDA OOM lands here): {e}"));
    assert_eq!(steps, vec![1, 2, 3, 4, 5, 6], "one forward per fixed step");
    let GenerationOutput::Images(images) = output else {
        panic!("an image generator must return images")
    };
    let img = &images[0];
    assert_eq!((img.width, img.height), (2048, 2048));
    let n = img.pixels.len() as f64;
    let mean = img.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
    let std = (img
        .pixels
        .iter()
        .map(|&p| (f64::from(p) - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    println!(
        "2048x2048 {label} render in {:.1} s: mean {mean:.1}, std {std:.1}",
        started.elapsed().as_secs_f64()
    );
    assert!(
        std > 10.0 && (5.0..250.0).contains(&mean),
        "degenerate {label} render: mean {mean:.1}, std {std:.1}"
    );
    if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_RENDER_OUT") {
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .expect("rgb")
            .save(format!("{out}-{label}.png"))
            .expect("write the render");
    }
}

/// AC1 (candle/CUDA): a 2048 x 2048 render at the q8 tier.
#[test]
#[ignore = "needs the converted q8 tier, MINIMAX_H3_IMAGE_SNAPSHOT, MiniMax-H3's q8 text encoder (MINIMAX_H3_TIER_SNAPSHOT) and CUDA; full 2048x2048 render"]
fn image_t2i_renders_2048_q8_on_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the 2048x2048 render runs the DiT on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        render_image_tier("q8", candle_gen::gen_core::Quant::Q8);
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_t2i_renders_2048_q8_on_cuda"
        );
    }
}

/// AC1 (candle/CUDA): a 2048 x 2048 render at the q4 tier.
#[test]
#[ignore = "needs the converted q4 tier, MINIMAX_H3_IMAGE_SNAPSHOT, MiniMax-H3's q4 text encoder (MINIMAX_H3_TIER_SNAPSHOT) and CUDA; full 2048x2048 render"]
fn image_t2i_renders_2048_q4_on_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the 2048x2048 render runs the DiT on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        render_image_tier("q4", candle_gen::gen_core::Quant::Q4);
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_t2i_renders_2048_q4_on_cuda"
        );
    }
}

// ---- SW-7095: the 16 MP streamed render (candle/CUDA rung 4) --------------------------------------

/// AC2 (candle/CUDA): a 16 MP (4096 x 4096) bf16 text-to-image request that does not fit
/// resident is admitted and completes through the streamed route.
///
/// The admission budget is the device's **actually-free** memory (`cuMemGetInfo` after a
/// synchronize), not its total: the shared runner's cards carry other work. The per-phase host
/// and GPU estimates of the staged and streamed rungs are printed in decimal GB and binary GiB.
/// The test asserts the streamed estimate fits that free memory, that the selector lands on rung 4
/// for any budget between the two estimates (on a busy card, the real one), that the loaded
/// generator refuses the staged rung whenever its estimate exceeds the free memory and admits the
/// streamed one, and then renders through the request scope that admission returns — the
/// one-block window over the text encoder and the DiT. The image is written to
/// `MINIMAX_H3_IMAGE_STREAMED_RENDER_OUT` when set.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT), MiniMax-H3's text encoder (MINIMAX_H3_SNAPSHOT) and CUDA; full 4096x4096 streamed render"]
fn image_t2i_renders_16mp_streamed_on_cuda() {
    #[cfg(not(feature = "cuda"))]
    panic!("the 16 MP streamed render runs on the GPU; build with --features cuda");
    #[cfg(feature = "cuda")]
    {
        use candle_gen::gen_core::{
            GenerationOutput, GenerationRequest, LoadShape, LoadSpec, MemoryBudget,
            MemoryCacheState, MemoryGeometry, MemoryMode, MemoryNumericTier,
            MemoryOptimizationAuthority, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision,
            MemoryStrategy, WeightsSource,
        };
        use candle_gen_minimax_h3::image_memory_strategy::admission::{
            admit, binary_gib, decimal_gb, estimate, ladder, ImageRequestRows, ImageRequestShape,
        };
        use candle_gen_minimax_h3::image_memory_strategy::facts_for;

        let root = image_dit_snapshot();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        if !root.join("text_encoder").is_dir() {
            let h3 = std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default();
            assert!(
                !h3.is_empty(),
                "the image snapshot has no text_encoder/; MINIMAX_H3_SNAPSHOT must point at a \
                 MiniMax-H3 snapshot holding text_encoder/ shards 1-12"
            );
            spec.components.insert(
                "text_encoder".into(),
                WeightsSource::Dir(std::path::PathBuf::from(h3).join("text_encoder")),
            );
        }
        let generator = candle_gen_minimax_h3::image_model::load_image(&spec)
            .expect("load the minimax_h3_image generator for deferred materialization");
        let contract = generator
            .memory_strategy_contract()
            .expect("the image generator publishes a memory contract")
            .clone();
        assert!(contract.conformance_errors().is_empty());

        // The device's actually-free memory, after any pending frees complete.
        let device = Device::new_cuda(0).expect("CUDA device 0");
        device.synchronize().expect("synchronize");
        let (free, total) = device
            .as_cuda_device()
            .expect("a CUDA device")
            .cuda_stream()
            .context()
            .mem_get_info()
            .expect("cuMemGetInfo");
        let (free, total) = (free as u64, total as u64);
        println!(
            "GPU 0: {:.2} GB ({:.2} GiB) free of {:.2} GB ({:.2} GiB)",
            decimal_gb(free),
            binary_gib(free),
            decimal_gb(total),
            binary_gib(total)
        );

        let req = GenerationRequest {
            prompt: "A red lighthouse on a rocky coast at dusk, waves breaking, warm light in the \
                     lantern room, photographed with a 35mm lens"
                .into(),
            width: 4096,
            height: 4096,
            seed: Some(7095),
            ..Default::default()
        };
        let facts = facts_for(&spec).expect("header-only facts of the real sources");
        assert!(
            facts.streamable,
            "the dense source must support the streamed route"
        );
        let stream = contract
            .phase_facts
            .as_ref()
            .and_then(|f| f.transformer_stream.as_ref())
            .expect("the contract publishes the DiT's trunk and per-block bytes");
        assert_eq!(stream, facts.transformer_stream());
        assert_eq!(stream.stacks[0].len(), 50);
        let shape = ImageRequestShape::from_request(&req).expect("shape");
        assert_eq!(
            ImageRequestRows::of(&shape).image_rows,
            16_384,
            "16 MP is 16,384 image tokens"
        );
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: None,
            component_precision_floors: &[],
        };
        let rungs = ladder(&contract, tier);
        let at = |strategy| {
            let selection = rungs
                .iter()
                .find(|s| s.strategy == strategy)
                .expect("the deferred contract offers the rung");
            estimate(&contract, &facts, &shape, selection).expect("estimate")
        };
        let staged = at(MemoryStrategy::StagedResidency);
        let streamed = at(MemoryStrategy::BoundedTransformerResidency);
        println!("staged:   {staged}");
        println!("streamed: {streamed}");
        assert!(
            streamed.peak_device_bytes() < staged.peak_device_bytes(),
            "streaming must lower the GPU peak"
        );
        assert!(
            streamed.peak_device_bytes() <= free,
            "the streamed 16 MP estimate ({streamed}) exceeds the {:.2} GB free on GPU 0",
            decimal_gb(free)
        );
        let budget_of = |bytes| MemoryBudget {
            total_bytes: bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        };
        let between = (staged.peak_device_bytes() + streamed.peak_device_bytes()) / 2;
        assert_eq!(
            admit(&contract, &facts, &shape, tier, budget_of(between))
                .expect("a rung fits")
                .estimate
                .selection
                .strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        let resident_fits = staged.peak_device_bytes() <= free;
        println!(
            "resident {} on this card; admission picks {:?}",
            if resident_fits {
                "fits"
            } else {
                "does NOT fit"
            },
            admit(&contract, &facts, &shape, tier, budget_of(free))
                .map(|a| a.estimate.selection.strategy)
                .ok()
        );

        // Admission through the loaded generator, at the real free memory.
        let calibration = contract.calibration.clone().expect("calibration identity");
        let context_for = |strategy, predicted| MemoryRunContext {
            reference_resources: None,
            selection: contract
                .representative_selection(strategy, tier, false)
                .expect("selection"),
            optimization_authority: MemoryOptimizationAuthority::Estimated,
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: LoadShape::DeferredMaterialization,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 4096,
                height: 4096,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: free,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: predicted,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "sw-7095-16mp-streamed".into(),
        };
        let staged_context = context_for(
            MemoryStrategy::StagedResidency,
            streamed.peak_device_bytes(),
        );
        if !resident_fits {
            assert!(
                matches!(
                    generator.memory_strategy_safety_check(&staged_context),
                    MemorySafetyDecision::Reject { .. }
                ),
                "the staged rung must be refused where its estimate exceeds the free memory"
            );
        }
        let context = context_for(
            MemoryStrategy::BoundedTransformerResidency,
            streamed.peak_device_bytes(),
        );
        assert_eq!(
            generator.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Accept
        );
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("admit the streamed route")
            .expect("a request scope");
        let mut request = req.clone();
        scope.configure_request(&mut request).expect("configure");
        let memory = request
            .memory
            .expect("the scope sets the request's memory policy");
        assert!(memory.stream_transformer_blocks && memory.chunk_attention);
        assert_eq!(memory.transformer_window_size, Some(1));

        let started = std::time::Instant::now();
        let mut steps = Vec::new();
        let output = generator
            .generate(&request, &mut |p| {
                if let candle_gen::gen_core::Progress::Step { current, .. } = p {
                    steps.push(current)
                }
            })
            .unwrap_or_else(|e| panic!("the streamed 16 MP render failed: {e}"));
        scope
            .finish(MemoryRunOutcome::Complete)
            .expect("finish the request scope");
        assert_eq!(steps, vec![1, 2, 3, 4, 5, 6], "one forward per fixed step");
        let GenerationOutput::Images(images) = output else {
            panic!("an image generator must return images")
        };
        let img = &images[0];
        assert_eq!((img.width, img.height), (4096, 4096));
        let n = img.pixels.len() as f64;
        let mean = img.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
        let std = (img
            .pixels
            .iter()
            .map(|&p| (f64::from(p) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        println!(
            "4096x4096 streamed render in {:.1} s: mean {mean:.1}, std {std:.1}",
            started.elapsed().as_secs_f64()
        );
        assert!(
            std > 10.0 && (5.0..250.0).contains(&mean),
            "degenerate streamed render: mean {mean:.1}, std {std:.1}"
        );
        if let Ok(out) = std::env::var("MINIMAX_H3_IMAGE_STREAMED_RENDER_OUT") {
            image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
                .expect("rgb")
                .save(&out)
                .expect("write the render");
            println!("wrote {out}");
        }
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::image_t2i_renders_16mp_streamed_on_cuda"
        );
    }
}
