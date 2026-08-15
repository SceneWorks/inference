//! sc-17154: real-weight smokes for the candle MiniMax-H3 VAE decode paths.
//!
//! These are `#[ignore]`d — they need the ~9.7 GB `vae/` and ~0.6 GB `audio_vae/` components of a
//! `MiniMaxAI/MiniMax-H3` snapshot. Point `MINIMAX_H3_SNAPSHOT` at the snapshot root (the directory
//! holding `vae/`) and run:
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root> \
//!   cargo test -p candle-gen-minimax-h3 --test real_weights -- --ignored --nocapture
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

mod common;

use std::collections::BTreeSet;

use common::{flat, read_snapshot_file, rel, safetensors_keys, snapshot, std_dev};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::loader::sorted_safetensors;
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
/// it cannot be held to the committed fixture suite's 1e-5 either. Measured worst across the three
/// probes on the CPU lane:
///
/// | probe | mean | std |
/// |---|---|---|
/// | `keyframe` (1 frame, tiled at 320 px) | 1.355e-5 | 5.690e-6 |
/// | `clip` (17 frames, untiled) | 6.949e-6 | 1.137e-5 |
/// | `multiclip` (20 frames, padded to 2 clips) | 4.899e-6 | **1.730e-5** |
///
/// so the floor is **1.730e-5**, and this bound is ~11.6x it. That is deliberately looser than the
/// ~7x [`TOL`] takes over the fixture lane's floor, for one stated reason: those numbers were
/// measured **here, on the CPU lane**, and this arm also runs on the `--features cuda` Windows
/// runner (sc-18677), whose reduction orders differ again and whose own floor has not been
/// measured. The headroom covers a device change; it is not a claim about one.
///
/// **The exposure sits on exactly the quantity that already sets the floor**: the worst entry in
/// the table is a `std`, and `std` is `exp(0.5 * logvar)` (`vae_encoder.rs`), so an absolute wobble
/// in `logvar` comes back as a *relative* one in `std`, magnified by the exponential — and a device
/// change is what reorders the reductions in the conv stack feeding it. That is a reason to expect
/// movement there first, not a measurement of it.
///
/// What it is *not* is the 2e-2 house value it replaced (sc-19008 review): that sat 1155x above the
/// measurement and so stated a convention in the voice of a derivation. The band between the two is
/// not hypothetical — a 1e-3 relative drift injected into the encoder output measures 1.862e-3,
/// which **passes** at 2e-2 and fails here. If the CUDA lane ever lands above this, the fix is to
/// re-derive from a measurement taken there, not to restore a convention.
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
    let cos = common::cosine(&got, &want);
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
            let cos = common::cosine(got, want);
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
}
