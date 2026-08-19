//! sc-18766 — **LTX-2.5 DiffVAE (`NADiffusionDecoder`) parity on real weights.**
//!
//! The committed goldens (`tests/fixtures/ltx25_diffvae_golden.safetensors`, from
//! `tools/dump_ltx25_diffvae_golden.py`) hold reference I/O taken from the **upstream**
//! `DiffusionVideoDecoder` at Lightricks/LTX-2 v1.2.0 running on the real
//! `vae/ltx-2.5-video-vae-bf16.safetensors`. These tests load the same checkpoint through
//! [`mlx_gen_ltx::convert_vae_components`] and the shipped
//! [`mlx_gen_ltx::diff_vae::NaDiffusionDecoder`] and must reproduce it.
//!
//! Comparisons are **absolute**, never cosine: the decoder is a resampling stack whose failure
//! modes — a wrong rotary split, a transposed pixel shuffle, a missed per-channel un-normalisation
//! — are largely scale-preserving, and cosine similarity is exactly blind to those.
//!
//! Four levels, so a failure names a stage:
//!
//! | fixture | what it isolates |
//! | --- | --- |
//! | `t_emb` / `adaln` | the timestep embedder and the shared AdaLN-Zero projection |
//! | `na_*` | one deterministic `NABlock` (det stage 3, 3x5x5 window) |
//! | `diff_*` | one diffusion block (11x11x11 window) with real modulation |
//! | `dec_*` + `s123_slice` / `ctx_slice` | stages 1-3, stage 4, and the whole decode |
//!
//! `#[ignore]`d — needs the gated `Lightricks/LTX-2.5` weights and a GPU. `LTX25_VAE_DIR` points at
//! a directory holding the component checkpoints; the tests never search a cache for them
//! (inference source may not reference model caches — `scripts/check-workspace.py`). Run:
//!
//! ```text
//! LTX25_VAE_DIR=/path/to/Lightricks--LTX-2.5/vae \
//! LTX25_SPLIT_DIR=/tmp/ltx25-diffvae \
//!   cargo test -p mlx-gen-ltx --release --test ltx_2_5_diffvae_parity -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::ops::{abs, max as max_op, mean, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen_ltx::config::LtxVaeConfig;
use mlx_gen_ltx::convert::convert_vae_components;
use mlx_gen_ltx::diff_vae::{
    expected_weight_keys, looks_like_diffusion_decoder, DiffVaeTiling, NaDiffusionDecoder,
    NaDiffusionDecoderConfig, DIFFUSION_DECODER_COMPONENT, UNUSED_DECODER_KEYS,
};
use mlx_gen_ltx::vae::LtxVideoVae;

const DIFF_VAE: &str = "ltx-2.5-video-vae-bf16.safetensors";
const CONV_VAE: &str = "ltx-2.5-video-vae-conv-bf16.safetensors";

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx25_diffvae_golden.safetensors"
);
const CONV_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx25_conv_vae_golden.safetensors"
);

/// Absolute-error bar for a block or a decode against the reference.
///
/// The reference ran in f32 on CPU; this runs in f32 on Metal, where matmul accumulates at reduced
/// precision (~1e-3 relative per GEMM) and the decoder stacks 22 attention blocks. `2e-3` sits an
/// order of magnitude below any structural error — a wrong rotary split, a transposed shuffle, a
/// dropped ghost crop all land at 1e-1 or worse — and above the platform's own floor.
const BLOCK_TOL: f32 = 2e-3;

/// The whole-decode bar. Looser than a single block because the error compounds through stages 1-5,
/// and because the golden's pixels are stored f16 (see the dump script), which is itself ~1e-4.
const DECODE_TOL: f32 = 6e-3;

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn max_abs(a: &Array) -> f32 {
    max_op(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// `max |got - want|`, in the units of the signal itself.
fn abs_err(got: &Array, want: &Array) -> f32 {
    assert_eq!(
        got.shape(),
        want.shape(),
        "shape mismatch before comparison"
    );
    max_abs(&subtract(to_f32(got), to_f32(want)).unwrap())
}

fn to_f32(a: &Array) -> Array {
    to_dtype(a, Dtype::Float32).unwrap()
}

fn mean_abs(a: &Array) -> f32 {
    mean(abs(a).unwrap(), None).unwrap().item::<f32>()
}

fn vae_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("LTX25_VAE_DIR")?);
    dir.join(DIFF_VAE).is_file().then_some(dir)
}

/// The converted DiffVAE components, cached in `LTX25_SPLIT_DIR` when set.
fn split_components() -> Option<(PathBuf, Option<tempfile::TempDir>)> {
    let src = match vae_dir() {
        Some(d) => d,
        None => {
            eprintln!("skip: set LTX25_VAE_DIR to a directory holding {DIFF_VAE}");
            return None;
        }
    };
    let (out, guard) = match std::env::var_os("LTX25_SPLIT_DIR") {
        Some(d) => (PathBuf::from(d), None),
        None => {
            let tmp = tempfile::tempdir().expect("tempdir");
            (tmp.path().to_path_buf(), Some(tmp))
        }
    };
    let component = out.join(format!("{DIFFUSION_DECODER_COMPONENT}.safetensors"));
    if component.is_file() && out.join("embedded_config.json").is_file() {
        eprintln!("[convert] reusing cached components at {}", out.display());
        return Some((out, guard));
    }
    let t = Instant::now();
    let emitted = convert_vae_components(src.join(DIFF_VAE), None::<&Path>, &out)
        .expect("convert the LTX-2.5 DiffVAE");
    eprintln!(
        "[convert] {emitted:?} -> {} in {:.1}s",
        out.display(),
        t.elapsed().as_secs_f64()
    );
    Some((out, guard))
}

fn decoder() -> Option<(
    NaDiffusionDecoder,
    NaDiffusionDecoderConfig,
    Weights,
    Option<tempfile::TempDir>,
)> {
    let (dir, guard) = split_components()?;
    let cfg =
        NaDiffusionDecoderConfig::from_model_dir(&dir).expect("embedded_config.json vae.decoder");
    let w = Weights::from_file(dir.join(format!("{DIFFUSION_DECODER_COMPONENT}.safetensors")))
        .expect("the converted diffusion decoder");
    let decoder = NaDiffusionDecoder::from_weights(&w, &cfg).expect("build NaDiffusionDecoder");
    let golden =
        Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx25_diffvae_golden.py)");
    Some((decoder, cfg, golden, guard))
}

// ---------------------------------------------------------------------------------------------
// Conversion / loading
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB)"]
fn the_diffvae_converts_to_an_encoder_and_a_diffusion_decoder() {
    let Some((dir, _guard)) = split_components() else {
        return;
    };
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("split_model.json")).unwrap())
            .unwrap();
    let components: Vec<&str> = manifest["components"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        components,
        vec!["vae_encoder", DIFFUSION_DECODER_COMPONENT],
        "a CausalDiffusionVAE yields the conv encoder plus the NA diffusion decoder"
    );
    // The conv decoder's own component name must stay free: `LtxVideoVae::from_weights` would
    // happily try to build a conv stack out of whatever sits there.
    assert!(
        !dir.join("vae_decoder.safetensors").exists(),
        "the DiffVAE must never be written where the conv decoder is looked for"
    );

    let w =
        Weights::from_file(dir.join(format!("{DIFFUSION_DECODER_COMPONENT}.safetensors"))).unwrap();
    assert!(looks_like_diffusion_decoder(&w));
    let cfg = NaDiffusionDecoderConfig::from_model_dir(&dir).unwrap();
    let expected: std::collections::BTreeSet<String> =
        expected_weight_keys(&cfg).into_iter().collect();
    let present: std::collections::BTreeSet<String> = w.keys().map(str::to_string).collect();
    let missing: Vec<&String> = expected.difference(&present).collect();
    assert!(missing.is_empty(), "checkpoint is missing {missing:?}");
    let extra: Vec<&String> = present.difference(&expected).collect();
    assert_eq!(
        extra.len(),
        UNUSED_DECODER_KEYS.len(),
        "unexpected leftover tensors {extra:?} — the port may be skipping something"
    );
    for key in UNUSED_DECODER_KEYS {
        assert!(
            present.contains(*key),
            "{key} should still be carried through"
        );
    }

    // The 2.5 encoder still loads through the shipped conv port alongside it (sc-18765).
    let enc = Weights::from_file(dir.join("vae_encoder.safetensors")).unwrap();
    let vae_cfg = LtxVaeConfig::from_model_dir(&dir).unwrap();
    let encoder_only = LtxVideoVae::encoder_only(&enc, &vae_cfg).expect("conv encoder");
    assert!(encoder_only.has_encoder() && !encoder_only.has_decoder());
}

// ---------------------------------------------------------------------------------------------
// Level 1-3: embedder, one deterministic block, one diffusion block
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB) + a GPU"]
fn the_timestep_embedder_and_shared_adaln_match_the_reference() {
    let Some((decoder, cfg, g, _guard)) = decoder() else {
        return;
    };
    assert_eq!(cfg.timestep_scale_multiplier, 1000.0);
    let t_emb = decoder.timestep_embedding(1.0).unwrap();
    let err = abs_err(&t_emb, g.require("t_emb").unwrap());
    eprintln!(
        "[t_emb] max|delta| = {err:.3e} over |v| <= {:.3}",
        max_abs(&t_emb)
    );
    assert!(err < BLOCK_TOL, "timestep embedding max|delta| = {err:.3e}");

    let adaln = decoder.adaln_chunks(1.0).unwrap();
    let err = abs_err(&adaln, g.require("adaln").unwrap());
    eprintln!(
        "[adaln] max|delta| = {err:.3e} over |v| <= {:.3}",
        max_abs(&adaln)
    );
    // The AdaLN projection amplifies (its outputs reach |52|), so scale the bar with the signal.
    assert!(
        err < BLOCK_TOL * max_abs(&adaln).max(1.0),
        "shared AdaLN max|delta| = {err:.3e}"
    );
}

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB) + a GPU"]
fn one_deterministic_na_block_matches_the_reference() {
    let Some((decoder, _cfg, g, _guard)) = decoder() else {
        return;
    };
    let x = to_f32(g.require("na_in").unwrap());
    let want = g.require("na_out").unwrap();
    // det stage 3 (the stage-4 blocks), window 3x5x5.
    let got = decoder.det_block(3, 0, &x).unwrap();
    let err = abs_err(&got, want);
    eprintln!(
        "[na block] max|delta| = {err:.3e}, mean|delta| = {:.3e}, |ref| <= {:.3}",
        mean_abs(&subtract(&got, to_f32(want)).unwrap()),
        max_abs(&to_f32(want))
    );
    assert!(
        err < BLOCK_TOL,
        "deterministic NA block max|delta| = {err:.3e}"
    );
}

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB) + a GPU"]
fn one_diffusion_block_matches_the_reference() {
    let Some((decoder, _cfg, g, _guard)) = decoder() else {
        return;
    };
    let context = to_f32(g.require("diff_ctx").unwrap());
    let x = to_f32(g.require("diff_x").unwrap());
    let want = g.require("diff_out").unwrap();
    let got = decoder.diffusion_block(0, &context, &x, 1.0).unwrap();
    let err = abs_err(&got, want);
    eprintln!(
        "[diff block] max|delta| = {err:.3e}, mean|delta| = {:.3e}, |ref| <= {:.3}",
        mean_abs(&subtract(&got, to_f32(want)).unwrap()),
        max_abs(&to_f32(want))
    );
    assert!(err < BLOCK_TOL, "diffusion block max|delta| = {err:.3e}");

    // The context really is load-bearing: zeroing it must move the output well past the bar, or the
    // comparison above would pass on a block that ignored its context entirely.
    let no_context = decoder
        .diffusion_block(0, &Array::zeros::<f32>(context.shape()).unwrap(), &x, 1.0)
        .unwrap();
    let without = abs_err(&no_context, want);
    eprintln!("[diff block] context zeroed -> max|delta| = {without:.3e}");
    assert!(
        without > 20.0 * err,
        "zeroing the context barely moved the output ({without:.3e} vs {err:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Level 4: the whole decode
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB) + a GPU"]
fn the_full_decode_matches_the_reference() {
    let Some((decoder, cfg, g, _guard)) = decoder() else {
        return;
    };
    let latent = to_f32(g.require("dec_latent").unwrap());
    let noise = to_f32(g.require("dec_noise").unwrap());
    let want = g.require("dec_out").unwrap();

    let sh = latent.shape().to_vec();
    assert_eq!(
        cfg.noise_shape(sh[2], sh[3], sh[4]),
        [noise.shape()[2], noise.shape()[3], noise.shape()[4]],
        "the port must ask for exactly the stage-5 canvas the reference used"
    );

    // Stage-level anchors first: a whole-decode number cannot say which stage drifted.
    let (stage123, context) = decoder.stage_features(&latent).unwrap();
    let cut = |x: &Array| {
        let sh = x.shape().to_vec();
        x.take_axis(Array::from_slice(&[0, 1], &[2]), 2)
            .unwrap()
            .take_axis(Array::from_slice(&[0, 1], &[2]), 3)
            .unwrap()
            .reshape(&[sh[0], sh[1], 2, 2, sh[4]])
            .unwrap()
    };
    let s123_err = abs_err(&cut(&stage123), g.require("s123_slice").unwrap());
    let ctx_err = abs_err(&cut(&context), g.require("ctx_slice").unwrap());
    eprintln!(
        "[stages 1-3] max|delta| = {s123_err:.3e}  [stage 4 context] max|delta| = {ctx_err:.3e}"
    );
    assert!(
        s123_err < BLOCK_TOL,
        "stages 1-3 max|delta| = {s123_err:.3e}"
    );
    assert!(
        ctx_err < BLOCK_TOL,
        "stage-4 context max|delta| = {ctx_err:.3e}"
    );
    drop(stage123);
    drop(context);
    clear_cache();

    let t = Instant::now();
    reset_peak_memory();
    let got = decoder.decode(&latent, &noise).unwrap();
    got.eval().unwrap();
    let seconds = t.elapsed().as_secs_f64();
    let peak = get_peak_memory();

    let err = abs_err(&got, want);
    let mean_err = mean_abs(&subtract(&got, to_f32(want)).unwrap());
    eprintln!(
        "[decode] {:?} -> {:?} in {seconds:.1}s, peak {:.2} GiB | max|delta| = {err:.3e}, \
         mean|delta| = {mean_err:.3e}, |ref| <= {:.3}",
        latent.shape(),
        got.shape(),
        gib(peak),
        max_abs(&to_f32(want))
    );
    assert!(err < DECODE_TOL, "full decode max|delta| = {err:.3e}");

    // Decoding a different latent must not land on the same pixels — otherwise the comparison
    // above would pass on a decoder that ignored its input.
    let other = decoder
        .decode(&multiply(&latent, Array::from_f32(-1.0)).unwrap(), &noise)
        .unwrap();
    let moved = abs_err(&other, want);
    assert!(
        moved > 50.0 * err,
        "negating the latent barely moved the picture ({moved:.3e} vs {err:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Added scope: the conv VAE's own decode golden, backend-neutral for sc-18767
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 conv VAE (1.45 GB) + a GPU"]
fn the_conv_vae_decode_matches_the_v1_2_0_reference() {
    // sc-18765 established the 2.5 conv VAE round-trips through the shipped port at 53-58 dB. That
    // is self-consistency. This is the external check the epic's validation matrix asks for: the
    // reference `ConvVideoDecoder` on the same weights, compared absolutely. The fixture holds only
    // a latent and reference pixels, so `candle-gen-ltx` (sc-18767) asserts against the same file.
    let Some(src) = vae_dir() else {
        eprintln!("skip: set LTX25_VAE_DIR");
        return;
    };
    let conv = src.join(CONV_VAE);
    if !conv.is_file() {
        eprintln!("skip: {} not cached", conv.display());
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("conv");
    let components =
        convert_vae_components(&conv, None::<&Path>, &out).expect("convert the conv VAE");
    assert!(components.iter().any(|c| c == "vae_decoder"));

    let cfg = LtxVaeConfig::from_model_dir(&out).unwrap();
    let dec = Weights::from_file(out.join("vae_decoder.safetensors")).unwrap();
    let vae = LtxVideoVae::from_weights(&dec, None, &cfg).expect("build LtxVideoVae");

    let g = Weights::from_file(CONV_GOLDEN).expect("run tools/dump_ltx25_conv_vae_golden.py");
    let latent = to_f32(g.require("dec_in").unwrap());
    let want = g.require("dec_out").unwrap();
    let got = vae.decode(&latent).expect("conv decode");
    got.eval().unwrap();
    let err = abs_err(&got, want);
    eprintln!(
        "[2.5 conv VAE vs v1.2.0] {:?} -> {:?} max|delta| = {err:.3e}, mean|delta| = {:.3e}, \
         |ref| <= {:.3}",
        latent.shape(),
        got.shape(),
        mean_abs(&subtract(&got, to_f32(want)).unwrap()),
        max_abs(&to_f32(want))
    );
    assert!(err < BLOCK_TOL, "conv VAE decode max|delta| = {err:.3e}");
}

// ---------------------------------------------------------------------------------------------
// Tiling: seams and temporal continuity
// ---------------------------------------------------------------------------------------------

/// Largest jump between neighbouring slices along `axis`, as a fraction of the clip's own dynamic
/// range. A blend that dims or doubles a seam shows up here; a smooth picture does not.
/// A deterministic, smooth probe volume — the same shape of signal the goldens use, so the tiling
/// comparison runs on something the decoder can actually reconstruct rather than on white noise.
fn probe(shape: &[i32], seed: i32) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f64;
            let s = seed as f64;
            ((x * 0.013_1 + s * 1.7).sin() * (x * 0.007_3 - s * 0.31).cos() * 0.9
                + 0.1 * (x * 0.000_37 + s).sin()) as f32
        })
        .collect();
    Array::from_slice(&data, shape)
}

fn max_step(x: &Array, axis: i32) -> f32 {
    let len = x.shape()[axis as usize];
    let idx: Vec<i32> = (0..len - 1).collect();
    let next: Vec<i32> = (1..len).collect();
    let a = x
        .take_axis(Array::from_slice(&idx, &[len - 1]), axis)
        .unwrap();
    let b = x
        .take_axis(Array::from_slice(&next, &[len - 1]), axis)
        .unwrap();
    max_abs(&subtract(&b, &a).unwrap())
}

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB) + a GPU"]
fn tiled_decode_keeps_its_seams_and_its_temporal_continuity() {
    let Some((decoder, cfg, _g, _guard)) = decoder() else {
        return;
    };
    // A clip long enough that BOTH the temporal and a spatial axis actually split. The stage-5
    // halo is `ceil(depth * (kernel / 2) / stride)` = 20 stage-4 cells, so a tile has to be wider
    // than that before a split is even legal: 7 latent frames (49 pixel frames) gives a 25-cell
    // temporal grid, which is the shortest clip this decoder can be tiled in time at all.
    let (lt, lh, lw) = (7, 7, 7);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 1);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]], 2);
    let stage4 = cfg.stage4_shape(lt, lh, lw);
    let halo = cfg.tile_halo();
    eprintln!(
        "[tiling] latent {lt}x{lh}x{lw} -> stage-4 grid {stage4:?}, halo {halo:?}, min tile {:?}",
        cfg.min_tile_shape()
    );

    let untiled = decoder.decode(&latent, &noise).unwrap();
    untiled.eval().unwrap();
    clear_cache();

    // Split the temporal axis and one spatial axis. Time is included deliberately: the conv
    // decoder's tiler starved the temporal axis and corrupted the clip without erroring
    // (`tests/vae_decode_tiling_parity.rs`), so a temporal split that is never exercised is a guard
    // that has never been asked its question.
    let tile = [halo[0] + 3, halo[1] + 4, stage4[2]];
    let tiling = DiffVaeTiling {
        tile,
        overlap: halo,
    };
    let split_axes: Vec<usize> = (0..3).filter(|&a| tile[a] < stage4[a]).collect();
    assert!(
        split_axes.contains(&0) && split_axes.len() >= 2,
        "the temporal axis and at least one spatial axis must actually split, got {split_axes:?}"
    );
    let t = Instant::now();
    reset_peak_memory();
    let tiled = decoder.decode_tiled(&latent, &noise, &tiling).unwrap();
    tiled.eval().unwrap();
    let seconds = t.elapsed().as_secs_f64();
    let peak = get_peak_memory();
    assert_eq!(tiled.shape(), untiled.shape());

    let err = abs_err(&tiled, &untiled);
    let mean_err = mean_abs(&subtract(&tiled, &untiled).unwrap());
    eprintln!(
        "[tiled] tile {tile:?} axes {split_axes:?} in {seconds:.1}s, peak {:.2} GiB | \
         vs untiled max|delta| = {err:.3e}, mean|delta| = {mean_err:.3e}",
        gib(peak)
    );
    // Tiling truncates each tile's neighbourhood at its own border, so the two decodes are close
    // rather than equal. What must NOT happen is a seam: a localized jump the untiled decode does
    // not have.
    assert!(
        mean_err < 0.02,
        "tiled decode drifts from untiled by mean {mean_err:.3e} — that is not a seam, that is a \
         different picture"
    );

    for (axis, name) in [(2i32, "temporal"), (3, "height"), (4, "width")] {
        let tiled_step = max_step(&tiled, axis);
        let untiled_step = max_step(&untiled, axis);
        eprintln!("[seam] {name}: tiled max step {tiled_step:.4}, untiled {untiled_step:.4}");
        assert!(
            tiled_step < untiled_step * 1.5 + 0.02,
            "{name} seam: the tiled decode has a {tiled_step:.4} jump where the untiled one has \
             {untiled_step:.4} — the blend is dimming or doubling a seam"
        );
    }

    // The blend is a partition of unity, so brightness must be preserved rather than scaled.
    let tiled_mean = mean(&tiled, None).unwrap().item::<f32>();
    let untiled_mean = mean(&untiled, None).unwrap().item::<f32>();
    eprintln!("[seam] mean level tiled {tiled_mean:.5} vs untiled {untiled_mean:.5}");
    assert!(
        (tiled_mean - untiled_mean).abs() < 0.01,
        "the blend changed the picture's mean level"
    );
}

#[test]
#[ignore = "sc-18766: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB) + a GPU"]
fn a_starved_tiling_is_refused_on_every_axis() {
    let Some((decoder, cfg, _g, _guard)) = decoder() else {
        return;
    };
    let halo = cfg.tile_halo();
    let legal = DiffVaeTiling {
        tile: [halo[0] * 3, halo[1] * 3, halo[2] * 3],
        overlap: halo,
    };
    let latent = Array::zeros::<f32>(&[1, cfg.in_channels, 3, 7, 7]).unwrap();
    let shape5 = cfg.noise_shape(3, 7, 7);
    let noise =
        Array::zeros::<f32>(&[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]]).unwrap();
    for (axis, &halo_axis) in halo.iter().enumerate() {
        let mut starved = legal;
        starved.overlap[axis] = halo_axis - 1;
        let err = match decoder.decode_tiled(&latent, &noise, &starved) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("axis {axis}: an under-haloed tiling must be refused, not smeared"),
        };
        assert!(err.contains("halo"), "axis {axis}: {err}");
    }
}

// ---------------------------------------------------------------------------------------------
// Production geometries: peak memory and a visual A/B against the conv decoder
// ---------------------------------------------------------------------------------------------

/// A smooth, band-limited clip in `[-1, 1]`, `(1, 3, F, H, W)` — the same generator sc-18765 uses.
/// White noise is meaningless for a 32x/8x autoencoder; low-frequency structure is what it carries.
fn synthetic_clip(frames: i32, h: i32, w: i32) -> Array {
    let mut data = Vec::with_capacity((3 * frames * h * w) as usize);
    for c in 0..3 {
        for f in 0..frames {
            let t = f as f32 / frames.max(2) as f32;
            for y in 0..h {
                let v = y as f32 / h as f32;
                for x in 0..w {
                    let u = x as f32 / w as f32;
                    let value =
                        0.55 * ((6.0 * u + 1.7 * t + c as f32).sin()) * ((4.0 * v - 2.3 * t).cos())
                            + 0.25 * (2.0 * v - 1.0)
                            + 0.10 * ((13.0 * u * v + 3.0 * t).sin());
                    data.push(value.clamp(-1.0, 1.0));
                }
            }
        }
    }
    Array::from_slice(&data, &[1, 3, frames, h, w])
}

fn psnr_db(got: &Array, want: &Array) -> f32 {
    let diff = subtract(got, want).unwrap();
    let mse = mean(multiply(&diff, &diff).unwrap(), None)
        .unwrap()
        .item::<f32>();
    20.0 * (2.0f32).log10() - 10.0 * mse.max(1e-20).log10()
}

/// Geometry to measure at. `LTX25_DIFFVAE_GEOMETRY` selects one (`WxHxF`); unset runs both.
fn geometries() -> Vec<(i32, i32, i32)> {
    match std::env::var("LTX25_DIFFVAE_GEOMETRY").ok().as_deref() {
        Some("768x512x25") => vec![(768, 512, 25)],
        Some("1280x704x25") => vec![(1280, 704, 25)],
        Some(other) => panic!("unknown LTX25_DIFFVAE_GEOMETRY {other:?}"),
        None => vec![(768, 512, 25), (1280, 704, 25)],
    }
}

#[test]
#[ignore = "sc-18766: real-weight decode at production geometry — needs the gated weights + a GPU"]
fn peak_memory_and_quality_at_production_geometries() {
    let Some((decoder, cfg, _g, _guard)) = decoder() else {
        return;
    };
    let Some(src) = vae_dir() else { return };

    // The conv VAE supplies the latent (the DiffVAE's own encoder is the same conv encoder) and the
    // A/B reference decode. Built, used, and dropped before the diffusion decode runs, so the
    // measured peak is the diffusion decoder's own.
    let tmp = tempfile::tempdir().unwrap();
    let conv_dir = tmp.path().join("conv");
    convert_vae_components(src.join(CONV_VAE), None::<&Path>, &conv_dir).expect("convert conv VAE");
    let conv_cfg = LtxVaeConfig::from_model_dir(&conv_dir).unwrap();

    for (w, h, frames) in geometries() {
        let clip = synthetic_clip(frames, h, w);
        clip.eval().unwrap();

        let (latent, conv_pixels) = {
            let enc = Weights::from_file(conv_dir.join("vae_encoder.safetensors")).unwrap();
            let dec = Weights::from_file(conv_dir.join("vae_decoder.safetensors")).unwrap();
            let conv = LtxVideoVae::from_weights(&dec, Some(&enc), &conv_cfg).unwrap();
            let latent = conv.encode(&clip).expect("encode");
            latent.eval().unwrap();
            let pixels = conv.decode(&latent).expect("conv decode");
            pixels.eval().unwrap();
            (latent, pixels)
        };
        clear_cache();

        let shape5 = cfg.noise_shape(latent.shape()[2], latent.shape()[3], latent.shape()[4]);
        eprintln!(
            "[{w}x{h}x{frames}] latent {:?} stage-5 canvas {shape5:?}",
            latent.shape()
        );

        reset_peak_memory();
        let t = Instant::now();
        let pixels = decoder
            .decode_seeded(&latent, 18766, None)
            .expect("diffusion decode");
        pixels.eval().unwrap();
        let seconds = t.elapsed().as_secs_f64();
        let peak = get_peak_memory();

        assert_eq!(
            pixels.shape(),
            clip.shape(),
            "decode must return the input geometry"
        );
        let amplitude = max_abs(&pixels);
        assert!(
            amplitude.is_finite() && amplitude < 5.0,
            "decode returned out-of-range values (max|v| = {amplitude})"
        );

        // Visual A/B: the diffusion decoder is a quality upgrade over the conv one, not a different
        // picture. Reconstruction PSNR against the source clip is the comparable number; agreement
        // with the conv decode is the "same picture" claim.
        let diff_psnr = psnr_db(&pixels, &clip);
        let conv_psnr = psnr_db(&conv_pixels, &clip);
        let ab_psnr = psnr_db(&pixels, &conv_pixels);
        eprintln!(
            "[{w}x{h}x{frames}] diffusion decode {seconds:.1}s peak {:.2} GiB | PSNR vs source: \
             diffusion {diff_psnr:.2} dB, conv {conv_psnr:.2} dB | diffusion-vs-conv A/B \
             {ab_psnr:.2} dB",
            gib(peak)
        );
        assert!(
            diff_psnr > 20.0,
            "{w}x{h}x{frames}: diffusion decode PSNR {diff_psnr:.2} dB is not a reconstruction"
        );
        assert!(
            ab_psnr > 15.0,
            "{w}x{h}x{frames}: the diffusion decode is {ab_psnr:.2} dB from the conv decode — that \
             is a different picture, not a refinement"
        );
        drop(conv_pixels);
        clear_cache();

        // The memory lever. Precision is not one here — the pmetal bf16 SDPA/GEMM hazards
        // (`tests/bf16_sdpa_bug.rs`) are exactly what a 22-block attention stack amplifies — so
        // tiling is what a machine that cannot hold the untiled peak uses instead.
        let stage4 = cfg.stage4_shape(latent.shape()[2], latent.shape()[3], latent.shape()[4]);
        let halo = cfg.tile_halo();
        let mut tile = stage4;
        for axis in 1..3 {
            let candidate = (stage4[axis] / 2).max(halo[axis] + 1);
            if candidate < stage4[axis] {
                tile[axis] = candidate;
            }
        }
        if tile == stage4 {
            eprintln!("[{w}x{h}x{frames}] too small to tile at this halo; untiled peak stands");
        } else {
            let tiling = DiffVaeTiling {
                tile,
                overlap: halo,
            };
            reset_peak_memory();
            let t = Instant::now();
            let tiled = decoder
                .decode_seeded(&latent, 18766, Some(&tiling))
                .expect("tiled diffusion decode");
            tiled.eval().unwrap();
            let tiled_seconds = t.elapsed().as_secs_f64();
            let tiled_peak = get_peak_memory();
            assert_eq!(tiled.shape(), clip.shape());
            let tiled_psnr = psnr_db(&tiled, &clip);
            eprintln!(
                "[{w}x{h}x{frames}] tiled {tile:?} overlap {halo:?}: {tiled_seconds:.1}s peak \
                 {:.2} GiB (untiled {:.2} GiB) | PSNR vs source {tiled_psnr:.2} dB, vs untiled \
                 decode {:.2} dB",
                gib(tiled_peak),
                gib(peak),
                psnr_db(&tiled, &pixels)
            );
            assert!(
                tiled_peak < peak,
                "tiling must lower the peak: {:.2} GiB tiled vs {:.2} GiB untiled",
                gib(tiled_peak),
                gib(peak)
            );
            assert!(
                tiled_psnr > diff_psnr - 3.0,
                "tiling cost {:.2} dB of reconstruction quality",
                diff_psnr - tiled_psnr
            );
        }

        drop(pixels);
        drop(latent);
        clear_cache();
    }
}
