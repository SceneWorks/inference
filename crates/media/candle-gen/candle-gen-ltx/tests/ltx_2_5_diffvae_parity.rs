//! sc-18767 — **LTX-2.5 DiffVAE (`NADiffusionDecoder`) on candle**: key-set conformance in ordinary
//! CI, and absolute-error parity against sc-18766's goldens on real weights.
//!
//! The goldens are **not re-recorded here**. `tests/ltx_2_5_diffvae_parity.rs` on the MLX side and
//! this file load the *same* committed files —
//! `mlx-gen-ltx/tests/fixtures/ltx25_diffvae_golden.safetensors` and
//! `ltx25_conv_vae_golden.safetensors`, both dumped from the upstream reference at
//! `Lightricks/LTX-2` v1.2.0 (`d1511477`) — so a divergence between the two backends is a red test
//! rather than two independently plausible pictures. Neither file is MLX-shaped: they hold a latent,
//! a noise canvas, per-stage inputs and reference outputs, and nothing else.
//!
//! Comparisons are **absolute**, never cosine: the decoder is a resampling stack whose failure
//! modes — a wrong rotary split, a transposed pixel shuffle, a missed per-channel un-normalisation
//! — are largely scale-preserving, and cosine similarity is exactly blind to those.
//!
//! Two tiers of test live here:
//!
//! * **Weights-free** (ordinary CI, every platform including the CUDA lane): the released
//!   checkpoint's recorded key set and shapes, from
//!   `docs/reference/sc-18765-vae-keysets/ltx-vae-keysets.json`, asserted against what the shipped
//!   loader reads — plus the two executed controls that the conv and diffusion decoders can never
//!   be loaded through each other.
//! * **Real weights** (`#[ignore]`d, `LTX25_VAE_DIR`-driven): the four golden levels, the tiling
//!   seam/continuity properties, and the 2.5 conv-VAE decode golden. The tests never search a cache
//!   for weights (inference source may not reference model caches — `scripts/check-workspace.py`).
//!
//! ```text
//! LTX25_VAE_DIR=/path/to/Lightricks--LTX-2.5/vae \
//!   cargo test -p candle-gen-ltx --release --test integration \
//!     -- ltx_2_5_diffvae_parity:: --ignored --nocapture
//! # on the CUDA lane, add --features cuda
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
#[cfg(feature = "cuda")]
use candle_gen::testkit::PeakSampler;
use candle_gen_ltx::config::LATENT_CHANNELS;
#[cfg(feature = "cuda")]
use candle_gen_ltx::diff_vae::budget::{
    estimated_diffvae_decode_peak_bytes, plan_diffvae_tiling, DecodeGeometry, DecodePlan,
    DiffVaeMode, HostNaSupport,
};
use candle_gen_ltx::diff_vae::{
    expected_weight_keys, looks_like_diffusion_decoder, DiffVaeTiling, NaDiffusionDecoder,
    NaDiffusionDecoderConfig, DECODER_PREFIX, STAT_MEAN_KEY, STAT_STD_KEY, UNUSED_DECODER_KEYS,
};
use candle_gen_ltx::vae::LtxVideoVae;

const DIFF_VAE: &str = "ltx-2.5-video-vae-bf16.safetensors";
const CONV_VAE: &str = "ltx-2.5-video-vae-conv-bf16.safetensors";

/// The DiffVAE goldens, owned by the MLX crate and consumed by both lanes (sc-18766).
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx25_diffvae_golden.safetensors"
);
/// The 2.5 conv-VAE decode golden, likewise shared (sc-18766's added scope, for this story).
const CONV_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/mlx-gen-ltx/tests/fixtures/ltx25_conv_vae_golden.safetensors"
);
/// The recorded LTX-2.5 VAE key sets (sc-18765), weights-free.
const KEYSETS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../docs/reference/sc-18765-vae-keysets/ltx-vae-keysets.json"
);

/// Absolute-error bar for a block or a decode against the reference — **the MLX lane's bar**, kept
/// identical so "the same goldens pass on both backends" is one number rather than two.
///
/// MLX sized it for Metal, where matmul accumulates at reduced precision (~1e-3 relative per GEMM)
/// across 22 attention blocks. candle's f32 GEMMs have headroom below it on both CPU and CUDA; the
/// bar is still the right contract, because what it excludes is structural error — a wrong rotary
/// split, a transposed shuffle, a dropped ghost crop all land at 1e-1 or worse.
const BLOCK_TOL: f32 = 2e-3;

/// The whole-decode bar. Looser than a single block because the error compounds through stages 1-5,
/// and because the golden's pixels are stored f16 (see the dump script), which is itself ~1e-4.
const DECODE_TOL: f32 = 6e-3;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn keysets() -> serde_json::Value {
    let text = std::fs::read_to_string(KEYSETS).unwrap_or_else(|e| {
        panic!("read {KEYSETS}: {e} (regenerate with tools/dump_ltx_vae_keysets.py)")
    });
    serde_json::from_str(&text).expect("parse ltx-vae-keysets.json")
}

fn component_tensors(component: &str) -> serde_json::Map<String, serde_json::Value> {
    keysets()["components"][component]["tensors"]
        .as_object()
        .unwrap_or_else(|| panic!("fixture component {component} has no tensors"))
        .clone()
}

/// The CUDA device when the feature is on and a GPU is present, else CPU. The port is
/// device-agnostic; only the affordable geometry differs.
fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

/// The provisioned VAE directory, or a panic.
///
/// Deliberately not a skip: every caller is already `#[ignore]`d, which is the opt-out. A silent
/// skip made an unprovisioned run report exactly the same green as a real one, so a GPU box with
/// `LTX25_VAE_DIR` unset or pointing at the wrong tree looked like passing goldens.
fn vae_dir() -> PathBuf {
    let dir = match std::env::var_os("LTX25_VAE_DIR") {
        Some(d) => PathBuf::from(d),
        None => panic!("LTX25_VAE_DIR must point at a directory holding {DIFF_VAE}"),
    };
    assert!(
        dir.join(DIFF_VAE).is_file(),
        "LTX25_VAE_DIR must point at a directory holding {DIFF_VAE}; {} has none",
        dir.display()
    );
    dir
}

fn to_f32(t: &Tensor) -> Tensor {
    t.to_dtype(DType::F32).expect("f32")
}

fn values(t: &Tensor) -> Vec<f32> {
    to_f32(t)
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host copy")
}

/// `max |got - want|`, in the units of the signal itself.
fn abs_err(got: &Tensor, want: &Tensor) -> f32 {
    assert_eq!(got.dims(), want.dims(), "shape mismatch before comparison");
    values(got)
        .iter()
        .zip(values(want))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_err(got: &Tensor, want: &Tensor) -> f32 {
    let a = values(got);
    let b = values(want);
    a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

fn max_abs(t: &Tensor) -> f32 {
    values(t).iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

fn mean(t: &Tensor) -> f32 {
    let v = values(t);
    v.iter().sum::<f32>() / v.len() as f32
}

/// The committed golden tensors, on `device`.
fn golden(path: &str, device: &Device) -> std::collections::HashMap<String, Tensor> {
    candle_gen::candle_core::safetensors::load(path, device)
        .unwrap_or_else(|e| panic!("load {path}: {e}"))
}

fn require<'a>(g: &'a std::collections::HashMap<String, Tensor>, key: &str) -> &'a Tensor {
    g.get(key).unwrap_or_else(|| panic!("golden has no {key}"))
}

/// The decoder built straight off the released checkpoint — no conversion, no key remap.
fn decoder(device: &Device) -> (NaDiffusionDecoder, NaDiffusionDecoderConfig) {
    let path = vae_dir().join(DIFF_VAE);
    let cfg =
        NaDiffusionDecoderConfig::from_checkpoint(&path).expect("the checkpoint's vae config");
    let t = Instant::now();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, device)
            .expect("mmap the DiffVAE checkpoint")
    };
    let decoder = NaDiffusionDecoder::load(vb.pp(DECODER_PREFIX), vb.clone(), &cfg)
        .expect("build NaDiffusionDecoder straight off the released checkpoint");
    eprintln!(
        "[load] {} in {:.1}s",
        path.display(),
        t.elapsed().as_secs_f64()
    );
    (decoder, cfg)
}

// ---------------------------------------------------------------------------------------------
// Weights-free: the released key set, and the two decoders never loading through each other
// ---------------------------------------------------------------------------------------------

#[test]
fn the_loader_reads_every_decoder_tensor_the_released_checkpoint_carries() {
    let recorded = component_tensors("ltx_2_5_video_vae_diffusion");
    let cfg = NaDiffusionDecoderConfig::from_embedded_vae(
        &keysets()["components"]["ltx_2_5_video_vae_diffusion"]["config"]["vae"],
    )
    .expect("the recorded config parses");

    let present: std::collections::BTreeSet<String> = recorded
        .keys()
        .filter_map(|k| k.strip_prefix("decoder.").map(str::to_string))
        .collect();
    let mut expected: std::collections::BTreeSet<String> =
        expected_weight_keys(&cfg).into_iter().collect();
    let missing: Vec<&String> = expected.difference(&present).collect();
    assert!(
        missing.is_empty(),
        "the port reads {missing:?}, which the released checkpoint does not carry"
    );
    for key in UNUSED_DECODER_KEYS {
        assert!(
            present.contains(*key),
            "{key} is listed as a known-unused key but the checkpoint does not carry it"
        );
        expected.insert((*key).to_string());
    }
    let extra: Vec<&String> = present.difference(&expected).collect();
    assert!(
        extra.is_empty(),
        "the released checkpoint carries {extra:?}, which the port neither reads nor declares dead \
         — the port may be skipping something"
    );

    // The statistics sit *above* the decoder, under the file root, and keep LTX-2.3's `-of-means`
    // spelling. That is the property that lets the candle port skip a conversion step entirely.
    for key in [STAT_MEAN_KEY, STAT_STD_KEY] {
        assert!(recorded.contains_key(key), "the checkpoint has no {key}");
        assert_eq!(
            recorded[key].as_array().expect("shape").len(),
            1,
            "{key} must be a 1-D per-channel vector"
        );
    }
}

#[test]
fn the_recorded_shapes_are_the_widths_the_config_declares() {
    // The loader cross-checks config against weights at load; this asserts the *released* pair
    // actually agrees, weights-free, so a config drift is caught without 1.5 GB of downloads.
    let recorded = component_tensors("ltx_2_5_video_vae_diffusion");
    let cfg = NaDiffusionDecoderConfig::from_embedded_vae(
        &keysets()["components"]["ltx_2_5_video_vae_diffusion"]["config"]["vae"],
    )
    .expect("config");
    let shape = |key: &str| -> Vec<usize> {
        recorded[key]
            .as_array()
            .unwrap_or_else(|| panic!("no recorded shape for {key}"))
            .iter()
            .map(|v| v.as_u64().expect("dim") as usize)
            .collect()
    };
    assert_eq!(
        shape("decoder.conv_in.weight"),
        vec![cfg.stage_channels[0], cfg.in_channels]
    );
    assert_eq!(
        shape("decoder.conv_out.weight"),
        vec![
            cfg.out_channels * cfg.patch_size * cfg.patch_size,
            cfg.stage5_width()
        ]
    );
    assert_eq!(
        shape("decoder.shared_adaln.proj.weight"),
        vec![7 * cfg.stage5_width(), cfg.t_emb_dim]
    );
    // The fused qkv is 3x the stage width — the split the loader performs.
    assert_eq!(
        shape("decoder.det_stages.0.0.attn.qkv.weight"),
        vec![3 * cfg.stage_channels[0], cfg.stage_channels[0]]
    );
    assert_eq!(
        shape("decoder.diff_blocks.0.scale_shift_table"),
        vec![7, cfg.stage5_width()]
    );
    assert_eq!(shape(STAT_MEAN_KEY), vec![LATENT_CHANNELS]);
}

#[test]
fn the_two_2_5_video_decoders_can_never_be_loaded_through_each_other() {
    let diffusion: Vec<String> = component_tensors("ltx_2_5_video_vae_diffusion")
        .keys()
        .cloned()
        .collect();
    let conv: Vec<String> = component_tensors("ltx_2_5_video_vae_conv")
        .keys()
        .cloned()
        .collect();
    assert!(looks_like_diffusion_decoder(
        diffusion.iter().map(String::as_str)
    ));
    assert!(
        !looks_like_diffusion_decoder(conv.iter().map(String::as_str)),
        "the conv VAE must not classify as a diffusion decoder"
    );

    // Executed controls, not just a classifier: build both loaders against both key sets.
    let device = Device::Cpu;
    let dummy = |keys: &[String]| -> VarBuilder<'static> {
        let tensors: std::collections::HashMap<String, Tensor> = keys
            .iter()
            .map(|k| {
                (
                    k.clone(),
                    Tensor::zeros(vec![2usize; 2], DType::F32, &device).expect("dummy"),
                )
            })
            .collect();
        VarBuilder::from_tensors(tensors, DType::F32, &device)
    };
    let cfg = NaDiffusionDecoderConfig::from_embedded_vae(
        &keysets()["components"]["ltx_2_5_video_vae_diffusion"]["config"]["vae"],
    )
    .expect("config");

    let vb = dummy(&conv);
    NaDiffusionDecoder::load(vb.pp(DECODER_PREFIX), vb.clone(), &cfg)
        .map(|_| ())
        .expect_err("the conv VAE's tensors must not build an NADiffusionDecoder");

    // ... and the conv port must not build itself out of the diffusion decoder's tensors. The conv
    // decode path is unchanged by this story, and this is what says so. `LtxVideoVae::new` takes the
    // file root and roots itself at `decoder.`, exactly as it does for a real 2.5 conv component.
    let vb = dummy(&diffusion);
    LtxVideoVae::new(vb, LATENT_CHANNELS, 4)
        .map(|_| ())
        .expect_err("the diffusion decoder's tensors must not build a conv LtxVideoVae");
}

// ---------------------------------------------------------------------------------------------
// Real weights, level 1-3: embedder, one deterministic block, one diffusion block
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "sc-18767: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB)"]
fn the_timestep_embedder_and_shared_adaln_match_the_reference() {
    let device = device();
    let (decoder, cfg) = decoder(&device);
    let g = golden(GOLDEN, &device);
    assert_eq!(cfg.timestep_scale_multiplier, 1000.0);

    let t_emb = decoder.timestep_embedding(1.0).expect("t_emb");
    let err = abs_err(&t_emb, require(&g, "t_emb"));
    eprintln!(
        "[t_emb] max|delta| = {err:.3e} over |v| <= {:.3}",
        max_abs(&t_emb)
    );
    assert!(err < BLOCK_TOL, "timestep embedding max|delta| = {err:.3e}");

    let adaln = decoder.adaln_chunks(1.0).expect("adaln");
    let err = abs_err(&adaln, require(&g, "adaln"));
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
#[ignore = "sc-18767: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB)"]
fn one_deterministic_na_block_matches_the_reference() {
    let device = device();
    let (decoder, _cfg) = decoder(&device);
    let g = golden(GOLDEN, &device);
    let x = to_f32(require(&g, "na_in"));
    let want = require(&g, "na_out");
    // det stage 3 (the stage-4 blocks), window 3x5x5 — the same block the MLX golden records.
    let got = decoder.det_block(3, 0, &x).expect("det block");
    let err = abs_err(&got, want);
    eprintln!(
        "[na block] max|delta| = {err:.3e}, mean|delta| = {:.3e}, |ref| <= {:.3}",
        mean_abs_err(&got, want),
        max_abs(want)
    );
    assert!(
        err < BLOCK_TOL,
        "deterministic NA block max|delta| = {err:.3e}"
    );
}

#[test]
#[ignore = "sc-18767: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB)"]
fn one_diffusion_block_matches_the_reference() {
    let device = device();
    let (decoder, _cfg) = decoder(&device);
    let g = golden(GOLDEN, &device);
    let context = to_f32(require(&g, "diff_ctx"));
    let x = to_f32(require(&g, "diff_x"));
    let want = require(&g, "diff_out");
    let got = decoder
        .diffusion_block(0, &context, &x, 1.0)
        .expect("diffusion block");
    let err = abs_err(&got, want);
    eprintln!(
        "[diff block] max|delta| = {err:.3e}, mean|delta| = {:.3e}, |ref| <= {:.3}",
        mean_abs_err(&got, want),
        max_abs(want)
    );
    assert!(err < BLOCK_TOL, "diffusion block max|delta| = {err:.3e}");

    // The context really is load-bearing: zeroing it must move the output well past the bar, or the
    // comparison above would pass on a block that ignored its context entirely.
    let zeros = context.zeros_like().expect("zeros");
    let no_context = decoder
        .diffusion_block(0, &zeros, &x, 1.0)
        .expect("context-free block");
    let without = abs_err(&no_context, want);
    eprintln!("[diff block] context zeroed -> max|delta| = {without:.3e}");
    assert!(
        without > 20.0 * err,
        "zeroing the context barely moved the output ({without:.3e} vs {err:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Real weights, level 4: the whole decode
// ---------------------------------------------------------------------------------------------

/// The `[:, :, :2, :2, :]` corner of a stage feature — the slice the golden records, so the
/// comparison does not need the whole 1x17x28x28x512 volume on the host.
fn corner(x: &Tensor) -> Tensor {
    x.narrow(2, 0, 2)
        .expect("h")
        .narrow(3, 0, 2)
        .expect("w")
        .contiguous()
        .expect("corner")
}

#[test]
#[ignore = "sc-18767: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB)"]
fn the_full_decode_matches_the_reference() {
    let device = device();
    let (decoder, cfg) = decoder(&device);
    let g = golden(GOLDEN, &device);
    let latent = to_f32(require(&g, "dec_latent"));
    let noise = to_f32(require(&g, "dec_noise"));
    let want = require(&g, "dec_out");

    let sh = latent.dims().to_vec();
    assert_eq!(
        cfg.noise_shape(sh[2], sh[3], sh[4]).to_vec(),
        noise.dims()[2..].to_vec(),
        "the port must ask for exactly the stage-5 canvas the reference used"
    );

    // Stage-level anchors first: a whole-decode number cannot say which stage drifted.
    let (stage123, context) = decoder.stage_features(&latent).expect("stage features");
    let s123_err = abs_err(&corner(&stage123), require(&g, "s123_slice"));
    let ctx_err = abs_err(&corner(&context), require(&g, "ctx_slice"));
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

    let t = Instant::now();
    let got = decoder.decode(&latent, &noise).expect("decode");
    let seconds = t.elapsed().as_secs_f64();

    let err = abs_err(&got, want);
    eprintln!(
        "[decode] {:?} -> {:?} in {seconds:.1}s | max|delta| = {err:.3e}, mean|delta| = {:.3e}, \
         |ref| <= {:.3}",
        latent.dims(),
        got.dims(),
        mean_abs_err(&got, want),
        max_abs(want)
    );
    assert!(err < DECODE_TOL, "full decode max|delta| = {err:.3e}");

    // Decoding a different latent must not land on the same pixels — otherwise the comparison
    // above would pass on a decoder that ignored its input.
    let other = decoder
        .decode(&(latent * -1.0).expect("negate"), &noise)
        .expect("negated decode");
    let moved = abs_err(&other, want);
    // Reported, like the diffusion block's zeroed-context control, so a run's evidence carries the
    // separation it achieved rather than only the fact that it cleared the bar.
    eprintln!(
        "[decode] latent negated -> max|delta| = {moved:.3e} ({:.0}x)",
        moved / err
    );
    assert!(
        moved > 50.0 * err,
        "negating the latent barely moved the picture ({moved:.3e} vs {err:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Real weights: the conv VAE's own decode golden — the path this story must leave working
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "sc-18767: needs the gated Lightricks/LTX-2.5 conv VAE (1.45 GB)"]
fn the_conv_vae_decode_still_matches_the_v1_2_0_reference() {
    let device = device();
    let conv = vae_dir().join(CONV_VAE);
    assert!(
        conv.is_file(),
        "LTX25_VAE_DIR must point at a directory holding {CONV_VAE}; {} is missing",
        conv.display()
    );
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&conv], DType::F32, &device).expect("mmap conv VAE")
    };
    let vae = LtxVideoVae::new(vb, LATENT_CHANNELS, 4).expect("build the conv LtxVideoVae");

    let g = golden(CONV_GOLDEN, &device);
    let latent = to_f32(require(&g, "dec_in"));
    let want = require(&g, "dec_out");
    let got = vae.decode(&latent).expect("conv decode");
    let err = abs_err(&got, want);
    eprintln!(
        "[2.5 conv VAE vs v1.2.0] {:?} -> {:?} max|delta| = {err:.3e}, mean|delta| = {:.3e}, \
         |ref| <= {:.3}",
        latent.dims(),
        got.dims(),
        mean_abs_err(&got, want),
        max_abs(want)
    );
    assert!(err < BLOCK_TOL, "conv VAE decode max|delta| = {err:.3e}");
}

// ---------------------------------------------------------------------------------------------
// Real weights: tiling seams and temporal continuity
// ---------------------------------------------------------------------------------------------

/// A deterministic, smooth probe volume — the same generator the MLX seam test uses, so the two
/// lanes' tiling numbers are comparable.
fn probe(shape: &[usize], seed: usize, device: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f64;
            let s = seed as f64;
            ((x * 0.013_1 + s * 1.7).sin() * (x * 0.007_3 - s * 0.31).cos() * 0.9
                + 0.1 * (x * 0.000_37 + s).sin()) as f32
        })
        .collect();
    Tensor::from_vec(data, shape, device).expect("probe")
}

/// Largest jump between neighbouring slices along `axis`. A blend that dims or doubles a seam shows
/// up here; a smooth picture does not.
fn max_step(x: &Tensor, axis: usize) -> f32 {
    let len = x.dims()[axis];
    let a = x.narrow(axis, 0, len - 1).expect("a");
    let b = x.narrow(axis, 1, len - 1).expect("b");
    abs_err(&b, &a)
}

#[test]
#[ignore = "sc-18767: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB)"]
fn tiled_decode_keeps_its_seams_and_its_temporal_continuity() {
    let device = device();
    let (decoder, cfg) = decoder(&device);
    // A clip long enough that BOTH the temporal and a spatial axis actually split. The stage-5
    // halo is 20 stage-4 cells, so a tile has to be wider than that before a split is even legal:
    // 7 latent frames (49 pixel frames) gives a 25-cell temporal grid, which is the shortest clip
    // this decoder can be tiled in time at all.
    let (lt, lh, lw) = (7usize, 7, 7);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 1, &device);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        2,
        &device,
    );
    let stage4 = cfg.stage4_shape(lt, lh, lw);
    let halo = cfg.tile_halo();
    eprintln!(
        "[tiling] latent {lt}x{lh}x{lw} -> stage-4 grid {stage4:?}, halo {halo:?}, min tile {:?}",
        cfg.min_tile_shape()
    );

    let untiled = decoder.decode(&latent, &noise).expect("untiled decode");

    // Split the temporal axis and one spatial axis. Time is included deliberately: a tiler that
    // starves the temporal axis smears the clip without erroring, so a temporal split that is never
    // exercised is a guard that has never been asked its question.
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
    let tiled = decoder
        .decode_tiled(&latent, &noise, &tiling)
        .expect("tiled decode");
    let seconds = t.elapsed().as_secs_f64();
    assert_eq!(tiled.dims(), untiled.dims());

    let err = abs_err(&tiled, &untiled);
    let mean_err = mean_abs_err(&tiled, &untiled);
    eprintln!(
        "[tiled] tile {tile:?} axes {split_axes:?} in {seconds:.1}s | vs untiled max|delta| = \
         {err:.3e}, mean|delta| = {mean_err:.3e}"
    );
    // Tiling truncates each tile's neighbourhood at its own border, so the two decodes are close
    // rather than equal. What must NOT happen is a seam: a localized jump the untiled decode does
    // not have.
    assert!(
        mean_err < 0.02,
        "tiled decode drifts from untiled by mean {mean_err:.3e} — that is not a seam, that is a \
         different picture"
    );

    for (axis, name) in [(2usize, "temporal"), (3, "height"), (4, "width")] {
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
    let (a, b) = (mean(&tiled), mean(&untiled));
    eprintln!("[seam] mean level tiled {a:.5} vs untiled {b:.5}");
    assert!(
        (a - b).abs() < 0.01,
        "the blend changed the picture's mean level"
    );
}

/// CUDA-only acceptance substrate for sc-18783.  This story deliberately does not run the
/// campaign or promote coefficients: it makes the production-latent measurement reject the exact
/// one-byte-under-observed edge, so an under-predicting future coefficient cannot look green.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "sc-18783: needs an idle CUDA host plus LTX25_VAE_DIR; terminal coefficient campaign"]
fn budgeted_diffvae_estimate_never_under_predicts_the_measured_peak() {
    let device = Device::new_cuda(0).expect("CUDA device");
    let (decoder, cfg) = decoder(&device);
    // 1280x704x25 — the normal LTX-2.5 production latent, rather than a tiny fixture which would
    // leave every accumulator and halo arm unexercised.
    let (lt, lh, lw) = (4usize, 22, 40);
    let latent = probe(&[1, cfg.in_channels, lt, lh, lw], 18799, &device);
    let shape5 = cfg.noise_shape(lt, lh, lw);
    let noise = probe(
        &[1, cfg.out_channels, shape5[0], shape5[1], shape5[2]],
        18800,
        &device,
    );
    let resolved = DiffVaeMode::ChunkedEager
        .resolve_for_host(HostNaSupport::detect(&device))
        .expect("Candle's eager DiffVAE mode runs on CUDA");
    let geometry = DecodeGeometry::new(&cfg, lt, lh, lw);

    // The exact, untiled decode is the measured reference.  Sampling begins after construction so
    // the peak covers the construction the estimator declares (resident weights/context) and the
    // decode, but no unrelated loading activity.  Run on an otherwise idle GPU.
    let sampler = PeakSampler::start(0);
    let t = Instant::now();
    let untiled = decoder
        .decode(&latent, &noise)
        .expect("exact untiled decode");
    device.synchronize().expect("synchronize untiled decode");
    let measured = sampler.stop() * 1024 * 1024;
    let estimated =
        estimated_diffvae_decode_peak_bytes(&geometry, DecodePlan::SinglePass, &resolved);
    assert!(
        estimated >= measured,
        "under-predicts exact 1280x704x25 untiled peak: estimated {estimated} < measured {measured} bytes"
    );

    // This is the important edge: subtracting one byte from the observed peak must not leave the
    // single-pass arm selectable.  `Some` exercises a bounded tiled plan; an `Err` would also be
    // safe, but this production geometry has legal tiles and must take the bounded path.
    let edge_gib =
        (measured + resolved.budget_safety_bytes - 1) as f64 / (1024.0 * 1024.0 * 1024.0);
    let tiling = plan_diffvae_tiling(&cfg, lt, lh, lw, edge_gib, &resolved)
        .expect("the one-byte-under edge still has a legal bounded tile")
        .expect("the one-byte-under edge must not select the untiled decode");
    assert!(
        (0..3).any(|axis| tiling.tile[axis] < geometry.stage4[axis]),
        "the edge plan must actually tile: {tiling:?} vs {:?}",
        geometry.stage4
    );

    let tiled = decoder
        .decode_tiled(&latent, &noise, &tiling)
        .expect("bounded decode at the under-peak edge");
    device.synchronize().expect("synchronize bounded decode");
    let drift = mean_abs_err(&tiled, &untiled);
    assert!(
        drift < 0.02,
        "bounded decode drifted from exact untiled decode by mean {drift:.3e} after {:.1}s",
        t.elapsed().as_secs_f64()
    );
}
