//! sc-17149: audio-VAE **encode** parity against `diffusers.AutoencoderKLMiniMaxH3Audio`.
//!
//! Fixture `tests/fixtures/audio_vae_encode.safetensors` ←
//! `tools/dump_minimax_h3_audio_vae_encode.py`.
//!
//! # Why diffusers is the reference here, and not the snapshot's own bundle
//!
//! The decode half (`tests/audio_vae_parity.rs`) runs against `FL2VA/audio_vae`'s `DacAudioVAE`.
//! That bundle is **inference-only and has no `encode` method at all** — it ships `preprocess`
//! and `decode` — so `AutoencoderKLMiniMaxH3Audio` is the only executable reference for this
//! half. That is also what `layout.rs` Rule 3 asks for: the golden comes from the graph that
//! loads the *published* tensors, and `fixture_provenance_is_the_converted_reference` asserts the
//! recorded `provenance` / `reference` / `reference_version` metadata rather than trusting it.
//!
//! # The geometry is deliberately harder than the shipped model
//!
//! Two knobs in the generator are chosen to be *worse* cases than production:
//!
//! * `encoder_rates = (2, 5)` keeps an ODD stride, whose `padding = ceil(stride / 2)` is what
//!   makes the shipped `(2, 4, 4, 5, 5)` chain land on exactly `samples / 800`;
//! * `num_attention_heads = 2` over a 96-wide trunk puts the attention head width at 48 against a
//!   32-channel output, so `adaptive_avg_pool1d` runs with **overlapping** windows. The shipped
//!   256 → 32 is an exact 8:1 that a `reshape(.., 32, 8).mean(-1)` also reproduces; the ragged
//!   case does not, and `out.pool.*` pins all three regimes against torch directly.
//!
//! # Tolerances — three tiers, matching the decode suite's structure
//!
//! `observed` is the dev-Mac figure — the worst of the hosts this runs on. Where a tier's floor is
//! "MLX's convolution kernel" the observed value is **host-dependent** and lands far lower on a
//! host whose `conv1d` does not lower to a reduced-precision matmul; the gates are upper bounds and
//! hold in both regimes.
//!
//! | tier | gate | observed | floor |
//! |---|---|---|---|
//! | adaptive pool, latent normalization, `std == exp(logs)`, matmul-free conv | 1e-5 | ~1e-7 | f32 round-off |
//! | ONE weight-normed convolution | 2e-3 | 8.4e-4 | MLX's convolution kernel |
//! | one residual unit / one `EncoderBlock` | 1e-2 | 2.2e-3 … 5.0e-3 | the same, 2 and 7 convolutions deep |
//! | `pre_block` and its attention branch | 3e-3 | 1.2e-3 | the same, over the attention + GeGLU |
//! | the trunk, `encode`, real weights | 2e-2 | 7.9e-3 … 8.8e-3 | the same, over ~30 convolutions |
//! | the posterior's `std` | 1e-1 | 4.5e-2 | `exp` amplifying `logs`' absolute error |
//!
//! **The convolutional floor is MLX's, and that is measured rather than assumed.** A single
//! 1 → 24 channel, 7-tap convolution comes back 8.4e-4 off on this repo's dev Macs — far too much
//! for seven f32 multiply-adds — so `a_matmul_free_convolution_reproduces_the_reference_exactly`
//! recomputes exactly that convolution from the same fused weights with **no matmul at all**
//! (seven shifted, broadcast products) and lands inside 1e-5, then reads `WnConv1d`'s own taps back
//! out with a unit impulse and checks that the difference between the two paths is no larger than
//! that impulse says the host's kernel can explain. The weight-norm fusion, the layout transpose,
//! the padding and the bias are therefore exact, and what `conv1d` adds is the reduced-precision
//! f32 matmul MLX dispatches to on Metal (`tests/audio_vae_parity.rs` documents the same floor for
//! the decode half).
//!
//! **How big that kernel share is depends on the host, so no test here asserts a floor for it.**
//! The hosted macOS CI runner's `conv1d` is exact: both paths return 1.967e-7 and the convolutional
//! tiers below sit three orders under their gates. The dev Macs' do not. Every gate in this file is
//! an upper bound, which is true in both regimes; see that test's docs for the assertion this
//! replaced and why a lower bound was also the wrong direction for catching a regression.
//!
//! `the_trunk_residual_is_accumulated_round_off` then walks one conv → one unit → one
//! block → the whole trunk and shows that floor accumulating monotonically, which a structural
//! error could not do: a wrong dilation, a wrong `ceil(stride/2)` padding or a mis-ordered Snake
//! is order-1 wrong at the FIRST stage containing it.
//!
//! Every test prints its residual so the real margin stays auditable, and the mutation table
//! clears the whole-encoder gate by one to two orders of magnitude.

use crate::common;

use std::collections::BTreeSet;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::audio_vae_encoder::{
    adaptive_avg_pool_last_axis, AttnProjection, CausalAttention, DacEncoder, EncoderBlock,
    ResidualUnit, WnConv1d, ATTN_PROJ_HEADS, LAYER_NORM_EPS,
};
use mlx_gen_minimax_h3::{
    AudioDiagonalGaussian, BigVganConfig, MiniMaxH3AudioVae, MiniMaxH3AudioVaeConfig,
    MiniMaxH3AudioVaeEncoder,
};

use common::{audio_fixture_config, rel, snapshot, std_dev, AUDIO_FIXTURE};

/// The committed audio **encode** parity fixture (sc-17149), produced by
/// `tools/dump_minimax_h3_audio_vae_encode.py` running the official
/// `diffusers.AutoencoderKLMiniMaxH3Audio`.
///
/// A separate file from `AUDIO_FIXTURE` for the same reason the video halves are separate: the
/// decode golden's bytes are shared verbatim with `candle-gen-minimax-h3`, and its geometry
/// (`encoder_rates = [2, 2]`, `latent_dim = 64`) cannot express the encode half's cases.
const ENCODE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/audio_vae_encode.safetensors"
);

/// Whole-encoder and real-weight parity. Observed 8.8e-3 — see the module docs for where that
/// comes from and why it cannot be tightened.
const TOL: f32 = 2e-2;
/// `pre_block` and its attention branch — the same floor without the trunk's ~30 convolutions.
const BLOCK_TOL: f32 = 3e-3;
/// One residual unit / one downsampling stage: a handful of multi-channel convolutions.
const STAGE_TOL: f32 = 1e-2;
/// The posterior's `std = exp(logs)`. Exponentiating turns `logs`' *absolute* error into `std`'s
/// *relative* one — `d(e^l)/e^l = dl` — and `logs` peaks around 5 here, so an 8.7e-3 relative
/// error on `logs` is ~4e-2 relative on `std`. Amplification, not a second defect: the closed
/// form `std == exp(logs)` is checked separately at [`UNIT_TOL`] on the port's own tensors.
const STD_TOL: f32 = 1e-1;
/// A SINGLE weight-normed convolution, which is where MLX's per-op floor is measured.
const CONV_TOL: f32 = 2e-3;
/// Pieces that run no matmul at all: the adaptive pool, the latent normalization, and the
/// matmul-free convolution reconstruction. Three orders tighter than the convolutional tiers,
/// deliberately — a loose bound here would hide exactly what these fixtures exist to catch.
const UNIT_TOL: f32 = 1e-5;
/// Mutation probes must clear the whole-encoder gate by 10x, or "the output moved" could be
/// numerical jitter rather than a wiring difference.
const MUTATION_FLOOR: f32 = 5e-2;
/// Slack on the impulse error budget in `a_matmul_free_convolution_reproduces_the_reference_exactly`.
///
/// `tap_err · taps · max|x|` is a worst case that assumes every tap errs in the same direction at
/// the input's peak, so the observed residual sits *under* it: 0.33x on a host whose `conv1d`
/// lowers to a reduced-precision matmul, ~1.0x on one where it does not and the budget is just the
/// structural term. 4x covers both regimes and still leaves nothing unexplained through.
const BUDGET_SLACK: f32 = 4.0;

/// The tiny geometry the fixture was dumped at. Mirrors `dump_minimax_h3_audio_vae_encode.py`.
///
/// Built by hand rather than through `MiniMaxH3AudioVaeConfig::from_source_files` because this is
/// not a published configuration — deliberately, see the module docs. The `latents_mean` /
/// `latents_std` are the REAL 32-entry statistics, so `normalize` is exercised verbatim and
/// round-trips against the decode half's `denormalize`.
fn encode_fixture_config() -> MiniMaxH3AudioVaeConfig {
    let shipped = MiniMaxH3AudioVaeConfig::default();
    MiniMaxH3AudioVaeConfig {
        sample_rate: 32_000,
        output_channels: 2,
        latent_channels: 32,
        encoder_dim: 24,
        encoder_rates: vec![2, 5],
        decoder_rates: vec![5, 2],
        decoder_dim: 8,
        attn_proj: true,
        decoder_type: "bigvgan".into(),
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        // `num_mels` IS the audio VAE's `latent_dim`. 96 = encoder_dim 24 · 2^2, the
        // constructor's own derivation, so this is a configuration `from_source_files` would
        // accept. The rest of the block is the fixture's decoder, which nothing here reads.
        bigvgan: BigVganConfig {
            num_mels: 96,
            upsample_rates: vec![5, 2],
            upsample_kernel_sizes: vec![10, 4],
            upsample_initial_channel: 8,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_tanh_at_final: false,
            use_bias_at_final: false,
            snake_logscale: true,
        },
    }
}

/// The head count the fixture was dumped at — NOT the shipped [`ATTN_PROJ_HEADS`].
const FIXTURE_HEADS: i32 = 2;

fn fixture() -> Weights {
    Weights::from_file(ENCODE_FIXTURE).unwrap()
}

/// The fixture minus the reference-side extras — i.e. exactly the model weights in the published
/// naming, which is what the loader consumes for real weights.
fn model_weights() -> Weights {
    let mut w = fixture();
    for prefix in ["in.", "out.", "const.", "real."] {
        w.remove_prefix(prefix);
    }
    w
}

fn encoder() -> MiniMaxH3AudioVaeEncoder {
    let mut w = model_weights();
    MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &encode_fixture_config(),
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap()
}

fn assert_parity(got: &Array, want: &Array, tol: f32, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (peak, mean) = rel(got, want);
    println!("  {what}: peak rel {peak:.3e} (mean {mean:.3e}, tol {tol:.1e})");
    assert!(
        peak < tol,
        "{what}: peak-relative error {peak:.3e} (mean {mean:.3e}) exceeds {tol:.1e}"
    );
}

/// Materialize a (possibly transposed) array's logical order — `as_slice` returns the PHYSICAL
/// buffer, so a strided view would be read in the wrong order.
fn values(x: &Array) -> Vec<f32> {
    let n = x.size() as i32;
    x.as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// Build the encoder from the fixture with one tensor replaced, and return its `encode` mean.
fn encode_with<F: FnOnce(&mut Weights)>(mutate: F) -> Array {
    let f = fixture();
    let mut w = model_weights();
    mutate(&mut w);
    let enc = MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &encode_fixture_config(),
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap();
    enc.encode(f.require("in.encode.waveform").unwrap())
        .unwrap()
        .mode()
        .clone()
}

// ---------------------------------------------------------------------------------------------
// Provenance — layout.rs Rule 3
// ---------------------------------------------------------------------------------------------

/// The golden must have come from the CONVERTED-layout reference, and say so in its own bytes.
///
/// sc-18740 shipped a functionally wrong decoder because a fixture generated from the reference
/// modules through a pure rename agreed with a loader that shared its layout. A regeneration that
/// silently reverted to that path must fail here rather than pass everywhere else.
#[test]
fn fixture_provenance_is_the_converted_reference() {
    let f = fixture();
    assert_eq!(
        f.metadata("provenance"),
        Some("converted-checkpoint"),
        "the encode golden must be built from the published/converted layout"
    );
    assert_eq!(
        f.metadata("reference"),
        Some("diffusers.AutoencoderKLMiniMaxH3Audio")
    );
    assert_eq!(f.metadata("half"), Some("encode"));
    assert_eq!(f.metadata("story"), Some("sc-17149"));
    let version = f
        .metadata("reference_version")
        .expect("the fixture records the diffusers version it was produced with");
    assert!(
        version.starts_with(char::is_numeric),
        "reference_version {version:?} is not a version"
    );
    assert!(
        f.metadata("snapshot").is_some_and(|s| s.len() >= 8),
        "the fixture records which MiniMax-H3 snapshot the real-weight goldens came from"
    );

    // The generator measured, on the REFERENCE, how far each defect class would move the golden.
    // If any of these collapsed, the fixture would no longer be able to catch that class.
    for key in [
        "mutation_swap_posterior_heads_rel",
        "mutation_interleaved_qkv_rel",
        "mutation_geglu_half_swap_rel",
        "mutation_weight_norm_skipped_rel",
        "causal_tail_only_rel",
    ] {
        let recorded: f32 = f
            .metadata(key)
            .unwrap_or_else(|| panic!("the fixture records {key}"))
            .parse()
            .unwrap();
        println!("  {key} = {recorded:.3e}");
        assert!(
            recorded > MUTATION_FLOOR,
            "{key} is only {recorded:.3e}; the golden cannot police that defect class"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// adaptive_avg_pool1d — targeted fixture #1
// ---------------------------------------------------------------------------------------------

/// `CausalAttention` pools the surviving head width down to `latent_channels`, and the three
/// dumped cases walk every regime of PyTorch's window rule
/// `[⌊i·L/out⌋, ⌈(i+1)·L/out⌉)`: the shipped exact 256 → 32 tiling, the fixture's OVERLAPPING
/// 48 → 32, and a 44 → 32 whose windows differ in width.
///
/// A port that implemented the pool as `reshape(.., out, L/out).mean(-1)` passes the first and
/// cannot even express the other two.
#[test]
fn adaptive_pool_matches_the_reference() {
    let f = fixture();
    for tag in ["uniform", "ragged", "varsize"] {
        let params = f.require(&format!("const.pool.{tag}")).unwrap();
        let params = params.as_slice::<i32>();
        let (len, out) = (params[0], params[1]);
        let x = f.require(&format!("in.pool.{tag}")).unwrap();
        assert_eq!(*x.shape().last().unwrap(), len);
        let got = adaptive_avg_pool_last_axis(x, out).unwrap();
        assert_parity(
            &got,
            f.require(&format!("out.pool.{tag}")).unwrap(),
            UNIT_TOL,
            &format!("adaptive_avg_pool1d({len} -> {out})"),
        );
    }

    // The three cases really are different regimes, so passing all three is not one assertion in
    // disguise: a reshape-and-mean reproduces `uniform` exactly and `ragged` not at all.
    let uniform = f.require("in.pool.uniform").unwrap();
    let naive = uniform
        .reshape(&[2, 5, 32, 8])
        .unwrap()
        .mean_axes(&[3], false)
        .unwrap();
    let (peak, _) = rel(&naive, f.require("out.pool.uniform").unwrap());
    println!("  reshape-and-mean on the SHIPPED 256->32 geometry: peak rel {peak:.3e}");
    assert!(
        peak < UNIT_TOL,
        "the shipped geometry is supposed to be an exact tiling; if it is not, the ragged case is \
         no longer the discriminating one"
    );
    let ragged_windows: Vec<(i32, i32)> = (0..32)
        .map(|i| {
            (
                i * 48 / 32,
                (i + 1) * 48 / 32 + i32::from(((i + 1) * 48) % 32 != 0),
            )
        })
        .collect();
    assert!(
        ragged_windows.windows(2).any(|w| w[0].1 > w[1].0),
        "the fixture's own pooling windows do not overlap"
    );
}

// ---------------------------------------------------------------------------------------------
// The trunk, pre_block, and the whole encode
// ---------------------------------------------------------------------------------------------

/// The DAC convolutional trunk alone: `block.0`, five (here two) `EncoderBlock`s of three dilated
/// residual units plus a strided channel-doubling convolution, then `Snake1d` and `block.N`.
///
/// Held separately from `encode` because an end-to-end golden cannot say which of the trunk and
/// `pre_block` is wrong.
#[test]
fn encoder_trunk_has_the_reference_geometry() {
    let f = fixture();
    let mut w = model_weights();
    let cfg = encode_fixture_config();
    let trunk = DacEncoder::from_weights(&mut w, "encoder", &cfg, Dtype::Float32).unwrap();
    let got = trunk
        .forward(
            &f.require("in.encode.waveform")
                .unwrap()
                .transpose_axes(&[0, 2, 1])
                .unwrap(),
        )
        .unwrap();
    // NLC: 2 batch items, 320 / hop 10 = 32 frames, latent_dim 96. The reference's own tensor is
    // NCL, and `the_trunk_residual_is_accumulated_round_off` is what checks the values.
    assert_eq!(got.shape(), &[2, 32, 96]);
    assert_eq!(
        f.require("out.trunk.hidden").unwrap().shape(),
        &[2, 96, 32],
        "the reference dumps NCL; a port that compared without transposing would be checking          a transposed tensor against itself only because 32 == 32 here"
    );
}

/// Where the trunk's residual comes from: one convolution, then one residual unit, then one
/// downsampling stage, then the whole stack.
///
/// MLX evaluates f32 matmul in **reduced precision on Metal**, so a convolutional stack
/// accumulates a floor that has nothing to do with the port being right. A single loose
/// end-to-end bound cannot distinguish that floor from a defect; this walk can, because a
/// structural error — a wrong dilation, a wrong `ceil(stride/2)` padding, a mis-ordered Snake —
/// is order-1 wrong at the FIRST stage that contains it, not 1e-6 wrong there and 1e-2 wrong
/// twenty layers later.
#[test]
fn the_trunk_residual_is_accumulated_round_off() {
    let f = fixture();
    let cfg = encode_fixture_config();
    let nlc = |key: &str| f.require(key).unwrap().transpose_axes(&[0, 2, 1]).unwrap();
    let x = nlc("in.encode.waveform");

    // ONE weight-normed convolution: 1 -> 24 channels, k7, 'same' padding. Seven multiply-adds
    // per output — and on a dev Mac it is already 8e-4 off, which is the whole point: see
    // `a_matmul_free_convolution_reproduces_the_reference_exactly` for the proof that the
    // weights, the layout transpose, the padding and the bias are all exact and this residual is
    // MLX's convolution kernel. How much of it there is depends on the host; the gate is an upper
    // bound, and a host with an exact `conv1d` simply lands three orders under it.
    let mut w = model_weights();
    let conv_in = WnConv1d::from_weights(&mut w, "encoder.block.0", 1, 3, 1, Dtype::Float32)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_parity(
        &conv_in,
        &nlc("out.stage.conv_in"),
        CONV_TOL,
        "encoder.block.0 (one weight-normed conv)",
    );

    // One residual unit: two convolutions and two Snakes, at 24 channels.
    let mut w = model_weights();
    let unit =
        ResidualUnit::from_weights(&mut w, "encoder.block.1.block.0", 1, Dtype::Float32).unwrap();
    assert_parity(
        &unit.forward(&conv_in).unwrap(),
        &nlc("out.stage.unit0"),
        STAGE_TOL,
        "one ResidualUnit (dilation 1)",
    );

    // One EncoderBlock: three residual units at dilations 1/3/9, a Snake, and the stride-2
    // channel-doubling convolution.
    let mut w = model_weights();
    let block = EncoderBlock::from_weights(&mut w, "encoder.block.1", 2, Dtype::Float32).unwrap();
    let staged = block.forward(&conv_in).unwrap();
    assert_eq!(staged.shape(), &[2, 160, 48], "stride 2 halves 320 -> 160");
    assert_parity(
        &staged,
        &nlc("out.stage.block1"),
        STAGE_TOL,
        "one EncoderBlock (stride 2)",
    );

    // ...and the whole trunk, whose residual is the same floor over ~30 convolutions.
    let mut w = model_weights();
    let trunk = DacEncoder::from_weights(&mut w, "encoder", &cfg, Dtype::Float32).unwrap();
    assert_parity(
        &trunk.forward(&x).unwrap(),
        &nlc("out.trunk.hidden"),
        TOL,
        "the whole DAC trunk",
    );
}

/// The attention branch on its own — `attn(norm1(x))`, before the residual sum can hide it.
///
/// This is the piece that carries the causal mask, the head **mean-pool** and the adaptive pool,
/// and the reference dumps it separately for exactly that reason. The LayerNorm is applied here
/// rather than inside [`AttnProjection`] so the branch is genuinely isolated.
#[test]
fn attention_branch_matches_the_reference() {
    let f = fixture();
    let cfg = encode_fixture_config();
    let x = f.require("in.pre_block.x").unwrap();

    let mut w = model_weights();
    let attn = CausalAttention::from_weights(
        &mut w,
        "pre_block.attn",
        cfg.bigvgan.num_mels,
        cfg.latent_channels,
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap();
    let normed = mlx_rs::fast::layer_norm(
        x,
        Some(f.require("pre_block.norm1.weight").unwrap()),
        Some(f.require("pre_block.norm1.bias").unwrap()),
        LAYER_NORM_EPS,
    )
    .unwrap();
    assert_parity(
        &attn.forward(&normed).unwrap(),
        f.require("out.pre_block.attn").unwrap(),
        BLOCK_TOL,
        "CausalAttention branch",
    );

    // The composed block: `proj(norm3(x)) + attn(norm1(x))`, then `+ mlp(norm2(·))`.
    let mut w = model_weights();
    let pre = AttnProjection::from_weights(
        &mut w,
        "pre_block",
        cfg.bigvgan.num_mels,
        cfg.latent_channels,
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap();
    assert_parity(
        &pre.forward(x).unwrap(),
        f.require("out.pre_block.y").unwrap(),
        BLOCK_TOL,
        "AttnProjection (pre_block)",
    );

    // The attention branch must be a genuinely different tensor from the block output, or the
    // golden above would not be isolating anything.
    let (branch_gap, _) = rel(
        f.require("out.pre_block.attn").unwrap(),
        f.require("out.pre_block.y").unwrap(),
    );
    println!("  attention branch vs block output: peak rel {branch_gap:.3e}");
    assert!(
        branch_gap > MUTATION_FLOOR,
        "the attention branch and the block output are near-identical ({branch_gap:.3e}); the \
         residual assembly is not being exercised"
    );
}

/// `AutoencoderKLMiniMaxH3Audio.encode`: trunk → `pre_block` → `mean_proj` / `logs_proj`.
///
/// Both heads are checked, and `std = exp(logs)` with them: the pipeline only ever consumes
/// `mode()`, so a wrong `logs_proj` would otherwise never be observed by anything.
#[test]
fn encode_matches_the_reference() {
    let f = fixture();
    let enc = encoder();
    let posterior = enc
        .encode(f.require("in.encode.waveform").unwrap())
        .unwrap();
    assert_eq!(
        posterior.mean().shape(),
        &[2, 32, 32],
        "320 samples at hop 10 -> 32 latent frames, 32 channels, 2 batch items"
    );
    assert_parity(
        posterior.mode(),
        f.require("out.encode.mean").unwrap(),
        TOL,
        "encode -> mean_proj",
    );
    assert_parity(
        posterior.logs(),
        f.require("out.encode.logs").unwrap(),
        TOL,
        "encode -> logs_proj",
    );
    assert_parity(
        posterior.std(),
        f.require("out.encode.std").unwrap(),
        STD_TOL,
        "posterior std = exp(logs)",
    );
    // The closed form itself, on the port's own tensors, where no amplification applies.
    assert_parity(
        posterior.std(),
        &mlx_rs::ops::exp(posterior.logs()).unwrap(),
        UNIT_TOL,
        "std == exp(logs)",
    );

    // `mode()` is bit-for-bit `mean_proj`'s output, as the reference documents.
    assert_eq!(values(posterior.mode()), values(posterior.mean()));
}

/// `encode` zero-pads on the RIGHT to a whole number of hops, so a clip that is not a multiple of
/// 800 samples still produces `ceil(S / 800)` frames — and the same ones the padded clip does.
#[test]
fn encode_right_pads_to_a_whole_hop() {
    let f = fixture();
    let enc = encoder();
    assert_eq!(enc.hop_length(), 10, "hop = 2 * 5");

    let ragged = f.require("in.encode_pad.waveform").unwrap();
    assert_eq!(
        ragged.shape(),
        &[2, 1, 311],
        "311 = 320 - 10 + 1, not a whole hop"
    );
    let got = enc.encode(ragged).unwrap();
    assert_eq!(
        got.mean().shape(),
        &[2, 32, 32],
        "ceil(311 / 10) = 32 frames"
    );
    assert_parity(
        got.mode(),
        f.require("out.encode_pad.mean").unwrap(),
        TOL,
        "encode with a right-pad",
    );

    // The pad is not a no-op: a ragged clip does NOT encode to the same latents as the full one,
    // so a port that silently truncated to a whole hop would be visible here.
    let full = enc
        .encode(f.require("in.encode.waveform").unwrap())
        .unwrap();
    let (peak, _) = rel(got.mode(), full.mode());
    println!("  ragged vs full clip: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "the padded and unpadded clips encode identically ({peak:.3e}); the pad is untested"
    );
}

/// Every model tensor must be consumed, and the declared name list must be exactly the fixture's.
/// A silently unmapped tensor still produces plausible-looking latents.
#[test]
fn weight_mapping_is_exhaustive() {
    let cfg = encode_fixture_config();
    let mut w = model_weights();
    let before: BTreeSet<String> = w.keys().map(str::to_string).collect();
    // 3 + 2 * 28 + 4 + 22 + 4 for a two-stage encoder.
    assert_eq!(before.len(), 89, "the fixture carries the encode half only");

    let _enc = MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &cfg,
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap();
    w.remove_accessed();
    let leftover: Vec<&str> = w.keys().collect();
    assert!(
        leftover.is_empty(),
        "these checkpoint tensors were never read: {leftover:?}"
    );

    let declared: BTreeSet<String> = MiniMaxH3AudioVaeEncoder::tensor_names(&cfg)
        .into_iter()
        .collect();
    assert_eq!(
        declared, before,
        "declared tensor names differ from the fixture's"
    );
}

/// A checkpoint or configuration that disagrees with the loader is a loud error, not a silent
/// re-interpretation.
#[test]
fn a_mismatched_configuration_is_rejected() {
    let cfg = encode_fixture_config();

    // `attn_proj = false` has no `pre_block`, so the trunk cannot reach the posterior heads.
    let mut without = cfg.clone();
    without.attn_proj = false;
    let mut w = model_weights();
    let err = MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &without,
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("attn_proj"), "unexpected error: {err}");

    // `latent_dim % latent_channels != 0` is the branch where the reference widens
    // `attn_proj_dim` to the next power of two and diffusers refuses outright.
    let mut indivisible = cfg.clone();
    indivisible.latent_channels = 7;
    let mut w = model_weights();
    assert!(MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &indivisible,
        FIXTURE_HEADS,
        Dtype::Float32
    )
    .is_err());

    // A GeGLU MLP at a different `mlp_ratio` is rejected: MLP_RATIO is enforced, not declared.
    let mut w = model_weights();
    let w0 = w.require("pre_block.mlp.w0.weight").unwrap().clone();
    let hidden = w0.shape()[0];
    assert_eq!(
        hidden,
        cfg.latent_channels * 2,
        "the published GeGLU hidden width is latent_channels * MLP_RATIO"
    );
    w.insert(
        "pre_block.mlp.w0.weight",
        mlx_gen_minimax_h3::tensor::slice_axis(&w0, 0, 0, hidden - 1).unwrap(),
    );
    let err = MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &cfg,
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("mlp_ratio"), "unexpected error: {err}");

    // A latent-statistics list of the wrong length is rejected rather than broadcast.
    let mut short = cfg.clone();
    short.latents_mean.truncate(8);
    let mut w = model_weights();
    assert!(MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &short,
        FIXTURE_HEADS,
        Dtype::Float32
    )
    .is_err());
}

/// The model is MONO. A stereo clip is two BATCH items, and a `[1, 2, samples]` packing — the
/// plausible transposition — is rejected rather than encoded as something the model never saw.
#[test]
fn mis_packed_waveforms_are_rejected() {
    let f = fixture();
    let enc = encoder();
    let wave = f.require("in.encode.waveform").unwrap();
    assert!(enc.encode(wave).is_ok());

    // Asserted by MESSAGE, not by `is_err()` (sc-19488). `encode`'s packing guard is one arm
    // (`s.len() != 3 || s[1] != 1`), and with it deleted every input below still errors — the
    // rank-2 and rank-4 cases fault on the `transpose_axes(&[0, 2, 1])` immediately after, and the
    // `[1, 2, samples]` case reaches the encoder stack with a channel axis it was never built for.
    // Naming the guard is what separates "the encoder is mono" from "something downstream tripped".
    let expect_packing_guard = |z: &Array, what: &str| {
        let msg = enc.encode(z).expect_err(what).to_string();
        assert!(
            msg.contains("expected [B, 1, samples]"),
            "{what}: the mono packing guard must be what rejects this, not a downstream \
             transpose/conv fault: {msg}"
        );
    };
    let stereo_packed = wave.reshape(&[1, 2, 320]).unwrap();
    expect_packing_guard(
        &stereo_packed,
        "[1, 2, samples] must be rejected: the encoder is mono",
    );
    // Rank 2 and rank 4 are errors too.
    expect_packing_guard(&wave.reshape(&[2, 320]).unwrap(), "rank 2 is not [B, 1, S]");
    expect_packing_guard(
        &wave.reshape(&[2, 1, 320, 1]).unwrap(),
        "rank 4 is not [B, 1, S]",
    );
}

// ---------------------------------------------------------------------------------------------
// Latent normalization — the decode half's exact inverse
// ---------------------------------------------------------------------------------------------

/// `normalize` is `(z − mean) / std` per channel, and composing it with the decode half's
/// `denormalize` is the identity.
///
/// Cross-checked against the *other* module's implementation rather than against a restatement of
/// the same formula: the two are supposed to be inverses, and only running both can show it.
#[test]
fn normalization_is_the_decoders_exact_inverse() {
    let f = fixture();
    let enc = encoder();
    let z = enc
        .encode(f.require("in.encode.waveform").unwrap())
        .unwrap()
        .mode()
        .clone();

    let normalized = enc.normalize(&z).unwrap();
    let (moved, _) = rel(&normalized, &z);
    assert!(moved > 1e-2, "normalization was a no-op ({moved:.3e})");

    // The closed form, element by element. `[B, 32, T]`: element (0, c, t) is at index c*T + t.
    let raw = values(&z);
    let got = values(&normalized);
    let cfg = MiniMaxH3AudioVaeConfig::default();
    let t = z.shape()[2] as usize;
    for c in [0usize, 5, 31] {
        let idx = c * t + 3;
        let want = (raw[idx] - cfg.latents_mean[c]) / cfg.latents_std[c];
        assert!(
            (got[idx] - want).abs() < 1e-4,
            "channel {c}: expected (z - mean)/std = {want}, got {}",
            got[idx]
        );
    }

    // Round trip through the DECODE half's `denormalize`, which is the actual contract.
    let mut dw = Weights::from_file(AUDIO_FIXTURE).unwrap();
    for prefix in ["in.", "out.", "const.", "amp."] {
        dw.remove_prefix(prefix);
    }
    let vae =
        MiniMaxH3AudioVae::from_weights(&mut dw, &audio_fixture_config(), Dtype::Float32).unwrap();
    let round_trip = vae.denormalize(&normalized).unwrap();
    assert_parity(&round_trip, &z, UNIT_TOL, "normalize -> denormalize");

    // Rank-4 stereo packing uses the same second-to-last channel axis.
    let stereo = z.reshape(&[1, 2, 32, 32]).unwrap();
    let both = enc.normalize(&stereo).unwrap();
    assert_eq!(both.shape(), stereo.shape());
    assert_parity(
        &vae.denormalize(&both).unwrap(),
        &stereo,
        UNIT_TOL,
        "normalize -> denormalize (stereo packing)",
    );

    // A latent whose channel count disagrees with the config is an error, not a broadcast. Both
    // clauses assert the MESSAGE (sc-19488), and they target DIFFERENT arms of `normalize`: the
    // first is the channel-count check, the second the rank check. `is_err()` alone could not tell
    // them apart, and a 16-channel latent would broadcast silently against a 32-entry mean/std
    // vector on some shapes rather than erroring at all.
    let msg = enc
        .normalize(&z.reshape(&[2, 16, 64]).unwrap())
        .expect_err("16 channels disagrees with the config's 32")
        .to_string();
    assert!(
        // Matched on the guard's FULL prefix, not the bare "config declares" tail: the audio VAE's
        // own de-normalize guard (`src/audio_vae.rs`) emits that same tail, so the short form would
        // go inert the moment this entry point delegated there (sc-19488).
        msg.contains("minimax-h3 audio encoder: latent has"),
        "the channel-count guard must be what rejects this, not a broadcast fault: {msg}"
    );
    let msg = enc
        .normalize(&Array::from_slice(&[1.0f32], &[1]))
        .expect_err("a rank-1 latent has no channel axis")
        .to_string();
    assert!(
        msg.contains("cannot normalize a rank-1 latent"),
        "the rank guard must be what rejects this: {msg}"
    );
}

/// The audio posterior's second parameter is a log **standard deviation** with **no clamp** — not
/// the video encoder's clamped log variance.
///
/// The two are shape-identical and `mode()` is unaffected by the difference, so nothing the
/// MiniMax-H3 pipeline consumes would ever reveal a port that reused
/// `vae_encoder::DiagonalGaussian` here. This is the assertion that does.
#[test]
fn log_std_is_not_log_variance() {
    let logs = Array::from_slice(&[0.5f32, 3.0, -40.0], &[1, 3, 1]);
    let mean = Array::from_slice(&[0.0f32, 0.0, 0.0], &[1, 3, 1]);
    let p = AudioDiagonalGaussian::new(mean, logs.clone()).unwrap();
    let std = p.std().as_slice::<f32>();

    for (i, l) in [0.5f32, 3.0, -40.0].iter().enumerate() {
        let audio = l.exp();
        let video = (0.5 * l.clamp(-30.0, 20.0)).exp();
        assert!(
            (std[i] - audio).abs() < 1e-5 * audio.max(1e-6),
            "std[{i}] = {} is not exp({l})",
            std[i]
        );
        // ...and the video convention would have given something measurably different.
        let gap = (audio - video).abs() / video.max(1e-12);
        println!(
            "  logs {l}: exp(logs) {audio:.6e} vs exp(0.5*clamp(logs)) {video:.6e} ({gap:.3e})"
        );
        assert!(
            gap > MUTATION_FLOOR,
            "the two posterior conventions agree at logs = {l}; the distinction is untestable \
             there"
        );
    }
    // The last case additionally proves the absence of the clamp: -40 is outside [-30, 20].
    assert!(
        std[2] > 0.0 && std[2] < 1e-15,
        "no clamp is applied to logs"
    );
}

// ---------------------------------------------------------------------------------------------
// False-green guards and mutation controls
// ---------------------------------------------------------------------------------------------

/// A constant, all-zero or degenerate golden is a false green.
#[test]
fn fixture_is_not_degenerate() {
    let f = fixture();
    for key in [
        "out.encode.mean",
        "out.encode.logs",
        "out.trunk.hidden",
        "out.pre_block.y",
        "out.pre_block.attn",
        "out.pool.ragged",
    ] {
        let t = f.require(key).unwrap();
        assert!(
            std_dev(t) > 1e-4,
            "{key} is ~constant; a constant golden is a false green"
        );
    }

    // The two posterior heads must be distinguishable, or swapping them would be invisible.
    let (gap, _) = rel(
        f.require("out.encode.mean").unwrap(),
        f.require("out.encode.logs").unwrap(),
    );
    println!("  mean vs logs: peak rel {gap:.3e}");
    assert!(gap > MUTATION_FLOOR, "mean and logs are indistinguishable");

    // The two batch items are genuinely different waveforms, so a port that collapsed the batch
    // (or broadcast one channel) could not pass.
    let wave = f.require("in.encode.waveform").unwrap();
    let (channel_gap, _) = rel(
        &wave.take_axis(Array::from_slice(&[0], &[1]), 0).unwrap(),
        &wave.take_axis(Array::from_slice(&[1], &[1]), 0).unwrap(),
    );
    assert!(
        channel_gap > MUTATION_FLOOR,
        "the two batch items are near-identical ({channel_gap:.3e})"
    );

    // `zero_k_bias` really is zero in this fixture, as it is in the checkpoint.
    let zk = f.require("pre_block.attn.zero_k_bias").unwrap();
    assert_eq!(
        zk.abs().unwrap().max(None).unwrap().item::<f32>(),
        0.0,
        "zero_k_bias is a zero buffer"
    );
    // ...but `q_bias` and `v_bias` are NOT, or the fused-bias assembly would be unobservable.
    for key in ["pre_block.attn.q_bias", "pre_block.attn.v_bias"] {
        assert!(
            std_dev(f.require(key).unwrap()) > 1e-3,
            "{key} is ~constant"
        );
    }
}

/// Mutation control: every distinct tensor ROLE in the encode path must move the output.
///
/// One representative per role, including both halves of a weight-norm pair (a port that read
/// only `weight_v` would encode plausibly), the `Snake1d` alphas, the three separately-stored
/// attention biases, and both ends of the residual stack.
#[test]
fn every_weight_is_load_bearing() {
    let baseline = encode_with(|_| {});

    let probes: [&str; 16] = [
        "encoder.block.0.weight_g",
        "encoder.block.0.weight_v",
        "encoder.block.0.bias",
        "encoder.block.1.block.0.block.0.alpha",
        "encoder.block.1.block.0.block.1.weight_v",
        "encoder.block.1.block.2.block.3.weight_g",
        "encoder.block.1.block.3.alpha",
        "encoder.block.1.block.4.weight_v",
        "encoder.block.2.block.1.block.1.bias",
        "encoder.block.3.alpha",
        "encoder.block.4.weight_v",
        "pre_block.norm1.weight",
        "pre_block.attn.qkv.weight",
        "pre_block.proj.weight",
        "pre_block.mlp.w2.bias",
        "mean_proj.weight",
    ];

    for key in probes {
        let mutated = encode_with(|w| {
            let original = w.require(key).unwrap().clone();
            // Scale AND shift: the shift makes an all-zero tensor observable, the scale keeps the
            // perturbation proportionate for tensors that are already large.
            // Scaled AND shifted, with the shift sized against the tensor's own magnitude:
            // `weight_v` is only ever used through `v / ||v||`, so a pure SCALE of it is a
            // mathematical no-op and a fixed +0.2 barely turns a tensor whose entries are ~3.
            let scale: f32 = original.abs().unwrap().mean(None).unwrap().item::<f32>();
            let bumped = mlx_rs::ops::add(
                mlx_rs::ops::multiply(&original, Array::from_f32(1.3)).unwrap(),
                Array::from_f32(0.2 + 0.5 * scale),
            )
            .unwrap();
            w.insert(key, bumped);
        });
        let (peak, _) = rel(&mutated, &baseline);
        println!("  {key}: peak rel {peak:.3e} (floor {MUTATION_FLOOR:.1e})");
        assert!(
            peak > MUTATION_FLOOR,
            "perturbing {key} moved the encode by only {peak:.3e}, under the \
             {MUTATION_FLOOR:.1e} floor — it is not wired into the graph"
        );
    }
}

/// **The key bias is mathematically inert, which is why it ships as a frozen zero buffer** — and
/// why `every_weight_is_load_bearing` cannot include it.
///
/// A bias added to the KEYS shifts every logit for a given query by the same `q·c`, and softmax
/// is shift-invariant, so any `zero_k_bias` — zero or not — leaves the attention unchanged up to
/// round-off. Two consequences, both asserted here rather than left as a comment:
///
/// * the tensor cannot be shown to be consumed by perturbing it, so it is shown by DELETING it:
///   the loader must fail rather than synthesize zeros;
/// * a checkpoint that shipped a non-zero key bias would still be honoured — the port
///   concatenates whatever is stored, and the measured inertness below is a property of softmax,
///   not of the loader ignoring the tensor.
#[test]
fn the_key_bias_is_read_and_is_inert_by_construction() {
    // Deleting it is a load error, so the tensor is genuinely required.
    let mut w = model_weights();
    w.remove("pre_block.attn.zero_k_bias");
    let err = MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &mut w,
        &encode_fixture_config(),
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("zero_k_bias"),
        "the loader must require the key bias; got: {err}"
    );

    // ...and replacing it with a non-zero vector changes nothing beyond round-off, because
    // softmax is invariant to a per-query constant.
    let baseline = encode_with(|_| {});
    let mutated = encode_with(|w| {
        let shape = w
            .require("pre_block.attn.zero_k_bias")
            .unwrap()
            .shape()
            .to_vec();
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin() * 0.5).collect();
        w.insert(
            "pre_block.attn.zero_k_bias",
            Array::from_slice(&vals, &shape),
        );
    });
    let (peak, _) = rel(&mutated, &baseline);
    println!("  non-zero key bias moves the encode by {peak:.3e} (softmax shift-invariance)");
    assert!(
        peak < CONV_TOL,
        "a key bias moved the encode by {peak:.3e}; softmax should be shift-invariant, so either          the bias is not being concatenated into the KEY third or the thirds are mis-ordered"
    );

    // The same probe applied to the QUERY bias is loud, so the test above is not vacuous — it is
    // the key third specifically that is inert.
    let queried = encode_with(|w| {
        let shape = w.require("pre_block.attn.q_bias").unwrap().shape().to_vec();
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin() * 0.5).collect();
        w.insert("pre_block.attn.q_bias", Array::from_slice(&vals, &shape));
    });
    let (q_peak, _) = rel(&queried, &baseline);
    println!(
        "  the same perturbation on q_bias moves it by {q_peak:.3e} ({:.0}x the key bias)",
        q_peak / peak.max(1e-12)
    );
    // Compared against the KEY bias rather than an absolute floor: the point is the contrast.
    // `q_bias` shifts every logit by `c·k_j`, which varies with the key and therefore survives
    // softmax, but its effect is then diluted by the head mean-pool, the adaptive pool and the
    // residual sum with `proj(norm3(x))` — so it is loud relative to the key bias, not loud in
    // absolute terms. A port that assembled the three biases in the wrong ORDER would make the
    // query third inert and the key third loud, which is exactly what this ratio catches.
    assert!(
        q_peak > peak * 50.0,
        "perturbing q_bias ({q_peak:.3e}) is no louder than perturbing the inert key bias          ({peak:.3e}); the three biases are not landing on their own thirds"
    );
}

/// The single-convolution residual is **MLX's convolution kernel**, not the port's weights — and
/// that conclusion is reached in a way that holds on any host.
///
/// `encoder.block.0` is a 1 -> 24 channel, 7-tap convolution: seven multiply-adds per output, which
/// f32 should evaluate to ~1e-7. On this repo's dev Macs it comes back ~8e-4 off, and this test is
/// what says whose 8e-4 it is. It makes three measurements and then does the arithmetic:
///
/// 1. **The reference's semantics, with no matmul at all.** The same convolution recomputed from
///    the same `weight_g` / `weight_v` / `bias` as seven shifted, broadcast products — landing
///    within [`UNIT_TOL`], which pins the weight-norm fusion (`g · v / ‖v‖`, norm over axes 1..),
///    the `[out, in, k] -> [out, k, in]` layout transpose, the 'same' padding and the bias add.
///    Anything wrong there is a *structural* error and would be order-1, not 1e-7.
/// 2. **The port's effective kernel, by unit impulse.** A single 1.0 in the input makes every
///    output exactly one multiply-add, so `WnConv1d`'s own taps read straight back out. They must
///    match (1)'s fusion within [`CONV_TOL`], and the residual they leave is the kernel's
///    per-product precision with no accumulation in it.
/// 3. **The port's convolution over the real input**, against the reference, within [`CONV_TOL`].
///
/// Then (3) must fit inside the budget (2) allows plus what (1) costs. It does, so nothing in the
/// residual is unaccounted for and none of it can be a port defect. The trunk's 7.9e-3 is that same
/// floor accumulating over ~30 convolutions.
///
/// # What this test deliberately does not assert
///
/// **How large the kernel's share is, is a property of the host.** An earlier revision asserted
/// that MLX's `conv1d` must be at least 100x worse than the matmul-free path. That is true where
/// `conv1d` lowers to the reduced-precision f32 Metal matmul `tests/audio_vae_parity.rs` documents
/// for the decode half (~8.4e-4 here, ~4000x), and false on the hosted CI runner, where `conv1d`
/// is exact and both paths return an identical 1.967e-7. The premise was machine-dependent, so the
/// test failed on half the fleet — and it was the wrong direction for catching regressions anyway:
/// a broken port makes `conv1d`'s residual *larger*, which satisfies a lower bound more easily.
/// Every gate here is an upper bound instead, and the budget in (4) degrades gracefully to
/// `struct_abs` on a host with an exact kernel rather than asserting a floor that host has no
/// reason to have.
#[test]
fn a_matmul_free_convolution_reproduces_the_reference_exactly() {
    let f = fixture();
    let x = f
        .require("in.encode.waveform")
        .unwrap()
        .transpose_axes(&[0, 2, 1])
        .unwrap();
    let want = f
        .require("out.stage.conv_in")
        .unwrap()
        .transpose_axes(&[0, 2, 1])
        .unwrap();

    // Re-fuse the weight-norm pair exactly as the loader does: `g · v / ‖v‖`, `[24, 1, 7]`.
    let g = f.require("encoder.block.0.weight_g").unwrap();
    let v = f.require("encoder.block.0.weight_v").unwrap();
    let norm = v
        .square()
        .unwrap()
        .sum_axes(&[1, 2], true)
        .unwrap()
        .sqrt()
        .unwrap();
    let fused = mlx_rs::ops::multiply(g, mlx_rs::ops::divide(v, &norm).unwrap()).unwrap();
    let taps = fused.shape()[2];
    assert_eq!(fused.shape(), &[24, 1, 7]);
    // A near-constant kernel would make the impulse readout below agree with anything, and a
    // near-zero one would make every peak-relative gate here divide by noise.
    assert!(
        std_dev(&fused) > 1e-2,
        "the fused kernel is nearly constant ({:.3e}); the comparisons below would be vacuous",
        std_dev(&fused)
    );

    // Zero-pad 3 either side ('same' for k7), then accumulate seven shifted, broadcast products.
    let s = x.shape().to_vec();
    let pad = Array::zeros::<f32>(&[s[0], taps / 2, s[2]]).unwrap();
    let padded = mlx_rs::ops::concatenate_axis(&[&pad, &x, &pad], 1).unwrap();

    let mut acc: Option<Array> = None;
    for k in 0..taps {
        // `[B, S, 1]` window times `[24]` taps broadcasts to `[B, S, 24]` — no matmul anywhere.
        let window = mlx_gen_minimax_h3::tensor::slice_axis(&padded, 1, k, k + s[1]).unwrap();
        let tap = mlx_gen_minimax_h3::tensor::slice_axis(&fused, 2, k, k + 1)
            .unwrap()
            .reshape(&[24])
            .unwrap();
        let term = mlx_rs::ops::multiply(&window, &tap).unwrap();
        acc = Some(match acc {
            Some(prev) => mlx_rs::ops::add(&prev, &term).unwrap(),
            None => term,
        });
    }
    let bias = f.require("encoder.block.0.bias").unwrap();
    let got = mlx_rs::ops::add(acc.unwrap(), bias).unwrap();

    assert_parity(&got, &want, UNIT_TOL, "matmul-free encoder.block.0");

    // 2. The PORT's effective kernel, read out by a unit impulse.
    //
    // `y[0, t, o] = bias[o] + Σ_k fused[o, k] · imp[t + k - pad]`, and `imp` is 1.0 at `p` and 0
    // everywhere else, so `y[0, p + pad - k, o] - bias[o]` IS `fused[o, k]` — one multiply-add per
    // output, no accumulation, no cancellation. That reads the fusion, the `[out, in, k] ->
    // [out, k, in]` transpose, the padding offset and the bias straight back out of the loaded
    // module, and the residual it leaves is the kernel's per-product precision with nothing else
    // mixed into it.
    let pad = taps / 2;
    let mut w = model_weights();
    let conv =
        WnConv1d::from_weights(&mut w, "encoder.block.0", 1, pad, 1, Dtype::Float32).unwrap();

    let span = 2 * taps;
    let p = span / 2;
    let mut imp = vec![0.0f32; span as usize];
    imp[p as usize] = 1.0;
    let response = conv
        .forward(&Array::from_slice(&imp, &[1, span, 1]))
        .unwrap();

    let (rv, bv, fv) = (values(&response), values(bias), values(&fused));
    let out = fused.shape()[0] as usize;
    let k_len = taps as usize;
    let mut recovered = vec![0.0f32; out * k_len];
    let mut tap_err = 0.0f32;
    for o in 0..out {
        for k in 0..k_len {
            let t = (p + pad) as usize - k;
            let v = rv[t * out + o] - bv[o];
            recovered[o * k_len + k] = v;
            tap_err = tap_err.max((v - fv[o * k_len + k]).abs());
        }
    }
    let recovered = Array::from_slice(&recovered, &[out as i32, 1, taps]);
    assert_parity(
        &recovered,
        &fused,
        CONV_TOL,
        "the port's impulse-recovered kernel",
    );

    // 3. ...and the port's convolution over the real input still matches the reference.
    let via_conv = conv.forward(&x).unwrap();
    assert_parity(&via_conv, &want, CONV_TOL, "WnConv1d encoder.block.0");

    // 4. The accounting: every part of that residual is already paid for.
    //
    // Deliberately NOT `kernel_gap > exact_gap * 100`. That held on a host whose `conv1d` lowers to
    // a reduced-precision matmul and failed on one whose `conv1d` is exact — and it is the wrong
    // direction for regression detection besides, since breaking the port makes `kernel_gap` LARGER
    // and a lower bound EASIER to satisfy. The portable claim is an upper bound: the impulse says
    // this host's kernel misrepresents a single tap by at most `tap_err`, so a `taps`-long
    // convolution over `x` can stray by at most `tap_err · taps · max|x|`, on top of what the
    // structure itself costs (`struct_abs`, measured at step 1). Nothing is left over to be a port
    // defect. On a host with an exact `conv1d` the first term vanishes and the bound collapses to
    // `struct_abs` — still true, and still the same conclusion.
    let peak = |a: &Array| -> f32 { a.abs().unwrap().max(None).unwrap().item() };
    let diff = |a: &Array, b: &Array| peak(&mlx_rs::ops::subtract(a, b).unwrap());
    let (x_max, obs_abs, struct_abs) = (peak(&x), diff(&via_conv, &want), diff(&got, &want));
    let budget = tap_err * taps as f32 * x_max + struct_abs;
    println!(
        "  conv1d vs reference {:.3e}; the SAME arithmetic without a matmul {:.3e}",
        obs_abs / peak(&want),
        struct_abs / peak(&want)
    );
    println!(
        "  per-tap kernel error {tap_err:.3e} -> budget {budget:.3e} vs observed {obs_abs:.3e} ({:.2}x)",
        obs_abs / budget.max(f32::MIN_POSITIVE)
    );
    assert!(
        obs_abs <= budget * BUDGET_SLACK,
        "WnConv1d strays {obs_abs:.3e} from the reference, but this host's kernel precision          ({tap_err:.3e} per tap) and the port's own structural residual ({struct_abs:.3e}) only          account for {budget:.3e} — the excess is NOT kernel round-off"
    );
}

/// The three structural transforms this port could plausibly get wrong, each shape-identical to
/// the right one.
///
/// The generator measured the same three on the REFERENCE and recorded them in the fixture's
/// metadata (`fixture_provenance_is_the_converted_reference` asserts those are large); this is the
/// in-repo half of that contract, applied to the loader.
#[test]
fn structural_transforms_are_load_bearing() {
    let baseline = encode_with(|_| {});

    // 1. Fused QKV read per-head interleaved instead of as contiguous thirds — layout.rs Rule 2's
    //    hazard, in the shape it would take here. It does NOT apply to the audio VAE, and
    //    applying it anyway is silently wrong.
    let mutated = encode_with(|w| {
        let qkv = w.require("pre_block.attn.qkv.weight").unwrap().clone();
        let d = qkv.shape()[1];
        let regrouped = qkv
            .reshape(&[3, FIXTURE_HEADS, d / FIXTURE_HEADS, d])
            .unwrap()
            .transpose_axes(&[1, 0, 2, 3])
            .unwrap()
            .reshape(&[3 * d, d])
            .unwrap();
        w.insert("pre_block.attn.qkv.weight", regrouped);
    });
    let (peak, _) = rel(&mutated, &baseline);
    println!("  interleaved QKV split: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "a per-head-interleaved QKV read does not change the encode ({peak:.3e}); the contiguous \
         thirds convention is not being exercised"
    );

    // 2. GeGLU halves swapped — `w0` is the GATE, `w1` the VALUE. Two separate tensors, so
    //    `layout::split_gate_value` does not apply, but the sc-18740 signature is identical.
    let mutated = encode_with(|w| {
        for leaf in ["weight", "bias"] {
            let a = w
                .require(&format!("pre_block.mlp.w0.{leaf}"))
                .unwrap()
                .clone();
            let b = w
                .require(&format!("pre_block.mlp.w1.{leaf}"))
                .unwrap()
                .clone();
            w.insert(format!("pre_block.mlp.w0.{leaf}"), b);
            w.insert(format!("pre_block.mlp.w1.{leaf}"), a);
        }
    });
    let (peak, _) = rel(&mutated, &baseline);
    println!("  GeGLU gate/value swap: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "the GeGLU halves are interchangeable"
    );

    // 3. The weight-norm rescale dropped: `weight_g` replaced by ‖weight_v‖, which makes
    //    `g · v / ‖v‖` collapse to `v` itself.
    let mutated = encode_with(|w| {
        let v = w.require("encoder.block.0.weight_v").unwrap().clone();
        let norm = v
            .square()
            .unwrap()
            .sum_axes(&[1, 2], true)
            .unwrap()
            .sqrt()
            .unwrap();
        w.insert("encoder.block.0.weight_g", norm);
    });
    let (peak, _) = rel(&mutated, &baseline);
    println!("  weight-norm rescale dropped: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "reading weight_v raw does not change the encode ({peak:.3e}); the fusion is untested"
    );

    // 4. The two posterior heads swapped.
    let mutated = encode_with(|w| {
        for leaf in ["weight", "bias"] {
            let a = w.require(&format!("mean_proj.{leaf}")).unwrap().clone();
            let b = w.require(&format!("logs_proj.{leaf}")).unwrap().clone();
            w.insert(format!("mean_proj.{leaf}"), b);
            w.insert(format!("logs_proj.{leaf}"), a);
        }
    });
    let (peak, _) = rel(&mutated, &baseline);
    println!("  mean_proj <-> logs_proj: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "the posterior heads are interchangeable"
    );
}

/// `pre_block`'s attention is **causal**: perturbing the last row of its input must leave every
/// earlier row untouched.
///
/// This cannot be checked end-to-end — the convolutional trunk is symmetrically padded and is not
/// causal — which is why `AttnProjection` is public.
#[test]
fn attention_does_not_look_ahead() {
    let f = fixture();
    let mut w = model_weights();
    let cfg = encode_fixture_config();
    let pre = AttnProjection::from_weights(
        &mut w,
        "pre_block",
        cfg.bigvgan.num_mels,
        cfg.latent_channels,
        FIXTURE_HEADS,
        Dtype::Float32,
    )
    .unwrap();

    let x = f.require("in.pre_block.x").unwrap().clone();
    let baseline = pre.forward(&x).unwrap();
    let shape = x.shape().to_vec();
    let (b, n, c) = (shape[0], shape[1], shape[2]);

    // A random perturbation of the LAST row only. NOT a constant offset: `norm1` and `norm3` are
    // LayerNorms, which mean-centre a constant straight back out and would make this inert.
    let mut bump = vec![0f32; (b * n * c) as usize];
    for bi in 0..b as usize {
        for ci in 0..c as usize {
            let idx = (bi * n as usize + (n as usize - 1)) * c as usize + ci;
            bump[idx] = ((idx as f32) * 0.53).sin() * 3.0;
        }
    }
    let moved = pre
        .forward(&mlx_rs::ops::add(&x, Array::from_slice(&bump, &shape)).unwrap())
        .unwrap();

    let head_before = mlx_gen_minimax_h3::tensor::slice_axis(&baseline, 1, 0, n - 1).unwrap();
    let head_after = mlx_gen_minimax_h3::tensor::slice_axis(&moved, 1, 0, n - 1).unwrap();
    let (head_gap, _) = rel(&head_after, &head_before);
    let tail_before = mlx_gen_minimax_h3::tensor::slice_axis(&baseline, 1, n - 1, n).unwrap();
    let tail_after = mlx_gen_minimax_h3::tensor::slice_axis(&moved, 1, n - 1, n).unwrap();
    let (tail_gap, _) = rel(&tail_after, &tail_before);
    println!(
        "  causal: rows 0..{} moved {head_gap:.3e}, the last row moved {tail_gap:.3e}",
        n - 1
    );

    assert!(
        head_gap < UNIT_TOL,
        "perturbing the LAST row moved earlier rows by {head_gap:.3e}; the attention is not causal"
    );
    assert!(
        tail_gap > MUTATION_FLOOR,
        "the perturbation never reached the last row ({tail_gap:.3e}); the probe is inert"
    );
}

/// The `pre_block` head count changes the result, so it is a real parameter rather than a
/// spelling — and the shipped [`ATTN_PROJ_HEADS`] is what `from_weights` uses.
#[test]
fn head_count_is_load_bearing() {
    let f = fixture();
    let cfg = encode_fixture_config();
    let x = f.require("in.pre_block.x").unwrap();

    let build = |heads: i32| {
        let mut w = model_weights();
        AttnProjection::from_weights(
            &mut w,
            "pre_block",
            cfg.bigvgan.num_mels,
            cfg.latent_channels,
            heads,
            Dtype::Float32,
        )
        .unwrap()
        .forward(x)
        .unwrap()
    };

    let (peak, _) = rel(&build(4), &build(FIXTURE_HEADS));
    println!("  4 heads vs {FIXTURE_HEADS}: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "the head count is inert ({peak:.3e}); head_dim drives both the attention scale and the \
         adaptive pool's window layout"
    );

    // A head count that does not divide the trunk is an error, not a truncation.
    let mut w = model_weights();
    assert!(AttnProjection::from_weights(
        &mut w,
        "pre_block",
        cfg.bigvgan.num_mels,
        cfg.latent_channels,
        7,
        Dtype::Float32
    )
    .is_err());
    assert_eq!(ATTN_PROJ_HEADS, 8, "the published checkpoint uses 8 heads");
}

// ---------------------------------------------------------------------------------------------
// Real weights — sc-17149's actual deliverable
// ---------------------------------------------------------------------------------------------

/// Total tensors in the published `audio_vae/` component, and the decode half of them.
const PUBLISHED_AUDIO_TENSORS: usize = 1087;
const AUDIO_ENCODE_TENSORS: usize = 173;

/// The declared encode-path key set must be EXACTLY the published checkpoint's complement of the
/// decode half's 914 — asserted against the real bytes, reading only the safetensors headers.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn declared_encode_tensor_names_match_the_published_checkpoint() {
    let root = snapshot();
    let published: BTreeSet<String> = Weights::from_dir(root.join("audio_vae"))
        .unwrap()
        .keys()
        .map(str::to_string)
        .collect();
    assert_eq!(published.len(), PUBLISHED_AUDIO_TENSORS);

    let cfg = MiniMaxH3AudioVaeConfig::default();
    let encode: BTreeSet<String> = MiniMaxH3AudioVaeEncoder::tensor_names(&cfg)
        .into_iter()
        .collect();
    let decode: BTreeSet<String> = MiniMaxH3AudioVae::tensor_names(&cfg).into_iter().collect();

    let missing: Vec<&String> = encode.difference(&published).collect();
    assert!(
        missing.is_empty(),
        "the encoder requires tensors the checkpoint does not have: {missing:?}"
    );
    assert_eq!(encode.len(), AUDIO_ENCODE_TENSORS);
    assert!(
        encode.is_disjoint(&decode),
        "the two halves must not claim the same tensor"
    );

    let union: BTreeSet<String> = encode.union(&decode).cloned().collect();
    assert_eq!(
        union, published,
        "encode + decode must be exactly the published checkpoint — nothing left over, nothing \
         invented"
    );
    println!(
        "AUDIO VAE: {} published tensors = {} decode + {} encode; the port now consumes ALL of them",
        published.len(),
        decode.len(),
        encode.len()
    );
}

/// Load the real 577 MB audio VAE and encode a stereo probe, against the committed
/// `diffusers.AutoencoderKLMiniMaxH3Audio` reference for the SAME probe and the SAME bytes.
///
/// This is the gate the committed tiny fixture cannot be: it runs the shipped 2048-wide trunk,
/// the shipped 8-head `pre_block` and its exact 256 → 32 adaptive pool. The reference tensors are
/// dumped by the generator and committed alongside the fixture (`real.*`), so the test needs only
/// the snapshot — and it asserts on that rather than skipping, because an `#[ignore]`d test that
/// quietly returns prints `ok` in 0.00 s and reads as a pass.
#[test]
#[ignore = "needs a real MiniMax-H3 snapshot (MINIMAX_H3_SNAPSHOT); dev box / macos-mlx only"]
fn real_weight_encode_matches_the_diffusers_reference() {
    let root = snapshot();
    let f = fixture();
    let started = std::time::Instant::now();

    let mut w = Weights::from_dir(root.join("audio_vae")).unwrap();
    assert_eq!(w.keys().count(), PUBLISHED_AUDIO_TENSORS);
    let cfg = MiniMaxH3AudioVaeConfig::default();
    let enc = MiniMaxH3AudioVaeEncoder::from_weights(&mut w, &cfg, Dtype::Float32).unwrap();
    assert_eq!(enc.hop_length(), 800, "40 latents/s at 32 kHz");

    let waveform = f.require("real.in.waveform").unwrap();
    assert_eq!(
        waveform.shape(),
        &[2, 1, 8000],
        "0.25 s of stereo, as 2 batch items"
    );
    let posterior = enc.encode(waveform).unwrap();
    assert_eq!(
        posterior.mean().shape(),
        &[2, 32, 10],
        "8000 / 800 = 10 frames"
    );

    assert_parity(
        posterior.mode(),
        f.require("real.out.mean").unwrap(),
        TOL,
        "real-weight encode -> mean_proj",
    );
    assert_parity(
        posterior.logs(),
        f.require("real.out.logs").unwrap(),
        TOL,
        "real-weight encode -> logs_proj",
    );

    // The port really ran: the result is finite, not constant, and the two channels differ.
    let samples = values(posterior.mode());
    assert!(
        samples.iter().all(|s| s.is_finite()),
        "real encode produced NaN/Inf"
    );
    let spread = std_dev(posterior.mode());
    assert!(
        spread > 1e-3,
        "the encoded latents are ~constant (std {spread:.3e})"
    );
    let (channel_gap, _) = rel(
        &posterior
            .mode()
            .take_axis(Array::from_slice(&[0], &[1]), 0)
            .unwrap(),
        &posterior
            .mode()
            .take_axis(Array::from_slice(&[1], &[1]), 0)
            .unwrap(),
    );
    println!(
        "  real-weight encode: {:.1}s, latent std {spread:.4}, L-vs-R rel {channel_gap:.3e}",
        started.elapsed().as_secs_f32()
    );
    assert!(
        channel_gap > MUTATION_FLOOR,
        "the two stereo channels encoded near-identically ({channel_gap:.3e}); a port that \
         collapsed the batch would pass"
    );

    // Every published tensor is now consumed by one half or the other.
    w.remove_accessed();
    let leftover: BTreeSet<String> = w.keys().map(str::to_string).collect();
    let decode: BTreeSet<String> = MiniMaxH3AudioVae::tensor_names(&cfg).into_iter().collect();
    let orphans: Vec<&String> = leftover.difference(&decode).collect();
    assert!(
        orphans.is_empty(),
        "published tensors that neither half reads: {orphans:?}"
    );
}
