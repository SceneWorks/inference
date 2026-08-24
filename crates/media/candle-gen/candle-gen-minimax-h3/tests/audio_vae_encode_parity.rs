//! sc-17157: audio-VAE **encode** parity against `diffusers.AutoencoderKLMiniMaxH3Audio`, candle
//! side — the half `ref2va` uses to turn a reference soundtrack into conditioning latents.
//!
//! Fixture `tests/fixtures/audio_vae_encode.safetensors`, copied byte-for-byte from the MLX lane
//! (`cross_backend.rs` asserts that identity). It was produced by
//! `crates/media/mlx-gen/tools/dump_minimax_h3_audio_vae_encode.py`.
//!
//! # Why diffusers is the reference here, and not the snapshot's own bundle
//!
//! The decode half (`tests/audio_vae_parity.rs`) runs against `FL2VA/audio_vae`'s `DacAudioVAE`.
//! That bundle is **inference-only and has no `encode` method at all** — it ships `preprocess` and
//! `decode` — so `AutoencoderKLMiniMaxH3Audio` is the only executable reference for this half. That
//! is also what `layout.rs` Rule 3 asks for: the golden comes from the graph that loads the
//! *published* tensors, and `fixture_provenance_is_the_converted_reference` asserts the recorded
//! `provenance` / `reference` / `reference_version` metadata rather than trusting it.
//!
//! # The geometry is deliberately harder than the shipped model
//!
//! Two knobs in the generator are chosen to be *worse* cases than production:
//!
//! * `encoder_rates = (2, 5)` keeps an ODD stride, whose `padding = ceil(stride / 2)` is what makes
//!   the shipped `(2, 4, 4, 5, 5)` chain land on exactly `samples / 800`;
//! * `num_attention_heads = 2` over a 96-wide trunk puts the attention head width at 48 against a
//!   32-channel output, so `adaptive_avg_pool1d` runs with **overlapping** windows. The shipped
//!   256 → 32 is an exact 8:1 that a `reshape(.., 32, 8).mean(-1)` also reproduces; the ragged case
//!   does not, and `out.pool.*` pins all three regimes against torch directly.
//!
//! # Tolerances
//!
//! This lane is f32 on the CPU and floors at round-off, so its gates are three to four orders
//! tighter than the MLX sibling's — MLX evaluates f32 matmul in **reduced precision on Metal** and
//! measures 8.8e-3 end to end where this measures ~1e-6. Every number below was **measured on this
//! fixture** and rounded up; each test prints its residual so the real margin stays auditable.
//!
//! The one loose gate is [`STD_TOL`], and it is loose for a stated reason rather than a hedged one:
//! exponentiating turns `logs`' *absolute* error into `std`'s *relative* one (`d(e^l)/e^l = dl`),
//! and `logs` peaks around 5 here. The closed form `std == exp(logs)` is checked separately at
//! [`UNIT_TOL`] on this port's own tensors, where no amplification applies.

use crate::common;

use std::collections::{BTreeSet, HashMap};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::Weights;
use candle_gen_minimax_h3::audio_vae_encoder::{
    adaptive_avg_pool_last_axis, AttnProjection, CausalAttention, DacEncoder, EncoderBlock,
    ResidualUnit, WnConv1d, LAYER_NORM_EPS,
};
use candle_gen_minimax_h3::{
    AudioDiagonalGaussian, BigVganConfig, MiniMaxH3AudioVae, MiniMaxH3AudioVaeConfig,
    MiniMaxH3AudioVaeEncoder,
};

use common::{
    audio_fixture_config, rel, std_dev, weights, Golden, AUDIO_ENCODE_FIXTURE, AUDIO_FIXTURE,
};

/// Whole encoder / whole trunk, through ~30 convolutions. Measured 1.7e-6 on this lane.
const TOL: f32 = 1e-4;
/// `pre_block` and its attention branch. Measured 2.6e-7.
const BLOCK_TOL: f32 = 1e-5;
/// One residual unit / one downsampling stage / one convolution. Measured 2.0e-7 … 4.6e-7.
const STAGE_TOL: f32 = 1e-5;
/// Pieces that run no matmul at all — the adaptive pool (0.0 … 2.8e-8), the latent normalization
/// round trip (9.1e-8) and `std == exp(logs)` (exactly 0.0).
const UNIT_TOL: f32 = 1e-5;
/// The posterior's `std = exp(logs)` against the reference's own. Measured 5.7e-6 — amplified from
/// `logs`' 1.5e-6 by the exponential, see the module docs.
const STD_TOL: f32 = 1e-3;
/// Mutation probes must clear the whole-encoder gate by orders, or "the output moved" could be
/// numerical jitter rather than a wiring difference.
const MUTATION_FLOOR: f32 = 5e-2;
/// The floor for the **per-tensor** probes in `every_weight_role_is_load_bearing`, 100x [`TOL`].
///
/// Lower than [`MUTATION_FLOOR`] because the smallest genuine reading in the encode path is
/// `pre_block.attn.q_bias` at 4.5e-2 — a query-side bias, whose effect the softmax damps by
/// construction. Set below that rather than above it: a floor a correctly-read tensor cannot clear
/// makes the probe unpassable, and 1e-2 is still four orders above this suite's parity gate, so
/// "the output moved" cannot be round-off.
const PROBE_FLOOR: f32 = 1e-2;

/// The head count the fixture was dumped at — NOT the shipped `ATTN_PROJ_HEADS` (8).
const FIXTURE_HEADS: usize = 2;

fn fixture() -> Golden {
    Golden::load(AUDIO_ENCODE_FIXTURE)
}

/// The fixture minus the reference-side extras — i.e. exactly the model weights in the published
/// naming, which is what the loader consumes for real weights.
fn model_map(g: &Golden) -> HashMap<String, Tensor> {
    g.model_map(&["in.", "out.", "const.", "real."])
}

fn model_weights(g: &Golden) -> Weights {
    weights(model_map(g))
}

/// The tiny geometry the fixture was dumped at. Mirrors `dump_minimax_h3_audio_vae_encode.py` and
/// the MLX sibling's `encode_fixture_config`, so the two lanes read the same bytes as the same
/// model.
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
        // `num_mels` IS the audio VAE's `latent_dim`. 96 = encoder_dim 24 · 2^2, the constructor's
        // own derivation. The rest of the block is the fixture's decoder, which nothing here reads.
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

fn encoder(g: &Golden) -> MiniMaxH3AudioVaeEncoder {
    MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &model_weights(g),
        &encode_fixture_config(),
        FIXTURE_HEADS,
        &Device::Cpu,
        DType::F32,
    )
    .unwrap()
}

fn assert_parity(got: &Tensor, want: &Tensor, tol: f32, what: &str) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let (peak, mean) = rel(got, want);
    println!("  {what}: peak rel {peak:.3e} (mean {mean:.3e}, tol {tol:.1e})");
    assert!(
        peak < tol,
        "{what}: peak-relative error {peak:.3e} (mean {mean:.3e}) exceeds {tol:.1e}"
    );
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
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
    let g = fixture();
    assert_eq!(
        g.meta("provenance"),
        Some("converted-checkpoint"),
        "the encode golden must be built from the published/converted layout"
    );
    assert_eq!(
        g.meta("reference"),
        Some("diffusers.AutoencoderKLMiniMaxH3Audio")
    );
    assert_eq!(g.meta("half"), Some("encode"));
    let version = g
        .meta("reference_version")
        .expect("the fixture records the diffusers version it was produced with");
    assert!(
        version.starts_with(char::is_numeric),
        "reference_version {version:?} is not a version"
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
        let recorded: f32 = g
            .meta(key)
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
// adaptive_avg_pool1d
// ---------------------------------------------------------------------------------------------

/// `CausalAttention` pools the surviving head width down to `latent_channels`, and the three
/// dumped cases walk every regime of PyTorch's window rule `[⌊i·L/out⌋, ⌈(i+1)·L/out⌉)`: the
/// shipped exact 256 → 32 tiling, the fixture's OVERLAPPING 48 → 32, and a 44 → 32 whose windows
/// differ in width.
///
/// A port that implemented the pool as `reshape(.., out, L/out).mean(-1)` passes the first and
/// cannot even express the other two — which the control at the end of this test measures rather
/// than asserts in prose.
#[test]
fn adaptive_pool_matches_the_reference() {
    let g = fixture();
    for tag in ["uniform", "ragged", "varsize"] {
        let params = g.i32_vec(&format!("const.pool.{tag}"));
        let (len, out) = (params[0] as usize, params[1] as usize);
        let x = g.tensor(&format!("in.pool.{tag}"));
        assert_eq!(*x.dims().last().unwrap(), len);
        let got = adaptive_avg_pool_last_axis(&x, out).unwrap();
        assert_parity(
            &got,
            &g.tensor(&format!("out.pool.{tag}")),
            UNIT_TOL,
            &format!("adaptive_avg_pool1d({len} -> {out})"),
        );
    }

    // The shipped 256 -> 32 geometry really IS an exact tiling, so a reshape-and-mean reproduces
    // it — which is precisely why the ragged case has to exist.
    let naive = g
        .tensor("in.pool.uniform")
        .reshape((2, 5, 32, 8))
        .unwrap()
        .mean(3)
        .unwrap();
    let (peak, _) = rel(&naive, &g.tensor("out.pool.uniform"));
    println!("  reshape-and-mean on the SHIPPED 256->32 geometry: peak rel {peak:.3e}");
    assert!(
        peak < UNIT_TOL,
        "the shipped geometry is supposed to be an exact tiling; if it is not, the ragged case is \
         no longer the discriminating one"
    );

    // ...and on the ragged 48 -> 32 it is wrong by orders, so the fixture discriminates.
    let ragged = g.tensor("in.pool.ragged");
    let naive_ragged = {
        let (b, n, l) = (2usize, 5usize, 48usize);
        // The nearest thing a reshape port can do: 48 does not divide by 32, so it must drop to a
        // coarser exact tiling. 48 -> 16 -> upsampled by repeat is the charitable reading.
        ragged
            .reshape((b, n, 16, l / 16))
            .unwrap()
            .mean(3)
            .unwrap()
            .repeat((1, 1, 2))
            .unwrap()
    };
    let (ragged_gap, _) = rel(&naive_ragged, &g.tensor("out.pool.ragged"));
    println!("  a tiling port on the RAGGED 48->32 geometry: peak rel {ragged_gap:.3e}");
    assert!(
        ragged_gap > MUTATION_FLOOR,
        "the ragged case does not discriminate against a tiling port ({ragged_gap:.3e})"
    );
}

// ---------------------------------------------------------------------------------------------
// The trunk, pre_block, and the whole encode
// ---------------------------------------------------------------------------------------------

/// One convolution, then one residual unit, then one downsampling stage, then the whole trunk.
///
/// Held as a walk rather than one end-to-end bound because a structural error — a wrong dilation, a
/// wrong `ceil(stride/2)` padding, a mis-ordered Snake — is order-1 wrong at the FIRST stage that
/// contains it, not 1e-7 wrong there and 1e-4 wrong twenty layers later. An end-to-end golden alone
/// cannot tell those apart.
///
/// Everything here is NCL, which is candle's conv layout *and* the reference's, so unlike the MLX
/// sibling this port compares the reference's tensors without a transpose.
#[test]
fn the_trunk_matches_the_reference_stage_by_stage() {
    let g = fixture();
    let cfg = encode_fixture_config();
    let w = model_weights(&g);
    let x = g.tensor("in.encode.waveform");
    assert_eq!(x.dims(), [2, 1, 320], "the reference dumps NCL");

    // ONE weight-normed convolution: 1 -> 24 channels, k7, 'same' padding.
    let conv_in = WnConv1d::from_weights(&w, "encoder.block.0", 1, 3, 1, DType::F32)
        .unwrap()
        .forward(&x)
        .unwrap();
    assert_parity(
        &conv_in,
        &g.tensor("out.stage.conv_in"),
        STAGE_TOL,
        "encoder.block.0 (one weight-normed conv)",
    );

    // One residual unit: two convolutions and two Snakes, at 24 channels.
    let unit = ResidualUnit::from_weights(&w, "encoder.block.1.block.0", 1, DType::F32).unwrap();
    assert_parity(
        &unit.forward(&conv_in).unwrap(),
        &g.tensor("out.stage.unit0"),
        STAGE_TOL,
        "one ResidualUnit (dilation 1)",
    );

    // One EncoderBlock: three residual units at dilations 1/3/9, a Snake, and the stride-2
    // channel-doubling convolution.
    let block = EncoderBlock::from_weights(&w, "encoder.block.1", 2, DType::F32).unwrap();
    let staged = block.forward(&conv_in).unwrap();
    assert_eq!(staged.dims(), [2, 48, 160], "stride 2 halves 320 -> 160");
    assert_parity(
        &staged,
        &g.tensor("out.stage.block1"),
        STAGE_TOL,
        "one EncoderBlock (stride 2)",
    );

    // ...and the whole trunk.
    let trunk = DacEncoder::from_weights(&w, "encoder", &cfg, DType::F32).unwrap();
    let hidden = trunk.forward(&x).unwrap();
    assert_eq!(
        hidden.dims(),
        [2, 96, 32],
        "320 samples at hop 10 -> 32 frames, latent_dim 96"
    );
    assert_parity(
        &hidden,
        &g.tensor("out.trunk.hidden"),
        TOL,
        "the whole DAC trunk",
    );
}

/// The attention branch on its own — `attn(norm1(x))`, before the residual sum can hide it.
///
/// This is the piece that carries the causal mask, the head **mean-pool** and the adaptive pool,
/// and the reference dumps it separately for exactly that reason.
#[test]
fn attention_branch_matches_the_reference() {
    let g = fixture();
    let cfg = encode_fixture_config();
    let w = model_weights(&g);
    let x = g.tensor("in.pre_block.x");

    let attn = CausalAttention::from_weights(
        &w,
        "pre_block.attn",
        cfg.bigvgan.num_mels,
        cfg.latent_channels,
        FIXTURE_HEADS,
        DType::F32,
    )
    .unwrap();
    let normed = candle_gen_minimax_h3::nn::layer_norm(
        &x,
        &g.tensor("pre_block.norm1.weight"),
        &g.tensor("pre_block.norm1.bias"),
        LAYER_NORM_EPS,
    )
    .unwrap();
    assert_parity(
        &attn.forward(&normed).unwrap(),
        &g.tensor("out.pre_block.attn"),
        BLOCK_TOL,
        "CausalAttention branch",
    );

    // The composed block: `proj(norm3(x)) + attn(norm1(x))`, then `+ mlp(norm2(·))`.
    let pre = AttnProjection::from_weights(
        &w,
        "pre_block",
        cfg.bigvgan.num_mels,
        cfg.latent_channels,
        FIXTURE_HEADS,
        DType::F32,
    )
    .unwrap();
    assert_parity(
        &pre.forward(&x).unwrap(),
        &g.tensor("out.pre_block.y"),
        BLOCK_TOL,
        "AttnProjection (pre_block)",
    );
    // `attention_branch` is the same computation the composed block performs, so the isolated
    // comparison above is genuinely of the shipped code path rather than of a test-local copy.
    assert_parity(
        &pre.attention_branch(&x).unwrap(),
        &g.tensor("out.pre_block.attn"),
        BLOCK_TOL,
        "AttnProjection::attention_branch",
    );

    // The attention branch must be a genuinely different tensor from the block output, or the
    // golden above would not be isolating anything.
    let (branch_gap, _) = rel(
        &g.tensor("out.pre_block.attn"),
        &g.tensor("out.pre_block.y"),
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
    let g = fixture();
    let enc = encoder(&g);
    let posterior = enc.encode(&g.tensor("in.encode.waveform")).unwrap();
    assert_eq!(
        posterior.mean().dims(),
        [2, 32, 32],
        "320 samples at hop 10 -> 32 latent frames, 32 channels, 2 batch items"
    );
    assert_parity(
        posterior.mode(),
        &g.tensor("out.encode.mean"),
        TOL,
        "encode -> mean_proj",
    );
    assert_parity(
        posterior.logs(),
        &g.tensor("out.encode.logs"),
        TOL,
        "encode -> logs_proj",
    );
    assert_parity(
        posterior.std(),
        &g.tensor("out.encode.std"),
        STD_TOL,
        "posterior std = exp(logs)",
    );
    // The closed form itself, on the port's own tensors, where no amplification applies.
    assert_parity(
        posterior.std(),
        &posterior.logs().exp().unwrap(),
        UNIT_TOL,
        "std == exp(logs)",
    );
    // `mode()` is bit-for-bit `mean_proj`'s output, as the reference documents.
    assert_eq!(flat(posterior.mode()), flat(posterior.mean()));

    // **The dtype is f32, asserted rather than assumed.** Every value in this fixture is
    // representable in bf16 to within this suite's gates on some tensors, so a dropped cast is not
    // reliably visible numerically — diffusers pins this whole component with
    // `_keep_in_fp32_modules` for that reason.
    assert_eq!(posterior.mean().dtype(), DType::F32);
    assert_eq!(posterior.std().dtype(), DType::F32);
}

/// `encode` zero-pads on the RIGHT to a whole number of hops, so a clip that is not a multiple of
/// the hop still produces `ceil(S / hop)` frames — and the same ones the reference's padded clip
/// does.
#[test]
fn encode_right_pads_to_a_whole_hop() {
    let g = fixture();
    let enc = encoder(&g);
    assert_eq!(enc.hop_length(), 10, "hop = 2 * 5");

    let ragged = g.tensor("in.encode_pad.waveform");
    assert_eq!(
        ragged.dims(),
        [2, 1, 311],
        "311 = 320 - 10 + 1, not a whole hop"
    );
    let got = enc.encode(&ragged).unwrap();
    assert_eq!(got.mean().dims(), [2, 32, 32], "ceil(311 / 10) = 32 frames");
    assert_parity(
        got.mode(),
        &g.tensor("out.encode_pad.mean"),
        TOL,
        "encode with a right-pad",
    );

    // The pad is not a no-op: a ragged clip does NOT encode to the same latents as the full one,
    // so a port that silently truncated to a whole hop would be visible here.
    let full = enc.encode(&g.tensor("in.encode.waveform")).unwrap();
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
    let g = fixture();
    let cfg = encode_fixture_config();
    let present: BTreeSet<String> = model_map(&g).into_keys().collect();
    // 3 + 2 * 28 + 4 + 22 + 4 for a two-stage encoder.
    assert_eq!(
        present.len(),
        89,
        "the fixture carries the encode half only"
    );

    let declared: BTreeSet<String> = MiniMaxH3AudioVaeEncoder::tensor_names(&cfg)
        .into_iter()
        .collect();
    assert_eq!(
        declared, present,
        "declared tensor names differ from the fixture's"
    );

    // ...and the loader really consumes exactly that set: building it from ONLY the declared names
    // succeeds, so nothing it reads is outside the inventory.
    let only_declared: HashMap<String, Tensor> = model_map(&g)
        .into_iter()
        .filter(|(k, _)| declared.contains(k))
        .collect();
    MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &weights(only_declared),
        &cfg,
        FIXTURE_HEADS,
        &Device::Cpu,
        DType::F32,
    )
    .expect("the declared inventory is sufficient to build the encoder");
}

/// A checkpoint or configuration that disagrees with the loader is a loud error, not a silent
/// re-interpretation. **Each mutation is applied on its own**, against a control that loads the
/// same fixture unmutated, so a refusal cannot be credited to a different check.
#[test]
fn a_mismatched_configuration_is_rejected() {
    let g = fixture();
    let cfg = encode_fixture_config();
    let build = |w: &Weights, c: &MiniMaxH3AudioVaeConfig| {
        MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
            w,
            c,
            FIXTURE_HEADS,
            &Device::Cpu,
            DType::F32,
        )
    };
    // Control.
    build(&model_weights(&g), &cfg).expect("the unmutated fixture must load");

    // `attn_proj = false` has no `pre_block`, so the trunk cannot reach the posterior heads.
    let mut without = cfg.clone();
    without.attn_proj = false;
    let err = build(&model_weights(&g), &without).unwrap_err().to_string();
    assert!(err.contains("attn_proj"), "unexpected error: {err}");

    // `latent_dim % latent_channels != 0` is the branch where the reference widens
    // `attn_proj_dim` to the next power of two and diffusers refuses outright.
    let mut indivisible = cfg.clone();
    indivisible.latent_channels = 7;
    let err = build(&model_weights(&g), &indivisible)
        .unwrap_err()
        .to_string();
    assert!(err.contains("latent_dim"), "unexpected error: {err}");

    // A GeGLU MLP at a different `mlp_ratio` is rejected: MLP_RATIO is enforced, not declared.
    let mut map = model_map(&g);
    let w0 = map["pre_block.mlp.w0.weight"].clone();
    let hidden = w0.dims()[0];
    assert_eq!(
        hidden,
        cfg.latent_channels * 2,
        "the published GeGLU hidden width is latent_channels * MLP_RATIO"
    );
    map.insert(
        "pre_block.mlp.w0.weight".into(),
        w0.narrow(0, 0, hidden - 1).unwrap().contiguous().unwrap(),
    );
    let err = build(&weights(map), &cfg).unwrap_err().to_string();
    assert!(err.contains("mlp_ratio"), "unexpected error: {err}");

    // A latent-statistics list of the wrong length is rejected rather than broadcast.
    let mut short = cfg.clone();
    short.latents_mean.truncate(8);
    let err = build(&model_weights(&g), &short).unwrap_err().to_string();
    assert!(err.contains("latents_mean"), "unexpected error: {err}");
}

/// The model is MONO. A stereo clip is two BATCH items, and a `[1, 2, samples]` packing — the
/// plausible transposition — is rejected rather than encoded as something the model never saw.
#[test]
fn mis_packed_waveforms_are_rejected() {
    let g = fixture();
    let enc = encoder(&g);
    let wave = g.tensor("in.encode.waveform");
    assert!(enc.encode(&wave).is_ok());

    let stereo_packed = wave.reshape((1, 2, 320)).unwrap();
    let e = enc.encode(&stereo_packed).unwrap_err().to_string();
    assert!(
        e.contains("stereo is B = 2"),
        "[1, 2, samples] must be rejected BY THE PACKING CHECK, not by something downstream: {e}"
    );
    // Rank 2 and rank 4 are errors too, and by the same check.
    for bad in [
        wave.reshape((2, 320)).unwrap(),
        wave.reshape((2, 1, 320, 1)).unwrap(),
    ] {
        let e = enc.encode(&bad).unwrap_err().to_string();
        assert!(e.contains("stereo is B = 2"), "{e}");
    }
}

/// `normalize` is `(z − mean) / std` per channel, and composing it with the decode half's
/// `denormalize` is the identity.
///
/// Cross-checked against the *other* module's implementation rather than against a restatement of
/// the same formula: the two are supposed to be inverses, and only running both can show it.
#[test]
fn normalization_is_the_decoders_exact_inverse() {
    let g = fixture();
    let enc = encoder(&g);
    let z = enc
        .encode(&g.tensor("in.encode.waveform"))
        .unwrap()
        .mode()
        .clone();

    let normalized = enc.normalize(&z).unwrap();
    let (moved, _) = rel(&normalized, &z);
    assert!(moved > 1e-2, "normalization was a no-op ({moved:.3e})");

    // The closed form, element by element. `[B, 32, T]`: element (0, c, t) is at index c*T + t.
    let raw = flat(&z);
    let got = flat(&normalized);
    let cfg = MiniMaxH3AudioVaeConfig::default();
    let t = z.dims()[2];
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
    let decode = Golden::load(AUDIO_FIXTURE);
    let dw = weights(decode.model_map(&["in.", "out.", "const.", "amp."]));
    let vae =
        MiniMaxH3AudioVae::from_weights(&dw, &audio_fixture_config(), &Device::Cpu, DType::F32)
            .unwrap();
    assert_parity(
        &vae.denormalize(&normalized).unwrap(),
        &z,
        UNIT_TOL,
        "normalize -> denormalize",
    );

    // Rank-4 stereo packing uses the same second-to-last channel axis.
    let stereo = z.reshape((1, 2, 32, 32)).unwrap();
    let both = enc.normalize(&stereo).unwrap();
    assert_eq!(both.dims(), stereo.dims());
    assert_parity(
        &vae.denormalize(&both).unwrap(),
        &stereo,
        UNIT_TOL,
        "normalize -> denormalize (stereo packing)",
    );

    // A latent whose channel count disagrees with the config is an error, not a broadcast.
    assert!(enc.normalize(&z.reshape((2, 16, 64)).unwrap()).is_err());
    assert!(enc
        .normalize(&Tensor::from_vec(vec![1.0f32], 1, &Device::Cpu).unwrap())
        .is_err());
}

/// The audio posterior's second parameter is a log **standard deviation** with **no clamp** — not
/// the video encoder's clamped log variance.
///
/// The two are shape-identical and `mode()` is unaffected by the difference, so nothing the
/// MiniMax-H3 pipeline consumes would ever reveal a port that reused `vae_encoder::DiagonalGaussian`
/// here. This is the assertion that does.
#[test]
fn log_std_is_not_log_variance() {
    let dev = Device::Cpu;
    let values = [0.5f32, 3.0, -40.0];
    let logs = Tensor::from_vec(values.to_vec(), (1, 3, 1), &dev).unwrap();
    let mean = Tensor::zeros((1, 3, 1), DType::F32, &dev).unwrap();
    let p = AudioDiagonalGaussian::new(mean, logs).unwrap();
    let std = flat(p.std());

    for (i, l) in values.iter().enumerate() {
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
            "the two posterior conventions agree at logs = {l}; the distinction is untestable there"
        );
    }
    // The last case additionally proves the absence of the clamp: -40 is outside [-30, 20].
    assert!(
        std[2] > 0.0 && std[2] < 1e-15,
        "no clamp is applied to logs"
    );
}

/// A constant, all-zero or degenerate golden is a false green.
#[test]
fn fixture_is_not_degenerate() {
    let g = fixture();
    for key in [
        "out.encode.mean",
        "out.encode.logs",
        "out.trunk.hidden",
        "out.pre_block.y",
        "out.pre_block.attn",
        "out.pool.ragged",
    ] {
        let t = g.tensor(key);
        assert!(
            std_dev(&t) > 1e-4,
            "{key} is ~constant; a constant golden is a false green"
        );
    }

    // The two posterior heads must be distinguishable, or swapping them would be invisible.
    let (gap, _) = rel(&g.tensor("out.encode.mean"), &g.tensor("out.encode.logs"));
    println!("  mean vs logs: peak rel {gap:.3e}");
    assert!(gap > MUTATION_FLOOR, "mean and logs are indistinguishable");

    // The two batch items are genuinely different waveforms, so a port that collapsed the batch
    // (or broadcast one channel) could not pass.
    let wave = g.tensor("in.encode.waveform");
    let (channel_gap, _) = rel(&wave.i(0).unwrap(), &wave.i(1).unwrap());
    assert!(
        channel_gap > MUTATION_FLOOR,
        "the two batch items are near-identical ({channel_gap:.3e})"
    );

    // `zero_k_bias` really is zero in this fixture, as it is in the checkpoint...
    assert!(
        flat(&g.tensor("pre_block.attn.zero_k_bias"))
            .iter()
            .all(|v| *v == 0.0),
        "zero_k_bias is a zero buffer"
    );
    // ...but `q_bias` and `v_bias` are NOT, or the fused-bias assembly would be unobservable.
    for key in ["pre_block.attn.q_bias", "pre_block.attn.v_bias"] {
        assert!(std_dev(&g.tensor(key)) > 1e-3, "{key} is ~constant");
    }
}

/// Rebuild the encoder with exactly one tensor replaced and return its `encode` mean.
fn encode_with(g: &Golden, key: &str, replacement: Tensor) -> Tensor {
    let mut map = model_map(g);
    map.insert(key.into(), replacement);
    MiniMaxH3AudioVaeEncoder::from_weights_with_heads(
        &weights(map),
        &encode_fixture_config(),
        FIXTURE_HEADS,
        &Device::Cpu,
        DType::F32,
    )
    .unwrap()
    .encode(&g.tensor("in.encode.waveform"))
    .unwrap()
    .mode()
    .clone()
}

/// **`weight_v` is scale-invariant, and that is the weight-norm fusion working.**
///
/// `w = g · v / ‖v‖` is homogeneous of degree zero in `v`, so scaling `weight_v` must change
/// nothing at all — while scaling `weight_g` scales the output. A port that skipped the
/// normalization (used `v` directly, or `g · v`) fails this in the first direction; a port that
/// dropped `g` fails it in the second. Neither failure is visible to any shape or key check, and
/// the fixture's own recorded `mutation_weight_norm_skipped_rel` says the reference moves by 1.0
/// under exactly this defect.
///
/// This is also why `every_weight_role_is_load_bearing` mutates `weight_v` **affinely** rather than
/// by a scale: a scale there would report "not read" for a tensor that is read correctly.
#[test]
fn weight_v_is_scale_invariant_and_weight_g_is_not() {
    let g = fixture();
    let baseline = encoder(&g)
        .encode(&g.tensor("in.encode.waveform"))
        .unwrap()
        .mode()
        .clone();

    let v = model_map(&g)["encoder.block.0.weight_v"].clone();
    let scaled_v = encode_with(&g, "encoder.block.0.weight_v", (v.clone() * 7.0).unwrap());
    let (peak, _) = rel(&scaled_v, &baseline);
    println!("  weight_v x7: peak rel {peak:.3e}");
    assert!(
        peak < UNIT_TOL,
        "scaling weight_v must be a NO-OP under `g · v / ‖v‖`; it moved by {peak:.3e}, so the \
         normalization is not being applied"
    );

    // ...and it is not simply ignored: rotating its direction does move the output.
    let flipped = encode_with(
        &g,
        "encoder.block.0.weight_v",
        v.flip(&[2]).unwrap().contiguous().unwrap(),
    );
    let (peak, _) = rel(&flipped, &baseline);
    println!("  weight_v kernel reversed: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "reversing weight_v's kernel taps did not move the output ({peak:.3e}); it is not read"
    );

    let gm = model_map(&g)["encoder.block.0.weight_g"].clone();
    let scaled_g = encode_with(&g, "encoder.block.0.weight_g", (gm * 7.0).unwrap());
    let (peak, _) = rel(&scaled_g, &baseline);
    println!("  weight_g x7: peak rel {peak:.3e}");
    assert!(
        peak > MUTATION_FLOOR,
        "scaling weight_g must move the output ({peak:.3e}); the gain is being dropped"
    );
}

/// **Mutation control: every distinct tensor ROLE in the encode path moves the output**, and each
/// is mutated ON ITS OWN against the unmutated baseline.
///
/// One representative per role, including both halves of a weight-norm pair (a port that read only
/// `weight_v` would encode plausibly), the `Snake1d` alphas, the three separately-stored attention
/// biases, both GeGLU projections, and both ends of the residual stack.
///
/// The mutation is **affine** (`1.5·t + 0.25`), not a scale: `weight_v` is scale-invariant by
/// construction (see `weight_v_is_scale_invariant_and_weight_g_is_not`), so a pure scale would
/// report a correctly-read tensor as unread.
#[test]
fn every_weight_role_is_load_bearing() {
    let g = fixture();
    let baseline = encoder(&g)
        .encode(&g.tensor("in.encode.waveform"))
        .unwrap()
        .mode()
        .clone();

    let probes: [&str; 14] = [
        "encoder.block.0.weight_g",
        "encoder.block.0.weight_v",
        "encoder.block.0.bias",
        "encoder.block.1.block.0.block.0.alpha",
        "encoder.block.1.block.0.block.1.weight_v",
        "encoder.block.1.block.2.block.3.weight_g",
        "encoder.block.1.block.3.alpha",
        "encoder.block.2.block.4.weight_v",
        "pre_block.norm1.weight",
        "pre_block.attn.qkv.weight",
        "pre_block.attn.q_bias",
        "pre_block.attn.v_bias",
        "pre_block.mlp.w0.weight",
        "logs_proj.weight",
    ];
    for key in probes {
        let t = model_map(&g)[key].clone();
        // `1.5·t + 0.25` — a change no shape check, key-count check or dtype check can see.
        let mutated = encode_with(&g, key, ((t * 1.5).unwrap() + 0.25).unwrap());
        // `logs_proj` feeds only the OTHER head, so it must NOT move the mean — the one probe whose
        // expected direction is inverted, and the reason this is not a blanket "everything moves".
        let (peak, _) = rel(&mutated, &baseline);
        println!("  {key}: peak rel {peak:.3e}");
        if key == "logs_proj.weight" {
            assert!(
                peak < UNIT_TOL,
                "{key} feeds logs_proj alone and must not move mean_proj's output ({peak:.3e})"
            );
        } else {
            assert!(
                peak > PROBE_FLOOR,
                "{key} does not move the encode output ({peak:.3e}); it is not being read"
            );
        }
    }
}
