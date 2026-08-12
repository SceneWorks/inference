//! sc-17154: **cross-backend agreement with the MLX lane**, and an honest statement of the floor
//! that comparison sits on.
//!
//! # How the two backends are compared at all
//!
//! `mlx-gen-minimax-h3` and `candle-gen-minimax-h3` cannot coexist in one build — MLX is
//! macOS/Metal, candle-cuda is Windows/Linux — so nothing can call both in one process. The repo's
//! established pattern is a **shared committed golden**: both crates assert against the same
//! `.safetensors`, and cross-backend agreement is inferred as the sum of two independent residuals.
//!
//! This file does that *and* one thing stronger. `fixtures_are_byte_identical_to_the_mlx_lanes`
//! pins the shared-golden half. Then `mlx-gen-minimax-h3/tests/cross_backend_record.rs` — an
//! `#[ignore]`d generator run on Metal — records the MLX lane's **own decode** of those goldens
//! into `tests/fixtures/mlx_cross_backend.safetensors`, and the tests below compare this port's
//! tensors against MLX's directly. So the headline number is measured, not bounded.
//!
//! # The tolerance, the floor, and what the floor is made of
//!
//! [`CROSS_TOL`] is `2e-2` peak-relative. That is not a statement about how close the two ports
//! are — they are far closer — it is the bound the **noisier** of the two lanes can support.
//!
//! `the_cross_backend_floor_is_the_mlx_lanes_own_reduced_precision` measures both halves from
//! committed data rather than quoting them:
//!
//! * **candle vs the reference** is 2.1e-7 … 3.3e-6 (see `video_vae_parity.rs` /
//!   `audio_vae_parity.rs`): f32 on the CPU, essentially round-off.
//! * **MLX vs the reference** is ~1e-3 for the video decode and 2.7e-3 … 4.4e-3 for the audio one.
//!   MLX evaluates f32 matmul in **reduced precision on Metal**, and the sibling MiniMax-H3 DiT
//!   suite measured the same effect at ~4.2e-3.
//!
//! The floor is therefore **entirely MLX's**, and it is ~1e-3 … 4e-3. `CROSS_TOL` sits about 5×
//! above it.
//!
//! # What this comparison CANNOT detect — read before citing it as coverage
//!
//! Any divergence smaller than ~4e-3 relative is **invisible** to this gate. Concretely:
//!
//! * an RMSNorm epsilon applied outside the square root instead of inside — measured at **5.9e-6**
//!   through the whole video decoder in `video_vae_parity.rs`, roughly **three orders below** this
//!   floor. `the_cross_backend_gate_cannot_resolve_an_eps_misplacement` pins that fact with the
//!   numbers rather than leaving it as prose;
//! * a wrong-but-close activation constant (`SnakeBeta`'s `1e-9` reciprocal guard written as
//!   `1e-6`, ~1e-6 relative per activation);
//! * any sub-1% numerical drift at all.
//!
//! Those classes are covered **structurally** instead — `candle_gen_minimax_h3::nn`'s
//! `rms_norm_puts_epsilon_inside_the_square_root` and `layer_norm_puts_epsilon_inside_the_square_root`
//! pin the formulations against hand-computed closed forms — and by the committed-fixture suites,
//! whose own gates are 1e-5 (video) and 1e-5 … 1e-4 (audio). **Cross-backend agreement is evidence
//! that the two ports implement the same model; it is not evidence that either is numerically
//! exact, and it must not be cited as the latter.**
//!
//! What it *does* catch is exactly the class this epic has been bitten by five times: a layout or
//! structural divergence. The sc-18740 gated-FFN half-swap moves the decode by 1.3e-1 here — two
//! orders clear of the floor — while leaving cosine at 0.997 and the L2 norm unchanged to three
//! digits.

mod common;

use common::{
    audio_fixture_config, cosine, fixture_config, l2_norm, rel, weights, Golden, AUDIO_FIXTURE,
    FIXTURE, MLX_FIXTURE_DIR, MLX_RECORD,
};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_minimax_h3::audio_vae::{AmpBlock1, BigVgan};
use candle_gen_minimax_h3::blocks::TransformerBlock;
use candle_gen_minimax_h3::{
    swap_gated_halves, MiniMaxH3AudioVae, MiniMaxH3VideoVae, Rope3d, SnakeBeta, ViT3dDecoder,
};

/// Peak-relative bound between the two backends. Set by the MLX lane's ~1e-3 … 4e-3 Metal
/// reduced-precision floor, not by this lane's ~1e-6. See the module docs.
const CROSS_TOL: f32 = 2e-2;

/// The MLX lane's documented residual against the reference — the floor `CROSS_TOL` has to clear.
/// Video decode ~1e-3; audio decode 2.7e-3 … 4.4e-3; the sibling DiT suite measured ~4.2e-3.
const MLX_REFERENCE_RESIDUAL: f32 = 4.4e-3;

/// FNV-1a over a file's bytes, matching the generator's, so the record can be bound to the exact
/// fixture bytes it was produced from without pulling in a hash crate.
fn digest(path: &str) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn video() -> Golden {
    Golden::load(FIXTURE)
}

fn audio() -> Golden {
    Golden::load(AUDIO_FIXTURE)
}

fn record() -> Golden {
    Golden::load(MLX_RECORD)
}

// ---------------------------------------------------------------------------------------------
// The shared-golden half
// ---------------------------------------------------------------------------------------------

/// Both lanes must be held to the SAME reference tensors.
///
/// The fixtures are committed twice — once per crate — because a candle-only checkout has to be
/// able to run its own suite. Two copies can silently drift, and if they did, each suite would stay
/// green while the "both backends agree with the reference" argument quietly stopped being about
/// one reference. This is the drift guard.
#[test]
fn fixtures_are_byte_identical_to_the_mlx_lanes() {
    for name in ["video_vae_decode", "audio_vae_decode"] {
        let here = std::fs::read(format!(
            "{}/tests/fixtures/{name}.safetensors",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap_or_else(|e| panic!("read the candle copy of {name}: {e}"));
        let there = std::fs::read(format!("{MLX_FIXTURE_DIR}/{name}.safetensors"))
            .unwrap_or_else(|e| panic!("read the mlx copy of {name}: {e}"));
        assert_eq!(
            here.len(),
            there.len(),
            "{name}: the two committed copies differ in length ({} vs {}); regenerate both from \
             the same tools/dump_minimax_h3_* run",
            here.len(),
            there.len()
        );
        assert!(
            here == there,
            "{name}: the candle and mlx copies of this golden have drifted apart. Both lanes must \
             assert against the same reference bytes or the cross-backend claim is vacuous."
        );
        println!(
            "  {name}: {} bytes, byte-identical across lanes",
            here.len()
        );
    }
}

/// The MLX record must have been produced from the fixtures this crate actually holds.
///
/// Without this, regenerating a golden would leave a stale record in place and the cross-backend
/// tests would compare today's candle against yesterday's MLX — green, and meaningless.
#[test]
fn mlx_record_is_bound_to_these_fixtures() {
    let r = record();
    assert_eq!(r.meta("backend"), Some("mlx"));
    assert_eq!(r.meta("dtype"), Some("float32"));
    for (key, path) in [
        ("video_fixture_fnv1a64", FIXTURE),
        ("audio_fixture_fnv1a64", AUDIO_FIXTURE),
    ] {
        let recorded = r
            .meta(key)
            .unwrap_or_else(|| panic!("the MLX record has no `{key}`; regenerate it"));
        let actual = digest(path);
        assert_eq!(
            recorded, actual,
            "the MLX record was produced from a different {key} ({recorded} vs {actual}). \
             Re-run mlx-gen-minimax-h3's cross_backend_record generator against the current \
             fixtures."
        );
    }
    println!(
        "  MLX record bound to video {} / audio {}",
        r.meta("video_fixture_fnv1a64").unwrap_or("?"),
        r.meta("audio_fixture_fnv1a64").unwrap_or("?")
    );
}

// ---------------------------------------------------------------------------------------------
// The measured half
// ---------------------------------------------------------------------------------------------

/// Every video-side tensor the MLX lane recorded, reproduced here and compared directly.
#[test]
fn video_decode_agrees_with_the_mlx_lane() {
    let f = video();
    let r = record();
    let cfg = fixture_config(3);
    let map = f.model_map(&["src.", "in.", "out.", "const."]);
    let w = weights(map.clone());

    let rope = Rope3d::new(cfg.rope_apply_dim(), cfg.rope_theta).expect("rope");
    let tables = rope.tables(&f.tensor("in.block.ids")).expect("tables");
    let block =
        TransformerBlock::from_weights(&w, "decoder.transformer_blocks.0", &cfg, DType::F32)
            .expect("block");

    let mut checked = 0usize;
    let mut worst = 0.0f32;
    let report = |name: &str, got: &Tensor, r: &Golden, checked: &mut usize, worst: &mut f32| {
        let want = r.tensor(name);
        assert_eq!(got.dims(), want.dims(), "{name}: shape");
        let (peak, mean) = rel(got, &want);
        println!(
            "  {name}: candle-vs-mlx peak rel {peak:.3e} (mean {mean:.3e}, cosine {:.7})",
            cosine(got, &want)
        );
        assert!(
            peak < CROSS_TOL,
            "{name}: the two backends disagree by {peak:.3e}, over the {CROSS_TOL:.0e} bound"
        );
        *checked += 1;
        *worst = worst.max(peak);
    };

    report(
        "video.block.rope_cos",
        &tables.cos,
        &r,
        &mut checked,
        &mut worst,
    );
    report(
        "video.block.hidden",
        &block
            .forward(&f.tensor("in.block.hidden"), &rope, &tables)
            .expect("block forward"),
        &r,
        &mut checked,
        &mut worst,
    );

    let decoder = ViT3dDecoder::from_weights(&w, "decoder", &cfg, DType::F32).expect("vit decoder");
    report(
        "video.vit.video",
        &decoder
            .forward(&f.tensor("in.vit.latent"))
            .expect("vit forward"),
        &r,
        &mut checked,
        &mut worst,
    );

    let vae = MiniMaxH3VideoVae::from_weights(&w, &cfg, &Device::Cpu, DType::F32).expect("vae");
    for tokens in [7, 12] {
        report(
            &format!("video.temporal{tokens}.video"),
            &vae.decode(&f.tensor(&format!("in.temporal{tokens}.latent")))
                .expect("decode"),
            &r,
            &mut checked,
            &mut worst,
        );
    }

    assert_eq!(checked, 5, "every recorded video tensor must be compared");
    println!("VIDEO cross-backend worst peak-relative: {worst:.3e} (bound {CROSS_TOL:.0e})");
}

/// Every audio-side tensor the MLX lane recorded, reproduced here and compared directly.
///
/// This half is the stronger evidence of the two, because the implementations genuinely differ:
/// MLX works in NLC and transposes at the boundary, this port is NCL throughout. Agreement here is
/// not two copies of one array layout agreeing with themselves.
#[test]
fn audio_decode_agrees_with_the_mlx_lane() {
    let f = audio();
    let r = record();
    let cfg = audio_fixture_config();
    let map = f.model_map(&["in.", "out.", "const.", "amp."]);
    let w = weights(map);

    let mut checked = 0usize;
    let mut worst = 0.0f32;
    let report = |name: &str, got: &Tensor, r: &Golden, checked: &mut usize, worst: &mut f32| {
        let want = r.tensor(name);
        assert_eq!(got.dims(), want.dims(), "{name}: shape");
        let (peak, mean) = rel(got, &want);
        println!(
            "  {name}: candle-vs-mlx peak rel {peak:.3e} (mean {mean:.3e}, cosine {:.7})",
            cosine(got, &want)
        );
        assert!(
            peak < CROSS_TOL,
            "{name}: the two backends disagree by {peak:.3e}, over the {CROSS_TOL:.0e} bound"
        );
        *checked += 1;
        *worst = worst.max(peak);
    };

    let snake = SnakeBeta::new(f.tensor("in.snake.alpha"), f.tensor("in.snake.beta"), true)
        .expect("snakebeta");
    report(
        "audio.snake.log",
        &snake.forward(&f.tensor("in.snake.x")).expect("snake"),
        &r,
        &mut checked,
        &mut worst,
    );

    let amp = AmpBlock1::from_weights(
        &weights(f.prefixed_map("amp.")),
        "amp",
        7,
        &[1, 3, 5],
        true,
        DType::F32,
    )
    .expect("amp block");
    report(
        "audio.amp.y",
        &amp.forward(&f.tensor("in.amp.x")).expect("amp forward"),
        &r,
        &mut checked,
        &mut worst,
    );

    let vocoder = BigVgan::from_weights(&w, "decoder", &cfg, DType::F32).expect("vocoder");
    report(
        "audio.bigvgan.y",
        &vocoder
            .forward(&f.tensor("in.bigvgan.x"))
            .expect("bigvgan forward"),
        &r,
        &mut checked,
        &mut worst,
    );

    let vae =
        MiniMaxH3AudioVae::from_weights(&w, &cfg, &Device::Cpu, DType::F32).expect("audio vae");
    report(
        "audio.decode.audio",
        &vae.decode(&f.tensor("in.decode.z")).expect("decode"),
        &r,
        &mut checked,
        &mut worst,
    );
    let z = f.tensor("in.stereo.z");
    report(
        "audio.stereo.audio",
        &vae.decode_stereo(&vae.denormalize(&z).expect("denormalize"))
            .expect("decode_stereo"),
        &r,
        &mut checked,
        &mut worst,
    );

    assert_eq!(checked, 5, "every recorded audio tensor must be compared");
    println!("AUDIO cross-backend worst peak-relative: {worst:.3e} (bound {CROSS_TOL:.0e})");
}

// ---------------------------------------------------------------------------------------------
// The floor, measured
// ---------------------------------------------------------------------------------------------

/// **The noise floor of the comparison above, computed from committed data rather than quoted.**
///
/// The MLX record and the reference golden are both committed here, so this measures MLX's own
/// residual against the reference directly — and then measures candle's, and shows which of the two
/// sets the bound. If the answer ever becomes "candle's", the module docs are wrong and this fails.
#[test]
fn the_cross_backend_floor_is_the_mlx_lanes_own_reduced_precision() {
    let f = video();
    let r = record();
    let cfg = fixture_config(3);
    let w = weights(f.model_map(&["src.", "in.", "out.", "const."]));

    let reference = f.tensor("out.vit.video");
    let mlx = r.tensor("video.vit.video");
    let candle = ViT3dDecoder::from_weights(&w, "decoder", &cfg, DType::F32)
        .expect("vit decoder")
        .forward(&f.tensor("in.vit.latent"))
        .expect("forward");

    let (mlx_vs_ref, _) = rel(&mlx, &reference);
    let (candle_vs_ref, _) = rel(&candle, &reference);
    let (candle_vs_mlx, _) = rel(&candle, &mlx);
    println!(
        "FLOOR (ViT3DDecoder): mlx-vs-reference {mlx_vs_ref:.3e}, candle-vs-reference \
         {candle_vs_ref:.3e}, candle-vs-mlx {candle_vs_mlx:.3e}; bound {CROSS_TOL:.0e}"
    );

    assert!(
        mlx_vs_ref > candle_vs_ref * 10.0,
        "the MLX lane's residual ({mlx_vs_ref:.3e}) is no longer an order above candle's \
         ({candle_vs_ref:.3e}); the module docs attribute the whole cross-backend floor to Metal's \
         reduced-precision matmul and would need re-stating"
    );
    assert!(
        mlx_vs_ref < MLX_REFERENCE_RESIDUAL,
        "the MLX lane's residual ({mlx_vs_ref:.3e}) exceeds its own documented {MLX_REFERENCE_RESIDUAL:.1e}"
    );
    assert!(
        CROSS_TOL > mlx_vs_ref * 2.0,
        "the cross-backend bound {CROSS_TOL:.0e} does not clear the measured floor \
         {mlx_vs_ref:.3e} with margin"
    );
    // The two backends must agree BETTER than either agrees with the reference is required to —
    // otherwise the bound is being met by luck rather than by the ports matching.
    assert!(
        candle_vs_mlx < CROSS_TOL,
        "candle and mlx disagree by {candle_vs_mlx:.3e}"
    );
}

/// **What this gate cannot see**, pinned with numbers.
///
/// An RMSNorm epsilon applied outside the square root rather than inside moves the whole video
/// decode by ~5.9e-6 (measured in `video_vae_parity.rs`). This test recomputes that displacement
/// and asserts it is far *below* the cross-backend floor — i.e. that the comparison above would
/// stay green on a port with that defect.
///
/// The assertion is deliberately in that direction. It is not a wish that the gate stay weak; it is
/// a pin on a claim the module docs make. If this ever fails, the MLX lane's precision improved by
/// orders and the "cannot detect sub-1% divergence" statement must be rewritten rather than
/// silently left standing.
///
/// The class itself is covered by `candle_gen_minimax_h3::nn`'s
/// `rms_norm_puts_epsilon_inside_the_square_root`, which pins the formulation structurally.
#[test]
fn the_cross_backend_gate_cannot_resolve_an_eps_misplacement() {
    let f = video();
    let r = record();
    let w = weights(f.model_map(&["src.", "in.", "out.", "const."]));

    let baseline = ViT3dDecoder::from_weights(&w, "decoder", &fixture_config(3), DType::F32)
        .expect("vit decoder")
        .forward(&f.tensor("in.vit.latent"))
        .expect("forward");

    // `norm_eps = 3e-5` displaces the norms by the ~2e-5 an eps-outside-the-root would at 1e-5.
    let mut cfg = fixture_config(3);
    cfg.norm_eps = 3e-5;
    let displaced = ViT3dDecoder::from_weights(&w, "decoder", &cfg, DType::F32)
        .expect("vit decoder")
        .forward(&f.tensor("in.vit.latent"))
        .expect("forward");

    let (eps_effect, _) = rel(&displaced, &baseline);
    let (floor, _) = rel(&r.tensor("video.vit.video"), &f.tensor("out.vit.video"));
    println!(
        "BLIND SPOT: an eps-outside-sqrt-sized displacement moves the decode by {eps_effect:.3e}; \
         the cross-backend floor is {floor:.3e} and the bound is {CROSS_TOL:.0e} — the \
         displacement is {:.0}x SMALLER than the floor",
        floor / eps_effect
    );
    assert!(
        eps_effect * 10.0 < floor,
        "an eps misplacement ({eps_effect:.3e}) is now within an order of the cross-backend floor \
         ({floor:.3e}). That is an improvement, but this file's docs claim the comparison cannot \
         resolve sub-1% divergences — re-state them before re-greening this."
    );
}

/// The other direction: a **layout** error is loud here, which is what makes this gate worth
/// having at all.
///
/// The sc-18740 gated-FFN half-swap is the exact defect that shipped past a fully green MLX parity
/// suite. Against the MLX record it diverges by two orders more than the floor — while leaving the
/// cosine near 1 and the L2 norm unchanged to three digits, which is why no magnitude, cosine or
/// checksum gate anywhere in this epic caught it.
#[test]
fn a_gated_ffn_half_swap_is_loud_against_the_mlx_record() {
    let f = video();
    let r = record();
    let cfg = fixture_config(3);
    let want = r.tensor("video.temporal7.video");

    let mut map = f.model_map(&["src.", "in.", "out.", "const."]);
    for block in 0..cfg.num_layers {
        for suffix in ["weight", "bias"] {
            let key = format!("decoder.transformer_blocks.{block}.ff.net.0.proj.{suffix}");
            let swapped = swap_gated_halves(&map[&key]).expect("swap");
            map.insert(key, swapped);
        }
    }
    let mutated = MiniMaxH3VideoVae::from_weights(&weights(map), &cfg, &Device::Cpu, DType::F32)
        .expect("vae")
        .decode(&f.tensor("in.temporal7.latent"))
        .expect("decode");

    let (peak, mean) = rel(&mutated, &want);
    println!(
        "sc-18740 half-swap vs the MLX record: peak rel {peak:.3e} (mean {mean:.3e}), cosine \
         {:.6}, ||mlx||={:.4} ||swapped||={:.4}; bound {CROSS_TOL:.0e}",
        cosine(&mutated, &want),
        l2_norm(&want),
        l2_norm(&mutated),
    );
    assert!(
        peak > CROSS_TOL * 5.0,
        "the half-swap moved the decode by only {peak:.3e}; this cross-backend gate would not \
         catch the sc-18740 defect and should not be cited as covering it"
    );
}
