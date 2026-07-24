//! sc-14104 — Gaussian-Shading watermarked initial noise + detection, against the vendored torch
//! reference (`_vendor/mage_flow/models/modules/mage_latent.py`).
//!
//! The reference *discards* its plain `randn` and starts the denoise loop from
//! `encode_noise(shape, key, seed)` on both the generation (`pipeline.py:307`) and edit (`:506`)
//! paths, with no toggle. So this is a correctness gate, not a nice-to-have: sc-14036 measured the
//! watermarked latent and the discarded `randn` **5.99 apart** in max_abs, which means a port that
//! got the RNG wrong would be loudly, not subtly, wrong at token 0.
//!
//! Two tiers, deliberately:
//!
//! * **Always-on** — every expected value below was captured from the pinned reference environment
//!   (`_vendor/mage_flow/requirements.txt`: numpy 2.4.3, torch 2.13.0) and is committed here, so CI
//!   exercises the parity claim even though the golden bundle is gitignored.
//! * **`#[ignore]`** — [`encode_noise_matches_the_committed_golden`] checks the whole 32 768-entry
//!   production tensor against `tools/golden/mage_flow_noise_golden.safetensors`. Regenerate with
//!   `MAGE_DEVICE=cpu … tools/dump_mage_flow_golden.py --stage noise` (sc-14250: an MPS dump is
//!   silently corrupt), then:
//!   `cargo test -p mlx-gen-mage --test gaussian_shading_parity -- --ignored --nocapture`

use std::path::Path;

use mlx_gen::Result;
use mlx_gen_mage::config::{GS_DEFAULT_KEY, GS_MESSAGE_BITS};
use mlx_gen_mage::latent::{
    decode_bits, decode_bits_host, encode_noise, encode_noise_host, invert_to_noise, message_bits,
    pad_and_pos, GsKey,
};
use mlx_rs::{Array, Dtype};

/// The golden's fixed configuration (`tools/dump_mage_flow_golden.py:102,116-121`).
const GOLDEN_SEED: i64 = 42;
/// 256²/16 = 16, and Mage-VAE latents are 128-channel.
const GOLDEN_SHAPE: (usize, usize, usize) = (128, 16, 16);
const GOLDEN_N: usize = 128 * 16 * 16;

const GOLDEN_FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mage_flow_noise_golden.safetensors"
);

// =================================================================================================
// (a) `encode_noise` reproduces the torch initial-noise tensor
// =================================================================================================

/// `encode_noise((8, 2, 2), key=20260720, seed=42, dtype=float32)` — the golden's `gs_noise_tiny`,
/// which is also the head of the production tensor's uniform stream.
#[rustfmt::skip]
const TINY_F32: [f32; 32] = [
    -1.894_531_6, -1.859_825_8, -1.539_894_5, -1.938_353_1, 0.716_264_4, 0.638_397_04,
    -0.056_133_196, -0.089_414_395, 0.104_898_15, -1.503_769_8, -1.415_051_1, 0.489_294_86,
    -0.198_671_49, -0.163_022_09, 0.492_356_93, 0.864_044_55, 0.110_654_645, 1.038_971_3,
    0.884_183_94, -0.776_796_2, -1.782_088, 1.001_531_8, 0.401_895, -1.017_199,
    -2.152_726_7, -2.049_979_2, -0.085_443_53, 0.191_831_53, 0.338_541_6, -1.512_394_7,
    -1.145_854_8, 0.296_835_87,
];

/// `encode_noise((8, 2, 2), …, dtype=bfloat16)` — the dtype the denoise loop actually receives
/// (`pipeline.py:307-308`), read back as f32.
#[rustfmt::skip]
const TINY_BF16: [f32; 32] = [
    -1.898_437_5, -1.859_375, -1.539_062_5, -1.937_5, 0.714_843_75, 0.636_718_75,
    -0.056_152_344, -0.089_355_47, 0.104_980_47, -1.5, -1.414_062_5, 0.490_234_38,
    -0.198_242_19, -0.163_085_94, 0.492_187_5, 0.863_281_25, 0.110_839_844, 1.039_062_5,
    0.882_812_5, -0.777_343_75, -1.781_25, 1.0, 0.402_343_75, -1.015_625,
    -2.156_25, -2.046_875, -0.085_449_22, 0.191_406_25, 0.337_890_63, -1.515_625,
    -1.148_437_5, 0.296_875,
];

/// Scattered exact values from the **production** tensor `(128, 16, 16)`, key 20260720, seed 42.
/// The tail entries are the discriminating ones: they depend on 32 767 correct prior uniform draws
/// *and* on the index map having started after 32 768 pad draws rather than after 32.
#[rustfmt::skip]
const FULL_F32: [(usize, f32); 13] = [
    (0, -1.894_531_6), (1, 0.078_928_076), (2, 0.155_516_77), (3, 0.065_947_235),
    (31, 0.296_835_87), (32, -1.211_601_1), (100, -0.902_157_25), (1_000, -1.431_651_2),
    (4_095, 0.012_590_552), (16_384, 0.292_654_8), (32_765, 1.149_232_1),
    (32_766, -0.734_464_65), (32_767, 0.554_717_2),
];

#[test]
fn encode_noise_reproduces_the_torch_tensor_bit_for_bit() {
    let key = GsKey::default();
    assert_eq!(key, GsKey::from_u64(GS_DEFAULT_KEY));

    let tiny = encode_noise_host((8, 2, 2), &key, GOLDEN_SEED).unwrap();
    let tiny_f32: Vec<f32> = tiny.iter().map(|&v| v as f32).collect();
    assert_eq!(tiny_f32, TINY_F32.to_vec(), "tiny (8,2,2) tensor");

    let full = encode_noise_host(GOLDEN_SHAPE, &key, GOLDEN_SEED).unwrap();
    assert_eq!(full.len(), GOLDEN_N);
    for (i, want) in FULL_F32 {
        assert_eq!(full[i] as f32, want, "production tensor entry {i}");
    }

    // The reference's own summary statistics for this tensor, so a wholesale reordering that
    // happened to preserve the thirteen sampled indices still fails. Tolerance is 1e-8, not 0:
    // torch reduces with pairwise summation and this loop is sequential, which disagree at ~1e-11
    // over 32 768 terms. The per-entry equalities above are the bit-exactness claim; this is the
    // whole-tensor cross-check.
    let mean = full.iter().sum::<f64>() / GOLDEN_N as f64;
    let var = full.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (GOLDEN_N as f64 - 1.0);
    assert!(
        (mean - 0.005_076_203_061_018_691_5).abs() < 1e-8,
        "mean {mean}"
    );
    assert!(
        (var.sqrt() - 1.003_521_587_810_495_7).abs() < 1e-8,
        "std {}",
        var.sqrt()
    );
    // Exact extrema (permutation-invariant, but free and reorder-proof against truncation).
    let (min, max) = full
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert_eq!(min as f32, -3.959_779);
    assert_eq!(max as f32, 4.147_058_5);
}

/// The pinned values above only prove something if a wrong-but-plausible implementation FAILS them.
/// Each variant here is a mistake a careful porter could realistically make, driven through the
/// public API and shown to produce a different tensor.
#[test]
fn the_pinned_values_discriminate_against_plausible_mistakes() {
    let key = GsKey::default();
    let good = encode_noise_host((8, 2, 2), &key, GOLDEN_SEED).unwrap();
    let good_f32: Vec<f32> = good.iter().map(|&v| v as f32).collect();
    assert_eq!(good_f32, TINY_F32.to_vec());

    let differs = |other: &[f64]| {
        let other: Vec<f32> = other.iter().map(|&v| v as f32).collect();
        other != TINY_F32.to_vec()
    };

    // 1. A neighbouring key — i.e. the env/keyfile fallback resolving to anything else at all.
    assert!(differs(
        &encode_noise_host((8, 2, 2), &GsKey::from_u64(20_260_721), GOLDEN_SEED).unwrap()
    ));
    // 2. Treating the all-digits key as a passphrase instead of an integer (`_key_to_int`'s branch).
    assert!(differs(
        &encode_noise_host((8, 2, 2), &GsKey::from_passphrase("20260720"), GOLDEN_SEED).unwrap()
    ));
    // 3. A neighbouring seed.
    assert!(differs(
        &encode_noise_host((8, 2, 2), &key, GOLDEN_SEED + 1).unwrap()
    ));
    // 4. The index map is drawn AFTER `n` pad values, so it is length-dependent: deriving the
    //    32-entry tensor as a prefix of the production one is wrong, and diverges by entry 1.
    let full = encode_noise_host(GOLDEN_SHAPE, &key, GOLDEN_SEED).unwrap();
    assert_ne!(full[1] as f32, good_f32[1]);
    let (_, pos_small) = pad_and_pos(32, &key, GS_MESSAGE_BITS).unwrap();
    let (_, pos_full) = pad_and_pos(GOLDEN_N, &key, GS_MESSAGE_BITS).unwrap();
    assert_ne!(pos_small[..], pos_full[..32]);
    // 5. The payload expansion is bit-order-sensitive: MSB-first packing gives a different message,
    //    which would relabel every entry's half.
    let msg = message_bits();
    let msb_first: Vec<u8> = msg
        .chunks_exact(8)
        .flat_map(|c| c.iter().rev().copied())
        .collect();
    assert_ne!(msb_first, msg);
}

#[test]
fn encode_noise_lifts_to_mlx_at_both_pipeline_dtypes() {
    let key = GsKey::default();

    let f32_arr = encode_noise((8, 2, 2), &key, GOLDEN_SEED, Dtype::Float32).unwrap();
    assert_eq!(f32_arr.shape(), &[1, 8, 2, 2]);
    assert_eq!(f32_arr.dtype(), Dtype::Float32);
    assert_eq!(host_f32(&f32_arr), TINY_F32.to_vec());

    // bfloat16 is what `pipeline.py:307-308` feeds the denoise loop, so the e2e golden's first
    // trajectory step is comparable to THIS tensor, not the float32 one.
    let bf16_arr = encode_noise((8, 2, 2), &key, GOLDEN_SEED, Dtype::Bfloat16).unwrap();
    assert_eq!(bf16_arr.dtype(), Dtype::Bfloat16);
    let got = host_f32(&bf16_arr);
    assert_eq!(
        got,
        TINY_BF16.to_vec(),
        "MLX's float32→bfloat16 rounding must match torch's"
    );
}

/// The `clamp(1e-6, 1 − 1e-6)` on the inverse-normal-CDF argument (`mage_latent.py:86`) is
/// **reachable in production and unreachable in the golden** — the same trap class as the
/// `TXT_MAX_LENGTH` gap recorded on sc-14037: a golden whose inputs are too small to exercise a
/// guard. Measured against the real torch reference at seed 42:
///
/// | geometry | latent entries | entries that clamp |
/// |---|---|---|
/// | 256² (the golden) | 32 768 | **0** — min argument 3.75e-5, 37× clear of the bound |
/// | 1024² (the epic default) | 524 288 | **2** (high side) |
/// | 2048² (the native-res cap) | 2 097 152 | **5** (2 low, 3 high) |
///
/// So the committed golden structurally cannot catch a clamp regression at the resolutions we
/// actually target, and without this test both widening the bound to `1e-5` and deleting the clamp
/// outright pass the whole suite. Each pinned entry below is a value the reference produces *only*
/// with the `1e-6` bound in place.
#[test]
fn the_argument_clamp_binds_at_production_resolutions() {
    let key = GsKey::default();

    // Low side, cheaply: at the golden's own 256² geometry, seed 20 drives one argument to
    // 2.2530e-9. Clamped it is Φ⁻¹(1e-6) = −4.753424; unclamped it would be −5.864458, and under a
    // 1e-5 bound −4.264891.
    let low = encode_noise_host(GOLDEN_SHAPE, &key, 20).unwrap();
    assert_eq!(low[5_833] as f32, -4.753_424, "low clamp at 256², seed 20");

    // High side at the epic's default 1024². Two entries with *different* arguments
    // (1 − 2.238e-7 and 1 − 8.835e-7) collapse to the same Φ⁻¹(1 − 1e-6) — that coincidence IS the
    // clamp's signature. Unclamped they would be 5.047527 and 4.778405.
    let high = encode_noise_host((128, 64, 64), &key, GOLDEN_SEED).unwrap();
    assert_eq!(high[252_919] as f32, 4.753_424, "high clamp #1 at 1024²");
    assert_eq!(high[500_143] as f32, 4.753_424, "high clamp #2 at 1024²");
    assert_eq!(
        high[252_919], high[500_143],
        "both pinned to the same bound"
    );

    // Ordinary entries of the same tensor, so this also pins the 524 288-length draw sequence
    // rather than only the two exceptional values.
    assert_eq!(high[262_144] as f32, 0.323_487_46);
    assert_eq!(high[524_287] as f32, -1.639_861_7);

    // ...and the golden's own geometry never reaches the bound, which is exactly why it is blind
    // here: a clamped entry is |4.753424|, and the golden's extrema are 4.147059 / −3.959779.
    let golden = encode_noise_host(GOLDEN_SHAPE, &key, GOLDEN_SEED).unwrap();
    assert!(golden.iter().all(|v| v.abs() < 4.75));
}

// =================================================================================================
// (b) detection: `invert_to_noise` + `decode_bits`
// =================================================================================================

/// A `steps`-entry Mage-Flow sigma ladder plus the terminal zero: `FlowMatchEulerDiscreteScheduler`
/// with `shift = 6.0` over `linspace(1, 1/N, N)` (`pipeline.py:37-50`, GAP 4 on sc-14036).
///
/// Test-local on purpose — the production scheduler is sc-14041's, and [`invert_to_noise`] takes the
/// ladder as an argument precisely so detection does not have to wait for it.
fn sigma_ladder(steps: usize) -> Vec<f32> {
    assert!(steps >= 2, "linspace needs at least two points");
    let n = steps as f64;
    let mut out: Vec<f32> = (0..steps)
        .map(|i| {
            let raw = 1.0 - (i as f64) * (1.0 - 1.0 / n) / (n - 1.0);
            (6.0 * raw / (1.0 + 5.0 * raw)) as f32
        })
        .collect();
    out.push(0.0);
    out
}

#[test]
fn the_sigma_ladder_helper_matches_the_reference_four_step_schedule() {
    // Independently recomputed in review on sc-14036 and pinned in the epic.
    let want = [1.0, 0.947_368_44, 0.857_142_87, 0.666_666_7, 0.0];
    for (got, want) in sigma_ladder(4).into_iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

/// **The (b) gate, with its boundary stated.** A watermarked latent is pushed forward through a
/// rectified-flow ODE to a clean latent, then recovered with [`invert_to_noise`] and read with
/// [`decode_bits`].
///
/// The velocity field here is a **synthetic affine flow**, not the real 12-block NR-MMDiT: the DiT
/// is sc-14040 and the assembled pipeline is sc-14041, neither of which exists yet. What that means
/// concretely:
///
/// * **Covered** — the reverse-Euler integration, the token↔latent layout round trip, the dtype
///   contract, the sign-vote decoder, the majority-vote message recovery, and the fact that the
///   watermark survives a *lossy* inversion rather than only a pristine one. The field is strongly
///   state-dependent (`κ = 1`), so the "evaluate the velocity at the point in hand" proxy the
///   reference uses is genuinely approximate here — the recovered noise is **not** the original.
/// * **NOT covered** — whether the real model's inversion error is small enough. That is a
///   real-weight question and belongs to sc-14041; this test cannot and does not answer it.
///
/// The control below is framed as *payload recovery*, not as "absent". A synthetic **linear** flow
/// necessarily leaves the clean latent partly sign-correlated with ε (measured: raw_acc 0.55,
/// z 8.97), because a linear map cannot scramble signs the way a 12-block nonlinear DiT does. What
/// the inversion changes is categorical: the 256-bit payload comes back exactly only after it. The
/// crisp "reads as absent" controls — plain `randn`, and the wrong key — live in
/// [`detection_rejects_unwatermarked_noise_and_the_wrong_key`], where they are not confounded by a
/// linear flow.
#[test]
fn the_watermark_survives_a_flow_ode_round_trip() {
    let key = GsKey::default();
    let shape = (128usize, 8usize, 8usize);
    let (c, gh, gw) = (shape.0 as i32, shape.1 as i32, shape.2 as i32);
    let tokens = gh * gw;
    let n = shape.0 * shape.1 * shape.2;

    let eps = encode_noise(shape, &key, GOLDEN_SEED, Dtype::Bfloat16).unwrap();
    let eps_tokens = eps
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap()
        .reshape(&[1, tokens, c])
        .unwrap();

    // v(x, σ) = κ·σ·x + drift·(0.5 + σ). Deterministic and elementwise, but **σ-dependent on both
    // terms** — that is deliberate and load-bearing. A velocity that ignored σ would make the
    // integrator blind to its own two most error-prone details: traversing the ladder in the wrong
    // direction, and perturbing Δσ, would both cancel out and the whole suite would still pass.
    // The drift also dominates ε, so the "no inversion" control below is not trivially detecting
    // through the linearity.
    const KAPPA: f32 = 1.0;
    let drift_host: Vec<f32> = (0..n)
        .map(|i| 3.0 * ((i as f32) * 0.7).sin() + 1.5 * ((i as f32) * 0.13).cos())
        .collect();
    let drift = Array::from_slice(&drift_host, &[1, tokens, c]);
    let velocity = |x: &Array, sigma: f32| -> Result<Array> {
        let a = Array::from_slice(&[KAPPA * sigma], &[1]);
        let b = Array::from_slice(&[0.5 + sigma], &[1]);
        // Computed in float32 and returned at bfloat16, like a real model forward.
        Ok(x.as_dtype(Dtype::Float32)?
            .multiply(&a)?
            .add(drift.multiply(&b)?)?
            .as_dtype(Dtype::Bfloat16)?)
    };

    let sigmas = sigma_ladder(30);

    // Forward: x_{i+1} = x_i + (σ_{i+1} − σ_i)·v(x_i, σ_i), exactly what the sampler does.
    let mut x = eps_tokens.clone();
    for si in 0..sigmas.len() - 1 {
        let d = Array::from_slice(&[sigmas[si + 1] - sigmas[si]], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let v = velocity(&x, sigmas[si]).unwrap();
        x = x
            .add(v.multiply(&d).unwrap())
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
    }
    let clean = x
        .reshape(&[1, gh, gw, c])
        .unwrap()
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();

    let recovered = invert_to_noise(&clean, &sigmas, velocity).unwrap();
    assert_eq!(recovered.shape(), &[1, c, gh, gw]);
    assert_eq!(recovered.dtype(), Dtype::Float32);

    // The inversion really is lossy — otherwise this test would be checking nothing but algebra.
    let eps_f32 = host_f32(&eps);
    let rec_f32 = host_f32(&recovered);
    let max_abs = eps_f32
        .iter()
        .zip(&rec_f32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs > 1e-3,
        "the synthetic flow must leave a real inversion residual, got max_abs {max_abs}"
    );

    // Pin the recovered TENSOR, not only the decode statistics. The decode is a sign vote with
    // enormous redundancy, so it survives large perturbations of the integrator: without these
    // equalities, reversing the ladder or rounding Δσ to bfloat16 before the multiply (the ~0.4%
    // per-step error fixed in 041e208f) both leave every statistical assertion above satisfied.
    // These values come from this implementation, so they are a change-detector rather than a
    // parity oracle — the parity oracle for the integrator itself is sc-14041, with the real model.
    // Every value is bfloat16-exact, so they are not sensitive to float accumulation order.
    #[rustfmt::skip]
    const RECOVERED: [(usize, f32); 8] = [
        (0, -1.921_875), (1, 0.011_840_82), (2, 0.192_382_81), (777, -1.281_25),
        (4_095, -0.065_917_97), (6_000, 0.890_625), (8_190, 0.906_25), (8_191, 1.968_75),
    ];
    for (i, want) in RECOVERED {
        assert_eq!(rec_f32[i], want, "recovered noise entry {i}");
    }
    assert!(
        (max_abs - 0.222_656_25).abs() < 1e-6,
        "inversion residual moved: {max_abs}"
    );

    let report = decode_bits(&recovered, &key).unwrap();
    println!(
        "round trip: inversion max_abs {max_abs:.4}, raw_acc {:.4}, msg_acc {:.4}, z {:.2}",
        report.raw_acc, report.msg_acc, report.z_score
    );
    assert!(report.present, "watermark must be detected: {report:?}");
    assert_eq!(report.n, n);
    assert_eq!(
        report.msg_hat, report.msg,
        "the 256-bit payload must be recovered exactly"
    );
    assert!(report.raw_acc > 0.9, "raw_acc {}", report.raw_acc);

    // Control: the CLEAN latent, read without inverting. The payload must NOT come back, and the
    // statistic must be far weaker — that is what makes the assertions above about
    // `invert_to_noise` rather than about `decode_bits` alone. See the doc comment for why this is
    // a margin rather than an "absent" claim under a linear flow.
    let without_inversion = decode_bits(&clean, &key).unwrap();
    println!(
        "no inversion: raw_acc {:.4}, msg_acc {:.4}, z {:.2}",
        without_inversion.raw_acc, without_inversion.msg_acc, without_inversion.z_score
    );
    assert_ne!(
        without_inversion.msg_hat, without_inversion.msg,
        "the payload must be recoverable only after inversion"
    );
    assert!(without_inversion.msg_acc < 0.85);
    assert!(without_inversion.raw_acc < 0.6);
    assert!(
        report.z_score > 5.0 * without_inversion.z_score,
        "inversion must dominate: z {} vs {}",
        report.z_score,
        without_inversion.z_score
    );
}

#[test]
fn detection_rejects_unwatermarked_noise_and_the_wrong_key() {
    let key = GsKey::default();
    let n = 8192;

    // A plain standard normal built host-side (Box–Muller over a deterministic LCG) — the "plain
    // randn" the reference computes and throws away.
    let mut state: u64 = 0x2026_0720_0000_002A;
    let mut plain = Vec::with_capacity(n);
    while plain.len() < n {
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let (u1, u2) = (next().max(1e-12), next());
        let r = (-2.0 * u1.ln()).sqrt();
        plain.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        plain.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
    }
    plain.truncate(n);

    let report = decode_bits_host(&plain, &key).unwrap();
    println!(
        "plain randn: raw_acc {:.4}, z {:.2}, p {:.3e}",
        report.raw_acc, report.z_score, report.p_value
    );
    assert!(!report.present, "plain randn must not detect: {report:?}");
    assert!((report.raw_acc - 0.5).abs() < 0.05);

    // And a genuinely watermarked latent read with the wrong key is equally silent.
    let watermarked = encode_noise_host((128, 8, 8), &key, GOLDEN_SEED).unwrap();
    let host: Vec<f32> = watermarked.iter().map(|&v| v as f32).collect();
    let wrong = decode_bits_host(&host, &GsKey::from_passphrase("not the key")).unwrap();
    assert!(!wrong.present, "a wrong key must not detect: {wrong:?}");
    // ...while the right key is unmistakable: all n entries match, so z = √n.
    let right = decode_bits_host(&host, &key).unwrap();
    assert_eq!(right.matches, host.len());
    assert!((right.z_score - (host.len() as f64).sqrt()).abs() < 1e-9);
    assert_eq!(right.p_value, 0.0, "z ≈ 90 underflows the p-value to zero");
    assert!(right.present);
}

/// numpy's `argmax` returns the **first** maximum, so a message bit with no votes — or with tied
/// votes — resolves to 0, never 1. `decode_bits` implements that as `v[1] > v[0]`; the
/// natural-looking `>=` is a *different decoder*, and every other test in this file tolerates it
/// because they run at redundancies where no bit goes unvoted.
///
/// At n = 256 the redundancy is one vote per bit on average and **91 of the 256 bits go unvoted**,
/// where the two rules disagree on all 91 (measured against the vendored `decode_bits`). Pinning
/// `msg_hat` exactly here is what turns the tie rule from a comment into a tested decision.
#[test]
fn the_tie_rule_follows_numpys_first_maximum() {
    let key = GsKey::default();
    let host: Vec<f32> = encode_noise_host((4, 8, 8), &key, GOLDEN_SEED)
        .unwrap()
        .iter()
        .map(|&v| v as f32)
        .collect();
    let report = decode_bits_host(&host, &key).unwrap();
    assert_eq!(report.n, 256);
    // Every *sign* matches — the latent is pristine. The message is nevertheless only partly
    // recoverable, because recovery needs coverage, not accuracy.
    assert_eq!(report.raw_acc, 1.0);
    assert!(
        (report.msg_acc - 206.0 / 256.0).abs() < 1e-12,
        "msg_acc {} (the vendored decoder reports 206/256 here)",
        report.msg_acc
    );
    assert_ne!(report.msg_hat, report.msg);

    // `mage_flow.models.modules.mage_latent.decode_bits(...)["msg_hat"]`, packed LSB-first per byte.
    #[rustfmt::skip]
    const MSG_HAT: [u8; 32] = [
        0x9c, 0xb0, 0x82, 0x11, 0x90, 0x18, 0x60, 0xd1, 0x34, 0x01, 0x90, 0x43, 0xe4, 0x78, 0x2c,
        0x7c, 0x01, 0x42, 0xe1, 0xb6, 0xb0, 0x29, 0x48, 0x67, 0x22, 0x24, 0x00, 0xb2, 0x00, 0x15,
        0x00, 0xee,
    ];
    let want: Vec<u8> = (0..GS_MESSAGE_BITS)
        .map(|i| (MSG_HAT[i / 8] >> (i % 8)) & 1)
        .collect();
    assert_eq!(
        report.msg_hat, want,
        "tie rule: unvoted bits must resolve to 0"
    );

    // With real redundancy the payload does come back exactly, so this is a coverage property of
    // the size, not a defect of the decoder.
    let big: Vec<f32> = encode_noise_host((128, 16, 16), &key, GOLDEN_SEED)
        .unwrap()
        .iter()
        .map(|&v| v as f32)
        .collect();
    let big_report = decode_bits_host(&big, &key).unwrap();
    assert_eq!(big_report.msg_hat, big_report.msg);
}

#[test]
fn invert_to_noise_rejects_malformed_inputs() {
    let sigmas = sigma_ladder(4);
    let identity = |x: &Array, _s: f32| -> Result<Array> { Ok(x.clone()) };

    let bad_rank = Array::from_slice(&[0.0f32; 8], &[8]);
    assert!(invert_to_noise(&bad_rank, &sigmas, identity).is_err());

    let batched = Array::from_slice(&[0.0f32; 16], &[2, 2, 2, 2]);
    assert!(invert_to_noise(&batched, &sigmas, identity).is_err());

    let ok = Array::from_slice(&[0.0f32; 16], &[1, 4, 2, 2]);
    assert!(invert_to_noise(&ok, &[1.0], identity).is_err());

    // A model returning the wrong shape is a typed error, not a silent broadcast.
    let wrong_shape =
        |_: &Array, _: f32| -> Result<Array> { Ok(Array::from_slice(&[0.0f32], &[1])) };
    assert!(invert_to_noise(&ok, &sigmas, wrong_shape).is_err());
}

// =================================================================================================
// Guardrails
// =================================================================================================

/// The reference resolves its key from `MAGEFLOW_GS_KEY` / `~/.mageflow/gs_key`
/// (`mage_latent.py:13-15`). Porting that would trip the epic-13657 guardrail in
/// `scripts/check-workspace.py`, which forbids production env side channels and path derivation in
/// this repository — so the key arrives through [`mlx_gen_mage::latent::resolve_gs_key`] instead.
/// This is the file that would have grown the side channel, so the ban is asserted, not assumed —
/// and it is the *only* thing standing there: `check-workspace.py`'s `DELETED_ENV_SIDE_CHANNELS`
/// list does not know about `MAGEFLOW_GS_KEY`, since the variable was never in this repository.
///
/// Needle-based, so it is a tripwire rather than a proof: it catches the realistic regression
/// (someone reaching for the reference's `expanduser` + env pattern) and any filesystem or process
/// -environment access at all, but a hardcoded literal path would still slip through. Widening it
/// further would start false-positiving on the prose above, which names both side channels
/// deliberately.
#[test]
fn the_watermark_module_reads_no_env_var_and_derives_no_path() {
    let src = include_str!("../src/latent.rs");
    for needle in [
        // process environment
        concat!("env", "::", "var"),
        concat!("std::", "env"),
        concat!("var", "_os"),
        concat!("get", "env"),
        // home / config directory derivation
        concat!("home_", "dir"),
        concat!("dirs", "::"),
        concat!("expand", "user"),
        // filesystem access of any kind
        concat!("std::", "fs"),
        concat!("Path", "Buf"),
        concat!("Path", "::new"),
        concat!("File", "::open"),
        concat!("include_", "str!"),
    ] {
        assert!(
            !src.contains(needle),
            "src/latent.rs must not contain `{needle}` — epic-13657 self-fetch boundary"
        );
    }
}

/// The Cephes `ndtri`/`erf`/`erfc` port carries Moshier's coefficient tables verbatim and **ships**
/// in binary bundles, which BSD-3 clause 2 makes a distribution obligation. Nothing in CI enforces
/// `NOTICE` — `cargo deny check licenses` only inspects Cargo dependencies, and no workflow or
/// script references the file — so this test is the enforcement point.
///
/// It also guards the wording: the crate NOTICE used to state flatly that *no* source code is
/// copied into the shipped Rust crates, which this port made false.
#[test]
fn the_cephes_port_is_attributed_in_the_crate_notice() {
    let notice = include_str!("../../NOTICE");
    for needle in [
        "Ported third-party source (IN the shipped Rust crates)",
        "Cephes Math Library Release 2.8",
        "Stephen L. Moshier",
        "BSD 3-Clause",
        "mlx-gen-mage/src/latent.rs",
        "DOES ship in binary bundles",
    ] {
        assert!(
            notice.contains(needle),
            "crates/media/mlx-gen/NOTICE must record `{needle}` for the Cephes port"
        );
    }
    // The blanket "nothing is copied" claim must stay qualified.
    let blanket =
        "No source code from the\nprojects listed below is copied into the shipped Rust crates";
    assert!(
        !notice.contains(blanket) || notice.contains("The one exception"),
        "the NOTICE's no-source-copied claim must name the Cephes exception"
    );
}

// =================================================================================================
// The full committed golden (needs the gitignored bundle)
// =================================================================================================

/// A minimal safetensors reader.
///
/// `mlx_gen::weights::Weights` cannot open this bundle: it stores the reference's detector
/// statistics as **float64**, and MLX has no float64 dtype, so `load_safetensors` rejects the whole
/// file (`[safetensor] unsupported dtype F64`). Rather than drop those tensors — they are the
/// reference's own answer for `raw_acc`/`msg_acc`/`z_score` and the only cross-check on the decoder
/// — the header is parsed directly here.
struct Golden {
    header: serde_json::Value,
    bytes: Vec<u8>,
    body: usize,
}

impl Golden {
    fn open(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
        let header = serde_json::from_slice(&bytes[8..8 + len]).expect("safetensors header");
        Ok(Self {
            header,
            bytes,
            body: 8 + len,
        })
    }

    fn raw(&self, name: &str) -> (&str, &[u8]) {
        let entry = self
            .header
            .get(name)
            .unwrap_or_else(|| panic!("{name} absent"));
        let dtype = entry["dtype"].as_str().expect("dtype");
        let start = entry["data_offsets"][0].as_u64().expect("start") as usize;
        let end = entry["data_offsets"][1].as_u64().expect("end") as usize;
        (dtype, &self.bytes[self.body + start..self.body + end])
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        let (dtype, raw) = self.raw(name);
        assert_eq!(dtype, "F32", "{name}");
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect()
    }

    fn f64s(&self, name: &str) -> Vec<f64> {
        let (dtype, raw) = self.raw(name);
        assert_eq!(dtype, "F64", "{name}");
        raw.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect()
    }

    fn i64s(&self, name: &str) -> Vec<i64> {
        let (dtype, raw) = self.raw(name);
        assert_eq!(dtype, "I64", "{name}");
        raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect()
    }
}

#[test]
#[ignore = "needs tools/golden/mage_flow_noise_golden.safetensors (gitignored; MAGE_DEVICE=cpu dump)"]
fn encode_noise_matches_the_committed_golden() {
    let g = Golden::open(Path::new(GOLDEN_FILE)).expect("noise golden — see the module docs");
    let key = GsKey::default();

    // The golden records its own configuration; read it rather than assuming it.
    assert_eq!(g.i64s("seed")[0], GOLDEN_SEED);
    assert_eq!(g.i64s("gs_key")[0] as u64, GS_DEFAULT_KEY);
    assert_eq!(
        g.header["gs_noise"]["shape"].as_array().unwrap().len(),
        4,
        "gs_noise is [1, C, gh, gw]"
    );

    // The key-schedule internals first, so a failure bisects instead of just saying "wrong tensor".
    let msg_golden: Vec<u8> = g.i64s("msg_bits").into_iter().map(|v| v as u8).collect();
    assert_eq!(message_bits(), msg_golden, "payload → 256-bit message");
    let (pad, pos) = pad_and_pos(32, &key, GS_MESSAGE_BITS).unwrap();
    let pad_golden: Vec<u8> = g.i64s("pad_tiny").into_iter().map(|v| v as u8).collect();
    let pos_golden: Vec<u32> = g.i64s("pos_tiny").into_iter().map(|v| v as u32).collect();
    assert_eq!(pad, pad_golden, "XOR pad");
    assert_eq!(pos, pos_golden, "message-index map");

    // The tiny tensor, then the production one.
    let tiny = host_f32(&encode_noise((8, 2, 2), &key, GOLDEN_SEED, Dtype::Float32).unwrap());
    assert_eq!(tiny, g.f32s("gs_noise_tiny"));

    let want = g.f32s("gs_noise");
    assert_eq!(want.len(), GOLDEN_N);
    let got = host_f32(&encode_noise(GOLDEN_SHAPE, &key, GOLDEN_SEED, Dtype::Float32).unwrap());
    let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("gs_noise: {mismatches}/{GOLDEN_N} float32 mismatches, max_abs {max_abs:e}");
    assert_eq!(mismatches, 0, "float32 tensor is not bit-exact");

    // The bfloat16 tensor the denoise loop actually consumes.
    let want_bf16 = g.f32s("gs_noise_bf16");
    let got_bf16 =
        host_f32(&encode_noise(GOLDEN_SHAPE, &key, GOLDEN_SEED, Dtype::Bfloat16).unwrap());
    let bf16_mismatches = got_bf16
        .iter()
        .zip(&want_bf16)
        .filter(|(a, b)| a != b)
        .count();
    println!("gs_noise_bf16: {bf16_mismatches}/{GOLDEN_N} mismatches");
    assert_eq!(bf16_mismatches, 0, "bfloat16 tensor is not bit-exact");

    // ...and it must NOT match the plain `randn` the reference computes and throws away.
    let plain = g.f32s("plain_randn");
    let apart = got
        .iter()
        .zip(&plain)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("distance from the discarded plain randn: max_abs {apart}");
    assert!(
        apart > 5.0,
        "the watermark must not collapse to plain randn"
    );

    // The reference's own detector statistics for this tensor.
    let report = decode_bits_host(&got, &key).unwrap();
    assert_eq!(report.raw_acc, g.f64s("detect_raw_acc")[0]);
    assert_eq!(report.msg_acc, g.f64s("detect_msg_acc")[0]);
    let z = g.f64s("detect_z_score")[0];
    assert!(
        (report.z_score - z).abs() < 1e-9,
        "{} vs {z}",
        report.z_score
    );
    assert!(report.present);
}

// =================================================================================================

fn host_f32(a: &Array) -> Vec<f32> {
    mlx_gen::array::contiguous(&a.as_dtype(Dtype::Float32).unwrap())
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}
