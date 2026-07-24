//! sc-14040: NR-MMDiT parity against the frozen torch reference, on **real bf16 weights**.
//!
//! Two gates, both fed the reference's own step-0 tensors so nothing upstream can mask a DiT bug:
//!
//! | gate | input | reference |
//! |---|---|---|
//! | one dual-stream block | `block_in.*` (post-`img_in`/`txt_in` streams, `temb`, the reference's own msrope table) | `block_out.{0,1}` |
//! | the full 12-block stack | `dit_in.*` (raw latents, raw conditioning, sigma, `img_shapes`) | `dit_out` |
//!
//! Plus a **front-end** gate (the input projections and the timestep conditioning against
//! `block_in.*`), the precision evidence behind the stack tolerance, and an explicit statement of
//! what this gate can and cannot discriminate.
//!
//! **A parity assertion nobody has tried to break is not evidence** — but on real bf16 weights the
//! two mutations that matter most (the sibling's SwiGLU activation; rotating the fused-CFG uncond
//! branch at msrope frame 0) both land *inside* the bf16 noise floor. That is measured and
//! asserted here rather than papered over, and both are caught in
//! `tests/mage_flow_small.rs` (f32 floor 2.4e-3, ~10× tighter) and `tests/msrope_golden.rs`
//! (table floor 6e-8, five orders tighter). What this file *does* discriminate on real weights is the deliberate bf16
//! rounding of the timestep frequency table: 1.1e-4 with it, 1.7e-2 without, against a 1e-3 gate.
//!
//! ## Why a tolerance and not equality
//!
//! The reference runs bf16 weights **and** bf16 activations with no autocast (`pipeline.py:753`),
//! and the goldens were dumped on CPU. This port runs the same dtypes on Metal, where MLX's matmul
//! is reduced precision (~1e-3 relative) and the reduction order differs from a CPU GEMM. Each of
//! the 12 blocks contains 8 GEMMs whose bf16 rounding is re-quantised at every op boundary, so the
//! per-block gap compounds roughly with the square root of the depth. Equality is not achievable
//! and is not the trained model's own reproducibility either — the reference's CPU and MPS runs of
//! the *same* stack disagree by ~2.7e-2 mean-relative (sc-14036 GAP 1). The gates below are stated
//! against the **measured** error with a small safety factor. The counter-probes that show a real
//! mistake landing outside a gate live in the two files named above, for the reason given there;
//! what this file contributes instead is the *evidence* that its own residual is precision
//! (`the_stack_residual_is_bf16_rounding_not_an_algorithmic_gap`) and an executable statement of
//! its limits (`the_real_weights_gate_cannot_separate_these_two_mistakes_and_says_so`).
//!
//! Run (`MAGE_SNAPSHOT` is a **passed-in path** to a `microsoft/Mage-Flow` snapshot — this
//! repository derives no cache location of its own):
//! ```text
//! MAGE_SNAPSHOT=/path/to/Mage-Flow-snapshot \
//!   cargo test --locked -p mlx-gen-mage --release --test dit_real_weights -- --ignored --nocapture
//! ```

mod common;

use common::{
    bf16_ulp_at, error, ints, peak_abs, require_golden, require_transformer_dir, BLOCK_GOLDEN,
    STACK_GOLDEN,
};

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mage::feed_forward::FfnActivation;
use mlx_gen_mage::{
    DualStream, ImgShape, MageTransformer, MsRope, PackContext, PackLayout, RopeTable,
};

// ---------------------------------------------------------------------------------------------
// Gate constants — the tolerances, with the measured numbers they were derived from.
// ---------------------------------------------------------------------------------------------

/// The pre-block projections (`img_in`, `txt_norm → txt_in`, the timestep conditioning). One or
/// two GEMMs deep, so this is held at bf16 round-off. **Measured** `max_rel` 4.8e-4 /
/// `mean_rel` 1.1e-4 (`img_in` alone: 2.4e-8 mean-relative — bit-level agreement). The gate is
/// ~4× the measurement, tight enough that the f32-frequency-table mutation (1.5e-2 / 1.7e-2)
/// fails it by an order of magnitude.
const FRONT_END_MAX_REL: f32 = 2.0e-3;
const FRONT_END_MEAN_REL: f32 = 1.0e-3;

/// One block, bf16 on Metal vs bf16 on CPU. **Measured** on the 512-token fused-CFG pack:
/// image stream `max_rel` 6.6e-3 / `mean_rel` 2.4e-3, text stream 1.2e-3 / 2.0e-3. The gate is
/// ~2.3× the measurement.
///
/// The measured `max_abs` of 5.2429e5 = 2¹⁹ is **within one bf16 rounding step of the largest
/// element**. bf16 carries 8 significand bits, so the ULP at a value in `[2ᵉ, 2ᵉ⁺¹)` is `2ᵉ⁻⁷`:
/// `block_out.1` peaks at 7.917e7 ∈ [2²⁶, 2²⁷), where the ULP is 2¹⁹ = 5.243e5 — the error is
/// exactly **one** ULP there; `block_out.0` peaks at 4.383e8 ∈ [2²⁸, 2²⁹), where the ULP is 2²¹ =
/// 2.097e6 — the error is a **quarter** ULP. (An earlier revision of this comment called it a half
/// ULP at the 4.4e8 peak, which is off by 2× in the wrong direction; the real result is tighter.)
///
/// `max_rel` is peak-relative (`max|Δ| / peak|ref|`), the honest denominator for an activation
/// tensor whose entries span eight orders of magnitude.
const BLOCK_MAX_REL: f32 = 1.5e-2;
const BLOCK_MEAN_REL: f32 = 6.0e-3;

/// The 12-block stack. **Measured** `max_rel` 2.1e-2 / `mean_rel` 2.1e-2 against the bf16 CPU
/// golden; the gate is ~1.9× that.
///
/// Why this is the right order and not evidence of a bug: re-running the identical port with the
/// weights upcast to f32 moves the output by `mean_rel` 2.8e-2 — **more** than the bf16 run's
/// distance from the golden. The model's own sensitivity to bf16 rounding at this depth is
/// therefore larger than the port-vs-reference gap, so the gap is bounded by precision, not by
/// arithmetic. (`the_stack_residual_is_bf16_rounding_not_an_algorithmic_gap` measures exactly
/// this; the sc-14036 record has the same finding on the reference itself — its own CPU and MPS
/// runs of the 36-layer text encoder disagree by ~2.7e-2 mean-relative.)
const STACK_MAX_REL: f32 = 4.0e-2;
const STACK_MEAN_REL: f32 = 4.0e-2;

// ---------------------------------------------------------------------------------------------

fn stack_layout(stack: &Weights) -> PackLayout {
    let shapes: Vec<ImgShape> = ints(stack, "img_shapes")
        .chunks(3)
        .map(|s| ImgShape::new(s[0], s[1], s[2]))
        .collect();
    let txt_cu = ints(stack, "dit_in.txt_cu_seqlens");
    let img_cu = ints(stack, "dit_in.img_cu_seqlens");
    let lens = |cu: &[i32]| cu.windows(2).map(|w| w[1] - w[0]).collect::<Vec<i32>>();
    let layout = PackLayout::new(shapes, lens(&img_cu), lens(&txt_cu)).unwrap();
    // The layout must reproduce the reference's own offsets, or the comparison below is
    // comparing two different packings.
    assert_eq!(layout.img_cu(), img_cu);
    assert_eq!(layout.txt_cu(), txt_cu);
    layout
}

fn transformer() -> MageTransformer {
    let dir = require_transformer_dir();
    let model =
        MageTransformer::load(&dir).unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    assert_eq!(
        model.dtype(),
        Dtype::Bfloat16,
        "the published transformer ships bf16 and the reference runs it at bf16 \
         (`pipeline.py:753`) — an upcast here would not be the trained configuration"
    );
    assert_eq!(model.blocks().len(), 12);
    model
}

/// One `MageFlowTransformerBlock` against the reference's own step-0 capture.
///
/// The block is fed the reference's `image_rotary_emb` directly rather than a locally-built table,
/// so a msrope bug cannot be laundered through this gate — msrope has its own gate in
/// `tests/msrope_golden.rs`.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + tools/golden/mage_flow_dit_block_golden.safetensors"]
fn one_block_matches_the_torch_golden() {
    let (block_err_txt, block_err_img) = run_block(FfnActivation::GeluApproximate);
    let (txt_abs, txt_rel, txt_mean) = block_err_txt;
    let (img_abs, img_rel, img_mean) = block_err_img;
    println!(
        "block txt: max_abs {txt_abs:.4e} max_rel {txt_rel:.4e} mean_rel {txt_mean:.4e}\n\
         block img: max_abs {img_abs:.4e} max_rel {img_rel:.4e} mean_rel {img_mean:.4e}"
    );
    let golden = require_golden(BLOCK_GOLDEN);
    for (name, (abs, rel, mean), key) in [
        ("txt", block_err_txt, "block_out.0"),
        ("img", block_err_img, "block_out.1"),
    ] {
        assert!(
            rel < BLOCK_MAX_REL,
            "block {name} stream max_rel {rel} exceeds {BLOCK_MAX_REL}"
        );
        assert!(
            mean < BLOCK_MEAN_REL,
            "block {name} stream mean_rel {mean} exceeds {BLOCK_MEAN_REL}"
        );
        // The ULP claim in BLOCK_MAX_REL's docs, made executable: the largest disagreement is
        // within ONE bf16 rounding step at that tensor's own peak magnitude. Hand exponent
        // arithmetic got this wrong by 2x once already.
        let peak = peak_abs(golden.require(key).unwrap());
        let ulp = bf16_ulp_at(peak);
        println!("block {name}: peak {peak:.4e} bf16 ULP there {ulp:.4e} vs max_abs {abs:.4e} ({:.2} ULP)", abs / ulp);
        assert!(
            abs <= ulp,
            "block {name} max_abs {abs} exceeds one bf16 ULP ({ulp}) at its {peak} peak"
        );
    }
}

/// The pre-block half — `img_in`, `txt_norm → txt_in`, and the timestep conditioning — against the
/// reference's own `block_in.*` capture.
///
/// This is the gate that keeps the 12-block number honest: it isolates the input projections and
/// the bf16-rounded sinusoid, so a front-end bug cannot hide inside "12 blocks of bf16
/// compounding". Everything here is one or two GEMMs deep, so it is held near bf16 round-off.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + tools/golden/mage_flow_dit_{block_,}golden.safetensors"]
fn stack_front_end_matches_the_reference_block_inputs() {
    let block = require_golden(BLOCK_GOLDEN);
    let stack = require_golden(STACK_GOLDEN);
    let layout = stack_layout(&stack);
    let model = transformer();
    let ctx = model.pack_context(layout).unwrap();

    let bf16 = |w: &Weights, key: &str| w.require(key).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    let (stream, temb) = model
        .embed(
            &bf16(&stack, "dit_in.img"),
            &bf16(&stack, "dit_in.txt"),
            &bf16(&stack, "dit_in.timesteps"),
            &ctx,
        )
        .unwrap();

    for (name, got, want) in [
        ("img_in", &stream.img, "block_in.hidden_states"),
        (
            "txt_norm→txt_in",
            &stream.txt,
            "block_in.encoder_hidden_states",
        ),
        ("time_text_embed", &temb, "block_in.temb"),
    ] {
        let (abs, rel, mean) = error(got, block.require(want).unwrap());
        println!("{name}: max_abs {abs:.4e} max_rel {rel:.4e} mean_rel {mean:.4e}");
        assert!(
            rel < FRONT_END_MAX_REL && mean < FRONT_END_MEAN_REL,
            "{name} diverged: max_rel {rel} (gate {FRONT_END_MAX_REL}), \
             mean_rel {mean} (gate {FRONT_END_MEAN_REL})"
        );
    }
}

/// The full 12-block stack, from raw latents + raw conditioning + sigma, with msrope built by this
/// crate from `img_shapes` — i.e. the whole DoD surface end to end.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + tools/golden/mage_flow_dit_golden.safetensors"]
fn full_twelve_block_stack_matches_the_torch_golden() {
    let (abs, rel, mean) = run_stack(FfnActivation::GeluApproximate, RopeChoice::Reference);
    println!("stack: max_abs {abs:.4e} max_rel {rel:.4e} mean_rel {mean:.4e}");
    assert!(
        rel < STACK_MAX_REL,
        "stack max_rel {rel} exceeds {STACK_MAX_REL}"
    );
    assert!(
        mean < STACK_MEAN_REL,
        "stack mean_rel {mean} exceeds {STACK_MEAN_REL}"
    );
}

/// Evidence that the stack residual is **bf16 rounding, not an algorithmic gap**.
///
/// The golden is a bf16 CPU run, so even a numerically perfect port is separated from it by the
/// reference's own accumulated bf16 round-off. Re-running this port with the same weights upcast
/// to f32 removes *our* half of that rounding: if the residual is precision, the f32 run lands
/// close to the bf16 run and no closer to the golden (both are limited by the reference's
/// rounding, which we cannot remove); if it were an algorithmic bug, the error would be
/// precision-insensitive in a different way — the f32 and bf16 runs would agree with each other
/// far more tightly than either agrees with the golden.
///
/// Measured: the two runs differ from each other by about the same amount each differs from the
/// golden, which is the signature of three independent roundings of the same computation.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + tools/golden/mage_flow_dit_golden.safetensors; ~16 GB f32 weights"]
fn the_stack_residual_is_bf16_rounding_not_an_algorithmic_gap() {
    let golden = require_golden(STACK_GOLDEN);
    let layout = stack_layout(&golden);
    let mut model = transformer();
    let ctx = model.pack_context(layout).unwrap();
    let want = golden.require("dit_out").unwrap();

    let at = |model: &MageTransformer, dtype: Dtype| {
        let cast = |key: &str| golden.require(key).unwrap().as_dtype(dtype).unwrap();
        model
            .forward(
                &cast("dit_in.img"),
                &cast("dit_in.txt"),
                &cast("dit_in.timesteps"),
                &ctx,
            )
            .unwrap()
    };
    let out_bf16 = at(&model, Dtype::Bfloat16);
    model.cast_weights(Dtype::Float32).unwrap();
    let out_f32 = at(&model, Dtype::Float32);

    let (_, bf16_rel, bf16_mean) = error(&out_bf16, want);
    let (_, f32_rel, f32_mean) = error(&out_f32, want);
    let (_, self_rel, self_mean) = error(&out_f32, &out_bf16);
    println!(
        "vs golden  — bf16: max_rel {bf16_rel:.4e} mean_rel {bf16_mean:.4e}\n\
         vs golden  — f32 : max_rel {f32_rel:.4e} mean_rel {f32_mean:.4e}\n\
         bf16 vs f32 (self): max_rel {self_rel:.4e} mean_rel {self_mean:.4e}"
    );
    // f32 is a diagnostic, not the shipped configuration: it is held to the same ORDER as the
    // bf16 gate rather than to the gate itself, because upcasting removes our rounding without
    // removing the reference's.
    assert!(
        f32_rel < STACK_MAX_REL * 2.0 && f32_mean < STACK_MEAN_REL * 2.0,
        "the f32 run ({f32_rel} / {f32_mean}) must stay the same order as the bf16 gate"
    );
    // ...and the bf16 run must be the CLOSER of the two: the golden is a bf16 run, so reproducing
    // its rounding is what parity means here.
    assert!(
        bf16_mean <= f32_mean,
        "the bf16 run ({bf16_mean}) should track the bf16 golden at least as well as an f32 run          ({f32_mean}) does"
    );
    // The port's own precision spread must be the same order as its distance from the golden —
    // i.e. the golden gap is not dominated by something precision cannot explain.
    assert!(
        self_mean > bf16_mean / 4.0,
        "bf16-vs-f32 self spread ({self_mean}) is far smaller than the golden gap ({bf16_mean}); \
         that would mean the residual is systematic, not rounding"
    );
}

/// The timestep frequency table's **deliberate bf16 rounding** (`mage_layers.py:32-46`) is
/// observable against the real conditioning: an f32 table moves `temb` measurably further from the
/// reference. Without this, [`crate::config::TIMESTEP_FREQS_BF16`] would be an unfalsifiable
/// comment.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + tools/golden/mage_flow_dit_{block_,}golden.safetensors"]
fn an_f32_timestep_frequency_table_moves_further_from_the_reference() {
    use mlx_gen_mage::MageTimestepEmbedder;

    let block = require_golden(BLOCK_GOLDEN);
    let stack = require_golden(STACK_GOLDEN);
    let weights = Weights::from_file(
        require_transformer_dir().join(mlx_gen_mage::transformer::TRANSFORMER_WEIGHTS_FILE),
    )
    .unwrap();
    let sigma = stack
        .require("dit_in.timesteps")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let want = block.require("block_in.temb").unwrap();

    let rounded = MageTimestepEmbedder::from_weights(&weights, "time_text_embed").unwrap();
    let mut exact = rounded.clone();
    exact.set_round_frequency_table(false);
    let (_, rounded_rel, rounded_mean) = error(&rounded.forward(&sigma).unwrap(), want);
    let (_, exact_rel, exact_mean) = error(&exact.forward(&sigma).unwrap(), want);
    println!(
        "temb bf16 table: max_rel {rounded_rel:.4e} mean_rel {rounded_mean:.4e}\n\
         temb f32  table: max_rel {exact_rel:.4e} mean_rel {exact_mean:.4e}"
    );
    assert!(
        exact_mean > rounded_mean,
        "the bf16-rounded frequency table must be the closer one (bf16 {rounded_mean}, \
         f32 {exact_mean}) — the model was trained with the rounding"
    );
}

/// **Where this gate's discrimination ends — measured, not assumed.**
///
/// The story this port was written from asked the parity gate to reject the `mlx-gen-z-image`
/// sibling's SwiGLU activation and the frame-index-0 rotation of the fused-CFG unconditional
/// branch. On the real checkpoint **it cannot**, and pretending otherwise by widening or narrowing
/// a tolerance would be a false green. Both are measured here and asserted to sit *inside* the
/// stack gate, with the place they ARE caught named:
///
/// * **SwiGLU's SiLU gate** moves `dit_out` by ~2e-2 — the same order as the bf16 floor. It is
///   caught in `tests/mage_flow_small.rs`, which runs the vendored reference itself at f32 where
///   the floor is 2.4e-3: the mutation lands 30× above it.
/// * **Rotating the uncond branch at frame 0** is a *real* difference on a generation pack, but a
///   small one: image↔image attention is unchanged (RoPE is relative and the duplication offsets
///   every shape in the second copy equally), so only image↔text attention leaks, and the text
///   stream is never rotated. At f32 that leak is 2.1e-3 on the unconditional branch against a
///   1.0e-1 bf16 spread on the guided velocity — see
///   `the_fused_cfg_frame_shift_is_precision_scale_not_semantic`, which measures it without the
///   rounding confound. It is caught at the **table** level (bit-level, `tests/msrope_golden.rs`)
///   and at the output level on an edit-shaped pack, where one attention window spans several
///   shapes (`tests/mage_flow_small.rs`).
///
/// Keeping this as an executable assertion rather than a comment means the classification cannot
/// rot: if a future change makes either mutation visible here, this test fails and says so.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + tools/golden/mage_flow_dit_golden.safetensors"]
fn the_real_weights_gate_cannot_separate_these_two_mistakes_and_says_so() {
    let (_, base, base_mean) = run_stack(FfnActivation::GeluApproximate, RopeChoice::Reference);
    let cases = [
        (
            "SwiGLU's SiLU gate",
            run_stack(FfnActivation::Silu, RopeChoice::Reference),
            "tests/mage_flow_small.rs (f32, 30x above its floor)",
        ),
        (
            "uncond branch rotated at msrope frame 0",
            run_stack(
                FfnActivation::GeluApproximate,
                RopeChoice::EveryBranchFrameZero,
            ),
            "tests/msrope_golden.rs (table level, 0.46 vs 2.4e-7) and tests/mage_flow_small.rs \
             (edit pack)",
        ),
    ];
    println!("unmutated stack: max_rel {base:.4e} mean_rel {base_mean:.4e}");
    for (what, (_, rel, mean), caught_by) in cases {
        println!(
            "{what}: max_rel {rel:.4e} mean_rel {mean:.4e} ({:.1}x the unmutated run) — caught by \
             {caught_by}",
            rel / base
        );
        assert!(
            rel < STACK_MAX_REL,
            "{what} now moves dit_out by max_rel {rel}, past the {STACK_MAX_REL} stack gate. \
             That is GOOD news — promote it to a real discrimination assertion here and update \
             the module docs, which currently state this gate cannot see it."
        );
    }
}

/// **The fused-CFG msrope frame shift, measured properly** — the number that settles whether
/// `batch_cfg` is a semantic or a precision-scale difference.
///
/// Three write-ups have now disagreed about this, so the measurement is made executable rather
/// than argued:
///
/// * sc-14036 said the fused path renders differently. Overstated.
/// * this port's first correction said the shift was inert in the output and the reference's
///   docstring at `pipeline.py:136-140` was right. **Also wrong** — the shift *is* a real
///   algebraic difference.
/// * what is true: it is real, and it is an order of magnitude below the model's own bf16 spread.
///
/// Two things make the measurement non-obvious. First, the conditional half is bit-identical by
/// construction (its shapes keep frame index 0 either way), so a whole-pack metric **halves** the
/// number — this test reports the unconditional branch separately. Second, measuring the shift at
/// bf16 measures mostly bf16 rounding; measuring it at **f32** isolates the algebra, and *that*
/// is the figure to compare against the bf16 spread.
///
/// Cond and uncond are isolated attention windows — `joint_cu_lens` is built per sample with
/// `batch_size = 2·na` (`mage_layers.py:430-441`) and the kernel loops per window
/// (`_attn_backend.py:192-208`) — so image↔image attention inside a window is invariant under a
/// constant frame offset (RoPE is relative). The leak is image↔**text** attention, because the
/// text stream is never rotated.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + tools/golden/mage_flow_dit_golden.safetensors; ~16 GB f32 weights"]
fn the_fused_cfg_frame_shift_is_precision_scale_not_semantic() {
    let golden = require_golden(STACK_GOLDEN);
    let cfg = golden.require("cfg").unwrap().as_slice::<f32>()[0];

    let shifted = |dt| run_stack_raw(FfnActivation::GeluApproximate, RopeChoice::Reference, dt);
    let flat = |dt| {
        run_stack_raw(
            FfnActivation::GeluApproximate,
            RopeChoice::EveryBranchFrameZero,
            dt,
        )
    };

    // --- f32: the shift's true algebraic size, with no rounding confound -----------------------
    let (f32_shifted, f32_flat) = (shifted(Dtype::Float32), flat(Dtype::Float32));
    let (cond_a, unc_a) = cfg_halves(&f32_shifted);
    let (cond_b, unc_b) = cfg_halves(&f32_flat);
    let (cond_abs, _, cond_mean) = error(&cond_b, &cond_a);
    let (_, _, unc_mean_f32) = error(&unc_b, &unc_a);
    let (_, _, guided_mean_f32) = error(&guided(&f32_flat, cfg), &guided(&f32_shifted, cfg));

    // --- bf16: the same shift, plus the model's own rounding spread ----------------------------
    let (bf_shifted, bf_flat) = (shifted(Dtype::Bfloat16), flat(Dtype::Bfloat16));
    let (_, unc_bf) = cfg_halves(&bf_shifted);
    let (_, unc_bf_flat) = cfg_halves(&bf_flat);
    let (_, _, unc_mean_bf16) = error(&unc_bf_flat, &unc_bf);
    let (_, _, whole_pack_bf16) = error(&bf_flat, &bf_shifted);
    // The precision yardstick: the same quantity's bf16-vs-f32 spread.
    let (_, _, spread_unc) = error(&unc_bf, &unc_a);
    let (_, _, spread_guided) = error(
        &guided(&bf_shifted, cfg),
        &guided(&f32_shifted, cfg).as_dtype(Dtype::Bfloat16).unwrap(),
    );

    println!(
        "f32  cond half   : max_abs {cond_abs:.4e} mean_rel {cond_mean:.4e}  (must be ~0)\n\
         f32  uncond half : mean_rel {unc_mean_f32:.4e}\n\
         f32  guided (cfg {cfg}) : mean_rel {guided_mean_f32:.4e}\n\
         bf16 uncond half : mean_rel {unc_mean_bf16:.4e}   [whole pack, diluted 2x: \
         {whole_pack_bf16:.4e}]\n\
         bf16-vs-f32 spread: uncond {spread_unc:.4e}  guided {spread_guided:.4e}"
    );

    // 1. The conditional branch does not move: its shapes keep frame index 0 either way. This is
    //    what makes a whole-pack metric exactly 2x diluted.
    assert!(
        cond_mean < 1.0e-6,
        "the conditional half must be unaffected by the duplicate's frame index (got {cond_mean})"
    );
    // 2. The shift IS a real algebraic difference — orders above the f32 floor. Asserting this
    //    stops the "it is inert" reading from coming back.
    assert!(
        unc_mean_f32 > 1.0e-4,
        "at f32 the frame shift moves the unconditional branch by {unc_mean_f32}, which would make \
         it numerically inert — the reference's 'numerically identical to two separate forwards' \
         docstring would then be true, contradicting the measurement this test records"
    );
    // 3. ...and it is an order of magnitude below the model's own bf16 sensitivity, which is why
    //    a frame-0 port renders indistinguishably ON GENERATION (it is still structurally wrong on
    //    edit, where one window spans several shapes — gated in tests/mage_flow_small.rs).
    assert!(
        guided_mean_f32 * 10.0 < spread_guided,
        "the frame shift ({guided_mean_f32}) is no longer an order of magnitude under the bf16 \
         spread ({spread_guided}) on the guided velocity — it has become semantic, and every doc \
         in this crate saying otherwise needs revisiting"
    );
}

// ---------------------------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------------------------

type Err3 = (f32, f32, f32);

fn run_block(activation: FfnActivation) -> (Err3, Err3) {
    let golden = require_golden(BLOCK_GOLDEN);
    let stack = require_golden(STACK_GOLDEN);
    let layout = stack_layout(&stack);
    // The block golden records the same packing; assert it rather than assume it.
    assert_eq!(layout.img_cu(), ints(&golden, "block_in.img_cu_lens"));
    assert_eq!(layout.txt_cu(), ints(&golden, "block_in.txt_cu_lens"));

    let ctx = PackContext::with_rope_table(
        layout,
        RopeTable {
            cos: golden
                .require("block_in.image_rotary_emb_re")
                .unwrap()
                .clone(),
            sin: golden
                .require("block_in.image_rotary_emb_im")
                .unwrap()
                .clone(),
        },
    )
    .unwrap();

    let mut model = transformer();
    model.set_ffn_activation(activation);
    let block = &model.blocks()[0];

    let bf16 = |key: &str| {
        golden
            .require(key)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    };
    let stream = DualStream {
        img: bf16("block_in.hidden_states"),
        txt: bf16("block_in.encoder_hidden_states"),
    };
    let temb = bf16("block_in.temb");
    let out = block.forward(&stream, &temb, &ctx).unwrap();

    (
        error(&out.txt, golden.require("block_out.0").unwrap()),
        error(&out.img, golden.require("block_out.1").unwrap()),
    )
}

enum RopeChoice {
    /// msrope built from `img_shapes` the way this crate does it — segment position is the frame
    /// index, so the fused-CFG duplicate rotates at 1.
    Reference,
    /// Every branch rotated at frame 0 — the "two separate forwards are identical" reading.
    EveryBranchFrameZero,
}

fn run_stack(activation: FfnActivation, rope_choice: RopeChoice) -> Err3 {
    let golden = require_golden(STACK_GOLDEN);
    let out = run_stack_raw(activation, rope_choice, Dtype::Bfloat16);
    error(&out, golden.require("dit_out").unwrap())
}

/// [`run_stack`] but returning the velocity itself, at a chosen weight/activation dtype — the seam
/// the fused-CFG measurement needs, since it compares runs against each other rather than against
/// the golden.
fn run_stack_raw(activation: FfnActivation, rope_choice: RopeChoice, dtype: Dtype) -> Array {
    let golden = require_golden(STACK_GOLDEN);
    let layout = stack_layout(&golden);

    let mut model = transformer();
    model.set_ffn_activation(activation);
    if dtype != Dtype::Bfloat16 {
        model.cast_weights(dtype).unwrap();
    }

    let ctx = match rope_choice {
        RopeChoice::Reference => model.pack_context(layout).unwrap(),
        RopeChoice::EveryBranchFrameZero => {
            let rope = MsRope::from_config(model.config()).unwrap();
            let per_segment: Vec<RopeTable> = layout
                .img_shapes()
                .iter()
                .map(|s| rope.forward(std::slice::from_ref(s)).unwrap())
                .collect();
            let cos: Vec<&Array> = per_segment.iter().map(|t| &t.cos).collect();
            let sin: Vec<&Array> = per_segment.iter().map(|t| &t.sin).collect();
            let table = RopeTable {
                cos: concatenate_axis(&cos, 0).unwrap(),
                sin: concatenate_axis(&sin, 0).unwrap(),
            };
            PackContext::with_rope_table(layout, table).unwrap()
        }
    };

    let cast = |key: &str| golden.require(key).unwrap().as_dtype(dtype).unwrap();
    model
        .forward(
            &cast("dit_in.img"),
            &cast("dit_in.txt"),
            &cast("dit_in.timesteps"),
            &ctx,
        )
        .unwrap()
}

/// Split a fused-CFG velocity `[1, 2·L, C]` into its conditional and unconditional halves.
fn cfg_halves(velocity: &Array) -> (Array, Array) {
    let tokens = velocity.shape()[1];
    assert_eq!(tokens % 2, 0, "a fused-CFG pack has two equal image halves");
    let half = tokens / 2;
    let idx =
        |from: i32, to: i32| Array::from_slice(&(from..to).collect::<Vec<i32>>(), &[to - from]);
    (
        velocity.take_axis(idx(0, half), 1).unwrap(),
        velocity.take_axis(idx(half, tokens), 1).unwrap(),
    )
}

/// The guided velocity the sampler actually steps with: `unc + cfg·(cond − unc)`
/// (`pipeline.py:230`). The frame shift only reaches an image through this combination, so it is
/// the quantity worth measuring.
fn guided(velocity: &Array, cfg: f32) -> Array {
    use mlx_rs::ops::{add, multiply, subtract};
    let (cond, unc) = cfg_halves(velocity);
    let scale = mlx_gen::array::scalar(cfg).as_dtype(cond.dtype()).unwrap();
    let delta = subtract(&cond, &unc).unwrap();
    add(&unc, multiply(scale, delta).unwrap()).unwrap()
}
