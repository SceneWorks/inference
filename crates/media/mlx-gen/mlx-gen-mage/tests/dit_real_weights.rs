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
//! `tests/mage_flow_small.rs` / `tests/msrope_golden.rs`, which have floors two to five orders of
//! magnitude tighter. What this file *does* discriminate on real weights is the deliberate bf16
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
//! against the **measured** error with a small safety factor, and each carries a counter-probe
//! showing a real mistake lands orders of magnitude outside it.
//!
//! Run (`MAGE_SNAPSHOT` is a **passed-in path** to a `microsoft/Mage-Flow` snapshot — this
//! repository derives no cache location of its own):
//! ```text
//! MAGE_SNAPSHOT=/path/to/Mage-Flow-snapshot \
//!   cargo test --locked -p mlx-gen-mage --release --test dit_real_weights -- --ignored --nocapture
//! ```

mod common;

use common::{error, ints, require_golden, require_transformer_dir, BLOCK_GOLDEN, STACK_GOLDEN};

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
/// image stream `max_rel` 6.6e-3 / `mean_rel` 2.4e-3, text stream 1.2e-3 / 2.0e-3. That is about
/// half a bf16 ULP per element — `block_out.0` peaks at 4.4e8, where a half-ULP is 5.2e5, which is
/// exactly the measured `max_abs`. The gate is ~2.3× the measurement.
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
    for (name, (_, rel, mean)) in [("txt", block_err_txt), ("img", block_err_img)] {
        assert!(
            rel < BLOCK_MAX_REL,
            "block {name} stream max_rel {rel} exceeds {BLOCK_MAX_REL}"
        );
        assert!(
            mean < BLOCK_MEAN_REL,
            "block {name} stream mean_rel {mean} exceeds {BLOCK_MEAN_REL}"
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
/// * **Rotating the uncond branch at frame 0** is *mathematically* inert on a generation pack:
///   RoPE encodes relative position, and the fused-CFG duplication offsets every shape in the
///   second copy by the same amount, so image↔image attention is unchanged. It leaks only through
///   image↔text attention (text is never rotated). It is caught at the **table** level — bit-level,
///   `tests/msrope_golden.rs` — and at the output level only on an edit-shaped pack, where one
///   attention window spans several shapes (`tests/mage_flow_small.rs`).
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
    let layout = stack_layout(&golden);

    let mut model = transformer();
    model.set_ffn_activation(activation);

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

    let bf16 = |key: &str| {
        golden
            .require(key)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    };
    let out = model
        .forward(
            &bf16("dit_in.img"),
            &bf16("dit_in.txt"),
            &bf16("dit_in.timesteps"),
            &ctx,
        )
        .unwrap();
    error(&out, golden.require("dit_out").unwrap())
}
